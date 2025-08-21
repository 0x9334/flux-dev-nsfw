import os
import torch
import asyncio
import uuid
import time
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any, Awaitable
from dotenv import load_dotenv
from fastapi import FastAPI
from mmgp import offload, profile_type
from loguru import logger
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download

from schema import (
    ImageGenerationRequest, 
    ImageSize,
)

load_dotenv()

class TaskStatus:
    """Task status constants"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class QueueTask:
    """Represents a task in the queue"""
    def __init__(self, task_id: str, func: Callable, *args, **kwargs):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = TaskStatus.PENDING
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.future: Optional[asyncio.Future] = None

class ProductionQueueManager:
    """Production-ready async task queue manager for image generation"""
    
    def __init__(self, max_queue_size: int = 100, max_concurrent: int = 1):
        self.max_queue_size = max_queue_size
        self.max_concurrent = max_concurrent
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.active_tasks: Dict[str, QueueTask] = {}
        self.completed_tasks: Dict[str, QueueTask] = {}
        self.worker_tasks: list = []
        self.running = False
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(1)  # Limit to 1 concurrent task
        
    async def start(self):
        """Start the queue manager and worker tasks"""
        if self.running:
            logger.warning("Queue manager is already running")
            return
            
        self.running = True
        logger.info(f"Starting queue manager with semaphore limit of 1")
        
        # Start single worker task
        worker_task = asyncio.create_task(self._worker("worker-0"))
        self.worker_tasks.append(worker_task)
            
        logger.info("Queue manager started successfully")
    
    async def stop(self):
        """Stop the queue manager and cancel all tasks"""
        if not self.running:
            return
            
        logger.info("Stopping queue manager...")
        self.running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
            
        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Cancel all pending tasks
        async with self._lock:
            for task in self.active_tasks.values():
                if task.future and not task.future.done():
                    task.future.cancel()
                    task.status = TaskStatus.CANCELLED
                    
        logger.info("Queue manager stopped")
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task to the queue and return task ID"""
        if not self.running:
            raise RuntimeError("Queue manager is not running")
            
        if self.queue.full():
            raise RuntimeError("Queue is full")
            
        task_id = str(uuid.uuid4())
        task = QueueTask(task_id, func, *args, **kwargs)
        task.future = asyncio.Future()
        
        async with self._lock:
            self.active_tasks[task_id] = task
            
        await self.queue.put(task)
        logger.debug(f"Task {task_id} submitted to queue")
        
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None):
        """Get the result of a task by ID"""
        async with self._lock:
            task = self.active_tasks.get(task_id) or self.completed_tasks.get(task_id)
            
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        if not task.future:
            raise ValueError(f"Task {task_id} has no future")
            
        try:
            result = await asyncio.wait_for(task.future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task by ID"""
        async with self._lock:
            task = self.active_tasks.get(task_id) or self.completed_tasks.get(task_id)
            
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        return {
            "task_id": task_id,
            "status": task.status,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "error": str(task.error) if task.error else None
        }
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "queue_size": self.queue.qsize(),
                "max_queue_size": self.max_queue_size,
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "max_concurrent": self.max_concurrent,
                "running": self.running
            }
    
    async def _worker(self, worker_name: str):
        """Worker coroutine that processes tasks from the queue"""
        logger.info(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                logger.debug(f"Worker {worker_name} processing task {task.task_id}")
                
                # Use semaphore to limit concurrent execution
                async with self._semaphore:
                    # Update task status
                    task.status = TaskStatus.PROCESSING
                    task.started_at = time.time()
                    
                    try:
                        # Execute the task
                        if asyncio.iscoroutinefunction(task.func):
                            result = await task.func(*task.args, **task.kwargs)
                        else:
                            result = task.func(*task.args, **task.kwargs)
                        
                        # Task completed successfully
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = time.time()
                        
                        if task.future and not task.future.done():
                            task.future.set_result(result)
                            
                        logger.debug(f"Task {task.task_id} completed successfully")
                        
                    except Exception as e:
                        # Task failed
                        task.error = e
                        task.status = TaskStatus.FAILED
                        task.completed_at = time.time()
                        
                        if task.future and not task.future.done():
                            task.future.set_exception(e)
                            
                        logger.error(f"Task {task.task_id} failed: {e}")
                    
                    finally:
                        # Move task from active to completed
                        async with self._lock:
                            if task.task_id in self.active_tasks:
                                self.completed_tasks[task.task_id] = self.active_tasks.pop(task.task_id)
                        
                        # Mark task as done in queue
                        self.queue.task_done()
                    
            except asyncio.TimeoutError:
                # No task available, continue
                continue
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                
        logger.info(f"Worker {worker_name} stopped")

# Configuration
@dataclass
class Config:
    """Application configuration"""
    hf_token: str = os.getenv("HF_TOKEN", "")
    model_id: str = os.getenv("MODEL_ID", "FLUX.1-dev-NSFW")
    lora_file_name: str = os.getenv("LORA_FILE_NAME", "NSFW_master.safetensors")
    lora_local_path: str = os.getenv("LORA_LOCAL_PATH", "NSFW_master.safetensors")
    lora_repo: str = os.getenv("LORA_REPO", "NikolaSigmoid/FLUX.1-dev-NSFW-Master")
    flux_dev_repo: str = os.getenv("FLUX_DEV_REPO", "black-forest-labs/FLUX.1-dev")
    max_queue_size: int = int(os.getenv("MAX_QUEUE_SIZE", "100"))
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "1"))
    sync_timeout: int = int(os.getenv("SYNC_TIMEOUT", "600"))
    cleanup_interval: int = int(os.getenv("CLEANUP_INTERVAL", "300"))
    task_ttl: int = int(os.getenv("TASK_TTL", "3600"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

config = Config()

async def initialize_pipeline():
    """Initialize the FLUX.1-Krea-dev image generation pipeline"""
    global pipeline
    try:
        logger.info(f"Downloading {config.flux_dev_repo}...")
        hf_hub_download(repo_id=config.flux_dev_repo)
        logger.info(f"Downloading {config.lora_file_name}...")
        hf_hub_download(repo_id=config.lora_repo, filename=config.lora_file_name, local_dir=Path.cwd())
        lora_path = Path.cwd() / config.lora_file_name
        pipeline = FluxPipeline.from_pretrained(config.model_id, torch_dtype=torch.bfloat16, token = config.hf_token).to("cpu")
        pipeline.load_lora(lora_path)
        offload.profile(pipeline, profile_type.HighRAM_HighVRAM)
        
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global queue_manager
    
    # Startup
    logger.info("Starting up the image generation API...")
    
    # Initialize pipeline
    await initialize_pipeline()
    
    # Initialize queue manager
    queue_manager = ProductionQueueManager(
        max_queue_size=config.max_queue_size, 
        max_concurrent=config.max_concurrent
    )
    await queue_manager.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down the image generation API...")
    if queue_manager:
        await queue_manager.stop()

# Create FastAPI app
app = FastAPI(
    title="OpenAI-Compatible Image Generation API",
    description="A FastAPI-based image generation API compatible with OpenAI's DALL-E API using NVIDIA Cosmos",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint with provider details."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
