import asyncio
import base64
import io
import os
import time
import uuid
import json
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import heapq
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
import torch
from diffusers import FluxPipeline
from PIL import Image
from mmgp import offload, profile_type
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download, snapshot_download

from schema import (
    ImageGenerationRequest, 
    ImageGenerationError, 
    ImageGenerationErrorResponse,
    ImageSize,
    TaskStatus,
    TaskInfo,
    TaskStatusResponse,
    Priority,
    ImageChunk,
    QueueStats,
)

load_dotenv()

MAX_IMAGE_CHUNK_SIZE = 512

# Configuration
@dataclass
class Config:
    """Application configuration"""
    hf_token: str = os.getenv("HF_TOKEN", "")
    model_id: str = os.getenv("MODEL_ID", "FLUX.1-dev-NSFW")
    lora_file_name: str = os.getenv("LORA_FILE_NAME", "NSFW_master.safetensors")
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

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level.upper()))
logger = logging.getLogger(__name__)

@dataclass
class QueuedTask:
    """Represents a task in the priority queue"""
    task_id: str
    request: ImageGenerationRequest
    created_at: datetime
    priority: int = field(default=1)  # Lower number = higher priority
    
    def __lt__(self, other):
        # First compare by priority, then by creation time for FIFO within same priority
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at

class ProductionQueueManager:
    """Production-ready queue manager with concurrency control"""
    
    def __init__(self, max_queue_size: int = 100, max_concurrent: int = 1):
        self.max_queue_size = max_queue_size
        self.max_concurrent = max_concurrent
        
        # Semaphore to control concurrent processing
        self.processing_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Priority queue for tasks
        self.task_queue: list[QueuedTask] = []
        self.queue_lock = asyncio.Lock()
        
        # Task tracking
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_errors: Dict[str, ImageGenerationErrorResponse] = {}
        
        # Statistics
        self.stats = {
            'total_queued': 0,
            'total_processed': 0,
            'total_completed': 0,
            'total_failed': 0,
            'processing_times': []
        }
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.processor_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the queue manager"""
        logger.info("Starting production queue manager...")
        self.processor_task = asyncio.create_task(self._process_queue())
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_tasks())
        
    async def stop(self):
        """Stop the queue manager"""
        logger.info("Stopping production queue manager...")
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
                
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
    
    def _get_priority_value(self, priority: Priority) -> int:
        """Convert priority enum to numeric value"""
        priority_map = {
            Priority.HIGH: 0,
            Priority.NORMAL: 1,
            Priority.LOW: 2
        }
        return priority_map.get(priority, 1)
    
    async def add_task(self, task_id: str, request: ImageGenerationRequest) -> int:
        """Add a task to the queue and return queue position"""
        async with self.queue_lock:
            # Check if queue is full
            if len(self.task_queue) >= self.max_queue_size:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Queue is full. Maximum {self.max_queue_size} tasks allowed."
                )
            
            # Create queued task
            queued_task = QueuedTask(
                task_id=task_id,
                request=request,
                created_at=datetime.now(),
                priority=self._get_priority_value(request.priority or Priority.NORMAL)
            )
            
            # Add to priority queue
            heapq.heappush(self.task_queue, queued_task)
            
            # Create task info
            self.tasks[task_id] = TaskInfo(
                task_id=task_id,
                status=TaskStatus.QUEUED,
                progress=0.0,
                created_at=datetime.now(),
                priority=request.priority or Priority.NORMAL,
                queue_position=self._get_queue_position(task_id)
            )
            
            self.stats['total_queued'] += 1
            logger.info(f"Added task {task_id} to queue (priority: {request.priority})")
            
            return len(self.task_queue)
    
    def _get_queue_position(self, task_id: str) -> int:
        """Get the current position of a task in the queue"""
        for i, task in enumerate(sorted(self.task_queue)):
            if task.task_id == task_id:
                return i + 1
        return -1
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatusResponse]:
        """Get the status of a task"""
        task_info = self.tasks.get(task_id)
        if not task_info:
            return None
            
        # Update queue position if still queued
        if task_info.status == TaskStatus.QUEUED:
            task_info.queue_position = self._get_queue_position(task_id)
            # Estimate completion time
            if task_info.queue_position > 0:
                avg_time = self._get_average_processing_time()
                if avg_time:
                    task_info.estimated_completion = datetime.now() + timedelta(
                        seconds=avg_time * task_info.queue_position
                    )
        
        return TaskStatusResponse(
            task_id=task_info.task_id,
            status=task_info.status,
            progress=task_info.progress,
            created_at=task_info.created_at,
            started_at=task_info.started_at,
            completed_at=task_info.completed_at,
            queue_position=task_info.queue_position,
            estimated_completion=task_info.estimated_completion,
            error=self.task_errors.get(task_id, {}).get('error') if task_id in self.task_errors else None
        )
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a completed task"""
        return self.task_results.get(task_id)
    
    def get_queue_stats(self) -> QueueStats:
        """Get current queue statistics"""
        processing_count = sum(1 for task in self.tasks.values() if task.status == TaskStatus.PROCESSING)
        completed_count = sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
        failed_count = sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)
        
        return QueueStats(
            total_queued=len(self.task_queue),
            total_processing=processing_count,
            total_completed=completed_count,
            total_failed=failed_count,
            queue_size=len(self.task_queue),
            max_queue_size=self.max_queue_size,
            average_processing_time=self._get_average_processing_time(),
            estimated_wait_time=self._get_estimated_wait_time()
        )
    
    def _get_average_processing_time(self) -> Optional[float]:
        """Calculate average processing time"""
        if not self.stats['processing_times']:
            return None
        return sum(self.stats['processing_times']) / len(self.stats['processing_times'])
    
    def _get_estimated_wait_time(self) -> Optional[float]:
        """Estimate wait time for new tasks"""
        avg_time = self._get_average_processing_time()
        if avg_time and len(self.task_queue) > 0:
            return avg_time * len(self.task_queue)
        return None
    
    async def _process_queue(self):
        """Main queue processing loop"""
        logger.info("Starting queue processor...")
        
        while True:
            try:
                # Get next task
                queued_task = None
                async with self.queue_lock:
                    if self.task_queue:
                        queued_task = heapq.heappop(self.task_queue)
                
                if queued_task:
                    # Acquire semaphore to limit concurrent processing
                    async with self.processing_semaphore:
                        await self._process_task(queued_task)
                else:
                    # No tasks to process, wait a bit
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(5)
    
    async def _process_task(self, queued_task: QueuedTask):
        """Process a single task"""
        task_id = queued_task.task_id
        start_time = time.time()
        
        try:
            # Update task status
            task_info = self.tasks[task_id]
            task_info.status = TaskStatus.PROCESSING
            task_info.started_at = datetime.now()
            task_info.progress = 0.1
            
            logger.info(f"Processing task {task_id} (priority: {queued_task.priority})")
            
            # Generate image
            result = await generate_image(queued_task.request, task_id, task_info)
            
            # Store result
            self.task_results[task_id] = result
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = datetime.now()
            task_info.progress = 1.0
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            if len(self.stats['processing_times']) > 100:  # Keep only last 100 times
                self.stats['processing_times'] = self.stats['processing_times'][-100:]
            
            self.stats['total_processed'] += 1
            self.stats['total_completed'] += 1
            
            logger.info(f"Completed task {task_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            
            # Create error response
            error_response = ImageGenerationErrorResponse(
                created=int(time.time()),
                error=ImageGenerationError(
                    code="generation_error",
                    message=f"Failed to generate image: {str(e)}",
                    type="processing_error"
                )
            )
            
            self.task_errors[task_id] = error_response
            task_info = self.tasks[task_id]
            task_info.status = TaskStatus.FAILED
            task_info.completed_at = datetime.now()
            
            self.stats['total_processed'] += 1
            self.stats['total_failed'] += 1
    
    async def _cleanup_expired_tasks(self):
        """Clean up expired tasks periodically"""
        while True:
            try:
                current_time = datetime.now()
                expired_tasks = []
                
                # Find expired tasks (older than configured TTL)
                for task_id, task_info in self.tasks.items():
                    if (current_time - task_info.created_at).total_seconds() > config.task_ttl:
                        expired_tasks.append(task_id)
                
                # Clean up expired tasks
                for task_id in expired_tasks:
                    logger.info(f"Cleaning up expired task {task_id}")
                    self.tasks.pop(task_id, None)
                    self.task_results.pop(task_id, None)
                    self.task_errors.pop(task_id, None)
                
                # Sleep for configured cleanup interval
                await asyncio.sleep(config.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)

# Global variables
pipeline = None
queue_manager: Optional[ProductionQueueManager] = None

async def initialize_pipeline():
    """Initialize the FLUX.1-Krea-dev image generation pipeline"""
    global pipeline
    try:
        logger.info(f"Downloading {config.flux_dev_repo}...")
        snapshot_download(repo_id=config.flux_dev_repo)
        logger.info(f"Downloading {config.lora_file_name}...")
        path = hf_hub_download(repo_id=config.lora_repo, filename=config.lora_file_name)
        pipeline = FluxPipeline.from_pretrained(config.flux_dev_repo, torch_dtype=torch.bfloat16, token = config.hf_token).to("cpu")
        pipeline.load_lora_weights(path)

        offload.profile(pipeline, profile_type.HighRAM_HighVRAM)
        
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise e

def parse_image_size(size: ImageSize) -> tuple:
    """Parse image size string to width, height tuple"""
    width, height = map(int, size.value.split('x'))
    return width, height

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

async def generate_image(request_data: ImageGenerationRequest, task_id: str, task_info: TaskInfo) -> str:
    """Generate image using the pipeline"""
    logger.info(f"Starting image generation for task {task_id}")
    
    # Set default negative prompt if not provided
    # negative_prompt = request_data.negative_prompt or None
    
    # Parse image size
    width, height = parse_image_size(request_data.size)
    
    # Set number of inference steps
    num_steps = request_data.steps or 28
    prompt = request_data.prompt
        
    # Generate image in thread pool to avoid blocking the event loop
    def _run_pipeline():
        with torch.no_grad():
            return pipeline(
                prompt=prompt,
                num_inference_steps=num_steps,
                width=width,
                height=height,
                guidance_scale=3.5,
                generator=torch.Generator().manual_seed(int(time.time()))
            )
    
    # Run the pipeline in a thread pool to keep the event loop responsive
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_pipeline)
        
    task_info.progress = 0.8
        
    # Get the generated image
    generated_image = result.images[0]
    
    # Convert to base64
    image_b64 = image_to_base64(generated_image)
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clean up result to free memory
    del result
    del generated_image
    
    return image_b64

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
    title="Image Generation API",
    description="A FastAPI-based image generation API",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/images/generations")
@app.post("/v1/images/generations")
async def create_image(request: ImageGenerationRequest) -> StreamingResponse:
    async def fake_stream_generator():
        task_id = str(uuid.uuid4())
        start_time = time.time()
        max_wait_time = 300  # 5 minutes timeout

        try:
            # add to queue
            queue_position = await queue_manager.add_task(task_id, request)
            logger.info(f"Added task {task_id} to queue at position {queue_position}")
            
            while True:
                # Check for timeout
                if time.time() - start_time > max_wait_time:
                    timeout_chunk = ImageChunk(
                        content="Request timed out after 5 minutes",
                        finish_reason="error"
                    )
                    timeout_json = await asyncio.to_thread(json.dumps, timeout_chunk.model_dump())
                    yield f"data: {timeout_json}\n\n"
                    yield "data: [DONE]\n\n"
                    break

                try:
                    task_status = await queue_manager.get_task_status(task_id)
                    if task_status.status == TaskStatus.COMPLETED:
                        result = await queue_manager.get_task_result(task_id)
                        # split result into many small chunks for streaming
                        is_final_chunk = False
                        for i in range(0, len(result), MAX_IMAGE_CHUNK_SIZE):
                            is_final_chunk = i + MAX_IMAGE_CHUNK_SIZE >= len(result)
                            chunk = ImageChunk(
                                image_base64=result[i : i+ MAX_IMAGE_CHUNK_SIZE],
                                finish_reason="stop" if is_final_chunk else None
                            )
                            # Use asyncio.to_thread for JSON serialization to avoid blocking
                            chunk_json = await asyncio.to_thread(json.dumps, chunk.model_dump())
                            yield f"data: {chunk_json}\n\n"
                            # Yield control back to event loop periodically
                            if i % (MAX_IMAGE_CHUNK_SIZE * 10) == 0:
                                await asyncio.sleep(0)
                        yield "data: [DONE]\n\n"
                        break
                    elif task_status.status == TaskStatus.FAILED:
                        error_chunk = ImageChunk(
                            content=task_status.error.message,
                            finish_reason="error"
                        )
                        # Use asyncio.to_thread for JSON serialization
                        error_json = await asyncio.to_thread(json.dumps, error_chunk.model_dump())
                        yield f"data: {error_json}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    elif task_status.status == TaskStatus.QUEUED:
                        queued_chunk = ImageChunk(
                            content="Still queued..."
                        )
                        # Use asyncio.to_thread for JSON serialization
                        queued_json = await asyncio.to_thread(json.dumps, queued_chunk.model_dump())
                        yield f"data: {queued_json}\n\n"
                        await asyncio.sleep(1)
                    else:
                        processing_chunk = ImageChunk(
                            content="Still processing..."
                        )
                        # Use asyncio.to_thread for JSON serialization
                        processing_json = await asyncio.to_thread(json.dumps, processing_chunk.model_dump())
                        yield f"data: {processing_json}\n\n"
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {e}")
                    error_chunk = ImageChunk(
                        content=f"Internal error: {str(e)}",
                        finish_reason="error"
                    )
                    error_json = await asyncio.to_thread(json.dumps, error_chunk.model_dump())
                    yield f"data: {error_json}\n\n"
                    yield "data: [DONE]\n\n"
                    break
        except Exception as e:
            logger.error(f"Error in stream generator for task {task_id}: {e}")
            error_chunk = ImageChunk(
                content=f"Stream error: {str(e)}",
                finish_reason="error"
            )
            error_json = await asyncio.to_thread(json.dumps, error_chunk.model_dump())
            yield f"data: {error_json}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(fake_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

@app.get("/v1/images/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of an image generation task"""
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")
    
    task_status = await queue_manager.get_task_status(task_id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": config.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "eternalai"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
