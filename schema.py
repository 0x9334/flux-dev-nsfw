from enum import Enum
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


MODEL_ID = "FLUX.1-dev-NSFW"

class ImageSize(str, Enum):
    FLUX_SIZE = "1024x1024"

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class ImageGenerationRequest(BaseModel):
    """Request schema for OpenAI-compatible image generation API"""
    prompt: str = Field(..., description="A text description of the desired image(s). The maximum length is 1000 characters.", max_length=1000)
    model: Optional[str] = Field(default=MODEL_ID, description="The model to use for image generation")
    size: Optional[ImageSize] = Field(default=ImageSize.FLUX_SIZE, description="The size of the generated images")
    steps: Optional[int] = Field(default=28, ge=1, le=50, description="The number of inference steps (1-50)")
    priority: Optional[Priority] = Field(default=Priority.NORMAL, description="Task priority in queue")

class ImageChunk(BaseModel):
    """Image configuration"""
    content: str = Field(..., description="The content of the image")
    finish_reason: str = Field(default=None, description="The reason the generation finished")

class ImageData(BaseModel):
    """Individual image data in the response"""
    url: Optional[str] = Field(None, description="The URL of the generated image, if response_format is url")
    b64_json: Optional[str] = Field(None, description="The base64-encoded JSON of the generated image, if response_format is b64_json")

class ImageGenerationError(BaseModel):
    """Error response schema"""
    code: str = Field(..., description="Error code (e.g., 'contentFilter', 'generation_error', 'queue_full')")
    message: str = Field(..., description="Human-readable error message")
    type: Optional[str] = Field(None, description="Error type")

class ImageGenerationErrorResponse(BaseModel):
    """Error response wrapper"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the error occurred")
    error: ImageGenerationError = Field(..., description="Error details")

class QueueStats(BaseModel):
    """Queue statistics schema"""
    total_queued: int = Field(..., description="Total number of queued tasks")
    total_processing: int = Field(..., description="Total number of processing tasks")
    total_completed: int = Field(..., description="Total number of completed tasks")
    total_failed: int = Field(..., description="Total number of failed tasks")
    queue_size: int = Field(..., description="Current queue size")
    max_queue_size: int = Field(..., description="Maximum queue size")
    average_processing_time: Optional[float] = Field(None, description="Average processing time in seconds")
    estimated_wait_time: Optional[float] = Field(None, description="Estimated wait time for new tasks")

class TaskInfo(BaseModel):
    """Task information schema"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Task progress (0.0 to 1.0)")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    priority: Priority = Field(default=Priority.NORMAL, description="Task priority")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    queue_position: Optional[int] = Field(None, description="Current position in queue")

class TaskStatusResponse(BaseModel):
    """Task status response schema"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Task progress (0.0 to 1.0)")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    queue_position: Optional[int] = Field(None, description="Current position in queue")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error: Optional[ImageGenerationError] = Field(None, description="Error information if task failed")