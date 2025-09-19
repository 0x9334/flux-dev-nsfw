from enum import Enum
from datetime import datetime
from typing import Optional, Union, Literal, List
from pydantic import BaseModel, Field, ConfigDict

MODEL_ID = "FLUX.1-dev-NSFW"

class BaseModelWithDatetime(BaseModel):
    """Base model with proper datetime serialization"""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        },
        use_enum_values=True
    )

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class Model(BaseModel):
    """
    Represents a model in the models list response.
    """
    id: str = Field(..., description="The model ID.")
    object: str = Field("model", description="The object type, always 'model'.")
    created: int = Field(..., description="The creation timestamp.")
    owned_by: str = Field("user", description="The owner of the model.")

class ModelsResponse(BaseModel):
    """
    Represents the response for the models list endpoint.
    """
    object: str = Field("list", description="The object type, always 'list'.")
    data: List[Model] = Field(..., description="List of models.")

class ResponseStatus(BaseModelWithDatetime):
    """Response status schema"""
    status: TaskStatus = Field(..., description="The status of the response")
    progress: float = Field(..., ge=0.0, le=100.0, description="The progress of the response as a percentage (0-100)")
    queue_position: Optional[int] = Field(None, description="Position in queue (only for queued status) - use with total_queue_size for format like '2/5'")
    total_queue_size: Optional[int] = Field(None, description="Total number of tasks in queue (only for queued status) - use with queue_position for format like '2/5'")
    started_at: Optional[datetime] = Field(None, description="When processing started (only for processing status)")
    estimated_wait_time: Optional[float] = Field(None, description="Estimated wait time in seconds (null if not in queue, queue_position * average_processing_time if in queue)")

class ImageChunk(BaseModelWithDatetime):
    """Individual image data in the response"""
    id: str = Field(..., description="Request identifier - same for all chunks in a request")
    content: Optional[str] = Field(None, description="The content of the chunk")
    image_base64: Optional[str] = Field(None, description="The base64-encoded image data")
    status: ResponseStatus = Field(..., description="The status of the response")
    finish_reason: Union[Literal["stop", "error"], None] = Field(None, description="The finish reason of the chunk")

class APIError(BaseModel):
    """Error response schema"""
    code: str = Field(..., description="Error code (e.g., 'contentFilter', 'processing_error', 'queue_full')")
    message: str = Field(..., description="Human-readable error message")
    type: Optional[str] = Field(None, description="Error type")

class APIErrorResponse(BaseModel):
    """Error response wrapper"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the error occurred")
    error: APIError = Field(..., description="Error details")

class FailureReason(BaseModelWithDatetime):
    """Failure reason details"""
    error_type: str = Field(..., description="Type of error (e.g., 'pipeline_error', 'memory_error', 'timeout')")
    error_message: str = Field(..., description="Human-readable error message")
    count: int = Field(..., description="Number of times this error occurred")
    last_occurrence: datetime = Field(..., description="When this error last occurred")

class QueueStats(BaseModel):
    """Queue statistics schema"""
    total_queued: int = Field(..., description="Total number of queued tasks")
    total_processing: int = Field(..., description="Total number of processing tasks")
    total_completed: int = Field(..., description="Total number of completed tasks")
    total_failed: int = Field(..., description="Total number of failed tasks")
    total_dropped: int = Field(..., description="Total number of dropped tasks (exceeded TTL)")
    queue_size: int = Field(..., description="Current queue size")
    max_queue_size: int = Field(..., description="Maximum queue size")
    average_processing_time: Optional[float] = Field(None, description="Average processing time in seconds")
    estimated_wait_time: Optional[float] = Field(None, description="Estimated wait time for new tasks")
    failure_reasons: List[FailureReason] = Field(default_factory=list, description="Detailed breakdown of failure reasons")

class TaskInfo(BaseModelWithDatetime):
    """Task information schema"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Task progress (0.0 to 1.0)")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    priority: Priority = Field(default=Priority.NORMAL, description="Task priority")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    queue_position: Optional[int] = Field(None, description="Current position in queue - use with total_queue_size for format like '2/5'")
    total_queue_size: Optional[int] = Field(None, description="Total number of tasks in queue - use with queue_position for format like '2/5'")

class TaskStatusResponse(BaseModelWithDatetime):
    """Task status response schema"""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Task progress (0.0 to 1.0)")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    queue_position: Optional[int] = Field(None, description="Current position in queue - use with total_queue_size for format like '2/5'")
    total_queue_size: Optional[int] = Field(None, description="Total number of tasks in queue - use with queue_position for format like '2/5'")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error: Optional[APIError] = Field(None, description="Error information if task failed")

class ImageGenerationRequest(BaseModel):
    """Request schema for OpenAI-compatible image generation API (for internal processing)"""
    model: str = Field(MODEL_ID, description="The model to use for image generation")
    prompt: str = Field(..., description="A text description of the desired image. The maximum length is 1000 characters.", max_length=1000)
    priority: Optional[Priority] = Field(default=Priority.NORMAL, description="Task priority in queue")