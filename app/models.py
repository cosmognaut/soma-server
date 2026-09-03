from pydantic import BaseModel
from typing import Literal, Optional

class Message(BaseModel):
    """A generic message given to the LLM"""
    role: Literal["user", "assistant"]
    content: str

class MessageRequest(BaseModel):
    """A message request to the LLM"""
    input: Message
    history: list[Message]
    file_id: Optional[str] = None

class StreamChunk(BaseModel):
    """A stream chunk to be sent to the user interface"""
    type: Literal["status", "answer", "failure"]
    payload: str

class UploadResponse(BaseModel):
    """Response after file upload"""
    file_id: str
    file_name: str
