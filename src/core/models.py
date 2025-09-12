from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
import uuid

class Document(BaseModel):
    id: UUID = Field(default_factory=uuid.uuid4)
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    include_metadata: bool = True

class SearchResult(BaseModel):
    document: Document
    score: float
    rank_explanation: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    processing_time_ms: float

class DocumentCreateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentUpdateRequest(BaseModel):
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]