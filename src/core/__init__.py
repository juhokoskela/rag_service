"""Core module containing configuration, models, and exceptions."""

from .config import settings
from .models import (
    Document,
    SearchRequest,
    SearchResult,
    SearchResponse,
    DocumentCreateRequest,
    DocumentUpdateRequest,
    HealthResponse,
)
from .exceptions import (
    RAGException,
    DocumentNotFoundError,
    SearchError,
    ValidationError,
    EmbeddingError,
    DatabaseError,
)

__all__ = [
    "settings",
    "Document",
    "SearchRequest",
    "SearchResult", 
    "SearchResponse",
    "DocumentCreateRequest",
    "DocumentUpdateRequest",
    "HealthResponse",
    "RAGServiceError",
    "DocumentNotFoundError",
    "SearchError", 
    "ValidationError",
    "EmbeddingError",
    "DatabaseError",
]