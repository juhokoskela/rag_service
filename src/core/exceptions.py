"""Custom exceptions for the RAG service."""

class RAGException(Exception):
    """Base exception for RAG service."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class DocumentNotFoundError(RAGException):
    """Raised when a document is not found."""
    pass

class SearchError(RAGException):
    """Raised when search operations fail."""
    pass

class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass

class ValidationError(RAGException):
    """Raised when input validation fails."""
    pass

class DatabaseError(RAGException):
    """Raised when database operations fail."""
    pass

class CacheError(RAGException):
    """Raised when cache operations fail."""
    pass