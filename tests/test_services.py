import pytest
from unittest.mock import Mock, AsyncMock
from src.services.embedding import EmbeddingService
from src.services.chunking import ChunkingService
from src.repositories.embedding_cache_repository import EmbeddingCacheRepository


@pytest.mark.asyncio
async def test_embedding_service_initialization():
    """Test that EmbeddingService initializes correctly."""
    mock_cache_repo = Mock(spec=EmbeddingCacheRepository)
    service = EmbeddingService(mock_cache_repo)
    
    assert service.cache_repo == mock_cache_repo
    assert service.client is not None


@pytest.mark.asyncio
async def test_embedding_service_empty_text_validation():
    """Test that EmbeddingService validates empty text."""
    mock_cache_repo = Mock(spec=EmbeddingCacheRepository)
    service = EmbeddingService(mock_cache_repo)
    
    with pytest.raises(ValueError, match="Text cannot be empty"):
        await service.embed_text("")
    
    with pytest.raises(ValueError, match="Text cannot be empty"):
        await service.embed_text("   ")  # Only whitespace


def test_chunking_service_initialization():
    """Test that ChunkingService initializes correctly."""
    service = ChunkingService()
    
    assert service.splitter._chunk_size > 0
    assert service.splitter._chunk_overlap >= 0
    assert service.splitter._chunk_overlap < service.splitter._chunk_size


def test_chunking_service_empty_text():
    """Test chunking service with empty text."""
    service = ChunkingService()
    
    chunks = service.chunk_text("")
    assert len(chunks) == 0
    
    chunks = service.chunk_text("   ")  # Only whitespace
    assert len(chunks) == 0


def test_chunking_service_basic_text():
    """Test chunking service with basic text."""
    service = ChunkingService()
    
    text = "This is a simple test document. It has multiple sentences. Each sentence adds content."
    chunks = service.chunk_text(text)
    
    assert len(chunks) >= 1
    assert all(chunk.content.strip() for chunk in chunks)
    assert all(chunk.metadata.get('token_count', 0) > 0 for chunk in chunks)
    assert all(chunk.metadata.get('chunk_index', -1) >= 0 for chunk in chunks)


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    mock_cache_repo = Mock(spec=EmbeddingCacheRepository)
    service = EmbeddingService(mock_cache_repo)
    
    # Test identical vectors
    vec = [1.0, 0.0, 0.0]
    similarity = service.cosine_similarity(vec, vec)
    assert abs(similarity - 1.0) < 1e-6
    
    # Test orthogonal vectors  
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    similarity = service.cosine_similarity(vec_a, vec_b)
    assert abs(similarity - 0.0) < 1e-6
    
    # Test zero vectors
    zero_vec = [0.0, 0.0, 0.0]
    similarity = service.cosine_similarity(zero_vec, vec)
    assert similarity == 0.0
    