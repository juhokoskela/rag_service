"""pytest configuration and fixtures."""

import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator
import asyncpg
import redis.asyncio as redis
from httpx import AsyncClient, ASGITransport

from src.main import app
from src.core.config import settings
from src.infrastructure.postgres import get_db_connection
from src.infrastructure.redis import redis_cache


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for the test session."""
    return asyncio.get_event_loop_policy()


@pytest_asyncio.fixture
async def client():
    """Create an async test client for the FastAPI app with proper lifecycle management."""
    # Manually trigger lifespan startup
    from src.infrastructure.postgres import initialize_database
    from src.infrastructure.redis import initialize_redis
    
    try:
        await initialize_database()
        await initialize_redis()
    except Exception as e:
        print(f"Warning: Failed to initialize services for testing: {e}")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    
    # Cleanup
    try:
        from src.infrastructure.postgres import close_database
        from src.infrastructure.redis import close_redis
        await close_database()
        await close_redis()
    except Exception as e:
        print(f"Warning: Failed to cleanup services: {e}")


@pytest.fixture
async def db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Create a database connection for testing."""
    async with get_db_connection() as conn:
        yield conn


@pytest.fixture
async def redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Create a Redis client for testing."""
    yield redis_cache.redis


@pytest.fixture
async def sample_document():
    """Create a sample document for testing."""
    return {
        "id": "test-doc-1",
        "title": "Test Document",
        "content": "This is a test document content for searching and testing purposes.",
        "url": "https://example.com/test-doc",
        "metadata": {"source": "test", "type": "article"}
    }


@pytest.fixture
async def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        {
            "chunk_index": 0,
            "text": "This is the first chunk of test content.",
            "token_count": 10,
            "metadata": {"section": "intro"}
        },
        {
            "chunk_index": 1, 
            "text": "This is the second chunk with more test content.",
            "token_count": 12,
            "metadata": {"section": "body"}
        }
    ]
