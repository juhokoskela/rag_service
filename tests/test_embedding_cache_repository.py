"""Integration tests for the embedding cache repository."""

import asyncio
from typing import List

import pytest

from src.core.config import settings
from src.repositories.embedding_cache_repository import EmbeddingCacheRepository
from src.infrastructure.redis import redis_cache
from src.infrastructure.postgres import db_pool, initialize_database, get_db_connection


pytestmark = pytest.mark.asyncio(loop_scope="module")


async def _ensure_pool_ready() -> None:
    """Ensure the asyncpg pool is bound to the current event loop."""
    pool = getattr(db_pool, "_pool", None)
    loop = asyncio.get_running_loop()
    needs_reset = (
        pool is None
        or getattr(pool, "_closed", False)
        or getattr(pool, "_loop", None) is not loop
    )
    if needs_reset:
        db_pool._pool = None
        try:
            await initialize_database()
        except Exception:
            pytest.skip("Database not configured for tests")


def _build_embedding(scale: float) -> List[float]:
    return [round((i + scale) / 1000.0, 6) for i in range(settings.embedding_dim)]


async def _reset_cache_state(text: str, model: str, repo: EmbeddingCacheRepository) -> None:
    await _ensure_pool_ready()
    async with get_db_connection() as conn:
        text_hash = repo._text_hash(text, model)
        await conn.execute("DELETE FROM embedding_cache WHERE text_hash = $1", text_hash)
    if redis_cache.redis:
        key = redis_cache._generate_key("embed", text)
        await redis_cache.redis.delete(key)


async def _fetch_cache_row(text: str, model: str, repo: EmbeddingCacheRepository):
    await _ensure_pool_ready()
    async with get_db_connection() as conn:
        text_hash = repo._text_hash(text, model)
        query = """
            SELECT embedding::text AS embedding_text,
                   access_count,
                   created_at,
                   last_accessed
            FROM embedding_cache
            WHERE text_hash = $1 AND model = $2
        """
        return await conn.fetchrow(query, text_hash, model)


async def _evict_from_redis(text: str) -> None:
    if redis_cache.redis:
        key = redis_cache._generate_key("embed", text)
        await redis_cache.redis.delete(key)


async def test_embedding_cache_round_trip(redis_client):
    if redis_client is None:
        pytest.skip("Redis not configured for tests")

    await _ensure_pool_ready()
    repo = EmbeddingCacheRepository()
    text = "Unit test embedding cache round trip"
    model = settings.embedding_model
    embedding = _build_embedding(scale=0.0)

    await _reset_cache_state(text, model, repo)

    assert await repo.cache_embedding(text, embedding, model=model)

    cached_redis = await repo.get_cached_embedding(text, model=model)
    assert cached_redis == embedding

    await _evict_from_redis(text)

    cached_db = await repo.get_cached_embedding(text, model=model)
    assert cached_db == pytest.approx(embedding, rel=5e-4, abs=1e-3)

    row = await _fetch_cache_row(text, model, repo)
    assert row is not None
    assert row["access_count"] == 2
    assert row["last_accessed"] >= row["created_at"]


async def test_embedding_cache_conflict_updates_existing_row(redis_client):
    if redis_client is None:
        pytest.skip("Redis not configured for tests")

    await _ensure_pool_ready()
    repo = EmbeddingCacheRepository()
    text = "Unit test embedding cache conflict"
    model = settings.embedding_model
    first_embedding = _build_embedding(scale=1.0)
    second_embedding = _build_embedding(scale=2.0)

    await _reset_cache_state(text, model, repo)

    assert await repo.cache_embedding(text, first_embedding, model=model)

    first_row = await _fetch_cache_row(text, model, repo)
    assert first_row["access_count"] == 1

    assert await repo.cache_embedding(text, second_embedding, model=model)

    second_row = await _fetch_cache_row(text, model, repo)
    assert second_row["access_count"] == 2

    parsed_embedding = [float(value.strip()) for value in second_row["embedding_text"].strip("[]").split(",")]
    assert len(parsed_embedding) == len(second_embedding)
    assert parsed_embedding == pytest.approx(second_embedding, rel=5e-4, abs=1e-3)
