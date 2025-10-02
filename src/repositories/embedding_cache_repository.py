from typing import Optional, List
import hashlib
import json
from datetime import datetime
import asyncpg
from src.infrastructure.postgres import get_db_connection
from src.infrastructure.redis import redis_cache
import logging

logger = logging.getLogger(__name__)

class EmbeddingCacheRepository:
    
    def _text_hash(self, text: str, model: str = "text-embedding-3-large") -> str:
        """Generate hash for text + model combination."""
        combined = f"{model}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()

    async def get_cached_embedding(self, text: str, model: str = "text-embedding-3-large") -> Optional[List[float]]:
        """Get cached embedding, checking Redis first, then PostgreSQL."""
        # Try Redis first (fastest)
        cached = await redis_cache.get_embedding(text)
        if cached:
            return cached
        
        # Try PostgreSQL cache
        async with get_db_connection() as conn:
            try:
                text_hash = self._text_hash(text, model)
                
                query = """
                SELECT embedding, access_count
                FROM embedding_cache
                WHERE text_hash = $1 AND model = $2
                """
                
                row = await conn.fetchrow(query, text_hash, model)
                if row:
                    # Update access count and timestamp
                    await conn.execute(
                        "UPDATE embedding_cache SET last_accessed = $1, access_count = $2 WHERE text_hash = $3 AND model = $4",
                        datetime.utcnow(), row['access_count'] + 1, text_hash, model
                    )
                    
                    # Parse embedding from string format
                    embedding_str = row['embedding']
                    if embedding_str:
                        # Remove brackets and split
                        embedding = [float(x) for x in embedding_str.strip('[]').split(',')]
                        
                        # Cache in Redis for next time
                        await redis_cache.set_embedding(text, embedding)
                        return embedding
                
                return None
            except Exception as e:
                logger.error(f"Failed to get cached embedding: {e}")
                return None

    async def cache_embedding(self, text: str, embedding: List[float], model: str = "text-embedding-3-large") -> bool:
        """Cache embedding in both Redis and PostgreSQL."""
        try:
            # Cache in Redis (fast access)
            await redis_cache.set_embedding(text, embedding)
            
            # Cache in PostgreSQL (persistent)
            async with get_db_connection() as conn:
                text_hash = self._text_hash(text, model)
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                now = datetime.utcnow()
                
                query = """
                INSERT INTO embedding_cache (id, text_hash, embedding, model, created_at, last_accessed, access_count)
                VALUES (uuid_generate_v4(), $1, $2, $3, $4, $5, 1)
                ON CONFLICT (text_hash) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    last_accessed = EXCLUDED.last_accessed,
                    access_count = embedding_cache.access_count + 1
                """
                
                await conn.execute(query, text_hash, embedding_str, model, now, now)
            
            return True
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
            return False

    async def cleanup_old_cache(self, days: int = 30) -> int:
        """Clean up old cached embeddings."""
        async with get_db_connection() as conn:
            try:
                query = """
                DELETE FROM embedding_cache
                WHERE last_accessed < NOW() - INTERVAL '%s days' AND access_count < 5
                """
                
                result = await conn.execute(query % days)
                deleted_count = int(result.split()[-1]) if result else 0
                
                logger.info(f"Cleaned up {deleted_count} old cached embeddings")
                return deleted_count
            except Exception as e:
                logger.error(f"Failed to cleanup cache: {e}")
                return 0
