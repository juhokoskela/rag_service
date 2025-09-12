import redis.asyncio as redis
import json
import hashlib
from typing import Optional, Any, List
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self):
        self.redis: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            
            # Test connection
            await self.redis.ping()
            logger.info("Redis connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.aclose()
            logger.info("Redis connection closed")

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key."""
        if isinstance(data, str):
            data_str = data
        else:
            data_str = json.dumps(data, sort_keys=True)
        
        hash_obj = hashlib.sha256(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"

    async def set_embedding(self, text: str, embedding: List[float], ttl: int = 86400) -> bool:
        """Cache embedding with TTL."""
        try:
            if not self.redis:
                return False
                
            key = self._generate_key("embed", text)
            value = json.dumps(embedding)
            
            await self.redis.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
            return False

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding."""
        try:
            if not self.redis:
                return None
                
            key = self._generate_key("embed", text)
            value = await self.redis.get(key)
            
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get cached embedding: {e}")
            return None

    async def set_search_results(self, query: str, results: Any, ttl: int = 3600) -> bool:
        """Cache search results."""
        try:
            if not self.redis:
                return False
                
            key = self._generate_key("search", query)
            value = json.dumps(results, default=str)
            
            await self.redis.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Failed to cache search results: {e}")
            return False

    async def get_search_results(self, query: str) -> Optional[Any]:
        """Get cached search results."""
        try:
            if not self.redis:
                return None
                
            key = self._generate_key("search", query)
            value = await self.redis.get(key)
            
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get cached search results: {e}")
            return None

    async def invalidate_search_cache(self) -> bool:
        """Invalidate all search caches (call when documents change)."""
        try:
            if not self.redis:
                return False
                
            keys = await self.redis.keys("search:*")
            if keys:
                await self.redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Failed to invalidate search cache: {e}")
            return False

    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            if not self.redis:
                return False
            await self.redis.ping()
            return True
        except Exception:
            return False

# Global Redis cache instance
redis_cache = RedisCache()

async def initialize_redis():
    """Initialize Redis on startup."""
    await redis_cache.initialize()

async def close_redis():
    """Close Redis on shutdown."""
    await redis_cache.close()