import asyncpg
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class DatabasePool:
    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self._pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'application_name': 'rag-service',
                }
            )
            logger.info("Database pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self):
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get database connection from pool."""
        if not self._pool:
            await self.initialize()
        
        async with self._pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise

# Global database pool instance
db_pool = DatabasePool()

@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get database connection (convenience function)."""
    async with db_pool.get_connection() as conn:
        yield conn

async def initialize_database():
    """Initialize database on startup."""
    await db_pool.initialize()

async def close_database():
    """Close database on shutdown."""
    await db_pool.close()