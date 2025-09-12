"""Infrastructure layer for external dependencies."""

from .postgres import (
    db_pool,
    initialize_database,
    close_database,
    get_db_connection,
)
from .redis import (
    redis_cache,
    initialize_redis,
    close_redis,
)

__all__ = [
    "db_pool",
    "initialize_database", 
    "close_database",
    "get_db_connection",
    "redis_cache",
    "initialize_redis",
    "close_redis",
]