from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from typing import Dict, Any
from src.core.models import HealthResponse
from src.infrastructure.postgres import db_pool
from src.infrastructure.redis import redis_cache

router = APIRouter()

async def check_database() -> bool:
    """Check PostgreSQL connection."""
    try:
        async with db_pool.get_connection() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        return False

async def check_redis() -> bool:
    """Check Redis connection."""
    return await redis_cache.health_check()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check for load balancers."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        services={}
    )

@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check including dependencies."""
    db_healthy = await check_database()
    redis_healthy = await check_redis()
    
    services = {
        "database": "healthy" if db_healthy else "unhealthy",
        "redis": "healthy" if redis_healthy else "unhealthy"
    }
    
    overall_status = "healthy" if all([db_healthy, redis_healthy]) else "unhealthy"
    
    response = {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "services": services
    }
    
    # Return 503 if any service is unhealthy
    if overall_status == "unhealthy":
        raise HTTPException(status_code=503, detail=response)
    
    return response
