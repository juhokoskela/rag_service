import time
import uuid
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

async def add_correlation_id(request: Request, call_next):
    """Add correlation ID for request tracing."""
    correlation_id = request.headers.get("x-correlation-id") or str(uuid.uuid4())
    
    # Add to request state
    request.state.correlation_id = correlation_id
    
    # Process request
    response = await call_next(request)
    
    # Add to response headers
    response.headers["x-correlation-id"] = correlation_id
    
    return response

async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    # Log request
    logger.info(
        f"Request started - {request.method} {request.url.path} "
        f"[{correlation_id}]"
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        f"Request completed - {request.method} {request.url.path} "
        f"[{correlation_id}] - {response.status_code} - {process_time:.3f}s"
    )
    
    # Add timing header
    response.headers["x-process-time"] = str(process_time)
    
    return response