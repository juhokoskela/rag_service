import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

from src.core.config import settings
from src.infrastructure.postgres import initialize_database, close_database
from src.infrastructure.redis import initialize_redis, close_redis
from src.api.routes import search, documents, health
from src.api.middleware import add_correlation_id, log_requests, enforce_authentication
import src.core.exceptions as exceptions

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Starting RAG service...")
    try:
        await initialize_database()
        await initialize_redis()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG service...")
    await close_database()
    await close_redis()
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A high-performance RAG (Retrieval-Augmented Generation) service for document search and retrieval",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Custom middleware (last registered runs first)
app.middleware("http")(log_requests)
app.middleware("http")(enforce_authentication)
app.middleware("http")(add_correlation_id)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(search.router, prefix=settings.api_prefix + "/search", tags=["search"])
app.include_router(documents.router, prefix=settings.api_prefix + "/documents", tags=["documents"])

# Global exception handlers
@app.exception_handler(exceptions.DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: exceptions.DocumentNotFoundError):
    return Response(
        status_code=404,
        content=f"Document not found: {exc.message}",
        media_type="application/json"
    )

@app.exception_handler(exceptions.SearchError)
async def search_error_handler(request: Request, exc: exceptions.SearchError):
    return Response(
        status_code=500,
        content=f"Search error: {exc.message}",
        media_type="application/json"
    )

@app.exception_handler(exceptions.ValidationError)
async def validation_error_handler(request: Request, exc: exceptions.ValidationError):
    return Response(
        status_code=400,
        content=f"Validation error: {exc.message}",
        media_type="application/json"
    )

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "status": "healthy",
        "docs_url": "/docs" if settings.debug else None
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
