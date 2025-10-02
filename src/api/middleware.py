import time
import uuid
import logging
import secrets
from typing import Any, Dict

import jwt
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from jwt import InvalidTokenError
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import settings

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


def _unauthorized(message: str) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=401)


def _decode_jwt(token: str) -> Dict[str, Any]:
    decode_kwargs: Dict[str, Any] = {"algorithms": [settings.jwt_algorithm]}
    if settings.jwt_audience:
        decode_kwargs["audience"] = settings.jwt_audience
    if settings.jwt_issuer:
        decode_kwargs["issuer"] = settings.jwt_issuer
    return jwt.decode(token, settings.jwt_secret, **decode_kwargs)


async def enforce_authentication(request: Request, call_next):
    """Require Authorization header when auth settings are configured."""
    if not settings.jwt_secret and not settings.api_auth_token:
        return await call_next(request)

    auth_header = request.headers.get("authorization")
    if not auth_header:
        return _unauthorized("Missing Authorization header")

    scheme, _, token = auth_header.partition(" ")
    scheme = scheme.strip().lower()
    token = token.strip()

    if not token:
        return _unauthorized("Missing authorization token")

    # Try JWT first if configured
    if settings.jwt_secret and scheme == "bearer":
        try:
            payload = _decode_jwt(token)
            request.state.jwt_payload = payload
            request.state.auth_method = "jwt"
            return await call_next(request)
        except InvalidTokenError:
            if not settings.api_auth_token:
                return _unauthorized("Invalid or expired token")
        except Exception:
            if not settings.api_auth_token:
                return _unauthorized("Authorization failed")

    # Fallback to shared token if provided
    if settings.api_auth_token:
        expected = settings.api_auth_token
        candidate = token if scheme in {"bearer", "token"} else auth_header
        if secrets.compare_digest(candidate, expected):
            request.state.auth_method = "shared-token"
            return await call_next(request)

    # No configured scheme succeeded
    if settings.jwt_secret:
        return _unauthorized("Invalid or expired token")
    return _unauthorized("Invalid authentication token")
