from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List
import time
import logging

from src.core.models import SearchRequest, SearchResponse
from src.core.config import settings
from src.api.dependencies import get_hybrid_search_service
from src.services.hybrid_search import HybridSearchService

router = APIRouter()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

@router.post("/", response_model=SearchResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def search_documents(
    request: Request,
    search_request: SearchRequest,
    search_service: HybridSearchService = Depends(get_hybrid_search_service)
):
    """Search documents using hybrid vector + BM25 approach with reranking."""
    start_time = time.time()
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        logger.info(f"Search request: '{search_request.query}' [{correlation_id}]")
        
        # Perform hybrid search
        results = await search_service.search(
            query=search_request.query,
            limit=search_request.limit,
            enable_reranking=True
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = SearchResponse(
            results=results,
            query=search_request.query,
            total_results=len(results),
            processing_time_ms=round(processing_time_ms, 2)
        )
        
        logger.info(
            f"Search completed: {len(results)} results in {processing_time_ms:.2f}ms [{correlation_id}]"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e} [{correlation_id}]")
        raise HTTPException(
            status_code=500,
            detail=f"Search operation failed: {str(e)}"
        )

@router.post("/vector", response_model=SearchResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def vector_search(
    request: Request,
    search_request: SearchRequest,
    search_service: HybridSearchService = Depends(get_hybrid_search_service)
):
    """Pure vector similarity search (no BM25)."""
    start_time = time.time()
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        logger.info(f"Vector search request: '{search_request.query}' [{correlation_id}]")
        
        # Vector-only search with weight 1.0 for vector, 0.0 for BM25
        results = await search_service.search(
            query=search_request.query,
            limit=search_request.limit,
            vector_weight=1.0,
            bm25_weight=0.0,
            enable_reranking=False  # Skip reranking for pure vector search
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = SearchResponse(
            results=results,
            query=search_request.query,
            total_results=len(results),
            processing_time_ms=round(processing_time_ms, 2)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Vector search failed: {e} [{correlation_id}]")
        raise HTTPException(
            status_code=500,
            detail=f"Vector search operation failed: {str(e)}"
        )

@router.post("/bm25", response_model=SearchResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def bm25_search(
    request: Request,
    search_request: SearchRequest,
    search_service: HybridSearchService = Depends(get_hybrid_search_service)
):
    """Pure BM25 full-text search (no vector similarity)."""
    start_time = time.time()
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        logger.info(f"BM25 search request: '{search_request.query}' [{correlation_id}]")
        
        # BM25-only search with weight 0.0 for vector, 1.0 for BM25
        results = await search_service.search(
            query=search_request.query,
            limit=search_request.limit,
            vector_weight=0.0,
            bm25_weight=1.0,
            enable_reranking=False
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = SearchResponse(
            results=results,
            query=search_request.query,
            total_results=len(results),
            processing_time_ms=round(processing_time_ms, 2)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"BM25 search failed: {e} [{correlation_id}]")
        raise HTTPException(
            status_code=500,
            detail=f"BM25 search operation failed: {str(e)}"
        )