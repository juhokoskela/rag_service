"""FastAPI dependency injection for services."""

from functools import lru_cache
from src.services.embedding import EmbeddingService
from src.services.chunking import ChunkingService
from src.services.ingestion import IngestionService
from src.services.vector_search import VectorSearchService
from src.services.bm25_search import BM25SearchService
from src.services.reranking import RerankingService
from src.services.hybrid_search import HybridSearchService
from src.repositories.document_repository import DocumentRepository
from src.repositories.embedding_cache_repository import EmbeddingCacheRepository

# Repositories
@lru_cache()
def get_document_repository() -> DocumentRepository:
    return DocumentRepository()

@lru_cache()
def get_embedding_cache_repository() -> EmbeddingCacheRepository:
    return EmbeddingCacheRepository()

# Core services
@lru_cache()
def get_embedding_service() -> EmbeddingService:
    cache_repo = get_embedding_cache_repository()
    return EmbeddingService(cache_repo)

@lru_cache()
def get_chunking_service() -> ChunkingService:
    return ChunkingService()

@lru_cache()
def get_ingestion_service() -> IngestionService:
    doc_repo = get_document_repository()
    embedding_service = get_embedding_service()
    chunking_service = get_chunking_service()
    return IngestionService(doc_repo, embedding_service, chunking_service)

# Search services
@lru_cache()
def get_vector_search_service() -> VectorSearchService:
    doc_repo = get_document_repository()
    return VectorSearchService(doc_repo)

@lru_cache()
def get_bm25_search_service() -> BM25SearchService:
    doc_repo = get_document_repository()
    return BM25SearchService(doc_repo)

@lru_cache()
def get_reranking_service() -> RerankingService:
    return RerankingService()

@lru_cache()
def get_hybrid_search_service() -> HybridSearchService:
    vector_service = get_vector_search_service()
    bm25_service = get_bm25_search_service()
    reranking_service = get_reranking_service()
    embedding_service = get_embedding_service()
    
    return HybridSearchService(
        vector_service,
        bm25_service, 
        reranking_service,
        embedding_service
    )