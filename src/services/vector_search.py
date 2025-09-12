from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from src.core.models import Document, SearchResult
from src.repositories.document_repository import DocumentRepository
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class VectorSearchService:
    def __init__(self, document_repo: DocumentRepository):
        self.document_repo = document_repo

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform vector similarity search using cosine similarity."""
        try:
            documents = await self.document_repo.vector_search(
                embedding=query_embedding,
                limit=limit,
                threshold=threshold,
                metadata_filter=metadata_filter
            )
            
            results = []
            for doc, score in documents:
                result = SearchResult(
                    document=doc,
                    score=float(score),
                    rank_explanation={"method": "vector", "similarity": score}
                )
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise