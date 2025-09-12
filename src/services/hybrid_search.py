from typing import List, Dict, Any, Optional
import asyncio
from src.core.models import SearchResult
from src.services.vector_search import VectorSearchService
from src.services.bm25_search import BM25SearchService
from src.services.reranking import RerankingService
from src.services.embedding import EmbeddingService
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class HybridSearchService:
    def __init__(
        self,
        vector_service: VectorSearchService,
        bm25_service: BM25SearchService,
        reranking_service: RerankingService,
        embedding_service: EmbeddingService
    ):
        self.vector_service = vector_service
        self.bm25_service = bm25_service
        self.reranking_service = reranking_service
        self.embedding_service = embedding_service

    def _normalize_scores(self, results: List[SearchResult], method: str) -> List[SearchResult]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for result in results:
                result.score = 1.0
        else:
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results

    def _merge_results(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        vector_weight: float = None,
        bm25_weight: float = None
    ) -> List[SearchResult]:
        """Merge and score results from different search methods."""
        vector_weight = vector_weight or settings.vector_weight
        bm25_weight = bm25_weight or settings.bm25_weight

        # Normalize scores
        vector_results = self._normalize_scores(vector_results, "vector")
        bm25_results = self._normalize_scores(bm25_results, "bm25")

        # Create combined results dictionary
        combined = {}
        
        # Add vector results
        for result in vector_results:
            doc_id = result.document.id
            combined[doc_id] = {
                'document': result.document,
                'vector_score': result.score * vector_weight,
                'bm25_score': 0.0,
                'explanation': result.rank_explanation
            }
        
        # Add BM25 results
        for result in bm25_results:
            doc_id = result.document.id
            if doc_id in combined:
                combined[doc_id]['bm25_score'] = result.score * bm25_weight
                combined[doc_id]['explanation']['bm25'] = result.rank_explanation
            else:
                combined[doc_id] = {
                    'document': result.document,
                    'vector_score': 0.0,
                    'bm25_score': result.score * bm25_weight,
                    'explanation': result.rank_explanation
                }
        
        # Create final results
        final_results = []
        for doc_id, data in combined.items():
            total_score = data['vector_score'] + data['bm25_score']
            
            result = SearchResult(
                document=data['document'],
                score=total_score,
                rank_explanation={
                    'hybrid_score': total_score,
                    'vector_score': data['vector_score'],
                    'bm25_score': data['bm25_score'],
                    'weights': {'vector': vector_weight, 'bm25': bm25_weight},
                    **data['explanation']
                }
            )
            final_results.append(result)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results

    async def search(
        self,
        query: str,
        limit: int = 10,
        enable_reranking: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
        vector_weight: float = None,
        bm25_weight: float = None
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and BM25 methods."""
        try:
            # Use default weights if not specified
            vector_weight = vector_weight if vector_weight is not None else settings.vector_weight
            bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight
            
            vector_results = []
            bm25_results = []
            
            # Only run vector search if weight > 0
            if vector_weight > 0:
                query_embedding = await self.embedding_service.embed_query(query)
                vector_results = await self.vector_service.search(
                    query_embedding=query_embedding,
                    limit=limit * 2,  # Get more candidates for better fusion
                    metadata_filter=metadata_filter
                )
            
            # Only run BM25 search if weight > 0  
            if bm25_weight > 0:
                # Adjust BM25 limit based on available corpus size
                bm25_corpus_size = len(getattr(self.bm25_service, 'documents', []))
                bm25_limit = min(limit * 2, max(bm25_corpus_size, 1)) if bm25_corpus_size > 0 else limit
                
                bm25_results = await self.bm25_service.search(
                    query=query,
                    limit=bm25_limit,
                    metadata_filter=metadata_filter
                )
            
            # Handle case where only one search method is used
            if vector_weight > 0 and bm25_weight == 0:
                # Pure vector search - return vector results directly with proper scoring
                results = []
                for result in vector_results:
                    new_result = SearchResult(
                        document=result.document,
                        score=result.score,
                        rank_explanation={
                            'hybrid_score': result.score,
                            'vector_score': result.score,
                            'bm25_score': 0.0,
                            'weights': {'vector': vector_weight, 'bm25': bm25_weight},
                            'method': 'vector',
                            'similarity': result.rank_explanation.get('similarity', result.score)
                        }
                    )
                    results.append(new_result)
                merged_results = results
            elif bm25_weight > 0 and vector_weight == 0:
                # Pure BM25 search - return BM25 results directly with proper scoring
                results = []
                for result in bm25_results:
                    new_result = SearchResult(
                        document=result.document,
                        score=result.score,
                        rank_explanation={
                            'hybrid_score': result.score,
                            'vector_score': 0.0,
                            'bm25_score': result.score,
                            'weights': {'vector': vector_weight, 'bm25': bm25_weight},
                            'method': 'bm25',
                            'score': result.rank_explanation.get('score', result.score)
                        }
                    )
                    results.append(new_result)
                merged_results = results
            else:
                # Hybrid search - merge results from both methods
                merged_results = self._merge_results(
                    vector_results, bm25_results, vector_weight, bm25_weight
                )
            
            # Limit to requested number
            merged_results = merged_results[:limit * 2]  # Get candidates for reranking
            
            # Apply reranking if enabled
            if enable_reranking and merged_results:
                merged_results = await self.reranking_service.rerank(query, merged_results)
            
            # Final limit
            return merged_results[:limit]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise