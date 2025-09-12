from typing import List, Optional
import httpx
from sentence_transformers import CrossEncoder
from src.core.models import SearchResult
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class RerankingService:
    def __init__(self):
        self.jina_client = httpx.AsyncClient() if settings.jina_api_key else None
        self.local_reranker = None

    def _load_local_reranker(self):
        """Load local cross-encoder model as fallback."""
        if self.local_reranker is None:
            try:
                self.local_reranker = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2')
                logger.info("Loaded local cross-encoder reranker")
            except Exception as e:
                logger.error(f"Failed to load local reranker: {e}")

    async def rerank_jina(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using Jina API."""
        if not self.jina_client or not settings.jina_api_key:
            return results

        try:
            documents = [{"text": result.document.content} for result in results]
            
            response = await self.jina_client.post(
                f"https://api.jina.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {settings.jina_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": settings.reranker_model,
                    "query": query,
                    "documents": documents,
                    "top_n": len(results)
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                rerank_data = response.json()
                reranked_results = []
                
                for item in rerank_data["results"]:
                    original_result = results[item["index"]]
                    original_result.score = item["relevance_score"]
                    original_result.rank_explanation = {
                        **original_result.rank_explanation,
                        "rerank_method": "jina",
                        "rerank_score": item["relevance_score"]
                    }
                    reranked_results.append(original_result)
                
                return reranked_results
            else:
                logger.warning(f"Jina reranking failed: {response.status_code}")
                return await self.rerank_local(query, results)
                
        except Exception as e:
            logger.error(f"Jina reranking error: {e}")
            return await self.rerank_local(query, results)

    async def rerank_local(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using local cross-encoder."""
        if not results:
            return results

        try:
            self._load_local_reranker()
            if self.local_reranker is None:
                return results

            pairs = [[query, result.document.content] for result in results]
            scores = self.local_reranker.predict(pairs)
            
            # Update results with new scores
            for i, result in enumerate(results):
                result.score = float(scores[i])
                result.rank_explanation = {
                    **result.rank_explanation,
                    "rerank_method": "local",
                    "rerank_score": scores[i]
                }
            
            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Local reranking failed: {e}")
            return results

    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Main reranking method - tries Jina first, falls back to local."""
        if settings.jina_api_key:
            return await self.rerank_jina(query, results)
        else:
            return await self.rerank_local(query, results)