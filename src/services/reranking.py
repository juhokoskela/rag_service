from typing import List, Optional, Tuple
import asyncio
import httpx
from httpx import Limits, Timeout
from sentence_transformers import CrossEncoder
from src.core.models import SearchResult
from src.core.config import settings
import logging

try:  # Optional dependency shipped with httpx[http2]
    import h2  # type: ignore  # noqa: F401
    _HTTP2_ENABLED = True
except ImportError:  # pragma: no cover - environment specific
    _HTTP2_ENABLED = False

logger = logging.getLogger(__name__)

class RerankingService:
    def __init__(self):
        self.jina_client = None
        self._jina_headers = None
        self.local_reranker = None

        if settings.jina_api_key:
            self.jina_client = httpx.AsyncClient(
                base_url="https://api.jina.ai",
                headers={
                    "Authorization": f"Bearer {settings.jina_api_key}",
                    "Accept": "application/json",
                },
                timeout=Timeout(connect=3.0, read=25.0, write=10.0, pool=None),
                limits=Limits(max_connections=10, max_keepalive_connections=5),
                http2=_HTTP2_ENABLED,
            )
            self._jina_headers = {"Content-Type": "application/json"}

            if not _HTTP2_ENABLED:
                logger.debug("HTTP/2 support unavailable; falling back to HTTP/1.1 for Jina client")

    def _load_local_reranker(self):
        """Load local cross-encoder model as fallback."""
        if self.local_reranker is None:
            try:
                self.local_reranker = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2')
                logger.info("Loaded local cross-encoder reranker")
            except Exception as e:
                logger.error(f"Failed to load local reranker: {e}")

    def _split_candidates(self, results: List[SearchResult]) -> Tuple[List[SearchResult], List[SearchResult]]:
        max_candidates = settings.rerank_top_k or 0
        if max_candidates and max_candidates < len(results):
            return results[:max_candidates], results[max_candidates:]
        return results, []

    def _truncate_content(self, content: str) -> str:
        max_chars = max(settings.rerank_max_chars, 0)
        if max_chars and len(content) > max_chars:
            return content[:max_chars]
        return content

    async def rerank_jina(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using Jina API."""
        if not self.jina_client or not settings.jina_api_key:
            return results

        try:
            candidates, remainder = self._split_candidates(results)
            if not candidates:
                return results

            documents = [
                {"text": self._truncate_content(result.document.content or "")}
                for result in candidates
            ]

            response = await self.jina_client.post(
                "/v1/rerank",
                headers=self._jina_headers,
                json={
                    "model": settings.reranker_model,
                    "query": query,
                    "documents": documents,
                    "top_n": len(candidates)
                },
                timeout=30.0
            )

            if response.status_code == 200:
                rerank_data = response.json()

                for item in rerank_data.get("results", []):
                    candidate = candidates[item["index"]]
                    rerank_score = item["relevance_score"]
                    candidate.score = rerank_score
                    candidate.rank_explanation = {
                        **(candidate.rank_explanation or {}),
                        "rerank_method": "jina",
                        "rerank_score": rerank_score
                    }

                candidates.sort(key=lambda x: x.score, reverse=True)

                return candidates + remainder
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

            candidates, remainder = self._split_candidates(results)
            if not candidates:
                return results

            pairs = [
                [query, self._truncate_content(result.document.content or "")]
                for result in candidates
            ]
            scores = await asyncio.to_thread(self.local_reranker.predict, pairs)

            # Update results with new scores
            for i, result in enumerate(candidates):
                rerank_score = float(scores[i])  # Cast numpy scalar to native float
                result.score = rerank_score
                result.rank_explanation = {
                    **(result.rank_explanation or {}),
                    "rerank_method": "local",
                    "rerank_score": rerank_score
                }

            # Sort by new scores
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates + remainder

        except Exception as e:
            logger.error(f"Local reranking failed: {e}")
            return results

    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Main reranking method - tries Jina first, falls back to local."""
        if settings.jina_api_key:
            return await self.rerank_jina(query, results)
        else:
            return await self.rerank_local(query, results)
