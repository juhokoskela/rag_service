from typing import List, Optional
import openai
import asyncio
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from src.core.config import settings
from src.repositories.embedding_cache_repository import EmbeddingCacheRepository
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, cache_repo: EmbeddingCacheRepository):
        self.cache_repo = cache_repo
        self.client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API with retries."""
        try:
            response = await self.client.embeddings.create(
                model=settings.embedding_model,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            if len(embedding) != settings.embedding_dim:
                raise ValueError(f"Expected embedding dimension {settings.embedding_dim}, got {len(embedding)}")
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for text with caching."""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Check cache first
        if use_cache:
            cached = await self.cache_repo.get_cached_embedding(text, settings.embedding_model)
            if cached:
                logger.debug(f"Using cached embedding for text: {text[:50]}...")
                return cached

        # Generate new embedding
        embedding = await self._generate_embedding(text)
        
        # Cache the result
        if use_cache:
            await self.cache_repo.cache_embedding(text, embedding, settings.embedding_model)

        return embedding

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        return await self.embed_text(query, use_cache=True)

    async def embed_documents(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        use_batch_api: bool = None,
        batch_threshold: int = 50
    ) -> List[List[float]]:
        """Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            use_batch_api: Force use of batch API (None = auto-decide based on size)
            batch_threshold: Minimum number of texts to use batch API
        """
        if not texts:
            return []

        # Decide whether to use batch API
        should_use_batch = (
            use_batch_api is True or 
            (use_batch_api is None and len(texts) >= batch_threshold)
        )
        
        if should_use_batch:
            logger.info(f"Using OpenAI Batch API for {len(texts)} texts (50% cost savings)")
            try:
                from src.services.batch_embedding import BatchEmbeddingService
                batch_service = BatchEmbeddingService(self.cache_repo)
                
                # Create batch job and wait for completion
                job_info = await batch_service.create_batch_embedding_job(texts, use_cache=use_cache)
                
                if job_info["status"] == "completed":
                    # All results were cached
                    return job_info["embeddings"]
                else:
                    # Wait for batch completion (this could take up to 24h)
                    logger.info(f"Batch job {job_info['job_id']} created, waiting for completion...")
                    embeddings = await batch_service.wait_for_batch_completion(job_info["job_id"])
                    return embeddings
                    
            except Exception as e:
                logger.error(f"Batch API failed, falling back to regular processing: {e}")
                # Fall through to regular processing

        # Regular processing for small batches or batch API failures
        logger.info(f"Using regular API for {len(texts)} texts")
        
        # Process in batches to avoid rate limits
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Create tasks for parallel processing
            tasks = [self.embed_text(text, use_cache) for text in batch]
            
            try:
                batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and handle exceptions
                for j, result in enumerate(batch_embeddings):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to embed document {i + j}: {result}")
                        # Use zero vector as fallback
                        all_embeddings.append([0.0] * settings.embedding_dim)
                    else:
                        all_embeddings.append(result)
                
                # Rate limiting between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Add zero vectors for failed batch
                for _ in batch:
                    all_embeddings.append([0.0] * settings.embedding_dim)

        return all_embeddings

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            a_np = np.array(a)
            b_np = np.array(b)
            
            # Calculate cosine similarity
            dot_product = np.dot(a_np, b_np)
            norm_a = np.linalg.norm(a_np)
            norm_b = np.linalg.norm(b_np)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0