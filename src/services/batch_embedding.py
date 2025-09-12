from typing import List, Dict, Optional, Any
import openai
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
import tempfile
import os
from pathlib import Path

from src.core.config import settings
from src.repositories.embedding_cache_repository import EmbeddingCacheRepository
import logging

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress" 
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class BatchEmbeddingService:
    """Service for handling large-scale embedding generation using OpenAI's Batch API."""
    
    def __init__(self, cache_repo: EmbeddingCacheRepository):
        self.cache_repo = cache_repo
        self.client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout
        )
        # Store active batches in memory (in production, use Redis or DB)
        self._active_batches: Dict[str, Dict[str, Any]] = {}

    async def create_batch_embedding_job(
        self,
        texts: List[str],
        job_id: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Create a batch embedding job using OpenAI's Batch API."""
        
        if not texts:
            raise ValueError("No texts provided for batch embedding")
        
        if len(texts) > 50000:  # OpenAI batch limit
            raise ValueError("Batch size exceeds OpenAI limit of 50,000 requests")
        
        job_id = job_id or str(uuid.uuid4())
        
        # Filter out texts that are already cached
        uncached_texts = []
        cached_results = {}
        text_to_index = {}
        
        if use_cache:
            for i, text in enumerate(texts):
                cached = await self.cache_repo.get_cached_embedding(text, settings.embedding_model)
                if cached:
                    cached_results[i] = cached
                    logger.debug(f"Using cached embedding for text {i}")
                else:
                    text_index = len(uncached_texts)
                    uncached_texts.append(text)
                    text_to_index[text_index] = i  # Map batch index to original index
        else:
            uncached_texts = texts
            text_to_index = {i: i for i in range(len(texts))}
        
        if not uncached_texts:
            logger.info(f"All {len(texts)} texts were cached, no batch job needed")
            return {
                "job_id": job_id,
                "status": "completed",
                "total_requests": len(texts),
                "cached_requests": len(texts),
                "uncached_requests": 0,
                "batch_id": None,
                "embeddings": [cached_results[i] for i in range(len(texts))]
            }
        
        logger.info(f"Creating batch job for {len(uncached_texts)} uncached texts ({len(cached_results)} cached)")
        
        try:
            # Create batch file
            batch_requests = []
            for i, text in enumerate(uncached_texts):
                request = {
                    "custom_id": f"{job_id}_{i}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": settings.embedding_model,
                        "input": text,
                        "encoding_format": "float"
                    }
                }
                batch_requests.append(request)
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for request in batch_requests:
                    f.write(json.dumps(request) + '\n')
                batch_file_path = f.name
            
            # Upload batch file
            with open(batch_file_path, 'rb') as f:
                batch_file = await self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            # Clean up temp file
            os.unlink(batch_file_path)
            
            # Create batch job
            batch_job = await self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={"job_id": job_id}
            )
            
            # Store job info
            job_info = {
                "job_id": job_id,
                "batch_id": batch_job.id,
                "file_id": batch_file.id,
                "status": batch_job.status,
                "created_at": datetime.now(),
                "total_requests": len(texts),
                "cached_requests": len(cached_results),
                "uncached_requests": len(uncached_texts),
                "text_to_index": text_to_index,
                "cached_results": cached_results,
                "use_cache": use_cache
            }
            
            self._active_batches[job_id] = job_info
            
            logger.info(f"Created batch job {job_id} with batch_id {batch_job.id}")
            
            return {
                "job_id": job_id,
                "batch_id": batch_job.id,
                "status": batch_job.status,
                "total_requests": len(texts),
                "cached_requests": len(cached_results),
                "uncached_requests": len(uncached_texts),
                "estimated_completion": datetime.now() + timedelta(hours=24)
            }
            
        except Exception as e:
            logger.error(f"Failed to create batch embedding job: {e}")
            raise

    async def get_batch_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a batch embedding job."""
        
        if job_id not in self._active_batches:
            # Try to find the batch by checking recent batches from OpenAI
            # This is a fallback for when service restarts and loses in-memory state
            try:
                batches = await self.client.batches.list(limit=100)
                for batch in batches.data:
                    if (batch.metadata and 
                        batch.metadata.get("job_id") == job_id):
                        # Reconstruct minimal job info
                        self._active_batches[job_id] = {
                            "job_id": job_id,
                            "batch_id": batch.id,
                            "status": batch.status,
                            "total_requests": batch.request_counts.total if batch.request_counts else 0,
                            "cached_requests": 0,  # Unknown after restart
                            "uncached_requests": batch.request_counts.total if batch.request_counts else 0,
                            "created_at": batch.created_at
                        }
                        break
                else:
                    raise ValueError(f"Job {job_id} not found in active batches or recent OpenAI batches")
            except Exception as e:
                logger.error(f"Failed to find batch job {job_id}: {e}")
                raise ValueError(f"Job {job_id} not found")
        
        job_info = self._active_batches[job_id]
        
        if job_info["status"] == "completed":
            return {
                "job_id": job_id,
                "status": "completed",
                "total_requests": job_info["total_requests"],
                "cached_requests": job_info["cached_requests"],
                "uncached_requests": job_info["uncached_requests"]
            }
        
        try:
            # Get batch status from OpenAI
            batch = await self.client.batches.retrieve(job_info["batch_id"])
            
            # Update stored status
            job_info["status"] = batch.status
            
            return {
                "job_id": job_id,
                "batch_id": batch.id,
                "status": batch.status,
                "total_requests": job_info["total_requests"],
                "cached_requests": job_info["cached_requests"],
                "uncached_requests": job_info["uncached_requests"],
                "request_counts": {
                    "total": batch.request_counts.total if batch.request_counts else 0,
                    "completed": batch.request_counts.completed if batch.request_counts else 0,
                    "failed": batch.request_counts.failed if batch.request_counts else 0
                },
                "created_at": job_info["created_at"],
                "completed_at": batch.completed_at,
                "failed_at": batch.failed_at,
                "expired_at": batch.expired_at
            }
            
        except Exception as e:
            logger.error(f"Failed to get batch status for job {job_id}: {e}")
            raise

    async def get_batch_results(self, job_id: str) -> List[List[float]]:
        """Get the results from a completed batch embedding job."""
        
        if job_id not in self._active_batches:
            raise ValueError(f"Job {job_id} not found")
        
        job_info = self._active_batches[job_id]
        
        # If job was completed using only cache
        if job_info["uncached_requests"] == 0:
            embeddings = [job_info["cached_results"][i] for i in range(job_info["total_requests"])]
            return embeddings
        
        try:
            # Get batch status
            batch = await self.client.batches.retrieve(job_info["batch_id"])
            
            if batch.status != "completed":
                raise ValueError(f"Batch job {job_id} is not completed (status: {batch.status})")
            
            if not batch.output_file_id:
                raise ValueError(f"Batch job {job_id} has no output file")
            
            # Download results
            result_file = await self.client.files.content(batch.output_file_id)
            result_content = result_file.read().decode('utf-8')
            
            # Parse results
            batch_results = {}
            for line in result_content.strip().split('\n'):
                if line:
                    result = json.loads(line)
                    custom_id = result["custom_id"]
                    batch_index = int(custom_id.split('_')[1])
                    
                    if result.get("response") and result["response"].get("body"):
                        embedding = result["response"]["body"]["data"][0]["embedding"]
                        batch_results[batch_index] = embedding
                        
                        # Cache the result
                        if job_info["use_cache"]:
                            original_index = job_info["text_to_index"][batch_index]
                            # We don't have the original text here, but we could store it in job_info if needed
                    else:
                        logger.error(f"Failed result for custom_id {custom_id}: {result.get('error')}")
                        # Use zero vector as fallback
                        batch_results[batch_index] = [0.0] * settings.embedding_dim
            
            # Combine cached and batch results in original order
            embeddings = []
            for i in range(job_info["total_requests"]):
                if i in job_info["cached_results"]:
                    embeddings.append(job_info["cached_results"][i])
                else:
                    # Find the batch index for this original index
                    batch_index = None
                    for b_idx, orig_idx in job_info["text_to_index"].items():
                        if orig_idx == i:
                            batch_index = b_idx
                            break
                    
                    if batch_index is not None and batch_index in batch_results:
                        embeddings.append(batch_results[batch_index])
                    else:
                        logger.error(f"No result found for text index {i}")
                        embeddings.append([0.0] * settings.embedding_dim)
            
            # Update job status
            job_info["status"] = "completed"
            
            logger.info(f"Retrieved {len(embeddings)} embeddings for job {job_id}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get batch results for job {job_id}: {e}")
            raise

    async def wait_for_batch_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        max_wait_time: int = 86400  # 24 hours
    ) -> List[List[float]]:
        """Wait for a batch job to complete and return results."""
        
        start_time = datetime.now()
        max_wait_delta = timedelta(seconds=max_wait_time)
        
        while datetime.now() - start_time < max_wait_delta:
            status = await self.get_batch_status(job_id)
            
            if status["status"] == "completed":
                logger.info(f"Batch job {job_id} completed")
                return await self.get_batch_results(job_id)
            elif status["status"] in ["failed", "expired", "cancelled"]:
                raise RuntimeError(f"Batch job {job_id} failed with status: {status['status']}")
            
            logger.info(f"Batch job {job_id} status: {status['status']}, waiting {poll_interval}s...")
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"Batch job {job_id} did not complete within {max_wait_time} seconds")

    async def cancel_batch_job(self, job_id: str) -> bool:
        """Cancel a batch embedding job."""
        
        if job_id not in self._active_batches:
            raise ValueError(f"Job {job_id} not found")
        
        job_info = self._active_batches[job_id]
        
        try:
            batch = await self.client.batches.cancel(job_info["batch_id"])
            job_info["status"] = batch.status
            logger.info(f"Cancelled batch job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel batch job {job_id}: {e}")
            return False

    def cleanup_completed_jobs(self, max_age_hours: int = 48):
        """Clean up completed jobs older than max_age_hours."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        jobs_to_remove = []
        for job_id, job_info in self._active_batches.items():
            if (job_info["status"] == "completed" and 
                job_info["created_at"] < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self._active_batches[job_id]
            logger.info(f"Cleaned up completed job {job_id}")
        
        return len(jobs_to_remove)