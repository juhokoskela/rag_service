from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import uuid
import logging
import json
import asyncio

if TYPE_CHECKING:
    from src.services.batch_embedding import BatchEmbeddingService

from src.core.models import Document, DocumentCreateRequest, DocumentUpdateRequest
from src.core.config import settings
from src.api.dependencies import get_ingestion_service, get_document_repository
from src.services.ingestion import IngestionService
from src.repositories.document_repository import DocumentRepository

router = APIRouter()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)


async def poll_and_complete_batch(
    job_id: str,
    original_documents: List[Dict[str, Any]],
    batch_service,  # BatchEmbeddingService instance
    correlation_id: str = "unknown"
):
    """Background task to poll batch status and complete automatically."""
    try:
        from src.api.dependencies import get_ingestion_service

        ingestion_service = get_ingestion_service()

        logger.info(f"Starting background polling for batch job {job_id} [{correlation_id}]")

        max_attempts = 480  # 4 hours with 30s intervals
        for attempt in range(max_attempts):
            try:
                status_info = await batch_service.get_batch_status(job_id)

                if status_info["status"] == "completed":
                    try:
                        embeddings = await batch_service.get_batch_results(job_id)

                        # Validate embeddings
                        if not embeddings or len(embeddings) != len(original_documents):
                            logger.error(f"Embedding count mismatch for job {job_id}: got {len(embeddings) if embeddings else 0}, expected {len(original_documents)} [{correlation_id}]")
                            return

                        # Store documents with embeddings in database
                        created_docs = await ingestion_service.ingest_documents_with_embeddings(
                            original_documents, embeddings
                        )

                        logger.info(f"Auto-completed batch job {job_id}: stored {len(created_docs)} documents [{correlation_id}]")

                        # Mark job as completed in batch service
                        job_info = batch_service._active_batches.get(job_id, {})
                        job_info["status"] = "completed"
                        job_info["documents_created"] = len(created_docs)
                        job_info["document_ids"] = [str(doc.id) for doc in created_docs]

                        return

                    except Exception as e:
                        logger.error(f"Failed to process completed batch job {job_id}: {e} [{correlation_id}]")
                        # Mark as failed so we don't keep retrying
                        job_info = batch_service._active_batches.get(job_id, {})
                        job_info["status"] = "processing_failed"
                        job_info["error"] = str(e)
                        return

                elif status_info["status"] in ["failed", "expired", "cancelled"]:
                    logger.error(f"Batch job {job_id} failed with status: {status_info['status']} [{correlation_id}]")
                    return

                # Still processing, wait before next poll
                if attempt % 10 == 0:  # Log every 5 minutes
                    logger.info(f"Batch job {job_id} still processing (attempt {attempt + 1}/{max_attempts}) [{correlation_id}]")

            except Exception as e:
                logger.error(f"Error polling batch job {job_id} (attempt {attempt + 1}): {e} [{correlation_id}]")

            await asyncio.sleep(30)  # Wait 30 seconds before next poll

        logger.error(f"Batch job {job_id} timed out after {max_attempts * 30} seconds [{correlation_id}]")

    except Exception as e:
        logger.error(f"Critical error in batch polling for job {job_id}: {e} [{correlation_id}]")

@router.post("/", response_model=List[Document], status_code=201)
@limiter.limit("10/minute")
async def create_document(
    request: Request,
    doc_request: DocumentCreateRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """Ingest a new document with automatic chunking and embedding generation."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        logger.info(f"Creating document [{correlation_id}]")
        
        documents = await ingestion_service.ingest_document(
            content=doc_request.content,
            metadata=doc_request.metadata,
            enable_chunking=True
        )
        
        logger.info(f"Created {len(documents)} document chunks [{correlation_id}]")
        return documents
        
    except ValueError as e:
        logger.error(f"Document validation failed: {e} [{correlation_id}]")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Document creation failed: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to create document")

@router.get("/", response_model=List[Document])
async def list_documents(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    doc_repo: DocumentRepository = Depends(get_document_repository)
):
    """List documents with pagination."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        documents = await doc_repo.get_all_documents(limit=limit)
        logger.info(f"Retrieved {len(documents)} documents [{correlation_id}]")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@router.get("/{document_id}", response_model=Document)
async def get_document(
    document_id: uuid.UUID,
    request: Request,
    doc_repo: DocumentRepository = Depends(get_document_repository)
):
    """Retrieve a specific document by ID."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        document = await doc_repo.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")

@router.put("/{document_id}", response_model=Document)
@limiter.limit("10/minute")
async def update_document(
    document_id: uuid.UUID,
    doc_request: DocumentUpdateRequest,
    request: Request,
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """Update an existing document."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        updated_doc = await ingestion_service.update_document(
            document_id=document_id,
            content=doc_request.content,
            metadata=doc_request.metadata,
            regenerate_embedding=doc_request.content is not None
        )
        
        if not updated_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Updated document {document_id} [{correlation_id}]")
        return updated_doc
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update document {document_id}: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to update document")

@router.delete("/{document_id}", status_code=204)
@limiter.limit("10/minute")
async def delete_document(
    document_id: uuid.UUID,
    request: Request,
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """Delete a document."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        success = await ingestion_service.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Deleted document {document_id} [{correlation_id}]")
        return  # 204 No Content - no response body needed
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@router.post("/upload", response_model=List[Document], status_code=201)
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """Upload a text file as a document with automatic chunking and embedding generation."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('text/'):
            if not file.filename or not file.filename.endswith(('.txt', '.md', '.rst', '.py', '.js', '.html', '.xml', '.json', '.csv')):
                raise HTTPException(status_code=400, detail="Only text files are supported")
        
        # Read file content
        content_bytes = await file.read()
        try:
            content = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
        
        if len(content.strip()) == 0:
            raise HTTPException(status_code=400, detail="File cannot be empty")
        
        # Parse metadata
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata field")
        
        # Add file info to metadata
        parsed_metadata.update({
            "filename": file.filename,
            "content_type": file.content_type,
            "source": "upload"
        })
        
        logger.info(f"Uploading file {file.filename} ({len(content)} characters) [{correlation_id}]")
        
        documents = await ingestion_service.ingest_document(
            content=content,
            metadata=parsed_metadata,
            enable_chunking=True
        )
        
        logger.info(f"Created {len(documents)} document chunks from {file.filename} [{correlation_id}]")
        return documents
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to upload document")

@router.post("/batch-job", status_code=202)
@limiter.limit("2/minute")
async def create_batch_job(
    request: Request,
    background_tasks: BackgroundTasks,
    doc_requests: List[DocumentCreateRequest],
    use_batch_api: bool = True,
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """Create a batch processing job for multiple documents using OpenAI Batch API."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')

    if len(doc_requests) < 10:
        raise HTTPException(
            status_code=400,
            detail="Batch jobs require at least 10 documents. Use regular endpoints for smaller batches."
        )

    if len(doc_requests) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 1000 documents per job."
        )

    try:
        logger.info(f"Creating batch job for {len(doc_requests)} documents [{correlation_id}]")

        from src.services.chunking import ChunkingService
        from src.services.batch_embedding import BatchEmbeddingService
        from src.api.dependencies import get_embedding_cache_repository

        chunking_service = ChunkingService()
        cache_repo = get_embedding_cache_repository()
        batch_service = BatchEmbeddingService(cache_repo)

        # Chunk all documents consistently (single pass with metadata)
        all_texts = []
        chunk_documents = []

        for doc_request in doc_requests:
            chunks = chunking_service.chunk_text(doc_request.content, doc_request.metadata)
            for chunk in chunks:
                all_texts.append(chunk.content)
                chunk_documents.append({
                    "content": chunk.content,
                    "metadata": chunk.metadata
                })

        # Validate minimum chunk requirement for batch API
        if use_batch_api and len(all_texts) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Batch API requires at least 50 chunks, got {len(all_texts)}. Use regular endpoints instead."
            )

        # Create batch embedding job
        job_info = await batch_service.create_batch_embedding_job(all_texts)

        # Store original documents in the batch service for later retrieval
        if job_info["job_id"] in batch_service._active_batches:
            batch_service._active_batches[job_info["job_id"]]["original_documents"] = chunk_documents

        # If job is already completed (all cached), process immediately
        if job_info["status"] == "completed":
            original_docs = job_info.get('original_documents', chunk_documents)
            embeddings = job_info.get('embeddings', [])

            if embeddings:
                created_docs = await ingestion_service.ingest_documents_with_embeddings(
                    original_docs, embeddings
                )
                logger.info(f"Immediately completed cached batch job {job_info['job_id']}: stored {len(created_docs)} documents [{correlation_id}]")

                return {
                    "job_id": job_info["job_id"],
                    "status": "completed",
                    "total_documents": len(doc_requests),
                    "total_chunks": len(all_texts),
                    "cached_chunks": job_info["cached_requests"],
                    "uncached_chunks": job_info["uncached_requests"],
                    "documents_created": len(created_docs),
                    "document_ids": [str(doc.id) for doc in created_docs],
                    "cost_savings": "N/A (all cached)"
                }

        # Start background polling for non-completed jobs
        background_tasks.add_task(
            poll_and_complete_batch,
            job_info["job_id"],
            chunk_documents,
            batch_service,
            correlation_id
        )

        job_response = {
            "job_id": job_info["job_id"],
            "status": "processing",
            "total_documents": len(doc_requests),
            "total_chunks": len(all_texts),
            "cached_chunks": job_info["cached_requests"],
            "uncached_chunks": job_info["uncached_requests"],
            "estimated_completion": job_info.get("estimated_completion"),
            "batch_id": job_info.get("batch_id"),
            "cost_savings": "50% compared to regular API",
            "message": "Documents will be automatically stored when batch processing completes. Check status with GET /batch-job/{job_id}"
        }

        logger.info(f"Created batch job {job_info['job_id']} for {len(all_texts)} chunks, started background polling [{correlation_id}]")

        return job_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch job creation failed: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to create batch job")

@router.post("/batch-job/{job_id}/complete", operation_id="complete_batch_job")
async def complete_batch_job(
    job_id: str,
    request: Request,
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """Complete a batch job by storing documents with their embeddings."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        from src.services.batch_embedding import BatchEmbeddingService
        from src.api.dependencies import get_embedding_cache_repository
        
        cache_repo = get_embedding_cache_repository()
        batch_service = BatchEmbeddingService(cache_repo)
        
        # Check if job is completed
        status_info = await batch_service.get_batch_status(job_id)
        if status_info["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Batch job is not completed yet. Current status: {status_info['status']}"
            )
        
        # Get embeddings and original documents 
        job_info = batch_service._active_batches.get(job_id)
        if not job_info:
            raise HTTPException(
                status_code=404,
                detail="Batch job data not found. Job may have been cleaned up."
            )
        
        embeddings = await batch_service.get_batch_results(job_id)
        original_docs = job_info.get('original_documents', [])
        
        if not original_docs:
            raise HTTPException(
                status_code=400,
                detail="Original document data not found for batch job"
            )
        
        # Store documents with embeddings
        created_docs = await ingestion_service.ingest_documents_with_embeddings(
            original_docs, embeddings
        )
        
        logger.info(f"Completed batch job {job_id}: stored {len(created_docs)} documents [{correlation_id}]")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "documents_created": len(created_docs),
            "document_ids": [str(doc.id) for doc in created_docs]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete batch job {job_id}: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to complete batch job")

@router.get("/batch-job/{job_id}")
async def get_batch_job_status(
    job_id: str,
    request: Request
):
    """Get the status of a batch processing job."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        from src.services.batch_embedding import BatchEmbeddingService
        from src.api.dependencies import get_embedding_cache_repository
        
        cache_repo = get_embedding_cache_repository()
        batch_service = BatchEmbeddingService(cache_repo)
        
        status_info = await batch_service.get_batch_status(job_id)

        # Check if documents have been created by the background task
        job_info = batch_service._active_batches.get(job_id, {})
        documents_created = job_info.get("documents_created", 0)
        document_ids = job_info.get("document_ids", [])

        response = {
            "job_id": job_id,
            "status": status_info["status"],
            "progress": {
                "total_requests": status_info["total_requests"],
                "cached_requests": status_info["cached_requests"],
                "uncached_requests": status_info["uncached_requests"],
                "completed": status_info.get("request_counts", {}).get("completed", 0),
                "failed": status_info.get("request_counts", {}).get("failed", 0)
            },
            "timestamps": {
                "created_at": status_info.get("created_at"),
                "completed_at": status_info.get("completed_at"),
                "failed_at": status_info.get("failed_at")
            }
        }

        # Add document creation info if documents have been stored
        if documents_created > 0:
            response["documents"] = {
                "created": documents_created,
                "document_ids": document_ids,
                "message": "Documents have been automatically stored in the database"
            }
        elif job_info.get("status") == "processing_failed":
            response["documents"] = {
                "message": "Automatic processing failed. Try manually completing with POST /batch-job/{job_id}/complete",
                "error": job_info.get("error", "Unknown processing error")
            }
        elif status_info["status"] == "completed":
            response["documents"] = {
                "message": "Batch completed but documents not yet stored. Automatic processing may still be running."
            }

        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get batch job status: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to get batch job status")