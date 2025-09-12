from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Optional
import uuid
import logging
import json

from src.core.models import Document, DocumentCreateRequest, DocumentUpdateRequest
from src.core.config import settings
from src.api.dependencies import get_ingestion_service, get_document_repository
from src.services.ingestion import IngestionService
from src.repositories.document_repository import DocumentRepository

router = APIRouter()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

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
        
        # Extract all text content for batch embedding
        all_texts = []
        for doc_request in doc_requests:
            # Use chunking service to split documents
            from src.services.chunking import ChunkingService
            chunking_service = ChunkingService()
            chunks = chunking_service.chunk_text(doc_request.content)
            all_texts.extend([chunk.content for chunk in chunks])
        
        # Create batch embedding job
        if use_batch_api and len(all_texts) >= 50:
            from src.services.batch_embedding import BatchEmbeddingService
            from src.api.dependencies import get_embedding_cache_repository
            
            cache_repo = get_embedding_cache_repository()
            batch_service = BatchEmbeddingService(cache_repo)
            
            job_info = await batch_service.create_batch_embedding_job(all_texts)
            
            # Store job info for later processing
            # In production, store this in database/Redis
            job_response = {
                "job_id": job_info["job_id"],
                "status": "processing" if job_info["status"] != "completed" else "completed", 
                "total_documents": len(doc_requests),
                "total_chunks": len(all_texts),
                "cached_chunks": job_info["cached_requests"],
                "uncached_chunks": job_info["uncached_requests"],
                "estimated_completion": job_info.get("estimated_completion"),
                "batch_id": job_info.get("batch_id"),
                "cost_savings": "50% compared to regular API" if job_info.get("batch_id") else "N/A (all cached)"
            }
            
            logger.info(f"Created batch job {job_info['job_id']} for {len(all_texts)} chunks [{correlation_id}]")
            
            return job_response
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Batch API requires at least 50 chunks, got {len(all_texts)}. Use regular endpoints instead."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch job creation failed: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to create batch job")

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
        
        return {
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
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get batch job status: {e} [{correlation_id}]")
        raise HTTPException(status_code=500, detail="Failed to get batch job status")