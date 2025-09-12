from typing import List, Dict, Any, Optional, Union
import asyncio
import uuid
from datetime import datetime
from src.core.models import Document, DocumentCreateRequest
from src.services.embedding import EmbeddingService
from src.services.chunking import ChunkingService
from src.repositories.document_repository import DocumentRepository
from src.infrastructure.redis import redis_cache
import logging

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(
        self,
        document_repo: DocumentRepository,
        embedding_service: EmbeddingService,
        chunking_service: ChunkingService
    ):
        self.document_repo = document_repo
        self.embedding_service = embedding_service
        self.chunking_service = chunking_service

    async def ingest_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        enable_chunking: bool = True
    ) -> List[Document]:
        """Ingest a single document, optionally splitting into chunks."""
        try:
            # Validate content
            if not content or not content.strip():
                raise ValueError("Document content cannot be empty")

            documents_to_create = []
            
            if enable_chunking:
                # Split into chunks
                chunks = self.chunking_service.chunk_text(content, metadata)
                
                if not chunks:
                    raise ValueError("No valid chunks created from document")
                
                # Merge small chunks if necessary
                chunks = self.chunking_service.merge_small_chunks(chunks)
                documents_to_create = chunks
            else:
                # Create single document
                doc = Document(
                    id=uuid.uuid4(),
                    content=content.strip(),
                    metadata=metadata or {}
                )
                documents_to_create = [doc]

            # Generate embeddings for all documents
            texts = [doc.content for doc in documents_to_create]
            embeddings = await self.embedding_service.embed_documents(texts)

            # Create documents with embeddings
            created_documents = []
            for doc, embedding in zip(documents_to_create, embeddings):
                # Add embedding to document (we'll handle this in the repo)
                setattr(doc, 'embedding', embedding)
                
                # Save to database
                created_doc = await self.document_repo.create_document(doc)
                created_documents.append(created_doc)

            # Invalidate search caches
            await redis_cache.invalidate_search_cache()

            logger.info(f"Successfully ingested document into {len(created_documents)} chunks")
            return created_documents

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise

    async def ingest_documents_with_embeddings(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[Document]:
        """Ingest documents that already have computed embeddings."""
        if len(documents) != len(embeddings):
            raise ValueError(f"Documents count ({len(documents)}) must match embeddings count ({len(embeddings)})")
        
        created_documents = []
        
        try:
            for doc_data, embedding in zip(documents, embeddings):
                content = doc_data.get('content', '').strip()
                if not content:
                    logger.warning("Skipping document with empty content")
                    continue
                
                # Create document object
                doc = Document(
                    id=uuid.uuid4(),
                    content=content,
                    metadata=doc_data.get('metadata', {})
                )
                
                # Add pre-computed embedding
                setattr(doc, 'embedding', embedding)
                
                # Save to database
                created_doc = await self.document_repo.create_document(doc)
                created_documents.append(created_doc)
            
            # Invalidate search caches
            await redis_cache.invalidate_search_cache()
            
            logger.info(f"Successfully ingested {len(created_documents)} documents with pre-computed embeddings")
            return created_documents
            
        except Exception as e:
            logger.error(f"Batch ingestion with embeddings failed: {e}")
            raise

    async def ingest_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        enable_chunking: bool = True,
        batch_size: int = 5
    ) -> List[Document]:
        """Ingest multiple documents in batches."""
        all_created = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Create tasks for parallel processing
            tasks = []
            for doc_data in batch:
                content = doc_data.get('content', '')
                metadata = doc_data.get('metadata', {})
                
                if content.strip():
                    task = self.ingest_document(content, metadata, enable_chunking)
                    tasks.append(task)

            try:
                # Process batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results and exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to ingest document {i + j}: {result}")
                    else:
                        all_created.extend(result)

                # Brief pause between batches
                if i + batch_size < len(documents):
                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Batch ingestion failed: {e}")
                continue

        logger.info(f"Batch ingestion completed: {len(all_created)} total documents created")
        return all_created

    async def update_document(
        self,
        document_id: uuid.UUID,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        regenerate_embedding: bool = True
    ) -> Optional[Document]:
        """Update an existing document."""
        try:
            # Check if document exists
            existing = await self.document_repo.get_document(document_id)
            if not existing:
                return None

            embedding = None
            if content and regenerate_embedding:
                # Generate new embedding if content changed
                embedding = await self.embedding_service.embed_text(content)

            # Update document
            updated = await self.document_repo.update_document(
                document_id=document_id,
                content=content,
                metadata=metadata,
                embedding=embedding
            )

            if updated:
                # Invalidate search caches
                await redis_cache.invalidate_search_cache()
                logger.info(f"Updated document {document_id}")

            return updated

        except Exception as e:
            logger.error(f"Document update failed: {e}")
            raise

    async def delete_document(self, document_id: uuid.UUID) -> bool:
        """Delete a document."""
        try:
            success = await self.document_repo.delete_document(document_id)
            
            if success:
                # Invalidate search caches
                await redis_cache.invalidate_search_cache()
                logger.info(f"Deleted document {document_id}")

            return success

        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            raise

    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        try:
            total_docs = await self.document_repo.count_documents()
            
            return {
                "total_documents": total_docs,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            return {
                "total_documents": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "error",
                "error": str(e)
            }