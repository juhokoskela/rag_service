from typing import List, Tuple, Optional, Dict, Any
import uuid
from datetime import datetime
import asyncpg
import json
from src.core.models import Document
from src.infrastructure.postgres import get_db_connection
import logging

logger = logging.getLogger(__name__)

class DocumentRepository:
    async def create_document(self, document: Document) -> Document:
        """Create a new document with embedding."""
        async with get_db_connection() as conn:
            try:
                # Convert embedding list to pgvector format
                embedding_str = '[' + ','.join(map(str, document.embedding)) + ']' if hasattr(document, 'embedding') else None
                
                query = """
                INSERT INTO documents (id, content, metadata, embedding, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, content, metadata, embedding, created_at, updated_at
                """
                
                now = datetime.now()
                row = await conn.fetchrow(
                    query,
                    document.id,
                    document.content,
                    json.dumps(document.metadata),
                    embedding_str,
                    now,
                    now
                )
                
                # Convert embedding back from pgvector format
                embedding = None
                if row['embedding']:
                    # Convert from pgvector string to list
                    embedding_str = str(row['embedding'])
                    if embedding_str.startswith('[') and embedding_str.endswith(']'):
                        embedding = [float(x) for x in embedding_str[1:-1].split(',')]
                
                return Document(
                    id=row['id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    embedding=embedding,
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            except Exception as e:
                logger.error(f"Failed to create document: {e}")
                raise

    async def get_document(self, document_id: uuid.UUID) -> Optional[Document]:
        """Get document by ID."""
        async with get_db_connection() as conn:
            try:
                query = """
                SELECT id, content, metadata, embedding, created_at, updated_at
                FROM documents 
                WHERE id = $1 AND deleted_at IS NULL
                """
                
                row = await conn.fetchrow(query, document_id)
                if not row:
                    return None
                
                # Convert embedding back from pgvector format
                embedding = None
                if row['embedding']:
                    # Convert from pgvector string to list
                    embedding_str = str(row['embedding'])
                    if embedding_str.startswith('[') and embedding_str.endswith(']'):
                        embedding = [float(x) for x in embedding_str[1:-1].split(',')]
                
                return Document(
                    id=row['id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    embedding=embedding,
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            except Exception as e:
                logger.error(f"Failed to get document {document_id}: {e}")
                raise

    async def update_document(self, document_id: uuid.UUID, content: str = None, metadata: Dict[str, Any] = None, embedding: List[float] = None) -> Optional[Document]:
        """Update document."""
        async with get_db_connection() as conn:
            try:
                updates = []
                params = []
                param_count = 1
                
                if content is not None:
                    updates.append(f"content = ${param_count}")
                    params.append(content)
                    param_count += 1
                
                if metadata is not None:
                    updates.append(f"metadata = ${param_count}")
                    params.append(json.dumps(metadata))
                    param_count += 1
                
                if embedding is not None:
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    updates.append(f"embedding = ${param_count}")
                    params.append(embedding_str)
                    param_count += 1
                
                if not updates:
                    return await self.get_document(document_id)
                
                updates.append(f"updated_at = ${param_count}")
                params.append(datetime.utcnow())
                param_count += 1
                
                params.append(document_id)
                
                query = f"""
                UPDATE documents 
                SET {', '.join(updates)}
                WHERE id = ${param_count} AND deleted_at IS NULL
                RETURNING id, content, metadata, created_at, updated_at
                """
                
                row = await conn.fetchrow(query, *params)
                if not row:
                    return None
                
                return Document(
                    id=row['id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            except Exception as e:
                logger.error(f"Failed to update document {document_id}: {e}")
                raise

    async def delete_document(self, document_id: uuid.UUID) -> bool:
        """Soft delete document."""
        async with get_db_connection() as conn:
            try:
                query = """
                UPDATE documents 
                SET deleted_at = $1, updated_at = $1
                WHERE id = $2 AND deleted_at IS NULL
                """
                
                now = datetime.utcnow()
                result = await conn.execute(query, now, document_id)
                return result.split()[-1] == '1'  # Check if one row was updated
            except Exception as e:
                logger.error(f"Failed to delete document {document_id}: {e}")
                raise

    async def vector_search(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search documents by vector similarity."""
        async with get_db_connection() as conn:
            try:
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                # Build query with optional metadata filter
                where_conditions = ["deleted_at IS NULL"]
                params = [embedding_str, threshold, limit]
                param_count = 4
                
                if metadata_filter:
                    for key, value in metadata_filter.items():
                        where_conditions.append(f"metadata->>'{key}' = ${param_count}")
                        params.append(str(value))
                        param_count += 1
                
                where_clause = " AND ".join(where_conditions)
                
                query = f"""
                SELECT id, content, metadata, created_at, updated_at,
                       1 - (embedding <=> $1) as similarity
                FROM documents
                WHERE {where_clause} AND (1 - (embedding <=> $1)) >= $2
                ORDER BY embedding <=> $1
                LIMIT $3
                """
                
                rows = await conn.fetch(query, *params)
                
                results = []
                for row in rows:
                    doc = Document(
                        id=row['id'],
                        content=row['content'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    results.append((doc, float(row['similarity'])))
                
                return results
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                raise

    async def get_all_documents(self, limit: Optional[int] = None) -> List[Document]:
        """Get all documents (for BM25 indexing)."""
        async with get_db_connection() as conn:
            try:
                query = """
                SELECT id, content, metadata, embedding, created_at, updated_at
                FROM documents
                WHERE deleted_at IS NULL
                ORDER BY created_at DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                rows = await conn.fetch(query)
                
                documents = []
                for row in rows:
                    # Convert embedding back from pgvector format
                    embedding = None
                    if row['embedding']:
                        # Convert from pgvector string to list
                        embedding_str = str(row['embedding'])
                        if embedding_str.startswith('[') and embedding_str.endswith(']'):
                            embedding = [float(x) for x in embedding_str[1:-1].split(',')]
                    
                    doc = Document(
                        id=row['id'],
                        content=row['content'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                        embedding=embedding,
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    documents.append(doc)
                
                return documents
            except Exception as e:
                logger.error(f"Failed to get all documents: {e}")
                raise

    async def count_documents(self) -> int:
        """Get total document count."""
        async with get_db_connection() as conn:
            try:
                result = await conn.fetchval("SELECT COUNT(*) FROM documents WHERE deleted_at IS NULL")
                return result or 0
            except Exception as e:
                logger.error(f"Failed to count documents: {e}")
                raise