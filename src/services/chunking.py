from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from src.core.config import settings
from src.core.models import Document
import uuid
import logging

logger = logging.getLogger(__name__)

class ChunkingService:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Used by text-embedding-3-large
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            keep_separator=False,
            add_start_index=True,
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            # Fallback to character count / 4 (rough estimation)
            return len(text) // 4

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()

    def chunk_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[uuid.UUID] = None
    ) -> List[Document]:
        """Split text into chunks and create Document objects."""
        if not text or not text.strip():
            return []

        try:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Split into chunks
            chunks = self.splitter.split_text(cleaned_text)
            
            if not chunks:
                return []

            documents = []
            base_metadata = metadata or {}
            
            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                # Create chunk metadata
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "token_count": self._count_tokens(chunk_text),
                    "char_count": len(chunk_text)
                }
                
                if parent_id:
                    chunk_metadata["parent_id"] = str(parent_id)

                # Create document
                doc = Document(
                    id=uuid.uuid4(),
                    content=chunk_text.strip(),
                    metadata=chunk_metadata
                )
                documents.append(doc)

            logger.info(f"Created {len(documents)} chunks from text of {len(text)} characters")
            return documents

        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            raise

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Process multiple documents and return all chunks."""
        all_chunks = []
        
        for doc_data in documents:
            content = doc_data.get('content', '')
            metadata = doc_data.get('metadata', {})
            doc_id = doc_data.get('id')
            
            if not content:
                continue
                
            try:
                chunks = self.chunk_text(
                    content, 
                    metadata=metadata,
                    parent_id=uuid.UUID(doc_id) if doc_id else None
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to chunk document {doc_id}: {e}")
                continue

        return all_chunks

    def estimate_chunks(self, text: str) -> int:
        """Estimate number of chunks that will be created."""
        if not text:
            return 0
        
        token_count = self._count_tokens(text)
        overlap_tokens = int(settings.chunk_size * (settings.chunk_overlap / settings.chunk_size))
        effective_chunk_size = settings.chunk_size - overlap_tokens
        
        return max(1, (token_count + effective_chunk_size - 1) // effective_chunk_size)

    def validate_chunk_size(self, text: str) -> bool:
        """Validate if text chunk is appropriate size."""
        token_count = self._count_tokens(text)
        return 50 <= token_count <= settings.chunk_size * 1.2  # Allow 20% over for edge cases

    def merge_small_chunks(self, chunks: List[Document], min_size: int = 100) -> List[Document]:
        """Merge consecutive small chunks to improve quality."""
        if not chunks:
            return chunks

        merged = []
        current_chunk = None

        for chunk in chunks:
            token_count = self._count_tokens(chunk.content)
            
            if token_count < min_size and current_chunk:
                # Merge with previous chunk
                merged_content = f"{current_chunk.content}\n\n{chunk.content}"
                merged_tokens = self._count_tokens(merged_content)
                
                if merged_tokens <= settings.chunk_size:
                    # Update current chunk
                    current_chunk.content = merged_content
                    current_chunk.metadata['token_count'] = merged_tokens
                    current_chunk.metadata['char_count'] = len(merged_content)
                    current_chunk.metadata['merged'] = True
                    continue
            
            # Add previous chunk to results
            if current_chunk:
                merged.append(current_chunk)
            
            current_chunk = chunk

        # Add the last chunk
        if current_chunk:
            merged.append(current_chunk)

        return merged