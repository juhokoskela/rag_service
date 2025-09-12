from typing import List, Dict, Any, Optional
import bm25s
import numpy as np
from Stemmer import Stemmer
from stopwordsiso import stopwords
from src.core.models import Document, SearchResult
from src.repositories.document_repository import DocumentRepository
import logging

logger = logging.getLogger(__name__)

class BM25SearchService:
    def __init__(self, document_repo: DocumentRepository):
        self.document_repo = document_repo
        self.stemmer = Stemmer('english')
        self.stop_words = set(stopwords('en'))
        self.index = None
        self.documents = []

    def _preprocess_text(self, text: str) -> List[str]:
        """Tokenize, stem, and remove stopwords."""
        tokens = text.lower().split()
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stemWord(token) for token in tokens]
        return tokens

    async def build_index(self):
        """Build BM25 index from all documents."""
        try:
            documents = await self.document_repo.get_all_documents()
            self.documents = documents
            
            corpus = [self._preprocess_text(doc.content) for doc in documents]
            
            # Ensure corpus is properly formatted for bm25s
            if not corpus:
                logger.warning("No documents found for BM25 index")
                self.index = None
                return
            
            # Validate corpus format - each doc should be a list of strings
            for i, doc_tokens in enumerate(corpus):
                if not isinstance(doc_tokens, list):
                    raise TypeError(f"Doc #{i} tokens must be list, got {type(doc_tokens).__name__}")
                if not all(isinstance(token, str) for token in doc_tokens):
                    raise TypeError(f"Doc #{i} contains non-string tokens")
            
            # Initialize BM25 without parameters and then index
            self.index = bm25s.BM25()
            self.index.index(corpus)
            
            logger.info(f"Built BM25 index with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            raise

    async def search(
        self,
        query: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform BM25 search."""
        if self.index is None:
            await self.build_index()
        
            # If still no index after building (no documents), return empty results                                                                     │ │
            if self.index is None:                                                                                                                      │ │
            logger.warning("BM25 search: No documents available for search")                                                                        │ │
            return []   

        try:
            query_tokens = self._preprocess_text(query)
            if not query_tokens:
                logger.debug("No valid tokens after query preprocessing")
                return []
            
            # Use retrieve method instead of get_scores for better compatibility
            # Wrap single query as batch: [query_tokens] -> List[List[str]]
            # Automatically clamp k to corpus size to avoid bm25s errors
            actual_k = min(limit, len(self.documents)) if self.documents else 1
            results = self.index.retrieve([query_tokens], k=actual_k)
            
            # Handle bm25s return format (indices, scores)
            if isinstance(results, tuple):
                indices, scores = results
            else:
                indices = getattr(results, 'indices', None)
                scores = getattr(results, 'scores', None)
            
            if indices is None or scores is None:
                return []
            
            # If batched results, take first batch
            if hasattr(indices, '__len__') and len(indices) > 0 and hasattr(indices[0], '__len__'):
                indices = indices[0]
                scores = scores[0]
            
            search_results = []
            n = min(len(indices), len(scores))
            
            for i in range(n):
                idx = int(indices[i])
                score = float(scores[i])
                
                if score > 0 and 0 <= idx < len(self.documents):  # Only include positive scores
                    doc = self.documents[idx]
                    
                    # Apply metadata filter if specified
                    if metadata_filter:
                        match = True
                        for key, value in metadata_filter.items():
                            if doc.metadata.get(key) != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    result = SearchResult(
                        document=doc,
                        score=score,
                        rank_explanation={"method": "bm25", "score": score}
                    )
                    search_results.append(result)
            
            return search_results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise