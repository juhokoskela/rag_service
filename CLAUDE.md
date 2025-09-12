# CLAUDE.md - RAG Retrieval Service

This document provides comprehensive technical documentation for AI assistants working with the RAG (Retrieval-Augmented Generation) service codebase. It covers architecture, implementation details, performance optimizations, and development guidelines.

## Project Overview

### Purpose
This RAG service is a focused, production-ready document retrieval system designed to be called as a tool or API by other applications. It provides intelligent document search using hybrid vector + keyword search with advanced reranking.

### Architecture Philosophy
- **Separation of Concerns**: Clean layering (API → Services → Repositories → Infrastructure)
- **Performance First**: Optimized for low latency with intelligent caching
- **Production Ready**: Health checks, rate limiting, correlation IDs, circuit breakers
- **Docker Native**: Complete containerization for consistent deployment
- **Tool-Friendly**: RESTful API designed for LLM tool calling

### Technology Stack
- **Framework**: FastAPI with async/await patterns
- **Database**: PostgreSQL 16 with pgvector extension
- **Cache**: Redis for embeddings and search results
- **ML**: OpenAI embeddings, Jina reranking, local cross-encoders
- **Search**: Hybrid vector (HNSW indexes) + BM25 full-text
- **Text Processing**: LangChain splitters with tiktoken tokenization

## System Architecture

### High-Level Component Flow
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Services       │    │  Repositories   │
│   Routes        │───▶│   Layer          │───▶│   & Storage     │
│                 │    │                  │    │                 │
│ • Health        │    │ • HybridSearch   │    │ • DocumentRepo  │
│ • Search        │    │ • VectorSearch   │    │ • EmbedCache    │
│ • Documents     │    │ • BM25Search     │    │ • PostgreSQL    │
│                 │    │ • Reranking      │    │ • Redis         │
│                 │    │ • Embedding      │    │                 │
│                 │    │ • Chunking       │    │                 │
│                 │    │ • Ingestion      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow for Document Ingestion
```
Text Document → Chunking Service → Embedding Service → Document Repository
                     ↓                    ↓                    ↓
               LangChain Split      OpenAI API         PostgreSQL+pgvector
               (800 tokens)      (text-embed-3-large)     (HNSW index)
                     ↓                    ↓                    ↓
               Overlap Merge        Redis Cache          Metadata Storage
```

### Search Flow Architecture
```
Query → Embedding Service → ┌─ Vector Search ─┐
  ↓         (cached)         │    (pgvector)   │
  ↓                         └─────────────────┘
  ↓                                 ↓
  └─── BM25 Search ────────── Hybrid Fusion ────── Reranking ────── Results
       (preprocessed)         (score normalize)    (Jina API)     (ranked)
```

## Core Components Deep Dive

### 1. Hybrid Search Algorithm (`src/services/hybrid_search.py`)

#### Score Fusion Strategy
- **Vector Search**: Cosine similarity (0-1 range)
- **BM25 Search**: TF-IDF based relevance scores
- **Normalization**: Min-max scaling to [0,1] range for fair weighting
- **Fusion Formula**: `final_score = (vector_score * v_weight) + (bm25_score * b_weight)`
- **Default Weights**: Vector 0.7, BM25 0.3 (tunable via config)

#### Implementation Details
```python
# Key method: _merge_results()
# 1. Normalize scores independently
# 2. Create combined document dictionary
# 3. Apply weighted scoring
# 4. Sort by final scores
# 5. Return top-k results
```

### 2. Vector Search (`src/services/vector_search.py`)

#### PostgreSQL with pgvector
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Distance Function**: Cosine distance (`<=>` operator)
- **Optimization**: HNSW parameters: `m=16, ef_construction=64`
- **Query Pattern**: `ORDER BY embedding <=> query_vector LIMIT k`

#### Performance Considerations
- **Batch Size**: Process embeddings in chunks of 10
- **Connection Pool**: 5-20 connections via asyncpg
- **Threshold**: Configurable similarity minimum (default 0.7)
- **Metadata Filtering**: JSON field queries with proper indexing

### 3. BM25 Search (`src/services/bm25_search.py`)

#### Implementation Strategy
- **Library**: `bm25s` (Rust-based, faster than `rank-bm25`)
- **Preprocessing**: Tokenization → Stemming → Stopword removal
- **Language**: English stemmer with ISO stopwords
- **Index Building**: Dynamic index creation from document corpus
- **Score Cutoff**: Only positive scores returned

#### Text Processing Pipeline
```python
text → tokenize() → remove_stopwords() → stem() → bm25_index
                                                      ↓
query → same_pipeline() → bm25.get_scores() → top_k_results
```

### 4. Reranking System (`src/services/reranking.py`)

#### Two-Tier Strategy
1. **Primary**: Jina AI multilingual reranker (`jina-reranker-v2-base-multilingual`)
2. **Fallback**: Local cross-encoder (`cross-encoder/ms-marco-minilm-l-6-v2`)

#### Quality Improvements
- **Input**: Query + candidate documents
- **Output**: Refined relevance scores
- **Failure Handling**: Automatic fallback to local model
- **Performance**: Timeout handling, error recovery

### 5. Embedding Service (`src/services/embedding.py`)

#### OpenAI Integration
- **Model**: `text-embedding-3-large` (3072 dimensions)
- **Client**: AsyncOpenAI for concurrent requests
- **Retry Logic**: Exponential backoff (3 attempts, 4-10s delays)
- **Batch Processing**: 10 embeddings per batch, 0.1s delays

#### Two-Tier Caching
- **L1 (Redis)**: Fast access, 24h TTL
- **L2 (PostgreSQL)**: Persistent storage with access tracking
- **Cache Key**: SHA256 hash of `model:text` combination
- **Hit Rate**: Optimized for development/testing workflows

## Database Schema

### Documents Table
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(3072),  -- pgvector type
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE  -- Soft delete
);
```

### Indexes for Performance
```sql
-- Vector similarity (HNSW for high performance)
CREATE INDEX idx_documents_embedding_hnsw 
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search (trigram for fuzzy matching)
CREATE INDEX idx_documents_content_trgm
ON documents USING gin (content gin_trgm_ops);

-- Metadata queries
CREATE INDEX idx_documents_metadata_gin
ON documents USING gin (metadata);

-- Active documents (soft delete aware)
CREATE INDEX idx_documents_active
ON documents (created_at) 
WHERE deleted_at IS NULL;
```

### Embedding Cache Table
```sql
CREATE TABLE embedding_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_hash VARCHAR(64) UNIQUE NOT NULL,
    embedding VECTOR(3072),
    model VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 1
);
```

## API Design

### RESTful Endpoints

#### Core Search
- `POST /api/v1/search` - Hybrid search (vector + BM25 + reranking)
- `POST /api/v1/search/vector` - Pure vector search
- `POST /api/v1/search/bm25` - Pure BM25 search

#### Document Management
- `POST /api/v1/documents` - Create document (auto-chunking)
- `GET /api/v1/documents/{id}` - Retrieve document
- `PUT /api/v1/documents/{id}` - Update document
- `DELETE /api/v1/documents/{id}` - Delete document

#### Health & Monitoring
- `GET /health` - Basic health check
- `GET /health/detailed` - Dependency health
- `GET /health/ready` - Kubernetes readiness
- `GET /health/live` - Kubernetes liveness

### Request/Response Models

#### SearchRequest
```python
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    include_metadata: bool = True
```

#### SearchResponse
```python
class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    processing_time_ms: float
```

## Performance Optimizations

### 1. Caching Strategy

#### Multi-Level Caching
```
Request → Redis Check → PostgreSQL Check → API Call → Cache Result
   ↓         (hit)         (hit)           (miss)      (all levels)
Result ←─────────────────────────────────────────────────────────
```

#### Cache Invalidation
- **Search Results**: Invalidated on document changes
- **Embeddings**: Long TTL (24h) with LRU eviction
- **Health Data**: No caching (real-time)

### 2. Database Optimizations

#### Connection Management
- **Pool Size**: 5-20 connections (tunable)
- **Connection Timeout**: 60 seconds
- **Query Timeout**: Configurable per operation
- **Health Checks**: Regular connection validation

#### Index Strategy
- **HNSW vs IVFFlat**: HNSW chosen for better query performance
- **Partial Indexes**: Only index active documents
- **Composite Indexes**: Optimized for common query patterns

### 3. Async Processing

#### Concurrent Operations
- **Search Parallelization**: Vector + BM25 searches run concurrently
- **Batch Embedding**: Process multiple documents simultaneously
- **Connection Pooling**: Shared connections across requests
- **Background Tasks**: Cache cleanup, index maintenance

## Search Quality Features

### 1. Score Normalization
```python
def normalize_scores(scores):
    min_score, max_score = min(scores), max(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]
```

### 2. Result Diversity (MMR Pattern)
- **Future Enhancement**: Maximal Marginal Relevance
- **Purpose**: Reduce duplicate/similar results
- **Algorithm**: Balance relevance vs diversity

### 3. Query Understanding
- **Current**: Direct embedding + keyword matching  
- **Future**: Intent classification, query expansion
- **Potential**: Multi-query generation, back-translation

## Development Guidelines

### Code Patterns to Follow

#### 1. Service Layer Pattern
```python
class ServiceName:
    def __init__(self, dependencies):
        self.deps = dependencies
    
    async def method_name(self, params) -> return_type:
        try:
            # Implementation
            return result
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            raise
```

#### 2. Error Handling
- **Custom Exceptions**: Use domain-specific exceptions
- **Correlation IDs**: Include in all log messages
- **Graceful Degradation**: Fallback when possible
- **Resource Cleanup**: Use context managers

#### 3. Testing Patterns
```python
@pytest.mark.asyncio
async def test_service_method():
    # Arrange
    service = create_test_service()
    
    # Act
    result = await service.method(test_data)
    
    # Assert
    assert result.expected_property == expected_value
```

### Common Pitfalls

1. **Vector Dimensions**: Ensure embedding dimensions match (3072)
2. **Connection Leaks**: Always use context managers
3. **Cache Invalidation**: Clear search caches on document updates
4. **Rate Limits**: Respect OpenAI API limits
5. **Memory Usage**: Monitor embedding cache size

### Extension Points

#### Adding New Search Methods
1. Implement search service in `src/services/`
2. Add to hybrid search fusion
3. Update API routes
4. Add configuration options

#### Custom Rerankers
1. Extend `RerankingService` base methods
2. Add configuration for new models
3. Implement fallback logic

#### Metadata Enhancements
1. Add new fields to Document model
2. Update database migration
3. Add filtering support in repositories

## Configuration Reference

### Environment Variables

#### Core Settings
- `OPENAI_API_KEY`: OpenAI API key (required)
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `JINA_API_KEY`: Jina reranking API key (optional)

#### Performance Tuning
- `CHUNK_SIZE`: Target tokens per chunk (default: 800)
- `CHUNK_OVERLAP`: Overlap tokens (default: 100)
- `VECTOR_WEIGHT`: Hybrid search vector weight (default: 0.7)
- `BM25_WEIGHT`: Hybrid search BM25 weight (default: 0.3)

#### Rate Limiting
- `RATE_LIMIT_PER_MINUTE`: API requests per minute (default: 60)
- `OPENAI_TIMEOUT`: API timeout in seconds (default: 60)

## Troubleshooting Guide

### Common Issues

#### 1. High Memory Usage
- **Cause**: Large embedding cache
- **Solution**: Reduce cache TTL, implement LRU eviction
- **Monitoring**: Track Redis memory usage

#### 2. Slow Search Performance
- **Cause**: Missing indexes, large result sets
- **Solution**: Check HNSW index, tune `ef_construction`
- **Debug**: Use `EXPLAIN ANALYZE` on PostgreSQL queries

#### 3. Embedding API Failures
- **Cause**: Rate limits, network issues
- **Solution**: Implement exponential backoff, batch sizing
- **Fallback**: Use cached embeddings when available

#### 4. Docker Issues
- **Cause**: Port conflicts, environment variables
- **Solution**: Check `docker-compose.yml`, verify `.env`
- **Debug**: Use `docker-compose logs` for diagnostics

### Debug Strategies

#### 1. Enable Detailed Logging
```bash
LOG_LEVEL=DEBUG docker-compose up
```

#### 2. Health Check Endpoints
- Check `/health/detailed` for dependency status
- Monitor processing times in response headers

#### 3. Database Queries
```sql
-- Check index usage
EXPLAIN ANALYZE SELECT ... FROM documents WHERE ...;

-- Monitor cache hit rates
SELECT * FROM embedding_cache ORDER BY access_count DESC LIMIT 10;
```

### Performance Profiling

#### 1. Request Timing
- Check `x-process-time` response header
- Monitor correlation IDs across logs
- Use `/metrics` endpoint for aggregated stats

#### 2. Database Performance
- Monitor connection pool usage
- Check query execution times
- Analyze index effectiveness

## Future Enhancements

### Planned Improvements

#### 1. Query Expansion
- **HyDE**: Hypothetical document generation
- **Synonym Expansion**: WordNet integration  
- **Multi-Query**: Generate multiple query variants

#### 2. Advanced Reranking
- **Learning to Rank**: Custom ranking models
- **User Feedback**: Incorporate click-through data
- **Context Awareness**: Consider user session

#### 3. Scalability
- **Horizontal Scaling**: Multiple API instances
- **Read Replicas**: Separate read/write databases
- **Caching Layers**: Multi-region Redis clusters

#### 4. Quality Metrics
- **Relevance Scoring**: A/B testing framework
- **User Analytics**: Query success tracking
- **Performance Monitoring**: Detailed metrics dashboard

### Architecture Evolution

#### Microservices Migration
- **Search Service**: Independent search processing
- **Embedding Service**: Dedicated embedding generation
- **Document Service**: Document CRUD operations

#### Event-Driven Architecture
- **Document Events**: Publish on create/update/delete
- **Async Processing**: Background embedding generation
- **Cache Warming**: Proactive cache population

## Development Commands

### Quick Reference
```bash
# Start services
make up

# Run migrations
make migrate  

# Health check
make health

# View logs
make logs

# Run tests
make test

# Clean rebuild
make clean && make up
```

### File Structure Navigation
```
src/
├── core/                    # Configuration, models, exceptions
├── services/                # Business logic layer
│   ├── hybrid_search.py    # Main search orchestration
│   ├── vector_search.py    # pgvector operations
│   ├── bm25_search.py      # BM25 full-text search
│   ├── reranking.py        # Result reranking
│   ├── embedding.py        # OpenAI embeddings
│   ├── chunking.py         # Text processing
│   └── ingestion.py        # Document pipeline
├── repositories/            # Data access layer
├── infrastructure/          # Database/cache connections
├── api/                     # FastAPI routes
└── main.py                 # Application entry point
```

---

## Summary

This RAG service provides a production-ready, high-performance document retrieval system with hybrid search capabilities. The architecture emphasizes clean separation of concerns, aggressive caching, and robust error handling. The implementation combines the semantic understanding of vector search with the precision of keyword search, enhanced by AI-powered reranking for optimal relevance.

Key strengths:
- **Performance**: Sub-second search with intelligent caching
- **Quality**: Multi-strategy search with reranking
- **Reliability**: Production middleware with health monitoring
- **Scalability**: Docker-native with clear extension points
- **Maintainability**: Clean architecture with comprehensive testing

This system serves as both a standalone RAG service and a reference implementation for advanced document retrieval systems.