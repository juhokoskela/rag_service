# RAG service for LLM tool use

A focused RAG service that handles document ingestion and hybrid search. This service provides intelligent document retrieval using vector similarity and BM25 search with reranking capabilities.

## Features

- **Document Management**: Add, update, soft-delete, and restore documents with metadata
- **Hybrid Search**: Combines vector similarity search (pgvector) with BM25 keyword search
- **Smart Reranking**: Uses Jina AI reranker (or local fallback) with tunable payload caps
- **High Performance**: PostgreSQL with HNSW indexes, Redis caching, and a persistent embedding cache
- **Production Ready**: Health checks, configurable rate limiting, circuit breakers, structured logging
- **Docker Support**: Full containerization with Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- (Optional) Jina AI API key for reranking

### Setup

1. **Clone and navigate to the service:**
   ```bash
   cd rag-service
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

3. **Start the services:**
   ```bash
   make up
   ```

4. **Initialize the database:**
   ```bash
   make migrate
   ```

5. **Check service health:**
   ```bash
   make health
   ```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs` (WIP).

## Authentication & Rate Limiting

- **Authentication**: Optional middleware enforces `Authorization: Bearer …` headers when configured. Supply a shared secret via `RAG_API_TOKEN` or a JWT signing key via `RAG_JWT_SECRET` (+ optional audience/issuer). If both are set, the service verifies JWTs first and falls back to the shared token. The request middleware automatically propagates `x-correlation-id` so upstream systems can trace requests end-to-end.
- **Rate Limiting**: All routes share a global SlowAPI limiter. Limits are enforced per client IP (default 60 requests/min) and are configurable with `RATE_LIMIT_PER_MINUTE`. Exceeding the threshold returns HTTP 429 with `Retry-After` headers.
- **Circuit Breaking**: Critical network calls (OpenAI/Jina) go through a simple circuit breaker to stop cascading failures; monitor logs for `circuitbreaker` warnings when upstreams misbehave.

## Adding Documents

### Easy Document Submission

Instead of using curl, you can easily add documents using several methods:

#### 1. Command Line Script
```bash
# Add text directly
python scripts/add_document.py "Your document content here"

# Upload a file
python scripts/add_document.py --file document.txt

# Upload with metadata
python scripts/add_document.py --file document.txt --metadata '{"source": "manual", "author": "John"}'

# Fetch from URL
python scripts/add_document.py --url "https://example.com/article"
```

#### 2. File Upload API
```bash
# Upload a single file
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.txt" \
  -F 'metadata={"source": "upload"}'
```

#### 3. Batch Processing

**Large Scale Processing with OpenAI Batch API:**
```bash
# Process all files in a directory (uses OpenAI Batch API for 50+ chunks)
python scripts/batch_add.py /path/to/documents/

# Process specific file types with batch optimization
python scripts/batch_add.py /path/to/docs/ --pattern "*.md" --use-batch-api

# Add metadata to all files
python scripts/batch_add.py /path/to/docs/ --metadata '{"project": "docs", "version": "1.0"}'
```

**API Batch Jobs:**
```bash
# Create a batch job for multiple documents (10+ documents required)
curl -X POST "http://localhost:8000/api/v1/documents/batch-job" \
  -H "Content-Type: application/json" \
  -d '[{"content": "Document 1..."}, {"content": "Document 2..."}]'

# Check batch job status
curl "http://localhost:8000/api/v1/documents/batch-job/{job_id}"
```

**Batch Processing Benefits:**
- **50% cost savings** compared to regular OpenAI API
- **Higher throughput** for large document collections
- **Automatic fallback** to regular API for small batches
- **Smart caching** reduces redundant API calls
- **Progress tracking** with detailed status updates

#### Supported File Types
- Text files: `.txt`, `.md`, `.rst`
- Code files: `.py`, `.js`, `.html`, `.xml`, `.json`
- Data files: `.csv`, `.log`

#### 4. Zendesk FAQ Import

**Import FAQ articles directly from Zendesk with Batch API:**
```bash
# Set up Zendesk credentials
export ZENDESK_SUBDOMAIN=your-company
export ZENDESK_EMAIL=admin@company.com
export ZENDESK_TOKEN=your-api-token

# Import all FAQ categories with batch processing (default)
python scripts/import_zendesk_faqs.py

# Import specific category
python scripts/import_zendesk_faqs.py --category-id 123456 --service-type your_service

# Import without batch API (individual processing)
python scripts/import_zendesk_faqs.py --no-batch

# Dry run to see what would be imported
python scripts/import_zendesk_faqs.py --dry-run
```

**Zendesk Integration Features:**
- **OpenAI Batch API Integration** - 50% cost savings on embedding generation
- **Smart batch processing** - Automatically uses batch API for 10+ articles
- **Progress monitoring** - Real-time batch job status tracking
- **Automatic fallback** - Falls back to individual processing if batch fails
- **HTML cleaning** - Converts HTML articles to clean text
- **Metadata extraction** - Preserves article titles, URLs, votes, etc.
- **Service type mapping** - Categorizes articles by business unit
- **Error resilience** - Failed articles don't stop the entire import

## Architecture

### Directory Structure

```
rag-service/
├── src/
│   ├── core/           # Configuration, models, exceptions
│   ├── services/       # Business logic (search, embedding, chunking)
│   ├── repositories/   # Database access layer
│   ├── api/            # FastAPI routes and endpoints
│   │   └── routes/
│   └── infrastructure/ # PostgreSQL, Redis implementations
├── alembic/           # Database migrations
├── tests/             # Test files
├── scripts/           # Utility scripts
└── docs/              # Documentation
```

### Core Components

- **FastAPI**: Modern Python web framework with automatic API docs
- **PostgreSQL + pgvector**: Vector database for embeddings with HNSW indexes
- **Redis**: Caching layer for embeddings and search results
- **OpenAI Embeddings**: High-quality text embeddings (text-embedding-3-large)
- **BM25 Search**: Traditional keyword search for hybrid retrieval
- **Jina Reranker**: Cross-encoder reranking for improved relevance

### Data Lifecycle & Caching

- **Soft Deletes**: Documents are never hard-deleted; instead we set `deleted_at` and filter them transparently. This keeps history for potential restores and analytics while letting the ingestion pipeline “undelete” records safely.
- **Embedding Cache**: A two-tier cache stores embeddings in Redis (hot path) and PostgreSQL (durable). Each entry tracks `last_accessed` and `access_count`, so stale rows can be trimmed with the `EmbeddingCacheRepository.cleanup_old_cache` helper or a cron job.
- **Redis Invalidation**: Document mutations automatically purge hybrid search caches to ensure fresh results without having to flush the entire Redis instance.

## API Endpoints

### Document Management
- `POST /api/v1/documents/` - Add new document (returns document chunks)
- `POST /api/v1/documents/upload` - Upload text file as document
- `POST /api/v1/documents/batch-job` - Create batch processing job (10+ documents, 50% cost savings)
- `GET /api/v1/documents/batch-job/{job_id}` - Check batch job status
- `GET /api/v1/documents/` - List all documents with pagination
- `GET /api/v1/documents/{id}` - Retrieve specific document
- `PUT /api/v1/documents/{id}` - Update document
- `DELETE /api/v1/documents/{id}` - Delete document

### Search
- `POST /api/v1/search/` - Hybrid search across documents
- `POST /api/v1/search/vector` - Pure vector similarity search
- `POST /api/v1/search/bm25` - Pure BM25 keyword search

### Health & Monitoring
- `GET /health/` - Basic health check
- `GET /health/detailed` - Detailed service health

## Configuration

The service uses flexible, component-based configuration that works with any deployment scenario - from local development to managed cloud services.

### Core Configuration

Key environment variables in `.env`:

```bash
# Required - OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Database Configuration (PostgreSQL)
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_pass  
POSTGRES_DB=rag_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis Configuration  
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SSL=false

# Optional - AI Services
JINA_API_KEY=your_jina_api_key  # For reranking

# Search Configuration
ENABLE_BM25=true               # Enable hybrid search
BM25_WEIGHT=0.3               # BM25 vs vector weight
VECTOR_WEIGHT=0.7
RATE_LIMIT_PER_MINUTE=60      # SlowAPI per-IP throttle
RERANK_TOP_K=10               # Max candidates sent to reranker
RERANK_MAX_CHARS=600          # Max characters per candidate payload
```

Tweak these knobs to balance relevance and latency. The default profile ranks the top 10 candidates with ~600-character payloads, which keeps Jina calls around 400 ms while preserving accuracy. Lower values reduce cost/latency further; higher values can improve recall at the expense of speed.

Leave the authentication variables blank to disable middleware (default). When set, every request must include `Authorization: Bearer <token>` where the token is either the shared secret (`RAG_API_TOKEN`) or a JWT signed with `RAG_JWT_SECRET`.

### Production Deployment Examples

#### AWS RDS + ElastiCache
```bash
# PostgreSQL (AWS RDS)
POSTGRES_HOST=your-db.abc123.us-east-1.rds.amazonaws.com
POSTGRES_USER=production_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=rag_production

# Redis (AWS ElastiCache)
REDIS_HOST=your-cluster.abc123.cache.amazonaws.com
REDIS_PASSWORD=your-auth-token
REDIS_SSL=true
```

#### Google Cloud SQL + Memorystore
```bash
# PostgreSQL (Cloud SQL)
POSTGRES_HOST=127.0.0.1  # via Cloud SQL Proxy
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password
POSTGRES_DB=rag_db

# Redis (Memorystore)
REDIS_HOST=10.1.2.3
REDIS_PORT=6379
REDIS_SSL=false
```

#### Azure Database + Cache
```bash
# PostgreSQL (Azure Database)
POSTGRES_HOST=your-server.postgres.database.azure.com
POSTGRES_USER=your-user@your-server
POSTGRES_PASSWORD=your-password
POSTGRES_DB=rag_db

# Redis (Azure Cache)
REDIS_HOST=your-cache.redis.cache.windows.net
REDIS_PORT=6380
REDIS_PASSWORD=your-access-key
REDIS_SSL=true
```

### Local Development
The default `.env.example` values work perfectly for local Docker development, no changes needed.

## Development

### Local Development

```bash
# Install dependencies
make dev-install

# Start development environment
make dev

# Run tests
make test

# Format code
make format

# Run linting
make lint
```

### Database Operations

```bash
# Apply migrations (database is auto-initialized on startup)
make migrate

# Reset database (WARNING: destroys data)
make reset-db

# Note: New migrations should be created manually in alembic/versions/
# as the service doesn't include alembic CLI in the container
```

### Common Tasks

```bash
# View logs
make logs

# Open container shell
make shell

# Check service health
make health-detailed

# Clean up everything
make clean
```

### Testing

```bash
# Run the fast smoke suite (no external services required)
pytest

# Exercise the embedding cache integration tests (requires Postgres + Redis running)
pytest tests/test_embedding_cache_repository.py

# Lint and type-check before committing
make lint
```

The CI smoke tests spin up the app with strict asyncio mode, so ensure you have valid `.env` values (or set `OPENAI_API_KEY=dummy` with `ENABLE_BM25=true` to avoid outbound calls) when running locally.

## Production Deployment

### Quick Production Setup

1. **Configure environment variables:**
   ```bash
   # Copy and customize for your environment
   cp .env.example .env
   
   # Set your production values
   ENVIRONMENT=production
   DEBUG=false
   POSTGRES_HOST=your-production-db-host.com
   REDIS_HOST=your-production-redis-host.com
   # ... etc
   ```

2. **Deploy with production settings:**
   ```bash
   ENVIRONMENT=production DEBUG=false docker-compose up -d
   ```

### Production Considerations

- **Managed Databases**: Use cloud-managed PostgreSQL and Redis for better reliability
- **Security**: 
  - Set strong passwords and enable SSL where possible
  - Use environment variables or secrets management for sensitive data
  - Configure proper CORS origins (not `*` in production)
- **Monitoring:**
  - Health check endpoints are available for load balancers
  - Structured logging
  - Consider adding metrics endpoints for monitoring

4. **Add this service as a tool an LLM can call**

### LLM Tool Integration Schemas

Tool use schemas for different APIs. Most LLM providers support Chat Completions API.
OpenAI additionally supports the Responses API. Use whichever suits your needs the best.

#### Chat Completions API

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_documents",
        "description": "Search through ingested documents using hybrid vector and keyword search",
        "parameters": {
          "type": "object",
          "properties": {
            "query": { "type": "string", "description": "The search query to find relevant documents" },
            "limit": { "type": "integer", "description": "Number of results to return (default: 10)", "default": 10 }
          },
          "required": ["query"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "add_document",
        "description": "Add a new document to the knowledge base",
        "parameters": {
          "type": "object",
          "properties": {
            "content": { "type": "string", "description": "The document content to ingest" },
            "metadata": { "type": "object", "description": "Optional metadata for the document" }
          },
          "required": ["content"]
        }
      }
    }
  ]
}
```

#### Responses API

```json
{
  "tools": [
    {
      "type": "function",
      "name": "search_documents",
      "description": "Search through ingested documents using hybrid vector and keyword search",
      "parameters": {
        "type": "object",
        "properties": {
          "query": { "type": "string", "description": "The search query to find relevant documents" },
          "limit": { "type": "integer", "description": "Number of results to return (default: 10)", "default": 10 }
        },
        "required": ["query"],
        "additionalProperties": false
      },
      "strict": true
    },
    {
      "type": "function",
      "name": "add_document",
      "description": "Add a new document to the knowledge base",
      "parameters": {
        "type": "object",
        "properties": {
          "content": { "type": "string", "description": "The document content to ingest" },
          "metadata": { "type": "object", "description": "Optional metadata for the document" }
        },
        "required": ["content"]
      },
      "strict": true
    }
  ]
}
```

## Performance Tuning

### Search Configuration

- **Vector vs BM25 weights**: Adjust `BM25_WEIGHT` and `VECTOR_WEIGHT`
- **Chunking**: Tune `CHUNK_SIZE` and `CHUNK_OVERLAP` for your documents
- **Reranking**: Enable with `JINA_API_KEY`, then use `RERANK_TOP_K` / `RERANK_MAX_CHARS` to keep Jina calls around ~400 ms
- **Rate Limits**: Set `RATE_LIMIT_PER_MINUTE` per environment to protect the service under load

### Database Optimization

- HNSW indexes for vector search
- Trigram indexes for text search  
- JSON indexes for metadata
- Connection pooling via asyncpg

## Troubleshooting

### Common Issues

1. **Service won't start:**
   ```bash
   make logs  # Check container logs
   make health-detailed  # Check service dependencies
   ```

2. **Database connection issues:**
   ```bash
   docker-compose exec postgres pg_isready -U rag_user -d rag_db
   ```

3. **Vector search not working:**
   - Ensure pgvector extension is installed
   - Check embeddings are being generated
   - Verify HNSW indexes are created

### Performance Issues

1. **Slow search:**
   - Check index usage in query plans
   - Monitor Redis cache hit rates
   - Consider adjusting search parameters and rerank caps (`RERANK_TOP_K`, `RERANK_MAX_CHARS`)

2. **High memory usage:**
   - Reduce batch sizes for embedding generation
   - Tune PostgreSQL memory settings
   - Monitor vector index memory usage

## License

This project is licensed under the MIT License.
