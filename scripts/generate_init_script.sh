#!/bin/bash
set -e

# Execute dynamic database initialization directly
echo "Initializing database with environment variables..."
echo "POSTGRES_USER: $POSTGRES_USER"
echo "POSTGRES_DB: $POSTGRES_DB"

# Connect to database and execute the initialization SQL directly
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
-- Ensure required extensions are available
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Verify user exists
DO \$\$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$POSTGRES_USER') THEN
        RAISE EXCEPTION 'User $POSTGRES_USER does not exist. Environment variables not processed correctly.';
    ELSE
        RAISE NOTICE 'User $POSTGRES_USER exists - continuing with setup';
    END IF;
END
\$\$;

-- Ensure user has necessary permissions
GRANT CONNECT ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;
GRANT USAGE ON SCHEMA public TO $POSTGRES_USER;
GRANT CREATE ON SCHEMA public TO $POSTGRES_USER;
GRANT ALL PRIVILEGES ON SCHEMA public TO $POSTGRES_USER;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding HALFVEC(3072),  -- halfvec type for OpenAI text-embedding-3-large (up to 4000 dimensions)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE  -- Soft delete support
);

-- Create embedding cache table
CREATE TABLE IF NOT EXISTS embedding_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_hash VARCHAR(64) UNIQUE NOT NULL,
    embedding HALFVEC(3072),
    model VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 1
);

-- Create indexes for performance
-- Vector similarity index (HNSW for halfvec - supports up to 4000 dimensions)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw 
ON documents USING hnsw (embedding halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search index (trigram for fuzzy matching)
CREATE INDEX IF NOT EXISTS idx_documents_content_trgm
ON documents USING gin (content gin_trgm_ops);

-- Metadata queries index
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin
ON documents USING gin (metadata);

-- Active documents index (soft delete aware)
CREATE INDEX IF NOT EXISTS idx_documents_active
ON documents (created_at) 
WHERE deleted_at IS NULL;

-- Embedding cache indexes
CREATE INDEX IF NOT EXISTS idx_embedding_cache_hash
ON embedding_cache (text_hash);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_model
ON embedding_cache (model);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_accessed
ON embedding_cache (last_accessed);

-- Grant permissions on tables to user
GRANT ALL PRIVILEGES ON TABLE documents TO $POSTGRES_USER;
GRANT ALL PRIVILEGES ON TABLE embedding_cache TO $POSTGRES_USER;

-- Grant sequence permissions (for auto-generated IDs)
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO $POSTGRES_USER;
EOSQL

echo "Database initialization completed successfully"