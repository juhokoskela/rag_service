"""Initial RAG database schema with pgvector support

Revision ID: 20250905_1200_initial_schema
Revises: 
Create Date: 2025-09-05 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision = '20250905_1200_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable required PostgreSQL extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    
    # Create documents table with vector embeddings
    op.create_table(
        'documents',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('metadata', sa.JSON, nullable=False, server_default='{}'),
        sa.Column('embedding', sa.VARCHAR, nullable=True),  # Will store vector as string initially
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Add vector column with proper type (3072 dimensions for text-embedding-3-large)
    op.execute('ALTER TABLE documents ALTER COLUMN embedding TYPE vector(3072) USING embedding::vector(3072)')
    
    # Create embedding cache table for performance
    op.create_table(
        'embedding_cache',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('text_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('embedding', sa.VARCHAR, nullable=False),  # Will convert to vector type
        sa.Column('model', sa.String(100), nullable=False, server_default='text-embedding-3-large'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('access_count', sa.Integer, nullable=False, server_default=sa.text('0')),
    )
    
    # Convert embedding cache column to vector type
    op.execute('ALTER TABLE embedding_cache ALTER COLUMN embedding TYPE vector(3072) USING embedding::vector(3072)')
    
    # Create optimal indexes
    # HNSW index for vector similarity search (most efficient for high-dimensional vectors)
    op.execute('CREATE INDEX documents_embedding_hnsw_idx ON documents USING hnsw (embedding vector_cosine_ops)')
    
    # Trigram index for full-text search support
    op.execute('CREATE INDEX documents_content_trgm_idx ON documents USING gin (content gin_trgm_ops)')
    
    # Metadata search support (GIN index for JSON)
    op.execute('CREATE INDEX documents_metadata_idx ON documents USING gin (metadata)')
    op.execute('CREATE INDEX documents_deleted_at_idx ON documents (deleted_at)')
    
    # Embedding cache indexes
    op.execute('CREATE INDEX embedding_cache_embedding_hnsw_idx ON embedding_cache USING hnsw (embedding vector_cosine_ops)')
    op.execute('CREATE INDEX embedding_cache_model_idx ON embedding_cache (model)')
    op.execute('CREATE INDEX embedding_cache_model_text_hash_idx ON embedding_cache (model, text_hash)')
    
    # Create trigger for updated_at timestamp
    op.execute('''
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    ''')
    
    op.execute('''
        CREATE TRIGGER update_documents_updated_at
        BEFORE UPDATE ON documents
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    ''')


def downgrade() -> None:
    # Drop triggers and functions
    op.execute('DROP TRIGGER IF EXISTS update_documents_updated_at ON documents')
    op.execute('DROP FUNCTION IF EXISTS update_updated_at_column()')
    
    # Drop tables (indexes will be dropped automatically)
    op.drop_table('embedding_cache')
    op.drop_table('documents')
    
    # Note: We don't drop extensions as they might be used by other applications
