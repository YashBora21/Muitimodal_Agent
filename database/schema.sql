CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

ALTER TABLE messages ALTER COLUMN created_at SET DEFAULT clock_timestamp();

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'messages' AND column_name = 'position'
    ) THEN
        EXECUTE 'UPDATE messages SET created_at = created_at + position * INTERVAL ''1 microsecond''';
        EXECUTE 'ALTER TABLE messages DROP COLUMN position';
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    asset_key VARCHAR(50) NOT NULL,
    kind VARCHAR(20) NOT NULL CHECK (kind IN ('image', 'pdf', 'audio', 'youtube')),
    original_name TEXT NOT NULL,
    content_text TEXT,
    raw_data BYTEA,
    file_size_bytes BIGINT CHECK (file_size_bytes >= 0),
    processing_status VARCHAR(20) NOT NULL DEFAULT 'ready'
        CHECK (processing_status IN ('pending', 'processing', 'ready', 'failed')),
    processing_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (thread_id, asset_key)
);



CREATE TABLE IF NOT EXISTS asset_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    page_number INTEGER CHECK (page_number IS NULL OR page_number >= 1),
    embedding VECTOR(384),
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    UNIQUE (asset_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_threads_user_updated ON threads(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_thread_created ON messages(thread_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_assets_thread ON assets(thread_id);
CREATE INDEX IF NOT EXISTS idx_chunks_asset ON asset_chunks(asset_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON asset_chunks USING gin(tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON asset_chunks USING hnsw (embedding vector_cosine_ops);
