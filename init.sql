CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE tickers (
    ticker        TEXT PRIMARY KEY,
    last_updated  TIMESTAMPTZ
);

CREATE TABLE transcripts (
    id           SERIAL PRIMARY KEY,
    ticker       TEXT NOT NULL REFERENCES tickers(ticker),
    quarter      TEXT NOT NULL,          -- e.g. '2025Q2'
    raw_json     JSONB NOT NULL,          -- cached Alpha Vantage response
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ticker, quarter)              -- enforces cache-aside at the DB level
);

CREATE TABLE chunks (
    id                  SERIAL PRIMARY KEY,
    transcript_id       INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    ticker              TEXT NOT NULL,    -- denormalized, see note below
    quarter             TEXT NOT NULL,
    speaker             TEXT,
    speaker_role        TEXT,
    is_qa               BOOLEAN NOT NULL DEFAULT false,
    turn_index          INTEGER NOT NULL, -- position within the call
    content             TEXT NOT NULL,
    embedding           VECTOR(1536),      -- match your embedding model's output dim
    av_sentiment        REAL,             -- cached Alpha Vantage sentiment response
    sentiment_label     TEXT,              -- from your Lambda
    sentiment_positive  REAL,
    sentiment_neutral   REAL,
    sentiment_negative  REAL
);

CREATE TABLE guidance (
    id               SERIAL PRIMARY KEY,
    ticker           TEXT NOT NULL,
    quarter          TEXT NOT NULL,
    metric           TEXT NOT NULL,        -- e.g. 'revenue_growth_guidance'
    value            TEXT,                  -- kept as text; guidance isn't always a clean number
    direction        TEXT,                  -- raised / cut / reiterated, filled in later
    quote_text       TEXT NOT NULL,
    source_chunk_id  INTEGER REFERENCES chunks(id),   -- the citation link
    extracted_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chat_messages (
    id          SERIAL PRIMARY KEY,
    ticker      TEXT NOT NULL REFERENCES tickers(ticker),
    role        TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chunks_ticker_quarter ON chunks (ticker, quarter);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_guidance_ticker_quarter ON guidance (ticker, quarter);
CREATE INDEX idx_chat_messages_ticker ON chat_messages (ticker, created_at);