from psycopg2.extras import Json, RealDictCursor


def save_transcript_to_db(conn, ticker, quarter, raw_json):
    """
    Caches the raw earnings call transcript response (as returned by the
    Alpha Vantage API, unmodified) in the transcripts table.

    Args:
        conn: A psycopg2 connection object to the PostgreSQL database.
        ticker: Stock ticker symbol.
        quarter: Quarter string, e.g. '2026Q1'.
        raw_json: The full dict returned by the Alpha Vantage API call.

    Returns:
        The id of the newly inserted transcripts row.
    """
    with conn.cursor() as cur:
        # transcripts.ticker has a FK to tickers(ticker) — make sure the
        # parent row exists before inserting, and bump last_updated either way.
        cur.execute("""
            INSERT INTO tickers (ticker, last_updated)
            VALUES (%s, now())
            ON CONFLICT (ticker) DO UPDATE SET last_updated = now();
        """, (ticker,))
        cur.execute("""
            INSERT INTO transcripts (ticker, quarter, raw_json)
            VALUES (%s, %s, %s)
            RETURNING id;
        """, (ticker, quarter, Json(raw_json)))
        transcript_id = cur.fetchone()[0]
        conn.commit()
    return transcript_id

def get_transcript_from_db(conn, ticker, quarter):
    """
    Looks up a cached transcript for a given ticker/quarter.

    Args:
        conn: A psycopg2 connection object to the PostgreSQL database.
        ticker: Stock ticker symbol.
        quarter: Quarter string, e.g. '2026Q1'.

    Returns:
        A dict of the transcripts row (including raw_json), or None if not cached.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT * FROM transcripts
            WHERE ticker = %s AND quarter = %s;
        """, (ticker, quarter))
        transcript = cur.fetchone()
    return transcript


def save_chunks_to_db(conn, transcript_id, ticker, quarter, chunks):
    """
    Saves the list of chunks to the database.

    Args:
        conn: A psycopg2 connection object to the PostgreSQL database.
        transcript_id: The id of the transcripts row this chunk belongs to.
        chunks: A list of dicts, each containing the chunk data.

    Returns:
        None
    """
    with conn.cursor() as cur:
        for chunk in chunks:
            cur.execute("""
                INSERT INTO chunks (transcript_id, ticker, quarter, speaker, speaker_role,
                                     is_qa, turn_index, content, embedding, av_sentiment,
                                     sentiment_label, sentiment_positive, sentiment_neutral,
                                     sentiment_negative)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                transcript_id,
                ticker,
                quarter,
                chunk['speaker'],
                chunk['speaker_role'],
                chunk['is_qa'],
                chunk['turn_index'],
                chunk['content'],
                chunk['embedding'],
                chunk['av_sentiment'],
                chunk['sentiment_label'],
                chunk['sentiment_positive'],
                chunk['sentiment_neutral'],
                chunk['sentiment_negative'],
            ))
        conn.commit()

def get_chunks_from_db(conn, ticker, quarter):
    """
    Retrieves all chunks for a given ticker and quarter.

    Args:
        conn: A psycopg2 connection object to the PostgreSQL database.
        ticker: Stock ticker symbol.
        quarter: Quarter string, e.g. '2026Q1'.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT * FROM chunks
            WHERE ticker = %s AND quarter = %s
            ORDER BY turn_index;
        """, (ticker, quarter))
        chunks = cur.fetchall()
    return chunks

def save_guidance_to_db(conn, guidance_items):
    """
    Saves already-extracted guidance records to the database.

    Args:
        conn: A psycopg2 connection object to the PostgreSQL database.
        guidance_items: A list of dicts, each with keys: ticker, quarter,
            metric, value, direction, quote_text, source_chunk_id.

    Returns:
        None
    """
    with conn.cursor() as cur:
        for item in guidance_items:
            cur.execute("""
                INSERT INTO guidance (ticker, quarter, metric, value, direction,
                                       quote_text, source_chunk_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (
                item['ticker'],
                item['quarter'],
                item['metric'],
                item['value'],
                item['direction'],
                item['quote_text'],
                item['source_chunk_id'],
            ))
        conn.commit()