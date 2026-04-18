"""
Fin catalogue — Supabase-backed storage with pgvector nearest-neighbour matching.

Requires the following Supabase setup (run once in the SQL editor):

    -- 1. Extension
    CREATE EXTENSION IF NOT EXISTS vector;

    -- 2. Table
    CREATE TABLE fins (
        id          TEXT PRIMARY KEY,
        label       TEXT NOT NULL,
        embedding   vector(512) NOT NULL,
        image_path  TEXT,
        added_at    TIMESTAMPTZ DEFAULT NOW()
    );

    -- 3. IVFFlat index for fast approximate nearest-neighbour search
    CREATE INDEX fins_embedding_idx
        ON fins USING ivfflat (embedding vector_l2_ops)
        WITH (lists = 100);

    -- 4. Row Level Security — table is private; only the service role can access it.
    --    The service role key bypasses RLS automatically, so the app keeps working.
    --    Anon and authenticated users are denied all access.
    ALTER TABLE fins ENABLE ROW LEVEL SECURITY;

    -- 5. Similarity search function.
    --    SECURITY INVOKER means it runs as the calling role, so RLS is respected.
    CREATE OR REPLACE FUNCTION match_fins(
        query_embedding vector(512),
        match_count     int DEFAULT 10
    )
    RETURNS TABLE (id text, label text, distance float)
    LANGUAGE sql
    SECURITY INVOKER
    AS $$
        SELECT id, label, (embedding <-> query_embedding)::float AS distance
        FROM   fins
        ORDER  BY embedding <-> query_embedding
        LIMIT  match_count;
    $$;
"""

from __future__ import annotations

import os

from supabase import Client, create_client

# Module-level singleton — created once per process.
_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        _client = create_client(url, key)
    return _client


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def add_fin(fin_id: str, label: str, embedding: list[float], image_path: str | None = None) -> None:
    """Insert or replace a fin record in the catalogue."""
    _get_client().table("fins").upsert({
        "id": fin_id,
        "label": label,
        "embedding": embedding,
        "image_path": image_path,
    }).execute()


def delete_fin(fin_id: str) -> bool:
    """Remove a fin from the catalogue. Returns True if a row was deleted."""
    resp = _get_client().table("fins").delete().eq("id", fin_id).execute()
    return len(resp.data) > 0


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def list_fins(limit: int = 200, offset: int = 0) -> list[dict]:
    """Return catalogue entries (without embeddings for brevity)."""
    resp = (
        _get_client()
        .table("fins")
        .select("id,label,image_path,added_at")
        .order("added_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return resp.data


def get_fin(fin_id: str) -> dict | None:
    resp = (
        _get_client()
        .table("fins")
        .select("*")
        .eq("id", fin_id)
        .maybe_single()
        .execute()
    )
    return resp.data


def catalogue_size() -> int:
    resp = _get_client().table("fins").select("id", count="exact").execute()
    return resp.count or 0


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def find_matches(
    query_embedding: list[float],
    top_n: int = 10,
) -> list[dict]:
    """
    pgvector L2 similarity search via a Postgres stored function.

    Returns list of {id, label, distance} sorted ascending by distance.
    """
    resp = _get_client().rpc(
        "match_fins",
        {"query_embedding": query_embedding, "match_count": top_n},
    ).execute()
    return resp.data or []
