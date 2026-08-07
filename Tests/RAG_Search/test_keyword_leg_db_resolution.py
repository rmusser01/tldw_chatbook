"""The keyword leg must use the configured media DB and never write.

Today it guesses paths (media_db.db / chacha_notes.db) that can never match
the real tldw_cli_media_v2.db, opens the ChaChaNotes DB with media-schema
SQL, and on a total miss CREATES a MediaDatabase as a search side effect.
"""
import asyncio
from pathlib import Path
import pytest

from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService


def _make_service(tmp_path, media_db_path=None):
    """Build a RAGService with the in-memory vector store and mock (offline,
    deterministic) embeddings backend -- same pattern as
    Tests/RAG/test_ingestion_indexing.py's `_make_real_service`.
    """
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.chunking.chunk_size = 60
    cfg.chunking.chunk_overlap = 10
    cfg.search.enable_cache = False
    if media_db_path is not None:
        cfg.search.media_db_path = Path(media_db_path)
    return RAGService(cfg)


def test_missing_media_db_returns_empty_and_creates_nothing(tmp_path):
    service = _make_service(tmp_path, media_db_path=tmp_path / "absent.db")
    results = asyncio.run(service._keyword_search("anything", top_k=5))
    assert results == []
    assert not (tmp_path / "absent.db").exists()          # no create-on-miss
    assert list(tmp_path.glob("*.db")) == []              # no rogue DB anywhere


def test_keyword_rows_carry_media_source_type(tmp_path):
    db_path = tmp_path / "tldw_cli_media_v2.db"
    # Create a real MediaDatabase and insert one item via its public API
    # (add_media_with_keywords) so media_fts is populated by triggers.
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    db = MediaDatabase(db_path=str(db_path), client_id="test_keyword_leg")
    media_id, media_uuid, message = db.add_media_with_keywords(
        title="Wombat Field Notes",
        content="Field notes mentioning wombat behavior in the wild.",
        media_type="article",
        author="J. Naturalist",
        url="https://example.com/wombat-field-notes",
    )
    assert media_id is not None, f"seed failed: {message}"
    db.close_connection()

    service = _make_service(tmp_path, media_db_path=db_path)
    results = asyncio.run(service._keyword_search("wombat", top_k=5))
    assert results, "FTS row expected"
    assert results[0].metadata["source_type"] == "media"


def test_default_resolution_uses_get_media_db_path(monkeypatch, tmp_path):
    sentinel = tmp_path / "sentinel_media.db"
    import tldw_chatbook.config as cfg
    monkeypatch.setattr(cfg, "get_media_db_path", lambda **kw: sentinel)
    service = _make_service(tmp_path)  # no explicit path configured
    results = asyncio.run(service._keyword_search("anything", top_k=5))
    assert results == []                                   # sentinel absent
    assert not sentinel.exists()                           # still no writes
