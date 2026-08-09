"""Per-token FTS5 quoting: multi-token queries must match non-contiguous text.

TASK-3995: `RAGService._escape_fts5_query` wrapped the ENTIRE query in a
single pair of double quotes, which FTS5 parses as a *phrase* query --
every token must appear as one contiguous run. That is strictly stronger
than AND-of-terms, not equivalent to it. Verified directly against a real
corpus document and the exact SQL join the engine uses (see task-3995's
description): the phrase form of a multi-token query matched 0 rows
against a document that contains the relevant terms but not as one
contiguous run.

The whole-query quoting was not an accident -- it existed to keep FTS5
from parsing a bare token as column-filter syntax. The documented
incident: the unquoted token `Obsidian-3` raises
`OperationalError('no such column: 3')` because FTS5's query grammar
treats the bareword after the hyphen specially. Any fix must drop phrase
semantics while keeping that safety net.

These tests build a plain sqlite3 stdlib in-memory FTS5 table (no app
DBs) so the MATCH behavior under test is the real FTS5 engine, not a
mock. `RAGService._escape_fts5_query` itself needs no DB at all -- but
`_make_service` seeds a real media DB anyway, and that is load-bearing
for exactly one test. `test_all_punctuation_query_short_circuits` spies
on `_perform_fts5_search` to prove the production short-circuit fires;
with no `media_db_path` configured, the media sub-leg bails on an
unresolvable database BEFORE reaching that call, so the spy could never
fire and the assertion held whether or not the guard existed. Verified:
with the short-circuit neutered (`if False:`) the unseeded file stayed
5/5 green; seeded, the same mutation yields three `_perform_fts5_search`
calls (one per retry attempt) and reds the test. Do not drop the seed.
"""
import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService


def _seed_media_db(tmp_path):
    """A real MediaDatabase with one row (FTS populated by its own triggers).

    Same pattern as `test_keyword_leg_chacha._seed_media`.
    """
    db_path = tmp_path / "tldw_cli_media_v2.db"
    db = MediaDatabase(db_path=str(db_path), client_id="test_fts5_escaping")
    try:
        media_id, _, message = db.add_media_with_keywords(
            title="Lathe Maintenance Log",
            content="The Obsidian-3 lathe shows spindle runout under load.",
            media_type="article",
            author="Tester",
            url="https://example.com/lathe-maintenance-log",
        )
        assert media_id is not None, f"media seed failed: {message}"
    finally:
        db.close_connection()
    return db_path


def _make_service(tmp_path):
    """A RAGService whose media sub-leg can actually reach FTS5.

    `media_db_path` is seeded deliberately -- see the module docstring:
    without it the media sub-leg returns early and the short-circuit spy
    in `test_all_punctuation_query_short_circuits` is unreachable.
    """
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    cfg.search.media_db_path = Path(_seed_media_db(tmp_path))
    return RAGService(cfg)


def _fts5_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE docs USING fts5(title, content)")
    return conn


def _insert(conn, title, content):
    conn.execute(
        "INSERT INTO docs(title, content) VALUES (?, ?)", (title, content)
    )
    conn.commit()


def _match(conn, query):
    return conn.execute(
        "SELECT rowid FROM docs WHERE docs MATCH ?", (query,)
    ).fetchall()


def test_multi_token_query_matches_non_contiguous_tokens(tmp_path):
    """The relevant tokens are all present but NOT contiguous in the seed
    document -- "Obsidian-3" ... "lathe shows" ... "spindle runout" -- so a
    phrase query over the whole 3-token search string must NOT match it,
    while an AND-of-terms query must.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Lathe Maintenance Log",
        "The Obsidian-3 lathe shows spindle runout under load.",
    )
    query = "Obsidian-3 spindle runout"

    # Sanity check on the OLD behavior: wrapping the whole query in one
    # pair of quotes is a phrase query requiring the 3 tokens contiguous,
    # which they are not in the seed doc.
    old_phrase_form = f'"{query}"'
    assert _match(conn, old_phrase_form) == [], (
        "test setup invalid: the seed doc must NOT satisfy the old "
        "whole-query phrase form"
    )

    service = _make_service(tmp_path)
    escaped = service._escape_fts5_query(query)
    assert escaped, "escaped query must not be empty for a real multi-token query"

    rows = _match(conn, escaped)
    assert len(rows) == 1, (
        f"per-token AND-of-terms form must match the non-contiguous seed "
        f"doc; _escape_fts5_query({query!r}) = {escaped!r}, rows={rows}"
    )
    conn.close()


def test_hyphen_numeric_token_still_safe(tmp_path):
    """A bare hyphenated-numeric token is unsafe unquoted -- FTS5's query
    grammar raises `OperationalError('no such column: 3')` on it (the
    documented incident this task's safety property guards against). The
    escaped form must execute cleanly and still match the seed doc.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Lathe Maintenance Log",
        "The Obsidian-3 lathe was serviced today.",
    )
    query = "Obsidian-3"

    # Sanity check: the raw, unquoted token really is unsafe.
    with pytest.raises(sqlite3.OperationalError):
        _match(conn, query)

    service = _make_service(tmp_path)
    escaped = service._escape_fts5_query(query)
    rows = _match(conn, escaped)
    assert len(rows) == 1, f"escaped={escaped!r} rows={rows}"
    conn.close()


def test_embedded_quotes_are_doubled_and_safe(tmp_path):
    """A query containing a raw double-quote character must not break the
    FTS5 MATCH expression -- the quote is doubled per FTS5's own escaping
    convention for a literal quote inside a quoted term.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Quote Test",
        'The technician wrote "runout confirmed" in the log.',
    )
    query = 'confirmed" runout'

    service = _make_service(tmp_path)
    escaped = service._escape_fts5_query(query)

    # Must execute without raising -- the safety property under test.
    rows = _match(conn, escaped)
    assert rows == [(1,)], f"escaped={escaped!r} rows={rows}"
    conn.close()


def test_single_token_behavior_unchanged(tmp_path):
    """A single-token query is a degenerate case where phrase quoting and
    per-token quoting produce the identical MATCH expression -- this must
    keep working exactly as before.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Wombat Field Notes",
        "Field notes mentioning wombat behavior in the wild.",
    )

    service = _make_service(tmp_path)
    escaped = service._escape_fts5_query("wombat")
    assert escaped == '"wombat"'

    rows = _match(conn, escaped)
    assert len(rows) == 1
    conn.close()


def test_all_punctuation_query_short_circuits(tmp_path):
    """A query with no alphanumeric content (FTS5's default tokenizer
    indexes nothing for it) must escape to "" -- and the keyword search
    must recognize "" as "no results" and skip the FTS5 call entirely
    rather than execute a query that can only ever match nothing.
    """
    service = _make_service(tmp_path)
    assert service._escape_fts5_query("!!! ...") == ""

    import asyncio

    calls = []
    original_perform = service._perform_fts5_search

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original_perform(*args, **kwargs)

    service._perform_fts5_search = _spy

    results = asyncio.run(service._keyword_search("!!! ...", top_k=5))
    assert results == []
    assert calls == [], (
        "an all-punctuation query must short-circuit before the FTS5 "
        f"call is ever made; calls={calls}"
    )
