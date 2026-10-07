"""B1: the conversations keyword sub-leg's content assembly is bounded, and
citation spans scan only the preview they will be cited against.

Review-B finding: ``_chacha_conversations_fts`` materialized EVERY matching
message of every matching conversation (one ``sender: content`` line per
row) before prompt assembly ever truncated the text, and the citation
builder's token regex scanned that same unbounded text even though only the
first 1000 characters are ever shown. A conversation with thousands of
matching messages paid for all of them on every search.

The bounds are a RETRIEVAL-PRESENTATION bound, not a recall change: the FTS
matching set is unchanged, the conversations returned are unchanged, and
within a conversation the retained messages are the OLDEST ones in the
ORM's own ``timestamp ASC`` order (with rowid as the deterministic
tie-break), so the document still reads chronologically.

Fixture conventions mirror ``test_keyword_leg_chacha.py``: a real
``CharactersRAGDB`` builds the schema (``messages_fts`` populated by its
triggers); the engine reads it through its own read-only connection.
"""

import asyncio

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.RAG_Search.simplified import rag_service as rs
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

# RAGService/RAGConfig construction reads the guarded config loader
# (``get_user_data_dir`` for the default chroma dir). Under the per-test
# TLDW_CONFIG_PATH redirect that read fails config-participant admission
# (RecoveryRequired("raw_source_selection_changed")) -- the same admission
# signature conftest.py documents for test_hosted_chat.py and friends. These
# tests never assert on sandboxed config contents (their databases are
# tmp_path-backed), so they keep the collection-time bootstrap profile.
pytestmark = pytest.mark.bootstrap_profile


def _make_service(chachanotes_db_path):
    """A RAGService pointed at the fixture DB (same recipe as
    ``test_keyword_leg_chacha._make_service``; only the chacha path matters
    here, so no media DB is seeded)."""
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    cfg.search.chachanotes_db_path = chachanotes_db_path
    return RAGService(cfg)


def _seed_conversation(db, title, messages):
    """One conversation, ``messages`` = list of ``(marker, timestamp)``.

    Each message's content embeds its marker (``alpha msg-0007 ...``) so a
    test can assert the RETAINED lines' chronological order by number.
    """
    conv_id = db.add_conversation({"title": title})
    assert conv_id, "conversation seed failed"
    for marker, timestamp in messages:
        assert db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "user",
                "content": f"alpha msg-{marker:04d} body text for the bound test",
                "timestamp": timestamp,
            }
        ), f"message seed failed at marker {marker}"
    return conv_id


def _conversation_docs(service, db_path, query, limit):
    """Run the raw sub-leg exactly as ``_chacha_fts_rows`` drives it."""
    conn = service._connect_chacha_readonly(db_path)
    try:
        expression = service._fts5_match_expressions(query)[0]
        return RAGService._chacha_conversations_fts(conn, expression, limit)
    finally:
        conn.close()


def _marker_of(line: str) -> int:
    """Extract the embedded ``msg-NNNN`` marker from a rendered line."""
    token = next(part for part in line.split() if part.startswith("msg-"))
    return int(token.split("-")[1])


@pytest.fixture
def bounded_conversation_db(tmp_path):
    """One conversation, 500 matching messages, inserted ANTI-chronologically.

    The highest-marker (newest) message is stored first, so insertion order
    and the ORM's ``timestamp ASC`` read order disagree -- the same trick
    ``test_conversation_messages_are_aggregated_chronologically`` uses to
    keep the ordering assertion honest.
    """
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path, "test_conv_bound")
    messages = [
        (i, f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}.000Z") for i in range(500)
    ]
    _seed_conversation(db, "Alpha Flood", list(reversed(messages)))
    db.close_connection()
    return db_path


@pytest.fixture
def many_conversations_db(tmp_path):
    """Ten conversations x 60 matching messages each (600 rows total)."""
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path, "test_conv_bound_total")
    for c in range(10):
        messages = [
            (
                c * 1000 + i,
                f"2026-01-0{c + 1}T00:{i // 60:02d}:{i % 60:02d}.000Z",
            )
            for i in range(60)
        ]
        _seed_conversation(db, f"Alpha Chat {c}", list(reversed(messages)))
    db.close_connection()
    return db_path


def test_content_leg_caps_messages_per_conversation(bounded_conversation_db):
    """500 matching messages must materialize at most the per-conversation
    cap, oldest first, from a single conversation row."""
    service = _make_service(bounded_conversation_db)
    docs = _conversation_docs(service, bounded_conversation_db, "alpha", limit=5)

    assert len(docs) == 1, "the conversation sub-leg returns one row per conversation"
    doc = docs[0]
    joined_lines = doc["content"].split("\n")
    assert len(joined_lines) <= rs._MAX_CONV_MESSAGES_PER_CONVERSATION, (
        f"content leg materialized {len(joined_lines)} messages for one "
        f"conversation (cap {rs._MAX_CONV_MESSAGES_PER_CONVERSATION})"
    )
    # Deterministic ordering: the retained window is the OLDEST messages in
    # timestamp order, so the first retained marker < the last retained one.
    markers = [_marker_of(line) for line in joined_lines]
    assert markers == sorted(markers), f"retained lines out of order: {markers[:5]}..."


def test_content_leg_caps_total_messages_across_conversations(many_conversations_db):
    """Ten conversations x 60 matches = 600 materialized lines before; the
    total cap must hold regardless of how many conversations matched."""
    service = _make_service(many_conversations_db)
    docs = _conversation_docs(service, many_conversations_db, "alpha", limit=10)

    assert len(docs) == 10
    # Empty docs (conversations that lost their window to the total cap)
    # contribute no lines; "".split("\n") would count a phantom one.
    total_lines = sum(len(d["content"].split("\n")) for d in docs if d["content"])
    assert total_lines <= rs._MAX_CONV_MESSAGES_TOTAL, (
        f"content leg materialized {total_lines} lines across conversations "
        f"(total cap {rs._MAX_CONV_MESSAGES_TOTAL})"
    )
    # Every conversation that fit under the running total keeps its lines;
    # conversations are filled in the sub-leg's existing top-k order, so the
    # FIRST conversations keep theirs and later ones may come back empty.
    for doc in docs:
        lines = doc["content"].split("\n") if doc["content"] else []
        if len(lines) >= 2:
            assert _marker_of(lines[0]) < _marker_of(lines[-1]), (
                f"retained lines out of order in {doc['title']!r}"
            )


def test_citation_spans_scan_preview_only():
    """``_keyword_citation_spans`` with ``text_limit`` scans only the prefix
    it is told to; the default still scans the whole text (existing callers
    unchanged)."""
    content = "x" * 500 + "needle" + "y" * 4300 + "needle"

    bounded = RAGService._keyword_citation_spans(content, ["needle"], text_limit=1000)
    assert len(bounded) == 1, f"preview scan found a span past the limit: {bounded}"
    assert bounded[0][1] <= 1000

    # A token STRADDLING the boundary is not matched at all: the scan runs
    # on exactly the first ``text_limit`` characters.
    straddling = "x" * 997 + "needle"
    assert (
        RAGService._keyword_citation_spans(straddling, ["needle"], text_limit=1000)
        == []
    )

    # Default behavior unchanged: both spans, whole text.
    unbounded = RAGService._keyword_citation_spans(content, ["needle"])
    assert len(unbounded) == 2


@pytest.mark.asyncio
async def test_conversation_citations_scan_preview_only(
    bounded_conversation_db, monkeypatch
):
    """End to end: the conversation-content caller passes the preview length,
    so the token regex never sees the full assembled document."""
    seen = []
    real = RAGService._keyword_citation_spans

    def spy(content, tokens, text_limit=None):
        seen.append({"len": len(content), "text_limit": text_limit})
        return real(content, tokens, text_limit=text_limit)

    monkeypatch.setattr(RAGService, "_keyword_citation_spans", staticmethod(spy))

    service = _make_service(bounded_conversation_db)
    # ~500 x ~45 chars = far more than the 1000-char preview, every message
    # matching "alpha".
    results = await service._keyword_search("alpha", top_k=5)

    conversation_rows = [
        r for r in results if r.metadata.get("source_type") == "conversation"
    ]
    assert conversation_rows, "the conversation sub-leg returned no rows"
    assert seen, "citation spans were never built for the conversation rows"
    for call in seen:
        assert call["text_limit"] == 1000, (
            "conversation citation scan must be bounded to the "
            f"[:1000] preview, got text_limit={call['text_limit']} "
            f"over {call['len']} chars"
        )
    for row in conversation_rows:
        assert row.citations, "a matched conversation row lost its citations"
        for citation in row.citations:
            assert citation.end_char <= 1000, (
                f"citation span past the preview: {citation.end_char}"
            )
