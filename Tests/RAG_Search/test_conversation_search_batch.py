"""Tests for task-260: RAG conversation search batch fetch + BLOB skip.

All tests run against a REAL file-backed CharactersRAGDB — no mocks. This is
also the regression net for the flattened-module import defect: the pipeline
function used to do ``from ...DB.ChaChaNotes_DB import ...`` (written when the
module lived one package deeper, in ``simplified/``), which raised
``ImportError: attempted relative import beyond top-level package`` at call
time; any end-to-end call below would resurface that class of break.
"""

import asyncio

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.RAG_Search.pipeline_functions_simple import search_conversations_fts5

_PNG_STUB = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class _FakeApp:
    """Bare app double carrying only the db_config the pipeline reads."""

    def __init__(self, db_path):
        self.db_config = {"chacha_db_path": str(db_path)}


def _seed(db: CharactersRAGDB, *, n_convs: int = 3, msgs_per_conv: int = 4) -> list[str]:
    """Seed conversations whose messages embed a searchable marker word."""
    conv_ids = []
    for c in range(n_convs):
        conv_id = db.add_conversation({"title": f"Conversation {c}"})
        conv_ids.append(conv_id)
        for m in range(msgs_per_conv):
            db.add_message({
                "conversation_id": conv_id,
                "sender": "User" if m % 2 == 0 else "AI",
                "content": f"glimmerfish message {m} of conversation {c}",
                # Explicit distinct timestamps: deterministic ordering.
                "timestamp": f"2026-07-17T10:0{c}:{m:02d}Z",
                # One BLOB-bearing message per conversation proves the
                # text-only path skips (not breaks on) image rows.
                **({"image_data": _PNG_STUB, "image_mime_type": "image/png"} if m == 1 else {}),
            })
    return conv_ids


@pytest.fixture()
def db(tmp_path):
    return CharactersRAGDB(tmp_path / "task260.db", client_id="task-260-test")


def test_batch_matches_per_conversation_loop(db):
    """AC#1/#3: the batch method returns exactly what the old loop fetched."""
    conv_ids = _seed(db)

    batched = db.get_messages_for_conversations_batch(
        conversation_ids=conv_ids, limit_per_conversation=5
    )

    for conv_id in conv_ids:
        loop_rows = db.get_messages_for_conversation(conversation_id=conv_id, limit=5)
        batch_rows = batched[conv_id]
        assert len(batch_rows) == len(loop_rows)
        # Compare on the columns both queries select (the batch query
        # predates the variant columns and doesn't return them).
        common = ("id", "conversation_id", "sender", "content", "timestamp", "image_data")
        for lr, br in zip(loop_rows, batch_rows):
            for key in common:
                assert br[key] == lr[key], (conv_id, key)


def test_include_image_data_false_skips_blob_both_methods(db):
    """AC#2: BLOB column comes back None, mime type and shape survive."""
    conv_ids = _seed(db, n_convs=1)
    conv_id = conv_ids[0]

    with_blob = db.get_messages_for_conversation(conversation_id=conv_id, limit=5)
    without_blob = db.get_messages_for_conversation(
        conversation_id=conv_id, limit=5, include_image_data=False
    )
    batch_without = db.get_messages_for_conversations_batch(
        conversation_ids=[conv_id], limit_per_conversation=5, include_image_data=False
    )[conv_id]

    image_rows = [r for r in with_blob if r["image_data"] is not None]
    assert image_rows, "seed must include a BLOB-bearing message"
    assert image_rows[0]["image_data"] == _PNG_STUB

    for rows in (without_blob, batch_without):
        assert [r["id"] for r in rows] == [r["id"] for r in with_blob]
        assert all(r["image_data"] is None for r in rows)
        # Callers can still tell an image exists.
        assert any(r["image_mime_type"] == "image/png" for r in rows)
        assert [r["content"] for r in rows] == [r["content"] for r in with_blob]


@pytest.mark.asyncio
async def test_search_conversations_fts5_end_to_end(db, tmp_path):
    """AC#3: the real pipeline function returns the same results the old
    per-conversation loop produced — snippet text, titles, and relevance
    order — via one batched query. Also pins the flattened-module import."""
    _seed(db)
    app = _FakeApp(tmp_path / "task260.db")

    results = await search_conversations_fts5(app, "glimmerfish", limit=10)

    assert results, "expected conversation hits for the seeded marker word"
    reference_order = [
        str(c["id"]) for c in db.search_conversations_by_content(
            search_query="glimmerfish", limit=20
        )
    ]
    assert [r.id for r in results] == reference_order[: len(results)]

    for result in results:
        assert result.source == "conversation"
        assert result.title.startswith("Conversation ")
        # Snippet identical to the old loop: first-5 ASC messages joined
        # as "sender: content" lines.
        expected = "\n".join(
            f"{m['sender']}: {m['content']}"
            for m in db.get_messages_for_conversation(conversation_id=result.id, limit=5)
        )
        assert result.content == expected


@pytest.mark.asyncio
async def test_search_conversations_fts5_no_db_config_returns_empty(tmp_path):
    """The production guard: no db_config on the app -> quiet empty result."""

    class _Bare:
        pass

    assert await search_conversations_fts5(_Bare(), "anything") == []
