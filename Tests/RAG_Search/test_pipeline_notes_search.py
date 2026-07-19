"""Tests for task-295: RAG pipeline notes search + DB resolution seams.

``search_notes_fts5`` used to import a ``Notes_DB`` module that does not
exist anywhere in the tree and call an API shape (``user_id=``) the real
store never had — invisible to mocked tests, so everything here runs
against a REAL CharactersRAGDB. Also covers the task-295 wiring decision:
the app's live ``chachanotes_db`` instance is preferred, with the
``db_config['chacha_db_path']`` construction seam kept for tests/probes.
"""

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.RAG_Search.pipeline_functions_simple import (
    _resolve_chacha_db,
    search_conversations_fts5,
    search_notes_fts5,
)


class _LiveDbApp:
    """App double exposing only the live-instance seam."""

    def __init__(self, db: CharactersRAGDB):
        self.chachanotes_db = db


class _PathApp:
    """App double exposing only the db_config construction seam."""

    def __init__(self, db_path):
        self.db_config = {"chacha_db_path": str(db_path)}


@pytest.fixture()
def db(tmp_path):
    return CharactersRAGDB(tmp_path / "task295.db", client_id="task-295-test")


def _seed_notes(db: CharactersRAGDB) -> list[str]:
    ids = []
    for i in range(3):
        note_id = db.add_note(
            title=f"Moon note {i}",
            content=f"starlight observation {i}: tides follow the moon.",
        )
        ids.append(note_id)
    db.add_note(title="Unrelated", content="grocery list: bread, eggs.")
    return ids


@pytest.mark.asyncio
async def test_search_notes_fts5_end_to_end_live_instance(db):
    """The rewired notes search returns real FTS hits via the live-DB seam."""
    seeded = _seed_notes(db)

    results = await search_notes_fts5(_LiveDbApp(db), "starlight", limit=10)

    assert {r.id for r in results} == set(seeded)
    for r in results:
        assert r.source == "note"
        assert r.title.startswith("Moon note")
        assert "starlight observation" in r.content
        assert r.metadata["created_at"] is not None
        assert r.metadata["last_modified"] is not None


@pytest.mark.asyncio
async def test_search_notes_fts5_end_to_end_db_config_path(db, tmp_path):
    """The db_config construction seam still works for notes."""
    seeded = _seed_notes(db)

    results = await search_notes_fts5(_PathApp(tmp_path / "task295.db"), "starlight")

    assert {r.id for r in results} == set(seeded)


@pytest.mark.asyncio
async def test_search_conversations_fts5_uses_live_instance(db):
    """The conversations search prefers the live instance too (no ctor)."""
    conv_id = db.add_conversation({"title": "Live seam conversation"})
    db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "User",
            "content": "glimmerfish sighting confirmed",
        }
    )

    results = await search_conversations_fts5(_LiveDbApp(db), "glimmerfish")

    assert [r.id for r in results] == [conv_id]
    assert "glimmerfish sighting confirmed" in results[0].content


def test_resolver_prefers_live_instance_over_db_config(db, tmp_path):
    """When both seams are present the live instance wins (no reconstruction)."""

    class _BothApp:
        chachanotes_db = db
        db_config = {"chacha_db_path": str(tmp_path / "other.db")}

    assert _resolve_chacha_db(_BothApp()) is db
    # And the other.db file was never created by a fallback construction.
    assert not (tmp_path / "other.db").exists()


@pytest.mark.asyncio
async def test_search_notes_fts5_no_seams_returns_empty():
    """No live instance and no db_config -> quiet empty result."""

    class _Bare:
        pass

    assert await search_notes_fts5(_Bare(), "anything") == []
