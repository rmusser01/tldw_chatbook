"""task-34399 item 1: ``list_notes`` preview projection.

The Library read path already ships ``substr(content, 1, ?) AS preview``
(``list_library_notes_page``); the older ``list_notes`` surface kept
transferring every full note body. It now accepts ``preview_chars`` --
matching the library path's naming -- while the default (``None``)
preserves the historical full-row behavior for untouched callers.
"""

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

PREVIEW_CHARS = 241  # same cap the library list surface uses

_LONG_CONTENT = "alpha " * 100  # 600 chars -- comfortably past the preview cut


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(tmp_path / "notes-preview.db", client_id="preview-test")


def test_preview_mode_returns_preview_not_full_content(db):
    note_id = db.add_note("Long note", _LONG_CONTENT)

    rows = db.list_notes(limit=10, preview_chars=PREVIEW_CHARS)

    assert [row["id"] for row in rows] == [note_id]
    row = rows[0]
    # The body is projected, not shipped: a `preview` key instead of `content`.
    assert "content" not in row
    assert row["preview"] == _LONG_CONTENT[:PREVIEW_CHARS]


def test_default_call_still_returns_full_bodies(db):
    db.add_note("Body note", "the entire body, untouched")

    rows = db.list_notes()

    assert rows[0]["content"] == "the entire body, untouched"
    assert "preview" not in rows[0]


def test_preview_mode_keeps_row_metadata_and_excludes_deleted(db):
    keep_id = db.add_note("Keep", _LONG_CONTENT)
    gone_id = db.add_note("Gone", _LONG_CONTENT)
    gone_version = db.get_note_by_id(gone_id)["version"]
    assert db.soft_delete_note(gone_id, gone_version) is True

    rows = db.list_notes(limit=10, preview_chars=PREVIEW_CHARS)

    assert [row["id"] for row in rows] == [keep_id]
    for key in (
        "id",
        "title",
        "created_at",
        "last_modified",
        "deleted",
        "client_id",
        "version",
    ):
        assert key in rows[0]


@pytest.mark.parametrize("value", [None, 0])
def test_non_positive_preview_chars_fall_back_to_full_content(db, value):
    db.add_note("Body", "full body here")

    rows = db.list_notes(preview_chars=value)

    assert rows[0]["content"] == "full body here"
