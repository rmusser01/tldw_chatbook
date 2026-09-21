"""A bare DML call must not swallow every later write on its connection.

TASK-32801.1, the core review's P0. ``update_keywords_for_media`` and
``create_document_version`` execute DML on the held connection and document
that they assume an existing transaction. The connection keeps the legacy
isolation level, so the first such call opens an *implicit* transaction that
nothing commits; ``transaction()`` then sees ``conn.in_transaction`` already
True, treats itself as nested, and skips its own commit. Every Media write on
that thread from then on rides the uncommitted transaction and is rolled back
when the connection closes.

The shipped path with no enclosing transaction is a reading-list re-import:

    Media/media_reading_scope_service.py:1799  import_reading_items
    Media/local_media_reading_service.py:2848  import_reading_items
                                        :2873  execute_reading_import_job
                                        :3383  _execute_reading_import_job  (per-row loop)
                                        :3485  _materialize_reading_import_row
                                        :3525  db.update_keywords_for_media(...)

These tests use a file-backed database on purpose: an in-memory one cannot
observe a close-time rollback, which is why the existing Media suites never
caught this.

NOTE ON RUNNING THESE: constructing a file-backed ``MediaDatabase`` currently
raises ``RecoveryRequired('raw_source_selection_changed')`` in a clean
``origin/dev`` worktree -- the existing ``file_db`` fixture in this directory
fails the same way, so this is environmental and not specific to these tests.
They pass in a checkout where that admission state is bound. The fix was
verified directly with the repro in the task notes.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


def _open(path) -> MediaDatabase:
    return MediaDatabase(db_path=str(path), client_id="bare-dml-test")


def _add(db: MediaDatabase, url: str, title: str) -> int:
    """Add a genuinely distinct row.

    Title and content must differ per row: `add_media_with_keywords`
    de-duplicates and returns "already exists. Overwrite not enabled." for a
    repeated title, which looks exactly like a lost write. A first draft of
    this test was wrong for that reason.
    """
    media_id, _uuid, _msg = db.add_media_with_keywords(
        url=url,
        title=title,
        media_type="document",
        content=f"body-{title}",
        keywords=["k1"],
    )
    return media_id


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "media-bare-dml.sqlite"


def test_bare_update_keywords_leaves_no_open_transaction(db_path):
    """The first bare call must not leave an implicit transaction open."""
    db = _open(db_path)
    try:
        media_id = _add(db, "https://example.test/a", "Alpha")
        db.update_keywords_for_media(media_id, ["k1", "k2"])
        assert db.get_connection().in_transaction is False, (
            "a bare DML call left an implicit transaction open; every later "
            "write on this connection will be rolled back at close"
        )
    finally:
        db.close_connection()


def test_writes_after_a_bare_update_survive_close(db_path):
    """The defect as a user meets it: a re-import, then later writes vanish."""
    db = _open(db_path)
    try:
        first_id = _add(db, "https://example.test/a", "Alpha")
        # The re-import path: merge tags into an existing row, no transaction.
        db.update_keywords_for_media(first_id, ["k1", "k2"])
        # Anything written afterwards rode the uncommitted transaction.
        _add(db, "https://example.test/b", "Beta")
    finally:
        db.close_connection()

    reopened = _open(db_path)
    try:
        conn = reopened.get_connection()
        media_rows = conn.execute(
            "SELECT COUNT(*) FROM Media WHERE deleted = 0"
        ).fetchone()[0]
        assert media_rows == 2, (
            f"expected both media rows to survive the close, found {media_rows}"
        )
        keywords = {
            row["keyword"]
            for row in conn.execute(
                "SELECT k.keyword FROM Keywords k "
                "JOIN MediaKeywords mk ON mk.keyword_id = k.id "
                "WHERE mk.media_id = ? AND k.deleted = 0",
                (first_id,),
            ).fetchall()
        }
        assert keywords == {"k1", "k2"}, (
            f"the merged keyword did not survive the close: {sorted(keywords)}"
        )
    finally:
        reopened.close_connection()


def test_bare_create_document_version_leaves_no_open_transaction(db_path):
    """The sibling method carries the same contract and the same defect."""
    db = _open(db_path)
    try:
        media_id = _add(db, "https://example.test/a", "Alpha")
        db.create_document_version(media_id=media_id, content="v2")
        assert db.get_connection().in_transaction is False, (
            "create_document_version left an implicit transaction open"
        )
    finally:
        db.close_connection()


def test_an_enclosing_transaction_still_owns_the_commit(db_path):
    """The fix must not break callers that already open their own transaction.

    ``transaction()`` joins an outer transaction and leaves the commit to it,
    so a caller that wraps these methods must still see all-or-nothing
    behaviour -- `Library/meeting_speaker_rename.py` and
    `local_media_reading_service.py:4875` both rely on this.
    """
    db = _open(db_path)
    try:
        media_id = _add(db, "https://example.test/a", "Alpha")
        with pytest.raises(RuntimeError):
            with db.transaction():
                db.update_keywords_for_media(media_id, ["k1", "rolled-back"])
                raise RuntimeError("caller aborts after the nested write")
        keywords = {
            row["keyword"]
            for row in db.get_connection()
            .execute(
                "SELECT k.keyword FROM Keywords k "
                "JOIN MediaKeywords mk ON mk.keyword_id = k.id "
                "WHERE mk.media_id = ? AND k.deleted = 0",
                (media_id,),
            )
            .fetchall()
        }
        assert "rolled-back" not in keywords, (
            "a nested write survived the outer transaction's rollback"
        )
    finally:
        db.close_connection()
