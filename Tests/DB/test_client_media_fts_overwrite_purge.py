"""Overwriting a Media row must purge its stale FTS5 tokens (TASK-38588).

``media_fts`` is an external-content FTS5 table (``content='Media'``,
``content_rowid='id'``).  For such tables, ``INSERT OR REPLACE`` never
removes the previous generation of tokens -- it only adds the new ones --
so every title/content overwrite left the OLD words searchable forever
(Library full-text search kept returning the item for words that no
longer existed anywhere in the row).

The fix follows the external-content idiom already used by the
ChaChaNotes_DB triggers: before re-inserting a rowid's new values, issue
the special FTS5 'delete' command with the TRUE previous values.

These tests run against a real in-memory SQLite DB with the production
schema (a genuine ``MediaDatabase``), drive the public write paths that
change title/content, and assert both directions:

* words only present in the OLD content/title no longer MATCH;
* words only present in the NEW content/title still MATCH the row;

plus an FTS5 integrity-check, since a 'delete' issued with values that
were not the previously indexed ones corrupts the index instead of
purging it.
"""

from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


@pytest.fixture()
def db() -> MediaDatabase:
    database = MediaDatabase(":memory:", client_id="fts-purge-test")
    yield database
    database.close_connection()


def _match_rowids(db: MediaDatabase, word: str) -> set[int]:
    """Rowids the media_fts index reports for a single-token MATCH."""
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT rowid FROM media_fts WHERE media_fts MATCH ?", (word,)
    ).fetchall()
    return {row["rowid"] if isinstance(row, sqlite3.Row) else row[0] for row in rows}


def _assert_fts_intact(db: MediaDatabase) -> None:
    """FTS5 integrity-check must pass (a wrong-valued 'delete' corrupts)."""
    conn = db.get_connection()
    conn.execute("INSERT INTO media_fts(media_fts, rank) VALUES('integrity-check', 1)")


class TestAddMediaOverwritePurgesStaleTokens:
    def test_full_content_overwrite_removes_old_content_tokens(self, db):
        """Case A.1.b (overwrite with different content): old words gone."""
        media_id, _, _ = db.add_media_with_keywords(
            url="ftspurge://doc", title="Alpha Report", content="kangaroo wallaby"
        )
        assert media_id is not None
        assert _match_rowids(db, "kangaroo") == {media_id}

        same_id, _, _ = db.add_media_with_keywords(
            url="ftspurge://doc",
            title="Alpha Report",
            content="zebra quagga",
            overwrite=True,
        )
        assert same_id == media_id

        # Words only in the OLD content must no longer match...
        assert _match_rowids(db, "kangaroo") == set()
        assert _match_rowids(db, "wallaby") == set()
        # ...while the NEW content words match exactly this row.
        assert _match_rowids(db, "zebra") == {media_id}
        assert _match_rowids(db, "quagga") == {media_id}
        _assert_fts_intact(db)

    def test_full_content_overwrite_removes_old_title_tokens(self, db):
        """Case A.1.b: the title column drifts exactly like content."""
        media_id, _, _ = db.add_media_with_keywords(
            url="ftspurge://title", title="Alpha Title", content="shared body"
        )
        assert _match_rowids(db, "alpha") == {media_id}

        db.add_media_with_keywords(
            url="ftspurge://title",
            title="Bravo Title",
            content="different body",
            overwrite=True,
        )

        assert _match_rowids(db, "alpha") == set()
        assert _match_rowids(db, "bravo") == {media_id}
        _assert_fts_intact(db)

    def test_metadata_only_update_removes_old_title_tokens(self, db):
        """Case A.1.a (identical content, new title): old title word gone."""
        media_id, _, _ = db.add_media_with_keywords(
            url="ftspurge://meta",
            title="Alpha Metadata",
            content="unchanged body text",
        )
        assert _match_rowids(db, "alpha") == {media_id}

        same_id, _, _ = db.add_media_with_keywords(
            url="ftspurge://meta",
            title="Bravo Metadata",
            content="unchanged body text",
            overwrite=True,
        )
        assert same_id == media_id

        assert _match_rowids(db, "alpha") == set()
        assert _match_rowids(db, "bravo") == {media_id}
        # Content was untouched and stays searchable.
        assert _match_rowids(db, "unchanged") == {media_id}
        _assert_fts_intact(db)


class TestMetadataAndRollbackPaths:
    def test_update_media_metadata_purges_old_title_tokens(self, db):
        """``update_media_metadata`` title change purges the old tokens."""
        media_id, _, _ = db.add_media_with_keywords(
            url="ftspurge://updmeta", title="Alpha Upd", content="steady content"
        )
        assert _match_rowids(db, "alpha") == {media_id}

        ok, _message = db.update_media_metadata(media_id, title="Bravo Upd")
        assert ok

        assert _match_rowids(db, "alpha") == set()
        assert _match_rowids(db, "bravo") == {media_id}
        assert _match_rowids(db, "steady") == {media_id}
        _assert_fts_intact(db)

    def test_rollback_to_previous_version_purges_rolled_away_tokens(self, db):
        """``rollback_media_content`` re-indexes without the rolled-away words."""
        media_id, _, _ = db.add_media_with_keywords(
            url="ftspurge://rollback", title="Rollback Doc", content="generation one"
        )
        db.add_media_with_keywords(
            url="ftspurge://rollback",
            title="Rollback Doc",
            content="generation two",
            overwrite=True,
        )
        assert _match_rowids(db, "two") == {media_id}

        result = db.rollback_to_version(media_id, 1)
        assert "success" in result

        # The rolled-away generation's word is gone...
        assert _match_rowids(db, "two") == set()
        # ...the rolled-back generation's word matches again.
        assert _match_rowids(db, "one") == {media_id}
        _assert_fts_intact(db)


class TestRepeatedOverwritesStayClean:
    def test_three_generation_chain_leaves_only_the_last(self, db):
        """Chained overwrites accumulate no drift across generations."""
        url = "ftspurge://chain"
        for i, word in enumerate(("first", "second", "third")):
            media_id, _, _ = db.add_media_with_keywords(
                url=url, title=f"Chain {word}", content=f"body {word}", overwrite=i > 0
            )

        assert media_id is not None
        for stale in ("first", "second"):
            assert _match_rowids(db, stale) == set()
        assert _match_rowids(db, "third") == {media_id}
        _assert_fts_intact(db)

    def test_soft_delete_then_undelete_keeps_fts_consistent(self, db):
        """Unchanged-value reinsert paths (undelete) stay correct."""
        media_id, _, _ = db.add_media_with_keywords(
            url="ftspurge://undelete", title="Alpha Live", content="resurrect me"
        )
        assert db.soft_delete_media(media_id)
        assert _match_rowids(db, "resurrect") == set()

        assert db.undelete_media(media_id)
        assert _match_rowids(db, "resurrect") == {media_id}
        assert _match_rowids(db, "alpha") == {media_id}
        _assert_fts_intact(db)
