# Tests/Dreams/test_profile_sources.py
"""Signal readers: notes keywords, media keywords, locked Personal Context."""
import json
from datetime import datetime, timezone

import pytest

from tldw_chatbook.Dreams.profile_sources import (
    read_media_topics,
    read_note_topics,
    read_personal_context_topics,
)


class LockedService:
    def list_records(self, *, scope_ids, include_archived=False):
        raise RuntimeError("profile-locked")  # wrapper re-raises ProfileLockedError


def test_locked_personal_context_returns_cached_distillate(tmp_path, monkeypatch):
    cache = tmp_path / "pc_distillate.json"
    cache.write_text(json.dumps([{"facet": "topic", "text": "visit japan",
                                  "weight": 1.0, "searchable": 1,
                                  "source": "personal_context"}]))
    out = read_personal_context_topics(LockedService(), cache_path=cache)
    assert [t["text"] for t in out] == ["visit japan"]


def test_corrupt_cache_entries_are_dropped(tmp_path):
    cache = tmp_path / "pc_distillate.json"
    cache.write_text(json.dumps([1, "x", None, {"facet": "topic"},
                                 {"facet": "bogus", "text": "t", "weight": 1},
                                 {"facet": "topic", "text": "ok", "weight": 1.0}]))
    out = read_personal_context_topics(LockedService(), cache_path=cache)
    assert [t["text"] for t in out] == ["ok"]


def test_workspace_only_profile_reads_no_scopes(tmp_path):
    from types import SimpleNamespace

    seen = []

    class WorkspaceOnly:
        def list_scopes(self):
            return [SimpleNamespace(scope_id="ws-1", kind="workspace")]

        def list_records(self, *, scope_ids, include_archived=False):
            seen.append(tuple(scope_ids))
            return []

    read_personal_context_topics(WorkspaceOnly(),
                                 cache_path=tmp_path / "c.json")
    # Workspace context never enters discovery, even with no global scope.
    assert seen == [()]


def test_successful_read_refreshes_cache(tmp_path):
    class OkService:
        def list_records(self, *, scope_ids, include_archived=False):
            return []  # empty tuple is a successful read
    cache = tmp_path / "pc_distillate.json"
    out = read_personal_context_topics(OkService(), cache_path=cache)
    assert out == []
    assert cache.exists()


def _fmt(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


_NOW = datetime.now(timezone.utc).replace(microsecond=0)
_STALE = "2020-01-01 00:00:00"


def test_read_note_topics_returns_normalized_recent_keywords(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "chachanotes.sqlite", "test-client")
    try:
        db.execute_query(
            "INSERT INTO notes (id, title, content, last_modified)"
            " VALUES ('n1', 't', 'c', ?)",
            (_fmt(_NOW),),
            commit=True,
        )
        db.execute_query(
            "INSERT INTO notes (id, title, content, last_modified)"
            " VALUES ('n2', 't2', 'c2', ?)",
            (_STALE,),
            commit=True,
        )
        db.execute_query(
            "INSERT INTO keywords (keyword, created_at, last_modified)"
            " VALUES ('  rust  ', ?, ?)",
            (_fmt(_NOW), _fmt(_NOW)),
            commit=True,
        )
        db.execute_query(
            "INSERT INTO keywords (keyword, created_at, last_modified)"
            " VALUES ('cobol', ?, ?)",
            (_STALE, _STALE),
            commit=True,
        )
        db.execute_query(
            "INSERT INTO note_keywords (note_id, keyword_id)"
            " VALUES ('n1', 1)",
            commit=True,
        )
        db.execute_query(
            "INSERT INTO note_keywords (note_id, keyword_id)"
            " VALUES ('n2', 2)",
            commit=True,
        )
        topics = read_note_topics(db)
    finally:
        db.close_connection()
    assert [t["text"] for t in topics] == ["rust"]  # trimmed, stale note gone
    assert topics[0]["facet"] == "topic"
    assert topics[0]["source"] == "notes"


def test_read_media_topics_counts_reading_progress_double(tmp_path):
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    db = MediaDatabase(tmp_path / "media.sqlite", "test-client")
    try:
        for media_id, url, touched in (
            (1, "https://x/1", _fmt(_NOW)),   # fresh ingestion
            (2, "https://x/2", _STALE),        # stale: outside the window
            (3, "https://x/3", _STALE),        # old media, fresh reading progress
        ):
            db.execute_query(
                "INSERT INTO Media (url, title, type, content_hash, uuid,"
                " last_modified, ingestion_date, client_id)"
                " VALUES (?, ?, 'article', ?, ?, ?, ?, 'test-client')",
                (url, f"m{media_id}", f"h{media_id}", f"u{media_id}",
                 touched, touched),
                commit=True,
            )
        db.execute_query(
            "INSERT INTO Keywords (keyword, uuid, last_modified, client_id)"
            " VALUES (' rust ', 'ku1', ?, 'test-client')",
            (_fmt(_NOW),),
            commit=True,
        )
        db.execute_query(
            "INSERT INTO Keywords (keyword, uuid, last_modified, client_id)"
            " VALUES ('cobol', 'ku2', ?, 'test-client')",
            (_STALE,),
            commit=True,
        )
        for media_id, keyword_id in ((1, 1), (2, 2), (3, 1)):
            db.execute_query(
                "INSERT INTO MediaKeywords (media_id, keyword_id)"
                " VALUES (?, ?)",
                (media_id, keyword_id),
                commit=True,
            )
        db.execute_query(
            "INSERT INTO ReadingProgress (media_id, progress_json, last_modified)"
            " VALUES (3, '{}', ?)",
            (_fmt(_NOW),),
            commit=True,
        )
        topics = read_media_topics(db)
    finally:
        db.close()
    assert [t["text"] for t in topics] == ["rust"]  # trimmed, stale media gone
    assert topics[0]["facet"] == "topic"
    assert topics[0]["source"] == "media"
    # fresh ingestion (1 touch) + fresh reading progress (1 touch, doubled in)
    assert topics[0]["weight"] == pytest.approx(0.5)
