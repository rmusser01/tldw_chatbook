from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from tldw_chatbook.DB.Subscriptions_DB import (
    SubscriptionsDB,
    SubscriptionsDBUnavailableError,
)
from tldw_chatbook.Subscriptions.briefing_service import INTERRUPTED_ERROR
from tldw_chatbook.Subscriptions.startup_reconcile import INTERRUPTED_RUN_ERROR


# Exact historical definitions of the v1 tables this migration owns.  This is
# intentionally literal: constructing a current database and relabelling its
# version would silently acquire the columns and indexes under test.
HISTORICAL_V1_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL);
INSERT INTO schema_version VALUES (1);
CREATE TABLE subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    description TEXT,
    tags TEXT,
    priority INTEGER DEFAULT 3,
    folder TEXT,
    check_frequency INTEGER DEFAULT 3600,
    last_checked TEXT,
    last_successful_check TEXT,
    last_error TEXT,
    error_count INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    is_paused INTEGER DEFAULT 0,
    auto_pause_threshold INTEGER DEFAULT 10,
    auth_config TEXT,
    custom_headers TEXT,
    rate_limit_config TEXT,
    extraction_method TEXT DEFAULT 'auto',
    extraction_rules TEXT,
    processing_options TEXT,
    auto_ingest INTEGER DEFAULT 0,
    notification_config TEXT,
    change_threshold REAL DEFAULT 0.0,
    ignore_selectors TEXT,
    etag TEXT,
    last_modified TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE subscription_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER NOT NULL REFERENCES subscriptions(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    content_hash TEXT,
    published_date TEXT,
    author TEXT,
    categories TEXT,
    enclosures TEXT,
    extracted_data TEXT,
    status TEXT DEFAULT 'new',
    media_id INTEGER,
    processing_error TEXT,
    previous_hash TEXT,
    change_percentage REAL,
    diff_summary TEXT,
    change_type TEXT,
    canonical_url TEXT,
    duplicate_of INTEGER REFERENCES subscription_items(id),
    queued_for_briefing INTEGER DEFAULT 0,
    run_id INTEGER,
    alert_matches TEXT,
    content TEXT,
    content_format TEXT,
    content_kind TEXT,
    is_flagged INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    effective_date TEXT GENERATED ALWAYS AS
        (COALESCE(datetime(published_date), datetime(created_at))) VIRTUAL
);
CREATE TABLE watchlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    tags TEXT,
    is_active INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0,
    briefing_selection_mode TEXT DEFAULT 'auto_featured',
    default_briefing_preset_id INTEGER,
    briefing_cadence_seconds INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE local_watchlist_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES subscriptions(id) ON DELETE CASCADE,
    job_id INTEGER,
    batch_id TEXT,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    stats_json TEXT,
    error_msg TEXT,
    log_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE briefings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watchlist_id INTEGER NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'generating',
    error TEXT,
    covers_through_item_id INTEGER,
    covers_from_ts TEXT,
    selection_mode TEXT,
    preset_id INTEGER,
    model_used TEXT,
    body_markdown TEXT,
    item_count INTEGER DEFAULT 0,
    featured_count INTEGER DEFAULT 0,
    overflow_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE briefing_items (
    briefing_id INTEGER NOT NULL REFERENCES briefings(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL REFERENCES subscription_items(id) ON DELETE CASCADE,
    featured INTEGER DEFAULT 0,
    PRIMARY KEY (briefing_id, item_id)
);
"""


def _columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))


def _indexes(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'index'"
        )
    }


def _build_v1(path: Path, *, fail_version_write: bool = False) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(HISTORICAL_V1_SCHEMA)
        assert _columns(conn, "briefing_items") == (
            "briefing_id",
            "item_id",
            "featured",
        )
        assert "uq_local_watchlist_runs_active_source" not in _indexes(conn)
        assert "uq_briefings_generating_watchlist" not in _indexes(conn)
        conn.execute(
            "INSERT INTO subscriptions "
            "(id, name, type, source, created_at, updated_at) VALUES "
            "(1, 'Old source', 'rss', "
            "'https://source-user:source-pass@example.test/feed?token=source#frag', "
            "'2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO subscription_items "
            "(id, subscription_id, url, title, published_date, status, "
            "canonical_url, created_at, updated_at) VALUES "
            "(11, 1, 'https://item-user:item-pass@example.test/story?token=item#frag', "
            "'Original title', '2026-08-02T00:00:00+00:00', 'new', NULL, "
            "'2026-08-02T01:00:00+00:00', '2026-08-02T01:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO watchlists (id, name, created_at, updated_at) VALUES "
            "(7, 'Threats', '2026-08-01T00:00:00+00:00', "
            "'2026-08-01T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO briefings VALUES "
            "(20, 7, 'complete', NULL, 11, NULL, 'auto', NULL, 'provider/model', "
            "'# Legacy', 1, 1, 0, '2026-08-03T00:00:00+00:00', "
            "'2026-08-03T00:00:00+00:00')"
        )
        conn.execute("INSERT INTO briefing_items VALUES (20, 11, 1)")
        for row in (
            (30, "queued", "2026-08-04T00:00:00+00:00"),
            (31, "running", "2026-08-05T00:00:00+00:00"),
        ):
            conn.execute(
                "INSERT INTO local_watchlist_runs "
                "(id, source_id, job_id, status, created_at, updated_at) "
                "VALUES (?, 1, 1, ?, ?, ?)",
                (row[0], row[1], row[2], row[2]),
            )
        for row in (
            (21, "2026-08-04T00:00:00+00:00"),
            (22, "2026-08-05T00:00:00+00:00"),
        ):
            conn.execute(
                "INSERT INTO briefings "
                "(id, watchlist_id, status, created_at, updated_at) "
                "VALUES (?, 7, 'generating', ?, ?)",
                (row[0], row[1], row[1]),
            )
        if fail_version_write:
            conn.execute(
                "CREATE TRIGGER fail_v2_version BEFORE DELETE ON schema_version "
                "BEGIN SELECT RAISE(ABORT, 'injected version failure'); END"
            )
        conn.commit()


def test_fresh_database_is_direct_v2_with_one_version_row(tmp_path: Path) -> None:
    db = SubscriptionsDB(tmp_path / "fresh.db")
    assert [tuple(row) for row in db.conn.execute("SELECT version FROM schema_version")] == [(2,)]
    assert "selection_position" in _columns(db.conn, "briefing_items")
    assert {
        "uq_local_watchlist_runs_active_source",
        "uq_briefings_generating_watchlist",
    } <= _indexes(db.conn)
    db.close()


def test_v1_upgrade_snapshots_legacy_rows_reconciles_duplicates_and_reopens(
    tmp_path: Path,
) -> None:
    path = tmp_path / "historical-v1.db"
    _build_v1(path)

    db = SubscriptionsDB(path)
    assert [tuple(row) for row in db.conn.execute("SELECT version FROM schema_version")] == [(2,)]
    row = dict(db.conn.execute("SELECT * FROM briefing_items").fetchone())
    assert row["item_id"] == 11
    assert row["live_item_id"] == 11
    assert row["selection_position"] is None
    assert row["citation_position"] is None
    assert row["featured"] == 1
    assert row["cited"] == 0
    assert row["item_title"] == "Original title"
    assert row["item_url"] == "https://example.test/story"
    assert row["source_id"] == 1
    assert row["source_name"] == "Old source"
    assert row["source_url"] == "https://example.test/feed"
    assert row["provenance_version"] == 1

    runs = [
        tuple(row)
        for row in db.conn.execute(
            "SELECT id, status, error_msg FROM local_watchlist_runs ORDER BY id"
        )
    ]
    assert runs == [(30, "failed", INTERRUPTED_RUN_ERROR), (31, "running", None)]
    briefings = [
        tuple(row)
        for row in db.conn.execute(
            "SELECT id, status, error FROM briefings WHERE id IN (21, 22) ORDER BY id"
        )
    ]
    assert briefings == [(21, "failed", INTERRUPTED_ERROR), (22, "generating", None)]

    db.conn.execute("DELETE FROM subscription_items WHERE id = 11")
    db.conn.execute("DELETE FROM subscriptions WHERE id = 1")
    db.conn.commit()
    surviving = dict(db.conn.execute("SELECT * FROM briefing_items").fetchone())
    assert surviving["live_item_id"] is None
    assert surviving["item_id"] == 11
    assert surviving["item_title"] == "Original title"
    assert surviving["source_name"] == "Old source"
    db.close()

    reopened = SubscriptionsDB(path)
    assert [tuple(row) for row in reopened.conn.execute("SELECT version FROM schema_version")] == [(2,)]
    assert reopened.conn.execute("SELECT COUNT(*) FROM briefing_items").fetchone()[0] == 1
    reopened.close()


def test_v1_upgrade_failure_rolls_back_table_rebuild_and_version(tmp_path: Path) -> None:
    path = tmp_path / "rollback-v1.db"
    _build_v1(path, fail_version_write=True)

    with pytest.raises(sqlite3.IntegrityError, match="injected version failure"):
        SubscriptionsDB(path)

    with closing(sqlite3.connect(path)) as conn:
        assert list(conn.execute("SELECT version FROM schema_version")) == [(1,)]
        assert _columns(conn, "briefing_items") == (
            "briefing_id",
            "item_id",
            "featured",
        )
        assert "uq_local_watchlist_runs_active_source" not in _indexes(conn)
        assert "uq_briefings_generating_watchlist" not in _indexes(conn)


def test_read_only_v1_is_unavailable_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "readonly-v1.db"
    _build_v1(path)
    before = path.read_bytes()

    db = SubscriptionsDB(path, read_only=True)
    with pytest.raises(SubscriptionsDBUnavailableError):
        db.assert_agent_read_ready()
    db.close()

    assert path.read_bytes() == before
    with closing(sqlite3.connect(path)) as conn:
        assert list(conn.execute("SELECT version FROM schema_version")) == [(1,)]
        assert _columns(conn, "briefing_items") == (
            "briefing_id",
            "item_id",
            "featured",
        )


def test_two_database_owners_resolve_source_run_claim_and_terminal_releases_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run-claim.db"
    first = SubscriptionsDB(path, client_id="first")
    second = SubscriptionsDB(path, client_id="second")
    source_id = first.add_subscription(
        name="Claimed", type="rss", source="https://example.test/feed"
    )
    barrier = threading.Barrier(2)

    def accept(db: SubscriptionsDB) -> dict[str, object]:
        barrier.wait()
        return db.accept_watchlist_run(
            source_id,
            created_at="2026-08-10T00:00:00+00:00",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(accept, (first, second)))

    assert receipts[0]["id"] == receipts[1]["id"]
    assert sorted(receipt["_claim_acquired"] for receipt in receipts) == [False, True]
    assert first.conn.execute(
        "SELECT COUNT(*) FROM local_watchlist_runs WHERE status IN ('queued', 'running')"
    ).fetchone()[0] == 1
    completed = first.transition_watchlist_run(
        int(receipts[0]["id"]),
        status="completed",
        finished_at="2026-08-10T00:01:00+00:00",
    )
    assert completed is not None
    assert first.transition_watchlist_run(
        int(receipts[0]["id"]),
        status="failed",
        finished_at="2026-08-10T00:01:30+00:00",
        error_msg="late loser",
    ) is None
    assert tuple(
        first.conn.execute(
            "SELECT status, error_msg FROM local_watchlist_runs WHERE id = ?",
            (receipts[0]["id"],),
        ).fetchone()
    ) == ("completed", None)
    replacement = second.accept_watchlist_run(
        source_id,
        created_at="2026-08-10T00:02:00+00:00",
    )
    assert replacement["id"] != receipts[0]["id"]
    assert replacement["_claim_acquired"] is True
    first.close()
    second.close()


def test_two_database_owners_resolve_briefing_claim_and_terminal_releases_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "briefing-claim.db"
    first = SubscriptionsDB(path, client_id="first")
    second = SubscriptionsDB(path, client_id="second")
    with first.transaction() as conn:
        watchlist_id = conn.execute(
            "INSERT INTO watchlists (name) VALUES ('Claimed')"
        ).lastrowid
    barrier = threading.Barrier(2)

    def accept(db: SubscriptionsDB) -> dict[str, object]:
        barrier.wait()
        return db.accept_briefing(
            watchlist_id,
            created_at="2026-08-10T00:00:00+00:00",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(accept, (first, second)))

    assert receipts[0]["id"] == receipts[1]["id"]
    assert sorted(receipt["_claim_acquired"] for receipt in receipts) == [False, True]
    failed = first.transition_briefing(
        int(receipts[0]["id"]), status="failed", error="interrupted"
    )
    assert failed is not None
    assert first.transition_briefing(
        int(receipts[0]["id"]), status="empty", error="late loser"
    ) is None
    assert first.get_briefing(int(receipts[0]["id"]))["status"] == "failed"
    assert first.get_briefing(int(receipts[0]["id"]))["error"] == "interrupted"
    replacement = second.accept_briefing(
        watchlist_id,
        created_at="2026-08-10T00:02:00+00:00",
    )
    assert replacement["id"] != receipts[0]["id"]
    assert replacement["_claim_acquired"] is True
    first.close()
    second.close()
