from __future__ import annotations

import hashlib
import inspect
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

import pytest

import tldw_chatbook.DB.Subscriptions_DB as subscriptions_module
from tldw_chatbook.DB.Subscriptions_DB import (
    SubscriptionError,
    SubscriptionsDB,
    SubscriptionsDBReadError,
)


def _create_subscriptions_database(path: Path) -> None:
    db = SubscriptionsDB(path, client_id="seed")
    db.add_subscription(
        name="Example source",
        type="rss",
        source="https://example.test/feed",
    )
    db.close()


def _database_snapshot(path: Path) -> tuple[str, tuple[str, ...], tuple[int, int, int]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with closing(sqlite3.connect(path)) as conn:
        schema = tuple(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY name"
            )
        )
        counts = (
            conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM subscription_items").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM watchlists").fetchone()[0],
        )
    return digest, schema, counts


def _sqlite_artifact_snapshot(
    path: Path,
) -> dict[str, tuple[int, int, int, str]]:
    snapshot: dict[str, tuple[int, int, int, str]] = {}
    for candidate in sorted(path.parent.iterdir()):
        if candidate.name != path.name and not candidate.name.startswith(
            f"{path.name}-"
        ):
            continue
        metadata = candidate.stat()
        snapshot[candidate.name] = (
            stat.S_IMODE(metadata.st_mode),
            metadata.st_size,
            metadata.st_mtime_ns,
            hashlib.sha256(candidate.read_bytes()).hexdigest(),
        )
    return snapshot


def test_read_only_is_keyword_only_and_rejects_memory_and_missing_files(
    tmp_path: Path,
) -> None:
    parameter = inspect.signature(SubscriptionsDB).parameters["read_only"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        SubscriptionsDB(tmp_path / "positional.db", "test", True)

    for path in (Path(":memory:"), tmp_path / "missing.db"):
        with pytest.raises(SubscriptionError) as exc_info:
            SubscriptionsDB(path, client_id="test", read_only=True)
        assert str(exc_info.value) == "Watchlists database is unavailable"
    assert not (tmp_path / "missing.db").exists()


def test_read_only_skips_schema_initialization_and_default_construction_does_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "subscriptions.db"
    _create_subscriptions_database(path)
    calls: list[str] = []
    original_initialize = SubscriptionsDB._initialize_schema
    original_migrate = SubscriptionsDB._ensure_watchlists_schema

    def fail_initialize(self: SubscriptionsDB) -> None:
        raise AssertionError("read-only construction attempted schema initialization")

    def fail_migration(self: SubscriptionsDB, conn=None) -> None:
        raise AssertionError("read-only construction attempted a migration probe")

    monkeypatch.setattr(SubscriptionsDB, "_initialize_schema", fail_initialize)
    monkeypatch.setattr(SubscriptionsDB, "_ensure_watchlists_schema", fail_migration)

    read_only = SubscriptionsDB(path, client_id="agent", read_only=True)
    read_only.close()

    def record_initialize(self: SubscriptionsDB) -> None:
        calls.append(self.db_path_str)
        original_initialize(self)

    monkeypatch.setattr(SubscriptionsDB, "_initialize_schema", record_initialize)
    monkeypatch.setattr(SubscriptionsDB, "_ensure_watchlists_schema", original_migrate)

    writable = SubscriptionsDB(path, client_id="app")
    assert calls == [str(path)]
    writable.close()


def test_read_only_uses_dedicated_owner_and_only_safe_connection_pragmas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "subscriptions.db"
    _create_subscriptions_database(path)
    calls: list[tuple[str, Path | str, dict[str, object]]] = []
    statements: list[str] = []
    real_connect = subscriptions_module.connect_private_sqlite

    def recording_connect(owner_id, database, **kwargs):
        calls.append((owner_id, database, kwargs))
        conn = real_connect(owner_id, database, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(
        subscriptions_module,
        "connect_private_sqlite",
        recording_connect,
    )

    db = SubscriptionsDB(path, client_id="agent", read_only=True)
    db.assert_agent_read_ready()
    assert [tuple(row) for row in db.conn.execute(
        "SELECT version FROM schema_version"
    )] == [(2,)]

    assert calls == [
        (
            "db.subscriptions.agent_read",
            str(path),
            {"read_only": True, "must_exist": True},
        )
    ]
    assert db.conn.row_factory is sqlite3.Row
    normalized = [" ".join(statement.lower().split()) for statement in statements]
    assert "pragma foreign_keys = on;" in normalized
    assert "pragma query_only = on;" in normalized
    forbidden = (
        "journal_mode",
        "synchronous",
        "locking_mode",
        "wal_checkpoint",
        "writable_schema",
    )
    assert not any(
        token in statement for token in forbidden for statement in normalized
    )
    db.close()


def test_read_only_view_cannot_mutate_file_schema_or_rows(tmp_path: Path) -> None:
    path = tmp_path / "subscriptions.db"
    _create_subscriptions_database(path)
    before = _database_snapshot(path)

    db = SubscriptionsDB(path, client_id="agent", read_only=True)
    db.assert_agent_read_ready()
    mutations = (
        (
            "INSERT INTO subscriptions (name, type, source) VALUES (?, ?, ?)",
            ("Injected", "rss", "https://evil.test/feed"),
        ),
        ("UPDATE subscriptions SET name = ? WHERE id = 1", ("Changed",)),
        ("CREATE TABLE agent_write_probe (secret TEXT)", ()),
    )
    for sql, params in mutations:
        with pytest.raises(sqlite3.OperationalError):
            with db.transaction() as conn:
                conn.execute(sql, params)
    db.close()

    assert _database_snapshot(path) == before


def test_read_only_view_preserves_live_wal_data_and_logical_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "subscriptions.db"
    writer = SubscriptionsDB(path, client_id="writer")
    writer.conn.execute("PRAGMA wal_autocheckpoint = 0")
    writer.add_subscription(
        name="Uncheckpointed source",
        type="rss",
        source="https://example.test/live-wal",
    )
    assert writer.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert Path(f"{path}-wal").exists()
    assert Path(f"{path}-shm").exists()
    before_artifacts = _sqlite_artifact_snapshot(path)
    before_schema = tuple(
        row[0]
        for row in writer.conn.execute(
            "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY name"
        )
    )
    before_rows = tuple(
        writer.conn.execute("SELECT id, name, source FROM subscriptions ORDER BY id")
    )

    reader = SubscriptionsDB(path, client_id="agent", read_only=True)
    reader.assert_agent_read_ready()
    assert (
        reader.conn.execute(
            "SELECT name FROM subscriptions WHERE source = ?",
            ("https://example.test/live-wal",),
        ).fetchone()[0]
        == "Uncheckpointed source"
    )
    reader.close()

    after_artifacts = _sqlite_artifact_snapshot(path)
    after_schema = tuple(
        row[0]
        for row in writer.conn.execute(
            "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY name"
        )
    )
    after_rows = tuple(
        writer.conn.execute("SELECT id, name, source FROM subscriptions ORDER BY id")
    )
    writer.close()

    assert after_schema == before_schema
    assert after_rows == before_rows
    assert after_artifacts.keys() == before_artifacts.keys()
    changed_artifacts = {
        name
        for name in before_artifacts
        if before_artifacts[name] != after_artifacts[name]
    }
    # A normal mode=ro connection participates in live WAL coordination. On
    # SQLite builds that update a read mark, the existing -shm bytes may
    # change even though no logical database write occurs. Requiring immutable
    # sidecars would force immutable=1 (which can ignore committed WAL frames)
    # or a separate snapshot architecture. Neither is this live read contract.
    assert changed_artifacts <= {f"{path.name}-shm"}
    assert after_artifacts[path.name] == before_artifacts[path.name]
    assert after_artifacts[f"{path.name}-wal"] == before_artifacts[f"{path.name}-wal"]


def test_standalone_read_only_view_creates_only_private_sqlite_sidecars(
    tmp_path: Path,
) -> None:
    path = tmp_path / "subscriptions.db"
    _create_subscriptions_database(path)
    before_database = _database_snapshot(path)
    before_artifacts = _sqlite_artifact_snapshot(path)
    assert before_artifacts.keys() == {path.name}

    reader = SubscriptionsDB(path, client_id="standalone-agent", read_only=True)
    reader.assert_agent_read_ready()
    assert reader.conn.execute("SELECT name FROM subscriptions").fetchone()[0] == (
        "Example source"
    )
    reader.close()

    after_artifacts = _sqlite_artifact_snapshot(path)
    expected_sidecars = {f"{path.name}-wal", f"{path.name}-shm"}
    assert after_artifacts.keys() - before_artifacts.keys() <= expected_sidecars
    assert after_artifacts.keys() <= {path.name, *expected_sidecars}
    assert all(mode & 0o077 == 0 for mode, *_rest in after_artifacts.values())
    assert after_artifacts[path.name] == before_artifacts[path.name]
    assert _database_snapshot(path) == before_database


def test_readiness_requires_only_agent_tool_core_schema_and_failure_is_closeable(
    tmp_path: Path,
) -> None:
    complete_path = tmp_path / "complete.db"
    _create_subscriptions_database(complete_path)
    with sqlite3.connect(complete_path) as conn:
        conn.execute("DROP TABLE subscription_items_fts")

    complete = SubscriptionsDB(complete_path, read_only=True)
    complete.assert_agent_read_ready()
    complete.close()

    incomplete_path = tmp_path / "operator-secret.db"
    _create_subscriptions_database(incomplete_path)
    with sqlite3.connect(incomplete_path) as conn:
        conn.execute("DROP TABLE watchlist_sources")

    incomplete = SubscriptionsDB(incomplete_path, read_only=True)
    with pytest.raises(SubscriptionError) as exc_info:
        incomplete.assert_agent_read_ready()
    assert str(exc_info.value) == "Watchlists database is unavailable"
    assert "watchlist_sources" not in str(exc_info.value)
    assert str(incomplete_path) not in str(exc_info.value)

    incomplete.close()
    assert incomplete._local.conn is None


@pytest.mark.parametrize(
    ("table", "column"),
    (
        ("subscriptions", "check_frequency"),
        ("subscriptions", "consecutive_failures"),
        ("watchlists", "is_active"),
        ("watchlists", "briefing_selection_mode"),
        ("watchlists", "default_briefing_preset_id"),
        ("watchlists", "briefing_cadence_seconds"),
        ("watchlists", "created_at"),
        ("watchlists", "updated_at"),
    ),
)
def test_readiness_rejects_each_missing_agent_metadata_projection_column(
    tmp_path: Path,
    table: str,
    column: str,
) -> None:
    path = tmp_path / "subscriptions.db"
    _create_subscriptions_database(path)
    db = SubscriptionsDB(path, read_only=True)
    original_connection = db.conn
    incomplete_connection = sqlite3.connect(":memory:")
    incomplete_connection.row_factory = sqlite3.Row
    original_connection.backup(incomplete_connection)
    incomplete_connection.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    db._local.conn = incomplete_connection
    try:
        with pytest.raises(SubscriptionError) as exc_info:
            db.assert_agent_read_ready()
    finally:
        incomplete_connection.close()
        db._local.conn = original_connection
        db.close()

    assert str(exc_info.value) == "Watchlists database is unavailable"


def test_readiness_operational_failure_uses_fixed_transient_exception(
    tmp_path: Path,
) -> None:
    path = tmp_path / "subscriptions.db"
    _create_subscriptions_database(path)
    db = SubscriptionsDB(path, read_only=True)
    original_connection = db.conn

    class FailingReadinessConnection:
        def execute(self, _statement: str):
            raise sqlite3.OperationalError(
                "database /operator/private.db is locked token=secret"
            )

    db._local.conn = FailingReadinessConnection()
    try:
        with pytest.raises(SubscriptionError) as exc_info:
            db.assert_agent_read_ready()
    finally:
        db._local.conn = original_connection
        db.close()

    assert type(exc_info.value) is SubscriptionsDBReadError
    assert str(exc_info.value) == "Watchlists database read failed"
    assert "operator" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)
