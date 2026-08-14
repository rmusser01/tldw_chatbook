from __future__ import annotations

import hashlib
import inspect
import sqlite3
from pathlib import Path

import pytest

import tldw_chatbook.DB.Subscriptions_DB as subscriptions_module
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionError, SubscriptionsDB


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
    with sqlite3.connect(path) as conn:
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
    assert not any(token in statement for token in forbidden for statement in normalized)
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
