"""Installed heads extend every qualified legacy schema without relabeling it."""

import sqlite3
from contextlib import closing
from threading import Event

import pytest

from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.recovery_core import core_adapters
from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA, recovery_adapters


@pytest.mark.parametrize(
    "leaf,owner_id,old,new",
    [
        ("chat.db", "db.chachanotes.primary", 81, 82),
        ("workspaces.db", "db.workspaces", 8, 9),
        ("runs.db", "db.agent_runs", 25, 26),
    ],
)
def test_real_legacy_source_heads_capture_and_migrate_exactly(
    tmp_path, leaf, owner_id, old, new
):
    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    path = source / leaf
    owner = next(
        item
        for item in (*core_adapters(), *recovery_adapters())
        if item.owner_id == owner_id
    )
    query = (
        "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
        if leaf == "chat.db"
        else "SELECT MAX(version) FROM schema_version"
    )
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(query).fetchone()[0] == old
    before = path.read_bytes()
    assert validate_candidate(owner, path, Event(), migrate=False) == ()
    assert path.read_bytes() == before
    assert validate_candidate(owner, path, Event(), migrate=True) == ()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(query).fetchone()[0] == new
        assert (
            conn.execute(
                "SELECT count(*) FROM mobile_consolidation_batches"
            ).fetchone()[0]
            == 0
        )
    assert validate_candidate(owner, path, Event(), migrate=False) == ()


@pytest.mark.parametrize(
    "catalog", [catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 25]
)
def test_every_qualified_v25_layout_migrates_to_exact_v26(tmp_path, catalog):
    path = tmp_path / "agent_runs.db"
    with closing(sqlite3.connect(path)) as conn:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.execute("INSERT INTO schema_version VALUES (25)")
        conn.commit()
    path.chmod(0o600)
    owner = next(
        item for item in recovery_adapters() if item.owner_id == "db.agent_runs"
    )
    assert validate_candidate(owner, path, Event(), migrate=False) == ()
    assert validate_candidate(owner, path, Event(), migrate=True) == ()
    assert validate_candidate(owner, path, Event(), migrate=False) == ()


def test_owner_migration_failure_rolls_back_new_tables_and_version(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB import mobile_import_authority
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    path = source / "runs.db"
    monkeypatch.setattr(
        mobile_import_authority,
        "AGENT_RUNS_MIGRATION",
        (
            *mobile_import_authority.AGENT_RUNS_MIGRATION,
            "SELECT * FROM deliberately_missing_table",
        ),
    )
    with (
        ConsoleExecutionOwner.acquire(path) as owner,
        pytest.raises(sqlite3.OperationalError, match="deliberately_missing"),
    ):
        AgentRunsDB(path, execution_owner=owner)
    with closing(sqlite3.connect(path)) as conn:
        assert (
            conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 25
        )
        assert (
            conn.execute(
                "SELECT name FROM sqlite_schema WHERE name LIKE 'mobile_%'"
            ).fetchall()
            == []
        )
