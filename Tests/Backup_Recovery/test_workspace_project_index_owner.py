"""Exact cache ownership: excluded payload, installed finite retirement."""

from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Event

import pytest

from Tests.private_profile import private_profile_test

from Tests.Workspaces.project_index_support import open_index_store


def test_project_cache_owner_is_excluded_and_capture_refused(tmp_path):
    from tldw_chatbook.Workspaces.recovery import recovery_adapters

    adapter = next(
        a for a in recovery_adapters() if a.owner_id == "db.workspace_project_index"
    )
    root = tmp_path / "test" / "project_index" / ("a" * 64)
    root.mkdir(parents=True)
    for name in (
        "workspace_project_index.db",
        "workspace_project_index.db-wal",
        "cleanup.receipt",
    ):
        (root / name).write_bytes(b"cache")
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )

    items = adapter.discover(
        {
            DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "selected.toml", "test"),
            "paths": {"data_dir": str(tmp_path)},
            "general": {"users_name": "test"},
        }
    )
    assert all(item.status == "intentionally_excluded" for item in items)
    assert {item.path.name for item in items} >= {
        "workspace_project_index.db",
        "workspace_project_index.db-wal",
        "cleanup.receipt",
    }
    with pytest.raises(ValueError, match="excluded"):
        adapter.capture(items[-1], tmp_path / "archive", Event())


def test_finite_worker_retires_exact_index_handle(tmp_path):
    from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker

    def worker():
        handles = []

        def operation():
            store = open_index_store(tmp_path)
            handles.append((store, store._held_connection()))

        run_finite_local_worker(operation)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            handles[0][1].execute("SELECT 1")

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result(timeout=30)


def test_workspace_successor_catalog_matches_fresh_schema(tmp_path):
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.DB.recovery_operations import _WORKSPACES_SCHEMA

    db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    with db.connection() as conn:
        actual = tuple(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
    assert (9, actual) in _WORKSPACES_SCHEMA
    assert any(version == 8 for version, _ in _WORKSPACES_SCHEMA)
    db.close()


def test_activation_denial_precedes_index_open(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from tldw_chatbook.Backup_Recovery import activation

    @contextmanager
    def denied(*args, **kwargs):
        yield False

    monkeypatch.setattr(activation, "execution_scope", denied)
    with pytest.raises(RuntimeError, match="activation"):
        open_index_store(tmp_path)
    assert not (tmp_path / "workspace_project_index.db").exists()


def test_constructor_free_recovery_import_is_inert():
    import subprocess
    import sys

    script = "import sys; from tldw_chatbook.Workspaces.recovery import recovery_adapters; assert any(a.owner_id=='db.workspace_project_index' for a in recovery_adapters()); assert 'tldw_chatbook.DB.Workspace_Project_Index_DB' not in sys.modules; assert 'tldw_chatbook.RAG_Search.chunking_service' not in sys.modules; assert 'tldw_chatbook.Workspaces.registry_service' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@private_profile_test
def test_maintenance_retires_index_cache_after_transaction_settles(tmp_path, request):
    import time
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _retire_current_thread_caches

    store = open_index_store(tmp_path)
    connection = store._held_connection()
    connection.execute("BEGIN")
    pause = storage._begin_local_pause()
    try:
        _retire_current_thread_caches(pause)
        assert connection.in_transaction
        assert not pause.drain(time.monotonic())
        connection.rollback()
        _retire_current_thread_caches(pause)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        assert pause.drain(time.monotonic() + 2)
    finally:
        pause.resume()
        store.close()
