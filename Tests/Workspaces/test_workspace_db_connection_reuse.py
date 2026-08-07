"""task-3011: WorkspaceDB holds one connection per thread.

cProfile of a single Console push (task-2902 round 2) showed
`connect_private_sqlite` invoked 1,352 times through WorkspaceDB — a brand
new SQLite connection per query, 0.64s of the ~2.5s first paint. Every
other heavy DB in the repo already holds a thread-local connection
(ChaChaNotes' `_get_thread_connection` idiom).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


@pytest.fixture()
def connection_spy(monkeypatch):
    """Count real private-sqlite connection opens through the base-db seam."""
    import tldw_chatbook.DB.base_db as base_db

    real_connect = base_db.connect_private_sqlite
    opened: list[object] = []

    def counting_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(base_db, "connect_private_sqlite", counting_connect)
    return opened


def test_repeated_reads_open_no_new_connections(tmp_path: Path, connection_spy):
    service = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    )
    service.ensure_default_workspace()  # warm-up: schema + first held conn

    baseline = len(connection_spy)
    for _ in range(10):
        service.get_active_workspace()
        service.list_workspaces()
    assert len(connection_spy) == baseline, (
        f"{len(connection_spy) - baseline} new connections opened for 20 "
        "reads — the per-query connect is back"
    )


def test_failed_transaction_rolls_back_and_connection_stays_usable(
    tmp_path: Path,
):
    db = WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    service = LocalWorkspaceRegistryService(db)
    service.create_workspace(workspace_id="ws-a", name="Alpha")

    with pytest.raises(RuntimeError, match="boom"):
        with db.transaction() as conn:
            conn.execute(
                "UPDATE workspace_records SET name = ? WHERE workspace_id = ?",
                ("Broken", "ws-a"),
            )
            raise RuntimeError("boom")

    # Rolled back...
    rows = [w for w in service.list_workspaces() if w.workspace_id == "ws-a"]
    assert rows and rows[0].name == "Alpha"
    # ...and the held connection keeps working for writes afterwards.
    service.create_workspace(workspace_id="ws-b", name="Beta")
    assert any(
        w.workspace_id == "ws-b" for w in service.list_workspaces()
    )


def test_each_thread_gets_its_own_connection(tmp_path: Path):
    db = WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    service = LocalWorkspaceRegistryService(db)
    service.ensure_default_workspace()

    seen: dict[str, object] = {}
    errors: list[BaseException] = []

    def read(tag: str) -> None:
        try:
            with db.connection() as conn:
                conn.execute("SELECT COUNT(*) FROM workspace_records").fetchone()
                seen[tag] = conn
        except BaseException as exc:  # noqa: BLE001 - surface to the test
            errors.append(exc)

    threads = [threading.Thread(target=read, args=(f"t{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"cross-thread use failed: {errors}"
    assert seen["t0"] is not seen["t1"], (
        "two threads shared one sqlite connection"
    )
