"""task-3012: AgentRunsDB holds one connection per thread.

Its module docstring said it "follows the Workspace_DB pattern: per-call
connections" — the anti-pattern task-3011 removed from WorkspaceDB after it
measured ~60% of the Console push. AgentRuns pays it per agent step.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


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
    db = AgentRunsDB(tmp_path / "agent_runs.sqlite", client_id="client-1")
    db.create_run(conversation_id="conv-1", agent_kind="primary")  # warm-up

    baseline = len(connection_spy)
    for _ in range(10):
        db.list_runs(conversation_id="conv-1")
    assert len(connection_spy) == baseline, (
        f"{len(connection_spy) - baseline} new connections opened for 10 "
        "reads — the per-query connect is back"
    )


def test_failed_transaction_rolls_back_and_connection_stays_usable(
    tmp_path: Path,
):
    db = AgentRunsDB(tmp_path / "agent_runs.sqlite", client_id="client-1")
    run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")

    with pytest.raises(RuntimeError, match="boom"):
        with db.transaction() as conn:
            conn.execute(
                "UPDATE agent_runs SET status = ? WHERE id = ?",
                ("failed", run_id),
            )
            raise RuntimeError("boom")

    # Rolled back (BEGIN IMMEDIATE path)...
    runs = {r["id"]: r for r in db.list_runs(conversation_id="conv-1")}
    assert runs[run_id]["status"] == "running"
    # ...and the held connection keeps working for writes afterwards.
    second = db.create_run(conversation_id="conv-1", agent_kind="primary")
    assert second in {
        r["id"] for r in db.list_runs(conversation_id="conv-1")
    }


def test_each_thread_gets_its_own_connection(tmp_path: Path):
    db = AgentRunsDB(tmp_path / "agent_runs.sqlite", client_id="client-1")
    db.create_run(conversation_id="conv-1", agent_kind="primary")

    seen: dict[str, object] = {}
    errors: list[BaseException] = []

    def read(tag: str) -> None:
        try:
            with db.connection() as conn:
                conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()
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
