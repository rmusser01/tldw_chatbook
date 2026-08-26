"""task-15863: a wake notice must report the child's terminal status as
of delivery time -- never a stale word from a pinned read snapshot.

The live finding (PR3a-2 Task 7, scenario 4): a wake deferred behind a
composer draft composed its notice with ``researcher — running`` for a
child whose ``agent_runs`` row had been ``done`` for a full minute.

The mechanism, identified and demonstrated here: ``_rows_for`` reads
``runs_db.get_run`` on the app-loop thread's per-thread HELD connection
(``AgentRunsDB._held_connection``). That connection is a WAL reader, and
in Python's ``sqlite3`` ANY unfinalized statement on a connection holds
its implicit read transaction open -- pinning the connection's snapshot
of the database at that moment. Every later read on the same connection
(including ``get_run`` at compose time, however much later) then reports
the world as of the pin: a child mid-run at pin time reads ``running``
forever, no matter that its terminal ``done`` + result committed long
ago through the child thread's own connection. The wake registry itself
can never produce the word: settle-hook and ledger entries are terminal
by construction (the settle hook fires strictly AFTER the terminal DB
write -- ``run_child``'s ``finally`` ordering).

The fix under test: ``_rows_for`` treats a NON-terminal status on a
pending run's row as the stale read it provably is and re-reads through
``AgentRunsDB.get_run_fresh`` -- a dedicated, immediately-closed
connection that cannot inherit any pinned snapshot -- recovering both
the terminal word and the result the snapshot was hiding.
"""

from __future__ import annotations

import threading

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _controller_rig,
    _drain,
    _quiet,
    _settle,
    _survivor,
)


@pytest.mark.asyncio
async def test_deferred_wake_notice_reports_done_despite_a_pinned_snapshot(
    tmp_path,
):
    """The live scenario-4 shape, with the stale-snapshot mechanism held
    open deliberately: the child is mid-run when the composing thread's
    held connection pins its snapshot; the child then settles ``done``
    (terminal write through ITS OWN connection); the wake defers behind a
    draft and later delivers. The notice must say ``done`` and carry the
    result -- against unfixed production it says ``running`` with no
    result, exactly the live finding."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        parent_id = runs_db.create_run(
            conversation_id=session.id, agent_kind="primary"
        )
        runs_db.set_status(parent_id, "done", "turn final")
        child_id = runs_db.create_run(
            conversation_id=session.id,
            agent_kind="subagent",
            task="long job",
            parent_run_id=parent_id,
        )

        # Pin THIS thread's held-connection snapshot while the child is
        # still mid-run: an unfinalized statement (any partially-consumed
        # cursor) holds the connection's WAL read transaction open. This
        # is the composing thread -- pytest-asyncio's loop runs here, so
        # `_attempt`/`_rows_for` read through this exact connection.
        with runs_db.connection() as pinned_conn:
            pinned_cursor = pinned_conn.execute("SELECT * FROM agent_runs")
            assert pinned_cursor.fetchone() is not None, (
                "precondition: the pinning statement must be mid-iteration"
            )

            # The child settles on its own thread (its own held
            # connection): terminal status + result committed to the FILE.
            done = threading.Event()

            def _settle_child() -> None:
                runs_db.set_status(child_id, "done", "late child answer")
                done.set()

            threading.Thread(target=_settle_child, daemon=True).start()
            assert done.wait(5), "the child's terminal write never committed"

            # The mechanism, demonstrated: this thread's held connection
            # still reports the pre-pin world.
            stale_row = runs_db.get_run(child_id)
            assert stale_row is not None
            assert stale_row["status"] == "running", (
                "mechanism precondition: the pinned snapshot must hide the "
                f"terminal write (read {stale_row['status']!r}); if this "
                "fails, the held-connection pin no longer reproduces and "
                "the fix needs a new mechanism test"
            )

            # Scenario 4's deferral: a composer draft holds the wake.
            draft_present = True
            controller.wake_user_priority_probe = lambda sid: draft_present
            wake = controller.fleet_wake
            wake.on_fleet_drained(
                _drain(session.id, _survivor(child_id, session_id=session.id))
            )
            assert await _quiet(lambda: gateway.payloads), (
                "the draft must defer the wake"
            )

            # The claim ends a minute later (the pin still held, exactly
            # the frozen-UI window the live pass sat in); the wake fires.
            draft_present = False
            wake.retry_soon()
            assert await _settle(lambda: gateway.payloads), (
                "clearing the draft never delivered the deferred wake"
            )

            notice = gateway.payloads[0][-1]["content"]
            assert "— running" not in notice, (
                "task-15863: the notice labelled a settled child with the "
                f"pinned snapshot's stale 'running':\n{notice}"
            )
            assert "— done" in notice, (
                "the notice must report the child's terminal status as of "
                f"delivery time:\n{notice}"
            )
            assert "late child answer" in notice, (
                "the fresh delivery-time read must also recover the result "
                f"the stale snapshot was hiding:\n{notice}"
            )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_rows_for_never_reports_running_without_a_fresh_read_seam(
    tmp_path,
):
    """The last honest resort: a runs-db handle WITHOUT ``get_run_fresh``
    (an older double, or a wrapper) whose held read reports a settled
    child ``running`` must still yield the settle-recorded terminal word
    -- the registry's status was taken strictly after the terminal write,
    so 'running' is never an honest thing to announce for a pending run."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = (
        _controller_rig(tmp_path)
    )
    try:
        class _StaleOnlyRunsDB:
            """get_run serves a stale non-terminal row; no fresh seam."""

            def get_run(self, run_id):
                return {
                    "id": run_id,
                    "agent_definition": "researcher",
                    "status": "running",
                    "task": "long job",
                    "result": None,
                    "wake_delivered_at": None,
                }

        bridge._runs_db = _StaleOnlyRunsDB()
        wake = controller.fleet_wake
        rows = wake._rows_for(session.id, {"run-stale": "done"})
        assert len(rows) == 1
        assert rows[0]["status"] == "done", (
            "task-15863: without a fresh-read seam the settle-recorded "
            f"terminal word must win over the stale 'running': {rows[0]!r}"
        )
    finally:
        chacha.close()
