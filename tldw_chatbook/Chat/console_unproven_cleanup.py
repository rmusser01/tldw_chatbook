"""Release rows for runs whose tool cleanup is unproven (TASK-40716).

A row appears on a run's terminal card for as long as the run has a retained
unproven cleanup, whatever the run's own outcome was: the retention can come
after a timeout, a cancel or a per-call abandon. Release returns only the
run's tool-worker slot. The tool's workspace, storage and claim stay held
until the app restarts (ADR-134). Imported only once such a row can exist.
"""

from __future__ import annotations

from typing import Any


def _capacity(controller: Any) -> Any:
    return getattr(getattr(controller, "_agent_bridge", None), "_runtime_capacity", None)


def _retained(controller: Any) -> dict[str, Any]:
    retained = getattr(_capacity(controller), "retained", None)
    records = retained() if callable(retained) else ()
    return {record.run_id: record for record in records if record.run_id}


def unproven_cleanup_rows(controller: Any) -> dict[str, bool]:
    """Terminal rows of runs with a retained cleanup; True once released.

    Also forgets each row nothing can still retain: its session closed, or its
    run drained without a retained record (including after a late proof). A
    released record stays listed while it is retained, so the map stays
    bounded by live runs plus retained records.
    """
    capacity = _capacity(controller)
    retained = _retained(controller)
    snapshot = getattr(capacity, "snapshot", None)
    live = (
        {entry.run_id for entry in snapshot().executions if entry.run_id}
        if callable(snapshot)
        else set()
    )
    open_messages = getattr(
        getattr(controller, "store", None), "_message_session_index", None
    )
    rows = {}
    for message_id, run_id in list(controller.run_terminal_rows.items()):
        closed = open_messages is not None and message_id not in open_messages
        if closed or (run_id not in retained and run_id not in live):
            controller.run_terminal_rows.pop(message_id, None)
        elif run_id in retained:
            rows[message_id] = retained[run_id].abandoned
    return rows


def release_unproven_tool_slot(controller: Any, message_id: str) -> bool:
    """Return the tool-worker slot of the run this row belongs to."""
    run_id = controller.run_terminal_rows.get(message_id)
    retention = _retained(controller).get(run_id)
    return retention is not None and retention.abandon()
