"""The Console rows that explain a run's unproven tool cleanup (TASK-40716).

The row's Release action returns only the run's tool-worker slot. The tool's
workspace, storage and claim stay held until the app restarts (ADR-134).
Imported only once such a row exists, so it never loads at boot.
"""

from __future__ import annotations

from typing import Any


def _retention(controller: Any, run_id: str | None) -> Any:
    bridge = getattr(controller, "_agent_bridge", None)
    capacity = getattr(bridge, "_runtime_capacity", None)
    retained = getattr(capacity, "retained", None)
    for retention in retained() if callable(retained) and run_id else ():
        if retention.run_id == run_id:
            return retention
    return None


def unproven_cleanup_rows(controller: Any) -> dict[str, bool]:
    """Explanation rows by message id; True once the run's slot is released."""
    rows = {}
    for message_id, run_id in controller.unproven_cleanup_runs.items():
        retention = _retention(controller, run_id)
        rows[message_id] = retention is None or retention.abandoned
    return rows


def release_unproven_tool_slot(controller: Any, message_id: str) -> bool:
    """Return the tool-worker slot of the run this row explains."""
    run_id = controller.unproven_cleanup_runs.get(message_id)
    retention = _retention(controller, run_id)
    return retention is not None and retention.abandon()
