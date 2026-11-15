"""Release rows for runs whose tool cleanup is unproven (TASK-40716).

A row appears on a run's terminal card for as long as the run has a retained
unproven cleanup, whatever the run's own outcome was: the retention can come
after a timeout, a cancel or a per-call abandon. Release returns only the
run's tool-worker slots. The tool's workspace, storage and claim stay held
until the app restarts (ADR-134). Imported only once such a row can exist.
"""

from __future__ import annotations

from typing import Any


def _capacity(controller: Any) -> Any:
    return getattr(getattr(controller, "_agent_bridge", None), "_runtime_capacity", None)


def _retained(controller: Any) -> dict[str, list[Any]]:
    """Every retained record, grouped by its run (a run can retain several)."""
    retained = getattr(_capacity(controller), "retained", None)
    runs: dict[str, list[Any]] = {}
    for record in retained() if callable(retained) else ():
        if record.run_id:
            runs.setdefault(record.run_id, []).append(record)
    return runs


def unproven_cleanup_rows(controller: Any, messages: Any = ()) -> dict[str, bool]:
    """Rows among ``messages`` whose run holds a retained cleanup; True once released.

    A card is found by its id or its persisted id, so a conversation closed
    and reopened shows its row again. Each row whose run is neither live nor
    retained is forgotten (including after a late proof), so the map stays
    bounded by live runs plus retained records.
    """
    retained = _retained(controller)
    snapshot = getattr(_capacity(controller), "snapshot", None)
    live = (
        {entry.run_id for entry in snapshot().executions if entry.run_id}
        if callable(snapshot)
        else set()
    )
    cards = controller.run_terminal_rows
    for card, run_id in list(cards.items()):
        if run_id not in retained and run_id not in live:
            cards.pop(card, None)
    rows = {}
    for message in messages:
        keep = (message.id, message.persisted_message_id)
        run_id = cards.get(message.id) or cards.get(message.persisted_message_id)
        if run_id in retained:
            # A reopened card releases by its new id; one alias per run.
            for card in [c for c, r in cards.items() if r == run_id and c not in keep]:
                del cards[card]
            cards[message.id] = run_id
            rows[message.id] = all(record.abandoned for record in retained[run_id])
    return rows


def release_unproven_tool_slot(controller: Any, message_id: str) -> bool:
    """Return every tool-worker slot the row's run still holds."""
    run_id = controller.run_terminal_rows.get(message_id)
    records = _retained(controller).get(run_id, ())
    return any([record.abandon() for record in records])
