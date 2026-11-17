"""Release rows and the held-slots list for unproven tool cleanups.

TASK-40716 put a row on an agent run's terminal card for as long as the run
has a retained unproven cleanup, whatever the run's own outcome was: the
retention can come after a timeout, a cancel or a per-call abandon.
TASK-41307 keys records by their run, else by their owner's execution, so a
Raw CLI marker and a turn stopped before its run can carry the same row, and
lists every record (Undo, change review, @git, sub-agents too) behind the
status line's "Held tool slots" segment.

Release returns only tool-worker slots. The tool's workspace, storage and
claim stay held until the app restarts (ADR-134). Imported only once such a
row or record can exist.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from tldw_chatbook.Agents.execution_capacity import (
    CleanupUnprovenRefused,
    cleanup_unproven_result,
)


def _capacity(controller: Any) -> Any:
    # The runtime owns capacity; the bridge allocates its copy only once an
    # agent turn runs, so a Raw CLI- or Undo-only session has none there.
    runtime = getattr(controller, "_hooks_v2_runtime", None)
    capacity = getattr(runtime, "_execution_capacity", None)
    if capacity is None:
        capacity = getattr(
            getattr(controller, "_agent_bridge", None), "_runtime_capacity", None
        )
    return capacity


def _key(record: Any) -> str:
    """A record's run, else its owner's execution (never bound to a run)."""
    return record.run_id or record.operation.owner.execution_id


def _retained(controller: Any) -> dict[str, list[Any]]:
    """Every retained record, grouped by key (a run can retain several)."""
    retained = getattr(_capacity(controller), "retained", None)
    groups: dict[str, list[Any]] = {}
    for record in retained() if callable(retained) else ():
        groups.setdefault(_key(record), []).append(record)
    return groups


def _labels(records: list[Any]) -> str:
    return ", ".join(dict.fromkeys(record.label for record in records))


def _explanation(records: list[Any]) -> str:
    return f"Agent stopped: {cleanup_unproven_result(_labels(records)).error}."


def unproven_cleanup_rows(
    controller: Any, messages: Any = ()
) -> dict[str, tuple[bool, str]]:
    """Rows among ``messages`` with a retained cleanup: (released, explanation).

    A row can carry several keys: a retried turn reuses its reply, whose
    pre-run record stays retained under the same row (TASK-41307). A card is
    found by its id or its persisted id, so a conversation closed and
    reopened shows its row again. A key that is neither live nor retained is
    forgotten (including after a late proof), so the map stays bounded by
    live work plus retained records. The explanation is set only for a turn
    stopped before its run existed: an agent run explains itself in its own
    row, a Raw CLI marker in its "Cleanup" line.
    """
    from .console_chat_models import ConsoleMessageRole

    retained = _retained(controller)
    snapshot = getattr(_capacity(controller), "snapshot", None)
    live = (
        {
            key
            for entry in snapshot().executions
            for key in (entry.run_id, entry.execution_id)
        }
        if callable(snapshot)
        else set()
    )
    cards = controller.run_terminal_rows
    for card, keys in list(cards.items()):
        keys &= retained.keys() | live
        if not keys:
            cards.pop(card, None)
    rows = {}
    for message in messages:
        keep = (message.id, message.persisted_message_id)
        held = {
            key
            for card in keep
            for key in cards.get(card, ())
            if key in retained
        }
        if not held:
            continue
        # A reopened card releases by its new id; one alias per key.
        for card, keys in list(cards.items()):
            if card not in keep:
                keys -= held
                if not keys:
                    del cards[card]
        cards.setdefault(message.id, set()).update(held)
        records = [record for key in held for record in retained[key]]
        pre_run = message.role is ConsoleMessageRole.ASSISTANT and not any(
            record.run_id for record in records
        )
        rows[message.id] = (
            all(record.abandoned for record in records),
            _explanation(records) if pre_run else "",
        )
    return rows


def release_unproven_tool_slot(controller: Any, message_id: str) -> bool:
    """Return every tool-worker slot the row's keys still hold."""
    keys = tuple(controller.run_terminal_rows.get(message_id, ()))
    return any([release_held_tool_slot(controller, key) for key in keys])


def held_tool_slots(controller: Any) -> dict[str, tuple[str, bool]]:
    """Every retained record by key: (what it was, True once released)."""
    return {
        key: (_labels(records), all(record.abandoned for record in records))
        for key, records in _retained(controller).items()
    }


def unreleased_slot_count(controller: Any) -> int:
    """Tool-worker slots still held by retained records (the status segment)."""
    retained = getattr(_capacity(controller), "retained", None)
    return sum(
        not record.abandoned for record in (retained() if callable(retained) else ())
    )


def release_held_tool_slot(controller: Any, key: str | None) -> bool:
    """Return the slots of every record under ``key``; the resource stays held."""
    records = _retained(controller).get(key, ())
    return any([record.abandon() for record in records])


def offer_release(controller: Any, row_id: str, retention: Any) -> None:
    """Show Release under transcript row ``row_id`` for ``retention``."""
    controller.run_terminal_rows.setdefault(row_id, set()).add(_key(retention))


def explain_pre_run_stop(
    controller: Any, assistant_id: str, error: BaseException
) -> None:
    """Name the cleanup that stopped a turn before its run, and offer Release.

    The stop copy names the tool instead of "Response recovery failed", and
    the reply gets the explanation with Release (TASK-41307). Callers run it
    just before the recovery restore that consumes the copy, so an early
    return leaves none behind. It runs inside the accepted-turn failure
    handler, so it never raises; a cancellation is never explained.
    """
    if isinstance(error, asyncio.CancelledError):
        return
    try:
        seen = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            retention = getattr(error, "retention", None)
            if isinstance(error, CleanupUnprovenRefused) and retention is not None:
                offer_release(controller, assistant_id, retention)
                controller._preflight_block_copies[assistant_id] = _explanation(
                    [retention]
                )
                return
            error = error.__cause__ or error.__context__
    except Exception as failure:  # noqa: BLE001 -- must not replace the one handled
        logger.warning(
            "Unproven-cleanup stop copy failed | exception_type={}",
            type(failure).__name__,
        )
