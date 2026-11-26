"""A spawned child's preset fallback chain (TASK-42564).

ADR-337 amendment 2026-11-26 (owner-approved): a child's preset fallback
keeps the child's actor and starts a new chain. A child segment holds the
calls of one run identity (schema v87), so the new chain's first call opens
a child segment of its own, linked to the same owner and turn. That call is
admitted only when it follows the child's failed call before any response,
and the check is repeated at dispatch binding.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from tldw_chatbook.Chat.console_trace_continuation import falls_back_from
from tldw_chatbook.Chat.console_trace_models import TraceCallState

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_trace_repository import (
        ConsoleTraceRepository,
        TraceCallRecord,
        RequestHeaderRecord,
    )


def child_fallback_admits(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    run_id: str,
    owner_id: str,
    turn_id: str,
    policy_id: str,
    provider: str,
    model: str,
) -> bool | None:
    """Whether a child chain's first call may follow the child's earlier chains.

    Args:
        cursor: Cursor for the caller-owned transaction.
        repository: The trace repository.
        run_id: The new chain's run identity (``actor:chain``).
        owner_id: The attached owner the call is reserved under.
        turn_id: The spawning turn.
        policy_id: The call's frozen policy.
        provider: The call's provider (execution key).
        model: The call's model.

    Returns:
        ``None`` when the actor has no other chain (an ordinary first call).
        Otherwise ``True`` only when every other chain of the actor is under
        the same owner and turn, none of their calls started a response (so
        no tool call came first), and the latest of them is a settled ERROR
        with the same policy, to another provider or model.
    """
    actor, _, chain = run_id.partition(":")
    if not actor or not chain:
        return None
    links = cursor.execute(
        "SELECT segment_id, owner_id, turn_id FROM console_trace_child_segments "
        "WHERE substr(run_id, 1, ?) = ? AND run_id <> ?",
        (len(actor) + 1, actor + ":", run_id),
    ).fetchall()
    if not links:
        return None
    if any((link[1], link[2]) != (owner_id, turn_id) for link in links):
        return False
    segments = tuple(link[0] for link in links)
    marks = ",".join("?" for _ in segments)
    responded = cursor.execute(
        f"SELECT 1 FROM console_trace_calls WHERE segment_id IN ({marks}) "  # noqa: S608 - placeholders only
        "AND response_started_at IS NOT NULL LIMIT 1",
        segments,
    ).fetchone()
    latest = cursor.execute(
        f"SELECT call_id FROM console_trace_calls WHERE segment_id IN ({marks}) "  # noqa: S608 - placeholders only
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        segments,
    ).fetchone()
    failed = None if latest is None else repository.get_call(cursor, latest[0])
    return bool(
        responded is None
        and failed is not None
        and failed.outcome == TraceCallState.ERROR.value
        and failed.settled_at is not None
        and failed.policy_id == policy_id
        and falls_back_from(failed, provider, model)
    )


def recheck_child_fallback(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    reserved: TraceCallRecord,
    header: RequestHeaderRecord,
) -> None:
    """Repeat a child chain's admission at dispatch binding (ADR-337 2026-11-26).

    Raises:
        ValueError: ``trace_child_segment_unavailable`` when it no longer holds.
    """
    if (
        child_fallback_admits(
            cursor,
            repository,
            run_id=reserved.run_id,
            owner_id=reserved.owner_id,
            turn_id=reserved.turn_id,
            policy_id=reserved.policy_id,
            provider=header.provider_name,
            model=header.model_name,
        )
        is False
    ):
        raise ValueError("trace_child_segment_unavailable")
