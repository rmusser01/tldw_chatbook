"""Durable proof for a rewritten trace surface tail.

ADR-097 amendment 2026-10-27 (TASK-33621.7). Edit & resend answers on a new
sibling branch with capture off, so the edited branch is never recorded in the
trace surface. The next captured send therefore carries a rewrite of the
recorded surface's tail together with the new turn, a shape the planner
otherwise refuses.

This module admits that one shape only when the replacement rows are the
current branch's exact saved history: each row's message is the direct parent
of the next, the last row is the current user, and the oldest row hangs off a
proven anchor (the last retained saved row, or the conversation root). It
reads messages only; it writes nothing.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import TYPE_CHECKING

from tldw_chatbook.Chat.console_trace_final_values import (
    RewrittenTailWitness,
    rewritten_history_revision_id,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    ProviderArtifactTraceProvenance,
    TraceProvenance,
    TraceProvenanceSource,
    current_turn_source_revision_id,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository


def _conversation_id(
    cursor: sqlite3.Cursor, repository: ConsoleTraceRepository, owner_id: str
) -> str | None:
    owner = repository.get_owner(cursor, owner_id)
    return None if owner is None else owner.conversation_id


def _message_rows(
    cursor: sqlite3.Cursor, conversation_id: str, message_ids: Sequence[str]
) -> dict[str, str | None] | None:
    """Map each undeleted message to its parent, or None if any is missing."""
    placeholders = ",".join("?" for _ in message_ids)
    rows = cursor.execute(
        "SELECT id, parent_message_id FROM messages "
        f"WHERE conversation_id = ? AND deleted = 0 AND id IN ({placeholders})",
        (conversation_id, *message_ids),
    ).fetchall()
    parents = {str(row[0]): row[1] for row in rows}
    return parents if len(parents) == len(set(message_ids)) else None


def _chain_messages(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    conversation_id: str,
    revision_ids: Sequence[str],
    current_turn_id: str,
) -> tuple[list[str], dict[str, str | None]] | None:
    """Resolve revisions to one parent-linked message chain ending at the user."""
    messages: list[str] = []
    for revision_id in revision_ids:
        revision = repository.get_semantic_revision(cursor, revision_id)
        if revision is None or revision.source_conversation_id != conversation_id:
            return None
        messages.append(revision.source_message_id)
    if messages[-1] != current_turn_id:
        return None
    parents = _message_rows(cursor, conversation_id, messages)
    if parents is None or any(
        parents[child] != parent for parent, child in zip(messages, messages[1:])
    ):
        return None
    return messages, parents


def build_rewritten_tail_witness(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    owner_id: str,
    retained: Sequence[TraceProvenance],
    incoming: Sequence[TraceProvenance],
    current_turn_id: str | None,
) -> RewrittenTailWitness | None:
    """Prove that ``incoming`` is the current saved branch after ``retained``.

    Args:
        cursor: Cursor in the caller-owned transaction.
        repository: Trace repository used for owner and revision reads.
        owner_id: Trace owner whose attached conversation holds every row.
        retained: Incoming descriptors that still match the recorded surface,
            in provider order, before the replacement.
        incoming: Descriptors replacing the recorded tail; the last one is the
            current user turn.
        current_turn_id: Saved user message that owns the incoming call.

    Returns:
        The witness when every row is proven, otherwise None.

    Raises:
        sqlite3.Error: If a revision or message read fails.
    """
    conversation_id = _conversation_id(cursor, repository, owner_id)
    if len(incoming) < 2 or current_turn_id is None or conversation_id is None:
        return None
    history = tuple(incoming[:-1])
    user_revision_id = current_turn_source_revision_id(incoming[-1])
    revision_ids = [rewritten_history_revision_id(row) for row in history]
    if user_revision_id is None or any(item is None for item in revision_ids):
        return None
    chain = _chain_messages(
        cursor,
        repository,
        conversation_id=conversation_id,
        revision_ids=(*revision_ids, user_revision_id),
        current_turn_id=current_turn_id,
    )
    if chain is None:
        return None
    messages, parents = chain
    anchor = parents[messages[0]]
    last_retained = next(
        (
            row
            for row in reversed(retained)
            if not (
                type(row) is ProviderArtifactTraceProvenance
                and row.source is TraceProvenanceSource.RENDERED_SYSTEM
            )
        ),
        None,
    )
    retained_id = (
        None if last_retained is None else rewritten_history_revision_id(last_retained)
    )
    retained_message = None
    if retained_id is not None:
        revision = repository.get_semantic_revision(cursor, retained_id)
        retained_message = None if revision is None else revision.source_message_id
    if not (
        (anchor is not None and anchor == retained_message)
        or (anchor is None and last_retained is None)
    ):
        return None
    try:
        return RewrittenTailWitness(history, user_revision_id, anchor_message_id=anchor)
    except ValueError:
        return None


def validate_rewritten_tail(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    owner_id: str,
    witness: RewrittenTailWitness,
    descriptors: Sequence[TraceProvenance],
    current_turn_id: str | None,
) -> None:
    """Recheck an admitted witness against durable state at final binding.

    Args:
        cursor: Cursor in the dispatch-binding transaction.
        repository: Trace repository used for owner and revision reads.
        owner_id: Trace owner whose attached conversation holds every row.
        witness: Witness admitted at preparation.
        descriptors: Exact descriptors being persisted for the replacement.
        current_turn_id: Saved user message owning the reserved call.

    Raises:
        ValueError: ``rewritten_tail_unavailable`` if any proof no longer holds.
        sqlite3.Error: If a revision or message read fails.
    """
    conversation_id = _conversation_id(cursor, repository, owner_id)
    if (
        not witness.matches_descriptors(tuple(descriptors))
        or current_turn_id is None
        or conversation_id is None
    ):
        raise ValueError("rewritten_tail_unavailable")
    chain = _chain_messages(
        cursor,
        repository,
        conversation_id=conversation_id,
        revision_ids=(
            *(rewritten_history_revision_id(row) for row in witness.history),
            witness.user_revision_id,
        ),
        current_turn_id=current_turn_id,
    )
    if chain is None or chain[1][chain[0][0]] != witness.anchor_message_id:
        raise ValueError("rewritten_tail_unavailable")
    anchor = witness.anchor_message_id
    if anchor is not None and _message_rows(cursor, conversation_id, (anchor,)) is None:
        raise ValueError("rewritten_tail_unavailable")
