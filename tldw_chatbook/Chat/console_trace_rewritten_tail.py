"""Durable proof for a rewritten trace surface tail.

ADR-337 amendment 2026-10-27 (TASK-33621.29, TASK-33621.7). Compact now drops
the compacted turns from the provider-visible history, and Edit & resend
answers on a new sibling branch with capture off. Neither records its change
in the trace surface. The next captured send therefore carries a rewrite of
the recorded surface's tail together with the new turn, a shape the planner
otherwise refuses.

This module admits that one shape only when the replacement rows are the
current branch's exact saved history: each row's message is the direct parent
of the next, the last row is the current user, and the oldest row hangs off a
proven anchor (the last retained saved row, the active conversation memory's
boundary, or the conversation root). It reads messages and memory records
only; it writes nothing.
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
    from tldw_chatbook.Chat.console_trace_action_calls import AfterActionCall
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService


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


def _active_memory_boundary(
    cursor: sqlite3.Cursor, conversation_id: str, message_id: str
) -> bool:
    row = cursor.execute(
        "SELECT 1 FROM console_conversation_memories "
        "WHERE conversation_id = ? AND active = 1 AND boundary_message_id = ? "
        "LIMIT 1",
        (conversation_id, message_id),
    ).fetchone()
    return row is not None


def _chain_messages(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    conversation_id: str,
    revision_ids: Sequence[str],
    current_turn_id: str,
    replies: int = 0,
    skipped_reply_id: str | None = None,
) -> tuple[list[str], dict[str, str | None]] | None:
    """Resolve revisions to one parent-linked message chain through the user.

    The user ends the chain, or is followed by ``replies`` saved replies (a
    Continue's, TASK-42560; the planner found them as assistant rows, and
    binding compares each row with its saved value). With
    ``skipped_reply_id``, the user hangs off that failed, empty reply, whose
    parent is the row before the user; the request skips it.
    """
    from tldw_chatbook.Chat.console_trace_action_calls import unanswered_reply

    messages: list[str] = []
    for revision_id in revision_ids:
        revision = repository.get_semantic_revision(cursor, revision_id)
        if revision is None or revision.source_conversation_id != conversation_id:
            return None
        messages.append(revision.source_message_id)
    user = len(messages) - 1 - replies
    if messages[user] != current_turn_id:
        return None
    parents = _message_rows(cursor, conversation_id, messages)
    if parents is not None and skipped_reply_id is not None:
        if (
            user < 1
            or parents[messages[user]] != skipped_reply_id
            or unanswered_reply(
                cursor, conversation_id=conversation_id, message_id=skipped_reply_id
            )
            != messages[user - 1]
        ):
            return None
        parents = {**parents, messages[user]: messages[user - 1]}
    if parents is None or any(
        parents[child] != parent for parent, child in zip(messages, messages[1:])
    ):
        return None
    return messages, parents


def _shape(
    incoming: Sequence[TraceProvenance], user_index: int | None
) -> tuple[int, tuple, TraceProvenance | None, int] | None:
    """Split ``incoming`` into (user index, reply, instruction, project count)."""
    project = 0
    while project < len(incoming) and _is_project_row(incoming[-1 - project]):
        project += 1
    if user_index is None:
        return len(incoming) - 1 - project, (), None, project
    if not 0 <= user_index < len(incoming) - project:
        return None
    middle = tuple(incoming[user_index + 1 : len(incoming) - project])
    if not middle:
        return user_index, (), None, project
    instruction = middle[-1]
    if (
        len(middle) < 2
        or type(instruction) is not ProviderArtifactTraceProvenance
        or instruction.source is not TraceProvenanceSource.ACTIVE_REQUEST
    ):
        return None
    return user_index, middle[:-1], instruction, project


def _is_project_row(descriptor: TraceProvenance) -> bool:
    return (
        type(descriptor) is ProviderArtifactTraceProvenance
        and descriptor.source is TraceProvenanceSource.PROJECT_INSTRUCTION
    )


def _after_action_proof(
    cursor: sqlite3.Cursor,
    *,
    after: AfterActionCall,
    history: Sequence[TraceProvenance],
    removed_start: int | None,
    segment_id: str,
) -> bool:
    """Prove the rows a send after an action's run carries before its user.

    The replacement removes only rows the run wrote after its re-recorded
    user row. It may first restore that row's saved source (the run's user
    was a transform). Then it carries the run's verified reply, or, when the
    run bore none, nothing or the reply an earlier call of the turn bore.
    """
    from tldw_chatbook.Chat.console_trace_action_calls import earlier_reply

    if removed_start is None or removed_start < after.user_sequence:
        return False
    rest = [rewritten_history_revision_id(row) for row in history]
    if removed_start == after.user_sequence:
        if not rest or rest[0] != after.pin_revision_id:
            return False
        rest = rest[1:]
    if after.response_revision_id is not None:
        return rest == [after.response_revision_id]
    return not rest or (
        len(rest) == 1
        and rest[0] is not None
        and earlier_reply(
            cursor,
            segment_id=segment_id,
            turn_id=after.turn_id,
            origin_call_id=after.origin_call_id,
            revision_id=rest[0],
        )
    )


def build_rewritten_tail_witness(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    owner_id: str,
    retained: Sequence[TraceProvenance],
    incoming: Sequence[TraceProvenance],
    current_turn_id: str | None,
    user_index: int | None = None,
    after_action: AfterActionCall | None = None,
    segment_id: str | None = None,
    removed_start: int | None = None,
    policy_id: str | None = None,
) -> RewrittenTailWitness | None:
    """Prove that ``incoming`` is the current saved branch after ``retained``.

    Args:
        cursor: Cursor in the caller-owned transaction.
        repository: Trace repository used for owner and revision reads.
        owner_id: Trace owner whose attached conversation holds every row.
        retained: Incoming descriptors that still match the recorded surface,
            in provider order, before the replacement.
        incoming: Descriptors replacing the recorded tail, ending at the
            current user turn (and, below, the call's own rows after it).
        current_turn_id: Saved user message that owns the incoming call.
        user_index: For an action's call (ADR-337 amendment 2026-11-26), the
            current user's position in ``incoming``; the user may be followed
            by a Continue's saved reply and instruction, then project context.
        after_action: For the send after an action's run, that run's
            evidence; the rows may then end with project context.
        segment_id: The call's segment.
        removed_start: First sequence the replacement removes.
        policy_id: The call's frozen disclosure policy.

    Returns:
        The witness when every row is proven, otherwise None.

    Raises:
        sqlite3.Error: If a revision, message or memory read fails.
    """
    from tldw_chatbook.Chat.console_trace_action_calls import unanswered_reply

    if after_action is not None and (
        removed_start is None
        or removed_start < after_action.user_sequence
        or removed_start == after_action.user_sequence
        and (
            not incoming
            or rewritten_history_revision_id(incoming[0])
            != after_action.pin_revision_id
        )
    ):
        # The replacement reaches before the run's own rows (Compact now, a
        # changed earlier row), or replaces the run's user row with another
        # (a Capture Off Edit & resend of it, or a switch back to its
        # earlier branch): an ordinary rewritten tail, as before.
        after_action = None
    conversation_id = _conversation_id(cursor, repository, owner_id)
    shape = _shape(incoming, user_index)
    if (
        shape is None
        or len(incoming)
        < (1 if user_index is not None or after_action is not None else 2)
        or current_turn_id is None
        or conversation_id is None
        or (user_index is not None and after_action is not None)
    ):
        return None
    position, reply, instruction, project = shape
    if project or instruction is not None:
        if policy_id is None or any(
            item.policy.policy_id != policy_id
            for item in (*incoming[len(incoming) - project :], instruction)
            if item is not None
        ):
            return None
    history = tuple(incoming[:position])
    if user_index is None and (
        after_action is None
        and project
        or after_action is not None
        and (
            segment_id is None
            or not _after_action_proof(
                cursor,
                after=after_action,
                history=history,
                removed_start=removed_start,
                segment_id=segment_id,
            )
        )
    ):
        return None
    user_revision_id = current_turn_source_revision_id(incoming[position])
    revision_ids = [rewritten_history_revision_id(row) for row in (*history, *reply)]
    if user_revision_id is None or any(item is None for item in revision_ids):
        return None
    chain_ids = (
        *revision_ids[: len(history)],
        user_revision_id,
        *revision_ids[len(history) :],
    )
    chain = _chain_messages(
        cursor,
        repository,
        conversation_id=conversation_id,
        revision_ids=chain_ids,
        current_turn_id=current_turn_id,
        replies=len(reply),
    )
    skipped_reply_id = None
    if (
        chain is None
        and after_action is not None
        and after_action.response_revision_id is None
        and removed_start == after_action.user_sequence
        and len(history) == 1
    ):
        # The run bore nothing and its user was a transform: the request
        # restores the user's saved source, and the new user hangs off the
        # run's failed, empty reply after it, which the request skips.
        parents = _message_rows(cursor, conversation_id, (current_turn_id,))
        skipped_reply_id = None if parents is None else parents[current_turn_id]
        chain = (
            None
            if skipped_reply_id is None
            else _chain_messages(
                cursor,
                repository,
                conversation_id=conversation_id,
                revision_ids=chain_ids,
                current_turn_id=current_turn_id,
                skipped_reply_id=skipped_reply_id,
            )
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
    unanswered = False
    if anchor is not None and anchor == retained_message:
        memory_anchor = False
    elif (
        anchor is not None
        and (
            user_index is not None
            or after_action is not None
            and after_action.response_revision_id is None
        )
        and retained_message is not None
        and unanswered_reply(
            cursor,
            conversation_id=conversation_id,
            message_id=anchor,
        )
        == retained_message
    ):
        # A reply that failed with nothing in it, after the last retained
        # row: the run's own, or one before an action's turn (ADR-337
        # amendment 2026-11-26, point 4); the request skips it.
        memory_anchor, unanswered = False, True
    elif anchor is not None and _active_memory_boundary(
        cursor, conversation_id, anchor
    ):
        # The memory reaches the provider as a system message outside the
        # history rows, so the active memory's boundary proves the drop.
        memory_anchor = True
    elif anchor is None and last_retained is None:
        memory_anchor = False
    else:
        return None
    try:
        return RewrittenTailWitness(
            history,
            user_revision_id,
            anchor_message_id=anchor,
            memory_anchor=memory_anchor,
            reply=reply,
            instruction=instruction,
            project_context_count=project,
            after_call_id=None if after_action is None else after_action.latest_call_id,
            unanswered_anchor=unanswered,
            skipped_reply_id=skipped_reply_id,
        )
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
    service: ConsoleTraceService | None = None,
    segment_id: str | None = None,
    reserved_call_id: str | None = None,
    removed_start: int | None = None,
) -> None:
    """Recheck an admitted witness against durable state at final binding.

    Args:
        cursor: Cursor in the dispatch-binding transaction.
        repository: Trace repository used for owner and revision reads.
        owner_id: Trace owner whose attached conversation holds every row.
        witness: Witness admitted at preparation.
        descriptors: Exact descriptors being persisted for the replacement.
        current_turn_id: Saved user message owning the reserved call.
        service: The trace service, to read an action's run again.
        segment_id: The reserved call's segment.
        reserved_call_id: The reserved call being bound.
        removed_start: First sequence the bound replacement removes.

    Raises:
        ValueError: ``rewritten_tail_unavailable`` if any proof no longer holds.
        sqlite3.Error: If a revision, message or memory read fails.
    """
    from tldw_chatbook.Chat.console_trace_action_calls import (
        after_action_call,
        unanswered_reply,
    )

    conversation_id = _conversation_id(cursor, repository, owner_id)
    if (
        not witness.matches_descriptors(tuple(descriptors))
        or current_turn_id is None
        or conversation_id is None
    ):
        raise ValueError("rewritten_tail_unavailable")
    if witness.after_call_id is not None:
        after = (
            None
            if service is None or segment_id is None or reserved_call_id is None
            else after_action_call(
                service, cursor, segment_id=segment_id, before_call_id=reserved_call_id
            )
        )
        if (
            after is None
            or after.latest_call_id != witness.after_call_id
            or not _after_action_proof(
                cursor,
                after=after,
                history=witness.history,
                removed_start=removed_start,
                segment_id=segment_id,
            )
            or (witness.unanswered_anchor or witness.skipped_reply_id is not None)
            and after.response_revision_id is not None
        ):
            raise ValueError("rewritten_tail_unavailable")
    if (
        witness.unanswered_anchor
        and unanswered_reply(
            cursor,
            conversation_id=conversation_id,
            message_id=witness.anchor_message_id,
        )
        is None
    ):
        raise ValueError("rewritten_tail_unavailable")
    chain = _chain_messages(
        cursor,
        repository,
        conversation_id=conversation_id,
        revision_ids=(
            *(rewritten_history_revision_id(row) for row in witness.history),
            witness.user_revision_id,
            *(rewritten_history_revision_id(row) for row in witness.reply),
        ),
        current_turn_id=current_turn_id,
        replies=len(witness.reply),
        skipped_reply_id=witness.skipped_reply_id,
    )
    if chain is None or chain[1][chain[0][0]] != witness.anchor_message_id:
        raise ValueError("rewritten_tail_unavailable")
    anchor = witness.anchor_message_id
    if anchor is not None and (
        _message_rows(cursor, conversation_id, (anchor,)) is None
        or (
            witness.memory_anchor
            and not _active_memory_boundary(cursor, conversation_id, anchor)
        )
    ):
        raise ValueError("rewritten_tail_unavailable")
