"""Read-only workspace activity projection over existing Console owners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .interaction import BuddyBinding


@dataclass(frozen=True, slots=True)
class BuddyInboxEntry:
    """One live conversation or one independently acknowledgeable result."""

    key: str
    title: str
    group: Literal["needs_you", "running", "results"]
    summary: str
    binding: BuddyBinding
    receipt_ids: tuple[str, ...] = ()


def project_workspace_inbox(
    workspace_id: str,
    *,
    sessions: Iterable[Any],
    receipts: Iterable[Any],
    run_states: Mapping[str, Any],
    activities: Mapping[str, Any],
    member_titles: Mapping[str, str],
) -> tuple[BuddyInboxEntry, ...]:
    """Project current local scope without marking viewed results or decisions.

    Args:
        workspace_id: Explicit workspace owner.
        sessions: Current runtime sessions, including other workspaces.
        receipts: Existing immutable unseen outcome records.
        run_states: Per-session run snapshots.
        activities: Queue-aware per-session activity snapshots.
        member_titles: Verified current durable workspace members and their titles.
    """
    live = tuple(sessions)
    scoped = {
        session.id: session
        for session in live
        if session.runtime_backend == "local" and session.workspace_id == workspace_id
    }
    rows: list[BuddyInboxEntry] = []
    for session in scoped.values():
        activity = activities.get(session.id)
        state = run_states.get(session.id)
        if getattr(activity, "needs_approval", False):
            group, summary = "needs_you", "A question or approval needs your response"
        elif getattr(activity, "queue_paused", False):
            group, summary = (
                "needs_you",
                "Queued follow-ups are paused; review in Console",
            )
        elif state is not None and not state.is_send_allowed:
            group, summary = "running", state.visible_copy or "Working"
        elif getattr(activity, "queued_count", 0):
            group, summary = "running", "Follow-ups queued"
        else:
            continue
        rows.append(
            BuddyInboxEntry(
                f"live:{session.id}",
                session.title,
                group,
                summary,
                BuddyBinding.for_session(session),
            )
        )

    for receipt in receipts:
        conversation_id = receipt.conversation_id
        if conversation_id:
            matches = [
                s for s in live if s.persisted_conversation_id == conversation_id
            ]
        else:
            matches = [
                s
                for s in live
                if s.id == receipt.session_id and not s.persisted_conversation_id
            ]
        if len(matches) > 1:
            continue  # No ambiguous live conversation authority.
        if matches:
            session = matches[0]
            if session.id not in scoped:
                continue  # Current live placement wins over stale membership.
            binding = BuddyBinding.for_session(session)
            title = session.title
        elif conversation_id and conversation_id in member_titles:
            binding = BuddyBinding(
                kind="conversation",
                target_id=f"saved:{conversation_id}",
                conversation_id=conversation_id,
            )
            title = member_titles[conversation_id]
        else:
            continue
        summary = {
            "done": "Response ready",
            "completed": "Response ready",
            "failed": "Run failed; review the result",
            "cancelled": "Run stopped",
            "stuck": "Run needs review",
        }.get(receipt.status, "New run result")
        rows.append(
            BuddyInboxEntry(
                f"result:{receipt.activity_id}",
                title,
                "results",
                summary,
                binding,
                (receipt.activity_id,),
            )
        )
    order = {"needs_you": 0, "running": 1, "results": 2}
    return tuple(sorted(rows, key=lambda row: order[row.group]))
