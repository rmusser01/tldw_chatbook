"""Checkpoint-backed send-history projection for one live verifier only."""

from __future__ import annotations

import json
from dataclasses import dataclass

from tldw_chatbook.Chat.provider_continuation import ContinuationOwnerGroup

from .agent_models import FENCE_TOOL_RESULT_PREFIX
from .run_log_eviction import _make_round_boundary


@dataclass(frozen=True)
class ReviewHistoryCheckpoint:
    attempt_id: str
    target_digest: str
    submission_revision: int
    acknowledged_cursors: tuple[str, ...] = ()
    cursor: str | None = None
    pending_unit_ids: tuple[str, ...] = ()
    evidence_refs: tuple[dict, ...] = ()
    check_refs: tuple[str, ...] = ()
    pending_check_refs: tuple[str, ...] = ()


def project_verifier_history(
    messages: list[dict],
    checkpoint: ReviewHistoryCheckpoint,
    *,
    continuation_groups: tuple[ContinuationOwnerGroup, ...],
    native: bool,
) -> list[dict]:
    """Replace only acknowledged whole rounds; preserve tasks and live protocol."""
    if not checkpoint.acknowledged_cursors:
        return messages
    boundary = _make_round_boundary(native=native)
    groups = []
    for row in messages:
        if not groups or boundary(row):
            groups.append([])
        groups[-1].append(row)
    acknowledged = -1
    for index, group in enumerate(groups):
        for row in group:
            text = row.get("content", "")
            if native:
                if row.get("role") != "tool":
                    continue
            elif (
                row.get("role") == "user"
                and isinstance(text, str)
                and text.startswith(FENCE_TOOL_RESULT_PREFIX + "code_review_submit: ")
            ):
                text = text.partition(": ")[2]
            else:
                continue
            try:
                ack = json.loads(text)
            except (ValueError, TypeError):
                continue
            if (
                isinstance(ack, dict)
                and ack.get("cursor") in checkpoint.acknowledged_cursors
                and ack.get("submission_revision", -1) <= checkpoint.submission_revision
                and "accepted_ids" in ack
            ):
                acknowledged = index
    if acknowledged < 0:
        return messages
    protected_owners = {owner.owner_message_id for owner in continuation_groups}
    protected_calls = {
        call.call_id
        for owner in continuation_groups
        for round_ in owner.rounds
        for call in round_.calls
    }
    retained, removed = [], False
    for index, group in enumerate(groups):
        protected = bool(checkpoint.pending_check_refs) or any(
            row.get("role") == "system"
            or any(
                value in protected_owners
                for value in row.values()
                if isinstance(value, str)
            )
            or row.get("_chatbook_project_instruction_row_key")
            or row.get("_chatbook_ephemeral_origin")
            or row.get("tool_call_id") in protected_calls
            or any(
                call.get("id") in protected_calls
                for call in row.get("tool_calls", ())
                if isinstance(call, dict)
            )
            for row in group
        )
        assistant = group[0].get("role") == "assistant"
        if index <= acknowledged and assistant and not protected:
            removed = True
        else:
            retained.extend(group)
    if not removed:
        return messages
    note = {
        "role": "user",
        "content": "Accepted verification checkpoint: "
        + json.dumps(
            {
                "submission_revision": checkpoint.submission_revision,
                "cursor": checkpoint.cursor,
                "pending_unit_ids": checkpoint.pending_unit_ids,
                "evidence_refs": checkpoint.evidence_refs,
                "check_refs": checkpoint.check_refs,
            },
            separators=(",", ":"),
        )
        + '. Accepted decisions are stored. Read code_review_read(view="checkpoint") '
        "for current submission_revision, batch_cursor and pending_unit_ids. Cached batch pages "
        "retain historical revisions. Do not resubmit accepted decisions; when all units are "
        "disposed submit verifier_complete at the current revision.",
    }
    insertion = next(
        (i for i, row in enumerate(retained) if row.get("role") == "assistant"),
        len(retained),
    )
    return retained[:insertion] + [note] + retained[insertion:]
