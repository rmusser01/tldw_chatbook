"""Checkpoint-backed send-history projection for one live verifier only."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_service import AgentConfig, ModelRequest

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


def review_history_for_request(
    self, config, messages, *, native, continuation_groups=()
):
    binding = config.review_binding
    service = getattr(self.registry, "_review_service", None)
    if (
        binding is None
        or binding.role != "verifier"
        or service is None
        or service.actor.kind != "subagent"
    ):
        return messages
    from .code_review_history import project_verifier_history

    checkpoint = service.verifier_checkpoint(binding.review_id)
    if (
        checkpoint.attempt_id != binding.attempt_id
        or checkpoint.target_digest != binding.target_digest
    ):
        from .code_review_models import ReviewRefused

        raise ReviewRefused("review_checkpoint_scope_mismatch")
    return project_verifier_history(
        messages, checkpoint, continuation_groups=continuation_groups, native=native
    )


def review_input_capacity(
    self, config: AgentConfig, api_endpoint: str, request: ModelRequest
) -> int | None:
    """Return remaining exact input tokens; unverified fallbacks are unknown."""
    from tldw_chatbook.Agents import agent_service as runtime
    from tldw_chatbook.Chat.console_prepared_request import resolve_request_capacity
    from tldw_chatbook.model_capabilities import get_model_capabilities
    from tldw_chatbook.Utils.token_counter import resolve_context_window

    try:
        window = resolve_context_window(api_endpoint, config.model)
        caps = get_model_capabilities().get_model_capabilities(
            api_endpoint, config.model
        )

        def cap(*names):
            return next(
                (
                    caps[name]
                    for name in names
                    if type(caps.get(name)) is int and caps[name] > 0
                ),
                None,
            )

        capacity = resolve_request_capacity(
            context_window_tokens=window.tokens if window.verified else None,
            context_window_verified=window.verified,
            provider_input_cap_tokens=cap(
                "max_input_tokens", "input_token_limit", "provider_input_cap"
            ),
            provider_output_cap_tokens=cap(
                "max_output_tokens", "output_token_limit", "provider_output_cap"
            ),
            requested_response_tokens=max(1, config.response_reserve_tokens),
        )
        ceiling = capacity.effective_input_ceiling_tokens
        if ceiling is None:
            return None
        used = runtime._count_model_messages(
            list(request.messages),
            config.model,
            api_endpoint,
            reasoning_replay=config.reasoning_replay,
        )
        if request.tools:
            used += runtime.estimate_tokens(
                runtime.json.dumps(list(request.tools), sort_keys=True, separators=(",", ":")),
                config.model,
                provider=api_endpoint,
            )
        if type(used) is not int or used < 0:
            return None
        return ceiling - used
    except Exception:  # noqa: BLE001 — unavailable accounting fails closed
        return None
