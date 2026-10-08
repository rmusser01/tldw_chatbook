"""Generic Console interrupt-round host (sub-project C1, task program spec
2026-08-20-console-interrupt-host-design.md).

One lifecycle, ONE lock, per-kind storage for the Console's five blocking
interrupt rounds: MCP approvals, skill-install confirms, skill-script
confirms, worktree-merge confirms, and ask_user questions (task-31384,
sub-project C of the 2026-08-19 design; the C1 spine from PR #1903 ported
to the five-kind controller).

Locking: ``lock`` is a plain NON-REENTRANT ``threading.Lock``. The
controller aliases its five historical lock names to this one object, so
nesting any two of them -- or calling a host method that takes the lock
from inside a ``with`` on any of the names -- self-deadlocks immediately.
Nothing nests today (verified at C1 design time, tests included); keep it
that way.

Dependencies: keyword-only named live accessors preserve action-time controller
and original-module lookups (ADR-220). Native registries, payloads and the one
lock remain here; separate chat-create, source, grant, launch and acceptance
authority remain controller-owned. Retained controller/module wrappers preserve
public and known patch routes. Seven private same-owner helpers call directly.
Full responsibility documentation follows each implementation below.
"""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tldw_chatbook.Agents.hook_permissions import HookPermissions
    from tldw_chatbook.Chat.console_chat_controller import (
        ApprovalDecision,
        ApprovalDecisions,
        BuiltinToolGate,
        BuiltinToolProvider,
        ConsolePendingDecisionProjection,
        LocalToolProvider,
        MCPPendingCall,
        MCPToolProvider,
        ManagedSkillProposalGate,
        RawShellToolProvider,
        ToolCall,
        ToolReviewDecision,
        ToolReviewValue,
        VirtualCliProvider,
    )

from loguru import logger

from tldw_chatbook.Agents.human_input_wait import use_human_input_wait
from tldw_chatbook.Chat.console_hook_review_host import InitialHookReviewMixin
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_PENDING_APPROVAL_KIND,
    CONSOLE_PENDING_HOOK_REVIEW_KIND,
    CONSOLE_PENDING_QUESTION_KIND,
    CONSOLE_PENDING_SKILL_INSTALL_KIND,
    CONSOLE_PENDING_SKILL_SCRIPT_KIND,
    CONSOLE_PENDING_WORKTREE_MERGE_KIND,
)

#: Kind -> the controller attribute holding that kind's UI setter. The
#: setters are attach-time assignments and may be absent entirely
#: (headless, or a kind not yet wired -- "question" until sub-project A);
#: every read goes through ``getattr(..., None)`` and treats None as
#: "no UI, no-op".
#: The kinds every session-activation site re-derives together
#: (`remount_for_session`); approvals are re-derived separately by the
#: sites' own approval block, which also drives the attach path.
SESSION_REMOUNT_KINDS: tuple[str, ...] = (
    CONSOLE_PENDING_SKILL_INSTALL_KIND,
    CONSOLE_PENDING_SKILL_SCRIPT_KIND,
    CONSOLE_PENDING_WORKTREE_MERGE_KIND,
    CONSOLE_PENDING_QUESTION_KIND,
)

KIND_SETTER_ATTRS: dict[str, str] = {
    CONSOLE_PENDING_APPROVAL_KIND: "set_pending_approval",
    CONSOLE_PENDING_HOOK_REVIEW_KIND: "set_pending_decision",
    CONSOLE_PENDING_SKILL_INSTALL_KIND: "set_pending_skill_install",
    CONSOLE_PENDING_SKILL_SCRIPT_KIND: "set_pending_skill_script",
    CONSOLE_PENDING_WORKTREE_MERGE_KIND: "set_pending_worktree_merge",
    CONSOLE_PENDING_QUESTION_KIND: "set_pending_question",
}


def head_payload_locked(
    store: dict[str, dict[str, Any]], session_id: str | None
) -> dict[str, Any] | None:
    """The session's oldest-armed payload in ``store``. Caller holds the lock.

    Args:
        store: A round-id -> payload dict.
        session_id: The session to look up.

    Returns:
        The first payload armed for ``session_id``, or None.
    """
    for payload in store.values():
        if payload.get("session_id") == session_id:
            return payload
    return None


def park_round_payload(
    lock: threading.Lock,
    store: dict[str, dict[str, Any]],
    round_id: str,
    payload: dict[str, Any],
) -> bool:
    """Retain ``payload`` under ``round_id``.

    Args:
        lock: The lock guarding ``store``.
        store: A round-id -> payload dict.
        round_id: The round's id.
        payload: The card payload; must carry ``session_id``.

    Returns:
        True when ``payload`` is now its session's head.
    """
    with lock:
        store[round_id] = payload
        head = head_payload_locked(store, payload.get("session_id"))
    return head is payload


def head_round_payload(
    lock: threading.Lock, store: dict[str, dict[str, Any]], session_id: str
) -> dict[str, Any] | None:
    """The payload whose card ``session_id`` should show now.

    Carries the PR #1836 remaining-time snapshot behaviour verbatim: a
    payload with a live ``deadline_monotonic`` is returned as a shallow
    copy whose ``timeout_seconds`` is the remaining window; the retained
    payload is never mutated.

    Args:
        lock: The lock guarding ``store``.
        store: A round-id -> payload dict.
        session_id: The session whose head to return.

    Returns:
        The head payload (or its remaining-time snapshot), or None.
    """
    with lock:
        payload = head_payload_locked(store, session_id)
    if payload is None:
        return None
    deadline = payload.get("deadline_monotonic")
    if not deadline:
        return payload
    snapshot = dict(payload)
    snapshot["timeout_seconds"] = max(0.0, deadline - time.monotonic())
    return snapshot


def session_round_payloads(
    lock: threading.Lock, store: dict[str, dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    """Every payload ``store`` retains for ``session_id``, arm order first.

    Args:
        lock: The lock guarding ``store``.
        store: A round-id -> payload dict.
        session_id: The session to collect.

    Returns:
        The session's payloads in arm order.
    """
    with lock:
        return [
            payload
            for payload in store.values()
            if payload.get("session_id") == session_id
        ]


def unpark_round_payload(
    lock: threading.Lock, store: dict[str, dict[str, Any]], round_id: str
) -> None:
    """Drop ``round_id``'s retained payload, if any.

    Args:
        lock: The lock guarding ``store``.
        store: A round-id -> payload dict.
        round_id: The round to forget.
    """
    with lock:
        store.pop(round_id, None)


@dataclass
class _DecisionClock:
    """Volatile answerable-time budget; never included in a card payload."""

    remaining: float
    updated_at: float
    payload: dict[str, Any]
    answerable: bool = False
    requires_head: bool = True

    def update(self, now: float, answerable: bool) -> None:
        if self.answerable:
            self.remaining = max(0.0, self.remaining - (now - self.updated_at))
        self.updated_at = now
        self.answerable = answerable
        self.payload["timeout_seconds"] = self.remaining
        self.payload["deadline_monotonic"] = (
            now + self.remaining if answerable else None
        )


class InterruptRoundHost(InitialHookReviewMixin):
    """Own the registries, payload maps, and FIFO-head render contract."""

    POLL_SECONDS = 1.0

    def notify_hook_interrupt(self, cancellation, lifecycle, turn_id: str) -> bool:
        """Publish once after the host's irreversible per-turn cancel seal.

        H2 owns notification deadlines and real process cleanup; no UI card,
        permission prompt or process wait runs under this host's registry lock.
        """
        if not cancellation.is_set():
            return False
        with self.lock:
            if cancellation in self._hook_interrupts:
                return False
            self._hook_interrupts.add(cancellation)
        return lifecycle.engine.notify_teardown(
            lifecycle.event("Interrupt", turn_id=turn_id, initiator="manual")
        )

    def __init__(
        self,
        *,
        read_controller__active_assistant_message_ids: Callable[[], Any],
        read_controller__advance_lifecycle_revision: Callable[[], Any],
        read_controller__agent_bridge: Callable[[], Any],
        read_controller__announce_detached_approval: Callable[[], Any],
        read_controller__announce_hidden_decision: Callable[[], Any],
        read_controller__announced_pending_decision_ids: Callable[[], Any],
        read_controller__answerable_decision_by_session: Callable[[], Any],
        read_controller__approval_view_is_detached: Callable[[], Any],
        read_controller__bind_round_cancel_signal: Callable[[], Any],
        read_controller__bind_visit_cancel_signal: Callable[[], Any],
        read_controller__buddy_sink: Callable[[], Any],
        read_controller__chat_create_session_grants: Callable[[], Any],
        read_controller__chat_creation_record_locked: Callable[[], Any],
        read_controller__chat_creation_records: Callable[[], Any],
        read_controller__chat_creation_revoked_runs: Callable[[], Any],
        read_controller__chat_start: Callable[[], Any],
        read_controller__console_answerable_decision_by_session: Callable[[], Any],
        read_controller__deliver_permission_summary: Callable[[], Any],
        read_controller__disposed: Callable[[], Any],
        read_controller__enrich_chat_create_confirm_payload: Callable[[], Any],
        read_controller__forget_hidden_decision: Callable[[], Any],
        read_controller__head_round_payload: Callable[[], Any],
        read_controller__interrupt_bell_enabled: Callable[[], Any],
        read_controller__is_session_cancelled: Callable[[], Any],
        read_controller__marshal_pending_chat_create: Callable[[], Any],
        read_controller__marshal_pending_decision_projection: Callable[[], Any],
        read_controller__maybe_fire_permission_summary: Callable[[], Any],
        read_controller__notify_run_hook_approval: Callable[[], Any],
        read_controller__observe_chat_creation_record: Callable[[], Any],
        read_controller__park_round_payload: Callable[[], Any],
        read_controller__parked_chat_create_payloads: Callable[[], Any],
        read_controller__pending_approvals: Callable[[], Any],
        read_controller__pending_chat_create_lock: Callable[[], Any],
        read_controller__pending_chat_create_rounds: Callable[[], Any],
        read_controller__pending_decision_order: Callable[[], Any],
        read_controller__pending_round_kinds: Callable[[], Any],
        read_controller__permission_summary_worker: Callable[[], Any],
        read_controller__provider_messages_for_session: Callable[[], Any],
        read_controller__publish_console_attention_change: Callable[[], Any],
        read_controller__publish_pending_decision: Callable[[], Any],
        read_controller__question_bounces: Callable[[], Any],
        read_controller__raw_shell_providers: Callable[[], Any],
        read_controller__record_cancelled_approval_decisions: Callable[[], Any],
        read_controller__refresh_answerable_decision: Callable[[], Any],
        read_controller__remount_head: Callable[[], Any],
        read_controller__remount_parked_chat_create: Callable[[], Any],
        read_controller__remount_parked_skill_install: Callable[[], Any],
        read_controller__remount_parked_skill_script: Callable[[], Any],
        read_controller__remount_session_kinds: Callable[[], Any],
        read_controller__reproject_pending_decision_for_session: Callable[[], Any],
        read_controller__resolve_ask_user_timeout_seconds: Callable[[], Any],
        read_controller__resolve_mcp_approval_timeout_seconds: Callable[[], Any],
        read_controller__revoke_chat_create_rounds: Callable[[], Any],
        read_controller__run_hooks_engine: Callable[[], Any],
        read_controller__session_close_generations: Callable[[], Any],
        read_controller__summary_tail_messages: Callable[[], Any],
        read_controller__unpark_round_payload: Callable[[], Any],
        read_controller_add_pending_round: Callable[[], Any],
        read_controller_announce_hidden_decision: Callable[[], Any],
        read_controller_app: Callable[[], Any],
        read_controller_ask_user_timeout_seconds: Callable[[], Any],
        read_controller_chat_create_confirm_timeout_seconds: Callable[[], Any],
        read_controller_decision_monotonic_clock: Callable[[], Any],
        read_controller_discard_pending_round: Callable[[], Any],
        read_controller_expire_pending_decisions: Callable[[], Any],
        read_controller_mcp_approval_timeout_seconds: Callable[[], Any],
        read_controller_on_console_attention_changed: Callable[[], Any],
        read_controller_on_pending_rounds_changed: Callable[[], Any],
        read_controller_park_pending_approval: Callable[[], Any],
        read_controller_pending_decision_projection: Callable[[], Any],
        read_controller_project_pending_decision_for_active_session: Callable[[], Any],
        read_controller_remount_pending_approval_for_active_session: Callable[[], Any],
        read_controller_set_answerable_decision: Callable[[], Any],
        read_controller_set_pending_approval: Callable[[], Any],
        read_controller_set_pending_chat_create: Callable[[], Any],
        read_controller_set_pending_decision: Callable[[], Any],
        read_controller_set_pending_question: Callable[[], Any],
        read_controller_set_pending_skill_install: Callable[[], Any],
        read_controller_set_pending_skill_script: Callable[[], Any],
        read_controller_set_pending_worktree_merge: Callable[[], Any],
        read_controller_set_task_panel: Callable[[], Any],
        read_controller_skill_install_confirm_timeout_seconds: Callable[[], Any],
        read_controller_skill_script_confirm_timeout_seconds: Callable[[], Any],
        read_controller_store: Callable[[], Any],
        read_controller_update_pending_approval_summary: Callable[[], Any],
        read_controller_worktree_merge_confirm_timeout_seconds: Callable[[], Any],
        write_controller__pending_decision_order: Callable[[Any], None],
        read_global_ASK_USER_TIMEOUT_ENV_VAR: Callable[[], Any],
        read_global_ApprovalDecisions: Callable[[], Any],
        read_global_CONSOLE_PENDING_APPROVAL_KIND: Callable[[], Any],
        read_global_CONSOLE_PENDING_CHAT_CREATE_KIND: Callable[[], Any],
        read_global_ConsolePendingDecisionProjection: Callable[[], Any],
        read_global_INTERRUPT_BELL_ENV_VAR: Callable[[], Any],
        read_global_ToolExecutionPolicy: Callable[[], Any],
        read_global_UNRESOLVED_DENIED_DECISION: Callable[[], Any],
        read_global__ChatCreationToken: Callable[[], Any],
        read_global__DEFAULT_ASK_USER_TIMEOUT_SECONDS: Callable[[], Any],
        read_global__DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS: Callable[[], Any],
        read_global__DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS: Callable[[], Any],
        read_global__DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS: Callable[[], Any],
        read_global__DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS: Callable[[], Any],
        read_global__DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS: Callable[[], Any],
        read_global__LEGACY_PENDING_APPROVAL_ROUND_ID: Callable[[], Any],
        read_global__MAX_TRACKED_QUESTION_BOUNCE_RUNS: Callable[[], Any],
        read_global__MCP_APPROVAL_POLL_SECONDS: Callable[[], Any],
        read_global__REVOCATION_STAMPS: Callable[[], Any],
        read_global__bool_or_none: Callable[[], Any],
        read_global__build_approval_payload: Callable[[], Any],
        read_global__normalize_world_info_history: Callable[[], Any],
        read_global_contextlib: Callable[[], Any],
        read_global_current_run_actor: Callable[[], Any],
        read_global_current_run_id: Callable[[], Any],
        read_global_escape_markup: Callable[[], Any],
        read_global_get_cli_setting: Callable[[], Any],
        read_global_get_runtime_config_snapshot: Callable[[], Any],
        read_global_logger: Callable[[], Any],
        read_global_os: Callable[[], Any],
        read_global_threading: Callable[[], Any],
        read_global_time: Callable[[], Any],
        read_global_uuid4: Callable[[], Any],
        read_hook_review_owner: Callable[[], HookPermissions | None] | None = None,
        read_hook_review_shutdown: Callable[[str], bool] | None = None,
    ):
        """Store named live accessors; preserve existing native host initialization separately."""
        self.read_controller__active_assistant_message_ids = (
            read_controller__active_assistant_message_ids
        )
        self.read_controller__advance_lifecycle_revision = (
            read_controller__advance_lifecycle_revision
        )
        self.read_controller__agent_bridge = read_controller__agent_bridge
        self.read_controller__announce_detached_approval = (
            read_controller__announce_detached_approval
        )
        self.read_controller__announce_hidden_decision = (
            read_controller__announce_hidden_decision
        )
        self.read_controller__announced_pending_decision_ids = (
            read_controller__announced_pending_decision_ids
        )
        self.read_controller__answerable_decision_by_session = (
            read_controller__answerable_decision_by_session
        )
        self.read_controller__approval_view_is_detached = (
            read_controller__approval_view_is_detached
        )
        self.read_controller__bind_round_cancel_signal = (
            read_controller__bind_round_cancel_signal
        )
        self.read_controller__bind_visit_cancel_signal = (
            read_controller__bind_visit_cancel_signal
        )
        self.read_controller__buddy_sink = read_controller__buddy_sink
        self.read_controller__chat_create_session_grants = (
            read_controller__chat_create_session_grants
        )
        self.read_controller__chat_creation_record_locked = (
            read_controller__chat_creation_record_locked
        )
        self.read_controller__chat_creation_records = (
            read_controller__chat_creation_records
        )
        self.read_controller__chat_creation_revoked_runs = (
            read_controller__chat_creation_revoked_runs
        )
        self.read_controller__chat_start = read_controller__chat_start
        self.read_controller__console_answerable_decision_by_session = (
            read_controller__console_answerable_decision_by_session
        )
        self.read_controller__deliver_permission_summary = (
            read_controller__deliver_permission_summary
        )
        self.read_controller__disposed = read_controller__disposed
        self.read_controller__enrich_chat_create_confirm_payload = (
            read_controller__enrich_chat_create_confirm_payload
        )
        self.read_controller__forget_hidden_decision = (
            read_controller__forget_hidden_decision
        )
        self.read_controller__head_round_payload = read_controller__head_round_payload
        self.read_controller__interrupt_bell_enabled = (
            read_controller__interrupt_bell_enabled
        )
        self.read_controller__is_session_cancelled = (
            read_controller__is_session_cancelled
        )
        self.read_controller__marshal_pending_chat_create = (
            read_controller__marshal_pending_chat_create
        )
        self.read_controller__marshal_pending_decision_projection = (
            read_controller__marshal_pending_decision_projection
        )
        self.read_controller__maybe_fire_permission_summary = (
            read_controller__maybe_fire_permission_summary
        )
        self.read_controller__notify_run_hook_approval = (
            read_controller__notify_run_hook_approval
        )
        self.read_controller__observe_chat_creation_record = (
            read_controller__observe_chat_creation_record
        )
        self.read_controller__park_round_payload = read_controller__park_round_payload
        self.read_controller__parked_chat_create_payloads = (
            read_controller__parked_chat_create_payloads
        )
        self.read_controller__pending_approvals = read_controller__pending_approvals
        self.read_controller__pending_chat_create_lock = (
            read_controller__pending_chat_create_lock
        )
        self.read_controller__pending_chat_create_rounds = (
            read_controller__pending_chat_create_rounds
        )
        self.read_controller__pending_decision_order = (
            read_controller__pending_decision_order
        )
        self.read_controller__pending_round_kinds = read_controller__pending_round_kinds
        self.read_controller__permission_summary_worker = (
            read_controller__permission_summary_worker
        )
        self.read_controller__provider_messages_for_session = (
            read_controller__provider_messages_for_session
        )
        self.read_controller__publish_console_attention_change = (
            read_controller__publish_console_attention_change
        )
        self.read_controller__publish_pending_decision = (
            read_controller__publish_pending_decision
        )
        self.read_controller__question_bounces = read_controller__question_bounces
        self.read_controller__raw_shell_providers = read_controller__raw_shell_providers
        self.read_controller__record_cancelled_approval_decisions = (
            read_controller__record_cancelled_approval_decisions
        )
        self.read_controller__refresh_answerable_decision = (
            read_controller__refresh_answerable_decision
        )
        self.read_controller__remount_head = read_controller__remount_head
        self.read_controller__remount_parked_chat_create = (
            read_controller__remount_parked_chat_create
        )
        self.read_controller__remount_parked_skill_install = (
            read_controller__remount_parked_skill_install
        )
        self.read_controller__remount_parked_skill_script = (
            read_controller__remount_parked_skill_script
        )
        self.read_controller__remount_session_kinds = (
            read_controller__remount_session_kinds
        )
        self.read_controller__reproject_pending_decision_for_session = (
            read_controller__reproject_pending_decision_for_session
        )
        self.read_controller__resolve_ask_user_timeout_seconds = (
            read_controller__resolve_ask_user_timeout_seconds
        )
        self.read_controller__resolve_mcp_approval_timeout_seconds = (
            read_controller__resolve_mcp_approval_timeout_seconds
        )
        self.read_controller__revoke_chat_create_rounds = (
            read_controller__revoke_chat_create_rounds
        )
        self.read_controller__run_hooks_engine = read_controller__run_hooks_engine
        self.read_controller__session_close_generations = (
            read_controller__session_close_generations
        )
        self.read_controller__summary_tail_messages = (
            read_controller__summary_tail_messages
        )
        self.read_controller__unpark_round_payload = (
            read_controller__unpark_round_payload
        )
        self.read_controller_add_pending_round = read_controller_add_pending_round
        self.read_controller_announce_hidden_decision = (
            read_controller_announce_hidden_decision
        )
        self.read_controller_app = read_controller_app
        self.read_controller_ask_user_timeout_seconds = (
            read_controller_ask_user_timeout_seconds
        )
        self.read_controller_chat_create_confirm_timeout_seconds = (
            read_controller_chat_create_confirm_timeout_seconds
        )
        self.read_controller_decision_monotonic_clock = (
            read_controller_decision_monotonic_clock
        )
        self.read_controller_discard_pending_round = (
            read_controller_discard_pending_round
        )
        self.read_controller_expire_pending_decisions = (
            read_controller_expire_pending_decisions
        )
        self.read_controller_mcp_approval_timeout_seconds = (
            read_controller_mcp_approval_timeout_seconds
        )
        self.read_controller_on_console_attention_changed = (
            read_controller_on_console_attention_changed
        )
        self.read_controller_on_pending_rounds_changed = (
            read_controller_on_pending_rounds_changed
        )
        self.read_controller_park_pending_approval = (
            read_controller_park_pending_approval
        )
        self.read_controller_pending_decision_projection = (
            read_controller_pending_decision_projection
        )
        self.read_controller_project_pending_decision_for_active_session = (
            read_controller_project_pending_decision_for_active_session
        )
        self.read_controller_remount_pending_approval_for_active_session = (
            read_controller_remount_pending_approval_for_active_session
        )
        self.read_controller_set_answerable_decision = (
            read_controller_set_answerable_decision
        )
        self.read_controller_set_pending_approval = read_controller_set_pending_approval
        self.read_controller_set_pending_chat_create = (
            read_controller_set_pending_chat_create
        )
        self.read_controller_set_pending_decision = read_controller_set_pending_decision
        self.read_controller_set_pending_question = read_controller_set_pending_question
        self.read_controller_set_pending_skill_install = (
            read_controller_set_pending_skill_install
        )
        self.read_controller_set_pending_skill_script = (
            read_controller_set_pending_skill_script
        )
        self.read_controller_set_pending_worktree_merge = (
            read_controller_set_pending_worktree_merge
        )
        self.read_controller_set_task_panel = read_controller_set_task_panel
        self.read_controller_skill_install_confirm_timeout_seconds = (
            read_controller_skill_install_confirm_timeout_seconds
        )
        self.read_controller_skill_script_confirm_timeout_seconds = (
            read_controller_skill_script_confirm_timeout_seconds
        )
        self.read_controller_store = read_controller_store
        self.read_controller_update_pending_approval_summary = (
            read_controller_update_pending_approval_summary
        )
        self.read_controller_worktree_merge_confirm_timeout_seconds = (
            read_controller_worktree_merge_confirm_timeout_seconds
        )
        self.write_controller__pending_decision_order = (
            write_controller__pending_decision_order
        )
        self.read_global_ASK_USER_TIMEOUT_ENV_VAR = read_global_ASK_USER_TIMEOUT_ENV_VAR
        self.read_global_ApprovalDecisions = read_global_ApprovalDecisions
        self.read_global_CONSOLE_PENDING_APPROVAL_KIND = (
            read_global_CONSOLE_PENDING_APPROVAL_KIND
        )
        self.read_global_CONSOLE_PENDING_CHAT_CREATE_KIND = (
            read_global_CONSOLE_PENDING_CHAT_CREATE_KIND
        )
        self.read_global_ConsolePendingDecisionProjection = (
            read_global_ConsolePendingDecisionProjection
        )
        self.read_global_INTERRUPT_BELL_ENV_VAR = read_global_INTERRUPT_BELL_ENV_VAR
        self.read_global_ToolExecutionPolicy = read_global_ToolExecutionPolicy
        self.read_global_UNRESOLVED_DENIED_DECISION = (
            read_global_UNRESOLVED_DENIED_DECISION
        )
        self.read_global__ChatCreationToken = read_global__ChatCreationToken
        self.read_global__DEFAULT_ASK_USER_TIMEOUT_SECONDS = (
            read_global__DEFAULT_ASK_USER_TIMEOUT_SECONDS
        )
        self.read_global__DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS = (
            read_global__DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS
        )
        self.read_global__DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = (
            read_global__DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS
        )
        self.read_global__DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS = (
            read_global__DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS
        )
        self.read_global__DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS = (
            read_global__DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS
        )
        self.read_global__DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS = (
            read_global__DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS
        )
        self.read_global__LEGACY_PENDING_APPROVAL_ROUND_ID = (
            read_global__LEGACY_PENDING_APPROVAL_ROUND_ID
        )
        self.read_global__MAX_TRACKED_QUESTION_BOUNCE_RUNS = (
            read_global__MAX_TRACKED_QUESTION_BOUNCE_RUNS
        )
        self.read_global__MCP_APPROVAL_POLL_SECONDS = (
            read_global__MCP_APPROVAL_POLL_SECONDS
        )
        self.read_global__REVOCATION_STAMPS = read_global__REVOCATION_STAMPS
        self.read_global__bool_or_none = read_global__bool_or_none
        self.read_global__build_approval_payload = read_global__build_approval_payload
        self.read_global__normalize_world_info_history = (
            read_global__normalize_world_info_history
        )
        self.read_global_contextlib = read_global_contextlib
        self.read_global_current_run_actor = read_global_current_run_actor
        self.read_global_current_run_id = read_global_current_run_id
        self.read_global_escape_markup = read_global_escape_markup
        self.read_global_get_cli_setting = read_global_get_cli_setting
        self.read_global_get_runtime_config_snapshot = (
            read_global_get_runtime_config_snapshot
        )
        self.read_global_logger = read_global_logger
        self.read_global_os = read_global_os
        self.read_global_threading = read_global_threading
        self.read_global_time = read_global_time
        self.read_global_uuid4 = read_global_uuid4
        self.read_hook_review_owner = read_hook_review_owner
        self.read_hook_review_shutdown = read_hook_review_shutdown
        self.lock = threading.Lock()
        self._hook_interrupts: set[object] = set()
        self.registries: dict[str, dict[str, dict[str, Any]]] = {
            kind: {} for kind in KIND_SETTER_ATTRS
        }
        # Host-lifetime fences: logical completion cannot prove that an
        # abandoned provider daemon will never reach its approval fallback.
        self._revoked_runs: dict[str, set[str]] = {
            kind: set() for kind in KIND_SETTER_ATTRS
        }
        #: Per-kind hook run on the UI thread after a head payload is pushed,
        #: whichever path pushed it (teardown promotion, revocation sweep,
        #: activation, attach). Approvals register their ADR-090 permission
        #: summary trigger here once, so no remount path can forget it.
        self.after_remount: dict[str, Callable[[dict[str, Any]], None]] = {}
        self.payloads: dict[str, dict[str, dict[str, Any]]] = {
            kind: {} for kind in KIND_SETTER_ATTRS
        }
        # None preserves the setter-based contract for non-Textual callers.
        # A reused screen explicitly reports suspend/resume, including modals.
        self.view_visible: bool | None = None
        self.decision_view_revision = 0
        self._decision_views: dict[
            object,
            tuple[
                str,
                frozenset[str],
                tuple[weakref.ReferenceType[Any], int, bool, str] | None,
            ],
        ] = {}
        self._retained_decision_targets: dict[
            str, tuple[weakref.ReferenceType[Any], int, bool]
        ] = {}

    def retain_decision_target(self, session_id: str) -> None:
        """An explicit Buddy opener makes this exact session's cards recoverable."""
        session = next(
            (
                row
                for row in self.read_controller_store().sessions()
                if row.id == session_id
            ),
            None,
        )
        if session is not None:
            with self.lock:
                self._retained_decision_targets[session_id] = (
                    weakref.ref(session),
                    session.conversation_binding_revision,
                    session.ephemeral,
                )

    def has_retained_decision_target(self, session_id: str | None) -> bool:
        """Wake-only, closed, replaced or repurposed slots gain no Buddy capability."""
        with self.lock:
            retained = self._retained_decision_targets.get(session_id)
        if retained is None:
            return False
        reference, revision, ephemeral = retained
        session = reference()
        return bool(
            session is not None
            and session.runtime_backend == "local"
            and session.conversation_binding_revision == revision
            and session.ephemeral == ephemeral
            and any(row is session for row in self.read_controller_store().sessions())
        )

    def set_decision_view(
        self,
        owner: object,
        session_id: str | None,
        *,
        kinds: tuple[str, ...] = tuple(KIND_SETTER_ATTRS),
        decision_id: str | None = None,
    ) -> None:
        """Claim/release a visible exact-session decision projection, such as Buddy."""
        session = (
            next(
                (
                    row
                    for row in self.read_controller_store().sessions()
                    if row.id == session_id
                ),
                None,
            )
            if decision_id is not None
            else None
        )
        rendered = (
            (
                weakref.ref(session),
                session.conversation_binding_revision,
                session.ephemeral,
                decision_id,
            )
            if session is not None
            else None
        )
        with self.lock:
            previous = self._decision_views.get(owner)
            self.decision_view_revision += 1
            if session_id is None:
                self._decision_views.pop(owner, None)
            else:
                self._decision_views[owner] = (session_id, frozenset(kinds), rendered)
        self.refresh_decision_clocks()
        refresh = self.read_controller__refresh_answerable_decision()
        if callable(refresh):
            for target in {session_id, previous[0] if previous else None} - {None}:
                refresh(target)

    def rendered_decision_ids(self, session_id: str) -> set[str]:
        """Snapshot live exact-binding card claims without retaining the host lock."""
        with self.lock:
            views = tuple(self._decision_views.values())
        sessions = self.read_controller_store().sessions()
        result = set()
        for target, _kinds, rendered in views:
            if target != session_id or rendered is None:
                continue
            reference, revision, ephemeral, decision_id = rendered
            session = reference()
            if (
                session is not None
                and session.conversation_binding_revision == revision
                and session.ephemeral == ephemeral
                and any(row is session for row in sessions)
            ):
                result.add(decision_id)
        return result

    def set_view_visible(self, visible: bool) -> None:
        """Account for the old visibility interval before changing clock activity."""
        self.view_visible = visible
        self.refresh_decision_clocks()
        if not visible:
            self.announce_hidden_decisions()

    def refresh_decision_clocks(self) -> None:
        """Charge only a visible session's FIFO head, across every decision kind."""
        with self.lock:
            active_session_id = self._active_session_id()
            now = time.monotonic()
            for kind, states in self.registries.items():
                for state in states.values():
                    clock = state.get("decision_clock")
                    if not isinstance(clock, _DecisionClock):
                        continue
                    session_id = state.get("session_id")
                    head = head_payload_locked(self.payloads[kind], session_id)
                    answerable = (
                        (
                            self.view_visible is not False
                            and self._setter(kind) is not None
                            and (not session_id or session_id == active_session_id)
                        )
                        or any(
                            session_id == target and kind in kinds
                            for target, kinds, _rendered in self._decision_views.values()
                        )
                    ) and (not clock.requires_head or head is clock.payload)
                    clock.update(now, answerable)

    def announce_hidden_decisions(self) -> None:
        """Notify once per live round, including one hidden after it was mounted."""
        announce = self.read_controller_announce_hidden_decision()
        if not callable(announce):
            return
        with self.lock:
            pending = []
            typed_pending = []
            for kind, states in self.registries.items():
                for state in states.values():
                    if state.get("decision_type"):
                        typed_pending.append(
                            (
                                kind,
                                state.get("session_id", ""),
                                state.get("decision_id", ""),
                            )
                        )
                        continue
                    if state.get("attention_announced") or state.get("revoked"):
                        continue
                    state["attention_announced"] = True
                    pending.append((state.get("session_id", ""), kind))
        for session_id, kind in pending:
            announce(session_id, kind)
        for kind, session_id, decision_id in typed_pending:
            self.read_controller__announce_hidden_decision()(
                kind, session_id, decision_id
            )

    # -- setter / app access (always late-bound) -----------------------

    def _setter(self, kind: str):
        if kind == CONSOLE_PENDING_HOOK_REVIEW_KIND:
            return lambda payload: self._hook_review_project_current()
        return {
            CONSOLE_PENDING_APPROVAL_KIND: self.read_controller_set_pending_approval,
            CONSOLE_PENDING_SKILL_INSTALL_KIND: self.read_controller_set_pending_skill_install,
            CONSOLE_PENDING_SKILL_SCRIPT_KIND: self.read_controller_set_pending_skill_script,
            CONSOLE_PENDING_WORKTREE_MERGE_KIND: self.read_controller_set_pending_worktree_merge,
            CONSOLE_PENDING_QUESTION_KIND: self.read_controller_set_pending_question,
        }[kind]()

    def _active_session_id(self) -> str:
        store = self.read_controller_store()
        return (getattr(store, "active_session_id", None) or "") if store else ""

    # -- payload layer (moved verbatim from ConsoleChatController) -----

    def park_round_payload(
        self, kind: str, round_id: str, payload: dict[str, Any]
    ) -> bool:
        """Retain ``payload``; return whether it is now its session's head."""
        return park_round_payload(self.lock, self.payloads[kind], round_id, payload)

    def head_round_payload(self, kind: str, session_id: str) -> dict[str, Any] | None:
        """The payload whose card ``session_id`` should show (remaining-time snapshot)."""
        return head_round_payload(self.lock, self.payloads[kind], session_id)

    def session_round_payloads(
        self, kind: str, session_id: str
    ) -> list[dict[str, Any]]:
        """Every payload the kind retains for ``session_id``, arm order first."""
        return session_round_payloads(self.lock, self.payloads[kind], session_id)

    def unpark_round_payload(self, kind: str, round_id: str) -> None:
        """Drop ``round_id``'s retained payload, if any."""
        unpark_round_payload(self.lock, self.payloads[kind], round_id)

    def remount_head(
        self,
        kind: str,
        session_id: str | None,
        *,
        after: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Enqueue a head re-derive onto the UI thread (worker-safe).

        The decision -- WHICH payload, and whether the session is still
        the one being viewed -- is computed INSIDE the callable, on the
        UI thread, never from a worker-thread snapshot: the invariant
        three pre-PR0 fix rounds converged on.

        ``session_id=None`` means "the session being VIEWED when the
        callback runs" (legacy no-session rounds mount unconditionally,
        so their card can sit over any session by teardown time).

        Args:
            kind: The round kind whose head to re-derive.
            session_id: The session to re-derive for, or None for the
                session viewed when the callback runs.
            after: task-31384: a hook run on the UI thread with the pushed
                payload, only when a head was pushed (the approvals bridge
                fires its permission summary here).
        """
        app = self.read_controller_app()
        if app is None or self._setter(kind) is None:
            return

        def _apply() -> None:
            setter = self._setter(kind)
            if setter is None:
                return
            target = session_id
            if target is None:
                target = self._active_session_id()
            elif target != self._active_session_id():
                return
            self.refresh_decision_clocks()
            payload = self.head_round_payload(kind, target)
            setter(payload)
            hook = after if after is not None else self.after_remount.get(kind)
            if hook is not None and isinstance(payload, dict):
                hook(payload)

        app.call_from_thread(_apply)

    def pending_total(self) -> int:
        """How many rounds of every kind are registered right now.

        Returns:
            The number of registered interrupt rounds across all kinds.
        """
        with self.lock:
            return sum(len(rounds) for rounds in self.registries.values())

    def _note_pending(self, kind: str, *, raised: bool) -> None:
        """task-31385: tell the seams the pending-round total changed.

        Late-bound like every other seam: a controller (or test double)
        without ``on_pending_rounds_changed`` hears nothing. ``raised`` is
        True right after a round mounted or parked and False after its
        registry entry was popped in teardown.

        Args:
            kind: The round kind that changed.
            raised: Whether this is the arm (True) or the teardown (False).
        """
        hook = self.read_controller_on_pending_rounds_changed()
        if hook is None:
            return
        try:
            hook(self.pending_total(), kind, raised)
        except Exception:  # noqa: BLE001 -- attention is best-effort; the round and its teardown are not
            logger.opt(exception=True).debug(
                f"Pending-round attention hook failed for {kind}"
            )

    def register_round(
        self,
        kind: str,
        round_id: str,
        state: dict[str, Any],
        *,
        check_revoked: bool = True,
    ) -> bool:
        """Admit a round atomically with its owner's revocation fence.

        Shared by early controller admission and the host lifecycle. A refused
        preregistration is removed only when it belongs to this exact state.

        Args:
            kind: Interrupt kind identifying the host's registry.
            round_id: Identifier under which to register this round.
            state: Mutable round state, including optional ``run_id`` and
                ``revoked`` fields. Refusal stamps ``revoked=True``.
            check_revoked: Whether to reject an already-revoked state or an
                owner fenced for this kind. False opts out of both checks and
                the unowned-arm warning for primary-only round lifecycles.

        Returns:
            True when the state is registered and admission may continue.
            False when the state was already revoked or its owner was fenced;
            callers must skip publication and return the normal denied outcome.

        Raises:
            KeyError: If ``kind`` has no host registry.
        """
        with self.lock:
            registry = self.registries[kind]
            run_id = state.get("run_id")
            if check_revoked and (
                state.get("revoked") or run_id in self._revoked_runs[kind]
            ):
                state["revoked"] = True
                if registry.get(round_id) is state:
                    registry.pop(round_id)
                return False
            warn_unowned = (
                check_revoked and not run_id and registry.get(round_id) is not state
            )
            registry[round_id] = state
        if warn_unowned:
            logger.warning("Arming a revocable interrupt round without a run owner")
        return True

    def revoke_for_run(
        self,
        run_id: str,
        stamps: dict[str, Callable[[dict[str, Any]], None]],
    ) -> dict[str, list[tuple[str, str | None]]]:
        """Fence future arms and fail armed rounds owned by ``run_id`` closed.

        Each swept round is marked ``revoked``, stamped closed by its
        kind's callable (approvals deny every undecided key; a skill
        script clears allow/remember; a question needs nothing), removed
        from its registry, unparked, and released via its Event -- the
        controller then discards the fleet badge and re-derives each
        affected session's head, exactly as the per-kind sweeps did.

        Args:
            run_id: The cancelled/abandoned run. Falsy ids sweep nothing.
            stamps: ``kind -> stamp(state)`` for every kind that is swept.
                A kind absent from the map is not swept (skill-install and
                worktree-merge are primary-agent-only and never are).

        Returns:
            ``kind -> [(round_id, session_id), ...]`` for the swept rounds.
        """
        swept: dict[str, list[tuple[str, str | None]]] = {kind: [] for kind in stamps}
        if not run_id:
            return swept
        with self.lock:
            for kind, stamp in stamps.items():
                self._revoked_runs[kind].add(run_id)
                registry = self.registries[kind]
                for round_id, state in list(registry.items()):
                    if state.get("run_id") != run_id:
                        continue
                    state["revoked"] = True
                    stamp(state)
                    registry.pop(round_id, None)
                    swept[kind].append((round_id, state.get("session_id") or None))
                    event = state.get("event")
                    if event is not None:
                        event.set()
        for kind, rounds in swept.items():
            for round_id, _session_id in rounds:
                self.unpark_round_payload(kind, round_id)
        return swept

    def remount_for_session(
        self,
        session_id: str,
        *,
        after: dict[str, Callable[[dict[str, Any]], None]] | None = None,
        kinds: tuple[str, ...] | None = None,
    ) -> None:
        """UI THREAD: push every kind's head for ``session_id`` in one call.

        Args:
            session_id: The session now being activated/viewed.
            after: Optional ``kind -> hook(payload)`` run after that kind's
                setter when a head payload was pushed (task-31384: the
                approvals bridge fires its permission summary here).
            kinds: The kinds to push; every kind when None.
        """
        for kind in kinds if kinds is not None else tuple(KIND_SETTER_ATTRS):
            setter = self._setter(kind)
            if setter is None:
                continue
            payload = self.head_round_payload(kind, session_id)
            setter(payload)
            hook = (after or {}).get(kind)
            if hook is not None and isinstance(payload, dict):
                hook(payload)

    # -- generic round lifecycle ----------------------------------------

    def run_round(
        self,
        kind: str,
        round_id: str,
        payload: dict[str, Any],
        state: dict[str, Any],
        *,
        session_id: str | None,
        owning_session_id: str,
        deadline: float | None,
        is_parked: bool,
        hard_deadline: float | None = None,
        announce_detached: Callable[[], bool] | None = None,
        human_wait_run_id: str | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_timeout: Callable[[], None] | None = None,
        check_revoked: bool = True,
        on_teardown: Callable[[], bool] | None = None,
        before_wait: Callable[[], None] | None = None,
        on_outcome: Callable[[str], None] | None = None,
    ) -> str:
        """One blocking interrupt round, registration through teardown.

        WORKER THREAD. Reproduces the (converged) bridge lifecycle:
        register -> badge -> park -> announce/park-toast/mount -> poll ->
        teardown (pop, unpark, badge-discard, head re-derive). The
        per-bridge deltas ride the hooks: ``announce_detached`` is the
        MCP detached-view leg, called after the park and returning True
        when it announced app-wide (the card is then not mounted);
        ``on_cancelled``/``on_timeout`` let the approvals wrapper stamp
        its decisions box and audit-log; ``human_wait_run_id`` wraps the
        wait in ``use_human_input_wait`` (every bridge wraps its wait
        today -- ``None`` selects ``nullcontext()`` instead, for a
        kind/test with no owning run to pause); ``check_revoked`` is False
        for skill-install and worktree-merge, which are never swept;
        ``on_outcome`` runs with the final outcome AFTER the wait and
        BEFORE teardown, so a wrapper can snapshot its decisions, write
        its audit rows or transcript marker while the round is still
        registered and its card still mounted; ``on_teardown``
        (task-31384) lets a kind retain its payload past teardown by
        returning True; ``before_wait`` runs once inside the wait mark
        before polling begins. The teardown head re-derive applies the
        kind's ``after_remount`` hook, like every other remount path.

        Args:
            kind: The round kind (a ``KIND_SETTER_ATTRS`` key).
            round_id: The round's unique id.
            payload: The card payload to park and mount.
            state: The registry entry; must carry an ``event`` and may
                carry ``cancel_event``/``visit_event``/``revoked``.
            session_id: The owning session for badge and park bookkeeping,
                or None for the legacy no-session shape.
            owning_session_id: The session whose head is re-derived at
                teardown.
            deadline: Initial ``time.monotonic()`` deadline, or None to wait
                indefinitely. Finite budgets advance only while the owning
                session's FIFO head can be answered on the visible Console.
            hard_deadline: Optional absolute host deadline, including parked and
                hidden time. It never resets or pauses with the UI clock.
            is_parked: True when the round belongs to a non-viewed session.
            announce_detached: Detached-view announcer; returns True when
                it announced instead of mounting.
            human_wait_run_id: Run id for ``use_human_input_wait``, or None.
            on_cancelled: Runs once when the session cancel fires mid-wait.
            on_timeout: Runs once when the answerable-time budget is exhausted.
            check_revoked: Whether a ``revoked`` stamp wins over "decided".
            on_teardown: Returns True to keep the payload parked.
            before_wait: Runs once inside the wait mark before polling.
            on_outcome: Receives the outcome before teardown.

        Returns:
            ``"decided"``, ``"cancelled"``, ``"timeout"`` or ``"revoked"``.
        """
        event: threading.Event = state["event"]

        def hard_expired():
            return hard_deadline is not None and time.monotonic() >= hard_deadline

        notified_outcomes = set()

        def notify_outcome(value):
            if value in notified_outcomes:
                return
            notified_outcomes.add(value)
            callback = on_timeout if value == "timeout" else on_cancelled
            if callback is not None:
                callback()

        if not self.register_round(kind, round_id, state, check_revoked=check_revoked):
            if on_outcome is not None:
                on_outcome("revoked")
            return "revoked"
        if not hard_expired() and kind in {
            CONSOLE_PENDING_APPROVAL_KIND,
            CONSOLE_PENDING_SKILL_INSTALL_KIND,
            CONSOLE_PENDING_SKILL_SCRIPT_KIND,
            CONSOLE_PENDING_WORKTREE_MERGE_KIND,
        }:
            with self.lock:
                notify_hook = not state.get("run_hook_notified", False)
                state["run_hook_notified"] = True
            notify = self.read_controller__notify_run_hook_approval()
            if notify_hook and callable(notify):
                try:
                    notify(kind, payload, state)
                except Exception:
                    logger.warning("ApprovalRequested hook notification failed")
        is_head = True
        publish_decision = self.read_controller__publish_pending_decision()
        retained_decision = (
            session_id is not None
            and kind
            in {
                CONSOLE_PENDING_APPROVAL_KIND,
                CONSOLE_PENDING_SKILL_INSTALL_KIND,
                CONSOLE_PENDING_SKILL_SCRIPT_KIND,
            }
            and callable(publish_decision)
        )
        if retained_decision and not hard_expired():
            # The controller's accepted-time metadata uses this same lock.
            # Enter the hook only after releasing the registration lock.
            is_head = publish_decision(
                round_state=state,
                payload=payload,
                decision_type=kind,
                decision_id=round_id,
                timeout_seconds=float(payload.get("timeout_seconds") or 0),
                retained_store=self.payloads[kind],
            )
            deadline = None
        clock = None
        if deadline is not None:
            now = time.monotonic()
            clock = _DecisionClock(
                max(0.0, float(payload.get("timeout_seconds", deadline - now))),
                now,
                payload,
                requires_head=session_id is not None,
            )
            with self.lock:
                state["decision_clock"] = clock
        if session_id is not None:
            add = self.read_controller_add_pending_round()
            if add is not None:
                # Qodo #4: the badge does not care which kind is waiting, but
                # the run chip and activity line do -- passing it here is
                # what lets them say "Waiting for your answer" for a question
                # instead of claiming an approval is pending. Optional by
                # keyword so the many two-argument controller doubles keep
                # working; `TypeError` means an older seam, not a bug.
                try:
                    add(session_id, round_id, kind=kind)
                except TypeError:
                    add(session_id, round_id)
            if not retained_decision:
                is_head = self.park_round_payload(kind, round_id, payload)
        try:
            app = self.read_controller_app()
            park_toast = self.read_controller_park_pending_approval()
            if hard_expired():
                pass
            elif retained_decision:
                if (
                    self.read_controller__approval_view_is_detached()()
                    or is_parked
                    or not is_head
                ):
                    self.read_controller__announce_hidden_decision()(
                        kind, owning_session_id, round_id
                    )
                elif self.read_controller_set_pending_decision() is not None:
                    self.read_controller__marshal_pending_decision_projection()()
                else:
                    setter = self._setter(kind)
                    if app is not None and setter is not None:
                        app.call_from_thread(setter, payload)
            elif announce_detached is not None and announce_detached():
                with self.lock:
                    state["attention_announced"] = True
            elif self.view_visible is False:
                self.announce_hidden_decisions()
            elif is_parked:
                if app is not None and park_toast is not None:
                    app.call_from_thread(park_toast, session_id)
            elif is_head:
                setter = self._setter(kind)
                if app is not None and setter is not None:
                    app.call_from_thread(setter, payload)
            self.refresh_decision_clocks()
            # A sweep can revoke and pop the round between registration and
            # here; a round that is no longer registered never announces
            # itself (no bell for a dead round, no zero-total "raised").
            with self.lock:
                still_live = self.registries[kind].get(round_id) is state and not bool(
                    state.get("revoked")
                )
            if still_live:
                self._note_pending(kind, raised=True)
            outcome = "decided"
            wait_cm = (
                use_human_input_wait(human_wait_run_id)
                if human_wait_run_id is not None
                else nullcontext()
            )
            with wait_cm:
                # task-31384: the approvals bridge fires its advisory
                # permission summary INSIDE the human-wait mark so the
                # summariser's own model call never counts against the
                # owning run's tool clock.
                if before_wait is not None and not hard_expired():
                    before_wait()
                while True:
                    if hard_expired():
                        notify_outcome("timeout")
                        outcome = "timeout"
                        break
                    remaining = (
                        max(0, hard_deadline - time.monotonic())
                        if hard_deadline is not None
                        else self.POLL_SECONDS
                    )
                    if event.wait(min(self.POLL_SECONDS, remaining)):
                        break
                    if self.read_controller__is_session_cancelled()(
                        session_id,
                        cancel_event=state.get("cancel_event"),
                        visit_event=state.get("visit_event"),
                    ):
                        notify_outcome("cancelled")
                        outcome = "cancelled"
                        break
                    if retained_decision:
                        self.read_controller_expire_pending_decisions()()
                    self.refresh_decision_clocks()
                    if clock is not None and clock.remaining <= 0:
                        notify_outcome("timeout")
                        outcome = "timeout"
                        break
            if retained_decision and state.get("terminal_reason") == "timeout":
                outcome = "timeout"
                notify_outcome("timeout")
            # A pre-set or just-arrived answer cannot evade the absolute bound.
            if hard_expired():
                outcome = "timeout"
                notify_outcome("timeout")
            elif (
                hard_deadline is not None
                and self.read_controller__is_session_cancelled()(
                    session_id,
                    cancel_event=state.get("cancel_event"),
                    visit_event=state.get("visit_event"),
                )
            ):
                outcome = "cancelled"
                notify_outcome("cancelled")
            if check_revoked:
                with self.lock:
                    if bool(state.get("revoked")):
                        outcome = "revoked"
            if on_outcome is not None:
                on_outcome(outcome)
            return outcome
        finally:
            with self.lock:
                self.registries[kind].pop(round_id, None)
            if retained_decision:
                self.read_controller__forget_hidden_decision()(round_id)
            self._note_pending(kind, raised=False)
            # task-31384: a kind may RETAIN its payload past teardown (the
            # approvals bridge keeps a definitive-after-start batch mounted
            # in its "finishing" phase). The hook runs OUTSIDE the lock and
            # returns True to keep the payload parked; it may take the lock
            # itself to mutate the retained payload.
            if on_teardown is None or not on_teardown():
                self.unpark_round_payload(kind, round_id)
            if session_id is not None:
                discard = self.read_controller_discard_pending_round()
                if discard is not None:
                    discard(session_id, round_id)
            try:
                if (
                    retained_decision
                    and self.read_controller_set_pending_decision() is not None
                ):
                    self.read_controller__marshal_pending_decision_projection()()
                else:
                    self.remount_head(
                        kind, owning_session_id if session_id is not None else None
                    )
            except Exception:  # noqa: BLE001 -- teardown must never raise
                logger.opt(exception=True).debug(
                    f"Failed to marshal {kind} remount during teardown"
                )

    def _announce_detached_approval(self, session_id: str, *, kind: str) -> None:
        """Raise the app-wide toast for a round with no visible Console view.

        WORKER THREAD. ``App.notify`` is documented thread-safe (it posts
        a message), so this needs no ``call_from_thread`` marshal -- and
        the toast renders on whatever screen the user is currently
        looking at, which is the whole point: the screen-owned seam
        (``ChatScreen._park_console_approval``) is unreachable here.

        Best-effort in both directions. An app double with no ``notify``
        (several controller-level tests) is silently skipped, and a
        raising/incompatible ``notify`` is logged rather than allowed to
        break the round -- a missing toast must never turn into a missing
        approval.

        Args:
            session_id: The round's owning session, used only to name the
                conversation in the notice.
        """
        app = self.read_controller_app()
        notify = getattr(app, "notify", None) if app is not None else None
        if not callable(notify):
            return
        title = ""
        try:
            for session in self.read_controller_store().sessions():
                if session.id == session_id:
                    title = str(getattr(session, "title", "") or "")
                    break
        except Exception:  # noqa: BLE001 -- a missing title never blocks the notice
            title = ""
        where = f" in {self.read_global_escape_markup()(title)}" if title else ""
        reason = (
            "needs your answer to a question"
            if kind == "question"
            else "needs approval to use a tool"
        )
        message = (
            f"Agent{where} {reason}. "
            "Open Console to review -- nothing runs until you answer."
        )
        try:
            notify(message, severity="warning")
        except TypeError:
            # An app double whose `notify` takes the message alone.
            try:
                notify(message)
            except Exception:  # noqa: BLE001
                self.read_global_logger().debug(
                    "Detached approval notice could not be delivered"
                )
        except Exception as exc:  # noqa: BLE001 -- surfacing is best-effort
            self.read_global_logger().debug(
                "Detached approval notice raised (exception_type={})",
                type(exc).__name__,
            )

    def _announce_hidden_decision(
        self,
        decision_type: Literal[
            "approval", "skill_install", "skill_script", "hook_review"
        ],
        session_id: str,
        decision_id: str,
    ) -> None:
        """Raise one content-free app notice for a hidden stable decision.

        WORKER THREAD. ``App.notify`` is documented thread-safe (it posts
        a message), so this needs no ``call_from_thread`` marshal -- and
        the toast renders on whatever screen the user is currently
        looking at, which is the whole point: the screen-owned seam
        (``ChatScreen._park_console_approval``) is unreachable here.

        Best-effort in both directions. An app double with no ``notify``
        (several controller-level tests) is silently skipped, and a
        raising/incompatible ``notify`` is logged rather than allowed to
        break the round -- a missing toast must never turn into a missing
        approval.

        Args:
            session_id: The round's owning session. It is routing context
                only and is never interpolated into the notice.
        """
        app = self.read_controller_app()
        notify = getattr(app, "notify", None) if app is not None else None
        if not callable(notify):
            return
        label = {
            "approval": "needs approval to use a tool",
            "skill_install": "needs confirmation for a skill install",
            "skill_script": "needs confirmation to run a skill script",
            "hook_review": "needs hook review before Send",
        }[decision_type]
        # Deliberately excludes title, ids, tool/skill names, URLs, paths,
        # arguments, and payload bodies.
        message = f"A Console session {label}. Return to Console to respond."

        def _emit() -> bool:
            """Post the fixed notice while the exact registry lock is held.

            Textual's ``App.notify`` posts a message; it does not synchronously
            re-enter controller decision state. Keeping this one fixed-string
            call inside the type registry lock gives teardown a total order:
            either emission wins and the later pop forgets its marker, or the
            pop wins and the liveness check below rejects the stale emission.

            Returns:
                True only when the app accepted the notification call.
            """
            try:
                notify(message, severity="warning")
            except TypeError:
                # An app double whose `notify` takes the message alone.
                try:
                    notify(message)
                except Exception:  # noqa: BLE001
                    self.read_global_logger().debug(
                        "Detached approval notice could not be delivered"
                    )
                    return False
            except Exception as exc:  # noqa: BLE001 -- surfacing is best-effort
                self.read_global_logger().debug(
                    "Detached approval notice raised (exception_type={})",
                    type(exc).__name__,
                )
                return False
            return True

        def _matches_live_round(
            state: dict[str, Any] | None,
        ) -> bool:
            return bool(
                state is not None
                and not state.get("settled")
                and state.get("session_id") == session_id
                and state.get("decision_type") == decision_type
                and state.get("decision_id") == decision_id
            )

        if decision_type == "approval":
            # MCP approvals use `_approval_state_lock` as both their registry
            # lock and the announcement-dedupe lock, so there is no second
            # lock to nest.
            with self.lock:
                if not _matches_live_round(
                    self.registries["approval"].get(decision_id)
                ):
                    return
                if (
                    decision_id
                    in self.read_controller__announced_pending_decision_ids()
                ):
                    return
                if _emit():
                    self.read_controller__announced_pending_decision_ids().add(
                        decision_id
                    )
            return

        registry_lock = self.lock if decision_type == "skill_install" else self.lock
        registry = (
            self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND]
            if decision_type == CONSOLE_PENDING_HOOK_REVIEW_KIND
            else self.registries["skill_install"]
            if decision_type == "skill_install"
            else self.registries["skill_script"]
        )
        # Dev's type registry and announcement state share the one
        # non-reentrant interrupt-host lock. Validate and dedupe together.
        with registry_lock:
            if not _matches_live_round(registry.get(decision_id)):
                return
            if decision_id in self.read_controller__announced_pending_decision_ids():
                return
            if not _emit():
                return
            self.read_controller__announced_pending_decision_ids().add(decision_id)

    def _approval_view_is_detached(self) -> bool:
        """True when Console is hidden or its approval view hooks are absent.

        TASK-31520 retains hooks during navigation. Attachment alone therefore
        cannot tell whether the user can see a card; modals also suspend it.
        """
        if self.view_visible is False:
            return True
        router = getattr(self.read_controller_set_pending_decision(), "__self__", None)
        available = getattr(router, "has_answerable_view", None)
        if callable(available):
            return not available()
        return (
            self.read_controller_set_pending_approval() is None
            and self.read_controller_set_pending_skill_install() is None
            and self.read_controller_set_pending_skill_script() is None
            and self.read_controller_park_pending_approval() is None
        )

    def _cancel_pending_decisions_for_session(self, session_id: str) -> None:
        """Fail closed only the rounds owned by a destructively closed session."""
        self.cancel_hook_reviews(session_id)
        events: list[self.read_global_threading().Event] = []
        self.read_controller_set_answerable_decision()(session_id, None)
        with self.lock:
            for state in self.registries["approval"].values():
                if state.get("session_id") != session_id:
                    continue
                state["revoked"] = True
                if not state.get("settled"):
                    state["settled"] = True
                    state["terminal_reason"] = "cancelled"
                    decisions = state.get("decisions")
                    if isinstance(decisions, dict):
                        for name in state.get("names", ()):
                            decisions[name] = "deny"
                event = state.get("event")
                if isinstance(event, self.read_global_threading().Event):
                    events.append(event)
        with self.lock:
            for state in self.registries["question"].values():
                if state.get("session_id") != session_id:
                    continue
                state["revoked"] = True
                if not state.get("settled"):
                    state["settled"] = True
                    state["terminal_reason"] = "cancelled"
                event = state.get("event")
                if isinstance(event, self.read_global_threading().Event):
                    events.append(event)
        with self.lock:
            for state in self.registries["skill_install"].values():
                if state.get("session_id") == session_id:
                    if not state.get("settled"):
                        state["settled"] = True
                        state["terminal_reason"] = "cancelled"
                        decision = state.get("decision")
                        if isinstance(decision, dict):
                            decision["allow"] = False
                    event = state.get("event")
                    if isinstance(event, self.read_global_threading().Event):
                        events.append(event)
        with self.lock:
            for state in self.registries["skill_script"].values():
                if state.get("session_id") != session_id:
                    continue
                state["revoked"] = True
                if not state.get("settled"):
                    state["settled"] = True
                    state["terminal_reason"] = "cancelled"
                    decision = state.get("decision")
                    if isinstance(decision, dict):
                        decision["allow"] = False
                        decision["remember"] = False
                event = state.get("event")
                if isinstance(event, self.read_global_threading().Event):
                    events.append(event)
        with self.read_controller__pending_chat_create_lock():
            for state in self.read_controller__pending_chat_create_rounds().values():
                if state.get("session_id") != session_id:
                    continue
                state["revoked"] = True
                decision = state.get("decision")
                if isinstance(decision, dict):
                    decision["allow"] = False
                    decision["remember"] = False
                event = state.get("event")
                if isinstance(event, self.read_global_threading().Event):
                    events.append(event)
        for event in events:
            event.set()

    def _console_tool_kill_switch_reader(self) -> Callable[[], bool] | None:
        """Return a fresh-per-call kill-switch reader, or ``None`` without a service.

        TASK-631. A callable rather than a bool so `build_tool_review_hook`
        observes a mid-run flip on the next batch; reading raises -> the
        hook fails CLOSED (refuses the turn), which is the only safe answer
        for a security control that cannot be read.

        Returns:
            A zero-arg callable returning the switch state, or ``None``
            when the app has no ``unified_mcp_service`` (nothing to honor).
        """
        service = getattr(self.read_controller_app(), "unified_mcp_service", None)
        if service is None:
            return None
        getter = getattr(service, "get_kill_switch", None)
        if not callable(getter):
            return None
        return lambda: bool(getter())

    def _deliver_permission_summary(
        self, round_id: str, payload: dict[str, Any], text: str
    ) -> None:
        """UI THREAD: store the summary, then patch the mounted card.

        Drops resolved/revoked rounds and unknown ids; writes the payload's
        ``summary`` slot (the source of truth for remounts) before the live
        patch. Never re-runs ``set_batch``.
        """
        with self.lock:
            state = self.registries["approval"].get(round_id)
            if state is None or state["event"].is_set():
                return
            state["summary"] = text
        payload["summary"] = text
        if self.read_controller_update_pending_approval_summary() is not None:
            try:
                self.read_controller_update_pending_approval_summary()(round_id, text)
            except Exception:  # noqa: BLE001 -- advisory only
                pass

    def _discard_approval_rows_for_closing_session(self, session_id: str) -> None:
        """Drop every approval payload owned by a closing session.

        This uses the same lock as the approval-to-finishing transition, so
        whichever operation wins first, no later transition can retain a row
        for a session that is being deleted.
        """
        removed = False
        with self.lock:
            for round_id, payload in list(self.payloads["approval"].items()):
                if payload.get("session_id") == session_id:
                    self.payloads["approval"].pop(round_id, None)
                    removed = True
        if removed and self.read_controller_store().active_session_id == session_id:
            self.read_controller__remount_head()(
                self.payloads["approval"],
                self.read_controller_set_pending_approval(),
                session_id,
            )

    def _enrich_chat_create_confirm_payload(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Fill the confirm card's fork facts, default title, and run id.

        Final-review fix wave (Finding 1): the card renders
        ``fork_source_title`` / ``fork_message_count`` (its "Copies N
        messages from '<title>'" line) and a non-empty header title, but no
        production payload producer ever set the fork keys -- every card
        read "Copies ? messages from ''" -- and an agent-omitted title left
        the header blank until the executor computed its default
        post-confirm. The controller is the only side holding
        store/persistence/db access at arm time, so it enriches here,
        BEFORE the round is armed.

        EVERYTHING is best-effort: any failure degrades -- the fork line's
        keys are omitted, a fork title falls back to the owning session's
        display title so the header still renders -- and can never block,
        delay, or deny the round.
        """
        from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

        enriched = dict(payload)
        tool = str(enriched.get("tool") or "")
        try:
            title = str(enriched.get("title") or "").strip()
            if tool == "fork_chat":
                session = next(
                    (
                        s
                        for s in self.read_controller_store().sessions()
                        if s.id == str(enriched.get("session_id") or "")
                    ),
                    None,
                )
                conversation_id = (
                    getattr(session, "persisted_conversation_id", None)
                    if session is not None
                    else None
                )
                if session is not None and not session.ephemeral and conversation_id:
                    # The tree-read leg carries its OWN guard so a failure
                    # here degrades to the session-title fallback below
                    # instead of skipping it.
                    try:
                        database = (
                            getattr(
                                self.read_controller_store().persistence, "db", None
                            )
                            if self.read_controller_store().persistence
                            else None
                        )
                        tree = ChatConversationService(database).get_conversation_tree(
                            str(conversation_id),
                            root_limit=10_000,
                            depth_cap=10_000,
                        )
                        conversation_row = dict(tree.get("conversation") or {})
                        source_title = str(conversation_row.get("title") or "")
                        # Count definition -- ACTIVE-PATH LENGTH, not total
                        # tree nodes: the executor's
                        # copy_conversation_active_path copies exactly the
                        # leaf-to-root ancestry, so the card must not count
                        # off-path siblings the fork drops.
                        nodes: dict[str, Mapping[str, Any]] = {}

                        def _walk(
                            node: Mapping[str, Any],
                        ) -> None:
                            nodes[str(node["id"])] = node
                            for child in node.get("children") or []:
                                _walk(child)

                        for root in tree.get("root_threads") or []:
                            _walk(root)
                        leaf = (
                            database.get_conversation_active_leaf(str(conversation_id))
                            if database is not None
                            else None
                        )
                        if leaf is None or leaf not in nodes:
                            # Mirrors copy_conversation_active_path's own
                            # fallback: a missing/dangling pointer means
                            # the most recent message by timestamp.
                            leaf = (
                                max(
                                    nodes,
                                    key=lambda i: str(nodes[i].get("timestamp") or ""),
                                )
                                if nodes
                                else None
                            )
                        count = 0
                        cursor = leaf
                        while cursor is not None and cursor in nodes:
                            count += 1
                            cursor = nodes[cursor].get("parent_message_id")
                        enriched["fork_source_title"] = source_title
                        enriched["fork_message_count"] = count
                        if not title:
                            # Same default formula the executor applies
                            # post-confirm, so the card header matches the
                            # title that would actually be created.
                            title = f"Fork of {source_title or 'chat'}"[:120]
                    except Exception:  # noqa: BLE001 — omit the fork line only
                        self.read_global_logger().opt(exception=True).debug(
                            "chat-create confirm fork enrichment degraded; "
                            "card omits the fork line"
                        )
                if not title and session is not None:
                    # Degraded fork path (tree read failed, ephemeral
                    # source, or unpersisted source): default the title
                    # from the SESSION's display title -- the header must
                    # never render empty.
                    title = f"Fork of {session.title or 'chat'}"[:120]
            elif not title:
                title = "New Chat"
            if title:
                enriched["title"] = title
            # Run attribution: the bridge payload already carries run_id;
            # normalize it so the card can render "requested by agent run".
            if enriched.get("run_id"):
                enriched["run_id"] = str(enriched["run_id"])
        except Exception:  # noqa: BLE001 — enrichment never blocks the round
            self.read_global_logger().opt(exception=True).debug(
                "chat-create confirm payload enrichment degraded"
            )
        return enriched

    def _expire_answerable_decision_if_due(
        self, session_id: str, decision_id: str, *, now: float
    ) -> bool:
        """Settle one still-mounted head only when its active allowance is due."""
        expired = False

        def _expire(
            state: dict[str, Any],
        ) -> self.read_global_threading().Event | None:
            nonlocal expired
            if (
                self.read_controller__answerable_decision_by_session().get(session_id)
                != decision_id
            ):
                return None
            remaining = state.get("remaining_active_seconds")
            active_since = state.get("active_since")
            if remaining is None or active_since is None:
                return None
            if max(0.0, now - float(active_since)) < float(remaining):
                return None
            self.read_controller__answerable_decision_by_session().pop(session_id, None)
            expired = True
            return self._settle_pending_decision_timeout_locked(
                state, read_global_threading=self.read_global_threading
            )

        event = self._mutate_exact_pending_decision(decision_id, _expire)
        if isinstance(event, self.read_global_threading().Event):
            event.set()
        return expired

    def _forget_hidden_decision(self, decision_id: str) -> None:
        """Release one terminal decision's app-wide announcement marker."""
        with self.lock:
            self.read_controller__announced_pending_decision_ids().discard(decision_id)

    def _head_round_payload(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> dict[str, Any] | None:
        """The payload whose card ``session_id`` should currently show (remaining-time snapshot)."""
        from tldw_chatbook.Chat.console_interrupt_rounds import head_round_payload

        return head_round_payload(self.lock, store, session_id)

    @staticmethod
    def _head_round_payload_locked(
        store: dict[str, dict[str, Any]], session_id: str | None
    ) -> dict[str, Any] | None:
        """The session's oldest-armed payload. Caller holds the lock."""
        from tldw_chatbook.Chat.console_interrupt_rounds import head_payload_locked

        return head_payload_locked(store, session_id)

    def _interrupt_bell_enabled(self) -> bool:
        """Resolve ``[console] interrupt_bell``: environment, then config, then on.

        Returns:
            False only when ``TLDW_CONSOLE_INTERRUPT_BELL`` (a non-empty
            value) or the config key coerces to False.
        """
        env_value = self.read_global__bool_or_none()(
            self.read_global_os()
            .environ.get(self.read_global_INTERRUPT_BELL_ENV_VAR(), "")
            .strip()
        )
        if env_value is not None:
            return env_value
        config_value = self.read_global__bool_or_none()(
            self.read_global_get_cli_setting()("console", "interrupt_bell", None)
        )
        return True if config_value is None else config_value

    def _marshal_pending_approval(
        self, payload: dict[str, Any] | None, *, fire_summary: bool
    ) -> None:
        """Push ``payload`` (or clear it) onto the UI thread, if wired.

        Args:
            payload: The approval payload dict, or ``None`` to clear.
            fire_summary: Whether to run the ADR-090 advisory-summary
                trigger check after delivery. The arm-time head-mount site
                passes ``False`` and fires the check itself inside the
                ``use_human_input_wait`` mark, so the hook's config read
                cannot sit between payload delivery and the wait mark.
        """
        if (
            self.read_controller_app() is not None
            and self.read_controller_set_pending_approval() is not None
        ):
            self.read_controller_app().call_from_thread(
                self.read_controller_set_pending_approval(), payload
            )
        if fire_summary and isinstance(payload, dict):
            self.read_controller__maybe_fire_permission_summary()(payload)

    def _marshal_pending_chat_create(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: project a current chat-create decision on the UI.

        Recheck scoped ownership after dispatch; legacy unparked rounds keep
        their unconditional initial projection. A clear derives the current head.

        Args:
            payload: Proposed confirmation, or None to rederive the active head.
        """
        if (
            self.read_controller_app() is None
            or self.read_controller_set_pending_chat_create() is None
        ):
            return

        def _apply() -> None:
            """UI THREAD: qualify the current owner before painting its decision."""
            setter = self.read_controller_set_pending_chat_create()
            if setter is None:
                return
            active_session_id = self.read_controller_store().active_session_id or ""
            if payload is None:
                setter(
                    self.read_controller__head_round_payload()(
                        self.read_controller__parked_chat_create_payloads(),
                        active_session_id,
                    )
                )
                return
            owning_session_id = str(payload.get("session_id") or "")
            if owning_session_id in self.read_controller__session_close_generations():
                return
            with self.read_controller__pending_chat_create_lock():
                state = self.read_controller__pending_chat_create_rounds().get(
                    payload.get("request_id")
                )
                if state is None or state.get("revoked") or state["event"].is_set():
                    return
                if state.get("session_scoped", True) and (
                    owning_session_id != active_session_id
                ):
                    return
            # The UI owns Close/navigation; external sinks run outside locks.
            setter(payload)

        self.read_controller_app().call_from_thread(_apply)

    def _marshal_pending_decision_projection(self) -> None:
        """Worker-thread marshal of the active-session derived head."""
        if self.read_controller_app() is None:
            return
        self.read_controller_app().call_from_thread(
            self.read_controller_project_pending_decision_for_active_session()
        )

    def _marshal_pending_question(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a question payload to the UI thread.

        Args:
            payload: The card payload to show, or None to hide the card.
        """
        if (
            self.read_controller_app() is not None
            and self.read_controller_set_pending_question() is not None
        ):
            self.read_controller_app().call_from_thread(
                self.read_controller_set_pending_question(), payload
            )

    def _marshal_pending_skill_install(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-install confirm payload to the UI thread.

        No-op when no UI bridge is wired (``self.app`` or
        ``set_pending_skill_install`` is None).

        Args:
            payload: The pending confirm's ``{"url", "timeout_seconds"}``
                dict to show, or None to clear/hide the card.
        """
        if (
            self.read_controller_app() is not None
            and self.read_controller_set_pending_skill_install() is not None
        ):
            self.read_controller_app().call_from_thread(
                self.read_controller_set_pending_skill_install(), payload
            )

    def _marshal_pending_skill_script(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-script confirm payload to the UI thread.

        Args:
            payload: The pending confirm dict to show, or None to hide it.
        """
        if (
            self.read_controller_app() is not None
            and self.read_controller_set_pending_skill_script() is not None
        ):
            self.read_controller_app().call_from_thread(
                self.read_controller_set_pending_skill_script(), payload
            )

    def _marshal_pending_worktree_merge(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a worktree-merge confirm payload to the UI thread.

        Args:
            payload: The pending confirm dict to show, or None to hide it.
        """
        if (
            self.read_controller_app() is not None
            and self.read_controller_set_pending_worktree_merge() is not None
        ):
            self.read_controller_app().call_from_thread(
                self.read_controller_set_pending_worktree_merge(), payload
            )

    def _marshal_task_panel(
        self, session_id: str, tasks: list[dict[str, object]]
    ) -> None:
        """WORKER THREAD: hand a session's task snapshot to the pinned panel.

        PRD Feature B (AC-B4): fires on every ``todo_*`` change alongside
        the transcript marker. The screen-side setter ignores snapshots
        for sessions that are not the viewed one.

        Args:
            session_id: The session whose todo store changed.
            tasks: Its full task list after the change.
        """
        if (
            self.read_controller_app() is not None
            and self.read_controller_set_task_panel() is not None
        ):
            self.read_controller_app().call_from_thread(
                self.read_controller_set_task_panel(), session_id, tasks
            )

    def _maybe_fire_permission_summary(self, payload: dict[str, Any]) -> None:
        """Fire the external summarizer once per round, if configured.

        ADR-090 trigger: ``fallback`` only when some pending row lacks a
        rationale, ``always`` for every round with rows. One call per
        ``round_id`` -- no-call outcomes also consume the once-flag, so
        exactly one trigger check runs per round no matter how many times
        it mounts. Called from EVERY path that marshals a stored approval
        payload to the UI: ``_marshal_pending_approval`` (arm-time head
        mount), the session-activation mounts (``new_session``/
        ``switch_session``/``close_session`` neighbor activation,
        ``remount_pending_approval_for_active_session`` headless attach)
        and ``_remount_head`` (sibling promotion on resolve/revoke) -- so
        a round that armed while parked fires when its card actually
        mounts. Never raises.
        """
        round_id = str(payload.get("round_id") or "")
        rows = payload.get("calls") or []
        if not round_id or not rows:
            return
        with self.lock:
            state = self.registries["approval"].get(round_id)
            if state is None or state.get("summary_fired"):
                return
        # Release the preliminary approval check before reading config.
        # Nested lock order stays config first, then approval (see
        # `run_if_runtime_config_generation_current` in config.py).
        try:
            from tldw_chatbook.Chat.permission_summary_service import (
                resolve_permission_summary,
            )

            resolution = resolve_permission_summary(
                self.read_global_get_runtime_config_snapshot()().values
            )
        except Exception:  # noqa: BLE001 -- advisory only
            resolution = None
        with self.lock:
            state = self.registries["approval"].get(round_id)
            if state is None or state.get("summary_fired"):
                return
            if resolution is None or not resolution.active:
                state["summary_fired"] = True
                return
            needs = resolution.mode == "always" or any(
                not str(row.get("rationale") or "") for row in rows
            )
            state["summary_fired"] = True
            if not needs:
                return
        # TASK-32801.4: the tail is read from the store, and the store is
        # owned by this thread (``store mutation always runs on the thread
        # that owns the store``). Reading it here rather than in the worker
        # keeps ``messages_for_session`` -- which folds buffered chunks and
        # can persist a pending row -- off the raw thread entirely. The
        # worker is left with the network call alone.
        try:
            from tldw_chatbook.Chat.permission_summary_service import (
                build_messages_tail,
            )

            tail = build_messages_tail(
                self.read_controller__summary_tail_messages()(payload),
                resolution.tail_max_chars,
            )
        except Exception:  # noqa: BLE001 -- advisory only
            tail = []
        try:
            self.read_global_threading().Thread(
                target=self.read_controller__permission_summary_worker(),
                args=(round_id, payload, resolution, tail),
                daemon=True,
                name=f"permission-summary-{round_id}",
            ).start()
        except Exception:  # noqa: BLE001 -- advisory only
            # A failed spawn must not destroy the approval round; the summary
            # lane is advisory-only, so swallow it (content-free log below).
            self.read_global_logger().debug("permission summary thread spawn failed")

    def _mutate_exact_pending_decision(
        self, decision_id: str, mutate: Callable[[dict[str, Any]], Any]
    ) -> Any:
        """Mutate one live round under its registry then projection lock.

        All five kinds share the host's non-reentrant lock. Mutations must
        never acquire a second aliased lock or project UI while holding it.
        """
        with self.lock:
            for registry in self.registries.values():
                state = registry.get(decision_id)
                if state is not None:
                    return mutate(state)
        return None

    def _notify_run_hook_approval(
        self, kind: str, payload: dict[str, Any], state: dict[str, Any]
    ) -> None:
        """Publish one successfully admitted permission round, including headless runs."""
        engine = self.read_controller__run_hooks_engine()()
        if engine is None:
            return
        from tldw_chatbook.Agents.run_hooks import summarize_hook_arguments

        session_id = payload.get("session_id") or state.get("session_id")
        run_id = (
            state.get("run_id")
            if kind == "worktree_merge"
            else payload.get("run_id") or state.get("run_id")
        )
        if kind == "approval":
            calls = [
                {
                    "name": row.get("llm_name") or row.get("tool_name") or "",
                    "args_summary": summarize_hook_arguments(
                        row.get("arguments") or {}
                    ),
                }
                for row in payload.get("calls", ())
            ]
        elif kind == "worktree_merge":
            action = payload.get("action") or payload.get("mode")
            arguments = {
                key: payload[key]
                for key in (
                    "handle_id",
                    "run_id",
                    "action",
                    "mode",
                    "branch",
                    "worktree",
                    "source",
                    "destination",
                )
                if key in payload
            }
            calls = [
                {
                    "name": (
                        "discard_agent_worktree"
                        if action == "discard"
                        else "merge_agent_worktree"
                    ),
                    "args_summary": summarize_hook_arguments(arguments),
                }
            ]
        else:
            arguments = (
                {"url": payload.get("url", "")}
                if kind == "skill_install"
                else {
                    key: payload[key]
                    for key in ("skill_name", "script_path", "mechanism", "args")
                    if key in payload
                }
            )
            calls = [
                {
                    "name": "install_skill"
                    if kind == "skill_install"
                    else "run_skill_script",
                    "args_summary": summarize_hook_arguments(arguments),
                }
            ]
        engine.notify(
            "ApprovalRequested",
            session_id=session_id,
            run_id=run_id,
            data={
                "calls": calls,
                "session_active": bool(
                    session_id == self.read_controller_store().active_session_id
                    and self.view_visible is not False
                    and (
                        self.read_controller_set_pending_decision() is not None
                        or self._setter(kind) is not None
                    )
                ),
            },
        )

    def _park_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str, payload: dict[str, Any]
    ) -> bool:
        """Retain ``payload``; return whether it is now its session's head."""
        from tldw_chatbook.Chat.console_interrupt_rounds import park_round_payload

        return park_round_payload(self.lock, store, round_id, payload)

    def _pause_answerable_decision(
        self,
        session_id: str,
        decision_id: str,
        *,
        now: float,
        claim_revision: int | None,
    ) -> bool:
        """Pause one exact head, terminally timing it out at zero."""
        expired = False

        def _pause(
            state: dict[str, Any],
        ) -> self.read_global_threading().Event | None:
            nonlocal expired
            if (
                claim_revision is not None
                and claim_revision != self.decision_view_revision
            ):
                return None
            if (
                self.read_controller__answerable_decision_by_session().get(session_id)
                != decision_id
            ):
                return None
            self._pause_pending_decision_state_locked(state, now=now)
            self.read_controller__answerable_decision_by_session().pop(session_id, None)
            remaining = state.get("remaining_active_seconds")
            if remaining is None or float(remaining) > 0:
                return None
            expired = True
            return self._settle_pending_decision_timeout_locked(
                state, read_global_threading=self.read_global_threading
            )

        event = self._mutate_exact_pending_decision(decision_id, _pause)
        if isinstance(event, self.read_global_threading().Event):
            event.set()
        if event is None:
            with self.lock:
                if (
                    claim_revision is not None
                    and claim_revision != self.decision_view_revision
                ):
                    return expired
                if (
                    self.read_controller__answerable_decision_by_session().get(
                        session_id
                    )
                    == decision_id
                ):
                    self.read_controller__answerable_decision_by_session().pop(
                        session_id, None
                    )
        return expired

    def _pause_pending_decision_state_locked(
        self, state: dict[str, Any], *, now: float
    ) -> None:
        active_since = state.get("active_since")
        if active_since is None:
            return
        remaining = state.get("remaining_active_seconds")
        if remaining is not None:
            state["remaining_active_seconds"] = max(
                0.0, float(remaining) - max(0.0, now - float(active_since))
            )
        state["active_since"] = None

    def _pending_decision_payloads_locked(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        payloads = [
            payload
            for store in (
                self.payloads["approval"],
                self.payloads["skill_install"],
                self.payloads["skill_script"],
                self.payloads[CONSOLE_PENDING_HOOK_REVIEW_KIND],
            )
            for payload in store.values()
            if payload.get("session_id") == session_id and payload.get("_decision_id")
        ]
        return sorted(payloads, key=lambda item: int(item["_decision_order"]))

    def _pending_round_states_snapshot(self) -> dict[str, dict[str, Any]]:
        """Snapshot existing round records without nesting registry locks.

        Lock order is type registry lock, release, then
        ``_approval_state_lock`` in callers. No code acquires a type lock while
        holding the shared projection lock.
        """
        with self.lock:
            states = dict(self.registries["approval"])
        with self.lock:
            states.update(self.registries["skill_install"])
        with self.lock:
            states.update(self.registries["skill_script"])
        with self.lock:
            states.update(self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND])
        return states

    def _permission_summary_worker(
        self, round_id: str, payload: dict[str, Any], resolution: object, tail: list
    ) -> None:
        """Worker THREAD: run the advisory call, deliver on the UI thread.

        The approval wait loop is never blocked and the round's deadline is
        unaffected; a slow call that outlives the round is dropped on
        delivery. Content-free failures only (ADR-090).

        ``tail`` arrives already built: it is store-derived, and the store
        belongs to the thread that spawned this one (TASK-32801.4). Nothing
        here may touch ``self.store``.
        """
        from tldw_chatbook.Chat.permission_summary_service import (
            pending_calls_info_from_payload,
            summarize_pending_round,
        )

        try:
            info = pending_calls_info_from_payload(payload.get("calls") or [])
            text = summarize_pending_round(resolution, tail, info)
        except Exception:  # noqa: BLE001 -- advisory only
            text = None
        if not text or self.read_controller_app() is None:
            return
        self.read_controller_app().call_from_thread(
            self.read_controller__deliver_permission_summary(), round_id, payload, text
        )

    def _publish_console_attention_change(self) -> None:
        """Best-effort notification that the runtime should re-derive attention."""
        callback = self.read_controller_on_console_attention_changed()
        if not callable(callback):
            return
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 -- decision ownership is authoritative
            self.read_global_logger().debug(
                "Console attention refresh failed (exception_type={})",
                type(exc).__name__,
            )

    def _publish_pending_decision(
        self,
        *,
        round_state: dict[str, Any],
        payload: dict[str, Any],
        decision_type: Literal[
            "approval", "skill_install", "skill_script", "hook_review"
        ],
        decision_id: str,
        timeout_seconds: float,
        retained_store: dict[str, dict[str, Any]] | None,
    ) -> bool:
        """Atomically admit, order, retain, and derive one decision head.

        Admission order is the order in which fully built decision payloads
        acquire ``_approval_state_lock`` here. The order stamp and retained
        payload become visible in the same transaction, so another decision
        can observe neither half of an admission. The five-kind host calls
        this after releasing its non-reentrant registration lock.
        """
        with self.lock:
            if round_state.get("revoked") or round_state.get("settled"):
                return False
            self.write_controller__pending_decision_order(
                self.read_controller__pending_decision_order() + 1
            )
            order = self.read_controller__pending_decision_order()
            remaining = max(0.0, timeout_seconds) if timeout_seconds > 0 else None
            round_state.update(
                {
                    "decision_type": decision_type,
                    "decision_id": decision_id,
                    "decision_order": order,
                    "remaining_active_seconds": remaining,
                    "active_since": None,
                    "settled": False,
                    "terminal_reason": None,
                }
            )
            payload.update(
                {
                    "_decision_type": decision_type,
                    "_decision_id": decision_id,
                    "_decision_order": order,
                }
            )
            if retained_store is None:
                return True
            retained_store[decision_id] = payload
            session_id = str(payload.get("session_id") or "")
            payloads = self._pending_decision_payloads_locked(session_id)
            return bool(payloads and payloads[0] is payload)

    def _record_cancelled_approval_decisions(
        self, keys: list[str], call_by_key: dict[str, "MCPPendingCall"]
    ) -> None:
        """Best-effort audit log for calls denied by a stop/unmount mid-approval.

        Finding I3: see the cancellation branch's own comment in
        ``request_mcp_approvals`` for why this direct call is necessary --
        `MCPToolProvider._record_decision_safe` (the normal recording
        path) is never reached for these calls, since `run_agent_loop`
        cancels the whole turn before dispatching any of them. Reached via
        `self.app.unified_mcp_service` (the same object
        `_compose_mcp_provider` built this run's `MCPToolProvider` from --
        see that method), never raises: a missing app/service, or the
        service lacking `record_tool_decision`, is a silent no-op, and any
        exception the real call raises is logged and swallowed, mirroring
        `MCPToolProvider._record_decision_safe`'s own never-raise
        contract.
        """
        service = getattr(self.read_controller_app(), "unified_mcp_service", None)
        if service is None:
            return
        record = getattr(service, "record_tool_decision", None)
        if not callable(record):
            return
        for key in keys:
            call = call_by_key.get(key)
            if call is None:
                continue
            try:
                record(
                    call.server_key,
                    call.tool_name,
                    # task-32280 fix round: the turn was stopped WHILE the
                    # card was up -- the user never answered it. Recording
                    # this as the bare "denied" made Audit report an
                    # explicit "Denied by you" for a question nobody got to
                    # answer.
                    decision=self.read_global_UNRESOLVED_DENIED_DECISION(),
                    initiator="agent",
                    error="run stopped while approval pending",
                )
            except Exception:  # noqa: BLE001 -- best-effort audit trail only
                self.read_global_logger().opt(exception=True).debug(
                    "Failed to record cancelled MCP approval decision"
                )

    def _refresh_answerable_decision(self, session_id: str) -> str | None:
        """Reconcile rendered Console/Buddy claims against one typed FIFO clock."""
        with self.lock:
            claim_revision = self.decision_view_revision
        rendered = self.rendered_decision_ids(session_id)
        projection = self.read_controller_pending_decision_projection()(session_id)
        with self.lock:
            console_id = (
                self.read_controller__console_answerable_decision_by_session().get(
                    session_id
                )
            )
        if (
            session_id == self.read_controller_store().active_session_id
            and console_id is not None
        ):
            rendered.add(console_id)
        decision_id = (
            projection.decision_id
            if projection is not None
            and projection.payload.get("phase") != "finishing"
            and projection.decision_id in rendered
            else None
        )
        now = self.read_controller_decision_monotonic_clock()()
        with self.lock:
            current_id = self.read_controller__answerable_decision_by_session().get(
                session_id
            )
        if current_id == decision_id:
            return current_id
        if current_id is not None:
            self._pause_answerable_decision(
                session_id, current_id, now=now, claim_revision=claim_revision
            )
        if decision_id is None:
            return None

        started = False

        def _start(state: dict[str, Any]) -> None:
            nonlocal started
            if claim_revision != self.decision_view_revision:
                return
            if state.get("settled"):
                return
            if state.get("session_id") != session_id:
                return
            payloads = self._pending_decision_payloads_locked(session_id)
            if (
                not payloads
                or payloads[0].get("_decision_id") != decision_id
                or payloads[0].get("phase") == "finishing"
            ):
                return
            remaining = state.get("remaining_active_seconds")
            if remaining is not None and float(remaining) <= 0:
                return
            state["active_since"] = now
            self.read_controller__answerable_decision_by_session()[session_id] = (
                decision_id
            )
            started = True

        self._mutate_exact_pending_decision(decision_id, _start)
        return decision_id if started else None

    def _remount_head(
        self,
        store: dict[str, dict[str, Any]],
        setter: Callable[[dict[str, Any] | None], None] | None,
        session_id: str | None,
    ) -> None:
        """Push ``store``'s head for ``session_id`` through ``setter`` on the UI thread.

        The pre-host body, kept verbatim: every kind's activation re-derive
        and the approval attach path call it with their own store and
        setter, and it fires the ADR-090 permission summary for any dict
        payload -- ``_maybe_fire_permission_summary`` itself ignores round
        ids it does not know, so a skill payload passes through untouched.

        Args:
            store: A per-kind parked-payload dict.
            setter: The kind's UI-thread setter, or None to do nothing.
            session_id: The session whose head to push; None means the
                store's active session.
        """
        if self.read_controller_app() is None:
            return

        if self.read_controller_set_pending_decision() is not None:
            self.read_controller_app().call_from_thread(
                self.read_controller_project_pending_decision_for_active_session()
            )
            return
        if setter is None:
            return

        def _apply() -> None:
            target = session_id
            if target is None:
                target = self.read_controller_store().active_session_id or ""
            elif target != (self.read_controller_store().active_session_id or ""):
                return
            payload = self.read_controller__head_round_payload()(store, target)
            setter(payload)
            if isinstance(payload, dict):
                self.read_controller__maybe_fire_permission_summary()(payload)

        self.read_controller_app().call_from_thread(_apply)

    def _remount_parked_chat_create(self, session_id: str) -> None:
        """Re-derive the mounted chat-create confirm card for ``session_id``.

        Called from `switch_session`/`new_session`/`close_session` exactly
        like the sibling confirm cards' own re-derive -- mounts
        ``session_id``'s retained payload (if any) and clears whatever the
        departing session had shown, all in one call. A no-op when no UI
        bridge is wired.

        Already runs on the UI thread, so it calls `_head_round_payload`
        directly rather than `_remount_head`.

        Args:
            session_id: The session now being activated/viewed.
        """
        if self.read_controller_set_pending_chat_create() is None:
            return
        self.read_controller_set_pending_chat_create()(
            self.read_controller__head_round_payload()(
                self.read_controller__parked_chat_create_payloads(), session_id
            )
        )

    def _remount_parked_question(self, session_id: str) -> None:
        """UI THREAD: re-derive the question card for the session now viewed.

        Called from ``switch_session``/``new_session``/``close_session``
        beside the other card re-derives (PRD A10).

        Args:
            session_id: The session being activated/viewed.
        """
        if self.read_controller_set_pending_question() is None:
            return
        self.read_controller_set_pending_question()(
            self.read_controller__head_round_payload()(
                self.payloads["question"], session_id
            )
        )

    def _remount_parked_skill_install(self, session_id: str) -> None:
        """Re-derive the mounted skill-install confirm card for ``session_id``.

        TASK-910: called from `switch_session`/`new_session`/`close_session`
        exactly like the MCP approval card's own re-derive -- mounts
        ``session_id``'s retained payload (if any) and clears whatever the
        departing session had shown, all in one call. A no-op when no UI
        bridge is wired.

        Args:
            session_id: The session now being activated/viewed.
        """
        if self.read_controller_set_pending_decision() is not None:
            self.read_controller_project_pending_decision_for_active_session()()
            return
        if self.read_controller_set_pending_skill_install() is None:
            return
        self.read_controller_set_pending_skill_install()(
            self.read_controller__head_round_payload()(
                self.payloads["skill_install"], session_id
            )
        )

    def _remount_parked_skill_script(self, session_id: str) -> None:
        """Re-derive the mounted skill-script confirm card for ``session_id``.

        TASK-910: called from `switch_session`/`new_session`/`close_session`
        exactly like the MCP approval card's own re-derive -- mounts
        ``session_id``'s retained payload (if any) and clears whatever the
        departing session had shown, all in one call. A no-op when no UI
        bridge is wired.

        PR0: re-keyed by round, so this now re-derives the session's FIFO
        head instead of a single per-session slot. It already runs on the
        UI thread, so it calls `_head_round_payload` directly rather than
        `_remount_head`.

        Args:
            session_id: The session now being activated/viewed.
        """
        if self.read_controller_set_pending_decision() is not None:
            self.read_controller_project_pending_decision_for_active_session()()
            return
        if self.read_controller_set_pending_skill_script() is None:
            return
        self.read_controller_set_pending_skill_script()(
            self.read_controller__head_round_payload()(
                self.payloads["skill_script"], session_id
            )
        )

    def _remount_parked_worktree_merge(self, session_id: str) -> None:
        """Re-derive the mounted worktree-merge confirm card for
        ``session_id``. Mirrors ``_remount_parked_skill_script``.

        Args:
            session_id: The session now being activated/viewed.
        """
        if self.read_controller_set_pending_worktree_merge() is None:
            return
        self.read_controller_set_pending_worktree_merge()(
            self.read_controller__head_round_payload()(
                self.payloads["worktree_merge"], session_id
            )
        )

    def _remount_session_kinds(self, session_id: str) -> None:
        """Re-derive every non-approval kind's head card for ``session_id``.

        The one call the three session-activation sites (new, switch,
        close) share; the kinds come from the host module's
        ``SESSION_REMOUNT_KINDS`` and approvals stay on the sites' own block.

        Args:
            session_id: The session being activated.
        """
        from tldw_chatbook.Chat.console_interrupt_rounds import SESSION_REMOUNT_KINDS

        self.refresh_decision_clocks()
        kinds = (
            ("worktree_merge", "question")
            if self.read_controller_set_pending_decision() is not None
            else SESSION_REMOUNT_KINDS
        )
        self.remount_for_session(session_id, kinds=kinds)

    def _remount_task_panel(self, session_id: str | None) -> None:
        """UI THREAD: re-derive the pinned task panel for ``session_id``.

        Called from `switch_session`/`new_session`/`close_session` next to
        the card re-derives, and from `ConsoleRuntime.attach_view` when a
        new screen claims the surviving runtime, so the panel always shows
        the VIEWED session's tasks (AC-B5) and hides when that session has
        none -- or when there is no session at all.

        Args:
            session_id: The session now being activated/viewed, or None
                when none is (the last session was just closed).
        """
        if self.read_controller_set_task_panel() is None:
            return
        session = next(
            (s for s in self.read_controller_store().sessions() if s.id == session_id),
            None,
        )
        tasks = session.todo_store.list_after(None) if session is not None else []
        self.read_controller_set_task_panel()(session_id, tasks)

    def _reproject_pending_decision_for_session(self, session_id: str) -> None:
        """Re-derive one session through the unified or legacy card seams."""
        # ADR-150: the chat-create card rides its OWN standalone registry
        # (pre-host design), so it must re-derive BEFORE the unified
        # decision projection's early return -- otherwise a resolved card
        # lingers on every session switch (live-UAT defect).
        self.read_controller__remount_parked_chat_create()(session_id)
        if self.read_controller_set_pending_decision() is not None:
            self.read_controller_project_pending_decision_for_active_session()()
            return
        if (setter := self.read_controller_set_pending_approval()) is not None:
            head = self.read_controller__head_round_payload()
            payload = head(self.payloads["approval"], session_id)
            setter(payload)
            if isinstance(payload, dict):
                self.read_controller__maybe_fire_permission_summary()(payload)
        self.read_controller__remount_parked_skill_install()(session_id)
        self.read_controller__remount_parked_skill_script()(session_id)

    def _resolve_ask_user_timeout_seconds(self) -> float:
        """PRD A7: the question deadline -- seam, else env, else config, else 0.

        The injected seam exists for tests; production precedence is
        ``TLDW_CONSOLE_ASK_USER_TIMEOUT_SECONDS`` -> ``[console]
        ask_user_timeout_seconds`` -> ``0``. An empty or unparseable env
        value is ignored.

        Returns:
            Seconds before an unanswered question auto-continues; ``0.0``
            (the default) means no deadline. Never negative.
        """
        if self.read_controller_ask_user_timeout_seconds() is not None:
            try:
                return max(
                    0.0, float(self.read_controller_ask_user_timeout_seconds()())
                )
            except Exception:  # noqa: BLE001 -- fail open to the documented default
                pass
        raw_env = (
            self.read_global_os()
            .environ.get(self.read_global_ASK_USER_TIMEOUT_ENV_VAR(), "")
            .strip()
        )
        if raw_env:
            try:
                return max(0.0, float(raw_env))
            except ValueError:
                pass  # an unparseable override is ignored, not fatal
        try:
            return max(
                0.0,
                float(
                    self.read_global_get_cli_setting()(
                        "console",
                        "ask_user_timeout_seconds",
                        self.read_global__DEFAULT_ASK_USER_TIMEOUT_SECONDS(),
                    )
                ),
            )
        except (TypeError, ValueError):
            return self.read_global__DEFAULT_ASK_USER_TIMEOUT_SECONDS()

    def _resolve_mcp_approval_timeout_seconds(self) -> float:
        if self.read_controller_mcp_approval_timeout_seconds() is not None:
            try:
                return float(self.read_controller_mcp_approval_timeout_seconds()())
            except Exception:  # noqa: BLE001 -- fail open to the documented default
                pass
        try:
            return float(
                self.read_global_get_cli_setting()(
                    "mcp",
                    "approval_timeout_seconds",
                    self.read_global__DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS(),
                )
            )
        except (TypeError, ValueError):
            return self.read_global__DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS()

    def _revoke_skill_script_rounds(self, run_id: str) -> list[tuple[str, str | None]]:
        """Fail this run's ``run_skill_script`` confirms closed.

        The host fences and sweeps under its shared lock, then removes only
        each swept round's retained payload. Same-session siblings retain
        their own round-keyed payloads through revocation and teardown.

        Args:
            run_id: The cancelled/abandoned run.

        Returns:
            ``(request_id, session_id)`` for each revoked confirm.
        """
        return self.revoke_for_run(
            run_id,
            {"skill_script": self.read_global__REVOCATION_STAMPS()["skill_script"]},
        )["skill_script"]

    def _revoke_tool_approval_rounds(self, run_id: str) -> list[tuple[str, str | None]]:
        """Fail this run's tool-approval rounds closed. Registry work only.

        Args:
            run_id: The cancelled/abandoned run.

        Returns:
            ``(round_id, session_id)`` for each revoked round, for the
            caller's badge/card teardown (which must run outside the lock
            held here).
        """
        return self.revoke_for_run(
            run_id, {"approval": self.read_global__REVOCATION_STAMPS()["approval"]}
        )["approval"]

    def _session_round_payloads(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Every payload ``store`` retains for ``session_id``, arm order first."""
        from tldw_chatbook.Chat.console_interrupt_rounds import session_round_payloads

        return session_round_payloads(self.lock, store, session_id)

    @staticmethod
    def _settle_pending_decision_timeout_locked(
        state: dict[str, Any], *, read_global_threading: Callable[[], Any]
    ) -> threading.Event | None:
        """Stamp one exact live round timeout once; caller holds its locks."""
        if state.get("settled"):
            return None
        state["settled"] = True
        state["terminal_reason"] = "timeout"
        state["remaining_active_seconds"] = 0.0
        state["active_since"] = None
        decision_type = state.get("decision_type")
        if decision_type == "approval":
            decisions = state.get("decisions")
            if isinstance(decisions, dict):
                for name in state.get("names", ()):
                    decisions[name] = "timeout"
        elif decision_type == "skill_install":
            decision = state.get("decision")
            if isinstance(decision, dict):
                decision["allow"] = False
        elif decision_type == "skill_script":
            decision = state.get("decision")
            if isinstance(decision, dict):
                decision["allow"] = False
                decision["remember"] = False
        event = state.get("event")
        return event if isinstance(event, read_global_threading().Event) else None

    def _summary_tail_messages(self, payload: dict[str, Any]) -> list:
        """User/assistant text projection of the round's stored conversation.

        Uses the same message flattening as world-info scanning
        (``_normalize_world_info_history``) and the same stored-message
        source its call sites feed it (``_provider_messages_for_session``,
        which reads ``self.store.messages_for_session`` and emits the
        provider-dict shape the flattener consumes); keeps the defensive
        no-raise posture.
        """
        try:
            session_id = str(payload.get("session_id") or "")
            messages: list = list(
                self.read_controller__provider_messages_for_session()(session_id)
                if session_id
                else []
            )
        except Exception:  # noqa: BLE001 -- advisory only
            return []
        return self.read_global__normalize_world_info_history()(messages)

    def _unpark_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str
    ) -> None:
        """Drop ``round_id``'s retained payload, if any."""
        from tldw_chatbook.Chat.console_interrupt_rounds import unpark_round_payload

        unpark_round_payload(self.lock, store, round_id)

    def active_session_changed(self) -> None:
        """Pause stale heads and derive the newly active session's head."""
        with self.lock:
            answerable = tuple(
                self.read_controller__answerable_decision_by_session().items()
            )
            self.read_controller__console_answerable_decision_by_session().clear()
            self.decision_view_revision += 1
        for answerable_session_id, decision_id in answerable:
            self.read_controller__refresh_answerable_decision()(answerable_session_id)
        session_id = self.read_controller_store().active_session_id
        if session_id:
            self.read_controller__reproject_pending_decision_for_session()(session_id)

    def add_pending_round(self, session_id: str, round_id: str, kind: str) -> None:
        """Register ``round_id`` as an outstanding approval-like round for ``session_id``.

        TASK-1050 (Defect A): the fleet-visible pending-approval badge used
        to be a single boolean per session (``_pending_approvals`` as a
        plain ``set[str]``, flipped by the now-deprecated ``set_run_
        pending_approval``) shared by THREE independent bridges -- MCP tool
        approvals, skill-install confirms, and skill-script confirms. Any
        one bridge's teardown cleared the badge for its own session_id
        regardless of whether a SIBLING round (same bridge or a different
        one) was still outstanding for that same session, so the badge
        could go dark while a live confirm was still waiting on the user.

        ``_pending_approvals`` is now keyed by session id to the SET of
        round ids currently outstanding for it -- a session reads as
        "pending" (``run_marker_for``/``fleet_summary_counts``) iff that
        set is non-empty. Idempotent: adding an already-registered
        ``round_id`` again is a no-op (set semantics), so a caller never
        needs to check first.

        Every genuine bridge round already mints a fresh ``uuid4()`` round/
        request id before arming (``request_mcp_approvals``'s ``round_id``,
        ``request_skill_install_confirm``'s/``request_skill_script_
        confirm``'s ``request_id``) -- this is the id each bridge now
        passes here instead of the old boolean.

        Args:
            session_id: The session the round belongs to.
            round_id: The round's own unique id (a real bridge round id, or
                the reserved ``_LEGACY_PENDING_APPROVAL_ROUND_ID`` sentinel
                -- see ``set_run_pending_approval``).
            kind: Which interrupt kind is waiting -- a
                registered interrupt key (``approval``, ``question``,
                ``skill_install``, ``skill_script``, ``worktree_merge``, or
                the standalone ``chat_create`` confirmation). Qodo #4: the badge and
                lifecycle do not care, but the run chip and activity line do
                -- they used to translate this registry's generic "something
                is pending" into "Waiting for your approval" even for a
                question. Defaults to ``approval``, which is what every
                caller without a kind of its own (the deprecated boolean
                shim, direct test drives) has always meant.
        """
        # F2b fix (Qodo wave), preserved: reachable from a worker thread
        # while the UI thread concurrently iterates `_pending_approvals`
        # via `fleet_summary_counts` -- guard the mutation with the shared
        # lock so iteration never observes a torn add/discard.
        with self.lock:
            rounds = self.read_controller__pending_approvals().setdefault(
                session_id, set()
            )
            changed = round_id not in rounds
            rounds.add(round_id)
            self.read_controller__pending_round_kinds().setdefault(session_id, {})[
                round_id
            ] = str(kind or self.read_global_CONSOLE_PENDING_APPROVAL_KIND())
        if changed:
            if self.read_controller__buddy_sink() is not None:
                self.read_controller__buddy_sink().approval_round(
                    session_id, round_id, pending=True
                )
            self.read_controller__advance_lifecycle_revision()(session_id)
            self.read_controller__publish_console_attention_change()()

    def announce_hidden_decision(self, session_id: str, kind: str) -> None:
        """Keep typed notices under their live stable-ID privacy authority."""
        if kind in ("approval", "skill_install", "skill_script"):
            with self.lock:
                decision_ids = tuple(
                    decision_id
                    for decision_id, state in self.registries[kind].items()
                    if state.get("session_id") == session_id
                )
            for decision_id in decision_ids:
                self.read_controller__announce_hidden_decision()(
                    kind, session_id, decision_id
                )
            return
        self.read_controller__announce_detached_approval()(session_id, kind=kind)

    def complete_definitive_run(self, run_id: str) -> None:
        """WORKER THREAD: remove finishing rows a run never dispatched."""
        if not run_id:
            return
        affected_sessions: set[str] = set()
        with self.lock:
            for round_id, payload in list(self.payloads["approval"].items()):
                if (
                    payload.get("phase") != "finishing"
                    or payload.get("run_id") != run_id
                ):
                    continue
                session_id = str(payload.get("session_id") or "")
                if session_id:
                    affected_sessions.add(session_id)
                self.payloads["approval"].pop(round_id, None)
        for session_id in affected_sessions:
            self.read_controller__remount_head()(
                self.payloads["approval"],
                self.read_controller_set_pending_approval(),
                session_id,
            )

    def complete_definitive_tool(
        self, run_id: str, call_key: str, tool_name: str
    ) -> None:
        """WORKER THREAD: clear one finishing row at its real terminal.

        The primary key is the provider call id.  Fence/local rows that did
        not carry one fall back to the tool name; only one matching row is
        consumed per callback so repeated same-name calls remain visible
        until each sequential mutation actually finishes.
        """
        affected_session: str | None = None
        with self.lock:
            for round_id, payload in list(self.payloads["approval"].items()):
                if payload.get("phase") != "finishing":
                    continue
                if payload.get("run_id") != run_id:
                    continue
                calls = list(payload.get("calls") or [])
                match_index = next(
                    (
                        index
                        for index, call in enumerate(calls)
                        if str(call.get("call_id") or call.get("llm_name") or "")
                        == call_key
                    ),
                    None,
                )
                if match_index is None:
                    match_index = next(
                        (
                            index
                            for index, call in enumerate(calls)
                            if not call.get("call_id")
                            and str(call.get("llm_name") or "") == tool_name
                        ),
                        None,
                    )
                if match_index is None:
                    continue
                calls.pop(match_index)
                affected_session = str(payload.get("session_id") or "") or None
                if calls:
                    payload["calls"] = calls
                else:
                    self.payloads["approval"].pop(round_id, None)
                break
        if affected_session is not None:
            self.read_controller__remount_head()(
                self.payloads["approval"],
                self.read_controller_set_pending_approval(),
                affected_session,
            )

    def discard_pending_round(self, session_id: str, round_id: str) -> None:
        """Clear ``round_id`` from ``session_id``'s outstanding approval-like rounds.

        TASK-1050 (Defect A) counterpart to ``add_pending_round``: discards
        only THIS round's id from the session's round-id set. The fleet
        badge (``run_marker_for``) clears only once that set is empty --
        i.e. once every bridge round for the session has resolved, not just
        this one. Idempotent: discarding an id that was never added (or was
        already discarded) is a safe no-op, and discarding the SAME id
        twice never double-decrements anything (set semantics -- there is
        nothing to corrupt).

        Args:
            session_id: The session the round belongs to.
            round_id: The round's own unique id, as passed to the matching
                ``add_pending_round`` call.
        """
        changed = False
        with self.lock:
            rounds = self.read_controller__pending_approvals().get(session_id)
            if rounds is None:
                return
            changed = round_id in rounds
            rounds.discard(round_id)
            kinds = self.read_controller__pending_round_kinds().get(session_id)
            if kinds is not None:
                kinds.pop(round_id, None)
                if not kinds:
                    self.read_controller__pending_round_kinds().pop(session_id, None)
            if not rounds:
                self.read_controller__pending_approvals().pop(session_id, None)
        if changed:
            if self.read_controller__buddy_sink() is not None:
                self.read_controller__buddy_sink().approval_round(
                    session_id, round_id, pending=False
                )
            self.read_controller__advance_lifecycle_revision()(session_id)
            self.read_controller__publish_console_attention_change()()

    def expire_pending_decisions(self) -> tuple[str, ...]:
        """Fail closed every answerable head whose active allowance elapsed."""
        now = self.read_controller_decision_monotonic_clock()()
        with self.lock:
            answerable = tuple(
                self.read_controller__answerable_decision_by_session().items()
            )
        for session_id, _decision_id in answerable:
            self.read_controller__refresh_answerable_decision()(session_id)
        expired = [
            decision_id
            for session_id, decision_id in answerable
            if self._expire_answerable_decision_if_due(session_id, decision_id, now=now)
        ]
        return tuple(expired)

    def has_pending_approval_round(self, session_id: str) -> bool:
        """Return whether ``session_id`` currently has ANY outstanding approval-like round.

        TASK-1050: exposed so a caller that lacks a round id of its own
        (see ``set_run_pending_approval``'s docstring) can check whether a
        REAL round is already registered before redundantly stamping the
        deprecated boolean shim -- ``ChatScreen._park_console_approval`` is
        the one production caller that needs this (its owning bridge always
        registers the real round id via ``add_pending_round`` moments
        before invoking the park callback, so by the time this runs, the
        real round is normally already present).

        Args:
            session_id: The session to check.

        Returns:
            ``True`` iff at least one round id is currently registered for
            ``session_id``.
        """
        with self.lock:
            return session_id in self.read_controller__pending_approvals()

    def on_console_view_visibility_changed(self, visible: bool) -> None:
        """Project screen visibility without changing execution or cancellation."""
        if self.read_controller__disposed():
            return
        self.set_view_visible(visible)
        if visible:
            self.read_controller_remount_pending_approval_for_active_session()()
            session_id = self.read_controller_store().active_session_id
            if session_id:
                self.read_controller__remount_session_kinds()(session_id)

    def on_pending_rounds_changed(self, total: int, kind: str, raised: bool) -> None:
        """task-31385: attention when a round blocks on the user off-screen.

        WORKER THREAD; the host calls this after every round arms (mounted
        or parked) and after every teardown. Two effects, both on the UI
        thread: the Console entry in the app navigation carries a
        pending-interrupt badge while ``total`` is non-zero, and a round
        that ARMS while Console is hidden or detached -- another screen or
        modal is visible, or Console has not been opened this launch --
        rings the terminal bell once. The bell is governed by
        ``[console] interrupt_bell`` (default on) and never fires in a
        headless app, so tests and embedded runs emit no control bytes.
        ``_approval_view_is_detached`` combines suspend/resume visibility
        with the absent-hook fallback for a truly detached view.

        Args:
            total: Rounds of every kind registered after this change.
            kind: The round kind that changed (unused; kept for callers
                that want to specialise).
            raised: True for an arm, False for a teardown.
        """
        app = self.read_controller_app()
        if app is None:
            return
        ring = (
            raised
            and total > 0
            and self.read_controller__interrupt_bell_enabled()()
            and self.read_controller__approval_view_is_detached()()
            and not bool(getattr(app, "is_headless", False))
        )
        host = self

        def _apply() -> None:
            from tldw_chatbook.UI.Navigation.main_navigation import (
                set_console_attention,
            )

            # Re-read the total on the UI thread: two workers' updates may
            # be marshalled out of order, and the badge must show the
            # host's current truth, not whichever dispatch ran last.
            current = host.pending_total() if host is not None else total
            set_console_attention(app, current)
            bell = getattr(app, "bell", None) if ring else None
            if callable(bell):
                bell()

        marshal = getattr(app, "call_from_thread", None)
        if not callable(marshal):
            return
        try:
            marshal(_apply)
        except Exception:  # noqa: BLE001 -- attention is best-effort, never the round
            self.read_global_logger().opt(exception=True).debug(
                "Console attention marshal failed"
            )

    def pending_chat_create_ids(self) -> list[str]:
        """Return the request ids of every currently-armed chat-create round.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending. Exposed for tests and for any surface that needs to
            know whether a decision is outstanding.
        """
        with self.read_controller__pending_chat_create_lock():
            return list(self.read_controller__pending_chat_create_rounds())

    def pending_decision_projection(
        self, session_id: str
    ) -> ConsolePendingDecisionProjection | None:
        """Return the session's one stable mixed-type FIFO head."""
        states = self._pending_round_states_snapshot()
        now = self.read_controller_decision_monotonic_clock()()
        with self.lock:
            payloads = self._pending_decision_payloads_locked(session_id)
            if not payloads:
                return None
            payload = payloads[0]
            decision_id = str(payload["_decision_id"])
            state = states.get(decision_id)
            finishing = payload.get("phase") == "finishing"
            if (state is None or state.get("settled")) and not finishing:
                return None
            state = state or {}
            remaining = state.get("remaining_active_seconds")
            active_since = state.get("active_since")
            if remaining is not None and active_since is not None:
                remaining = max(
                    0.0, float(remaining) - max(0.0, now - float(active_since))
                )
            snapshot = dict(payload)
            snapshot["timeout_seconds"] = remaining or 0.0
            return self.read_global_ConsolePendingDecisionProjection()(
                decision_type=str(payload["_decision_type"]),
                session_id=session_id,
                decision_id=decision_id,
                remaining_active_seconds=remaining,
                payload=snapshot,
            )

    def pending_question_ids(self) -> list[str]:
        """Return the request ids of every armed question round, arm order.

        Returns:
            The armed round ids; empty when none is pending.
        """
        with self.lock:
            return list(self.registries["question"])

    def pending_round_count(self, session_id: str, *, kind: str) -> int:
        """Count one session's outstanding rounds of the requested kind.

        Args:
            session_id: The owning session to inspect.
            kind: Interrupt kind to count, including queued or hidden rounds.

        Returns:
            Number of registered rounds of this kind for the session.
        """
        with self.lock:
            return sum(
                value == kind
                for value in self.read_controller__pending_round_kinds()
                .get(session_id, {})
                .values()
            )

    def pending_round_kinds(self, session_id: str) -> frozenset[str]:
        """Return the KINDS of ``session_id``'s outstanding interrupt rounds.

        Qodo #4: ``has_pending_approval_round`` answers "is anything waiting
        on the user", which is the right question for the badge and the
        lifecycle but the wrong one for copy -- the shared registry holds
        questions, skill-install/skill-script confirms and worktree-merge
        confirms as well as MCP approvals, and translating the generic
        predicate into "Waiting for your approval" mislabels the other four
        (and can disagree with the inspector, which counts mounted approval
        cards only).

        Args:
            session_id: The session to read.

        Returns:
            Every distinct registered kind outstanding for ``session_id``,
            including standalone chat creation, empty when nothing is.
        """
        with self.lock:
            return frozenset(
                self.read_controller__pending_round_kinds().get(session_id, {}).values()
            )

    def pending_skill_install_ids(self) -> list[str]:
        """Return the request ids of every currently-armed install-confirm round.

        Mirrors ``pending_skill_script_ids`` -- exposed for tests and for
        any surface that needs to know whether a decision is outstanding.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending.
        """
        with self.lock:
            return list(self.registries["skill_install"])

    def pending_skill_script_ids(self) -> list[str]:
        """Return the request ids of every currently-armed confirm round.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending. Exposed for tests and for any surface that needs to
            know whether a decision is outstanding.
        """
        with self.lock:
            return list(self.registries["skill_script"])

    def pending_worktree_merge_ids(self) -> list[str]:
        """Return the request ids of every currently-armed worktree-merge
        confirm round. Mirrors ``pending_skill_script_ids``.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending.
        """
        with self.lock:
            return list(self.registries["worktree_merge"])

    def project_pending_decision_for_active_session(self) -> bool:
        """Project only the active session's ordered mixed-type head."""
        session_id = self.read_controller_store().active_session_id
        projection = (
            self.read_controller_pending_decision_projection()(session_id)
            if session_id
            else None
        )
        if self.read_controller_set_pending_decision() is not None:
            mounted = self.read_controller_set_pending_decision()(projection) is True
            if (
                mounted
                and projection is not None
                and projection.decision_type == "approval"
            ):
                self.read_controller__maybe_fire_permission_summary()(
                    dict(projection.payload)
                )
            return mounted
        if projection is None:
            for setter in (
                self.read_controller_set_pending_approval(),
                self.read_controller_set_pending_skill_install(),
                self.read_controller_set_pending_skill_script(),
            ):
                if setter is not None:
                    setter(None)
            return False
        setter = {
            "approval": self.read_controller_set_pending_approval(),
            "skill_install": self.read_controller_set_pending_skill_install(),
            "skill_script": self.read_controller_set_pending_skill_script(),
        }[projection.decision_type]
        if setter is None:
            return False
        setter(dict(projection.payload))
        return True

    def remount_pending_approval_for_active_session(self) -> bool:
        """Mount the ACTIVE session's still-armed approval round, if any.

        task-15860 Task 5. UI THREAD (called from
        ``ConsoleRuntime.attach_view``, which runs on it). Re-derives the
        card from ``_parked_approval_payloads`` exactly as
        ``switch_session`` does -- same single source of truth, no second
        copy of "what is this session's card showing".

        Deliberately mounts NOTHING when no round is armed: pushing
        ``None`` here would clear a card on every new claim, and an attach
        is not a reason to hide anything.

        PR0 (task-15661, fixed): ``_parked_approval_payloads`` is keyed by
        ROUND now, so two rounds armed for one session each keep their own
        payload. This mounts the session's FIFO HEAD -- the oldest-armed
        round -- and each later sibling mounts in turn as the head ahead of
        it resolves. Covered by ``Tests/UI/test_console_headless_approval.
        py::test_two_headless_rounds_each_mount_in_turn``.

        Returns:
            True when a card was mounted.
        """
        session_id = self.read_controller_store().active_session_id
        if not session_id:
            return False
        if self.read_controller_set_pending_decision() is not None:
            projection = self.read_controller_pending_decision_projection()(session_id)
            self.read_controller_project_pending_decision_for_active_session()()
            return projection is not None
        if self.read_controller_set_pending_approval() is None:
            return False
        # The pre-PR0 `still_armed` pre-test is redundant: a round unparks
        # its own payload in its own teardown, so a payload present here
        # necessarily belongs to a live round.
        payload = self.read_controller__head_round_payload()(
            self.payloads["approval"], session_id
        )
        if payload is None:
            return False
        self.read_controller_set_pending_approval()(payload)
        # ADR-090: a headless attach is often the FIRST mount a parked
        # round ever gets -- arm the summary trigger here (bypasses
        # `_marshal_pending_approval`; fire-once makes re-attaches safe).
        if isinstance(payload, dict):
            self.read_controller__maybe_fire_permission_summary()(payload)
        return True

    def request_chat_create_confirm(
        self, payload: dict[str, Any], *, session_id: str | None
    ) -> dict[str, bool]:
        """WORKER THREAD: ask the user to confirm an agent-initiated chat create.

        Mirrors request_skill_script_confirm for the fork_chat/new_chat
        tools, with one addition: a session-scoped "remember" grant store.
        A prior allow+remember decision for ``(session_id, tool)`` (recorded
        in ``_chat_create_session_grants``) short-circuits this call with
        ``{"allow": True, "remember": True}`` before any round is armed --
        no card, no wait. Grants die with their session (``close_session``
        pops the whole set).

        Each call arms a fresh round under a newly-generated request id
        (embedded in the payload handed to the UI as ``"request_id"``) so
        that ``resolve_pending_chat_create`` can reject a decision left
        over from a prior, already-torn-down round -- see that method's
        docstring for why this matters.

        Carries the SAME park/mount/retain contract as
        ``request_skill_script_confirm`` -- see that method's docstring for
        the full mount-vs-park/retain rationale, identical here.

        Args:
            payload: Confirm details to render ({"tool" ("fork_chat"|
                "new_chat"), "title", "opening_prompt", "instructions"});
                "fork_source_title"/"fork_message_count" (fork_chat only),
                a default "title" when the agent omitted one, and a
                normalized "run_id" are enriched by
                ``_enrich_chat_create_confirm_payload`` before arming, and
                "timeout_seconds", "request_id", "session_id" and
                "deadline_monotonic" keys are added before marshaling to
                the UI.
            session_id: The run's OWNING session, scoping the cancel check
                (``_is_session_cancelled``), the park/mount decision, and
                the remember-grant lookup. ``None`` preserves the
                viewed-session fallback and never parks.

        Returns:
            ``{"allow": bool, "remember": bool}``. Every non-Allow path
            (deny, cancel, stop, timeout, no wired UI) returns
            ``allow=False``.
        """
        from tldw_chatbook.Agents.human_input_wait import use_human_input_wait
        from tldw_chatbook.Agents.agent_models import AGENT_KIND_PRIMARY

        owning_session_id = (
            session_id
            if session_id is not None
            else (self.read_controller_store().active_session_id or "")
        )
        tool = str(payload.get("tool") or "")
        # PR review #13: the card must display the TRUE owning run id. The
        # bridge closure rides the originating assistant message id under
        # `run_id` (the run's own id does not exist when the closure is
        # built); the round's arm-time stamp knows the real one.
        true_run_id = self.read_global_current_run_id()()
        if true_run_id:
            payload = {**payload, "run_id": str(true_run_id)}
        # TASK-32531: stamp the REQUESTING run's identity (kind + parent)
        # from the agent_runs row -- the card must name WHO is asking, and
        # a sub-agent requester never rides a session grant (below).
        # Qodo 2761 finding 2: UNKNOWN is fail-safe -- when the run row
        # cannot be read (no bridge db, raise, missing row) the requester
        # is treated as NOT primary, so a session grant can never ride on
        # an unverified identity.
        requesting_kind = "unknown"
        requesting_parent: str | None = None
        requesting_task = ""
        agent_db = getattr(
            self.read_controller__agent_bridge(), "runs_db", None
        ) or getattr(self.read_controller__agent_bridge(), "agent_runs_db", None)
        get_run = getattr(agent_db, "get_run", None)
        if true_run_id and callable(get_run):
            try:
                from tldw_chatbook.DB.base_db import operation_owned_connection

                with operation_owned_connection(agent_db):
                    row = get_run(str(true_run_id))
            except Exception:  # noqa: BLE001 -- identity is best-effort
                row = None
            if isinstance(row, dict) and row.get("agent_kind"):
                requesting_kind = str(row["agent_kind"])
                parent = row.get("parent_run_id")
                requesting_parent = str(parent) if parent else None
                requesting_task = str(row.get("task") or "")[:120]
        payload = {
            **payload,
            "agent_kind": requesting_kind,
            "parent_run_id": requesting_parent,
            "agent_task": requesting_task,
        }
        observation = (
            self.read_controller__observe_chat_creation_record()(payload)
            if tool == "new_chat"
            else None
        )
        # Decide a remembered grant atomically with Close and revocation.
        with self.read_controller__pending_chat_create_lock():
            record = (
                self.read_controller__chat_creation_record_locked()(
                    payload, observation
                )
                if tool == "new_chat"
                else None
            )
            refused = (
                owning_session_id in self.read_controller__session_close_generations()
                or (tool == "new_chat" and record is None)
            )
            grant = payload["_grant_scope"] if record is not None else tool
            if (
                not refused
                and requesting_kind == AGENT_KIND_PRIMARY
                and grant
                in self.read_controller__chat_create_session_grants().get(
                    owning_session_id, set()
                )
            ):
                if record is not None:
                    record["approved"] = True
                return {"allow": True, "remember": True}
        if refused:
            token = payload.get("_creation_token")
            if isinstance(token, self.read_global__ChatCreationToken()):
                token.close()
            return {"allow": False, "remember": False}
        if (
            self.read_controller_app() is None
            or self.read_controller_set_pending_chat_create() is None
        ):
            if record is not None:
                payload["_creation_token"].close()
            return {"allow": False, "remember": False}

        # Final-review fix wave (Finding 1): enrich the payload BEFORE the
        # round is armed -- fork_source_title/fork_message_count (the card's
        # fork line), a default title when the agent omitted one, and run-id
        # attribution. Entirely best-effort; see the helper.
        enriched_payload = self.read_controller__enrich_chat_create_confirm_payload()(
            payload
        )

        event = self.read_global_threading().Event()
        decision: dict[str, bool] = {}
        request_id = str(self.read_global_uuid4()())
        # Arm-time cancel binding, identical to the sibling bridges' -- see
        # `_bind_round_cancel_signal`.
        round_cancel_event = self.read_controller__bind_round_cancel_signal()(
            session_id
        )
        if (
            record is not None
            and requesting_kind == "subagent"
            and self.read_controller__active_assistant_message_ids().get(
                owning_session_id
            )
            != record["payload"]["source_message_id"]
        ):
            round_cancel_event = None
        # The visit's teardown Event, captured at ARM time for the same
        # reason the run's cancel event is -- see `_bind_visit_cancel_signal`.
        visit_cancel_event = self.read_controller__bind_visit_cancel_signal()()
        # Same run-ownership stamp the sibling bridges carry -- a revoked
        # round must fail closed even against a late Allow.
        chat_create_round_state: dict[str, Any] = {
            "event": event,
            "decision": decision,
            "session_id": owning_session_id,
            # Legacy callers never park; their initial card remains unscoped.
            "session_scoped": session_id is not None,
            "run_id": self.read_global_current_run_id()(),
            # Re-read after the wait: a late Allow must not stick. See
            # `revoke_approval_rounds_for_run`.
            "revoked": False,
        }
        with self.read_controller__pending_chat_create_lock():
            # Enrichment can finish after Close has swept the standalone rounds.
            if owning_session_id in self.read_controller__session_close_generations():
                return {"allow": False, "remember": False}
            if record is not None and (
                true_run_id in self.read_controller__chat_creation_revoked_runs()
                or self.read_controller__chat_creation_records().get(
                    payload["_creation_token"]
                )
                is not record
            ):
                return {"allow": False, "remember": False}
            self.read_controller__pending_chat_create_rounds()[request_id] = (
                chat_create_round_state
            )

        timeout_seconds = (
            self.read_controller_chat_create_confirm_timeout_seconds()()
            if self.read_controller_chat_create_confirm_timeout_seconds() is not None
            else self.read_global__DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS()
        )
        # ADR-067: <= 0 arms NO deadline (the default) -- the round waits
        # for a decision or the owning run's cancellation.
        deadline = (
            self.read_global_time().monotonic() + timeout_seconds
            if timeout_seconds > 0
            else None
        )
        card_payload = dict(enriched_payload)
        card_payload["timeout_seconds"] = timeout_seconds
        card_payload["request_id"] = request_id
        card_payload["session_id"] = owning_session_id
        # See `_head_round_payload`'s remaining-time snapshot. None when
        # ADR-067 armed no deadline.
        card_payload["deadline_monotonic"] = deadline
        is_parked = session_id is not None and session_id != (
            self.read_controller_store().active_session_id or ""
        )
        # Legacy `session_id is None` callers never park and never queue --
        # they keep the unconditional mount below.
        is_head = True
        if session_id is not None:
            self.read_controller_add_pending_round()(
                session_id,
                request_id,
                kind=self.read_global_CONSOLE_PENDING_CHAT_CREATE_KIND(),
            )
            # Keyed by ROUND; the return says whether THIS round is its
            # session's FIFO head. A non-head round must not mount: an
            # older sibling is still holding the card.
            is_head = self.read_controller__park_round_payload()(
                self.read_controller__parked_chat_create_payloads(),
                request_id,
                card_payload,
            )
        try:
            if is_parked:
                if (
                    self.read_controller_app() is not None
                    and self.read_controller_park_pending_approval() is not None
                ):
                    self.read_controller_app().call_from_thread(
                        self.read_controller_park_pending_approval(), session_id
                    )
            elif is_head:
                self.read_controller__marshal_pending_chat_create()(card_payload)
            # ADR-067: mark the owning run as waiting on a human decision
            # (see the sibling bridges' identical wrap for the why).
            with use_human_input_wait(str(chat_create_round_state.get("run_id") or "")):
                while not event.wait(self.read_global__MCP_APPROVAL_POLL_SECONDS()):
                    if self.read_controller__is_session_cancelled()(
                        session_id,
                        cancel_event=round_cancel_event,
                        visit_event=visit_cancel_event,
                    ):
                        break
                    if (
                        deadline is not None
                        and self.read_global_time().monotonic() >= deadline
                    ):
                        break
            # A revoked round denies unconditionally, without consulting
            # `decision` at all -- an Allow delivered just after the child
            # was cancelled must not authorize the create. Mirrors the
            # sibling bridges' identical post-wait guard.
            # Human approval is a new observation phase; never reuse its entry rows.
            observation = (
                self.read_controller__observe_chat_creation_record()(payload)
                if tool == "new_chat" and decision.get("allow", False)
                else None
            )
            with self.read_controller__pending_chat_create_lock():
                if (
                    chat_create_round_state.get("revoked")
                    or owning_session_id
                    in self.read_controller__session_close_generations()
                ):
                    return {"allow": False, "remember": False}
                allow = bool(decision.get("allow", False))
                remember = requesting_kind == AGENT_KIND_PRIMARY and bool(
                    decision.get("remember", False)
                )
                # Decide and remember atomically with the Close/revocation sweep.
                # A remembered deny must never become a standing grant.
                if allow and tool == "new_chat":
                    allow = (
                        self.read_controller__chat_creation_record_locked()(
                            payload, observation
                        )
                        is record
                    )
                    if allow:
                        record["approved"] = True
                if allow and remember:
                    self.read_controller__chat_create_session_grants().setdefault(
                        owning_session_id, set()
                    ).add(grant)
                return {"allow": allow, "remember": remember}
        finally:
            if record is not None and not record["approved"]:
                payload["_creation_token"].close()
            with self.read_controller__pending_chat_create_lock():
                self.read_controller__pending_chat_create_rounds().pop(request_id, None)
            # Drop exactly THIS round's retained payload -- each round owns
            # its own key, so no still-armed sibling guard is needed.
            self.read_controller__unpark_round_payload()(
                self.read_controller__parked_chat_create_payloads(), request_id
            )
            if session_id is not None:
                # Discard ONLY this round's own id -- the badge clears only
                # once every bridge round for this session has resolved.
                self.read_controller_discard_pending_round()(session_id, request_id)
            # Re-derive the card from the session's remaining FIFO head
            # rather than deciding whether to CLEAR it -- a legacy
            # no-session round passes None so `_remount_head` re-derives
            # for the session active WHEN THE CALLBACK RUNS; a
            # session-attributed round keeps its exact-match owning id.
            try:
                # Live-UAT fix: `_remount_head` early-returns into the
                # decision-projection system when `set_pending_decision`
                # is wired and would never clear OUR standalone map's
                # payload, leaving a resolved card re-appearing on every
                # re-render. Push the head (or None) directly instead.
                target_session = (
                    owning_session_id
                    if session_id is not None
                    else (self.read_controller_store().active_session_id or "")
                )
                if target_session == (
                    self.read_controller_store().active_session_id or ""
                ):
                    self.read_controller__marshal_pending_chat_create()(
                        self.read_controller__head_round_payload()(
                            self.read_controller__parked_chat_create_payloads(),
                            target_session,
                        )
                    )
            except Exception:  # noqa: BLE001 -- suppress teardown-time errors
                self.read_global_logger().opt(exception=True).debug(
                    "Failed to marshal chat-create remount during teardown"
                )

    def request_mcp_approvals(
        self, pending: list[MCPPendingCall], *, session_id: str | None
    ) -> dict[str, str]:
        """Bridge one batch of pending tool-approval rows to the Console UI and back.

        TASK-630: OWNER-AGNOSTIC, despite the legacy ``mcp`` in the name.
        Since TASK-545/P1's run-level ``build_tool_review_hook``, the rows
        handed here may come from MCP tools OR from built-in agent-runtime
        tools (``server_key="agent:builtin"``); every row is marshalled to
        the same ``ChatApprovalCard`` and resolved through the same
        Event-polling loop below. There is no separate approval path for
        built-ins -- a reader assuming "MCP-only" here would go looking for
        one that does not exist. (The name is kept: it is the wire between
        this method, ``resolve_pending_approval``, and the round-id
        plumbing, and renaming it is churn without a defect.)

        WORKER THREAD. Bound (via a ``functools.partial`` binding this
        run's ``session_id``, Task 9) as ``MCPToolProvider``'s
        ``approval_callback`` and ``build_tool_review_hook``'s
        ``request_approvals``, so this runs on the agent bridge's
        background OS thread (the ``asyncio.to_thread`` call inside
        ``_run_agent_reply``) -- it must never touch a widget directly,
        only through ``self.app.call_from_thread``.

        Builds a fresh ``threading.Event`` + shared decisions dict (stored
        under this round's own entry in ``_pending_approval_rounds``, keyed
        by a freshly minted ``round_id`` -- see that map's own docstring
        for why a single shared slot, or a slot keyed by session id alone,
        could not survive concurrent sessions or same-session round
        replacement). Either MOUNTS the card immediately (``session_id`` is
        the currently ACTIVE/viewed session, or unknown -- legacy
        no-session callers keep the pre-Task-9 always-mount behavior) or
        PARKS it (``session_id`` is a DIFFERENT, background session --
        Task 9: the retained ``payload`` goes into
        ``_parked_approval_payloads`` for ``switch_session`` to mount
        later, while the controller raises one sanitized app-wide notice
        for this exact stable decision id instead of touching a screen-owned
        parking hook). PR0
        adds a third case: an ACTIVE-session round that is not its
        session's FIFO head neither mounts nor parks -- an older sibling
        still owns the card, and this round's payload is retained under
        its own ``round_id`` until that sibling's teardown promotes it.
        Either way it then polls ``event.wait(1.0)`` re-checking this run's OWN
        cancel signal (``_is_session_cancelled``) and -- only when a
        POSITIVE timeout is configured (ADR-067: the default is 0 = none)
        -- a deadline, every second until one of three things happens: the
        user submits a decision (``resolve_pending_approval``, called from
        the UI thread once the card's own stamped ``round_id`` is delivered
        back, sets the Event -- Fix round 1: NOT "whichever round belongs
        to the active session", see ``resolve_pending_approval``'s own
        docstring for why that was a real cross-session misattribution
        hazard), the run is cancelled/torn down (``_is_session_cancelled``
        -- F5 fix, Qodo wave: this round's OWN cancel event, or real
        process teardown via ``_shutdown_requested``, never any OTHER
        session's bare Stop -- see that method's own docstring), or the
        configured approval timeout elapses. With no deadline armed the
        round simply waits for one of the first two, however long the
        human takes -- the wait is marked in ``Agents.human_input_wait``
        so a per-call wrapper hosting it pauses its ceiling. Whichever
        addressable verdict key (native ``call_id`` when present, otherwise
        ``llm_name``) never received an explicit decision by then
        fails closed to ``"deny"``
        (cancellation) or ``"timeout"`` (deadline) -- see
        ``MCPToolProvider._apply_verdict`` for how each decision string is
        consumed. The mounted card (if any) is always cleared afterwards
        (``finally``), regardless of outcome -- but ONLY if this round's
        session is STILL the active one at that moment, so a background
        round resolving (timeout/cancel) while some OTHER session's card is
        showing never clobbers it.

        Args:
            pending: One turn's pending tool calls awaiting approval. Native
                calls use their call id as the verdict key; id-less fence
                calls sharing a name share one name-keyed verdict.
            session_id: The run's OWNING session (Task 3 threads it through
                ``_run_agent_reply``). ``None`` preserves every pre-Task-9
                call site's behavior (always mounts against whatever
                session is active at ROUND-key time; no parking).

        Returns:
            An `ApprovalDecisions` (a plain verdict `dict` carrying
            ``unresolved_keys``) holding a decision string
            (``approve_once``/``approve_session``/``always_allow``/
            ``deny``/``timeout``) for every addressable call-id-or-name
            verdict key in ``pending``. Keys listed in ``unresolved_keys``
            hold the fail-closed ``"deny"`` default of a round nobody
            answered, not a user refusal -- see that class.
        """
        start = self.read_controller__chat_start()._active.get(session_id)
        if start is not None and not start.accepted:
            start.withdrawal_reason = "target_consent_required"
            return {
                str(call.call_id or "") or call.llm_name: "deny" for call in pending
            }
        unique_keys: list[str] = []
        seen: set[str] = set()
        call_by_key: dict[str, "MCPPendingCall"] = {}
        for call in pending:
            key = str(call.call_id or "") or call.llm_name
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
                call_by_key[key] = call
        if not unique_keys:
            return {}
        if self.read_controller_app() is None:
            # Qodo #2597 #8: no UI is wired, so no card can be shown and
            # nobody can answer -- this fails CLOSED, but it is NOT a user
            # denial. Returned as a bare dict it looked exactly like one to
            # `approval_was_unanswered()`, and both review hooks then wrote
            # a `record_user_denial()` audit row claiming a person picked
            # Deny. Every key here is unresolved, by construction.
            # No `denied-unresolved` audit row is written (or possible)
            # here: the execution log is reached through
            # `self.app.unified_mcp_service`, and this branch exists
            # precisely because there is no app. `unresolved_keys` is what
            # keeps the hooks from inventing a user decision instead.
            headless = self.read_global_ApprovalDecisions()(
                {key: "deny" for key in unique_keys}
            )
            headless.unresolved_keys = frozenset(unique_keys)
            return headless
        event = self.read_global_threading().Event()
        decisions = self.read_global_ApprovalDecisions()()
        round_id = str(self.read_global_uuid4()())
        owning_session_id = (
            session_id
            if session_id is not None
            else (self.read_controller_store().active_session_id or "")
        )
        round_cancel_event = self.read_controller__bind_round_cancel_signal()(
            session_id
        )
        from tldw_chatbook.Agents.mcp_tool_provider import (
            current_mcp_invocation_policies,
        )

        hook_policies = current_mcp_invocation_policies()
        if hook_policies:
            if any(not policy.allow_approval for policy in hook_policies):
                return self.read_global_ApprovalDecisions()(
                    {key: "deny" for key in unique_keys}
                )
            original_cancel_event = round_cancel_event

            class HookRoundCancellation:
                def is_set(self):
                    return (
                        original_cancel_event is not None
                        and original_cancel_event.is_set()
                    ) or any(policy.cancel_event.is_set() for policy in hook_policies)

            round_cancel_event = HookRoundCancellation()
        visit_cancel_event = self.read_controller__bind_visit_cancel_signal()()
        owning_run_id = self.read_global_current_run_id()()
        round_state: dict[str, Any] = {
            "event": event,
            "decisions": decisions,
            "session_id": owning_session_id,
            "run_id": owning_run_id,
            "names": tuple(unique_keys),
            "calls": tuple(pending),
            "revoked": False,
            "summary": None,
            "summary_fired": False,
            "cancel_event": round_cancel_event,
            "visit_event": visit_cancel_event,
        }
        # Registered before the timeout config read so sweeps and
        # `pending_*` readers see the round across that (lock-taking) call;
        # `run_round` re-registers the same object, harmlessly.
        if not self.register_round("approval", round_id, round_state):
            self.read_controller__record_cancelled_approval_decisions()(
                list(unique_keys), call_by_key
            )
            denied = self.read_global_ApprovalDecisions()(
                {key: "deny" for key in unique_keys}
            )
            denied.unresolved_keys = frozenset(unique_keys)
            return denied
        timeout_seconds = self.read_controller__resolve_mcp_approval_timeout_seconds()()
        deadline = (
            self.read_global_time().monotonic() + timeout_seconds
            if session_id is None and timeout_seconds > 0
            else None
        )
        if hook_policies:
            deadline = min(
                ([deadline] if deadline is not None else [])
                + [
                    min(
                        policy.deadline,
                        (
                            policy.approval_deadline()
                            if policy.approval_deadline is not None
                            else policy.deadline
                        ),
                    )
                    for policy in hook_policies
                ]
            )
        payload = self.read_global__build_approval_payload()(
            round_id,
            owning_session_id,
            owning_run_id,
            pending,
            timeout_seconds,
            deadline,
        )
        is_parked = session_id is not None and session_id != (
            self.read_controller_store().active_session_id or ""
        )
        approved_values = {"approve_once", "approve_session", "always_allow"}
        # task-32280 fix round (R23): the keys whose "deny" below is a
        # fail-closed DEFAULT, not a user decision. `_record_cancelled_
        # approval_decisions` already wrote the honest `denied-unresolved`
        # audit row for exactly these; carried out on the returned map so
        # the review hooks can skip recording a second, dishonest "Denied
        # by you" row for the same call. See `ApprovalDecisions`.
        unresolved_keys: set[str] = set()

        def _on_cancelled() -> None:
            if hook_policies:
                decisions.clear()
            cancelled_keys = [key for key in unique_keys if key not in decisions]
            for key in unique_keys:
                decisions.setdefault(key, "deny")
            unresolved_keys.update(cancelled_keys)
            self.read_controller__record_cancelled_approval_decisions()(
                cancelled_keys, call_by_key
            )

        def _on_timeout() -> None:
            if hook_policies:
                decisions.clear()
            for key in unique_keys:
                decisions.setdefault(key, "timeout")

        # Filled by `_on_outcome` BEFORE the host pops the registry, so a
        # `resolve_pending_approval` landing after the snapshot is ignored
        # (TASK-913) and the audit rows for a revoked round precede its
        # card being cleared, as before the host.
        result: dict[str, dict[str, str]] = {}

        def _on_outcome(outcome: str) -> None:
            # Commit the whole batch under the sweep lock. Cancellation can
            # win before this snapshot, but cannot retract a completed one.
            with self.lock:
                if hook_policies:
                    if (
                        deadline is not None
                        and self.read_global_time().monotonic() >= deadline
                    ):
                        outcome = "timeout"
                        decisions.update({key: "timeout" for key in unique_keys})
                    elif round_cancel_event.is_set():
                        outcome = "cancelled"
                        decisions.update({key: "deny" for key in unique_keys})
                        unresolved_keys.update(unique_keys)
                revoked = outcome == "revoked" or bool(round_state.get("revoked"))
                if revoked:
                    unresolved_keys.update(unique_keys)
                    result["map"] = {key: "deny" for key in unique_keys}
                else:
                    for key in unique_keys:
                        decisions.setdefault(key, "deny")
                    result["map"] = {
                        key: decisions.get(key, "deny") for key in unique_keys
                    }
            if revoked:
                self.read_controller__record_cancelled_approval_decisions()(
                    list(unique_keys), call_by_key
                )

        def _announce_if_detached() -> bool:
            # Sampled after the park, at the same moment the pre-host body
            # did, so an attach landing meanwhile mounts the card instead.
            if not self.read_controller__approval_view_is_detached()():
                return False
            self.read_controller__announce_hidden_decision()(
                "approval", owning_session_id, round_id
            )
            return True

        def _on_teardown() -> bool:
            # Runs inside the host's teardown, BEFORE the payload is
            # unparked: a definitive-after-start batch that was approved
            # keeps its card mounted in the "finishing" phase, which means
            # returning True to retain it. Only a wait that completed
            # (decided/timeout/cancelled) can do so; an exception mid-wait
            # leaves no snapshot and unparks, as before the host.
            snapshot = result.get("map")
            if snapshot is None or round_state.get("revoked"):
                return False
            finishing_calls = [
                call_payload
                for call_payload in payload["calls"]
                if call_payload.get("execution_policy")
                == self.read_global_ToolExecutionPolicy().DEFINITIVE_AFTER_START.value
                and snapshot.get(
                    str(
                        call_payload.get("call_id")
                        or call_payload.get("llm_name")
                        or ""
                    )
                )
                in approved_values
            ]
            if not finishing_calls or session_id is None:
                return False
            with self.lock:
                retained = self.payloads["approval"].get(round_id)
                if retained is not None:
                    retained["phase"] = "finishing"
                    retained["calls"] = finishing_calls
                    retained["timeout_seconds"] = 0.0
                    retained["deadline_monotonic"] = None
            return True

        # task-31384: one host lifecycle; the approvals-only legs ride the
        # hooks -- the detached-view announce (task-15860) instead of a
        # mount, the advisory permission summary fired INSIDE the human-wait
        # mark, decision stamping on cancel/timeout, and finishing-phase
        # retention at teardown.
        activity_bridge = self.read_controller__agent_bridge()
        project_wait = getattr(activity_bridge, "set_tool_approval_pending", None)

        def project_tool_wait(pending: bool) -> None:
            if callable(project_wait):
                try:
                    project_wait(owning_session_id, owning_run_id, unique_keys, pending)
                except Exception as exc:  # noqa: BLE001 — display cannot interrupt approval
                    self.read_global_logger().warning(
                        "Console tool approval display could not be updated ({})",
                        type(exc).__name__,
                    )

        project_tool_wait(True)
        try:
            self.run_round(
                "approval",
                round_id,
                payload,
                round_state,
                session_id=session_id,
                owning_session_id=owning_session_id,
                deadline=deadline,
                is_parked=is_parked,
                **({"hard_deadline": deadline} if hook_policies else {}),
                announce_detached=_announce_if_detached,
                human_wait_run_id=owning_run_id,
                on_cancelled=_on_cancelled,
                on_timeout=_on_timeout,
                before_wait=lambda: (
                    self.read_controller__maybe_fire_permission_summary()(payload)
                ),
                on_teardown=_on_teardown,
                on_outcome=_on_outcome,
            )
        finally:
            project_tool_wait(False)
        verdicts_out = self.read_global_ApprovalDecisions()(
            result.get("map") or {key: "deny" for key in unique_keys}
        )
        verdicts_out.unresolved_keys = frozenset(unresolved_keys)
        verdicts_out.denial_reasons = {
            key: reason
            for key, reason in decisions.denial_reasons.items()
            if verdicts_out.get(key) == "deny" and key not in unresolved_keys
        }
        return verdicts_out

    def request_skill_install_confirm(
        self, url: str, *, session_id: str | None
    ) -> bool:
        """WORKER THREAD: ask the user to confirm a skill install before any fetch.

        TASK-910: mirrors ``request_mcp_approvals``' park/mount/retain
        contract. Registers a fresh round (event + decision box + owning
        session id) under a freshly minted request id in
        ``_pending_skill_install_rounds`` (mirrors ``_pending_skill_script_
        rounds``' identical per-round design -- the pre-TASK-910 single
        ``_pending_skill_install_event``/``_pending_skill_install_decision``
        pair could not survive two DIFFERENT sessions each raising their own
        install confirm concurrently, exactly the hazard task-581 already
        fixed for skill-script). Either MOUNTS the card immediately
        (``session_id`` is the active/viewed session, or unknown -- legacy
        no-session callers keep the pre-TASK-910 always-mount behavior) or
        PARKS it (a different, background session -- the retained payload
        goes into ``_parked_skill_install_payloads`` for ``switch_session``/
        ``new_session``/``close_session`` to remount later, while the
        controller raises one sanitized app-wide notice for this exact
        stable decision id).

        Then polls re-checking this round's OWN cancel signal
        (``_is_session_cancelled``, scoped to ``session_id`` when known) and
        a deadline. Cancel/stop (of the OWNING session, or real process
        teardown via ``_shutdown_requested``), timeout, or no wired UI all
        resolve to DENY (fail-closed). A plain switch away no longer denies
        -- the round parks and stays alive until its own resolution,
        cancellation, or shutdown. Returns True only on an explicit Allow.

        Args:
            url: The skill source URL the model wants to install, surfaced
                verbatim on the confirm card for the user to inspect.
            session_id: The run's OWNING session (Task 3/9/TASK-910).
                ``None`` preserves the pre-Task-9 VIEWED-session/global-flag
                fallback (see ``_is_session_cancelled``) and never parks.

        Returns:
            True only on an explicit Allow; every other path (deny, cancel,
            stop, timeout, or no wired UI) returns False.
        """
        if self.read_controller_app() is None or (
            self.read_controller_set_pending_skill_install() is None
            and not self.has_retained_decision_target(session_id)
        ):
            return False
        event = self.read_global_threading().Event()
        decision: dict[str, bool] = {}
        request_id = str(self.read_global_uuid4()())
        owning_session_id = (
            session_id
            if session_id is not None
            else (self.read_controller_store().active_session_id or "")
        )
        round_cancel_event = self.read_controller__bind_round_cancel_signal()(
            session_id
        )
        visit_cancel_event = self.read_controller__bind_visit_cancel_signal()()
        owning_run_id = self.read_global_current_run_id()()
        install_round_state: dict[str, Any] = {
            "event": event,
            "decision": decision,
            "session_id": owning_session_id,
            "run_id": owning_run_id,
            "cancel_event": round_cancel_event,
            "visit_event": visit_cancel_event,
        }
        with self.lock:
            self.registries["skill_install"][request_id] = install_round_state
        timeout_seconds = (
            self.read_controller_skill_install_confirm_timeout_seconds()()
            if self.read_controller_skill_install_confirm_timeout_seconds() is not None
            else self.read_global__DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS()
        )
        # ADR-067: <= 0 arms NO deadline (the default) -- the round waits
        # for a decision or the owning run's cancellation.
        deadline = (
            self.read_global_time().monotonic() + timeout_seconds
            if session_id is None and timeout_seconds > 0
            else None
        )
        payload = {
            "url": url,
            "timeout_seconds": timeout_seconds,
            "request_id": request_id,
            "session_id": owning_session_id,
            "run_id": owning_run_id,
            "deadline_monotonic": deadline,
        }
        is_parked = session_id is not None and session_id != (
            self.read_controller_store().active_session_id or ""
        )
        # task-31384: register/badge/park-or-mount/poll/teardown is the
        # host's one lifecycle. Install is primary-agent-only and never
        # swept, so the revoked flag is not consulted.
        self.run_round(
            "skill_install",
            request_id,
            payload,
            install_round_state,
            session_id=session_id,
            owning_session_id=owning_session_id,
            deadline=deadline,
            is_parked=is_parked,
            human_wait_run_id=owning_run_id,
            check_revoked=False,
        )
        return bool(decision.get("allow", False))

    def request_skill_script_confirm(
        self, payload: dict[str, Any], *, session_id: str | None
    ) -> dict[str, bool]:
        """WORKER THREAD: ask the user to confirm running a skill's script.

        Mirrors request_skill_install_confirm, but carries a two-part decision:
        allow this run, and whether to remember the choice for this skill.

        Each call arms a fresh round under a newly-generated request id
        (embedded in the payload handed to the UI as ``"request_id"``) so
        that ``resolve_pending_skill_script`` can reject a decision left
        over from a prior, already-torn-down round -- see that method's
        docstring for why this matters.

        TASK-910: also carries the SAME park/mount/retain contract as
        ``request_mcp_approvals``/``request_skill_install_confirm`` -- see
        ``request_skill_install_confirm``'s docstring for the full
        mount-vs-park/retain rationale, identical here. The per-round
        registry (keyed by ``request_id``, task-581) now also stores this
        round's owning session id, so teardown can distinguish "another
        round for a DIFFERENT session is still armed" (must not suppress
        clearing THIS session's card) from "another round for the SAME
        session is still armed" (must not clear it out from under that
        sibling round, preserving task-581's original guarantee).

        Args:
            payload: Confirm details to render ({"skill_name", "script_path",
                "mechanism", "args", ...}); "timeout_seconds" and
                "request_id" keys are added before marshaling to the UI.
            session_id: The run's OWNING session (Task 3/9/TASK-910), scoping
                the cancel check (``_is_session_cancelled`` -- PA-T9 finding
                #1) and the park/mount decision. ``None`` preserves the
                pre-Task-9 VIEWED-session/global-flag fallback and never
                parks.

        Returns:
            ``{"allow": bool, "remember": bool}``. Every non-Allow path (deny,
            cancel, stop, timeout, no wired UI) returns ``allow=False``.
        """
        if self.read_controller_app() is None or (
            self.read_controller_set_pending_skill_script() is None
            and not self.has_retained_decision_target(session_id)
        ):
            return {"allow": False, "remember": False}
        event = self.read_global_threading().Event()
        decision: dict[str, bool] = {}
        request_id = str(self.read_global_uuid4()())
        owning_session_id = (
            session_id
            if session_id is not None
            else (self.read_controller_store().active_session_id or "")
        )
        round_cancel_event = self.read_controller__bind_round_cancel_signal()(
            session_id
        )
        visit_cancel_event = self.read_controller__bind_visit_cancel_signal()()
        owning_run_id = self.read_global_current_run_id()()
        script_round_state: dict[str, Any] = {
            "event": event,
            "decision": decision,
            "session_id": owning_session_id,
            "run_id": owning_run_id,
            "revoked": False,
            "cancel_event": round_cancel_event,
            "visit_event": visit_cancel_event,
        }
        if not self.register_round("skill_script", request_id, script_round_state):
            return {"allow": False, "remember": False}
        timeout_seconds = (
            self.read_controller_skill_script_confirm_timeout_seconds()()
            if self.read_controller_skill_script_confirm_timeout_seconds() is not None
            else self.read_global__DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS()
        )
        # ADR-067: <= 0 arms NO deadline (the default) -- the round waits
        # for a decision or the owning run's cancellation.
        deadline = (
            self.read_global_time().monotonic() + timeout_seconds
            if session_id is None and timeout_seconds > 0
            else None
        )
        card_payload = dict(payload)
        card_payload["timeout_seconds"] = timeout_seconds
        card_payload["request_id"] = request_id
        card_payload["session_id"] = owning_session_id
        card_payload["deadline_monotonic"] = deadline
        is_parked = session_id is not None and session_id != (
            self.read_controller_store().active_session_id or ""
        )
        # task-31384: one host lifecycle; a swept (revoked) round fails closed.
        outcome = self.run_round(
            "skill_script",
            request_id,
            card_payload,
            script_round_state,
            session_id=session_id,
            owning_session_id=owning_session_id,
            deadline=deadline,
            is_parked=is_parked,
            human_wait_run_id=owning_run_id,
        )
        if outcome == "revoked":
            return {"allow": False, "remember": False}
        return {
            "allow": bool(decision.get("allow", False)),
            "remember": bool(decision.get("remember", False)),
        }

    def request_user_questions(
        self, questions: list[dict[str, Any]], *, session_id: str | None
    ) -> dict[str, Any]:
        """WORKER THREAD: show ``questions`` on a card and wait for the answers.

        PRD Feature A (A5-A7, A9-A11, A14). Clones
        ``request_worktree_merge_confirm``'s round machinery -- fresh
        request id, park-or-mount under the TASK-910 contract, poll under
        ``use_human_input_wait`` so the owning run's tool clock pauses,
        cancel/deadline checks -- with a question-shaped decision. Two
        differences: a second call while this session already has a live
        round returns ``busy`` at once (A9: depth is expressed by batching
        questions, never by queueing rounds), and every outcome is recorded
        in the transcript on resolve (A14).

        Args:
            questions: Validated questions (``ask_user_questions.
                validate_questions`` output).
            session_id: The run's OWNING session; ``None`` never parks.

        Returns:
            ``{"answered": True, "answers": [...]}`` or ``{"answered":
            False, "reason": "timeout" | "cancelled" | "busy"}``.

        Raises:
            AskUserBusyRefusal: ``MAX_CONSECUTIVE_BUSY`` consecutive busy
                results in one run (A9's retry-loop ceiling).
        """
        from tldw_chatbook.Agents.ask_user_questions import (
            ASK_USER_REFUSAL_COPY,
            MAX_CONSECUTIVE_BUSY,
            AskUserBusyRefusal,
            answered_result,
            busy_result,
            empty_answers,
            unanswered_result,
        )
        from tldw_chatbook.Chat.console_agent_bridge import format_question_marker

        if self.read_controller_app() is None or (
            self.read_controller_set_pending_question() is None
            and not self.has_retained_decision_target(session_id)
        ):
            return unanswered_result("cancelled")
        owning_session_id = (
            session_id
            if session_id is not None
            else (self.read_controller_store().active_session_id or "")
        )
        owning_run_id = self.read_global_current_run_id()()
        event = self.read_global_threading().Event()
        decision: dict[str, Any] = {}
        request_id = str(self.read_global_uuid4()())
        round_state: dict[str, Any] = {
            "event": event,
            "decision": decision,
            "session_id": owning_session_id,
            "run_id": owning_run_id,
            "revoked": False,
        }
        # The live check and the registration share ONE critical section
        # (Qodo #2379): two sibling workers asking at once must not both
        # see "no live round" and both arm -- exactly one arms, the other
        # gets `busy`. The host re-registers the same state object at
        # run_round entry, which is idempotent.
        with self.lock:
            # Close publishes its fence and sweeps questions under this same
            # lock; no delayed worker may register after the sweep has passed.
            if owning_session_id in self.read_controller__session_close_generations():
                return unanswered_result("cancelled")
            live = any(
                state.get("session_id") == owning_session_id
                for state in self.registries["question"].values()
            )
            if live:
                bounces = (
                    self.read_controller__question_bounces().get(owning_run_id, 0) + 1
                )
                self.read_controller__question_bounces()[owning_run_id] = bounces
                while (
                    len(self.read_controller__question_bounces())
                    > self.read_global__MAX_TRACKED_QUESTION_BOUNCE_RUNS()
                ):
                    self.read_controller__question_bounces().pop(
                        next(iter(self.read_controller__question_bounces()))
                    )
            else:
                bounces = 0
                self.read_controller__question_bounces().pop(owning_run_id, None)
                self.registries["question"][request_id] = round_state
        if live:
            if bounces >= MAX_CONSECUTIVE_BUSY:
                raise AskUserBusyRefusal(ASK_USER_REFUSAL_COPY)
            return busy_result()
        round_state["cancel_event"] = self.read_controller__bind_round_cancel_signal()(
            session_id
        )
        round_state["visit_event"] = self.read_controller__bind_visit_cancel_signal()()
        timeout_seconds = self.read_controller__resolve_ask_user_timeout_seconds()()
        deadline = (
            self.read_global_time().monotonic() + timeout_seconds
            if timeout_seconds > 0
            else None
        )
        actor = self.read_global_current_run_actor()()
        asked_by = (
            "sub-agent" if actor is not None and actor.kind == "subagent" else "agent"
        )
        # task-31382: name WHICH sub-agent is asking when the run carries a label.
        asker_label = (
            actor.label if asked_by == "sub-agent" and actor is not None else None
        )
        card_payload: dict[str, Any] = {
            "questions": [dict(question) for question in questions],
            "asked_by": asked_by,
            "asker_label": asker_label,
            "timeout_seconds": timeout_seconds,
            "request_id": request_id,
            "session_id": owning_session_id,
            "deadline_monotonic": deadline,
        }
        is_parked = session_id is not None and session_id != (
            self.read_controller_store().active_session_id or ""
        )
        # task-31384: one host lifecycle; the outcome maps onto PRD A6.
        results: list[dict[str, Any]] = []

        def _on_outcome(outcome: str) -> None:
            # Runs before the host's teardown so the transcript marker
            # precedes the card clear and badge discard, as before.
            if outcome == "decided":
                answers = decision.get("answers")
                result = answered_result(
                    answers if answers else empty_answers(questions)
                )
            else:
                result = unanswered_result(
                    "timeout" if outcome == "timeout" else "cancelled"
                )
            results.append(result)
            bridge = self.read_controller__agent_bridge()
            if bridge is not None:
                with self.read_global_contextlib().suppress(Exception):
                    bridge.append_question_marker(
                        owning_session_id,
                        format_question_marker(
                            asked_by, questions, result, asker_label=asker_label
                        ),
                    )

        self.run_round(
            "question",
            request_id,
            card_payload,
            round_state,
            session_id=session_id,
            owning_session_id=owning_session_id,
            deadline=deadline,
            is_parked=is_parked,
            human_wait_run_id=owning_run_id,
            on_outcome=_on_outcome,
        )
        return results[0] if results else unanswered_result("cancelled")

    def request_worktree_merge_confirm(
        self,
        payload: dict[str, Any],
        *,
        session_id: str | None,
        operation_cancel_event: threading.Event | None,
    ) -> dict[str, bool]:
        """WORKER THREAD: ask the user to confirm merging/discarding an
        agent worktree before ``AgentService`` mutates anything.

        Clones ``request_skill_script_confirm``'s round machinery -- arm a
        fresh request id, park-or-mount under the same TASK-910 contract
        (mount when ``session_id`` is the active/viewed session or
        unknown, park a different background session's round for
        ``switch_session``/``new_session``/``close_session`` to remount
        later), poll under ``use_human_input_wait`` so the owning run's
        tool-call deadline pauses while the card is up, and fail closed on
        cancel/stop/timeout/no-UI -- with a single-key decision instead of
        that method's two-part one: worktree merge/discard has no
        "remember" concept, just Allow/Deny.

        ``merge_agent_worktree``/``discard_agent_worktree`` are wired
        PRIMARY-agent-only (``AgentService.run_turn``'s ``fleet_active``
        gate -- a worktree only ever exists for a fleet-launched CHILD,
        merged/discarded by the PRIMARY that launched it), so unlike the
        skill-script confirm this never needs
        ``revoke_approval_rounds_for_run``'s sweep: there is no separate
        "a child was abandoned but its session lives on" case to guard --
        the round's own ``_is_session_cancelled`` check already covers
        "the primary's turn stopped."

        Args:
            payload: Confirm details to render (``{"handle_id", "mode" |
                "action", "branch", "worktree", "diffstat"}``, built by
                the ``AgentService`` closures -- see
                ``merge_agent_worktree_tool``/``discard_agent_worktree_
                tool``); ``"timeout_seconds"``, ``"request_id"``,
                ``"session_id"``, and ``"deadline_monotonic"`` are added
                before marshaling to the UI.
            session_id: The run's OWNING session -- always the PRIMARY's,
                per the gate above. ``None`` preserves the legacy VIEWED-
                session/global-flag fallback and never parks.

        Returns:
            ``{"allow": bool}`` -- the exact shape ``AgentService``'s
            ``merge_agent_worktree_tool``/``discard_agent_worktree_tool``
            closures read via ``decision.get("allow", False)``. Every
            non-Allow path (deny, cancel, stop, timeout, no wired UI)
            returns ``allow=False``.
        """
        if self.read_controller_app() is None or (
            self.read_controller_set_pending_worktree_merge() is None
            and not self.has_retained_decision_target(session_id)
        ):
            return {"allow": False}
        event = self.read_global_threading().Event()
        decision: dict[str, bool] = {}
        request_id = str(self.read_global_uuid4()())
        owning_session_id = (
            session_id
            if session_id is not None
            else (self.read_controller_store().active_session_id or "")
        )
        if operation_cancel_event is not None and session_id is None:
            return {"allow": False}
        round_cancel_event = (
            operation_cancel_event
            if operation_cancel_event is not None
            else self.read_controller__bind_round_cancel_signal()(session_id)
        )
        visit_cancel_event = self.read_controller__bind_visit_cancel_signal()()
        owning_run_id = self.read_global_current_run_id()()
        merge_round_state: dict[str, Any] = {
            "event": event,
            "decision": decision,
            "session_id": owning_session_id,
            # The payload's run_id is the recovered child, not the requester.
            "run_id": owning_run_id or None,
            "cancel_event": round_cancel_event,
            "visit_event": visit_cancel_event,
        }
        with self.lock:
            self.registries["worktree_merge"][request_id] = merge_round_state
        timeout_seconds = (
            self.read_controller_worktree_merge_confirm_timeout_seconds()()
            if self.read_controller_worktree_merge_confirm_timeout_seconds() is not None
            else self.read_global__DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS()
        )
        deadline = (
            self.read_global_time().monotonic() + timeout_seconds
            if timeout_seconds > 0
            else None
        )
        card_payload = dict(payload)
        card_payload["timeout_seconds"] = timeout_seconds
        card_payload["request_id"] = request_id
        card_payload["session_id"] = owning_session_id
        card_payload["deadline_monotonic"] = deadline
        is_parked = session_id is not None and session_id != (
            self.read_controller_store().active_session_id or ""
        )
        # task-31384: one host lifecycle. Only an explicit decision allows;
        # cancel, stop, and timeout all fail closed. Primary-agent-only, so
        # never swept: the revoked flag is not consulted.
        outcome = self.run_round(
            "worktree_merge",
            request_id,
            card_payload,
            merge_round_state,
            session_id=session_id,
            owning_session_id=owning_session_id,
            deadline=deadline,
            is_parked=is_parked,
            human_wait_run_id=owning_run_id,
            check_revoked=False,
        )
        if outcome != "decided":
            return {"allow": False}
        return {"allow": bool(decision.get("allow", False))}

    def resolve_pending_approval(
        self, decisions: dict[str, str], *, round_id: str | None
    ) -> None:
        """UI THREAD: apply the user's batch decision, releasing the waiting worker thread.

        Called by ``ChatScreen``'s ``ChatApprovalCard.ApprovalDecided``
        handler, which forwards ``event.round_id`` -- the SAME id
        ``request_mcp_approvals`` stamped into the payload the card was
        built from (``ChatApprovalCard.set_batch`` stashes it;
        ``_submit_batch_decisions`` echoes it back on submit, mirroring
        ``resolve_pending_skill_script``'s identical ``request_id``
        round-trip).

        Fix round 1 (review CRITICAL finding): resolves ONLY the round
        whose id matches ``round_id`` -- never "whichever round belongs to
        the currently active session". ``ApprovalDecided`` travels as an
        async Textual message: a ``switch_session`` landing in the gap
        between the user's click and this handler running would otherwise
        let session A's decision resolve session B's completely different,
        unreviewed batch (or, for the same session, let a STALE decision
        from an already-ended round 1 resolve a newer round 2 that
        happened to arm before the stale message was delivered). A
        mismatched or stale ``round_id`` -- including one belonging to a
        round that already resolved and was popped -- is a safe no-op: the
        real round (if any) stays pending and its card re-derives
        unchanged on the next visit; nothing is ever auto-approved or
        denied-by-accident here.

        TASK-913 (AC#2): ``round_id=None`` no longer falls back to
        "whichever round belongs to the currently active session" -- it
        fails closed immediately, mirroring
        ``resolve_pending_skill_script``'s/``resolve_pending_skill_install``'s
        identical ``if request_id is None: return`` contract. Production
        (``ChatApprovalCard``/``ChatScreen``) has only ever had a single
        emitter (``ChatApprovalCard._submit_batch_decisions``) and it
        always threads the real ``round_id`` through; the active-session
        fallback existed only for legacy direct-call tests, which have
        been migrated to pass the real round id captured from the
        mounted/parked payload instead.

        A no-op both when ``round_id`` is ``None`` and when it doesn't
        match any currently-armed round (e.g. a stale message arriving
        after a timeout/cancellation already resolved and cleared it) --
        the real round (if any) stays pending and undecided; nothing is
        ever auto-approved or denied-by-accident here.

        NOTE: Snapshots the round's ``decisions``/``event`` into locals to
        avoid TOCTOU race: the worker thread's ``finally`` block pops the
        round entry out of ``_pending_approval_rounds`` concurrently. Guard
        and act only on the snapshots.

        Args:
            decisions: The user's per-``llm_name`` decision strings
                (``approve_once``/``approve_session``/``always_allow``/
                ``deny``) to merge into the round's shared decisions dict.
            round_id: The specific round to resolve (the id stamped onto
                the card the user actually decided). ``None`` (the
                default) never matches an armed round, so an un-migrated
                or malformed caller fails closed by omission.
        """
        # TASK-913 (AC#2): fail closed on a missing round_id rather than
        # scanning `_pending_approval_rounds.values()` for "whichever round
        # belongs to the active session" -- that active-session fallback
        # was production-unreachable (see docstring) and is now removed
        # entirely, taking its AC#1 lock-guarded-snapshot protection with
        # it (moot once the scan itself is gone). The remaining branch's
        # `.get()` read stays guarded: the worker thread's own registration
        # (`request_mcp_approvals`) and teardown (its `finally`) can mutate
        # this dict concurrently.
        if round_id is None:
            return
        with self.lock:
            round_state = self.registries["approval"].get(round_id)
            if round_state is None or round_state.get("settled"):
                return
            round_state["settled"] = True
            round_state["terminal_reason"] = "user"
            decisions_dict = round_state["decisions"]
            decisions_dict.update(decisions or {})
            if isinstance(decisions_dict, self.read_global_ApprovalDecisions()):
                answers = self.read_global_ApprovalDecisions()(
                    decisions or {},
                    denial_reasons=getattr(decisions, "denial_reasons", {}),
                )
                decisions_dict.denial_reasons = {
                    key: reason
                    for key, reason in answers.denial_reasons.items()
                    if key in round_state["names"]
                }
            approval_event = round_state["event"]
        approval_event.set()

    def resolve_pending_chat_create(
        self, allow: bool, remember: bool, request_id: str | None
    ) -> None:
        """UI THREAD: apply the user's decision, releasing the worker thread.

        ``request_id`` must be the exact ``"request_id"`` value the pending
        confirm's payload carried (``request_chat_create_confirm`` embeds a
        fresh one per round, and the confirm card built in a later task
        MUST echo it back here unchanged). This is a strict match: a
        resolve carrying no id, or an id from any round other than the one
        currently armed, is silently dropped rather than resolved.

        Same hazard class as ``resolve_pending_skill_script``: if round 1
        ends (deadline, cancel, stop, conversation switch) and the agent
        immediately issues a second fork_chat/new_chat call arming round 2,
        a ``Button.Pressed`` queued for round 1 just before its teardown
        could otherwise be handled after round 2 is armed -- resolving
        round 2 (a chat the user never saw) with round 1's stale click.
        Widget messages and ``call_from_thread`` calls are separate
        queues, so ordering across a round boundary is not guaranteed.

        Args:
            allow: True to create the chat this once.
            remember: True to also grant this tool standing permission in
                the owning session.
            request_id: The armed round's id, as echoed back by the UI.
                ``None`` (the default) never matches an armed round, so an
                un-migrated or malformed caller fails closed by omission.
        """
        if request_id is None:
            return
        with self.read_controller__pending_chat_create_lock():
            round_state = self.read_controller__pending_chat_create_rounds().get(
                request_id
            )
            if round_state is None:
                return
            round_state["decision"]["allow"] = bool(allow)
            round_state["decision"]["remember"] = bool(remember)
            round_state["event"].set()

    def resolve_pending_question(
        self, answers: list[dict[str, Any]], request_id: str | None
    ) -> None:
        """UI THREAD: hand the card's answers to the waiting worker thread.

        Strict ``request_id`` match, exactly like
        ``resolve_pending_skill_script``: a resolve with no id, or an id
        from any round but the armed one, is silently dropped. The answers
        are validated (``ask_user_questions.validate_answers``) before the
        worker sees them; a malformed list is dropped the same way.

        Args:
            answers: One PRD A6 answer dict per question, in order.
            request_id: The armed round's id as echoed back by the card.
        """
        from tldw_chatbook.Agents.ask_user_questions import (
            AskUserValidationError,
            validate_answers,
        )

        if request_id is None:
            return
        try:
            clean_answers = validate_answers(answers)
        except AskUserValidationError:
            return  # fail closed: a malformed resolve is dropped, never partially applied
        with self.lock:
            round_state = self.registries["question"].get(request_id)
        if round_state is None:
            return
        round_state["decision"]["answers"] = clean_answers
        round_state["event"].set()

    def resolve_pending_skill_install(
        self, allow: bool, *, request_id: str | None
    ) -> None:
        """UI THREAD: apply the user's Allow/Deny, releasing the worker thread.

        TASK-910: strict match against ``request_id``, mirroring
        ``resolve_pending_skill_script``'s identical contract -- a resolve
        carrying no id, or an id belonging to any round other than the one
        it names, is silently dropped rather than resolved. This closes the
        same stale-late-click hazard ``resolve_pending_skill_script``'s own
        docstring documents: once two sessions can each have their own
        concurrent install-confirm round (TASK-910 parking), "whichever
        round happens to be active" is no longer a safe fallback the way it
        was pre-TASK-910 (a single global slot could only ever have one
        candidate).

        Args:
            allow: True to allow the pending install, False to deny it.
            request_id: The armed round's id, as echoed back by the UI
                (``SkillInstallConfirmCard.InstallDecided.request_id``).
                ``None`` (the default) never matches an armed round, so an
                un-migrated or malformed caller fails closed by omission.
        """
        if request_id is None:
            return
        with self.lock:
            round_state = self.registries["skill_install"].get(request_id)
            if round_state is None or round_state.get("settled"):
                return
            round_state["settled"] = True
            round_state["terminal_reason"] = "user"
            round_state["decision"]["allow"] = bool(allow)
            event = round_state["event"]
        event.set()

    def resolve_pending_skill_script(
        self, allow: bool, remember: bool, request_id: str | None
    ) -> None:
        """UI THREAD: apply the user's decision, releasing the worker thread.

        ``request_id`` must be the exact ``"request_id"`` value the pending
        confirm's payload carried (``request_skill_script_confirm`` embeds
        a fresh one per round, and the confirm card built in a later task
        MUST echo it back here unchanged). This is a strict match: a
        resolve carrying no id, or an id from any round other than the one
        currently armed, is silently dropped rather than resolved.

        This guards against a real arbitrary-code-execution hazard: if
        round 1 ends (deadline, cancel, stop, conversation switch) and the
        agent immediately issues a second ``run_skill_script`` call
        arming round 2, a ``Button.Pressed`` queued for round 1 just
        before its teardown could otherwise be handled after round 2 is
        armed -- resolving round 2 (a script the user never saw) with
        round 1's stale click. Widget messages and ``call_from_thread``
        calls are separate queues, so ordering across a round boundary is
        not guaranteed.

        Args:
            allow: True to run the script this once.
            remember: True to also grant this skill standing permission.
            request_id: The armed round's id, as echoed back by the UI.
                ``None`` (the default) never matches an armed round, so an
                un-migrated or malformed caller fails closed by omission.
        """
        if request_id is None:
            return
        with self.lock:
            round_state = self.registries["skill_script"].get(request_id)
            if round_state is None or round_state.get("settled"):
                return
            round_state["settled"] = True
            round_state["terminal_reason"] = "user"
            round_state["decision"]["allow"] = bool(allow)
            round_state["decision"]["remember"] = bool(remember)
            event = round_state["event"]
        event.set()

    def resolve_pending_worktree_merge(
        self, allow: bool, *, request_id: str | None
    ) -> None:
        """UI THREAD: apply the user's Allow/Deny, releasing the worker thread.

        Strict ``request_id`` match, mirroring
        ``resolve_pending_skill_script`` -- see that method's docstring
        for why a resolve carrying no id, or an id belonging to any round
        other than the one it names, is silently dropped.

        Args:
            allow: True to allow the pending merge/discard, False to deny.
            request_id: The armed round's id, as echoed back by the UI.
                ``None`` (the default) never matches an armed round.
        """
        if request_id is None:
            return
        with self.lock:
            round_state = self.registries["worktree_merge"].get(request_id)
            if round_state is None or round_state["event"].is_set():
                return
            round_state["decision"]["allow"] = bool(allow)
            round_state["event"].set()

    def revoke_approval_rounds_for_run(self, run_id: str) -> int:
        """Fail every approval round owned by ``run_id`` closed, right now.

        PR2a Task 7 (safety). The approval wait blocks inside
        ``_call_with_timeout``'s per-call daemon thread, which keeps
        running after the fleet cooperatively cancels -- or outright
        ABANDONS -- the child that owns it. Until this existed, that
        child's card stayed on screen and stayed live: pressing Approve
        resolved the round, the waiting thread returned the approval, and
        the tool EXECUTED FOR REAL (a file written, a message sent) for a
        run whose handle and run row already read ``cancelled``. The
        documented ``approval_timeout < max_tool_call_seconds`` invariant
        (see ``_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS``) bounds the same
        class of hazard for the timeout path; this closes the
        cancellation path.

        Called by ``AgentService`` (through its injected
        ``revoke_approvals`` seam) at both moments a child stops being
        allowed to act: the cooperative cancel and the end-of-turn
        abandon. Safe to call for a run that never armed a card -- the
        common case -- and never touches another run's rounds, which
        matters because every child of a fleet turn shares ONE console
        session: session-keyed teardown could not tell a cancelled child's
        card from its live sibling's.

        Covers tool-call approvals, run_skill_script confirms, and ask_user
        questions. Skill-install and worktree-merge confirms are primary-only
        and are not swept. The host also fences future arms for the revoked
        run and these kinds for its lifetime, even when no round exists yet.

        Each revoked round is (a) marked ``revoked`` so the waiting thread
        fails closed even if a click lands in its shared decision box
        afterwards, (b) pre-filled with the closed verdict, (c) removed
        from its registry, so a late ``resolve_pending_approval``/
        ``resolve_pending_skill_script`` finds nothing to resolve, (d)
        released via its Event, so the waiting thread returns immediately
        rather than at its auto-deny deadline, (e) discarded from
        ``_pending_approvals`` so the session's NEEDS_APPROVAL badge
        clears once its last round is gone, and (f) taken off screen
        through the SAME FIFO-head re-derive (``_remount_head``) that the
        round's own teardown uses, so a sibling round's card is never
        clobbered.

        Thread-safe. InterruptRoundHost records the per-kind revocation
        fences and sweeps the registries under one shared non-reentrant lock.
        Exact-round payload cleanup and badge/UI callbacks run after that
        critical section; callbacks may acquire the same lock themselves.

        Args:
            run_id: The cancelled/abandoned run whose cards must die. A
                falsy id is a no-op -- ``""`` is the "no run bound" key
                that rounds armed outside any agent run carry, and
                sweeping those would deny cards no run owns.

        Returns:
            How many existing rounds were revoked across the swept kinds
            (``0`` when the run had none; the late-arm fence still persists).
        """
        if not run_id:
            return 0
        # task-31384: one host sweep, parameterised by each kind's stamp;
        # the badge discard and the FIFO-head re-derive per affected
        # session stay here, exactly as the per-kind sweeps did them.
        swept = self.revoke_for_run(run_id, self.read_global__REVOCATION_STAMPS())
        chat_create_swept = self.read_controller__revoke_chat_create_rounds()(run_id)
        with self.lock:
            self.read_controller__question_bounces().pop(run_id, None)
        total = 0
        for kind, rounds in swept.items():
            for round_id, session_id in rounds:
                total += 1
                if session_id is not None:
                    self.read_controller_discard_pending_round()(session_id, round_id)
                with self.read_global_contextlib().suppress(Exception):
                    self.remount_head(kind, session_id)
        total += len(chat_create_swept)
        if total:
            self.read_global_logger().info(
                "Revoked pending approval rounds for cancelled run"
            )
        return total

    def revoke_raw_shell_authority(self) -> int:
        """Fail closed only raw-shell stamps and approval rounds on disarm."""
        from tldw_chatbook.Agents.raw_shell_tool_provider import (
            RAW_SHELL_SERVER_KEY,
            RAW_SHELL_TOOL_NAME,
        )

        providers = self.read_controller__raw_shell_providers()
        for provider in tuple(providers):
            provider.revoke_approval_stamps()

        revoked: list[tuple[str, str | None]] = []
        with self.lock:
            for round_id, state in list(self.registries["approval"].items()):
                calls = tuple(state.get("calls") or ())
                if not calls or not all(
                    call.server_key == RAW_SHELL_SERVER_KEY
                    and call.tool_name == RAW_SHELL_TOOL_NAME
                    for call in calls
                ):
                    continue
                state["revoked"] = True
                decisions = state.get("decisions")
                if isinstance(decisions, dict):
                    for name in state.get("names") or ():
                        decisions[name] = "deny"
                self.registries["approval"].pop(round_id, None)
                session_id = state.get("session_id") or None
                revoked.append((round_id, session_id))
                event = state.get("event")
                if event is not None:
                    event.set()

        for round_id, session_id in revoked:
            self.read_controller__unpark_round_payload()(
                self.payloads["approval"], round_id
            )
            if session_id is not None:
                self.read_controller_discard_pending_round()(session_id, round_id)
            try:
                self.read_controller__remount_head()(
                    self.payloads["approval"],
                    self.read_controller_set_pending_approval(),
                    session_id,
                )
            except Exception:  # noqa: BLE001 -- revocation must continue
                self.read_global_logger().debug(
                    "Failed to remount after raw shell revocation"
                )
        if revoked:
            self.read_global_logger().info(
                "Revoked pending raw shell approval rounds on disarm"
            )
        return len(revoked)

    def set_answerable_decision(self, session_id: str, decision_id: str | None) -> bool:
        """Update Console's claim without erasing another visible owner's claim."""
        with self.lock:
            self.decision_view_revision += 1
            self.read_controller__console_answerable_decision_by_session().pop(
                session_id, None
            )
            if (
                decision_id is not None
                and session_id == self.read_controller_store().active_session_id
            ):
                self.read_controller__console_answerable_decision_by_session()[
                    session_id
                ] = decision_id
        current_id = self.read_controller__refresh_answerable_decision()(session_id)
        return decision_id is None or (
            session_id == self.read_controller_store().active_session_id
            and current_id == decision_id
        )

    def set_run_pending_approval(self, session_id: str, pending: bool) -> None:
        """DEPRECATED boolean shim -- prefer ``add_pending_round``/``discard_pending_round``.

        Parallel-agents spec §6 (Task 7 stores/exposes the flag; Task 9
        wired the approval paths -- MCP batch approvals, skill-install/
        script confirms -- that originally called this). TASK-1050 (Defect
        A) migrated all three bridges to the round-keyed ``add_pending_
        round``/``discard_pending_round`` instead, since a plain boolean
        cannot represent "N independent rounds outstanding for one
        session" without one clobbering another's clear.

        This shim survives for the ONE remaining caller genuinely without a
        round id of its own: ``ChatScreen._park_console_approval`` (wired
        as ``park_pending_approval``), whose own public contract is a
        single-arg ``Callable[[str], None]`` with no room for a round id --
        changing that would ripple into every test that wires ``park_
        pending_approval = some_list.append`` -- and it is ALSO used
        directly, standalone, by tests exercising the marker/badge
        lifecycle without a live round (mirrors how those tests already
        drive other controller seams directly).

        Internally represented as the reserved
        ``_LEGACY_PENDING_APPROVAL_ROUND_ID`` sentinel round id, so it
        composes safely alongside real round ids in the same per-session
        set -- ``pending=True`` adds the sentinel, ``pending=False``
        discards ONLY the sentinel (a real round registered separately via
        ``add_pending_round`` is untouched either way). Because of this, a
        caller that calls this with ``pending=True`` while a REAL round is
        ALREADY registered for the session adds a harmless, redundant
        no-op-visible entry -- but that same caller must not rely on this
        call's own ``pending=False`` (or a real round's ``discard_pending_
        round``) to fully clear the badge on its own; whichever one runs
        last is the one that actually clears it. ``ChatScreen._park_
        console_approval`` avoids this ambiguity by checking ``has_
        pending_approval_round`` first and only falling back to this shim
        when no real round is registered yet.

        Args:
            session_id: The session whose pending-approval flag to update.
            pending: ``True`` to mark the session as awaiting a decision,
                ``False`` to clear it.
        """
        if pending:
            self.read_controller_add_pending_round()(
                session_id, self.read_global__LEGACY_PENDING_APPROVAL_ROUND_ID()
            )
        else:
            self.read_controller_discard_pending_round()(
                session_id, self.read_global__LEGACY_PENDING_APPROVAL_ROUND_ID()
            )

    @property
    def worktree_confirmation_enabled(self) -> bool:
        """Only the real disposable worktree surface enables new tool disclosure."""
        return (
            self.read_controller_app() is not None
            and self.read_controller_set_pending_worktree_merge() is not None
        )


def _approval_decision_fact(
    decision: object, *, unanswered: bool
) -> ApprovalDecision | None:
    if unanswered:
        return None
    if decision == "deny":
        return "denied"
    if isinstance(decision, str) and decision in {
        "approve_once",
        "approve_session",
        "always_allow",
    }:
        return "approved"
    return None


def _build_approval_payload(
    round_id: str,
    session_id: str,
    run_id: str,
    pending: "list[MCPPendingCall]",
    timeout_seconds: float,
    deadline: float | None,
    *,
    read_global_ToolExecutionPolicy: Callable[[], Any],
) -> dict[str, Any]:
    """Marshal one approval round's card payload.

    ADR-090: rows carry ``rationale`` (the model's advisory context) and
    ``description`` (the tool definition's own text, for the external
    summarizer); the payload carries a ``summary`` slot that starts ``None``
    and is filled by the advisory summarizer -- payload-carried so any
    remount re-renders it rather than depending on a live patch surviving.
    """
    return {
        "round_id": round_id,
        "session_id": session_id,
        "run_id": run_id,
        "calls": [
            {
                "llm_name": call.llm_name,
                "server_key": call.server_key,
                "tool_name": call.tool_name,
                "server_label": call.server_label,
                "arguments": dict(call.arguments or {}),
                "reason": call.reason,
                "options": list(call.options),
                "effects": list(call.effects),
                "execution_policy": (
                    call.execution_policy.value
                    if isinstance(
                        call.execution_policy, read_global_ToolExecutionPolicy()
                    )
                    else read_global_ToolExecutionPolicy().BOUNDED_ABANDONABLE.value
                ),
                "path_precheck_failed": call.path_precheck_failed,
                "call_id": call.call_id,
                "full_command": call.full_command,
                "warning": call.warning,
                "scope_notice": call.scope_notice,
                "rationale": str(getattr(call, "rationale", "") or ""),
                "description": str(getattr(call, "description", "") or ""),
            }
            for call in pending
        ],
        "timeout_seconds": timeout_seconds,
        "deadline_monotonic": deadline,
        "summary": None,
    }


def _collect_mcp_pending(
    provider: MCPToolProvider, calls: list["ToolCall"]
) -> list["MCPPendingCall"]:
    """Resolve each call's MCP gate; return the subset that needs asking.

    Extracted so `build_mcp_review_hook` (MCP-only, still used directly by
    its own long-standing tests) and `build_tool_review_hook` (T6: the
    run-level hook that folds built-ins in too) share this ONE walk over
    `provider.pending_gate_for` rather than one copying the other's body.
    `None` per call means either "not an MCP call this provider owns" or
    "an MCP call whose current state doesn't need asking" -- see
    `pending_gate_for`'s own docstring for why callers do not need to
    distinguish those two cases.
    """
    pending: list["MCPPendingCall"] = []
    for call in calls:
        gate = provider.pending_gate_for(
            call.name,
            call.args,
            str(getattr(call, "call_id", "") or ""),
            rationale=str(getattr(call, "rationale", "") or ""),
        )
        if gate is not None:
            pending.append(gate)
    return pending


def _review_decision(
    row: MCPPendingCall,
    decisions: Mapping[str, str],
    verdict: str,
    *,
    allowing: tuple[str, ...],
    name_fallback: bool,
    read_global_ToolReviewDecision: Callable[[], Any],
    read_global__approval_decision_fact: Callable[[], Any],
    read_global_append_denial_reason: Callable[[], Any],
    read_global_approval_key_unanswered: Callable[[], Any],
    read_global_selected_approval_key: Callable[[], Any],
) -> ToolReviewDecision:
    """Attach an answered raw choice to the owner's unchanged verdict."""
    key = (
        read_global_selected_approval_key()(decisions, row.call_id, row.llm_name)
        if name_fallback
        else row.call_id or row.llm_name
    )
    decision = decisions.get(key)
    unanswered = read_global_approval_key_unanswered()(decisions, key)
    fact = read_global__approval_decision_fact()(decision, unanswered=unanswered)
    if fact == "approved" and decision not in allowing:
        fact = None
    if decision == "allow_matching" and decision in allowing and not unanswered:
        fact = "approved"
    return read_global_ToolReviewDecision()(
        read_global_append_denial_reason()(verdict, decisions, key), fact
    )


def _sibling_approval_refusals(
    rows: Sequence[MCPPendingCall],
    decision_for: Callable[[MCPPendingCall], str | None],
    decisions: Mapping[str, str],
    allowing_for: Callable[[MCPPendingCall], tuple[str, ...]],
    record_refusal: Callable[[MCPPendingCall, bool], None],
    *,
    read_global_TIMEOUT_REFUSAL: Callable[[], Any],
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_UNRESOLVED_REFUSAL: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
) -> dict[str, ToolReviewValue]:
    """Refuse rows that would run only on a same-name sibling's approval.

    TASK-33082. A tool's stamp is name-keyed and keeps the broadest approval
    any row of that name received, so it cannot say "this call, not that
    one". A row whose own answer is missing, ``"timeout"`` or unknown would
    then run on its approved sibling's stamp. Each such row is refused here,
    by its own key, and audited through ``record_refusal``, because the
    runtime never dispatches it to the owner that would otherwise record the
    outcome. A row with no approved sibling is left alone: its name's stamp
    is not an approval, so the owner refuses and audits it at dispatch, as
    before.

    Args:
        rows: The batch's pending approval rows.
        decision_for: Resolves one row's own answer (call id first, then name).
        decisions: The approval round's answers, for the review fact.
        allowing_for: The answers that approve a given row's owner.
        record_refusal: Audits one refused row; the flag is whether its own
            answer was ``"timeout"``.

    Returns:
        Refusal verdicts keyed by call id, or by name for an id-less row.
    """
    approved = {row.llm_name for row in rows if decision_for(row) in allowing_for(row)}
    refusals: dict[str, read_global_ToolReviewValue()] = {}
    for row in rows:
        decision = decision_for(row)
        if row.llm_name not in approved or decision == "deny":
            continue
        if decision in allowing_for(row):
            continue
        timed_out = decision == "timeout"
        refusals[row.call_id or row.llm_name] = read_global__review_decision()(
            row,
            decisions,
            read_global_TIMEOUT_REFUSAL()
            if timed_out
            else read_global_UNRESOLVED_REFUSAL(),
        )
        record_refusal(row, timed_out)
    return refusals


def _stamp_answer_provenance(
    stamps: dict[str, str],
    rows: Sequence[MCPPendingCall],
    decisions: Mapping[str, str],
    *,
    read_global_ApprovalDecisions: Callable[[], Any],
    read_global_approval_was_unanswered: Callable[[], Any],
    read_global_selected_approval_key: Callable[[], Any],
) -> ApprovalDecisions:
    """Keep unresolved denies attached to the selected name-scoped stamp."""
    result = read_global_ApprovalDecisions()(stamps)
    result.unresolved_keys = frozenset(
        row.llm_name
        for row in rows
        if stamps.get(row.llm_name) == "deny"
        and decisions.get(
            read_global_selected_approval_key()(decisions, row.call_id, row.llm_name)
        )
        == "deny"
        and read_global_approval_was_unanswered()(row, decisions)
    )
    return result


def _stamp_approval_round_closed(state: dict[str, Any]) -> None:
    """Revocation stamp for an approval round: deny every undecided key."""
    decisions = state.get("decisions")
    if isinstance(decisions, dict):
        for name in state.get("names") or ():
            decisions[name] = "deny"


def _stamp_question_round_closed(state: dict[str, Any]) -> None:
    """Revocation stamp for a question round: the revoked flag says it all."""


def _stamp_skill_script_round_closed(state: dict[str, Any]) -> None:
    """Revocation stamp for a skill-script round: allow/remember both off."""
    decision = state.get("decision")
    if isinstance(decision, dict):
        decision["allow"] = False
        decision["remember"] = False


def approval_was_unanswered(
    row: "MCPPendingCall",
    decisions: Mapping[str, str],
    *,
    read_global_approval_key_unanswered: Callable[[], Any],
    read_global_selected_approval_key: Callable[[], Any],
) -> bool:
    """True when ``row``'s deny came from a Stop/revoke, not from the user.

    Args:
        row: The pending call whose verdict is being recorded.
        decisions: The map ``request_mcp_approvals`` returned -- an
            `ApprovalDecisions` in production, a bare dict in tests and in
            any other `request_approvals` shape (which then reports
            "answered", the pre-fix behaviour).

    Returns:
        Whether the verdict for ``row`` was defaulted by an unresolved round.
    """
    key = read_global_selected_approval_key()(
        decisions, str(getattr(row, "call_id", "") or ""), row.llm_name
    )
    return read_global_approval_key_unanswered()(decisions, key)


def build_combined_review_hook(
    hooks: list[Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_logger: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Fan one batch through every provider's hook; merge verdict maps.

    Each hook gates only the calls its provider owns (pending_gate_for
    returns None for foreign tools), so merging is collision-free --
    except when two providers own the SAME name (not possible today:
    local names carry fs_/web_/todo_ prefixes, virtual CLI owns only
    virtual_cli, and MCP names carry mcp__*), where
    the later hook's "proceed" would simply win; both stamps are still
    applied by each provider's own hook regardless.

    I3 across providers: every hook runs even when an earlier one RAISES.
    `run_agent_loop` fails the batch OPEN on hook exception
    (agent_runtime.py:367-376), and each hook's clear-first stamp wipe is
    the only thing standing between a stale prior-turn stamp and the
    fail-open runtime handing it to `invoke()`. A naive sequential loop
    would let one hook's raising approval round trip (the documented I3
    mid-shutdown case) skip every LATER hook -- including its entry clear
    -- stranding that provider's stale stamp. So each hook is invoked
    under its own try/except and the FIRST exception is re-raised after
    all hooks have run: every provider gets its clear (and, when its own
    round trip succeeds, its fresh this-turn decisions), and the runtime
    still sees the raise and applies its fail-open policy against stamps
    that are guaranteed non-stale.

    Args:
        hooks: The per-provider review hooks to fan each batch through,
            in application order.

    Returns:
        A `review_tool_calls`-shaped callable that merges every hook's
        verdict map into one.
    """

    def review_tool_calls(
        calls: list["ToolCall"], run_id: str
    ) -> dict[str, read_global_ToolReviewValue()]:
        verdicts: dict[str, read_global_ToolReviewValue()] = {}
        first_exc: Exception | None = None
        for hook in hooks:
            try:
                verdicts.update(hook(calls, run_id))
            except Exception as exc:  # noqa: BLE001 -- re-raised after ALL hooks ran
                read_global_logger().opt(exception=True).warning(
                    "combined review_tool_calls: a provider hook raised; "
                    "running remaining hooks so their entry clears still fire"
                )
                if first_exc is None:
                    first_exc = exc
        if first_exc is not None:
            raise first_exc
        return verdicts

    return review_tool_calls


def build_local_review_hook(
    provider: "LocalToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_USER_DENIED_REFUSAL: Callable[[], Any],
    read_global__APPROVAL_SCOPE_RANK: Callable[[], Any],
    read_global__APPROVING_DECISIONS: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
    read_global__sibling_approval_refusals: Callable[[], Any],
    read_global__stamp_answer_provenance: Callable[[], Any],
    read_global_approval_was_unanswered: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Build this run's review_tool_calls hook for the local provider.

    Identical discipline to build_mcp_review_hook (see its docstring for
    the full rationale -- every binding point applies unchanged here):
    clear-first stamps at entry (I3: a raising approval round trip must
    never leave a stale prior-turn stamp live for the fail-open runtime
    to hand to `invoke()`), and exactly ONE approval round trip per batch.
    Name-keyed stamps keep the widest approved scope because the provider
    gate is tool-scoped; per-call refusal verdicts stop only the denied
    sibling before dispatch. Calls the provider doesn't own resolve
    `None` from `pending_gate_for` and never enter the batch.

    Args:
        provider: This run's already-composed `LocalToolProvider` (built
            by `_compose_local_provider` on the main loop before the
            run's worker thread starts).
        request_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run -- the same
            approval-card bridge the MCP hook uses; it consumes
            `MCPPendingCall` payloads regardless of origin.

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """

    def review_tool_calls(
        calls: list["ToolCall"], run_id: str
    ) -> dict[str, read_global_ToolReviewValue()]:
        # I3: clear THIS turn's stamps FIRST -- see build_mcp_review_hook.
        # PR2a Task 5: scoped to `run_id`, so the clear cannot reach a
        # concurrent sibling run's live verdicts.
        provider.apply_batch_decisions(run_id, {})
        pending: list["MCPPendingCall"] = []
        for call in calls:
            gate = provider.pending_gate_for(
                call.name,
                call.args,
                str(getattr(call, "call_id", "") or ""),
                # Qodo review #10: the local owner must receive the call's
                # advisory rationale like the MCP and builtin owners do, or
                # every local approval row renders without model context.
                rationale=str(getattr(call, "rationale", "") or ""),
                run_id=run_id,
            )
            if gate is not None:
                pending.append(gate)
        if not pending:
            return {}
        decisions = request_approvals(pending)

        def _decision_for(row: "MCPPendingCall") -> str | None:
            key = str(getattr(row, "call_id", "") or "")
            if key and key in decisions:
                return decisions[key]
            return decisions.get(row.llm_name)

        approvals: dict[str, str] = {}
        denied: set[str] = set()
        for row in pending:
            decision = _decision_for(row)
            if decision is None:
                continue
            if decision == "deny":
                denied.add(row.llm_name)
                continue
            current = approvals.get(row.llm_name)
            if current is None or read_global__APPROVAL_SCOPE_RANK().get(
                decision, 0
            ) > read_global__APPROVAL_SCOPE_RANK().get(current, 0):
                approvals[row.llm_name] = decision
        stamps = dict(approvals)
        for name in denied:
            stamps.setdefault(name, "deny")
        provider.apply_batch_decisions(
            run_id, read_global__stamp_answer_provenance()(stamps, pending, decisions)
        )
        reviewed_call_ids = {row.call_id for row in pending if row.call_id}
        provider.apply_promotion_decisions(
            run_id,
            [call for call in calls if call.call_id in reviewed_call_ids],
            decisions,
        )

        # task-32280 fix round (Critical): mirrors the MCP hook's own
        # record_user_denial call a few hundred lines up. `run_agent_loop`
        # turns any non-"proceed" verdict straight into the call's result
        # and skips dispatch entirely, so `LocalToolProvider.invoke_detailed`
        # -- the only thing that otherwise records a local refusal -- never
        # runs for a hook-level denied call. Record at the point the denial
        # becomes final, through the provider's own audit seam.
        verdicts: dict[str, read_global_ToolReviewValue()] = {
            row.llm_name: "proceed" for row in pending
        }
        for row in pending:
            if _decision_for(row) != "deny":
                continue
            # R23: as in the MCP hook -- a deny the user never chose (Stop
            # mid-card) is already logged as `denied-unresolved`; only the
            # REFUSAL below applies to it, not a second audit row.
            if not read_global_approval_was_unanswered()(row, decisions):
                provider.record_user_denial(row.llm_name)
            key = str(getattr(row, "call_id", "") or "") or row.llm_name
            verdicts[key] = read_global__review_decision()(
                row,
                decisions,
                read_global_USER_DENIED_REFUSAL().format(name=row.llm_name),
            )
        verdicts.update(
            read_global__sibling_approval_refusals()(
                pending,
                _decision_for,
                decisions,
                lambda _row: read_global__APPROVING_DECISIONS(),
                lambda row, timed_out: provider.record_hook_refusal(
                    row.llm_name, timed_out=timed_out
                ),
            )
        )
        # Preserve settled name-wide refusal fallback before adding facts.
        verdicts.update(
            {
                row.call_id or row.llm_name: read_global__review_decision()(
                    row, decisions, "proceed"
                )
                for row in pending
                if verdicts.get(row.call_id or row.llm_name, verdicts[row.llm_name])
                == "proceed"
            }
        )
        return verdicts

    return review_tool_calls


def build_managed_skill_promotion_review_hook(
    gate: "ManagedSkillProposalGate",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_USER_DENIED_REFUSAL: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Build the primary-only approval hook for read-only skill proposals."""

    def review_tool_calls(
        calls: list["ToolCall"], run_id: str
    ) -> dict[str, read_global_ToolReviewValue()]:
        gate.clear(run_id)
        pending = [
            row
            for call in calls
            if (
                row := gate.pending_gate_for(
                    call.name,
                    call.args,
                    run_id=run_id,
                    call_id=call.call_id,
                )
            )
            is not None
        ]
        if not pending:
            return {}
        decisions = request_approvals(pending)
        reviewed_call_ids = {row.call_id for row in pending if row.call_id}
        gate.apply_decisions(
            run_id,
            [call for call in calls if call.call_id in reviewed_call_ids],
            decisions,
        )
        verdicts: dict[str, read_global_ToolReviewValue()] = {}
        for row in pending:
            key = row.call_id or row.llm_name
            verdicts[key] = read_global__review_decision()(
                row,
                decisions,
                "proceed"
                if decisions.get(key) == "approve_once"
                else read_global_USER_DENIED_REFUSAL().format(name=row.llm_name),
                allowing=("approve_once",),
                name_fallback=False,
            )
        return verdicts

    return review_tool_calls


def build_mcp_review_hook(
    provider: MCPToolProvider,
    request_mcp_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global__collect_mcp_pending: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Build this run's T4 `review_tool_calls` hook for one composed MCP provider.

    Handed to `ConsoleAgentBridge.run_reply` (P5-T6), which forwards it
    straight through to `AgentService`/`LoopDeps.review_tool_calls` (T4):
    called ONCE per turn with the full batch of tool calls about to be
    dispatched, before any of them is invoked.

    For every call in the batch, `provider.pending_gate_for(name, args)`
    resolves whether it needs human gating (`None` for both "not an MCP
    call this provider owns" and "an MCP call whose current state doesn't
    need asking" -- `invoke()` re-resolves either case for itself, so
    this hook does not need to distinguish them). When at least one call
    needs asking, this makes exactly ONE `request_mcp_approvals` round
    trip for the whole batch (never one per call) and hands the resulting
    decisions to `provider.apply_batch_decisions` -- a per-turn stamp
    every same-named call `invoke()` makes THIS turn peeks (Finding F1:
    never popped, so two calls to the same tool in one batch both see the
    approval, not just the first).

    Finding F1 also requires this hook to call
    `provider.apply_batch_decisions` on EVERY invocation, even when
    `pending` ends up empty (a turn whose calls are all non-MCP, or all
    already resolved without asking) -- passing `{}` in that case.
    `apply_batch_decisions` REPLACES the stamp set rather than merging, so
    this is what guarantees a stamp from an earlier turn can never survive
    into a later one and be misread as this turn's verdict for a
    repeated tool name.

    I3 (probe-verified): that clear happens at hook ENTRY, before
    `pending_gate_for` is even resolved and before the
    `request_mcp_approvals` round trip -- not only after a successful one.
    `request_mcp_approvals` can raise (e.g. the unguarded
    `_marshal_pending_approval` call mid-shutdown); `run_agent_loop`'s own
    hook-exception handling fails the WHOLE batch open (treats every call
    in it as `"proceed"`) when that happens. If the clear only ran after a
    successful round trip, a raise would leave THIS turn's stamp set
    exactly as the PREVIOUS turn left it -- so the fail-open runtime would
    hand `invoke()` a stale prior-turn stamp (e.g. a real `"approve_once"`)
    for a call the user never decided on this turn. Clearing first means a
    raised round trip always leaves `invoke()` with no stamp to peek,
    falling through to its own fresh gate -- which fails closed for an
    `"ask"` tool with no approval_callback wired.

    Design choice (binding, per the Phase-5 plan): this hook never
    returns a refusal string itself. Every MCP call it stamped is left to
    resolve through `invoke()`'s own gate on dispatch -- `invoke()`
    already handles every decision string uniformly (`approve_once`/
    `approve_session`/`always_allow` execute; `deny`/`timeout` refuse with
    the exact model-facing copy AND record the audit decision), so
    routing every decision through that ONE place keeps the refusal copy
    and the audit trail single-sourced instead of duplicating that logic
    here. The verdict map this hook returns therefore only ever contains
    `"proceed"` entries (for calls it gated this turn) -- purely
    documentary, since `run_agent_loop` already treats any name this hook
    doesn't mention as `"proceed"` by default; returning `{}` when nothing
    needed gating is exactly as correct as omitting entries would be.
    Non-MCP calls are untouched either way: `pending_gate_for` returns
    `None` for any name the provider doesn't own, so they never enter
    `pending` and are never mentioned in the returned map.

    Args:
        provider: This run's already-composed `MCPToolProvider` (P5-T6:
            built and `compose_catalog()`-ed by the caller on the main
            loop before the run's worker thread starts).
        request_mcp_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run -- runs on the
            agent bridge's worker thread and blocks until the batch is
            decided, cancelled, or times out (T5).

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """

    def review_tool_calls(
        calls: list["ToolCall"], run_id: str
    ) -> dict[str, read_global_ToolReviewValue()]:
        # I3: clear THIS turn's stamps FIRST, before pending_gate_for/the
        # approval round trip even run -- subsumes the `if not pending`
        # branch's own clear below (every invocation of this hook clears,
        # unconditionally). See this function's own docstring for why the
        # clear must happen at entry, not only after a successful round
        # trip: a raising `request_mcp_approvals` must never leave a stale
        # prior-turn stamp live for the fail-open runtime to hand straight
        # to `invoke()`. PR2a Task 5: that clear is scoped to `run_id` --
        # it still wipes THIS run's prior turn, and no longer wipes a
        # concurrent sibling's live verdicts.
        provider.apply_batch_decisions(run_id, {})
        pending = read_global__collect_mcp_pending()(provider, calls)
        if not pending:
            return {}
        decisions = request_mcp_approvals(pending)
        provider.apply_batch_decisions(run_id, decisions)
        return {
            call.call_id or call.llm_name: read_global__review_decision()(
                call,
                decisions,
                "proceed",
                allowing=(
                    "approve_once",
                    "approve_session",
                    "always_allow",
                    "allow_matching",
                ),
            )
            for call in pending
        }

    return review_tool_calls


def build_raw_shell_review_hook(
    provider: "RawShellToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_USER_DENIED_REFUSAL: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Gate model-authored host-shell calls independently by native call id."""

    def review_tool_calls(
        calls: list["ToolCall"], run_id: str
    ) -> dict[str, read_global_ToolReviewValue()]:
        authority_generation = provider.authority_generation
        provider.apply_batch_decisions(run_id, {})
        pending = [
            row
            for call in calls
            if (row := provider.pending_gate_for(call)) is not None
        ]
        if not pending:
            return {}
        decisions = request_approvals(pending)
        provider.apply_batch_decisions(
            run_id,
            decisions,
            pending,
            authority_generation=authority_generation,
        )
        verdicts: dict[str, read_global_ToolReviewValue()] = {}
        for row in pending:
            key = row.call_id or row.llm_name
            decision = decisions.get(key)
            verdicts[key] = read_global__review_decision()(
                row,
                decisions,
                "proceed"
                if decision in ("approve_once", "approve_session")
                else read_global_USER_DENIED_REFUSAL().format(name=row.llm_name),
                allowing=("approve_once", "approve_session"),
                name_fallback=False,
            )
        return verdicts

    return review_tool_calls


def build_tool_review_hook(
    builtin_gate: "BuiltinToolGate",
    builtin_provider: "BuiltinToolProvider",
    mcp_provider: MCPToolProvider | None,
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    workspace_id: str | None,
    kill_switch: Callable[[], bool] | None,
    library_provider: Any | None,
    read_global_AGENT_LESSON_APPROVAL_REQUIRED: Callable[[], Any],
    read_global_AGENT_LESSON_DENIED: Callable[[], Any],
    read_global_AGENT_LESSON_FOREGROUND_REQUIRED: Callable[[], Any],
    read_global_Any: Callable[[], Any],
    read_global_ApprovalDecisions: Callable[[], Any],
    read_global_BUILTIN_TOOL_SERVER_KEY: Callable[[], Any],
    read_global_KILL_SWITCH_REFUSAL: Callable[[], Any],
    read_global_MCPPendingCall: Callable[[], Any],
    read_global_TOOL_DESCRIPTION_CAPTURE_CAP: Callable[[], Any],
    read_global_ToolReviewValue: Callable[[], Any],
    read_global_USER_DENIED_REFUSAL: Callable[[], Any],
    read_global__APPROVAL_SCOPE_RANK: Callable[[], Any],
    read_global__APPROVING_DECISIONS: Callable[[], Any],
    read_global__collect_mcp_pending: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
    read_global__sibling_approval_refusals: Callable[[], Any],
    read_global__stamp_answer_provenance: Callable[[], Any],
    read_global_approval_effects_for_tool: Callable[[], Any],
    read_global_approval_key_unanswered: Callable[[], Any],
    read_global_approval_was_unanswered: Callable[[], Any],
    read_global_current_run_actor: Callable[[], Any],
    read_global_logger: Callable[[], Any],
    read_global_path_precheck_failed: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Build THIS run's run-level `review_tool_calls` hook (P5-T6/task-545).

    TASK-631: when ``kill_switch`` reports on, EVERY call in the batch is
    refused here, without prompting -- the runtime turns any non-"proceed"
    verdict into the call's result and skips dispatch. MCP composition is
    already skipped and ``BuiltinToolGate.check`` already refuses with the
    switch on, but names neither provider claims (skills,
    ``spawn_subagent``, ``find_tools``, ``load_tools``) used to pass
    through unreviewed and RUN NORMALLY -- the switch's label promises
    "block tool calls in chat", and this hook is the one place every
    parsed call passes, so this is where the promise is kept. Read fresh
    per turn (a callable, not a bool) so flipping the switch mid-run takes
    effect on the next batch.

    Unlike `build_mcp_review_hook`, this is wired UNCONDITIONALLY -- every
    run gets one, even a user with no MCP servers configured at all --
    because built-in tools (calculator/datetime today, more later) must be
    gated regardless of whether MCP happens to be composed this turn.
    `BuiltinToolProvider.invoke` already enforces the gate as defense in
    depth, but without this hook the ONLY review a built-in call would ever
    get is that per-call fallback -- never the batched, one-card-per-turn
    review MCP calls already get, and never a chance to ask before
    dispatch for calls this hook doesn't stamp.

    Routing per call, MCP first: `mcp_provider.pending_gate_for` (when a
    provider was composed this run) is asked before the built-in provider,
    so a name that provider actually owns is never mistakenly re-resolved
    against the built-in side too. Note this hook's own precedent is the
    OPPOSITE of `console_agent_bridge._non_colliding_mcp_names`, which
    resolves a name collision the other way -- it drops the colliding MCP
    name from the run's registry so the built-in wins composition. That
    inconsistency is moot in practice: `MCP/tool_naming.py:106` always
    mints MCP tool names as `mcp__<server>__<tool>`, which can never equal
    a bare built-in name like `calculator`/`get_current_datetime`, so no
    call is ever ambiguous between the two orders. A name neither provider
    claims (a skill, `spawn_subagent`, `find_tools`, ...) passes through
    unreviewed, exactly as it does for `build_mcp_review_hook` today.

    Built-in rows use `server_key=BUILTIN_TOOL_SERVER_KEY`
    (`"agent:builtin"`), `server_label="Built-in"`, and `reason=
    "risk_floored"` when `EffectiveToolState.risk_floored` else `"ask"`
    (built-ins never set `config_changed` -- see `resolve_builtin_state`'s
    own docstring for why). Every built-in row's `path_precheck_failed`
    (TASK-1231/F3 AC2) is set via `Tools.file_operation_tools.
    path_precheck_failed`: for `read_file`/`list_directory`/`write_file`
    this pre-flights the SAME `allowed_file_roots`/`validate_path_multi`
    check `invoke()` runs at dispatch, so the approval card can warn the
    user this exact call will fail even if approved -- it never gates or
    auto-denies; `False` for every other builtin tool and every MCP row.
    Only a resolved `"ask"` state ever produces a row: `"allow"` never
    prompts, and `"deny"` is refused outright by
    `invoke()`'s own gate WITHOUT ever reaching the user -- a tool the
    operator switched Off must not appear on the approval card at all.
    Nor does an `"ask"` tool that already has a live session approval
    (`builtin_gate.is_session_approved(name)`) -- review finding 1
    (T6 review): `resolve()`/`resolve_builtin_state` read the permission
    store ONLY, never session approvals, so without this check a user who
    picked "Approve for session" on turn 1 would be re-prompted on turn 2
    even though `invoke()`'s own `check()` already honors that same
    session approval and would execute it anyway. Mirrors MCP's own
    `pending_gate_for`, which applies the identical
    `_is_session_approved_safe` skip for exactly this reason.

    `options=("approve_once", "approve_session", "deny")` -- deliberately
    excluding ONLY `"always_allow"` (verified at
    `Agents/mcp_tool_provider.py:556-564`: `always_allow` is the sole
    PERSISTENT write via `set_tool_state`; `approve_session` is an
    in-memory session cache and `deny`/`timeout` are turn-scoped refusals
    that persist nothing). `"deny"` MUST stay offered -- an earlier draft
    of this design mistakenly dropped it too, which would have made a
    built-in row impossible to refuse from the card at all (the bulk "Deny
    all" button would silently leave it on whatever the row's default
    was).

    Mirrors `build_mcp_review_hook`'s I3 clear-at-entry discipline, extended
    to the built-in side: `builtin_gate.begin_turn(run_id)` runs FIRST,
    unconditionally -- before the MCP stamp clear, before any
    `pending_gate_for`/`resolve` call, before the `request_approvals` round
    trip -- so a raising round trip can never leave a stale built-in stamp
    (or a stale cached permission payload) live for the next turn to
    consume. `mcp_provider.apply_batch_decisions(run_id, {})` follows the
    same reasoning for the MCP side, only when a provider was actually
    composed this run.

    PR2a Task 5: every one of those mutations is scoped to `run_id`, the
    second argument this hook now receives (`AgentService` binds its own
    run id into the callable it hands `LoopDeps`). The gate and the MCP
    provider are shared by a parent run and every sub-agent it spawns, so
    an unscoped clear here wipes -- and an unscoped stamp overwrites --
    verdicts another run in the tree has already been granted and has not
    yet consumed. It still clears THIS run's own previous turn, which is
    what the I3 discipline above requires.

    Exactly ONE `request_approvals` round trip is made per turn, carrying
    BOTH the MCP and built-in pending rows together -- never one call per
    owner. Decisions are then applied back to each owner separately:
    `mcp_provider.apply_batch_decisions(run_id, ...)` for MCP rows,
    `builtin_gate.stamp(run_id, name, decision)` for built-in rows. The returned
    verdict map carries "proceed" for approved calls and REFUSAL STRINGS
    for per-call denials (TASK-1861), kill-switch blocks (TASK-631), and
    calls that lack an approval of their own while a same-name sibling was
    approved (TASK-33082) -- the runtime enforces those directly, skipping
    dispatch. Approvals are
    still left to `invoke()`'s gate on dispatch, which records the audit
    decision.

    Args:
        builtin_gate: THIS run's `BuiltinToolGate` -- the SAME instance
            the run's `BuiltinToolProvider.invoke` checks, so a stamp
            written here is visible there. Two separate instances would
            mean a decision made here is invisible to `invoke()`, silently
            re-prompting (a stamp `invoke()` never sees) or failing closed
            (an approval that never reaches the gate that checks it).
        builtin_provider: THIS run's `BuiltinToolProvider` (only
            `.tool_for(name)` is used here, to resolve a `ToolCall.name`
            to the `Tool` object `builtin_gate.resolve` needs).
        mcp_provider: THIS run's already-composed `MCPToolProvider`, or
            `None` when no MCP tools should be offered this run (no
            service, kill switch on, or composition yielded nothing) --
            the entire point of this hook existing separately from
            `build_mcp_review_hook` is that built-in gating must not
            depend on this being non-`None`.
        request_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run (the name predates
            built-in gating; the method itself is owner-agnostic -- it
            only reads `MCPPendingCall` fields, never assumes MCP
            ownership).
        workspace_id: THIS run's OWN workspace id (round 1 review CRITICAL
            1) -- e.g. `self.store.session_workspace_id(session_id)` --
            threaded into every builtin file-tool row's `path_precheck_
            failed` computation via `Tools.file_operation_tools.
            path_precheck_failed`'s own `workspace_id=` parameter. Must be
            the SAME workspace id `ConsoleAgentBridge.run_reply` resolves
            for this run's real dispatch (`BuiltinToolProvider(workspace_
            id=...)`) -- otherwise the pre-flight can resolve a DIFFERENT
            workspace than the one the call will actually run against
            (e.g. whatever happens to be active in the UI for a parked
            background session), making the warning wrong in either
            direction. `None` (the default) reproduces the pre-existing
            active-workspace fallback for a caller with no session
            context at all; every caller that has a real session id MUST
            resolve and pass its workspace id.

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """

    def review_tool_calls(
        calls: list["ToolCall"], run_id: str
    ) -> dict[str, read_global_ToolReviewValue()]:
        # PR2a Task 5: every gate mutation below is scoped to `run_id` --
        # the run whose batch this is, supplied by `AgentService` (which
        # binds its own run id into the hook it puts on `LoopDeps`). The
        # gate and provider instances are shared by a parent and every
        # sub-agent it spawns, so an unscoped clear/stamp here reaches
        # verdicts a concurrent sibling has not yet consumed.
        lesson_clear_failed = False
        clear_lesson_approvals = getattr(
            library_provider, "clear_agent_lesson_approvals", None
        )
        if callable(clear_lesson_approvals):
            try:
                clear_lesson_approvals(run_id)
            except Exception:  # noqa: BLE001 - fail closed without payload logging
                lesson_clear_failed = True
        builtin_gate.begin_turn(run_id)
        # TASK-631: the kill switch outranks everything -- no prompting, no
        # stamps, every call refused. Per-call keys where the runtime can
        # address them; an id-less (fence-path) call is refused by NAME,
        # which stops every same-name call -- fail-closed, same reasoning
        # as TASK-1861's refusal fallback.
        if kill_switch is not None:
            try:
                switch_on = bool(kill_switch())
            except Exception:  # noqa: BLE001 -- an unreadable switch fails CLOSED
                read_global_logger().opt(exception=True).warning(
                    "build_tool_review_hook: kill-switch read failed; "
                    "refusing this turn's tool calls"
                )
                switch_on = True
            if switch_on:
                if mcp_provider is not None:
                    mcp_provider.apply_batch_decisions(run_id, {})
                return {
                    (str(getattr(call, "call_id", "") or "") or call.name): (
                        read_global_KILL_SWITCH_REFUSAL()
                    )
                    for call in calls
                }
        if mcp_provider is not None:
            mcp_provider.apply_batch_decisions(run_id, {})

        mcp_pending = (
            read_global__collect_mcp_pending()(mcp_provider, calls)
            if mcp_provider is not None
            else []
        )
        mcp_claimed_names = {row.llm_name for row in mcp_pending}

        lesson_pending: list["MCPPendingCall"] = []
        lesson_preflights: list[tuple["MCPPendingCall", read_global_Any()]] = []
        lesson_refusals: dict[str, str] = {}
        actor = read_global_current_run_actor()()
        preflight_lesson = getattr(
            library_provider, "preflight_agent_lesson_save", None
        )
        if callable(preflight_lesson):
            batch_call_ids = [str(getattr(call, "call_id", "") or "") for call in calls]
            duplicate_batch_call_ids = {
                call_id
                for call_id in batch_call_ids
                if call_id and batch_call_ids.count(call_id) > 1
            }
            for call in calls:
                if call.name != "library_save_note":
                    continue
                refusal_key = str(getattr(call, "call_id", "") or "") or call.name
                if lesson_clear_failed:
                    lesson_refusals[refusal_key] = (
                        read_global_AGENT_LESSON_APPROVAL_REQUIRED()
                    )
                    continue
                try:
                    preflight = preflight_lesson(
                        call.name,
                        dict(call.args or {}),
                        str(getattr(call, "call_id", "") or ""),
                    )
                except Exception:  # noqa: BLE001 - classification is content-free
                    lesson_refusals[refusal_key] = (
                        read_global_AGENT_LESSON_APPROVAL_REQUIRED()
                    )
                    continue
                if preflight is None:
                    continue
                if preflight.call_id in duplicate_batch_call_ids:
                    lesson_refusals[refusal_key] = (
                        read_global_AGENT_LESSON_APPROVAL_REQUIRED()
                    )
                    continue
                if actor is None or actor.run_id != run_id:
                    lesson_refusals[refusal_key] = (
                        read_global_AGENT_LESSON_APPROVAL_REQUIRED()
                    )
                    continue
                if actor.kind != "primary":
                    lesson_refusals[refusal_key] = (
                        read_global_AGENT_LESSON_FOREGROUND_REQUIRED()
                    )
                    continue
                row = read_global_MCPPendingCall()(
                    llm_name=call.name,
                    server_key="agent:library",
                    tool_name=call.name,
                    server_label="Agent Lessons",
                    arguments={
                        "operation": preflight.operation,
                        "title": preflight.title,
                        "classification": preflight.classification.reason,
                        "call_digest": preflight.call_digest,
                    },
                    call_id=preflight.call_id,
                    reason="ask",
                    options=("approve_once", "deny"),
                )
                lesson_pending.append(row)
                lesson_preflights.append((row, preflight))

        # Minor (round 1 review): memoize `allowed_file_roots` across every
        # builtin file-tool row THIS batch checks -- `workspace_id` is fixed
        # for the whole call, so a turn with several read_file/write_file
        # rows would otherwise re-hit the workspace registry (a repeat query
        # against `WorkspaceDB`'s held, per-thread connection -- task-3011)
        # once per row.
        # Fresh dict per `review_tool_calls` call -- never reused across
        # turns, so a folder binding added/removed between turns is still
        # picked up on the very next call.
        path_roots_cache: dict[bool, tuple] = {}

        builtin_pending: list["MCPPendingCall"] = []
        for call in calls:
            if call.name in mcp_claimed_names:
                continue
            tool = builtin_provider.tool_for(call.name)
            if tool is None:
                continue  # not ours either -- a skill/native tool, unreviewed
            state = builtin_gate.resolve(tool)
            if state.state != "ask":
                # "allow" never prompts; "deny" is refused outright by
                # invoke()'s own gate -- neither is offered a card.
                continue
            if builtin_gate.is_session_approved(call.name):
                # Review finding 1 (T6 review): already approved for this
                # session -- `invoke()`'s own `check()` will honor it via
                # the identical `is_session_approved` read, so re-asking
                # here would just re-prompt for a decision the user
                # already made. Not added to `builtin_pending` and so
                # never mentioned in the returned verdict map either --
                # exactly as undecided-but-not-needed-this-turn MCP calls
                # already work (see this function's own docstring).
                continue
            builtin_pending.append(
                read_global_MCPPendingCall()(
                    llm_name=call.name,
                    server_key=read_global_BUILTIN_TOOL_SERVER_KEY(),
                    tool_name=call.name,
                    server_label="Built-in",
                    arguments=dict(call.args or {}),
                    # Per-call verdict key: lets the user allow one target and
                    # refuse another in the same batch. Empty on the fence
                    # path, where the runtime falls back to the name.
                    call_id=str(getattr(call, "call_id", "") or ""),
                    rationale=str(getattr(call, "rationale", "") or ""),
                    description=str(getattr(tool, "description", "") or "")[
                        : read_global_TOOL_DESCRIPTION_CAPTURE_CAP()
                    ],
                    reason="risk_floored" if state.risk_floored else "ask",
                    # task-32278: the card's high-risk sentence must say
                    # "changes" for a mutating built-in, and `effects` is the
                    # only signal it reads. Derived from the same tags the
                    # floor above keys on, so the two cannot disagree.
                    effects=read_global_approval_effects_for_tool()(tool),
                    options=("approve_once", "approve_session", "deny"),
                    # TASK-1231/F3 AC2: pre-flight the roots check for the
                    # three file tools -- never gates or auto-denies, just
                    # tells the card this specific path is doomed even if
                    # approved (see path_precheck_failed's own docstring).
                    # `workspace_id=workspace_id` (round 1 review CRITICAL
                    # 1): the pre-flight MUST resolve THIS run's own
                    # workspace, never whatever happens to be active in the
                    # UI -- see this function's own docstring.
                    path_precheck_failed=read_global_path_precheck_failed()(
                        call.name,
                        call.args,
                        workspace_id=workspace_id,
                        roots_cache=path_roots_cache,
                        sandbox_root=getattr(builtin_provider, "sandbox_root", None),
                        sandbox_lease=getattr(
                            builtin_provider,
                            "sandbox_lease",
                            None,
                        ),
                    ),
                )
            )

        all_pending = mcp_pending + builtin_pending + lesson_pending
        if not all_pending:
            return lesson_refusals
        try:
            decisions = request_approvals(all_pending)
        except BaseException:
            if callable(clear_lesson_approvals):
                try:
                    clear_lesson_approvals(run_id)
                except Exception:  # noqa: BLE001 - cleanup remains best effort
                    pass
            raise

        def _decision_for(row: "MCPPendingCall") -> str | None:
            """Resolve one row's verdict, per-call id first then name.

            The card now keys verdicts by `call_id` where the runtime can
            address them (so two reads of two files are two decisions), but
            BOTH consumers below are name-keyed by contract:
            `MCPToolProvider.apply_batch_decisions` takes llm_names, and
            `builtin_gate.stamp` records a grant against a tool NAME because
            a session/always grant is per tool, not per call. Resolving here
            keeps the finer-grained card from silently starving them --
            without this, MCP received {} and no gate grant was ever stamped.
            """
            key = str(getattr(row, "call_id", "") or "")
            if key and key in decisions:
                return decisions[key]
            return decisions.get(row.llm_name)

        def _stamps_for(
            rows: "list[MCPPendingCall]",
        ) -> read_global_ApprovalDecisions():
            """Name-keyed stamps for `rows`: approvals win, all-denied denies.

            TASK-1861. A refusal must NOT be stamped against the name when a
            sibling call of the same tool was approved -- the stamp is what
            `invoke()` peeks at, and it cannot express "allow this one,
            refuse that one", so stamping the refusal would also stop the
            call the user allowed. Refusals are enforced per call by the
            verdict map below instead.

            Stamping the approval is safe even with a refused sibling,
            because that sibling is stopped before dispatch and never
            reaches `invoke()`. When EVERY call of a name was refused there
            is no approval to preserve, so "deny" is stamped as defense in
            depth for any path that bypasses the verdict map.
            """
            approvals: dict[str, str] = {}
            denied: set[str] = set()
            for row in rows:
                decision = _decision_for(row)
                if decision is None:
                    continue
                if decision == "deny":
                    denied.add(row.llm_name)
                    continue
                # Per-call rows can disagree on SCOPE, not just allow/refuse,
                # and only one scope per name can be stamped. Taking the last
                # silently downgraded "Approve for session" to "approve once"
                # whenever a later row of the same tool was approved once --
                # dropping the grant the user asked for and re-prompting on
                # the next call. Choosing "for session" on ANY call of a tool
                # is choosing to grant that tool for the session (that is what
                # the control means, and its label says so), so the broadest
                # chosen scope wins.
                current = approvals.get(row.llm_name)
                if current is None or read_global__APPROVAL_SCOPE_RANK().get(
                    decision, 0
                ) > read_global__APPROVAL_SCOPE_RANK().get(current, 0):
                    approvals[row.llm_name] = decision
            stamps = dict(approvals)
            for name in denied:
                stamps.setdefault(name, "deny")
            return read_global__stamp_answer_provenance()(stamps, rows, decisions)

        if mcp_provider is not None:
            mcp_provider.apply_batch_decisions(
                run_id,
                _stamps_for(
                    [r for r in mcp_pending if r.llm_name in mcp_claimed_names]
                ),
            )
        builtin_stamps = _stamps_for(builtin_pending)
        for name, decision in builtin_stamps.items():
            if read_global_approval_key_unanswered()(builtin_stamps, name):
                builtin_gate.stamp(run_id, name, decision, unanswered=True)
            else:
                builtin_gate.stamp(run_id, name, decision)

        # task-32280: because the runtime turns the refusal below into the
        # call's result and never dispatches it, `MCPToolProvider.invoke` --
        # which records every refusal IT reaches -- never runs for a denied
        # call. Live on dev 3315241674 that left three approvals of one tool
        # in the execution log and no row at all for the Deny. Record at the
        # point the denial becomes final, through the provider's own audit
        # seam. Built-in rows are left alone: nothing records their
        # approvals either, so a denial-only trail would be worse than none.
        # R23: skip rows whose "deny" was DEFAULTED by a Stop/revoke --
        # `request_mcp_approvals` already logged those as
        # `denied-unresolved`, and recording them again here claimed the
        # user pressed Deny on a card they never saw resolved.
        if mcp_provider is not None:
            for row in mcp_pending:
                if _decision_for(
                    row
                ) == "deny" and not read_global_approval_was_unanswered()(
                    row, decisions
                ):
                    mcp_provider.record_user_denial(row.llm_name)

        # The refusal half, enforced HERE rather than through the stamps.
        # The runtime resolves `call_id` before name and turns any
        # non-"proceed" verdict string into that call's result without
        # dispatching it, so this is the only layer that can refuse one
        # target while running another.
        verdicts: dict[str, read_global_ToolReviewValue()] = {
            row.llm_name: "proceed" for row in mcp_pending + builtin_pending
        }
        for row in mcp_pending + builtin_pending:
            if _decision_for(row) != "deny":
                continue
            # Prefer the per-call key. A row with no `call_id` -- the fence
            # path, or an MCP row whose provider omitted an id -- can only be
            # addressed by name, which stops every same-name call in the
            # batch. That is fail-closed, and the only honest option when the
            # runtime cannot tell those calls apart.
            key = str(getattr(row, "call_id", "") or "") or row.llm_name
            verdicts[key] = read_global__review_decision()(
                row,
                decisions,
                read_global_USER_DENIED_REFUSAL().format(name=row.llm_name),
            )

        def _allowing(row: read_global_MCPPendingCall()) -> tuple[str, ...]:
            if row in mcp_pending:
                return (*read_global__APPROVING_DECISIONS(), "allow_matching")
            return read_global__APPROVING_DECISIONS()

        def _record_sibling_refusal(
            row: read_global_MCPPendingCall(), timed_out: bool
        ) -> None:
            # Built-in rows are left unaudited, as for a Deny above.
            if mcp_provider is not None and row in mcp_pending:
                mcp_provider.record_hook_refusal(row.llm_name, timed_out=timed_out)

        verdicts.update(
            read_global__sibling_approval_refusals()(
                mcp_pending + builtin_pending,
                _decision_for,
                decisions,
                _allowing,
                _record_sibling_refusal,
            )
        )
        # Settle all name-wide refusals first. Metadata must never add an
        # exact proceed that bypasses an id-less sibling's refusal fallback.
        verdicts.update(
            {
                row.call_id or row.llm_name: read_global__review_decision()(
                    row, decisions, "proceed", allowing=_allowing(row)
                )
                for row in mcp_pending + builtin_pending
                if verdicts.get(row.call_id or row.llm_name, verdicts[row.llm_name])
                == "proceed"
            }
        )
        verdicts.update(lesson_refusals)
        issue_lesson_approval = getattr(
            library_provider, "issue_agent_lesson_approval", None
        )
        lesson_issue_failed = False
        for row, preflight in lesson_preflights:
            decision = decisions.get(row.call_id)
            if decision == "approve_once" and callable(issue_lesson_approval):
                try:
                    issue_lesson_approval(run_id, preflight)
                except Exception:  # noqa: BLE001 - no call data crosses refusal
                    verdicts[row.call_id] = read_global_AGENT_LESSON_APPROVAL_REQUIRED()
                    lesson_issue_failed = True
                else:
                    verdicts[row.call_id] = read_global__review_decision()(
                        row,
                        decisions,
                        "proceed",
                        allowing=("approve_once",),
                        name_fallback=False,
                    )
            elif decision == "deny":
                verdicts[row.call_id] = read_global__review_decision()(
                    row,
                    decisions,
                    read_global_AGENT_LESSON_DENIED(),
                    allowing=("approve_once",),
                    name_fallback=False,
                )
            else:
                verdicts[row.call_id] = read_global_AGENT_LESSON_APPROVAL_REQUIRED()
        if lesson_issue_failed:
            if callable(clear_lesson_approvals):
                try:
                    clear_lesson_approvals(run_id)
                except Exception:  # noqa: BLE001 - already failing closed
                    pass
            for row, _preflight in lesson_preflights:
                verdicts[row.call_id] = read_global_AGENT_LESSON_APPROVAL_REQUIRED()
        return verdicts

    return review_tool_calls


def build_virtual_cli_review_hook(
    provider: "VirtualCliProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    read_global_ToolReviewValue: Callable[[], Any],
    read_global__review_decision: Callable[[], Any],
    read_global_append_denial_reason: Callable[[], Any],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Gate each selected virtual command while exposing one model tool.

    Approval rows are command-specific Hub entries but verdict stamps are
    keyed by native call id, so multiple ``virtual_cli`` calls in one model
    response remain independently addressable.
    """

    def review_tool_calls(
        calls: list["ToolCall"], run_id: str
    ) -> dict[str, read_global_ToolReviewValue()]:
        provider.apply_batch_decisions(run_id, {})
        pending = [
            row
            for call in calls
            if (row := provider.pending_gate_for(call)) is not None
        ]
        if not pending:
            return {}
        from tldw_chatbook.Agents.local_tool_provider import LOCAL_USER_DENY_REFUSAL

        decisions = request_approvals(pending)
        provider.apply_batch_decisions(run_id, decisions, pending)
        reason_keys = {
            row.call_id or row.llm_name
            for row in pending
            if read_global_append_denial_reason()(
                LOCAL_USER_DENY_REFUSAL, decisions, row.call_id or row.llm_name
            )
            != LOCAL_USER_DENY_REFUSAL
        }
        for row in pending:
            if (row.call_id or row.llm_name) in reason_keys:
                provider.record_user_denial(row.tool_name)
        return {
            row.call_id or row.llm_name: read_global__review_decision()(
                row,
                decisions,
                LOCAL_USER_DENY_REFUSAL
                if (row.call_id or row.llm_name) in reason_keys
                else "proceed",
                allowing=(
                    "approve_once",
                    "approve_session",
                    "always_allow",
                    "allow_matching",
                ),
                name_fallback=False,
            )
            for row in pending
        }

    return review_tool_calls


# Definition-time sources for optional in-memory Inspector pending display only.
_CONSOLE_PENDING_FACTS_READERS = tuple(
    (
        name,
        method,
        method.__code__,
        method.__globals__,
        method.__defaults__,
        method.__kwdefaults__,
        tuple((method.__kwdefaults__ or {}).items()),
        method.__closure__,
        tuple((cell, cell.cell_contents) for cell in method.__closure__ or ()),
    )
    for name in (
        "pending_round_count",
        "pending_round_kinds",
        "has_pending_approval_round",
    )
    for method in (getattr(InterruptRoundHost, name),)
)
