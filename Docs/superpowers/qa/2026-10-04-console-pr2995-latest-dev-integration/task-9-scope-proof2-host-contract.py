from __future__ import annotations
from collections.abc import Callable
from typing import Any


class InterruptRoundHost:
    """Proposed keyword-only dependency amendment; body relocation is not implemented."""

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
        read_global_Any: Callable[[], Any],
        read_global_ApprovalDecisions: Callable[[], Any],
        read_global_CONSOLE_PENDING_APPROVAL_KIND: Callable[[], Any],
        read_global_CONSOLE_PENDING_CHAT_CREATE_KIND: Callable[[], Any],
        read_global_ConsolePendingDecisionProjection: Callable[[], Any],
        read_global_INTERRUPT_BELL_ENV_VAR: Callable[[], Any],
        read_global_Mapping: Callable[[], Any],
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
        self.read_global_Any = read_global_Any
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
        self.read_global_Mapping = read_global_Mapping
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
        ...

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
        ...

    def _publish_console_attention_change(self) -> None:
        """Best-effort notification that the runtime should re-derive attention."""
        ...

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
        ...

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
        ...

    def pending_round_count(self, session_id: str, *, kind: str) -> int:
        """Count one session's outstanding rounds of the requested kind.

        Args:
            session_id: The owning session to inspect.
            kind: Interrupt kind to count, including queued or hidden rounds.

        Returns:
            Number of registered rounds of this kind for the session.
        """
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

    def _summary_tail_messages(self, payload: dict[str, Any]) -> list:
        """User/assistant text projection of the round's stored conversation.

        Uses the same message flattening as world-info scanning
        (``_normalize_world_info_history``) and the same stored-message
        source its call sites feed it (``_provider_messages_for_session``,
        which reads ``self.store.messages_for_session`` and emits the
        provider-dict shape the flattener consumes); keeps the defensive
        no-raise posture.
        """
        ...

    def _deliver_permission_summary(
        self, round_id: str, payload: dict[str, Any], text: str
    ) -> None:
        """UI THREAD: store the summary, then patch the mounted card.

        Drops resolved/revoked rounds and unknown ids; writes the payload's
        ``summary`` slot (the source of truth for remounts) before the live
        patch. Never re-runs ``set_batch``.
        """
        ...

    def _publish_pending_decision(
        self,
        *,
        round_state: dict[str, Any],
        payload: dict[str, Any],
        decision_type: Literal["approval", "skill_install", "skill_script"],
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
        ...

    def _pending_round_states_snapshot(self) -> dict[str, dict[str, Any]]:
        """Snapshot existing round records without nesting registry locks.

        Lock order is type registry lock, release, then
        ``_approval_state_lock`` in callers. No code acquires a type lock while
        holding the shared projection lock.
        """
        ...

    def _pending_decision_payloads_locked(
        self, session_id: str
    ) -> list[dict[str, Any]]: ...

    def _pause_pending_decision_state_locked(
        self, state: dict[str, Any], *, now: float
    ) -> None: ...

    def _mutate_exact_pending_decision(
        self, decision_id: str, mutate: Callable[[dict[str, Any]], Any]
    ) -> Any:
        """Mutate one live round under its registry then projection lock.

        All five kinds share the host's non-reentrant lock. Mutations must
        never acquire a second aliased lock or project UI while holding it.
        """
        ...

    @staticmethod
    def _settle_pending_decision_timeout_locked(
        state: dict[str, Any], *, read_global_threading: Callable[[], Any]
    ) -> threading.Event | None:
        """Stamp one exact live round timeout once; caller holds its locks."""
        ...

    def _pause_answerable_decision(
        self,
        session_id: str,
        decision_id: str,
        *,
        now: float,
        claim_revision: int | None,
    ) -> bool:
        """Pause one exact head, terminally timing it out at zero."""
        ...

    def _expire_answerable_decision_if_due(
        self, session_id: str, decision_id: str, *, now: float
    ) -> bool:
        """Settle one still-mounted head only when its active allowance is due."""
        ...

    def pending_decision_projection(
        self, session_id: str
    ) -> ConsolePendingDecisionProjection | None:
        """Return the session's one stable mixed-type FIFO head."""
        ...

    def project_pending_decision_for_active_session(self) -> bool:
        """Project only the active session's ordered mixed-type head."""
        ...

    def _reproject_pending_decision_for_session(self, session_id: str) -> None:
        """Re-derive one session through the unified or legacy card seams."""
        ...

    def active_session_changed(self) -> None:
        """Pause stale heads and derive the newly active session's head."""
        ...

    def _cancel_pending_decisions_for_session(self, session_id: str) -> None:
        """Fail closed only the rounds owned by a destructively closed session."""
        ...

    def _marshal_pending_decision_projection(self) -> None:
        """Worker-thread marshal of the active-session derived head."""
        ...

    def set_answerable_decision(self, session_id: str, decision_id: str | None) -> bool:
        """Update Console's claim without erasing another visible owner's claim."""
        ...

    def _refresh_answerable_decision(self, session_id: str) -> str | None:
        """Reconcile rendered Console/Buddy claims against one typed FIFO clock."""
        ...

    def expire_pending_decisions(self) -> tuple[str, ...]:
        """Fail closed every answerable head whose active allowance elapsed."""
        ...

    @staticmethod
    def _head_round_payload_locked(
        store: dict[str, dict[str, Any]], session_id: str | None
    ) -> dict[str, Any] | None:
        """The session's oldest-armed payload. Caller holds the lock."""
        ...

    def _park_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str, payload: dict[str, Any]
    ) -> bool:
        """Retain ``payload``; return whether it is now its session's head."""
        ...

    def _head_round_payload(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> dict[str, Any] | None:
        """The payload whose card ``session_id`` should currently show (remaining-time snapshot)."""
        ...

    def _session_round_payloads(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Every payload ``store`` retains for ``session_id``, arm order first."""
        ...

    def _unpark_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str
    ) -> None:
        """Drop ``round_id``'s retained payload, if any."""
        ...

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
        ...

    def _remount_session_kinds(self, session_id: str) -> None:
        """Re-derive every non-approval kind's head card for ``session_id``.

        The one call the three session-activation sites (new, switch,
        close) share; the kinds come from the host module's
        ``SESSION_REMOUNT_KINDS`` and approvals stay on the sites' own block.

        Args:
            session_id: The session being activated.
        """
        ...

    def on_console_view_visibility_changed(self, visible: bool) -> None:
        """Project screen visibility without changing execution or cancellation."""
        ...

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
        ...

    def _approval_view_is_detached(self) -> bool:
        """True when Console is hidden or its approval view hooks are absent.

        TASK-31520 retains hooks during navigation. Attachment alone therefore
        cannot tell whether the user can see a card; modals also suspend it.
        """
        ...

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
        ...

    def _interrupt_bell_enabled(self) -> bool:
        """Resolve ``[console] interrupt_bell``: environment, then config, then on.

        Returns:
            False only when ``TLDW_CONSOLE_INTERRUPT_BELL`` (a non-empty
            value) or the config key coerces to False.
        """
        ...

    def announce_hidden_decision(self, session_id: str, kind: str) -> None:
        """Keep typed notices under their live stable-ID privacy authority."""
        ...

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
        ...

    def _announce_hidden_decision(
        self,
        decision_type: Literal["approval", "skill_install", "skill_script"],
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
        ...

    def _forget_hidden_decision(self, decision_id: str) -> None:
        """Release one terminal decision's app-wide announcement marker."""
        ...

    def _resolve_mcp_approval_timeout_seconds(self) -> float: ...

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
        ...

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
        ...

    def complete_definitive_tool(
        self, run_id: str, call_key: str, tool_name: str
    ) -> None:
        """WORKER THREAD: clear one finishing row at its real terminal.

        The primary key is the provider call id.  Fence/local rows that did
        not carry one fall back to the tool name; only one matching row is
        consumed per callback so repeated same-name calls remain visible
        until each sequential mutation actually finishes.
        """
        ...

    def complete_definitive_run(self, run_id: str) -> None:
        """WORKER THREAD: remove finishing rows a run never dispatched."""
        ...

    def _discard_approval_rows_for_closing_session(self, session_id: str) -> None:
        """Drop every approval payload owned by a closing session.

        This uses the same lock as the approval-to-finishing transition, so
        whichever operation wins first, no later transition can retain a row
        for a session that is being deleted.
        """
        ...

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
        ...

    def revoke_raw_shell_authority(self) -> int:
        """Fail closed only raw-shell stamps and approval rounds on disarm."""
        ...

    def _revoke_tool_approval_rounds(self, run_id: str) -> list[tuple[str, str | None]]:
        """Fail this run's tool-approval rounds closed. Registry work only.

        Args:
            run_id: The cancelled/abandoned run.

        Returns:
            ``(round_id, session_id)`` for each revoked round, for the
            caller's badge/card teardown (which must run outside the lock
            held here).
        """
        ...

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
        ...

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
        ...

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
        ...

    def _marshal_pending_skill_install(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-install confirm payload to the UI thread.

        No-op when no UI bridge is wired (``self.app`` or
        ``set_pending_skill_install`` is None).

        Args:
            payload: The pending confirm's ``{"url", "timeout_seconds"}``
                dict to show, or None to clear/hide the card.
        """
        ...

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
        ...

    def pending_skill_install_ids(self) -> list[str]:
        """Return the request ids of every currently-armed install-confirm round.

        Mirrors ``pending_skill_script_ids`` -- exposed for tests and for
        any surface that needs to know whether a decision is outstanding.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending.
        """
        ...

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
        ...

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
        ...

    def _marshal_pending_skill_script(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-script confirm payload to the UI thread.

        Args:
            payload: The pending confirm dict to show, or None to hide it.
        """
        ...

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
        ...

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
        ...

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
        ...

    def pending_skill_script_ids(self) -> list[str]:
        """Return the request ids of every currently-armed confirm round.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending. Exposed for tests and for any surface that needs to
            know whether a decision is outstanding.
        """
        ...

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
        ...

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
        ...

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
        ...

    def _marshal_pending_chat_create(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: project a current chat-create decision on the UI.

        Recheck scoped ownership after dispatch; legacy unparked rounds keep
        their unconditional initial projection. A clear derives the current head.

        Args:
            payload: Proposed confirmation, or None to rederive the active head.
        """
        ...

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
        ...

    def pending_chat_create_ids(self) -> list[str]:
        """Return the request ids of every currently-armed chat-create round.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending. Exposed for tests and for any surface that needs to
            know whether a decision is outstanding.
        """
        ...

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
        ...

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
        ...

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
        ...

    def pending_question_ids(self) -> list[str]:
        """Return the request ids of every armed question round, arm order.

        Returns:
            The armed round ids; empty when none is pending.
        """
        ...

    def _marshal_pending_question(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a question payload to the UI thread.

        Args:
            payload: The card payload to show, or None to hide the card.
        """
        ...

    def _remount_parked_question(self, session_id: str) -> None:
        """UI THREAD: re-derive the question card for the session now viewed.

        Called from ``switch_session``/``new_session``/``close_session``
        beside the other card re-derives (PRD A10).

        Args:
            session_id: The session being activated/viewed.
        """
        ...

    @property
    def worktree_confirmation_enabled(self) -> bool:
        """Only the real disposable worktree surface enables new tool disclosure."""
        ...

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
        ...

    def _remount_parked_worktree_merge(self, session_id: str) -> None:
        """Re-derive the mounted worktree-merge confirm card for
        ``session_id``. Mirrors ``_remount_parked_skill_script``.

        Args:
            session_id: The session now being activated/viewed.
        """
        ...

    def _marshal_pending_worktree_merge(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a worktree-merge confirm payload to the UI thread.

        Args:
            payload: The pending confirm dict to show, or None to hide it.
        """
        ...

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
        ...

    def pending_worktree_merge_ids(self) -> list[str]:
        """Return the request ids of every currently-armed worktree-merge
        confirm round. Mirrors ``pending_skill_script_ids``.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending.
        """
        ...

    def _notify_run_hook_approval(
        self, kind: str, payload: dict[str, Any], state: dict[str, Any]
    ) -> None:
        """Publish one successfully admitted permission round, including headless runs."""
        ...
