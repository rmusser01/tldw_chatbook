from __future__ import annotations


def approval_was_unanswered(
    row: "MCPPendingCall", decisions: Mapping[str, str]
) -> bool:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        approval_was_unanswered as _owned_approval_was_unanswered,
    )

    return _owned_approval_was_unanswered(
        row,
        decisions,
        read_global_approval_key_unanswered=lambda: approval_key_unanswered,
        read_global_selected_approval_key=lambda: selected_approval_key,
    )


def _approval_decision_fact(
    decision: object, *, unanswered: bool = False
) -> ApprovalDecision | None:
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _approval_decision_fact as _owned__approval_decision_fact,
    )

    return _owned__approval_decision_fact(decision, unanswered=unanswered)


def _review_decision(
    row: MCPPendingCall,
    decisions: Mapping[str, str],
    verdict: str,
    *,
    allowing: tuple[str, ...] = _APPROVING_DECISIONS,
    name_fallback: bool = True,
) -> ToolReviewDecision:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _review_decision as _owned__review_decision,
    )

    return _owned__review_decision(
        row,
        decisions,
        verdict,
        allowing=allowing,
        name_fallback=name_fallback,
        read_global_ToolReviewDecision=lambda: ToolReviewDecision,
        read_global__approval_decision_fact=lambda: _approval_decision_fact,
        read_global_append_denial_reason=lambda: append_denial_reason,
        read_global_approval_key_unanswered=lambda: approval_key_unanswered,
        read_global_selected_approval_key=lambda: selected_approval_key,
    )


def _stamp_answer_provenance(
    stamps: dict[str, str], rows: Sequence[MCPPendingCall], decisions: Mapping[str, str]
) -> ApprovalDecisions:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _stamp_answer_provenance as _owned__stamp_answer_provenance,
    )

    return _owned__stamp_answer_provenance(
        stamps,
        rows,
        decisions,
        read_global_ApprovalDecisions=lambda: ApprovalDecisions,
        read_global_approval_was_unanswered=lambda: approval_was_unanswered,
        read_global_selected_approval_key=lambda: selected_approval_key,
    )


def _sibling_approval_refusals(
    rows: Sequence[MCPPendingCall],
    decision_for: Callable[[MCPPendingCall], str | None],
    decisions: Mapping[str, str],
    allowing_for: Callable[[MCPPendingCall], tuple[str, ...]],
    record_refusal: Callable[[MCPPendingCall, bool], None],
) -> dict[str, ToolReviewValue]:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _sibling_approval_refusals as _owned__sibling_approval_refusals,
    )

    return _owned__sibling_approval_refusals(
        rows,
        decision_for,
        decisions,
        allowing_for,
        record_refusal,
        read_global_TIMEOUT_REFUSAL=lambda: TIMEOUT_REFUSAL,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_UNRESOLVED_REFUSAL=lambda: UNRESOLVED_REFUSAL,
        read_global__review_decision=lambda: _review_decision,
    )


def _collect_mcp_pending(
    provider: MCPToolProvider, calls: list["ToolCall"]
) -> list["MCPPendingCall"]:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _collect_mcp_pending as _owned__collect_mcp_pending,
    )

    return _owned__collect_mcp_pending(provider, calls)


def _build_approval_payload(
    round_id: str,
    session_id: str,
    run_id: str,
    pending: "list[MCPPendingCall]",
    timeout_seconds: float,
    deadline: float | None,
) -> dict[str, Any]:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _build_approval_payload as _owned__build_approval_payload,
    )

    return _owned__build_approval_payload(
        round_id,
        session_id,
        run_id,
        pending,
        timeout_seconds,
        deadline,
        read_global_ToolExecutionPolicy=lambda: ToolExecutionPolicy,
    )


def build_mcp_review_hook(
    provider: MCPToolProvider,
    request_mcp_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_mcp_review_hook as _owned_build_mcp_review_hook,
    )

    return _owned_build_mcp_review_hook(
        provider,
        request_mcp_approvals,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global__collect_mcp_pending=lambda: _collect_mcp_pending,
        read_global__review_decision=lambda: _review_decision,
    )


def build_tool_review_hook(
    builtin_gate: "BuiltinToolGate",
    builtin_provider: "BuiltinToolProvider",
    mcp_provider: MCPToolProvider | None,
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    workspace_id: str | None = None,
    kill_switch: Callable[[], bool] | None = None,
    library_provider: Any | None = None,
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_tool_review_hook as _owned_build_tool_review_hook,
    )

    return _owned_build_tool_review_hook(
        builtin_gate,
        builtin_provider,
        mcp_provider,
        request_approvals,
        workspace_id=workspace_id,
        kill_switch=kill_switch,
        library_provider=library_provider,
        read_global_AGENT_LESSON_APPROVAL_REQUIRED=lambda: (
            AGENT_LESSON_APPROVAL_REQUIRED
        ),
        read_global_AGENT_LESSON_DENIED=lambda: AGENT_LESSON_DENIED,
        read_global_AGENT_LESSON_FOREGROUND_REQUIRED=lambda: (
            AGENT_LESSON_FOREGROUND_REQUIRED
        ),
        read_global_Any=lambda: Any,
        read_global_ApprovalDecisions=lambda: ApprovalDecisions,
        read_global_BUILTIN_TOOL_SERVER_KEY=lambda: BUILTIN_TOOL_SERVER_KEY,
        read_global_KILL_SWITCH_REFUSAL=lambda: KILL_SWITCH_REFUSAL,
        read_global_MCPPendingCall=lambda: MCPPendingCall,
        read_global_TOOL_DESCRIPTION_CAPTURE_CAP=lambda: TOOL_DESCRIPTION_CAPTURE_CAP,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_USER_DENIED_REFUSAL=lambda: USER_DENIED_REFUSAL,
        read_global__APPROVAL_SCOPE_RANK=lambda: _APPROVAL_SCOPE_RANK,
        read_global__APPROVING_DECISIONS=lambda: _APPROVING_DECISIONS,
        read_global__collect_mcp_pending=lambda: _collect_mcp_pending,
        read_global__review_decision=lambda: _review_decision,
        read_global__sibling_approval_refusals=lambda: _sibling_approval_refusals,
        read_global__stamp_answer_provenance=lambda: _stamp_answer_provenance,
        read_global_approval_effects_for_tool=lambda: approval_effects_for_tool,
        read_global_approval_key_unanswered=lambda: approval_key_unanswered,
        read_global_approval_was_unanswered=lambda: approval_was_unanswered,
        read_global_current_run_actor=lambda: current_run_actor,
        read_global_logger=lambda: logger,
        read_global_path_precheck_failed=lambda: path_precheck_failed,
    )


def build_local_review_hook(
    provider: "LocalToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_local_review_hook as _owned_build_local_review_hook,
    )

    return _owned_build_local_review_hook(
        provider,
        request_approvals,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_USER_DENIED_REFUSAL=lambda: USER_DENIED_REFUSAL,
        read_global__APPROVAL_SCOPE_RANK=lambda: _APPROVAL_SCOPE_RANK,
        read_global__APPROVING_DECISIONS=lambda: _APPROVING_DECISIONS,
        read_global__review_decision=lambda: _review_decision,
        read_global__sibling_approval_refusals=lambda: _sibling_approval_refusals,
        read_global__stamp_answer_provenance=lambda: _stamp_answer_provenance,
        read_global_approval_was_unanswered=lambda: approval_was_unanswered,
    )


def build_managed_skill_promotion_review_hook(
    gate: "ManagedSkillProposalGate",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_managed_skill_promotion_review_hook as _owned_build_managed_skill_promotion_review_hook,
    )

    return _owned_build_managed_skill_promotion_review_hook(
        gate,
        request_approvals,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_USER_DENIED_REFUSAL=lambda: USER_DENIED_REFUSAL,
        read_global__review_decision=lambda: _review_decision,
    )


def build_virtual_cli_review_hook(
    provider: "VirtualCliProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_virtual_cli_review_hook as _owned_build_virtual_cli_review_hook,
    )

    return _owned_build_virtual_cli_review_hook(
        provider,
        request_approvals,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global__review_decision=lambda: _review_decision,
        read_global_append_denial_reason=lambda: append_denial_reason,
    )


def build_raw_shell_review_hook(
    provider: "RawShellToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_raw_shell_review_hook as _owned_build_raw_shell_review_hook,
    )

    return _owned_build_raw_shell_review_hook(
        provider,
        request_approvals,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_USER_DENIED_REFUSAL=lambda: USER_DENIED_REFUSAL,
        read_global__review_decision=lambda: _review_decision,
    )


def build_combined_review_hook(
    hooks: list[Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]],
) -> Callable[[list["ToolCall"], str], dict[str, ToolReviewValue]]:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        build_combined_review_hook as _owned_build_combined_review_hook,
    )

    return _owned_build_combined_review_hook(
        hooks,
        read_global_ToolReviewValue=lambda: ToolReviewValue,
        read_global_logger=lambda: logger,
    )


def _stamp_approval_round_closed(state: dict[str, Any]) -> None:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _stamp_approval_round_closed as _owned__stamp_approval_round_closed,
    )

    return _owned__stamp_approval_round_closed(state)


def _stamp_skill_script_round_closed(state: dict[str, Any]) -> None:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _stamp_skill_script_round_closed as _owned__stamp_skill_script_round_closed,
    )

    return _owned__stamp_skill_script_round_closed(state)


def _stamp_question_round_closed(state: dict[str, Any]) -> None:
    """Forward to the documented console_interrupt_rounds implementation."""
    from tldw_chatbook.Chat.console_interrupt_rounds import (
        _stamp_question_round_closed as _owned__stamp_question_round_closed,
    )

    return _owned__stamp_question_round_closed(state)


class ConsoleChatController:
    def __init__(self):
        """Scope fragment replacing the original host import and construction only."""

        def write_controller__pending_decision_order(value):
            """Write through to the controller-owned _pending_decision_order value."""
            self._pending_decision_order = value

        from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost

        self._interrupt_host = InterruptRoundHost(
            read_controller__active_assistant_message_ids=lambda: (
                self._active_assistant_message_ids
            ),
            read_controller__advance_lifecycle_revision=lambda: (
                self._advance_lifecycle_revision
            ),
            read_controller__agent_bridge=lambda: getattr(self, "_agent_bridge", None),
            read_controller__announce_detached_approval=lambda: (
                self._announce_detached_approval
            ),
            read_controller__announce_hidden_decision=lambda: (
                self._announce_hidden_decision
            ),
            read_controller__announced_pending_decision_ids=lambda: (
                self._announced_pending_decision_ids
            ),
            read_controller__answerable_decision_by_session=lambda: (
                self._answerable_decision_by_session
            ),
            read_controller__approval_view_is_detached=lambda: (
                self._approval_view_is_detached
            ),
            read_controller__bind_round_cancel_signal=lambda: (
                self._bind_round_cancel_signal
            ),
            read_controller__bind_visit_cancel_signal=lambda: (
                self._bind_visit_cancel_signal
            ),
            read_controller__buddy_sink=lambda: self._buddy_sink,
            read_controller__chat_create_session_grants=lambda: (
                self._chat_create_session_grants
            ),
            read_controller__chat_creation_record_locked=lambda: (
                self._chat_creation_record_locked
            ),
            read_controller__chat_creation_records=lambda: self._chat_creation_records,
            read_controller__chat_creation_revoked_runs=lambda: (
                self._chat_creation_revoked_runs
            ),
            read_controller__chat_start=lambda: self._chat_start,
            read_controller__console_answerable_decision_by_session=lambda: (
                self._console_answerable_decision_by_session
            ),
            read_controller__deliver_permission_summary=lambda: (
                self._deliver_permission_summary
            ),
            read_controller__disposed=lambda: self._disposed,
            read_controller__enrich_chat_create_confirm_payload=lambda: (
                self._enrich_chat_create_confirm_payload
            ),
            read_controller__forget_hidden_decision=lambda: (
                self._forget_hidden_decision
            ),
            read_controller__head_round_payload=lambda: self._head_round_payload,
            read_controller__interrupt_bell_enabled=lambda: (
                self._interrupt_bell_enabled
            ),
            read_controller__is_session_cancelled=lambda: self._is_session_cancelled,
            read_controller__marshal_pending_chat_create=lambda: (
                self._marshal_pending_chat_create
            ),
            read_controller__marshal_pending_decision_projection=lambda: (
                self._marshal_pending_decision_projection
            ),
            read_controller__maybe_fire_permission_summary=lambda: (
                self._maybe_fire_permission_summary
            ),
            read_controller__notify_run_hook_approval=lambda: (
                self._notify_run_hook_approval
            ),
            read_controller__observe_chat_creation_record=lambda: (
                self._observe_chat_creation_record
            ),
            read_controller__park_round_payload=lambda: self._park_round_payload,
            read_controller__parked_chat_create_payloads=lambda: (
                self._parked_chat_create_payloads
            ),
            read_controller__pending_approvals=lambda: self._pending_approvals,
            read_controller__pending_chat_create_lock=lambda: (
                self._pending_chat_create_lock
            ),
            read_controller__pending_chat_create_rounds=lambda: (
                self._pending_chat_create_rounds
            ),
            read_controller__pending_decision_order=lambda: (
                self._pending_decision_order
            ),
            read_controller__pending_round_kinds=lambda: self._pending_round_kinds,
            read_controller__permission_summary_worker=lambda: (
                self._permission_summary_worker
            ),
            read_controller__provider_messages_for_session=lambda: (
                self._provider_messages_for_session
            ),
            read_controller__publish_console_attention_change=lambda: (
                self._publish_console_attention_change
            ),
            read_controller__publish_pending_decision=lambda: (
                self._publish_pending_decision
            ),
            read_controller__question_bounces=lambda: self._question_bounces,
            read_controller__raw_shell_providers=lambda: getattr(
                self, "_raw_shell_providers", ()
            ),
            read_controller__record_cancelled_approval_decisions=lambda: (
                self._record_cancelled_approval_decisions
            ),
            read_controller__refresh_answerable_decision=lambda: (
                self._refresh_answerable_decision
            ),
            read_controller__remount_head=lambda: self._remount_head,
            read_controller__remount_parked_chat_create=lambda: (
                self._remount_parked_chat_create
            ),
            read_controller__remount_parked_skill_install=lambda: (
                self._remount_parked_skill_install
            ),
            read_controller__remount_parked_skill_script=lambda: (
                self._remount_parked_skill_script
            ),
            read_controller__remount_session_kinds=lambda: self._remount_session_kinds,
            read_controller__reproject_pending_decision_for_session=lambda: (
                self._reproject_pending_decision_for_session
            ),
            read_controller__resolve_ask_user_timeout_seconds=lambda: (
                self._resolve_ask_user_timeout_seconds
            ),
            read_controller__resolve_mcp_approval_timeout_seconds=lambda: (
                self._resolve_mcp_approval_timeout_seconds
            ),
            read_controller__revoke_chat_create_rounds=lambda: (
                self._revoke_chat_create_rounds
            ),
            read_controller__run_hooks_engine=lambda: self._run_hooks_engine,
            read_controller__session_close_generations=lambda: (
                self._session_close_generations
            ),
            read_controller__summary_tail_messages=lambda: self._summary_tail_messages,
            read_controller__unpark_round_payload=lambda: self._unpark_round_payload,
            read_controller_add_pending_round=lambda: self.add_pending_round,
            read_controller_announce_hidden_decision=lambda: (
                self.announce_hidden_decision
            ),
            read_controller_app=lambda: self.app,
            read_controller_ask_user_timeout_seconds=lambda: (
                self.ask_user_timeout_seconds
            ),
            read_controller_chat_create_confirm_timeout_seconds=lambda: (
                self.chat_create_confirm_timeout_seconds
            ),
            read_controller_decision_monotonic_clock=lambda: (
                self.decision_monotonic_clock
            ),
            read_controller_discard_pending_round=lambda: self.discard_pending_round,
            read_controller_expire_pending_decisions=lambda: (
                self.expire_pending_decisions
            ),
            read_controller_mcp_approval_timeout_seconds=lambda: (
                self.mcp_approval_timeout_seconds
            ),
            read_controller_on_console_attention_changed=lambda: (
                self.on_console_attention_changed
            ),
            read_controller_on_pending_rounds_changed=lambda: (
                self.on_pending_rounds_changed
            ),
            read_controller_park_pending_approval=lambda: self.park_pending_approval,
            read_controller_pending_decision_projection=lambda: (
                self.pending_decision_projection
            ),
            read_controller_project_pending_decision_for_active_session=lambda: (
                self.project_pending_decision_for_active_session
            ),
            read_controller_remount_pending_approval_for_active_session=lambda: (
                self.remount_pending_approval_for_active_session
            ),
            read_controller_set_answerable_decision=lambda: (
                self.set_answerable_decision
            ),
            read_controller_set_pending_approval=lambda: self.set_pending_approval,
            read_controller_set_pending_chat_create=lambda: (
                self.set_pending_chat_create
            ),
            read_controller_set_pending_decision=lambda: self.set_pending_decision,
            read_controller_set_pending_question=lambda: self.set_pending_question,
            read_controller_set_pending_skill_install=lambda: (
                self.set_pending_skill_install
            ),
            read_controller_set_pending_skill_script=lambda: (
                self.set_pending_skill_script
            ),
            read_controller_set_pending_worktree_merge=lambda: (
                self.set_pending_worktree_merge
            ),
            read_controller_set_task_panel=lambda: self.set_task_panel,
            read_controller_skill_install_confirm_timeout_seconds=lambda: (
                self.skill_install_confirm_timeout_seconds
            ),
            read_controller_skill_script_confirm_timeout_seconds=lambda: (
                self.skill_script_confirm_timeout_seconds
            ),
            read_controller_store=lambda: self.store,
            read_controller_update_pending_approval_summary=lambda: (
                self.update_pending_approval_summary
            ),
            read_controller_worktree_merge_confirm_timeout_seconds=lambda: (
                self.worktree_merge_confirm_timeout_seconds
            ),
            write_controller__pending_decision_order=write_controller__pending_decision_order,
            read_global_ASK_USER_TIMEOUT_ENV_VAR=lambda: ASK_USER_TIMEOUT_ENV_VAR,
            read_global_Any=lambda: Any,
            read_global_ApprovalDecisions=lambda: ApprovalDecisions,
            read_global_CONSOLE_PENDING_APPROVAL_KIND=lambda: (
                CONSOLE_PENDING_APPROVAL_KIND
            ),
            read_global_CONSOLE_PENDING_CHAT_CREATE_KIND=lambda: (
                CONSOLE_PENDING_CHAT_CREATE_KIND
            ),
            read_global_ConsolePendingDecisionProjection=lambda: (
                ConsolePendingDecisionProjection
            ),
            read_global_INTERRUPT_BELL_ENV_VAR=lambda: INTERRUPT_BELL_ENV_VAR,
            read_global_Mapping=lambda: Mapping,
            read_global_ToolExecutionPolicy=lambda: ToolExecutionPolicy,
            read_global_UNRESOLVED_DENIED_DECISION=lambda: UNRESOLVED_DENIED_DECISION,
            read_global__ChatCreationToken=lambda: _ChatCreationToken,
            read_global__DEFAULT_ASK_USER_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_ASK_USER_TIMEOUT_SECONDS
            ),
            read_global__DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS
            ),
            read_global__DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS
            ),
            read_global__DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS
            ),
            read_global__DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS
            ),
            read_global__DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS=lambda: (
                _DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS
            ),
            read_global__LEGACY_PENDING_APPROVAL_ROUND_ID=lambda: (
                _LEGACY_PENDING_APPROVAL_ROUND_ID
            ),
            read_global__MAX_TRACKED_QUESTION_BOUNCE_RUNS=lambda: (
                _MAX_TRACKED_QUESTION_BOUNCE_RUNS
            ),
            read_global__MCP_APPROVAL_POLL_SECONDS=lambda: _MCP_APPROVAL_POLL_SECONDS,
            read_global__REVOCATION_STAMPS=lambda: _REVOCATION_STAMPS,
            read_global__bool_or_none=lambda: _bool_or_none,
            read_global__build_approval_payload=lambda: _build_approval_payload,
            read_global__normalize_world_info_history=lambda: (
                _normalize_world_info_history
            ),
            read_global_contextlib=lambda: contextlib,
            read_global_current_run_actor=lambda: current_run_actor,
            read_global_current_run_id=lambda: current_run_id,
            read_global_escape_markup=lambda: escape_markup,
            read_global_get_cli_setting=lambda: get_cli_setting,
            read_global_get_runtime_config_snapshot=lambda: get_runtime_config_snapshot,
            read_global_logger=lambda: logger,
            read_global_os=lambda: os,
            read_global_threading=lambda: threading,
            read_global_time=lambda: time,
            read_global_uuid4=lambda: uuid4,
        )
        self._interrupt_host.after_remount["approval"] = (
            self._maybe_fire_permission_summary
        )

    def add_pending_round(
        self, session_id: str, round_id: str, kind: str = CONSOLE_PENDING_APPROVAL_KIND
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.add_pending_round(session_id, round_id, kind)

    def discard_pending_round(self, session_id: str, round_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.discard_pending_round(session_id, round_id)

    def _publish_console_attention_change(self) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._publish_console_attention_change()

    def has_pending_approval_round(self, session_id: str) -> bool:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.has_pending_approval_round(session_id)

    def pending_round_kinds(self, session_id: str) -> frozenset[str]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.pending_round_kinds(session_id)

    def pending_round_count(
        self, session_id: str, *, kind: str = CONSOLE_PENDING_APPROVAL_KIND
    ) -> int:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.pending_round_count(session_id, kind=kind)

    def set_run_pending_approval(self, session_id: str, pending: bool) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.set_run_pending_approval(session_id, pending)

    def request_mcp_approvals(
        self, pending: list[MCPPendingCall], *, session_id: str | None = None
    ) -> dict[str, str]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.request_mcp_approvals(
            pending, session_id=session_id
        )

    def _record_cancelled_approval_decisions(
        self, keys: list[str], call_by_key: dict[str, "MCPPendingCall"]
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._record_cancelled_approval_decisions(
            keys, call_by_key
        )

    def _marshal_pending_approval(
        self, payload: dict[str, Any] | None, *, fire_summary: bool = True
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._marshal_pending_approval(
            payload, fire_summary=fire_summary
        )

    def _maybe_fire_permission_summary(self, payload: dict[str, Any]) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._maybe_fire_permission_summary(payload)

    def _permission_summary_worker(
        self, round_id: str, payload: dict[str, Any], resolution: object, tail: list
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._permission_summary_worker(
            round_id, payload, resolution, tail
        )

    def _summary_tail_messages(self, payload: dict[str, Any]) -> list:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._summary_tail_messages(payload)

    def _deliver_permission_summary(
        self, round_id: str, payload: dict[str, Any], text: str
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._deliver_permission_summary(round_id, payload, text)

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
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._publish_pending_decision(
            round_state=round_state,
            payload=payload,
            decision_type=decision_type,
            decision_id=decision_id,
            timeout_seconds=timeout_seconds,
            retained_store=retained_store,
        )

    def pending_decision_projection(
        self, session_id: str
    ) -> ConsolePendingDecisionProjection | None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.pending_decision_projection(session_id)

    def project_pending_decision_for_active_session(self) -> bool:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.project_pending_decision_for_active_session()

    def _reproject_pending_decision_for_session(self, session_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._reproject_pending_decision_for_session(session_id)

    def active_session_changed(self) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.active_session_changed()

    def _cancel_pending_decisions_for_session(self, session_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._cancel_pending_decisions_for_session(session_id)

    def _marshal_pending_decision_projection(self) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._marshal_pending_decision_projection()

    def set_answerable_decision(self, session_id: str, decision_id: str | None) -> bool:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.set_answerable_decision(session_id, decision_id)

    def _refresh_answerable_decision(self, session_id: str) -> str | None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._refresh_answerable_decision(session_id)

    def expire_pending_decisions(self) -> tuple[str, ...]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.expire_pending_decisions()

    @staticmethod
    def _head_round_payload_locked(
        store: dict[str, dict[str, Any]], session_id: str | None
    ) -> dict[str, Any] | None:
        """Forward to the documented InterruptRoundHost implementation."""
        from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost

        return InterruptRoundHost._head_round_payload_locked(store, session_id)

    def _park_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str, payload: dict[str, Any]
    ) -> bool:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._park_round_payload(store, round_id, payload)

    def _head_round_payload(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> dict[str, Any] | None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._head_round_payload(store, session_id)

    def _session_round_payloads(
        self, store: dict[str, dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._session_round_payloads(store, session_id)

    def _unpark_round_payload(
        self, store: dict[str, dict[str, Any]], round_id: str
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._unpark_round_payload(store, round_id)

    def _remount_head(
        self,
        store: dict[str, dict[str, Any]],
        setter: Callable[[dict[str, Any] | None], None] | None,
        session_id: str | None,
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._remount_head(store, setter, session_id)

    def _remount_session_kinds(self, session_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._remount_session_kinds(session_id)

    def on_console_view_visibility_changed(self, visible: bool) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.on_console_view_visibility_changed(visible)

    def remount_pending_approval_for_active_session(self) -> bool:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.remount_pending_approval_for_active_session()

    def _approval_view_is_detached(self) -> bool:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._approval_view_is_detached()

    def on_pending_rounds_changed(self, total: int, kind: str, raised: bool) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.on_pending_rounds_changed(total, kind, raised)

    def _interrupt_bell_enabled(self) -> bool:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._interrupt_bell_enabled()

    def announce_hidden_decision(self, session_id: str, kind: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.announce_hidden_decision(session_id, kind)

    def _announce_detached_approval(
        self, session_id: str, *, kind: str = CONSOLE_PENDING_APPROVAL_KIND
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._announce_detached_approval(session_id, kind=kind)

    def _announce_hidden_decision(
        self,
        decision_type: Literal["approval", "skill_install", "skill_script"],
        session_id: str,
        decision_id: str,
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._announce_hidden_decision(
            decision_type, session_id, decision_id
        )

    def _forget_hidden_decision(self, decision_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._forget_hidden_decision(decision_id)

    def _resolve_mcp_approval_timeout_seconds(self) -> float:
        return self._interrupt_host._resolve_mcp_approval_timeout_seconds()

    def _console_tool_kill_switch_reader(self) -> Callable[[], bool] | None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._console_tool_kill_switch_reader()

    def resolve_pending_approval(
        self, decisions: dict[str, str], *, round_id: str | None = None
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.resolve_pending_approval(
            decisions, round_id=round_id
        )

    def complete_definitive_tool(
        self, run_id: str, call_key: str, tool_name: str
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.complete_definitive_tool(
            run_id, call_key, tool_name
        )

    def complete_definitive_run(self, run_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.complete_definitive_run(run_id)

    def _discard_approval_rows_for_closing_session(self, session_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._discard_approval_rows_for_closing_session(
            session_id
        )

    def revoke_approval_rounds_for_run(self, run_id: str) -> int:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.revoke_approval_rounds_for_run(run_id)

    def revoke_raw_shell_authority(self) -> int:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.revoke_raw_shell_authority()

    def _revoke_tool_approval_rounds(self, run_id: str) -> list[tuple[str, str | None]]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._revoke_tool_approval_rounds(run_id)

    def _revoke_skill_script_rounds(self, run_id: str) -> list[tuple[str, str | None]]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._revoke_skill_script_rounds(run_id)

    def request_skill_install_confirm(
        self, url: str, *, session_id: str | None = None
    ) -> bool:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.request_skill_install_confirm(
            url, session_id=session_id
        )

    def _remount_parked_skill_install(self, session_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._remount_parked_skill_install(session_id)

    def _marshal_pending_skill_install(self, payload: dict[str, Any] | None) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._marshal_pending_skill_install(payload)

    def resolve_pending_skill_install(
        self, allow: bool, *, request_id: str | None = None
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.resolve_pending_skill_install(
            allow, request_id=request_id
        )

    def pending_skill_install_ids(self) -> list[str]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.pending_skill_install_ids()

    def request_skill_script_confirm(
        self, payload: dict[str, Any], *, session_id: str | None = None
    ) -> dict[str, bool]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.request_skill_script_confirm(
            payload, session_id=session_id
        )

    def _remount_parked_skill_script(self, session_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._remount_parked_skill_script(session_id)

    def _marshal_pending_skill_script(self, payload: dict[str, Any] | None) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._marshal_pending_skill_script(payload)

    def _marshal_task_panel(
        self, session_id: str, tasks: list[dict[str, object]]
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._marshal_task_panel(session_id, tasks)

    def _remount_task_panel(self, session_id: str | None) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._remount_task_panel(session_id)

    def resolve_pending_skill_script(
        self, allow: bool, remember: bool, request_id: str | None = None
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.resolve_pending_skill_script(
            allow, remember, request_id
        )

    def pending_skill_script_ids(self) -> list[str]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.pending_skill_script_ids()

    def _enrich_chat_create_confirm_payload(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._enrich_chat_create_confirm_payload(payload)

    def request_chat_create_confirm(
        self, payload: dict[str, Any], *, session_id: str | None = None
    ) -> dict[str, bool]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.request_chat_create_confirm(
            payload, session_id=session_id
        )

    def _remount_parked_chat_create(self, session_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._remount_parked_chat_create(session_id)

    def _marshal_pending_chat_create(self, payload: dict[str, Any] | None) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._marshal_pending_chat_create(payload)

    def resolve_pending_chat_create(
        self, allow: bool, remember: bool, request_id: str | None = None
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.resolve_pending_chat_create(
            allow, remember, request_id
        )

    def pending_chat_create_ids(self) -> list[str]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.pending_chat_create_ids()

    def _resolve_ask_user_timeout_seconds(self) -> float:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._resolve_ask_user_timeout_seconds()

    def request_user_questions(
        self, questions: list[dict[str, Any]], *, session_id: str | None = None
    ) -> dict[str, Any]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.request_user_questions(
            questions, session_id=session_id
        )

    def resolve_pending_question(
        self, answers: list[dict[str, Any]], request_id: str | None = None
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.resolve_pending_question(answers, request_id)

    def pending_question_ids(self) -> list[str]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.pending_question_ids()

    def _marshal_pending_question(self, payload: dict[str, Any] | None) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._marshal_pending_question(payload)

    def _remount_parked_question(self, session_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._remount_parked_question(session_id)

    @property
    def worktree_confirmation_enabled(self) -> bool:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.worktree_confirmation_enabled

    def request_worktree_merge_confirm(
        self,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
        operation_cancel_event: threading.Event | None = None,
    ) -> dict[str, bool]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.request_worktree_merge_confirm(
            payload,
            session_id=session_id,
            operation_cancel_event=operation_cancel_event,
        )

    def _remount_parked_worktree_merge(self, session_id: str) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._remount_parked_worktree_merge(session_id)

    def _marshal_pending_worktree_merge(self, payload: dict[str, Any] | None) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._marshal_pending_worktree_merge(payload)

    def resolve_pending_worktree_merge(
        self, allow: bool, *, request_id: str | None = None
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.resolve_pending_worktree_merge(
            allow, request_id=request_id
        )

    def pending_worktree_merge_ids(self) -> list[str]:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host.pending_worktree_merge_ids()

    def _notify_run_hook_approval(
        self, kind: str, payload: dict[str, Any], state: dict[str, Any]
    ) -> None:
        """Forward to the documented InterruptRoundHost implementation."""
        return self._interrupt_host._notify_run_hook_approval(kind, payload, state)
