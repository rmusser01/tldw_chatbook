"""Explicit test-only wiring for pre-ADR220 minimal and bare fixtures.

No production module imports this helper. Absent fake dependencies stay None;
real methods/globals are read at invocation. Native fixture aliases are retained.
"""

from tldw_chatbook.Chat import console_chat_controller as ccc
from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost


def make_interrupt_host(seams):
    host = InterruptRoundHost(
        read_controller__active_assistant_message_ids=lambda: getattr(
            seams, "_active_assistant_message_ids", None
        ),
        read_controller__advance_lifecycle_revision=lambda: getattr(
            seams, "_advance_lifecycle_revision", None
        ),
        read_controller__agent_bridge=lambda: getattr(seams, "_agent_bridge", None),
        read_controller__announce_detached_approval=lambda: getattr(
            seams, "_announce_detached_approval", None
        ),
        read_controller__announce_hidden_decision=lambda: getattr(
            seams, "_announce_hidden_decision", None
        ),
        read_controller__announced_pending_decision_ids=lambda: getattr(
            seams, "_announced_pending_decision_ids", None
        ),
        read_controller__answerable_decision_by_session=lambda: getattr(
            seams, "_answerable_decision_by_session", None
        ),
        read_controller__approval_view_is_detached=lambda: getattr(
            seams, "_approval_view_is_detached", None
        ),
        read_controller__bind_round_cancel_signal=lambda: getattr(
            seams, "_bind_round_cancel_signal", None
        ),
        read_controller__bind_visit_cancel_signal=lambda: getattr(
            seams, "_bind_visit_cancel_signal", None
        ),
        read_controller__buddy_sink=lambda: getattr(seams, "_buddy_sink", None),
        read_controller__chat_create_session_grants=lambda: getattr(
            seams, "_chat_create_session_grants", None
        ),
        read_controller__chat_creation_record_locked=lambda: getattr(
            seams, "_chat_creation_record_locked", None
        ),
        read_controller__chat_creation_records=lambda: getattr(
            seams, "_chat_creation_records", None
        ),
        read_controller__chat_creation_revoked_runs=lambda: getattr(
            seams, "_chat_creation_revoked_runs", None
        ),
        read_controller__chat_start=lambda: getattr(seams, "_chat_start", None),
        read_controller__console_answerable_decision_by_session=lambda: getattr(
            seams, "_console_answerable_decision_by_session", None
        ),
        read_controller__deliver_permission_summary=lambda: getattr(
            seams, "_deliver_permission_summary", None
        ),
        read_controller__disposed=lambda: getattr(seams, "_disposed", None),
        read_controller__enrich_chat_create_confirm_payload=lambda: getattr(
            seams, "_enrich_chat_create_confirm_payload", None
        ),
        read_controller__forget_hidden_decision=lambda: getattr(
            seams, "_forget_hidden_decision", None
        ),
        read_controller__head_round_payload=lambda: getattr(
            seams, "_head_round_payload", None
        ),
        read_controller__interrupt_bell_enabled=lambda: getattr(
            seams, "_interrupt_bell_enabled", None
        ),
        read_controller__is_session_cancelled=lambda: getattr(
            seams, "_is_session_cancelled", None
        ),
        read_controller__marshal_pending_chat_create=lambda: getattr(
            seams, "_marshal_pending_chat_create", None
        ),
        read_controller__marshal_pending_decision_projection=lambda: getattr(
            seams, "_marshal_pending_decision_projection", None
        ),
        read_controller__maybe_fire_permission_summary=lambda: getattr(
            seams, "_maybe_fire_permission_summary", None
        ),
        read_controller__notify_run_hook_approval=lambda: getattr(
            seams, "_notify_run_hook_approval", None
        ),
        read_controller__observe_chat_creation_record=lambda: getattr(
            seams, "_observe_chat_creation_record", None
        ),
        read_controller__park_round_payload=lambda: getattr(
            seams, "_park_round_payload", None
        ),
        read_controller__parked_chat_create_payloads=lambda: getattr(
            seams, "_parked_chat_create_payloads", None
        ),
        read_controller__pending_approvals=lambda: getattr(
            seams, "_pending_approvals", None
        ),
        read_controller__pending_chat_create_lock=lambda: getattr(
            seams, "_pending_chat_create_lock", None
        ),
        read_controller__pending_chat_create_rounds=lambda: getattr(
            seams, "_pending_chat_create_rounds", None
        ),
        read_controller__pending_decision_order=lambda: getattr(
            seams, "_pending_decision_order", None
        ),
        read_controller__pending_round_kinds=lambda: getattr(
            seams, "_pending_round_kinds", None
        ),
        read_controller__permission_summary_worker=lambda: getattr(
            seams, "_permission_summary_worker", None
        ),
        read_controller__provider_messages_for_session=lambda: getattr(
            seams, "_provider_messages_for_session", None
        ),
        read_controller__publish_console_attention_change=lambda: getattr(
            seams, "_publish_console_attention_change", None
        ),
        read_controller__publish_pending_decision=lambda: getattr(
            seams, "_publish_pending_decision", None
        ),
        read_controller__question_bounces=lambda: getattr(
            seams, "_question_bounces", None
        ),
        read_controller__raw_shell_providers=lambda: getattr(
            seams, "_raw_shell_providers", None
        ),
        read_controller__record_cancelled_approval_decisions=lambda: getattr(
            seams, "_record_cancelled_approval_decisions", None
        ),
        read_controller__refresh_answerable_decision=lambda: getattr(
            seams, "_refresh_answerable_decision", None
        ),
        read_controller__remount_head=lambda: getattr(seams, "_remount_head", None),
        read_controller__remount_parked_chat_create=lambda: getattr(
            seams, "_remount_parked_chat_create", None
        ),
        read_controller__remount_parked_skill_install=lambda: getattr(
            seams, "_remount_parked_skill_install", None
        ),
        read_controller__remount_parked_skill_script=lambda: getattr(
            seams, "_remount_parked_skill_script", None
        ),
        read_controller__remount_session_kinds=lambda: getattr(
            seams, "_remount_session_kinds", None
        ),
        read_controller__reproject_pending_decision_for_session=lambda: getattr(
            seams, "_reproject_pending_decision_for_session", None
        ),
        read_controller__resolve_ask_user_timeout_seconds=lambda: getattr(
            seams, "_resolve_ask_user_timeout_seconds", None
        ),
        read_controller__resolve_mcp_approval_timeout_seconds=lambda: getattr(
            seams, "_resolve_mcp_approval_timeout_seconds", None
        ),
        read_controller__revoke_chat_create_rounds=lambda: getattr(
            seams, "_revoke_chat_create_rounds", None
        ),
        read_controller__run_hooks_engine=lambda: getattr(
            seams, "_run_hooks_engine", None
        ),
        read_controller__session_close_generations=lambda: getattr(
            seams, "_session_close_generations", None
        ),
        read_controller__summary_tail_messages=lambda: getattr(
            seams, "_summary_tail_messages", None
        ),
        read_controller__unpark_round_payload=lambda: getattr(
            seams, "_unpark_round_payload", None
        ),
        read_controller_add_pending_round=lambda: getattr(
            seams, "add_pending_round", None
        ),
        read_controller_announce_hidden_decision=lambda: getattr(
            seams, "announce_hidden_decision", None
        ),
        read_controller_app=lambda: getattr(seams, "app", None),
        read_controller_ask_user_timeout_seconds=lambda: getattr(
            seams, "ask_user_timeout_seconds", None
        ),
        read_controller_chat_create_confirm_timeout_seconds=lambda: getattr(
            seams, "chat_create_confirm_timeout_seconds", None
        ),
        read_controller_decision_monotonic_clock=lambda: getattr(
            seams, "decision_monotonic_clock", None
        ),
        read_controller_discard_pending_round=lambda: getattr(
            seams, "discard_pending_round", None
        ),
        read_controller_expire_pending_decisions=lambda: getattr(
            seams, "expire_pending_decisions", None
        ),
        read_controller_mcp_approval_timeout_seconds=lambda: getattr(
            seams, "mcp_approval_timeout_seconds", None
        ),
        read_controller_on_console_attention_changed=lambda: getattr(
            seams, "on_console_attention_changed", None
        ),
        read_controller_on_pending_rounds_changed=lambda: getattr(
            seams, "on_pending_rounds_changed", None
        ),
        read_controller_park_pending_approval=lambda: getattr(
            seams, "park_pending_approval", None
        ),
        read_controller_pending_decision_projection=lambda: getattr(
            seams, "pending_decision_projection", None
        ),
        read_controller_project_pending_decision_for_active_session=lambda: getattr(
            seams, "project_pending_decision_for_active_session", None
        ),
        read_controller_remount_pending_approval_for_active_session=lambda: getattr(
            seams, "remount_pending_approval_for_active_session", None
        ),
        read_controller_set_answerable_decision=lambda: getattr(
            seams, "set_answerable_decision", None
        ),
        read_controller_set_pending_approval=lambda: getattr(
            seams, "set_pending_approval", None
        ),
        read_controller_set_pending_chat_create=lambda: getattr(
            seams, "set_pending_chat_create", None
        ),
        read_controller_set_pending_decision=lambda: getattr(
            seams, "set_pending_decision", None
        ),
        read_controller_set_pending_question=lambda: getattr(
            seams, "set_pending_question", None
        ),
        read_controller_set_pending_skill_install=lambda: getattr(
            seams, "set_pending_skill_install", None
        ),
        read_controller_set_pending_skill_script=lambda: getattr(
            seams, "set_pending_skill_script", None
        ),
        read_controller_set_pending_worktree_merge=lambda: getattr(
            seams, "set_pending_worktree_merge", None
        ),
        read_controller_set_task_panel=lambda: getattr(seams, "set_task_panel", None),
        read_controller_skill_install_confirm_timeout_seconds=lambda: getattr(
            seams, "skill_install_confirm_timeout_seconds", None
        ),
        read_controller_skill_script_confirm_timeout_seconds=lambda: getattr(
            seams, "skill_script_confirm_timeout_seconds", None
        ),
        read_controller_store=lambda: getattr(seams, "store", None),
        read_controller_update_pending_approval_summary=lambda: getattr(
            seams, "update_pending_approval_summary", None
        ),
        read_controller_worktree_merge_confirm_timeout_seconds=lambda: getattr(
            seams, "worktree_merge_confirm_timeout_seconds", None
        ),
        write_controller__pending_decision_order=lambda value: setattr(
            seams, "_pending_decision_order", value
        ),
        read_global_ASK_USER_TIMEOUT_ENV_VAR=lambda: ccc.ASK_USER_TIMEOUT_ENV_VAR,
        read_global_ApprovalDecisions=lambda: ccc.ApprovalDecisions,
        read_global_CONSOLE_PENDING_APPROVAL_KIND=lambda: (
            ccc.CONSOLE_PENDING_APPROVAL_KIND
        ),
        read_global_CONSOLE_PENDING_CHAT_CREATE_KIND=lambda: (
            ccc.CONSOLE_PENDING_CHAT_CREATE_KIND
        ),
        read_global_ConsolePendingDecisionProjection=lambda: (
            ccc.ConsolePendingDecisionProjection
        ),
        read_global_INTERRUPT_BELL_ENV_VAR=lambda: ccc.INTERRUPT_BELL_ENV_VAR,
        read_global_ToolExecutionPolicy=lambda: ccc.ToolExecutionPolicy,
        read_global_UNRESOLVED_DENIED_DECISION=lambda: ccc.UNRESOLVED_DENIED_DECISION,
        read_global__ChatCreationToken=lambda: ccc._ChatCreationToken,
        read_global__DEFAULT_ASK_USER_TIMEOUT_SECONDS=lambda: (
            ccc._DEFAULT_ASK_USER_TIMEOUT_SECONDS
        ),
        read_global__DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS=lambda: (
            ccc._DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS
        ),
        read_global__DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS=lambda: (
            ccc._DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS
        ),
        read_global__DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS=lambda: (
            ccc._DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS
        ),
        read_global__DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS=lambda: (
            ccc._DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS
        ),
        read_global__DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS=lambda: (
            ccc._DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS
        ),
        read_global__LEGACY_PENDING_APPROVAL_ROUND_ID=lambda: (
            ccc._LEGACY_PENDING_APPROVAL_ROUND_ID
        ),
        read_global__MAX_TRACKED_QUESTION_BOUNCE_RUNS=lambda: (
            ccc._MAX_TRACKED_QUESTION_BOUNCE_RUNS
        ),
        read_global__MCP_APPROVAL_POLL_SECONDS=lambda: ccc._MCP_APPROVAL_POLL_SECONDS,
        read_global__REVOCATION_STAMPS=lambda: ccc._REVOCATION_STAMPS,
        read_global__bool_or_none=lambda: ccc._bool_or_none,
        read_global__build_approval_payload=lambda: ccc._build_approval_payload,
        read_global__normalize_world_info_history=lambda: (
            ccc._normalize_world_info_history
        ),
        read_global_contextlib=lambda: ccc.contextlib,
        read_global_current_run_actor=lambda: ccc.current_run_actor,
        read_global_current_run_id=lambda: ccc.current_run_id,
        read_global_escape_markup=lambda: ccc.escape_markup,
        read_global_get_cli_setting=lambda: ccc.get_cli_setting,
        read_global_get_runtime_config_snapshot=lambda: ccc.get_runtime_config_snapshot,
        read_global_logger=lambda: ccc.logger,
        read_global_os=lambda: ccc.os,
        read_global_threading=lambda: ccc.threading,
        read_global_time=lambda: ccc.time,
        read_global_uuid4=lambda: ccc.uuid4,
    )
    # The historical assertions inspect the fake, never a production receiver.
    host._seams = seams
    if hasattr(seams, "_approval_state_lock"):
        host.lock = seams._approval_state_lock
    if hasattr(seams, "_pending_approval_rounds"):
        host.registries["approval"] = seams._pending_approval_rounds
    return host
