"""Architecture evidence for Console decomposition Wave 6 (TASK-3070.1)."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCREEN_PATH = _REPO_ROOT / "tldw_chatbook/UI/Screens/chat_screen.py"
IMPLEMENTATION_BASE = "bed39af6b004e4db86218fad01d2ea515b332135"
BASELINE_LINES = 22_338
BASELINE_METHODS = 705
POST_IMAGE_IMPLEMENTATION_BASE = "8d806b71d9c5ae7ed333ccb42780f6b2ea68acd0"
POST_IMAGE_BASELINE_LINES = 22_172
POST_IMAGE_BASELINE_METHODS = 712
TASK10_SCREEN_LINE_CEILING = 20_943
LINE_OVERAGE = 4_445
METHOD_OVERAGE = 119
DESCRIPTOR_LINE_BUDGET = 64
FLEET_TASK0_IMPLEMENTATION_BASE = "d4f3f97763ddf3fa46eeb35ae9473827e72695bc"
FLEET_TASK0_BASE_SCREEN_LINES = 20_428
FLEET_TASK0_BASE_METHODS = 653
FLEET_TASK0_DEFINITION_LINES = 421
FLEET_TASK0_MAX_SCREEN_LINES = 20_007
FLEET_TASK0_MAX_METHODS = 637
FLEET_FINAL_REBASE_BASE = "02cd80b33004305765b5cd91b3d264aa3664596e"
FLEET_FINAL_REBASE_BASE_SCREEN_LINES = 20_486
FLEET_FINAL_REBASE_BASE_METHODS = 656
FLEET_FINAL_REBASE_DEFINITION_LINES = 421
FLEET_FINAL_REBASE_MAX_SCREEN_LINES = 20_065
FLEET_FINAL_REBASE_MAX_METHODS = 640
FLEET_FINAL_REBASE_ADDED_SCREEN_METHODS = frozenset(
    {
        "_console_inspector_active",
        "_request_console_context_allocation_reconcile",
        "_request_console_live_work_reconcile",
    }
)
TASK_3070_9_DESIGN_BASE = "0da426e1e4c2846f13671690b8f981f72e673359"
TASK_3070_9_TASK0_IMPLEMENTATION_BASE = "ede2162143331e324c44832ff6a3910e1185cf58"
TASK_3070_9_TASK0_BASE_SCREEN_LINES = 19_995
TASK_3070_9_TASK0_BASE_METHODS = 640
TASK_3070_9_DEFINITION_LINES = 328
TASK_3070_9_TASK0_MAX_SCREEN_LINES = 19_667
TASK_3070_9_TASK0_MAX_METHODS = 632
TASK_3070_9_FAMILY_SHA256 = (
    "3a2968883c63dc89de430ee72b40444ebd97fb9b36c1dbc8a46e19d063a715ee"
)
TASK_3070_9_FAMILY_NAMES = (
    "_first_chat_defaults_match",
    "_current_first_chat_defaults",
    "eligible_console_first_chat_session_id",
    "_release_first_chat_claim",
    "_log_first_chat_handoff_exception",
    "_resync_console_after_first_chat_rollback",
    "_resync_mounted_console_after_first_chat_rollback",
    "consume_pending_console_first_chat_intent",
)
FIRST_CHAT_CONTROLLER_CALLBACKS = frozenset(
    {
        "screen_mounted_accessor",
        "first_chat_presentation_snapshot",
        "apply_first_chat_control_selection",
        "restore_first_chat_focus",
    }
)
FIRST_CHAT_PRESENTATION_ATTRIBUTES = frozenset(
    {
        "_screen_mounted_accessor",
        "_first_chat_presentation_snapshot_fn",
        "_apply_first_chat_control_selection_fn",
        "_restore_first_chat_focus_fn",
    }
)
FLEET_CONTROLLER_CALLBACKS = frozenset(
    {
        "pending_handoffs_accessor",
        "ensure_chat_store",
        "ensure_chat_controller",
        "activate_workspace_for_session",
        "switch_chat_session",
        "schedule_native_console_sync",
        "ensure_agent_bridge",
        "wire_wake_coordinator",
        "seed_wake_from_marks",
        "retry_wake_soon",
        "wake_has_pending",
        "wake_delivering_conversation_id",
        "displayed_composer_draft_accessor",
        "screen_displayed_accessor",
        "screen_mounted_accessor",
        "active_session_id_accessor",
        "chat_sessions_accessor",
        "defer_on_message_pump",
        "start_transcript_sync_timer",
        "transcript_sync_timer_active",
        "sync_native_console_ui",
        "create_interval",
        "record_timer_created",
        "record_timer_stopped",
        "chat_controller_available",
        "fleet_has_unsettled_children",
        "run_marker_for_session",
        "fleet_teardown_split",
        "leave_runtime",
        "stage_teardown_notices",
        "fleet_unseen_revision_accessor",
        "read_fleet_unseen_ids",
        "clear_fleet_unseen",
    }
)


@dataclass(frozen=True)
class Wave6Group:
    """Describe one reviewed Wave 6 ownership family.

    The record locks the implementation-base definitions, their destination,
    and the conservative residue retained on ``ChatScreen``.
    """

    target_path: str
    target_class: str
    owner_name: str
    moved: frozenset[str]
    delegates: frozenset[str] = frozenset()
    stays: frozenset[str] = frozenset()
    deleted: frozenset[str] = frozenset()
    raw_lines: int = 0
    residue_lines: int = 0
    source_revision: str = IMPLEMENTATION_BASE

    @property
    def raw_names(self) -> frozenset[str]:
        """All implementation-base definitions counted by this family."""
        return self.moved | self.delegates | self.stays | self.deleted

    @property
    def removed_methods(self) -> int:
        """Direct ChatScreen methods removed when this family completes."""
        return len(self.moved | self.deleted)


WAVE6_GROUPS = {
    "image": Wave6Group(
        target_path="tldw_chatbook/UI/Console_Modules/image.py",
        target_class="ConsoleImageController",
        owner_name="_image",
        moved=frozenset(
            {
                "_build_console_image_specs",
                "_extend_specs_with_remote_images",
                "_fetch_remote_transcript_image",
                "_build_generation_card_specs",
                "_pending_console_generation_card_images",
                "_console_imagegen_inflight_sessions",
                "_console_imagegen_inflight_message_ids",
                "_console_generate_image_conversation_pairs",
                "_console_generate_image_llm_context_options",
                "_h3_image_edit_registry",
                "_h3_reference_snapshot",
                "_h3_reference_from_snapshot",
                "_filter_h3_attachment_from_app_stash",
                "_h3_origin_screen_is_live",
                "_cleanup_h3_completion_in_store",
                "_reconcile_h3_image_edit_completions",
                "_merge_h3_failure_notice_in_store",
                "_settle_current_h3_outcome",
                "_schedule_current_h3_settlement",
                "_append_h3_image_edit_error",
                "_run_h3_image_edit_command",
                "_regenerate_console_generation_variant",
                "_select_console_generation_variant",
                "_keep_console_generation_variant",
                "_handle_console_toggle_image_view",
            }
        ),
        delegates=frozenset({"_console_command_generate_image"}),
        stays=frozenset(
            {
                "_ensure_console_image_view",
                "_console_generation_browse",
                "_prep_console_images",
                "_open_console_generate_image_modal",
                "_paste_console_generate_image_command",
            }
        ),
        raw_lines=1_335,
        residue_lines=79,
    ),
    "video": Wave6Group(
        target_path="tldw_chatbook/UI/Console_Modules/video.py",
        target_class="ConsoleVideoController",
        owner_name="_video",
        moved=frozenset(
            {
                "_console_videogen_inflight_sessions",
                "_console_videogen_cancel_events",
                "_ensure_console_video_store",
                "_build_video_card_specs",
                "_video_storage_message_id",
                "_pending_console_video_artifacts",
                "_owns_pending_console_video",
                "_close_pending_console_video",
                "_register_console_video_publication_gate",
                "_release_console_video_publication_gate",
                "_begin_pending_console_video_operation",
                "_end_pending_console_video_operation",
                "_await_shielded_console_video_task",
                "_run_pending_console_video_operation",
                "_run_console_video_generation_operation",
                "_drain_pending_console_videos",
                "_external_video_target_identity",
                "_external_video_stat_identity",
                "_external_video_cleanup_identity",
                "_external_video_parent_identity",
                "_require_external_video_pinned_capabilities",
                "_external_video_precommit_check",
                "_copy_pending_video_external",
                "_retry_pending_console_video",
                "_save_pending_console_video_external",
                "_normalize_pending_video_target",
                "_resolve_generated_video_outcome",
                "_persist_generated_video_tuple",
                "_play_console_video",
                "_save_console_video_copy",
                "_regenerate_console_video_message",
            }
        ),
        delegates=frozenset(
            {"_console_command_generate_video", "_console_command_stream_video"}
        ),
        raw_lines=1_292,
        residue_lines=10,
    ),
    "browser": Wave6Group(
        target_path="tldw_chatbook/UI/Console_Modules/workspace.py",
        target_class="ConsoleWorkspaceController",
        owner_name="_workspace",
        moved=frozenset(
            {
                "_start_console_conversation_browser_search",
                "_console_browser_row_key",
                "_console_browser_row_scope_copy",
                "_console_browser_row_matches_query",
                "_filter_console_browser_rows_for_query",
                "_find_console_browser_row",
                "_console_browser_display_identity",
                "_starred_console_conversation_ids",
                "_apply_console_browser_star_state",
                "_native_console_browser_rows",
                "_membership_console_browser_rows",
                "_persisted_console_browser_rows",
                "_invalidate_console_persisted_rows_cache",
                "_sync_persisted_console_browser_rows",
                "_compute_persisted_console_browser_rows",
                "_merge_console_browser_rows",
                "_current_console_browser_rows",
                "_refresh_console_conversation_browser_search",
                "_refresh_console_conversation_browser_after_selection",
                "_with_console_conversation_browser_state",
                "_console_browser_unseen_marker",
            }
        ),
        delegates=frozenset({"on_console_workspace_conversation_search_changed"}),
        raw_lines=959,
        residue_lines=5,
        source_revision=POST_IMAGE_IMPLEMENTATION_BASE,
    ),
    "retrieval": Wave6Group(
        target_path="tldw_chatbook/UI/Console_Modules/retrieval.py",
        target_class="ConsoleRetrievalController",
        owner_name="_retrieval",
        moved=frozenset(
            {
                "_capture_console_staged_rag",
                "_build_console_retrieval_scope_state",
                "_console_retrieval_scope_run_recipe_count",
                "_resolve_console_effective_scope_state",
                "_refresh_console_effective_scope_and_sync",
                "_warm_console_effective_scope_cache_if_stale",
                "_read_console_retrieval_scope",
                "_write_console_retrieval_scope",
                "_console_scope_picker_listers",
                "_apply_console_retrieval_scope_save",
                "_console_rag_source_status",
                "_active_console_dictionary_scope_ids",
                "_refresh_active_dictionaries_summary_if_scope_changed",
                "refresh_active_dictionaries_summary",
                "_active_console_world_book_scope_ids",
                "refresh_active_world_books_summary",
                "_refresh_active_world_books_summary_if_scope_changed",
                "_console_dictionary_inspector_rows",
                "_console_world_book_inspector_rows",
                "_console_dictionary_inspector_actions",
                "_console_world_book_inspector_actions",
                "_console_library_rag_scope_label",
                "_stage_console_library_rag_launch",
                "_maybe_auto_retrieve_for_send",
                "_apply_console_rag_settings_choice",
                "_resolve_console_library_rag_scope",
                "_apply_console_library_rag_search_outcome",
                "_rag_service_still_initializing",
                "_notify_console_auto_rag_scope_empty",
                "_notify_auto_rag_degraded",
                "_notify_console_auto_rag",
                "_clear_console_auto_rag_placeholder",
            }
        ),
        delegates=frozenset(
            {
                "_persist_console_rag_auto_retrieve_on_send",
                "_execute_console_library_rag_search",
            }
        ),
        raw_lines=992,
        residue_lines=10,
    ),
    "skill": Wave6Group(
        target_path="tldw_chatbook/UI/Console_Modules/skill.py",
        target_class="ConsoleSkillController",
        owner_name="_skill",
        moved=frozenset(
            {
                "_fetch_console_skill_context",
                "_console_skill_trusted_candidates_from_context",
                "_console_skill_blocked_summaries",
                "_refresh_console_skill_candidates",
                "_split_console_skill_name_args",
                "_console_skill_blocked_match_response",
                "_append_skill_refuse_row",
                "_set_console_pending_skill_install",
                "_set_console_pending_skill_script",
            }
        ),
        delegates=frozenset(
            {
                "_console_command_skills",
                "handle_console_skill_install_decided",
                "handle_console_skill_script_decided",
            }
        ),
        deleted=frozenset(
            {
                "_console_skill_search",
                "_console_command_run_skill",
                "_run_resolved_console_skill",
                "_open_console_skill_picker",
            }
        ),
        raw_lines=339,
        residue_lines=15,
    ),
    "character": Wave6Group(
        target_path="tldw_chatbook/UI/Console_Modules/character.py",
        target_class="ConsoleCharacterController",
        owner_name="_character",
        moved=frozenset(
            {
                "_console_character_picker_options",
                "_current_console_rail_conversation_id",
                "_current_console_rail_character_id",
                "_current_console_rail_character_name",
                "_fetch_character_card_for_avatar",
                "_apply_console_character_choice_async",
                "_refresh_active_character_avatar_if_scope_changed",
            }
        ),
        deleted=frozenset({"_fetch_expression_image_bytes"}),
        raw_lines=281,
    ),
    "fleet": Wave6Group(
        target_path="tldw_chatbook/UI/Console_Modules/fleet.py",
        target_class="ConsoleFleetLifecycleController",
        owner_name="_fleet",
        moved=frozenset(
            {
                "consume_pending_console_fleet_completion",
                "_claim_console_fleet_wake_marks",
                "_console_wake_user_priority",
                "_console_wake_probe_composer",
                "_console_screen_displayed",
                "_console_wake_conversation_in_view",
                "_poke_console_wake_retry",
                "_on_console_wake_delivery_started",
                "_console_wake_turn_active",
                "_record_console_fleet_teardown",
                "_console_fleet_unseen_ids",
                "_console_run_marker_with_unseen",
                "_console_fleet_survivors_live",
                "_maybe_start_console_fleet_survivor_tick",
                "_stop_console_fleet_survivor_tick",
                "_console_fleet_survivor_tick",
            }
        ),
        raw_lines=401,
        source_revision=POST_IMAGE_IMPLEMENTATION_BASE,
    ),
    "first_chat": Wave6Group(
        target_path="tldw_chatbook/UI/Console_Modules/session.py",
        target_class="ConsoleSessionController",
        owner_name="_session",
        moved=frozenset(
            {
                "_first_chat_defaults_match",
                "_current_first_chat_defaults",
                "eligible_console_first_chat_session_id",
                "_release_first_chat_claim",
                "_log_first_chat_handoff_exception",
                "_resync_console_after_first_chat_rollback",
                "_resync_mounted_console_after_first_chat_rollback",
                "consume_pending_console_first_chat_intent",
            }
        ),
        raw_lines=328,
        source_revision=POST_IMAGE_IMPLEMENTATION_BASE,
    ),
    "auto_speak": Wave6Group(
        target_path="tldw_chatbook/UI/Console_Modules/hands_free.py",
        target_class="ConsoleHandsFreeController",
        owner_name="_hands_free",
        moved=frozenset(
            {
                "_resolve_console_auto_speak_destination",
                "_sync_console_auto_speak_controls",
            }
        ),
        delegates=frozenset(
            {
                "on_console_auto_speak_changed",
                "on_console_auto_speak_resume_requested",
                "on_console_auto_speak_retry_requested",
            }
        ),
        raw_lines=48,
        residue_lines=15,
        source_revision=POST_IMAGE_IMPLEMENTATION_BASE,
    ),
}

COMPATIBILITY_TARGETS = {
    "_image": frozenset(
        {
            "_imagegen_inflight_sessions",
            "_imagegen_inflight_message_ids",
            "_console_h3_ui_generations",
        }
    ),
    "_video": frozenset(
        {
            "_console_videogen_inflight",
            "_console_videogen_cancels",
            "_console_video_store",
            "_pending_video_artifacts",
            "_pending_video_artifacts_closed",
            "_pending_video_operation_cancels",
            "_pending_video_active_operations",
            "_pending_video_deferred_closes",
        }
    ),
    "_workspace": frozenset(
        {
            "_console_persisted_rows_cache",
            "_console_persisted_rows_cache_key",
            "_console_persisted_rows_cache_at",
            "_console_conversation_browser_query",
            "_console_conversation_browser_search_timer",
            "_console_conversation_browser_search_token",
            "_console_conversation_browser_rows",
            "_console_conversation_browser_total",
            "_console_conversation_browser_error",
        }
    ),
    "_retrieval": frozenset(
        {
            "_console_retrieval_scope_cache",
            "_console_effective_scope_cache",
            "_active_dictionaries_summary",
            "_last_console_dictionary_scope_ids",
            "_active_world_books_summary",
            "_last_console_world_book_scope_ids",
        }
    ),
    "_skill": frozenset({"_console_skill_candidates"}),
    "_character": frozenset(
        {
            "_active_character_avatar",
            "_active_character_avatar_name",
            "_last_console_avatar_scope",
            "_console_expression_spec_cache",
        }
    ),
}
BROWSER_LEGACY_STATE_NAMES = frozenset(
    {
        "_console_workspace_conversation_query",
        "_console_workspace_conversation_search_timer",
        "_console_workspace_conversation_search_token",
        "_console_workspace_conversation_search_rows",
        "_console_workspace_conversation_search_total",
        "_console_workspace_conversation_search_error",
    }
)
EXTERNAL_COMPATIBILITY_NAME = "_console_video_store"
EXTERNAL_WRITE_SOURCES = {
    "Tests/Chat/test_console_video_actions.py",
    "Tests/Chat/test_console_video_message.py",
}
BASELINE_DEFAULTS = {
    "_imagegen_inflight_sessions": ("set",),
    "_imagegen_inflight_message_ids": ("set",),
    "_console_h3_ui_generations": ("dict",),
    "_console_videogen_inflight": ("set",),
    "_console_videogen_cancels": ("dict",),
    "_console_video_store": ("constant", "NoneType", None),
    "_pending_video_artifacts": ("dict",),
    "_pending_video_artifacts_closed": ("constant", "bool", False),
    "_pending_video_operation_cancels": ("dict",),
    "_pending_video_active_operations": ("dict",),
    "_pending_video_deferred_closes": ("dict",),
    "_console_persisted_rows_cache": ("constant", "NoneType", None),
    "_console_persisted_rows_cache_key": ("constant", "NoneType", None),
    "_console_persisted_rows_cache_at": ("constant", "float", 0.0),
    "_console_conversation_browser_query": ("constant", "str", ""),
    "_console_conversation_browser_search_timer": ("constant", "NoneType", None),
    "_console_conversation_browser_search_token": ("constant", "int", 0),
    "_console_conversation_browser_rows": ("tuple",),
    "_console_conversation_browser_total": ("constant", "NoneType", None),
    "_console_conversation_browser_error": ("constant", "str", ""),
    "_console_retrieval_scope_cache": ("dict",),
    "_console_effective_scope_cache": ("dict",),
    "_active_dictionaries_summary": ("constant", "NoneType", None),
    "_last_console_dictionary_scope_ids": ("constant", "NoneType", None),
    "_active_world_books_summary": ("constant", "NoneType", None),
    "_last_console_world_book_scope_ids": ("constant", "NoneType", None),
    "_console_skill_candidates": ("tuple",),
    "_active_character_avatar": ("constant", "NoneType", None),
    "_active_character_avatar_name": ("constant", "NoneType", None),
    "_last_console_avatar_scope": ("constant", "NoneType", None),
    "_console_expression_spec_cache": ("dict",),
}
DELEGATE_BINDINGS = {
    "_console_command_generate_image": "_dispatch_console_command",
    "_console_command_generate_video": "_dispatch_console_command",
    "_console_command_stream_video": "_dispatch_console_command",
    "on_console_workspace_conversation_search_changed": "@on",
    "_persist_console_rag_auto_retrieve_on_send": "_open_console_rag_settings",
    "_execute_console_library_rag_search": "_run_console_library_rag_from_visible_action",
    "_console_command_skills": "_dispatch_console_command",
    "handle_console_skill_install_decided": "@on",
    "handle_console_skill_script_decided": "@on",
    "on_console_auto_speak_changed": "@on",
    "on_console_auto_speak_resume_requested": "@on",
    "on_console_auto_speak_retry_requested": "@on",
}


def _class_node(path: Path, class_name: str) -> tuple[str, ast.ClassDef]:
    source = path.read_text(encoding="utf-8")
    return source, _class_node_from_source(source, class_name, path)


def _class_node_from_source(
    source: str, class_name: str, source_name: Path | str
) -> ast.ClassDef:
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1, f"expected exactly one {class_name} in {source_name}"
    return classes[0]


def _source_at_revision(revision: str, path: Path) -> str:
    relative_path = path.relative_to(_REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout


def _methods_from_class(class_node: ast.ClassDef) -> dict[str, ast.AST]:
    return {node.name: node for node in _method_definitions(class_node)}


def _method_definitions(
    class_node: ast.ClassDef,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return every direct method definition in source order."""
    return [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _method_name_counts(class_node: ast.ClassDef) -> Counter[str]:
    """Count direct method definitions without collapsing duplicate names."""
    return Counter(node.name for node in _method_definitions(class_node))


def _method_count(class_node: ast.ClassDef) -> int:
    return len(_method_definitions(class_node))


def _methods(path: Path, class_name: str) -> dict[str, ast.AST]:
    _, class_node = _class_node(path, class_name)
    return _methods_from_class(class_node)


def _span(node: ast.AST) -> int:
    return node.end_lineno - node.lineno + 1  # type: ignore[attr-defined]


def _is_property(node: ast.AST) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "property"
        for decorator in node.decorator_list  # type: ignore[attr-defined]
    )


def _self_assignments(root: ast.AST) -> set[str]:
    assigned: set[str] = set()
    for node in ast.walk(root):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assigned.add(target.attr)
    return assigned


def _default_signature(node: ast.AST) -> tuple[object, ...]:
    if isinstance(node, ast.Constant):
        return ("constant", type(node.value).__name__, node.value)
    if isinstance(node, ast.Dict) and not node.keys:
        return ("dict",)
    if isinstance(node, ast.Tuple) and not node.elts:
        return ("tuple",)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"dict", "set", "tuple"}
        and not node.args
        and not node.keywords
    ):
        return (node.func.id,)
    return ("expression", ast.dump(node, include_attributes=False))


def _resolved_default_signature(
    value: ast.AST, method: ast.AST, before_line: int
) -> tuple[object, ...]:
    if not isinstance(value, ast.Name):
        return _default_signature(value)
    candidates: list[tuple[int, ast.AST]] = []
    for node in ast.walk(method):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if node.lineno >= before_line:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == value.id for target in targets
        ):
            candidates.append((node.lineno, node.value))
    if not candidates:
        return _default_signature(value)
    line, initializer = max(
        candidates,
        key=lambda item: (item[0], item[1].col_offset),
    )
    return _resolved_default_signature(initializer, method, line)


def _assignment_signatures(
    class_node: ast.ClassDef, attribute: str
) -> set[tuple[object, ...]]:
    signatures: set[tuple[object, ...]] = set()
    for method in _methods_from_class(class_node).values():
        for node in ast.walk(method):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == attribute
                for target in targets
            ):
                signatures.add(
                    _resolved_default_signature(node.value, method, node.lineno)
                )
    return signatures


def _getattr_default_signatures(
    class_node: ast.ClassDef, attribute: str
) -> set[tuple[object, ...]]:
    return {
        _default_signature(node.args[2])
        for node in ast.walk(class_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 3
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "self"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == attribute
    }


def _first_self_attribute_access(
    method: ast.AST, attribute: str
) -> tuple[str, ast.AST] | None:
    accesses: list[tuple[int, int, str, ast.AST]] = []
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == attribute
        ):
            kind = "store" if isinstance(node.ctx, ast.Store) else "load"
            accesses.append((node.lineno, node.col_offset, kind, node))
    if not accesses:
        return None
    _, _, kind, node = min(accesses)
    return kind, node


def _build_call_line(method: ast.AST) -> int:
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_console_controllers"
    ]
    assert len(calls) == 1
    return calls[0].lineno


def _first_assignment_value(method: ast.AST, attribute: str) -> ast.AST:
    assignments: list[tuple[int, ast.AST]] = []
    for node in ast.walk(method):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == attribute
            for target in targets
        ):
            assignments.append((node.lineno, node.value))
    assert assignments, f"{attribute} has no controller default"
    return min(
        assignments,
        key=lambda item: (item[0], item[1].col_offset),
    )[1]


def _screen_read_lines(method: ast.AST, attribute: str) -> list[int]:
    lines = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == attribute
        and isinstance(node.ctx, ast.Load)
    ]
    lines.extend(
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"getattr", "hasattr"}
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "self"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == attribute
    )
    return lines


def _assert_controller_default(
    method: ast.AST, attribute: str, expected: tuple[object, ...]
) -> None:
    first_access = _first_self_attribute_access(method, attribute)
    assert first_access is not None and first_access[0] == "store"
    assert _default_signature(_first_assignment_value(method, attribute)) == expected


def _assert_screen_reads_after_build(
    method: ast.AST, attribute: str, build_line: int
) -> None:
    assert all(line >= build_line for line in _screen_read_lines(method, attribute))


def _has_external_default_none(base_methods: dict[str, ast.AST]) -> bool:
    method = base_methods["_ensure_console_video_store"]
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 3
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "self"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == EXTERNAL_COMPATIBILITY_NAME
        and isinstance(node.args[2], ast.Constant)
        and node.args[2].value is None
        for node in ast.walk(method)
    )


def _assert_owner_is_wired(group: Wave6Group) -> None:
    wiring_path = _REPO_ROOT / "tldw_chatbook/UI/Console_Modules/wiring.py"
    wiring_tree = ast.parse(wiring_path.read_text(encoding="utf-8"))
    build = next(
        node
        for node in wiring_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_console_controllers"
    )
    assignments = []
    for node in ast.walk(build):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "screen"
            and target.attr == group.owner_name
            for target in node.targets
        ):
            assignments.append(node)
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    assert value.func.id == group.target_class


def _class_assignments(class_node: ast.ClassDef) -> set[str]:
    assigned: set[str] = set()
    for node in class_node.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(node, ast.AnnAssign) and node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        assigned.update(target.id for target in targets if isinstance(target, ast.Name))
    return assigned


def _calls_query_one(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "query_one"
        for child in ast.walk(node)
    )


def _calls_dom_query(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr in {"query", "query_one"}
        for child in ast.walk(node)
    )


def _has_on_binding(method: ast.AST) -> bool:
    for decorator in method.decorator_list:  # type: ignore[attr-defined]
        function = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(function, ast.Name) and function.id == "on":
            return True
    return False


def _has_browser_search_on_binding(method: ast.AST) -> bool:
    for decorator in method.decorator_list:  # type: ignore[attr-defined]
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "on"
            and len(decorator.args) == 2
        ):
            continue
        event_type, selector = decorator.args
        if (
            isinstance(event_type, ast.Attribute)
            and isinstance(event_type.value, ast.Name)
            and event_type.value.id == "Input"
            and event_type.attr == "Changed"
            and isinstance(selector, ast.Constant)
            and selector.value == "#console-workspace-conversation-search"
        ):
            return True
    return False


def _controller_state_assignments(
    class_node: ast.ClassDef, names: frozenset[str]
) -> dict[str, str]:
    targets: dict[str, str] = {}
    for node in class_node.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not (
            isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_ControllerState"
            and len(node.value.args) == 2
            and all(isinstance(argument, ast.Constant) for argument in node.value.args)
        ):
            continue
        owner_name, state_name = (argument.value for argument in node.value.args)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                assert state_name == target.id
                targets[target.id] = owner_name
    return targets


def _self_owner_accesses(method: ast.AST, names: frozenset[str]) -> set[str]:
    return {
        node.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in names
    }


def _direct_self_writes(method: ast.AST, names: frozenset[str]) -> set[str]:
    def targets_inside(target: ast.AST):
        yield target
        if isinstance(target, (ast.List, ast.Tuple)):
            for element in target.elts:
                yield from targets_inside(element)
        elif isinstance(target, ast.Starred):
            yield from targets_inside(target.value)

    writes: set[str] = set()
    for node in ast.walk(method):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        writes.update(
            target.attr
            for assignment_target in targets
            for target in targets_inside(assignment_target)
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr in names
        )
    return writes


def _direct_writer_inventory(
    methods: dict[str, ast.AST],
    names: frozenset[str],
    *,
    excluded: frozenset[str] = frozenset(),
) -> dict[str, list[str]]:
    return {
        name: sorted(writes)
        for name, method in methods.items()
        if name not in excluded and (writes := _direct_self_writes(method, names))
    }


def _workspace_delegate_calls(method: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == "_workspace"
    ]


def _workspace_delegate_targets(method: ast.AST) -> set[str]:
    return {call.func.attr for call in _workspace_delegate_calls(method)}  # type: ignore[union-attr]


def _assert_browser_search_delegate_contract(method: ast.AST) -> None:
    """Require the Textual handler to pass only plain search values."""
    assert isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
    assert len(method.args.args) == 2
    expected_annotation = ast.parse("Changed", mode="eval").body
    assert method.args.args[1].annotation is not None
    assert ast.dump(method.args.args[1].annotation) == ast.dump(expected_annotation)
    event_name = method.args.args[1].arg
    assert len(method.body) == 4

    stop_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == event_name
        and node.func.attr == "stop"
    ]
    assert len(stop_calls) == 1
    assert not stop_calls[0].args and not stop_calls[0].keywords
    first = method.body[0]
    assert isinstance(first, ast.Expr) and first.value is stop_calls[0]

    workspace_calls = _workspace_delegate_calls(method)
    assert len(workspace_calls) == 1
    transition_call = workspace_calls[0]
    assert len(transition_call.args) == 2 and not transition_call.keywords
    assert all(isinstance(argument, ast.Name) for argument in transition_call.args)
    query_name, disabled_name = (
        argument.id
        for argument in transition_call.args  # type: ignore[union-attr]
    )
    assert event_name not in {query_name, disabled_name}

    query_statement, disabled_statement, final_statement = method.body[1:]
    assert isinstance(query_statement, ast.Assign)
    assert isinstance(disabled_statement, ast.Assign)
    assert isinstance(final_statement, ast.Expr)
    assert final_statement.value is transition_call
    assert len(query_statement.targets) == 1
    assert isinstance(query_statement.targets[0], ast.Name)
    assert query_statement.targets[0].id == query_name
    assert len(disabled_statement.targets) == 1
    assert isinstance(disabled_statement.targets[0], ast.Name)
    assert disabled_statement.targets[0].id == disabled_name
    expected_query = ast.parse(f"str({event_name}.value or '')", mode="eval").body
    expected_disabled = ast.parse(
        f"bool(getattr(getattr({event_name}, 'input', None), 'disabled', False))",
        mode="eval",
    ).body
    assert ast.dump(query_statement.value) == ast.dump(expected_query)
    assert ast.dump(disabled_statement.value) == ast.dump(expected_disabled)
    assert not _direct_self_writes(
        method,
        COMPATIBILITY_TARGETS["_workspace"] | BROWSER_LEGACY_STATE_NAMES,
    )


def _has_real_delegate_binding(
    screen_methods: dict[str, ast.AST], method_name: str
) -> bool:
    binding = DELEGATE_BINDINGS[method_name]
    if binding == "@on":
        return _has_on_binding(screen_methods[method_name])
    caller = screen_methods.get(binding)
    return caller is not None and any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == method_name
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(caller)
    )


def _assert_no_dom_access(methods: object) -> None:
    assert not any(_calls_query_one(node) for node in methods)  # type: ignore[arg-type]


def _assert_skill_dead_tokens_absent(sources: dict[str, str]) -> None:
    dead_tokens = {
        "KIND_FALLBACK",
        "ConsoleSkillPickerModal",
        "_CONSOLE_SKILL_SEARCH_LIMIT",
        "_console_skill_search",
        "_console_command_run_skill",
        "_run_resolved_console_skill",
        "_open_console_skill_picker",
    }
    offenders = {
        path: sorted(token for token in dead_tokens if token in source)
        for path, source in sources.items()
        if any(token in source for token in dead_tokens)
    }
    assert not offenders, f"dead Console skill surface remains: {offenders}"


def _assert_delegate_contract(
    screen_methods: dict[str, ast.AST], method_name: str, *, complete: bool
) -> None:
    assert _has_real_delegate_binding(screen_methods, method_name)
    if complete:
        assert _span(screen_methods[method_name]) <= 5


def _assigns_attribute(path: Path, attribute: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Attribute) and target.attr == attribute
            for target in targets
        ):
            return True
    return False


def _assert_descriptor_contract(
    screen_type: type, owner_name: str, state_name: str
) -> None:
    screen = screen_type.__new__(screen_type)
    for read in (
        lambda: getattr(screen, state_name),
        lambda: getattr(screen, state_name, object()),
        lambda: hasattr(screen, state_name),
    ):
        try:
            read()
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{state_name} did not fail loudly before wiring")
    try:
        setattr(screen, state_name, object())
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"{state_name} accepted a pre-wiring write")

    initial = object()
    owner = SimpleNamespace(**{state_name: initial})
    object.__setattr__(screen, owner_name, owner)
    assert getattr(screen, state_name) is initial

    replacement = object()
    setattr(screen, state_name, replacement)
    assert getattr(owner, state_name) is replacement
    assert state_name not in screen.__dict__


@pytest.mark.unit
def test_wave6_inventory_matches_the_implementation_base() -> None:
    """Lock every reviewed definition to its implementation-base evidence.

    Each definition retains an exact source span and one migration owner while
    the current tree moves through the extraction phases.
    """
    base_source = _source_at_revision(IMPLEMENTATION_BASE, _SCREEN_PATH)
    base_class = _class_node_from_source(base_source, "ChatScreen", _SCREEN_PATH)
    base_methods = _methods_from_class(base_class)
    post_image_source = _source_at_revision(
        POST_IMAGE_IMPLEMENTATION_BASE, _SCREEN_PATH
    )
    post_image_class = _class_node_from_source(
        post_image_source, "ChatScreen", _SCREEN_PATH
    )
    post_image_methods = _methods_from_class(post_image_class)
    reviewed_methods = {
        IMPLEMENTATION_BASE: base_methods,
        POST_IMAGE_IMPLEMENTATION_BASE: post_image_methods,
    }
    screen_source, screen_class = _class_node(_SCREEN_PATH, "ChatScreen")
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")

    assert len(base_source.splitlines()) == BASELINE_LINES
    assert _method_count(base_class) == BASELINE_METHODS
    assert len(post_image_source.splitlines()) == POST_IMAGE_BASELINE_LINES
    assert _method_count(post_image_class) == POST_IMAGE_BASELINE_METHODS
    assert len(screen_source.splitlines()) <= POST_IMAGE_BASELINE_LINES, (
        POST_IMAGE_IMPLEMENTATION_BASE
    )
    assert len(screen_source.splitlines()) <= TASK10_SCREEN_LINE_CEILING
    assert _method_count(screen_class) <= POST_IMAGE_BASELINE_METHODS, (
        POST_IMAGE_IMPLEMENTATION_BASE
    )
    assert set(WAVE6_GROUPS) == {
        "image",
        "video",
        "browser",
        "retrieval",
        "skill",
        "character",
        "fleet",
        "first_chat",
        "auto_speak",
    }
    assert set(DELEGATE_BINDINGS) == set().union(
        *(group.delegates for group in WAVE6_GROUPS.values())
    )
    all_names: set[str] = set()
    for name, group in WAVE6_GROUPS.items():
        overlap = all_names & group.raw_names
        assert not overlap, f"Wave 6 methods counted twice: {sorted(overlap)}"
        all_names.update(group.raw_names)

        target_path = _REPO_ROOT / group.target_path
        target_methods = (
            _methods(target_path, group.target_class) if target_path.exists() else {}
        )
        target_owners = {
            method_name: node
            for method_name, node in target_methods.items()
            if not _is_property(node)
        }
        for method_name in group.moved:
            owners = int(method_name in screen_methods) + int(
                method_name in target_owners
            )
            assert owners == 1, f"{name}.{method_name} must have exactly one owner"
            if method_name in target_owners:
                assert not _calls_query_one(target_owners[method_name])
        assert group.delegates <= screen_methods.keys()
        assert group.stays <= screen_methods.keys()
        assert not (group.deleted & target_owners.keys())

        source_methods = reviewed_methods[group.source_revision]
        assert group.raw_names <= source_methods.keys()
        assert sum(_span(source_methods[item]) for item in group.raw_names) == (
            group.raw_lines
        )

        complete = not (group.moved & screen_methods.keys()) and not (
            group.deleted & screen_methods.keys()
        )
        if complete:
            residue = group.delegates | group.stays
            assert sum(_span(screen_methods[item]) for item in residue) <= (
                group.residue_lines
            )
            for method_name in group.delegates:
                _assert_delegate_contract(screen_methods, method_name, complete=True)
        else:
            for method_name in group.delegates:
                _assert_delegate_contract(screen_methods, method_name, complete=False)

        if target_path.exists():
            methods_to_scan = (
                target_methods[method_name]
                for method_name in group.moved | group.delegates
                if method_name in target_methods
            )
            _assert_no_dom_access(methods_to_scan)


@pytest.mark.unit
def test_browser_family_has_completed_workspace_ownership() -> None:
    """Require every reviewed browser method to have its final sole owner."""
    group = WAVE6_GROUPS["browser"]
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")
    target_methods = _methods(_REPO_ROOT / group.target_path, group.target_class)

    assert not (group.moved & screen_methods.keys()), (
        "browser methods still owned by ChatScreen: "
        f"{sorted(group.moved & screen_methods.keys())}"
    )
    assert group.moved <= target_methods.keys(), (
        "browser methods missing from ConsoleWorkspaceController: "
        f"{sorted(group.moved - target_methods.keys())}"
    )
    assert not (group.delegates & target_methods.keys())
    assert group.delegates <= screen_methods.keys()


@pytest.mark.unit
def test_retrieval_family_has_completed_controller_ownership() -> None:
    """Require every reviewed retrieval method to have its final owner."""
    group = WAVE6_GROUPS["retrieval"]
    target_path = _REPO_ROOT / group.target_path
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")

    assert target_path.exists(), "ConsoleRetrievalController module is missing"
    target_methods = _methods(target_path, group.target_class)
    assert not (group.moved & screen_methods.keys()), (
        "retrieval methods still owned by ChatScreen: "
        f"{sorted(group.moved & screen_methods.keys())}"
    )
    assert group.moved <= target_methods.keys(), (
        "retrieval methods missing from ConsoleRetrievalController: "
        f"{sorted(group.moved - target_methods.keys())}"
    )
    assert group.delegates <= screen_methods.keys()
    assert group.delegates <= target_methods.keys()
    owned_methods = [target_methods[name] for name in group.moved | group.delegates]
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"query", "query_one", "push_screen"}
        for method in owned_methods
        for node in ast.walk(method)
    )
    assert "_screen" not in _self_assignments(target_methods["__init__"])


@pytest.mark.unit
def test_retrieval_compatibility_descriptors_all_target_retrieval() -> None:
    """Require all six assignable retrieval names to proxy to one owner."""
    _, screen_class = _class_node(_SCREEN_PATH, "ChatScreen")
    names = COMPATIBILITY_TARGETS["_retrieval"]

    assert _controller_state_assignments(screen_class, names) == {
        name: "_retrieval" for name in names
    }


@pytest.mark.unit
def test_skill_family_has_completed_controller_ownership() -> None:
    """Require the exact nine-M/three-D/four-X skill inventory."""
    group = WAVE6_GROUPS["skill"]
    target_path = _REPO_ROOT / group.target_path
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")

    assert target_path.exists(), "ConsoleSkillController module is missing"
    target_methods = _methods(target_path, group.target_class)
    assert not (group.moved & screen_methods.keys()), (
        "skill methods still owned by ChatScreen: "
        f"{sorted(group.moved & screen_methods.keys())}"
    )
    assert group.moved <= target_methods.keys()
    assert group.delegates <= screen_methods.keys()
    assert group.delegates <= target_methods.keys()
    assert not (group.deleted & screen_methods.keys())
    assert not (group.deleted & target_methods.keys())
    owned = [target_methods[name] for name in group.moved | group.delegates]
    _assert_no_dom_access(owned)
    assert "_screen" not in _self_assignments(target_methods["__init__"])


@pytest.mark.unit
def test_skill_compatibility_descriptor_targets_skill_controller() -> None:
    """Keep the assignable candidate cache on the `_skill` owner."""
    _, screen_class = _class_node(_SCREEN_PATH, "ChatScreen")
    names = COMPATIBILITY_TARGETS["_skill"]

    assert _controller_state_assignments(screen_class, names) == {
        name: "_skill" for name in names
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "method_name",
    [
        "_console_command_skills",
        "handle_console_skill_install_decided",
        "handle_console_skill_script_decided",
    ],
)
def test_skill_screen_entry_points_are_bounded_controller_delegates(
    method_name: str,
) -> None:
    """Keep framework-bound screen names and one exact `_skill` call."""
    method = _methods(_SCREEN_PATH, "ChatScreen")[method_name]
    _assert_delegate_contract(
        _methods(_SCREEN_PATH, "ChatScreen"), method_name, complete=True
    )
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method_name
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == "_skill"
    ]
    assert len(calls) == 1


@pytest.mark.unit
def test_skill_dead_fallback_and_picker_surface_is_absent() -> None:
    """Reject every unreachable fallback/picker production and test surface."""
    target_path = _REPO_ROOT / WAVE6_GROUPS["skill"].target_path
    scanned = {
        path.relative_to(_REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in (
            target_path,
            _SCREEN_PATH,
            _REPO_ROOT / "tldw_chatbook/Chat/console_skill_resolver.py",
            _REPO_ROOT / "tldw_chatbook/Widgets/Console/console_style_picker_modal.py",
            _REPO_ROOT / "Tests/Chat/test_console_style_picker.py",
        )
    }
    _assert_skill_dead_tokens_absent(scanned)

    assert not (
        _REPO_ROOT / "tldw_chatbook/Widgets/Console/console_skill_picker_modal.py"
    ).exists()
    assert not (_REPO_ROOT / "Tests/UI/test_console_skill_picker.py").exists()
    for css_path in (
        _REPO_ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss",
        _REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss",
    ):
        assert "console-skill-picker" not in css_path.read_text(encoding="utf-8")

    removed_owner_tokens = {
        f"ChatScreen.{name}" for name in WAVE6_GROUPS["skill"].moved
    }
    production_offenders = {
        path.relative_to(_REPO_ROOT).as_posix(): sorted(
            token for token in removed_owner_tokens if token in source
        )
        for path in (_REPO_ROOT / "tldw_chatbook").rglob("*.py")
        if (source := path.read_text(encoding="utf-8"))
        and any(token in source for token in removed_owner_tokens)
    }
    assert not production_offenders


@pytest.mark.unit
def test_skill_dead_surface_oracle_is_non_vacuous() -> None:
    """Prove one synthetic dead token makes the deletion oracle fail."""
    with pytest.raises(AssertionError, match="dead Console skill surface"):
        _assert_skill_dead_tokens_absent(
            {"synthetic.py": "self._open_console_skill_picker('x', '')"}
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method_name", "awaited"),
    [
        ("_persist_console_rag_auto_retrieve_on_send", False),
        ("_execute_console_library_rag_search", True),
    ],
)
def test_retrieval_worker_delegates_call_exact_controller_methods(
    method_name: str, awaited: bool
) -> None:
    """Both decorated screen names contain one exact controller call."""
    method = _methods(_SCREEN_PATH, "ChatScreen")[method_name]
    assert len(method.body) == 1
    statement = method.body[0]
    assert isinstance(statement, ast.Expr)
    value = statement.value
    if awaited:
        assert isinstance(value, ast.Await)
        value = value.value
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Attribute)
    assert value.func.attr == method_name
    owner = value.func.value
    assert isinstance(owner, ast.Attribute)
    assert isinstance(owner.value, ast.Name)
    assert owner.value.id == "self"
    assert owner.attr == "_retrieval"


@pytest.mark.unit
def test_browser_search_handler_is_exact_bounded_textual_delegate() -> None:
    """Keep the framework handler bound and within its five-line residue."""
    group = WAVE6_GROUPS["browser"]
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")
    method_name = next(iter(group.delegates))
    method = screen_methods[method_name]

    assert _has_browser_search_on_binding(method)
    assert _span(method) <= 5
    _assert_browser_search_delegate_contract(method)


@pytest.mark.unit
def test_browser_compatibility_descriptors_all_target_workspace() -> None:
    """Require all nine assignable browser names to proxy through Workspace."""
    _, screen_class = _class_node(_SCREEN_PATH, "ChatScreen")
    names = COMPATIBILITY_TARGETS["_workspace"]

    assert _controller_state_assignments(screen_class, names) == {
        name: "_workspace" for name in names
    }


@pytest.mark.unit
def test_workspace_browser_methods_never_query_the_dom() -> None:
    """Reject both Textual query APIs in every moved Workspace method."""
    group = WAVE6_GROUPS["browser"]
    target_methods = _methods(_REPO_ROOT / group.target_path, group.target_class)

    assert group.moved <= target_methods.keys(), (
        "cannot prove the DOM boundary until every browser method has moved"
    )
    offenders = {name for name in group.moved if _calls_dom_query(target_methods[name])}
    assert not offenders, (
        f"Workspace browser methods query the DOM: {sorted(offenders)}"
    )


@pytest.mark.unit
def test_workspace_browser_methods_have_no_sibling_controller_reach_through() -> None:
    """Require moved browser dependencies to use named constructor callables."""
    group = WAVE6_GROUPS["browser"]
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")
    target_methods = _methods(_REPO_ROOT / group.target_path, group.target_class)
    forbidden = frozenset(
        {
            "_screen",
            "_session",
            "_agent",
            "_workspace",
            "_fleet",
            "_hands_free",
            "_retrieval",
            "_image",
            "_video",
            "_skill",
            "_character",
            "_message",
            "_prompts",
            "_prompt_queue",
        }
    )
    owners = {
        name: target_methods.get(name) or screen_methods.get(name)
        for name in group.moved
    }
    assert all(owners.values()), "reviewed browser inventory contains a missing method"
    offenders = {
        name: sorted(_self_owner_accesses(method, forbidden))
        for name, method in owners.items()
        if method is not None and _self_owner_accesses(method, forbidden)
    }

    delegate_name = next(iter(group.delegates))
    transition_names = _workspace_delegate_targets(screen_methods[delegate_name])
    if transition_names:
        assert len(transition_names) == 1
        transition_name = next(iter(transition_names))
        assert transition_name in target_methods
        transition_offenders = _self_owner_accesses(
            target_methods[transition_name], forbidden
        )
        if transition_offenders:
            offenders[transition_name] = sorted(transition_offenders)

    assert not offenders, f"browser methods reach sibling owners directly: {offenders}"


@pytest.mark.unit
def test_chat_screen_has_no_direct_browser_state_writers() -> None:
    """Keep browser/cache writes behind Workspace, including the Clear branch."""
    group = WAVE6_GROUPS["browser"]
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")
    state_names = COMPATIBILITY_TARGETS["_workspace"] | BROWSER_LEGACY_STATE_NAMES
    writers = _direct_writer_inventory(
        screen_methods,
        state_names,
        excluded=group.delegates,
    )

    assert "on_button_pressed" not in writers, (
        "the browser Clear button is still a duplicate state writer: "
        f"{writers.get('on_button_pressed')}"
    )
    assert not writers, f"ChatScreen still writes Workspace browser state: {writers}"


@pytest.mark.unit
def test_wave6_projection_clears_both_ratchet_overages() -> None:
    """Require the remaining post-image projection to clear both overages.

    The conservative line and method estimates must retain implementation
    margin after all documented residue is included.
    """
    remaining = {name: group for name, group in WAVE6_GROUPS.items() if name != "image"}
    raw_lines = sum(group.raw_lines for group in remaining.values())
    residue_lines = sum(group.residue_lines for group in remaining.values())
    removed_methods = sum(group.removed_methods for group in remaining.values())

    assert raw_lines == 4_640
    assert residue_lines + DESCRIPTOR_LINE_BUDGET == 119
    assert raw_lines - residue_lines - DESCRIPTOR_LINE_BUDGET == 4_521
    assert removed_methods == 131
    assert 4_521 > LINE_OVERAGE
    assert removed_methods > METHOD_OVERAGE


@pytest.mark.unit
def test_wave6_compatibility_inventory_is_complete_and_phase_safe() -> None:
    """Lock compatibility assignments across every extraction phase.

    Baseline assignments remain authoritative until an entire family moves;
    afterward, descriptors must preserve the reviewed controller contract.
    """
    base_source = _source_at_revision(IMPLEMENTATION_BASE, _SCREEN_PATH)
    base_class = _class_node_from_source(base_source, "ChatScreen", _SCREEN_PATH)
    base_methods = _methods_from_class(base_class)
    _, screen_class = _class_node(_SCREEN_PATH, "ChatScreen")
    screen_methods = _methods_from_class(screen_class)
    compatibility_names = frozenset().union(*COMPATIBILITY_TARGETS.values())
    base_assigned_names = _self_assignments(base_class) & compatibility_names
    assigned_names = _self_assignments(screen_class) & compatibility_names
    descriptors = _class_assignments(screen_class) & compatibility_names

    assert len(compatibility_names) == 31
    assert set(BASELINE_DEFAULTS) == compatibility_names
    assert base_assigned_names == compatibility_names - {EXTERNAL_COMPATIBILITY_NAME}
    for state_name in base_assigned_names:
        observed_defaults = _assignment_signatures(
            base_class, state_name
        ) | _getattr_default_signatures(base_class, state_name)
        assert BASELINE_DEFAULTS[state_name] in observed_defaults
    assert _has_external_default_none(base_methods)
    assert all(
        _assigns_attribute(_REPO_ROOT / path, EXTERNAL_COMPATIBILITY_NAME)
        for path in EXTERNAL_WRITE_SOURCES
    )
    build_line = _build_call_line(screen_methods["__init__"])

    for owner_name, names in COMPATIBILITY_TARGETS.items():
        family_descriptors = descriptors & names
        baseline_assignments = names - {EXTERNAL_COMPATIBILITY_NAME}
        if not family_descriptors:
            assert assigned_names & names == baseline_assignments
            continue
        assert family_descriptors == names
        assert not (assigned_names & names)

        group = next(
            group for group in WAVE6_GROUPS.values() if group.owner_name == owner_name
        )
        target_path = _REPO_ROOT / group.target_path
        assert target_path.exists()
        target_methods = _methods(target_path, group.target_class)
        assert "__init__" in target_methods
        target_init = target_methods["__init__"]
        assert names <= _self_assignments(target_init)
        for state_name in names:
            _assert_controller_default(
                target_init, state_name, BASELINE_DEFAULTS[state_name]
            )
            _assert_screen_reads_after_build(
                screen_methods["__init__"], state_name, build_line
            )
        _assert_owner_is_wired(group)

        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        for state_name in names:
            _assert_descriptor_contract(ChatScreen, owner_name, state_name)


def _assert_character_ownership_contract(
    screen_methods: dict[str, ast.AST],
    target_methods: dict[str, ast.AST],
) -> None:
    """Require character policy and presentation to have one approved owner."""
    group = WAVE6_GROUPS["character"]
    assert not (group.moved & screen_methods.keys()), (
        "character methods still owned by ChatScreen: "
        f"{sorted(group.moved & screen_methods.keys())}"
    )
    assert group.moved <= target_methods.keys(), (
        "character methods missing from ConsoleCharacterController: "
        f"{sorted(group.moved - target_methods.keys())}"
    )
    assert not (group.deleted & screen_methods.keys())
    assert not (group.deleted & target_methods.keys())
    dom_offenders = {
        name for name in group.moved if _calls_dom_query(target_methods[name])
    }
    assert not dom_offenders, (
        f"character controller methods query DOM: {sorted(dom_offenders)}"
    )

    presentation = {
        "_open_console_character_picker",
        "_apply_console_character_choice",
        "_render_character_avatar_into_section",
    }
    assert presentation <= screen_methods.keys()
    assert not (presentation & target_methods.keys()), (
        "character presentation duplicated on controller: "
        f"{sorted(presentation & target_methods.keys())}"
    )


@pytest.mark.unit
def test_character_family_has_completed_controller_ownership() -> None:
    """Require the current seven-M/deleted-one character inventory."""
    group = WAVE6_GROUPS["character"]
    target_path = _REPO_ROOT / group.target_path
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")
    character_state = {
        "_active_character_avatar",
        "_active_character_avatar_name",
        "_last_console_avatar_scope",
        "_console_expression_spec_cache",
    }

    assert target_path.exists(), "ConsoleCharacterController module is missing"
    assert COMPATIBILITY_TARGETS[group.owner_name] == character_state
    assert set(BASELINE_DEFAULTS) & character_state == character_state
    target_methods = _methods(target_path, group.target_class)
    _assert_character_ownership_contract(screen_methods, target_methods)
    assert "_screen" not in _self_assignments(target_methods["__init__"])


@pytest.mark.unit
def test_character_move_ownership_oracle_is_non_vacuous() -> None:
    """Prove the shared oracle rejects screen ownership and DOM queries."""
    group = WAVE6_GROUPS["character"]
    moved_method = "_console_character_picker_options"
    assert moved_method in group.moved
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")
    extracted_screen = {
        name: method
        for name, method in screen_methods.items()
        if name not in group.moved
    }
    target_methods = {
        name: ast.parse(f"def {name}(self): pass").body[0] for name in group.moved
    }
    synthetic_screen = {
        **extracted_screen,
        moved_method: target_methods[moved_method],
    }
    with pytest.raises(AssertionError, match="still owned by ChatScreen"):
        _assert_character_ownership_contract(synthetic_screen, target_methods)

    synthetic_target = dict(target_methods)
    synthetic_target[moved_method] = ast.parse(
        "def _console_character_picker_options(self): self.query('.row')"
    ).body[0]
    with pytest.raises(AssertionError, match="query DOM"):
        _assert_character_ownership_contract(
            extracted_screen,
            synthetic_target,
        )


@pytest.mark.unit
def test_character_controller_has_only_named_non_dom_dependencies() -> None:
    """Lock Task10 orchestration behind the approved controller boundary."""

    path = _REPO_ROOT / "tldw_chatbook/UI/Console_Modules/character.py"
    _, controller = _class_node(path, "ConsoleCharacterController")
    methods = _methods_from_class(controller)
    init = methods["__init__"]
    assert isinstance(init, ast.FunctionDef)
    assert not init.args.posonlyargs
    assert [argument.arg for argument in init.args.args] == ["self"]
    assert {argument.arg for argument in init.args.kwonlyargs} == {
        "app_config_accessor",
        "chat_store_accessor",
        "active_native_session_accessor",
        "current_conversation_id_accessor",
        "character_db_accessor",
        "ensure_chat_store",
        "provider_readiness_config_accessor",
        "default_session_settings",
        "swap_session_character",
        "sync_temporary_chip",
        "sync_native_chat_ui",
        "notify",
        "actor_scope_accessor",
        "manual_reaction_key",
        "resolve_visual_identity",
        "resolve_historical_visual_identity",
        "ensure_console_image_view",
        "console_image_default_mode",
        "is_mounted",
        "render_character_avatar",
    }
    assert init.args.kw_defaults == [None] * len(init.args.kwonlyargs)
    assert init.args.vararg is None
    assert init.args.kwarg is None
    assert "_screen" not in _self_assignments(controller)
    assert not any(_calls_dom_query(method) for method in methods.values())


def _assert_first_chat_ownership_multiplicity(
    screen_class: ast.ClassDef,
    target_class: ast.ClassDef,
) -> None:
    """Require zero screen and exactly one controller definition per method."""
    group = WAVE6_GROUPS["first_chat"]
    screen_counts = _method_name_counts(screen_class)
    target_counts = _method_name_counts(target_class)
    screen_owners = {
        name: screen_counts[name] for name in group.moved if screen_counts[name]
    }
    target_multiplicity = {
        name: target_counts[name] for name in group.moved if target_counts[name] != 1
    }
    assert not screen_owners, (
        f"first-chat methods still owned by ChatScreen: {screen_owners}"
    )
    assert not target_multiplicity, (
        f"first-chat controller methods must occur exactly once: {target_multiplicity}"
    )


def _assert_first_chat_revision_state_ownership(
    screen_class: ast.ClassDef,
    target_class: ast.ClassDef,
) -> None:
    """Reject every screen-side revision-state shim, including class proxies."""
    state_name = "_first_chat_handoff_notified_revision"

    def assigned_names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.List, ast.Tuple)):
            return set().union(*(assigned_names(item) for item in target.elts))
        if isinstance(target, ast.Starred):
            return assigned_names(target.value)
        return set()

    screen_class_members: set[str] = set()
    for node in screen_class.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            screen_class_members.add(node.name)
        elif isinstance(node, ast.Assign):
            screen_class_members.update(
                set().union(*(assigned_names(target) for target in node.targets))
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            screen_class_members.add(node.target.id)

    assert state_name not in screen_class_members, (
        f"first-chat revision-state shim remains on ChatScreen: {state_name}"
    )
    assert state_name not in _self_assignments(screen_class), (
        f"first-chat revision state remains on ChatScreen: {state_name}"
    )
    assert state_name in _self_assignments(target_class), (
        f"first-chat revision state missing from controller: {state_name}"
    )


def _assert_first_chat_presentation_callback_usage(
    methods: dict[str, ast.AST],
) -> None:
    """Require the moved policy family to consume every presentation edge."""
    called = {
        node.func.attr
        for method in methods.values()
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr in FIRST_CHAT_PRESENTATION_ATTRIBUTES
    }
    missing = FIRST_CHAT_PRESENTATION_ATTRIBUTES - called
    assert not missing, f"first-chat presentation callbacks unused: {sorted(missing)}"


def _assert_first_chat_method_boundaries(
    methods: dict[str, ast.AST],
) -> None:
    """Reject DOM queries and direct screen/sibling-controller reach-through."""
    dom_offenders = {
        name for name, method in methods.items() if _calls_dom_query(method)
    }
    assert not dom_offenders, (
        f"first-chat controller methods query DOM: {sorted(dom_offenders)}"
    )

    forbidden_owners = frozenset(
        {
            "_screen",
            "_workspace",
            "_fleet",
            "_agent",
            "_image",
            "_video",
            "_retrieval",
            "_skill",
            "_character",
            "_dictation",
            "_hands_free",
            "_message",
            "_prompts",
            "_prompt_queue",
        }
    )
    sibling_offenders = {
        name: sorted(_self_owner_accesses(method, forbidden_owners))
        for name, method in methods.items()
        if _self_owner_accesses(method, forbidden_owners)
    }
    assert not sibling_offenders, (
        "first-chat controller methods reach screen/sibling owners: "
        f"{sibling_offenders}"
    )

    presentation_names = frozenset({"focus", "focused", "is_attached", "is_mounted"})

    def direct_presentation_accesses(method: ast.AST) -> set[str]:
        return {
            node.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Attribute)
            and (
                node.attr in presentation_names
                or (
                    node.attr == "app"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                )
            )
        }

    presentation_offenders = {
        name: sorted(direct_presentation_accesses(method))
        for name, method in methods.items()
        if direct_presentation_accesses(method)
    }
    assert not presentation_offenders, (
        "first-chat controller methods access presentation directly: "
        f"{presentation_offenders}"
    )


def _assert_no_first_chat_compatibility_delegates(
    screen_methods: dict[str, ast.AST],
) -> None:
    """Reject same-name ChatScreen delegates into the session owner."""
    offenders: set[str] = set()
    for name in WAVE6_GROUPS["first_chat"].moved & screen_methods.keys():
        method = screen_methods[name]
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_session"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "self"
            for node in ast.walk(method)
        ):
            offenders.add(name)
    assert not offenders, (
        f"first-chat compatibility delegates remain: {sorted(offenders)}"
    )


@pytest.mark.unit
def test_first_chat_family_has_completed_controller_ownership() -> None:
    """Require the exact eight-method family and revision state on Session."""
    group = WAVE6_GROUPS["first_chat"]
    target_path = _REPO_ROOT / group.target_path

    assert target_path.exists(), "ConsoleSessionController module is missing"
    _, screen_class = _class_node(_SCREEN_PATH, "ChatScreen")
    _, target_class = _class_node(target_path, group.target_class)
    screen_methods = _methods_from_class(screen_class)
    _assert_first_chat_ownership_multiplicity(screen_class, target_class)
    _assert_no_first_chat_compatibility_delegates(screen_methods)
    _assert_first_chat_revision_state_ownership(screen_class, target_class)


@pytest.mark.unit
def test_first_chat_controller_has_only_named_non_dom_dependencies() -> None:
    """Lock first-chat policy behind narrow, presentation-only callbacks."""
    group = WAVE6_GROUPS["first_chat"]
    _, controller = _class_node(
        _REPO_ROOT / group.target_path,
        group.target_class,
    )
    methods = _methods_from_class(controller)
    missing = group.moved - methods.keys()
    assert not missing, f"first-chat controller methods missing: {sorted(missing)}"
    moved_methods = {name: methods[name] for name in group.moved}
    _assert_first_chat_method_boundaries(moved_methods)
    _assert_first_chat_presentation_callback_usage(moved_methods)

    init = methods["__init__"]
    assert isinstance(init, ast.FunctionDef)
    assert init.args.vararg is None
    assert init.args.kwarg is None
    keyword_arguments = {argument.arg for argument in init.args.kwonlyargs}
    assert FIRST_CHAT_CONTROLLER_CALLBACKS <= keyword_arguments
    callback_annotations = {
        argument.arg: ast.unparse(argument.annotation)
        for argument in init.args.kwonlyargs
        if argument.arg in FIRST_CHAT_CONTROLLER_CALLBACKS
        and argument.annotation is not None
    }
    assert callback_annotations == {
        "screen_mounted_accessor": "Callable[[], bool]",
        "first_chat_presentation_snapshot": (
            "Callable[[], tuple[Any, Any, object | None]]"
        ),
        "apply_first_chat_control_selection": "Callable[[Any, Any], None]",
        "restore_first_chat_focus": "Callable[[object | None], None]",
    }
    stored_callbacks = {
        (target.attr, node.value.id)
        for node in ast.walk(init)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert {
        ("_screen_mounted_accessor", "screen_mounted_accessor"),
        (
            "_first_chat_presentation_snapshot_fn",
            "first_chat_presentation_snapshot",
        ),
        (
            "_apply_first_chat_control_selection_fn",
            "apply_first_chat_control_selection",
        ),
        ("_restore_first_chat_focus_fn", "restore_first_chat_focus"),
    } <= stored_callbacks
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FIRST_CHAT_CONTROLLER_CALLBACKS
        for node in ast.walk(init)
    ), "first-chat presentation callbacks must be stored, not called, at wiring time"


@pytest.mark.unit
def test_first_chat_task_ratchet_is_earned() -> None:
    """Require immutable design/Task 0 provenance and the earned move ceiling."""
    group = WAVE6_GROUPS["first_chat"]
    design_source = _source_at_revision(TASK_3070_9_DESIGN_BASE, _SCREEN_PATH)
    design_class = _class_node_from_source(
        design_source,
        "ChatScreen",
        _SCREEN_PATH,
    )
    task0_source = _source_at_revision(
        TASK_3070_9_TASK0_IMPLEMENTATION_BASE,
        _SCREEN_PATH,
    )
    task0_class = _class_node_from_source(task0_source, "ChatScreen", _SCREEN_PATH)
    current_source, current_class = _class_node(_SCREEN_PATH, "ChatScreen")

    def family_nodes(
        owner: ast.ClassDef,
    ) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        return [
            node
            for node in owner.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in group.moved
        ]

    design_family = family_nodes(design_class)
    task0_family = family_nodes(task0_class)
    design_counts = _method_name_counts(design_class)
    task0_counts = _method_name_counts(task0_class)
    current_counts = _method_name_counts(current_class)
    design_normalized = [
        ast.dump(node, include_attributes=False) for node in design_family
    ]
    task0_normalized = [
        ast.dump(node, include_attributes=False) for node in task0_family
    ]
    design_digest = hashlib.sha256("\n".join(design_normalized).encode()).hexdigest()
    task0_digest = hashlib.sha256("\n".join(task0_normalized).encode()).hexdigest()

    assert group.source_revision == POST_IMAGE_IMPLEMENTATION_BASE
    assert group.raw_lines == TASK_3070_9_DEFINITION_LINES
    assert frozenset(TASK_3070_9_FAMILY_NAMES) == group.moved
    assert tuple(node.name for node in design_family) == TASK_3070_9_FAMILY_NAMES
    assert tuple(node.name for node in task0_family) == TASK_3070_9_FAMILY_NAMES
    assert all(design_counts[name] == task0_counts[name] == 1 for name in group.moved)
    assert sum(_span(node) for node in design_family) == TASK_3070_9_DEFINITION_LINES
    assert sum(_span(node) for node in task0_family) == TASK_3070_9_DEFINITION_LINES
    assert design_normalized == task0_normalized
    assert design_digest == task0_digest == TASK_3070_9_FAMILY_SHA256
    assert len(design_source.splitlines()) == TASK_3070_9_TASK0_BASE_SCREEN_LINES
    assert _method_count(design_class) == TASK_3070_9_TASK0_BASE_METHODS
    assert len(task0_source.splitlines()) == TASK_3070_9_TASK0_BASE_SCREEN_LINES
    assert _method_count(task0_class) == TASK_3070_9_TASK0_BASE_METHODS
    assert TASK_3070_9_TASK0_MAX_SCREEN_LINES == (
        TASK_3070_9_TASK0_BASE_SCREEN_LINES - TASK_3070_9_DEFINITION_LINES
    )
    assert TASK_3070_9_TASK0_MAX_METHODS == (
        TASK_3070_9_TASK0_BASE_METHODS - len(TASK_3070_9_FAMILY_NAMES)
    )
    assert not any(current_counts[name] for name in group.moved), (
        "first-chat task methods returned to ChatScreen: "
        f"{sorted(name for name in group.moved if current_counts[name])}"
    )
    assert len(current_source.splitlines()) <= TASK_3070_9_TASK0_MAX_SCREEN_LINES
    assert _method_count(current_class) <= TASK_3070_9_TASK0_MAX_METHODS


@pytest.mark.unit
def test_first_chat_move_oracles_are_non_vacuous() -> None:
    """Prove every first-chat ownership and boundary oracle rejects a mutant."""
    target_source = "class ConsoleSessionController:\n" + "".join(
        f"    def {name}(self): return None\n" for name in TASK_3070_9_FAMILY_NAMES
    )
    target_class = ast.parse(target_source).body[0]
    assert isinstance(target_class, ast.ClassDef)

    screen_owner_mutant = ast.parse(
        "class ChatScreen:\n    def _first_chat_defaults_match(self): return True\n"
    ).body[0]
    assert isinstance(screen_owner_mutant, ast.ClassDef)
    with pytest.raises(AssertionError, match="still owned by ChatScreen"):
        _assert_first_chat_ownership_multiplicity(
            screen_owner_mutant,
            target_class,
        )

    target_methods = _methods_from_class(target_class)
    dom_mutant = dict(target_methods)
    dom_mutant["_current_first_chat_defaults"] = ast.parse(
        "def _current_first_chat_defaults(self): return self.query_one('#provider')"
    ).body[0]
    with pytest.raises(AssertionError, match="query DOM"):
        _assert_first_chat_method_boundaries(dom_mutant)

    sibling_mutant = dict(target_methods)
    sibling_mutant["eligible_console_first_chat_session_id"] = ast.parse(
        "def eligible_console_first_chat_session_id(self): "
        "return self._workspace.active_session_id"
    ).body[0]
    with pytest.raises(AssertionError, match="screen/sibling owners"):
        _assert_first_chat_method_boundaries(sibling_mutant)

    callback_complete = dict(target_methods)
    callback_complete["_current_first_chat_defaults"] = ast.parse(
        "def _current_first_chat_defaults(self):\n"
        "    return (\n"
        "        self._screen_mounted_accessor(),\n"
        "        self._first_chat_presentation_snapshot_fn(),\n"
        "        self._apply_first_chat_control_selection_fn(None, None),\n"
        "        self._restore_first_chat_focus_fn(None),\n"
        "    )\n"
    ).body[0]
    _assert_first_chat_presentation_callback_usage(callback_complete)
    callback_omission_mutant = dict(callback_complete)
    callback_omission_mutant["_current_first_chat_defaults"] = ast.parse(
        "def _current_first_chat_defaults(self):\n"
        "    return (\n"
        "        self._screen_mounted_accessor(),\n"
        "        self._first_chat_presentation_snapshot_fn(),\n"
        "        self._apply_first_chat_control_selection_fn(None, None),\n"
        "    )\n"
    ).body[0]
    with pytest.raises(AssertionError, match="presentation callbacks unused"):
        _assert_first_chat_presentation_callback_usage(callback_omission_mutant)

    direct_focus_mutant = dict(target_methods)
    direct_focus_mutant["_resync_console_after_first_chat_rollback"] = ast.parse(
        "def _resync_console_after_first_chat_rollback(self, token):\n"
        "    if token.is_mounted:\n"
        "        token.focus()\n"
    ).body[0]
    with pytest.raises(AssertionError, match="access presentation directly"):
        _assert_first_chat_method_boundaries(direct_focus_mutant)

    duplicate_target = ast.parse(
        target_source
        + "    def consume_pending_console_first_chat_intent(self): return True\n"
    ).body[0]
    empty_screen = ast.parse("class ChatScreen:\n    pass\n").body[0]
    assert isinstance(duplicate_target, ast.ClassDef)
    assert isinstance(empty_screen, ast.ClassDef)
    with pytest.raises(AssertionError, match="exactly once"):
        _assert_first_chat_ownership_multiplicity(empty_screen, duplicate_target)

    target_with_revision_state = ast.parse(
        "class ConsoleSessionController:\n"
        "    def __init__(self):\n"
        "        self._first_chat_handoff_notified_revision = None\n"
    ).body[0]
    revision_property_mutant = ast.parse(
        "class ChatScreen:\n"
        "    @property\n"
        "    def _first_chat_handoff_notified_revision(self):\n"
        "        return self._session._first_chat_handoff_notified_revision\n"
    ).body[0]
    assert isinstance(target_with_revision_state, ast.ClassDef)
    assert isinstance(revision_property_mutant, ast.ClassDef)
    with pytest.raises(AssertionError, match="revision-state shim"):
        _assert_first_chat_revision_state_ownership(
            revision_property_mutant,
            target_with_revision_state,
        )

    compatibility_mutant = ast.parse(
        "class ChatScreen:\n"
        "    def consume_pending_console_first_chat_intent(self):\n"
        "        return self._session.consume_pending_console_first_chat_intent()\n"
    ).body[0]
    assert isinstance(compatibility_mutant, ast.ClassDef)
    with pytest.raises(AssertionError, match="compatibility delegates"):
        _assert_no_first_chat_compatibility_delegates(
            _methods_from_class(compatibility_mutant)
        )


def _assert_fleet_ownership_contract(
    screen_methods: dict[str, ast.AST],
    target_methods: dict[str, ast.AST],
) -> None:
    """Require the reviewed fleet family to have one controller owner."""
    group = WAVE6_GROUPS["fleet"]
    missing = group.moved - target_methods.keys()
    assert not missing, (
        f"fleet methods missing from ConsoleFleetLifecycleController: {sorted(missing)}"
    )
    duplicates = group.moved & screen_methods.keys()
    assert not duplicates, (
        f"fleet methods still owned by ChatScreen: {sorted(duplicates)}"
    )


def _assert_fleet_ownership_multiplicity(
    screen_class: ast.ClassDef,
    target_class: ast.ClassDef,
) -> None:
    """Require zero screen and exactly one controller definition per fleet name."""
    group = WAVE6_GROUPS["fleet"]
    screen_counts = _method_name_counts(screen_class)
    target_counts = _method_name_counts(target_class)
    screen_duplicates = {
        name: screen_counts[name] for name in group.moved if screen_counts[name]
    }
    target_multiplicity = {
        name: target_counts[name] for name in group.moved if target_counts[name] != 1
    }
    assert not screen_duplicates, (
        f"fleet methods still owned by ChatScreen: {screen_duplicates}"
    )
    assert not target_multiplicity, (
        f"fleet controller methods must occur exactly once: {target_multiplicity}"
    )


def _assert_fleet_method_boundaries(methods: dict[str, ast.AST]) -> None:
    """Reject DOM access and direct screen/sibling handles."""
    dom_offenders = {
        name for name, method in methods.items() if _calls_dom_query(method)
    }
    assert not dom_offenders, (
        f"fleet controller methods query DOM: {sorted(dom_offenders)}"
    )
    composer_widget_offenders = {
        name
        for name, method in methods.items()
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "draft_text"
            for node in ast.walk(method)
        )
    }
    assert not composer_widget_offenders, (
        "fleet controller methods reach composer widgets: "
        f"{sorted(composer_widget_offenders)}"
    )

    forbidden_owners = frozenset(
        {
            "screen",
            "_screen",
            "app",
            "_app",
            "app_instance",
            "_app_instance",
            "controller",
            "_controller",
            "_console_chat_controller",
            "wake",
            "_wake",
            "fleet_wake",
            "_fleet_wake",
            "_workspace",
            "_session",
            "_agent",
            "_image",
            "_video",
            "_retrieval",
            "_skill",
            "_character",
            "_dictation",
            "_hands_free",
            "_message",
            "_prompts",
            "_prompt_queue",
        }
    )
    sibling_offenders = {
        name: sorted(_self_owner_accesses(method, forbidden_owners))
        for name, method in methods.items()
        if _self_owner_accesses(method, forbidden_owners)
    }
    assert not sibling_offenders, (
        f"fleet controller methods reach screen/sibling owners: {sibling_offenders}"
    )


def _assert_fleet_controller_boundary(controller: ast.ClassDef) -> None:
    """Require exact named callbacks and reject DOM/sibling capabilities."""
    methods = _methods_from_class(controller)
    group = WAVE6_GROUPS["fleet"]
    expected_methods = group.moved | {"prepare_session_run_markers"}
    assert _method_name_counts(controller) == Counter(
        {name: 1 for name in expected_methods | {"__init__"}}
    )

    init = methods["__init__"]
    assert isinstance(init, ast.FunctionDef)
    assert not init.args.posonlyargs
    assert [argument.arg for argument in init.args.args] == ["self"]
    assert {argument.arg for argument in init.args.kwonlyargs} == (
        FLEET_CONTROLLER_CALLBACKS
    )
    assert init.args.kw_defaults == [None] * len(init.args.kwonlyargs)
    assert init.args.vararg is None
    assert init.args.kwarg is None
    callback_annotations = {
        argument.arg: ast.unparse(argument.annotation)
        for argument in init.args.kwonlyargs
        if argument.annotation is not None
    }
    assert callback_annotations.keys() == FLEET_CONTROLLER_CALLBACKS
    assert not {
        name: annotation
        for name, annotation in callback_annotations.items()
        if "Callable[...," in annotation
    }, "fleet constructor callbacks must have explicit arity"

    expected_assignments = {
        *(f"_{name}" for name in FLEET_CONTROLLER_CALLBACKS),
        "_console_fleet_survivor_timer",
        "_console_fleet_unseen_cache",
    }
    assert _self_assignments(controller) == expected_assignments
    stored_callbacks = {
        (target.attr, node.value.id)
        for node in ast.walk(init)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert stored_callbacks == {
        (f"_{name}", name) for name in FLEET_CONTROLLER_CALLBACKS
    }
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FLEET_CONTROLLER_CALLBACKS
        for node in ast.walk(init)
    )
    _assert_fleet_method_boundaries(methods)
    assert methods.keys() - {"__init__"} == expected_methods


@pytest.mark.unit
def test_fleet_family_has_completed_controller_ownership() -> None:
    """Require all 16 fleet lifecycle methods solely on the controller."""
    group = WAVE6_GROUPS["fleet"]
    target_path = _REPO_ROOT / group.target_path

    assert target_path.exists(), "ConsoleFleetLifecycleController module is missing"
    _, screen_class = _class_node(_SCREEN_PATH, "ChatScreen")
    _, target_class = _class_node(target_path, group.target_class)
    screen_methods = _methods_from_class(screen_class)
    target_methods = _methods_from_class(target_class)
    _assert_fleet_ownership_contract(screen_methods, target_methods)
    _assert_fleet_ownership_multiplicity(screen_class, target_class)

    replacement_helpers = {
        "_displayed_console_composer_draft",
        "_console_screen_is_displayed",
    }
    assert not (replacement_helpers & screen_methods.keys()), (
        "ChatScreen gained replacement fleet helpers: "
        f"{sorted(replacement_helpers & screen_methods.keys())}"
    )


@pytest.mark.unit
def test_fleet_controller_has_only_named_non_dom_dependencies() -> None:
    """Lock the fleet owner behind its reviewed callback-only boundary."""
    group = WAVE6_GROUPS["fleet"]
    _, controller = _class_node(_REPO_ROOT / group.target_path, group.target_class)

    _assert_fleet_controller_boundary(controller)


@pytest.mark.unit
def test_fleet_task_ratchet_is_earned() -> None:
    """Require the immutable Task 0 and frozen final-rebase ratchets."""
    group = WAVE6_GROUPS["fleet"]
    task0_source = _source_at_revision(FLEET_TASK0_IMPLEMENTATION_BASE, _SCREEN_PATH)
    task0_class = _class_node_from_source(task0_source, "ChatScreen", _SCREEN_PATH)
    task0_methods = _methods_from_class(task0_class)
    task0_counts = _method_name_counts(task0_class)
    final_source = _source_at_revision(FLEET_FINAL_REBASE_BASE, _SCREEN_PATH)
    final_class = _class_node_from_source(final_source, "ChatScreen", _SCREEN_PATH)
    final_methods = _methods_from_class(final_class)
    final_counts = _method_name_counts(final_class)
    current_source, current_class = _class_node(_SCREEN_PATH, "ChatScreen")
    current_counts = _method_name_counts(current_class)

    assert len(task0_source.splitlines()) == FLEET_TASK0_BASE_SCREEN_LINES
    assert _method_count(task0_class) == FLEET_TASK0_BASE_METHODS
    assert group.moved <= task0_methods.keys()
    assert sum(_span(task0_methods[name]) for name in group.moved) == (
        FLEET_TASK0_DEFINITION_LINES
    )
    assert len(group.moved) == 16

    assert len(final_source.splitlines()) == FLEET_FINAL_REBASE_BASE_SCREEN_LINES
    assert _method_count(final_class) == FLEET_FINAL_REBASE_BASE_METHODS
    assert group.moved <= final_methods.keys()
    assert sum(_span(final_methods[name]) for name in group.moved) == (
        FLEET_FINAL_REBASE_DEFINITION_LINES
    )
    assert all(task0_counts[name] == final_counts[name] == 1 for name in group.moved)
    multiplicity_delta = {
        name: final_counts[name] - task0_counts[name]
        for name in task0_counts.keys() | final_counts.keys()
        if final_counts[name] != task0_counts[name]
    }
    assert multiplicity_delta == {
        name: 1 for name in FLEET_FINAL_REBASE_ADDED_SCREEN_METHODS
    }
    assert [
        ast.dump(task0_methods[name], include_attributes=False)
        for name in sorted(group.moved)
    ] == [
        ast.dump(final_methods[name], include_attributes=False)
        for name in sorted(group.moved)
    ]

    assert FLEET_TASK0_MAX_SCREEN_LINES == (
        FLEET_TASK0_BASE_SCREEN_LINES - FLEET_TASK0_DEFINITION_LINES
    )
    assert FLEET_TASK0_MAX_METHODS == FLEET_TASK0_BASE_METHODS - len(group.moved)
    assert FLEET_FINAL_REBASE_MAX_SCREEN_LINES == (
        FLEET_FINAL_REBASE_BASE_SCREEN_LINES - FLEET_FINAL_REBASE_DEFINITION_LINES
    )
    assert FLEET_FINAL_REBASE_MAX_METHODS == (
        FLEET_FINAL_REBASE_BASE_METHODS - len(group.moved)
    )
    assert not any(current_counts[name] for name in group.moved), (
        "fleet task methods returned to ChatScreen: "
        f"{sorted(name for name in group.moved if current_counts[name])}"
    )
    assert len(current_source.splitlines()) <= FLEET_FINAL_REBASE_MAX_SCREEN_LINES, (
        "fleet task screen-line ratchet remains RED: "
        f"{len(current_source.splitlines())} > {FLEET_FINAL_REBASE_MAX_SCREEN_LINES}"
    )
    assert _method_count(current_class) <= FLEET_FINAL_REBASE_MAX_METHODS, (
        "fleet task method ratchet remains RED: "
        f"{_method_count(current_class)} > {FLEET_FINAL_REBASE_MAX_METHODS}"
    )


@pytest.mark.unit
def test_fleet_move_oracles_are_non_vacuous() -> None:
    """Prove fleet ownership, DOM, and sibling checks reject exact mutants."""
    group = WAVE6_GROUPS["fleet"]
    target_methods = {
        name: ast.parse(f"def {name}(self): return None").body[0]
        for name in group.moved
    }
    screen_mutant = ast.parse(
        "class ChatScreen:\n"
        "    def _console_fleet_survivors_live(self):\n"
        "        return True\n"
    ).body[0]
    assert isinstance(screen_mutant, ast.ClassDef)
    with pytest.raises(AssertionError, match="still owned by ChatScreen"):
        _assert_fleet_ownership_contract(
            _methods_from_class(screen_mutant),
            target_methods,
        )

    duplicate_name = "_console_fleet_survivors_live"
    controller_duplicate_mutant = ast.parse(
        "class ConsoleFleetLifecycleController:\n"
        + "".join(
            f"    def {name}(self): return None\n" for name in sorted(group.moved)
        )
        + f"    def {duplicate_name}(self): return True\n"
    ).body[0]
    assert isinstance(controller_duplicate_mutant, ast.ClassDef)
    empty_screen_mutant = ast.parse("class ChatScreen:\n    pass\n").body[0]
    assert isinstance(empty_screen_mutant, ast.ClassDef)
    with pytest.raises(AssertionError, match="exactly once"):
        _assert_fleet_ownership_multiplicity(
            empty_screen_mutant,
            controller_duplicate_mutant,
        )

    controller_mutant = ast.parse(
        "class ConsoleFleetLifecycleController:\n"
        "    def leaked_dom(self):\n"
        "        return self.query_one('#composer')\n"
        "    def sibling_reach_through(self):\n"
        "        return self._workspace._poke_console_wake_retry()\n"
        "    def leaked_composer_widget(self):\n"
        "        return self._displayed_composer_draft_accessor().draft_text()\n"
    ).body[0]
    assert isinstance(controller_mutant, ast.ClassDef)
    mutant_methods = _methods_from_class(controller_mutant)
    with pytest.raises(AssertionError, match="query DOM"):
        _assert_fleet_method_boundaries({"leaked_dom": mutant_methods["leaked_dom"]})

    with pytest.raises(AssertionError, match="screen/sibling owners"):
        _assert_fleet_method_boundaries(
            {"sibling_reach_through": mutant_methods["sibling_reach_through"]}
        )

    with pytest.raises(AssertionError, match="composer widgets"):
        _assert_fleet_method_boundaries(
            {"leaked_composer_widget": mutant_methods["leaked_composer_widget"]}
        )


@pytest.mark.unit
def test_descriptor_contract_oracle_is_non_vacuous() -> None:
    """Prove the phase-safe descriptor oracle rejects known regressions.

    Pre-wiring access and screen-local shadow state must both fail the oracle.
    """

    class ReferenceDescriptor:
        def __get__(self, instance: object, owner: type | None = None) -> object:
            if instance is None:
                return self
            try:
                target = object.__getattribute__(instance, "_owner")
            except AttributeError as exc:
                raise RuntimeError("controller not wired") from exc
            return target.state

        def __set__(self, instance: object, value: object) -> None:
            try:
                target = object.__getattribute__(instance, "_owner")
            except AttributeError as exc:
                raise RuntimeError("controller not wired") from exc
            target.state = value

    class ReferenceScreen:
        state = ReferenceDescriptor()

    _assert_descriptor_contract(ReferenceScreen, "_owner", "state")

    class ShadowDescriptor:
        def __get__(self, instance: object, owner: type | None = None) -> object:
            if instance is None:
                return self
            return instance.__dict__.get("state")

        def __set__(self, instance: object, value: object) -> None:
            instance.__dict__["state"] = value

    class ShadowScreen:
        state = ShadowDescriptor()

    with pytest.raises(AssertionError, match="did not fail loudly"):
        _assert_descriptor_contract(ShadowScreen, "_owner", "state")


@pytest.mark.unit
def test_wave6_structural_oracles_are_non_vacuous() -> None:
    """Prove the structural oracles reject representative regressions.

    The fixture exercises DOM access, delegate reachability, physical span,
    initialization order, and controller-default failures.
    """
    sample = ast.parse(
        """
class Sample:
    browser_state = _ControllerState('_workspace', 'browser_state')

    def __init__(self):
        seen = self.state
        self.state = {}
        self.browser_state = ()

    def caller(self):
        return self.delegate()

    def delegate(self):
        first = 1
        second = 2
        third = 3
        fourth = 4
        return first + second + third + fourth

    def touches_dom(self):
        return self.query_one('#forbidden')

    def touches_dom_collection(self):
        return self.query('.forbidden')

    def reaches_sibling(self):
        return self._agent.refresh()

    def workspace_delegate(self):
        return self._workspace.transition('query', False)

    def unpacks_browser_state(self, values):
        self.other, [*self.browser_state] = values

    def writes_legacy_browser_state(self):
        self._console_workspace_conversation_query = 'legacy'
"""
    ).body[0]
    assert isinstance(sample, ast.ClassDef)
    sample_methods = _methods_from_class(sample)
    with pytest.raises(AssertionError):
        _assert_controller_default(sample_methods["__init__"], "state", ("dict",))
    with pytest.raises(AssertionError):
        _assert_screen_reads_after_build(sample_methods["__init__"], "state", 100)
    assert _controller_state_assignments(sample, frozenset({"browser_state"})) == {
        "browser_state": "_workspace"
    }
    assert _direct_self_writes(
        sample_methods["__init__"], frozenset({"browser_state"})
    ) == {"browser_state"}
    assert _direct_self_writes(
        sample_methods["unpacks_browser_state"], frozenset({"browser_state"})
    ) == {"browser_state"}
    assert _direct_writer_inventory(sample_methods, BROWSER_LEGACY_STATE_NAMES) == {
        "writes_legacy_browser_state": ["_console_workspace_conversation_query"]
    }
    assert _self_owner_accesses(
        sample_methods["reaches_sibling"], frozenset({"_agent"})
    ) == {"_agent"}
    assert _workspace_delegate_targets(sample_methods["workspace_delegate"]) == {
        "transition"
    }
    assert _calls_dom_query(sample_methods["touches_dom"])
    assert _calls_dom_query(sample_methods["touches_dom_collection"])
    assert not _calls_dom_query(sample_methods["caller"])
    original_binding = DELEGATE_BINDINGS.get("delegate")
    DELEGATE_BINDINGS["delegate"] = "caller"
    try:
        assert _has_real_delegate_binding(sample_methods, "delegate")
        with pytest.raises(AssertionError):
            _assert_delegate_contract(sample_methods, "delegate", complete=True)
        with pytest.raises(AssertionError):
            _assert_no_dom_access(sample_methods.values())
        DELEGATE_BINDINGS["delegate"] = "missing_caller"
        with pytest.raises(AssertionError):
            _assert_delegate_contract(sample_methods, "delegate", complete=False)
    finally:
        if original_binding is None:
            del DELEGATE_BINDINGS["delegate"]
        else:
            DELEGATE_BINDINGS["delegate"] = original_binding


@pytest.mark.unit
def test_browser_search_delegate_oracle_is_non_vacuous() -> None:
    """Prove the bounded delegate oracle rejects event and writer leakage."""
    reference = """
class Sample:
    def handler(self, event: Changed):
        event.stop()
        query = str(event.value or '')
        disabled = bool(getattr(getattr(event, 'input', None), 'disabled', False))
        self._workspace.transition(query, disabled)
"""

    def handler(source: str) -> ast.AST:
        sample = ast.parse(source).body[0]
        assert isinstance(sample, ast.ClassDef)
        return _methods_from_class(sample)["handler"]

    _assert_browser_search_delegate_contract(handler(reference))
    mutations = (
        reference.replace("event.stop()", "event.stop(); event.stop()"),
        reference.replace("str(event.value or '')", "str(event)"),
        reference.replace(
            "bool(getattr(getattr(event, 'input', None), 'disabled', False))",
            "bool(event)",
        ),
        reference.replace("transition(query, disabled)", "transition(event, disabled)"),
        reference.replace(
            "event.stop()",
            "event.stop()\n        self._console_conversation_browser_query = ''",
        ),
    )
    for mutant in mutations:
        with pytest.raises(AssertionError):
            _assert_browser_search_delegate_contract(handler(mutant))

    writer_mutant = handler(mutations[-1])
    assert _direct_self_writes(writer_mutant, COMPATIBILITY_TARGETS["_workspace"]) == {
        "_console_conversation_browser_query"
    }


@pytest.mark.unit
def test_assignment_oracles_order_same_line_ast_nodes_without_comparing_them() -> None:
    """Select assignments by source position when several share one line.

    Bare tuple ordering falls through from equal line numbers to comparing
    ``ast.AST`` objects, which raises ``TypeError``.
    """
    sample = ast.parse(
        "class Sample:\n"
        "    def initialize(self):\n"
        "        value = {}; value = set()\n"
        "        self.state = value; self.state = tuple()\n"
    ).body[0]
    assert isinstance(sample, ast.ClassDef)
    method = _methods_from_class(sample)["initialize"]
    state_assignments = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "state"
            for target in node.targets
        )
    ]

    assert _resolved_default_signature(
        ast.Name(id="value"), method, state_assignments[0].lineno
    ) == ("set",)
    first = _first_assignment_value(method, "state")
    assert isinstance(first, ast.Name) and first.id == "value"
