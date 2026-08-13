"""Architecture evidence for Console decomposition Wave 6 (TASK-3070.1)."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCREEN_PATH = _REPO_ROOT / "tldw_chatbook/UI/Screens/chat_screen.py"
IMPLEMENTATION_BASE = "bed39af6b004e4db86218fad01d2ea515b332135"
BASELINE_LINES = 22_338
BASELINE_METHODS = 705
LINE_OVERAGE = 4_611
METHOD_OVERAGE = 112
DESCRIPTOR_LINE_BUDGET = 64


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
            }
        ),
        delegates=frozenset({"on_console_workspace_conversation_search_changed"}),
        raw_lines=912,
        residue_lines=5,
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
                "_fetch_expression_image_bytes",
                "_apply_console_character_choice_async",
                "_refresh_active_character_avatar_if_scope_changed",
            }
        ),
        raw_lines=281,
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
    return {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _method_count(class_node: ast.ClassDef) -> int:
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in class_node.body
    )


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


def _has_on_binding(method: ast.AST) -> bool:
    for decorator in method.decorator_list:  # type: ignore[attr-defined]
        function = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(function, ast.Name) and function.id == "on":
            return True
    return False


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
    screen_source, screen_class = _class_node(_SCREEN_PATH, "ChatScreen")
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")

    assert len(base_source.splitlines()) == BASELINE_LINES
    assert _method_count(base_class) == BASELINE_METHODS
    assert len(screen_source.splitlines()) <= BASELINE_LINES, IMPLEMENTATION_BASE
    assert _method_count(screen_class) <= BASELINE_METHODS, IMPLEMENTATION_BASE
    assert set(WAVE6_GROUPS) == {
        "image",
        "video",
        "browser",
        "retrieval",
        "skill",
        "character",
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

        assert group.raw_names <= base_methods.keys()
        assert sum(_span(base_methods[item]) for item in group.raw_names) == (
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
                target_methods.values()
                if name != "browser"
                else (
                    target_methods[method_name]
                    for method_name in group.moved
                    if method_name in target_methods
                )
            )
            _assert_no_dom_access(methods_to_scan)


@pytest.mark.unit
def test_wave6_projection_clears_both_ratchet_overages() -> None:
    """Require the reviewed projection to clear both ratchet overages.

    The conservative line and method estimates must retain implementation
    margin after all documented residue is included.
    """
    raw_lines = sum(group.raw_lines for group in WAVE6_GROUPS.values())
    residue_lines = sum(group.residue_lines for group in WAVE6_GROUPS.values())
    removed_methods = sum(group.removed_methods for group in WAVE6_GROUPS.values())

    assert raw_lines == 5_151
    assert residue_lines + DESCRIPTOR_LINE_BUDGET == 183
    assert raw_lines - residue_lines - DESCRIPTOR_LINE_BUDGET == 4_968
    assert removed_methods == 129
    assert 4_968 > LINE_OVERAGE
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
    def __init__(self):
        seen = self.state
        self.state = {}

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
"""
    ).body[0]
    assert isinstance(sample, ast.ClassDef)
    sample_methods = _methods_from_class(sample)
    with pytest.raises(AssertionError):
        _assert_controller_default(sample_methods["__init__"], "state", ("dict",))
    with pytest.raises(AssertionError):
        _assert_screen_reads_after_build(sample_methods["__init__"], "state", 5)
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
