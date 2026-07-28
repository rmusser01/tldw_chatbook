from __future__ import annotations

import ast
from functools import cache
from pathlib import Path
import re
import warnings

import pytest

from Tests.reactive_ownership_contract import (
    RETAINED_TLDW_REACTIVES,
    RETIRED_TLDW_REACTIVES,
)
from tldw_chatbook.runtime_policy.bootstrap import RuntimePolicyContext
from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.state import (
    AppState,
    ChatState,
    NavigationState,
    NotesState,
    UIState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = PROJECT_ROOT / "tldw_chatbook"
APP_PATH = PRODUCTION_ROOT / "app.py"
BOOTSTRAP_PATH = PRODUCTION_ROOT / "runtime_policy" / "bootstrap.py"
SOURCE_STATE_PATH = PRODUCTION_ROOT / "runtime_policy" / "source_state.py"
SCHEDULES_WORKBENCH_PATH = (
    PRODUCTION_ROOT / "UI" / "Screens" / "scheduling" / "schedules_workbench.py"
)
SCREEN_STATE_STORE_PATH = (
    PRODUCTION_ROOT / "UI" / "Navigation" / "screen_state_store.py"
)
HANDOFF_STORE_PATH = PRODUCTION_ROOT / "UI" / "Navigation" / "pending_handoff_store.py"
CHAT_SCREEN_PATH = PRODUCTION_ROOT / "UI" / "Screens" / "chat_screen.py"
CHAT_SCREEN_STATE_PATH = PRODUCTION_ROOT / "UI" / "Screens" / "chat_screen_state.py"
MEDIA_WINDOW_PATH = PRODUCTION_ROOT / "UI" / "MediaWindow_v2.py"
MEDIA_SCREEN_PATH = PRODUCTION_ROOT / "UI" / "Screens" / "media_screen.py"
MEDIA_EVENTS_PATH = PRODUCTION_ROOT / "Event_Handlers" / "media_events.py"
LEGACY_CHAT_ROOT_NAMES = (
    "rag_expansion_provider_value",
    "chat_sidebar_collapsed",
    "chat_right_sidebar_collapsed",
    "chat_right_sidebar_width",
    "chat_sidebar_selected_prompt_id",
    "chat_sidebar_selected_prompt_system",
    "chat_sidebar_selected_prompt_user",
    "current_chat_is_ephemeral",
    "current_chat_conversation_id",
    "current_chat_active_character_data",
    "active_chat_tab_id",
    "chat_sessions",
    "chat_sidebar_loaded_prompt_id",
    "chat_sidebar_loaded_prompt_title_text",
    "chat_sidebar_loaded_prompt_system_text",
    "chat_sidebar_loaded_prompt_user_text",
    "chat_sidebar_loaded_prompt_keywords_text",
    "chat_sidebar_prompt_display_visible",
    "chat_settings_mode",
    "chat_settings_search_query",
    "_chat_state_lock",
    "current_ai_message_widget",
    "current_chat_worker",
    "current_chat_is_streaming",
    "current_chat_note_id",
    "current_chat_note_version",
    "_conversation_search_timer",
    "_chat_sidebar_prompt_search_timer",
    "_media_sidebar_search_timer",
    "media_search_current_page",
    "media_search_total_pages",
    "current_sidebar_media_item",
)
LEGACY_CHAT_ROOT_METHOD_NAMES = (
    "get_current_ai_message_widget",
    "set_current_ai_message_widget",
    "get_current_chat_worker",
    "set_current_chat_worker",
    "get_current_chat_is_streaming",
    "set_current_chat_is_streaming",
    *(f"watch_{name}" for name in LEGACY_CHAT_ROOT_NAMES),
)
LEGACY_CHAT_COMPOSITION_PATHS = (
    PRODUCTION_ROOT / "UI" / "Chat_Window.py",
    PRODUCTION_ROOT / "UI" / "Chat_Window_Enhanced.py",
)
LEGACY_CHAT_COMPOSITION_MODULES = {
    "Chat_Window",
    "Chat_Window_Enhanced",
}
LEGACY_CHAT_COMPOSITION_CLASSES = {
    "ChatWindow",
    "ChatWindowEnhanced",
}
LEGACY_CHAT_ROOT_MODULE_PATHS = (
    PRODUCTION_ROOT / "Event_Handlers" / "Chat_Events" / "chat_events.py",
    PRODUCTION_ROOT / "Event_Handlers" / "Chat_Events" / "chat_events_sidebar.py",
    PRODUCTION_ROOT
    / "Event_Handlers"
    / "Chat_Events"
    / "chat_events_sidebar_resize.py",
    PRODUCTION_ROOT / "Event_Handlers" / "Chat_Events" / "chat_events_tabs.py",
    PRODUCTION_ROOT / "Event_Handlers" / "Chat_Events" / "chat_events_worldbooks.py",
    PRODUCTION_ROOT / "Event_Handlers" / "Chat_Events" / "chat_streaming_events.py",
    PRODUCTION_ROOT / "Event_Handlers" / "sidebar_events.py",
    PRODUCTION_ROOT / "Event_Handlers" / "tab_initializers" / "chat_tab_initializer.py",
    PRODUCTION_ROOT / "Event_Handlers" / "worker_handlers" / "chat_worker_handler.py",
)
LEGACY_CCP_ROOT_NAMES = (
    "ccp_active_view",
    "ccp_api_provider_value",
    "current_editing_character_id",
    "current_editing_character_data",
    "conv_char_sidebar_left_collapsed",
    "conv_char_sidebar_right_collapsed",
    "current_conv_char_tab_conversation_id",
    "current_ccp_character_details",
    "current_prompt_id",
    "current_prompt_uuid",
    "current_prompt_name",
    "current_prompt_author",
    "current_prompt_details",
    "current_prompt_system",
    "current_prompt_user",
    "current_prompt_keywords_str",
    "current_prompt_version",
    "current_ccp_character_image",
    "_conv_char_search_timer",
    "_ccp_conversation_search_generation",
)
LEGACY_CCP_ROOT_METHOD_NAMES = (
    "switch_ccp_center_view",
    "_clear_prompt_fields",
    "_load_prompt_for_editing",
    "update_ccp_provider_reactive",
    "_update_model_select",
    "on_ccp_conversations_collapsible_toggle",
    *(f"watch_{name}" for name in LEGACY_CCP_ROOT_NAMES),
)
LEGACY_CCP_HANDLER_PATHS = (
    PRODUCTION_ROOT / "Event_Handlers" / "conv_char_events.py",
    PRODUCTION_ROOT / "Event_Handlers" / "character_ingest_events.py",
    PRODUCTION_ROOT / "Event_Handlers" / "prompt_ingest_events.py",
    PRODUCTION_ROOT / "Event_Handlers" / "worker_handlers" / "ai_generation_handler.py",
)
LEGACY_MEDIA_ROOT_NAMES = (
    "media_active_view",
    "_initial_media_view_slug",
    "current_media_type_filter_slug",
    "current_media_type_filter_display_name",
    "media_current_page",
    "current_loaded_media_item",
    "_media_search_timers",
    "_media_search_generation",
    "_initial_media_view",
    "media_runtime_state",
)
RETIRED_DESTINATION_ROOT_NAMES = (
    "current_selected_note_id",
    "current_selected_note_version",
    "current_selected_note_title",
    "current_selected_note_content",
    "notes_sort_by",
    "notes_sort_ascending",
    "notes_preview_mode",
    "notes_auto_save_enabled",
    "notes_auto_save_timer",
    "notes_last_save_time",
    "search_active_sub_tab",
    "ingest_active_view",
    "tools_settings_active_view",
    "evals_sidebar_collapsed",
    "_notes_search_timer",
    "_initial_search_sub_tab_view",
    "_initial_ingest_view",
    "_initial_tools_settings_view",
)
RETIRED_DESTINATION_ROOT_METHOD_NAMES = (
    "_activate_initial_ingest_view",
    "handle_notes_auto_save_toggle",
    *(f"watch_{name}" for name in RETIRED_DESTINATION_ROOT_NAMES),
)
RETIRED_APP_COMPANION_NAMES = (
    "USE_REBUILT_INGEST",
    "INGEST_NAV_BUTTON_IDS",
    "INGEST_VIEW_IDS",
    "ALL_INGEST_VIEW_IDS",
    "SEARCH_NAV_RAG_QA",
    "SEARCH_NAV_RAG_CHAT",
    "SEARCH_NAV_RAG_MANAGEMENT",
    "SEARCH_NAV_WEB_SEARCH",
    "SEARCH_NAV_EMBEDDINGS_CREATE",
    "SEARCH_NAV_EMBEDDINGS_MANAGE",
    "PlaceholderWindow",
)
RETIRED_TAB_INITIALIZERS_PATH = PRODUCTION_ROOT / "Event_Handlers" / "tab_initializers"
RETIRED_TAB_INITIALIZERS_MODULE = "tldw_chatbook.Event_Handlers.tab_initializers"
INGEST_EVENTS_PATH = PRODUCTION_ROOT / "Event_Handlers" / "ingest_events.py"
INGEST_UTILS_PATH = PRODUCTION_ROOT / "Event_Handlers" / "ingest_utils.py"
WORKER_EVENTS_PATH = PRODUCTION_ROOT / "Event_Handlers" / "worker_events.py"
MEDIA_INGEST_WORKERS_PATH = (
    PRODUCTION_ROOT / "Event_Handlers" / "media_ingest_workers.py"
)
TLDW_API_EVENTS_PATH = PRODUCTION_ROOT / "Event_Handlers" / "tldw_api_events.py"
STUDY_SCREEN_PATH = PRODUCTION_ROOT / "UI" / "Screens" / "study_screen.py"
ARTIFACTS_SCREEN_PATH = PRODUCTION_ROOT / "UI" / "Screens" / "artifacts_screen.py"
ACP_SCREEN_PATH = PRODUCTION_ROOT / "UI" / "Screens" / "acp_screen.py"
RECENT_WORK_SCREEN_PATHS = (
    PRODUCTION_ROOT / "UI" / "Screens" / "home_screen.py",
    PRODUCTION_ROOT / "UI" / "Screens" / "workflows_screen.py",
    PRODUCTION_ROOT / "UI" / "Screens" / "schedules_screen.py",
    PRODUCTION_ROOT / "UI" / "Screens" / "scheduling" / "schedules_workbench.py",
)
RETIRED_HANDOFF_FIELDS = (
    "pending_chat_handoff",
    "pending_console_launch",
    "pending_console_prompt_insert",
    "pending_study_scope_context",
    "pending_study_initial_section",
    "pending_notes_workspace_context",
    "pending_artifacts_chatbook_target_id",
    "pending_acp_session_target_id",
    "_screen_states",
)
PROJECTION_NAMES = (
    "current_runtime_backend",
    "runtime_backend",
    "active_server_id",
)
PROJECTION_SNAPSHOT = "_runtime_policy_projection_snapshot"
PROJECTION_PUBLISHER = "_publish_runtime_policy_projection"
PROJECTION_BOUNDARY = "_apply_runtime_policy_to_app"
PRIVATE_CONTEXT_STORE = "__runtime_policy_state_store"
PRIVATE_CONTEXT_CALLBACK = "__runtime_policy_projection_callback"
RUNTIME_POLICY_LOADER = "load_runtime_policy_for_app"


@cache
def _parse(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _chain(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _chain(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _constant_dynamic_name(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"getattr", "setattr", "delattr", "hasattr"}
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return node.args[1].value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and (
            (
                isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "vars"
                and len(node.func.value.args) == 1
            )
            or (
                isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "__dict__"
            )
        )
    ):
        return node.args[0].value
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


class _NamedOccurrenceCollector(ast.NodeVisitor):
    """Collect only syntax fields that can bind or reference one exact name."""

    def __init__(self, path: Path, target: str) -> None:
        self.path = path
        self.target = target
        self.scopes: list[str] = []
        self.occurrences: list[tuple[str, str, tuple[str, ...], int]] = []

    def _record(self, kind: str, lineno: int) -> None:
        self.occurrences.append(
            (
                str(self.path.relative_to(PROJECT_ROOT)),
                kind,
                tuple(self.scopes),
                lineno,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == self.target:
            self._record("class_definition", node.lineno)
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.target:
            self._record("function_definition", node.lineno)
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name == self.target:
            self._record("async_function_definition", node.lineno)
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.target:
            self._record(f"name_{type(node.ctx).__name__.lower()}", node.lineno)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == self.target:
            self._record(
                f"attribute_{type(node.ctx).__name__.lower()}",
                node.lineno,
            )
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        if node.name.rsplit(".", 1)[-1] == self.target:
            self._record("import_name", node.lineno)
        if node.asname == self.target:
            self._record("import_alias", node.lineno)

    def visit_arg(self, node: ast.arg) -> None:
        if node.arg == self.target:
            self._record("argument", node.lineno)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg == self.target:
            self._record("keyword", node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _constant_dynamic_name(node) == self.target:
            self._record("dynamic_name", node.lineno)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _constant_dynamic_name(node) == self.target:
            self._record("subscript_name", node.lineno)
        self.generic_visit(node)


def _occurrences(
    path: Path, target: str
) -> list[tuple[str, str, tuple[str, ...], int]]:
    collector = _NamedOccurrenceCollector(path, target)
    collector.visit(_parse(path))
    return collector.occurrences


def _production_occurrences(
    target: str,
) -> list[tuple[str, str, tuple[str, ...], int]]:
    found: list[tuple[str, str, tuple[str, ...], int]] = []
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        found.extend(_occurrences(path, target))
    return found


def _is_root_app_expression(node: ast.AST) -> bool:
    """Return whether an expression denotes the production app root."""
    chain = _chain(node)
    return bool(chain) and chain.rsplit(".", 1)[-1] in {"app", "app_instance"}


def _is_root_mapping_expression(
    node: ast.AST,
    root_predicate,
) -> bool:
    """Recognize direct, ``__dict__``, and ``vars`` root storage."""
    if root_predicate(node):
        return True
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "__dict__"
        and root_predicate(node.value)
    ):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "vars"
        and len(node.args) == 1
        and root_predicate(node.args[0])
    )


def _root_app_occurrences(
    path: Path,
    target: str,
) -> list[tuple[str, str, int]]:
    """Collect exact root-app access without rejecting destination-owned names."""
    relative = str(path.relative_to(PROJECT_ROOT))
    found: list[tuple[str, str, int]] = []
    for node in ast.walk(_parse(path)):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == target
            and _is_root_app_expression(node.value)
        ):
            found.append(
                (relative, f"attribute_{type(node.ctx).__name__.lower()}", node.lineno)
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr", "delattr", "hasattr"}
            and len(node.args) >= 2
            and _is_root_app_expression(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == target
        ):
            found.append((relative, f"dynamic_{node.func.id}", node.lineno))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _constant_dynamic_name(node) == target
            and _is_root_mapping_expression(
                node.func.value,
                _is_root_app_expression,
            )
            and not _is_root_app_expression(node.func.value)
        ):
            found.append((relative, "mapping_get", node.lineno))
        elif (
            isinstance(node, ast.Subscript)
            and _is_root_mapping_expression(
                node.value,
                _is_root_app_expression,
            )
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == target
        ):
            found.append(
                (
                    relative,
                    f"mapping_{type(node.ctx).__name__.lower()}",
                    node.lineno,
                )
            )
    return found


def _root_app_target_occurrences(
    path: Path,
    targets: frozenset[str],
) -> list[tuple[str, str, str, int]]:
    """Collect root-app access for a set of exact state names in one AST walk."""
    relative = str(path.relative_to(PROJECT_ROOT))
    found: list[tuple[str, str, str, int]] = []
    for node in ast.walk(_parse(path)):
        target: str | None = None
        kind: str | None = None
        if (
            isinstance(node, ast.Attribute)
            and node.attr in targets
            and _is_root_app_expression(node.value)
        ):
            target = node.attr
            kind = f"attribute_{type(node.ctx).__name__.lower()}"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr", "delattr", "hasattr"}
            and len(node.args) >= 2
            and _is_root_app_expression(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in targets
        ):
            target = node.args[1].value
            kind = f"dynamic_{node.func.id}"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _constant_dynamic_name(node) in targets
            and _is_root_mapping_expression(
                node.func.value,
                _is_root_app_expression,
            )
            and not _is_root_app_expression(node.func.value)
        ):
            target = _constant_dynamic_name(node)
            kind = "mapping_get"
        elif (
            isinstance(node, ast.Subscript)
            and _is_root_mapping_expression(
                node.value,
                _is_root_app_expression,
            )
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in targets
        ):
            target = node.slice.value
            kind = f"mapping_{type(node.ctx).__name__.lower()}"
        if target is not None and kind is not None:
            found.append((target, relative, kind, node.lineno))
    return found


def _class_body_bound_names(node: ast.AST) -> tuple[str, ...]:
    """Return names bound by one direct class-body assignment target."""
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(
            name for element in node.elts for name in _class_body_bound_names(element)
        )
    return ()


def _class_body_reactive_names(class_node: ast.ClassDef) -> frozenset[str]:
    """Return names assigned by direct class-body ``reactive(...)`` calls."""
    names: set[str] = set()
    for statement in class_node.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
            value = statement.value
        else:
            continue
        if not (
            isinstance(value, ast.Call)
            and _chain(value.func).rsplit(".", 1)[-1] == "reactive"
        ):
            continue
        for target in targets:
            names.update(_class_body_bound_names(target))
    return frozenset(names)


def _local_tldw_root_classes(path: Path) -> tuple[ast.ClassDef, ...]:
    """Return ``TldwCli`` and its transitive, in-module class mixins."""
    module = _parse(path)
    classes = {
        node.name: node for node in module.body if isinstance(node, ast.ClassDef)
    }
    root = classes["TldwCli"]
    ordered: list[ast.ClassDef] = []
    seen: set[str] = set()

    def add_with_local_bases(class_node: ast.ClassDef) -> None:
        if class_node.name in seen:
            return
        seen.add(class_node.name)
        ordered.append(class_node)
        for base in class_node.bases:
            base_class = classes.get(base.id) if isinstance(base, ast.Name) else None
            if base_class is not None:
                add_with_local_bases(base_class)

    add_with_local_bases(root)
    return tuple(ordered)


class _TldwCliRootOccurrenceCollector(ast.NodeVisitor):
    """Collect only syntax that can store retired state on ``TldwCli``."""

    def __init__(self, path: Path, target: str) -> None:
        self.path = path
        self.target = target
        self.scopes: list[str] = []
        self.nested_class_depth = 0
        self.occurrences: list[tuple[str, str, tuple[str, ...], int]] = []

    def _record(self, kind: str, lineno: int) -> None:
        self.occurrences.append(
            (
                str(self.path.relative_to(PROJECT_ROOT)),
                kind,
                tuple(self.scopes),
                lineno,
            )
        )

    def collect(self, app_class: ast.ClassDef) -> None:
        self.scopes.append(app_class.name)
        for statement in app_class.body:
            assignment_targets: tuple[ast.AST, ...] = ()
            if isinstance(statement, ast.Assign):
                assignment_targets = tuple(statement.targets)
            elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
                assignment_targets = (statement.target,)
            if any(
                self.target in _class_body_bound_names(target)
                for target in assignment_targets
            ):
                self._record("class_declaration", statement.lineno)
            self.visit(statement)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if len(self.scopes) == 1 and node.name == self.target:
            self._record("class_definition", node.lineno)
        self.scopes.append(node.name)
        self.nested_class_depth += 1
        self.generic_visit(node)
        self.nested_class_depth -= 1
        self.scopes.pop()

    def _is_root_receiver(self, node: ast.AST) -> bool:
        if _is_root_app_expression(node):
            return True
        return self.nested_class_depth == 0 and _chain(node) == "self"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if len(self.scopes) == 1 and node.name == self.target:
            self._record("function_definition", node.lineno)
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if len(self.scopes) == 1 and node.name == self.target:
            self._record("async_function_definition", node.lineno)
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == self.target and self._is_root_receiver(node.value):
            self._record(
                f"attribute_{type(node.ctx).__name__.lower()}",
                node.lineno,
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr", "delattr", "hasattr"}
            and len(node.args) >= 2
            and self._is_root_receiver(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == self.target
        ):
            self._record(f"dynamic_{node.func.id}", node.lineno)
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _constant_dynamic_name(node) == self.target
            and _is_root_mapping_expression(
                node.func.value,
                self._is_root_receiver,
            )
            and not self._is_root_receiver(node.func.value)
        ):
            self._record("mapping_get", node.lineno)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            _is_root_mapping_expression(
                node.value,
                self._is_root_receiver,
            )
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == self.target
        ):
            self._record(
                f"mapping_{type(node.ctx).__name__.lower()}",
                node.lineno,
            )
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if (
            self.nested_class_depth == 0
            and node.arg == "reactive_attr"
            and isinstance(node.value, ast.Constant)
            and node.value.value == self.target
        ):
            self._record("reactive_attr", node.lineno)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if (
            self.nested_class_depth == 0
            and self.target == "notes-auto-save-toggle"
            and node.value in {self.target, f"#{self.target}"}
        ):
            self._record("selector_literal", node.lineno)


def _tldw_cli_occurrences(
    target: str,
) -> list[tuple[str, str, tuple[str, ...], int]]:
    """Collect root-owned syntax inside the production ``TldwCli`` class."""
    collector = _TldwCliRootOccurrenceCollector(APP_PATH, target)
    collector.collect(_class_definition(APP_PATH, "TldwCli"))
    return collector.occurrences


def _class_definition(path: Path, name: str) -> ast.ClassDef:
    return next(
        node
        for node in _parse(path).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method_definition(
    class_node: ast.ClassDef,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _top_level_function(path: Path, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _parse(path).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _is_exception_category(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "__name__"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "type"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "exc"
    )


def _is_logger_call(node: ast.Call) -> bool:
    return any(
        isinstance(part, ast.Name) and part.id in {"logger", "logging"}
        for part in ast.walk(node.func)
    )


def _unsafe_handoff_log_parts(node: ast.Call) -> list[str]:
    unsafe: list[str] = []
    safe_chains = {"claim.channel.value", "claim.revision"}

    for argument in node.args:
        chain = _chain(argument)
        if isinstance(argument, ast.Constant):
            continue
        if chain in safe_chains or _is_exception_category(argument):
            continue
        unsafe.append(f"argument:{ast.unparse(argument)}")

    for keyword in node.keywords:
        if keyword.arg in {"exception", "exc_info"} and not (
            isinstance(keyword.value, ast.Constant) and keyword.value.value is False
        ):
            unsafe.append(f"traceback:{keyword.arg}")
        elif keyword.arg not in {"exception", "exc_info"}:
            unsafe.append(f"keyword:{keyword.arg or '**'}")

    return unsafe


def test_named_occurrence_guard_covers_all_retired_state_access_forms() -> None:
    path = PROJECT_ROOT / "direct-ast-guard-check.py"
    collector = _NamedOccurrenceCollector(path, "retired_state")
    collector.visit(
        ast.parse(
            """
owner.retired_state = value
owner.retired_state: str = value
owner.retired_state += value
del owner.retired_state
getattr(owner, "retired_state")
setattr(owner, "retired_state", value)
delattr(owner, "retired_state")
hasattr(owner, "retired_state")
vars(owner).get("retired_state")
owner.__dict__.get("retired_state")
mapping["retired_state"]
"""
        )
    )
    kinds = [kind for _path, kind, _scopes, _line in collector.occurrences]

    assert kinds.count("attribute_store") == 3
    assert kinds.count("attribute_del") == 1
    assert kinds.count("dynamic_name") == 6
    assert kinds.count("subscript_name") == 1


def test_class_body_reactive_guard_detects_assignments_and_annotations() -> None:
    tree = ast.parse(
        """class TldwCli:
    direct = reactive(0)
    annotated: reactive[str] = reactive("")
    qualified = textual.reactive(False)
    unrelated = other_factory()
"""
    )
    app_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    assert _class_body_reactive_names(app_class) == {
        "direct",
        "annotated",
        "qualified",
    }


def test_root_app_guard_detects_chained_dynamic_and_mapping_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = PROJECT_ROOT / "synthetic-root-app-guard.py"
    tree = ast.parse(
        """screen.app.ingest_active_view
self.window.app.ingest_active_view = value
getattr(screen.app, "ingest_active_view")
setattr(screen.app, "ingest_active_view", value)
delattr(self.window.app, "ingest_active_view")
hasattr(self.window.app, "ingest_active_view")
screen.app.__dict__["ingest_active_view"]
screen.app.__dict__["ingest_active_view"] = value
del screen.app.__dict__["ingest_active_view"]
vars(self.window.app)["ingest_active_view"]
vars(self.window.app)["ingest_active_view"] = value
del vars(self.window.app)["ingest_active_view"]
vars(screen.app).get("ingest_active_view")
self.window.app.__dict__.get("ingest_active_view")
destination.ingest_active_view
"""
    )
    monkeypatch.setitem(globals(), "_parse", lambda _path: tree)

    assert _is_root_app_expression(ast.parse("screen.app", mode="eval").body)
    assert _is_root_app_expression(
        ast.parse("self.window.app_instance", mode="eval").body
    )
    occurrences = _root_app_occurrences(path, "ingest_active_view")

    assert {line for _path, _kind, line in occurrences} == set(range(1, 15))
    assert sorted(kind for _path, kind, _line in occurrences) == sorted(
        (
            "attribute_load",
            "attribute_store",
            "dynamic_getattr",
            "dynamic_setattr",
            "dynamic_delattr",
            "dynamic_hasattr",
            "mapping_load",
            "mapping_store",
            "mapping_del",
            "mapping_load",
            "mapping_store",
            "mapping_del",
            "mapping_get",
            "mapping_get",
        )
    )


def test_root_app_target_guard_detects_only_root_retired_names_in_one_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = PROJECT_ROOT / "synthetic-root-app-target-guard.py"
    tree = ast.parse(
        """screen.app.retired_one
setattr(screen.app, "retired_two", value)
screen.app.__dict__["retired_one"] = value
vars(screen.app).get("retired_two")
handler(reactive_attr="retired_one")
destination.retired_one
"""
    )
    monkeypatch.setitem(globals(), "_parse", lambda _path: tree)

    occurrences = _root_app_target_occurrences(
        path,
        frozenset({"retired_one", "retired_two"}),
    )

    assert occurrences == [
        ("retired_one", "synthetic-root-app-target-guard.py", "attribute_load", 1),
        ("retired_two", "synthetic-root-app-target-guard.py", "dynamic_setattr", 2),
        ("retired_one", "synthetic-root-app-target-guard.py", "mapping_store", 3),
        ("retired_two", "synthetic-root-app-target-guard.py", "mapping_get", 4),
    ]


def test_local_tldw_root_classes_include_transitive_in_module_mixins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = ast.parse(
        """class RootStateMixin:
    inherited = reactive(0)

class QueueMixin(RootStateMixin):
    pass

class ExternalBase:
    pass

class App:
    externally_qualified = reactive("must not be inherited")

class TldwCli(QueueMixin, external.App):
    direct = reactive(False)
"""
    )
    monkeypatch.setitem(globals(), "_parse", lambda _path: tree)

    assert [node.name for node in _local_tldw_root_classes(APP_PATH)] == [
        "TldwCli",
        "QueueMixin",
        "RootStateMixin",
    ]


def test_tldw_cli_root_guard_detects_only_root_owned_syntax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = ast.parse(
        """class TldwCli:
    ingest_active_view = None

    def watch_ingest_active_view(self):
        self.ingest_active_view
        self.ingest_active_view = value
        del self.ingest_active_view
        getattr(self, "ingest_active_view")
        setattr(self, "ingest_active_view", value)
        delattr(self, "ingest_active_view")
        hasattr(self, "ingest_active_view")
        self.__dict__["ingest_active_view"]
        self.__dict__["ingest_active_view"] = value
        del self.__dict__["ingest_active_view"]
        vars(self)["ingest_active_view"]
        vars(self)["ingest_active_view"] = value
        del vars(self)["ingest_active_view"]
        vars(self).get("ingest_active_view")
        self.__dict__.get("ingest_active_view")
        handler(reactive_attr="ingest_active_view")
        destination.ingest_active_view = value
        ingest_active_view = value
"""
    )
    app_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    monkeypatch.setitem(
        globals(),
        "_class_definition",
        lambda _path, _name: app_class,
    )

    occurrences = _tldw_cli_occurrences("ingest_active_view")
    assert {line for _path, _kind, _scopes, line in occurrences} == {
        2,
        *range(5, 21),
    }
    assert all(line not in {21, 22} for _path, _kind, _scopes, line in occurrences)

    method_occurrences = _tldw_cli_occurrences("watch_ingest_active_view")
    assert [(kind, line) for _path, kind, _scopes, line in method_occurrences] == [
        ("function_definition", 4)
    ]


def test_tldw_cli_root_guard_recognizes_bare_and_selector_notes_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = ast.parse(
        """class TldwCli:
    def handle(self):
        self.query_one("#notes-auto-save-toggle")
        register("notes-auto-save-toggle")
"""
    )
    app_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    monkeypatch.setitem(
        globals(),
        "_class_definition",
        lambda _path, _name: app_class,
    )

    occurrences = _tldw_cli_occurrences("notes-auto-save-toggle")

    assert [(kind, line) for _path, kind, _scopes, line in occurrences] == [
        ("selector_literal", 3),
        ("selector_literal", 4),
    ]


def test_tldw_cli_root_guard_ignores_nested_owner_self_but_detects_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = ast.parse(
        """class TldwCli:
    def watch_ingest_active_view(self):
        pass

    class NestedOwner:
        ingest_active_view = None

        def mutate(self):
            self.ingest_active_view = value
            getattr(self, "ingest_active_view")
            self.__dict__["ingest_active_view"] = value
            handler(reactive_attr="ingest_active_view")

        def watch_ingest_active_view(self):
            pass

    class ingest_active_view:
        pass
"""
    )
    app_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    monkeypatch.setitem(
        globals(),
        "_class_definition",
        lambda _path, _name: app_class,
    )

    state_occurrences = _tldw_cli_occurrences("ingest_active_view")
    assert [(kind, line) for _path, kind, _scopes, line in state_occurrences] == [
        ("class_definition", 17)
    ]

    method_occurrences = _tldw_cli_occurrences("watch_ingest_active_view")
    assert [(kind, line) for _path, kind, _scopes, line in method_occurrences] == [
        ("function_definition", 2)
    ]


def test_legacy_chat_composition_modules_imports_and_classes_are_absent() -> None:
    violations: list[tuple[str, str, int]] = []

    for path in LEGACY_CHAT_COMPOSITION_PATHS:
        if path.exists():
            violations.append(
                (str(path.relative_to(PROJECT_ROOT)), "retired_module", 1)
            )

    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ClassDef):
                if node.name in LEGACY_CHAT_COMPOSITION_CLASSES:
                    violations.append(
                        (
                            str(path.relative_to(PROJECT_ROOT)),
                            f"class:{node.name}",
                            node.lineno,
                        )
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.rsplit(".", 1)[-1] in LEGACY_CHAT_COMPOSITION_MODULES:
                        violations.append(
                            (
                                str(path.relative_to(PROJECT_ROOT)),
                                f"import:{alias.name}",
                                node.lineno,
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                module_name = (node.module or "").rsplit(".", 1)[-1]
                imported_names = {alias.name for alias in node.names}
                if (
                    module_name in LEGACY_CHAT_COMPOSITION_MODULES
                    or imported_names.intersection(LEGACY_CHAT_COMPOSITION_CLASSES)
                ):
                    violations.append(
                        (
                            str(path.relative_to(PROJECT_ROOT)),
                            f"import_from:{node.module}",
                            node.lineno,
                        )
                    )

    assert violations == []


def test_legacy_chat_root_handler_modules_and_stream_messages_are_absent() -> None:
    assert [
        str(path.relative_to(PROJECT_ROOT))
        for path in LEGACY_CHAT_ROOT_MODULE_PATHS
        if path.exists()
    ] == []

    worker_source = WORKER_EVENTS_PATH.read_text(encoding="utf-8")
    assert "class StreamingChunk" not in worker_source
    assert "class StreamDone" not in worker_source
    assert "STREAMING_HANDLED_BY_EVENTS" not in worker_source


def test_chat_screen_has_no_legacy_composition_field_or_adapter() -> None:
    violations = {
        target: _occurrences(CHAT_SCREEN_PATH, target)
        for target in ("chat_window", "_ensure_chat_window")
        if _occurrences(CHAT_SCREEN_PATH, target)
    }

    assert violations == {}
    assert "#chat-window" not in CHAT_SCREEN_PATH.read_text(encoding="utf-8")


def test_chat_screen_state_module_contains_only_native_task_resume_state() -> None:
    violations = {
        target: _occurrences(CHAT_SCREEN_STATE_PATH, target)
        for target in ("MessageData", "TabState", "ChatScreenState")
        if _occurrences(CHAT_SCREEN_STATE_PATH, target)
    }
    assert violations == {}


def test_legacy_chat_root_state_and_accessors_are_absent() -> None:
    """Reject root mirrors while allowing same-named destination-owner fields."""
    violations: dict[
        str, list[tuple[str, str, tuple[str, ...], int] | tuple[str, str, int]]
    ] = {}
    for name in LEGACY_CHAT_ROOT_NAMES:
        occurrences = _occurrences(APP_PATH, name)
        for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
            if path == APP_PATH:
                continue
            occurrences.extend(_root_app_occurrences(path, name))
        if occurrences:
            violations[name] = occurrences

    for method_name in LEGACY_CHAT_ROOT_METHOD_NAMES:
        occurrences = _production_occurrences(method_name)
        if occurrences:
            violations[method_name] = occurrences

    assert violations == {}


def test_legacy_ccp_prompt_root_state_and_accessors_are_absent() -> None:
    """Reject CCP/prompt mirrors while allowing destination-owned state."""
    violations: dict[
        str, list[tuple[str, str, tuple[str, ...], int] | tuple[str, str, int]]
    ] = {}
    for name in LEGACY_CCP_ROOT_NAMES:
        occurrences = _occurrences(APP_PATH, name)
        for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
            if path == APP_PATH:
                continue
            occurrences.extend(_root_app_occurrences(path, name))
        if occurrences:
            violations[name] = occurrences

    for method_name in LEGACY_CCP_ROOT_METHOD_NAMES:
        occurrences = _production_occurrences(method_name)
        if occurrences:
            violations[method_name] = occurrences

    assert violations == {}


def test_legacy_ccp_prompt_handlers_and_compatibility_exports_are_absent() -> None:
    assert [
        str(path.relative_to(PROJECT_ROOT))
        for path in LEGACY_CCP_HANDLER_PATHS
        if path.exists()
    ] == []

    ingest_source = INGEST_EVENTS_PATH.read_text(encoding="utf-8")
    assert "character_ingest_events" not in ingest_source
    assert "prompt_ingest_events" not in ingest_source
    assert "INGEST_BUTTON_HANDLERS" not in ingest_source

    ingest_utils_source = INGEST_UTILS_PATH.read_text(encoding="utf-8")
    for retired_name in (
        "MAX_PROMPT_PREVIEWS",
        "PROMPT_FILE_FILTERS",
        "MAX_CHARACTER_PREVIEWS",
        "CHARACTER_FILE_FILTERS",
    ):
        assert retired_name not in ingest_utils_source


def test_tldw_cli_final_reactive_ownership_contract_is_exact() -> None:
    """Freeze the reviewed 61-descriptor disposition at the app boundary."""
    root_owner_classes = _local_tldw_root_classes(APP_PATH)
    assert len(RETAINED_TLDW_REACTIVES) == 2
    assert len(RETIRED_TLDW_REACTIVES) == 59
    assert RETAINED_TLDW_REACTIVES.isdisjoint(RETIRED_TLDW_REACTIVES)
    assert (
        frozenset().union(
            *(_class_body_reactive_names(node) for node in root_owner_classes)
        )
        == RETAINED_TLDW_REACTIVES
    )

    violations: dict[
        str, list[tuple[str, str, tuple[str, ...], int] | tuple[str, str, int]]
    ] = {}
    for name in sorted(RETIRED_TLDW_REACTIVES):
        occurrences: list[
            tuple[str, str, tuple[str, ...], int] | tuple[str, str, int]
        ] = []
        for root_owner_class in root_owner_classes:
            collector = _TldwCliRootOccurrenceCollector(APP_PATH, name)
            collector.collect(root_owner_class)
            occurrences.extend(collector.occurrences)
        if occurrences:
            violations[name] = [*occurrences]

    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        for name, relative, kind, line in _root_app_target_occurrences(
            path,
            RETIRED_TLDW_REACTIVES,
        ):
            violations.setdefault(name, []).append((relative, kind, line))

    root_methods = {
        (owner.name, node.name, node.lineno)
        for owner in root_owner_classes
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in sorted(RETIRED_TLDW_REACTIVES):
        for owner, method, line in root_methods:
            if method == f"watch_{name}":
                violations.setdefault(name, []).append(
                    (
                        str(APP_PATH.relative_to(PROJECT_ROOT)),
                        "watcher_definition",
                        (owner, method),
                        line,
                    )
                )

    assert violations == {}
    assert all(method != "watch_current_tab" for _owner, method, _line in root_methods)


def test_retired_destination_root_state_and_handlers_are_absent() -> None:
    """Reject root mirrors while allowing destination-owned view state."""
    violations: dict[
        str, list[tuple[str, str, tuple[str, ...], int] | tuple[str, str, int]]
    ] = {}
    for name in RETIRED_DESTINATION_ROOT_NAMES:
        occurrences: list[
            tuple[str, str, tuple[str, ...], int] | tuple[str, str, int]
        ] = [*_tldw_cli_occurrences(name)]
        for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
            occurrences.extend(_root_app_occurrences(path, name))
        if occurrences:
            violations[name] = occurrences

    for method_name in RETIRED_DESTINATION_ROOT_METHOD_NAMES:
        occurrences = _tldw_cli_occurrences(method_name)
        if occurrences:
            violations[method_name] = occurrences

    notes_toggle_registration = _tldw_cli_occurrences("notes-auto-save-toggle")
    if notes_toggle_registration:
        violations["notes-auto-save-toggle"] = notes_toggle_registration

    assert violations == {}


def test_retired_destination_root_companions_are_absent() -> None:
    """Keep directly orphaned constants and the lazy placeholder retired."""
    violations = {
        name: _occurrences(APP_PATH, name)
        for name in RETIRED_APP_COMPANION_NAMES
        if _occurrences(APP_PATH, name)
    }

    assert violations == {}


def test_retired_tldw_api_worker_context_and_pipeline_are_absent() -> None:
    """Keep the unproducible pre-Library ingest completion graph deleted."""
    context_name = "_last_tldw_api_request_context"
    context_occurrences = _production_occurrences(context_name)

    retired_paths = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (MEDIA_INGEST_WORKERS_PATH, TLDW_API_EVENTS_PATH)
        if path.exists()
    ]
    retired_screen_paths = [
        str(path.relative_to(PROJECT_ROOT))
        for path in sorted(PRODUCTION_ROOT.rglob("*.py"))
        if path.stem.casefold() in {"mediaingestscreen", "media_ingest_screen"}
    ]

    graph_violations: list[tuple[str, str, int]] = []
    exact_api_calls = re.compile(r"""(["'])api_calls\1""")
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        relative = str(path.relative_to(PROJECT_ROOT))
        source = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), start=1):
            if exact_api_calls.search(line):
                graph_violations.append(
                    (relative, "worker_group:api_calls", line_number)
                )
            if "media_ingest_workers" in line:
                graph_violations.append(
                    (relative, "module:media_ingest_workers", line_number)
                )
            if "#tldw-api-" in line:
                graph_violations.append((relative, "selector:#tldw-api-*", line_number))
            if "tldw_api_events" in line:
                graph_violations.append(
                    (relative, "module:tldw_api_events", line_number)
                )

    retired_symbols = {
        name: _production_occurrences(name)
        for name in (
            "MediaIngestScreen",
            "handle_tldw_api_worker_failure",
            "handle_tldw_api_worker_success",
            "_handle_api_calls",
        )
        if _production_occurrences(name)
    }

    assert context_occurrences == []
    assert retired_paths == []
    assert retired_screen_paths == []
    assert graph_violations == []
    assert retired_symbols == {}


def test_retired_tab_initializer_package_and_imports_stay_absent() -> None:
    """Keep the already-deleted legacy initializer entrypoint from reviving."""
    violations: list[tuple[str, str, int]] = []
    if RETIRED_TAB_INITIALIZERS_PATH.exists():
        violations.append(
            (
                str(RETIRED_TAB_INITIALIZERS_PATH.relative_to(PROJECT_ROOT)),
                "retired_package",
                1,
            )
        )

    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        relative = str(path.relative_to(PROJECT_ROOT))
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(RETIRED_TAB_INITIALIZERS_MODULE):
                        violations.append((relative, alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(RETIRED_TAB_INITIALIZERS_MODULE):
                    violations.append((relative, module, node.lineno))

    assert violations == []


def test_tldw_cli_does_not_render_or_retain_prompt_bodies() -> None:
    app_source = APP_PATH.read_text(encoding="utf-8")
    prohibited_prompt_widget_ids = (
        "#ccp-prompt-editor-view",
        "#ccp-editor-prompt-",
        "#prompt-import-",
    )

    assert all(
        widget_id not in app_source for widget_id in prohibited_prompt_widget_ids
    )
    assert _occurrences(APP_PATH, "current_prompt_system") == []
    assert _occurrences(APP_PATH, "current_prompt_user") == []


def test_tldw_cli_neither_imports_nor_instantiates_app_state() -> None:
    tree = _parse(APP_PATH)
    imported: list[str] = []
    constructed: list[int] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.extend(
                alias.name for alias in node.names if alias.name == "AppState"
            )
        elif isinstance(node, ast.Import):
            imported.extend(
                alias.name for alias in node.names if alias.name.endswith(".AppState")
            )
        elif isinstance(node, ast.Call) and _chain(node.func).endswith("AppState"):
            constructed.append(node.lineno)

    assert imported == []
    assert constructed == []


def test_tldw_cli_runtime_projections_are_getter_only_properties() -> None:
    from tldw_chatbook.app import TldwCli

    for name in PROJECTION_NAMES:
        descriptor = vars(TldwCli).get(name)
        assert isinstance(descriptor, property), name
        assert descriptor.fset is None, name


def test_runtime_projection_snapshot_has_exact_read_write_shape() -> None:
    relative_app = str(APP_PATH.relative_to(PROJECT_ROOT))
    observed = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(PROJECTION_SNAPSHOT)
    ]

    assert observed == [
        (relative_app, "name_store", ("TldwCli",)),
        (
            relative_app,
            "attribute_load",
            ("TldwCli", "current_runtime_backend"),
        ),
        (relative_app, "attribute_load", ("TldwCli", "runtime_backend")),
        (relative_app, "attribute_load", ("TldwCli", "active_server_id")),
        (
            relative_app,
            "attribute_store",
            ("TldwCli", PROJECTION_PUBLISHER),
        ),
    ]

    app_class = _class_definition(APP_PATH, "TldwCli")
    default_assignment = next(
        node
        for node in app_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == PROJECTION_SNAPSHOT
    )
    assert ast.literal_eval(default_assignment.value) == ("local", None)


def test_projection_boundary_and_publisher_have_exact_reference_allowlists() -> None:
    relative_app = str(APP_PATH.relative_to(PROJECT_ROOT))
    relative_bootstrap = str(BOOTSTRAP_PATH.relative_to(PROJECT_ROOT))

    boundary = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(PROJECTION_BOUNDARY)
    ]
    assert boundary == [
        (
            relative_bootstrap,
            "name_load",
            ("load_runtime_policy_for_app",),
        ),
        (relative_bootstrap, "function_definition", ()),
    ]

    publisher = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(PROJECTION_PUBLISHER)
    ]
    assert publisher == [
        (
            relative_app,
            "function_definition",
            ("TldwCli",),
        ),
        (
            relative_bootstrap,
            "dynamic_name",
            (PROJECTION_BOUNDARY,),
        ),
    ]


def test_projection_boundary_uses_only_the_private_publisher() -> None:
    boundary = _top_level_function(BOOTSTRAP_PATH, PROJECTION_BOUNDARY)
    dynamic_names = [
        _constant_dynamic_name(node)
        for node in ast.walk(boundary)
        if _constant_dynamic_name(node) is not None
    ]
    attribute_writes = [
        node.attr
        for node in ast.walk(boundary)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
    ]

    assert dynamic_names == [PROJECTION_PUBLISHER]
    assert attribute_writes == []
    assert not set(dynamic_names).intersection(PROJECTION_NAMES)


def test_tldw_cli_contains_no_public_projection_assignments() -> None:
    app_class = _class_definition(APP_PATH, "TldwCli")
    writes = [
        (node.attr, node.lineno)
        for node in ast.walk(app_class)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and node.attr in PROJECTION_NAMES
    ]
    dynamic_writes = [
        (_constant_dynamic_name(node), node.lineno)
        for node in ast.walk(app_class)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"setattr", "delattr"}
        and _constant_dynamic_name(node) in PROJECTION_NAMES
    ]

    assert writes == []
    assert dynamic_writes == []


def test_projection_boundary_never_accesses_app_state() -> None:
    boundary = _top_level_function(BOOTSTRAP_PATH, PROJECTION_BOUNDARY)
    accesses: list[int] = []
    for node in ast.walk(boundary):
        if isinstance(node, ast.Attribute) and node.attr == "app_state":
            accesses.append(node.lineno)
        elif _constant_dynamic_name(node) == "app_state":
            accesses.append(node.lineno)

    assert accesses == []


def test_runtime_policy_context_has_no_public_mutation_or_persistence_escape(
    tmp_path: Path,
) -> None:
    context = RuntimePolicyContext(
        RuntimeSourceState(),
        RuntimeSourceStateStore(tmp_path / "runtime-policy.json"),
    )
    alias = context

    assert isinstance(RuntimePolicyContext.state, property)
    assert RuntimePolicyContext.state.fset is None
    assert not hasattr(context, "persist")
    assert not hasattr(context, "store")
    with pytest.raises(AttributeError):
        alias.state = RuntimeSourceState(active_source="server")


def test_runtime_policy_context_private_fields_have_exact_structural_shape() -> None:
    relative_bootstrap = str(BOOTSTRAP_PATH.relative_to(PROJECT_ROOT))
    context_class = _class_definition(BOOTSTRAP_PATH, "RuntimePolicyContext")
    slots_assignment = next(
        node
        for node in context_class.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__slots__"
    )
    assert ast.literal_eval(slots_assignment.value) == (
        "_owner_thread_id",
        "_snapshot",
        PRIVATE_CONTEXT_CALLBACK,
        PRIVATE_CONTEXT_STORE,
    )

    store_occurrences = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(PRIVATE_CONTEXT_STORE)
    ]
    assert store_occurrences == [
        (
            relative_bootstrap,
            "attribute_store",
            ("RuntimePolicyContext", "__init__"),
        ),
        (
            relative_bootstrap,
            "attribute_load",
            ("RuntimePolicyContext", "commit_state"),
        ),
    ]

    callback_occurrences = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(
            PRIVATE_CONTEXT_CALLBACK
        )
    ]
    assert callback_occurrences == [
        (
            relative_bootstrap,
            "attribute_store",
            ("RuntimePolicyContext", "__init__"),
        ),
        (
            relative_bootstrap,
            "attribute_load",
            ("RuntimePolicyContext", "commit_state"),
        ),
        (
            relative_bootstrap,
            "attribute_load",
            ("RuntimePolicyContext", "commit_state"),
        ),
    ]

    assert (
        _production_occurrences("_RuntimePolicyContext__runtime_policy_state_store")
        == []
    )
    assert (
        _production_occurrences(
            "_RuntimePolicyContext__runtime_policy_projection_callback"
        )
        == []
    )


def test_runtime_policy_context_commit_has_one_immediate_private_store_save() -> None:
    context_class = _class_definition(BOOTSTRAP_PATH, "RuntimePolicyContext")
    commit = _method_definition(context_class, "commit_state")
    direct_saves = [
        statement
        for statement in commit.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "save"
        and isinstance(statement.value.func.value, ast.Attribute)
        and statement.value.func.value.attr == PRIVATE_CONTEXT_STORE
    ]

    assert len(direct_saves) == 1
    assert isinstance(direct_saves[0].value.args[0], ast.Name)
    assert direct_saves[0].value.args[0].id == "candidate"


def test_runtime_policy_preparation_precedes_single_direct_app_attachment() -> None:
    loader = _top_level_function(BOOTSTRAP_PATH, "load_runtime_policy_for_app")
    prepare_indices = [
        index
        for index, statement in enumerate(loader.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_prepare_runtime_policy_context"
    ]
    attach_indices = [
        index
        for index, statement in enumerate(loader.body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "app"
            and target.attr == "runtime_policy"
            for target in statement.targets
        )
    ]

    assert len(prepare_indices) == 1
    assert len(attach_indices) == 1
    assert prepare_indices[0] < attach_indices[0]


def test_tldw_cli_constructor_invokes_runtime_loader_as_standalone_expression() -> None:
    app_class = _class_definition(APP_PATH, "TldwCli")
    constructor = _method_definition(app_class, "__init__")
    standalone_calls = [
        statement
        for statement in constructor.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "load_runtime_policy_for_app"
    ]
    assigned_calls = [
        statement
        for statement in constructor.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "load_runtime_policy_for_app"
    ]

    assert len(standalone_calls) == 1
    assert assigned_calls == []


def test_retained_note_ingest_state_is_initialized_per_app_instance() -> None:
    app_class = _class_definition(APP_PATH, "TldwCli")
    constructor = _method_definition(app_class, "__init__")
    class_annotations = {
        statement.target.id: statement
        for statement in app_class.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    }
    constructor_assignments = [
        target.attr
        for statement in ast.walk(constructor)
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ]

    assert class_annotations["selected_note_files_for_import"].value is None
    assert constructor_assignments.count("selected_note_files_for_import") == 1
    assert constructor_assignments.count("last_note_import_dir") == 1
    assert "selected_notes_files_for_import" not in constructor_assignments
    assert "last_notes_import_dir" not in constructor_assignments


def test_runtime_policy_loader_has_exact_production_reference_allowlist() -> None:
    relative_app = str(APP_PATH.relative_to(PROJECT_ROOT))
    relative_bootstrap = str(BOOTSTRAP_PATH.relative_to(PROJECT_ROOT))
    observed = sorted(
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(RUNTIME_POLICY_LOADER)
    )

    assert observed == sorted(
        [
            (relative_app, "import_name", ()),
            (
                relative_app,
                "name_load",
                ("TldwCli", "__init__"),
            ),
            (
                relative_bootstrap,
                "function_definition",
                (),
            ),
            (
                relative_bootstrap,
                "name_load",
                ("ensure_runtime_policy_for_app",),
            ),
        ]
    )


def test_schedules_calls_authoritative_runtime_source_with_context_and_config() -> None:
    schedules_calls = [
        node
        for node in ast.walk(_parse(SCHEDULES_WORKBENCH_PATH))
        if isinstance(node, ast.Call)
        and _chain(node.func).endswith("set_authoritative_runtime_source")
    ]
    assert len(schedules_calls) == 1
    schedules_call = schedules_calls[0]
    assert _chain(schedules_call.args[0]) == "self.app_instance.runtime_policy"
    assert {
        keyword.arg: _chain(keyword.value)
        for keyword in schedules_call.keywords
        if keyword.arg is not None
    }["app_config"] == "self.app_instance.app_config"

    production_calls = [
        (path, node)
        for path in sorted(PRODUCTION_ROOT.rglob("*.py"))
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.Call)
        and _chain(node.func).endswith("set_authoritative_runtime_source")
    ]
    assert production_calls
    for path, call in production_calls:
        assert call.args, path
        first_argument = _chain(call.args[0])
        assert first_argument.endswith(".runtime_policy"), (
            path,
            first_argument,
        )
        app_config_keywords = [
            keyword for keyword in call.keywords if keyword.arg == "app_config"
        ]
        assert len(app_config_keywords) == 1, path


def test_runtime_source_state_store_references_are_confined_to_owner_modules() -> None:
    observed_paths = {
        PROJECT_ROOT / path
        for path, _kind, _scopes, _line in _production_occurrences(
            "RuntimeSourceStateStore"
        )
    }

    assert observed_paths == {BOOTSTRAP_PATH, SOURCE_STATE_PATH}


def test_legacy_runtime_policy_snapshot_symbol_is_absent_from_production() -> None:
    violations: list[tuple[str, str]] = []
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "runtime_policy_snapshot" in source:
            violations.append(
                (str(path.relative_to(PROJECT_ROOT)), "runtime_policy_snapshot")
            )

    assert violations == []


def test_screen_state_store_backing_entries_stay_private_to_owner_module() -> None:
    violations: list[tuple[str, int]] = []
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        if path == SCREEN_STATE_STORE_PATH:
            continue
        tree = _parse(path)
        imports_screen_state_store = any(
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.endswith("screen_state_store")
            and any(alias.name == "ScreenStateStore" for alias in node.names)
            for node in ast.walk(tree)
        )
        for node in ast.walk(tree):
            direct_access = isinstance(node, ast.Attribute) and node.attr == "_entries"
            dynamic_access = _constant_dynamic_name(node) == "_entries"
            if (direct_access or dynamic_access) and (
                imports_screen_state_store or "screen_state_store" in _chain(node)
            ):
                violations.append((str(path.relative_to(PROJECT_ROOT)), node.lineno))

    assert violations == []


def test_recent_work_consumers_use_owner_api_outside_threaded_workers() -> None:
    expected_scopes = {
        "tldw_chatbook/UI/Screens/home_screen.py": {
            ("HomeScreen", "_build_dashboard_input"),
        },
        "tldw_chatbook/UI/Screens/workflows_screen.py": {
            ("WorkflowsScreen", "on_mount"),
            ("WorkflowsScreen", "_latest_console_follow_item"),
        },
        "tldw_chatbook/UI/Screens/schedules_screen.py": {
            ("SchedulesScreen", "on_mount"),
            ("SchedulesScreen", "_latest_console_follow_item"),
        },
        "tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py": {
            ("SchedulesWorkbench", "_latest_console_follow_item_from_adapter"),
        },
    }
    observed: dict[str, set[tuple[str, ...]]] = {}
    for path in RECENT_WORK_SCREEN_PATHS:
        relative = str(path.relative_to(PROJECT_ROOT))
        scoped_calls: set[tuple[str, ...]] = set()
        for class_node in (
            node for node in _parse(path).body if isinstance(node, ast.ClassDef)
        ):
            for method in (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ):
                if any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "has_snapshots"
                    and _chain(node.func.value).endswith(".screen_state_store")
                    for node in ast.walk(method)
                ):
                    scoped_calls.add((class_node.name, method.name))
        observed[relative] = scoped_calls

    assert observed == expected_scopes
    assert all(
        method_name != "_refresh_latest_console_context"
        for path_scopes in observed.values()
        for _class_name, method_name in path_scopes
    )


def test_legacy_state_exports_remain_serialization_compatible() -> None:
    assert all(
        state_type is not None
        for state_type in (
            AppState,
            ChatState,
            NavigationState,
            NotesState,
            UIState,
        )
    )
    original = AppState(
        runtime_source=RuntimeSourceState(
            active_source="server",
            active_server_id="server-compatible",
            server_configured=True,
        )
    )
    payload = original.to_dict()

    assert AppState.from_dict(payload).to_dict() == payload


def test_legacy_state_docs_describe_compatibility_not_live_authority() -> None:
    app_state_source = (PRODUCTION_ROOT / "state" / "app_state.py").read_text(
        encoding="utf-8"
    )
    package_source = (PRODUCTION_ROOT / "state" / "__init__.py").read_text(
        encoding="utf-8"
    )
    rendered = f"{app_state_source}\n{package_source}".lower()

    assert "single source of truth" not in rendered
    assert "centralized state" not in rendered
    assert "compatibility" in rendered


def test_tldw_cli_has_no_retired_llm_destination_state_or_dispatcher() -> None:
    retired_names = (
        "llm_active_view",
        "button_handler_map",
        "_build_handler_map",
        "_update_llamacpp_log",
        "_update_llamafile_log",
        "_update_vllm_log",
        "_update_mlx_log",
        "_update_model_download_log",
    )
    violations = {
        name: _occurrences(APP_PATH, name)
        for name in retired_names
        if _occurrences(APP_PATH, name)
    }

    assert violations == {}


def test_tldw_cli_has_no_root_provider_descriptor_or_access() -> None:
    app_class = _class_definition(APP_PATH, "TldwCli")
    collector = _NamedOccurrenceCollector(APP_PATH, "chat_api_provider_value")
    collector.visit(app_class)

    assert collector.occurrences == []


def test_tldw_cli_has_no_retired_media_destination_state() -> None:
    app_class = _class_definition(APP_PATH, "TldwCli")
    violations = {}
    for name in LEGACY_MEDIA_ROOT_NAMES:
        collector = _NamedOccurrenceCollector(APP_PATH, name)
        collector.visit(app_class)
        if collector.occurrences:
            violations[name] = collector.occurrences

    assert violations == {}


def test_media_runtime_state_is_constructed_only_by_the_destination() -> None:
    app_occurrences = _occurrences(APP_PATH, "MediaRuntimeState")
    screen_occurrences = _occurrences(MEDIA_SCREEN_PATH, "MediaRuntimeState")
    window_occurrences = _occurrences(MEDIA_WINDOW_PATH, "MediaRuntimeState")

    assert app_occurrences == []
    assert screen_occurrences == []
    assert any(
        kind == "name_load" and "MediaWindow.__init__" in ".".join(scopes)
        for _path, kind, scopes, _line in window_occurrences
    )


def test_media_window_has_no_duplicate_media_active_view_descriptor() -> None:
    media_window_class = _class_definition(MEDIA_WINDOW_PATH, "MediaWindow")
    collector = _NamedOccurrenceCollector(MEDIA_WINDOW_PATH, "media_active_view")
    collector.visit(media_window_class)

    assert collector.occurrences == []


def test_media_events_module_contains_contracts_not_root_handlers() -> None:
    module = _parse(MEDIA_EVENTS_PATH)
    module_functions = [
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    module_classes = {
        node.name for node in module.body if isinstance(node, ast.ClassDef)
    }

    assert module_functions == []
    assert "MediaMetadataUpdateEvent" in module_classes
    assert "MediaTypeSelectedEvent" not in module_classes


def test_tldw_cli_has_no_constant_reactive_attribute_dispatch() -> None:
    app_class = _class_definition(APP_PATH, "TldwCli")
    violations = [
        (node.lineno, ast.unparse(node.value))
        for node in ast.walk(app_class)
        if isinstance(node, ast.keyword)
        and node.arg == "reactive_attr"
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]

    assert violations == []


def test_production_does_not_import_retired_llm_navigation_events() -> None:
    violations: list[tuple[str, int, str]] = []
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.rsplit(".", 1)[-1] == "llm_nav_events":
                        violations.append(
                            (
                                str(path.relative_to(PROJECT_ROOT)),
                                node.lineno,
                                alias.name,
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "llm_nav_events":
                        violations.append(
                            (
                                str(path.relative_to(PROJECT_ROOT)),
                                node.lineno,
                                f"{node.module}.{alias.name}",
                            )
                        )

    assert violations == []


def test_retired_raw_handoff_fields_are_absent_from_production() -> None:
    violations = {
        field: _production_occurrences(field)
        for field in RETIRED_HANDOFF_FIELDS
        if _production_occurrences(field)
    }

    assert violations == {}


def test_pending_handoff_slots_are_private_to_the_owner_module() -> None:
    relative_owner = str(HANDOFF_STORE_PATH.relative_to(PROJECT_ROOT))
    owner_occurrences = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences("_slots")
        if path == relative_owner
    ]
    assert owner_occurrences == [
        (
            relative_owner,
            "attribute_store",
            ("PendingHandoffStore", "__init__"),
        ),
        (
            relative_owner,
            "attribute_load",
            ("PendingHandoffStore", "_slot_for"),
        ),
    ]

    external_accesses: list[tuple[str, int]] = []
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        if path == HANDOFF_STORE_PATH:
            continue
        source = path.read_text(encoding="utf-8")
        if "pending_handoffs" not in source and "PendingHandoffStore" not in source:
            continue
        for node in ast.walk(_parse(path)):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "_slots"
                or _constant_dynamic_name(node) == "_slots"
            ):
                external_accesses.append(
                    (str(path.relative_to(PROJECT_ROOT)), node.lineno)
                )

    assert external_accesses == []


def test_pending_handoff_owner_has_no_persistence_or_serialization_calls() -> None:
    forbidden_calls = {
        "asdict",
        "dump",
        "dumps",
        "json",
        "model_dump",
        "model_dump_json",
        "open",
        "save",
        "serialize",
        "to_dict",
        "to_json",
        "write",
        "write_bytes",
        "write_text",
    }
    forbidden_imports = {"json", "pickle", "shelve"}
    tree = _parse(HANDOFF_STORE_PATH)
    calls = [
        (_chain(node.func), node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _chain(node.func).rsplit(".", 1)[-1] in forbidden_calls
    ]
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                (alias.name, node.lineno)
                for alias in node.names
                if alias.name.split(".", 1)[0] in forbidden_imports
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".", 1)[0] in forbidden_imports
        ):
            imports.append((node.module, node.lineno))

    assert calls == []
    assert imports == []


def test_handoff_exception_logs_are_metadata_only() -> None:
    app_class = _class_definition(APP_PATH, "TldwCli")
    chat_class = _class_definition(CHAT_SCREEN_PATH, "ChatScreen")
    study_class = _class_definition(STUDY_SCREEN_PATH, "StudyScreen")
    artifacts_class = _class_definition(ARTIFACTS_SCREEN_PATH, "ArtifactsScreen")
    acp_class = _class_definition(ACP_SCREEN_PATH, "ACPScreen")
    methods = (
        (APP_PATH, _method_definition(app_class, "_stage_handoff")),
        (
            CHAT_SCREEN_PATH,
            _method_definition(chat_class, "_consume_pending_console_launch"),
        ),
        (
            CHAT_SCREEN_PATH,
            _method_definition(chat_class, "_consume_pending_console_prompt_insert"),
        ),
        (
            CHAT_SCREEN_PATH,
            _method_definition(chat_class, "_consume_pending_chat_handoff"),
        ),
        (
            CHAT_SCREEN_PATH,
            _method_definition(chat_class, "_stage_handoff_as_console_live_work"),
        ),
        (
            CHAT_SCREEN_PATH,
            _method_definition(
                chat_class,
                "consume_pending_console_provider_intent",
            ),
        ),
        (
            STUDY_SCREEN_PATH,
            _method_definition(study_class, "_apply_pending_scope_handoff"),
        ),
        (
            STUDY_SCREEN_PATH,
            _method_definition(study_class, "_apply_pending_section_handoff"),
        ),
        (
            ARTIFACTS_SCREEN_PATH,
            _method_definition(artifacts_class, "_start_chatbook_refresh"),
        ),
        (
            ARTIFACTS_SCREEN_PATH,
            _method_definition(artifacts_class, "_apply_chatbook_refresh_outcome"),
        ),
        (
            ARTIFACTS_SCREEN_PATH,
            _method_definition(artifacts_class, "_exact_local_chatbook_console_launch"),
        ),
        (
            ACP_SCREEN_PATH,
            _method_definition(acp_class, "_consume_pending_session_target"),
        ),
    )
    violations: list[tuple[str, str, int, list[str]]] = []
    for path, method in methods:
        for node in ast.walk(method):
            if not isinstance(node, ast.Call) or not _is_logger_call(node):
                continue
            unsafe = _unsafe_handoff_log_parts(node)
            if unsafe:
                violations.append(
                    (
                        str(path.relative_to(PROJECT_ROOT)),
                        method.name,
                        node.lineno,
                        unsafe,
                    )
                )

    assert violations == []
