"""Construction and live dependency contracts for existing Library owners."""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens import library_screen
from tldw_chatbook.UI.Library_Modules import wiring
from tldw_chatbook.UI.Library_Modules.wiring import build_library_controllers

_OWNERS = (
    (
        "_conversation_reader_controller",
        "LibraryConversationReaderController",
        "_conversations_state",
    ),
    (
        "_conversations_controller",
        "LibraryConversationsController",
        "_conversations_state",
    ),
    ("_export_controller", "LibraryExportController", "_export_state"),
    ("_collections_controller", "LibraryCollectionsController", "_collections_state"),
    ("_rag_search_controller", "LibraryRagSearchController", "_rag_search_state"),
    ("_skills_controller", "LibrarySkillsController", "_skills_state"),
)


@pytest.mark.parametrize(("owner_name", "class_name", "state_name"), _OWNERS)
def test_existing_controller_reads_replaced_state_at_call_time(
    owner_name: str, class_name: str, state_name: str
) -> None:
    screen = library_screen.LibraryScreen(SimpleNamespace(app_config={}))
    controller = getattr(screen, owner_name)
    assert type(controller) is getattr(wiring, class_name)
    accessor = getattr(controller, f"{state_name}_accessor")
    assert accessor() is getattr(screen, state_name)

    replacement = object()
    setattr(screen, state_name, replacement)

    assert accessor() is replacement


def test_conversation_sibling_lookup_is_late_bound() -> None:
    screen = library_screen.LibraryScreen(SimpleNamespace(app_config={}))
    controller = screen._conversations_controller
    first, second = object(), object()
    screen._conversation_reader_controller = SimpleNamespace(
        _ensure_library_conversation_reader_selection=lambda: first
    )
    assert controller._ensure_reader_selection_fn() is first

    screen._conversation_reader_controller = SimpleNamespace(
        _ensure_library_conversation_reader_selection=lambda: second
    )
    assert controller._ensure_reader_selection_fn() is second


def test_existing_controller_assembly_keeps_order_and_explicit_live_ports() -> None:
    source = inspect.getsource(build_library_controllers)
    tree = ast.parse(textwrap.dedent(source))
    expected = [class_name for _, class_name, _ in _OWNERS]
    calls = [
        node.value
        for node in tree.body[0].body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id in expected
    ]
    assert [call.func.id for call in calls] == expected
    for call in calls:
        assert len(call.args) == 1
        assert all(keyword.arg is not None for keyword in call.keywords)
        assert all(isinstance(keyword.value, ast.Lambda) for keyword in call.keywords)


def test_assembly_stays_between_state_creation_and_preference_loading() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(library_screen.LibraryScreen.__init__))
    )
    statements = tree.body[0].body
    positions = [
        index
        for index, node in enumerate(statements)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "build_library_controllers"
    ]
    assert len(positions) == 1
    position = positions[0]
    previous, following = statements[position - 1], statements[position + 1]
    assert ast.unparse(statements[position - 2].targets[0]) == "self._skills_state"
    assert ast.unparse(previous.targets[0]) == "self._prompts_state"
    assert ast.unparse(following.targets[0]) == "self._ingest_controller"
    assert ast.unparse(following.value.func) == "LibraryIngestController"
    assert (
        ast.unparse(statements[position + 2].targets[0]) == "self._prompts_controller"
    )
    assert (
        ast.unparse(statements[position + 2].value.func) == "LibraryPromptsController"
    )
    assert (
        ast.unparse(statements[position + 3].value.func)
        == "self._load_library_reader_preference_snapshot"
    )
