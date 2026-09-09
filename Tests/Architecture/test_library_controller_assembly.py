"""Construction and live dependency contracts for existing Library owners."""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens import library_screen
from tldw_chatbook.UI.Library_Modules.library_collections_controller import (
    LibraryCollectionsController,
)
from tldw_chatbook.UI.Library_Modules.library_conversation_reader_controller import (
    LibraryConversationReaderController,
)
from tldw_chatbook.UI.Library_Modules.library_conversations_controller import (
    LibraryConversationsController,
)
from tldw_chatbook.UI.Library_Modules.library_export_controller import (
    LibraryExportController,
)
from tldw_chatbook.UI.Library_Modules.library_rag_search_controller import (
    LibraryRagSearchController,
)
from tldw_chatbook.UI.Library_Modules.library_skills_controller import (
    LibrarySkillsController,
)
from tldw_chatbook.UI.Library_Modules.wiring import build_library_controllers

_OWNERS = (
    (
        "_conversation_reader_controller",
        LibraryConversationReaderController,
        "_conversations_state",
    ),
    (
        "_conversations_controller",
        LibraryConversationsController,
        "_conversations_state",
    ),
    ("_export_controller", LibraryExportController, "_export_state"),
    ("_collections_controller", LibraryCollectionsController, "_collections_state"),
    ("_rag_search_controller", LibraryRagSearchController, "_rag_search_state"),
    ("_skills_controller", LibrarySkillsController, "_skills_state"),
)


@pytest.mark.parametrize(("owner_name", "owner_type", "state_name"), _OWNERS)
def test_existing_controller_reads_replaced_state_at_call_time(
    owner_name: str, owner_type: type, state_name: str
) -> None:
    """Resolve replacement state through the existing controller's live accessor.

    Args:
        owner_name: Screen attribute holding the controller.
        owner_type: Expected controller class.
        state_name: Screen state attribute read by the controller's accessor.
    """
    screen = library_screen.LibraryScreen(SimpleNamespace(app_config={}))
    controller = getattr(screen, owner_name)
    assert type(controller) is owner_type
    accessor = getattr(controller, f"{state_name}_accessor")
    assert accessor() is getattr(screen, state_name)

    replacement = object()
    setattr(screen, state_name, replacement)

    assert accessor() is replacement


def test_conversation_sibling_lookup_is_late_bound() -> None:
    """Resolve the current sibling controller after its screen attribute changes."""
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
    """Preserve controller construction order and explicit live lambda ports."""
    source = inspect.getsource(build_library_controllers)
    tree = ast.parse(textwrap.dedent(source))
    expected = [owner_type.__name__ for _, owner_type, _ in _OWNERS]
    assignments = [
        node
        for node in tree.body[0].body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id in expected
    ]
    assert [node.value.func.id for node in assignments] == expected
    for node, (owner_name, _, _) in zip(assignments, _OWNERS, strict=True):
        call = node.value
        assert len(node.targets) == 1
        assert ast.unparse(node.targets[0]) == f"screen.{owner_name}"
        assert [ast.unparse(arg) for arg in call.args] == ["screen"]
        assert all(keyword.arg is not None for keyword in call.keywords)
        assert all(isinstance(keyword.value, ast.Lambda) for keyword in call.keywords)


def test_assembly_preserves_media_state_and_later_controller_order() -> None:
    """Keep media state before assembly and subsequent controller order intact."""
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
    assert [ast.unparse(arg) for arg in statements[position].value.args] == ["self"]
    assert statements[position].value.keywords == []
    previous, following = statements[position - 1], statements[position + 1]
    assert ast.unparse(previous.targets[0]) == "self._media_state"
    assert ast.unparse(previous.value.func) == "LibraryMediaState"
    assert ast.unparse(following.targets[0]) == "self._ingest_controller"
    assert ast.unparse(following.value.func) == "LibraryIngestController"
    later_calls = [
        ast.unparse(node.value.func)
        for node in statements[position + 1 :]
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func)
        in {
            "LibraryIngestController",
            "LibraryPromptsController",
            "LibraryMediaController",
            "self._load_library_reader_preference_snapshot",
        }
    ]
    assert later_calls == [
        "LibraryIngestController",
        "LibraryPromptsController",
        "LibraryMediaController",
        "self._load_library_reader_preference_snapshot",
    ]
