"""Tests for the manual-only Console Library search surface."""

from unittest.mock import Mock

import pytest

from tldw_chatbook.Chat.console_display_state import ConsoleRetrievalScopeState
from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController
from tldw_chatbook.Widgets.Console.console_library_search_modal import (
    CONSOLE_RAG_DEFAULT_SOURCE_TYPES,
    ConsoleLibrarySearchModal,
)


def _wire_retrieval(
    screen: Mock,
    *,
    draft: str | None,
    state: ConsoleRetrievalScopeState | None = None,
) -> None:
    controller = object.__new__(ConsoleRetrievalController)
    controller.app_instance = screen.app
    controller._push_screen = screen.app.push_screen
    controller._composer_draft = lambda: draft
    controller._library_rag_query = lambda: screen._console_library_rag_query
    controller._library_rag_source_scope = lambda: CONSOLE_RAG_DEFAULT_SOURCE_TYPES
    controller._build_console_retrieval_scope_state = lambda: (
        state or ConsoleRetrievalScopeState.unscoped()
    )
    screen._retrieval = controller


@pytest.mark.unit
@pytest.mark.parametrize(
    "draft",
    (
        "/Users/x/notes.md",
        "file:///Users/x/notes.md",
        "https://example.com/incident-notes",
        "x" * 201,
        "  preserve this spacing exactly  ",
    ),
)
def test_search_modal_always_prefills_the_exact_composer_draft(draft: str) -> None:

    screen = Mock()
    composer = Mock()
    composer.draft_text.return_value = draft
    screen._console_composer_or_none.return_value = composer
    _wire_retrieval(screen, draft=draft)

    screen._retrieval.open_library_search()

    modal = screen.app.push_screen.call_args.args[0]
    assert isinstance(modal, ConsoleLibrarySearchModal)
    assert modal._query == draft


def test_search_modal_receives_the_current_item_scope_summary() -> None:

    screen = Mock()
    screen._console_composer_or_none.return_value = None
    screen._console_library_rag_query = "find this"
    _wire_retrieval(
        screen,
        draft=None,
        state=ConsoleRetrievalScopeState(is_scoped=True, item_count=3),
    )

    screen._retrieval.open_library_search()

    modal = screen.app.push_screen.call_args.args[0]
    assert modal._item_scope_summary == "Scope: 3 items"


def test_search_modal_copy_says_this_search_only_and_names_policy_separation() -> None:
    modal = ConsoleLibrarySearchModal(query="find this")

    copy = modal._status_copy()
    assert "this send only" in copy
    assert "Automatic retrieval" in copy
    assert "assistant Library access" in copy
    assert modal._scope_summary().startswith("This search only · Sources:")


@pytest.mark.asyncio
async def test_workbench_search_library_action_always_opens_search_modal() -> None:
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = Mock()
    event = Mock(action_id="run-library-rag")

    await ChatScreen.on_console_workbench_action_requested(screen, event)

    event.stop.assert_called_once_with()
    screen._retrieval.open_library_search.assert_called_once_with()
    screen._run_console_library_rag_from_visible_action.assert_not_called()


def test_inspector_search_library_action_always_opens_search_modal() -> None:
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = Mock()
    event = Mock()

    ChatScreen.handle_console_run_library_rag(screen, event)

    event.stop.assert_called_once_with()
    screen._retrieval.open_library_search.assert_called_once_with()
    screen._run_console_library_rag_from_visible_action.assert_not_called()
