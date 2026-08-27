"""Tests for the manual-only Console Library search surface."""

from unittest.mock import Mock

import pytest

from tldw_chatbook.Widgets.Console.console_library_search_modal import (
    ConsoleLibrarySearchModal,
)


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
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = Mock()
    composer = Mock()
    composer.draft_text.return_value = draft
    screen._console_composer_or_none.return_value = composer

    ChatScreen._open_console_library_search(screen)

    modal = screen.app.push_screen.call_args.args[0]
    assert isinstance(modal, ConsoleLibrarySearchModal)
    assert modal._query == draft


def test_search_modal_copy_says_this_search_only_and_names_policy_separation() -> None:
    modal = ConsoleLibrarySearchModal(query="find this")

    copy = modal._status_copy()
    assert "this send only" in copy
    assert "Automatic retrieval" in copy
    assert "assistant Library access" in copy
    assert modal._scope_summary().startswith("This search only · Sources:")
