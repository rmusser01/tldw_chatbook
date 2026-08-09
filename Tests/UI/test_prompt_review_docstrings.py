"""Documentation contracts for public Prompt Library UI handlers."""

from __future__ import annotations

import inspect

import pytest

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.prompt_delete_confirmation_modal import (
    PromptDeleteConfirmationModal,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor


@pytest.mark.parametrize(
    "handler",
    (
        PromptDeleteConfirmationModal.on_button_pressed,
        LibraryScreen.handle_library_prompt_copy,
        LibraryScreen.handle_library_prompt_delete,
    ),
)
def test_prompt_library_button_handlers_document_event_argument(
    handler: object,
) -> None:
    docstring = inspect.getdoc(handler) or ""

    assert "Args:" in docstring
    assert "event:" in docstring


def test_prompt_block_editor_compose_documents_embedded_layout_contract() -> None:
    docstring = inspect.getdoc(PromptBlockEditor.compose) or ""

    assert "embedded" in docstring.lower()
    assert "scroll" in docstring.lower()
    assert "Yields:" in docstring
