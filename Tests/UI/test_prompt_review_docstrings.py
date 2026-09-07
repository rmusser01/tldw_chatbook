"""Documentation contracts for public Prompt Library UI handlers.

**Owner retarget, 2026-09-05 (wave-6 task 2, the prompts controller move).**
``handle_library_prompt_copy``/``handle_library_prompt_delete`` moved, bodies
and docstrings byte-for-byte, from ``LibraryScreen`` to
``LibraryPromptsController``; ``LibraryScreen`` keeps a one-line delegator
under each name, and a delegator carries no docstring, so
``inspect.getdoc(LibraryScreen.<name>)`` returns ``None`` from that commit
onward. This is the recipe's own "hardcoded census ships red" shape
(``backlog/docs/library-decomposition-recipe.md`` §3): unlike every other
bypass shape it is loudly RED at the exact commit boundary that moves the
code rather than silently green, so the no-red-ships rule requires
retargeting it in the SAME PR-stage as the move instead of deferring to the
subsystem's cleanup PR. The parametrize rows below therefore name the new
owner; **both assertions are byte-for-byte unchanged**, and the contract they
express -- a public Prompt Library button handler documents its ``event``
argument -- is unchanged too, now checked where the handler actually lives.
"""

from __future__ import annotations

import inspect

import pytest

from tldw_chatbook.UI.Library_Modules.library_prompts_controller import (
    LibraryPromptsController,
)
from tldw_chatbook.Widgets.Library.prompt_delete_confirmation_modal import (
    PromptDeleteConfirmationModal,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor


@pytest.mark.parametrize(
    "handler",
    (
        PromptDeleteConfirmationModal.on_button_pressed,
        LibraryPromptsController.handle_library_prompt_copy,
        LibraryPromptsController.handle_library_prompt_delete,
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
