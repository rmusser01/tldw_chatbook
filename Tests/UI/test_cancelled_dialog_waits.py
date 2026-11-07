"""A cancelled worker never strands the dialog it was waiting on (TASK-39146).

``App.push_screen_wait`` awaits the dialog's result future through
``asyncio.shield``. A bare ``await app.push_screen(..., wait_for_dismiss=True)``
does not, so cancelling the waiting worker (shutdown, an exclusive restart)
cancels that future, and the dialog's later close raises ``InvalidStateError``
inside ``Screen.dismiss`` before it pops, leaving the dialog on the stack.

Each case drives one production wait site in a worker, cancels the worker
while the dialog is open, then closes the dialog. The dialogs are replaced by
a plain modal: the subject is the wait, not the dialog.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.screen import ModalScreen
from textual.widgets import Static
from textual.worker import WorkerCancelled, WorkerState

pytestmark = pytest.mark.bootstrap_profile


class _Dialog(ModalScreen):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

    def compose(self):
        yield Static("dialog")


class _Host(App):
    pass


class _Notes:
    async def list_notes(self, **_kwargs):
        return [{"id": 1, "title": "Note", "content": "Text"}]


class _Conversations:
    async def list_conversations(self, **_kwargs):
        return {"items": [{"id": 1, "title": "Chat"}]}


def _export_delete(app):
    from tldw_chatbook.UI.ChatbookExportManagementWindow import (
        ChatbookExportManagementWindow,
    )

    owner = SimpleNamespace(
        app_instance=app,
        chatbook_files=[{"name": "book", "path": Path("unused.zip")}],
        selected_chatbook=0,
    )
    return ChatbookExportManagementWindow._delete_selected(owner)


def _stts_notes(app):
    from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget

    app.notes_scope_service = _Notes()
    worker = inspect.unwrap(AudioBookGenerationWidget._import_from_notes_worker)
    return worker(SimpleNamespace(app=app))


def _stts_conversation(app):
    from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget

    app.chat_conversation_scope_service = _Conversations()
    worker = inspect.unwrap(AudioBookGenerationWidget._import_from_conversation_worker)
    return worker(SimpleNamespace(app=app))


def _chatbooks(method):
    def start(app):
        from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved

        return getattr(ChatbooksWindowImproved, method)(
            SimpleNamespace(app_instance=app)
        )

    return start


def _character_import(app):
    from tldw_chatbook.UI.CCP_Modules.ccp_character_handler import CCPCharacterHandler

    return CCPCharacterHandler.handle_import(
        SimpleNamespace(window=SimpleNamespace(app=app))
    )


def _media_analysis_delete(app):
    from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel

    worker = inspect.unwrap(MediaViewerPanel._run_delete_confirmation)
    return worker(
        SimpleNamespace(
            app=app,
            media_data={"id": 1},
            all_analyses=[{"version_number": 1, "uuid": "analysis"}],
            current_analysis_index=0,
        )
    )


_WIZARD = "tldw_chatbook.UI.Wizards.ChatbookCreationWizard.ChatbookCreationWizard"
_TEMPLATES = "tldw_chatbook.UI.ChatbookTemplatesWindow.ChatbookTemplatesWindow"

#: id -> (dialog classes the site imports, start(app) -> the site's coroutine,
#: dismiss results that walk it to the wait under test)
SITES = {
    "export-delete": (
        ["tldw_chatbook.Widgets.confirmation_dialog.ConfirmationDialog"],
        _export_delete,
        [],
    ),
    "stts-notes": (
        [
            "tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog.NoteSelectionDialog"
        ],
        _stts_notes,
        [],
    ),
    "stts-conversation": (
        [
            "tldw_chatbook.Widgets.conversation_selection_dialog.ConversationSelectionDialog"
        ],
        _stts_conversation,
        [],
    ),
    "chatbooks-create": ([_WIZARD], _chatbooks("action_create_chatbook"), []),
    "chatbooks-templates": (
        [_TEMPLATES, _WIZARD],
        _chatbooks("action_browse_templates"),
        [],
    ),
    "chatbooks-template-wizard": (
        [_TEMPLATES, _WIZARD],
        _chatbooks("action_browse_templates"),
        [{"name": "template"}],
    ),
    "chatbooks-manage-exports": (
        [
            "tldw_chatbook.UI.ChatbookExportManagementWindow.ChatbookExportManagementWindow"
        ],
        _chatbooks("action_manage_exports"),
        [],
    ),
    "character-import": (
        ["tldw_chatbook.Widgets.enhanced_file_picker.EnhancedFileOpen"],
        _character_import,
        [],
    ),
    "media-analysis-delete": (
        ["tldw_chatbook.Widgets.delete_confirmation_dialog.DeleteConfirmationDialog"],
        _media_analysis_delete,
        [],
    ),
}


async def _next_dialog(pilot, app, previous=None):
    async with asyncio.timeout(10):
        while not isinstance(app.screen, _Dialog) or app.screen is previous:
            await pilot.pause()
    return app.screen


@pytest.mark.asyncio
@pytest.mark.parametrize("site", sorted(SITES))
async def test_cancelling_the_waiting_worker_leaves_the_dialog_closable(
    monkeypatch, site
):
    dialogs, start, answers = SITES[site]
    for dotted in dialogs:
        monkeypatch.setattr(dotted, _Dialog)
    app = _Host()
    async with app.run_test() as pilot:
        worker = app.run_worker(start(app), group="dialog-wait-probe")
        dialog = await _next_dialog(pilot, app)
        for answer in answers:
            await dialog.dismiss(answer)
            dialog = await _next_dialog(pilot, app, previous=dialog)
        worker.cancel()
        with pytest.raises(WorkerCancelled):
            await asyncio.wait_for(worker.wait(), 10)
        assert worker.state == WorkerState.CANCELLED
        # Before TASK-39146 this raised InvalidStateError: the cancel had
        # cancelled the result future the close resolves.
        await dialog.dismiss(None)
        await pilot.pause()
        assert dialog not in app.screen_stack
