"""Read-only maintenance refusal over installed in-memory editor owners."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas


def test_unknown_owner_refuses_without_invoking_dirty_method():
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors

    class Unknown:
        def is_dirty(self):
            raise AssertionError("unqualified owner executed")

    result = probe_unsaved_editors(editors=(Unknown(),))
    assert result[0].reason == "unknown-editor-state"


@pytest.mark.parametrize("draft", ["", "unsaved", "   "])
def test_actual_console_store_draft_is_preserved(draft):
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    runtime = ConsoleRuntime(SimpleNamespace())
    store = ConsoleChatStore()
    session = store.create_session()
    store.set_session_draft(session.id, draft)
    runtime.set_chat_store(store)
    before = vars(session).copy()
    result = probe_unsaved_editors(console_runtime=runtime)
    assert bool(result) == bool(draft)
    assert vars(session) == before


def test_uninitialized_console_runtime_remains_uninitialized():
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    runtime = ConsoleRuntime(SimpleNamespace())
    assert probe_unsaved_editors(console_runtime=runtime) == ()
    assert runtime.chat_store is None


def test_actual_console_attachment_is_preserved():
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Chat.attachment_core import PendingAttachment
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    runtime = ConsoleRuntime(SimpleNamespace())
    store = ConsoleChatStore()
    session = store.create_session()
    attachment = PendingAttachment(
        "/synthetic", "attachment", "image", "attachment", data=b"private"
    )
    session.pending_attachments.append(attachment)
    runtime.set_chat_store(store)
    assert (
        probe_unsaved_editors(console_runtime=runtime)[0].reason
        == "needs-user-save-discard"
    )
    assert session.pending_attachments == [attachment]


@pytest.mark.parametrize("text", ["", "unprocessed change"])
def test_console_visible_control_without_session_is_preserved(text):
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    composer = ConsoleComposerBar()
    composer.load_draft(text)
    result = probe_unsaved_editors(editors=(composer,))
    assert bool(result) == bool(text)
    assert composer.draft_text() == text


@pytest.mark.parametrize("changed", [False, True])
def test_notes_visible_controls_can_lead_clean_snapshot(changed):
    from textual.widgets import Input, TextArea

    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Library.library_notes_state import (
        DatabaseNoteDraft,
        LibraryNoteSessionSnapshot,
        NormalizedDatabaseNote,
    )
    from tldw_chatbook.Widgets.Library.library_notes_canvas import (
        LibraryNotePresentationState,
    )

    snapshot = LibraryNoteSessionSnapshot(
        NormalizedDatabaseNote("n", "title", "body", (), 1, "", ""),
        DatabaseNoteDraft("n", "title", "body", "", 0),
        1,
        0,
        False,
        False,
        False,
        0,
        "",
    )
    canvas = LibraryNotesCanvas(
        mode="editor", presentation_state=LibraryNotePresentationState(snapshot, "", "")
    )
    controls = {
        "#library-note-title": Input(),
        "#library-note-body": TextArea("changed" if changed else "body"),
        "#library-note-keywords": Input(""),
    }
    controls["#library-note-title"].set_reactive(Input.value, "title")
    # Real controls before delivery of their Changed message; no app/service.
    canvas.query_one = lambda selector, expected: controls[selector]
    result = probe_unsaved_editors(editors=(canvas,))
    assert bool(result) is changed
    assert canvas.presentation_state.snapshot is snapshot
    assert controls["#library-note-body"].text == ("changed" if changed else "body")


@pytest.mark.parametrize(
    "state,body,blocked",
    [("saved", "body", False), ("saved", "typed", True), ("dirty", "body", True)],
)
def test_file_notes_checks_live_body_before_changed_message(state, body, blocked):
    from textual.widgets import TextArea

    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Notes.file_notes_service import OpenedFileNote

    owner = object.__new__(LibraryFileNotesWorkspace)
    owner._save_state = state
    owner._opened = OpenedFileNote(
        "/synthetic",
        "note.md",
        "body",
        b"",
        "hash",
        "\n",
        False,
        4,
        0,
        True,
        None,
        False,
        b"body",
        4,
        False,
    )
    owner._editor_widget = TextArea(body)
    assert bool(probe_unsaved_editors(editors=(owner,))) is blocked
    assert owner._save_state == state
    assert owner._editor_widget.text == body


@pytest.mark.parametrize("location", ["transition", "pending", "inflight"])
def test_console_transition_and_send_stashes_are_not_reconciled(location):
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Widgets.Console.console_composer_bar import (
        ConsoleComposerBar,
        ConsoleDraftStash,
    )

    composer = ConsoleComposerBar()
    composer.load_draft("")
    view = object.__new__(ChatScreen)
    view.query = lambda kind: [composer]
    snapshot = ("session", "transition draft", 1) if location == "transition" else None
    stash = ConsoleDraftStash([], "send draft", False)
    view._session = SimpleNamespace(_console_draft_switch_snapshot=snapshot)
    view._console_pending_send_stash = stash if location == "pending" else None
    view._console_inflight_send_stashes = (
        {"session": stash} if location == "inflight" else {}
    )
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.view = view
    result = probe_unsaved_editors(console_runtime=runtime)
    assert result[0].reason == "needs-user-save-discard"
    assert view._session._console_draft_switch_snapshot is snapshot
    assert stash.text == "send draft"
    assert composer.draft_text() == ""


def test_unknown_store_does_not_execute_sessions():
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(
        SimpleNamespace(sessions=lambda: pytest.fail("unknown store called"))
    )
    assert (
        probe_unsaved_editors(console_runtime=runtime)[0].reason
        == "unknown-editor-state"
    )


def test_viewless_app_attachment_stash_survives_probe_without_store():
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Chat.attachment_core import PendingAttachment
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    attachment = PendingAttachment(
        "/synthetic",
        "clipboard",
        "image",
        "attachment",
        data=b"unsaved clipboard bytes",
    )
    stash = {"session": (attachment,)}
    app = SimpleNamespace(_console_pending_attachment_stash=stash)
    runtime = ConsoleRuntime(app)
    result = probe_unsaved_editors(console_runtime=runtime)
    assert result[0].reason == "needs-user-save-discard"
    assert runtime.chat_store is None and runtime.view is None
    assert app._console_pending_attachment_stash is stash
    assert stash["session"][0] is attachment
    assert attachment.data == b"unsaved clipboard bytes"


@pytest.mark.parametrize("stash", [None, [], {"session": []}, {"session": (object(),)}])
def test_malformed_app_attachment_stash_refuses(stash):
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    app = SimpleNamespace(_console_pending_attachment_stash=stash)
    assert (
        probe_unsaved_editors(console_runtime=ConsoleRuntime(app))[0].reason
        == "unknown-editor-state"
    )
    assert app._console_pending_attachment_stash is stash
