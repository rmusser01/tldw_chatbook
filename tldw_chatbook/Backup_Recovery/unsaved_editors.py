"""Read-only refusal for unsaved installed editor state before maintenance."""

import sys
from dataclasses import dataclass

from textual.css.query import QueryError
from textual.widgets import Input, TextArea


@dataclass(frozen=True)
class UnsavedEditor:
    """Content-free label and refusal suitable for existing save/discard UI."""

    label: str
    reason: str


def _exact(owner: object, module: str, name: str) -> bool:
    # Inspect only already loaded installed classes. Importing a screen here can
    # initialize config/services, which is not part of a read-only editor probe.
    loaded = sys.modules.get("tldw_chatbook." + module)
    return loaded is not None and type(owner) is vars(loaded).get(name)


def _text(value: object) -> bool:
    if type(value) is not str:
        raise TypeError("unknown draft shape")
    return len(value) != 0


def _console(runtime: object) -> list[UnsavedEditor]:
    if not _exact(runtime, "Chat.console_runtime", "ConsoleRuntime"):
        return [UnsavedEditor("Console", "unknown-editor-state")]
    issues = []
    # ChatScreen's navigation stash can be the sole owner of clipboard bytes
    # while the runtime has neither a view nor a restored store.
    stash = getattr(runtime.app, "_console_pending_attachment_stash", {})
    if type(stash) is not dict:
        raise TypeError("unknown app attachment stash")
    for session_id, attachments in stash.items():
        if type(session_id) is not str or type(attachments) is not tuple:
            raise TypeError("unknown app attachment stash")
        if any(
            not _exact(item, "Chat.attachment_core", "PendingAttachment")
            for item in attachments
        ):
            raise TypeError("unknown stashed attachment")
        if attachments:
            issues.append(
                UnsavedEditor("Console attachment stash", "needs-user-save-discard")
            )
    store = runtime.chat_store
    if store is not None:
        if not _exact(store, "Chat.console_chat_store", "ConsoleChatStore"):
            return [UnsavedEditor("Console", "unknown-editor-state")]
        for session in store.sessions():
            if not _exact(session, "Chat.console_chat_store", "ConsoleChatSession"):
                raise TypeError("unknown session shape")
            if type(session.pending_attachments) is not list:
                raise TypeError("unknown attachments shape")
            if (
                _text(session.draft)
                or len(session.pending_attachments)
                or (
                    session.one_shot_prefill is not None
                    and _text(session.one_shot_prefill)
                )
            ):
                issues.append(
                    UnsavedEditor(
                        "Console drafts and attachments", "needs-user-save-discard"
                    )
                )
    if runtime.view is not None:
        issues.extend(_editor(runtime.view))
    return issues


def _editor(owner: object) -> list[UnsavedEditor]:
    dirty = False
    label = "Editor"
    if _exact(owner, "UI.Screens.chat_screen", "ChatScreen"):
        label = "Console composer"
        # Read the controls directly: the normal composer accessor memoizes,
        # while draft reconciliation writes back to sessions and clears stashes.
        from tldw_chatbook.Widgets.Console.console_composer_bar import (
            ConsoleComposerBar,
        )

        composers = list(owner.query(ConsoleComposerBar))
        if len(composers) != 1 or type(composers[0]) is not ConsoleComposerBar:
            raise TypeError("unknown mounted composer")
        dirty = _text(composers[0].draft_text())
        snapshot = owner._session._console_draft_switch_snapshot
        if snapshot is not None:
            if type(snapshot) is not tuple or len(snapshot) != 3:
                raise TypeError("unknown draft transition")
            dirty = _text(snapshot[1]) or dirty
        stashes = owner._console_inflight_send_stashes
        if type(stashes) is not dict:
            raise TypeError("unknown inflight draft stashes")
        for stash in (owner._console_pending_send_stash, *stashes.values()):
            if stash is None:
                continue
            if not _exact(
                stash, "Widgets.Console.console_composer_bar", "ConsoleDraftStash"
            ):
                raise TypeError("unknown draft stash")
            dirty = _text(stash.text) or bool(stash.segments) or dirty
    elif _exact(owner, "Widgets.Console.console_composer_bar", "ConsoleComposerBar"):
        label, dirty = "Console composer", _text(owner.draft_text())
    elif _exact(owner, "UI.Screens.library_screen", "LibraryScreen"):
        label = "Database Notes"
        snapshot = owner._library_note_session.snapshot
        if snapshot is not None:
            dirty = snapshot.dirty or snapshot.saving or snapshot.in_conflict
        # Library also owns prompt/recipe authoring. That editor has no installed
        # read-only dirty baseline here, so an active authoring canvas refuses.
        if owner._library_prompts_mutation_in_flight:
            return [UnsavedEditor("Library prompts", "unknown-editor-state")]
    elif _exact(owner, "Widgets.Library.library_notes_canvas", "LibraryNotesCanvas"):
        label = "Database Notes"
        if owner.mode == "editor":
            snapshot = owner.presentation_state.snapshot
            title = owner.query_one("#library-note-title", Input).value
            expected_title = "" if owner.title_placeholder_only else snapshot.title
            dirty = (
                snapshot.dirty
                or snapshot.saving
                or snapshot.in_conflict
                or title != expected_title
                or owner.query_one("#library-note-body", TextArea).text != snapshot.body
                or owner.query_one("#library-note-keywords", Input).value
                != snapshot.keywords_text
            )
        elif owner.mode not in {"list", "loading", "create", "sync"}:
            raise TypeError("unknown notes mode")
    elif _exact(
        owner,
        "Widgets.Library.library_file_notes_workspace",
        "LibraryFileNotesWorkspace",
    ):
        label = "File Notes"
        if owner.save_state not in {
            "idle",
            "dirty",
            "saving",
            "saved",
            "conflict",
            "error",
        }:
            raise TypeError("unknown file notes state")
        dirty = owner.save_state in {"dirty", "saving", "conflict", "error"}
        if owner._opened is not None:
            dirty = owner._editor_widget.text != owner._opened.body or dirty
        else:
            dirty = _text(owner._editor_widget.text) or dirty
    elif _exact(owner, "UI.Evals.bench_editor", "BenchEditor") or _exact(
        owner, "UI.Evals.character_bench_editor", "CharacterBenchEditor"
    ):
        label, dirty = "Evaluation bench", owner.is_dirty()
    elif _exact(owner, "UI.Screens.settings_screen", "SettingsScreen"):
        label = "Settings"
        dirty = any(
            owner._category_has_unsaved_changes(category)
            for category in owner._settings_drafts
        )
        if not dirty:
            return [UnsavedEditor(label, "unknown-editor-state")]
    elif _exact(owner, "UI.Screens.personas_screen", "PersonasScreen"):
        label = "Personas"
        dirty = (
            owner.state.has_unsaved_changes
            or owner._persona_visual_has_unsaved_authoring()
            or owner._visual_identity_has_unsaved_authoring()
        )
        if not dirty:
            return [UnsavedEditor(label, "unknown-editor-state")]
    else:
        for module, name, known_label in (
            ("UI.Screens.stts_screen", "STTSScreen", "Speech Studio"),
            ("UI.STTS_Window", "STTSWindow", "Speech Studio"),
            (
                "UI.stts_profile_library",
                "TTSProfileEditorModal",
                "Speech profile editor",
            ),
            (
                "Widgets.Library.library_prompts_canvas",
                "LibraryPromptsListCanvas",
                "Library prompts",
            ),
        ):
            if _exact(owner, module, name):
                label = known_label
                break
        return [UnsavedEditor(label, "unknown-editor-state")]
    if type(dirty) is not bool:
        raise TypeError("unknown dirty state")
    return [UnsavedEditor(label, "needs-user-save-discard")] if dirty else []


def probe_unsaved_editors(
    *, console_runtime: object | None = None, editors: tuple[object, ...] = ()
) -> tuple[UnsavedEditor, ...]:
    """Inspect actual runtime plus explicitly enumerated mounted editor owners.

    The app coordinator supplies authoring screens AND their mounted controls;
    this function does not discover/open screens or claim enumeration coverage.
    Unknown/custom owner types and unavailable controls refuse conservatively.
    No user text, source path, or attachment bytes are returned.
    """
    issues = []
    if console_runtime is not None:
        try:
            issues.extend(_console(console_runtime))
        except (AttributeError, TypeError, KeyError, QueryError):
            issues.append(UnsavedEditor("Console", "unknown-editor-state"))
    for owner in editors:
        try:
            issues.extend(_editor(owner))
        except (AttributeError, TypeError, KeyError, QueryError):
            issues.append(UnsavedEditor("Editor", "unknown-editor-state"))
    return tuple(dict.fromkeys(issues))
