"""Read-only refusal for unsaved installed editor state before maintenance."""

import sys
from dataclasses import dataclass

from textual.containers import Vertical
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


def _settings_drafts_dirty(drafts) -> bool:
    from tldw_chatbook.UI.Screens.settings_config_models import (
        SettingsCategoryId,
        SettingsDraft,
    )

    if type(drafts) is not dict:
        raise TypeError("unknown settings drafts")
    dirty = False
    for category, draft in drafts.items():
        if (
            type(category) is not SettingsCategoryId
            or type(draft) is not SettingsDraft
            or draft.category is not category
            or type(draft.originals) is not dict
            or type(draft.values) is not dict
            or any(type(key) is not str for key in (*draft.originals, *draft.values))
        ):
            raise TypeError("unknown settings draft")
        dirty = draft.is_dirty or dirty
    return dirty


def _speech_snapshot_dirty(snapshot) -> bool:
    if snapshot is None:
        return False
    if not _exact(
        snapshot,
        "Widgets.Settings_Widgets.speech_tts_settings_panel",
        "SpeechTTSPanelDraftSnapshot",
    ):
        raise TypeError("unknown speech settings snapshot")
    return (
        snapshot.state != snapshot.original_state
        or snapshot.realtime_draft != snapshot.realtime_original
    )


def _clean_settings_form(owner) -> bool:
    """Qualify only source-backed Overview/Storage forms, without UI sync."""
    if owner._category_pane_swap_pending is not False:
        raise TypeError("settings pane transition unresolved")
    if owner._speech_tts_draft_snapshot is None and (
        owner._speech_tts_draft_state is not None
        or owner._speech_tts_original_state is not None
    ):
        raise TypeError("unqualified cached speech draft")
    if owner.active_category == "overview":
        if type(owner.query_one("#settings-overview-card", Vertical)) is not Vertical:
            raise TypeError("unknown overview control")
        return True
    if owner.active_category == "storage":
        from tldw_chatbook.UI.Screens.settings_storage_defaults import (
            STORAGE_FIELD_LABELS,
        )

        values = type(owner)._storage_loaded_values(owner)
        if set(values) != set(STORAGE_FIELD_LABELS):
            raise TypeError("unknown storage form")
        for key, value in values.items():
            selector = type(owner)._storage_field_selector(owner, key)
            control = owner.query_one(selector, Input)
            if type(control) is not Input:
                raise TypeError("unknown storage control")
            if control.value != str(value):
                return False
        return True
    raise TypeError("unqualified settings form")


def _console_snapshot_dirty(state) -> bool:
    if type(state) is not dict or set(state) - {
        "interface_type",
        "native_console_state",
    }:
        raise TypeError("unknown retained Console state")
    if not state:
        return False
    if state.get("interface_type") != "native_console":
        raise TypeError("unknown retained Console interface")
    native = state.get("native_console_state")
    keys = {
        "version",
        "task_resume_state",
        "image_view_modes",
        "library_rag_source_types",
        "pending_console_launch",
        "console_evidence_sent_notice",
    }
    if type(native) is not dict or set(native) != keys or native["version"] != "1.0":
        raise TypeError("unknown retained Console snapshot")
    resume = native["task_resume_state"]
    text_keys = {"summary", "last_step", "diff_summary", "next_action"}
    pending_keys = {"pending_approval", "pending_skill_install", "pending_skill_script"}
    if type(resume) is not dict or set(resume) != text_keys | pending_keys:
        raise TypeError("unknown retained task state")
    if any(type(resume[key]) is not str for key in text_keys):
        raise TypeError("unknown retained task summary")
    for key in pending_keys:
        if resume[key] is not None:
            # These rounds are owned by live controller workers, not restorable
            # draft text; never silently classify their retained input as clean.
            raise TypeError("retained approval requires live owner settlement")
    modes = native["image_view_modes"]
    if type(modes) is not dict or any(
        type(key) is not str or value not in {"pixels", "graphics", "hidden"}
        for key, value in modes.items()
    ):
        raise TypeError("unknown retained image modes")
    sources = native["library_rag_source_types"]
    if type(sources) is not list or any(type(value) is not str for value in sources):
        raise TypeError("unknown retained retrieval scope")
    notice = native["console_evidence_sent_notice"]
    if notice is not None and type(notice) is not str:
        raise TypeError("unknown retained evidence notice")
    launch = native["pending_console_launch"]
    if launch is not None and type(launch) is not dict:
        raise TypeError("unknown retained Console launch")
    return launch is not None


def _retained_editors(store) -> list[UnsavedEditor]:
    if not _exact(store, "UI.Navigation.screen_state_store", "ScreenStateStore"):
        raise TypeError("unknown screen state owner")
    store._assert_owner_thread()
    if type(store._entries) is not dict:
        raise TypeError("unknown screen entries")
    issues = []
    for route, envelope in store._entries.items():
        if (
            type(route) is not str
            or not _exact(
                envelope, "UI.Navigation.screen_state_store", "_SnapshotEnvelope"
            )
            or envelope.canonical_route != route
            or not _exact(
                envelope.runtime_identity,
                "UI.Navigation.screen_state_store",
                "RuntimeIdentity",
            )
            or type(envelope.snapshot) is not dict
        ):
            raise TypeError("unknown retained screen shape")
        state = envelope.snapshot
        if route == "settings":
            if set(state) - {
                "active_category",
                "category_search_query",
                "settings_drafts",
                "speech_tts_panel_draft",
            }:
                raise TypeError("unknown retained Settings field")
            if any(
                type(state[key]) is not str
                for key in ("active_category", "category_search_query")
                if key in state
            ):
                raise TypeError("unknown retained Settings navigation")
            dirty = _settings_drafts_dirty(state.get("settings_drafts", {}))
            dirty = _speech_snapshot_dirty(state.get("speech_tts_panel_draft")) or dirty
        elif route == "chat":
            dirty = _console_snapshot_dirty(state)
        elif route == "library":
            _library_snapshot(state)
            dirty = False
        elif route == "personas":
            dirty = _personas_snapshot_dirty(state)
        elif route == "stts":
            if set(state) - {"speech_playground_axes"}:
                raise TypeError("unknown retained Speech fields")
            axes = state.get("speech_playground_axes", {})
            loaded = sys.modules.get("tldw_chatbook.UI.Screens.stts_screen")
            if (
                loaded is None
                or type(axes) is not dict
                or loaded._bounded_playground_axes(axes) != axes
            ):
                raise TypeError("unknown retained Speech axes")
            dirty = False
        elif route == "home" and not state:
            dirty = False
        else:
            # A route without its installed snapshot contract cannot be inferred
            # clean from its name or from superficially passive fields.
            raise TypeError("unqualified retained authoring state")
        if dirty:
            issues.append(
                UnsavedEditor(
                    "Retained "
                    + {
                        "settings": "Settings",
                        "chat": "Console",
                        "personas": "Personas",
                    }[route],
                    "needs-user-save-discard",
                )
            )
    return issues


def _library_snapshot(state) -> None:
    """Library.save_state carries applied browse state, never note editor text."""
    from tldw_chatbook.Library.library_media_state import MediaBrowseScope
    from tldw_chatbook.Library.library_prompts_state import PromptBrowseScope

    strings = {
        "library_selected_row_id",
        "selected_conversation_id",
        "selected_note_id",
        "library_notes_view",
        "selected_media_id",
        "library_media_view",
        "library_rag_query",
        "library_rag_mode",
        "library_rag_selected_result_id",
        "library_rag_retrieval_status",
        "library_rag_searched_query",
        "library_rag_answer_query",
        "library_rag_answer_mode",
        "library_notes_sort",
        "library_notes_filter",
        "library_conversation_query",
        "library_export_last_path",
    }
    special = {
        "library_rag_scope_deselected",
        "library_rag_results",
        "library_rag_recovery_state",
        "library_rag_diagnostics",
        "library_rag_answer",
        "library_media_scope",
        "library_prompts_scope",
        "selected_prompt_id",
        "library_conversation_page",
        "library_export_last_at",
    }
    if set(state) - strings - special or any(
        type(state[key]) is not str for key in strings & state.keys()
    ):
        raise TypeError("unknown retained Library fields")
    for key, value in state.items():
        if key == "library_rag_scope_deselected":
            if type(value) is not set or any(type(item) is not str for item in value):
                raise TypeError("unknown retained retrieval selection")
        elif key == "library_rag_results":
            if type(value) is not tuple or any(
                not _exact(item, "Library.library_rag_state", "LibraryRagResultRow")
                for item in value
            ):
                raise TypeError("unknown retained retrieval results")
        elif key == "library_rag_diagnostics":
            if type(value) is not dict or any(type(item) is not str for item in value):
                raise TypeError("unknown retained diagnostics")
        elif key in {"library_rag_recovery_state", "library_rag_answer"}:
            module, name = (
                ("UI.destination_recovery", "DestinationRecoveryState")
                if key.endswith("recovery_state")
                else ("Library.library_rag_answer_service", "LibraryRagAnswer")
            )
            if value is not None and not _exact(value, module, name):
                raise TypeError("unknown retained retrieval outcome")
        elif key in {"library_media_scope", "library_prompts_scope"}:
            cls = (
                MediaBrowseScope if key == "library_media_scope" else PromptBrowseScope
            )
            if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
                raise TypeError("unknown retained browse scope")
            try:
                cls(**value)
            except ValueError:
                raise TypeError("unknown retained browse values") from None
        elif (
            key == "selected_prompt_id" and value is not None and type(value) is not int
        ):
            raise TypeError("unknown retained prompt selection")
        elif key == "library_conversation_page" and (
            type(value) is not int or value < 1
        ):
            raise TypeError("unknown retained conversation page")
        elif (
            key == "library_export_last_at"
            and value is not None
            and type(value) not in {int, float}
        ):
            raise TypeError("unknown retained export receipt")


def _personas_snapshot_dirty(state) -> bool:
    from dataclasses import asdict

    from tldw_chatbook.Widgets.Persona_Widgets.personas_state import (
        PersonasWorkbenchState,
    )

    if set(state) - {"personas_workbench", "personas_preview"}:
        raise TypeError("unknown retained Personas fields")
    workbench = state.get("personas_workbench", asdict(PersonasWorkbenchState()))
    defaults = asdict(PersonasWorkbenchState())
    if type(workbench) is not dict or set(workbench) != set(defaults):
        raise TypeError("unknown retained Personas workbench")
    for key, value in workbench.items():
        expected = defaults[key]
        if (expected is None and value is not None and type(value) is not str) or (
            expected is not None and type(value) is not type(expected)
        ):
            raise TypeError("unknown retained Personas value")
    preview = state.get("personas_preview")
    if preview is not None:
        if type(preview) is not dict or set(preview) != {
            "greeting",
            "history",
            "seeded_for",
            "greeting_index",
        }:
            raise TypeError("unknown retained Personas preview")
        if (
            type(preview["greeting"]) is not str
            or type(preview["greeting_index"]) is not int
            or (
                preview["seeded_for"] is not None
                and type(preview["seeded_for"]) is not str
            )
        ):
            raise TypeError("unknown retained Personas preview values")
        history = preview["history"]
        if type(history) is not list or any(
            type(row) is not dict
            or set(row) != {"role", "content"}
            or row["role"] not in {"user", "assistant"}
            or type(row["content"]) is not str
            for row in history
        ):
            raise TypeError("unknown retained Personas preview history")
    return workbench["has_unsaved_changes"]


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
        dirty = _settings_drafts_dirty(owner._settings_drafts)
        dirty = _speech_snapshot_dirty(owner._speech_tts_draft_snapshot) or dirty
        if not dirty:
            dirty = not _clean_settings_form(owner)
    elif _exact(owner, "UI.Screens.personas_screen", "PersonasScreen"):
        label = "Personas"
        dirty = (
            owner.state.has_unsaved_changes
            or owner._persona_visual_has_unsaved_authoring()
            or owner._visual_identity_has_unsaved_authoring()
        )
        if not dirty:
            return [UnsavedEditor(label, "unknown-editor-state")]
    elif _exact(owner, "UI.Screens.stts_screen", "STTSScreen"):
        if not _exact(owner.stts_window, "UI.STTS_Window", "STTSWindow"):
            raise TypeError("Speech body unresolved")
        return _editor(owner.stts_window)
    elif _exact(owner, "UI.STTS_Window", "STTSWindow"):
        if owner._view_mount_lock.locked() or owner._mounted_view != owner.current_view:
            raise TypeError("Speech pane transition unresolved")
        if owner._pending_adopted_preset is not None:
            return [UnsavedEditor("Studio preferences", "needs-user-save-discard")]
        panes = {
            "settings": ("UI.Speech.speech_settings_pane", "SpeechSettingsPane"),
            "profiles": ("UI.stts_profile_library", "STTSProfileLibrary"),
            "playground": ("UI.Speech.speech_playground_pane", "SpeechPlaygroundPane"),
        }
        if owner.current_view not in panes:
            raise TypeError("unqualified Speech tool")
        module, name = panes[owner.current_view]
        loaded = sys.modules.get("tldw_chatbook." + module)
        cls = vars(loaded).get(name) if loaded is not None else None
        if cls is None:
            raise TypeError("Speech pane unavailable")
        pane = owner.query_one(cls)
        if not _exact(pane, module, name):
            raise TypeError("unknown Speech pane")
        return _editor(pane)
    elif _exact(owner, "UI.Speech.speech_settings_pane", "SpeechSettingsPane"):
        label, dirty = "Studio preferences", owner.is_dirty
    elif _exact(owner, "UI.stts_profile_library", "STTSProfileLibrary"):
        state = type(owner).profile_maintenance_state(owner)
        if state == "needs-user-save/discard":
            return [UnsavedEditor("Speech profiles", "needs-user-save-discard")]
        if state != "ready":
            raise TypeError("Speech profile work unresolved")
    elif _exact(owner, "UI.stts_profile_library", "TTSProfileEditorModal"):
        label = "Speech profile editor"
        # A retry can open with a draft from an unsuccessful save. Its initial
        # form values are not a persisted baseline even before further typing.
        dirty = owner.initial_draft is not None
        for name, initial in (
            ("name", owner.initial_name),
            ("model", owner.initial_model_id),
            ("voice", owner.initial_voice_id or ""),
        ):
            control = owner.query_one("#stts-profile-editor-" + name, Input)
            if type(control) is not Input or type(initial) is not str:
                raise TypeError("unknown Speech profile field")
            dirty = control.value != initial or dirty
    elif _exact(owner, "UI.Speech.speech_playground_pane", "SpeechPlaygroundPane"):
        from tldw_chatbook.UI.Speech.speech_playback_mixin import EXAMPLE_TEXTS

        label = "Speech Playground"
        control = owner.query_one("#tts-text-input", TextArea)
        if type(control) is not TextArea:
            raise TypeError("unknown Speech text editor")
        dirty = control.text not in ("", *EXAMPLE_TEXTS)
        # No clone/reference draft has a persisted baseline here. Preserve it
        # for explicit user action even if the synthesis text is the example.
        dirty = (
            any(
                value is not None
                for value in (
                    owner.reference_audio_path,
                    owner.higgs_reference_audio_path,
                    owner._clone_setup_source_path,
                    owner._clone_setup_canonical,
                )
            )
            or dirty
        )
        if (
            owner._active_profile_name_modal is not None
            or owner._clone_setup_validation_task is not None
            or owner._clone_setup_retained_tasks
        ):
            raise TypeError("Speech draft operation unresolved")
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
    *,
    console_runtime: object | None = None,
    editors: tuple[object, ...] = (),
    screen_state_store: object | None = None,
) -> tuple[UnsavedEditor, ...]:
    """Inspect actual runtime plus explicitly enumerated mounted editor owners.

    The app coordinator supplies authoring screens AND their mounted controls;
    this function does not discover/open screens or claim enumeration coverage.
    Unknown/custom owner types and unavailable controls refuse conservatively.
    No user text, source path, or attachment bytes are returned.
    """
    issues = []
    if screen_state_store is not None:
        try:
            issues.extend(_retained_editors(screen_state_store))
        except (AttributeError, TypeError, KeyError, RuntimeError):
            issues.append(UnsavedEditor("Retained screens", "unknown-editor-state"))
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
