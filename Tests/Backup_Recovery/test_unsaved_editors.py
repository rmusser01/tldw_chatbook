"""Read-only maintenance refusal over installed in-memory editor owners."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen
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


def test_retained_settings_draft_is_not_restored_or_discarded():
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.UI.Navigation.screen_state_store import (
        RuntimeIdentity,
        ScreenStateStore,
    )
    from tldw_chatbook.UI.Screens.settings_config_models import (
        SettingsCategoryId,
        SettingsDraft,
    )

    draft = SettingsDraft(
        SettingsCategoryId.STORAGE, {"media_db_path": "old"}, {"media_db_path": "new"}
    )
    store = ScreenStateStore()
    store.save(
        "settings",
        {
            "active_category": "storage",
            "settings_drafts": {SettingsCategoryId.STORAGE: draft},
        },
        RuntimeIdentity("server", "other"),
    )
    envelope = store._entries["settings"]
    result = probe_unsaved_editors(screen_state_store=store)
    assert result[0].reason == "needs-user-save-discard"
    assert store._entries["settings"] is envelope
    assert envelope.snapshot["settings_drafts"][SettingsCategoryId.STORAGE] is draft
    assert draft.values == {"media_db_path": "new"}


def test_clean_settings_overview_passes_without_constructing_forms():
    from textual.containers import Vertical
    from textual.screen import Screen

    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors

    screen = object.__new__(SettingsScreen)
    Screen.__init__(screen)
    screen._settings_drafts = {}
    screen._speech_tts_draft_snapshot = None
    screen._speech_tts_draft_state = None
    screen._speech_tts_original_state = None
    screen._category_pane_swap_pending = False
    screen.set_reactive(SettingsScreen.active_category, "overview")
    screen.query_one = lambda *args: Vertical(id="settings-overview-card")
    assert probe_unsaved_editors(editors=(screen,)) == ()


@pytest.mark.parametrize("changed", [False, True])
def test_settings_storage_reads_live_inputs_before_draft_event(changed):
    from textual.screen import Screen
    from textual.widgets import Input

    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors

    screen = object.__new__(SettingsScreen)
    Screen.__init__(screen)
    screen.app_instance = SimpleNamespace(app_config={"database": {}})
    screen._settings_drafts = {}
    screen._speech_tts_draft_snapshot = None
    screen._speech_tts_draft_state = None
    screen._speech_tts_original_state = None
    screen._category_pane_swap_pending = False
    screen.set_reactive(SettingsScreen.active_category, "storage")
    values = SettingsScreen._storage_loaded_values(screen)
    controls = {}
    for key, value in values.items():
        control = Input()
        control.set_reactive(
            Input.value, "unsaved" if changed and key == "media_db_path" else str(value)
        )
        controls[SettingsScreen._storage_field_selector(screen, key)] = control
    screen.query_one = lambda selector, expected: controls[selector]
    result = probe_unsaved_editors(editors=(screen,))
    assert bool(result) is changed
    assert screen._settings_drafts == {}


def _retained_console_snapshot():
    from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState

    return {
        "interface_type": "native_console",
        "native_console_state": {
            "version": "1.0",
            "task_resume_state": TaskResumeState().to_dict(),
            "image_view_modes": {},
            "library_rag_source_types": ["media"],
            "pending_console_launch": None,
            "console_evidence_sent_notice": None,
        },
    }


@pytest.mark.parametrize(
    "pending",
    [
        None,
        "pending_console_launch",
        "pending_approval",
        "pending_skill_install",
        "pending_skill_script",
        "unknown_nested",
    ],
)
def test_retained_console_nested_authoring_is_not_assumed_passive(pending):
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.UI.Navigation.screen_state_store import (
        RuntimeIdentity,
        ScreenStateStore,
    )

    snapshot = _retained_console_snapshot()
    native = snapshot["native_console_state"]
    if pending == "pending_console_launch":
        native[pending] = {"evidence": "not submitted"}
    elif pending is not None:
        native["task_resume_state"][pending] = {"draft": "pending"}
    store = ScreenStateStore()
    store.save("chat", snapshot, RuntimeIdentity("local"))
    envelope = store._entries["chat"]
    result = probe_unsaved_editors(screen_state_store=store)
    assert bool(result) is (pending is not None)
    assert store._entries["chat"] is envelope
    assert envelope.snapshot["native_console_state"] is native


@pytest.mark.parametrize(
    "route,snapshot",
    [
        ("unknown", {}),
        ("settings", {"new_editor_draft": "text"}),
        ("chat", {"native_console_state": {"new_editor_draft": "text"}}),
    ],
)
def test_retained_unknown_route_or_shape_refuses_without_discard(route, snapshot):
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.UI.Navigation.screen_state_store import (
        RuntimeIdentity,
        ScreenStateStore,
    )

    store = ScreenStateStore()
    store.save(route, snapshot, RuntimeIdentity("local"))
    envelope = store._entries[route]
    assert (
        probe_unsaved_editors(screen_state_store=store)[0].reason
        == "unknown-editor-state"
    )
    assert store._entries[route] is envelope


def test_empty_retained_settings_and_home_do_not_construct_owners():
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.UI.Navigation.screen_state_store import (
        RuntimeIdentity,
        ScreenStateStore,
    )

    store = ScreenStateStore()
    store.save(
        "settings",
        {"active_category": "overview", "settings_drafts": {}},
        RuntimeIdentity("local"),
    )
    store.save("home", {}, RuntimeIdentity("local"))
    assert probe_unsaved_editors(screen_state_store=store) == ()


@pytest.mark.parametrize("changed", [False, True])
def test_retained_speech_realtime_draft_is_compared_without_panel_sync(changed):
    from dataclasses import replace

    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.UI.Navigation.screen_state_store import (
        RuntimeIdentity,
        ScreenStateStore,
    )
    from tldw_chatbook.UI.Screens.settings_speech_tts import (
        load_global_speech_tts_state,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSPanelDraftSnapshot,
        _RealtimeSettingsDraft,
    )

    realtime = _RealtimeSettingsDraft(
        False, "openai", "gpt-realtime", "", "30", "auto", "semantic_vad", "0.5", "500"
    )
    snapshot = SpeechTTSPanelDraftSnapshot(
        state=load_global_speech_tts_state({}),
        original_state=load_global_speech_tts_state({}),
        realtime_draft=replace(realtime, enabled=changed),
        realtime_original=realtime,
        configure_provider="audio_cpp",
        draft_revision=1,
    )
    store = ScreenStateStore()
    store.save(
        "settings", {"speech_tts_panel_draft": snapshot}, RuntimeIdentity("local")
    )
    assert bool(probe_unsaved_editors(screen_state_store=store)) is changed
    assert store._entries["settings"].snapshot["speech_tts_panel_draft"] is snapshot


@pytest.mark.parametrize("route", ["library", "personas", "stts"])
@pytest.mark.parametrize("changed", [False, True])
def test_retained_navigation_snapshots_are_clean_without_losing_drafts(route, changed):
    from dataclasses import asdict

    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.Library.library_media_state import MediaBrowseScope
    from tldw_chatbook.Library.library_prompts_state import PromptBrowseScope
    from tldw_chatbook.UI.Navigation.screen_state_store import (
        RuntimeIdentity,
        ScreenStateStore,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.personas_state import (
        PersonasWorkbenchState,
    )

    snapshots = {
        "library": {
            "selected_note_id": "saved-note",
            "library_notes_view": "editor",
            "library_media_scope": asdict(MediaBrowseScope()),
            "library_prompts_scope": asdict(PromptBrowseScope()),
            "library_rag_results": (),
            "library_rag_diagnostics": {},
        },
        "personas": {
            "personas_workbench": asdict(PersonasWorkbenchState()),
            "personas_preview": {
                "greeting": "Hello",
                "history": [{"role": "user", "content": "Test"}],
                "seeded_for": "saved",
                "greeting_index": 0,
            },
        },
        "stts": {"speech_playground_axes": {"tts-provider-select": "audio_cpp"}},
    }
    snapshot = snapshots[route]
    if changed:
        if route == "personas":
            snapshot["personas_workbench"]["has_unsaved_changes"] = True
        else:
            snapshot["unrecognized_draft"] = "unsaved"
    store = ScreenStateStore()
    store.save(route, snapshot, RuntimeIdentity("local"))
    envelope = store._entries[route]
    assert bool(probe_unsaved_editors(screen_state_store=store)) is changed
    assert store._entries[route] is envelope


@pytest.mark.parametrize("changed", [False, True])
def test_speech_profile_modal_reads_actual_input_baselines(changed):
    from textual.screen import Screen
    from textual.widgets import Input

    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.UI.stts_profile_library import TTSProfileEditorModal

    modal = object.__new__(TTSProfileEditorModal)
    Screen.__init__(modal)
    modal.initial_draft = None
    modal.initial_name, modal.initial_model_id, modal.initial_voice_id = (
        "Saved",
        "model",
        None,
    )
    controls = {}
    for name, value in (
        ("name", "Changed" if changed else "Saved"),
        ("model", "model"),
        ("voice", ""),
    ):
        control = Input()
        control.set_reactive(Input.value, value)
        controls[f"#stts-profile-editor-{name}"] = control
    modal.query_one = lambda selector, cls: controls[selector]
    assert bool(probe_unsaved_editors(editors=(modal,))) is changed


@pytest.mark.parametrize("change", [None, "text", "reference", "clone", "pending"])
def test_speech_screen_routes_to_existing_playground_controls(change):
    import asyncio

    from textual.screen import Screen
    from textual.widget import Widget
    from textual.widgets import TextArea

    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.UI.Speech.speech_playback_mixin import EXAMPLE_TEXTS
    from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
    from tldw_chatbook.UI.STTS_Window import STTSWindow

    pane = object.__new__(SpeechPlaygroundPane)
    Widget.__init__(pane)
    control = TextArea("User draft" if change == "text" else EXAMPLE_TEXTS[0])
    pane.query_one = lambda selector, cls: control
    pane.reference_audio_path = "reference" if change == "reference" else None
    pane.higgs_reference_audio_path = None
    pane._clone_setup_source_path = "clone" if change == "clone" else None
    pane._clone_setup_canonical = None
    pane._active_profile_name_modal = None
    pane._clone_setup_validation_task = object() if change == "pending" else None
    pane._clone_setup_retained_tasks = set()
    window = object.__new__(STTSWindow)
    Widget.__init__(window)
    window._view_mount_lock = asyncio.Lock()
    window._mounted_view = "playground"
    window.set_reactive(STTSWindow.current_view, "playground")
    window._pending_adopted_preset = None
    window.query_one = lambda cls: pane
    screen = object.__new__(STTSScreen)
    Screen.__init__(screen)
    screen.stts_window = window
    assert bool(probe_unsaved_editors(editors=(screen, window))) is (change is not None)
    assert control.text == ("User draft" if change == "text" else EXAMPLE_TEXTS[0])


@pytest.mark.parametrize("changed", [False, True])
def test_speech_preferences_reads_visible_controls_without_staging(changed):
    from textual.widget import Widget
    from textual.widgets import Input, Select, Switch

    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
    from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot
    from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane

    pane = object.__new__(SpeechSettingsPane)
    Widget.__init__(pane)
    pane._is_mounted = True
    pane._dirty = False
    pane._saved_snapshot = StudioTTSPreferencesSnapshot()
    pane._global_preferences = TTSPreferencesSnapshot(
        "audio_cpp", "first_available", None, "server_default", None, "wav", 1.0
    )
    controls = {}
    for name in ("provider", "model-mode", "voice-mode"):
        select = Select([("Inherit", "__inherit__")])
        select.set_reactive(Select.value, "__inherit__")
        controls[f"#studio-tts-{name}"] = select
    for name in ("model-id", "voice-id"):
        controls[f"#studio-tts-{name}"] = Input()
    toggle = Switch()
    toggle.set_reactive(
        Switch.value,
        not pane._saved_snapshot.auto_play
        if changed
        else pane._saved_snapshot.auto_play,
    )
    controls["#studio-tts-auto-play"] = toggle
    pane.query_one = lambda selector, cls: controls[selector]
    assert bool(probe_unsaved_editors(editors=(pane,))) is changed
    assert pane._dirty is False


@pytest.mark.parametrize("state", ["ready", "draft", "pending"])
def test_speech_profile_library_uses_existing_maintenance_state(state):
    from textual.widget import Widget

    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.UI.stts_profile_library import STTSProfileLibrary

    pane = object.__new__(STTSProfileLibrary)
    Widget.__init__(pane)
    draft = object() if state == "draft" else None
    pane._retained_editor_draft = draft
    pane._active_modal = None
    pane._active_bundle_handle = None
    pane._bundle_cleanup_failed = False
    pane._action_calls = {object()} if state == "pending" else set()
    pane._export_workers = set()
    pane._bundle_invalidation_tasks = set()
    pane._active_page_task = None
    pane._retained_cleanup_task = None
    pane._export_operations = set()
    pane._live = True
    pane._export_unqualified = False
    result = probe_unsaved_editors(editors=(pane,))
    assert bool(result) is (state != "ready")
    assert pane._retained_editor_draft is draft


@pytest.mark.parametrize(
    "route,snapshot",
    [
        ("library", {"library_media_scope": {"query": "draft", "unexpected": True}}),
        (
            "personas",
            {
                "personas_preview": {
                    "greeting": "test",
                    "history": [{"role": "user", "content": "test", "draft": "hidden"}],
                    "seeded_for": None,
                    "greeting_index": 0,
                }
            },
        ),
        (
            "stts",
            {
                "speech_playground_axes": {
                    "tts-provider-select": "audio_cpp",
                    "hidden_draft": "text",
                }
            },
        ),
    ],
)
def test_retained_navigation_unknown_nested_shape_refuses(route, snapshot):
    from tldw_chatbook.Backup_Recovery.unsaved_editors import probe_unsaved_editors
    from tldw_chatbook.UI.Navigation.screen_state_store import (
        RuntimeIdentity,
        ScreenStateStore,
    )

    store = ScreenStateStore()
    store.save(route, snapshot, RuntimeIdentity("local"))
    assert (
        probe_unsaved_editors(screen_state_store=store)[0].reason
        == "unknown-editor-state"
    )
