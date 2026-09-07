"""Studio-only Speech preference editor contracts (TASK-1986)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Static, Switch, TextArea

from Tests.UI.speech_playground_fixtures import (
    FakeTTSService,
    _resolved,
    _wait_until,
)
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSPlaygroundGenerateEvent,
    STTSProviderConfigurationChanged,
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_service import TTSPlaygroundSelectionPreset
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSLoadResult,
    StudioTTSLoadState,
    StudioTTSPreferencesSnapshot,
    StudioTTSSelectionOverrides,
    StudioTTSWriteResult,
    StudioTTSWriteStatus,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_profile_mixin import (
    AdoptStudioPreferencesRequested,
)
from tldw_chatbook.UI.Speech.speech_settings_pane import (
    SpeechSettingsPane,
    StudioPreferencesSaved,
)
from tldw_chatbook.UI.STTS_Window import STTSWindow


def _global_openai() -> TTSPreferencesSnapshot:
    return TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="tts-1-hd",
        voice_mode="exact",
        voice_id="shimmer",
        response_format="mp3",
        speed=1.0,
    )


def _global_pocket() -> TTSPreferencesSnapshot:
    return TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="pocket-tts",
        voice_mode="exact",
        voice_id="alba",
        response_format="wav",
        speed=1.2,
    )


def _global_audio_cpp() -> TTSPreferencesSnapshot:
    return TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="exact",
        model_id="kokoro-82m",
        voice_mode="exact",
        voice_id="af_heart",
        response_format="wav",
        speed=1.0,
    )


class _Store:
    def __init__(
        self,
        snapshot: StudioTTSPreferencesSnapshot | None = None,
        *,
        save_status: StudioTTSWriteStatus = StudioTTSWriteStatus.SAVED,
    ) -> None:
        self.snapshot = snapshot or StudioTTSPreferencesSnapshot()
        self.save_status = save_status
        self.saved: list[StudioTTSPreferencesSnapshot] = []
        self.reset: list[StudioTTSPreferencesSnapshot] = []

    def load(self, *, migrate: bool = True) -> StudioTTSLoadResult:
        del migrate
        return StudioTTSLoadResult(self.snapshot, StudioTTSLoadState.LOADED)

    def save(self, snapshot: StudioTTSPreferencesSnapshot) -> StudioTTSWriteResult:
        self.saved.append(snapshot)
        if self.save_status not in {
            StudioTTSWriteStatus.SAVED,
            StudioTTSWriteStatus.UNCHANGED,
            StudioTTSWriteStatus.SAVED_CACHE_RELOAD_FAILED,
        }:
            return StudioTTSWriteResult(self.save_status, None)
        persisted = replace(snapshot, revision=snapshot.revision + 1)
        self.snapshot = persisted
        return StudioTTSWriteResult(self.save_status, persisted)

    def reset_to_global(
        self, snapshot: StudioTTSPreferencesSnapshot
    ) -> StudioTTSWriteResult:
        self.reset.append(snapshot)
        persisted = StudioTTSPreferencesSnapshot(revision=snapshot.revision + 1)
        self.snapshot = persisted
        return StudioTTSWriteResult(StudioTTSWriteStatus.SAVED, persisted)


class _Host(App[None]):
    def __init__(self, pane: SpeechSettingsPane) -> None:
        super().__init__()
        self.pane = pane
        self.navigation: list[NavigateToScreen] = []
        self.global_saves: list[STTSSettingsSaveEvent] = []
        self.studio_snapshots: list[StudioTTSPreferencesSnapshot] = []
        self.studio_messages: list[StudioPreferencesSaved] = []
        self.notices: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield self.pane

    def post_message(self, message: Any) -> bool:
        if isinstance(message, NavigateToScreen):
            self.navigation.append(message)
            return True
        if isinstance(message, STTSSettingsSaveEvent):
            self.global_saves.append(message)
            return True
        if isinstance(message, StudioPreferencesSaved):
            self.studio_messages.append(message)
            self.studio_snapshots.append(message.snapshot)
            return True
        return super().post_message(message)

    def notify(
        self,
        message: str,
        *,
        severity: str = "information",
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.notices.append((message, severity))


class _STTSHost(App[None]):
    def __init__(self, window: STTSWindow) -> None:
        super().__init__()
        self.window = window

    def compose(self) -> ComposeResult:
        yield self.window


def _pane(
    store: _Store,
    *,
    load_state: StudioTTSLoadState = StudioTTSLoadState.LOADED,
    issues: tuple[str, ...] = (),
    adopted_preset: TTSPlaygroundSelectionPreset | None = None,
) -> SpeechSettingsPane:
    return SpeechSettingsPane(
        id="speech-settings-pane",
        store=store,
        global_preferences=_global_openai(),
        load_result=StudioTTSLoadResult(store.snapshot, load_state, issues),
        adopted_preset=adopted_preset,
    )


@pytest.mark.asyncio
async def test_studio_surface_states_scope_and_excludes_every_global_owner() -> None:
    app = _Host(_pane(_Store()))

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        title = str(app.query_one("#studio-tts-title", Static).render())
        scope = str(app.query_one("#studio-tts-scope", Static).render())

        assert title == "Studio TTS Preferences"
        assert "only the Speech Studio" in scope
        assert "never change global defaults or character TTS profiles" in scope
        assert app.query_one("#studio-tts-save-btn", Button).label.plain == (
            "Save Studio Preferences"
        )
        assert app.query_one("#studio-tts-revert-btn", Button).label.plain == "Revert"
        assert app.query_one("#studio-tts-reset-btn", Button).label.plain == (
            "Reset to Global"
        )
        assert str(
            app.query_one("#studio-tts-voice-tools-heading", Static).render()
        ) == "Voice tools"
        assert app.query_one("#voice-profiles", Button).label.plain == (
            "Voice Profiles"
        )
        assert app.query_one("#voice-blends", Button).label.plain == "Voice Blends"
        assert not app.query("#studio-tts-voice-profile-heading")

        for forbidden in (
            "openai-api-key-input",
            "openai-base-url-input",
            "audio-cpp-base-url-input",
            "kokoro-device-select",
            "higgs-model-path-input",
            "alltalk-url-input",
        ):
            assert not app.query(f"#{forbidden}"), forbidden


@pytest.mark.asyncio
async def test_global_settings_link_carries_only_canonical_provider_context() -> None:
    store = _Store(
        StudioTTSPreferencesSnapshot(
            revision=3,
            selection=StudioTTSSelectionOverrides(provider_id="chatterbox"),
        )
    )
    app = _Host(_pane(store))

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        await pilot.click("#studio-tts-open-global-btn")
        await pilot.pause()

    assert len(app.navigation) == 1
    message = app.navigation[0]
    assert message.screen_name == "settings"
    assert message.screen_context == {
        "category": "speech-tts",
        "provider": "chatterbox",
        "intent": "configure",
    }


@pytest.mark.asyncio
async def test_global_link_keeps_selected_provider_when_draft_is_discarded() -> None:
    store = _Store()
    pane = _pane(store)
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane._apply_provider("chatterbox")
        assert pane.is_dirty
        pane._ask_leave_choice = AsyncMock(return_value="discard")

        await pane._open_global_settings()

        assert pane.query_one("#studio-tts-provider", Select).value == "__inherit__"

    assert len(app.navigation) == 1
    assert app.navigation[0].screen_context == {
        "category": "speech-tts",
        "provider": "chatterbox",
        "intent": "configure",
    }


@pytest.mark.asyncio
async def test_save_writes_only_sparse_studio_snapshot_and_never_global_settings() -> (
    None
):
    store = _Store()
    pane = _pane(store)
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane._apply_provider("chatterbox")
        pane.query_one("#studio-tts-model-mode", Select).value = "exact"
        pane.query_one("#studio-tts-model-id", Input).value = "chatterbox"
        pane.query_one("#studio-tts-voice-mode", Select).value = "exact"
        pane.query_one("#studio-tts-voice-id", Input).value = "warm"
        pane.query_one("#studio-tts-format", Select).value = "wav"
        pane.query_one("#studio-tts-speed", Input).value = "1.15"
        pane.query_one("#chatterbox-exaggeration-input", Input).value = "0.7"
        pane.query_one("#chatterbox-cfg-weight-input", Input).value = "0.4"

        assert await pane.save_preferences()

    assert not app.global_saves
    assert len(store.saved) == 1
    saved = store.saved[0]
    assert saved.selection == StudioTTSSelectionOverrides(
        provider_id="chatterbox",
        model_mode="exact",
        model_id="chatterbox",
        voice_mode="exact",
        voice_id="warm",
        response_format="wav",
        speed=1.15,
    )
    assert dict(saved.provider_options["chatterbox"]) == {
        "exaggeration": 0.7,
        "cfg_weight": 0.4,
    }


@pytest.mark.asyncio
async def test_inherited_values_are_visible_but_not_copied_on_save() -> None:
    store = _Store()
    pane = _pane(store)
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        assert "Inherited — Global defaults: OpenAI" in str(
            app.query_one("#studio-tts-provider-source", Static).render()
        )
        assert "tts-1-hd" in str(
            app.query_one("#studio-tts-model-source", Static).render()
        )
        assert await pane.save_preferences()

    assert store.saved[0].selection == StudioTTSSelectionOverrides()


@pytest.mark.asyncio
async def test_mounted_settings_refreshes_global_sources_without_losing_draft_focus() -> (
    None
):
    pane = _pane(_Store())
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        model_mode = pane.query_one("#studio-tts-model-mode", Select)
        model_id = pane.query_one("#studio-tts-model-id", Input)
        model_mode.value = "exact"
        await pilot.pause()
        model_id.value = "unsaved-studio-model"
        model_id.focus()
        await pilot.pause()
        assert pane.is_dirty
        assert model_id.has_focus

        pane.refresh_global_preferences(_global_pocket())
        await pilot.pause()

        _global_pocket_snapshot = pane._global_preferences
        assert _global_pocket_snapshot == _global_pocket()
        assert model_mode.value == "exact"
        assert model_id.value == "unsaved-studio-model"
        assert model_id.has_focus
        assert pane.is_dirty
        assert "alba" in str(
            pane.query_one("#studio-tts-voice-source", Static).render()
        )
        assert "Studio override" in str(
            pane.query_one("#studio-tts-model-source", Static).render()
        )


@pytest.mark.asyncio
async def test_inherited_provider_refresh_repopulates_dependent_controls_both_ways() -> (
    None
):
    pane = _pane(_Store())
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        provider = pane.query_one("#studio-tts-provider", Select)
        model_mode = pane.query_one("#studio-tts-model-mode", Select)
        model_id = pane.query_one("#studio-tts-model-id", Input)
        voice_mode = pane.query_one("#studio-tts-voice-mode", Select)
        voice_id = pane.query_one("#studio-tts-voice-id", Input)
        output_format = pane.query_one("#studio-tts-format", Select)
        speed = pane.query_one("#studio-tts-speed", Input)

        model_mode.value = "exact"
        await pilot.pause()
        model_id.value = "unsaved-cross-provider-model"
        output_format.value = "flac"
        speed.value = "1.4"
        model_id.focus()
        await pilot.pause()

        pane.refresh_global_preferences(_global_audio_cpp())
        await pilot.pause()

        assert provider.value == "__inherit__"
        assert model_mode.value == "exact"
        assert model_id.value == "unsaved-cross-provider-model"
        assert model_id.has_focus
        assert voice_mode.value == "__inherit__"
        assert voice_id.value == ""
        assert output_format.value == "wav"
        assert output_format.disabled
        assert speed.value == "1.0"
        assert speed.disabled
        assert "audio.cpp" in str(
            pane.query_one("#studio-tts-provider-source", Static).render()
        )
        assert "Fixed by audio.cpp" in str(
            pane.query_one("#studio-tts-format-source", Static).render()
        )

        pane.refresh_global_preferences(_global_pocket())
        await pilot.pause()

        assert provider.value == "__inherit__"
        assert model_mode.value == "exact"
        assert model_id.value == "unsaved-cross-provider-model"
        assert model_id.has_focus
        assert voice_mode.value == "__inherit__"
        assert voice_id.value == ""
        assert output_format.value == "flac"
        assert not output_format.disabled
        assert speed.value == "1.4"
        assert not speed.disabled
        assert "alba" in str(
            pane.query_one("#studio-tts-voice-source", Static).render()
        )
        assert "Studio override" in str(
            pane.query_one("#studio-tts-model-source", Static).render()
        )
        assert "Studio override" in str(
            pane.query_one("#studio-tts-format-source", Static).render()
        )
        assert "Studio override" in str(
            pane.query_one("#studio-tts-speed-source", Static).render()
        )


def test_settings_refresh_requires_exact_global_snapshot_type() -> None:
    pane = _pane(_Store())

    with pytest.raises(TypeError, match="global preferences"):
        pane.refresh_global_preferences(object())  # type: ignore[arg-type]


def test_retained_window_reloads_one_snapshot_and_ignores_stale_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter((_global_openai(), _global_pocket(), _global_openai()))
    reads: list[TTSPreferencesSnapshot] = []

    def read_snapshot() -> TTSPreferencesSnapshot:
        snapshot = next(snapshots)
        reads.append(snapshot)
        return snapshot

    monkeypatch.setattr(
        SpeechSettingsPane,
        "_read_global_preferences",
        staticmethod(read_snapshot),
    )
    window = STTSWindow(None)
    reads.clear()
    settings_refreshes: list[TTSPreferencesSnapshot] = []
    playground_refreshes: list[TTSPreferencesSnapshot] = []
    settings = SimpleNamespace(refresh_global_preferences=settings_refreshes.append)
    playground = SimpleNamespace(refresh_global_preferences=playground_refreshes.append)
    monkeypatch.setattr(
        window,
        "query",
        lambda pane_type: (
            [settings]
            if pane_type is SpeechSettingsPane
            else [playground]
            if pane_type is SpeechPlaygroundPane
            else []
        ),
    )

    window.receive_provider_configuration_changed(
        STTSProviderConfigurationChanged("openai", 4, 9)
    )
    window.receive_provider_configuration_changed(
        STTSProviderConfigurationChanged("openai", 4, 8)
    )
    window.receive_provider_configuration_changed(
        STTSProviderConfigurationChanged("openai", 4, 9)
    )

    assert reads == [_global_pocket()]
    assert settings_refreshes == reads
    assert playground_refreshes == reads
    assert window._global_preferences == _global_pocket()

    window.receive_provider_configuration_changed(
        STTSProviderConfigurationChanged("openai", 5, 10)
    )

    assert reads == [_global_pocket(), _global_openai()]
    assert settings_refreshes == reads
    assert playground_refreshes == reads


def test_retained_window_safely_caches_refresh_when_no_pane_is_mounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _global_pocket()
    monkeypatch.setattr(
        SpeechSettingsPane,
        "_read_global_preferences",
        staticmethod(lambda: snapshot),
    )
    window = STTSWindow(None)
    monkeypatch.setattr(window, "query", lambda _pane_type: [])

    window.receive_provider_configuration_changed(
        STTSProviderConfigurationChanged("openai", 4, 9)
    )

    assert window._global_preferences is snapshot


@pytest.mark.asyncio
async def test_absent_playground_rebases_inherited_axes_before_actual_remount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_global = _global_openai()
    new_global = _global_pocket()
    monkeypatch.setattr(
        SpeechSettingsPane,
        "_read_global_preferences",
        staticmethod(lambda: old_global),
    )
    monkeypatch.setattr(STTSWindow, "on_mount", lambda self: None)
    monkeypatch.setattr(
        STTSWindow,
        "_global_provider_configuration_states",
        staticmethod(dict),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )
    window = STTSWindow(SimpleNamespace())
    window._studio_load_result = StudioTTSLoadResult(
        StudioTTSPreferencesSnapshot(),
        StudioTTSLoadState.LOADED,
    )
    app = _STTSHost(window)

    async with app.run_test(size=(160, 60)) as pilot:
        window._mount_view("playground", force=True)
        await pilot.pause()
        first = window.query_one(SpeechPlaygroundPane)
        first.axis_values = {
            "tts-provider-select": "openai",
            "tts-model-select": "tts-1-hd",
            "tts-voice-select": "shimmer",
            "tts-format-select": "mp3",
            "tts-speed-input": "1.7",
        }
        first._refresh_axis_markers()

        window.select_view("settings")
        await pilot.pause()
        assert not window.query(SpeechPlaygroundPane)

        monkeypatch.setattr(
            SpeechSettingsPane,
            "_read_global_preferences",
            staticmethod(lambda: new_global),
        )
        window.receive_provider_configuration_changed(
            STTSProviderConfigurationChanged("openai", 4, 9)
        )
        window.select_view("playground")
        await pilot.pause()

        remounted = window.query_one(SpeechPlaygroundPane)
        assert remounted is not first
        assert remounted.axis_values == {
            "tts-provider-select": "openai",
            "tts-model-select": "pocket-tts",
            "tts-voice-select": "alba",
            "tts-format-select": "wav",
            "tts-speed-input": "1.7",
        }
        for inherited_axis in (
            "tts-provider-select",
            "tts-model-select",
            "tts-voice-select",
            "tts-format-select",
        ):
            chip = remounted.query_one(f"#speech-axis-{inherited_axis}", Static)
            assert not chip.has_class("speech-chip-override"), inherited_axis
        speed_chip = remounted.query_one("#speech-axis-tts-speed-input", Static)
        assert speed_chip.has_class("speech-chip-override")
        assert "1.2" in str(speed_chip.tooltip)


@pytest.mark.asyncio
async def test_revert_and_reset_restore_only_the_studio_scope() -> None:
    saved = StudioTTSPreferencesSnapshot(
        revision=7,
        selection=StudioTTSSelectionOverrides(
            provider_id="chatterbox",
            model_mode="exact",
            model_id="chatterbox",
        ),
        provider_options={"chatterbox": {"exaggeration": 0.6}},
    )
    store = _Store(saved)
    pane = _pane(store)
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane.query_one("#studio-tts-model-id", Input).value = "temporary"
        assert pane.is_dirty
        assert await pane.revert_preferences()
        assert pane.query_one("#studio-tts-model-id", Input).value == "chatterbox"
        assert not pane.is_dirty

        assert await pane.reset_to_global()
        assert pane.saved_snapshot.selection == StudioTTSSelectionOverrides()
        assert not pane.is_dirty

    assert store.reset == [saved]
    assert not app.global_saves


@pytest.mark.asyncio
async def test_reset_to_global_discards_the_bounded_playground_draft() -> None:
    """A reset must not leave stale exact axes ahead of inherited globals."""

    saved = StudioTTSPreferencesSnapshot(
        revision=7,
        selection=StudioTTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="supertonic-3",
            voice_mode="exact",
            voice_id="F1",
        ),
    )
    store = _Store(saved)
    pane = _pane(store)
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        assert await pane.reset_to_global()

    reset_message = app.studio_messages[-1]
    assert reset_message.reset_to_global is True

    window = STTSWindow(
        None,
        playground_axis_values={
            "tts-provider-select": "audio_cpp",
            "tts-model-select": "supertonic-3",
            "tts-voice-select": "F1",
            "tts-format-select": "wav",
            "tts-speed-input": "1.0",
        },
    )
    window.on_studio_preferences_saved(reset_message)

    assert window._playground_axis_values == {}
    assert window._studio_load_result is not None
    assert window._studio_load_result.snapshot.selection == (
        StudioTTSSelectionOverrides()
    )


@pytest.mark.asyncio
async def test_revert_reloads_latest_snapshot_after_a_save_conflict() -> None:
    saved = StudioTTSPreferencesSnapshot(
        revision=2,
        selection=StudioTTSSelectionOverrides(
            provider_id="chatterbox",
            model_mode="exact",
            model_id="original-model",
        ),
    )
    store = _Store(saved, save_status=StudioTTSWriteStatus.CONFLICT)
    pane = _pane(store)
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane.query_one("#studio-tts-model-id", Input).value = "conflicting-draft"
        assert not await pane.save_preferences()

        store.snapshot = StudioTTSPreferencesSnapshot(
            revision=3,
            selection=StudioTTSSelectionOverrides(
                provider_id="chatterbox",
                model_mode="exact",
                model_id="newer-saved-model",
            ),
        )
        assert await pane.revert_preferences()

        assert pane.saved_snapshot.revision == 3
        assert pane.query_one("#studio-tts-model-id", Input).value == (
            "newer-saved-model"
        )
        assert not pane.is_dirty

    assert app.studio_snapshots == [store.snapshot]


@pytest.mark.asyncio
async def test_failed_or_cancelled_leave_keeps_draft_and_focus() -> None:
    store = _Store(save_status=StudioTTSWriteStatus.FAILED)
    pane = _pane(store)
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        model_mode = pane.query_one("#studio-tts-model-mode", Select)
        model_mode.value = "exact"
        await pilot.pause()
        model_id = pane.query_one("#studio-tts-model-id", Input)
        model_id.value = "draft-model"
        model_id.focus()
        await pilot.pause()

        pane._ask_leave_choice = AsyncMock(return_value="cancel")
        assert not await pane.confirm_leave()
        assert model_id.value == "draft-model"
        assert app.focused is model_id

        pane._ask_leave_choice = AsyncMock(return_value="save")
        assert not await pane.confirm_leave()
        assert model_id.value == "draft-model"
        assert app.focused is model_id


@pytest.mark.asyncio
async def test_global_link_dirty_guard_supports_keyboard_cancel_and_save() -> None:
    store = _Store()
    pane = _pane(store)
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane.query_one("#studio-tts-model-mode", Select).value = "exact"
        await pilot.pause()
        model_id = pane.query_one("#studio-tts-model-id", Input)
        model_id.value = "keyboard-model"
        await pilot.pause()

        await pilot.click("#studio-tts-open-global-btn")
        for _ in range(40):
            if app.screen.query("#studio-tts-leave-modal"):
                break
            await pilot.pause(0.01)
        assert app.screen.query("#studio-tts-leave-modal")
        await pilot.press("escape")
        await pilot.pause()
        assert not app.navigation
        assert model_id.value == "keyboard-model"
        assert app.focused is pane.query_one("#studio-tts-open-global-btn", Button)

        await pilot.click("#studio-tts-open-global-btn")
        for _ in range(40):
            if app.screen.query("#studio-tts-leave-modal"):
                break
            await pilot.pause(0.01)
        assert app.screen.query("#studio-tts-leave-modal")
        assert app.focused is app.screen.query_one("#studio-tts-leave-cancel", Button)
        await pilot.press("tab")
        await pilot.press("tab")
        assert app.focused is app.screen.query_one("#studio-tts-leave-save", Button)
        await pilot.press("enter")
        for _ in range(80):
            if app.navigation:
                break
            await pilot.pause(0.01)

    assert len(store.saved) == 1
    assert store.saved[0].selection.model_id == "keyboard-model"
    assert len(app.navigation) == 1


@pytest.mark.asyncio
async def test_explicit_profile_adoption_is_draft_only_until_studio_save() -> None:
    store = _Store()
    preset = TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="supertonic",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
    )
    pane = _pane(store, adopted_preset=preset)
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        assert "Adopted profile preview" in str(
            app.query_one("#studio-tts-adoption-status", Static).render()
        )
        assert not store.saved
        assert pane.is_dirty
        assert await pane.save_preferences()

    assert store.saved[0].selection.provider_id == "audio_cpp"
    assert store.saved[0].selection.model_id == "supertonic"


@pytest.mark.asyncio
async def test_revert_discards_an_adopted_but_unsaved_profile() -> None:
    store = _Store()
    preset = TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="supertonic",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
    )
    pane = _pane(store, adopted_preset=preset)
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        assert pane.is_dirty
        assert await pane.revert_preferences()

        assert not pane.is_dirty
        assert pane.query_one("#studio-tts-provider", Select).value == "__inherit__"
        assert pane.query_one("#studio-tts-adoption-status", Static).has_class("hidden")

    assert not store.saved


@pytest.mark.asyncio
async def test_corrupt_record_offers_a_studio_only_reset() -> None:
    store = _Store()
    pane = _pane(
        store,
        load_state=StudioTTSLoadState.CORRUPT,
        issues=("speech_studio.selection.unknown_field",),
    )
    app = _Host(pane)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        status = str(app.query_one("#studio-tts-status", Static).render())
        assert "Studio preferences" in status
        assert "Reset to Global" in status
        assert "global settings" not in status.casefold()
        assert "character" not in status.casefold()


class _PlaygroundHost(App[None]):
    def __init__(
        self,
        preferences: StudioTTSPreferencesSnapshot,
        *,
        preset: TTSPlaygroundSelectionPreset | None = None,
    ) -> None:
        super().__init__()
        self.preferences = preferences
        self.preset = preset
        self.generated: list[STTSPlaygroundGenerateEvent] = []
        self.adoptions: list[AdoptStudioPreferencesRequested] = []

    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(
            profile_preset=self.preset,
            studio_preferences=self.preferences,
            global_preferences=_global_openai(),
        )

    def post_message(self, message: Any) -> bool:
        if isinstance(message, STTSPlaygroundGenerateEvent):
            self.generated.append(message)
            return True
        if isinstance(message, AdoptStudioPreferencesRequested):
            self.adoptions.append(message)
            return True
        return super().post_message(message)

    def notify(self, *_args: Any, **_kwargs: Any) -> None:
        return


@pytest.mark.asyncio
async def test_current_playground_controls_are_frozen_as_unsaved_studio_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_check_higgs_installation",
        lambda self: None,
    )
    preferences = StudioTTSPreferencesSnapshot(
        revision=6,
        selection=StudioTTSSelectionOverrides(provider_id="audio_cpp"),
    )
    app = _PlaygroundHost(preferences)

    async with app.run_test(size=(160, 60)) as pilot:
        provider = app.query_one("#tts-provider-select", Select)
        await _wait_until(pilot, lambda: provider.value == "audio_cpp")
        app.query_one("#tts-text-input", TextArea).load_text("Studio draft text")
        await pilot.pause()
        await pilot.click("#tts-generate-btn")
        await pilot.pause()

    assert len(app.generated) == 1
    request = app.generated[0].request
    assert request.studio_preferences is preferences
    assert request.studio_draft is not None
    assert request.studio_draft.base_revision == 6
    assert request.studio_draft.selection.provider_id == "audio_cpp"
    assert request.studio_draft.selection.model_id == "<opaque:model>"
    assert request.studio_draft.preview is False


@pytest.mark.asyncio
async def test_saved_provider_tuning_seeds_playground_then_current_edit_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_check_higgs_installation",
        lambda self: None,
    )
    preferences = StudioTTSPreferencesSnapshot(
        revision=9,
        selection=StudioTTSSelectionOverrides(provider_id="chatterbox"),
        provider_options={"chatterbox": {"exaggeration": 0.65, "cfg_weight": 0.35}},
    )
    app = _PlaygroundHost(preferences)

    async with app.run_test(size=(160, 60)) as pilot:
        provider = app.query_one("#tts-provider-select", Select)
        await _wait_until(pilot, lambda: provider.value == "chatterbox")
        await _wait_until(
            pilot,
            lambda: bool(app.query("#tts-exaggeration-input")),
        )
        exaggeration = app.query_one("#tts-exaggeration-input", Input)
        cfg_weight = app.query_one("#tts-cfg-weight-input", Input)
        assert exaggeration.value == "0.65"
        assert cfg_weight.value == "0.35"

        exaggeration.value = "0.8"
        app.query_one("#tts-text-input", TextArea).load_text("Current draft wins")
        await pilot.pause()
        await pilot.click("#tts-generate-btn")
        await pilot.pause()

    request = app.generated[0].request
    assert request.studio_draft is not None
    assert dict(request.studio_draft.selection.provider_options or {}) == {
        "exaggeration": 0.8,
        "cfg_weight": 0.35,
        "temperature": 0.5,
        "num_candidates": 1,
        "validate_with_whisper": False,
    }


@pytest.mark.asyncio
async def test_unsaved_request_tuning_reaches_generation_without_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_check_higgs_installation",
        lambda self: None,
    )
    preferences = StudioTTSPreferencesSnapshot(
        revision=10,
        selection=StudioTTSSelectionOverrides(provider_id="chatterbox"),
        provider_options={"chatterbox": {"exaggeration": 0.6, "cfg_weight": 0.4}},
    )
    app = _PlaygroundHost(preferences)

    async with app.run_test(size=(160, 60)) as pilot:
        provider = app.query_one("#tts-provider-select", Select)
        await _wait_until(pilot, lambda: provider.value == "chatterbox")
        await _wait_until(
            pilot,
            lambda: bool(app.query("#tts-temperature-input")),
        )
        app.query_one("#tts-temperature-input", Input).value = "1.2"
        app.query_one("#tts-num-candidates-input", Input).value = "3"
        app.query_one("#tts-validate-whisper-switch", Switch).value = True
        app.query_one("#tts-text-input", TextArea).load_text("Ephemeral tuning")
        await pilot.pause()
        await pilot.click("#tts-generate-btn")
        await pilot.pause()

    request = app.generated[0].request
    assert request.studio_draft is not None
    assert dict(request.studio_draft.selection.provider_options or {}) == {
        "exaggeration": 0.6,
        "cfg_weight": 0.4,
        "temperature": 1.2,
        "num_candidates": 3,
        "validate_with_whisper": True,
    }
    assert dict(preferences.provider_options["chatterbox"]) == {
        "exaggeration": 0.6,
        "cfg_weight": 0.4,
    }


@pytest.mark.asyncio
async def test_character_profile_is_a_preview_until_explicit_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_check_higgs_installation",
        lambda self: None,
    )
    preferences = StudioTTSPreferencesSnapshot(revision=12)
    preset = TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="<opaque:model>",
        voice_id="[voice]",
        response_format="wav",
        speed=1.0,
        options={},
        availability="available",
    )
    app = _PlaygroundHost(preferences, preset=preset)

    async with app.run_test(size=(160, 60)) as pilot:
        adopt = app.query_one("#tts-adopt-studio-preferences-btn", Button)
        await _wait_until(
            pilot,
            lambda: not adopt.disabled and not adopt.has_class("hidden"),
        )
        app.query_one("#tts-text-input", TextArea).load_text("Preview voice")
        await pilot.pause()
        await pilot.click("#tts-generate-btn")
        await pilot.pause()

        request = app.generated[0].request
        assert request.studio_draft is not None
        assert request.studio_draft.preview is True
        assert request.studio_draft.selection.provider_id == preset.provider_id
        assert request.studio_draft.selection.model_id == preset.model_id
        assert request.studio_draft.selection.voice_id == preset.voice_id
        assert request.studio_preferences is preferences
        assert not app.adoptions

        await pilot.click("#tts-adopt-studio-preferences-btn")
        await pilot.pause()

    assert len(app.adoptions) == 1
    assert app.adoptions[0].preset is preset
    assert preferences == StudioTTSPreferencesSnapshot(revision=12)


@pytest.mark.asyncio
@pytest.mark.parametrize("edited_control", ("provider", "model", "voice"))
async def test_editing_a_profile_preview_axis_ends_preview_before_adoption(
    monkeypatch: pytest.MonkeyPatch,
    edited_control: str,
) -> None:
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_check_higgs_installation",
        lambda self: None,
    )
    preset = TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="<opaque:model>",
        voice_id="[voice]",
        response_format="wav",
        speed=1.0,
        options={},
        availability="available",
    )
    app = _PlaygroundHost(StudioTTSPreferencesSnapshot(revision=13), preset=preset)

    async with app.run_test(size=(160, 60)) as pilot:
        pane = app.query_one(SpeechPlaygroundPane)
        adopt = app.query_one("#tts-adopt-studio-preferences-btn", Button)
        await _wait_until(
            pilot,
            lambda: not adopt.disabled and not adopt.has_class("hidden"),
        )
        await app.workers.wait_for_complete()
        await pilot.pause()

        control = app.query_one(f"#tts-{edited_control}-select", Select)
        replacement = {
            "provider": "openai",
            "model": "second-model",
            "voice": "<script>alert(1)</script>",
        }[edited_control]
        await _wait_until(
            pilot,
            lambda: any(
                value == replacement
                for _label, value in getattr(control, "_options", ())
            ),
        )
        control.value = replacement
        await pilot.pause()

        assert pane._profile_preset is None
        assert adopt.disabled
        assert adopt.has_class("hidden")
        assert not app.adoptions


@pytest.mark.asyncio
async def test_unadopted_character_preview_is_discarded_on_dismissal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_check_higgs_installation",
        lambda self: None,
    )
    preferences = StudioTTSPreferencesSnapshot(revision=13)
    preset = TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="<opaque:model>",
        voice_id="[voice]",
        response_format="wav",
        speed=1.0,
        options={},
        availability="available",
    )
    app = _PlaygroundHost(preferences, preset=preset)

    async with app.run_test(size=(160, 60)) as pilot:
        adopt = app.query_one("#tts-adopt-studio-preferences-btn", Button)
        await _wait_until(
            pilot,
            lambda: not adopt.disabled and not adopt.has_class("hidden"),
        )
        assert not app.adoptions

    assert not app.adoptions
    assert preferences == StudioTTSPreferencesSnapshot(revision=13)
