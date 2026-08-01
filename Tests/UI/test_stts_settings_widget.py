"""Behavioral coverage for the Studio-only TTS preference widget."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Select, Static

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSLoadResult,
    StudioTTSLoadState,
    StudioTTSPreferencesSnapshot,
    StudioTTSSelectionOverrides,
    StudioTTSWriteResult,
    StudioTTSWriteStatus,
)
from tldw_chatbook.UI.STTS_Window import STTSWindow
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane


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


def _global_audio_cpp() -> TTSPreferencesSnapshot:
    return TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="first_available",
        model_id=None,
        voice_mode="server_default",
        voice_id=None,
        response_format="wav",
        speed=1.0,
    )


class _Store:
    def __init__(self, snapshot: StudioTTSPreferencesSnapshot | None = None) -> None:
        self.snapshot = snapshot or StudioTTSPreferencesSnapshot()
        self.saved: list[StudioTTSPreferencesSnapshot] = []
        self.load_threads: list[int] = []

    def load(self, *, migrate: bool = True) -> StudioTTSLoadResult:
        del migrate
        self.load_threads.append(threading.get_ident())
        return StudioTTSLoadResult(self.snapshot, StudioTTSLoadState.LOADED)

    def save(self, snapshot: StudioTTSPreferencesSnapshot) -> StudioTTSWriteResult:
        self.saved.append(snapshot)
        self.snapshot = replace(snapshot, revision=snapshot.revision + 1)
        return StudioTTSWriteResult(StudioTTSWriteStatus.SAVED, self.snapshot)


class _Host(App[None]):
    def __init__(
        self,
        store: _Store,
        *,
        inject_load: bool = True,
        global_preferences: TTSPreferencesSnapshot | None = None,
    ) -> None:
        super().__init__()
        self.store = store
        self.inject_load = inject_load
        self.global_preferences = global_preferences or _global_openai()
        self.global_saves: list[STTSSettingsSaveEvent] = []

    def compose(self) -> ComposeResult:
        yield SpeechSettingsPane(
            store=self.store,
            global_preferences=self.global_preferences,
            load_result=(
                StudioTTSLoadResult(self.store.snapshot, StudioTTSLoadState.LOADED)
                if self.inject_load
                else None
            ),
            id="speech-settings-pane",
        )

    def post_message(self, message: Any) -> bool:
        if isinstance(message, STTSSettingsSaveEvent):
            self.global_saves.append(message)
            return True
        return super().post_message(message)


class _RecoveredHost(App[None]):
    def compose(self) -> ComposeResult:
        yield SpeechSettingsPane(
            store=_Store(),
            global_preferences=_global_openai(),
            load_result=StudioTTSLoadResult(
                StudioTTSPreferencesSnapshot(),
                StudioTTSLoadState.RECOVERED,
                ("speech_studio.provider_options.chatterbox.exaggeration",),
            ),
            id="speech-settings-pane",
        )


class _WindowHost(App[None]):
    def __init__(self, store: _Store) -> None:
        super().__init__()
        self.window = STTSWindow(self)
        self.window._studio_store = store

    def compose(self) -> ComposeResult:
        yield self.window


@pytest.mark.asyncio
async def test_mount_loads_studio_storage_off_the_message_pump() -> None:
    store = _Store()
    test_thread = threading.get_ident()
    app = _Host(store, inject_load=False)

    async with app.run_test(size=(120, 48)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert len(store.load_threads) == 1
    assert store.load_threads[0] != test_thread


@pytest.mark.asyncio
async def test_speech_window_loads_before_mounting_an_editable_view() -> None:
    store = _Store()
    test_thread = threading.get_ident()
    app = _WindowHost(store)

    async with app.run_test(size=(140, 52)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one(SpeechPlaygroundPane)

    assert len(store.load_threads) == 1
    assert store.load_threads[0] != test_thread


@pytest.mark.asyncio
async def test_audio_cpp_contract_is_visible_and_cannot_be_overridden() -> None:
    store = _Store(
        StudioTTSPreferencesSnapshot(
            selection=StudioTTSSelectionOverrides(provider_id="audio_cpp")
        )
    )
    app = _Host(store)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        output = app.query_one("#studio-tts-format", Select)
        speed = app.query_one("#studio-tts-speed", Input)
        assert output.value == "wav"
        assert output.disabled
        assert speed.value == "1.0"
        assert speed.disabled
        assert "Fixed by audio.cpp" in str(
            app.query_one("#studio-tts-format-source", Static).render()
        )


@pytest.mark.asyncio
async def test_valid_explicit_audio_cpp_fixed_values_do_not_create_false_dirty_state() -> (
    None
):
    store = _Store(
        StudioTTSPreferencesSnapshot(
            revision=2,
            selection=StudioTTSSelectionOverrides(
                provider_id="audio_cpp",
                response_format="wav",
                speed=1.0,
            ),
        )
    )
    app = _Host(store)

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        assert not app.query_one(SpeechSettingsPane).is_dirty


@pytest.mark.asyncio
async def test_inherited_audio_cpp_surfaces_incompatible_sparse_axes() -> None:
    store = _Store(
        StudioTTSPreferencesSnapshot(
            revision=2,
            selection=StudioTTSSelectionOverrides(
                response_format="mp3",
                speed=1.2,
            ),
        )
    )
    app = _Host(store, global_preferences=_global_audio_cpp())

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        assert pane.is_dirty
        for field in ("format", "speed"):
            error = app.query_one(f"#studio-tts-{field}-error", Static)
            assert not error.has_class("hidden")
            assert "incompatible with audio.cpp" in str(error.render())


@pytest.mark.asyncio
async def test_saved_provider_values_restore_after_switching_away_and_back() -> None:
    saved = StudioTTSPreferencesSnapshot(
        revision=4,
        selection=StudioTTSSelectionOverrides(
            provider_id="chatterbox",
            model_mode="exact",
            model_id="saved-model",
            voice_mode="exact",
            voice_id="saved-voice",
        ),
        provider_options={"chatterbox": {"exaggeration": 0.6}},
    )
    app = _Host(_Store(saved))

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        pane._apply_provider("audio_cpp")
        pane._apply_provider("chatterbox")

        assert app.query_one("#studio-tts-model-id", Input).value == "saved-model"
        assert app.query_one("#studio-tts-voice-id", Input).value == "saved-voice"
        assert app.query_one("#chatterbox-exaggeration-input", Input).value == "0.6"


@pytest.mark.asyncio
async def test_clean_provider_switch_does_not_prompt_for_unsaved_changes() -> None:
    app = _Host(_Store())
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        ask_leave = AsyncMock(return_value="cancel")
        pane._ask_leave_choice = ask_leave

        app.query_one("#studio-tts-provider", Select).value = "audio_cpp"
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.query_one("#studio-tts-provider", Select).value == "audio_cpp"
        ask_leave.assert_not_awaited()


@pytest.mark.asyncio
async def test_dirty_provider_switch_cancel_preserves_draft_and_focus() -> None:
    app = _Host(_Store())
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        app.query_one("#studio-tts-model-mode", Select).value = "exact"
        await pilot.pause()
        model = app.query_one("#studio-tts-model-id", Input)
        model.value = "unsaved-model"
        model.focus()
        await pilot.pause()
        pane._ask_leave_choice = AsyncMock(return_value="cancel")

        app.query_one("#studio-tts-provider", Select).value = "audio_cpp"
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.query_one("#studio-tts-provider", Select).value == "__inherit__"
        assert model.value == "unsaved-model"
        assert app.focused is model


@pytest.mark.asyncio
async def test_discarded_provider_draft_is_not_retained_when_switching_back() -> None:
    saved = StudioTTSPreferencesSnapshot(
        revision=3,
        selection=StudioTTSSelectionOverrides(
            provider_id="chatterbox",
            model_mode="exact",
            model_id="saved-model",
        ),
    )
    app = _Host(_Store(saved))
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        model = app.query_one("#studio-tts-model-id", Input)
        model.value = "discard-me"
        await pilot.pause()
        pane._ask_leave_choice = AsyncMock(return_value="discard")

        app.query_one("#studio-tts-provider", Select).value = "audio_cpp"
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one("#studio-tts-provider", Select).value == "audio_cpp"

        pane._apply_provider("chatterbox")
        assert model.value == "saved-model"


@pytest.mark.asyncio
async def test_unsupported_tuning_fails_at_its_field_without_persisting() -> None:
    store = _Store()
    app = _Host(store)
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        pane._apply_provider("chatterbox")
        app.query_one("#chatterbox-exaggeration-input", Input).value = "2.0"
        assert not await pane.save_preferences()

        error = app.query_one("#studio-tts-exaggeration-error", Static)
        assert not error.has_class("hidden")
        assert "not supported" in str(error.render()).casefold()

    assert not store.saved
    assert not app.global_saves


@pytest.mark.asyncio
async def test_invalid_exact_identifier_fails_at_its_own_field() -> None:
    store = _Store()
    app = _Host(store)
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        app.query_one("#studio-tts-model-mode", Select).value = "exact"
        await pilot.pause()
        app.query_one("#studio-tts-model-id", Input).value = "********"
        assert not await pane.save_preferences()

        error = app.query_one("#studio-tts-model-id-error", Static)
        assert not error.has_class("hidden")
        assert "not supported" in str(error.render()).casefold()
        assert app.focused is app.query_one("#studio-tts-model-id", Input)

    assert not store.saved


@pytest.mark.asyncio
async def test_recovered_saved_tuning_marks_its_field_with_safe_copy() -> None:
    app = _RecoveredHost()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        error = app.query_one("#studio-tts-exaggeration-error", Static)
        assert not error.has_class("hidden")
        assert str(error.render()) == "Ignored unsupported saved Studio value"


@pytest.mark.asyncio
async def test_studio_save_does_not_reconfigure_or_mutate_global_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook import config as config_module

    store = _Store()
    app = _Host(store)
    forbidden = Mock(side_effect=AssertionError("Studio save crossed its owner"))
    for helper_name in (
        "apply_settings_mutation_to_cli_config",
        "save_settings_to_cli_config",
        "save_setting_to_cli_config",
        "delete_settings_from_cli_config",
    ):
        monkeypatch.setattr(config_module, helper_name, forbidden)
    monkeypatch.setattr(
        SpeechSettingsPane,
        "_tts_service_factory",
        forbidden,
    )

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        assert await app.query_one(SpeechSettingsPane).save_preferences()

    assert len(store.saved) == 1
    assert not app.global_saves
    forbidden.assert_not_called()
