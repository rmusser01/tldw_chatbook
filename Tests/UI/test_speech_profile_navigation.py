"""Process-local Speech navigation for character TTS profile actions."""

from __future__ import annotations

import asyncio
import struct
import wave
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, DataTable, Select, Static

from Tests.UI.speech_playground_fixtures import FakeTTSService, _resolved
from tldw_chatbook.TTS import (
    LoadedTTSProfile,
    STTSGeneratedAudio,
    STTSPlaygroundResultProjection,
    TTSPlaygroundSelectionPreset,
    TTSProfileDraft,
    TTSProfileService,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.adapter_types import TTSNativeCapabilitySnapshot
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.UI import STTS_Window as stts_window_module
from tldw_chatbook.UI import stts_profile_library as profile_library_module
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen
from tldw_chatbook.UI.Speech.speech_playground_pane import (
    OpenVoiceProfilesRequested,
    SpeechPlaygroundPane,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSNavigationIntent,
)
from tldw_chatbook.UI.Speech.speech_settings_pane import VoiceBlendsPane
from tldw_chatbook.UI.stts_profile_library import STTSProfileLibrary
from tldw_chatbook.UI.STTS_Window import VoiceProfilePickerModal


def test_voice_profiles_action_never_opens_blend_editor() -> None:
    resolver = getattr(stts_window_module, "resolve_speech_navigation", None)

    assert callable(resolver), "Speech destinations need one explicit resolver"
    target = resolver("voice-profiles")
    assert target.view == "profiles"
    assert target.provider_id is None
    assert target.label == "Voice Profiles"


def test_voice_blends_are_labeled_and_scoped_to_kokoro() -> None:
    resolver = getattr(stts_window_module, "resolve_speech_navigation", None)

    assert callable(resolver), "Speech destinations need one explicit resolver"
    target = resolver("voice-blends")
    assert target.view == "blends"
    assert target.provider_id == "kokoro"
    assert target.label == "Voice Blends"


@pytest.fixture(autouse=True)
def _isolated_profile_test_contexts():
    """Every navigation test must release process-local profile authority."""

    profile_library_module._PROFILE_TEST_CONTEXTS.clear()
    yield
    assert profile_library_module._profile_test_context_count() == 0
    profile_library_module._PROFILE_TEST_CONTEXTS.clear()


def _preset() -> TTSPlaygroundSelectionPreset:
    return TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="roleplay/model",
        voice_id="roleplay-voice",
        response_format="wav",
        speed=1.0,
        options={},
        availability="available",
    )


def _artifact(tmp_path, operation_id: str) -> STTSGeneratedAudio:
    path = tmp_path / f"{operation_id}.wav"
    path.write_bytes(b"RIFFold-audio")
    return STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="old-model",
        voice_id="old-voice",
        source_text="Old result.",
        operation_id=operation_id,
        audio_format="wav",
        content_type="audio/wav",
        metadata={},
    )


class _SpeechHost(App[None]):
    def __init__(
        self,
        context: dict[str, object] | None = None,
        *,
        profile_service: TTSProfileService | None = None,
    ) -> None:
        super().__init__()
        self.profile_service = profile_service
        self.screen_under_test = STTSScreen(self)
        if context is not None:
            self.screen_under_test.apply_navigation_context(context)

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)

    async def _ensure_tts_profile_service(self) -> TTSProfileService | None:
        return self.profile_service


class _OpenAIProfileTTSService:
    """Small app-service collaborator; OpenAI profiles never use native catalogs."""

    revision = 7

    def configuration_revision(self, _provider_id: str) -> int:
        return self.revision

    async def require_current_configuration_revision(
        self,
        _provider_id: str,
        expected_revision: int,
    ) -> None:
        assert expected_revision == self.revision

    async def get_native_capability_snapshot(
        self,
        _provider_id: str,
        _exact_voice_model_ids: Iterable[str],
    ) -> TTSNativeCapabilitySnapshot:
        raise AssertionError("OpenAI-compatible profile tests must not use native catalogs")

    async def audio_cpp_guided_dependency_snapshot(
        self,
        _requirement: object,
    ) -> object:
        raise AssertionError("OpenAI-compatible profile tests must not use audio.cpp")


async def _real_openai_profile_service(
    tmp_path: Path,
    *,
    profile_count: int = 32,
) -> tuple[TTSProfileService, TTSProfileRepository, LoadedTTSProfile]:
    repository = TTSProfileRepository(tmp_path / "voice-profiles.sqlite3")
    await repository.open()
    target_id = None
    for index in range(profile_count):
        created = await repository.create_profile(
            TTSProfileDraft(
                display_name=f"Voice {index:02d}",
                provider_id="openai",
                model_id="pocket-tts",
                voice_id=f"alba-{index:02d}",
                response_format="wav",
                speed=1.0,
                options={},
            )
        )
        if index == 24:
            target_id = created.value.profile_id
    service = TTSProfileService(repository, _OpenAIProfileTTSService())
    assert target_id is not None
    return service, repository, await service.get_profile(target_id)


def _valid_openai_artifact(
    tmp_path: Path,
    loaded: LoadedTTSProfile,
    *,
    model_id: str | None = None,
) -> STTSGeneratedAudio:
    profile = loaded.profile
    path = (tmp_path / "profile-sample.wav").resolve()
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(struct.pack("<h", 100) * 1_600)
    selected_model = profile.model_id if model_id is None else model_id
    selection = TTSRequestedSelectionSnapshot(
        provider_id=profile.provider_id,
        model_id=selected_model,
        voice_id=profile.voice_id,
        response_format=profile.response_format,
        speed=profile.speed,
        options=profile.options,
        configuration_revision=_OpenAIProfileTTSService.revision,
    )
    return STTSGeneratedAudio(
        path=path,
        provider_id=profile.provider_id,
        model_id=selected_model,
        voice_id=profile.voice_id,
        source_text="Profile verification sample.",
        operation_id="profile-test-operation",
        audio_format="wav",
        content_type="audio/wav",
        metadata={},
        requested_selection=selection,
    )


async def _wait_until(pilot, predicate, *, timeout: float = 30.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition did not become true")


def _playground_ready(screen: STTSScreen) -> bool:
    """Return whether the dynamically mounted Playground is interactive."""

    return (
        len(screen.query(SpeechPlaygroundPane)) == 1
        and len(screen.query("#audio-play-btn")) == 1
        and len(screen.query("#tts-provider-select SelectOverlay")) == 1
    )


async def _open_real_profile_test(
    screen: STTSScreen,
    pilot,
    target: LoadedTTSProfile,
) -> tuple[int, SpeechPlaygroundPane]:
    target_key = str(target.profile.profile_id)
    await _wait_until(
        pilot,
        lambda: (
            len(screen.query(STTSProfileLibrary)) == 1
            and target_key in screen.query_one(STTSProfileLibrary)._row_availability
        ),
    )
    library = screen.query_one(STTSProfileLibrary)
    table = library.query_one("#stts-profile-table", DataTable)
    target_row = library._rendered_profile_ids.index(target_key)
    assert str(table.get_row_at(target_row)[3]) == "Needs test"
    table.move_cursor(row=target_row, animate=False)
    table.action_select_cursor()
    table.scroll_to(y=max(1, target_row - 2), animate=False)
    table.focus()
    await pilot.pause()
    selected_scroll = table.scroll_offset.y

    library.query_one("#stts-profile-preview-btn", Button).press()
    await _wait_until(pilot, lambda: _playground_ready(screen))
    return selected_scroll, screen.query_one(SpeechPlaygroundPane)


async def _deliver_profile_sample(
    pilot,
    playground: SpeechPlaygroundPane,
    artifact: STTSGeneratedAudio,
    *,
    expect_save: bool,
) -> None:
    playground._generation_operation_id = artifact.operation_id
    playground._generation_complete(artifact)
    if expect_save:
        await _wait_until(
            pilot,
            lambda: not playground.query_one(
                "#audio-save-profile-btn", Button
            ).disabled,
        )
    else:
        await pilot.pause()
        await pilot.pause()
        assert playground.query_one("#audio-save-profile-btn", Button).disabled


def _track_live_reconciliations(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    reconciled_rows: list[str] = []
    original_publish = STTSProfileLibrary.publish_profile_test_availability

    def track(self, loaded, availability) -> None:
        if self._live:
            reconciled_rows.append(str(loaded.profile.profile_id))
        original_publish(self, loaded, availability)

    monkeypatch.setattr(
        STTSProfileLibrary,
        "publish_profile_test_availability",
        track,
    )
    return reconciled_rows


def test_speech_screen_state_keeps_only_bounded_playground_axes() -> None:
    """Fresh-screen restore cannot retain text, unknown keys, or unsafe axes."""

    app = _SpeechHost()
    screen = app.screen_under_test
    screen.restore_state(
        {
            "speech_playground_axes": {
                "tts-provider-select": "audio_cpp",
                "tts-model-select": "safe-model",
                "tts-voice-select": "unsafe\nvoice",
                "tts-speed-input": "1.25",
                "tts-text-input": "private synthesis text",
                "unknown-control": "private provider body",
                "tts-language-select": "x" * 4097,
            }
        }
    )

    assert screen.save_state()["speech_playground_axes"] == {
        "tts-provider-select": "audio_cpp",
        "tts-model-select": "safe-model",
        "tts-speed-input": "1.25",
    }


@pytest.mark.asyncio
async def test_profile_library_navigation_waits_for_deferred_speech_body() -> None:
    app = _SpeechHost({"view": "profiles"})
    screen = app.screen_under_test

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window is not None
                and screen.stts_window.current_view == "profiles"
                and len(screen.query(STTSProfileLibrary)) == 1
            ),
        )


@pytest.mark.asyncio
async def test_profile_library_bundle_service_is_lazy_until_warning_acknowledged() -> (
    None
):
    app = _SpeechHost({"view": "profiles"})
    app._ensure_tts_voice_bundle_service = AsyncMock()  # type: ignore[attr-defined]
    screen = app.screen_under_test

    async with app.run_test(size=(80, 24)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                len(screen.query(STTSProfileLibrary)) == 1
                and len(screen.query("#stts-profile-import-btn")) == 1
            ),
        )
        screen.query_one("#stts-profile-import-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: len(app.screen.query("#bundle-warning-ack")) == 1,
        )

        app._ensure_tts_voice_bundle_service.assert_not_awaited()  # type: ignore[attr-defined]
        app.screen.query_one("#bundle-warning-cancel", Button).press()


@pytest.mark.asyncio
async def test_voice_profiles_returns_to_originating_studio_action() -> None:
    app = _SpeechHost({"view": "settings"})
    screen = app.screen_under_test

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window is not None
                and screen.stts_window.current_view == "settings"
                and len(screen.query("#voice-profiles")) == 1
            ),
        )
        profiles_action = screen.query_one("#voice-profiles", Button)
        profiles_action.focus()
        profiles_action.press()
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window.current_view == "profiles"
                and len(screen.query(STTSProfileLibrary)) == 1
            ),
        )

        back = screen.query_one("#speech-destination-back", Button)
        assert back.label.plain == "Back to previous Speech view"
        await _wait_until(pilot, lambda: not back.disabled)
        back.press()
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window.current_view == "settings"
                and getattr(app.focused, "id", None) == "voice-profiles"
            ),
        )


@pytest.mark.asyncio
async def test_voice_blends_opens_kokoro_tool_and_returns_to_origin() -> None:
    app = _SpeechHost({"view": "settings"})
    screen = app.screen_under_test

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window is not None
                and screen.stts_window.current_view == "settings"
                and len(screen.query("#voice-blends")) == 1
            ),
        )
        blends_action = screen.query_one("#voice-blends", Button)
        blends_action.focus()
        blends_action.press()
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window.current_view == "blends"
                and len(screen.query(VoiceBlendsPane)) == 1
            ),
        )

        pane = screen.query_one(VoiceBlendsPane)
        assert str(pane.query_one("#voice-blends-heading", Static).render()) == (
            "Voice Blends"
        )
        assert "Kokoro only" in str(
            pane.query_one("#voice-blends-scope", Static).render()
        )
        assert not pane.query(STTSProfileLibrary)

        pane.query_one("#speech-destination-back", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window.current_view == "settings"
                and getattr(app.focused, "id", None) == "voice-blends"
            ),
        )


@pytest.mark.asyncio
async def test_main_navigation_clears_voice_tool_origin_before_reopen() -> None:
    app = _SpeechHost({"view": "settings"})
    screen = app.screen_under_test

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(pilot, lambda: len(screen.query("#voice-profiles")) == 1)
        screen.query_one("#voice-profiles", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window.current_view == "profiles"
                and len(screen.query(STTSProfileLibrary)) == 1
                and len(screen.query("#speech-destination-back")) == 1
            ),
        )
        assert screen.stts_window._voice_tool_origin is not None
        assert len(screen.query("#speech-destination-back")) == 1

        screen.query_one("#lab-speech-row-playground", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window.current_view == "playground"
                and len(screen.query(SpeechPlaygroundPane)) == 1
            ),
        )
        assert screen.stts_window._voice_tool_origin is None

        screen.query_one("#lab-speech-row-profiles", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window.current_view == "profiles"
                and len(screen.query(STTSProfileLibrary)) == 1
            ),
        )
        assert not screen.query("#speech-destination-back")


@pytest.mark.asyncio
async def test_voice_tool_back_is_single_flight_and_recovers_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _SpeechHost({"view": "settings"})
    screen = app.screen_under_test

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(pilot, lambda: len(screen.query("#voice-blends")) == 1)
        screen.query_one("#voice-blends", Button).press()
        await _wait_until(
            pilot,
            lambda: screen.stts_window.current_view == "blends",
        )
        window = screen.stts_window
        origin = window._voice_tool_origin
        assert origin is not None
        back = screen.query_one("#speech-destination-back", Button)
        gate = asyncio.Event()
        requests: list[tuple[str, dict[str, object]]] = []

        async def delayed_request(view: str, **kwargs: object) -> bool:
            requests.append((view, kwargs))
            await gate.wait()
            return True

        monkeypatch.setattr(window, "request_view", delayed_request)
        first = SimpleNamespace(stop=Mock())
        second = SimpleNamespace(stop=Mock())
        window.on_speech_destination_back_requested(first)
        window.on_speech_destination_back_requested(second)
        await pilot.pause()

        assert len(requests) == 1
        assert window._voice_tool_back_in_progress is True
        assert back.disabled is True
        gate.set()
        await _wait_until(pilot, lambda: not window._voice_tool_back_in_progress)
        assert window._voice_tool_origin is None

        window._voice_tool_origin = origin
        back.disabled = False

        async def failed_request(_view: str, **_kwargs: object) -> bool:
            return False

        monkeypatch.setattr(window, "request_view", failed_request)
        window.on_speech_destination_back_requested(SimpleNamespace(stop=Mock()))
        await _wait_until(pilot, lambda: not window._voice_tool_back_in_progress)
        assert window._voice_tool_origin == origin
        assert back.disabled is False

        async def raised_request(_view: str, **_kwargs: object) -> bool:
            raise RuntimeError("navigation failed")

        monkeypatch.setattr(window, "request_view", raised_request)
        window.on_speech_destination_back_requested(SimpleNamespace(stop=Mock()))
        await pilot.pause()
        assert window._voice_tool_origin == origin
        assert window._voice_tool_back_in_progress is False
        assert back.disabled is False


@pytest.mark.asyncio
async def test_real_profile_verification_reconciles_on_library_remount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, repository, target = await _real_openai_profile_service(tmp_path)
    reconciled_rows = _track_live_reconciliations(monkeypatch)
    app = _SpeechHost({"view": "profiles"}, profile_service=service)
    screen = app.screen_under_test

    try:
        async with app.run_test(size=(100, 32)) as pilot:
            selected_scroll, playground = await _open_real_profile_test(
                screen,
                pilot,
                target,
            )
            artifact = _valid_openai_artifact(tmp_path, target)
            await _deliver_profile_sample(
                pilot,
                playground,
                artifact,
                expect_save=True,
            )

            screen.stts_window.select_view("profiles")
            await _wait_until(
                pilot,
                lambda: (
                    len(screen.query(STTSProfileLibrary)) == 1
                    and str(target.profile.profile_id)
                    in screen.query_one(STTSProfileLibrary)._row_availability
                    and screen.query_one(STTSProfileLibrary)
                    ._row_availability[str(target.profile.profile_id)]
                    .state
                    == "available"
                ),
            )
            restored = screen.query_one(STTSProfileLibrary)
            restored_table = restored.query_one("#stts-profile-table", DataTable)
            await _wait_until(
                pilot,
                lambda: (
                    getattr(app.focused, "id", None) == "stts-profile-table"
                    and restored_table.scroll_offset.y == selected_scroll
                ),
            )

            assert restored._selected_profile is not None
            assert restored._selected_profile.profile.profile_id == target.profile.profile_id
            assert getattr(app.focused, "id", None) == "stts-profile-table"
            assert restored_table.scroll_offset.y == selected_scroll
            assert reconciled_rows == [str(target.profile.profile_id)]
            assert profile_library_module._profile_test_context_count() == 0
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_profile_return_does_not_steal_focus_after_user_input(
    tmp_path: Path,
) -> None:
    service, repository, target = await _real_openai_profile_service(tmp_path)
    app = _SpeechHost({"view": "profiles"}, profile_service=service)
    screen = app.screen_under_test

    try:
        async with app.run_test(size=(100, 32)) as pilot:
            _scroll, playground = await _open_real_profile_test(
                screen,
                pilot,
                target,
            )
            await _deliver_profile_sample(
                pilot,
                playground,
                _valid_openai_artifact(tmp_path, target),
                expect_save=True,
            )

            screen.stts_window.select_view("profiles")
            await _wait_until(
                pilot,
                lambda: len(screen.query(STTSProfileLibrary)) == 1,
            )
            await pilot.press("tab")
            user_focus = app.focused
            assert screen.stts_window._profile_focus_restore_token is None
            await _wait_until(
                pilot,
                lambda: (
                    str(target.profile.profile_id)
                    in screen.query_one(STTSProfileLibrary)._row_availability
                    and screen.query_one(STTSProfileLibrary)
                    ._row_availability[str(target.profile.profile_id)]
                    .state
                    == "available"
                ),
            )
            await pilot.pause()
            await pilot.pause()

            assert app.focused is user_focus
            assert getattr(app.focused, "id", None) != "stts-profile-table"
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_edited_profile_cannot_publish_stale_verified_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, repository, target = await _real_openai_profile_service(tmp_path)
    reconciled_rows = _track_live_reconciliations(monkeypatch)
    app = _SpeechHost({"view": "profiles"}, profile_service=service)
    screen = app.screen_under_test

    try:
        async with app.run_test(size=(100, 32)) as pilot:
            _scroll, playground = await _open_real_profile_test(
                screen,
                pilot,
                target,
            )
            await _deliver_profile_sample(
                pilot,
                playground,
                _valid_openai_artifact(tmp_path, target),
                expect_save=True,
            )
            profile = target.profile
            updated = await service.update_profile(
                target,
                TTSProfileDraft(
                    display_name="Edited while testing",
                    provider_id=profile.provider_id,
                    model_id=profile.model_id,
                    voice_id=profile.voice_id,
                    response_format=profile.response_format,
                    speed=profile.speed,
                    options=profile.options,
                ),
            )

            screen.stts_window.select_view("profiles")
            await _wait_until(
                pilot,
                lambda: (
                    len(screen.query(STTSProfileLibrary)) == 1
                    and str(profile.profile_id)
                    in screen.query_one(STTSProfileLibrary)._row_availability
                    and screen.query_one(STTSProfileLibrary)
                    ._row_availability[str(profile.profile_id)]
                    .state
                    == "unverified"
                    and screen.stts_window._pending_profile_verification is None
                ),
            )
            restored = screen.query_one(STTSProfileLibrary)
            row = restored._rendered_profile_ids.index(str(profile.profile_id))

            assert restored._selected_profile is not None
            assert restored._selected_profile.profile.profile_id == profile.profile_id
            assert restored._selected_profile.profile.revision == updated.profile.revision
            assert str(
                restored.query_one("#stts-profile-table", DataTable).get_row_at(row)[3]
            ) == "Needs test"
            assert reconciled_rows == []
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_deleted_profile_discards_verified_result_and_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, repository, target = await _real_openai_profile_service(tmp_path)
    reconciled_rows = _track_live_reconciliations(monkeypatch)
    app = _SpeechHost({"view": "profiles"}, profile_service=service)
    screen = app.screen_under_test

    try:
        async with app.run_test(size=(100, 32)) as pilot:
            _scroll, playground = await _open_real_profile_test(
                screen,
                pilot,
                target,
            )
            await _deliver_profile_sample(
                pilot,
                playground,
                _valid_openai_artifact(tmp_path, target),
                expect_save=True,
            )
            await service.delete_profile(target)

            screen.stts_window.select_view("profiles")
            await _wait_until(
                pilot,
                lambda: (
                    len(screen.query(STTSProfileLibrary)) == 1
                    and str(target.profile.profile_id)
                    not in screen.query_one(STTSProfileLibrary)._rendered_profile_ids
                    and len(screen.query_one(STTSProfileLibrary)._row_availability) == 31
                    and screen.stts_window._pending_profile_verification is None
                ),
            )
            restored = screen.query_one(STTSProfileLibrary)

            assert restored._selected_profile is None
            assert all(
                availability.state == "unverified"
                for availability in restored._row_availability.values()
            )
            assert reconciled_rows == []
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_mismatched_sample_never_updates_real_profile_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, repository, target = await _real_openai_profile_service(tmp_path)
    reconciled_rows = _track_live_reconciliations(monkeypatch)
    app = _SpeechHost({"view": "profiles"}, profile_service=service)
    screen = app.screen_under_test

    try:
        async with app.run_test(size=(100, 32)) as pilot:
            _scroll, playground = await _open_real_profile_test(
                screen,
                pilot,
                target,
            )
            await _deliver_profile_sample(
                pilot,
                playground,
                _valid_openai_artifact(
                    tmp_path,
                    target,
                    model_id="different/model",
                ),
                expect_save=False,
            )

            screen.stts_window.select_view("profiles")
            await _wait_until(
                pilot,
                lambda: (
                    len(screen.query(STTSProfileLibrary)) == 1
                    and str(target.profile.profile_id)
                    in screen.query_one(STTSProfileLibrary)._row_availability
                    and screen.query_one(STTSProfileLibrary)
                    ._row_availability[str(target.profile.profile_id)]
                    .state
                    == "unverified"
                ),
            )
            restored = screen.query_one(STTSProfileLibrary)
            row = restored._rendered_profile_ids.index(
                str(target.profile.profile_id)
            )

            assert screen.stts_window._pending_profile_verification is None
            assert restored._selected_profile is not None
            assert restored._selected_profile.profile.profile_id == target.profile.profile_id
            assert str(
                restored.query_one("#stts-profile-table", DataTable).get_row_at(row)[3]
            ) == "Needs test"
            assert reconciled_rows == []
            assert profile_library_module._profile_test_context_count() == 0
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_profile_test_cancel_retires_context_and_restores_library(
    tmp_path: Path,
) -> None:
    service, repository, target = await _real_openai_profile_service(tmp_path)
    app = _SpeechHost({"view": "profiles"}, profile_service=service)
    screen = app.screen_under_test

    try:
        async with app.run_test(size=(100, 32)) as pilot:
            selected_scroll, _playground = await _open_real_profile_test(
                screen,
                pilot,
                target,
            )
            assert profile_library_module._profile_test_context_count() == 1

            screen.stts_window.select_view("profiles")
            await _wait_until(
                pilot,
                lambda: (
                    len(screen.query(STTSProfileLibrary)) == 1
                    and profile_library_module._profile_test_context_count() == 0
                ),
            )
            restored = screen.query_one(STTSProfileLibrary)
            restored_table = restored.query_one("#stts-profile-table", DataTable)
            await _wait_until(
                pilot,
                lambda: restored_table.scroll_offset.y == selected_scroll,
            )

            assert restored._selected_profile is not None
            assert restored._selected_profile.profile.profile_id == target.profile.profile_id
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_stale_profile_completion_retires_context_without_verification(
    tmp_path: Path,
) -> None:
    service, repository, target = await _real_openai_profile_service(tmp_path)
    app = _SpeechHost({"view": "profiles"}, profile_service=service)
    screen = app.screen_under_test

    try:
        async with app.run_test(size=(100, 32)) as pilot:
            _scroll, playground = await _open_real_profile_test(screen, pilot, target)
            profile = target.profile
            await service.update_profile(
                target,
                TTSProfileDraft(
                    display_name="Edited before completion",
                    provider_id=profile.provider_id,
                    model_id=profile.model_id,
                    voice_id=profile.voice_id,
                    response_format=profile.response_format,
                    speed=profile.speed,
                    options=profile.options,
                ),
            )

            await _deliver_profile_sample(
                pilot,
                playground,
                _valid_openai_artifact(tmp_path, target),
                expect_save=False,
            )
            await _wait_until(
                pilot,
                lambda: profile_library_module._profile_test_context_count() == 0,
            )

            assert screen.stts_window._pending_profile_verification is None
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_deleted_profile_completion_retires_context_without_verification(
    tmp_path: Path,
) -> None:
    service, repository, target = await _real_openai_profile_service(tmp_path)
    app = _SpeechHost({"view": "profiles"}, profile_service=service)
    screen = app.screen_under_test

    try:
        async with app.run_test(size=(100, 32)) as pilot:
            _scroll, playground = await _open_real_profile_test(screen, pilot, target)
            await service.delete_profile(target)

            await _deliver_profile_sample(
                pilot,
                playground,
                _valid_openai_artifact(tmp_path, target),
                expect_save=False,
            )
            await _wait_until(
                pilot,
                lambda: profile_library_module._profile_test_context_count() == 0,
            )

            assert screen.stts_window._pending_profile_verification is None
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_profile_test_context_retires_when_screen_unmounts(
    tmp_path: Path,
) -> None:
    service, repository, target = await _real_openai_profile_service(tmp_path)
    app = _SpeechHost({"view": "profiles"}, profile_service=service)
    screen = app.screen_under_test

    try:
        async with app.run_test(size=(100, 32)) as pilot:
            await _open_real_profile_test(screen, pilot, target)
            assert profile_library_module._profile_test_context_count() == 1

        assert profile_library_module._profile_test_context_count() == 0
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_out_of_order_superseded_callback_cannot_consume_newer_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, repository, target = await _real_openai_profile_service(tmp_path)
    app = _SpeechHost({"view": "profiles"}, profile_service=service)
    screen = app.screen_under_test

    try:
        async with app.run_test(size=(100, 32)) as pilot:
            _scroll, playground = await _open_real_profile_test(screen, pilot, target)
            stale_token = playground._profile_test_context_token
            stale_preset = playground._profile_preset
            stale_mount_generation = playground._profile_mount_generation
            registration = profile_library_module._remember_profile_test_context(
                service,
                target,
                stale_preset,
            )

            playground.apply_profile_preset(
                registration.preset,
                context_token=registration.context_token,
            )
            current_mount_generation = playground._profile_mount_generation
            stale_prime = Mock()
            stale_catalog_load = Mock()
            monkeypatch.setattr(
                playground,
                "_prime_profile_preset_controls",
                stale_prime,
            )
            monkeypatch.setattr(
                playground,
                "_load_provider_catalog",
                stale_catalog_load,
            )

            playground._finish_profile_preset_mount(stale_mount_generation)

            stale_prime.assert_not_called()
            stale_catalog_load.assert_not_called()
            assert profile_library_module._profile_test_context_count() == 1
            assert playground._profile_mount_generation == current_mount_generation
            assert playground._profile_test_context_token == registration.context_token
            assert profile_library_module._consume_profile_test_context(
                stale_token,
                stale_preset,
            ) is None
            assert profile_library_module._resolve_profile_test_context(
                registration.context_token,
                registration.preset,
            ) is not None

            screen.stts_window.select_view("profiles")
            await _wait_until(
                pilot,
                lambda: profile_library_module._profile_test_context_count() == 0,
            )
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_exact_playground_preset_survives_deferred_speech_body_mount() -> None:
    preset = _preset()
    app = _SpeechHost({"view": "playground", "profile_preset": preset})
    screen = app.screen_under_test

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                _playground_ready(screen)
                and screen.query_one(SpeechPlaygroundPane)._profile_preset is preset
                and screen.query_one("#tts-provider-select", Select).value
                == "audio_cpp"
            ),
        )


@pytest.mark.asyncio
async def test_exact_preset_applies_to_an_already_open_playground() -> None:
    app = _SpeechHost()
    screen = app.screen_under_test
    preset = _preset()

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: _playground_ready(screen),
        )
        original = screen.query_one(SpeechPlaygroundPane)

        screen.apply_navigation_context(
            {"view": "playground", "profile_preset": preset}
        )
        await _wait_until(
            pilot,
            lambda: (
                len(screen.query(SpeechPlaygroundPane)) == 1
                and screen.query_one(SpeechPlaygroundPane) is original
                and screen.query_one(SpeechPlaygroundPane)._profile_preset is preset
            ),
        )


@pytest.mark.asyncio
async def test_exact_preset_preserves_existing_playground_audio(tmp_path) -> None:
    app = _SpeechHost()
    screen = app.screen_under_test
    artifact = _artifact(tmp_path, "old-complete-operation")
    retire = Mock()

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: _playground_ready(screen),
        )
        playground = screen.query_one(SpeechPlaygroundPane)
        playground._store_delivered_artifact(artifact, announce=False)
        app._stts_handler = SimpleNamespace(retire_playground_generation=retire)

        screen.apply_navigation_context(
            {"view": "playground", "profile_preset": _preset()}
        )
        await _wait_until(
            pilot,
            lambda: playground._profile_preset is not None,
        )

        assert retire.call_count == 0
        assert type(playground.current_audio_artifact) is STTSPlaygroundResultProjection
        assert playground.current_audio_artifact.operation_id == artifact.operation_id
        assert playground.current_audio_file == artifact.path
        assert playground.query_one("#audio-play-btn", Button).disabled is False
        assert playground.query_one("#audio-export-btn", Button).disabled is False
        assert (
            str(playground.query_one("#audio-player-status", Static).renderable)
            == "Ready · WAV"
        )


@pytest.mark.asyncio
async def test_voice_profile_handoff_keeps_mounted_clone_draft_until_cancel() -> None:
    """Browsing the existing library must not unmount or clear setup state."""

    app = _SpeechHost()
    screen = app.screen_under_test
    retained_draft = object()

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(pilot, lambda: _playground_ready(screen))
        playground = screen.query_one(SpeechPlaygroundPane)
        playground._clone_setup_canonical = retained_draft

        playground.post_message(OpenVoiceProfilesRequested())
        await _wait_until(
            pilot, lambda: isinstance(app.screen, VoiceProfilePickerModal)
        )

        assert screen.query_one(SpeechPlaygroundPane) is playground
        assert playground._clone_setup_canonical is retained_draft
        app.screen.query_one("#speech-voice-profile-picker-cancel", Button).press()
        await _wait_until(pilot, lambda: app.screen is screen)

        assert screen.query_one(SpeechPlaygroundPane) is playground
        assert playground._clone_setup_canonical is retained_draft


@pytest.mark.asyncio
async def test_exact_preset_rejects_late_prior_generation_completion(tmp_path) -> None:
    app = _SpeechHost()
    screen = app.screen_under_test
    artifact = _artifact(tmp_path, "old-in-flight-operation")
    retire = Mock()

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: _playground_ready(screen),
        )
        playground = screen.query_one(SpeechPlaygroundPane)
        playground._generation_operation_id = artifact.operation_id
        app._stts_handler = SimpleNamespace(retire_playground_generation=retire)

        screen.apply_navigation_context(
            {"view": "playground", "profile_preset": _preset()}
        )
        await _wait_until(pilot, lambda: retire.call_count == 1)
        playground._generation_complete(artifact)

        assert playground.current_audio_artifact is None
        assert playground.current_audio_file is None
        assert playground.query_one("#audio-play-btn", Button).disabled is True
        assert playground.query_one("#audio-export-btn", Button).disabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent", "focused_id"),
    (
        (SpeechTTSNavigationIntent.CONFIGURE, "tts-provider-select"),
        (SpeechTTSNavigationIntent.TEST, "tts-test-connection-btn"),
        (SpeechTTSNavigationIntent.REFRESH_MODELS, "tts-refresh-catalog-btn"),
        (SpeechTTSNavigationIntent.REFRESH_VOICES, "tts-voice-select"),
    ),
)
async def test_bounded_lab_navigation_restores_provider_and_focus_without_action(
    monkeypatch,
    intent: SpeechTTSNavigationIntent,
    focused_id: str,
) -> None:
    service = FakeTTSService()

    catalog_calls: list[bool] = []
    original_load = SpeechPlaygroundPane._load_provider_catalog
    generate = Mock()

    def record_catalog_load(self, *args, **kwargs) -> None:
        catalog_calls.append(bool(kwargs.get("refresh", False)))
        original_load(self, *args, **kwargs)

    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        record_catalog_load,
    )
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
    monkeypatch.setattr(SpeechPlaygroundPane, "_generate_tts", generate)
    app = _SpeechHost(
        {
            "view": "playground",
            "provider": "audio_cpp",
            "intent": intent.value,
        }
    )
    screen = app.screen_under_test

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                _playground_ready(screen)
                and screen.query_one("#tts-provider-select", Select).value
                == "audio_cpp"
                and getattr(app.focused, "id", None) == focused_id
            ),
        )

    generate.assert_not_called()
    assert True not in catalog_calls


@pytest.mark.asyncio
async def test_test_connection_and_refresh_are_distinct_explicit_actions(
    monkeypatch,
) -> None:
    service = FakeTTSService()

    async def start_and_test_audio_cpp() -> object:
        return await service.get_catalog("audio_cpp", refresh=True)

    service.start_and_test_audio_cpp = start_and_test_audio_cpp  # type: ignore[attr-defined]
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
    app = _SpeechHost(
        {
            "view": "playground",
            "provider": "audio_cpp",
            "intent": "test",
        }
    )
    screen = app.screen_under_test

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                _playground_ready(screen)
                and service.catalog_calls
                and all(not refresh for _provider, refresh in service.catalog_calls)
            ),
        )
        initial_refreshes = sum(refresh for _provider, refresh in service.catalog_calls)

        screen.query_one("#tts-test-connection-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                sum(refresh for _provider, refresh in service.catalog_calls)
                == initial_refreshes + 1
            ),
        )

        refresh_button = screen.query_one("#tts-refresh-catalog-btn", Button)
        await _wait_until(pilot, lambda: not refresh_button.disabled)
        refresh_button.press()
        await _wait_until(
            pilot,
            lambda: (
                sum(refresh for _provider, refresh in service.catalog_calls)
                == initial_refreshes + 2
            ),
        )

    assert [call for call in service.catalog_calls if call[1]][-2:] == [
        ("audio_cpp", True),
        ("audio_cpp", True),
    ]


@pytest.mark.parametrize(
    "context",
    [
        {},
        {"view": 1},
        {"view": "unknown"},
        {"view": "playground", "profile_preset": object()},
        {"view": "profiles", "profile_preset": _preset()},
        {
            "view": "playground",
            "provider": "audio_cpp",
            "intent": "test",
            "text": "private",
        },
        {
            "view": "playground",
            "provider": "audio_cpp",
            "intent": "generate",
        },
    ],
)
def test_malformed_speech_navigation_context_is_rejected(
    context: dict[str, object],
) -> None:
    screen = STTSScreen(App())

    screen.apply_navigation_context(context)

    assert screen._pending_navigation_context is None
