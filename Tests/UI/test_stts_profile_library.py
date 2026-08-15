from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
import wave
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.widget import Widget
from textual.widgets import Button, Checkbox, DataTable, Input, Select, Static, TextArea
from textual.worker import WorkerFailed

from Tests.UI.background_signals import wait_for_background_signal
from tldw_chatbook.TTS import (
    LoadedTTSProfile,
    ProfileRepositoryError,
    ProfileServiceError,
    STTSGeneratedAudio,
    TTSCloneRecipeRequirement,
    TTSCloneReferenceSummary,
    TTSGenerationProfile,
    TTSPlaygroundSelectionPreset,
    TTSProfileAvailability,
    TTSProfileAvailabilitySnapshot,
    TTSProfileDraft,
    TTSProfilePageSnapshot,
    TTSRequestedSelectionSnapshot,
    TTSVoiceBundleHandle,
    TTSVoiceBundleImportChoice,
    TTSVoiceBundleImportResult,
    TTSVoiceBundleReview,
)
from tldw_chatbook.TTS.profile_service import TTSProfileDependencyProjection
from tldw_chatbook.TTS.profile_portability import (
    PortableTTSProfile,
    portable_profile_payload,
)
from tldw_chatbook.UI import STTS_Window as stts_window_module
from tldw_chatbook.UI import stts_profile_library as profile_library_module
from tldw_chatbook.UI.Dictation_Window_Improved import ImprovedDictationWindow
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_settings_mixin import SpeechSettingsMixin
from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane
from tldw_chatbook.UI.stts_profile_library import (
    PROFILE_STORE_UNAVAILABLE_COPY,
    STTSProfileLibrary,
    TTSProfileExportChoiceModal,
    TTSVoiceBundleConsentModal,
    VoiceBundleActionProjection,
    voice_bundle_export_actions,
    voice_bundle_import_choice,
    voice_bundle_review_action,
)
from tldw_chatbook.UI.tts_profile_recovery import dependency_recovery_actions
from tldw_chatbook.UI.STTS_Window import (
    AudioBookGenerationWidget,
    STTSWindow,
)


def test_audiobook_narrator_separator_preserves_voice_and_blend_ids() -> None:
    class _NarratorSelect:
        id = "narrator-voice-select"
        _options = [
            ("Bella (US Female)", "af_bella"),
            ("──── Voice Blends ────", "_separator"),
            ("Duet", "blend:duet"),
        ]

        def __init__(self, value: object) -> None:
            self.value = value

    widget = AudioBookGenerationWidget()
    narrator = _NarratorSelect("af_bella")

    AudioBookGenerationWidget.on_audiobook_selects_changed(
        widget,
        SimpleNamespace(select=narrator, value="af_bella"),
    )
    narrator.value = "_separator"
    AudioBookGenerationWidget.on_audiobook_selects_changed(
        widget,
        SimpleNamespace(select=narrator, value="_separator"),
    )
    assert narrator.value == "af_bella"
    assert narrator.value != "Bella (US Female)"

    narrator.value = "blend:duet"
    AudioBookGenerationWidget.on_audiobook_selects_changed(
        widget,
        SimpleNamespace(select=narrator, value="blend:duet"),
    )
    assert narrator.value == "blend:duet"

    fresh_widget = AudioBookGenerationWidget()
    fresh_narrator = _NarratorSelect("_separator")
    AudioBookGenerationWidget.on_audiobook_selects_changed(
        fresh_widget,
        SimpleNamespace(select=fresh_narrator, value="_separator"),
    )
    assert fresh_narrator.value in {Select.BLANK, None}


def _profile(index: int) -> TTSGenerationProfile:
    timestamp = datetime(2026, 7, 27, tzinfo=UTC)
    display_name = f"Voice {index:02d}"
    return TTSGenerationProfile(
        profile_id=UUID(int=index + 1),
        display_name=display_name,
        normalized_name=display_name.casefold(),
        provider_id="audio_cpp",
        model_id=f"model/{index}",
        voice_id=f"voice/{index}",
        response_format="wav",
        speed=1.0,
        options={},
        revision=1,
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_profile_test_context_consume_is_atomic_and_idempotent() -> None:
    profile_library_module._PROFILE_TEST_CONTEXTS.clear()
    service = _ControlledProfileService()
    loaded = LoadedTTSProfile(repository_generation=3, profile=_profile(0))
    registration = profile_library_module._remember_profile_test_context(
        service,
        loaded,
        TTSPlaygroundSelectionPreset(
            provider_id=loaded.profile.provider_id,
            model_id=loaded.profile.model_id,
            voice_id=loaded.profile.voice_id,
            response_format=loaded.profile.response_format,
            speed=loaded.profile.speed,
            options=loaded.profile.options,
            availability="unverified",
        ),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        consumed = list(
            executor.map(
                lambda _index: profile_library_module._consume_profile_test_context(
                    registration.context_token,
                    registration.preset,
                ),
                range(2),
            )
        )

    assert sum(context is not None for context in consumed) == 1
    assert profile_library_module._profile_test_context_count() == 0


def test_stale_profile_test_token_cannot_consume_newer_same_profile_context() -> None:
    profile_library_module._PROFILE_TEST_CONTEXTS.clear()
    service = _ControlledProfileService()
    loaded = LoadedTTSProfile(repository_generation=3, profile=_profile(0))
    preset = TTSPlaygroundSelectionPreset(
        provider_id=loaded.profile.provider_id,
        model_id=loaded.profile.model_id,
        voice_id=loaded.profile.voice_id,
        response_format=loaded.profile.response_format,
        speed=loaded.profile.speed,
        options=loaded.profile.options,
        availability="unverified",
    )
    stale = profile_library_module._remember_profile_test_context(
        service,
        loaded,
        preset,
    )
    current = profile_library_module._remember_profile_test_context(
        service,
        loaded,
        preset,
    )

    assert profile_library_module._retire_profile_test_context(
        stale.context_token
    )
    assert profile_library_module._resolve_profile_test_context(
        current.context_token,
        current.preset,
    ) is not None
    assert not profile_library_module._retire_profile_test_context(
        stale.context_token
    )
    assert profile_library_module._consume_profile_test_context(
        current.context_token,
        current.preset,
    ) is not None
    assert profile_library_module._profile_test_context_count() == 0


class _ControlledProfileService:
    def __init__(
        self,
        profiles: tuple[TTSGenerationProfile, ...] = (),
        *,
        total: int | None = None,
    ) -> None:
        self.page = TTSProfilePageSnapshot(
            repository_generation=3,
            profiles=profiles,
            total=len(profiles) if total is None else total,
        )
        self.list_calls: list[tuple[str | None, int]] = []
        self.availability_calls: list[TTSProfilePageSnapshot] = []
        self.availability_future: (
            asyncio.Future[TTSProfileAvailabilitySnapshot] | None
        ) = None

    async def list_profiles(
        self,
        *,
        search: str | None = None,
        offset: int = 0,
    ) -> TTSProfilePageSnapshot:
        self.list_calls.append((search, offset))
        return self.page

    async def observe_availability(
        self,
        page: TTSProfilePageSnapshot,
    ) -> TTSProfileAvailabilitySnapshot:
        self.availability_calls.append(page)
        if self.availability_future is None:
            self.availability_future = asyncio.get_running_loop().create_future()
        return await self.availability_future


class _STTSHost(App[None]):
    def __init__(self, service: object | None) -> None:
        super().__init__()
        self.profile_service = service
        self.profile_service_requests = 0

    def compose(self) -> ComposeResult:
        yield STTSWindow(self)

    async def _ensure_tts_profile_service(self) -> object | None:
        self.profile_service_requests += 1
        return self.profile_service


class _PipelineProfileService:
    def __init__(self) -> None:
        self.list_calls: list[tuple[str | None, int]] = []
        self.list_futures: list[asyncio.Future[TTSProfilePageSnapshot]] = []
        self.availability_calls: list[TTSProfilePageSnapshot] = []
        self.availability_futures: list[
            asyncio.Future[TTSProfileAvailabilitySnapshot]
        ] = []
        self.active_list_calls = 0
        self.maximum_active_list_calls = 0
        self.cancelled_list_calls = 0
        self.cancelled_availability_calls = 0
        self.preview_preset_calls: list[
            tuple[LoadedTTSProfile, TTSProfileAvailability]
        ] = []

    async def list_profiles(
        self,
        *,
        search: str | None = None,
        offset: int = 0,
    ) -> TTSProfilePageSnapshot:
        self.list_calls.append((search, offset))
        future = asyncio.get_running_loop().create_future()
        self.list_futures.append(future)
        self.active_list_calls += 1
        self.maximum_active_list_calls = max(
            self.maximum_active_list_calls,
            self.active_list_calls,
        )
        try:
            return await future
        except asyncio.CancelledError:
            self.cancelled_list_calls += 1
            raise
        finally:
            self.active_list_calls -= 1

    async def observe_availability(
        self,
        page: TTSProfilePageSnapshot,
    ) -> TTSProfileAvailabilitySnapshot:
        self.availability_calls.append(page)
        future = asyncio.get_running_loop().create_future()
        self.availability_futures.append(future)
        try:
            return await future
        except asyncio.CancelledError:
            self.cancelled_availability_calls += 1
            raise

    def preview_preset(
        self,
        loaded: LoadedTTSProfile,
        availability: TTSProfileAvailability,
    ) -> TTSPlaygroundSelectionPreset:
        self.preview_preset_calls.append((loaded, availability))
        profile = loaded.profile
        return TTSPlaygroundSelectionPreset(
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
            availability=availability.state,
        )


class _StoreFailureProfileService:
    def __init__(self, code: str) -> None:
        self.code = code
        self.list_calls: list[tuple[str | None, int]] = []

    async def list_profiles(
        self,
        *,
        search: str | None = None,
        offset: int = 0,
    ) -> TTSProfilePageSnapshot:
        self.list_calls.append((search, offset))
        raise ProfileRepositoryError(self.code)

    async def observe_availability(
        self,
        page: TTSProfilePageSnapshot,
    ) -> TTSProfileAvailabilitySnapshot:
        raise AssertionError("availability must not run after store failure")


class _CancellationResistantAvailabilityService(_PipelineProfileService):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_started: list[asyncio.Event] = []
        self.cleanup_releases: list[asyncio.Event] = []
        self.cleanup_settled_by_unmount: list[int] = []

    async def observe_availability(
        self,
        page: TTSProfilePageSnapshot,
    ) -> TTSProfileAvailabilitySnapshot:
        self.availability_calls.append(page)
        future = asyncio.get_running_loop().create_future()
        self.availability_futures.append(future)
        started = asyncio.Event()
        release = asyncio.Event()
        self.cleanup_started.append(started)
        self.cleanup_releases.append(release)
        call_index = len(self.availability_futures) - 1
        try:
            return await future
        except asyncio.CancelledError:
            self.cancelled_availability_calls += 1
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                self.cleanup_settled_by_unmount.append(call_index)
            return _availability(
                page,
                state="unavailable",
                configuration_revision=99,
                catalog_revision=99,
            )


class _ActionProfileService:
    def __init__(
        self,
        profile: TTSGenerationProfile,
        *,
        availability_state: str = "available",
        availability_recovery: str | None = None,
        dependency: TTSProfileDependencyProjection | None = None,
    ) -> None:
        self.page = _page(profile, generation=11)
        self.availability_state = availability_state
        self.availability_recovery = availability_recovery
        self.dependency = dependency
        self.list_calls: list[tuple[str | None, int]] = []
        self.availability_calls: list[TTSProfilePageSnapshot] = []
        self.create_calls: list[tuple[str, STTSGeneratedAudio]] = []
        self.update_calls: list[tuple[LoadedTTSProfile, TTSProfileDraft]] = []
        self.duplicate_calls: list[tuple[LoadedTTSProfile, str]] = []
        self.assignment_count_calls: list[LoadedTTSProfile] = []
        self.delete_calls: list[LoadedTTSProfile] = []
        self.preview_preset_calls: list[
            tuple[LoadedTTSProfile, TTSProfileAvailability]
        ] = []
        self.sample_evidence_calls: list[
            tuple[LoadedTTSProfile, STTSGeneratedAudio]
        ] = []
        self.assignment_total = 0
        self.create_error: BaseException | None = None
        self.update_error: BaseException | None = None
        self.duplicate_error: BaseException | None = None
        self.delete_error: BaseException | None = None
        self.created_result = LoadedTTSProfile(
            repository_generation=11,
            profile=replace(profile, profile_id=uuid4()),
        )
        self.updated_result = LoadedTTSProfile(
            repository_generation=11,
            profile=replace(profile, revision=profile.revision + 1),
        )
        self.duplicate_result = LoadedTTSProfile(
            repository_generation=11,
            profile=replace(profile, profile_id=uuid4()),
        )

    async def list_profiles(
        self,
        *,
        search: str | None = None,
        offset: int = 0,
    ) -> TTSProfilePageSnapshot:
        self.list_calls.append((search, offset))
        return self.page

    async def observe_availability(
        self,
        page: TTSProfilePageSnapshot,
    ) -> TTSProfileAvailabilitySnapshot:
        self.availability_calls.append(page)
        return _availability(
            page,
            state=self.availability_state,
            recovery_action=self.availability_recovery,
            dependency=self.dependency,
        )

    async def create_from_artifact(
        self,
        display_name: str,
        artifact: STTSGeneratedAudio,
    ) -> LoadedTTSProfile:
        self.create_calls.append((display_name, artifact))
        if self.create_error is not None:
            raise self.create_error
        return self.created_result

    async def update_profile(
        self,
        loaded: LoadedTTSProfile,
        draft: TTSProfileDraft,
    ) -> LoadedTTSProfile:
        self.update_calls.append((loaded, draft))
        if self.update_error is not None:
            raise self.update_error
        return self.updated_result

    async def duplicate_profile(
        self,
        loaded: LoadedTTSProfile,
        display_name: str,
    ) -> LoadedTTSProfile:
        self.duplicate_calls.append((loaded, display_name))
        if self.duplicate_error is not None:
            raise self.duplicate_error
        return self.duplicate_result

    async def assignment_count(self, loaded: LoadedTTSProfile) -> int:
        self.assignment_count_calls.append(loaded)
        return self.assignment_total

    async def delete_profile(self, loaded: LoadedTTSProfile) -> None:
        self.delete_calls.append(loaded)
        if self.delete_error is not None:
            raise self.delete_error

    def preview_preset(
        self,
        loaded: LoadedTTSProfile,
        availability: TTSProfileAvailability,
    ) -> TTSPlaygroundSelectionPreset:
        self.preview_preset_calls.append((loaded, availability))
        profile = loaded.profile
        return TTSPlaygroundSelectionPreset(
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
            availability=availability.state,
        )

    def record_sample_evidence(
        self,
        loaded: LoadedTTSProfile,
        artifact: STTSGeneratedAudio,
    ) -> None:
        self.sample_evidence_calls.append((loaded, artifact))
        selection = artifact.requested_selection
        profile = loaded.profile
        if (
            type(selection) is TTSRequestedSelectionSnapshot
            and artifact.path.is_file()
            and selection.provider_id == profile.provider_id
            and selection.model_id == profile.model_id
            and selection.voice_id == profile.voice_id
            and selection.response_format == profile.response_format
            and selection.speed == profile.speed
            and dict(selection.options) == dict(profile.options)
        ):
            self.availability_state = "available"
            self.availability_recovery = "none"


class _PendingAvailabilityActionProfileService(_ActionProfileService):
    def __init__(self, profile: TTSGenerationProfile) -> None:
        super().__init__(profile)
        self.availability_future: asyncio.Future[Any] | None = None

    async def observe_availability(
        self,
        page: TTSProfilePageSnapshot,
    ) -> TTSProfileAvailabilitySnapshot:
        self.availability_calls.append(page)
        if self.availability_future is None:
            self.availability_future = asyncio.get_running_loop().create_future()
        return await self.availability_future


class _ActionHost(_STTSHost):
    def __init__(self, service: object | None) -> None:
        super().__init__(service)
        self.preview_messages: list[object] = []
        self.nav_routes: list[str] = []
        self.nav_contexts: list[dict[str, object]] = []

    def on_profile_preview_requested(self, message: object) -> None:
        self.preview_messages.append(message)

    def on_navigate_to_screen(self, message: object) -> None:
        self.nav_routes.append(str(getattr(message, "screen_name", "")))
        self.nav_contexts.append(dict(getattr(message, "screen_context", {}) or {}))


def _page(
    profile: TTSGenerationProfile,
    *,
    generation: int,
    total: int = 1,
) -> TTSProfilePageSnapshot:
    return TTSProfilePageSnapshot(
        repository_generation=generation,
        profiles=(profile,),
        total=total,
    )


def _availability(
    page: TTSProfilePageSnapshot,
    *,
    state: str = "available",
    configuration_revision: int = 1,
    catalog_revision: int | None = 1,
    recovery_action: str | None = None,
    dependency: TTSProfileDependencyProjection | None = None,
) -> TTSProfileAvailabilitySnapshot:
    recovery = (
        recovery_action
        or {
            "available": "none",
            "unavailable": "edit",
            "unverified": "refresh",
        }[state]
    )
    return TTSProfileAvailabilitySnapshot(
        repository_generation=page.repository_generation,
        configuration_revision=configuration_revision,
        catalog_revision=catalog_revision,
        profiles=tuple(
            TTSProfileAvailability(
                profile_id=profile.profile_id,
                state=state,  # type: ignore[arg-type]
                recovery_action=recovery,  # type: ignore[arg-type]
                dependency=dependency or TTSProfileDependencyProjection(),
            )
            for profile in page.profiles
        ),
    )


@pytest.mark.parametrize(
    ("dependency", "expected"),
    (
        (
            TTSProfileDependencyProjection(
                reason="recipe_missing",
                display="Needs compatible model",
                action="open_audio_cpp_settings",
            ),
            ("open_audio_cpp_settings",),
        ),
        (
            TTSProfileDependencyProjection(
                reason="recipe_pending_apply",
                display="Compatible model saved; apply settings",
                action="open_speech_lab_apply",
            ),
            ("open_speech_lab_apply",),
        ),
        (
            TTSProfileDependencyProjection(
                advisory="recipe_provenance_unavailable",
                advisory_display="Recipe provenance unavailable",
                advisory_action="generate_new_profile",
            ),
            ("generate_new_profile",),
        ),
        (
            TTSProfileDependencyProjection(
                reason="recipe_mismatch",
                display="Needs compatible model",
                action="open_audio_cpp_settings",
                advisory="recipe_provenance_unavailable",
                advisory_display="Recipe provenance unavailable",
                advisory_action="generate_new_profile",
            ),
            ("open_audio_cpp_settings", "generate_new_profile"),
        ),
    ),
)
def test_dependency_recovery_projection_preserves_blocker_then_advisory_truth(
    dependency: TTSProfileDependencyProjection,
    expected: tuple[str, ...],
) -> None:
    actions = dependency_recovery_actions(dependency)

    assert tuple(action.operation for action in actions) == expected
    assert all(action.label and action.tooltip for action in actions)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((80, 24), (100, 30)))
@pytest.mark.parametrize(
    ("dependency", "button_id", "expected_route", "expects_preview"),
    (
        (
            TTSProfileDependencyProjection(
                reason="recipe_missing",
                display="Needs compatible model",
                action="open_audio_cpp_settings",
            ),
            "stts-profile-dependency-primary-btn",
            "settings",
            False,
        ),
        (
            TTSProfileDependencyProjection(
                reason="recipe_pending_apply",
                display="Compatible model saved; apply settings",
                action="open_speech_lab_apply",
            ),
            "stts-profile-dependency-primary-btn",
            "",
            True,
        ),
        (
            TTSProfileDependencyProjection(
                advisory="recipe_provenance_unavailable",
                advisory_display="Recipe provenance unavailable",
                advisory_action="generate_new_profile",
            ),
            "stts-profile-dependency-advisory-btn",
            "",
            True,
        ),
    ),
)
async def test_profile_library_routes_projected_dependency_recovery_exactly(
    size: tuple[int, int],
    dependency: TTSProfileDependencyProjection,
    button_id: str,
    expected_route: str,
    expects_preview: bool,
) -> None:
    profile = _profile(0)
    service = _ActionProfileService(profile, dependency=dependency)
    app = _ActionHost(service)

    async with app.run_test(size=size) as pilot:
        library, _loaded = await _select_action_profile(app, pilot)
        actions = dependency_recovery_actions(dependency)
        projected = actions[-1] if "advisory" in button_id else actions[0]
        button = app.query_one(f"#{button_id}", Button)
        assert str(button.label) == projected.label
        assert button.tooltip == projected.tooltip
        assert button.region.width > 0
        assert button.region.height == 1
        button.press()
        await pilot.pause()

        assert bool(app.preview_messages) is expects_preview
        assert app.nav_routes == ([expected_route] if expected_route else [])
        if expected_route == "settings":
            assert app.nav_contexts[-1]["category"] == "speech-tts"
        assert library._selected_profile is not None


def _table_cell(table: DataTable[Any], row: int, column: int) -> str:
    return str(table.get_row_at(row)[column])


def _playable_profile_artifact(
    tmp_path: Path,
    profile: TTSGenerationProfile,
    *,
    model_id: str | None = None,
    operation_id: str = "profile-test-operation",
) -> STTSGeneratedAudio:
    path = tmp_path / f"{operation_id}.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 160)
    requested_model = profile.model_id if model_id is None else model_id
    selection = TTSRequestedSelectionSnapshot(
        provider_id=profile.provider_id,
        model_id=requested_model,
        voice_id=profile.voice_id,
        response_format=profile.response_format,
        speed=profile.speed,
        options=profile.options,
        configuration_revision=1,
    )
    return STTSGeneratedAudio(
        path=path,
        provider_id=profile.provider_id,
        model_id=requested_model,
        voice_id=profile.voice_id,
        source_text="Profile verification sample",
        operation_id=operation_id,
        audio_format=profile.response_format,
        content_type="audio/wav",
        requested_selection=selection,
    )


def _visible_content_rows(widget: Widget) -> tuple[str, ...]:
    strips = widget.screen._compositor.render_strips()
    region = widget.content_region
    return tuple(
        "".join(segment.text for segment in strips[y])[
            region.x : region.x + region.width
        ].strip()
        for y in range(region.y, region.y + region.height)
    )


def _relative_luminance(color: Any) -> float:
    triplet = color.get_truecolor()

    def channel(value: int) -> float:
        component = value / 255
        return (
            component / 12.92
            if component <= 0.04045
            else ((component + 0.055) / 1.055) ** 2.4
        )

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


def _painted_contrast(first: Any, second: Any) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_style_of_text(app: App[Any], region: Any, needle: str) -> Any:
    strips = list(app.screen._compositor.render_strips())
    for y in range(region.y, region.y + region.height):
        if y >= len(strips):
            break
        segments = list(strips[y]._segments)
        row_text = "".join(segment.text for segment in segments)
        index = row_text.find(needle)
        if index < 0:
            continue
        x = 0
        for segment in segments:
            if x + len(segment.text) > index:
                return segment.style
            x += len(segment.text)
    return None


_BUNDLED_CSS = str(
    Path(profile_library_module.__file__).parent.parent
    / "css"
    / "tldw_cli_modular.tcss"
)


class _DisabledImportContrastHost(App[None]):
    CSS_PATH = _BUNDLED_CSS

    def compose(self) -> ComposeResult:
        async def no_service() -> None:
            return None

        yield STTSProfileLibrary(
            no_service,
            voice_bundle_service_loader=no_service,
            bundle_platform_supported=False,
        )


class _DisabledPortabilityContrastHost(App[None]):
    CSS_PATH = _BUNDLED_CSS

    def compose(self) -> ComposeResult:
        with Vertical(classes="stts-portability-dialog"):
            with Horizontal(classes="stts-portability-actions"):
                yield Button(
                    "Export portable voice bundle",
                    id="stts-export-choice-bundle",
                    disabled=True,
                )


@pytest.mark.parametrize(
    "theme",
    ("textual-dark", "textual-light", "tokyo-night", "monokai", "dracula"),
)
@pytest.mark.parametrize(
    ("host_type", "selector", "needle"),
    (
        (
            _DisabledImportContrastHost,
            "#stts-profile-import-btn",
            "Import bundle",
        ),
        (
            _DisabledPortabilityContrastHost,
            "#stts-export-choice-bundle",
            "Export portable voice bundle",
        ),
    ),
)
async def test_portability_disabled_actions_paint_at_three_to_one_across_themes(
    theme: str,
    host_type: type[App[None]],
    selector: str,
    needle: str,
) -> None:
    app = host_type()
    app.theme = theme
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        button = app.query_one(selector, Button)
        assert button.disabled is True
        assert button.styles.opacity == 1.0
        painted = _painted_style_of_text(app, button.region, needle)
        assert painted is not None
        assert painted.color is not None and painted.bgcolor is not None
        ratio = _painted_contrast(painted.color, painted.bgcolor)
        assert ratio >= 3.0, f"{selector} is {ratio:.2f}:1 under {theme}"


def _status_copy(app: App[Any]) -> str:
    return str(app.query_one("#stts-profile-status-copy", Static).render())


def _identifier_copy(app: App[Any]) -> str:
    identifiers = app.query_one("#stts-profile-identifiers")
    return str(identifiers.text)  # type: ignore[attr-defined]


def _artifact() -> STTSGeneratedAudio:
    requested = TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id="artifact/model",
        voice_id="artifact/voice",
        response_format="wav",
        speed=1.0,
        options={},
        configuration_revision=7,
    )
    return STTSGeneratedAudio(
        path=Path("artifact.wav"),
        provider_id="audio_cpp",
        model_id="artifact/model",
        voice_id="artifact/voice",
        source_text="must never enter profile UI copy",
        operation_id="operation",
        audio_format="wav",
        content_type="audio/wav",
        requested_selection=requested,
    )


async def _open_stts_view(
    app: App[Any],
    pilot: Any,
    view: str,
) -> None:
    """Open an STTS body view without depending on the screen-owned Lab rail."""
    app.query_one(STTSWindow).current_view = view
    await pilot.pause()


async def _select_action_profile(
    app: _ActionHost,
    pilot: Any,
) -> tuple[STTSProfileLibrary, LoadedTTSProfile]:
    await _open_stts_view(app, pilot, "profiles")
    await _wait_until(
        pilot,
        lambda: app.query_one("#stts-profile-table", DataTable).row_count == 1,
    )
    table = app.query_one("#stts-profile-table", DataTable)
    table.move_cursor(row=0)
    table.action_select_cursor()
    await pilot.pause()
    library = app.query_one(STTSProfileLibrary)
    selected = library._selected_profile
    assert selected is not None
    return library, selected


async def _wait_until(
    pilot: Any,
    predicate: Any,
    *,
    attempts: int = 100,
) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition did not become true")


def _playground_is_mounted(app: App[Any]) -> bool:
    """Return whether the async view replacement mounted strict descendants."""
    try:
        window = app.query_one(STTSWindow)
        playground = app.query_one(SpeechPlaygroundPane)
        playground.query_one("#tts-provider-select", Select).query_one(
            "SelectOverlay"
        )
    except QueryError:
        return False
    return (
        window.current_view == "playground"
        and window._pending_playground_preset is None
    )


@pytest.mark.asyncio
async def test_voice_profiles_view_mounts_focused_library_without_hiding_other_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OptionalAudioBookWidget(Widget):
        pass

    monkeypatch.setitem(
        sys.modules,
        "tldw_chatbook.Widgets.TTS.chapter_editor_widget",
        SimpleNamespace(ChapterEditorWidget=_OptionalAudioBookWidget),
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_chatbook.Widgets.TTS.character_voice_widget",
        SimpleNamespace(CharacterVoiceWidget=_OptionalAudioBookWidget),
    )
    service = _ControlledProfileService()
    app = _STTSHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        await pilot.pause()
        assert isinstance(
            app.query_one(".stts-content").children[0],
            SpeechPlaygroundPane,
        )
        assert app.query_one("#tts-generate-btn", Button)

        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: bool(service.list_calls))
        assert app.query_one(STTSProfileLibrary)

        await _open_stts_view(app, pilot, "settings")
        assert isinstance(
            app.query_one(".stts-content").children[0],
            SpeechSettingsPane,
        )

        await _open_stts_view(app, pilot, "audiobook")
        assert isinstance(
            app.query_one(".stts-content").children[0],
            AudioBookGenerationWidget,
        )

        await _open_stts_view(app, pilot, "dictation")
        assert isinstance(
            app.query_one(".stts-content").children[0],
            ImprovedDictationWindow,
        )

        await _open_stts_view(app, pilot, "playground")
        assert isinstance(
            app.query_one(".stts-content").children[0],
            SpeechPlaygroundPane,
        )
        assert app.query_one("#tts-generate-btn", Button)


@pytest.mark.asyncio
async def test_audiobook_kokoro_blend_group_is_not_a_keyboard_select_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _OptionalAudioBookWidget(Widget):
        pass

    class _AudioBookHost(App[None]):
        def compose(self) -> ComposeResult:
            yield AudioBookGenerationWidget()

    monkeypatch.setitem(
        sys.modules,
        "tldw_chatbook.Widgets.TTS.chapter_editor_widget",
        SimpleNamespace(ChapterEditorWidget=_OptionalAudioBookWidget),
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_chatbook.Widgets.TTS.character_voice_widget",
        SimpleNamespace(CharacterVoiceWidget=_OptionalAudioBookWidget),
    )
    blend_file = tmp_path / "voice-blends.json"
    blend_file.write_text(
        json.dumps({"duet": {"description": "Two voices"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(stts_window_module, "kokoro_ui_blend_file", lambda: blend_file)
    app = _AudioBookHost()

    async with app.run_test(size=(120, 48)) as pilot:
        widget = app.query_one(AudioBookGenerationWidget)
        # Let `_initialize_audiobook_defaults` (armed via `set_timer(0.1, ...)`
        # on mount) settle first -- since task-15772 fixed the
        # audiobook-provider-select tuple order, this now actually succeeds
        # and cascades into `_update_voice_options("openai")`. Without
        # waiting for it, that cascade could otherwise land *after* the
        # kokoro setup below (depending on how much the subsequent
        # `pilot.press` calls advance the clock) and wipe out this test's
        # kokoro voice options -- exactly the race a real user would never
        # hit, since they can't interact with the widget before it finishes
        # mounting.
        #
        # Poll on the actual condition rather than a fixed sleep: a fixed
        # `pilot.pause(0.15)` races a *real* wall-clock 0.1s timer plus a
        # message-pump round trip plus a reactive-watcher cascade against a
        # hardcoded 0.05s margin -- comfortable under normal load, but a
        # genuine flake risk under a contended runner (task-15772 review
        # round 2). Bounded poll instead, so this settles as soon as it
        # actually does and fails loudly (not silently, late) if it never
        # does.
        provider_select = app.query_one("#audiobook-provider-select", Select)
        for _ in range(100):
            if provider_select.value == "openai":
                break
            await pilot.pause()
        else:
            raise AssertionError("mount-time provider default never settled")

        widget._update_voice_options("kokoro")
        narrator = app.query_one("#narrator-voice-select", Select)
        option_values = tuple(value for _label, value in narrator._options)
        assert "_separator" not in option_values
        assert "blend:duet" in option_values
        grouping = app.query_one("#audiobook-voice-blends-label", Static)
        assert grouping.display
        assert str(grouping.render()) == "Voice Blends"

        narrator.focus()
        await pilot.press("enter")
        await pilot.press("end")
        await pilot.press("enter")
        await pilot.pause()
        assert narrator.value == "blend:duet"
        assert narrator.value != "_separator"


@pytest.mark.asyncio
async def test_legacy_default_voice_select_has_no_keyboard_separator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _DefaultVoiceWidget(SpeechSettingsMixin, Widget):
        def __init__(self) -> None:
            super().__init__()
            self.init_settings_state()

        def compose(self) -> ComposeResult:
            yield Select(
                [("Alloy", "alloy")],
                id="default-voice-select",
                allow_blank=False,
            )

    class _DefaultVoiceHost(App[None]):
        def compose(self) -> ComposeResult:
            yield _DefaultVoiceWidget()

    blend_file = tmp_path / "voice-blends.json"
    blend_file.write_text(
        json.dumps({"duet": {"description": "Two voices"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Speech.speech_settings_mixin.kokoro_ui_blend_file",
        lambda: blend_file,
    )
    app = _DefaultVoiceHost()

    async with app.run_test() as pilot:
        widget = app.query_one(_DefaultVoiceWidget)
        widget._update_default_voice_options("kokoro")
        voice = app.query_one("#default-voice-select", Select)
        option_values = tuple(value for _label, value in voice._options)
        assert "_separator" not in option_values
        assert "blend:duet" in option_values

        voice.focus()
        await pilot.press("enter")
        await pilot.press("end")
        await pilot.press("enter")
        assert voice.value == "blend:duet"


@pytest.mark.asyncio
async def test_profile_store_unavailable_isolated_to_stable_library_recovery() -> None:
    app = _STTSHost(None)

    async with app.run_test(size=(150, 55)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: app.profile_service_requests == 1)

        assert _status_copy(app) == PROFILE_STORE_UNAVAILABLE_COPY
        assert app.query_one("#stts-profile-table", DataTable).row_count == 0
        assert not app.query_one("#stts-profile-refresh-btn", Button).disabled

        await _open_stts_view(app, pilot, "playground")
        assert app.query_one("#tts-generate-btn", Button)


@pytest.mark.asyncio
async def test_store_unavailable_recovery_wraps_without_becoming_a_tab_stop_at_80x24() -> (
    None
):
    app = _STTSHost(None)

    async with app.run_test(size=(80, 24)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: app.profile_service_requests == 1)

        status = app.query_one("#stts-profile-status")
        assert not status.can_focus
        assert "Choose Refresh" in "\n".join(_visible_content_rows(status))


@pytest.mark.asyncio
async def test_empty_profile_page_offers_only_current_library_recovery() -> None:
    service = _ControlledProfileService()
    app = _STTSHost(service)

    async with app.run_test(size=(80, 24)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: bool(service.availability_calls))

        status = _status_copy(app)
        assert status == (
            "No voice profiles match. Change the search or choose Refresh "
            "to check again."
        )
        assert "save" not in status.casefold()


@pytest.mark.parametrize(
    "store_code",
    ["unavailable", "closed", "terminal", "restoring"],
)
@pytest.mark.asyncio
async def test_refresh_reloads_service_after_store_level_list_failure(
    store_code: str,
) -> None:
    failed_service = _StoreFailureProfileService(store_code)
    replacement = _ActionProfileService(_profile(0))
    app = _STTSHost(failed_service)

    async with app.run_test(size=(150, 55)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: len(failed_service.list_calls) == 1)

        await _wait_until(
            pilot,
            lambda: _status_copy(app) == PROFILE_STORE_UNAVAILABLE_COPY,
        )
        assert app.profile_service_requests == 1
        assert app.query_one("#stts-profile-table", DataTable).row_count == 0

        app.profile_service = replacement
        app.query_one("#stts-profile-refresh-btn", Button).press()
        await _wait_until(pilot, lambda: app.profile_service_requests == 2)
        await _wait_until(
            pilot,
            lambda: (
                app.query_one(
                    "#stts-profile-table",
                    DataTable,
                ).row_count
                == 1
            ),
        )

        assert failed_service.list_calls == [(None, 0)]
        assert replacement.list_calls == [(None, 0)]
        assert (
            _table_cell(
                app.query_one("#stts-profile-table", DataTable),
                0,
                0,
            )
            == "Voice 00"
        )


@pytest.mark.asyncio
async def test_repository_page_renders_before_availability_and_selection_arms_actions() -> (
    None
):
    profiles = tuple(_profile(index) for index in range(50))
    service = _ControlledProfileService(profiles, total=51)
    app = _STTSHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: bool(service.availability_calls))

        table = app.query_one("#stts-profile-table", DataTable)
        assert table.row_count == 50
        assert service.availability_future is not None
        assert not service.availability_future.done()

        for selector in (
            "#stts-profile-preview-btn",
            "#stts-profile-edit-btn",
            "#stts-profile-duplicate-btn",
            "#stts-profile-delete-btn",
        ):
            assert app.query_one(selector, Button).disabled
        assert not app.query_one("#stts-profile-refresh-btn", Button).disabled

        table.move_cursor(row=0)
        table.action_select_cursor()
        await pilot.pause()

        for selector in (
            "#stts-profile-edit-btn",
            "#stts-profile-duplicate-btn",
            "#stts-profile-delete-btn",
        ):
            assert not app.query_one(selector, Button).disabled
        assert app.query_one("#stts-profile-preview-btn", Button).disabled

        assert table.row_count <= 50
        service.availability_future.set_result(
            _availability(service.page, state="unavailable")
        )
        await _wait_until(
            pilot,
            lambda: _table_cell(table, 0, 3) == "Unavailable",
        )
        assert _status_copy(app).startswith("Unavailable — Refresh, then Edit.")
        assert app.query_one("#stts-profile-status-copy").has_class("selected-detail")


@pytest.mark.asyncio
async def test_voice_profile_actions_fit_and_remain_keyboard_reachable_at_80x24() -> (
    None
):
    service = _ActionProfileService(_profile(0))
    app = _ActionHost(service)
    action_ids = {
        "stts-profile-preview-btn",
        "stts-profile-edit-btn",
        "stts-profile-duplicate-btn",
        "stts-profile-refresh-btn",
        "stts-profile-delete-btn",
    }

    async with app.run_test(size=(80, 24)) as pilot:
        await _select_action_profile(app, pilot)
        table = app.query_one("#stts-profile-table", DataTable)
        status = app.query_one("#stts-profile-status")
        buttons = tuple(
            app.query_one(f"#{button_id}", Button) for button_id in action_ids
        )

        assert table.region.height >= 3
        assert status.region.height > 0
        assert status.region.right <= app.size.width
        assert status.region.bottom <= app.size.height
        for button in buttons:
            assert button.region.width > 0
            assert button.region.height > 0
            assert button.region.right <= app.size.width
            assert button.region.bottom <= app.size.height

        table.focus()
        reached: set[str] = set()
        for _ in range(24):
            await pilot.press("tab")
            focused = app.focused
            if focused is not None and focused.id is not None:
                reached.add(focused.id)
        assert action_ids <= reached


@pytest.mark.parametrize(
    ("availability_state", "expected_recovery"),
    [
        ("unavailable", "Unavailable — Refresh, then Edit."),
        ("unverified", "Needs test — Open in Playground."),
    ],
)
@pytest.mark.asyncio
async def test_profile_recovery_copy_is_visible_at_80x24(
    availability_state: str,
    expected_recovery: str,
) -> None:
    service = _ActionProfileService(
        _profile(0),
        availability_state=availability_state,
    )
    app = _ActionHost(service)

    async with app.run_test(size=(80, 24)) as pilot:
        await _select_action_profile(app, pilot)
        status = app.query_one("#stts-profile-status")

        assert _visible_content_rows(status) == (
            expected_recovery,
            "Selected: Voice 00",
            "audio_cpp / model/0 / voice/0",
        )


@pytest.mark.asyncio
async def test_unverified_legacy_profile_offers_an_exact_playground_test() -> (
    None
):
    """A provider without catalog authority needs sample evidence, not Ready."""

    legacy = replace(
        _profile(0),
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        response_format="mp3",
    )
    service = _ActionProfileService(
        legacy,
        availability_state="unverified",
        availability_recovery="none",
    )
    app = _ActionHost(service)

    async with app.run_test(size=(80, 24)) as pilot:
        await _select_action_profile(app, pilot)
        status = app.query_one("#stts-profile-status")

        rows = _visible_content_rows(status)
        assert rows == (
            "Needs test — Open in Playground.",
            "Selected: Voice 00",
            "openai / tts-1 / alloy",
        )
        assert "refresh" not in rows[0].casefold()
        assert "unverified" not in rows[0].casefold()
        assert app.query_one("#stts-profile-preview-btn", Button).label.plain == (
            "Test in Playground"
        )


@pytest.mark.parametrize(
    ("provider_id", "state", "recovery_action", "expected_cell"),
    [
        ("openai", "unverified", "none", "Needs test"),
        ("audio_cpp", "unverified", "refresh", "Needs test"),
        ("audio_cpp", "available", "none", "Verified"),
        ("audio_cpp", "unavailable", "edit", "Unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_availability_cell_uses_the_three_truthful_profile_states(
    provider_id: str,
    state: str,
    recovery_action: str,
    expected_cell: str,
) -> None:
    """Library rows map service evidence to the user-facing status contract."""

    profile = replace(_profile(0), provider_id=provider_id)
    service = _ActionProfileService(
        profile,
        availability_state=state,
        availability_recovery=recovery_action,
    )
    app = _STTSHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        table = app.query_one("#stts-profile-table", DataTable)
        await _wait_until(pilot, lambda: bool(service.availability_calls))
        await _wait_until(pilot, lambda: _table_cell(table, 0, 3) != "Checking")

        assert _table_cell(table, 0, 3) == expected_cell


@pytest.mark.asyncio
async def test_publish_page_preserves_needs_test_during_same_page_refresh() -> (
    None
):
    """The availability cell is filled at two sites -- `_publish_page` (from
    preserved availability across a same-page refresh) and
    `_publish_availability` (from a fresh observation) -- and both must go
    through the same helper so they cannot diverge. This pins
    `_publish_page`'s direct rendering by re-listing the identical page and
    checking the cell BEFORE the new availability observation resolves, when
    the cell can only have come from `_publish_page`'s preserved-availability
    branch."""

    legacy = replace(
        _profile(0),
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
    )
    service = _PipelineProfileService()
    app = _STTSHost(service)
    page = _page(legacy, generation=5)

    async with app.run_test(size=(150, 55)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: len(service.list_futures) == 1)
        service.list_futures[0].set_result(page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 1)
        service.availability_futures[0].set_result(
            _availability(page, state="unverified", recovery_action="none")
        )
        await pilot.pause()
        table = app.query_one("#stts-profile-table", DataTable)
        assert _table_cell(table, 0, 3) == "Needs test"

        app.query_one("#stts-profile-refresh-btn", Button).press()
        await pilot.pause()
        await _wait_until(pilot, lambda: len(service.list_futures) == 2)
        service.list_futures[1].set_result(page)
        await pilot.pause()

        # The second availability observation has not resolved yet -- this
        # cell can only have come from `_publish_page`'s direct render of
        # the preserved availability, not from `_publish_availability`.
        assert _table_cell(table, 0, 3) == "Needs test"


@pytest.mark.asyncio
async def test_long_profile_identifiers_are_keyboard_scrollable_at_80x24() -> None:
    display_name = "profile-" + ("n" * 120)
    model_id = f"model/{'opaque-model-segment/' * 5}model-tail"
    voice_id = f"voice/{'opaque-voice-segment/' * 5}voice-tail"
    profile = replace(
        _profile(0),
        display_name=display_name,
        normalized_name=display_name,
        model_id=model_id,
        voice_id=voice_id,
    )
    service = _ActionProfileService(profile, availability_state="unavailable")
    app = _ActionHost(service)
    detail_copy = f"audio_cpp / {model_id} / {voice_id}"

    async with app.run_test(size=(80, 24)) as pilot:
        library, _selected = await _select_action_profile(app, pilot)
        status = app.query_one("#stts-profile-status")
        visible_lines = _visible_content_rows(status)

        assert visible_lines[0] == "Unavailable — Refresh, then Edit."
        assert visible_lines[1].startswith("Selected: profile-")
        assert visible_lines[2].startswith("audio_cpp / model/opaque-model")
        assert status.region.height <= 5
        identifier_control = app.query_one("#stts-profile-identifiers")
        assert status.region.contains_region(identifier_control.region)

        table = app.query_one("#stts-profile-table", DataTable)
        table.focus()
        for _ in range(8):
            await pilot.press("tab")
            if app.focused is not None and app.focused.id == "stts-profile-identifiers":
                break
        identifiers = app.focused
        assert identifiers is not None
        assert identifiers is not status
        assert identifiers is identifier_control
        assert identifiers.text == detail_copy  # type: ignore[attr-defined]

        await pilot.press("end")
        await _wait_until(pilot, lambda: identifiers.scroll_x > 0)
        visible_lines = _visible_content_rows(status)
        assert visible_lines[0] == "Unavailable — Refresh, then Edit."
        assert visible_lines[1].startswith("Selected: profile-")
        assert visible_lines[2].endswith("voice-tail")

        unchanged = identifiers.text  # type: ignore[attr-defined]
        await pilot.press("x")
        assert identifiers.text == unchanged  # type: ignore[attr-defined]

        library._set_status(PROFILE_STORE_UNAVAILABLE_COPY)
        await pilot.pause()
        assert not identifiers.display
        assert identifiers.scroll_x == 0
        assert app.focused is table
        assert "Choose Refresh" in "\n".join(_visible_content_rows(status))

        refresh = app.query_one("#stts-profile-refresh-btn", Button)
        refresh.focus()
        library._set_status(profile_library_module.PROFILE_ACTION_FAILED_COPY)
        await pilot.pause()
        assert app.focused is refresh

        for selector in (
            "#stts-profile-preview-btn",
            "#stts-profile-edit-btn",
            "#stts-profile-duplicate-btn",
            "#stts-profile-refresh-btn",
            "#stts-profile-delete-btn",
        ):
            button = app.query_one(selector, Button)
            assert button.region.bottom <= app.size.height


@pytest.mark.asyncio
async def test_search_validates_shared_text_bound_before_queueing_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profile_library_module,
        "PROFILE_SEARCH_DEBOUNCE_SECONDS",
        0.01,
        raising=False,
    )
    service = _PipelineProfileService()
    app = _STTSHost(service)
    initial_page = _page(_profile(0), generation=5)

    async with app.run_test(size=(150, 55)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: len(service.list_futures) == 1)
        service.list_futures[0].set_result(initial_page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 1)
        service.availability_futures[0].set_result(_availability(initial_page))
        await pilot.pause()

        search = app.query_one("#stts-profile-search", Input)
        search.value = "x" * 129
        await pilot.pause(0.03)

        assert service.list_calls == [(None, 0)]
        assert "128 characters or fewer" in _status_copy(app)

        search.value = "x" * 128
        await _wait_until(pilot, lambda: len(service.list_futures) == 2)
        assert service.list_calls[-1] == ("x" * 128, 0)


@pytest.mark.asyncio
async def test_search_debounces_before_one_active_and_one_latest_page_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profile_library_module,
        "PROFILE_SEARCH_DEBOUNCE_SECONDS",
        0.03,
        raising=False,
    )
    service = _PipelineProfileService()
    app = _STTSHost(service)
    initial_page = _page(_profile(0), generation=5)
    latest_page = _page(_profile(5), generation=5)

    async with app.run_test(size=(150, 55)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: len(service.list_futures) == 1)
        service.list_futures[0].set_result(initial_page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 1)
        service.availability_futures[0].set_result(_availability(initial_page))
        await pilot.pause()

        search = app.query_one("#stts-profile-search")
        search.value = "a"
        search.value = "ab"
        search.value = "abc"
        await pilot.pause(0.01)
        assert service.list_calls == [(None, 0)]

        await _wait_until(pilot, lambda: len(service.list_futures) == 2)
        assert service.list_calls[-1] == ("abc", 0)

        search.value = "abcd"
        await pilot.pause(0.05)
        search.value = "abcde"
        await pilot.pause(0.05)
        assert service.list_calls == [(None, 0), ("abc", 0)]

        service.list_futures[1].set_result(_page(_profile(3), generation=5))
        await _wait_until(pilot, lambda: len(service.list_futures) == 3)
        assert service.list_calls == [
            (None, 0),
            ("abc", 0),
            ("abcde", 0),
        ]
        assert service.maximum_active_list_calls == 1

        service.list_futures[2].set_result(latest_page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 2)
        service.availability_futures[1].set_result(_availability(latest_page))
        await pilot.pause()


@pytest.mark.asyncio
async def test_stale_rendered_rows_cannot_arm_actions_or_emit_preview() -> None:
    service = _PipelineProfileService()
    app = _ActionHost(service)
    initial_page = _page(_profile(0), generation=5)
    replacement_page = _page(_profile(1), generation=6)
    action_selectors = (
        "#stts-profile-preview-btn",
        "#stts-profile-edit-btn",
        "#stts-profile-duplicate-btn",
        "#stts-profile-delete-btn",
    )

    async with app.run_test(size=(150, 55)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: len(service.list_futures) == 1)
        service.list_futures[0].set_result(initial_page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 1)
        service.availability_futures[0].set_result(_availability(initial_page))
        await pilot.pause()

        library = app.query_one(STTSProfileLibrary)
        table = app.query_one("#stts-profile-table", DataTable)
        initial_loaded = library._loaded_rows[str(initial_page.profiles[0].profile_id)]
        app.query_one("#stts-profile-refresh-btn", Button).press()
        await _wait_until(pilot, lambda: len(service.list_futures) == 2)
        assert _table_cell(table, 0, 0) == "Voice 00"

        table.move_cursor(row=0)
        table.action_select_cursor()
        await pilot.pause()
        stale_selected = library._selected_profile
        stale_disabled = tuple(
            app.query_one(selector, Button).disabled for selector in action_selectors
        )

        library._selected_profile = initial_loaded
        target_is_current = library._action_target_is_current(initial_loaded)
        preview = app.query_one("#stts-profile-preview-btn", Button)
        preview.disabled = False
        preview.press()
        await pilot.pause()
        stale_preview_count = len(app.preview_messages)
        app.preview_messages.clear()

        service.list_futures[1].set_result(replacement_page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 2)
        service.availability_futures[1].set_result(_availability(replacement_page))
        await _wait_until(
            pilot,
            lambda: library._rendered_repository_generation == 6,
        )
        table.move_cursor(row=0)
        table.action_select_cursor()
        await pilot.pause()
        current_loaded = library._selected_profile
        current_disabled = tuple(
            app.query_one(selector, Button).disabled for selector in action_selectors
        )
        assert current_loaded is not None
        assert current_loaded.repository_generation == 6
        assert current_disabled == (False, False, False, False)
        assert library._action_target_is_current(current_loaded)
        preview.press()
        await _wait_until(pilot, lambda: bool(app.preview_messages))

        assert (
            stale_selected,
            stale_disabled,
            target_is_current,
            stale_preview_count,
        ) == (None, (True, True, True, True), False, 0)
        message = app.preview_messages[0]
        assert isinstance(
            message,
            profile_library_module.ProfilePreviewRequested,
        )
        assert message.preset.model_id == current_loaded.profile.model_id
        assert service.preview_preset_calls[-1][0] is current_loaded


@pytest.mark.asyncio
async def test_cancellation_resistant_availability_keeps_one_cleanup_and_drains_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profile_library_module,
        "PROFILE_SEARCH_DEBOUNCE_SECONDS",
        0.02,
        raising=False,
    )
    service = _CancellationResistantAvailabilityService()
    app = _STTSHost(service)
    initial_page = _page(_profile(0), generation=5)
    middle_page = _page(_profile(1), generation=6)
    latest_page = _page(_profile(2), generation=7)
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    try:
        async with app.run_test(size=(150, 55)) as pilot:
            await _open_stts_view(app, pilot, "profiles")
            await _wait_until(pilot, lambda: len(service.list_futures) == 1)
            service.list_futures[0].set_result(initial_page)
            await _wait_until(
                pilot,
                lambda: len(service.availability_futures) == 1,
            )
            library = app.query_one(STTSProfileLibrary)
            table = app.query_one("#stts-profile-table", DataTable)

            search = app.query_one("#stts-profile-search", Input)
            search.value = "middle"
            await _wait_until(
                pilot,
                lambda: service.cleanup_started[0].is_set(),
            )
            assert not service.cleanup_releases[0].is_set()
            await _wait_until(pilot, lambda: len(service.list_futures) == 2)
            assert service.list_calls[-1] == ("middle", 0)

            service.list_futures[1].set_result(middle_page)
            await _wait_until(
                pilot,
                lambda: len(service.availability_futures) == 2,
            )
            search.value = "latest"
            await _wait_until(
                pilot,
                lambda: service.cleanup_started[1].is_set(),
            )
            await pilot.pause(0.05)

            assert len(service.list_futures) == 2
            assert library._retained_cleanup_task is not None
            assert library._active_page_task is not None
            assert library._pending_page_request is not None
            assert library._pending_page_request.search == "latest"

            service.cleanup_releases[0].set()
            await _wait_until(pilot, lambda: len(service.list_futures) == 3)
            assert service.list_calls[-1] == ("latest", 0)
            service.list_futures[2].set_result(latest_page)
            await _wait_until(
                pilot,
                lambda: len(service.availability_futures) == 3,
            )
            service.availability_futures[2].set_result(_availability(latest_page))
            await _wait_until(
                pilot,
                lambda: _table_cell(table, 0, 3) == "Verified",
            )

            assert _table_cell(table, 0, 0) == "Voice 02"
            assert library._retained_cleanup_task is not None
            assert not service.cleanup_releases[1].is_set()
            assert service.maximum_active_list_calls == 1

            await _open_stts_view(app, pilot, "settings")
            await _wait_until(
                pilot,
                lambda: service.cleanup_settled_by_unmount == [1],
            )
            assert library.parent is None
            assert library._active_page_task is None
            assert library._retained_cleanup_task is None
            assert _table_cell(table, 0, 0) == "Voice 02"
        assert _table_cell(table, 0, 3) == "Verified"
    finally:
        loop.set_exception_handler(previous_handler)

    await asyncio.sleep(0)
    assert unhandled == []


@pytest.mark.asyncio
async def test_late_search_and_repository_generation_pages_never_replace_current_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profile_library_module,
        "PROFILE_SEARCH_DEBOUNCE_SECONDS",
        0.02,
        raising=False,
    )
    service = _PipelineProfileService()
    app = _STTSHost(service)
    current_page = _page(_profile(0), generation=5)
    old_search_page = _page(_profile(1), generation=6)
    new_search_page = _page(_profile(2), generation=7)

    async with app.run_test(size=(150, 55)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: len(service.list_futures) == 1)
        service.list_futures[0].set_result(current_page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 1)
        service.availability_futures[0].set_result(_availability(current_page))
        await pilot.pause()

        table = app.query_one("#stts-profile-table", DataTable)
        assert _table_cell(table, 0, 0) == "Voice 00"

        search = app.query_one("#stts-profile-search")
        search.value = "old"
        await _wait_until(pilot, lambda: len(service.list_futures) == 2)
        search.value = "new"
        await pilot.pause(0.04)

        service.list_futures[1].set_result(old_search_page)
        await _wait_until(pilot, lambda: len(service.list_futures) == 3)
        assert _table_cell(table, 0, 0) == "Voice 00"
        assert service.list_calls[-1] == ("new", 0)

        service.list_futures[2].set_result(new_search_page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 2)
        assert _table_cell(table, 0, 0) == "Voice 02"
        service.availability_futures[1].set_result(_availability(new_search_page))
        await pilot.pause()

        app.query_one("#stts-profile-refresh-btn", Button).press()
        await pilot.pause()
        await _wait_until(pilot, lambda: len(service.list_futures) == 4)
        stale_generation_page = _page(_profile(9), generation=6)
        service.list_futures[3].set_result(stale_generation_page)
        await pilot.pause(0.05)

        assert _table_cell(table, 0, 0) == "Voice 02"
        assert len(service.availability_calls) == 2


@pytest.mark.asyncio
async def test_availability_revisions_cannot_regress_the_rendered_page() -> None:
    service = _PipelineProfileService()
    app = _STTSHost(service)
    page = _page(_profile(0), generation=5)

    async with app.run_test(size=(150, 55)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: len(service.list_futures) == 1)
        service.list_futures[0].set_result(page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 1)
        service.availability_futures[0].set_result(
            _availability(
                page,
                configuration_revision=5,
                catalog_revision=9,
            )
        )
        await pilot.pause()
        table = app.query_one("#stts-profile-table", DataTable)
        assert _table_cell(table, 0, 3) == "Verified"

        app.query_one("#stts-profile-refresh-btn", Button).press()
        await pilot.pause()
        await _wait_until(pilot, lambda: len(service.list_futures) == 2)
        service.list_futures[1].set_result(page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 2)
        service.availability_futures[1].set_result(
            _availability(
                page,
                state="unavailable",
                configuration_revision=4,
                catalog_revision=10,
            )
        )
        await pilot.pause(0.05)
        assert _table_cell(table, 0, 3) == "Verified"

        app.query_one("#stts-profile-refresh-btn", Button).press()
        await pilot.pause()
        await _wait_until(pilot, lambda: len(service.list_futures) == 3)
        service.list_futures[2].set_result(page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 3)
        service.availability_futures[2].set_result(
            _availability(
                page,
                state="unavailable",
                configuration_revision=5,
                catalog_revision=8,
            )
        )
        await pilot.pause(0.05)
        assert _table_cell(table, 0, 3) == "Verified"

        app.query_one("#stts-profile-refresh-btn", Button).press()
        await pilot.pause()
        await _wait_until(pilot, lambda: len(service.list_futures) == 4)
        service.list_futures[3].set_result(page)
        await _wait_until(pilot, lambda: len(service.availability_futures) == 4)
        service.availability_futures[3].set_result(
            _availability(
                page,
                state="unavailable",
                configuration_revision=6,
                catalog_revision=1,
            )
        )
        await pilot.pause()
        assert _table_cell(table, 0, 3) == "Unavailable"


@pytest.mark.asyncio
async def test_unmount_cancels_and_settles_the_retained_page_pipeline() -> None:
    service = _PipelineProfileService()
    app = _STTSHost(service)
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    try:
        async with app.run_test(size=(150, 55)) as pilot:
            await _open_stts_view(app, pilot, "profiles")
            await _wait_until(pilot, lambda: len(service.list_futures) == 1)
            library = app.query_one(STTSProfileLibrary)

            await _open_stts_view(app, pilot, "settings")
            await _wait_until(pilot, lambda: service.cancelled_list_calls == 1)

            assert library.parent is None
            assert library._live is False
            assert getattr(library, "_active_page_task", None) is None
    finally:
        loop.set_exception_handler(previous_handler)

    await asyncio.sleep(0)
    assert unhandled == []


@pytest.mark.asyncio
async def test_preview_posts_the_exact_loaded_profile_and_current_availability() -> (
    None
):
    service = _ActionProfileService(_profile(0))
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)
        await _wait_until(
            pilot,
            lambda: str(selected.profile.profile_id) in library._row_availability,
        )

        app.query_one("#stts-profile-preview-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: _playground_is_mounted(app),
        )
        preset = app.query_one(SpeechPlaygroundPane)._profile_preset
        assert type(preset) is TTSPlaygroundSelectionPreset
        current_availability = library._row_availability[
            str(selected.profile.profile_id)
        ]
        assert service.preview_preset_calls == [(selected, current_availability)]
        assert (
            preset.model_id,
            preset.voice_id,
            preset.availability,
        ) == (
            selected.profile.model_id,
            selected.profile.voice_id,
            current_availability.state,
        )
        assert preset.profile_id == selected.profile.profile_id
        assert preset.repository_generation == selected.repository_generation
        assert preset.profile_revision == selected.profile.revision
        assert "endpoint" not in repr(preset).casefold()
        assert "credential" not in repr(preset).casefold()


@pytest.mark.asyncio
async def test_unavailable_profile_disables_playground_action_with_clear_recovery() -> (
    None
):
    service = _ActionProfileService(
        _profile(0),
        availability_state="unavailable",
    )
    app = _ActionHost(service)

    async with app.run_test(size=(120, 40)) as pilot:
        await _select_action_profile(app, pilot)
        action = app.query_one("#stts-profile-preview-btn", Button)

        assert action.disabled
        assert "Unavailable" in str(action.label)
        assert _status_copy(app).startswith("Unavailable — Refresh, then Edit.")


@pytest.mark.asyncio
async def test_matching_profile_sample_records_evidence_and_enables_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(0)
    service = _ActionProfileService(
        profile,
        availability_state="unverified",
        availability_recovery="none",
    )
    app = _ActionHost(service)
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )

    async with app.run_test(size=(140, 50)) as pilot:
        await _select_action_profile(app, pilot)
        app.query_one("#stts-profile-preview-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: _playground_is_mounted(app),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        await _wait_until(pilot, lambda: len(pane.query("#audio-play-btn")) == 1)
        artifact = _playable_profile_artifact(tmp_path, profile)
        pane._generation_operation_id = artifact.operation_id
        pane._generation_complete(artifact)
        await _wait_until(pilot, lambda: service.availability_state == "available")
        await _wait_until(
            pilot,
            lambda: not app.query_one("#audio-save-profile-btn", Button).disabled,
        )

        assert service.sample_evidence_calls == [
            (
                LoadedTTSProfile(repository_generation=11, profile=profile),
                artifact,
            )
        ]
        assert "Verified" in str(
            app.query_one("#tts-profile-preview-status", Static).render()
        )


@pytest.mark.asyncio
async def test_different_profile_sample_cannot_enable_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(0)
    service = _ActionProfileService(
        profile,
        availability_state="unverified",
        availability_recovery="none",
    )
    app = _ActionHost(service)
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )

    async with app.run_test(size=(140, 50)) as pilot:
        await _select_action_profile(app, pilot)
        app.query_one("#stts-profile-preview-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: _playground_is_mounted(app),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        await _wait_until(pilot, lambda: len(pane.query("#audio-play-btn")) == 1)
        artifact = _playable_profile_artifact(
            tmp_path,
            profile,
            model_id="different/model",
        )
        pane._generation_operation_id = artifact.operation_id
        pane._generation_complete(artifact)
        await pilot.pause()
        await pilot.pause()

        assert app.query_one("#audio-save-profile-btn", Button).disabled
        assert "Needs test" in str(
            app.query_one("#tts-profile-preview-status", Static).render()
        )


@pytest.mark.asyncio
async def test_same_page_refresh_preserves_selected_profile_focus_and_scroll() -> None:
    profiles = tuple(_profile(index) for index in range(50))
    service = _ControlledProfileService(profiles)
    service.availability_future = asyncio.get_running_loop().create_future()
    service.availability_future.set_result(
        TTSProfileAvailabilitySnapshot(
            repository_generation=service.page.repository_generation,
            configuration_revision=1,
            catalog_revision=1,
            profiles=tuple(
                TTSProfileAvailability(
                    profile_id=profile.profile_id,
                    state="unverified",
                    recovery_action="refresh",
                )
                for profile in profiles
            ),
        )
    )
    app = _STTSHost(service)

    async with app.run_test(size=(80, 24)) as pilot:
        await _open_stts_view(app, pilot, "profiles")
        await _wait_until(pilot, lambda: len(service.availability_calls) == 1)
        library = app.query_one(STTSProfileLibrary)
        table = app.query_one("#stts-profile-table", DataTable)
        table.focus()
        table.move_cursor(row=35)
        table.action_select_cursor()
        table.scroll_to(y=35, animate=False)
        await pilot.pause()
        selected_id = library._selected_profile.profile.profile_id
        scroll_y = table.scroll_offset.y

        app.query_one("#stts-profile-refresh-btn", Button).press()
        await _wait_until(pilot, lambda: len(service.list_calls) == 2)
        await _wait_until(pilot, lambda: len(service.availability_calls) == 2)
        await pilot.pause()

        assert library._selected_profile is not None
        assert library._selected_profile.profile.profile_id == selected_id
        assert table.cursor_row == 35
        assert table.scroll_offset.y == scroll_y
        assert app.focused is table


@pytest.mark.asyncio
async def test_stts_window_consumes_exact_profile_preview_once_on_playground_remount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ActionProfileService(
        replace(
            _profile(0),
            model_id="missing/exact-model",
            voice_id="missing/exact-voice",
        )
    )
    app = _ActionHost(service)

    async def _unavailable_tts_service() -> object:
        raise RuntimeError("catalog deliberately unavailable")

    # `SpeechCatalogMixin._tts_service_factory` resolves `get_tts_service`
    # from its own direct import, never from `STTS_Window`'s module
    # namespace -- patching that module attribute was always inert here
    # (TASK-2951's second widget-deletion pass only surfaced it: deleting
    # the retired widget, the sole other user of that name in `STTS_Window`,
    # left the import genuinely unused, so ruff correctly dropped it and
    # this patch started raising `AttributeError` instead of silently
    # doing nothing).
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _unavailable_tts_service(),
    )

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)
        await _wait_until(
            pilot,
            lambda: str(selected.profile.profile_id) in library._row_availability,
        )

        app.query_one("#stts-profile-preview-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: _playground_is_mounted(app),
        )
        first_playground = app.query_one(SpeechPlaygroundPane)
        window = app.query_one(STTSWindow)

        assert first_playground._profile_preset is not None
        assert first_playground._profile_preset.model_id == "missing/exact-model"
        assert first_playground._profile_preset.voice_id == "missing/exact-voice"
        assert window._pending_playground_preset is None

        await pilot.pause(0.1)
        await _open_stts_view(app, pilot, "settings")
        await _open_stts_view(app, pilot, "playground")
        second_playground = app.query_one(SpeechPlaygroundPane)

        assert second_playground is not first_playground
        assert second_playground._profile_preset is None


@pytest.mark.asyncio
async def test_exact_preview_at_80x24_focuses_playground_with_visible_recovery_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ActionProfileService(
        replace(
            _profile(0),
            model_id="missing/exact-model",
            voice_id="missing/exact-voice",
        ),
        availability_state="unverified",
        availability_recovery="none",
    )
    app = _ActionHost(service)

    async def _unavailable_tts_service() -> object:
        raise RuntimeError("catalog deliberately unavailable")

    # `SpeechCatalogMixin._tts_service_factory` resolves `get_tts_service`
    # from its own direct import, never from `STTS_Window`'s module
    # namespace -- patching that module attribute was always inert here
    # (TASK-2951's second widget-deletion pass only surfaced it: deleting
    # the retired widget, the sole other user of that name in `STTS_Window`,
    # left the import genuinely unused, so ruff correctly dropped it and
    # this patch started raising `AttributeError` instead of silently
    # doing nothing).
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _unavailable_tts_service(),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        library, selected = await _select_action_profile(app, pilot)
        await _wait_until(
            pilot,
            lambda: str(selected.profile.profile_id) in library._row_availability,
        )
        preview = app.query_one("#stts-profile-preview-btn", Button)
        preview.focus()
        preview.scroll_visible(animate=False)
        await pilot.pause()
        preview.press()
        await _wait_until(
            pilot,
            lambda: _playground_is_mounted(app),
        )
        await pilot.pause()

        playground = app.query_one(SpeechPlaygroundPane)
        text_input = playground.query_one("#tts-text-input", TextArea)
        banner = playground.query_one("#tts-profile-preview-status", Static)
        copy = str(banner.render())

        assert app.focused is text_input
        assert banner.display is True
        assert "Needs test" in copy
        assert "exact sample" in copy
        assert banner.region.height > 0
        assert app.screen.region.contains_region(banner.region)


@pytest.mark.asyncio
async def test_profile_editor_preserves_opaque_values_and_builds_exact_draft() -> None:
    profile = replace(
        _profile(0),
        model_id="Opaque/Model:V1",
        voice_id="Voice Exact/β",
    )
    loaded = LoadedTTSProfile(repository_generation=11, profile=profile)
    modal = profile_library_module.TTSProfileEditorModal(
        loaded,
        assignment_count=3,
        mode="edit",
    )
    results: list[TTSProfileDraft | None] = []

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(100, 36)) as pilot:
        app.push_screen(modal, results.append)
        await pilot.pause()

        assert modal.query_one("#stts-profile-editor-name", Input).value == "Voice 00"
        assert (
            modal.query_one("#stts-profile-editor-model", Input).value
            == "Opaque/Model:V1"
        )
        assert (
            modal.query_one("#stts-profile-editor-voice", Input).value
            == "Voice Exact/β"
        )
        assert "3 assignments" in str(
            modal.query_one("#stts-profile-editor-scope", Static).render()
        )
        modal.query_one("#stts-profile-editor-name", Input).value = "Renamed"
        modal.query_one("#stts-profile-editor-model", Input).value = "New Opaque/Model"
        modal.query_one("#stts-profile-editor-voice", Input).value = "New Exact/Voice"
        await pilot.click("#stts-profile-editor-save")
        await _wait_until(pilot, lambda: len(results) == 1)

    assert results == [
        TTSProfileDraft(
            display_name="Renamed",
            provider_id="audio_cpp",
            model_id="New Opaque/Model",
            voice_id="New Exact/Voice",
            response_format="wav",
            speed=1.0,
            options={},
        )
    ]


@pytest.mark.asyncio
async def test_save_result_name_dialog_focuses_name_and_returns_trimmed_value() -> None:
    modal = profile_library_module.TTSProfileNameModal()
    results: list[str | None] = []

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(modal, results.append)
        await pilot.pause()
        name = modal.query_one("#stts-profile-name-input", Input)
        assert name.has_focus

        name.value = "  Character narrator  "
        await pilot.click("#stts-profile-name-save")
        await pilot.pause()

    assert results == ["Character narrator"]


@pytest.mark.asyncio
async def test_save_result_name_dialog_and_input_fit_at_50x24() -> None:
    modal = profile_library_module.TTSProfileNameModal()

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(50, 24)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        dialog = modal.query_one("#stts-profile-name-dialog")
        name = modal.query_one("#stts-profile-name-input", Input)

        for widget in (dialog, name):
            assert widget.region.width > 0
            assert widget.region.height > 0
            assert widget.region.x >= 0
            assert widget.region.right <= app.size.width
            assert widget.region.bottom <= app.size.height
        assert name.has_focus


@pytest.mark.asyncio
async def test_save_result_blank_name_uses_name_specific_not_saved_copy() -> None:
    modal = profile_library_module.TTSProfileNameModal()

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(50, 24)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        name = modal.query_one("#stts-profile-name-input", Input)
        name.value = "   "

        await pilot.click("#stts-profile-name-save")
        await pilot.pause()

        assert app.screen is modal
        assert (
            str(modal.query_one("#stts-profile-name-error", Static).render())
            == "Enter a profile name. The result was not saved."
        )


@pytest.mark.asyncio
async def test_clone_profile_review_requires_name_and_returns_explicit_post_save_choice() -> (
    None
):
    """Clone review distinguishes an unassigned save from a Roleplay handoff."""

    modal = profile_library_module.TTSCloneProfileSaveReviewModal()
    results: list[object] = []

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(modal, results.append)
        await pilot.pause()

        modal.query_one("#stts-clone-profile-name", Input).value = "   "
        await pilot.click("#stts-clone-profile-save-unassigned")
        await pilot.pause()
        assert results == []
        assert "Enter a profile name" in str(
            modal.query_one("#stts-clone-profile-error", Static).render()
        )

        modal.query_one("#stts-clone-profile-name", Input).value = "  Story voice  "
        await pilot.click("#stts-clone-profile-save-choose-character")
        await _wait_until(pilot, lambda: len(results) == 1)

    assert results == [
        profile_library_module.TTSCloneProfileSaveReview(
            display_name="Story voice",
            choose_character=True,
        )
    ]


@pytest.mark.parametrize("size", ((80, 24), (100, 30), (120, 35)))
@pytest.mark.asyncio
async def test_clone_profile_review_actions_fit_and_are_keyboard_reachable(
    size: tuple[int, int],
) -> None:
    modal = profile_library_module.TTSCloneProfileSaveReviewModal()

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=size) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        dialog = modal.query_one("#stts-clone-profile-dialog")
        controls = (
            modal.query_one("#stts-clone-profile-name", Input),
            modal.query_one("#stts-clone-profile-save-unassigned", Button),
            modal.query_one("#stts-clone-profile-save-choose-character", Button),
            modal.query_one("#stts-clone-profile-cancel", Button),
        )
        for control in controls:
            assert control.region.width > 0
            assert dialog.region.contains_region(control.region)
            control.focus()
            await pilot.pause()
            assert control.has_focus

        painted = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in modal._compositor.render_strips()
        )
        for label in ("Save unassigned", "Save & choose character", "Cancel"):
            assert label in painted


@pytest.mark.asyncio
async def test_profile_editor_refuses_a_cleared_voice_for_a_legacy_provider() -> None:
    """Clearing "Exact voice" on a legacy profile must not save silently.

    Legacy providers cannot speak without an exact voice, so the profile
    domain refuses the shape; the modal must surface its existing validation
    copy and keep the dialog open rather than dismissing with a draft.
    """

    legacy = replace(
        _profile(0),
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        response_format="mp3",
    )
    loaded = LoadedTTSProfile(repository_generation=11, profile=legacy)
    modal = profile_library_module.TTSProfileEditorModal(
        loaded,
        assignment_count=1,
        mode="edit",
    )
    results: list[TTSProfileDraft | None] = []

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(100, 36)) as pilot:
        app.push_screen(modal, results.append)
        await pilot.pause()

        voice_input = modal.query_one("#stts-profile-editor-voice", Input)
        assert voice_input.placeholder == "Required"
        voice_input.value = ""
        await pilot.click("#stts-profile-editor-save")
        await pilot.pause()

        assert results == []
        assert modal.is_running
        assert str(modal.query_one("#stts-profile-editor-error", Static).render()) == (
            "Review the profile name, model, and voice. Exact values were not saved."
        )


@pytest.mark.asyncio
async def test_profile_editor_still_accepts_a_cleared_voice_for_audio_cpp() -> None:
    loaded = LoadedTTSProfile(repository_generation=11, profile=_profile(0))
    modal = profile_library_module.TTSProfileEditorModal(
        loaded,
        assignment_count=1,
        mode="edit",
    )
    results: list[TTSProfileDraft | None] = []

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(100, 36)) as pilot:
        app.push_screen(modal, results.append)
        await pilot.pause()

        voice_input = modal.query_one("#stts-profile-editor-voice", Input)
        assert voice_input.placeholder == "Server default"
        voice_input.value = ""
        await pilot.click("#stts-profile-editor-save")
        await _wait_until(pilot, lambda: len(results) == 1)

    assert results[0] is not None
    assert results[0].voice_id is None


@pytest.mark.parametrize("mode", ["edit", "duplicate"])
@pytest.mark.asyncio
async def test_profile_editor_modal_controls_fit_at_80x24(mode: str) -> None:
    loaded = LoadedTTSProfile(repository_generation=11, profile=_profile(0))
    modal = profile_library_module.TTSProfileEditorModal(
        loaded,
        assignment_count=3,
        mode=mode,  # type: ignore[arg-type]
    )

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(modal)
        await pilot.pause()

        dialog = modal.query_one("#stts-profile-editor-dialog")
        name = modal.query_one("#stts-profile-editor-name", Input)
        model = modal.query_one("#stts-profile-editor-model", Input)
        voice = modal.query_one("#stts-profile-editor-voice", Input)
        error = modal.query_one("#stts-profile-editor-error", Static)
        cancel = modal.query_one("#stts-profile-editor-cancel", Button)
        save = modal.query_one("#stts-profile-editor-save", Button)

        name.value = ""
        save.press()
        await pilot.pause()

        for widget in (dialog, name, model, voice, error, cancel, save):
            region = widget.region
            assert region.width > 0
            assert region.height > 0
            assert region.x >= 0
            assert region.y >= 0
            assert region.right <= app.size.width
            assert region.bottom <= app.size.height

        assert name.can_focus
        assert model.can_focus
        assert voice.can_focus
        assert "Review the profile name" in str(error.render())

        painted = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in modal._compositor.render_strips()
        )
        for visible_copy in (
            "Name",
            "Exact model",
            "Exact voice",
            "Review the profile name",
            "Cancel",
            "Save",
        ):
            assert visible_copy in painted


@pytest.mark.parametrize(
    ("assignment_count", "assignment_copy", "recovery_copy", "cancel_label"),
    [
        (
            0,
            "No assignments were observed.",
            "profile storage remains the final authority.",
            "Cancel",
        ),
        (
            3,
            "3 assignments",
            "Remove them before deletion.",
            "Close",
        ),
    ],
)
@pytest.mark.asyncio
async def test_delete_modal_identifies_long_literal_profile_name_at_80x24(
    assignment_count: int,
    assignment_copy: str,
    recovery_copy: str,
    cancel_label: str,
) -> None:
    display_name = (
        "[bold]opaque target[/] "
        + " ".join(f"segment-{index:02d}" for index in range(8))
        + " tail"
    )
    modal = profile_library_module.TTSProfileDeleteModal(
        display_name=display_name,
        assignment_count=assignment_count,
    )

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(modal)
        await pilot.pause()

        dialog = modal.query_one("#stts-profile-delete-dialog")
        target = modal.query_one("#stts-profile-delete-target", Static)
        copy = modal.query_one("#stts-profile-delete-copy", Static)
        cancel = modal.query_one("#stts-profile-delete-cancel", Button)
        confirm = modal.query_one("#stts-profile-delete-confirm", Button)

        assert str(target.render()) == f"Profile: {display_name}"
        visible_target = " ".join(" ".join(_visible_content_rows(target)).split())
        assert visible_target == f"Profile: {display_name}"
        assert target.region.height >= 2
        assert assignment_copy in str(copy.render())
        assert recovery_copy in str(copy.render())
        assert str(cancel.label) == cancel_label
        assert confirm.disabled is (assignment_count > 0)

        for widget in (dialog, target, copy, cancel, confirm):
            assert widget.region.width > 0
            assert widget.region.height > 0
            assert widget.region.right <= app.size.width
            assert widget.region.bottom <= app.size.height


@pytest.mark.parametrize(
    ("modal_kind", "expected_result"),
    [("editor", None), ("delete", False)],
)
@pytest.mark.asyncio
async def test_profile_modals_escape_returns_their_cancel_result(
    modal_kind: str,
    expected_result: object,
) -> None:
    loaded = LoadedTTSProfile(repository_generation=11, profile=_profile(0))
    if modal_kind == "editor":
        modal = profile_library_module.TTSProfileEditorModal(
            loaded,
            assignment_count=0,
            mode="edit",
        )
    else:
        modal = profile_library_module.TTSProfileDeleteModal(
            display_name=loaded.profile.display_name,
            assignment_count=0,
        )
    results: list[object] = []

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(100, 36)) as pilot:
        app.push_screen(modal, results.append)
        await pilot.pause()
        await pilot.press("escape")
        await _wait_until(pilot, lambda: len(results) == 1)

    assert results == [expected_result]


@pytest.mark.parametrize(
    ("button_selector", "modal_type"),
    [
        (
            "#stts-profile-edit-btn",
            profile_library_module.TTSProfileEditorModal,
        ),
        (
            "#stts-profile-delete-btn",
            profile_library_module.TTSProfileDeleteModal,
        ),
    ],
)
@pytest.mark.asyncio
async def test_switching_stts_view_dismisses_owned_profile_modal_and_worker(
    button_selector: str,
    modal_type: type[object],
) -> None:
    service = _ActionProfileService(_profile(0))
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, _selected = await _select_action_profile(app, pilot)
        await pilot.click(button_selector)
        await _wait_until(pilot, lambda: isinstance(app.screen, modal_type))
        modal = app.screen
        action_workers = tuple(
            worker
            for worker in app.workers
            if getattr(worker, "group", None) == "voice_profile_action"
        )
        assert len(action_workers) == 1
        assert not action_workers[0].is_finished

        app.query_one(STTSWindow).current_view = "settings"
        await _wait_until(pilot, lambda: library.parent is None)
        await _wait_until(pilot, lambda: not isinstance(app.screen, modal_type))
        await _wait_until(
            pilot,
            lambda: isinstance(
                app.query_one(".stts-content").children[0],
                SpeechSettingsPane,
            ),
        )
        await _wait_until(
            pilot,
            lambda: all(worker.is_finished for worker in action_workers),
        )

        assert app.screen is not modal
        assert all(worker.is_finished for worker in action_workers)


@pytest.mark.parametrize(
    ("action_name", "modal_type", "expected_mode", "cancel_result"),
    [
        (
            "edit_selected_profile",
            profile_library_module.TTSProfileEditorModal,
            "edit",
            None,
        ),
        (
            "duplicate_selected_profile",
            profile_library_module.TTSProfileEditorModal,
            "duplicate",
            None,
        ),
        (
            "delete_selected_profile",
            profile_library_module.TTSProfileDeleteModal,
            None,
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_cancelled_profile_action_preserves_exact_current_detail(
    monkeypatch: pytest.MonkeyPatch,
    action_name: str,
    modal_type: type[object],
    expected_mode: str | None,
    cancel_result: object,
) -> None:
    profile = replace(
        _profile(0),
        model_id="cancel/exact-model",
        voice_id="cancel/exact-voice",
    )
    service = _ActionProfileService(profile, availability_state="unavailable")
    service.assignment_total = 0
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)
        status_before = _status_copy(app)
        identifiers_before = _identifier_copy(app)

        async def _cancel(screen: object) -> object:
            assert isinstance(screen, modal_type)
            if isinstance(screen, profile_library_module.TTSProfileEditorModal):
                assert screen.loaded is selected
                assert screen.mode == expected_mode
            else:
                assert screen.display_name == selected.profile.display_name
                assert screen.assignment_count == 0
            return cancel_result

        monkeypatch.setattr(app, "push_screen_wait", _cancel)
        action = getattr(library, action_name)
        result = await action()

        assert result is cancel_result
        assert service.assignment_count_calls == [selected]
        assert service.update_calls == []
        assert service.duplicate_calls == []
        assert service.delete_calls == []
        assert status_before.startswith("Unavailable — Refresh, then Edit.")
        assert _status_copy(app) == status_before
        assert identifiers_before == (
            "audio_cpp / cancel/exact-model / cancel/exact-voice"
        )
        assert _identifier_copy(app) == identifiers_before
        assert app.query_one("#stts-profile-identifiers").display


@pytest.mark.asyncio
async def test_edit_cancel_preserves_pending_availability_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _PendingAvailabilityActionProfileService(
        replace(
            _profile(0),
            model_id="pending/original-model",
            voice_id="pending/original-voice",
        )
    )
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)
        assert service.availability_future is not None

        async def _cancel_after_availability_failure(screen: object) -> None:
            assert isinstance(
                screen,
                profile_library_module.TTSProfileEditorModal,
            )
            assert screen.loaded is selected
            service.availability_future.set_exception(
                RuntimeError("must remain bounded")
            )
            await _wait_until(
                pilot,
                lambda: (
                    _status_copy(app)
                    == (
                        "Profiles are loaded, but availability is unverified. "
                        "Choose Refresh to retry; exact persisted values were not "
                        "changed."
                    )
                ),
            )

        monkeypatch.setattr(
            app,
            "push_screen_wait",
            _cancel_after_availability_failure,
        )
        assert await library.edit_selected_profile() is None

        assert _status_copy(app) == (
            "Profiles are loaded, but availability is unverified. Choose Refresh "
            "to retry; exact persisted values were not changed."
        )
        assert _identifier_copy(app) == ""
        assert not app.query_one("#stts-profile-identifiers").display
        assert service.update_calls == []


@pytest.mark.parametrize(
    "availability_outcome",
    ["success", "failure", "invalid"],
)
@pytest.mark.asyncio
async def test_late_availability_does_not_replace_newer_action_failure(
    availability_outcome: str,
) -> None:
    service = _PendingAvailabilityActionProfileService(_profile(0))
    service.assignment_total = -1
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, _selected = await _select_action_profile(app, pilot)
        assert service.availability_future is not None

        assert await library.edit_selected_profile() is None
        assert _status_copy(app) == profile_library_module.PROFILE_ACTION_FAILED_COPY
        assert not app.query_one("#stts-profile-status-copy").has_class(
            "selected-detail"
        )

        if availability_outcome == "success":
            service.availability_future.set_result(
                _availability(service.page, state="unavailable")
            )
        elif availability_outcome == "failure":
            service.availability_future.set_exception(
                RuntimeError("must remain bounded")
            )
        else:
            service.availability_future.set_result(object())
        await _wait_until(pilot, lambda: library._active_page_task is None)

        assert _status_copy(app) == profile_library_module.PROFILE_ACTION_FAILED_COPY
        assert _identifier_copy(app) == ""
        assert not app.query_one("#stts-profile-identifiers").display
        assert service.update_calls == []
        if availability_outcome == "success":
            assert (
                _table_cell(
                    app.query_one("#stts-profile-table", DataTable),
                    0,
                    3,
                )
                == "Unavailable"
            )


@pytest.mark.parametrize("generation_edit", [False, True])
@pytest.mark.asyncio
async def test_edit_passes_exact_loaded_token_for_rename_and_generation_changes(
    monkeypatch: pytest.MonkeyPatch,
    generation_edit: bool,
) -> None:
    service = _ActionProfileService(_profile(0))
    service.assignment_total = 4
    app = _ActionHost(service)
    returned_draft: TTSProfileDraft | None = None

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)

        async def _edit(screen: object) -> TTSProfileDraft:
            nonlocal returned_draft
            assert isinstance(
                screen,
                profile_library_module.TTSProfileEditorModal,
            )
            assert screen.loaded is selected
            assert screen.assignment_count == 4
            assert screen.mode == "edit"
            profile = screen.loaded.profile
            returned_draft = TTSProfileDraft(
                display_name="Renamed voice",
                provider_id=profile.provider_id,
                model_id=("changed/model" if generation_edit else profile.model_id),
                voice_id=("changed/voice" if generation_edit else profile.voice_id),
                response_format=profile.response_format,
                speed=profile.speed,
                options=profile.options,
            )
            return returned_draft

        monkeypatch.setattr(app, "push_screen_wait", _edit)
        result = await library.edit_selected_profile()

        assert service.assignment_count_calls[0] is selected
        assert service.update_calls[0][0] is selected
        assert service.update_calls[0][1] is returned_draft
        assert result is service.updated_result


@pytest.mark.asyncio
async def test_edit_conflict_retains_the_exact_draft_without_leaking_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ActionProfileService(_profile(0))
    service.update_error = ProfileRepositoryError("conflict")
    app = _ActionHost(service)
    submitted: TTSProfileDraft | None = None
    dialog_calls = 0

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)

        async def _edit(screen: object) -> TTSProfileDraft | None:
            nonlocal dialog_calls, submitted
            assert isinstance(
                screen,
                profile_library_module.TTSProfileEditorModal,
            )
            dialog_calls += 1
            if dialog_calls == 2:
                assert screen.initial_draft is submitted
                return None
            if dialog_calls == 3:
                assert screen.loaded.repository_generation == 12
                assert screen.initial_draft is None
                assert screen.initial_name == "Replacement voice"
                assert screen.initial_model_id == "replacement/model"
                assert screen.initial_voice_id == "replacement/voice"
                return None
            profile = screen.loaded.profile
            submitted = TTSProfileDraft(
                display_name="Conflict draft",
                provider_id=profile.provider_id,
                model_id=profile.model_id,
                voice_id=profile.voice_id,
                response_format=profile.response_format,
                speed=profile.speed,
                options=profile.options,
            )
            return submitted

        monkeypatch.setattr(app, "push_screen_wait", _edit)
        assert await library.edit_selected_profile() is None
        assert service.update_calls[0][0] is selected
        assert service.update_calls[0][1] is submitted
        assert _status_copy(app) == profile_library_module.PROFILE_CONFLICT_COPY

        await library.edit_selected_profile()
        assert len(service.update_calls) == 1

        replacement = replace(
            selected.profile,
            display_name="Replacement voice",
            normalized_name="replacement voice",
            model_id="replacement/model",
            voice_id="replacement/voice",
        )
        service.page = _page(replacement, generation=12)
        app.query_one("#stts-profile-refresh-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: library._rendered_repository_generation == 12,
        )
        assert library._retained_editor_draft is None
        table = app.query_one("#stts-profile-table", DataTable)
        table.move_cursor(row=0)
        table.action_select_cursor()
        await pilot.pause()

        await library.edit_selected_profile()
        assert dialog_calls == 3


@pytest.mark.asyncio
async def test_duplicate_uses_source_token_and_name_as_the_only_modal_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ActionProfileService(_profile(0))
    service.assignment_total = 2
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)

        async def _duplicate(screen: object) -> TTSProfileDraft:
            assert isinstance(
                screen,
                profile_library_module.TTSProfileEditorModal,
            )
            assert screen.loaded is selected
            assert screen.assignment_count == 2
            assert screen.mode == "duplicate"
            assert screen.initial_name == ""
            return TTSProfileDraft(
                display_name="Explicit copy",
                provider_id="audio_cpp",
                model_id="ignored/model",
                voice_id="ignored/voice",
                response_format="wav",
                speed=1.0,
                options={},
            )

        monkeypatch.setattr(app, "push_screen_wait", _duplicate)
        result = await library.duplicate_selected_profile()

        assert service.assignment_count_calls[0] is selected
        assert service.duplicate_calls == [(selected, "Explicit copy")]
        assert service.duplicate_calls[0][0] is selected
        assert result is service.duplicate_result
        assert result.profile.profile_id != selected.profile.profile_id
        assert result.profile.revision == 1


@pytest.mark.asyncio
async def test_create_from_artifact_handoff_preserves_the_exact_artifact() -> None:
    service = _ActionProfileService(_profile(0))
    artifact = _artifact()
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, _selected = await _select_action_profile(app, pilot)
        result = await library.create_from_artifact(
            "Saved artifact",
            artifact,
        )

        assert service.create_calls == [("Saved artifact", artifact)]
        assert service.create_calls[0][1] is artifact
        assert result is service.created_result
        assert "must never enter profile UI copy" not in _status_copy(app)


@pytest.mark.asyncio
async def test_create_from_artifact_maps_stale_configuration_to_honest_copy_for_any_provider() -> (
    None
):
    """`create_from_artifact` checks the exact provider's configuration
    revision with NO provider gate (`profile_service.py`'s
    `_require_configuration_revision` call at the top of
    `create_from_artifact`), so `stale_configuration` is reachable for a
    legacy provider here -- unlike every other action on this widget, whose
    `profile_unverified`/`stale_configuration` raises are all gated behind
    `provider_id != _PROFILE_PROVIDER_ID: return`. The two live callers
    (`STTS_Window.py`, `speech_profile_mixin.py`) already special-case this
    code with provider-agnostic copy instead of falling through to the
    "Refresh and retry" toast, which only makes sense for audio_cpp; this
    widget method must do the same, not launder a legacy staleness failure
    into a promise of a capability check that provider never gets."""

    legacy = replace(
        _profile(0),
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
    )
    service = _ActionProfileService(legacy)
    service.create_error = ProfileServiceError("stale_configuration")
    artifact = _artifact()
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, _selected = await _select_action_profile(app, pilot)
        result = await library.create_from_artifact("Saved artifact", artifact)

        assert result is None
        assert service.create_calls == [("Saved artifact", artifact)]
        status = _status_copy(app)
        assert status == profile_library_module._PROFILE_RESULT_STALE_COPY
        assert status != profile_library_module._PROFILE_UNVERIFIED_COPY
        assert "verified" not in status.casefold()
        assert "refresh" not in status.casefold()


@pytest.mark.asyncio
async def test_delete_shows_advisory_count_but_repository_conflict_is_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ActionProfileService(_profile(0))
    service.assignment_total = 0
    service.delete_error = ProfileRepositoryError("conflict")
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)

        async def _confirm(screen: object) -> bool:
            assert isinstance(
                screen,
                profile_library_module.TTSProfileDeleteModal,
            )
            assert screen.assignment_count == 0
            assert screen.display_name == selected.profile.display_name
            return True

        monkeypatch.setattr(app, "push_screen_wait", _confirm)
        assert await library.delete_selected_profile() is False

        assert service.assignment_count_calls[0] is selected
        assert service.delete_calls[0] is selected
        assert _status_copy(app) == profile_library_module.PROFILE_CONFLICT_COPY


@pytest.mark.asyncio
async def test_assigned_profile_delete_is_blocked_before_repository_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ActionProfileService(_profile(0))
    service.assignment_total = 2
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)

        async def _confirm(screen: object) -> bool:
            assert isinstance(
                screen,
                profile_library_module.TTSProfileDeleteModal,
            )
            assert screen.assignment_count == 2
            assert screen.display_name == selected.profile.display_name
            return True

        monkeypatch.setattr(app, "push_screen_wait", _confirm)
        assert await library.delete_selected_profile() is False

        assert service.assignment_count_calls[0] is selected
        assert service.delete_calls == []
        assert _status_copy(app) == profile_library_module.PROFILE_DELETE_PROTECTED_COPY


@pytest.mark.parametrize(
    ("configured_default", "expected_is_app_default"),
    [
        ("self", True),
        ("other", False),
        (None, False),
    ],
)
@pytest.mark.asyncio
async def test_delete_confirmation_flags_the_configured_app_default_profile(
    monkeypatch: pytest.MonkeyPatch,
    configured_default: str | None,
    expected_is_app_default: bool,
) -> None:
    """The delete confirmation must know a profile is `[app_tts]
    default_profile_id` -- a fact `assignment_count` structurally cannot
    see, since it lives in config, not the profile store. Deleting an
    unrelated profile (or with nothing configured) must not claim it.
    """

    profile = _profile(0)
    service = _ActionProfileService(profile)
    service.assignment_total = 0
    app = _ActionHost(service)

    if configured_default == "self":
        configured_value: str | None = str(profile.profile_id)
    elif configured_default == "other":
        configured_value = str(uuid4())
    else:
        configured_value = None

    monkeypatch.setattr(
        stts_window_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            configured_value
            if (section, key) == ("app_tts", "default_profile_id")
            else default
        ),
    )

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)

        captured: dict[str, object] = {}

        async def _confirm(screen: object) -> bool:
            assert isinstance(
                screen,
                profile_library_module.TTSProfileDeleteModal,
            )
            assert screen.display_name == selected.profile.display_name
            captured["is_app_default"] = screen.is_app_default
            return False

        monkeypatch.setattr(app, "push_screen_wait", _confirm)
        await library.delete_selected_profile()

        assert captured["is_app_default"] is expected_is_app_default


@pytest.mark.parametrize(
    ("assignment_count", "is_app_default", "should_mention_default"),
    [
        (0, True, True),
        (0, False, False),
        (3, True, True),
    ],
)
@pytest.mark.asyncio
async def test_delete_modal_copy_names_the_app_default_voice(
    assignment_count: int,
    is_app_default: bool,
    should_mention_default: bool,
) -> None:
    modal = profile_library_module.TTSProfileDeleteModal(
        display_name="Narrator voice",
        assignment_count=assignment_count,
        is_app_default=is_app_default,
    )

    class _ModalHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ModalHost()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(modal)
        await pilot.pause()

        copy = str(modal.query_one("#stts-profile-delete-copy", Static).render())
        assert ("app-wide default voice" in copy) is should_mention_default


@pytest.mark.asyncio
async def test_refresh_and_editor_repair_keep_unavailable_persisted_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = replace(
        _profile(0),
        model_id="missing/exact-model",
        voice_id="missing/exact-voice",
    )
    service = _ActionProfileService(profile, availability_state="unavailable")
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)
        await _wait_until(pilot, lambda: len(service.availability_calls) == 1)

        assert "missing/exact-model" in _identifier_copy(app)
        assert "missing/exact-voice" in _identifier_copy(app)
        assert "Refresh, then Edit" in _status_copy(app)

        app.query_one("#stts-profile-refresh-btn", Button).press()
        await _wait_until(pilot, lambda: len(service.availability_calls) == 2)
        table = app.query_one("#stts-profile-table", DataTable)
        table.move_cursor(row=0)
        table.action_select_cursor()
        await pilot.pause()
        refreshed = library._selected_profile
        assert refreshed is not None

        async def _repair(screen: object) -> None:
            assert isinstance(
                screen,
                profile_library_module.TTSProfileEditorModal,
            )
            assert screen.loaded is refreshed
            assert screen.initial_draft is None
            assert screen.initial_model_id == "missing/exact-model"
            assert screen.initial_voice_id == "missing/exact-voice"

        monkeypatch.setattr(app, "push_screen_wait", _repair)
        assert await library.edit_selected_profile() is None
        assert service.update_calls == []
        assert refreshed.profile == selected.profile


@pytest.mark.asyncio
async def test_unexpected_action_errors_render_only_value_independent_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ActionProfileService(_profile(0))
    service.duplicate_error = RuntimeError(
        "must never leak /Users/private/key https://upstream.invalid body"
    )
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)

        async def _duplicate(screen: object) -> TTSProfileDraft:
            assert isinstance(
                screen,
                profile_library_module.TTSProfileEditorModal,
            )
            profile = selected.profile
            return TTSProfileDraft(
                display_name="Copy",
                provider_id=profile.provider_id,
                model_id=profile.model_id,
                voice_id=profile.voice_id,
                response_format=profile.response_format,
                speed=profile.speed,
                options=profile.options,
            )

        monkeypatch.setattr(app, "push_screen_wait", _duplicate)
        assert await library.duplicate_selected_profile() is None

        status = _status_copy(app)
        assert status == profile_library_module.PROFILE_ACTION_FAILED_COPY
        assert "/Users/private/key" not in status
        assert "upstream.invalid" not in status


@pytest.mark.asyncio
async def test_selected_profile_exports_the_sanitized_standalone_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = _profile(0)
    service = _ActionProfileService(profile)
    app = _ActionHost(service)
    target = tmp_path / "voice-profile.json"

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)
        export_button = app.query_one("#stts-profile-export-btn", Button)
        assert export_button.disabled is False

        async def _choose_path() -> Path:
            return target

        monkeypatch.setattr(library, "_choose_profile_export_path", _choose_path)

        assert await library.export_selected_profile() is True

        payload = json.loads(target.read_text(encoding="utf-8"))
        expected = portable_profile_payload(
            PortableTTSProfile(
                profile_id=selected.profile.profile_id,
                draft=TTSProfileDraft(
                    display_name=selected.profile.display_name,
                    provider_id=selected.profile.provider_id,
                    model_id=selected.profile.model_id,
                    voice_id=selected.profile.voice_id,
                    response_format=selected.profile.response_format,
                    speed=selected.profile.speed,
                    options=selected.profile.options,
                ),
            )
        )
        assert payload == expected
        assert "revision" not in payload
        assert "created_at" not in payload
        assert _status_copy(app) == "Voice profile exported."


@pytest.mark.asyncio
async def test_profile_export_cancellation_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _ActionProfileService(_profile(0))
    app = _ActionHost(service)

    async with app.run_test(size=(150, 55)) as pilot:
        library, _selected = await _select_action_profile(app, pilot)

        async def _cancel() -> None:
            return None

        monkeypatch.setattr(library, "_choose_profile_export_path", _cancel)

        assert await library.export_selected_profile() is False
        assert list(tmp_path.glob("*.json")) == []


@pytest.mark.asyncio
async def test_profile_export_path_failure_never_logs_sensitive_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from loguru import logger as loguru_logger

    secret = "credential-private-origin-message-text"
    hidden_parent = tmp_path / f".{secret}"
    hidden_parent.mkdir()
    target = hidden_parent / "voice-profile.json"
    service = _ActionProfileService(_profile(0))
    app = _ActionHost(service)
    messages: list[str] = []

    async with app.run_test(size=(150, 55)) as pilot:
        library, _selected = await _select_action_profile(app, pilot)

        async def _choose_path() -> Path:
            return target

        monkeypatch.setattr(library, "_choose_profile_export_path", _choose_path)
        sink = loguru_logger.add(
            lambda message: messages.append(str(message)),
            level="DEBUG",
        )
        try:
            assert await library.export_selected_profile() is False
        finally:
            loguru_logger.remove(sink)

        status_copy = _status_copy(app)

    assert secret not in "".join(messages)
    assert status_copy == profile_library_module.PROFILE_ACTION_FAILED_COPY


def _safe_bundle_review(
    *,
    dependency_state: str = "exact",
    allowed_choices: tuple[str, ...] = ("create", "reuse", "copy"),
    exact_private_duplicate: bool = True,
    uuid_conflict: bool = False,
    name_conflict: bool = False,
) -> TTSVoiceBundleReview:
    handle = object.__new__(TTSVoiceBundleHandle)
    return TTSVoiceBundleReview(
        handle=handle,
        profile_id=uuid4(),
        profile_name="Imported voice",
        provider_id="audio_cpp",
        model_id="model/a",
        voice_id="voice/a",
        recipe_id="recipe/a",
        recipe_revision=2,
        dependency_state=dependency_state,  # type: ignore[arg-type]
        allowed_choices=allowed_choices,  # type: ignore[arg-type]
        copy_profile_id=uuid4() if "copy" in allowed_choices else None,
        copy_profile_name="Imported voice copy" if "copy" in allowed_choices else None,
        exact_private_duplicate=exact_private_duplicate,
        uuid_conflict=uuid_conflict,
        name_conflict=name_conflict,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("uuid_conflict", "name_conflict"),
    ((False, False), (True, False), (False, True), (True, True)),
)
async def test_bundle_review_renders_explicit_safe_collision_facts(
    uuid_conflict: bool,
    name_conflict: bool,
) -> None:
    review = _safe_bundle_review(
        uuid_conflict=uuid_conflict,
        name_conflict=name_conflict,
    )

    class _ReviewHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(profile_library_module.TTSVoiceBundleReviewModal(review))

    app = _ReviewHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        facts = app.screen.query_one("#stts-bundle-review-facts", TextArea).text

        assert f"UUID conflict: {'yes' if uuid_conflict else 'no'}" in facts
        assert f"Name conflict: {'yes' if name_conflict else 'no'}" in facts


@pytest.mark.parametrize(
    ("choice", "dependency", "consent", "disabled"),
    (
        ("create", "exact", False, False),
        ("create", "exact", True, False),
        ("create", "missing", False, True),
        ("create", "missing", True, False),
        ("reuse", "exact", False, False),
        ("reuse", "exact", True, False),
        ("reuse", "missing", False, False),
        ("reuse", "missing", True, False),
        ("copy", "exact", False, False),
        ("copy", "exact", True, False),
        ("copy", "missing", False, True),
        ("copy", "missing", True, False),
    ),
)
def test_bundle_review_action_projection_is_the_single_operation_truth(
    choice: str,
    dependency: str,
    consent: bool,
    disabled: bool,
) -> None:
    review = _safe_bundle_review(dependency_state=dependency)
    action = voice_bundle_review_action(
        review,
        choice,  # type: ignore[arg-type]
        inactive_consent=consent,
    )

    assert action.operation == f"import_{choice}"
    assert action.disabled is disabled
    assert bool(action.recovery) is disabled


def test_bundle_review_never_reuses_a_nonexact_or_migrated_candidate() -> None:
    review = _safe_bundle_review(
        allowed_choices=("create", "copy"),
        exact_private_duplicate=False,
    )

    action = voice_bundle_review_action(review, "reuse", inactive_consent=False)

    assert action.operation == "import_reuse"
    assert action.disabled is True
    assert action.recovery == "Choose an available destination."


@pytest.mark.asyncio
async def test_export_choice_modal_renders_and_returns_only_supplied_projection() -> (
    None
):
    sanitized = VoiceBundleActionProjection(
        operation="sanitized_export",
        label="Safe metadata copy",
        tooltip="No private reference material.",
        disabled=False,
    )
    bundle = VoiceBundleActionProjection(
        operation="bundle_export",
        label="Private material bundle",
        tooltip="Includes plaintext private material.",
        disabled=False,
    )
    results: list[VoiceBundleActionProjection | None] = []

    class _ExportHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(
                TTSProfileExportChoiceModal(sanitized, bundle),
                results.append,
            )

    app = _ExportHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        safe = app.screen.query_one("#stts-export-choice-sanitized", Button)
        private = app.screen.query_one("#stts-export-choice-bundle", Button)
        assert (str(safe.label), safe.tooltip) == (
            sanitized.label,
            sanitized.tooltip,
        )
        assert (str(private.label), private.tooltip) == (
            bundle.label,
            bundle.tooltip,
        )

        safe.press()
        await pilot.pause()

        assert results == [sanitized]
        assert results[0] is not None
        assert results[0].operation == "sanitized_export"


@pytest.mark.asyncio
@pytest.mark.parametrize("dependency", ("exact", "missing", "mismatch", "pending"))
@pytest.mark.parametrize("choice", ("create", "reuse", "copy"))
@pytest.mark.parametrize("consent", (False, True))
async def test_bundle_review_mounted_matrix_uses_one_projection_and_service_choice(
    dependency: str,
    choice: str,
    consent: bool,
) -> None:
    review = _safe_bundle_review(
        dependency_state=dependency,
        allowed_choices=("create", "reuse", "copy"),
        exact_private_duplicate=True,
    )
    results: list[profile_library_module.VoiceBundleReviewDecision | None] = []

    class _ReviewHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(
                profile_library_module.TTSVoiceBundleReviewModal(review),
                results.append,
            )

    app = _ReviewHost()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        select = app.screen.query_one("#stts-bundle-review-choice", Select)
        select.value = choice
        await pilot.pause()
        consent_control = app.screen.query_one(
            "#stts-bundle-inactive-consent", Checkbox
        )
        consent_control.value = consent
        await pilot.pause()
        projected = voice_bundle_review_action(
            review,
            choice,  # type: ignore[arg-type]
            inactive_consent=consent,
        )
        allowed = choice != "reuse" or review.exact_private_duplicate
        needs_consent = choice in {"create", "copy"} and dependency != "exact"
        expected_recovery = (
            "Choose an available destination."
            if not allowed
            else (
                "Acknowledge that the imported profile will remain inactive."
                if needs_consent and not consent
                else None
            )
        )
        expected_label = {
            "create": "Create",
            "reuse": "Reuse",
            "copy": "Create copy",
        }[choice]
        expected_tooltip = (
            expected_recovery or "Confirm the reviewed import destination."
        )
        confirm = app.screen.query_one("#stts-bundle-review-confirm", Button)
        recovery = app.screen.query_one("#stts-bundle-review-recovery", Static)
        assert (str(confirm.label), confirm.tooltip, confirm.disabled) == (
            expected_label,
            expected_tooltip,
            projected.disabled,
        )
        assert projected.label == expected_label
        assert projected.tooltip == expected_tooltip
        assert projected.recovery == expected_recovery
        assert str(recovery.renderable) == (expected_recovery or "")

        confirm.press()
        await pilot.pause()
        if projected.disabled:
            assert results == []
        else:
            assert results and results[0] is not None
            assert results[0].choice == choice
            service_choice = voice_bundle_import_choice(
                projected,
                inactive_consent=consent,
            )
            assert service_choice.choice == choice
            assert service_choice.inactive_consent is consent


def test_export_action_projection_supplies_both_visible_and_executable_truths() -> None:
    sanitized, bundle = voice_bundle_export_actions(
        bundle_disabled=True,
        bundle_recovery="Use a supported platform.",
    )

    assert (
        sanitized.operation,
        sanitized.label,
        sanitized.tooltip,
        sanitized.disabled,
    ) == (
        "sanitized_export",
        "Export sanitized profile",
        "Export profile settings without voice audio or transcript.",
        False,
    )
    assert (
        bundle.operation,
        bundle.label,
        bundle.tooltip,
        bundle.disabled,
        bundle.recovery,
    ) == (
        "bundle_export",
        "Export portable voice bundle",
        "Use a supported platform.",
        True,
        "Use a supported platform.",
    )


@pytest.mark.asyncio
async def test_profile_export_rejects_a_hostile_non_export_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _clone_profile()
    app = _ActionHost(_ActionProfileService(profile))

    async with app.run_test(size=(80, 24)) as pilot:
        library, _loaded = await _select_action_profile(app, pilot)
        path_calls = 0
        hostile = VoiceBundleActionProjection(
            operation="import_create",
            label="CANARY wrong operation",
            tooltip="CANARY wrong operation",
            disabled=False,
        )

        async def _hostile_choice(_modal: object) -> VoiceBundleActionProjection:
            return hostile

        async def _forbidden_path() -> None:
            nonlocal path_calls
            path_calls += 1
            return None

        monkeypatch.setattr(library, "_push_owned_modal", _hostile_choice)
        monkeypatch.setattr(library, "_choose_profile_export_path", _forbidden_path)

        assert await library.export_selected_profile() is False
        assert path_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((80, 24), (100, 30)))
async def test_bundle_warning_acknowledgement_has_initial_focus_at_narrow_sizes(
    size: tuple[int, int],
) -> None:
    class _ConsentHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(TTSVoiceBundleConsentModal(mode="import"))

    app = _ConsentHost()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        assert app.focused is not None
        assert app.focused.id == "bundle-warning-ack"
        assert app.screen.query_one("#bundle-warning-continue", Button).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((80, 24), (100, 30)))
async def test_bundle_review_focus_order_is_facts_choice_consent_confirm_cancel(
    size: tuple[int, int],
) -> None:
    review = _safe_bundle_review(
        dependency_state="missing",
        allowed_choices=("create",),
        exact_private_duplicate=False,
    )

    class _ReviewHost(App[None]):
        def on_mount(self) -> None:
            self.push_screen(profile_library_module.TTSVoiceBundleReviewModal(review))

    app = _ReviewHost()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        expected = (
            "stts-bundle-review-facts",
            "stts-bundle-review-choice",
            "stts-bundle-inactive-consent",
            "stts-bundle-review-confirm",
            "stts-bundle-review-cancel",
        )
        observed: list[str | None] = [app.focused.id if app.focused else None]
        await pilot.press("tab")
        observed.append(app.focused.id if app.focused else None)
        await pilot.press("tab")
        observed.append(app.focused.id if app.focused else None)
        await pilot.press("space")
        for _ in range(2):
            await pilot.press("tab")
            observed.append(app.focused.id if app.focused else None)

        assert tuple(observed) == expected


@pytest.mark.asyncio
async def test_windows_disables_bundle_import_truthfully_but_keeps_json_export() -> (
    None
):
    profile = _profile(0)
    service = _ActionProfileService(profile)

    async def _profile_loader() -> object:
        return service

    async def _bundle_loader() -> object:
        raise AssertionError("disabled import must not resolve the service")

    class _WindowsHost(App[None]):
        def compose(self) -> ComposeResult:
            yield STTSProfileLibrary(
                _profile_loader,
                voice_bundle_service_loader=_bundle_loader,
                bundle_platform_supported=False,
            )

    app = _WindowsHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await _wait_until(
            pilot,
            lambda: app.query_one("#stts-profile-table", DataTable).row_count == 1,
        )
        table = app.query_one("#stts-profile-table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        table.action_select_cursor()
        await pilot.pause()

        assert app.query_one("#stts-profile-export-btn", Button).disabled is False
        import_button = app.query_one("#stts-profile-import-btn", Button)
        assert import_button.disabled is True
        assert "Windows" in str(import_button.tooltip)


def _clone_profile(index: int = 0) -> TTSGenerationProfile:
    profile = _profile(index)
    timestamp = datetime(2026, 7, 27, tzinfo=UTC)
    requirement = TTSCloneRecipeRequirement(
        recipe_id="audio_cpp_clone_v1",
        recipe_revision=1,
        model_id=profile.model_id,
    )
    return replace(
        profile,
        reference=TTSCloneReferenceSummary(
            reference_id=uuid4(),
            byte_length=32000,
            duration_ms=1000,
            sample_rate_hz=16000,
            channels=1,
            sample_encoding="pcm_s16le",
            created_at=timestamp,
            updated_at=timestamp,
            recipe_requirement=requirement,
        ),
    )


@pytest.mark.asyncio
async def test_windows_clone_export_keeps_sanitized_default_and_disables_bundle() -> (
    None
):
    profile = _clone_profile()
    service = _ActionProfileService(profile)

    async def _profile_loader() -> object:
        return service

    async def _bundle_loader() -> object:
        raise AssertionError("disabled bundle export must not resolve the service")

    class _WindowsCloneHost(App[None]):
        def compose(self) -> ComposeResult:
            yield STTSProfileLibrary(
                _profile_loader,
                voice_bundle_service_loader=_bundle_loader,
                bundle_platform_supported=False,
            )

    app = _WindowsCloneHost()
    async with app.run_test(size=(80, 24)) as pilot:
        await _wait_until(
            pilot,
            lambda: app.query_one("#stts-profile-table", DataTable).row_count == 1,
        )
        table = app.query_one("#stts-profile-table", DataTable)
        table.focus()
        table.move_cursor(row=0)
        table.action_select_cursor()
        await pilot.pause()
        app.query_one("#stts-profile-export-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: len(app.screen.query("#stts-export-choice-bundle")) == 1,
        )

        sanitized = app.screen.query_one("#stts-export-choice-sanitized", Button)
        bundle = app.screen.query_one("#stts-export-choice-bundle", Button)
        assert app.focused is sanitized
        assert sanitized.disabled is False
        assert bundle.disabled is True
        assert "Windows" in str(bundle.tooltip)


class _BundleUIService:
    def __init__(self, profile: TTSGenerationProfile) -> None:
        self.profile = profile
        self.exports: list[tuple[object, ...]] = []
        self.inspect_calls: list[Path] = []
        self.commit_calls: list[
            tuple[TTSVoiceBundleHandle, TTSVoiceBundleImportChoice]
        ] = []
        self.invalidated: list[TTSVoiceBundleHandle] = []
        self.reviews = [self._review()]
        self.results: list[TTSVoiceBundleImportResult] = [
            TTSVoiceBundleImportResult(status="created", profile=profile)
        ]

    def _review(self) -> TTSVoiceBundleReview:
        return TTSVoiceBundleReview(
            handle=object.__new__(TTSVoiceBundleHandle),
            profile_id=self.profile.profile_id,
            profile_name=self.profile.display_name,
            provider_id=self.profile.provider_id,
            model_id=self.profile.model_id,
            voice_id=self.profile.voice_id,
            recipe_id="audio_cpp_clone_v1",
            recipe_revision=1,
            dependency_state="exact",
            allowed_choices=("create",),
            copy_profile_id=None,
            copy_profile_name=None,
            exact_private_duplicate=False,
        )

    async def export(
        self,
        profile_id: UUID,
        destination: Path,
        *,
        expected_generation: int,
        expected_revision: int,
        acknowledged: bool,
    ) -> None:
        self.exports.append(
            (
                profile_id,
                destination,
                expected_generation,
                expected_revision,
                acknowledged,
            )
        )

    async def inspect(self, source: Path) -> TTSVoiceBundleReview:
        self.inspect_calls.append(source)
        return self.reviews.pop(0)

    async def commit(
        self,
        handle: TTSVoiceBundleHandle,
        choice: TTSVoiceBundleImportChoice,
    ) -> TTSVoiceBundleImportResult:
        self.commit_calls.append((handle, choice))
        return self.results.pop(0)

    async def invalidate(self, handle: TTSVoiceBundleHandle) -> None:
        self.invalidated.append(handle)


class _BundleActionHost(_ActionHost):
    def __init__(self, service: object, bundle_service: _BundleUIService) -> None:
        super().__init__(service)
        self.bundle_service = bundle_service
        self.bundle_service_requests = 0

    async def _ensure_tts_voice_bundle_service(self) -> object:
        self.bundle_service_requests += 1
        return self.bundle_service


class _LateInspectBundleService(_BundleUIService):
    def __init__(self, profile: TTSGenerationProfile) -> None:
        super().__init__(profile)
        self.inspect_started = asyncio.Event()

    async def inspect(self, source: Path) -> TTSVoiceBundleReview:
        self.inspect_calls.append(source)
        self.inspect_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            return self.reviews.pop(0)


class _ReviewPushAbort(BaseException):
    """Non-Exception terminal used to prove finally-owned invalidation."""


@pytest.mark.asyncio
async def test_reference_export_defaults_sanitized_and_bundle_requires_ack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = _clone_profile()
    service = _ActionProfileService(profile)
    bundle_service = _BundleUIService(profile)
    app = _BundleActionHost(service, bundle_service)
    destination = tmp_path / "voice.tldw-voice.zip"

    async with app.run_test(size=(80, 24)) as pilot:
        library, loaded = await _select_action_profile(app, pilot)

        async def _destination() -> Path:
            return destination

        monkeypatch.setattr(library, "_choose_voice_bundle_export_path", _destination)
        app.query_one("#stts-profile-export-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: len(app.screen.query("#stts-export-choice-sanitized")) == 1,
        )
        assert app.focused is not None
        assert app.focused.id == "stts-export-choice-sanitized"
        app.screen.query_one("#stts-export-choice-bundle", Button).press()
        await _wait_until(
            pilot,
            lambda: len(app.screen.query("#bundle-warning-ack")) == 1,
        )
        assert bundle_service.exports == []
        assert app.focused is not None and app.focused.id == "bundle-warning-ack"
        app.screen.query_one("#bundle-warning-ack", Checkbox).toggle()
        await pilot.pause()
        app.screen.query_one("#bundle-warning-continue", Button).press()

        await _wait_until(pilot, lambda: len(bundle_service.exports) == 1)
        assert bundle_service.exports == [
            (
                profile.profile_id,
                destination,
                loaded.repository_generation,
                profile.revision,
                True,
            )
        ]


@pytest.mark.asyncio
async def test_bundle_export_failure_never_surfaces_private_canaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from loguru import logger as loguru_logger

    canaries = (
        "CANARY-source-path",
        "CANARY-destination-path",
        "CANARY-staging-path",
        "manifest.json",
        "CANARY-transcript",
        "CANARY-wav-bytes",
        "CANARY-checksum",
        "CANARY-provider-origin",
        "CANARY-generated-config",
        "CANARY-collaborator-error",
    )
    profile = _clone_profile()
    service = _ActionProfileService(profile)
    bundle_service = _BundleUIService(profile)
    app = _BundleActionHost(service, bundle_service)
    destination = tmp_path / "voice.tldw-voice.zip"
    messages: list[str] = []

    async def _fail_export(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(" ".join(canaries))

    bundle_service.export = _fail_export  # type: ignore[method-assign]

    async with app.run_test(size=(80, 24)) as pilot:
        library, loaded = await _select_action_profile(app, pilot)

        async def _destination() -> Path:
            return destination

        async def _acknowledge(_modal: object) -> bool:
            return True

        monkeypatch.setattr(library, "_choose_voice_bundle_export_path", _destination)
        monkeypatch.setattr(library, "_push_owned_modal", _acknowledge)
        sink = loguru_logger.add(
            lambda message: messages.append(str(message)),
            level="DEBUG",
        )
        try:
            assert await library._export_selected_voice_bundle(loaded) is False
        finally:
            loguru_logger.remove(sink)
        status_copy = _status_copy(app)

    rendered = "\n".join((*messages, status_copy))
    assert status_copy == profile_library_module.PROFILE_ACTION_FAILED_COPY
    assert all(canary not in rendered for canary in canaries)


@pytest.mark.asyncio
async def test_import_warns_before_picker_and_stale_successor_requires_reconfirm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = _profile(0)
    service = _ActionProfileService(profile)
    bundle_service = _BundleUIService(profile)
    successor = bundle_service._review()
    bundle_service.results = [
        TTSVoiceBundleImportResult(status="stale_inspection", review=successor),
        TTSVoiceBundleImportResult(status="created", profile=profile),
    ]
    app = _BundleActionHost(service, bundle_service)
    source = tmp_path / "voice.tldw-voice.zip"
    picker_calls = 0

    async with app.run_test(size=(100, 30)) as pilot:
        library, _loaded = await _select_action_profile(app, pilot)

        async def _source() -> Path:
            nonlocal picker_calls
            picker_calls += 1
            return source

        monkeypatch.setattr(library, "_choose_voice_bundle_import_path", _source)
        app.query_one("#stts-profile-import-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: len(app.screen.query("#bundle-warning-ack")) == 1,
        )
        assert picker_calls == 0
        app.screen.query_one("#bundle-warning-ack", Checkbox).toggle()
        await pilot.pause()
        app.screen.query_one("#bundle-warning-continue", Button).press()
        await _wait_until(
            pilot,
            lambda: len(app.screen.query("#stts-bundle-review-confirm")) == 1,
        )
        assert picker_calls == 1
        assert bundle_service.inspect_calls == [source]
        app.screen.query_one("#stts-bundle-review-confirm", Button).press()
        await _wait_until(pilot, lambda: len(bundle_service.commit_calls) == 1)
        await _wait_until(
            pilot,
            lambda: (
                len(app.screen.query("#stts-bundle-review-confirm")) == 1
                and app.screen.query_one("#stts-bundle-review-confirm", Button).disabled
                is False
            ),
        )
        assert len(bundle_service.commit_calls) == 1
        app.screen.query_one("#stts-bundle-review-confirm", Button).press()

        await _wait_until(pilot, lambda: len(bundle_service.commit_calls) == 2)
        assert [call[1].choice for call in bundle_service.commit_calls] == [
            "create",
            "create",
        ]
        assert bundle_service.commit_calls[0][0] is not successor.handle
        assert bundle_service.commit_calls[1][0] is successor.handle


@pytest.mark.asyncio
async def test_import_review_cancel_invalidates_handle_without_committing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = _profile(0)
    service = _ActionProfileService(profile)
    bundle_service = _BundleUIService(profile)
    review_handle = bundle_service.reviews[0].handle
    app = _BundleActionHost(service, bundle_service)

    async with app.run_test(size=(80, 24)) as pilot:
        library, _loaded = await _select_action_profile(app, pilot)

        async def _source() -> Path:
            return tmp_path / "voice.tldw-voice.zip"

        monkeypatch.setattr(library, "_choose_voice_bundle_import_path", _source)
        app.query_one("#stts-profile-import-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: len(app.screen.query("#bundle-warning-ack")) == 1,
        )
        app.screen.query_one("#bundle-warning-ack", Checkbox).toggle()
        await pilot.pause()
        app.screen.query_one("#bundle-warning-continue", Button).press()
        await _wait_until(
            pilot,
            lambda: len(app.screen.query("#stts-bundle-review-cancel")) == 1,
        )
        app.screen.query_one("#stts-bundle-review-cancel", Button).press()
        await _wait_until(pilot, lambda: bundle_service.invalidated == [review_handle])

        assert bundle_service.commit_calls == []
        assert library._active_bundle_handle is None


@pytest.mark.asyncio
async def test_repeated_import_cancels_release_completed_invalidation_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = _profile(0)
    service = _ActionProfileService(profile)
    bundle_service = _BundleUIService(profile)
    bundle_service.reviews = [bundle_service._review() for _ in range(20)]
    app = _BundleActionHost(service, bundle_service)

    async with app.run_test(size=(80, 24)) as pilot:
        library, _loaded = await _select_action_profile(app, pilot)

        async def _source() -> Path:
            return tmp_path / "voice.tldw-voice.zip"

        monkeypatch.setattr(library, "_choose_voice_bundle_import_path", _source)
        for expected in range(1, 21):
            app.query_one("#stts-profile-import-btn", Button).press()
            await _wait_until(
                pilot,
                lambda: len(app.screen.query("#bundle-warning-ack")) == 1,
            )
            app.screen.query_one("#bundle-warning-ack", Checkbox).toggle()
            await pilot.pause()
            app.screen.query_one("#bundle-warning-continue", Button).press()
            await _wait_until(
                pilot,
                lambda: len(app.screen.query("#stts-bundle-review-cancel")) == 1,
            )
            app.screen.query_one("#stts-bundle-review-cancel", Button).press()
            await _wait_until(
                pilot,
                lambda: len(bundle_service.invalidated) == expected,
            )

        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(bundle_service.invalidated) == 20
        assert len(set(bundle_service.invalidated)) == 20
        assert library._bundle_invalidation_tasks == {}


@pytest.mark.asyncio
async def test_unmount_racing_existing_invalidation_joins_once_and_releases_task() -> (
    None
):
    profile = _profile(0)
    bundle_service = _BundleUIService(profile)
    handle = bundle_service.reviews[0].handle
    started = asyncio.Event()
    release = asyncio.Event()

    async def _delayed_invalidate(candidate: TTSVoiceBundleHandle) -> None:
        started.set()
        await release.wait()
        bundle_service.invalidated.append(candidate)

    bundle_service.invalidate = _delayed_invalidate  # type: ignore[method-assign]
    app = _BundleActionHost(_ActionProfileService(profile), bundle_service)
    async with app.run_test(size=(80, 24)) as pilot:
        library, _loaded = await _select_action_profile(app, pilot)
        library._voice_bundle_service = bundle_service
        library._active_bundle_handle = handle
        cleanup = asyncio.create_task(library._invalidate_bundle_handle(handle))
        await wait_for_background_signal(
            started, cleanup, what="the bundle invalidation starting"
        )

        async def _remove_library() -> None:
            await library.remove()

        removal = asyncio.create_task(_remove_library())
        await pilot.pause()
        release.set()
        await asyncio.gather(cleanup, removal)

        assert bundle_service.invalidated == [handle]
        assert library._bundle_invalidation_tasks == {}


@pytest.mark.asyncio
async def test_completed_invalidation_cannot_clear_a_replacement_task() -> None:
    profile = _profile(0)
    bundle_service = _BundleUIService(profile)
    handle = bundle_service.reviews[0].handle
    started = asyncio.Event()
    release = asyncio.Event()

    async def _delayed_invalidate(candidate: TTSVoiceBundleHandle) -> None:
        started.set()
        await release.wait()
        bundle_service.invalidated.append(candidate)

    bundle_service.invalidate = _delayed_invalidate  # type: ignore[method-assign]
    app = _BundleActionHost(_ActionProfileService(profile), bundle_service)
    async with app.run_test(size=(80, 24)) as pilot:
        library, _loaded = await _select_action_profile(app, pilot)
        library._voice_bundle_service = bundle_service
        cleanup = asyncio.create_task(library._invalidate_bundle_handle(handle))
        await wait_for_background_signal(
            started, cleanup, what="the bundle invalidation starting"
        )
        original = library._bundle_invalidation_tasks[handle]
        replacement = asyncio.create_task(asyncio.sleep(60))
        library._bundle_invalidation_tasks[handle] = replacement

        release.set()
        await cleanup
        await pilot.pause()

        assert library._bundle_invalidation_tasks.get(handle) is replacement
        assert original.done()
        replacement.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replacement
        library._bundle_invalidation_tasks.pop(handle)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", (RuntimeError, _ReviewPushAbort))
async def test_import_review_push_failure_invalidates_handle_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_type: type[BaseException],
) -> None:
    profile = _profile(0)
    service = _ActionProfileService(profile)
    bundle_service = _BundleUIService(profile)
    review_handle = bundle_service.reviews[0].handle
    app = _BundleActionHost(service, bundle_service)

    async with app.run_test(size=(80, 24)) as pilot:
        library, _loaded = await _select_action_profile(app, pilot)

        async def _source() -> Path:
            return tmp_path / "voice.tldw-voice.zip"

        original_push = library._push_owned_modal

        async def _push(modal: object) -> object:
            if isinstance(modal, profile_library_module.TTSVoiceBundleReviewModal):
                raise failure_type("CANARY review push failure")
            return await original_push(modal)  # type: ignore[arg-type]

        monkeypatch.setattr(library, "_choose_voice_bundle_import_path", _source)
        monkeypatch.setattr(library, "_push_owned_modal", _push)
        app.query_one("#stts-profile-import-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: len(app.screen.query("#bundle-warning-ack")) == 1,
        )
        app.screen.query_one("#bundle-warning-ack", Checkbox).toggle()
        await pilot.pause()
        app.screen.query_one("#bundle-warning-continue", Button).press()
        expected_failure = (
            _ReviewPushAbort if failure_type is _ReviewPushAbort else WorkerFailed
        )
        with pytest.raises(expected_failure):
            await app.workers.wait_for_complete()

        assert bundle_service.invalidated == [review_handle]
        assert library._active_bundle_handle is None
        assert bundle_service.commit_calls == []


@pytest.mark.asyncio
async def test_cancelled_handle_cleanup_joins_without_masking_cancellation() -> None:
    profile = _profile(0)
    bundle_service = _BundleUIService(profile)
    handle = bundle_service.reviews[0].handle
    started = asyncio.Event()
    release = asyncio.Event()

    async def _failing_invalidate(candidate: TTSVoiceBundleHandle) -> None:
        started.set()
        await release.wait()
        bundle_service.invalidated.append(candidate)
        raise RuntimeError("CANARY cleanup failure")

    bundle_service.invalidate = _failing_invalidate  # type: ignore[method-assign]
    app = _BundleActionHost(_ActionProfileService(profile), bundle_service)
    async with app.run_test(size=(80, 24)) as pilot:
        library, _loaded = await _select_action_profile(app, pilot)
        library._voice_bundle_service = bundle_service
        cleanup = asyncio.create_task(library._invalidate_bundle_handle(handle))
        await wait_for_background_signal(
            started, cleanup, what="the bundle invalidation starting"
        )
        cleanup.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await cleanup
        await pilot.pause()

        assert bundle_service.invalidated == [handle]


@pytest.mark.asyncio
async def test_unmount_invalidates_handle_returned_by_late_inspection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = _profile(0)
    service = _ActionProfileService(profile)
    bundle_service = _LateInspectBundleService(profile)
    late_handle = bundle_service.reviews[0].handle
    app = _BundleActionHost(service, bundle_service)

    async with app.run_test(size=(80, 24)) as pilot:
        library, _loaded = await _select_action_profile(app, pilot)

        async def _source() -> Path:
            return tmp_path / "voice.tldw-voice.zip"

        monkeypatch.setattr(library, "_choose_voice_bundle_import_path", _source)
        app.query_one("#stts-profile-import-btn", Button).press()
        await _wait_until(
            pilot,
            lambda: len(app.screen.query("#bundle-warning-ack")) == 1,
        )
        app.screen.query_one("#bundle-warning-ack", Checkbox).toggle()
        await pilot.pause()
        app.screen.query_one("#bundle-warning-continue", Button).press()
        await asyncio.wait_for(bundle_service.inspect_started.wait(), timeout=1)

        await library.remove()
        await _wait_until(pilot, lambda: bundle_service.invalidated == [late_handle])

        assert bundle_service.commit_calls == []
        assert len(app.screen.query("#stts-bundle-review-confirm")) == 0
