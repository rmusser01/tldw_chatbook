from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Static, TextArea

from tldw_chatbook.TTS import (
    LoadedTTSProfile,
    ProfileRepositoryError,
    STTSGeneratedAudio,
    TTSGenerationProfile,
    TTSPlaygroundSelectionPreset,
    TTSProfileAvailability,
    TTSProfileAvailabilitySnapshot,
    TTSProfileDraft,
    TTSProfilePageSnapshot,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.profile_portability import (
    PortableTTSProfile,
    portable_profile_payload,
)
from tldw_chatbook.UI import STTS_Window as stts_window_module
from tldw_chatbook.UI import stts_profile_library as profile_library_module
from tldw_chatbook.UI.Dictation_Window_Improved import ImprovedDictationWindow
from tldw_chatbook.UI.stts_profile_library import (
    PROFILE_STORE_UNAVAILABLE_COPY,
    STTSProfileLibrary,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_settings_pane import SpeechSettingsPane
from tldw_chatbook.UI.STTS_Window import (
    AudioBookGenerationWidget,
    STTSWindow,
)


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
    ) -> None:
        self.page = _page(profile, generation=11)
        self.availability_state = availability_state
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
        self.assignment_total = 0
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
        return _availability(page, state=self.availability_state)

    async def create_from_artifact(
        self,
        display_name: str,
        artifact: STTSGeneratedAudio,
    ) -> LoadedTTSProfile:
        self.create_calls.append((display_name, artifact))
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

    def on_profile_preview_requested(self, message: object) -> None:
        self.preview_messages.append(message)


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
) -> TTSProfileAvailabilitySnapshot:
    recovery = {
        "available": "none",
        "unavailable": "edit",
        "unverified": "refresh",
    }[state]
    return TTSProfileAvailabilitySnapshot(
        repository_generation=page.repository_generation,
        configuration_revision=configuration_revision,
        catalog_revision=catalog_revision,
        profiles=tuple(
            TTSProfileAvailability(
                profile_id=profile.profile_id,
                state=state,  # type: ignore[arg-type]
                recovery_action=recovery,  # type: ignore[arg-type]
            )
            for profile in page.profiles
        ),
    )


def _table_cell(table: DataTable[Any], row: int, column: int) -> str:
    return str(table.get_row_at(row)[column])


def _visible_content_rows(widget: Widget) -> tuple[str, ...]:
    strips = widget.screen._compositor.render_strips()
    region = widget.content_region
    return tuple(
        "".join(segment.text for segment in strips[y])[
            region.x : region.x + region.width
        ].strip()
        for y in range(region.y, region.y + region.height)
    )


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
            "#stts-profile-preview-btn",
            "#stts-profile-edit-btn",
            "#stts-profile-duplicate-btn",
            "#stts-profile-delete-btn",
        ):
            assert not app.query_one(selector, Button).disabled

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
        ("unverified", "Unverified — Refresh and retry."),
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
                lambda: _table_cell(table, 0, 3) == "Available",
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
            assert _table_cell(table, 0, 3) == "Available"
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
        assert _table_cell(table, 0, 3) == "Available"

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
        assert _table_cell(table, 0, 3) == "Available"

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
        assert _table_cell(table, 0, 3) == "Available"

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

        await pilot.click("#stts-profile-preview-btn")
        await _wait_until(pilot, lambda: len(app.preview_messages) == 1)

        message = app.preview_messages[0]
        assert isinstance(
            message,
            profile_library_module.ProfilePreviewRequested,
        )
        current_availability = library._row_availability[
            str(selected.profile.profile_id)
        ]
        assert service.preview_preset_calls == [(selected, current_availability)]
        assert (
            message.preset.model_id,
            message.preset.voice_id,
            message.preset.availability,
        ) == (
            selected.profile.model_id,
            selected.profile.voice_id,
            current_availability.state,
        )


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

    monkeypatch.setattr(
        stts_window_module,
        "get_tts_service",
        _unavailable_tts_service,
    )

    async with app.run_test(size=(150, 55)) as pilot:
        library, selected = await _select_action_profile(app, pilot)
        await _wait_until(
            pilot,
            lambda: str(selected.profile.profile_id) in library._row_availability,
        )

        await pilot.click("#stts-profile-preview-btn")
        await _wait_until(
            pilot,
            lambda: app.query_one(STTSWindow).current_view == "playground",
        )
        first_playground = app.query_one(SpeechPlaygroundPane)
        window = app.query_one(STTSWindow)

        assert first_playground._profile_preset is not None
        assert first_playground._profile_preset.model_id == "missing/exact-model"
        assert first_playground._profile_preset.voice_id == "missing/exact-voice"
        assert window._pending_playground_preset is None

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
        availability_state="unavailable",
    )
    app = _ActionHost(service)

    async def _unavailable_tts_service() -> object:
        raise RuntimeError("catalog deliberately unavailable")

    monkeypatch.setattr(
        stts_window_module,
        "get_tts_service",
        _unavailable_tts_service,
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
            lambda: app.query_one(STTSWindow).current_view == "playground",
        )
        await pilot.pause()

        playground = app.query_one(SpeechPlaygroundPane)
        text_input = playground.query_one("#tts-text-input", TextArea)
        banner = playground.query_one("#tts-profile-preview-status", Static)
        copy = str(banner.render())

        assert app.focused is text_input
        assert banner.display is True
        assert "unavailable" in copy.lower()
        assert "Edit" in copy
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
