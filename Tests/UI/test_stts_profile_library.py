from __future__ import annotations

import asyncio
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
    TTSProfileAvailability,
    TTSProfileAvailabilitySnapshot,
    TTSProfileDraft,
    TTSProfilePageSnapshot,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.UI import stts_profile_library as profile_library_module
from tldw_chatbook.UI.Dictation_Window_Improved import ImprovedDictationWindow
from tldw_chatbook.UI.stts_profile_library import (
    PROFILE_STORE_UNAVAILABLE_COPY,
    STTSProfileLibrary,
)
from tldw_chatbook.UI.STTS_Window import (
    AudioBookGenerationWidget,
    STTSWindow,
    TTSPlaygroundWidget,
    TTSSettingsWidget,
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


async def _select_action_profile(
    app: _ActionHost,
    pilot: Any,
) -> tuple[STTSProfileLibrary, LoadedTTSProfile]:
    await pilot.click("#view-profiles-btn")
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
async def test_voice_profiles_sidebar_mounts_focused_library_without_hiding_legacy_views(
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
            TTSPlaygroundWidget,
        )
        assert app.query_one("#tts-generate-btn", Button)

        await pilot.click("#view-profiles-btn")
        await _wait_until(pilot, lambda: bool(service.list_calls))
        assert app.query_one(STTSProfileLibrary)

        await pilot.click("#view-settings-btn")
        await pilot.pause()
        assert isinstance(
            app.query_one(".stts-content").children[0],
            TTSSettingsWidget,
        )

        await pilot.click("#view-audiobook-btn")
        await pilot.pause()
        assert isinstance(
            app.query_one(".stts-content").children[0],
            AudioBookGenerationWidget,
        )

        await pilot.click("#view-stt-btn")
        await pilot.pause()
        assert isinstance(
            app.query_one(".stts-content").children[0],
            ImprovedDictationWindow,
        )

        await pilot.click("#view-playground-btn")
        await pilot.pause()
        assert isinstance(
            app.query_one(".stts-content").children[0],
            TTSPlaygroundWidget,
        )
        assert app.query_one("#tts-generate-btn", Button)


@pytest.mark.asyncio
async def test_profile_store_unavailable_isolated_to_stable_library_recovery() -> None:
    app = _STTSHost(None)

    async with app.run_test(size=(150, 55)) as pilot:
        await pilot.click("#view-profiles-btn")
        await _wait_until(pilot, lambda: app.profile_service_requests == 1)

        status = app.query_one("#stts-profile-status", TextArea)
        assert status.text == PROFILE_STORE_UNAVAILABLE_COPY
        assert app.query_one("#stts-profile-table", DataTable).row_count == 0
        assert not app.query_one("#stts-profile-refresh-btn", Button).disabled

        await pilot.click("#view-playground-btn")
        await pilot.pause()
        assert app.query_one("#tts-generate-btn", Button)


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
        await pilot.click("#view-profiles-btn")
        await _wait_until(pilot, lambda: len(failed_service.list_calls) == 1)

        status = app.query_one("#stts-profile-status", TextArea)
        await _wait_until(
            pilot,
            lambda: status.text == PROFILE_STORE_UNAVAILABLE_COPY,
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
        await pilot.click("#view-profiles-btn")
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
        status = app.query_one("#stts-profile-status", TextArea)
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
        status = app.query_one("#stts-profile-status", TextArea)

        assert _visible_content_rows(status) == (
            expected_recovery,
            "Selected: Voice 00",
            "audio_cpp / model/0 / voice/0",
        )


@pytest.mark.asyncio
async def test_long_profile_identifiers_are_keyboard_scrollable_at_80x24() -> None:
    model_id = f"model/{'opaque-model-segment/' * 5}model-tail"
    voice_id = f"voice/{'opaque-voice-segment/' * 5}voice-tail"
    profile = replace(_profile(0), model_id=model_id, voice_id=voice_id)
    service = _ActionProfileService(profile, availability_state="unavailable")
    app = _ActionHost(service)
    detail_copy = f"audio_cpp / {model_id} / {voice_id}"

    async with app.run_test(size=(80, 24)) as pilot:
        await _select_action_profile(app, pilot)
        status = app.query_one("#stts-profile-status")
        visible_lines = _visible_content_rows(status)

        assert visible_lines[0] == "Unavailable — Refresh, then Edit."
        assert visible_lines[1] == "Selected: Voice 00"
        assert visible_lines[2].startswith("audio_cpp / model/opaque-model")
        assert status.region.height <= 5

        app.query_one("#stts-profile-table", DataTable).focus()
        for _ in range(8):
            await pilot.press("tab")
            if app.focused is status:
                break
        assert app.focused is status
        assert isinstance(status, TextArea)
        assert status.read_only
        assert not status.soft_wrap
        assert status.text == (
            f"Unavailable — Refresh, then Edit.\nSelected: Voice 00\n{detail_copy}"
        )

        await pilot.press("down", "down", "end")
        await _wait_until(pilot, lambda: status.scroll_x > 0)
        assert _visible_content_rows(status)[2].endswith("voice-tail")

        unchanged = status.text
        await pilot.press("x")
        assert status.text == unchanged

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
        await pilot.click("#view-profiles-btn")
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
        await pilot.click("#view-profiles-btn")
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
        preview.press()
        await _wait_until(pilot, lambda: bool(app.preview_messages))

        assert (
            stale_selected,
            stale_disabled,
            target_is_current,
            stale_preview_count,
        ) == (None, (True, True, True, True), False, 0)
        assert current_loaded is not None
        assert current_loaded.repository_generation == 6
        assert current_disabled == (False, False, False, False)
        assert library._action_target_is_current(current_loaded)
        message = app.preview_messages[0]
        assert isinstance(
            message,
            profile_library_module.ProfilePreviewRequested,
        )
        assert message.loaded is current_loaded


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
            await pilot.click("#view-profiles-btn")
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

            await pilot.click("#view-settings-btn")
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
        await pilot.click("#view-profiles-btn")
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
        await pilot.click("#view-profiles-btn")
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
            await pilot.click("#view-profiles-btn")
            await _wait_until(pilot, lambda: len(service.list_futures) == 1)
            library = app.query_one(STTSProfileLibrary)

            await pilot.click("#view-settings-btn")
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
        assert message.loaded is selected
        assert (
            message.availability
            is library._row_availability[str(selected.profile.profile_id)]
        )
        assert not hasattr(service, "preview_preset_calls")


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
                TTSSettingsWidget,
            ),
        )
        await _wait_until(
            pilot,
            lambda: all(worker.is_finished for worker in action_workers),
        )

        assert app.screen is not modal
        assert all(worker.is_finished for worker in action_workers)


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
        assert (
            app.query_one("#stts-profile-status", TextArea).text
            == profile_library_module.PROFILE_CONFLICT_COPY
        )

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
        assert (
            "must never enter profile UI copy"
            not in app.query_one("#stts-profile-status", TextArea).text
        )


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
            return True

        monkeypatch.setattr(app, "push_screen_wait", _confirm)
        assert await library.delete_selected_profile() is False

        assert service.assignment_count_calls[0] is selected
        assert service.delete_calls[0] is selected
        assert (
            app.query_one("#stts-profile-status", TextArea).text
            == profile_library_module.PROFILE_CONFLICT_COPY
        )


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
            return True

        monkeypatch.setattr(app, "push_screen_wait", _confirm)
        assert await library.delete_selected_profile() is False

        assert service.assignment_count_calls[0] is selected
        assert service.delete_calls == []
        assert (
            app.query_one("#stts-profile-status", TextArea).text
            == profile_library_module.PROFILE_DELETE_PROTECTED_COPY
        )


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

        detail = app.query_one("#stts-profile-status", TextArea).text
        assert "missing/exact-model" in detail
        assert "missing/exact-voice" in detail
        assert "Refresh, then Edit" in detail

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

        status = app.query_one("#stts-profile-status", TextArea).text
        assert status == profile_library_module.PROFILE_ACTION_FAILED_COPY
        assert "/Users/private/key" not in status
        assert "upstream.invalid" not in status
