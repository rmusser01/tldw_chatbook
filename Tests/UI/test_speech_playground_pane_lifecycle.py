"""TASK-2951 port: the retired legacy playground widget's remaining
behavioral coverage, proven against the live `SpeechPlaygroundPane`.

`tldw_chatbook/UI/STTS_Window.py` used to carry a duplicate, independently
written TTS Playground widget -- dead code (never mounted in production) by
the time this port happened, deleted in the same branch along with its
dedicated test file, `Tests/UI/test_stts_playground_audio_cpp.py` (94
tests, all mounting only that widget). A prior research pass confirmed
several mechanisms named in that file's tests -- `_pending_voice_selections`,
`_remember_current_controls`, `_control_snapshot_for`,
`_catalog_request_generations`, the "stts-playback" worker group,
`_rehydrate_handler_state`, and the `controls_from_profile_preset`/
`profile_availability_from_catalog` pure functions -- exist verbatim in the
still-live replacement, `SpeechPlaygroundPane`
(`tldw_chatbook/UI/Speech/speech_playground_pane.py`, built from
`SpeechSynthesisMixin`, `SpeechCatalogMixin`, `SpeechPlaybackMixin` and
`SpeechProfileMixin`) -- independently written, not inherited, but
functionally parallel. This file ports the tests that exercise those shared
mechanisms so their guarantee is proven against the pane that is actually
reachable from the app, not the deleted widget.

3 of the widget file's tests were already covered by the pane's own suite
and are not ported again here (`test_unavailable_profile_preset_disables_
generation_and_points_to_edit`, `test_profile_service_acquisition_failure_
projects_exact_disabled_recovery`,
`test_save_profile_action_is_hidden_for_legacy_failed_and_in_progress_
states`). 14 more were RED against the widget itself (dead code) and are
excluded outright. Two pure-function tests
(`test_profile_preset_projection_*`) were ported into
`Tests/UI/test_stts_playground_catalog.py` instead, next to that module's
existing `controls_from_profile_preset` coverage.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Select, Static, TextArea

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.TTS import (
    ProfileRepositoryError,
    ProfileServiceError,
    TTSPlaygroundSelectionPreset,
)
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.audio_player import PlaybackState
from tldw_chatbook.TTS.playground_types import STTSGeneratedAudio
from tldw_chatbook.UI import stts_profile_library as profile_library_module
from tldw_chatbook.UI.stts_playground_catalog import (
    LOADING_SELECT_VALUE,
    SERVER_DEFAULT_VOICE_ID,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane

from Tests.UI.app_factory import _build_test_app
from Tests.UI.speech_playground_fixtures import (
    FakeTTSService,
    _native_profile_artifact,
    _profile_preset,
    _resolved,
    _wait_until,
)

#: Every provider id the fake service and catalogs below know about. Copied
#: locally (not imported) per the porting plan: it names a local test
#: constant, not a shared fixture.
PROVIDER_IDS = (
    "audio_cpp",
    "openai",
    "elevenlabs",
    "kokoro",
    "chatterbox",
    "higgs",
    "alltalk",
)


def _audio_catalog(
    *,
    health: ProviderHealth | None = None,
    revision: int = 11,
) -> TTSProviderCatalog:
    """Copied verbatim from the retired widget's test file (and identical to
    the private helper `FakeTTSService` builds internally in
    `speech_playground_fixtures.py`) so tests here can swap in a differently
    shaped audio_cpp catalog (stale health, a new revision, ...).
    """
    return TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=revision,
        health=health or ProviderHealth(state="available", fresh=True),
        models=(
            TTSModelInfo(
                model_id="<opaque:model>",
                display_name="[bold red]Opaque model[/]",
                family="test",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
            TTSModelInfo(
                model_id="second-model",
                display_name="Second model",
                family="test",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
        ),
    )


def _option_values(select: Select[Any]) -> tuple[Any, ...]:
    return tuple(value for _label, value in select._options)


def _option_labels(select: Select[Any]) -> tuple[str, ...]:
    labels = []
    for label, _value in select._options:
        labels.append(label.plain if isinstance(label, Text) else str(label))
    return tuple(labels)


def _label_for_value(select: Select[Any], value: str) -> str:
    for label, option_value in select._options:
        if option_value == value:
            return label.plain if isinstance(label, Text) else str(label)
    raise AssertionError(f"Missing Select value: {value}")


class _SaveProfileService:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, STTSGeneratedAudio]] = []
        self.error: BaseException | None = None

    async def create_from_artifact(
        self,
        display_name: str,
        artifact: STTSGeneratedAudio,
    ) -> object:
        self.create_calls.append((display_name, artifact))
        if self.error is not None:
            raise self.error
        return object()


class _PaneHost(App[None]):
    """Hosts `SpeechPlaygroundPane` alone -- the pane-side equivalent of the
    retired widget test file's `_PlaygroundHost`. Captures generation events
    and notifications the same way, and answers `_ensure_tts_profile_
    service` for the profile-save path.
    """

    def __init__(
        self,
        *,
        provider: str = "audio_cpp",
        preset: TTSPlaygroundSelectionPreset | None = None,
        profile_service: object | None = None,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.preset = preset
        self.profile_service = profile_service
        self.profile_service_requests = 0
        self.notices: list[tuple[str, str]] = []
        self.generation_events: list[STTSPlaygroundGenerateEvent] = []

    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(
            id="speech-playground-pane",
            provider=self.provider,
            profile_preset=self.preset,
        )

    async def _ensure_tts_profile_service(self) -> object | None:
        self.profile_service_requests += 1
        return self.profile_service

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: str = "information",
        timeout: float | None = None,
    ) -> None:
        del title, timeout
        self.notices.append((message, severity))

    def post_message(self, message: Any) -> bool:
        if isinstance(message, STTSPlaygroundGenerateEvent):
            self.generation_events.append(message)
            return True
        return super().post_message(message)


class _PaneScreen(Screen):
    """Hosts the pane under the real app-wide CSS bundle.

    Used by exactly one test in this file (the narrow-width save-button
    geometry check). `_PaneHost` above -- a bare `App[None]`, the same
    shape the retired widget's own `_PlaygroundHost` used -- loads no CSS
    at all. That was harmless for the widget (its layout came from an
    inline `DEFAULT_CSS` on the widget class itself), but verified
    empirically NOT harmless for `SpeechPlaygroundPane`: every axis
    `Select` reports a degenerate `Region(x=<viewport width>, width=1,
    ...)` under a bare `_PaneHost`, reproducible on a plain no-preset mount
    with nothing test-specific involved, because the pane's own CSS classes
    (`.speech-axis-control`, `.speech-pane`, ...) live in the app-wide TCSS
    bundle a real `TldwCli` loads, not on the widget. Mirrors
    `test_speech_playground_pane.py`'s own `_PaneScreen`, the pattern that
    file's geometry-sensitive tests already rely on.
    """

    def compose(self) -> ComposeResult:
        body = Vertical(
            SpeechPlaygroundPane(id="speech-playground-pane"),
            id="lab-body",
        )
        body.styles.width = "100%"
        body.styles.height = "100%"
        yield body


@pytest.fixture
def audio_cpp_playground(monkeypatch: pytest.MonkeyPatch) -> FakeTTSService:
    """Point the pane's service hook at a fresh `FakeTTSService`.

    Recipe step 3/4: replaces the retired widget test's `monkeypatch.
    setattr(STTS_Window, "get_tts_service", ...)` with the pane's own
    `_tts_service_factory` hook, and its `_check_higgs_installation` patch
    the same way. `provider=` on `_PaneHost` (recipe step 1) replaces the
    widget fixture's `get_cli_setting` override that forced
    ``default_provider == "audio_cpp"``.
    """
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
    return service


def _patch_cli_setting(
    monkeypatch: pytest.MonkeyPatch,
    configured: dict[tuple[str, str], Any],
) -> None:
    """Patch `_cli_setting` for a test whose original `get_cli_setting`
    fixture patch configured more than `default_provider` (e.g. a
    `default_voice`) -- recipe step 2's "keep an equivalent" case.
    """

    def _cli_setting(
        self: SpeechPlaygroundPane,
        section: str,
        key: str,
        default: Any = None,
    ) -> Any:
        return configured.get((section, key), default)

    monkeypatch.setattr(SpeechPlaygroundPane, "_cli_setting", _cli_setting)


# =====================================================================
# Section A -- mount, catalog loading, voice discovery, provider switching
# =====================================================================


@pytest.mark.asyncio
async def test_mount_uses_descriptors_and_resolves_only_selected_provider(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        provider_select = app.query_one("#tts-provider-select", Select)
        assert _option_values(provider_select) == PROVIDER_IDS
        assert provider_select.value == "audio_cpp"
        assert _option_labels(provider_select)[0] == "[b]audio.cpp[/]"

        model_select = app.query_one("#tts-model-select", Select)
        voice_select = app.query_one("#tts-voice-select", Select)
        assert model_select.value == "<opaque:model>"
        assert _option_labels(model_select)[0] == "[bold red]Opaque model[/]"
        assert voice_select.value == SERVER_DEFAULT_VOICE_ID
        assert _option_values(voice_select) == (
            SERVER_DEFAULT_VOICE_ID,
            "[voice]",
            "<script>alert(1)</script>",
        )
        assert _option_labels(voice_select)[2] == "<script>alert(1)</script>"

        assert app.query_one("#tts-format-select", Select).value == "wav"
        assert app.query_one("#tts-format-select", Select).disabled is True
        assert app.query_one("#tts-speed-input", Input).value == "1.0"
        assert app.query_one("#tts-speed-input", Input).disabled is True
        restriction = app.query_one("#tts-audio-cpp-restrictions", Static)
        assert "complete wav" in str(restriction.render()).lower()

    assert service.descriptor_calls == 1
    assert service.catalog_calls == [("audio_cpp", False)]
    assert service.voice_calls == [
        ("audio_cpp", "<opaque:model>", False),
    ]
    assert service.voice_observation_calls == [
        ("audio_cpp", "<opaque:model>", False),
    ]
    assert service.synthesize_calls == 0


@pytest.mark.asyncio
async def test_configuration_change_marks_catalog_stale_without_connecting(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        model_values = _option_values(app.query_one("#tts-model-select", Select))
        calls_before = list(service.catalog_calls)

        pane.current_audio_file = tmp_path / "existing.wav"
        app.query_one("#audio-play-btn", Button).disabled = False
        app.query_one("#audio-export-btn", Button).disabled = False
        service.revisions["audio_cpp"] = 2
        pane.mark_provider_configuration_changed("audio_cpp", 2)
        await pilot.pause()

        assert _option_values(app.query_one("#tts-model-select", Select)) == (
            model_values
        )
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert app.query_one("#audio-play-btn", Button).disabled is False
        assert app.query_one("#audio-export-btn", Button).disabled is False
        assert (
            "refresh"
            in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )
        assert service.catalog_calls == calls_before


@pytest.mark.asyncio
async def test_catalog_result_is_discarded_when_configuration_revision_changes(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    service.catalog_started = asyncio.Event()
    service.allow_catalog = asyncio.Event()
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await service.catalog_started.wait()
        service.revisions["audio_cpp"] = 2
        service.allow_catalog.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert _option_values(app.query_one("#tts-model-select", Select)) == (
            LOADING_SELECT_VALUE,
        )
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        status = str(app.query_one("#tts-provider-status", Static).render()).lower()
        assert "settings changed" in status


@pytest.mark.asyncio
async def test_exact_profile_revision_invalidated_catalog_projects_but_stays_blocked(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    service.catalog_started = asyncio.Event()
    service.allow_catalog = asyncio.Event()
    app = _PaneHost(preset=_profile_preset())

    async with app.run_test(size=(180, 70)) as pilot:
        await service.catalog_started.wait()
        service.revisions["audio_cpp"] = 2
        service.allow_catalog.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-model-select", Select).value == "profile/model"
        assert app.query_one("#tts-voice-select", Select).value == "profile/voice"
        assert app.query_one("#tts-format-select", Select).value == "wav"
        assert app.query_one("#tts-speed-input", Input).value == "1.0"
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        assert (
            "settings changed"
            in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )
        banner = app.query_one("#tts-profile-preview-status", Static)
        banner_copy = str(banner.render()).lower()
        assert "blocked" in banner_copy
        assert "refresh" in banner_copy
        assert "exact saved selection" not in banner_copy
        assert banner.has_class("profile-preview-unverified")


@pytest.mark.asyncio
async def test_superseded_catalog_failure_cannot_overwrite_newer_success(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        call_count = 0
        newer_catalog = _audio_catalog(revision=12)

        async def get_catalog(
            provider_id: str,
            refresh: bool = False,
        ) -> TTSProviderCatalog:
            nonlocal call_count
            del refresh
            call_count += 1
            if call_count == 1:
                first_started.set()
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    await release_first.wait()
                raise RuntimeError("obsolete refresh failed")
            assert provider_id == "audio_cpp"
            service.catalogs[provider_id] = newer_catalog
            return newer_catalog

        monkeypatch.setattr(service, "get_catalog", get_catalog)

        pane._load_provider_catalog("audio_cpp", refresh=True)
        await first_started.wait()
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await _wait_until(
            pilot,
            lambda: (
                call_count == 2
                and pane._catalogs.get("audio_cpp") is newer_catalog
                and "ready"
                in str(app.query_one("#tts-provider-status", Static).render()).lower()
            ),
        )

        release_first.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert pane._catalogs["audio_cpp"] is newer_catalog
        assert "audio_cpp" not in pane._stale_providers
        assert (
            "ready"
            in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )


@pytest.mark.asyncio
async def test_superseded_catalog_success_cannot_invalidate_newer_success(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        call_count = 0
        older_catalog = _audio_catalog(revision=11)
        newer_catalog = _audio_catalog(revision=12)

        async def get_catalog(
            provider_id: str,
            refresh: bool = False,
        ) -> TTSProviderCatalog:
            nonlocal call_count
            del refresh
            assert provider_id == "audio_cpp"
            call_count += 1
            if call_count == 1:
                first_started.set()
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    await release_first.wait()
                return older_catalog
            service.catalogs[provider_id] = newer_catalog
            return newer_catalog

        monkeypatch.setattr(service, "get_catalog", get_catalog)

        pane._load_provider_catalog("audio_cpp", refresh=True)
        await first_started.wait()
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await _wait_until(
            pilot,
            lambda: (
                call_count == 2
                and pane._catalogs.get("audio_cpp") is newer_catalog
                and "ready"
                in str(app.query_one("#tts-provider-status", Static).render()).lower()
            ),
        )

        release_first.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert pane._catalogs["audio_cpp"] is newer_catalog
        assert "audio_cpp" not in pane._stale_providers
        assert pane._catalog_generation_allowed is True
        assert (
            "ready"
            in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("obsolete_fails", (False, True), ids=("success", "failure"))
async def test_superseded_same_model_voice_result_cannot_overwrite_newer_success(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    obsolete_fails: bool,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        call_count = 0
        model_id = "<opaque:model>"
        catalog_revision = service.catalogs["audio_cpp"].revision

        async def get_voices(
            provider_id: str,
            requested_model_id: str,
            refresh: bool = False,
        ) -> tuple[str, ...]:
            nonlocal call_count
            del refresh
            assert (provider_id, requested_model_id) == ("audio_cpp", model_id)
            call_count += 1
            if call_count == 1:
                first_started.set()
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    await release_first.wait()
                if obsolete_fails:
                    raise RuntimeError("obsolete voice request failed")
                return ("obsolete-voice",)
            return ("new-voice",)

        monkeypatch.setattr(service, "get_voices", get_voices)

        pane._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )
        await first_started.wait()
        pane._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )
        await _wait_until(
            pilot,
            lambda: (
                call_count == 2
                and pane._discovered_voices.get(("audio_cpp", model_id))
                == ("new-voice",)
            ),
        )

        release_first.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert pane._discovered_voices[("audio_cpp", model_id)] == ("new-voice",)
        assert _option_values(app.query_one("#tts-voice-select", Select)) == (
            SERVER_DEFAULT_VOICE_ID,
            "new-voice",
        )
        assert (
            "voices are unavailable"
            not in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )


@pytest.mark.asyncio
async def test_catalog_generation_is_reserved_before_exclusive_worker_cancellation(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        baseline_catalog = pane._catalogs["audio_cpp"]
        first_started = asyncio.Event()
        first_returned_on_cancel = asyncio.Event()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        call_count = 0
        obsolete_catalog = _audio_catalog(revision=10)
        newer_catalog = _audio_catalog(revision=12)

        async def get_catalog(
            provider_id: str,
            refresh: bool = False,
        ) -> TTSProviderCatalog:
            nonlocal call_count
            del refresh
            assert provider_id == "audio_cpp"
            call_count += 1
            if call_count == 1:
                first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_returned_on_cancel.set()
                    return obsolete_catalog
            second_started.set()
            await release_second.wait()
            service.catalogs[provider_id] = newer_catalog
            return newer_catalog

        monkeypatch.setattr(service, "get_catalog", get_catalog)

        pane._load_provider_catalog("audio_cpp", refresh=True)
        await first_started.wait()
        first_generation = pane._catalog_request_generations["audio_cpp"]

        pane._load_provider_catalog("audio_cpp", refresh=True)

        assert pane._catalog_request_generations["audio_cpp"] == (first_generation + 1)
        await first_returned_on_cancel.wait()
        await second_started.wait()
        await pilot.pause()
        assert pane._catalogs["audio_cpp"] is baseline_catalog

        release_second.set()
        await app.workers.wait_for_complete()
        assert pane._catalogs["audio_cpp"] is newer_catalog


@pytest.mark.asyncio
async def test_voice_generation_is_reserved_before_exclusive_worker_cancellation(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        model_id = "<opaque:model>"
        request_key = ("audio_cpp", model_id)
        baseline_voices = pane._discovered_voices[request_key]
        catalog_revision = service.catalogs["audio_cpp"].revision
        first_started = asyncio.Event()
        first_returned_on_cancel = asyncio.Event()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        call_count = 0

        async def get_voices(
            provider_id: str,
            requested_model_id: str,
            refresh: bool = False,
        ) -> tuple[str, ...]:
            nonlocal call_count
            del refresh
            assert (provider_id, requested_model_id) == request_key
            call_count += 1
            if call_count == 1:
                first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_returned_on_cancel.set()
                    return ("obsolete-on-cancel",)
            second_started.set()
            await release_second.wait()
            return ("new-voice",)

        monkeypatch.setattr(service, "get_voices", get_voices)

        pane._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )
        await first_started.wait()
        first_generation = pane._voice_request_generations[request_key]

        pane._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )

        assert pane._voice_request_generations[request_key] == first_generation + 1
        await first_returned_on_cancel.wait()
        await second_started.wait()
        await pilot.pause()
        assert pane._discovered_voices[request_key] == baseline_voices

        release_second.set()
        await app.workers.wait_for_complete()
        assert pane._discovered_voices[request_key] == ("new-voice",)


@pytest.mark.asyncio
async def test_voice_discovery_does_not_cancel_inflight_catalog_refresh(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        service.catalog_started = asyncio.Event()
        service.allow_catalog = asyncio.Event()
        pane = app.query_one(SpeechPlaygroundPane)

        pane._load_provider_catalog("audio_cpp", refresh=True)
        await service.catalog_started.wait()
        app.query_one("#tts-model-select", Select).value = "second-model"
        await _wait_until(
            pilot,
            lambda: any(
                provider_id == "audio_cpp" and model_id == "second-model"
                for provider_id, model_id, _refresh in service.voice_calls
            ),
        )

        assert service.catalog_cancelled is False
        service.allow_catalog.set()
        await _wait_until(
            pilot,
            lambda: (
                "ready"
                in str(app.query_one("#tts-provider-status", Static).render()).lower()
            ),
        )


@pytest.mark.asyncio
async def test_catalog_revision_invalidates_old_voices_before_rediscovery(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.value = "[voice]"
        app.query_one("#tts-text-input", TextArea).text = "pending voice"
        await pilot.pause()

        service.catalogs["audio_cpp"] = _audio_catalog(revision=12)
        service.voice_started = asyncio.Event()
        service.allow_voices = asyncio.Event()
        pane = app.query_one(SpeechPlaygroundPane)
        notices_before = list(app.notices)
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await service.voice_started.wait()
        await pilot.pause()

        assert _option_values(voice_select) == (
            SERVER_DEFAULT_VOICE_ID,
            "[voice]",
        )
        assert voice_select.value == "[voice]"
        assert app.notices == notices_before
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        pending_notices = [
            *notices_before,
            ("Voices are still loading; wait before generating", "warning"),
        ]
        assert app.notices == pending_notices

        service.allow_voices.set()
        await app.workers.wait_for_complete()

        assert voice_select.value == "[voice]"
        assert app.notices == pending_notices
        assert app.query_one("#tts-generate-btn", Button).disabled is False


@pytest.mark.asyncio
async def test_catalog_revision_preserves_exact_voice_removed_by_refresh(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.value = "[voice]"
        await pilot.pause()

        service.catalogs["audio_cpp"] = _audio_catalog(revision=12)
        service.voices[("audio_cpp", "<opaque:model>")] = ("replacement",)
        service.voice_started = asyncio.Event()
        service.allow_voices = asyncio.Event()
        notices_before = list(app.notices)
        pane = app.query_one(SpeechPlaygroundPane)
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await service.voice_started.wait()
        await pilot.pause()

        assert app.notices == notices_before

        service.allow_voices.set()
        await app.workers.wait_for_complete()

        assert voice_select.value == "[voice]"
        assert "[voice]" in _option_values(voice_select)
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert app.notices == notices_before


@pytest.mark.asyncio
async def test_voice_discovery_failure_preserves_pending_explicit_voice(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.value = "[voice]"
        app.query_one("#tts-text-input", TextArea).text = "fallback voice"
        await pilot.pause()

        service.catalogs["audio_cpp"] = _audio_catalog(revision=12)
        service.voice_error = RuntimeError("untrusted upstream detail")
        app.query_one(SpeechPlaygroundPane)._load_provider_catalog(
            "audio_cpp",
            refresh=True,
        )
        await app.workers.wait_for_complete()

        assert voice_select.value == "[voice]"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert (
            str(app.query_one("#tts-provider-status", Static).render())
            == "Voices are unavailable; the exact selection remains unverified"
        )

        app.query_one(SpeechPlaygroundPane).action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        assert "untrusted upstream detail" not in str(app.notices)


@pytest.mark.asyncio
async def test_voice_discovery_failure_preserves_configured_explicit_default(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    _patch_cli_setting(
        monkeypatch,
        {
            ("app_tts", "default_provider"): "audio_cpp",
            ("app_tts", "default_voice"): "[voice]",
        },
    )
    service.voice_error = RuntimeError("untrusted upstream detail")
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.query_one("#tts-text-input", TextArea).text = "configured fallback"
        await pilot.pause()

        pane = app.query_one(SpeechPlaygroundPane)
        assert app.query_one("#tts-voice-select", Select).value == "[voice]"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert (
            str(app.query_one("#tts-provider-status", Static).render())
            == "Voices are unavailable; the exact selection remains unverified"
        )

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        assert "untrusted upstream detail" not in str(app.notices)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lifecycle_error", "expected_status"),
    (
        (
            TTSProviderReconfiguringError("PRIVATE_RECONFIGURING"),
            "settings are being applied",
        ),
        (
            TTSRegistryClosedError("PRIVATE_REGISTRY_CLOSED"),
            "tts service is unavailable",
        ),
    ),
)
async def test_voice_discovery_lifecycle_failure_preserves_pending_selection(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_error: Exception,
    expected_status: str,
) -> None:
    service = audio_cpp_playground
    _patch_cli_setting(
        monkeypatch,
        {
            ("app_tts", "default_provider"): "audio_cpp",
            ("app_tts", "default_voice"): "[voice]",
        },
    )
    service.voice_error = lifecycle_error
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.query_one("#tts-text-input", TextArea).text = "lifecycle pending"
        await pilot.pause()

        pane = app.query_one(SpeechPlaygroundPane)
        status = str(app.query_one("#tts-provider-status", Static).render()).lower()
        assert pane._pending_voice_selections == {"audio_cpp": "[voice]"}
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert expected_status in status

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        assert "PRIVATE_" not in str(app.notices)
        assert "PRIVATE_" not in status


@pytest.mark.asyncio
async def test_server_default_override_survives_provider_switch(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del audio_cpp_playground
    _patch_cli_setting(
        monkeypatch,
        {
            ("app_tts", "default_provider"): "audio_cpp",
            ("app_tts", "default_voice"): "[voice]",
        },
    )
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        provider_select = app.query_one("#tts-provider-select", Select)
        voice_select = app.query_one("#tts-voice-select", Select)
        assert voice_select.value == "[voice]"

        voice_select.value = SERVER_DEFAULT_VOICE_ID
        provider_select.value = "openai"
        await _wait_until(
            pilot,
            lambda: app.query_one("#tts-model-select", Select).value == "tts-1",
        )
        provider_select.value = "audio_cpp"
        await _wait_until(
            pilot,
            lambda: (
                app.query_one("#tts-model-select", Select).value == "<opaque:model>"
            ),
        )

        assert voice_select.value == SERVER_DEFAULT_VOICE_ID


@pytest.mark.asyncio
async def test_stale_model_voice_failure_cannot_release_current_pending_voice(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        voice_select = app.query_one("#tts-voice-select", Select)
        model_select = app.query_one("#tts-model-select", Select)
        voice_select.value = "[voice]"
        app.query_one("#tts-text-input", TextArea).text = "current pending voice"
        await pilot.pause()

        old_key = ("audio_cpp", "<opaque:model>")
        current_key = ("audio_cpp", "second-model")
        service.voice_started_by_request[old_key] = asyncio.Event()
        service.voice_finished_by_request[old_key] = asyncio.Event()
        service.voice_gates[old_key] = asyncio.Event()
        service.voice_errors[old_key] = RuntimeError("late old-model failure")
        service.voice_ignore_cancellation.add(old_key)
        service.voice_started_by_request[current_key] = asyncio.Event()
        service.voice_finished_by_request[current_key] = asyncio.Event()
        service.voice_gates[current_key] = asyncio.Event()

        pane._load_provider_voices(
            "audio_cpp",
            "<opaque:model>",
            service.catalogs["audio_cpp"].revision,
            refresh=True,
        )
        await service.voice_started_by_request[old_key].wait()

        model_select.value = "second-model"
        await service.voice_started_by_request[current_key].wait()
        await pilot.pause()

        assert pane._pending_voice_selections == {"audio_cpp": "[voice]"}
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        service.voice_gates[old_key].set()
        await service.voice_finished_by_request[old_key].wait()
        await pilot.pause()

        assert pane._pending_voice_selections == {"audio_cpp": "[voice]"}
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        pane.action_generate_tts()
        await pilot.pause()
        assert app.generation_events == []
        assert app.notices[-1] == (
            "Voices are still loading; wait before generating",
            "warning",
        )

        service.voice_gates[current_key].set()
        await service.voice_finished_by_request[current_key].wait()
        await _wait_until(pilot, lambda: pane._pending_voice_selections == {})


@pytest.mark.asyncio
async def test_legacy_control_state_is_restored_after_audio_cpp_switch(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        provider = app.query_one("#tts-provider-select", Select)
        provider.value = "openai"
        await _wait_until(
            pilot,
            lambda: app.query_one("#tts-model-select", Select).value == "tts-1",
        )

        model = app.query_one("#tts-model-select", Select)
        voice = app.query_one("#tts-voice-select", Select)
        response_format = app.query_one("#tts-format-select", Select)
        speed = app.query_one("#tts-speed-input", Input)
        model.value = "tts-1-hd"
        voice.value = "nova"
        response_format.value = "flac"
        speed.value = "1.35"
        await pilot.pause()

        provider.value = "audio_cpp"
        await _wait_until(pilot, lambda: response_format.disabled)
        assert response_format.value == "wav"
        assert response_format.disabled is True
        assert speed.value == "1.0"
        assert speed.disabled is True

        provider.value = "openai"
        await _wait_until(
            pilot,
            lambda: model.value == "tts-1-hd" and not response_format.disabled,
        )
        assert model.value == "tts-1-hd"
        assert voice.value == "nova"
        assert response_format.value == "flac"
        assert response_format.disabled is False
        assert speed.value == "1.35"
        assert speed.disabled is False

    assert [provider_id for provider_id, _refresh in service.catalog_calls] == [
        "audio_cpp",
        "openai",
        "audio_cpp",
        "openai",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "provider_id",
        "model_id",
        "model_label",
        "voice_id",
        "voice_label",
    ),
    (
        ("openai", "tts-1", "TTS-1 (Standard)", "alloy", "Alloy"),
        (
            "elevenlabs",
            "eleven_multilingual_v2",
            "Eleven Multilingual v2 (Default)",
            "21m00Tcm4TlvDq8ikWAM",
            "Rachel",
        ),
        ("kokoro", "kokoro", "Kokoro 82M", "af_alloy", "Alloy (US Female)"),
        (
            "chatterbox",
            "chatterbox",
            "Chatterbox 0.5B",
            "default",
            "Default Voice",
        ),
        (
            "higgs",
            "higgs-audio-v2",
            "Higgs Audio V2 3B",
            "professional_female",
            "Professional Female",
        ),
        ("alltalk", "alltalk", "AllTalk TTS", "female_01.wav", "Female 01"),
    ),
)
async def test_legacy_provider_defaults_and_labels_are_preserved(
    audio_cpp_playground: FakeTTSService,
    provider_id: str,
    model_id: str,
    model_label: str,
    voice_id: str,
    voice_label: str,
) -> None:
    del audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.query_one("#tts-provider-select", Select).value = provider_id
        await _wait_until(
            pilot,
            lambda: app.query_one("#tts-model-select", Select).value == model_id,
        )

        model_select = app.query_one("#tts-model-select", Select)
        voice_select = app.query_one("#tts-voice-select", Select)
        assert model_select.value == model_id
        assert _label_for_value(model_select, model_id) == model_label
        assert voice_select.value == voice_id
        assert _label_for_value(voice_select, voice_id) == voice_label


@pytest.mark.asyncio
async def test_higgs_saved_profile_is_prefixed_exactly_once_in_request(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del audio_cpp_playground
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_higgs_profile_choices",
        staticmethod(lambda: [("Saved voice", "profile:saved-voice")]),
    )
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.query_one("#tts-provider-select", Select).value = "higgs"
        await _wait_until(
            pilot,
            lambda: (
                app.query_one("#tts-model-select", Select).value == "higgs-audio-v2"
            ),
        )
        app.query_one("#tts-voice-select", Select).value = "profile:saved-voice"
        app.query_one("#tts-text-input", TextArea).text = "use saved profile"
        await pilot.pause()

        app.query_one(SpeechPlaygroundPane)._generate_tts()
        await pilot.pause()

        assert len(app.generation_events) == 1
        assert app.generation_events[0].request.voice_id == "profile:saved-voice"


#: PANE DEFECT (see the `xfail`s below): `_load_provider_catalog_worker`
#: adds a provider to `_stale_providers` whenever the *freshly-fetched*
#: catalog itself reports `health.fresh is False` --
#:
#:     if catalog.health.fresh:
#:         self._stale_providers.discard(provider_id)
#:     else:
#:         self._stale_providers.add(provider_id)
#:
#: -- and `_catalog_health_copy` checks `_stale_providers` FIRST, ahead of
#: `health.state`, so any fresh (first-ever) load whose health happens to
#: report `fresh=False` (a "reconfiguring" provider, or a naturally stale
#: "available" catalog) is shown the generic "settings changed; refresh
#: models" instead of the accurate, state-specific recovery copy -- even
#: though nothing has changed; this is simply how the very first read came
#: back. Confirmed NOT present in the retired widget: its equivalent
#: success path was an unconditional `self._stale_providers.discard(
#: provider_id)` with no `health.fresh` branch at all (verified by diffing
#: `git show HEAD:tldw_chatbook/UI/STTS_Window.py` against the pane before
#: this branch's own sibling task deleted the widget). `_catalog_health_
#: copy` itself is byte-for-byte identical between the two -- the
#: divergence is entirely in what populates `_stale_providers`.
_STALE_ON_FIRST_LOAD_DEFECT = (
    "PANE DEFECT (TASK-2951 finding): a fresh (first-ever) catalog load "
    "whose reported health.fresh is False is shown the generic 'settings "
    "changed; refresh models' instead of the accurate state-specific "
    "copy, because _load_provider_catalog_worker marks the provider "
    "'stale' purely from health.fresh on ANY load, not only a load that "
    "followed a real configuration change. The retired widget's "
    "equivalent success path never did this (unconditional discard, no "
    "health.fresh branch)."
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("health", "expected_copy"),
    (
        (
            ProviderHealth(state="unavailable", fresh=True),
            "unavailable",
        ),
        (
            ProviderHealth(state="not_configured", fresh=True),
            "not configured",
        ),
        pytest.param(
            ProviderHealth(state="reconfiguring", fresh=False),
            "settings are being applied",
            marks=pytest.mark.xfail(
                reason=_STALE_ON_FIRST_LOAD_DEFECT,
                strict=True,
            ),
        ),
        pytest.param(
            ProviderHealth(state="available", fresh=False),
            "catalog is stale",
            marks=pytest.mark.xfail(
                reason=_STALE_ON_FIRST_LOAD_DEFECT,
                strict=True,
            ),
        ),
    ),
)
async def test_audio_cpp_health_states_use_fixed_safe_recovery_copy(
    audio_cpp_playground: FakeTTSService,
    health: ProviderHealth,
    expected_copy: str,
) -> None:
    service = audio_cpp_playground
    service.catalogs["audio_cpp"] = _audio_catalog(health=health)
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        status = str(app.query_one("#tts-provider-status", Static).render()).lower()
        assert expected_copy in status
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert app.query_one("#tts-refresh-catalog-btn", Button).disabled is False
        assert app.query_one("#tts-provider-select", Select).value == "audio_cpp"
        assert _option_values(app.query_one("#tts-model-select", Select)) == (
            "<opaque:model>",
            "second-model",
        )
        assert service.catalog_calls == [("audio_cpp", False)]


# =====================================================================
# Section B -- playback, export, unmount cleanup, mount rehydration
# =====================================================================


@pytest.mark.asyncio
async def test_playback_uses_dedicated_widget_worker_group(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    path = tmp_path / "artifact.wav"
    path.write_bytes(b"RIFF")
    artifact = STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="model",
        voice_id=None,
        source_text="source",
        operation_id="operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    captured: dict[str, object] = {}
    worker_calls = 0

    def run_worker(
        _self: SpeechPlaygroundPane,
        awaitable: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        nonlocal worker_calls
        worker_calls += 1
        captured.update(kwargs)
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        return SimpleNamespace(is_finished=False, cancel=lambda: None)

    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        monkeypatch.setattr(SpeechPlaygroundPane, "run_worker", run_worker)
        monkeypatch.setattr(
            SpeechPlaygroundPane,
            "_ensure_audio_player",
            lambda _self: True,
        )
        pane = app.query_one(SpeechPlaygroundPane)
        pane.current_audio_artifact = artifact
        pane.current_audio_file = artifact.path

        pane._play_audio()
        pane._play_audio()

        assert captured["group"] == "stts-playback"
        assert captured["exclusive"] is True
        assert worker_calls == 1


@pytest.mark.asyncio
async def test_playback_uses_artifact_captured_before_worker_runs(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    old_path = tmp_path / "old.wav"
    new_path = tmp_path / "new.wav"
    old_path.write_bytes(b"old")
    new_path.write_bytes(b"new")
    old_artifact = STTSGeneratedAudio(
        path=old_path,
        provider_id="audio_cpp",
        model_id="old-model",
        voice_id=None,
        source_text="old",
        operation_id="old-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    new_artifact = STTSGeneratedAudio(
        path=new_path,
        provider_id="audio_cpp",
        model_id="new-model",
        voice_id=None,
        source_text="new",
        operation_id="new-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    jobs: list[object] = []

    def run_worker(
        _self: SpeechPlaygroundPane,
        job: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        jobs.append(job)
        return SimpleNamespace(is_finished=False, cancel=lambda: None)

    class Player:
        def __init__(self) -> None:
            self.played: list[Path] = []

        async def get_state(self) -> str:
            return "stopped"

        async def stop(self) -> bool:
            return True

        async def play(self, path: Path) -> bool:
            self.played.append(path)
            return False

    lease_handler = SimpleNamespace(
        lease_playground_artifact=Mock(return_value=True),
        release_playground_artifact=Mock(),
    )
    app = _PaneHost()
    app._stts_handler = lease_handler
    player = Player()
    app.audio_player = player

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        monkeypatch.setattr(SpeechPlaygroundPane, "run_worker", run_worker)
        pane = app.query_one(SpeechPlaygroundPane)
        pane.current_audio_artifact = old_artifact
        pane.current_audio_file = old_path

        pane._play_audio()
        pane.current_audio_artifact = new_artifact
        pane.current_audio_file = new_path

        job = jobs[0]
        if callable(job):
            await job()
        else:
            await job  # type: ignore[misc]

        assert player.played == [old_path]
        lease_handler.lease_playground_artifact.assert_called_once_with(old_artifact)
        lease_handler.release_playground_artifact.assert_called_once_with(old_artifact)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pause_before_replacement", "terminal_action"),
    (
        (False, "stop"),
        (True, "stop"),
        (False, "finish"),
        (False, "unmount"),
    ),
    ids=("playing-stop", "paused-stop", "natural-finish", "unmount"),
)
async def test_playback_lease_survives_replacement_until_playback_ends(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
    pause_before_replacement: bool,
    terminal_action: str,
) -> None:
    del audio_cpp_playground
    old_path = tmp_path / "old.wav"
    new_path = tmp_path / "new.wav"
    old_path.write_bytes(b"old")
    new_path.write_bytes(b"new")
    old_artifact = STTSGeneratedAudio(
        path=old_path,
        provider_id="audio_cpp",
        model_id="old-model",
        voice_id=None,
        source_text="old",
        operation_id="old-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    new_artifact = STTSGeneratedAudio(
        path=new_path,
        provider_id="audio_cpp",
        model_id="new-model",
        voice_id=None,
        source_text="new",
        operation_id="new-operation",
        audio_format="wav",
        content_type="audio/wav",
    )

    class Player:
        def __init__(self) -> None:
            self.state = PlaybackState.IDLE

        async def get_state(self) -> PlaybackState:
            return self.state

        async def stop(self) -> bool:
            self.state = PlaybackState.IDLE
            return True

        async def play(self, _path: Path) -> bool:
            self.state = PlaybackState.PLAYING
            return True

        async def is_playing(self) -> bool:
            return self.state == PlaybackState.PLAYING

        async def pause(self) -> bool:
            self.state = PlaybackState.PAUSED
            return True

        async def resume(self) -> bool:
            self.state = PlaybackState.PLAYING
            return True

        async def get_position(self) -> float:
            return 0.0

        async def get_duration(self) -> float:
            return 1.0

    app = _PaneHost()
    handler = STTSEventHandler(app)
    handler._accept_playground_artifact(old_artifact)
    app._stts_handler = handler
    player = Player()
    app.audio_player = player

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)

        pane._play_audio()
        await _wait_until(
            pilot,
            lambda: (
                player.state == PlaybackState.PLAYING and pane._play_worker_task is None
            ),
        )
        if pause_before_replacement:
            pane._pause_audio()
            await _wait_until(
                pilot,
                lambda: player.state == PlaybackState.PAUSED,
            )

        handler._accept_playground_artifact(new_artifact)
        pane._store_delivered_artifact(new_artifact, announce=False)

        assert old_path.exists()
        assert handler._playground_file_leases[old_path] == 1

        if terminal_action == "stop":
            await pane._stop_audio_async()
        elif terminal_action == "finish":
            player.state = PlaybackState.FINISHED
            await _wait_until(pilot, lambda: not old_path.exists())

    assert not old_path.exists()
    assert old_path not in handler._playground_file_leases


@pytest.mark.asyncio
async def test_export_uses_artifact_captured_before_dialog_completes(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    old_path = tmp_path / "old.wav"
    new_path = tmp_path / "new.wav"
    destination = tmp_path / "export.wav"
    old_path.write_bytes(b"old")
    new_path.write_bytes(b"new")
    old_artifact = STTSGeneratedAudio(
        path=old_path,
        provider_id="audio_cpp",
        model_id="old-model",
        voice_id=None,
        source_text="old",
        operation_id="old-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    new_artifact = STTSGeneratedAudio(
        path=new_path,
        provider_id="audio_cpp",
        model_id="new-model",
        voice_id=None,
        source_text="new",
        operation_id="new-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    callbacks: list[Callable[[str | None], None]] = []
    lease_handler = SimpleNamespace(
        lease_playground_artifact=Mock(return_value=True),
        release_playground_artifact=Mock(),
    )
    app = _PaneHost()
    app._stts_handler = lease_handler

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda _screen, callback: callbacks.append(callback),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        pane.current_audio_artifact = old_artifact
        pane.current_audio_file = old_path

        pane._export_audio()
        pane.current_audio_artifact = new_artifact
        pane.current_audio_file = new_path
        callbacks[0](str(destination))

        assert destination.read_bytes() == b"old"
        lease_handler.lease_playground_artifact.assert_called_once_with(old_artifact)
        lease_handler.release_playground_artifact.assert_called_once_with(old_artifact)


@pytest.mark.asyncio
async def test_export_cancel_releases_captured_artifact(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    path = tmp_path / "artifact.wav"
    path.write_bytes(b"audio")
    artifact = STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="model",
        voice_id=None,
        source_text="source",
        operation_id="operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    callbacks: list[Callable[[str | None], None]] = []
    lease_handler = SimpleNamespace(
        lease_playground_artifact=Mock(return_value=True),
        release_playground_artifact=Mock(),
    )
    app = _PaneHost()
    app._stts_handler = lease_handler

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda _screen, callback: callbacks.append(callback),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        pane.current_audio_artifact = artifact
        pane.current_audio_file = path

        pane._export_audio()
        callbacks[0](None)

        lease_handler.release_playground_artifact.assert_called_once_with(artifact)


@pytest.mark.asyncio
async def test_audio_export_rejects_unsafe_dialog_destination(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    source = tmp_path / "source.wav"
    unsafe_destination = tmp_path / "bad;name.wav"
    source.write_bytes(b"audio")
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)

        pane._handle_audio_export(
            str(unsafe_destination),
            source_path=source,
        )

    assert not unsafe_destination.exists()
    assert app.notices[-1][1] == "error"
    assert "dangerous pattern" in app.notices[-1][0]


@pytest.mark.asyncio
async def test_unmount_cancels_only_widget_owned_worker_groups(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del audio_cpp_playground
    app = _PaneHost()
    cleanup = Mock()
    app._stts_handler = SimpleNamespace(cleanup_tts_resources=cleanup)
    cancelled_groups: list[str] = []

    async with app.run_test(size=(180, 70)):
        original_cancel_group = app.workers.cancel_group

        def cancel_group(node: object, group: str) -> None:
            cancelled_groups.append(group)
            original_cancel_group(node, group)

        monkeypatch.setattr(app.workers, "cancel_group", cancel_group)
        await app.workers.wait_for_complete()

    assert {
        "stts-catalog-discovery",
        "stts-voice-discovery",
        "stts-playback",
    }.issubset(cancelled_groups)
    cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_new_mount_rehydrates_handler_owned_artifact(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    path = tmp_path / "retained.wav"
    path.write_bytes(b"RIFF")
    artifact = STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="response-model",
        voice_id=None,
        source_text="source",
        operation_id="completed-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    state = SimpleNamespace(
        active_operation_id=None,
        artifact=artifact,
        generation_active=False,
    )
    cleanup = Mock()
    app = _PaneHost()
    app._stts_handler = SimpleNamespace(
        playground_state=lambda: state,
        cleanup_tts_resources=cleanup,
    )

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert pane.current_audio_artifact is artifact
        assert pane.current_audio_file == path
        assert app.query_one("#audio-play-btn", Button).disabled is False
        assert app.query_one("#audio-export-btn", Button).disabled is False

    assert path.exists()
    cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_new_mount_rehydrates_active_generation_without_starting_another(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    state = SimpleNamespace(
        active_operation_id="active-operation",
        artifact=None,
        generation_active=True,
    )
    app = _PaneHost()
    app._stts_handler = SimpleNamespace(playground_state=lambda: state)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert pane._generation_operation_id == "active-operation"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert (
            "in progress"
            in str(app.query_one("#generation-status-text", Static).render()).lower()
        )
        assert service.synthesize_calls == 0


# =====================================================================
# Section C -- exact profile preview and adoption
# =====================================================================


@pytest.mark.asyncio
async def test_fresh_catalog_missing_exact_profile_stays_visible_but_unavailable(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    preset = _profile_preset()
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-provider-select", Select).value == "audio_cpp"
        assert app.query_one("#tts-model-select", Select).value == "profile/model"
        assert app.query_one("#tts-voice-select", Select).value == "profile/voice"
        assert app.query_one("#tts-profile-preview-status", Static).has_class(
            "profile-preview-unavailable"
        )
        assert pane._profile_effective_availability == "unavailable"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert pane._profile_preset is preset
        assert app.generation_events == []

    assert service.voice_calls == []
    assert service.synthesize_calls == 0


@pytest.mark.asyncio
async def test_complete_voice_observation_missing_exact_voice_blocks_profile(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    preset = _profile_preset(
        model_id="<opaque:model>",
        voice_id="missing-profile-voice",
    )
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert pane._profile_effective_availability == "unavailable"
        assert app.query_one("#tts-voice-select", Select).value == preset.voice_id
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert app.query_one("#tts-profile-preview-status", Static).has_class(
            "profile-preview-unavailable"
        )

    assert service.voice_calls == [("audio_cpp", "<opaque:model>", False)]


@pytest.mark.asyncio
async def test_long_exact_profile_ids_stay_reachable_via_tooltip(
    audio_cpp_playground: FakeTTSService,
) -> None:
    """MECHANICAL adaptation, reduced scope: the retired widget rendered
    each axis select in its own fixed-height "row" (a `#label` Static plus
    the Select, stacked one row per axis) and asserted that row's exact
    geometry (`select.region.height == 3`, single visible truncated line
    ending in "…", and the voice row sitting directly above the model row).
    `SpeechAxisRow` (the live replacement) lays every axis out as a `Grid`
    of reflowing chips -- there is no equivalent fixed single-row container
    or vertical row adjacency to assert.

    Verified empirically (not assumed) that this specific scenario cannot
    be ported with real layout assertions at all: `SpeechPlaygroundPane`'s
    own CSS classes (`.speech-axis-control`, `.speech-pane`, ...) are
    defined in the app-wide TCSS bundle, not on the widget itself. A bare
    `App[None]` test host -- the shape both the retired widget's
    `_PlaygroundHost` and this file's `_PaneHost` use, and the only
    lightweight host available outside a full `TldwCli` -- loads no CSS at
    all, so every `Select` on the axis row (even in tests that never
    reference geometry) reports a degenerate `Region(x=<viewport-width>,
    width=1, ...)`, reproducible on a plain no-preset mount with no long
    id involved. This is an artifact of the harness, not of the id length or
    the preset path, and does not reproduce inside the pane's own Screen-
    hosted test suite (`test_speech_playground_pane.py`, built on
    `Tests.UI.app_factory._build_test_app`'s real `TldwCli` app and its
    real CSS).

    What survives without any layout dependency, because it is pure
    application state (`Select.tooltip` is set to the exact preset id in
    `SpeechCatalogMixin._apply_controls`, shared code, unaffected by
    whether CSS ever painted a pixel): the tooltip carries the untruncated
    id. That is the one guarantee this reduced test still proves.
    """
    del audio_cpp_playground
    model_id = "[opaque-model]" + "/segment" * 24
    voice_id = "<opaque-voice>" + ":segment" * 24
    app = _PaneHost(preset=_profile_preset(model_id=model_id, voice_id=voice_id))

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        voice_select = app.query_one("#tts-voice-select", Select)
        model_select = app.query_one("#tts-model-select", Select)

        for select, exact_id in (
            (voice_select, voice_id),
            (model_select, model_id),
        ):
            assert isinstance(select.tooltip, Text)
            assert select.tooltip.plain == exact_id


@pytest.mark.asyncio
async def test_unavailable_exact_profile_catalog_failure_projects_but_stays_blocked(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    service.catalog_error = RuntimeError("private catalog detail")
    app = _PaneHost(preset=_profile_preset(availability="unavailable"))

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-model-select", Select).value == "profile/model"
        assert app.query_one("#tts-voice-select", Select).value == "profile/voice"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        status = str(app.query_one("#tts-provider-status", Static).render())
        assert "unavailable" in status.lower()
        assert "Edit" in status

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "voice_error",
    (
        None,
        RuntimeError("private voice detail"),
        TTSProviderReconfiguringError("private reconfiguring detail"),
        TTSRegistryClosedError("private closed detail"),
    ),
    ids=("success", "generic", "reconfiguring", "closed"),
)
async def test_fresh_not_configured_catalog_makes_exact_profile_unavailable(
    audio_cpp_playground: FakeTTSService,
    voice_error: Exception | None,
) -> None:
    service = audio_cpp_playground
    service.catalogs["audio_cpp"] = _audio_catalog(
        health=ProviderHealth(state="not_configured", fresh=True)
    )
    service.voice_error = voice_error
    preset = _profile_preset()
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-model-select", Select).value == preset.model_id
        assert app.query_one("#tts-voice-select", Select).value == preset.voice_id
        assert pane._profile_effective_availability == "unavailable"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        status = str(app.query_one("#tts-provider-status", Static).render())
        assert "unavailable" in status.lower()
        assert "Edit" in status
        banner = app.query_one("#tts-profile-preview-status", Static)
        assert "unavailable" in str(banner.render()).lower()
        assert "Edit" in str(banner.render())
        assert banner.has_class("profile-preview-unavailable")

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("availability", "expected_status"),
    (
        ("available", "service is unavailable"),
        ("unavailable", "choose Edit"),
    ),
)
async def test_closed_catalog_health_projects_exact_profile_but_stays_blocked(
    audio_cpp_playground: FakeTTSService,
    availability: str,
    expected_status: str,
) -> None:
    service = audio_cpp_playground
    service.catalogs["audio_cpp"] = _audio_catalog(
        health=ProviderHealth(state="closed", fresh=False)
    )
    preset = _profile_preset(availability=availability)
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-model-select", Select).value == preset.model_id
        assert app.query_one("#tts-voice-select", Select).value == preset.voice_id
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        assert (
            expected_status.lower()
            in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )
        banner_copy = str(
            app.query_one("#tts-profile-preview-status", Static).render()
        ).lower()
        assert "generate makes one exact attempt" not in banner_copy
        assert "blocked" in banner_copy or "unavailable" in banner_copy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("original_availability", "failure_availability"),
    (
        ("available", "unverified"),
        ("unverified", "unverified"),
        ("unavailable", "unavailable"),
    ),
)
async def test_current_voice_discovery_success_restores_only_original_preset_state(
    audio_cpp_playground: FakeTTSService,
    original_availability: str,
    failure_availability: str,
) -> None:
    service = audio_cpp_playground
    service.voice_error = RuntimeError("private upstream detail")
    preset = _profile_preset(
        model_id="<opaque:model>",
        voice_id="[voice]",
        availability=original_availability,
    )
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        assert pane._profile_effective_availability == failure_availability
        assert preset.availability == original_availability
        if original_availability == "unavailable":
            status = str(app.query_one("#tts-provider-status", Static).render())
            assert "unavailable" in status.lower()
            assert "Edit" in status

        service.voice_error = None
        pane._load_provider_voices(
            "audio_cpp",
            preset.model_id,
            service.catalogs["audio_cpp"].revision,
            refresh=True,
        )
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert pane._profile_effective_availability == original_availability
        assert preset.availability == original_availability


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edited_control", ("provider", "model", "voice", "format", "speed", "options")
)
async def test_user_generation_selection_edits_end_profile_preset_association(
    audio_cpp_playground: FakeTTSService,
    edited_control: str,
) -> None:
    """MECHANICAL adaptation for the "options" case: the retired widget
    mounted every provider's parameter container up front (hidden but
    focusable), so it could edit `#tts-stability-input` (an ElevenLabs-only
    control) while previewing an *audio.cpp* preset. `SpeechParamGroup`
    mounts only the SELECTED provider's own knobs, and
    `PROVIDER_PARAMS["audio_cpp"] == ()` -- audio.cpp has no options-class
    control at all in the redesigned split, so there is nothing to edit
    there anymore. This parametrization previews an ElevenLabs preset
    instead (whose `SpeechParamGroup`, including `#tts-stability-input`, is
    mounted straight from `compose()` because the pane is constructed with
    `provider="elevenlabs"` up front). The guarantee under test -- editing
    ANY control category ends the profile preset association -- is
    unchanged; only the fixture needed to reach an "options" control at all
    had to change.
    """
    del audio_cpp_playground
    if edited_control == "options":
        app = _PaneHost(
            provider="elevenlabs",
            preset=_profile_preset(
                provider_id="elevenlabs",
                model_id="eleven_multilingual_v2",
                voice_id="21m00Tcm4TlvDq8ikWAM",
            ),
        )
    else:
        app = _PaneHost(preset=_profile_preset())

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        assert pane._profile_preset is not None

        if edited_control == "provider":
            app.query_one("#tts-provider-select", Select).value = "openai"
        elif edited_control == "model":
            app.query_one("#tts-model-select", Select).value = "second-model"
        elif edited_control == "voice":
            app.query_one("#tts-voice-select", Select).value = SERVER_DEFAULT_VOICE_ID
        elif edited_control == "format":
            select = app.query_one("#tts-format-select", Select)
            select.disabled = False
            select.set_options([("WAV", "wav"), ("MP3", "mp3")])
            select.value = "mp3"
        elif edited_control == "speed":
            speed = app.query_one("#tts-speed-input", Input)
            speed.disabled = False
            speed.value = "1.1"
        else:
            options = app.query_one("#tts-stability-input", Input)
            options.focus()
            options.value = "0.6"
        await pilot.pause()

        assert pane._profile_preset is None


@pytest.mark.asyncio
async def test_provider_edit_during_initial_profile_catalog_load_ends_preset(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    service.catalog_started = asyncio.Event()
    service.allow_catalog = asyncio.Event()
    app = _PaneHost(preset=_profile_preset())

    async with app.run_test(size=(180, 70)) as pilot:
        await service.catalog_started.wait()
        pane = app.query_one(SpeechPlaygroundPane)
        provider = app.query_one("#tts-provider-select", Select)
        assert pane._profile_controls_applied is True
        assert app.query_one("#tts-model-select", Select).value == "profile/model"
        assert app.query_one("#tts-voice-select", Select).value == "profile/voice"
        banner = app.query_one("#tts-profile-preview-status", Static)
        banner_copy = str(banner.render()).lower()
        assert "loading" in banner_copy
        assert "blocked" not in banner_copy
        assert "unavailable" not in banner_copy
        assert banner.has_class("profile-preview-loading")

        provider.value = "openai"
        await _wait_until(pilot, lambda: pane._profile_preset is None)
        assert pane._profile_controls_applied is True

        service.allow_catalog.set()
        await app.workers.wait_for_complete()


# =====================================================================
# Section D -- save current result as a voice profile
# =====================================================================


@pytest.mark.asyncio
async def test_save_profile_action_is_visible_and_focusable_at_narrow_width(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
) -> None:
    """Needs the real app CSS bundle to be a meaningful geometry check --
    see `_PaneScreen`'s docstring for why this test alone uses
    `_build_test_app()`/`_PaneScreen` instead of this file's usual bare
    `_PaneHost`.
    """
    del audio_cpp_playground
    artifact = _native_profile_artifact(tmp_path / "narrow.wav")
    app = _build_test_app()

    async with app.run_test(size=(50, 24)) as pilot:
        screen = _PaneScreen()
        await app.push_screen(screen)
        await pilot.pause()
        # No `app.workers.wait_for_complete()` here: unlike `_PaneHost` (a
        # bare `App[None]` with no other workers), a real `TldwCli` from
        # `_build_test_app()` carries its own persistent background
        # workers, so waiting for the WHOLE app's worker set to go idle
        # never returns. `test_speech_playground_pane.py`'s own
        # `_build_test_app()`-based tests use plain `pilot.pause()` for the
        # same reason -- this test doesn't need the catalog to finish
        # loading, only the pane mounted.
        await pilot.pause()
        pane = screen.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(artifact, announce=False)
        button = screen.query_one("#audio-save-profile-btn", Button)
        player = screen.query_one("#audio-player-container")
        button.focus()
        button.scroll_visible(animate=False)
        await pilot.pause()

        assert button.has_focus
        assert player.content_region.contains_region(button.region)


@pytest.mark.asyncio
async def test_successful_native_artifact_save_uses_only_immutable_provenance(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = audio_cpp_playground
    profile_service = _SaveProfileService()
    app = _PaneHost(profile_service=profile_service)
    artifact_path = tmp_path / "result.wav"
    artifact_path.write_bytes(b"RIFF")
    artifact = _native_profile_artifact(artifact_path)
    lease = Mock(return_value=True)
    release = Mock()
    app._stts_handler = SimpleNamespace(
        playground_state=lambda: SimpleNamespace(
            active_operation_id=None,
            artifact=None,
            generation_active=False,
        ),
        lease_playground_artifact=lease,
        release_playground_artifact=release,
    )

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(artifact, announce=False)
        save_button = app.query_one("#audio-save-profile-btn", Button)
        assert save_button.display is True
        assert save_button.disabled is False

        model_select = app.query_one("#tts-model-select", Select)
        model_select.set_options([("Changed model", "changed/model")])
        model_select.value = "changed/model"
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.set_options([("Changed voice", "changed/voice")])
        voice_select.value = "changed/voice"
        await pilot.pause()
        discovery_before = (list(service.catalog_calls), list(service.voice_calls))

        async def _name(screen: object) -> str:
            assert isinstance(screen, profile_library_module.TTSProfileNameModal)
            return "Saved exact voice"

        monkeypatch.setattr(app, "push_screen_wait", _name)
        await pane._save_current_result_as_profile()

        assert profile_service.create_calls == [("Saved exact voice", artifact)]
        saved = profile_service.create_calls[0][1].requested_selection
        assert saved is artifact.requested_selection
        assert saved is not None
        assert (saved.model_id, saved.voice_id) == (
            "artifact/model",
            "artifact/voice",
        )
        assert app.profile_service_requests == 1
        assert (service.catalog_calls, service.voice_calls) == discovery_before
        assert pane.current_audio_artifact is artifact
        assert artifact_path.exists()
        lease.assert_not_called()
        release.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_mode", ("worker", "unmount"))
async def test_cancelled_profile_name_modal_is_dismissed_without_saving(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
    cancel_mode: str,
) -> None:
    del audio_cpp_playground
    profile_service = _SaveProfileService()
    app = _PaneHost(profile_service=profile_service)
    artifact = _native_profile_artifact(tmp_path / "cancelled.wav")

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(artifact, announce=False)
        save_button = app.query_one("#audio-save-profile-btn", Button)
        save_button.focus()
        await pilot.pause()

        assert save_button.has_focus
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: isinstance(
                app.screen,
                profile_library_module.TTSProfileNameModal,
            ),
        )
        modal = app.screen
        save_workers = tuple(
            worker
            for worker in app.workers
            if getattr(worker, "group", None) == "save_tts_result_as_profile"
        )
        assert len(save_workers) == 1
        assert not save_workers[0].is_finished

        if cancel_mode == "worker":
            save_workers[0].cancel()
        else:
            await pane.remove()

        await _wait_until(pilot, lambda: app.screen is not modal)
        await _wait_until(
            pilot,
            lambda: all(worker.is_finished for worker in save_workers),
        )

        assert not isinstance(
            app.screen,
            profile_library_module.TTSProfileNameModal,
        )
        assert profile_service.create_calls == []
        assert app.profile_service_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_copy"),
    (
        (
            ProfileRepositoryError("conflict"),
            profile_library_module.PROFILE_CONFLICT_COPY,
        ),
        (
            ProfileRepositoryError("stale"),
            profile_library_module.PROFILE_CONFLICT_COPY,
        ),
        (
            ProfileRepositoryError("unavailable"),
            profile_library_module.PROFILE_STORE_UNAVAILABLE_COPY,
        ),
        (
            ProfileServiceError("stale_configuration"),
            (
                "TTS settings changed after this audio was generated. Generate a new "
                "result before saving it as a profile."
            ),
        ),
    ),
)
async def test_save_profile_failures_use_value_independent_recovery_copy(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: BaseException,
    expected_copy: str,
) -> None:
    del audio_cpp_playground
    profile_service = _SaveProfileService()
    profile_service.error = error
    app = _PaneHost(profile_service=profile_service)
    artifact = _native_profile_artifact(tmp_path / "result.wav")

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(artifact, announce=False)

        async def _name(_screen: object) -> str:
            return "Safe name"

        monkeypatch.setattr(app, "push_screen_wait", _name)
        await pane._save_current_result_as_profile()

        status = str(app.query_one("#audio-player-status", Static).render())
        assert status == expected_copy
        assert app.profile_service_requests == 1


@pytest.mark.asyncio
async def test_unavailable_lazy_profile_service_keeps_audio_artifact_owned_by_handler(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    app = _PaneHost(profile_service=None)
    artifact = _native_profile_artifact(tmp_path / "retained.wav")

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(artifact, announce=False)

        async def _name(_screen: object) -> str:
            return "Unavailable store"

        monkeypatch.setattr(app, "push_screen_wait", _name)
        await pane._save_current_result_as_profile()

        assert (
            str(app.query_one("#audio-player-status", Static).render())
            == profile_library_module.PROFILE_STORE_UNAVAILABLE_COPY
        )
        assert app.profile_service_requests == 1
