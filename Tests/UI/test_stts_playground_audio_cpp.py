"""Playground behaviour, driven through the rebuilt pane.

These tests were written against `TTSPlaygroundWidget` and moved with the
code: the catalog, synthesis and playback paths they cover now live in the
shared mixins, and `SpeechPlaygroundPane` is the surface that hosts them.
Porting them found nine real defects in the pane (task-1266), so the
assertions are unchanged -- only the host and the two patch seams differ.

Service and settings are patched through the pane's `_tts_service_factory`
and `_cli_setting` hooks rather than on the `STTS_Window` module, which is
where they lived back when the code did.
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
from textual.widgets import Button, Input, Select, Static, TextArea

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.TTS.audio_player import PlaybackState
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.playground_types import STTSGeneratedAudio
from tldw_chatbook.TTS.legacy_catalogs import legacy_catalog
from tldw_chatbook.UI import STTS_Window
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.stts_playground_catalog import (
    LOADING_SELECT_VALUE,
    SERVER_DEFAULT_VOICE_ID,
)


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


class FakeTTSService:
    def __init__(self) -> None:
        self.descriptor_calls = 0
        self.catalog_calls: list[tuple[str, bool]] = []
        self.voice_calls: list[tuple[str, str, bool]] = []
        self.synthesize_calls = 0
        self.revisions = {provider_id: 1 for provider_id in PROVIDER_IDS}
        self.catalogs = {
            "audio_cpp": _audio_catalog(),
            **{
                provider_id: legacy_catalog(provider_id)
                for provider_id in PROVIDER_IDS
                if provider_id != "audio_cpp"
            },
        }
        self.voices: dict[tuple[str, str], tuple[str, ...]] = {
            ("audio_cpp", "<opaque:model>"): (
                "[voice]",
                "<script>alert(1)</script>",
            ),
            ("audio_cpp", "second-model"): ("second-voice",),
        }
        self.catalog_started: asyncio.Event | None = None
        self.allow_catalog: asyncio.Event | None = None
        self.catalog_cancelled = False
        self.voice_started: asyncio.Event | None = None
        self.allow_voices: asyncio.Event | None = None
        self.voice_error: Exception | None = None
        self.voice_started_by_request: dict[
            tuple[str, str],
            asyncio.Event,
        ] = {}
        self.voice_finished_by_request: dict[
            tuple[str, str],
            asyncio.Event,
        ] = {}
        self.voice_gates: dict[tuple[str, str], asyncio.Event] = {}
        self.voice_errors: dict[tuple[str, str], Exception] = {}
        self.voice_ignore_cancellation: set[tuple[str, str]] = set()

    def provider_descriptors(self) -> tuple[TTSProviderDescriptor, ...]:
        self.descriptor_calls += 1
        return tuple(
            TTSProviderDescriptor(
                provider_id=provider_id,
                display_name=(
                    "[b]audio.cpp[/]"
                    if provider_id == "audio_cpp"
                    else provider_id.title()
                ),
                native=provider_id == "audio_cpp",
            )
            for provider_id in PROVIDER_IDS
        )

    def configuration_revision(self, provider_id: str) -> int:
        return self.revisions[provider_id]

    async def get_catalog(
        self,
        provider_id: str,
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        self.catalog_calls.append((provider_id, refresh))
        if self.catalog_started is not None:
            self.catalog_started.set()
        if self.allow_catalog is not None:
            try:
                await self.allow_catalog.wait()
            except asyncio.CancelledError:
                self.catalog_cancelled = True
                await self.allow_catalog.wait()
        return self.catalogs[provider_id]

    async def get_voices(
        self,
        provider_id: str,
        model_id: str,
        refresh: bool = False,
    ) -> tuple[str, ...]:
        self.voice_calls.append((provider_id, model_id, refresh))
        request_key = (provider_id, model_id)
        if self.voice_started is not None:
            self.voice_started.set()
        request_started = self.voice_started_by_request.get(request_key)
        if request_started is not None:
            request_started.set()
        try:
            gate = self.voice_gates.get(request_key, self.allow_voices)
            if gate is not None:
                try:
                    await gate.wait()
                except asyncio.CancelledError:
                    if request_key not in self.voice_ignore_cancellation:
                        raise
                    await gate.wait()
            error = self.voice_errors.get(request_key, self.voice_error)
            if error is not None:
                raise error
            return self.voices.get(request_key, ())
        finally:
            request_finished = self.voice_finished_by_request.get(request_key)
            if request_finished is not None:
                request_finished.set()

    async def synthesize(self, *_args: Any, **_kwargs: Any) -> None:
        self.synthesize_calls += 1
        raise AssertionError("Task 4 must not synthesize")


class _PlaygroundHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.notices: list[tuple[str, str]] = []
        self.generation_events: list[STTSPlaygroundGenerateEvent] = []

    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane()

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


@pytest.fixture
def audio_cpp_playground(
    monkeypatch: pytest.MonkeyPatch,
) -> FakeTTSService:
    service = FakeTTSService()

    def get_setting(section: str, key: str, default: Any = None) -> Any:
        if (section, key) == ("app_tts", "default_provider"):
            return "audio_cpp"
        return default

    monkeypatch.setattr(SpeechPlaygroundPane, "_cli_setting",
                        lambda self, *a, **k: get_setting(*a, **k))
    monkeypatch.setattr(
        SpeechPlaygroundPane, "_tts_service_factory", lambda self: _resolved(service)
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_check_higgs_installation",
        lambda self: None,
    )
    return service


async def _resolved(value: Any) -> Any:
    return value


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


async def _wait_until(
    pilot: Any,
    predicate: Callable[[], bool],
) -> None:
    for _ in range(100):
        if predicate():
            return
        await pilot.pause(0.02)
    pytest.fail("Timed out waiting for Playground state")


@pytest.mark.asyncio
async def test_mount_uses_descriptors_and_resolves_only_selected_provider(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
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
    assert service.synthesize_calls == 0


@pytest.mark.asyncio
async def test_playground_generates_with_sentinel_shaped_remote_ids(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    remote_model_id = "__opaque_model__"
    remote_voice_id = "__server_default__"
    service.catalogs["audio_cpp"] = TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=12,
        health=ProviderHealth(state="available", fresh=True),
        models=(
            TTSModelInfo(
                model_id=remote_model_id,
                display_name="Opaque model",
                family="test",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
        ),
    )
    service.voices[("audio_cpp", remote_model_id)] = (remote_voice_id,)
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        widget = app.query_one(SpeechPlaygroundPane)
        model_select = app.query_one("#tts-model-select", Select)
        voice_select = app.query_one("#tts-voice-select", Select)

        assert model_select.value == remote_model_id
        assert service.voice_calls == [("audio_cpp", remote_model_id, False)]
        assert _option_values(voice_select) == (
            SERVER_DEFAULT_VOICE_ID,
            remote_voice_id,
        )

        voice_select.value = remote_voice_id
        await pilot.pause()
        widget._generate_tts()

        request = app.generation_events[-1].request
        assert request.model_id == remote_model_id
        assert request.voice_id == remote_voice_id


@pytest.mark.asyncio
async def test_configuration_change_marks_catalog_stale_without_connecting(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
) -> None:
    service = audio_cpp_playground
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        widget = app.query_one(SpeechPlaygroundPane)
        model_values = _option_values(app.query_one("#tts-model-select", Select))
        calls_before = list(service.catalog_calls)

        widget.current_audio_file = tmp_path / "existing.wav"
        app.query_one("#audio-play-btn", Button).disabled = False
        app.query_one("#audio-export-btn", Button).disabled = False
        service.revisions["audio_cpp"] = 2
        widget.mark_provider_configuration_changed("audio_cpp", 2)
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
    app = _PlaygroundHost()

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
async def test_superseded_catalog_failure_cannot_overwrite_newer_success(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)
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
            return newer_catalog

        monkeypatch.setattr(service, "get_catalog", get_catalog)

        widget._load_provider_catalog("audio_cpp", refresh=True)
        await first_started.wait()
        widget._load_provider_catalog("audio_cpp", refresh=True)
        await _wait_until(
            pilot,
            lambda: (
                call_count == 2
                and widget._catalogs.get("audio_cpp") is newer_catalog
                and "ready"
                in str(app.query_one("#tts-provider-status", Static).render()).lower()
            ),
        )

        release_first.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert widget._catalogs["audio_cpp"] is newer_catalog
        assert "audio_cpp" not in widget._stale_providers
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
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)
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
            return newer_catalog

        monkeypatch.setattr(service, "get_catalog", get_catalog)

        widget._load_provider_catalog("audio_cpp", refresh=True)
        await first_started.wait()
        widget._load_provider_catalog("audio_cpp", refresh=True)
        await _wait_until(
            pilot,
            lambda: (
                call_count == 2
                and widget._catalogs.get("audio_cpp") is newer_catalog
                and "ready"
                in str(app.query_one("#tts-provider-status", Static).render()).lower()
            ),
        )

        release_first.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert widget._catalogs["audio_cpp"] is newer_catalog
        assert "audio_cpp" not in widget._stale_providers
        assert widget._catalog_generation_allowed is True
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
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)
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

        widget._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )
        await first_started.wait()
        widget._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )
        await _wait_until(
            pilot,
            lambda: (
                call_count == 2
                and widget._discovered_voices.get(("audio_cpp", model_id))
                == ("new-voice",)
            ),
        )

        release_first.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert widget._discovered_voices[("audio_cpp", model_id)] == ("new-voice",)
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
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)
        baseline_catalog = widget._catalogs["audio_cpp"]
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
            return newer_catalog

        monkeypatch.setattr(service, "get_catalog", get_catalog)

        widget._load_provider_catalog("audio_cpp", refresh=True)
        await first_started.wait()
        first_generation = widget._catalog_request_generations["audio_cpp"]

        widget._load_provider_catalog("audio_cpp", refresh=True)

        assert widget._catalog_request_generations["audio_cpp"] == (
            first_generation + 1
        )
        await first_returned_on_cancel.wait()
        await second_started.wait()
        await pilot.pause()
        assert widget._catalogs["audio_cpp"] is baseline_catalog

        release_second.set()
        await app.workers.wait_for_complete()
        assert widget._catalogs["audio_cpp"] is newer_catalog


@pytest.mark.asyncio
async def test_voice_generation_is_reserved_before_exclusive_worker_cancellation(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)
        model_id = "<opaque:model>"
        request_key = ("audio_cpp", model_id)
        baseline_voices = widget._discovered_voices[request_key]
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

        widget._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )
        await first_started.wait()
        first_generation = widget._voice_request_generations[request_key]

        widget._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )

        assert widget._voice_request_generations[request_key] == first_generation + 1
        await first_returned_on_cancel.wait()
        await second_started.wait()
        await pilot.pause()
        assert widget._discovered_voices[request_key] == baseline_voices

        release_second.set()
        await app.workers.wait_for_complete()
        assert widget._discovered_voices[request_key] == ("new-voice",)


@pytest.mark.asyncio
async def test_voice_discovery_does_not_cancel_inflight_catalog_refresh(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        service.catalog_started = asyncio.Event()
        service.allow_catalog = asyncio.Event()
        widget = app.query_one(SpeechPlaygroundPane)

        widget._load_provider_catalog("audio_cpp", refresh=True)
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
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.value = "[voice]"
        app.query_one("#tts-text-input", TextArea).text = "pending voice"
        await pilot.pause()

        service.catalogs["audio_cpp"] = _audio_catalog(revision=12)
        service.voice_started = asyncio.Event()
        service.allow_voices = asyncio.Event()
        widget = app.query_one(SpeechPlaygroundPane)
        notices_before = list(app.notices)
        widget._load_provider_catalog("audio_cpp", refresh=True)
        await service.voice_started.wait()
        await pilot.pause()

        assert _option_values(voice_select) == (SERVER_DEFAULT_VOICE_ID,)
        assert voice_select.value == SERVER_DEFAULT_VOICE_ID
        assert app.notices == notices_before
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        widget.action_generate_tts()
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
async def test_catalog_revision_falls_back_only_after_refreshed_voice_is_removed(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.value = "[voice]"
        await pilot.pause()

        service.catalogs["audio_cpp"] = _audio_catalog(revision=12)
        service.voices[("audio_cpp", "<opaque:model>")] = ("replacement",)
        service.voice_started = asyncio.Event()
        service.allow_voices = asyncio.Event()
        notices_before = list(app.notices)
        widget = app.query_one(SpeechPlaygroundPane)
        widget._load_provider_catalog("audio_cpp", refresh=True)
        await service.voice_started.wait()
        await pilot.pause()

        assert app.notices == notices_before

        service.allow_voices.set()
        await app.workers.wait_for_complete()

        assert voice_select.value == SERVER_DEFAULT_VOICE_ID
        assert app.notices == [
            *notices_before,
            (
                "Available models or voices changed; a valid selection was chosen",
                "warning",
            ),
        ]


@pytest.mark.asyncio
async def test_voice_discovery_failure_releases_pending_explicit_voice(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
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

        assert voice_select.value == SERVER_DEFAULT_VOICE_ID
        assert app.query_one("#tts-generate-btn", Button).disabled is False
        assert (
            str(app.query_one("#tts-provider-status", Static).render())
            == "Voices are unavailable; the provider default remains available"
        )

        app.query_one(SpeechPlaygroundPane).action_generate_tts()
        await pilot.pause()

        assert len(app.generation_events) == 1
        assert app.generation_events[0].request.voice_id is None
        assert "untrusted upstream detail" not in str(app.notices)


@pytest.mark.asyncio
async def test_voice_discovery_failure_overrides_configured_explicit_default(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground

    def get_setting(section: str, key: str, default: Any = None) -> Any:
        configured = {
            ("app_tts", "default_provider"): "audio_cpp",
            ("app_tts", "default_voice"): "[voice]",
        }
        return configured.get((section, key), default)

    monkeypatch.setattr(SpeechPlaygroundPane, "_cli_setting",
                        lambda self, *a, **k: get_setting(*a, **k))
    service.voice_error = RuntimeError("untrusted upstream detail")
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        app.query_one("#tts-text-input", TextArea).text = "configured fallback"
        await pilot.pause()

        widget = app.query_one(SpeechPlaygroundPane)
        assert app.query_one("#tts-voice-select", Select).value == (
            SERVER_DEFAULT_VOICE_ID
        )
        assert widget._pending_voice_selections == {}
        assert app.query_one("#tts-generate-btn", Button).disabled is False

        widget.action_generate_tts()
        await pilot.pause()

        assert len(app.generation_events) == 1
        assert app.generation_events[0].request.voice_id is None
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

    def get_setting(section: str, key: str, default: Any = None) -> Any:
        configured = {
            ("app_tts", "default_provider"): "audio_cpp",
            ("app_tts", "default_voice"): "[voice]",
        }
        return configured.get((section, key), default)

    monkeypatch.setattr(SpeechPlaygroundPane, "_cli_setting",
                        lambda self, *a, **k: get_setting(*a, **k))
    service.voice_error = lifecycle_error
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        app.query_one("#tts-text-input", TextArea).text = "lifecycle pending"
        await pilot.pause()

        widget = app.query_one(SpeechPlaygroundPane)
        status = str(app.query_one("#tts-provider-status", Static).render()).lower()
        assert widget._pending_voice_selections == {"audio_cpp": "[voice]"}
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert expected_status in status

        widget.action_generate_tts()
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

    def get_setting(section: str, key: str, default: Any = None) -> Any:
        configured = {
            ("app_tts", "default_provider"): "audio_cpp",
            ("app_tts", "default_voice"): "[voice]",
        }
        return configured.get((section, key), default)

    monkeypatch.setattr(SpeechPlaygroundPane, "_cli_setting",
                        lambda self, *a, **k: get_setting(*a, **k))
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
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
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)
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

        widget._load_provider_voices(
            "audio_cpp",
            "<opaque:model>",
            service.catalogs["audio_cpp"].revision,
            refresh=True,
        )
        await service.voice_started_by_request[old_key].wait()

        model_select.value = "second-model"
        await service.voice_started_by_request[current_key].wait()
        await pilot.pause()

        assert widget._pending_voice_selections == {"audio_cpp": "[voice]"}
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        service.voice_gates[old_key].set()
        await service.voice_finished_by_request[old_key].wait()
        await pilot.pause()

        assert widget._pending_voice_selections == {"audio_cpp": "[voice]"}
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        widget.action_generate_tts()
        await pilot.pause()
        assert app.generation_events == []
        assert app.notices[-1] == (
            "Voices are still loading; wait before generating",
            "warning",
        )

        service.voice_gates[current_key].set()
        await service.voice_finished_by_request[current_key].wait()
        await _wait_until(pilot, lambda: widget._pending_voice_selections == {})


@pytest.mark.asyncio
async def test_legacy_control_state_is_restored_after_audio_cpp_switch(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
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
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
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
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
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
        (
            ProviderHealth(state="reconfiguring", fresh=False),
            "settings are being applied",
        ),
        (
            ProviderHealth(state="available", fresh=False),
            "catalog is stale",
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
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
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

    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)):
        await app.workers.wait_for_complete()
        monkeypatch.setattr(SpeechPlaygroundPane, "run_worker", run_worker)
        monkeypatch.setattr(
            SpeechPlaygroundPane,
            "_ensure_audio_player",
            lambda _self: True,
        )
        widget = app.query_one(SpeechPlaygroundPane)
        widget.current_audio_artifact = artifact
        widget.current_audio_file = artifact.path

        widget._play_audio()
        widget._play_audio()

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
    app = _PlaygroundHost()
    app._stts_handler = lease_handler
    player = Player()
    app.audio_player = player

    async with app.run_test(size=(180, 70)):
        await app.workers.wait_for_complete()
        monkeypatch.setattr(SpeechPlaygroundPane, "run_worker", run_worker)
        widget = app.query_one(SpeechPlaygroundPane)
        widget.current_audio_artifact = old_artifact
        widget.current_audio_file = old_path

        widget._play_audio()
        widget.current_audio_artifact = new_artifact
        widget.current_audio_file = new_path

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

    app = _PlaygroundHost()
    handler = STTSEventHandler(app)
    handler._accept_playground_artifact(old_artifact)
    app._stts_handler = handler
    player = Player()
    app.audio_player = player

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)

        widget._play_audio()
        await _wait_until(
            pilot,
            lambda: (
                player.state == PlaybackState.PLAYING
                and widget._play_worker_task is None
            ),
        )
        if pause_before_replacement:
            widget._pause_audio()
            await _wait_until(
                pilot,
                lambda: player.state == PlaybackState.PAUSED,
            )

        handler._accept_playground_artifact(new_artifact)
        widget._store_delivered_artifact(new_artifact, announce=False)

        assert old_path.exists()
        assert handler._playground_file_leases[old_path] == 1

        if terminal_action == "stop":
            await widget._stop_audio_async()
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
    app = _PlaygroundHost()
    app._stts_handler = lease_handler

    async with app.run_test(size=(180, 70)):
        await app.workers.wait_for_complete()
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda _screen, callback: callbacks.append(callback),
        )
        widget = app.query_one(SpeechPlaygroundPane)
        widget.current_audio_artifact = old_artifact
        widget.current_audio_file = old_path

        widget._export_audio()
        widget.current_audio_artifact = new_artifact
        widget.current_audio_file = new_path
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
    app = _PlaygroundHost()
    app._stts_handler = lease_handler

    async with app.run_test(size=(180, 70)):
        await app.workers.wait_for_complete()
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda _screen, callback: callbacks.append(callback),
        )
        widget = app.query_one(SpeechPlaygroundPane)
        widget.current_audio_artifact = artifact
        widget.current_audio_file = path

        widget._export_audio()
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
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)):
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)

        widget._handle_audio_export(
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
    app = _PlaygroundHost()
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
    app = _PlaygroundHost()
    app._stts_handler = SimpleNamespace(
        playground_state=lambda: state,
        cleanup_tts_resources=cleanup,
    )

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        widget = app.query_one(SpeechPlaygroundPane)

        assert widget.current_audio_artifact is artifact
        assert widget.current_audio_file == path
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
    app = _PlaygroundHost()
    app._stts_handler = SimpleNamespace(playground_state=lambda: state)

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        widget = app.query_one(SpeechPlaygroundPane)

        assert widget._generation_operation_id == "active-operation"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert (
            "in progress"
            in str(app.query_one("#generation-status-text", Static).render()).lower()
        )
        assert service.synthesize_calls == 0
