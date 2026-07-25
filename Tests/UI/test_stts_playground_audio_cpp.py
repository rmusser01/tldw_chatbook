from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderDescriptor,
)
from tldw_chatbook.TTS.legacy_catalogs import legacy_catalog
from tldw_chatbook.UI import STTS_Window
from tldw_chatbook.UI.STTS_Window import TTSPlaygroundWidget
from tldw_chatbook.UI.stts_playground_catalog import SERVER_DEFAULT_VOICE_ID


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
                await self.allow_catalog.wait()
        return self.catalogs[provider_id]

    async def get_voices(
        self,
        provider_id: str,
        model_id: str,
        refresh: bool = False,
    ) -> tuple[str, ...]:
        self.voice_calls.append((provider_id, model_id, refresh))
        return self.voices.get((provider_id, model_id), ())

    async def synthesize(self, *_args: Any, **_kwargs: Any) -> None:
        self.synthesize_calls += 1
        raise AssertionError("Task 4 must not synthesize")


class _PlaygroundHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.notices: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield TTSPlaygroundWidget()

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


@pytest.fixture
def audio_cpp_playground(
    monkeypatch: pytest.MonkeyPatch,
) -> FakeTTSService:
    service = FakeTTSService()

    def get_setting(section: str, key: str, default: Any = None) -> Any:
        if (section, key) == ("app_tts", "default_provider"):
            return "audio_cpp"
        return default

    monkeypatch.setattr(STTS_Window, "get_cli_setting", get_setting)
    monkeypatch.setattr(
        STTS_Window,
        "get_tts_service",
        lambda: _resolved(service),
    )
    monkeypatch.setattr(
        TTSPlaygroundWidget,
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
async def test_configuration_change_marks_catalog_stale_without_connecting(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
) -> None:
    service = audio_cpp_playground
    app = _PlaygroundHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        widget = app.query_one(TTSPlaygroundWidget)
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
            "__loading__",
        )
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        status = str(app.query_one("#tts-provider-status", Static).render()).lower()
        assert "settings changed" in status


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
        assert _option_values(app.query_one("#tts-model-select", Select)) == (
            "<opaque:model>",
            "second-model",
        )
