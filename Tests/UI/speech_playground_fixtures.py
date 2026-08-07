"""Shared fakes and helpers for `SpeechPlaygroundPane` test coverage.

Extracted from `test_stts_playground_audio_cpp.py` (TASK-2951) when that
file -- which drove the retired legacy playground widget exclusively -- was
deleted. `FakeTTSService`, `_resolved`, `_wait_until`, `_profile_preset` and
`_native_profile_artifact` were never widget-specific: they are the
provider/service fake and small builders several pane-focused test modules
already depended on (imported straight from the widget's test module, which
is why this file exists rather than everyone reimplementing them).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.TTS import (
    TTSPlaygroundSelectionPreset,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.legacy_catalogs import legacy_catalog
from tldw_chatbook.TTS.playground_types import STTSGeneratedAudio

#: Every provider id the fake service and catalogs below know about.
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
        self.voice_observation_calls: list[tuple[str, str, bool]] = []
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
        self.voice_states: dict[tuple[str, str], str] = {}
        self.catalog_started: asyncio.Event | None = None
        self.allow_catalog: asyncio.Event | None = None
        self.catalog_cancelled = False
        self.catalog_error: Exception | None = None
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
        if self.catalog_error is not None:
            raise self.catalog_error
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

    async def observe_voices(
        self,
        provider_id: str,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        self.voice_observation_calls.append((provider_id, model_id, refresh))
        voices = await self.get_voices(provider_id, model_id, refresh=refresh)
        catalog = self.catalogs[provider_id]
        model_ids = {model.model_id for model in catalog.models}
        state = self.voice_states.get(
            (provider_id, model_id),
            "complete" if model_id in model_ids else "model_missing",
        )
        return TTSVoiceDiscoveryResult(
            provider_id=provider_id,
            model_id=model_id,
            catalog_revision=catalog.revision,
            voices=voices,
            state=state,  # type: ignore[arg-type]
        )

    async def synthesize(self, *_args: Any, **_kwargs: Any) -> None:
        self.synthesize_calls += 1
        raise AssertionError("Task 4 must not synthesize")


async def _resolved(value: Any) -> Any:
    return value


async def _wait_until(
    pilot: Any,
    predicate: Callable[[], bool],
) -> None:
    for _ in range(100):
        if predicate():
            return
        await pilot.pause(0.02)
    pytest.fail("Timed out waiting for Playground state")


def _profile_preset(
    *,
    provider_id: str = "audio_cpp",
    model_id: str = "profile/model",
    voice_id: str | None = "profile/voice",
    availability: str = "available",
) -> TTSPlaygroundSelectionPreset:
    return TTSPlaygroundSelectionPreset(
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        response_format="wav",
        speed=1.0,
        options={},
        availability=availability,  # type: ignore[arg-type]
    )


def _native_profile_artifact(
    path: Path,
    *,
    model_id: str = "artifact/model",
    voice_id: str | None = "artifact/voice",
    operation_id: str = "profile-operation",
) -> STTSGeneratedAudio:
    selection = TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id=model_id,
        voice_id=voice_id,
        response_format="wav",
        speed=1.0,
        options={},
        configuration_revision=4,
    )
    return STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="response/model",
        voice_id=None,
        source_text="private preview text",
        operation_id=operation_id,
        audio_format="wav",
        content_type="audio/wav",
        requested_selection=selection,
    )
