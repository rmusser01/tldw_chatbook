from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

import pytest

from tldw_chatbook.TTS.adapter_types import (
    TTSAudioResponse,
    TTSProgress,
    TTSRequest,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import (
    LEGACY_PROVIDER_IDS,
    LEGACY_ROUTES,
    LegacyBackendHost,
    LegacyTTSAdapter,
    UnknownLegacyModelError,
    legacy_provider_specs,
    resolve_legacy_route,
)
from tldw_chatbook.TTS.legacy_catalogs import legacy_catalog

ProgressCallback = Callable[[Mapping[str, Any]], Awaitable[None]]

EXPECTED_ROUTES = {
    "openai_official_tts-1": "openai",
    "openai_official_tts-1-hd": "openai",
    "openai_official_tts1": "openai",
    "openai_official_tts1hd": "openai",
    "elevenlabs_eleven_monolingual_v1": "elevenlabs",
    "elevenlabs_eleven_multilingual_v1": "elevenlabs",
    "elevenlabs_eleven_multilingual_v2": "elevenlabs",
    "elevenlabs_eleven_turbo_v2": "elevenlabs",
    "elevenlabs_eleven_turbo_v2_5": "elevenlabs",
    "elevenlabs_eleven_flash_v2": "elevenlabs",
    "elevenlabs_eleven_flash_v2_5": "elevenlabs",
    "elevenlabs_english_v1": "elevenlabs",
    "elevenlabs_elevenlabs": "elevenlabs",
    "local_kokoro_default_onnx": "kokoro",
    "local_kokoro_default_pytorch": "kokoro",
    "local_chatterbox_default": "chatterbox",
    "local_higgs_default": "higgs",
    "local_higgs_v2": "higgs",
    "alltalk_default": "alltalk",
    "alltalk_alltalk": "alltalk",
}
REPRESENTATIVE_ROUTES = {
    "openai": "openai_official_tts-1",
    "elevenlabs": "elevenlabs_eleven_multilingual_v2",
    "kokoro": "local_kokoro_default_onnx",
    "chatterbox": "local_chatterbox_default",
    "higgs": "local_higgs_v2",
    "alltalk": "alltalk_default",
}
EXPECTED_MODELS = {
    "openai": ("tts-1", "tts-1-hd"),
    "elevenlabs": (
        "eleven_monolingual_v1",
        "eleven_multilingual_v1",
        "eleven_multilingual_v2",
        "eleven_turbo_v2",
        "eleven_turbo_v2_5",
        "eleven_flash_v2",
        "eleven_flash_v2_5",
    ),
    "kokoro": ("kokoro",),
    "chatterbox": ("chatterbox",),
    "higgs": ("higgs-audio-v2",),
    "alltalk": ("alltalk",),
}
EXPECTED_VOICES = {
    "openai": (
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "onyx",
        "nova",
        "sage",
        "shimmer",
        "verse",
    ),
    "elevenlabs": (
        "21m00Tcm4TlvDq8ikWAM",
        "AZnzlk1XvdvUeBnXmlld",
        "EXAVITQu4vr4xnSDxMaL",
        "ErXwobaYiN019PkySvjV",
        "MF3mGyEYCl7XYWbV9V6O",
        "TxGEqnHWrfWFTfGW9XjX",
        "VR6AewLTigWG4xSOukaG",
        "pNInz6obpgDQGcFmaJgB",
        "yoZ06aMxZJJ28mfd3POQ",
    ),
    "kokoro": (
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_michael",
        "bf_emma",
        "bf_isabella",
        "bm_george",
        "bm_lewis",
    ),
    "chatterbox": ("default",),
    "higgs": ("default",),
    "alltalk": ("female_01.wav", "male_01.wav"),
}
EXPECTED_OPTIONS = {
    "openai": (),
    "elevenlabs": (
        "stability",
        "similarity_boost",
        "style",
        "use_speaker_boost",
    ),
    "kokoro": ("language", "use_onnx"),
    "chatterbox": (
        "exaggeration",
        "cfg_weight",
        "temperature",
        "num_candidates",
        "validate_with_whisper",
    ),
    "higgs": (
        "temperature",
        "top_p",
        "repetition_penalty",
        "language",
    ),
    "alltalk": ("language",),
}
EXPECTED_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "elevenlabs": "ElevenLabs",
    "kokoro": "Kokoro (Local)",
    "chatterbox": "Chatterbox (Local)",
    "higgs": "Higgs Audio (Local)",
    "alltalk": "AllTalk (Local)",
}
ALL_VISIBLE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")


class FakeLegacyBackend:
    def __init__(self) -> None:
        self.progress_callback: ProgressCallback | None = None
        self.generated_requests: list[OpenAISpeechRequest] = []

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        self.progress_callback = callback

    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncIterator[bytes]:
        self.generated_requests.append(request)
        if self.progress_callback is not None:
            await self.progress_callback(
                {
                    "status": f"Generating {request.input}",
                    "progress": 0.5,
                    "processed": 2,
                    "total": 4,
                    "metrics": {
                        "rate": 1.25,
                        "cached": False,
                        "ignored": object(),
                    },
                }
            )
        yield b"audio"


class BlockingLegacyBackend(FakeLegacyBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.allow_finish = asyncio.Event()
        self.active_generations = 0
        self.max_concurrent_generations = 0

    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncIterator[bytes]:
        self.generated_requests.append(request)
        self.active_generations += 1
        self.max_concurrent_generations = max(
            self.max_concurrent_generations,
            self.active_generations,
        )
        self.started.set()
        try:
            await self.allow_finish.wait()
            if self.progress_callback is not None:
                await self.progress_callback(
                    {
                        "status": f"Complete {request.input}",
                        "progress": 1.0,
                    }
                )
            yield b"audio"
        finally:
            self.active_generations -= 1


class FakeLegacyManager:
    def __init__(
        self,
        backend: FakeLegacyBackend | None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.config = dict(config or {})
        self.backend_ids: list[str] = []
        self.close_calls = 0

    async def get_backend(self, internal_model_id: str) -> FakeLegacyBackend | None:
        self.backend_ids.append(internal_model_id)
        return self.backend

    async def close_all_backends(self) -> None:
        self.close_calls += 1


def speech_request(text: str = "hello") -> OpenAISpeechRequest:
    return OpenAISpeechRequest(
        model="kokoro",
        input=text,
        voice="af_heart",
        response_format="wav",
    )


def adapter_request(
    provider_id: str,
    internal_model_id: str,
    *,
    legacy_request: object | None = None,
    extra_options: Mapping[str, Any] | None = None,
) -> TTSRequest:
    options = {
        "_legacy_openai_request": (
            speech_request() if legacy_request is None else legacy_request
        ),
        "_legacy_internal_model_id": internal_model_id,
    }
    options.update(extra_options or {})
    return TTSRequest(
        provider_id=provider_id,
        model_id=EXPECTED_MODELS[provider_id][0],
        text="hello",
        voice=EXPECTED_VOICES[provider_id][0],
        response_format="wav",
        options=options,
    )


async def collect(stream: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in stream])


def progress_sink(
    target: list[TTSProgress],
) -> Callable[[TTSProgress], Awaitable[None]]:
    async def record(progress: TTSProgress) -> None:
        target.append(progress)

    return record


@pytest.mark.parametrize(
    ("internal_id", "provider_id"),
    EXPECTED_ROUTES.items(),
)
def test_resolver_uses_only_enumerated_routes(
    internal_id: str,
    provider_id: str,
) -> None:
    route = resolve_legacy_route(internal_id)

    assert route.provider_id == provider_id
    assert route.internal_model_id == internal_id


def test_route_table_is_exact() -> None:
    assert LEGACY_ROUTES == EXPECTED_ROUTES


@pytest.mark.parametrize(
    "internal_id",
    [
        "openai_official_",
        "openai_official_new-model",
        "elevenlabs_custom_unknown",
        "local_kokoro_evil",
        "local_chatterbox_",
        "local_higgs_v3",
        "alltalk_custom",
    ],
)
def test_resolver_rejects_unlisted_internal_ids(
    internal_id: str,
) -> None:
    with pytest.raises(
        UnknownLegacyModelError,
        match="selected TTS model is not available",
    ):
        resolve_legacy_route(internal_id)


def test_provider_specs_and_catalogs_preserve_compatibility_values() -> None:
    specs = legacy_provider_specs({})

    assert LEGACY_PROVIDER_IDS == tuple(EXPECTED_MODELS)
    assert tuple(spec.descriptor.provider_id for spec in specs) == LEGACY_PROVIDER_IDS
    assert {
        spec.descriptor.provider_id: spec.descriptor.display_name for spec in specs
    } == EXPECTED_DISPLAY_NAMES
    assert all(spec.descriptor.native is False for spec in specs)

    for provider_id in LEGACY_PROVIDER_IDS:
        catalog = legacy_catalog(provider_id)
        assert catalog.provider_id == provider_id
        assert catalog.revision == 1
        assert catalog.health.state == "available"
        assert catalog.health.fresh is True
        assert catalog.approximate is True
        assert (
            tuple(model.model_id for model in catalog.models)
            == EXPECTED_MODELS[provider_id]
        )
        for model in catalog.models:
            assert model.family == provider_id
            assert model.upstream_mode == "legacy"
            assert model.formats == ALL_VISIBLE_FORMATS
            assert model.voices == EXPECTED_VOICES[provider_id]
            assert model.supports_speed is True
            assert model.supports_options == EXPECTED_OPTIONS[provider_id]


@pytest.mark.asyncio
async def test_manager_and_backend_construction_are_lazy() -> None:
    managers: dict[str, FakeLegacyManager] = {}

    def create_manager(
        provider_id: str,
        config: dict[str, Any],
    ) -> FakeLegacyManager:
        manager = FakeLegacyManager(FakeLegacyBackend(), config)
        managers[provider_id] = manager
        return manager

    adapters = {
        spec.descriptor.provider_id: spec.factory(spec.initial_config)
        for spec in legacy_provider_specs(
            {"app_tts": {"default_format": "wav"}},
            manager_factory=create_manager,
        )
    }

    assert managers == {}
    for adapter in adapters.values():
        await adapter.ensure_ready()
        await adapter.get_catalog()
    assert managers == {}

    response = await adapters["openai"].synthesize(
        adapter_request("openai", REPRESENTATIVE_ROUTES["openai"])
    )
    assert managers == {}

    assert await collect(response.byte_stream) == b"audio"
    assert set(managers) == {"openai"}
    assert managers["openai"].backend_ids == ["openai_official_tts-1"]

    for adapter in adapters.values():
        await adapter.close()
    assert managers["openai"].close_calls == 1


@pytest.mark.asyncio
async def test_provider_hosts_copy_config_and_close_materialized_managers_once() -> (
    None
):
    managers: dict[str, FakeLegacyManager] = {}

    def create_manager(
        provider_id: str,
        config: dict[str, Any],
    ) -> FakeLegacyManager:
        manager = FakeLegacyManager(FakeLegacyBackend(), config)
        managers[provider_id] = manager
        return manager

    source_config = {"app_tts": {"default_format": "wav"}}
    specs = legacy_provider_specs(
        source_config,
        manager_factory=create_manager,
    )
    source_config["app_tts"]["default_format"] = "mp3"
    adapters = {
        spec.descriptor.provider_id: spec.factory(spec.initial_config) for spec in specs
    }

    assert len({id(adapter.host) for adapter in adapters.values()}) == 6
    for provider_id, adapter in adapters.items():
        response = await adapter.synthesize(
            adapter_request(provider_id, REPRESENTATIVE_ROUTES[provider_id])
        )
        assert await collect(response.byte_stream) == b"audio"

    assert set(managers) == set(LEGACY_PROVIDER_IDS)
    assert len({id(manager) for manager in managers.values()}) == 6
    assert len({id(manager.config) for manager in managers.values()}) == 6
    assert all(
        manager.config == {"app_tts": {"default_format": "wav"}}
        for manager in managers.values()
    )

    for adapter in adapters.values():
        await adapter.close()
        await adapter.close()
    assert all(manager.close_calls == 1 for manager in managers.values())


@pytest.mark.asyncio
async def test_same_backend_serializes_progress_through_stream_consumption() -> None:
    backend = BlockingLegacyBackend()
    manager = FakeLegacyManager(backend)
    host = LegacyBackendHost(
        provider_id="kokoro",
        app_config={"app_tts": {"KOKORO_DEVICE": "cpu"}},
        manager_factory=lambda _: manager,
    )
    first_progress: list[TTSProgress] = []
    second_progress: list[TTSProgress] = []

    first_task = asyncio.create_task(
        collect(
            host.generate(
                "local_kokoro_default_onnx",
                speech_request("first"),
                progress_sink(first_progress),
            )
        )
    )
    await backend.started.wait()
    second_task = asyncio.create_task(
        collect(
            host.generate(
                "local_kokoro_default_onnx",
                speech_request("second"),
                progress_sink(second_progress),
            )
        )
    )
    await asyncio.sleep(0)

    assert manager.backend_ids == ["local_kokoro_default_onnx"]
    assert backend.active_generations == 1

    backend.allow_finish.set()
    assert await first_task == b"audio"
    assert await second_task == b"audio"

    assert manager.backend_ids == [
        "local_kokoro_default_onnx",
        "local_kokoro_default_onnx",
    ]
    assert backend.max_concurrent_generations == 1
    assert [progress.status for progress in first_progress] == ["Complete first"]
    assert [progress.status for progress in second_progress] == ["Complete second"]
    assert backend.progress_callback is None


@pytest.mark.asyncio
async def test_different_provider_hosts_generate_concurrently() -> None:
    openai_backend = BlockingLegacyBackend()
    kokoro_backend = BlockingLegacyBackend()
    openai = LegacyBackendHost(
        provider_id="openai",
        app_config={},
        manager_factory=lambda _: FakeLegacyManager(openai_backend),
    )
    kokoro = LegacyBackendHost(
        provider_id="kokoro",
        app_config={},
        manager_factory=lambda _: FakeLegacyManager(kokoro_backend),
    )
    first = asyncio.create_task(
        collect(
            openai.generate(
                "openai_official_tts-1",
                speech_request("openai"),
                None,
            )
        )
    )
    second = asyncio.create_task(
        collect(
            kokoro.generate(
                "local_kokoro_default_onnx",
                speech_request("kokoro"),
                None,
            )
        )
    )

    await asyncio.wait_for(
        asyncio.gather(
            openai_backend.started.wait(),
            kokoro_backend.started.wait(),
        ),
        timeout=1,
    )
    openai_backend.allow_finish.set()
    kokoro_backend.allow_finish.set()

    assert await asyncio.gather(first, second) == [b"audio", b"audio"]


@pytest.mark.asyncio
async def test_adapter_maps_progress_and_partial_response_close_releases_backend() -> (
    None
):
    backend = FakeLegacyBackend()
    manager = FakeLegacyManager(backend)
    host = LegacyBackendHost(
        provider_id="kokoro",
        app_config={},
        manager_factory=lambda _: manager,
    )
    adapter = LegacyTTSAdapter(
        "kokoro",
        host,
        legacy_catalog("kokoro"),
    )
    progress: list[TTSProgress] = []

    response = await adapter.synthesize(
        adapter_request("kokoro", "local_kokoro_default_onnx"),
        progress_sink(progress),
    )

    assert isinstance(response, TTSAudioResponse)
    assert response.provider_id == "kokoro"
    assert response.model_id == "kokoro"
    assert response.audio_format == "wav"
    assert response.content_type == "audio/wav"
    assert await anext(response.byte_stream) == b"audio"
    assert backend.progress_callback is not None
    assert progress == [
        TTSProgress(
            status="Generating hello",
            fraction=0.5,
            processed=2,
            total=4,
            metrics={"rate": 1.25, "cached": False},
        )
    ]

    await response.aclose()
    assert backend.progress_callback is None

    next_response = await adapter.synthesize(
        adapter_request("kokoro", "local_kokoro_default_onnx")
    )
    assert (
        await asyncio.wait_for(
            collect(next_response.byte_stream),
            timeout=1,
        )
        == b"audio"
    )


@pytest.mark.asyncio
async def test_adapter_rejects_invalid_or_cross_provider_routes_before_host_use() -> (
    None
):
    manager_factory_calls = 0

    def create_manager(_: dict[str, Any]) -> FakeLegacyManager:
        nonlocal manager_factory_calls
        manager_factory_calls += 1
        return FakeLegacyManager(FakeLegacyBackend())

    adapter = LegacyTTSAdapter(
        "kokoro",
        LegacyBackendHost(
            provider_id="kokoro",
            app_config={},
            manager_factory=create_manager,
        ),
        legacy_catalog("kokoro"),
    )

    with pytest.raises(ValueError, match="Invalid legacy adapter options"):
        await adapter.synthesize(
            adapter_request(
                "kokoro",
                "local_kokoro_default_onnx",
                extra_options={"unexpected": True},
            )
        )
    with pytest.raises(TypeError, match="Legacy request must be"):
        await adapter.synthesize(
            adapter_request(
                "kokoro",
                "local_kokoro_default_onnx",
                legacy_request={"input": "hello"},
            )
        )
    with pytest.raises(ValueError, match="route does not match provider"):
        await adapter.synthesize(adapter_request("kokoro", "openai_official_tts-1"))
    with pytest.raises(UnknownLegacyModelError):
        await adapter.synthesize(adapter_request("kokoro", "local_kokoro_unlisted"))

    assert manager_factory_calls == 0


@pytest.mark.asyncio
async def test_missing_backend_fails_and_closed_host_stays_unmaterialized() -> None:
    manager = FakeLegacyManager(None)
    host = LegacyBackendHost(
        provider_id="alltalk",
        app_config={},
        manager_factory=lambda _: manager,
    )

    with pytest.raises(ValueError, match="is not available"):
        await collect(
            host.generate(
                "alltalk_default",
                speech_request(),
                None,
            )
        )

    await host.close()
    await host.close()
    assert manager.close_calls == 1
    with pytest.raises(RuntimeError, match="host is closed"):
        await collect(
            host.generate(
                "alltalk_default",
                speech_request(),
                None,
            )
        )

    unmaterialized_factory_calls = 0

    def create_unmaterialized(_: dict[str, Any]) -> FakeLegacyManager:
        nonlocal unmaterialized_factory_calls
        unmaterialized_factory_calls += 1
        return FakeLegacyManager(FakeLegacyBackend())

    unmaterialized = LegacyBackendHost(
        provider_id="openai",
        app_config={},
        manager_factory=create_unmaterialized,
    )
    await unmaterialized.close()
    await unmaterialized.close()

    assert unmaterialized_factory_calls == 0
    with pytest.raises(RuntimeError, match="host is closed"):
        await collect(
            unmaterialized.generate(
                "openai_official_tts-1",
                speech_request(),
                None,
            )
        )
