from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from Tests.TTS.adapter_fakes import FakeAdapter
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSNativeCapabilityObservation,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderSpec,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.TTS_Generation import TTSService


class RevisionedCapabilityAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__("audio_cpp")
        self.catalog_revision = 1
        self.voice_revision = 1

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        del refresh
        return TTSProviderCatalog(
            provider_id="audio_cpp",
            revision=self.catalog_revision,
            health=ProviderHealth(state="available", fresh=True),
            models=(
                TTSModelInfo(
                    model_id="model-a",
                    display_name="Model A",
                    family="fake",
                    upstream_mode="tts",
                    formats=("wav",),
                    voices=(),
                    supports_speed=False,
                    omit_voice_uses_server_default=True,
                ),
            ),
        )

    async def observe_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        self.get_voices_requests.append((model_id, refresh))
        return TTSVoiceDiscoveryResult(
            provider_id="audio_cpp",
            model_id=model_id,
            catalog_revision=self.voice_revision,
            voices=("voice-a",),
            state="complete",
        )


def _service() -> tuple[TTSService, RevisionedCapabilityAdapter, list[int]]:
    adapter = RevisionedCapabilityAdapter()
    factory_calls: list[int] = []

    def factory(_config: Mapping[str, object]) -> RevisionedCapabilityAdapter:
        factory_calls.append(1)
        return adapter

    registry = TTSAdapterRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config={"revision": 1},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    return TTSService(registry), adapter, factory_calls


def test_latest_observation_read_is_empty_and_does_not_materialize_adapter() -> None:
    service, _adapter, factory_calls = _service()

    assert service.latest_native_capability_observation("audio_cpp") is None
    assert factory_calls == []


@pytest.mark.asyncio
async def test_explicit_catalog_and_voice_calls_publish_one_coherent_observation() -> (
    None
):
    service, _adapter, factory_calls = _service()

    await service.get_catalog("audio_cpp", refresh=True)
    await service.observe_voices("audio_cpp", "model-a", refresh=True)

    observation = service.latest_native_capability_observation("audio_cpp")
    assert type(observation) is TTSNativeCapabilityObservation
    assert observation.snapshot.configuration_revision == 1
    assert observation.snapshot.catalog is not None
    assert observation.snapshot.catalog.revision == 1
    assert observation.snapshot.voice_results["model-a"].voices == ("voice-a",)
    assert observation.observed_at.tzinfo is not None
    assert factory_calls == [1]
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_newer_catalog_discards_voice_results_from_the_prior_revision() -> None:
    service, adapter, _factory_calls = _service()
    await service.get_catalog("audio_cpp", refresh=True)
    await service.observe_voices("audio_cpp", "model-a", refresh=True)

    adapter.catalog_revision = 2
    adapter.voice_revision = 2
    await service.get_catalog("audio_cpp", refresh=True)

    observation = service.latest_native_capability_observation("audio_cpp")
    assert observation is not None
    assert observation.snapshot.catalog is not None
    assert observation.snapshot.catalog.revision == 2
    assert observation.snapshot.voice_results == {}
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_voice_result_for_an_older_catalog_cannot_overwrite_newer_state() -> None:
    service, adapter, _factory_calls = _service()
    adapter.catalog_revision = 2
    await service.get_catalog("audio_cpp", refresh=True)

    adapter.voice_revision = 1
    await service.observe_voices("audio_cpp", "model-a", refresh=True)

    observation = service.latest_native_capability_observation("audio_cpp")
    assert observation is not None
    assert observation.snapshot.catalog is not None
    assert observation.snapshot.catalog.revision == 2
    assert observation.snapshot.voice_results == {}
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_reconfiguration_retains_prior_observation_for_stale_projection() -> None:
    service, _adapter, _factory_calls = _service()
    await service.get_catalog("audio_cpp", refresh=True)

    await service.reconfigure_provider("audio_cpp", {"revision": 2})

    observation = service.latest_native_capability_observation("audio_cpp")
    assert observation is not None
    assert observation.snapshot.configuration_revision == 1
    assert service.configuration_revision("audio_cpp") == 2
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_older_catalog_request_cannot_overwrite_a_newer_request() -> None:
    service, adapter, _factory_calls = _service()
    old_started = asyncio.Event()
    release_old = asyncio.Event()
    call_count = 0
    original_catalog = adapter.get_catalog

    async def controlled_catalog(
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        nonlocal call_count
        del refresh
        call_count += 1
        if call_count == 1:
            old_started.set()
            await release_old.wait()
            adapter.catalog_revision = 1
        else:
            adapter.catalog_revision = 2
        return await original_catalog()

    adapter.get_catalog = controlled_catalog  # type: ignore[method-assign]
    old_request = asyncio.create_task(service.get_catalog("audio_cpp", refresh=True))
    await old_started.wait()
    await service.get_catalog("audio_cpp", refresh=True)
    release_old.set()
    await old_request

    observation = service.latest_native_capability_observation("audio_cpp")
    assert observation is not None
    assert observation.snapshot.catalog is not None
    assert observation.snapshot.catalog.revision == 2
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_older_voice_request_cannot_overwrite_a_newer_model_result() -> None:
    service, adapter, _factory_calls = _service()
    await service.get_catalog("audio_cpp", refresh=True)
    old_started = asyncio.Event()
    release_old = asyncio.Event()
    call_count = 0

    async def controlled_voices(
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        nonlocal call_count
        assert model_id == "model-a"
        del refresh
        call_count += 1
        if call_count == 1:
            old_started.set()
            await release_old.wait()
            voices = ("voice-old",)
        else:
            voices = ("voice-new",)
        return TTSVoiceDiscoveryResult(
            provider_id="audio_cpp",
            model_id="model-a",
            catalog_revision=1,
            voices=voices,
            state="complete",
        )

    adapter.observe_voices = controlled_voices  # type: ignore[method-assign]
    old_request = asyncio.create_task(
        service.observe_voices("audio_cpp", "model-a", refresh=True)
    )
    await old_started.wait()
    await service.observe_voices("audio_cpp", "model-a", refresh=True)
    release_old.set()
    await old_request

    observation = service.latest_native_capability_observation("audio_cpp")
    assert observation is not None
    assert observation.snapshot.voice_results["model-a"].voices == ("voice-new",)
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_voice_cache_stays_bounded_to_the_accepted_catalog_models() -> None:
    service, _adapter, _factory_calls = _service()
    await service.get_catalog("audio_cpp", refresh=True)

    async def missing_model(
        provider_id: str,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        del refresh
        return TTSVoiceDiscoveryResult(
            provider_id=provider_id,
            model_id=model_id,
            catalog_revision=1,
            voices=(),
            state="model_missing",
        )

    service.registry.observe_voices = missing_model  # type: ignore[method-assign]
    await service.observe_voices("audio_cpp", "not-in-catalog", refresh=True)

    observation = service.latest_native_capability_observation("audio_cpp")
    assert observation is not None
    assert observation.snapshot.voice_results == {}
    await service.close()
    await service.wait_closed()
