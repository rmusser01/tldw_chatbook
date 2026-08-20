from __future__ import annotations

import asyncio
import hashlib
import threading
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any
from uuid import UUID
from datetime import UTC, datetime

import pytest
from loguru import logger as loguru_logger

from Tests.TTS.adapter_fakes import FakeAdapter
from tldw_chatbook.TTS import TTS_Generation as generation_module
from tldw_chatbook.TTS import (
    STTSPlaygroundCloneSnapshot,
    STTSPlaygroundProfilePreview,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
    TTSAdapterLease,
    TTSAdapterRegistry,
)
from tldw_chatbook.TTS.adapter_types import (
    AudioCppCloneCapabilityAdmission,
    ProgressSink,
    ProviderHealth,
    TTSAudioResponse,
    TTSConfigurationRevisionError,
    TTSModelInfo,
    TTSNativeCapabilitySnapshot,
    TTSOperationError,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderReconfiguringError,
    TTSProviderSpec,
    TTSProviderUnavailableError,
    TTSRegistryClosedError,
    TTSRequest,
    TTSVoiceDiscoveryResult,
    _AdmittedAudioCppCloneRequest,
    _new_audio_cpp_clone_capability,
)
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.effective_settings import (
    TTSCharacterProfileSelection,
    TTSEffectiveResolutionError,
    TTSSelectionOverrides,
    TTSSelectionSource,
    TTSStudioDraftSelection,
)
from tldw_chatbook.TTS.legacy_bridge import (
    LegacyBackendHost,
    LegacyTTSAdapter,
    legacy_provider_specs,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_reference_materialization import (
    TTSCloneReferenceMaterializer,
)
from tldw_chatbook.TTS.profile_reference_types import (
    CanonicalTTSCloneReference,
    TTSCloneReference,
    TTSCloneRecipeRequirement,
    TTSCloneReferenceSummary,
)
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSPreferencesSnapshot,
    StudioTTSSelectionOverrides,
)
from tldw_chatbook.TTS.TTS_Generation import TTSService

_WAIT_SECONDS = 1.0


async def _wait_bounded(awaitable: Awaitable[Any]) -> Any:
    """Join one test synchronization point with a finite deadline."""
    protected = (
        asyncio.shield(awaitable)
        if isinstance(awaitable, asyncio.Future)
        else awaitable
    )
    return await asyncio.wait_for(
        protected,
        timeout=_WAIT_SECONDS,
    )


def _admission_api() -> tuple[type[Any], type[Any]]:
    try:
        from tldw_chatbook.TTS.request_admission import (
            TTSRequestAdmissionCoordinator,
            _WriterPreferredGate,
        )
    except ModuleNotFoundError:
        pytest.fail("the TTS request-admission module is not implemented")
    return _WriterPreferredGate, TTSRequestAdmissionCoordinator


async def _hold_gate(
    context: Any,
    entered: asyncio.Event,
    release: asyncio.Event,
    order: list[str] | None = None,
    label: str = "",
) -> None:
    async with context:
        if order is not None:
            order.append(label)
        entered.set()
        await release.wait()


@pytest.mark.asyncio
async def test_gate_allows_concurrent_readers() -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    release = asyncio.Event()
    entered = (asyncio.Event(), asyncio.Event())
    readers = tuple(
        asyncio.create_task(_hold_gate(gate.read(), event, release))
        for event in entered
    )

    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered)),
            timeout=_WAIT_SECONDS,
        )
    finally:
        release.set()
        await asyncio.gather(*readers, return_exceptions=True)


@pytest.mark.asyncio
async def test_waiting_writer_blocks_later_readers() -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    first_reader_entered = asyncio.Event()
    release_first_reader = asyncio.Event()
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()
    late_reader_entered = asyncio.Event()
    release_late_reader = asyncio.Event()
    order: list[str] = []

    first_reader = asyncio.create_task(
        _hold_gate(
            gate.read(),
            first_reader_entered,
            release_first_reader,
            order,
            "first-reader",
        )
    )
    await asyncio.wait_for(first_reader_entered.wait(), timeout=_WAIT_SECONDS)
    writer = asyncio.create_task(
        _hold_gate(
            gate.write(),
            writer_entered,
            release_writer,
            order,
            "writer",
        )
    )
    await asyncio.sleep(0)
    late_reader = asyncio.create_task(
        _hold_gate(
            gate.read(),
            late_reader_entered,
            release_late_reader,
            order,
            "late-reader",
        )
    )

    try:
        await asyncio.sleep(0)
        assert not late_reader_entered.is_set()
        release_first_reader.set()
        await asyncio.wait_for(writer_entered.wait(), timeout=_WAIT_SECONDS)
        assert not late_reader_entered.is_set()
        release_writer.set()
        await asyncio.wait_for(late_reader_entered.wait(), timeout=_WAIT_SECONDS)
        assert order == ["first-reader", "writer", "late-reader"]
    finally:
        release_first_reader.set()
        release_writer.set()
        release_late_reader.set()
        await asyncio.gather(
            first_reader,
            writer,
            late_reader,
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_cancelling_waiting_writer_unblocks_later_readers() -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    first_reader_entered = asyncio.Event()
    release_first_reader = asyncio.Event()
    writer_started = asyncio.Event()

    async def wait_for_write() -> None:
        writer_started.set()
        async with gate.write():
            pytest.fail("cancelled writer must not enter")

    first_reader = asyncio.create_task(
        _hold_gate(
            gate.read(),
            first_reader_entered,
            release_first_reader,
        )
    )
    await asyncio.wait_for(first_reader_entered.wait(), timeout=_WAIT_SECONDS)
    writer = asyncio.create_task(wait_for_write())
    await asyncio.wait_for(writer_started.wait(), timeout=_WAIT_SECONDS)
    await asyncio.sleep(0)
    writer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer

    later_reader_entered = asyncio.Event()
    release_later_reader = asyncio.Event()
    later_reader = asyncio.create_task(
        _hold_gate(
            gate.read(),
            later_reader_entered,
            release_later_reader,
        )
    )
    try:
        await asyncio.wait_for(later_reader_entered.wait(), timeout=_WAIT_SECONDS)
    finally:
        release_first_reader.set()
        release_later_reader.set()
        await asyncio.gather(first_reader, later_reader, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_waiting_reader_leaves_gate_usable() -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()
    writer = asyncio.create_task(
        _hold_gate(gate.write(), writer_entered, release_writer)
    )
    await asyncio.wait_for(writer_entered.wait(), timeout=_WAIT_SECONDS)

    async def wait_for_read() -> None:
        async with gate.read():
            pytest.fail("cancelled reader must not enter")

    reader = asyncio.create_task(wait_for_read())
    await asyncio.sleep(0)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader

    release_writer.set()
    await writer
    next_writer_entered = asyncio.Event()
    release_next_writer = asyncio.Event()
    next_writer = asyncio.create_task(
        _hold_gate(gate.write(), next_writer_entered, release_next_writer)
    )
    try:
        await asyncio.wait_for(next_writer_entered.wait(), timeout=_WAIT_SECONDS)
    finally:
        release_next_writer.set()
        await asyncio.gather(next_writer, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancelled_mode", "next_mode"), (("read", "write"), ("write", "read"))
)
async def test_cancelling_entered_context_releases_exactly_once(
    cancelled_mode: str,
    next_mode: str,
) -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    entered = asyncio.Event()
    never_release = asyncio.Event()
    cancelled_context = getattr(gate, cancelled_mode)()
    holder = asyncio.create_task(_hold_gate(cancelled_context, entered, never_release))
    await asyncio.wait_for(entered.wait(), timeout=_WAIT_SECONDS)

    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder

    next_entered = asyncio.Event()
    release_next = asyncio.Event()
    next_holder = asyncio.create_task(
        _hold_gate(getattr(gate, next_mode)(), next_entered, release_next)
    )
    try:
        await asyncio.wait_for(next_entered.wait(), timeout=_WAIT_SECONDS)
    finally:
        release_next.set()
        await asyncio.gather(next_holder, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancelled_mode", "next_mode"),
    (("read", "write"), ("write", "read")),
)
async def test_repeated_cancellation_cannot_interrupt_context_release(
    cancelled_mode: str,
    next_mode: str,
) -> None:
    gate_type, _ = _admission_api()
    gate = gate_type()
    entered = asyncio.Event()
    never_release = asyncio.Event()
    holder = asyncio.create_task(
        _hold_gate(
            getattr(gate, cancelled_mode)(),
            entered,
            never_release,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=_WAIT_SECONDS)

    async with gate._condition:
        holder.cancel()
        await asyncio.sleep(0)
        assert not holder.done()
        holder.cancel()
        await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await holder

    next_entered = asyncio.Event()
    release_next = asyncio.Event()
    next_holder = asyncio.create_task(
        _hold_gate(getattr(gate, next_mode)(), next_entered, release_next)
    )
    try:
        await asyncio.wait_for(next_entered.wait(), timeout=_WAIT_SECONDS)
    finally:
        release_next.set()
        next_holder.cancel()
        await asyncio.gather(next_holder, return_exceptions=True)


class _RecordingRegistry(TTSAdapterRegistry):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.expected_revisions: list[tuple[str, int | None]] = []

    async def acquire(
        self,
        provider_id: str,
        *,
        expected_revision: int | None = None,
    ) -> TTSAdapterLease:
        self.expected_revisions.append((provider_id, expected_revision))
        return await super().acquire(
            provider_id,
            expected_revision=expected_revision,
        )


class _CapturingAdapter(FakeAdapter):
    def __init__(
        self,
        provider_id: str,
        *,
        models: tuple[TTSModelInfo, ...] = (),
        synthesis_error: BaseException | None = None,
        generation: str = "",
    ) -> None:
        super().__init__(provider_id)
        self.models = models
        self.synthesis_error = synthesis_error
        self.generation = generation
        self.catalog_calls = 0
        self.requests: list[TTSRequest] = []

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        assert refresh is False
        self.catalog_calls += 1
        return TTSProviderCatalog(
            provider_id=self.provider_id,
            revision=self.catalog_calls,
            health=ProviderHealth(state="available", fresh=True),
            models=self.models,
        )

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        self.requests.append(request)
        if self.synthesis_error is not None:
            raise self.synthesis_error
        return await super().synthesize(request, progress_sink)


class _CloneCapturingAdapter(_CapturingAdapter):
    def __init__(self) -> None:
        super().__init__("audio_cpp", models=(_model("clone-model"),))
        self._identity = object()
        self._capability: AudioCppCloneCapabilityAdmission | None = None
        self.clone_requests: list[_AdmittedAudioCppCloneRequest] = []
        self.events: list[str] = []
        self.capability_recipe_id = "pocket_tts"
        self.capability_recipe_revision = 1
        self.capability_process_generation = 7

    def preflight_clone_source(self) -> None:
        self.events.append("preflight")

    def preflight_clone_dependency(
        self,
        requirement: TTSCloneRecipeRequirement,
    ) -> None:
        self.events.append("dependency_preflight")

    def preflight_clone_request_dependency(
        self,
        request: TTSRequest,
        requirement: TTSCloneRecipeRequirement,
    ) -> None:
        self.preflight_clone_dependency(requirement)
        self.events.append("request_dependency_preflight")
        if request.model_id != requirement.model_id:
            raise RuntimeError("dependency drift")

    def admit_clone_capability(
        self, request: TTSRequest
    ) -> AudioCppCloneCapabilityAdmission:
        self.events.append("capability")
        capability = _new_audio_cpp_clone_capability(
            adapter_identity=self._identity,
            capability_token=object(),
            model_id=request.model_id,
            recipe_id=self.capability_recipe_id,
            recipe_revision=self.capability_recipe_revision,
            process_generation=self.capability_process_generation,
            request=request,
        )
        self._capability = capability
        return capability

    def release_clone_capability(
        self, capability: AudioCppCloneCapabilityAdmission
    ) -> None:
        if self._capability is capability:
            self.events.append("capability_released")
            self._capability = None

    async def synthesize_clone(
        self,
        request: _AdmittedAudioCppCloneRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        self.events.append("clone_synthesize")
        self.clone_requests.append(request)
        response = await super().synthesize(request.request, progress_sink)

        async def observe_adapter_cleanup() -> None:
            assert request.materialization.voice_ref.exists()
            self.events.append("adapter_cleanup")

        response.add_cleanup(observe_adapter_cleanup)
        return response


class _BlockingCloneCapturingAdapter(_CloneCapturingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.ensure_started = asyncio.Event()
        self.allow_ensure = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.allow_cleanup = asyncio.Event()

    async def ensure_ready(self) -> None:
        await super().ensure_ready()
        self.ensure_started.set()
        await self.allow_ensure.wait()

    async def synthesize_clone(
        self,
        request: _AdmittedAudioCppCloneRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        response = await super().synthesize_clone(request, progress_sink)

        async def block_adapter_cleanup() -> None:
            self.cleanup_started.set()
            await self.allow_cleanup.wait()

        response.add_cleanup(block_adapter_cleanup)
        return response


class _RejectedCloneSourceAdapter(_CloneCapturingAdapter):
    def preflight_clone_source(self) -> None:
        self.events.append("preflight_rejected")
        raise RuntimeError("rejected clone source")


def _clone_reference(
    requirement: TTSCloneRecipeRequirement | None = None,
) -> TTSCloneReference:
    wav = b"private-clone-reference"
    now = datetime(2026, 8, 10, tzinfo=UTC)
    return TTSCloneReference(
        summary=TTSCloneReferenceSummary(
            reference_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            byte_length=len(wav),
            duration_ms=250,
            sample_rate_hz=24_000,
            channels=1,
            sample_encoding="pcm_s16le",
            created_at=now,
            updated_at=now,
            recipe_requirement=requirement,
        ),
        reference_text="Private reference transcript",
        sha256=hashlib.sha256(wav).hexdigest(),
        wav_bytes=wav,
        recipe_requirement=requirement,
    )


def _canonical_clone_reference() -> CanonicalTTSCloneReference:
    stored = _clone_reference()
    return CanonicalTTSCloneReference(
        wav_bytes=stored.wav_bytes,
        reference_text=stored.reference_text,
        sha256=stored.sha256,
        byte_length=stored.summary.byte_length,
        duration_ms=stored.summary.duration_ms,
        sample_rate_hz=stored.summary.sample_rate_hz,
        channels=stored.summary.channels,
        sample_encoding=stored.summary.sample_encoding,
    )


def _model(model_id: str) -> TTSModelInfo:
    return TTSModelInfo(
        model_id=model_id,
        display_name=model_id,
        family="tts",
        upstream_mode="offline",
        formats=("wav",),
        voices=(),
        supports_speed=False,
        omit_voice_uses_server_default=True,
    )


def _snapshot(
    *,
    provider_id: str = "audio_cpp",
    model_mode: str = "exact",
    model_id: str | None = "Model/Case-Sensitive",
    voice_mode: str = "server_default",
    voice_id: str | None = None,
    response_format: str = "wav",
    speed: float = 1.0,
) -> TTSPreferencesSnapshot:
    return TTSPreferencesSnapshot(
        provider_id=provider_id,
        model_mode=model_mode,  # type: ignore[arg-type]
        model_id=model_id,
        voice_mode=voice_mode,  # type: ignore[arg-type]
        voice_id=voice_id,
        response_format=response_format,
        speed=speed,
    )


def _accepted_native_capability_reader(
    registry: TTSAdapterRegistry,
) -> Callable[[str, str, str | None], Awaitable[TTSNativeCapabilitySnapshot]]:
    """Provide explicit authoritative evidence for admission unit tests."""

    async def read(
        provider_id: str,
        model_id: str,
        voice_id: str | None,
    ) -> TTSNativeCapabilitySnapshot:
        catalog_revision = 19
        catalog = TTSProviderCatalog(
            provider_id=provider_id,
            revision=catalog_revision,
            health=ProviderHealth(state="available", fresh=True),
            models=(_model(model_id),),
        )
        voice_results = (
            {}
            if voice_id is None
            else {
                model_id: TTSVoiceDiscoveryResult(
                    provider_id=provider_id,
                    model_id=model_id,
                    catalog_revision=catalog_revision,
                    voices=(voice_id,),
                    state="complete",
                )
            }
        )
        return TTSNativeCapabilitySnapshot(
            provider_id=provider_id,
            configuration_revision=registry.configuration_revision(provider_id),
            state="complete",
            catalog=catalog,
            voice_results=voice_results,
        )

    return read


def _test_service(
    registry: TTSAdapterRegistry,
    **kwargs: Any,
) -> TTSService:
    """Construct a service with explicit native evidence for unit tests."""
    kwargs.setdefault(
        "native_capability_reader",
        _accepted_native_capability_reader(registry),
    )
    return TTSService(registry, **kwargs)


def _native_service(
    adapter: _CapturingAdapter,
    snapshot: TTSPreferencesSnapshot,
    studio_preferences_loader: Callable[[], StudioTTSPreferencesSnapshot] | None = None,
) -> tuple[TTSService, _RecordingRegistry]:
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id=adapter.provider_id,
                    display_name=adapter.provider_id,
                    native=True,
                ),
                factory=lambda _config: adapter,
                initial_config={"generation": adapter.generation or "initial"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    return (
        TTSService(
            registry,
            preferences_snapshot=snapshot,
            studio_preferences_loader=studio_preferences_loader,
            native_capability_reader=_accepted_native_capability_reader(registry),
        ),
        registry,
    )


class _ManagedPromotionSupervisor:
    def __init__(self) -> None:
        self.state = "running"
        self.draining_started = asyncio.Event()

    def admission_snapshot(self) -> Any:
        return generation_module.AudioCppProcessAdmissionSnapshot(
            lifecycle_epoch=1,
            process_generation=1,
            state=self.state,
            stage_application_eligible=self.state == "stopped",
        )

    async def begin_draining(self) -> None:
        self.state = "draining"
        self.draining_started.set()

    async def stop(self) -> None:
        self.state = "stopped"

    close = stop

    async def wait_closed(self) -> None:
        return None


def _managed_config(timeout: float) -> dict[str, Any]:
    return {"mode": "managed", "connect_timeout_seconds": timeout}


def _managed_promotion_service():
    adapters: list[_CapturingAdapter] = []

    def audio_factory(config: Mapping[str, Any]) -> _CapturingAdapter:
        adapter = _CapturingAdapter(
            "audio_cpp",
            generation=str(config["connect_timeout_seconds"]),
        )
        adapters.append(adapter)
        return adapter

    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                TTSProviderDescriptor("audio_cpp", "audio.cpp", True),
                audio_factory,
                _managed_config(5.0),
                True,
            ),
            TTSProviderSpec(
                TTSProviderDescriptor("other", "Other", True),
                lambda _config: _CapturingAdapter("other"),
                {"generation": "one"},
            ),
        ),
        aliases={},
    )
    supervisor = _ManagedPromotionSupervisor()
    service = _test_service(
        registry,
        preferences_snapshot=_snapshot(model_id="Model/A"),
        audio_cpp_supervisor=supervisor,
    )
    service._publish_native_catalog = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    return service, adapters, supervisor


async def _publish_settings(
    service: TTSService,
    preferences: TTSPreferencesSnapshot,
    provider_configs: Mapping[str, Mapping[str, Any]],
) -> Any:
    ticket = service.begin_preferences_publication(
        preferences,
        provider_configs,
        lambda: generation_module.TTSSettingsPersistenceOutcome(True, True, None),
        foreground_timeout_seconds=0,
    )
    return await _wait_bounded(ticket.completion)


@pytest.mark.asyncio
async def test_invalid_initial_provider_is_unconfigured_and_publication_recovers() -> (
    None
):
    private_provider = "PRIVATE_INITIAL_PROVIDER"
    adapter = _CapturingAdapter("audio_cpp")
    service, registry = _native_service(
        adapter,
        _snapshot(
            provider_id=private_provider,
            model_id="Private/Initial/Model",
        ),
    )
    recovered_snapshot = _snapshot(model_id="Model/Recovered")
    response: TTSAudioResponse | None = None
    try:
        with pytest.raises(TTSProviderUnavailableError) as captured:
            await service.synthesize_default(text="Blocked initial request")

        assert str(captured.value) == "TTS default provider is not configured"
        assert private_provider not in repr(captured.value)
        assert service.preferences_snapshot() is None
        assert service._operation_limit._value == 4
        assert registry._total_leases() == 0
        assert adapter.requests == []

        publication = service.begin_preferences_publication(
            recovered_snapshot,
            {},
            lambda: generation_module.TTSSettingsPersistenceOutcome(
                True,
                True,
                None,
            ),
        )
        result = await _wait_bounded(publication.completion)

        assert result.published is True
        assert service.preferences_snapshot() == recovered_snapshot
        assert service.preferences_generation() == publication.generation

        response = await service.synthesize_default(text="Recovered request")
        assert [request.provider_id for request in adapter.requests] == ["audio_cpp"]
        assert [request.model_id for request in adapter.requests] == ["Model/Recovered"]
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_invalid_global_provider_cannot_be_hidden_by_sparse_override() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    service, registry = _native_service(
        adapter,
        _snapshot(provider_id="PRIVATE_INITIAL_PROVIDER"),
    )

    try:
        with pytest.raises(TTSProviderUnavailableError) as caught:
            await service.synthesize_default(
                text="Do not select another provider.",
                voice_override="voice-only",
            )

        assert str(caught.value) == "TTS default provider is not configured"
        assert adapter.requests == []
        assert registry._total_leases() == 0
        assert service._operation_limit._value == 4
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_explicit_provider_without_global_uses_provider_fallback_axes() -> None:
    adapter = _CapturingAdapter("openai")
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="openai",
                    display_name="OpenAI",
                    native=False,
                ),
                factory=lambda _config: adapter,
                initial_config={},
            ),
        ),
        aliases={},
    )
    service = _test_service(
        registry,
        preferences_snapshot=_snapshot(provider_id="PRIVATE_INITIAL_PROVIDER"),
    )
    response: TTSAudioResponse | None = None

    try:
        response, effective = await service.synthesize_effective(
            text="Use declared fallbacks only.",
            explicit=TTSSelectionOverrides(provider_id="openai"),
        )

        assert effective.model_id == "tts-1"
        assert effective.voice_id == "alloy"
        assert effective.sources["provider_id"] is TTSSelectionSource.EXPLICIT
        assert effective.sources["model_id"] is TTSSelectionSource.PROVIDER_FALLBACK
        assert effective.sources["voice_id"] is TTSSelectionSource.PROVIDER_FALLBACK
        assert TTSSelectionSource.GLOBAL not in effective.sources.values()
        assert adapter.requests[0].model_id == "tts-1"
        assert adapter.requests[0].voice == "alloy"
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_openai_exact_custom_model_default_is_admitted_with_passthrough() -> None:
    """A custom OpenAI model id must survive admission untouched.

    OpenAI-compatible servers (TASK-2260) define their own model and voice
    names; the Console default path resolves them as exact global values, so
    admission must route on the provider and pass both through rather than
    re-imposing the official-catalog model list (TASK-15420).
    """
    adapter = _CapturingAdapter("openai")
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="openai",
                    display_name="OpenAI",
                    native=False,
                ),
                factory=lambda _config: adapter,
                initial_config={},
            ),
        ),
        aliases={},
    )
    service = _test_service(
        registry,
        preferences_snapshot=_snapshot(
            provider_id="openai",
            model_id="pocket-tts-model",
            voice_mode="exact",
            voice_id="pocket-voice",
        ),
    )
    response: TTSAudioResponse | None = None

    try:
        response = await service.synthesize_default(
            text="Speak through the custom endpoint.",
        )

        assert response.provider_id == "openai"
        assert response.model_id == "pocket-tts-model"
        assert adapter.requests[0].model_id == "pocket-tts-model"
        assert adapter.requests[0].voice == "pocket-voice"
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_audio_cpp_exact_default_is_admitted_without_rewriting_values() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    snapshot = _snapshot(
        model_id="Model/Byte-For-Byte",
        voice_mode="exact",
        voice_id="Voice/Byte-For-Byte",
    )
    service, registry = _native_service(adapter, snapshot)

    response = await service.synthesize_default(text="Character response")
    try:
        assert adapter.requests == [
            TTSRequest(
                provider_id="audio_cpp",
                model_id="Model/Byte-For-Byte",
                text="Character response",
                voice="Voice/Byte-For-Byte",
                response_format="wav",
                speed=1.0,
                options={},
            )
        ]
        assert registry.expected_revisions == [("audio_cpp", 1)]
    finally:
        await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_audio_cpp_first_available_is_resolved_once_and_never_falls_back() -> (
    None
):
    failure = RuntimeError("synthetic generation failure")
    adapter = _CapturingAdapter(
        "audio_cpp",
        models=(_model("First/Model"), _model("Second/Model")),
        synthesis_error=failure,
    )
    service, registry = _native_service(
        adapter,
        _snapshot(model_mode="first_available", model_id=None),
    )

    try:
        with pytest.raises(RuntimeError) as raised:
            await service.synthesize_default(text="Character response")
        assert raised.value is failure
        assert adapter.catalog_calls == 1
        assert [request.model_id for request in adapter.requests] == ["First/Model"]
        assert registry.expected_revisions == [
            ("audio_cpp", None),
            ("audio_cpp", 1),
        ]
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_audio_cpp_server_default_omits_voice_and_uses_locked_options() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    service, _registry = _native_service(
        adapter,
        _snapshot(voice_mode="server_default", voice_id=None),
    )

    response = await service.synthesize_default(
        text="Character response",
        voice_override=None,
    )
    try:
        request = adapter.requests[0]
        assert request.voice is None
        assert request.response_format == "wav"
        assert request.speed == 1.0
        assert request.options == {}
    finally:
        await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_empty_explicit_voice_blocks_instead_of_falling_through() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    service, registry = _native_service(
        adapter,
        _snapshot(voice_mode="exact", voice_id="saved-voice"),
    )

    try:
        with pytest.raises(TTSEffectiveResolutionError) as caught:
            await service.synthesize_default(
                text="Do not use the saved voice.",
                voice_override="",
            )

        assert caught.value.code == "invalid_selection"
        assert caught.value.axis == "voice_id"
        assert adapter.requests == []
        assert registry._total_leases() == 0
        assert service._operation_limit._value == 4
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_effective_admission_retains_character_profile_sources_and_revisions() -> (
    None
):
    adapter = _CapturingAdapter("audio_cpp")
    service, registry = _native_service(adapter, _snapshot())
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="Character/Model",
            voice_mode="exact",
            voice_id="Character/Voice",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=13,
        profile_revision=8,
        profile_id=UUID("11111111-1111-4111-8111-111111111111"),
    )
    response: TTSAudioResponse | None = None

    try:
        response, effective = await service.synthesize_effective(
            text="Character-authored response.",
            character_profile=character,
        )

        assert effective.sources["provider_id"] is TTSSelectionSource.CHARACTER_PROFILE
        assert effective.sources["model_id"] is TTSSelectionSource.CHARACTER_PROFILE
        assert effective.sources["voice_id"] is TTSSelectionSource.CHARACTER_PROFILE
        assert effective.revisions.character_repository == 13
        assert effective.revisions.character_profile == 8
        assert not hasattr(effective, "text")
        assert adapter.requests == [
            TTSRequest(
                provider_id="audio_cpp",
                model_id="Character/Model",
                text="Character-authored response.",
                voice="Character/Voice",
                response_format="wav",
                speed=1.0,
                options={},
            )
        ]
        assert registry.expected_revisions == [("audio_cpp", 1)]
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_closed_service_rejects_profile_preview_before_reference_read(
    tmp_path: Any,
) -> None:
    adapter = _CloneCapturingAdapter()
    saved = StudioTTSPreferencesSnapshot(revision=2)
    service, _registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
        studio_preferences_loader=lambda: saved,
    )
    service._clone_materializer = TTSCloneReferenceMaterializer(
        tmp_path / "clone-runtime"
    )
    await service.close()
    await service.wait_closed()
    resolver_calls = 0

    async def resolver(
        _profile_id: UUID,
        _repository_generation: int,
        _profile_revision: int,
    ) -> TTSCloneReference:
        nonlocal resolver_calls
        resolver_calls += 1
        return _clone_reference()

    saved = StudioTTSPreferencesSnapshot(revision=2)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=saved.revision,
        preview=True,
    )

    with pytest.raises(TTSRegistryClosedError):
        await service.synthesize_effective(
            text="Profile preview.",
            studio_draft=draft,
            studio_preferences=saved,
            profile_preview=STTSPlaygroundProfilePreview(
                profile_id=UUID("99999999-9999-4999-8999-999999999999"),
                repository_generation=1,
                profile_revision=1,
            ),
            profile_reference_resolver=resolver,
        )

    assert resolver_calls == 0


@pytest.mark.asyncio
async def test_profile_preview_reference_stays_private_below_service_admission(
    tmp_path: Any,
) -> None:
    adapter = _CloneCapturingAdapter()
    saved = StudioTTSPreferencesSnapshot(revision=2)
    service, registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
        studio_preferences_loader=lambda: saved,
    )
    service._clone_materializer = TTSCloneReferenceMaterializer(
        tmp_path / "clone-runtime"
    )
    preview = STTSPlaygroundProfilePreview(
        profile_id=UUID("88888888-8888-4888-8888-888888888888"),
        repository_generation=9,
        profile_revision=6,
    )
    resolved_reference = _clone_reference()
    resolver_calls: list[tuple[UUID, int, int]] = []

    async def resolver(
        profile_id: UUID,
        repository_generation: int,
        profile_revision: int,
    ) -> TTSCloneReference:
        resolver_calls.append((profile_id, repository_generation, profile_revision))
        return resolved_reference

    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=saved.revision,
        preview=True,
    )
    response: TTSAudioResponse | None = None
    try:
        response, _selection = await service.synthesize_effective(
            text="Profile preview.",
            studio_draft=draft,
            studio_preferences=saved,
            profile_preview=preview,
            profile_reference_resolver=resolver,
        )

        assert resolver_calls == [(preview.profile_id, 9, 6)]
        assert len(adapter.clone_requests) == 1
        materialization = adapter.clone_requests[0].materialization
        assert materialization.reference_text == resolved_reference.reference_text
        assert materialization.voice_ref.read_bytes() == resolved_reference.wav_bytes
        assert registry._total_leases() == 1

        await response.aclose()
        response = None
        assert not materialization.voice_ref.exists()
        assert registry._total_leases() == 0
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_stale_profile_preview_resolves_before_registry_or_provider_work(
    tmp_path: Any,
) -> None:
    adapter = _CloneCapturingAdapter()
    saved = StudioTTSPreferencesSnapshot(revision=2)
    service, registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
        studio_preferences_loader=lambda: saved,
    )
    service._clone_materializer = TTSCloneReferenceMaterializer(
        tmp_path / "clone-runtime"
    )
    preview = STTSPlaygroundProfilePreview(
        profile_id=UUID("66666666-6666-4666-8666-666666666666"),
        repository_generation=7,
        profile_revision=4,
    )
    resolver_calls: list[tuple[UUID, int, int]] = []

    async def stale_resolver(
        profile_id: UUID,
        repository_generation: int,
        profile_revision: int,
    ) -> TTSCloneReference:
        resolver_calls.append((profile_id, repository_generation, profile_revision))
        raise RuntimeError("PRIVATE_STALE_PROFILE_DETAIL")

    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=saved.revision,
        preview=True,
    )
    try:
        with pytest.raises(TTSEffectiveResolutionError) as caught:
            await service.synthesize_effective(
                text="Profile preview.",
                studio_draft=draft,
                studio_preferences=saved,
                profile_preview=preview,
                profile_reference_resolver=stale_resolver,
            )

        assert caught.value.code == "revision_incoherent"
        assert caught.value.axis == "profile_reference"
        assert resolver_calls == [
            (preview.profile_id, 7, 4),
        ]
        assert registry.expected_revisions == []
        assert registry._total_leases() == 0
        assert adapter.ensure_ready_calls == 0
        assert adapter.catalog_calls == 0
        assert adapter.events == []
        assert not (tmp_path / "clone-runtime").exists()
        assert "PRIVATE_STALE_PROFILE_DETAIL" not in repr(caught.value)
        assert service._operation_limit._value == 4
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_recipe_mismatch_blocks_before_provider_lease_or_adapter_work(
    tmp_path: Any,
) -> None:
    adapter = _CloneCapturingAdapter()
    saved = StudioTTSPreferencesSnapshot(revision=2)
    service, registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
        studio_preferences_loader=lambda: saved,
    )
    service._clone_materializer = TTSCloneReferenceMaterializer(
        tmp_path / "clone-runtime"
    )
    requirement = TTSCloneRecipeRequirement(
        recipe_id="pocket_tts",
        recipe_revision=1,
        model_id="clone-model",
    )
    preview = STTSPlaygroundProfilePreview(
        profile_id=UUID("66666666-6666-4666-8666-666666666666"),
        repository_generation=7,
        profile_revision=4,
    )

    async def resolver(
        _profile_id: UUID,
        _repository_generation: int,
        _profile_revision: int,
    ) -> TTSCloneReference:
        return _clone_reference(requirement)

    dependency_calls: list[TTSCloneRecipeRequirement] = []

    async def mismatch(current: TTSCloneRecipeRequirement):
        dependency_calls.append(current)
        return generation_module.AudioCppGuidedDependencySnapshot(
            state="mismatch",
            provider_configuration_revision=1,
            saved_generation=1,
            applied_generation=1,
            pending_configuration=False,
            saved_requirement=None,
            applied_requirement=None,
        )

    service.audio_cpp_guided_dependency_snapshot = mismatch  # type: ignore[method-assign]
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=saved.revision,
        preview=True,
    )
    try:
        with pytest.raises(TTSOperationError) as caught:
            await service.synthesize_effective(
                text="Profile preview.",
                studio_draft=draft,
                studio_preferences=saved,
                profile_preview=preview,
                profile_reference_resolver=resolver,
            )

        assert caught.value.code == "dependency_changed"
        assert caught.value.recovery_action == "open_settings"
        assert dependency_calls == [requirement]
        assert registry.expected_revisions == []
        assert registry._total_leases() == 0
        assert adapter.ensure_ready_calls == 0
        assert adapter.catalog_calls == 0
        assert adapter.events == []
        assert not (tmp_path / "clone-runtime").exists()
        assert service._operation_limit._value == 4
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_clone_dependency_collaborator_failure_is_bounded() -> None:
    canary = "CANARY-private-provider-origin-generated-config"
    logs: list[str] = []

    class PrivateProviderOriginError(RuntimeError):
        pass

    adapter = _CloneCapturingAdapter()
    saved = StudioTTSPreferencesSnapshot(revision=2)
    service, registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
        studio_preferences_loader=lambda: saved,
    )
    requirement = TTSCloneRecipeRequirement(
        recipe_id="pocket_tts",
        recipe_revision=1,
        model_id="clone-model",
    )

    async def exact(current: TTSCloneRecipeRequirement):
        return generation_module.AudioCppGuidedDependencySnapshot(
            state="exact",
            provider_configuration_revision=1,
            saved_generation=1,
            applied_generation=1,
            pending_configuration=False,
            saved_requirement=current,
            applied_requirement=current,
        )

    def fail_dependency(*_args: object, **_kwargs: object) -> None:
        private_traceback_local = canary
        raise PrivateProviderOriginError(private_traceback_local)

    adapter.preflight_clone_request_dependency = fail_dependency  # type: ignore[method-assign]
    service.audio_cpp_guided_dependency_snapshot = exact  # type: ignore[method-assign]
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=1,
        profile_revision=1,
        profile_id=UUID("55555555-5555-4555-8555-555555555555"),
        reference=_clone_reference(requirement),
    )
    sink = loguru_logger.add(lambda message: logs.append(str(message)), level="DEBUG")
    try:
        with pytest.raises(TTSOperationError) as caught:
            await service.synthesize_effective(
                text="private submitted text",
                character_profile=character,
            )

        assert caught.value.code == "dependency_changed"
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert canary not in str(caught.value)
        assert canary not in repr(caught.value)
        rendered_exception = "".join(traceback.format_exception(caught.value))
        assert canary not in rendered_exception
        assert PrivateProviderOriginError.__name__ not in rendered_exception

        pending: list[BaseException] = [caught.value]
        seen: set[int] = set()
        product_traceback_locals: list[str] = []
        while pending:
            error = pending.pop()
            if id(error) in seen:
                continue
            seen.add(id(error))
            pending.extend(
                linked
                for linked in (error.__cause__, error.__context__)
                if linked is not None
            )
            current = error.__traceback__
            while current is not None:
                if current.tb_frame.f_globals.get("__name__") == generation_module.__name__:
                    product_traceback_locals.extend(
                        repr(value) for value in current.tb_frame.f_locals.values()
                    )
                current = current.tb_next

        rendered_surfaces = "\n".join(
            (
                *logs,
                *product_traceback_locals,
                repr(adapter.events),
                repr(adapter.clone_requests),
            )
        )
        assert canary not in rendered_surfaces
        assert PrivateProviderOriginError.__name__ not in rendered_surfaces
        assert registry._total_leases() == 0
        assert adapter.clone_requests == []
    finally:
        loguru_logger.remove(sink)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_kind",
    ("boolean_generation", "hollow_snapshot", "hollow_requirement"),
)
async def test_forged_pure_dependency_evidence_is_rejected_before_provider_work(
    tmp_path: Any,
    invalid_kind: str,
) -> None:
    adapter = _CloneCapturingAdapter()
    saved = StudioTTSPreferencesSnapshot(revision=2)
    service, registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
        studio_preferences_loader=lambda: saved,
    )
    service._clone_materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    requirement = TTSCloneRecipeRequirement(
        recipe_id="pocket_tts",
        recipe_revision=1,
        model_id="clone-model",
    )

    async def forged(current: TTSCloneRecipeRequirement):
        if invalid_kind == "hollow_snapshot":
            return object.__new__(generation_module.AudioCppGuidedDependencySnapshot)
        if invalid_kind == "hollow_requirement":
            hollow = object.__new__(TTSCloneRecipeRequirement)
            return generation_module.AudioCppGuidedDependencySnapshot(
                state="exact",
                provider_configuration_revision=1,
                saved_generation=1,
                applied_generation=1,
                pending_configuration=False,
                saved_requirement=hollow,
                applied_requirement=hollow,
            )
        return generation_module.AudioCppGuidedDependencySnapshot(
            state="exact",
            provider_configuration_revision=True,  # type: ignore[arg-type]
            saved_generation=1,
            applied_generation=1,
            pending_configuration=False,
            saved_requirement=current,
            applied_requirement=current,
        )

    service.audio_cpp_guided_dependency_snapshot = forged  # type: ignore[method-assign]
    preview = STTSPlaygroundProfilePreview(
        profile_id=UUID("99999999-9999-4999-8999-999999999998"),
        repository_generation=7,
        profile_revision=4,
    )

    async def resolver(*_args: object) -> TTSCloneReference:
        return _clone_reference(requirement)

    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=saved.revision,
        preview=True,
    )
    try:
        with pytest.raises(TTSOperationError) as caught:
            await service.synthesize_effective(
                text="Profile preview.",
                studio_draft=draft,
                studio_preferences=saved,
                profile_preview=preview,
                profile_reference_resolver=resolver,
            )

        assert caught.value.code == "dependency_changed"
        assert registry.expected_revisions == []
        assert adapter.events == []
        assert adapter.ensure_ready_calls == 0
        assert not (tmp_path / "runtime").exists()
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_post_ready_recipe_drift_blocks_before_private_materialization(
    tmp_path: Any,
) -> None:
    adapter = _CloneCapturingAdapter()
    adapter.capability_recipe_revision = 2
    saved = StudioTTSPreferencesSnapshot(revision=2)
    service, registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
        studio_preferences_loader=lambda: saved,
    )
    runtime_root = tmp_path / "clone-runtime"
    service._clone_materializer = TTSCloneReferenceMaterializer(runtime_root)
    requirement = TTSCloneRecipeRequirement(
        recipe_id="pocket_tts",
        recipe_revision=1,
        model_id="clone-model",
    )

    async def exact(current: TTSCloneRecipeRequirement):
        return generation_module.AudioCppGuidedDependencySnapshot(
            state="exact",
            provider_configuration_revision=1,
            saved_generation=1,
            applied_generation=1,
            pending_configuration=False,
            saved_requirement=current,
            applied_requirement=current,
        )

    async def resolver(*_args: object) -> TTSCloneReference:
        return _clone_reference(requirement)

    service.audio_cpp_guided_dependency_snapshot = exact  # type: ignore[method-assign]
    preview = STTSPlaygroundProfilePreview(
        profile_id=UUID("77777777-7777-4777-8777-777777777777"),
        repository_generation=7,
        profile_revision=4,
    )

    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=saved.revision,
        preview=True,
    )
    try:
        with pytest.raises(TTSOperationError) as caught:
            await service.synthesize_effective(
                text="Profile preview.",
                studio_draft=draft,
                studio_preferences=saved,
                profile_preview=preview,
                profile_reference_resolver=resolver,
            )

        assert caught.value.code == "dependency_changed"
        assert adapter.ensure_ready_calls == 1
        assert adapter.events == [
            "dependency_preflight",
            "preflight",
            "dependency_preflight",
            "request_dependency_preflight",
            "capability",
            "capability_released",
        ]
        assert adapter.clone_requests == []
        assert not runtime_root.exists()
        assert registry._total_leases() == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_exact_pure_dependency_still_requires_adapter_config_preflight(
    tmp_path: Any,
) -> None:
    class _DriftedAdapter(_CloneCapturingAdapter):
        def preflight_clone_dependency(
            self,
            requirement: TTSCloneRecipeRequirement,
        ) -> None:
            self.events.append("dependency_preflight_rejected")
            raise TTSOperationError(
                code="dependency_changed",
                message="The clone voice dependency changed",
                retryable=False,
                operation_id="bounded",
                recovery_action="open_settings",
            )

    adapter = _DriftedAdapter()
    saved = StudioTTSPreferencesSnapshot(revision=2)
    service, registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
        studio_preferences_loader=lambda: saved,
    )
    runtime_root = tmp_path / "clone-runtime"
    service._clone_materializer = TTSCloneReferenceMaterializer(runtime_root)
    requirement = TTSCloneRecipeRequirement(
        recipe_id="pocket_tts",
        recipe_revision=1,
        model_id="clone-model",
    )

    async def exact(current: TTSCloneRecipeRequirement):
        return generation_module.AudioCppGuidedDependencySnapshot(
            state="exact",
            provider_configuration_revision=1,
            saved_generation=1,
            applied_generation=1,
            pending_configuration=False,
            saved_requirement=current,
            applied_requirement=current,
        )

    service.audio_cpp_guided_dependency_snapshot = exact  # type: ignore[method-assign]
    preview = STTSPlaygroundProfilePreview(
        profile_id=UUID("88888888-8888-4888-8888-888888888888"),
        repository_generation=7,
        profile_revision=4,
    )

    async def resolver(*_args: object) -> TTSCloneReference:
        return _clone_reference(requirement)

    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=saved.revision,
        preview=True,
    )
    try:
        with pytest.raises(TTSOperationError) as caught:
            await service.synthesize_effective(
                text="Profile preview.",
                studio_draft=draft,
                studio_preferences=saved,
                profile_preview=preview,
                profile_reference_resolver=resolver,
            )

        assert caught.value.code == "dependency_changed"
        assert adapter.events == ["dependency_preflight_rejected"]
        assert adapter.ensure_ready_calls == 0
        assert adapter.catalog_calls == 0
        assert registry._total_leases() == 0
        assert not runtime_root.exists()
    finally:
        await service.close()
        await service.wait_closed()


class _PolicyCloneAdapter(_CloneCapturingAdapter):
    def __init__(self, *, voice_required: bool) -> None:
        super().__init__()
        self.voice_required = voice_required

    def preflight_clone_request_dependency(
        self,
        request: TTSRequest,
        requirement: TTSCloneRecipeRequirement,
    ) -> None:
        super().preflight_clone_request_dependency(request, requirement)
        if (request.voice is not None) is self.voice_required:
            return
        raise TTSOperationError(
            code="dependency_changed",
            message="The clone voice dependency changed",
            retryable=False,
            operation_id="bounded",
            recovery_action="open_settings",
        )


@pytest.mark.asyncio
async def test_clone_reference_only_policy_uses_explicit_voice_override(
    tmp_path: Any,
) -> None:
    adapter = _PolicyCloneAdapter(voice_required=False)
    service, registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
    )
    runtime_root = tmp_path / "clone-runtime"
    service._clone_materializer = TTSCloneReferenceMaterializer(runtime_root)
    requirement = TTSCloneRecipeRequirement(
        recipe_id="pocket_tts",
        recipe_revision=1,
        model_id="clone-model",
    )
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=7,
        profile_revision=4,
        profile_id=UUID("77777777-7777-4777-8777-777777777777"),
        reference=_clone_reference(requirement),
    )

    async def exact(current: TTSCloneRecipeRequirement):
        return generation_module.AudioCppGuidedDependencySnapshot(
            state="exact",
            provider_configuration_revision=1,
            saved_generation=1,
            applied_generation=1,
            pending_configuration=False,
            saved_requirement=current,
            applied_requirement=current,
        )

    service.audio_cpp_guided_dependency_snapshot = exact  # type: ignore[method-assign]
    try:
        with pytest.raises(TTSOperationError) as caught:
            await service.synthesize_effective(
                text="Character response.",
                explicit=TTSSelectionOverrides(
                    voice_mode="exact",
                    voice_id="explicit-voice",
                ),
                character_profile=character,
            )

        assert caught.value.code == "dependency_changed"
        assert adapter.ensure_ready_calls == 0
        assert adapter.catalog_calls == 0
        assert adapter.clone_requests == []
        assert not runtime_root.exists()
        assert registry._total_leases() == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_source", ("studio_saved", "global"))
async def test_clone_reference_only_policy_uses_inherited_studio_voice(
    tmp_path: Any,
    voice_source: str,
) -> None:
    adapter = _PolicyCloneAdapter(voice_required=False)
    global_preferences = _snapshot(
        model_id="clone-model",
        voice_mode="exact",
        voice_id="global-voice",
    )
    saved = StudioTTSPreferencesSnapshot(
        revision=2,
        selection=(
            StudioTTSSelectionOverrides(
                voice_mode="exact",
                voice_id="saved-voice",
            )
            if voice_source == "studio_saved"
            else StudioTTSSelectionOverrides()
        ),
    )
    service, registry = _native_service(
        adapter,
        global_preferences,
        studio_preferences_loader=lambda: saved,
    )
    runtime_root = tmp_path / "clone-runtime"
    service._clone_materializer = TTSCloneReferenceMaterializer(runtime_root)
    requirement = TTSCloneRecipeRequirement(
        recipe_id="pocket_tts",
        recipe_revision=1,
        model_id="clone-model",
    )
    preview = STTSPlaygroundProfilePreview(
        profile_id=UUID("88888888-8888-4888-8888-888888888888"),
        repository_generation=7,
        profile_revision=4,
    )
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=saved.revision,
        preview=True,
    )

    async def exact(current: TTSCloneRecipeRequirement):
        return generation_module.AudioCppGuidedDependencySnapshot(
            state="exact",
            provider_configuration_revision=1,
            saved_generation=1,
            applied_generation=1,
            pending_configuration=False,
            saved_requirement=current,
            applied_requirement=current,
        )

    async def resolver(*_args: object) -> TTSCloneReference:
        return _clone_reference(requirement)

    service.audio_cpp_guided_dependency_snapshot = exact  # type: ignore[method-assign]
    try:
        with pytest.raises(TTSOperationError) as caught:
            await service.synthesize_effective(
                text="Profile preview.",
                studio_draft=draft,
                studio_preferences=saved,
                profile_preview=preview,
                profile_reference_resolver=resolver,
            )

        assert caught.value.code == "dependency_changed"
        assert adapter.ensure_ready_calls == 0
        assert adapter.catalog_calls == 0
        assert adapter.clone_requests == []
        assert not runtime_root.exists()
        assert registry._total_leases() == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_clone_both_required_policy_rejects_missing_effective_voice(
    tmp_path: Any,
) -> None:
    adapter = _PolicyCloneAdapter(voice_required=True)
    service, registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
    )
    runtime_root = tmp_path / "clone-runtime"
    service._clone_materializer = TTSCloneReferenceMaterializer(runtime_root)
    requirement = TTSCloneRecipeRequirement(
        recipe_id="pocket_tts",
        recipe_revision=1,
        model_id="clone-model",
    )
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=7,
        profile_revision=4,
        profile_id=UUID("77777777-7777-4777-8777-777777777777"),
        reference=_clone_reference(requirement),
    )

    async def exact(current: TTSCloneRecipeRequirement):
        return generation_module.AudioCppGuidedDependencySnapshot(
            state="exact",
            provider_configuration_revision=1,
            saved_generation=1,
            applied_generation=1,
            pending_configuration=False,
            saved_requirement=current,
            applied_requirement=current,
        )

    service.audio_cpp_guided_dependency_snapshot = exact  # type: ignore[method-assign]
    try:
        with pytest.raises(TTSOperationError) as caught:
            await service.synthesize_effective(
                text="Character response.",
                character_profile=character,
            )

        assert caught.value.code == "dependency_changed"
        assert adapter.ensure_ready_calls == 0
        assert adapter.catalog_calls == 0
        assert adapter.clone_requests == []
        assert not runtime_root.exists()
        assert registry._total_leases() == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_transient_clone_uses_existing_typed_materialization_lifetime(
    tmp_path: Any,
) -> None:
    adapter = _CloneCapturingAdapter()
    saved = StudioTTSPreferencesSnapshot(revision=2)
    service, registry = _native_service(
        adapter,
        _snapshot(model_id="clone-model"),
        studio_preferences_loader=lambda: saved,
    )
    materializer = TTSCloneReferenceMaterializer(tmp_path / "clone-runtime")
    service._clone_materializer = materializer
    clone_audition = STTSPlaygroundCloneSnapshot(
        draft_revision=3,
        canonical_reference=_canonical_clone_reference(),
    )
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=saved.revision,
    )
    response: TTSAudioResponse | None = None
    try:
        (
            response,
            selection,
            evidence,
        ) = await service.synthesize_effective_with_evidence(
            text="Transient clone.",
            studio_draft=draft,
            studio_preferences=saved,
            clone_audition=clone_audition,
        )

        assert selection.provider_id == "audio_cpp"
        assert evidence is not None
        assert repr(evidence) == "TTSCloneGenerationEvidence(<private>)"
        assert evidence.canonical_reference == clone_audition.canonical_reference
        assert evidence.model_id == "clone-model"
        assert evidence.recipe_id == "pocket_tts"
        assert evidence.recipe_revision == 1
        assert evidence.provider_configuration_revision == 1
        assert evidence.applied_provider_generation == 0
        assert evidence.process_generation == 7
        assert len(adapter.clone_requests) == 1
        materialization = adapter.clone_requests[0].materialization
        assert materialization.reference_text == "Private reference transcript"
        assert materialization.voice_ref.read_bytes() == _clone_reference().wav_bytes
        assert registry._total_leases() == 1

        await response.aclose()
        response = None

        assert not materialization.voice_ref.exists()
        assert registry._total_leases() == 0
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_character_clone_materialization_lives_through_response_cleanup(
    tmp_path: Any,
) -> None:
    adapter = _CloneCapturingAdapter()
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=lambda _config: adapter,
                initial_config={"mode": "managed"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    materializer = TTSCloneReferenceMaterializer(tmp_path / "clone-runtime")
    service = TTSService(
        registry,
        preferences_snapshot=_snapshot(model_id="clone-model"),
        native_capability_reader=_accepted_native_capability_reader(registry),
        clone_materializer=materializer,
    )
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=13,
        profile_revision=8,
        profile_id=UUID("11111111-1111-4111-8111-111111111111"),
        reference=_clone_reference(),
    )
    response: TTSAudioResponse | None = None
    try:
        response, _selection = await service.synthesize_effective(
            text="Character-authored response.",
            character_profile=character,
        )
        assert len(adapter.clone_requests) == 1
        clone_request = adapter.clone_requests[0]
        assert clone_request.provider_revision == 1
        assert clone_request.applied_provider_generation == 0
        assert clone_request.materialization.voice_ref.exists()
        assert registry._total_leases() == 1

        await response.aclose()
        response = None

        assert adapter.events.index("adapter_cleanup") < len(adapter.events)
        assert not clone_request.materialization.voice_ref.exists()
        assert registry._total_leases() == 0
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_public_service_rejects_forged_internal_clone_request_before_admission() -> (
    None
):
    adapter = _CloneCapturingAdapter()
    service, registry = _native_service(adapter, _snapshot(model_id="clone-model"))
    forged = object.__new__(_AdmittedAudioCppCloneRequest)
    try:
        with pytest.raises(TypeError, match="TTS request is invalid"):
            await service.synthesize(forged)  # type: ignore[arg-type]
        assert registry._total_leases() == 0
        assert adapter.ensure_ready_calls == 0
        assert adapter.events == []
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_staged_generation_does_not_change_admitted_clone_generation(
    tmp_path: Any,
) -> None:
    adapter = _BlockingCloneCapturingAdapter()
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("audio_cpp", "audio.cpp", True),
                factory=lambda _config: adapter,
                initial_config={"version": "applied"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=_snapshot(model_id="clone-model"),
        native_capability_reader=_accepted_native_capability_reader(registry),
        clone_materializer=TTSCloneReferenceMaterializer(tmp_path / "runtime"),
    )
    requirement = TTSCloneRecipeRequirement(
        recipe_id="pocket_tts",
        recipe_revision=1,
        model_id="clone-model",
    )

    async def exact(current: TTSCloneRecipeRequirement):
        return generation_module.AudioCppGuidedDependencySnapshot(
            state="exact",
            provider_configuration_revision=1,
            saved_generation=0,
            applied_generation=0,
            pending_configuration=False,
            saved_requirement=current,
            applied_requirement=current,
        )

    service.audio_cpp_guided_dependency_snapshot = exact  # type: ignore[method-assign]
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=3,
        profile_revision=2,
        profile_id=UUID("22222222-2222-4222-8222-222222222222"),
        reference=_clone_reference(requirement),
    )
    task = asyncio.create_task(
        service.synthesize_effective(text="hello", character_profile=character)
    )
    response: TTSAudioResponse | None = None
    try:
        await _wait_bounded(adapter.ensure_started.wait())
        result = await registry.stage_provider_configuration(
            "audio_cpp", {"version": "saved"}, generation=5
        )
        assert result is ReconfigureResult.CHANGED
        adapter.allow_ensure.set()
        response, _selection = await _wait_bounded(task)
        assert adapter.clone_requests[0].provider_revision == 1
        assert adapter.clone_requests[0].applied_provider_generation == 0
        assert adapter.clone_requests[0].process_generation == 7
        assert adapter.clone_requests[0].recipe_revision == 1
        snapshot = await registry.provider_configuration_snapshot("audio_cpp")
        assert snapshot.staged_generation == 5
        assert snapshot.applied_generation == 0
    finally:
        adapter.allow_ensure.set()
        adapter.allow_cleanup.set()
        if response is not None:
            await response.aclose()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_clone_direct_resource_release_cannot_bypass_response_cleanup(
    tmp_path: Any,
) -> None:
    adapter = _BlockingCloneCapturingAdapter()
    adapter.allow_ensure.set()
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("audio_cpp", "audio.cpp", True),
                factory=lambda _config: adapter,
                initial_config={},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=_snapshot(model_id="clone-model"),
        native_capability_reader=_accepted_native_capability_reader(registry),
        clone_materializer=TTSCloneReferenceMaterializer(tmp_path / "runtime"),
    )
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=3,
        profile_revision=2,
        profile_id=UUID("33333333-3333-4333-8333-333333333333"),
        reference=_clone_reference(),
    )
    response: Any = None
    release_task: asyncio.Task[None] | None = None
    try:
        response, _selection = await service.synthesize_effective(
            text="hello", character_profile=character
        )
        path = adapter.clone_requests[0].materialization.voice_ref
        release_task = response.start_resource_release()
        await _wait_bounded(adapter.cleanup_started.wait())
        assert path.exists()
        assert registry._total_leases() == 1
        assert not release_task.done()

        adapter.allow_cleanup.set()
        await _wait_bounded(release_task)
        assert not path.exists()
        assert registry._total_leases() == 0
        response = None
    finally:
        adapter.allow_cleanup.set()
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_transient_clone_source_rejection_precedes_provider_evidence(
    tmp_path: Any,
) -> None:
    adapter = _RejectedCloneSourceAdapter()
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("audio_cpp", "audio.cpp", True),
                factory=lambda _config: adapter,
                initial_config={},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    native_reads = 0

    async def native_reader(*args: Any) -> TTSNativeCapabilitySnapshot:
        nonlocal native_reads
        native_reads += 1
        return await _accepted_native_capability_reader(registry)(*args)

    saved = StudioTTSPreferencesSnapshot(revision=2)
    service = TTSService(
        registry,
        preferences_snapshot=_snapshot(model_id="clone-model"),
        studio_preferences_loader=lambda: saved,
        native_capability_reader=native_reader,
        clone_materializer=TTSCloneReferenceMaterializer(tmp_path / "runtime"),
    )
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=saved.revision,
    )
    try:
        with pytest.raises(RuntimeError, match="rejected clone source"):
            await service.synthesize_effective(
                text="hello",
                studio_draft=draft,
                studio_preferences=saved,
                clone_audition=STTSPlaygroundCloneSnapshot(
                    draft_revision=3,
                    canonical_reference=_canonical_clone_reference(),
                ),
            )
        assert native_reads == 0
        assert adapter.ensure_ready_calls == 0
        assert adapter.catalog_calls == 0
        assert not (tmp_path / "runtime").exists()
        assert registry._total_leases() == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_clone_source_rejection_precedes_readiness_and_catalog_evidence(
    tmp_path: Any,
) -> None:
    adapter = _RejectedCloneSourceAdapter()
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("audio_cpp", "audio.cpp", True),
                factory=lambda _config: adapter,
                initial_config={},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    native_reads = 0

    async def native_reader(*_args: Any) -> TTSNativeCapabilitySnapshot:
        nonlocal native_reads
        native_reads += 1
        return await _accepted_native_capability_reader(registry)(*_args)

    service = TTSService(
        registry,
        preferences_snapshot=_snapshot(model_id="clone-model"),
        native_capability_reader=native_reader,
        clone_materializer=TTSCloneReferenceMaterializer(tmp_path / "runtime"),
    )
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=1,
        profile_revision=1,
        profile_id=UUID("44444444-4444-4444-8444-444444444444"),
        reference=_clone_reference(),
    )
    try:
        with pytest.raises(RuntimeError, match="rejected clone source"):
            await service.synthesize_effective(
                text="hello", character_profile=character
            )
        assert native_reads == 0
        assert adapter.ensure_ready_calls == 0
        assert adapter.catalog_calls == 0
        assert not (tmp_path / "runtime").exists()
        assert registry._total_leases() == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_shutdown_does_not_release_executing_clone_lease_before_materialization(
    tmp_path: Any,
) -> None:
    adapter = _BlockingCloneCapturingAdapter()
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("audio_cpp", "audio.cpp", True),
                factory=lambda _config: adapter,
                initial_config={},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
        shutdown_timeout_seconds=0.01,
    )
    service = TTSService(
        registry,
        preferences_snapshot=_snapshot(model_id="clone-model"),
        native_capability_reader=_accepted_native_capability_reader(registry),
        clone_materializer=TTSCloneReferenceMaterializer(tmp_path / "runtime"),
    )
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="clone-model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=1,
        profile_revision=1,
        profile_id=UUID("55555555-5555-4555-8555-555555555555"),
        reference=_clone_reference(),
    )
    generation = asyncio.create_task(
        service.synthesize_effective(text="hello", character_profile=character)
    )
    try:
        await _wait_bounded(adapter.ensure_started.wait())
        await service.close()
        await asyncio.sleep(0.02)

        assert sum(record.leases for record in registry._closing_records) == 1
        assert service._operation_limit._value == 3
        assert not (tmp_path / "runtime").exists()

        adapter.allow_ensure.set()
        result = await asyncio.gather(generation, return_exceptions=True)
        assert isinstance(result[0], BaseException)
        await service.wait_closed()
        assert sum(record.leases for record in registry._closing_records) == 0
        assert service._operation_limit._value == 4
    finally:
        adapter.allow_ensure.set()
        adapter.allow_cleanup.set()
        if not generation.done():
            generation.cancel()
            await asyncio.gather(generation, return_exceptions=True)
        await service.wait_closed()


@pytest.mark.asyncio
async def test_effective_admission_marks_unsaved_studio_preview() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    saved = StudioTTSPreferencesSnapshot()
    service, _registry = _native_service(
        adapter,
        _snapshot(),
        studio_preferences_loader=lambda: saved,
    )
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="Preview/Model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=0,
        preview=True,
    )
    response: TTSAudioResponse | None = None

    try:
        response, effective = await service.synthesize_effective(
            text="Preview only.",
            studio_draft=draft,
            studio_preferences=saved,
        )

        assert effective.studio_preview is True
        assert effective.sources["provider_id"] is TTSSelectionSource.STUDIO_DRAFT
        assert effective.sources["model_id"] is TTSSelectionSource.STUDIO_DRAFT
        assert effective.revisions.studio_preferences == 0
        assert saved == StudioTTSPreferencesSnapshot()
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_studio_admission_rejects_snapshot_behind_current_store_revision() -> (
    None
):
    adapter = _CapturingAdapter("openai")
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="openai",
                    display_name="OpenAI",
                    native=False,
                ),
                factory=lambda _config: adapter,
                initial_config={},
            ),
        ),
        aliases={},
    )
    current = StudioTTSPreferencesSnapshot(revision=6)
    service = _test_service(
        registry,
        preferences_snapshot=_snapshot(
            provider_id="openai",
            model_id="tts-1",
            voice_mode="exact",
            voice_id="alloy",
            response_format="mp3",
        ),
        studio_preferences_loader=lambda: current,
    )

    try:
        with pytest.raises(TTSEffectiveResolutionError) as caught:
            await service.synthesize_effective(
                text="Do not synthesize stale Studio state.",
                studio_draft=TTSStudioDraftSelection(
                    selection=TTSSelectionOverrides(voice_id="echo"),
                    base_revision=5,
                ),
                studio_preferences=StudioTTSPreferencesSnapshot(revision=5),
            )

        assert caught.value.code == "revision_incoherent"
        assert caught.value.axis == "studio_preferences"
        assert adapter.synthesize_calls == 0
        assert registry._total_leases() == 0
    finally:
        await service.close()
        await service.wait_closed()


class _PauseOnceService(TTSService):
    def __init__(
        self,
        registry: TTSAdapterRegistry,
        snapshot: TTSPreferencesSnapshot,
    ) -> None:
        self.admission_started = asyncio.Event()
        self.allow_admission = asyncio.Event()
        self.frozen_requests: list[TTSRequest] = []
        self._pause_next_admission = True
        super().__init__(
            registry,
            preferences_snapshot=snapshot,
            native_capability_reader=_accepted_native_capability_reader(registry),
        )

    async def _pause_admission(self, request: TTSRequest) -> None:
        if self._pause_next_admission:
            self._pause_next_admission = False
            self.frozen_requests.append(request)
            self.admission_started.set()
            await self.allow_admission.wait()

    async def admit(
        self,
        request: TTSRequest,
        *,
        expected_configuration_revision: int | None = None,
    ) -> Any:
        await self._pause_admission(request)
        return await super().admit(
            request,
            expected_configuration_revision=expected_configuration_revision,
        )

    async def _admit_reserved(
        self,
        request: TTSRequest,
        reservation: Any,
        *,
        expected_configuration_revision: int | None = None,
        **kwargs: Any,
    ) -> Any:
        await self._pause_admission(request)
        return await super()._admit_reserved(
            request,
            reservation,
            expected_configuration_revision=expected_configuration_revision,
            **kwargs,
        )


class _CountingRecordingRegistry(_RecordingRegistry):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.release_calls = 0

    async def _release(self, slot: Any, record: Any) -> None:
        self.release_calls += 1
        await super()._release(slot, record)


class _BlockingExactAdapter(_CapturingAdapter):
    def __init__(self) -> None:
        super().__init__("audio_cpp")
        self.synthesis_started = asyncio.Event()
        self.allow_synthesis = asyncio.Event()

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        self.requests.append(request)
        self.synthesis_started.set()
        await self.allow_synthesis.wait()
        return await FakeAdapter.synthesize(self, request, progress_sink)


class _RevisionCountingRegistry(_RecordingRegistry):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.revision_reads = 0

    def configuration_revision(self, provider_id: str) -> int:
        self.revision_reads += 1
        return super().configuration_revision(provider_id)


def _counting_native_registry(
    adapter: _CapturingAdapter,
    *,
    shutdown_timeout_seconds: float = 10.0,
) -> _CountingRecordingRegistry:
    return _CountingRecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=lambda _config: adapter,
                initial_config={"generation": "initial"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )


@pytest.mark.asyncio
async def test_exact_native_admission_freezes_text_free_selection_and_releases_gate() -> (
    None
):
    adapter = _BlockingExactAdapter()
    registry = _counting_native_registry(adapter)
    service = _test_service(registry)
    mutable_options: dict[str, Any] = {}
    request = TTSRequest(
        provider_id="audio_cpp",
        model_id="Model/Exact",
        text="private submitted text",
        voice="Voice/Exact",
        response_format="wav",
        speed=1.0,
        options=mutable_options,
    )
    synthesis = asyncio.create_task(service.synthesize_exact(request))
    await _wait_bounded(adapter.synthesis_started.wait())
    writer_entered = asyncio.Event()

    async def enter_writer() -> None:
        async with service._request_admission._gate.write():
            writer_entered.set()

    writer = asyncio.create_task(enter_writer())
    response: TTSAudioResponse | None = None
    try:
        await _wait_bounded(writer_entered.wait())
        mutable_options["late"] = "private control value"
        adapter.allow_synthesis.set()
        response, selection = await _wait_bounded(synthesis)

        assert isinstance(selection, TTSRequestedSelectionSnapshot)
        assert selection.provider_id == "audio_cpp"
        assert selection.model_id == "Model/Exact"
        assert selection.voice_id == "Voice/Exact"
        assert selection.response_format == "wav"
        assert selection.speed == 1.0
        assert selection.options == {}
        assert selection.configuration_revision == 1
        assert not hasattr(selection, "text")
        assert "private submitted text" not in repr(selection)
        assert adapter.requests[0].options == {}
        assert registry.expected_revisions == [("audio_cpp", 1)]
        assert registry._total_leases() == 1
        response.model_id = "server-reported-model"
        assert selection.model_id == "Model/Exact"
    finally:
        adapter.allow_synthesis.set()
        await asyncio.gather(synthesis, writer, return_exceptions=True)
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    (
        {"options": {"origin": "https://user:password@example.invalid"}},
        {"options": {"credential": "PRIVATE_API_KEY"}},
        {"options": {"raw_body": bytearray(b"PRIVATE_RAW_BODY")}},
        {"options": {1: "PRIVATE_NON_STRING_KEY"}},
        {"response_format": "mp3"},
        {"speed": 1.1},
    ),
)
async def test_exact_audio_cpp_admission_rejects_unreviewed_contract_values(
    updates: dict[str, object],
) -> None:
    private_values = (
        "PRIVATE_SUBMITTED_TEXT",
        "https://user:password@example.invalid",
        "PRIVATE_API_KEY",
        "PRIVATE_RAW_BODY",
        "PRIVATE_NON_STRING_KEY",
    )
    adapter = _CapturingAdapter("audio_cpp")
    registry = _counting_native_registry(adapter)
    service = _test_service(registry)
    values: dict[str, object] = {
        "provider_id": "audio_cpp",
        "model_id": "Model/Exact",
        "text": "PRIVATE_SUBMITTED_TEXT",
        "voice": None,
        "response_format": "wav",
        "speed": 1.0,
        "options": {},
    }
    values.update(updates)
    response: TTSAudioResponse | None = None
    try:
        with pytest.raises((TypeError, ValueError)) as captured:
            response, _selection = await service.synthesize_exact(
                TTSRequest(**values)  # type: ignore[arg-type]
            )

        rendered = f"{captured.value!s} {captured.value!r}"
        for private_value in private_values:
            assert private_value not in rendered
        assert adapter.synthesize_calls == 0
        assert registry._total_leases() == 0
        assert service._operation_limit._value == 4
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_exact_admission_rejects_unreviewed_native_provider() -> None:
    audio_cpp = _CapturingAdapter("audio_cpp")
    future_native = _CapturingAdapter("future_native")
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=lambda _config: audio_cpp,
                initial_config={},
            ),
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="future_native",
                    display_name="Future native",
                    native=True,
                ),
                factory=lambda _config: future_native,
                initial_config={},
            ),
        ),
        aliases={},
    )
    service = _test_service(registry)
    response: TTSAudioResponse | None = None
    try:
        with pytest.raises(ValueError, match="exact audio_cpp"):
            response, _selection = await service.synthesize_exact(
                TTSRequest(
                    provider_id="future_native",
                    model_id="model",
                    text="private text",
                    voice=None,
                    response_format="wav",
                    speed=1.0,
                    options={},
                )
            )

        assert future_native.synthesize_calls == 0
        assert registry._total_leases() == 0
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_exact_native_entrypoint_rejects_legacy_provider_before_synthesis() -> (
    None
):
    adapter = _CapturingAdapter("openai")
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="openai",
                    display_name="OpenAI",
                    native=False,
                ),
                factory=lambda _config: adapter,
                initial_config={},
            ),
        ),
        aliases={},
    )
    service = _test_service(registry)

    try:
        with pytest.raises(ValueError, match="exact audio_cpp"):
            await service.synthesize_exact(
                TTSRequest(
                    provider_id="openai",
                    model_id="tts-1",
                    text="private text",
                    voice="alloy",
                    response_format="mp3",
                    speed=1.0,
                    options={},
                )
            )

        assert adapter.synthesize_calls == 0
        assert registry._total_leases() == 0
        assert service._operation_limit._value == 4
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_revision_decision_waits_for_queued_writer_and_reads_once() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    registry = _RevisionCountingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=lambda _config: adapter,
                initial_config={"generation": "one"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    service = _test_service(registry)
    first_reader_entered = asyncio.Event()
    release_first_reader = asyncio.Event()
    first_reader = asyncio.create_task(
        _hold_gate(
            service._request_admission._gate.read(),
            first_reader_entered,
            release_first_reader,
        )
    )
    await _wait_bounded(first_reader_entered.wait())
    writer_entered = asyncio.Event()

    async def publish_new_revision() -> None:
        async with service._request_admission._gate.write():
            writer_entered.set()
            await registry.reconfigure_provider(
                "audio_cpp",
                {"generation": "two"},
            )

    writer = asyncio.create_task(publish_new_revision())
    while service._request_admission._gate._waiting_writer_count == 0:
        await asyncio.sleep(0)
    registry.revision_reads = 0
    decision = asyncio.create_task(
        service.require_current_configuration_revision("audio_cpp", 1)
    )
    await asyncio.sleep(0)
    assert not decision.done()

    release_first_reader.set()
    await _wait_bounded(writer_entered.wait())
    await _wait_bounded(writer)
    with pytest.raises(
        TTSConfigurationRevisionError,
        match="TTS provider configuration changed: audio_cpp",
    ):
        await _wait_bounded(decision)

    assert registry.revision_reads == 1
    assert registry.configuration_revision("audio_cpp") == 2
    await first_reader
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_current_revision_decision_returns_without_gate_or_lease() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    service, registry = _native_service(adapter, _snapshot())

    await service.require_current_configuration_revision("audio_cpp", 1)
    writer_entered = asyncio.Event()
    async with service._request_admission._gate.write():
        writer_entered.set()

    assert writer_entered.is_set()
    assert registry._total_leases() == 0
    await service.close()
    await service.wait_closed()


class _GateExitPauseService(TTSService):
    def __init__(
        self,
        registry: TTSAdapterRegistry,
        snapshot: TTSPreferencesSnapshot,
    ) -> None:
        self.operation_admitted = asyncio.Event()
        self.allow_admit_return = asyncio.Event()
        self._admit_return_paused = False
        super().__init__(
            registry,
            max_concurrent_operations=1,
            preferences_snapshot=snapshot,
            native_capability_reader=_accepted_native_capability_reader(registry),
        )

    async def _pause_admit_return(self, operation: Any) -> Any:
        if self._admit_return_paused:
            return operation
        self._admit_return_paused = True
        self.operation_admitted.set()
        await self.allow_admit_return.wait()
        return operation

    async def admit(
        self,
        request: TTSRequest,
        *,
        expected_configuration_revision: int | None = None,
    ) -> Any:
        operation = await super().admit(
            request,
            expected_configuration_revision=expected_configuration_revision,
        )
        return await self._pause_admit_return(operation)

    async def _admit_reserved(
        self,
        request: TTSRequest,
        reservation: Any,
        *,
        expected_configuration_revision: int | None = None,
        **kwargs: Any,
    ) -> Any:
        operation = await super()._admit_reserved(
            request,
            reservation,
            expected_configuration_revision=expected_configuration_revision,
            **kwargs,
        )
        return await self._pause_admit_return(operation)


@pytest.mark.asyncio
async def test_cancellation_during_read_gate_exit_closes_claimed_operation() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    registry = _counting_native_registry(adapter)
    service = _GateExitPauseService(registry, _snapshot())
    generation = asyncio.create_task(
        service.synthesize_default(text="Character response")
    )
    await asyncio.wait_for(service.operation_admitted.wait(), timeout=_WAIT_SECONDS)

    try:
        async with service._request_admission._gate._condition:
            service.allow_admit_return.set()
            await asyncio.sleep(0)
            assert not generation.done()
            generation.cancel("cancelled during read-gate exit")
            await asyncio.sleep(0)
            assert not generation.done()

        with pytest.raises(asyncio.CancelledError) as cancellation:
            await generation
        assert cancellation.value.args == ("cancelled during read-gate exit",)
        assert adapter.synthesize_calls == 0
        assert service._admitted_operations == set()
        assert service._operation_limit._value == 1
        assert registry._total_leases() == 0
        assert registry.release_calls == 1
    finally:
        if not generation.done():
            generation.cancel()
            await asyncio.gather(generation, return_exceptions=True)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_shutdown_during_read_gate_exit_cannot_consume_claimed_operation() -> (
    None
):
    adapter = _CapturingAdapter("audio_cpp")
    registry = _counting_native_registry(
        adapter,
        shutdown_timeout_seconds=0,
    )
    service = _GateExitPauseService(registry, _snapshot())
    generation = asyncio.create_task(
        service.synthesize_default(text="Character response")
    )
    await asyncio.wait_for(service.operation_admitted.wait(), timeout=_WAIT_SECONDS)
    close_task: asyncio.Task[None] | None = None

    try:
        async with service._request_admission._gate._condition:
            service.allow_admit_return.set()
            await asyncio.sleep(0)
            assert not generation.done()
            close_task = asyncio.create_task(service.close())
            await asyncio.wait_for(
                service._close_signal.wait(),
                timeout=_WAIT_SECONDS,
            )
            await asyncio.sleep(0)

        with pytest.raises(TTSRegistryClosedError, match="service is closed"):
            await generation
        assert adapter.synthesize_calls == 0
        assert service._admitted_operations == set()
        assert service._operation_limit._value == 1
        assert registry._total_leases() == 0
        assert registry.release_calls == 1
    finally:
        if not generation.done():
            generation.cancel()
        tasks: list[asyncio.Future[Any]] = [generation]
        if close_task is not None:
            tasks.append(close_task)
        await asyncio.gather(*tasks, return_exceptions=True)
        await service.close()
        await service.wait_closed()


class _CapacityObservedService(TTSService):
    def __init__(
        self,
        registry: TTSAdapterRegistry,
        snapshot: TTSPreferencesSnapshot,
    ) -> None:
        self.capacity_acquisitions = 0
        self.second_capacity_wait_started = asyncio.Event()
        super().__init__(
            registry,
            max_concurrent_operations=1,
            preferences_snapshot=snapshot,
            native_capability_reader=_accepted_native_capability_reader(registry),
        )

    async def _acquire_operation_slot(self) -> None:
        self.capacity_acquisitions += 1
        if self.capacity_acquisitions == 2:
            self.second_capacity_wait_started.set()
        await super()._acquire_operation_slot()


@pytest.mark.asyncio
async def test_saturated_capacity_waiter_does_not_block_waiting_writer() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    registry = _counting_native_registry(adapter)
    service = _CapacityObservedService(registry, _snapshot())
    first_response = await service.synthesize_default(text="First response")
    second_generation = asyncio.create_task(
        service.synthesize_default(text="Second response")
    )
    await asyncio.wait_for(
        service.second_capacity_wait_started.wait(),
        timeout=_WAIT_SECONDS,
    )
    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()

    async def publish() -> None:
        async with service._request_admission._gate.write():
            writer_entered.set()
            await release_writer.wait()

    publication = asyncio.create_task(publish())
    try:
        for _ in range(3):
            await asyncio.sleep(0)
        assert writer_entered.is_set()
        assert adapter.response_close_calls == 0
        assert not second_generation.done()
    finally:
        second_generation.cancel()
        release_writer.set()
        await first_response.aclose()
        await asyncio.gather(
            second_generation,
            publication,
            return_exceptions=True,
        )
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_cancelled_capacity_reservation_does_not_leak_or_overrelease() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    registry = _counting_native_registry(adapter)
    service = _CapacityObservedService(registry, _snapshot())
    first_response = await service.synthesize_default(text="First response")
    waiting_generation = asyncio.create_task(
        service.synthesize_default(text="Waiting response")
    )
    await asyncio.wait_for(
        service.second_capacity_wait_started.wait(),
        timeout=_WAIT_SECONDS,
    )

    waiting_generation.cancel("cancelled while reserving capacity")
    with pytest.raises(asyncio.CancelledError) as cancellation:
        await waiting_generation
    assert cancellation.value.args == ("cancelled while reserving capacity",)
    assert service._operation_limit._value == 0
    assert registry._total_leases() == 1

    await first_response.aclose()
    next_response = await service.synthesize_default(text="Next response")
    await next_response.aclose()

    assert service._operation_limit._value == 1
    assert registry._total_leases() == 0
    assert registry.release_calls == 2
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_default_request_never_mixes_preference_and_adapter_generations() -> None:
    adapters: list[_CapturingAdapter] = []

    def factory(config: Mapping[str, Any]) -> _CapturingAdapter:
        adapter = _CapturingAdapter(
            "audio_cpp",
            generation=str(config["generation"]),
        )
        adapters.append(adapter)
        return adapter

    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config={"generation": "old"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    old_snapshot = _snapshot(model_id="Old/Model")
    new_snapshot = _snapshot(model_id="New/Model")
    service = _PauseOnceService(registry, old_snapshot)
    coordinator = service._request_admission
    request_task = asyncio.create_task(
        service.synthesize_default(text="Character response")
    )
    await asyncio.wait_for(service.admission_started.wait(), timeout=_WAIT_SECONDS)

    async def publish_new_generation() -> None:
        async with coordinator._gate.write():
            await service.reconfigure_provider(
                "audio_cpp",
                {"generation": "new"},
            )
            coordinator._preferences = new_snapshot

    publication = asyncio.create_task(publish_new_generation())
    await asyncio.sleep(0)
    service.allow_admission.set()
    first_response = await asyncio.wait_for(request_task, timeout=_WAIT_SECONDS)

    try:
        assert adapters[0].generation == "old"
        assert [request.model_id for request in adapters[0].requests] == ["Old/Model"]
        assert registry.expected_revisions == [("audio_cpp", 1)]
        assert not publication.done()
    finally:
        await first_response.aclose()

    await asyncio.wait_for(publication, timeout=_WAIT_SECONDS)
    assert service.preferences_snapshot() == new_snapshot
    second_response = await service.synthesize_default(text="Next response")
    try:
        assert [
            (adapter.generation, adapter.requests[0].model_id) for adapter in adapters
        ] == [
            ("old", "Old/Model"),
            ("new", "New/Model"),
        ]
        assert registry.expected_revisions == [
            ("audio_cpp", 1),
            ("audio_cpp", 2),
        ]
    finally:
        await second_response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_settings_publication_times_out_without_cancelling_old_speech() -> None:
    adapters: list[_CapturingAdapter] = []

    def factory(config: Mapping[str, Any]) -> _CapturingAdapter:
        adapter = _CapturingAdapter(
            "audio_cpp",
            generation=str(config["generation"]),
        )
        adapters.append(adapter)
        return adapter

    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config={"generation": "one"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    old_snapshot = _snapshot(model_id="Old/Model")
    new_snapshot = _snapshot(model_id="New/Model")
    service = _test_service(registry, preferences_snapshot=old_snapshot)
    response = await service.synthesize_default(text="Generation one")

    outcome_type = getattr(
        generation_module,
        "TTSSettingsPersistenceOutcome",
        None,
    )
    assert outcome_type is not None, "settings persistence outcome is missing"
    ticket = service.begin_preferences_publication(
        new_snapshot,
        {"audio_cpp": {"generation": "two"}},
        lambda: outcome_type(True, True, None),
        foreground_timeout_seconds=0,
    )
    foreground = await asyncio.shield(ticket.foreground)

    assert foreground.generation == ticket.generation
    assert foreground.persistence.file_replaced is True
    assert foreground.provider_statuses == {"audio_cpp": "pending"}
    # Activation is fenced behind the provider handoff (37da4620a): while
    # the old speech's open response holds the exclusive lease, the handoff
    # -- and therefore the in-memory default -- is still the OLD snapshot.
    assert service.preferences_snapshot() == old_snapshot
    assert registry.configuration_revision("audio_cpp") == 1
    with pytest.raises(TTSProviderReconfiguringError):
        await registry.acquire("audio_cpp")
    assert len(adapters) == 1
    assert adapters[0].close_calls == 0

    chunks = [chunk async for chunk in response.byte_stream]
    assert chunks == [b"audio"]
    await response.aclose()
    completion = await asyncio.shield(ticket.completion)

    assert completion.provider_statuses == {"audio_cpp": "applied"}
    assert registry.configuration_revision("audio_cpp") == 2
    assert adapters[0].close_calls == 1
    assert len(adapters) == 1
    # The handoff applied once the old speech released its lease, so the
    # new default activates only now -- the ordering 37da4620a guarantees.
    assert service.preferences_snapshot() == new_snapshot

    replacement = await service.synthesize_default(text="Generation two")
    assert adapters[1].generation == "two"
    assert adapters[1].requests[0].model_id == "New/Model"
    await replacement.aclose()
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_unrelated_publication_cannot_activate_staged_managed_preferences() -> (
    None
):
    service, adapters, _supervisor = _managed_promotion_service()
    registry = service.registry
    saved = _snapshot(model_id="Model/B")
    staged = await _publish_settings(
        service, saved, {"audio_cpp": _managed_config(6.0)}
    )
    assert staged.provider_statuses == {"audio_cpp": "pending"}
    await _publish_settings(service, saved, {"other": {"generation": "two"}})

    response = await _wait_bounded(service.synthesize_default(text="Still applied A"))
    assert (adapters[0].generation, adapters[0].requests[-1].model_id) == (
        "5.0",
        "Model/A",
    )
    await _wait_bounded(response.aclose())

    await _wait_bounded(service.shutdown_audio_cpp())
    assert service.preferences_snapshot() == saved
    assert registry.configuration_generation("audio_cpp") == staged.generation
    await _wait_bounded(service.close())
    await _wait_bounded(service.wait_closed())


@pytest.mark.asyncio
async def test_cancelled_managed_transition_settles_preferences_before_caller() -> None:
    service, adapters, supervisor = _managed_promotion_service()
    saved = _snapshot(model_id="Model/B")
    active = await _wait_bounded(service.synthesize_default(text="Hold applied A"))
    await _publish_settings(service, saved, {"audio_cpp": _managed_config(6.0)})
    supervisor.state = "stopped"
    transition = asyncio.create_task(
        service.synthesize_default(text="Cancelled promotion")
    )
    await _wait_bounded(supervisor.draining_started.wait())
    transition.cancel("caller cancelled")
    await asyncio.sleep(0)
    assert transition.done() is False

    await _wait_bounded(active.aclose())
    with pytest.raises(asyncio.CancelledError):
        await _wait_bounded(transition)

    assert service.preferences_snapshot() == saved
    replacement = await _wait_bounded(
        service.synthesize_default(text="Coherent after cancellation")
    )
    assert (adapters[1].generation, adapters[1].requests[-1].model_id) == (
        "6.0",
        "Model/B",
    )
    await _wait_bounded(replacement.aclose())
    await _wait_bounded(service.close())
    await _wait_bounded(service.wait_closed())


@pytest.mark.asyncio
async def test_pre_replacement_failure_changes_no_preferences_or_provider() -> None:
    adapter = _CapturingAdapter("audio_cpp", generation="one")
    old_snapshot = _snapshot(model_id="Model/One")
    service, registry = _native_service(adapter, old_snapshot)
    lease = await registry.acquire("audio_cpp")
    await lease.release()
    outcome_type = generation_module.TTSSettingsPersistenceOutcome

    ticket = service.begin_preferences_publication(
        _snapshot(model_id="Model/Two"),
        {"audio_cpp": {"generation": "two"}},
        lambda: outcome_type(False, False, "before_replace"),
        foreground_timeout_seconds=0,
    )
    foreground = await asyncio.shield(ticket.foreground)
    completion = await asyncio.shield(ticket.completion)

    assert foreground == completion
    assert foreground.published is False
    assert foreground.provider_statuses == {"audio_cpp": "unchanged"}
    assert service.preferences_snapshot() == old_snapshot
    assert service.preferences_generation() == 0
    assert registry.configuration_revision("audio_cpp") == 1
    assert adapter.close_calls == 0

    response = await service.synthesize_default(text="Still generation one")
    assert adapter.requests[-1].model_id == "Model/One"
    await response.aclose()
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_durable_save_advances_saved_generation_when_runtime_staging_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _CapturingAdapter("audio_cpp", generation="one")
    old_snapshot = _snapshot(model_id="Model/One")
    saved_snapshot = _snapshot(model_id="Model/Two")
    service, registry = _native_service(adapter, old_snapshot)

    async def fail_runtime_stage(
        _provider_id: str,
        _config: Mapping[str, Any],
        *,
        generation: int,
    ) -> str | None:
        del generation
        raise RuntimeError("private runtime transition failure")

    monkeypatch.setattr(service, "_stage_managed_boundary", fail_runtime_stage)
    ticket = service.begin_preferences_publication(
        saved_snapshot,
        {"audio_cpp": {"generation": "two"}},
        lambda: generation_module.TTSSettingsPersistenceOutcome(
            True,
            True,
            None,
        ),
        foreground_timeout_seconds=0,
    )

    try:
        completion = await asyncio.shield(ticket.completion)

        assert completion.published is True
        assert completion.provider_statuses == {"audio_cpp": "unavailable"}
        # The durable save advanced the SAVED generation, but activation is
        # fenced behind the provider handoff (37da4620a): the runtime
        # transition failed, so the in-memory default stays on the last
        # snapshot the runtime actually accepted.
        assert service.preferences_snapshot() == old_snapshot
        assert service.saved_configuration_revision("audio_cpp") == ticket.generation
        assert service.applied_configuration_revision("audio_cpp") == 0
        # The PUBLICATION failed, not the provider: the slot is not sealed
        # (sealing is reserved for a failed reviewed handoff), so the
        # runtime stays usable for the next attempt.
        lease = await registry.acquire("audio_cpp")
        await lease.release()
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_first_publication_cannot_collide_with_compatibility_reconfigure() -> (
    None
):
    adapters: list[_CapturingAdapter] = []

    def factory(config: Mapping[str, Any]) -> _CapturingAdapter:
        adapter = _CapturingAdapter(
            "audio_cpp",
            generation=str(config["generation"]),
        )
        adapters.append(adapter)
        return adapter

    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config={"generation": "initial"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    old_snapshot = _snapshot(model_id="Model/Initial")
    saved_snapshot = _snapshot(model_id="Model/Saved")
    service = _test_service(registry, preferences_snapshot=old_snapshot)

    try:
        assert (
            await service.reconfigure_provider(
                "audio_cpp",
                {"generation": "compatibility"},
            )
            is ReconfigureResult.CHANGED
        )
        assert registry.configuration_revision("audio_cpp") == 2

        ticket = service.begin_preferences_publication(
            saved_snapshot,
            {"audio_cpp": {"generation": "saved"}},
            lambda: generation_module.TTSSettingsPersistenceOutcome(
                True,
                True,
                None,
            ),
            foreground_timeout_seconds=0,
        )
        foreground = await asyncio.shield(ticket.foreground)
        completion = await asyncio.shield(ticket.completion)

        assert foreground.provider_statuses == {"audio_cpp": "applied"}
        assert completion.provider_statuses == {"audio_cpp": "applied"}
        assert service.preferences_snapshot() == saved_snapshot
        assert service.preferences_generation() == ticket.generation
        assert registry.configuration_generation("audio_cpp") == ticket.generation
        assert registry.configuration_revision("audio_cpp") == 3

        response = await service.synthesize_default(text="Saved generation")
        assert adapters[0].generation == "saved"
        assert adapters[0].requests[0].model_id == "Model/Saved"
        await response.aclose()
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_compatibility_reconfigure_cannot_supersede_pending_publication() -> None:
    adapters: list[_CapturingAdapter] = []

    def factory(config: Mapping[str, Any]) -> _CapturingAdapter:
        adapter = _CapturingAdapter(
            "audio_cpp",
            generation=str(config["generation"]),
        )
        adapters.append(adapter)
        return adapter

    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config={"generation": "initial"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    saved_snapshot = _snapshot(model_id="Model/Saved")
    service = _test_service(
        registry,
        preferences_snapshot=_snapshot(model_id="Model/Initial"),
    )
    response = await service.synthesize_default(text="Active speech")

    try:
        publication = service.begin_preferences_publication(
            saved_snapshot,
            {"audio_cpp": {"generation": "saved"}},
            lambda: generation_module.TTSSettingsPersistenceOutcome(
                True,
                True,
                None,
            ),
            foreground_timeout_seconds=0,
        )
        foreground = await asyncio.shield(publication.foreground)
        assert foreground.provider_statuses == {"audio_cpp": "pending"}
        # Fenced activation (37da4620a): pending means not yet activated --
        # the in-memory default is still the construction-time snapshot.
        assert service.preferences_snapshot() != saved_snapshot

        failed_publication = service.begin_preferences_publication(
            _snapshot(model_id="Model/Not-Replaced"),
            {},
            lambda: generation_module.TTSSettingsPersistenceOutcome(
                False,
                False,
                "before_replace",
            ),
            foreground_timeout_seconds=0,
        )
        failed_result = await asyncio.shield(failed_publication.completion)
        assert failed_result.published is False
        # The first publication is still pending, the second failed before
        # replace: fenced activation leaves the default untouched.
        assert service.preferences_snapshot() == _snapshot(model_id="Model/Initial")

        compatibility = asyncio.create_task(
            service.reconfigure_provider(
                "audio_cpp",
                {"generation": "compatibility"},
            )
        )
        await asyncio.sleep(0)
        assert compatibility.done() is False

        await response.aclose()
        assert await compatibility is ReconfigureResult.CHANGED
        completion = await asyncio.shield(publication.completion)

        assert completion.provider_statuses == {"audio_cpp": "unavailable"}
        # Fenced activation (37da4620a): the publication's handoff never
        # applied -- the old speech held the lease until the compatibility
        # reconfigure took the generation -- so the default stays on the
        # snapshot the runtime last accepted: the construction-time
        # Model/Initial (compatibility reconfigures provider config, not
        # preferences).
        assert service.preferences_snapshot() == _snapshot(model_id="Model/Initial")
        lease = await registry.acquire("audio_cpp")
        await lease.release()
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_provider_id", ("missing", "audio.cpp"))
async def test_publication_rejects_noncanonical_preference_provider_synchronously(
    invalid_provider_id: str,
) -> None:
    adapter = _CapturingAdapter("audio_cpp", generation="initial")
    initial_snapshot = _snapshot(model_id="Model/Initial")
    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=lambda _config: adapter,
                initial_config={"generation": "initial"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={"audio.cpp": "audio_cpp"},
    )
    service = _test_service(registry, preferences_snapshot=initial_snapshot)
    persistence_calls = 0

    def persistence() -> Any:
        nonlocal persistence_calls
        persistence_calls += 1
        return generation_module.TTSSettingsPersistenceOutcome(True, True, None)

    try:
        with pytest.raises(
            ValueError,
            match="preferences must use a canonical registered provider ID",
        ):
            service.begin_preferences_publication(
                _snapshot(
                    provider_id=invalid_provider_id,
                    model_id="Model/Invalid",
                ),
                {},
                persistence,
            )

        await asyncio.sleep(0)
        assert persistence_calls == 0
        assert service._settings_publication_tasks == set()
        assert service.preferences_snapshot() == initial_snapshot
        assert service.preferences_generation() == 0
        assert service._settings_generation == 0
        assert registry._generation_sequence == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_newer_settings_publication_supersedes_pending_handoff() -> None:
    adapters: list[_CapturingAdapter] = []

    def factory(config: Mapping[str, Any]) -> _CapturingAdapter:
        adapter = _CapturingAdapter(
            "audio_cpp",
            generation=str(config["generation"]),
        )
        adapters.append(adapter)
        return adapter

    registry = _RecordingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config={"generation": "one"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    service = _test_service(
        registry,
        preferences_snapshot=_snapshot(model_id="Model/One"),
    )
    response = await service.synthesize_default(text="Generation one")
    outcome_type = generation_module.TTSSettingsPersistenceOutcome

    generation_two = service.begin_preferences_publication(
        _snapshot(model_id="Model/Two"),
        {"audio_cpp": {"generation": "two"}},
        lambda: outcome_type(True, True, None),
        foreground_timeout_seconds=0,
    )
    assert (await asyncio.shield(generation_two.foreground)).provider_statuses == {
        "audio_cpp": "pending"
    }
    generation_three = service.begin_preferences_publication(
        _snapshot(model_id="Model/Three"),
        {"audio_cpp": {"generation": "three"}},
        lambda: outcome_type(True, True, None),
        foreground_timeout_seconds=0,
    )
    assert (await asyncio.shield(generation_three.foreground)).provider_statuses == {
        "audio_cpp": "pending"
    }

    assert registry.configuration_revision("audio_cpp") == 1
    assert len(adapters) == 1
    await response.aclose()
    second, third = await asyncio.gather(
        asyncio.shield(generation_two.completion),
        asyncio.shield(generation_three.completion),
    )

    assert second.provider_statuses == {"audio_cpp": "superseded"}
    assert third.provider_statuses == {"audio_cpp": "applied"}
    assert service.preferences_snapshot().model_id == "Model/Three"
    assert service.preferences_generation() == generation_three.generation
    assert registry.configuration_revision("audio_cpp") == 2
    assert adapters[0].close_calls == 1
    assert len(adapters) == 1

    replacement = await service.synthesize_default(text="Generation three")
    assert adapters[1].generation == "three"
    await replacement.aclose()
    await service.close()
    await service.wait_closed()


class _OlderPublicationObserverFirstService(TTSService):
    """Force an older pending publication to classify its result first."""

    def __init__(
        self,
        registry: TTSAdapterRegistry,
        snapshot: TTSPreferencesSnapshot,
    ) -> None:
        self.older_generation: int | None = None
        self.newer_generation: int | None = None
        self.older_status_classified = asyncio.Event()
        super().__init__(
            registry,
            preferences_snapshot=snapshot,
            native_capability_reader=_accepted_native_capability_reader(registry),
        )

    async def _reconfiguration_status(
        self,
        provider_id: str,
        ticket: Any,
    ) -> Any:
        if ticket.generation == self.newer_generation:
            await _wait_bounded(self.older_status_classified.wait())
        status = await super()._reconfiguration_status(provider_id, ticket)
        if ticket.generation == self.older_generation:
            self.older_status_classified.set()
        return status


class _ReconfigurationObservedRegistry(_RecordingRegistry):
    """Expose when a publication has registered its provider transition."""

    def __init__(self, **kwargs: Any) -> None:
        self.reconfiguration_begun = asyncio.Event()
        super().__init__(**kwargs)

    async def begin_reconfigure_provider(
        self,
        provider_id: str,
        config: Mapping[str, Any],
        *,
        generation: int | None = None,
    ) -> Any:
        ticket = await super().begin_reconfigure_provider(
            provider_id,
            config,
            generation=generation,
        )
        self.reconfiguration_begun.set()
        return ticket


@pytest.mark.asyncio
async def test_persisted_newer_handoff_supersedes_older_before_snapshot_publish() -> (
    None
):
    adapters: list[_CapturingAdapter] = []

    def factory(config: Mapping[str, Any]) -> _CapturingAdapter:
        adapter = _CapturingAdapter(
            "audio_cpp",
            generation=str(config["generation"]),
        )
        adapters.append(adapter)
        return adapter

    registry = _ReconfigurationObservedRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config={"generation": "one"},
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    service = _OlderPublicationObserverFirstService(
        registry,
        _snapshot(model_id="Model/One"),
    )
    active_response = await service.synthesize_default(text="Generation one")
    outcome_type = generation_module.TTSSettingsPersistenceOutcome
    replacement: TTSAudioResponse | None = None

    try:
        older = service.begin_preferences_publication(
            _snapshot(model_id="Model/Two"),
            {"audio_cpp": {"generation": "two"}},
            lambda: outcome_type(True, True, None),
            foreground_timeout_seconds=0,
        )
        service.older_generation = older.generation
        assert (await _wait_bounded(older.foreground)).provider_statuses == {
            "audio_cpp": "pending"
        }
        registry.reconfiguration_begun.clear()

        newer = service.begin_preferences_publication(
            _snapshot(model_id="Model/Three"),
            {"audio_cpp": {"generation": "three"}},
            lambda: outcome_type(True, True, None),
            foreground_timeout_seconds=_WAIT_SECONDS,
        )
        service.newer_generation = newer.generation

        await _wait_bounded(registry.reconfiguration_begun.wait())
        assert registry._slots["audio_cpp"].pending_generation == newer.generation
        await _wait_bounded(active_response.aclose())

        older_result, newer_foreground, newer_result = await _wait_bounded(
            asyncio.gather(
                asyncio.shield(older.completion),
                asyncio.shield(newer.foreground),
                asyncio.shield(newer.completion),
            )
        )

        assert older_result.provider_statuses == {"audio_cpp": "superseded"}
        assert newer_foreground.provider_statuses == {"audio_cpp": "applied"}
        assert newer_result.provider_statuses == {"audio_cpp": "applied"}
        assert service.preferences_snapshot().model_id == "Model/Three"
        assert service.preferences_generation() == newer.generation
        assert registry.configuration_generation("audio_cpp") == newer.generation
        assert registry.configuration_revision("audio_cpp") == 2
        assert len(adapters) == 1
        assert adapters[0].generation == "one"
        assert adapters[0].close_calls == 1

        replacement = await _wait_bounded(
            service.synthesize_default(text="Generation three")
        )
        assert len(adapters) == 2
        assert adapters[0].close_calls == 1
        assert adapters[1].generation == "three"
        assert adapters[1].requests[0].model_id == "Model/Three"
        assert registry.expected_revisions[-1] == ("audio_cpp", 2)
    finally:
        await _wait_bounded(active_response.aclose())
        if replacement is not None:
            await _wait_bounded(replacement.aclose())
        await _wait_bounded(service.close())
        await _wait_bounded(service.wait_closed())

    assert [adapter.close_calls for adapter in adapters] == [1, 1]


@pytest.mark.asyncio
async def test_persistence_runs_off_loop_and_publications_remain_serialized() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    service, _registry = _native_service(
        adapter,
        _snapshot(model_id="Model/One"),
    )
    loop = asyncio.get_running_loop()
    first_started = asyncio.Event()
    release_first = threading.Event()
    second_started = asyncio.Event()
    release_second = threading.Event()
    second_publication_entered = asyncio.Event()
    outcome_type = generation_module.TTSSettingsPersistenceOutcome
    heartbeat_ticks = 0
    heartbeat_advanced = asyncio.Event()
    heartbeat_running = True
    publication_entries = 0
    publications: list[Any] = []
    original_run = service._run_preferences_publication

    async def observe_publication_entry(*args: Any, **kwargs: Any) -> Any:
        nonlocal publication_entries
        publication_entries += 1
        if publication_entries == 2:
            second_publication_entered.set()
        return await original_run(*args, **kwargs)

    service._run_preferences_publication = (  # type: ignore[method-assign]
        observe_publication_entry
    )

    def heartbeat() -> None:
        nonlocal heartbeat_ticks
        if not heartbeat_running:
            return
        heartbeat_ticks += 1
        heartbeat_advanced.set()
        loop.call_soon(heartbeat)

    def first_persistence() -> Any:
        loop.call_soon_threadsafe(first_started.set)
        release_first.wait()
        return outcome_type(True, True, None)

    def second_persistence() -> Any:
        loop.call_soon_threadsafe(second_started.set)
        release_second.wait()
        return outcome_type(True, True, None)

    loop.call_soon(heartbeat)
    try:
        first = service.begin_preferences_publication(
            _snapshot(model_id="Model/Two"),
            {},
            first_persistence,
            foreground_timeout_seconds=0,
        )
        publications.append(first)
        await _wait_bounded(first_started.wait())
        observed_ticks = heartbeat_ticks
        heartbeat_advanced.clear()
        await _wait_bounded(heartbeat_advanced.wait())
        assert heartbeat_ticks > observed_ticks

        async def wait_for_foreground() -> Any:
            return await asyncio.shield(first.foreground)

        initiating_waiter = asyncio.create_task(wait_for_foreground())
        initiating_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await _wait_bounded(initiating_waiter)
        assert first.completion.cancelled() is False

        second = service.begin_preferences_publication(
            _snapshot(model_id="Model/Three"),
            {},
            second_persistence,
            foreground_timeout_seconds=0,
        )
        publications.append(second)
        await _wait_bounded(second_publication_entered.wait())
        assert second_started.is_set() is False

        release_first.set()
        await _wait_bounded(first.completion)
        await _wait_bounded(second_started.wait())
        release_second.set()
        await _wait_bounded(second.completion)

        assert service.preferences_snapshot().model_id == "Model/Three"
    finally:
        heartbeat_running = False
        release_first.set()
        release_second.set()
        await _wait_bounded(
            asyncio.gather(
                *(publication.completion for publication in publications),
                return_exceptions=True,
            )
        )
        await _wait_bounded(service.close())
        await _wait_bounded(service.wait_closed())


@pytest.mark.asyncio
async def test_service_shutdown_joins_retained_settings_publication() -> None:
    adapter = _CapturingAdapter("audio_cpp")
    service, _registry = _native_service(
        adapter,
        _snapshot(model_id="Model/One"),
    )
    loop = asyncio.get_running_loop()
    persistence_started = asyncio.Event()
    release_persistence = threading.Event()
    outcome_type = generation_module.TTSSettingsPersistenceOutcome
    ticket: Any = None
    wait_closed: asyncio.Task[None] | None = None

    def persistence() -> Any:
        loop.call_soon_threadsafe(persistence_started.set)
        release_persistence.wait()
        return outcome_type(True, True, None)

    try:
        ticket = service.begin_preferences_publication(
            _snapshot(model_id="Model/Two"),
            {},
            persistence,
            foreground_timeout_seconds=0,
        )
        await _wait_bounded(persistence_started.wait())

        await _wait_bounded(service.close())
        wait_closed_entered = asyncio.Event()

        async def join_service() -> None:
            wait_closed_entered.set()
            await service.wait_closed()

        wait_closed = asyncio.create_task(join_service())
        await _wait_bounded(wait_closed_entered.wait())

        assert wait_closed.done() is False
        assert ticket.completion.done() is False

        release_persistence.set()
        await _wait_bounded(ticket.completion)
        await _wait_bounded(wait_closed)
    finally:
        release_persistence.set()
        pending = [
            task
            for task in (
                ticket.completion if ticket is not None else None,
                wait_closed,
            )
            if task is not None
        ]
        if pending:
            await _wait_bounded(asyncio.gather(*pending, return_exceptions=True))
        if wait_closed is None:
            await _wait_bounded(service.wait_closed())


@pytest.mark.asyncio
async def test_failed_multi_provider_begin_seals_in_reverse_and_joins_started_work() -> (
    None
):
    events: list[str] = []
    alpha_started = asyncio.Event()
    allow_alpha = asyncio.Event()
    secret = "PRIVATE_TRANSITION_VALUE"

    class OrderedFailingRegistry(TTSAdapterRegistry):
        async def begin_reconfigure_provider(
            self,
            provider_id: str,
            config: Mapping[str, Any],
            *,
            generation: int | None = None,
        ) -> Any:
            events.append(f"begin-{provider_id}")
            if provider_id == "beta":
                if events[:2] != ["begin-alpha", "begin-beta"]:
                    raise RuntimeError("providers started out of canonical order")
                await _wait_bounded(alpha_started.wait())
                raise RuntimeError(secret)
            return await super().begin_reconfigure_provider(
                provider_id,
                config,
                generation=generation,
            )

        async def _reconfigure_retiring(
            self,
            slot: Any,
            new_config: dict[str, Any],
        ) -> Any:
            alpha_started.set()
            await _wait_bounded(allow_alpha.wait())
            return await super()._reconfigure_retiring(slot, new_config)

        async def seal_provider_unavailable(self, provider_id: str) -> None:
            events.append(f"seal-{provider_id}")
            await super().seal_provider_unavailable(provider_id)

    registry = OrderedFailingRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("alpha", "alpha", True),
                factory=lambda _config: _CapturingAdapter("alpha"),
                initial_config={"generation": "one"},
            ),
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("beta", "beta", True),
                factory=lambda _config: _CapturingAdapter("beta"),
                initial_config={"generation": "one"},
            ),
        ),
        aliases={},
    )
    service = _test_service(
        registry,
        preferences_snapshot=_snapshot(
            provider_id="alpha",
            model_id="Model/One",
            voice_mode="exact",
            voice_id="voice",
        ),
    )
    outcome_type = generation_module.TTSSettingsPersistenceOutcome
    publication: Any = None
    try:
        publication = service.begin_preferences_publication(
            _snapshot(
                provider_id="alpha",
                model_id="Model/Two",
                voice_mode="exact",
                voice_id="voice",
            ),
            {
                "beta": {"generation": "two"},
                "alpha": {"generation": "two"},
            },
            lambda: outcome_type(True, True, None),
            foreground_timeout_seconds=0,
        )
        foreground = await _wait_bounded(publication.foreground)

        assert foreground.provider_statuses == {
            "alpha": "unavailable",
            "beta": "unavailable",
        }
        # Canonical begin order is still enforced; the reverse SEAL pass is
        # gone (37da4620a and successors: a failed publication marks its
        # providers unavailable in the publication result without sealing
        # the slots -- the providers themselves did not fail, so the next
        # publication may retry them).
        assert events == [
            "begin-alpha",
            "begin-beta",
        ]
        assert publication.completion.done() is False

        allow_alpha.set()
        completion = await _wait_bounded(publication.completion)
        assert completion.provider_statuses == foreground.provider_statuses
        # No seal: the slot is usable again immediately.
        lease = await _wait_bounded(registry.acquire("alpha"))
        await lease.release()
        assert secret not in repr(completion)
    finally:
        allow_alpha.set()
        if publication is not None:
            await _wait_bounded(
                asyncio.gather(
                    publication.completion,
                    return_exceptions=True,
                )
            )
        await _wait_bounded(service.close())
        await _wait_bounded(service.wait_closed())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "provider_id",
        "configured_model",
        "configured_format",
        "expected_model",
        "expected_format",
        "expected_internal_id",
    ),
    (
        (
            "openai",
            "tts-1-hd",
            "opus",
            "tts-1-hd",
            "opus",
            "openai_official_tts-1-hd",
        ),
        (
            "elevenlabs",
            "eleven_multilingual_v2",
            "wav",
            "eleven_multilingual_v2",
            "wav",
            "elevenlabs_eleven_multilingual_v2",
        ),
        (
            "kokoro",
            "kokoro",
            "mp3",
            "kokoro",
            "mp3",
            "local_kokoro_default_onnx",
        ),
        (
            "chatterbox",
            "chatterbox",
            "mp3",
            "chatterbox",
            "mp3",
            "local_chatterbox_default",
        ),
        (
            "higgs",
            "higgs-audio-v2",
            "wav",
            "higgs-audio-v2",
            "wav",
            "local_higgs_v2",
        ),
        (
            "alltalk",
            "alltalk",
            "mp3",
            "alltalk",
            "mp3",
            "alltalk_default",
        ),
    ),
)
async def test_retained_provider_defaults_admit_the_legacy_bridge(
    monkeypatch: pytest.MonkeyPatch,
    provider_id: str,
    configured_model: str,
    configured_format: str,
    expected_model: str,
    expected_format: str,
    expected_internal_id: str,
) -> None:
    captured: list[tuple[str, OpenAISpeechRequest]] = []

    async def audio() -> AsyncIterator[bytes]:
        yield b"audio"

    def capture_generate(
        _host: LegacyBackendHost,
        internal_model_id: str,
        request: OpenAISpeechRequest,
        _progress_sink: ProgressSink | None,
    ) -> AsyncIterator[bytes]:
        captured.append((internal_model_id, request))
        return audio()

    monkeypatch.setattr(LegacyBackendHost, "generate", capture_generate)
    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(
            {},
            manager_factory=lambda _provider, _config: pytest.fail(
                "request admission must stop at the legacy adapter boundary"
            ),
        ),
        aliases={},
    )
    service = _test_service(
        registry,
        preferences_snapshot=_snapshot(
            provider_id=provider_id,
            model_id=configured_model,
            voice_mode="exact",
            voice_id="Voice/Case",
            response_format=configured_format,
        ),
    )

    response = await service.synthesize_default(text="Character response")
    try:
        active = registry._slots[provider_id].active
        assert active is not None
        assert isinstance(active.adapter, LegacyTTSAdapter)
        assert captured == [
            (
                expected_internal_id,
                OpenAISpeechRequest(
                    model=expected_model,
                    input="Character response",
                    voice="Voice/Case",
                    response_format=expected_format,
                    speed=1.0,
                ),
            )
        ]
        assert all(
            slot.active is None
            for candidate_id, slot in registry._slots.items()
            if candidate_id != provider_id
        )
    finally:
        await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_effective_legacy_snapshot_matches_the_admitted_exact_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[OpenAISpeechRequest] = []

    async def audio() -> AsyncIterator[bytes]:
        yield b"audio"

    def capture_generate(
        _host: LegacyBackendHost,
        _internal_model_id: str,
        request: OpenAISpeechRequest,
        _progress_sink: ProgressSink | None,
    ) -> AsyncIterator[bytes]:
        captured.append(request)
        return audio()

    monkeypatch.setattr(LegacyBackendHost, "generate", capture_generate)
    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(
            {},
            manager_factory=lambda _provider, _config: pytest.fail(
                "request admission must stop at the legacy adapter boundary"
            ),
        ),
        aliases={},
    )
    model_id = "eleven_multilingual_v2"
    voice_id = "AZnzlk1XvdvUeBnXmlld"
    service = _test_service(
        registry,
        preferences_snapshot=_snapshot(
            provider_id="elevenlabs",
            model_id=model_id,
            voice_mode="exact",
            voice_id=voice_id,
            response_format="wav",
        ),
    )
    response: TTSAudioResponse | None = None

    try:
        response, selection = await service.synthesize_effective(
            text="Preserve exact values."
        )

        assert selection.model_id == model_id
        assert selection.voice_id == voice_id
        assert selection.response_format == "wav"
        assert captured == [
            OpenAISpeechRequest(
                model=model_id,
                input="Preserve exact values.",
                voice=voice_id,
                response_format="wav",
                speed=1.0,
            )
        ]
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_supported_studio_options_reach_the_legacy_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[OpenAISpeechRequest] = []

    async def audio() -> AsyncIterator[bytes]:
        yield b"audio"

    def capture_generate(
        _host: LegacyBackendHost,
        _internal_model_id: str,
        request: OpenAISpeechRequest,
        _progress_sink: ProgressSink | None,
    ) -> AsyncIterator[bytes]:
        captured.append(request)
        return audio()

    monkeypatch.setattr(LegacyBackendHost, "generate", capture_generate)
    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(
            {},
            manager_factory=lambda _provider, _config: pytest.fail(
                "request admission must stop at the legacy adapter boundary"
            ),
        ),
        aliases={},
    )
    saved = StudioTTSPreferencesSnapshot(
        revision=2,
        auto_play=True,
        provider_options={"chatterbox": {"exaggeration": 0.8, "cfg_weight": 0.3}},
    )
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_options={
                "temperature": 1.2,
                "num_candidates": 3,
                "validate_with_whisper": True,
            }
        ),
        base_revision=2,
    )
    service = _test_service(
        registry,
        preferences_snapshot=_snapshot(
            provider_id="chatterbox",
            model_id="chatterbox",
            voice_mode="exact",
            voice_id="default",
            response_format="wav",
        ),
        studio_preferences_loader=lambda: saved,
    )
    response: TTSAudioResponse | None = None

    try:
        response, selection = await service.synthesize_effective(
            text="Studio response",
            studio_draft=draft,
            studio_preferences=saved,
        )

        assert dict(selection.provider_options) == {
            "exaggeration": 0.8,
            "cfg_weight": 0.3,
            "temperature": 1.2,
            "num_candidates": 3,
            "validate_with_whisper": True,
        }
        assert captured == [
            OpenAISpeechRequest(
                model="chatterbox",
                input="Studio response",
                voice="default",
                response_format="wav",
                speed=1.0,
                extra_params={
                    "temperature": 1.2,
                    "num_candidates": 3,
                    "validate_with_whisper": True,
                    "exaggeration": 0.8,
                    "cfg_weight": 0.3,
                },
            )
        ]
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()
