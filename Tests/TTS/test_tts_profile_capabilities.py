from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import pytest

from Tests.TTS.adapter_fakes import FakeAdapter
from tldw_chatbook.TTS import (
    CapabilitySnapshotState,
    ProviderHealth,
    TTSModelInfo,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSService,
    TTSStructuredVoiceAdapter,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.adapter_registry import (
    TTSAdapterLease,
    TTSAdapterRegistry,
)
from tldw_chatbook.TTS.adapter_types import TTSProviderSpec
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot

_WAIT_SECONDS = 1.0


def _model(model_id: str) -> TTSModelInfo:
    return TTSModelInfo(
        model_id=model_id,
        display_name=model_id,
        family="test",
        upstream_mode="tts",
        formats=("wav",),
        voices=(),
        supports_speed=False,
        omit_voice_uses_server_default=True,
    )


def _catalog(
    revision: int,
    model_ids: tuple[str, ...],
    *,
    fresh: bool = True,
    health_state: str = "available",
) -> TTSProviderCatalog:
    return TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=revision,
        health=ProviderHealth(
            state=health_state,  # type: ignore[arg-type]
            fresh=fresh,
        ),
        models=tuple(_model(model_id) for model_id in model_ids),
    )


class _CapabilityAdapter(FakeAdapter):
    def __init__(
        self,
        model_ids: tuple[str, ...],
        *,
        revisions: tuple[int, ...] = (1,),
        freshness: tuple[bool, ...] = (True,),
        health_states: tuple[str, ...] = ("available",),
    ) -> None:
        super().__init__("audio_cpp")
        self.model_ids = model_ids
        self.revisions = revisions
        self.freshness = freshness
        self.health_states = health_states
        self.catalog_calls = 0
        self.voice_calls: list[str] = []
        self.active_voice_calls = 0
        self.max_active_voice_calls = 0
        self.catalog_started = asyncio.Event()
        self.catalog_release: asyncio.Event | None = None
        self.voice_release: asyncio.Event | None = None
        self.four_voice_calls_started = asyncio.Event()
        self.observed_revision = revisions[0]

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        assert refresh is True
        self.catalog_started.set()
        if self.catalog_release is not None:
            await self.catalog_release.wait()
        revision = self.revisions[min(self.catalog_calls, len(self.revisions) - 1)]
        fresh = self.freshness[min(self.catalog_calls, len(self.freshness) - 1)]
        health_state = self.health_states[
            min(self.catalog_calls, len(self.health_states) - 1)
        ]
        self.catalog_calls += 1
        self.observed_revision = revision
        return _catalog(
            revision,
            self.model_ids,
            fresh=fresh,
            health_state=health_state,
        )

    async def observe_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        assert refresh is True
        self.voice_calls.append(model_id)
        self.active_voice_calls += 1
        self.max_active_voice_calls = max(
            self.max_active_voice_calls,
            self.active_voice_calls,
        )
        if self.active_voice_calls == 4:
            self.four_voice_calls_started.set()
        try:
            if self.voice_release is not None:
                await self.voice_release.wait()
            state = "complete" if model_id in self.model_ids else "model_missing"
            return TTSVoiceDiscoveryResult(
                provider_id="audio_cpp",
                model_id=model_id,
                catalog_revision=self.observed_revision,
                voices=(f"{model_id}/voice",) if state == "complete" else (),
                state=state,
            )
        finally:
            self.active_voice_calls -= 1


class _ExceptionalSiblingAdapter(_CapabilityAdapter):
    def __init__(self, *, raise_one: bool) -> None:
        super().__init__(tuple(f"model-{number}" for number in range(4)))
        self.raise_one = raise_one
        self.all_started = asyncio.Event()
        self.failure_raised = asyncio.Event()
        self.cancellation_seen = asyncio.Event()
        self.all_sibling_cancellations_seen = asyncio.Event()
        self.release_siblings = asyncio.Event()
        self.release_finalizers = asyncio.Event()
        self.active = 0
        self.finalized = 0
        self.cancelled = 0

    async def observe_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        assert refresh is True
        self.voice_calls.append(model_id)
        self.active += 1
        if self.active == 4:
            self.all_started.set()
        try:
            await self.all_started.wait()
            if self.raise_one and model_id == "model-0":
                self.failure_raised.set()
                raise RuntimeError("synthetic voice failure")
            try:
                await self.release_siblings.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                self.cancellation_seen.set()
                if self.cancelled == 3:
                    self.all_sibling_cancellations_seen.set()
                await self.release_finalizers.wait()
                raise
            return TTSVoiceDiscoveryResult(
                provider_id="audio_cpp",
                model_id=model_id,
                catalog_revision=1,
                voices=(f"{model_id}/voice",),
                state="complete",
            )
        finally:
            self.active -= 1
            self.finalized += 1


class _PhasedDeadlineAdapter(_CapabilityAdapter):
    def __init__(self) -> None:
        super().__init__(("model",), revisions=(1, 2, 2))
        self.catalog_entries = 0

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        self.catalog_entries += 1
        await asyncio.sleep(0.04)
        return await super().get_catalog(refresh=refresh)

    async def observe_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        await asyncio.sleep(0.04)
        return await super().observe_voices(model_id, refresh=refresh)


class _RecordingRegistry(TTSAdapterRegistry):
    def __init__(self, adapter: FakeAdapter) -> None:
        super().__init__(
            specs=(
                TTSProviderSpec(
                    descriptor=TTSProviderDescriptor(
                        provider_id="audio_cpp",
                        display_name="audio.cpp",
                        native=True,
                    ),
                    factory=lambda _config: adapter,
                    initial_config={},
                    exclusive_reconfigure=True,
                ),
            ),
            aliases={},
        )
        self.expected_revisions: list[tuple[str, int | None]] = []
        self.release_started = asyncio.Event()
        self.release_allowed: asyncio.Event | None = None

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

    async def _release(self, slot: Any, record: Any) -> None:
        self.release_started.set()
        if self.release_allowed is not None:
            await self.release_allowed.wait()
        await super()._release(slot, record)


def _service(
    adapter: FakeAdapter,
) -> tuple[TTSService, _RecordingRegistry]:
    registry = _RecordingRegistry(adapter)
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="model",
            voice_mode="server_default",
            voice_id=None,
            response_format="wav",
            speed=1.0,
        ),
    )
    return service, registry


async def _close_service(service: TTSService) -> None:
    await service.close()
    await service.wait_closed()


def test_capability_snapshot_is_frozen_and_copies_voice_results() -> None:
    catalog = _catalog(7, ("model",))
    voice = TTSVoiceDiscoveryResult(
        provider_id="audio_cpp",
        model_id="model",
        catalog_revision=7,
        voices=("voice",),
        state="complete",
    )
    source = {"model": voice}

    snapshot = TTSNativeCapabilitySnapshot(
        provider_id="audio_cpp",
        configuration_revision=3,
        state="complete",
        catalog=catalog,
        voice_results=source,
    )
    source.clear()

    assert snapshot.voice_results == {"model": voice}
    assert isinstance(snapshot.voice_results, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot.voice_results["other"] = voice  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        snapshot.state = "unverified"  # type: ignore[misc]


@pytest.mark.parametrize(
    "updates",
    (
        {"provider_id": "audio.cpp"},
        {"configuration_revision": True},
        {"configuration_revision": -1},
        {"state": "unknown"},
        {"catalog": None},
        {"voice_results": []},
        {
            "voice_results": {
                "other": TTSVoiceDiscoveryResult(
                    provider_id="audio_cpp",
                    model_id="model",
                    catalog_revision=7,
                    voices=(),
                    state="complete",
                )
            }
        },
        {
            "voice_results": {
                "model": TTSVoiceDiscoveryResult(
                    provider_id="audio_cpp",
                    model_id="model",
                    catalog_revision=8,
                    voices=(),
                    state="complete",
                )
            }
        },
        {
            "voice_results": {
                "model": TTSVoiceDiscoveryResult(
                    provider_id="audio_cpp",
                    model_id="model",
                    catalog_revision=7,
                    voices=(),
                    state="unverified",
                )
            }
        },
    ),
)
def test_complete_capability_snapshot_rejects_incoherent_state(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "provider_id": "audio_cpp",
        "configuration_revision": 3,
        "state": "complete",
        "catalog": _catalog(7, ("model",)),
        "voice_results": {
            "model": TTSVoiceDiscoveryResult(
                provider_id="audio_cpp",
                model_id="model",
                catalog_revision=7,
                voices=(),
                state="complete",
            )
        },
    }
    values.update(updates)

    with pytest.raises((TypeError, ValueError)):
        TTSNativeCapabilitySnapshot(**values)  # type: ignore[arg-type]


def test_unverified_capability_snapshot_may_retain_partial_diagnostics() -> None:
    result = TTSVoiceDiscoveryResult(
        provider_id="audio_cpp",
        model_id="model",
        catalog_revision=5,
        voices=("partial",),
        state="unverified",
    )

    snapshot = TTSNativeCapabilitySnapshot(
        provider_id="audio_cpp",
        configuration_revision=2,
        state="unverified",
        catalog=_catalog(6, ("model",)),
        voice_results={"model": result},
    )

    assert snapshot.voice_results == {"model": result}


@pytest.mark.parametrize(
    "catalog",
    (
        None,
        _catalog(6, ("model",)),
    ),
)
@pytest.mark.parametrize("state", ("complete", "model_missing"))
def test_unverified_snapshot_rejects_stale_authoritative_children(
    catalog: TTSProviderCatalog | None,
    state: str,
) -> None:
    result = TTSVoiceDiscoveryResult(
        provider_id="audio_cpp",
        model_id="model",
        catalog_revision=5,
        voices=("voice",) if state == "complete" else (),
        state=state,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError):
        TTSNativeCapabilitySnapshot(
            provider_id="audio_cpp",
            configuration_revision=2,
            state="unverified",
            catalog=catalog,
            voice_results={"model": result},
        )


def test_capability_snapshot_state_literal_is_public() -> None:
    from typing import get_args

    assert get_args(CapabilitySnapshotState) == ("complete", "unverified")


@pytest.mark.asyncio
async def test_capability_lease_is_revision_matched_and_gate_exits_before_network() -> (
    None
):
    adapter = _CapabilityAdapter(("model",))
    adapter.catalog_release = asyncio.Event()
    service, registry = _service(adapter)
    observation = asyncio.create_task(
        service.get_native_capability_snapshot("audio_cpp", {"model"})
    )
    await asyncio.wait_for(adapter.catalog_started.wait(), timeout=_WAIT_SECONDS)
    writer_entered = asyncio.Event()

    async def enter_writer() -> None:
        async with service._request_admission._gate.write():
            writer_entered.set()

    writer = asyncio.create_task(enter_writer())
    try:
        await asyncio.wait_for(writer_entered.wait(), timeout=_WAIT_SECONDS)
        assert registry.expected_revisions == [("audio_cpp", 1)]
        assert registry._total_leases() == 1
        adapter.catalog_release.set()
        snapshot = await asyncio.wait_for(observation, timeout=_WAIT_SECONDS)
        assert snapshot.configuration_revision == 1
        assert not isinstance(snapshot, (TTSAdapterLease, TTSStructuredVoiceAdapter))
        assert not hasattr(snapshot, "adapter")
        assert not hasattr(snapshot, "lease")
    finally:
        adapter.catalog_release.set()
        await asyncio.gather(observation, writer, return_exceptions=True)
        await _close_service(service)


@pytest.mark.asyncio
async def test_capability_snapshot_deduplicates_models_and_limits_voice_concurrency() -> (
    None
):
    model_ids = tuple(f"model-{number}" for number in range(6))
    adapter = _CapabilityAdapter(model_ids)
    adapter.voice_release = asyncio.Event()
    service, registry = _service(adapter)
    observation = asyncio.create_task(
        service.get_native_capability_snapshot(
            "audio_cpp",
            [*model_ids, "model-0", "model-1"],
        )
    )
    try:
        await asyncio.wait_for(
            adapter.four_voice_calls_started.wait(),
            timeout=_WAIT_SECONDS,
        )
        await asyncio.sleep(0)
        assert adapter.max_active_voice_calls == 4
        adapter.voice_release.set()
        snapshot = await asyncio.wait_for(observation, timeout=_WAIT_SECONDS)
        assert snapshot.state == "complete"
        assert set(adapter.voice_calls) == set(model_ids)
        assert len(adapter.voice_calls) == len(model_ids)
        assert set(snapshot.voice_results) == set(model_ids)
        assert registry._total_leases() == 0
    finally:
        adapter.voice_release.set()
        await asyncio.gather(observation, return_exceptions=True)
        await _close_service(service)


@pytest.mark.asyncio
async def test_voice_failure_cancels_and_joins_siblings_before_releasing_lease() -> (
    None
):
    adapter = _ExceptionalSiblingAdapter(raise_one=True)
    service, registry = _service(adapter)
    observation = asyncio.create_task(
        service.get_native_capability_snapshot(
            "audio_cpp",
            adapter.model_ids,
        )
    )
    try:
        await asyncio.wait_for(adapter.failure_raised.wait(), timeout=_WAIT_SECONDS)
        await asyncio.wait_for(
            adapter.all_sibling_cancellations_seen.wait(),
            timeout=_WAIT_SECONDS,
        )

        assert not observation.done()
        assert not registry.release_started.is_set()
        assert adapter.active == 3
        assert adapter.cancelled == 3

        adapter.release_finalizers.set()
        snapshot = await asyncio.wait_for(observation, timeout=_WAIT_SECONDS)

        assert snapshot.state == "unverified"
        assert snapshot.voice_results == {}
        assert adapter.active == 0
        assert adapter.finalized == 4
        assert registry._total_leases() == 0
    finally:
        adapter.release_siblings.set()
        adapter.release_finalizers.set()
        await asyncio.gather(observation, return_exceptions=True)
        await _close_service(service)


@pytest.mark.asyncio
async def test_repeated_caller_cancellation_cannot_bypass_voice_task_cleanup() -> None:
    adapter = _ExceptionalSiblingAdapter(raise_one=False)
    service, registry = _service(adapter)
    observation = asyncio.create_task(
        service.get_native_capability_snapshot(
            "audio_cpp",
            adapter.model_ids,
        )
    )
    await asyncio.wait_for(adapter.all_started.wait(), timeout=_WAIT_SECONDS)

    observation.cancel("first capability cancellation")
    await asyncio.wait_for(adapter.cancellation_seen.wait(), timeout=_WAIT_SECONDS)
    observation.cancel("second capability cancellation")
    await asyncio.sleep(0)

    assert not observation.done()
    assert not registry.release_started.is_set()
    assert adapter.active > 0

    adapter.release_finalizers.set()
    with pytest.raises(asyncio.CancelledError):
        await observation
    assert adapter.active == 0
    assert adapter.finalized == 4
    assert registry._total_leases() == 0
    await _close_service(service)


@pytest.mark.asyncio
async def test_server_default_only_snapshot_performs_no_voice_observation() -> None:
    adapter = _CapabilityAdapter(("model",))
    service, registry = _service(adapter)
    try:
        snapshot = await service.get_native_capability_snapshot("audio_cpp", ())

        assert snapshot.state == "complete"
        assert snapshot.voice_results == {}
        assert adapter.voice_calls == []
        assert registry.expected_revisions == [("audio_cpp", 1)]
    finally:
        await _close_service(service)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_ids", ((), ("model",)))
async def test_stale_initial_catalog_is_unverified_without_voice_observation(
    model_ids: tuple[str, ...],
) -> None:
    adapter = _CapabilityAdapter(("model",), freshness=(False,))
    service, registry = _service(adapter)
    try:
        snapshot = await service.get_native_capability_snapshot(
            "audio_cpp",
            model_ids,
        )

        assert snapshot.state == "unverified"
        assert snapshot.catalog is not None
        assert snapshot.catalog.health.fresh is False
        assert snapshot.voice_results == {}
        assert adapter.voice_calls == []
        assert registry._total_leases() == 0
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_catalog_becoming_stale_clears_authoritative_voice_results() -> None:
    adapter = _CapabilityAdapter(
        ("model",),
        revisions=(1, 1),
        freshness=(True, False),
    )
    service, _registry = _service(adapter)
    try:
        snapshot = await service.get_native_capability_snapshot(
            "audio_cpp",
            ("model",),
        )

        assert snapshot.state == "unverified"
        assert snapshot.catalog is not None
        assert snapshot.catalog.health.fresh is False
        assert snapshot.voice_results == {}
        assert adapter.voice_calls == ["model"]
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_fresh_not_configured_catalog_remains_authoritative() -> None:
    adapter = _CapabilityAdapter(
        (),
        freshness=(True,),
        health_states=("not_configured",),
    )
    service, _registry = _service(adapter)
    try:
        snapshot = await service.get_native_capability_snapshot("audio_cpp", ())

        assert snapshot.state == "complete"
        assert snapshot.catalog is not None
        assert snapshot.catalog.health.state == "not_configured"
        assert snapshot.catalog.health.fresh is True
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_capability_snapshot_uses_one_aggregate_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.TTS_Generation as generation_module

    adapter = _CapabilityAdapter(tuple(f"model-{number}" for number in range(5)))
    adapter.voice_release = asyncio.Event()
    service, registry = _service(adapter)
    monkeypatch.setattr(
        generation_module,
        "_NATIVE_CAPABILITY_TIMEOUT_SECONDS",
        0.03,
    )
    started = time.monotonic()
    try:
        snapshot = await service.get_native_capability_snapshot(
            "audio_cpp",
            adapter.model_ids,
        )
        elapsed = time.monotonic() - started

        assert snapshot.state == "unverified"
        assert elapsed < 0.2
        assert adapter.max_active_voice_calls == 4
        assert registry._total_leases() == 0
    finally:
        adapter.voice_release.set()
        await _close_service(service)


@pytest.mark.asyncio
async def test_capability_deadline_is_established_once_across_all_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.TTS.TTS_Generation as generation_module

    adapter = _PhasedDeadlineAdapter()
    service, registry = _service(adapter)
    deadline_calls = 0

    def one_deadline() -> float:
        nonlocal deadline_calls
        deadline_calls += 1
        return asyncio.get_running_loop().time() + 0.14

    monkeypatch.setattr(
        generation_module,
        "_native_capability_deadline",
        one_deadline,
    )
    started = time.monotonic()
    try:
        snapshot = await service.get_native_capability_snapshot(
            "audio_cpp",
            ("model",),
        )
        elapsed = time.monotonic() - started

        assert snapshot.state == "unverified"
        assert snapshot.voice_results == {}
        assert deadline_calls == 1
        assert adapter.catalog_entries == 3
        assert adapter.active_voice_calls == 0
        assert elapsed < 0.3
        assert registry._total_leases() == 0
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_one_catalog_advance_retries_once_and_returns_one_revision() -> None:
    adapter = _CapabilityAdapter(
        ("model",),
        revisions=(1, 2, 2, 2),
    )
    service, _registry = _service(adapter)
    try:
        snapshot = await service.get_native_capability_snapshot(
            "audio_cpp",
            ("model",),
        )

        assert snapshot.state == "complete"
        assert snapshot.catalog is not None
        assert snapshot.catalog.revision == 2
        assert snapshot.voice_results["model"].catalog_revision == 2
        assert adapter.catalog_calls == 4
        assert adapter.voice_calls == ["model", "model"]
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_second_catalog_advance_returns_unverified_without_mixed_authority() -> (
    None
):
    adapter = _CapabilityAdapter(
        ("model",),
        revisions=(1, 2, 2, 3),
    )
    service, _registry = _service(adapter)
    try:
        snapshot = await service.get_native_capability_snapshot(
            "audio_cpp",
            ("model",),
        )

        assert snapshot.state == "unverified"
        assert adapter.catalog_calls == 4
        assert adapter.voice_calls == ["model", "model"]
        assert snapshot.catalog is not None
        assert snapshot.catalog.revision == 3
        assert snapshot.voice_results == {}
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_service_shutdown_during_observation_returns_only_unverified() -> None:
    adapter = _CapabilityAdapter(("model",))
    adapter.catalog_release = asyncio.Event()
    service, registry = _service(adapter)
    observation = asyncio.create_task(
        service.get_native_capability_snapshot("audio_cpp", ("model",))
    )
    await asyncio.wait_for(adapter.catalog_started.wait(), timeout=_WAIT_SECONDS)
    close_task = asyncio.create_task(service.close())
    await asyncio.wait_for(service._close_signal.wait(), timeout=_WAIT_SECONDS)
    try:
        adapter.catalog_release.set()
        snapshot, _ = await asyncio.wait_for(
            asyncio.gather(observation, close_task),
            timeout=_WAIT_SECONDS,
        )

        assert snapshot.state == "unverified"
        assert snapshot.voice_results == {}
        assert registry._total_leases() == 0
        assert observation.done()
        assert close_task.done()
        await service.wait_closed()
    finally:
        adapter.catalog_release.set()
        await asyncio.gather(observation, close_task, return_exceptions=True)
        await service.wait_closed()


@pytest.mark.asyncio
async def test_cancellation_propagates_only_after_capability_lease_release() -> None:
    adapter = _CapabilityAdapter(("model",))
    adapter.catalog_release = asyncio.Event()
    service, registry = _service(adapter)
    registry.release_allowed = asyncio.Event()
    observation = asyncio.create_task(
        service.get_native_capability_snapshot("audio_cpp", ("model",))
    )
    await asyncio.wait_for(adapter.catalog_started.wait(), timeout=_WAIT_SECONDS)

    observation.cancel("capability caller cancelled")
    await asyncio.wait_for(registry.release_started.wait(), timeout=_WAIT_SECONDS)
    await asyncio.sleep(0)
    assert not observation.done()
    assert registry._total_leases() == 1

    registry.release_allowed.set()
    with pytest.raises(asyncio.CancelledError) as cancellation:
        await observation
    assert cancellation.value.args == ("capability caller cancelled",)
    assert registry._total_leases() == 0
    await _close_service(service)


@pytest.mark.asyncio
async def test_missing_structured_voice_capability_fails_safe_as_unverified() -> None:
    adapter = FakeAdapter("audio_cpp")
    service, registry = _service(adapter)
    try:
        snapshot = await service.get_native_capability_snapshot(
            "audio_cpp",
            ("model",),
        )

        assert snapshot.state == "unverified"
        assert snapshot.voice_results == {}
        assert registry._total_leases() == 0
    finally:
        await _close_service(service)
