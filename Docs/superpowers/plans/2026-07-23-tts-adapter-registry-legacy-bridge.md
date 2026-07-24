# TTS Adapter Registry and Legacy Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one application-owned, sealed TTS adapter registry authoritative while preserving all existing TTS callers and providers through a temporary legacy bridge.

**Architecture:** Add provider-neutral request, response, catalog, health, and progress contracts in front of an exact-ID registry. The registry lazily owns adapters and keeps operation leases alive through response consumption; six provider-scoped legacy adapters quarantine the existing wildcard manager until later native migrations. `TTSService` becomes the only synthesis entry point, while its existing byte-generator signature remains as a compatibility facade.

**Tech Stack:** Python 3.11+, asyncio, standard-library dataclasses and protocols, existing Pydantic `OpenAISpeechRequest`, existing `TTSBackendManager`, pytest, pytest-asyncio.

## Global Constraints

- This plan implements only delivery slice 1 from the approved design: registry authority, legacy containment, compatibility generation, and progress delivery.
- Do not implement audio.cpp HTTP discovery, synthesis, configuration, process supervision, or STTS catalog-driven controls in TASK-402.
- Canonical provider IDs are exactly `openai`, `elevenlabs`, `kokoro`, `chatterbox`, `higgs`, and `alltalk` for this slice; `audio_cpp` is added by the next ordered slice.
- The initial provider alias map is empty. Display labels never determine provider identity.
- Runtime provider registration is sealed. The retained class-global wildcard registry is private to the legacy bridge and closed to new providers.
- Adapter construction and legacy backend/model loading remain lazy.
- The default service concurrency limit remains four and is instance-scoped.
- Existing callers retain `generate_audio_stream(request: OpenAISpeechRequest, internal_model_id: str)`.
- Progress callbacks are operation-scoped. Sink exceptions are isolated and never fail synthesis.
- Tests verify lazy materialization and lease release through observable
  factory, retirement, and cleanup effects; do not add production
  introspection methods used only by tests.
- Architecture-boundary tests parse Python syntax rather than matching raw
  source text. The final independent `rg` check remains required.
- New registry, bridge, and service diagnostics may not log configuration
  values or synthesis text; TASK-402 also removes the existing OpenAI
  API-key-fragment disclosure.
- No new dependency is added.
- ADR required: yes
- ADR path: `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`
- Reason: ADR-023 already governs the provider boundary, lifecycle, compatibility bridge, and future native-adapter migration.

## File Structure

### New production files

- `tldw_chatbook/TTS/adapter_types.py` — provider-neutral DTOs, adapter protocol, response cleanup, provider definitions, and public registry errors.
- `tldw_chatbook/TTS/adapter_registry.py` — exact provider lookup, lazy materialization, operation leases, targeted reconfiguration, retirement, and shutdown.
- `tldw_chatbook/TTS/legacy_catalogs.py` — approximate compatibility metadata for the six retained providers.
- `tldw_chatbook/TTS/legacy_bridge.py` — enumerated legacy routing, provider-scoped hosts, progress translation, and six adapter factories.
- `tldw_chatbook/TTS/adapter_bootstrap.py` — construction of the default six-provider registry and application TTS service.

### Modified production files

- `tldw_chatbook/TTS/TTS_Backends.py` — quarantine and deterministically reset the old class registry; preserve the existing manager only for legacy hosts.
- `tldw_chatbook/TTS/TTS_Generation.py` — make `TTSService` registry-backed, retain the compatibility generator, and replace first-config singleton construction with explicit binding.
- `tldw_chatbook/TTS/__init__.py` — export the new public service contracts without exporting legacy hosts or concrete adapters.
- `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py` — pass progress through `TTSService` instead of retrieving a concrete backend.
- `tldw_chatbook/app.py` — construct, bind, close, and reset the app-owned service exactly once; delete direct backend-manager shutdown inspection.
- `tldw_chatbook/TTS/backends/openai.py` — remove the existing API-key-prefix diagnostic.
- `Docs/Development/TTS/TTS_MODULE_GUIDE.md` — document the authoritative registry, compatibility API, ownership, and bridge deletion boundary.

### New tests

- `Tests/TTS/adapter_fakes.py` — focused fake adapters and async streams shared by registry/service tests.
- `Tests/TTS/test_adapter_types.py` — response context/cleanup semantics and immutable request options.
- `Tests/TTS/test_adapter_registry.py` — exact lookup, sealing, lazy concurrency, leases, retirement, reconfiguration, and shutdown.
- `Tests/TTS/test_legacy_bridge.py` — route table, provider isolation, per-backend locking, progress, catalog, and close behavior.
- `Tests/TTS/test_tts_registry_service.py` — native service path, compatibility generator, progress isolation, and explicit binding.
- `Tests/TTS/test_tts_app_ownership.py` — application construction/binding/teardown and forbidden concrete-manager access.
- `Tests/TTS/test_tts_logging_privacy.py` — API-key-fragment regression.

---

### Task 1: Define provider-neutral adapter contracts

**Files:**
- Create: `tldw_chatbook/TTS/adapter_types.py`
- Create: `Tests/TTS/adapter_fakes.py`
- Create: `Tests/TTS/test_adapter_types.py`

**Interfaces:**
- Consumes: `collections.abc.AsyncIterator`, `Awaitable`, `Callable`, and `Mapping`.
- Produces: `TTSRequest`, `TTSAudioResponse`, `TTSProgress`, `TTSModelInfo`, `TTSProviderCatalog`, `ProviderHealth`, `TTSProviderDescriptor`, `TTSAdapter`, `TTSProviderSpec`, `ProgressSink`, `UnknownTTSProviderError`, `TTSRegistryClosedError`, and `TTSProviderReconfiguringError`.

- [ ] **Step 1: Write response-lifetime and request-immutability tests**

```python
# Tests/TTS/test_adapter_types.py
import pytest

from tldw_chatbook.TTS.adapter_types import TTSAudioResponse, TTSRequest


@pytest.mark.asyncio
async def test_audio_response_closes_stream_and_callbacks_once() -> None:
    events: list[str] = []

    async def stream():
        try:
            yield b"first"
            yield b"second"
        finally:
            events.append("stream")

    async def cleanup() -> None:
        events.append("cleanup")

    response = TTSAudioResponse(
        provider_id="openai",
        model_id="tts-1",
        audio_format="mp3",
        content_type="audio/mpeg",
        byte_stream=stream(),
        cleanup=cleanup,
    )
    assert await anext(response.byte_stream) == b"first"

    await response.aclose()
    await response.aclose()

    assert events == ["stream", "cleanup"]


@pytest.mark.asyncio
async def test_audio_response_context_manager_closes_after_consumer_failure() -> None:
    closed = False

    async def stream():
        nonlocal closed
        try:
            yield b"audio"
        finally:
            closed = True

    with pytest.raises(RuntimeError, match="consumer"):
        async with TTSAudioResponse(
            provider_id="openai",
            model_id="tts-1",
            audio_format="mp3",
            content_type="audio/mpeg",
            byte_stream=stream(),
        ) as response:
            assert await anext(response.byte_stream) == b"audio"
            raise RuntimeError("consumer")

    assert closed is True


def test_tts_request_copies_options_at_the_boundary() -> None:
    source = {"temperature": 0.5}
    request = TTSRequest(
        provider_id="chatterbox",
        model_id="chatterbox",
        text="hello",
        voice="default",
        response_format="wav",
        speed=1.0,
        options=source,
    )
    source["temperature"] = 1.0

    assert request.options == {"temperature": 0.5}
    with pytest.raises(TypeError):
        request.options["temperature"] = 0.2  # type: ignore[index]
```

- [ ] **Step 2: Run the contract tests and confirm the module is absent**

Run: `pytest Tests/TTS/test_adapter_types.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'tldw_chatbook.TTS.adapter_types'`.

- [ ] **Step 3: Implement the contracts and idempotent response cleanup**

```python
# tldw_chatbook/TTS/adapter_types.py
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

ProgressSink = Callable[["TTSProgress"], Awaitable[None]]
CleanupCallback = Callable[[], Awaitable[None]]
ProviderFactory = Callable[[Mapping[str, Any]], "TTSAdapter"]
ProviderState = Literal[
    "available", "unavailable", "not_configured", "reconfiguring", "closed"
]


class UnknownTTSProviderError(LookupError):
    """Raised when an exact canonical provider ID is not registered."""


class TTSRegistryClosedError(RuntimeError):
    """Raised when a registry no longer admits operations."""


class TTSProviderReconfiguringError(RuntimeError):
    """Raised when an exclusive provider handoff blocks new operations."""


@dataclass(frozen=True, slots=True)
class TTSRequest:
    provider_id: str
    model_id: str
    text: str
    voice: str | None
    response_format: str
    speed: float = 1.0
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "options", MappingProxyType(dict(self.options))
        )


@dataclass(frozen=True, slots=True)
class TTSProgress:
    status: str
    fraction: float | None = None
    processed: int | None = None
    total: int | None = None
    metrics: Mapping[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    state: ProviderState
    fresh: bool
    diagnostic: str | None = None
    retryable: bool = False
    recovery_action: str | None = None


@dataclass(frozen=True, slots=True)
class TTSModelInfo:
    model_id: str
    display_name: str
    family: str
    upstream_mode: str
    formats: tuple[str, ...]
    voices: tuple[str, ...]
    supports_speed: bool
    supports_options: tuple[str, ...] = ()
    omit_voice_uses_server_default: bool = False


@dataclass(frozen=True, slots=True)
class TTSProviderCatalog:
    provider_id: str
    revision: int
    health: ProviderHealth
    models: tuple[TTSModelInfo, ...]
    approximate: bool = False


@dataclass(frozen=True, slots=True)
class TTSProviderDescriptor:
    provider_id: str
    display_name: str
    native: bool


class TTSAudioResponse:
    def __init__(
        self,
        *,
        provider_id: str,
        model_id: str,
        audio_format: str,
        content_type: str,
        byte_stream: AsyncIterator[bytes],
        sample_rate: int | None = None,
        cleanup: CleanupCallback | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.audio_format = audio_format
        self.content_type = content_type
        self.byte_stream = byte_stream
        self.sample_rate = sample_rate
        self._cleanup_callbacks = [cleanup] if cleanup is not None else []
        self._close_lock = asyncio.Lock()
        self._closed = False

    def add_cleanup(self, callback: CleanupCallback) -> None:
        if self._closed:
            raise RuntimeError("Cannot add cleanup to a closed audio response")
        self._cleanup_callbacks.append(callback)

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            first_error: BaseException | None = None
            stream_close = getattr(self.byte_stream, "aclose", None)
            if callable(stream_close):
                try:
                    await stream_close()
                except BaseException as error:
                    first_error = error
            for callback in self._cleanup_callbacks:
                try:
                    await callback()
                except BaseException as error:
                    first_error = first_error or error
            if first_error is not None:
                raise first_error

    async def __aenter__(self) -> "TTSAudioResponse":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()


class TTSAdapter(Protocol):
    async def ensure_ready(self) -> None:
        raise NotImplementedError

    async def get_catalog(
        self, refresh: bool = False
    ) -> TTSProviderCatalog:
        raise NotImplementedError

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TTSProviderSpec:
    descriptor: TTSProviderDescriptor
    factory: ProviderFactory
    initial_config: Mapping[str, Any]
    exclusive_reconfigure: bool = False
```

Create the shared fakes with these exact public helpers:

```python
# Tests/TTS/adapter_fakes.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    ProgressSink,
    TTSAudioResponse,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderSpec,
    TTSRequest,
)
from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
    TTSAdapterRegistry,
)


class FakeAdapter:
    def __init__(
        self,
        provider_id: str,
        *,
        chunks: tuple[bytes, ...] = (b"audio",),
        close_order: list[str] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.chunks = chunks
        self.close_order = close_order
        self.ensure_ready_calls = 0
        self.synthesize_calls = 0
        self.close_calls = 0
        self.response_close_calls = 0

    async def ensure_ready(self) -> None:
        self.ensure_ready_calls += 1

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        del refresh
        return TTSProviderCatalog(
            provider_id=self.provider_id,
            revision=1,
            health=ProviderHealth(state="available", fresh=True),
            models=(
                TTSModelInfo(
                    model_id="model",
                    display_name="Model",
                    family="fake",
                    upstream_mode="tts",
                    formats=("wav",),
                    voices=("default",),
                    supports_speed=True,
                ),
            ),
        )

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        self.synthesize_calls += 1
        if progress_sink is not None:
            from tldw_chatbook.TTS.adapter_types import TTSProgress

            await progress_sink(TTSProgress(status="Generating"))

        async def stream():
            yield_chunks = self.chunks
            for chunk in yield_chunks:
                yield chunk

        async def cleanup() -> None:
            self.response_close_calls += 1

        return TTSAudioResponse(
            provider_id=self.provider_id,
            model_id=request.model_id,
            audio_format=request.response_format,
            content_type="audio/wav",
            byte_stream=stream(),
            cleanup=cleanup,
        )

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_order is not None:
            self.close_order.append(self.provider_id)


class FakeAdapterFactory:
    def __init__(
        self,
        provider_id: str,
        *,
        close_order: list[str] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.close_order = close_order
        self.calls = 0
        self.instances: list[FakeAdapter] = []

    def __call__(self, config: Mapping[str, Any]) -> FakeAdapter:
        del config
        self.calls += 1
        adapter = FakeAdapter(
            self.provider_id, close_order=self.close_order
        )
        self.instances.append(adapter)
        return adapter


def provider_spec(
    provider_id: str,
    factory: FakeAdapterFactory,
    config: Mapping[str, Any] | None = None,
    *,
    exclusive: bool = False,
) -> TTSProviderSpec:
    return TTSProviderSpec(
        descriptor=TTSProviderDescriptor(
            provider_id=provider_id,
            display_name=provider_id,
            native=True,
        ),
        factory=factory,
        initial_config={} if config is None else config,
        exclusive_reconfigure=exclusive,
    )
```

- [ ] **Step 4: Run the contract tests**

Run: `pytest Tests/TTS/test_adapter_types.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the contract boundary**

```bash
git add tldw_chatbook/TTS/adapter_types.py Tests/TTS/adapter_fakes.py Tests/TTS/test_adapter_types.py
git commit -m "feat(tts): define adapter contracts"
```

---

### Task 2: Implement the sealed adapter registry and operation leases

**Files:**
- Create: `tldw_chatbook/TTS/adapter_registry.py`
- Create: `Tests/TTS/test_adapter_registry.py`

**Interfaces:**
- Consumes: `TTSAdapter`, `TTSProviderSpec`, and registry errors from Task 1.
- Produces: `TTSAdapterRegistry.descriptors()`, `acquire()`, `get_catalog()`, `reconfigure_provider()`, `close()`, `TTSAdapterLease`, and `ReconfigureResult`.

- [ ] **Step 1: Write exact-registration and concurrent-materialization tests**

```python
@pytest.mark.asyncio
async def test_registry_uses_exact_ids_and_materializes_once() -> None:
    factory = FakeAdapterFactory("openai")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("openai", factory),),
        aliases={},
    )

    leases = await asyncio.gather(
        *(registry.acquire("openai") for _ in range(20))
    )

    assert factory.calls == 1
    assert [item.provider_id for item in registry.descriptors()] == ["openai"]
    assert registry.aliases() == {}
    with pytest.raises(UnknownTTSProviderError):
        await registry.acquire("open")
    for lease in leases:
        await lease.release()
    await registry.close()


def test_registry_rejects_duplicate_ids_and_alias_collisions() -> None:
    factory = FakeAdapterFactory("openai")
    spec = provider_spec("openai", factory)
    with pytest.raises(ValueError, match="Duplicate provider"):
        TTSAdapterRegistry(specs=(spec, spec), aliases={})
    with pytest.raises(ValueError, match="Alias"):
        TTSAdapterRegistry(specs=(spec,), aliases={"openai": "openai"})
```

- [ ] **Step 2: Write lease, retirement, no-op, exclusive-handoff, and shutdown tests**

```python
@pytest.mark.asyncio
async def test_changed_config_retires_only_selected_adapter_after_lease() -> None:
    openai_factory = FakeAdapterFactory("openai")
    kokoro_factory = FakeAdapterFactory("kokoro")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec("openai", openai_factory, {"key": "first"}),
            provider_spec("kokoro", kokoro_factory, {"device": "cpu"}),
        ),
        aliases={},
    )
    old_openai = await registry.acquire("openai")
    kokoro = await registry.acquire("kokoro")

    assert await registry.reconfigure_provider(
        "openai", {"key": "second"}
    ) is ReconfigureResult.CHANGED
    assert registry.configuration_revision("openai") == 2
    assert registry.configuration_revision("kokoro") == 1
    assert old_openai.adapter.close_calls == 0
    assert kokoro.adapter.close_calls == 0

    await old_openai.release()
    assert old_openai.adapter.close_calls == 1
    replacement = await registry.acquire("openai")
    assert replacement.adapter is not old_openai.adapter
    await replacement.release()
    await kokoro.release()
    await registry.close()


@pytest.mark.asyncio
async def test_identical_config_is_a_no_op() -> None:
    factory = FakeAdapterFactory("openai")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("openai", factory, {"key": "same"}),),
        aliases={},
    )
    lease = await registry.acquire("openai")
    await lease.release()

    result = await registry.reconfigure_provider("openai", {"key": "same"})

    assert result is ReconfigureResult.UNCHANGED
    assert factory.calls == 1
    assert registry.configuration_revision("openai") == 1
    await registry.close()


@pytest.mark.asyncio
async def test_exclusive_reconfigure_blocks_until_old_lease_releases() -> None:
    factory = FakeAdapterFactory("exclusive")
    registry = TTSAdapterRegistry(
        specs=(provider_spec(
            "exclusive", factory, {"revision": 1}, exclusive=True
        ),),
        aliases={},
    )
    old_lease = await registry.acquire("exclusive")
    reconfigure = asyncio.create_task(
        registry.reconfigure_provider("exclusive", {"revision": 2})
    )
    await asyncio.sleep(0)

    with pytest.raises(TTSProviderReconfiguringError):
        await registry.acquire("exclusive")
    assert reconfigure.done() is False

    await old_lease.release()
    assert await reconfigure is ReconfigureResult.CHANGED
    assert old_lease.adapter.close_calls == 1
    await registry.close()


@pytest.mark.asyncio
async def test_shutdown_is_ordered_bounded_and_idempotent() -> None:
    close_order: list[str] = []
    openai_factory = FakeAdapterFactory("openai", close_order=close_order)
    kokoro_factory = FakeAdapterFactory("kokoro", close_order=close_order)
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec("openai", openai_factory),
            provider_spec("kokoro", kokoro_factory),
        ),
        aliases={},
        shutdown_timeout_seconds=0.01,
    )
    openai = await registry.acquire("openai")
    kokoro = await registry.acquire("kokoro")

    await registry.close()
    await registry.close()

    assert close_order == ["openai", "kokoro"]
    with pytest.raises(TTSRegistryClosedError):
        await registry.acquire("openai")
    await openai.release()
    await kokoro.release()
```

- [ ] **Step 3: Run the registry tests and confirm they fail**

Run: `pytest Tests/TTS/test_adapter_registry.py -q`

Expected: collection fails because `adapter_registry.py` does not exist.

- [ ] **Step 4: Implement registry state, exact resolution, and leases**

```python
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tldw_chatbook.TTS.adapter_types import (
    TTSAdapter,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderReconfiguringError,
    TTSProviderSpec,
    TTSRegistryClosedError,
    UnknownTTSProviderError,
)


class ReconfigureResult(Enum):
    UNCHANGED = "unchanged"
    CHANGED = "changed"


@dataclass(slots=True)
class _AdapterRecord:
    adapter: TTSAdapter
    leases: int = 0
    retired: bool = False
    closed: bool = False


@dataclass(slots=True)
class _ProviderSlot:
    spec: TTSProviderSpec
    config: dict[str, Any]
    revision: int = 1
    active: _AdapterRecord | None = None
    retired: list[_AdapterRecord] = field(default_factory=list)
    reconfiguring: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    transition_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lease_changed: asyncio.Event = field(default_factory=asyncio.Event)


class TTSAdapterLease:
    def __init__(
        self,
        provider_id: str,
        adapter: TTSAdapter,
        release_callback: Callable[[], Awaitable[None]],
    ) -> None:
        self.provider_id = provider_id
        self.adapter = adapter
        self._release_callback = release_callback
        self._released = False
        self._release_lock = asyncio.Lock()

    async def release(self) -> None:
        async with self._release_lock:
            if self._released:
                return
            self._released = True
            await self._release_callback()

    async def __aenter__(self) -> "TTSAdapterLease":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class TTSAdapterRegistry:
    def __init__(
        self,
        *,
        specs: Iterable[TTSProviderSpec],
        aliases: Mapping[str, str],
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        if shutdown_timeout_seconds < 0:
            raise ValueError("shutdown_timeout_seconds cannot be negative")
        self._slots: dict[str, _ProviderSlot] = {}
        for spec in tuple(specs):
            provider_id = spec.descriptor.provider_id
            if not provider_id:
                raise ValueError("Provider IDs cannot be empty")
            if provider_id in self._slots:
                raise ValueError(f"Duplicate provider: {provider_id}")
            self._slots[provider_id] = _ProviderSlot(
                spec=spec,
                config=deepcopy(dict(spec.initial_config)),
            )
        self._aliases = dict(aliases)
        for alias, target in self._aliases.items():
            if not alias or alias in self._slots:
                raise ValueError(f"Alias collides with provider: {alias}")
            if target not in self._slots:
                raise ValueError(f"Alias target is not registered: {target}")
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._closed = False
        self._close_complete = False
        self._close_lock = asyncio.Lock()
        self._lease_changed = asyncio.Event()

    def descriptors(self) -> tuple[TTSProviderDescriptor, ...]:
        return tuple(slot.spec.descriptor for slot in self._slots.values())

    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def _resolve_id(self, provider_id: str) -> str:
        canonical_id = self._aliases.get(provider_id, provider_id)
        if canonical_id not in self._slots:
            raise UnknownTTSProviderError(
                f"Unknown TTS provider: {provider_id}"
            )
        return canonical_id

    def configuration_revision(self, provider_id: str) -> int:
        return self._slots[self._resolve_id(provider_id)].revision

    async def acquire(self, provider_id: str) -> TTSAdapterLease:
        canonical_id = self._resolve_id(provider_id)
        if self._closed:
            raise TTSRegistryClosedError("The TTS registry is closed")
        slot = self._slots[canonical_id]
        async with slot.lock:
            if self._closed:
                raise TTSRegistryClosedError("The TTS registry is closed")
            if slot.reconfiguring:
                raise TTSProviderReconfiguringError(
                    f"TTS provider is reconfiguring: {canonical_id}"
                )
            if slot.active is None:
                adapter = slot.spec.factory(deepcopy(slot.config))
                slot.active = _AdapterRecord(adapter=adapter)
            record = slot.active
            record.leases += 1

        async def release() -> None:
            await self._release(slot, record)

        return TTSAdapterLease(canonical_id, record.adapter, release)

    async def _release(
        self, slot: _ProviderSlot, record: _AdapterRecord
    ) -> None:
        close_record = False
        async with slot.lock:
            if record.leases == 0:
                return
            record.leases -= 1
            if record.leases == 0:
                slot.lease_changed.set()
                self._lease_changed.set()
                if record.retired and not record.closed:
                    record.closed = True
                    if record in slot.retired:
                        slot.retired.remove(record)
                    close_record = True
        if close_record:
            await record.adapter.close()

    async def get_catalog(
        self, provider_id: str, refresh: bool = False
    ) -> TTSProviderCatalog:
        lease = await self.acquire(provider_id)
        try:
            await lease.adapter.ensure_ready()
            return await lease.adapter.get_catalog(refresh=refresh)
        finally:
            await lease.release()

    async def reconfigure_provider(
        self, provider_id: str, config: Mapping[str, Any]
    ) -> ReconfigureResult:
        canonical_id = self._resolve_id(provider_id)
        if self._closed:
            raise TTSRegistryClosedError("The TTS registry is closed")
        slot = self._slots[canonical_id]
        new_config = deepcopy(dict(config))
        if slot.spec.exclusive_reconfigure:
            return await self._reconfigure_exclusive(slot, new_config)
        return await self._reconfigure_retiring(slot, new_config)

    async def _reconfigure_retiring(
        self, slot: _ProviderSlot, new_config: dict[str, Any]
    ) -> ReconfigureResult:
        close_record: _AdapterRecord | None = None
        async with slot.transition_lock:
            async with slot.lock:
                if slot.config == new_config:
                    return ReconfigureResult.UNCHANGED
                old_record = slot.active
                slot.active = None
                slot.config = new_config
                slot.revision += 1
                if old_record is not None:
                    old_record.retired = True
                    slot.retired.append(old_record)
                    if old_record.leases == 0:
                        old_record.closed = True
                        slot.retired.remove(old_record)
                        close_record = old_record
            if close_record is not None:
                await close_record.adapter.close()
        return ReconfigureResult.CHANGED

    async def _reconfigure_exclusive(
        self, slot: _ProviderSlot, new_config: dict[str, Any]
    ) -> ReconfigureResult:
        async with slot.transition_lock:
            async with slot.lock:
                if slot.config == new_config:
                    return ReconfigureResult.UNCHANGED
                slot.reconfiguring = True
                old_record = slot.active

            if old_record is not None:
                while True:
                    slot.lease_changed.clear()
                    async with slot.lock:
                        if old_record.leases == 0:
                            slot.active = None
                            if not old_record.closed:
                                old_record.closed = True
                                close_old = True
                            else:
                                close_old = False
                            break
                    await slot.lease_changed.wait()
                if close_old:
                    await old_record.adapter.close()

            async with slot.lock:
                slot.config = new_config
                slot.revision += 1
                slot.reconfiguring = False
        return ReconfigureResult.CHANGED

    async def close(self) -> None:
        async with self._close_lock:
            if self._close_complete:
                return
            self._closed = True
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._shutdown_timeout_seconds
            while self._total_leases() > 0:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                self._lease_changed.clear()
                if self._total_leases() == 0:
                    break
                try:
                    await asyncio.wait_for(
                        self._lease_changed.wait(), timeout=remaining
                    )
                except TimeoutError:
                    break

            records: list[_AdapterRecord] = []
            for slot in self._slots.values():
                async with slot.lock:
                    candidates = (
                        ([slot.active] if slot.active is not None else [])
                        + slot.retired
                    )
                    slot.active = None
                    slot.retired = []
                    for record in candidates:
                        if not record.closed:
                            record.closed = True
                            records.append(record)

            first_error: BaseException | None = None
            for record in records:
                try:
                    await record.adapter.close()
                except BaseException as error:
                    first_error = first_error or error
            self._close_complete = True
            if first_error is not None:
                raise first_error

    def _total_leases(self) -> int:
        total = 0
        for slot in self._slots.values():
            if slot.active is not None:
                total += slot.active.leases
            total += sum(record.leases for record in slot.retired)
        return total
```

Keep all mutable state registry-instance scoped; no class variables are allowed
in this file.

- [ ] **Step 5: Run the registry tests**

Run: `pytest Tests/TTS/test_adapter_registry.py -q`

Expected: all tests pass, including the concurrent factory-count assertion.

- [ ] **Step 6: Commit the registry**

```bash
git add tldw_chatbook/TTS/adapter_registry.py Tests/TTS/test_adapter_registry.py
git commit -m "feat(tts): add sealed adapter registry"
```

---

### Task 3: Quarantine the existing backend registry

**Files:**
- Modify: `tldw_chatbook/TTS/TTS_Backends.py:31-146`
- Create: `Tests/TTS/test_legacy_backend_registry.py`

**Interfaces:**
- Consumes: existing backend classes and `TTSBackendManager`.
- Produces: bridge-only `BackendRegistry.ensure_builtins()`, `get()`, `list_backends()`, and `_reset_for_tests()`. Public `register()` rejects new runtime registrations.

- [ ] **Step 1: Write quarantine and deterministic-reset tests**

```python
def test_legacy_registry_is_closed_to_new_providers() -> None:
    BackendRegistry._reset_for_tests()
    BackendRegistry.ensure_builtins()

    with pytest.raises(RuntimeError, match="sealed legacy registry"):
        BackendRegistry.register("new_provider_*", object)  # type: ignore[arg-type]


def test_legacy_registry_reset_is_deterministic() -> None:
    BackendRegistry._reset_for_tests()
    first = tuple(BackendRegistry.ensure_builtins())
    second = tuple(BackendRegistry.ensure_builtins())

    assert first == second
    assert len(first) == len(set(first))
    assert set(first) <= {
        "openai_official_*",
        "local_kokoro_*",
        "elevenlabs_*",
        "local_chatterbox_*",
        "alltalk_*",
        "local_higgs_*",
    }
```

- [ ] **Step 2: Run the focused tests and observe registration remains open**

Run: `pytest Tests/TTS/test_legacy_backend_registry.py -q`

Expected: the new-provider test fails because `BackendRegistry.register()` currently mutates `_registry`.

- [ ] **Step 3: Move built-in registration behind an idempotent private path**

```python
class BackendRegistry:
    _registry: dict[str, type[TTSBackendBase]] = {}
    _builtins_loaded = False
    _builtin_ids = frozenset({
        "openai_official_*",
        "local_kokoro_*",
        "elevenlabs_*",
        "local_chatterbox_*",
        "alltalk_*",
        "local_higgs_*",
    })

    @classmethod
    def register(
        cls, backend_id: str, backend_class: type[TTSBackendBase]
    ) -> None:
        raise RuntimeError(
            "The sealed legacy registry accepts no new providers"
        )

    @classmethod
    def _register_builtin(
        cls, backend_id: str, backend_class: type[TTSBackendBase]
    ) -> None:
        if backend_id not in cls._builtin_ids:
            raise ValueError(f"Unknown legacy backend ID: {backend_id}")
        existing = cls._registry.get(backend_id)
        if existing is not None and existing is not backend_class:
            raise RuntimeError(f"Conflicting legacy backend: {backend_id}")
        cls._registry[backend_id] = backend_class

    @classmethod
    def ensure_builtins(cls) -> tuple[str, ...]:
        if not cls._builtins_loaded:
            cls._load_builtin_classes()
            cls._builtins_loaded = True
        return tuple(cls._registry)

    @classmethod
    def _load_builtin_classes(cls) -> None:
        builtin_imports = (
            (
                "openai_official_*",
                "tldw_chatbook.TTS.backends.openai",
                "OpenAITTSBackend",
            ),
            (
                "local_kokoro_*",
                "tldw_chatbook.TTS.backends.kokoro",
                "KokoroTTSBackend",
            ),
            (
                "elevenlabs_*",
                "tldw_chatbook.TTS.backends.elevenlabs",
                "ElevenLabsTTSBackend",
            ),
            (
                "local_chatterbox_*",
                "tldw_chatbook.TTS.backends.chatterbox",
                "ChatterboxTTSBackend",
            ),
            (
                "alltalk_*",
                "tldw_chatbook.TTS.backends.alltalk",
                "AllTalkTTSBackend",
            ),
            (
                "local_higgs_*",
                "tldw_chatbook.TTS.backends.higgs",
                "HiggsAudioTTSBackend",
            ),
        )
        for backend_id, module_name, class_name in builtin_imports:
            try:
                module = importlib.import_module(module_name)
                backend_class = getattr(module, class_name)
                cls._register_builtin(backend_id, backend_class)
            except ImportError:
                logger.warning(
                    "Legacy TTS backend is unavailable: {}", backend_id
                )

    @classmethod
    def _reset_for_tests(cls) -> None:
        cls._registry.clear()
        cls._builtins_loaded = False
```

Make `TTSBackendManager.__init__()` call `BackendRegistry.ensure_builtins()`.
Delete `TTSBackendManager._register_builtin_backends()`. Preserve the old
`get()` wildcard behavior only inside `BackendRegistry`; no new module may
import `BackendRegistry`.

- [ ] **Step 4: Run the focused and existing backend tests**

Run: `pytest Tests/TTS/test_legacy_backend_registry.py Tests/TTS/test_kokoro_validation.py Tests/TTS/test_chatterbox_validation.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the quarantine**

```bash
git add tldw_chatbook/TTS/TTS_Backends.py Tests/TTS/test_legacy_backend_registry.py
git commit -m "refactor(tts): quarantine legacy backend registry"
```

---

### Task 4: Add provider-scoped legacy adapters and compatibility catalogs

**Files:**
- Create: `tldw_chatbook/TTS/legacy_catalogs.py`
- Create: `tldw_chatbook/TTS/legacy_bridge.py`
- Create: `Tests/TTS/test_legacy_bridge.py`

**Interfaces:**
- Consumes: Task 1 contracts, `OpenAISpeechRequest`, and bridge-only `TTSBackendManager`.
- Produces: `LegacyRoute`, `resolve_legacy_route()`, `LegacyBackendHost`, `LegacyTTSAdapter`, `legacy_provider_specs()`, and `LEGACY_PROVIDER_IDS`.

- [ ] **Step 1: Write route-table and fail-closed tests**

```python
@pytest.mark.parametrize(
    ("internal_id", "provider_id"),
    [
        ("openai_official_tts-1", "openai"),
        ("openai_official_tts-1-hd", "openai"),
        ("openai_official_tts1", "openai"),
        ("openai_official_tts1hd", "openai"),
        ("elevenlabs_eleven_multilingual_v2", "elevenlabs"),
        ("local_kokoro_default_onnx", "kokoro"),
        ("local_kokoro_default_pytorch", "kokoro"),
        ("local_chatterbox_default", "chatterbox"),
        ("local_higgs_default", "higgs"),
        ("local_higgs_v2", "higgs"),
        ("alltalk_default", "alltalk"),
        ("alltalk_alltalk", "alltalk"),
    ],
)
def test_resolver_uses_enumerated_routes(
    internal_id: str, provider_id: str
) -> None:
    route = resolve_legacy_route(internal_id)
    assert route.provider_id == provider_id
    assert route.internal_model_id == internal_id


@pytest.mark.parametrize(
    "internal_id",
    ["openai_official_", "elevenlabs_custom_unknown", "local_kokoro_evil"],
)
def test_resolver_rejects_unlisted_internal_ids(internal_id: str) -> None:
    with pytest.raises(UnknownLegacyModelError):
        resolve_legacy_route(internal_id)
```

Build `LEGACY_ROUTES` from explicit model constants:

```python
OPENAI_INTERNAL_IDS = (
    "openai_official_tts-1",
    "openai_official_tts-1-hd",
    "openai_official_tts1",
    "openai_official_tts1hd",
)
ELEVENLABS_MODELS = (
    "eleven_monolingual_v1",
    "eleven_multilingual_v1",
    "eleven_multilingual_v2",
    "eleven_turbo_v2",
    "eleven_turbo_v2_5",
    "eleven_flash_v2",
    "eleven_flash_v2_5",
    "english_v1",
    "elevenlabs",
)
```

Define `ELEVENLABS_MODELS` in `legacy_catalogs.py` and import it from
`legacy_bridge.py`; keep `OPENAI_INTERNAL_IDS` in `legacy_bridge.py`. This
direction prevents a catalog/bridge import cycle.

The route table contains `elevenlabs_{model}` for each listed ElevenLabs model
and the exact Kokoro, Chatterbox, Higgs, and AllTalk IDs shown by the test. Do
not use `startswith()`, regular-expression suffixes, or wildcard lookup in the
resolver.

- [ ] **Step 2: Write host-isolation, lock, progress, catalog, and close tests**

```python
class FakeLegacyBackend:
    def __init__(self) -> None:
        self.progress_callback = None

    def set_progress_callback(self, callback) -> None:
        self.progress_callback = callback

    async def generate_speech_stream(self, request: OpenAISpeechRequest):
        del request
        if self.progress_callback is not None:
            await self.progress_callback(
                {"status": "Generating", "progress": 0.5}
            )
        yield b"audio"


class BlockingLegacyBackend(FakeLegacyBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.allow_finish = asyncio.Event()
        self.active_generations = 0
        self.max_concurrent_generations = 0

    async def generate_speech_stream(self, request: OpenAISpeechRequest):
        del request
        self.active_generations += 1
        self.max_concurrent_generations = max(
            self.max_concurrent_generations, self.active_generations
        )
        self.started.set()
        try:
            await self.allow_finish.wait()
            if self.progress_callback is not None:
                await self.progress_callback(
                    {"status": "Complete", "progress": 1.0}
                )
            yield b"audio"
        finally:
            self.active_generations -= 1


class FakeLegacyManager:
    def __init__(self, backend: FakeLegacyBackend) -> None:
        self.backend = backend
        self.get_backend_calls = 0
        self.close_calls = 0

    async def get_backend(self, internal_model_id: str):
        del internal_model_id
        self.get_backend_calls += 1
        return self.backend

    async def close_all_backends(self) -> None:
        self.close_calls += 1


def speech_request() -> OpenAISpeechRequest:
    return OpenAISpeechRequest(
        model="kokoro",
        input="hello",
        voice="af_heart",
        response_format="wav",
    )


async def collect(stream) -> bytes:
    return b"".join([chunk async for chunk in stream])


def sink(label: str):
    async def record(progress: TTSProgress) -> None:
        assert progress.status
        assert label in {"first", "second"}

    return record


@pytest.mark.asyncio
async def test_same_backend_serializes_callback_through_stream_consumption() -> None:
    backend = BlockingLegacyBackend()
    manager = FakeLegacyManager(backend)
    host = LegacyBackendHost(
        provider_id="kokoro",
        app_config={"app_tts": {"KOKORO_DEVICE": "cpu"}},
        manager_factory=lambda _: manager,
    )
    first = collect(host.generate(
        "local_kokoro_default_onnx", speech_request(), sink("first")
    ))
    second = collect(host.generate(
        "local_kokoro_default_onnx", speech_request(), sink("second")
    ))

    first_task = asyncio.create_task(first)
    await backend.started.wait()
    second_task = asyncio.create_task(second)
    await asyncio.sleep(0)
    assert manager.get_backend_calls == 1

    backend.allow_finish.set()
    await first_task
    await second_task

    assert backend.max_concurrent_generations == 1
    assert backend.progress_callback is None


@pytest.mark.asyncio
async def test_provider_hosts_are_isolated_and_close_once() -> None:
    managers: dict[str, FakeLegacyManager] = {}
    specs = legacy_provider_specs(
        {"app_tts": {"default_format": "wav"}},
        manager_factory=lambda provider, config: managers.setdefault(
            provider, FakeLegacyManager(FakeLegacyBackend())
        ),
    )
    adapters = [spec.factory(spec.initial_config) for spec in specs]

    assert len({id(adapter.host) for adapter in adapters}) == 6
    for adapter in adapters:
        await adapter.close()
        await adapter.close()
    assert all(manager.close_calls == 1 for manager in managers.values())


@pytest.mark.asyncio
async def test_different_provider_hosts_can_generate_concurrently() -> None:
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
    first = asyncio.create_task(collect(openai.generate(
        "openai_official_tts-1", speech_request(), None
    )))
    second = asyncio.create_task(collect(kokoro.generate(
        "local_kokoro_default_onnx", speech_request(), None
    )))

    await asyncio.wait_for(
        asyncio.gather(
            openai_backend.started.wait(),
            kokoro_backend.started.wait(),
        ),
        timeout=1,
    )
    openai_backend.allow_finish.set()
    kokoro_backend.allow_finish.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_legacy_catalogs_are_approximate_and_provider_scoped() -> None:
    spec = next(
        spec
        for spec in legacy_provider_specs({})
        if spec.descriptor.provider_id == "openai"
    )
    adapter = spec.factory(spec.initial_config)
    catalog = await adapter.get_catalog()

    assert catalog.provider_id == "openai"
    assert catalog.approximate is True
    assert {model.model_id for model in catalog.models} == {"tts-1", "tts-1-hd"}
    assert "alloy" in catalog.models[0].voices
```

- [ ] **Step 3: Run the bridge tests and confirm they fail**

Run: `pytest Tests/TTS/test_legacy_bridge.py -q`

Expected: collection fails because `legacy_bridge.py` and `legacy_catalogs.py`
do not exist.

- [ ] **Step 4: Add complete approximate catalogs**

In `legacy_catalogs.py`, define one `TTSProviderCatalog` per provider. Preserve
the currently visible Playground model IDs and voices:

```python
LEGACY_MODELS = {
    "openai": ("tts-1", "tts-1-hd"),
    "elevenlabs": ELEVENLABS_MODELS[:7],
    "kokoro": ("kokoro",),
    "chatterbox": ("chatterbox",),
    "higgs": ("higgs-audio-v2",),
    "alltalk": ("alltalk",),
}
OPENAI_VOICES = (
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "onyx", "nova", "sage", "shimmer", "verse",
)
ELEVENLABS_VOICES = (
    "21m00Tcm4TlvDq8ikWAM", "AZnzlk1XvdvUeBnXmlld",
    "EXAVITQu4vr4xnSDxMaL", "ErXwobaYiN019PkySvjV",
    "MF3mGyEYCl7XYWbV9V6O", "TxGEqnHWrfWFTfGW9XjX",
    "VR6AewLTigWG4xSOukaG", "pNInz6obpgDQGcFmaJgB",
    "yoZ06aMxZJJ28mfd3POQ",
)
KOKORO_VOICES = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
    "af_sky", "am_adam", "am_michael", "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
)

_ALL_VISIBLE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
_VOICES = {
    "openai": OPENAI_VOICES,
    "elevenlabs": ELEVENLABS_VOICES,
    "kokoro": KOKORO_VOICES,
    "chatterbox": ("default",),
    "higgs": ("default",),
    "alltalk": ("female_01.wav", "male_01.wav"),
}
_OPTIONS = {
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


def legacy_catalog(provider_id: str) -> TTSProviderCatalog:
    models = LEGACY_MODELS.get(provider_id)
    if models is None:
        raise KeyError(f"Unknown legacy provider: {provider_id}")
    return TTSProviderCatalog(
        provider_id=provider_id,
        revision=1,
        health=ProviderHealth(state="available", fresh=True),
        models=tuple(
            TTSModelInfo(
                model_id=model_id,
                display_name=model_id.replace("_", " ").title(),
                family=provider_id,
                upstream_mode="legacy",
                formats=_ALL_VISIBLE_FORMATS,
                voices=_VOICES[provider_id],
                supports_speed=True,
                supports_options=_OPTIONS[provider_id],
            )
            for model_id in models
        ),
        approximate=True,
    )
```

The generic visible format list intentionally preserves current Playground
behavior. Later native adapters replace approximate format claims with
authoritative provider catalogs.

- [ ] **Step 5: Implement the provider-scoped host and adapter**

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager


class LegacyBackendHost:
    def __init__(
        self,
        *,
        provider_id: str,
        app_config: Mapping[str, Any],
        manager_factory: Callable[[dict[str, Any]], TTSBackendManager],
    ) -> None:
        self.provider_id = provider_id
        self._app_config = deepcopy(dict(app_config))
        self._manager_factory = manager_factory
        self._manager: TTSBackendManager | None = None
        self._manager_lock = asyncio.Lock()
        self._operation_locks: dict[str, asyncio.Lock] = {}
        self._closed = False

    async def _get_manager(self) -> TTSBackendManager:
        async with self._manager_lock:
            if self._closed:
                raise RuntimeError("Legacy TTS host is closed")
            if self._manager is None:
                self._manager = self._manager_factory(
                    deepcopy(self._app_config)
                )
            return self._manager

    async def generate(
        self,
        internal_model_id: str,
        request: OpenAISpeechRequest,
        progress_sink: ProgressSink | None,
    ) -> AsyncIterator[bytes]:
        lock = self._operation_locks.setdefault(
            internal_model_id, asyncio.Lock()
        )
        async with lock:
            manager = await self._get_manager()
            backend = await manager.get_backend(internal_model_id)
            if backend is None:
                raise ValueError(
                    f"TTS model '{request.model}' is not available"
                )
            backend.set_progress_callback(
                _legacy_progress_callback(progress_sink)
                if progress_sink is not None
                else None
            )
            try:
                async for chunk in backend.generate_speech_stream(request):
                    yield bytes(chunk)
            finally:
                backend.set_progress_callback(None)

    async def close(self) -> None:
        async with self._manager_lock:
            if self._closed:
                return
            self._closed = True
            manager = self._manager
            self._manager = None
        if manager is not None:
            await manager.close_all_backends()


class LegacyTTSAdapter:
    _allowed_options = {
        "_legacy_openai_request",
        "_legacy_internal_model_id",
    }

    def __init__(
        self,
        provider_id: str,
        host: LegacyBackendHost,
        catalog: TTSProviderCatalog,
    ) -> None:
        self.provider_id = provider_id
        self.host = host
        self._catalog = catalog

    async def ensure_ready(self) -> None:
        return

    async def get_catalog(
        self, refresh: bool = False
    ) -> TTSProviderCatalog:
        del refresh
        return self._catalog

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        if set(request.options) != self._allowed_options:
            raise ValueError("Invalid legacy adapter options")
        legacy_request = request.options["_legacy_openai_request"]
        internal_id = request.options["_legacy_internal_model_id"]
        if not isinstance(legacy_request, OpenAISpeechRequest):
            raise TypeError("Legacy request must be OpenAISpeechRequest")
        route = resolve_legacy_route(str(internal_id))
        if route.provider_id != self.provider_id:
            raise ValueError("Legacy route does not match provider")
        return TTSAudioResponse(
            provider_id=self.provider_id,
            model_id=request.model_id,
            audio_format=legacy_request.response_format,
            content_type=_content_type(legacy_request.response_format),
            byte_stream=self.host.generate(
                route.internal_model_id, legacy_request, progress_sink
            ),
        )

    async def close(self) -> None:
        await self.host.close()
```

Add exact route construction and provider factories:

```python
class UnknownLegacyModelError(LookupError):
    """Raised when a compatibility internal model ID is not enumerated."""


@dataclass(frozen=True, slots=True)
class LegacyRoute:
    provider_id: str
    internal_model_id: str


LEGACY_PROVIDER_IDS = (
    "openai",
    "elevenlabs",
    "kokoro",
    "chatterbox",
    "higgs",
    "alltalk",
)
_STATIC_ROUTES = {
    "local_kokoro_default_onnx": "kokoro",
    "local_kokoro_default_pytorch": "kokoro",
    "local_chatterbox_default": "chatterbox",
    "local_higgs_default": "higgs",
    "local_higgs_v2": "higgs",
    "alltalk_default": "alltalk",
    "alltalk_alltalk": "alltalk",
}
LEGACY_ROUTES = {
    **{internal_id: "openai" for internal_id in OPENAI_INTERNAL_IDS},
    **{
        f"elevenlabs_{model_id}": "elevenlabs"
        for model_id in ELEVENLABS_MODELS
    },
    **_STATIC_ROUTES,
}


def resolve_legacy_route(internal_model_id: str) -> LegacyRoute:
    provider_id = LEGACY_ROUTES.get(internal_model_id)
    if provider_id is None:
        raise UnknownLegacyModelError(
            "The selected TTS model is not available"
        )
    return LegacyRoute(provider_id, internal_model_id)


def _legacy_progress_callback(progress_sink: ProgressSink):
    async def report(info: Mapping[str, Any]) -> None:
        raw_fraction = info.get("progress")
        fraction = (
            max(0.0, min(1.0, float(raw_fraction)))
            if isinstance(raw_fraction, (int, float))
            else None
        )
        raw_metrics = info.get("metrics")
        metrics = {
            str(key): value
            for key, value in (
                raw_metrics.items()
                if isinstance(raw_metrics, Mapping)
                else ()
            )
            if isinstance(value, (str, int, float, bool))
        }
        await progress_sink(
            TTSProgress(
                status=str(info.get("status") or "Generating"),
                fraction=fraction,
                processed=(
                    int(info["processed"])
                    if isinstance(info.get("processed"), int)
                    else None
                ),
                total=(
                    int(info["total"])
                    if isinstance(info.get("total"), int)
                    else None
                ),
                metrics=metrics,
            )
        )

    return report


_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "elevenlabs": "ElevenLabs",
    "kokoro": "Kokoro (Local)",
    "chatterbox": "Chatterbox (Local)",
    "higgs": "Higgs Audio (Local)",
    "alltalk": "AllTalk (Local)",
}


def legacy_provider_specs(
    app_config: Mapping[str, Any],
    *,
    manager_factory: Callable[
        [str, dict[str, Any]], TTSBackendManager
    ] | None = None,
) -> tuple[TTSProviderSpec, ...]:
    def default_manager_factory(
        _provider_id: str, config: dict[str, Any]
    ) -> TTSBackendManager:
        from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager

        return TTSBackendManager(app_config=config)

    create_manager = manager_factory or default_manager_factory
    config_snapshot = deepcopy(dict(app_config))
    specs: list[TTSProviderSpec] = []
    for provider_id in LEGACY_PROVIDER_IDS:
        def create_adapter(
            config: Mapping[str, Any],
            selected_provider: str = provider_id,
        ) -> LegacyTTSAdapter:
            provider_config = deepcopy(dict(config["app_config"]))
            host = LegacyBackendHost(
                provider_id=selected_provider,
                app_config=provider_config,
                manager_factory=lambda current_config: create_manager(
                    selected_provider, current_config
                ),
            )
            return LegacyTTSAdapter(
                selected_provider,
                host,
                legacy_catalog(selected_provider),
            )

        specs.append(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id=provider_id,
                    display_name=_DISPLAY_NAMES[provider_id],
                    native=False,
                ),
                factory=create_adapter,
                initial_config={"app_config": config_snapshot},
            )
        )
    return tuple(specs)
```

Every provider factory creates a fresh `LegacyBackendHost`; no host or manager
is shared between provider specs.

- [ ] **Step 6: Run bridge and backend tests**

Run: `pytest Tests/TTS/test_legacy_bridge.py Tests/TTS/test_legacy_backend_registry.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the legacy bridge**

```bash
git add tldw_chatbook/TTS/legacy_catalogs.py tldw_chatbook/TTS/legacy_bridge.py Tests/TTS/test_legacy_bridge.py
git commit -m "feat(tts): bridge legacy providers into adapters"
```

---

### Task 5: Move TTSService onto the registry and preserve compatibility

**Files:**
- Create: `tldw_chatbook/TTS/adapter_bootstrap.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py:1-145`
- Create: `Tests/TTS/test_tts_registry_service.py`

**Interfaces:**
- Consumes: `TTSAdapterRegistry`, `TTSRequest`, `ProgressSink`, `legacy_provider_specs()`, `resolve_legacy_route()`, and `OpenAISpeechRequest`.
- Produces: registry-backed `TTSService.synthesize()`, `get_catalog()`, `reconfigure_provider()`, `generate_audio_stream()`, `close()`, `build_default_tts_service()`, `bind_tts_service()`, `get_tts_service()`, `reset_tts_service_binding()`, and `close_tts_resources()`.

- [ ] **Step 1: Write native response-lease and compatibility cleanup tests**

```python
def tts_request(provider_id: str) -> TTSRequest:
    return TTSRequest(
        provider_id=provider_id,
        model_id="tts-1",
        text="hello",
        voice="alloy",
        response_format="mp3",
    )


def speech_request() -> OpenAISpeechRequest:
    return OpenAISpeechRequest(
        model="tts-1",
        input="hello",
        voice="alloy",
        response_format="mp3",
    )


def registry_for_adapter(adapter: FakeAdapter) -> TTSAdapterRegistry:
    replacements = FakeAdapterFactory(adapter.provider_id)
    calls = 0

    def factory(config):
        nonlocal calls
        del config
        calls += 1
        return adapter if calls == 1 else replacements({})

    spec = TTSProviderSpec(
        descriptor=TTSProviderDescriptor(
            provider_id=adapter.provider_id,
            display_name=adapter.provider_id,
            native=True,
        ),
        factory=factory,
        initial_config={"revision": 1},
    )
    return TTSAdapterRegistry(specs=(spec,), aliases={})


def service_for_adapter(adapter: FakeAdapter) -> TTSService:
    return TTSService(registry_for_adapter(adapter))


@pytest.mark.asyncio
async def test_synthesize_holds_lease_until_response_close() -> None:
    adapter = FakeAdapter("openai", chunks=(b"a", b"b"))
    registry = registry_for_adapter(adapter)
    service = TTSService(registry, max_concurrent_operations=4)

    response = await service.synthesize(tts_request("openai"))
    assert adapter.close_calls == 0
    await registry.reconfigure_provider("openai", {"revision": 2})
    assert adapter.close_calls == 0

    await response.aclose()
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_compatibility_generator_closes_after_partial_consumption() -> None:
    adapter = FakeAdapter("openai", chunks=(b"one", b"two"))
    service = service_for_adapter(adapter)
    stream = service.generate_audio_stream(
        speech_request(), "openai_official_tts-1"
    )

    assert await anext(stream) == b"one"
    await stream.aclose()

    assert adapter.response_close_calls == 1
    await service.registry.reconfigure_provider("openai", {"revision": 2})
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_compatibility_generator_releases_response_on_cancellation() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class CancellationAdapter(FakeAdapter):
        async def synthesize(
            self,
            request: TTSRequest,
            progress_sink: ProgressSink | None = None,
        ) -> TTSAudioResponse:
            del progress_sink

            async def stream():
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()
                yield b"unreachable"

            async def cleanup() -> None:
                self.response_close_calls += 1

            return TTSAudioResponse(
                provider_id=self.provider_id,
                model_id=request.model_id,
                audio_format=request.response_format,
                content_type="audio/mpeg",
                byte_stream=stream(),
                cleanup=cleanup,
            )

    adapter = CancellationAdapter("openai")
    service = service_for_adapter(adapter)
    task = asyncio.create_task(anext(service.generate_audio_stream(
        speech_request(), "openai_official_tts-1"
    )))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert adapter.response_close_calls == 1
    await service.registry.reconfigure_provider("openai", {"revision": 2})
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_progress_sink_failure_does_not_fail_synthesis() -> None:
    async def broken_sink(_progress: TTSProgress) -> None:
        raise RuntimeError("display failed")

    service = service_for_adapter(FakeAdapter("openai"))
    response = await service.synthesize(
        tts_request("openai"), progress_sink=broken_sink
    )
    assert b"".join([chunk async for chunk in response.byte_stream]) == b"audio"
    await response.aclose()


def test_service_concurrency_limit_is_instance_scoped_across_event_loops() -> None:
    first = service_for_adapter(FakeAdapter("openai"))
    second = service_for_adapter(FakeAdapter("openai"))

    async def consume(service: TTSService) -> bytes:
        response = await service.synthesize(tts_request("openai"))
        try:
            return b"".join(
                [chunk async for chunk in response.byte_stream]
            )
        finally:
            await response.aclose()

    assert asyncio.run(consume(first)) == b"audio"
    assert asyncio.run(consume(second)) == b"audio"
    assert first._operation_limit is not second._operation_limit


def test_bootstrap_preserves_nested_raw_provider_configuration() -> None:
    snapshot = _legacy_config_snapshot({
        "COMPREHENSIVE_CONFIG_RAW": {
            "API": {"openai_api_key": "secret"},
            "app_tts": {"default_format": "wav"},
        },
        "APP_TTS_CONFIG": {"default_format": "mp3"},
    })

    assert snapshot == {
        "API": {"openai_api_key": "secret"},
        "app_tts": {"default_format": "wav"},
    }


def test_default_bootstrap_has_six_exact_ids_no_aliases_and_is_lazy(
    monkeypatch,
) -> None:
    adapter_calls: list[str] = []

    def factory_builder(provider_id: str):
        def build(_config):
            adapter_calls.append(provider_id)
            return FakeAdapter(provider_id)

        return build

    monkeypatch.setattr(
        "tldw_chatbook.TTS.adapter_bootstrap.make_legacy_adapter_factory",
        factory_builder,
    )
    service = build_default_tts_service({})

    assert tuple(
        item.provider_id for item in service.registry.descriptors()
    ) == (
        "openai",
        "elevenlabs",
        "kokoro",
        "chatterbox",
        "higgs",
        "alltalk",
    )
    assert service.registry.aliases() == {}
    assert adapter_calls == []
```

The concrete test may instead observe zero adapter-factory calls through a
small test-side factory harness. Do not add a production materialization
introspection method merely to support this assertion.

- [ ] **Step 2: Write explicit application-binding tests**

```python
@pytest.mark.asyncio
async def test_accessor_requires_an_explicit_binding() -> None:
    reset_tts_service_binding()
    with pytest.raises(RuntimeError, match="not bound"):
        await get_tts_service({"app_tts": {"default_provider": "openai"}})


@pytest.mark.asyncio
async def test_accessor_returns_bound_service_without_retaining_config() -> None:
    first = service_for_adapter(FakeAdapter("openai"))
    bind_tts_service(first)
    try:
        assert await get_tts_service({"value": "first"}) is first
        assert await get_tts_service({"value": "second"}) is first
    finally:
        reset_tts_service_binding(expected=first)


def test_binding_rejects_a_different_live_service() -> None:
    first = service_for_adapter(FakeAdapter("openai"))
    second = service_for_adapter(FakeAdapter("openai"))
    bind_tts_service(first)
    try:
        with pytest.raises(RuntimeError, match="already bound"):
            bind_tts_service(second)
    finally:
        reset_tts_service_binding(expected=first)
```

- [ ] **Step 3: Run service tests and observe old manager coupling**

Run: `pytest Tests/TTS/test_tts_registry_service.py -q`

Expected: tests fail because `TTSService` still requires `TTSBackendManager` and
the accessor constructs a singleton from its first configuration.

- [ ] **Step 4: Implement the registry-backed service**

```python
class TTSService:
    def __init__(
        self,
        registry: TTSAdapterRegistry,
        *,
        max_concurrent_operations: int = 4,
    ) -> None:
        if max_concurrent_operations < 1:
            raise ValueError("max_concurrent_operations must be positive")
        self.registry = registry
        self._operation_limit = asyncio.Semaphore(max_concurrent_operations)

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        await self._operation_limit.acquire()
        try:
            lease = await self.registry.acquire(request.provider_id)
        except BaseException:
            self._operation_limit.release()
            raise
        safe_sink = _isolate_progress_sink(progress_sink)
        try:
            await lease.adapter.ensure_ready()
            response = await lease.adapter.synthesize(request, safe_sink)
        except BaseException:
            await lease.release()
            self._operation_limit.release()
            raise
        response.add_cleanup(lease.release)
        response.add_cleanup(self._release_operation_slot)
        return response

    async def _release_operation_slot(self) -> None:
        self._operation_limit.release()

    async def generate_audio_stream(
        self,
        request: OpenAISpeechRequest,
        internal_model_id: str,
        progress_sink: ProgressSink | None = None,
    ) -> AsyncIterator[bytes]:
        route = resolve_legacy_route(internal_model_id)
        native_request = TTSRequest(
            provider_id=route.provider_id,
            model_id=request.model,
            text=request.input,
            voice=request.voice,
            response_format=request.response_format,
            speed=request.speed,
            options={
                "_legacy_openai_request": request,
                "_legacy_internal_model_id": internal_model_id,
            },
        )
        response = await self.synthesize(native_request, progress_sink)
        try:
            async for chunk in response.byte_stream:
                yield chunk
        finally:
            await response.aclose()

    async def get_catalog(
        self, provider_id: str, refresh: bool = False
    ) -> TTSProviderCatalog:
        return await self.registry.get_catalog(provider_id, refresh=refresh)

    async def reconfigure_provider(
        self, provider_id: str, config: Mapping[str, Any]
    ) -> ReconfigureResult:
        return await self.registry.reconfigure_provider(provider_id, config)

    async def close(self) -> None:
        await self.registry.close()
```

Use this exact progress wrapper:

```python
def _isolate_progress_sink(
    progress_sink: ProgressSink | None,
) -> ProgressSink | None:
    if progress_sink is None:
        return None

    async def report(progress: TTSProgress) -> None:
        try:
            await progress_sink(progress)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("TTS progress sink failed")

    return report
```

- [ ] **Step 5: Replace singleton construction with explicit binding**

```python
_bound_tts_service: TTSService | None = None


def bind_tts_service(service: TTSService) -> None:
    global _bound_tts_service
    if _bound_tts_service is not None and _bound_tts_service is not service:
        raise RuntimeError("A different TTS service is already bound")
    _bound_tts_service = service


async def get_tts_service(
    app_config: Mapping[str, Any] | None = None,
) -> TTSService:
    del app_config
    if _bound_tts_service is None:
        raise RuntimeError("The application TTS service is not bound")
    return _bound_tts_service


def reset_tts_service_binding(
    *, expected: TTSService | None = None
) -> None:
    global _bound_tts_service
    if expected is not None and _bound_tts_service not in (None, expected):
        raise RuntimeError("Refusing to reset a different TTS service")
    _bound_tts_service = None


async def close_tts_resources() -> None:
    service = _bound_tts_service
    if service is None:
        return
    try:
        await service.close()
    finally:
        reset_tts_service_binding(expected=service)
```

Use the raw CLI configuration nested by `load_settings()` so the new
application-owned service preserves the API keys and provider settings that
handlers previously assembled:

```python
def _legacy_config_snapshot(
    app_config: Mapping[str, Any],
) -> dict[str, Any]:
    nested_raw = app_config.get("COMPREHENSIVE_CONFIG_RAW")
    source = (
        nested_raw
        if isinstance(nested_raw, Mapping)
        else app_config
    )
    snapshot = deepcopy(dict(source))
    if "app_tts" not in snapshot:
        normalized_tts = app_config.get("APP_TTS_CONFIG", {})
        snapshot["app_tts"] = (
            deepcopy(dict(normalized_tts))
            if isinstance(normalized_tts, Mapping)
            else {}
        )
    return snapshot


def build_default_tts_service(
    app_config: Mapping[str, Any],
) -> TTSService:
    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(_legacy_config_snapshot(app_config)),
        aliases={},
    )
    return TTSService(registry, max_concurrent_operations=4)
```

The builder performs no backend imports, model loading, network calls, or
process launch.

- [ ] **Step 6: Run service and registry tests**

Run: `pytest Tests/TTS/test_tts_registry_service.py Tests/TTS/test_adapter_registry.py Tests/TTS/test_legacy_bridge.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the service migration**

```bash
git add tldw_chatbook/TTS/adapter_bootstrap.py tldw_chatbook/TTS/TTS_Generation.py Tests/TTS/test_tts_registry_service.py
git commit -m "refactor(tts): route service through adapters"
```

---

### Task 6: Bind lifecycle and progress at the application boundary

**Files:**
- Modify: `tldw_chatbook/app.py:2994-3045`
- Modify: `tldw_chatbook/app.py:7671-7795`
- Modify: `tldw_chatbook/app.py:9960-10035`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py:350-425`
- Create: `Tests/TTS/test_tts_app_ownership.py`

**Interfaces:**
- Consumes: `build_default_tts_service()`, `bind_tts_service()`,
  `reset_tts_service_binding()`, and the
  `TTSService.generate_audio_stream(request, internal_model_id,
  progress_sink)` compatibility method.
- Produces: `TldwCli.tts_service`, `_bind_tts_service()`, and
  `_close_tts_service()`.

- [ ] **Step 1: Write application-ownership and boundary tests**

```python
class FakeOwnedService:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def test_app_constructs_one_tts_service(
    monkeypatch,
) -> None:
    service = FakeOwnedService()
    builder = Mock(return_value=service)
    monkeypatch.setattr("tldw_chatbook.app.build_default_tts_service", builder)

    app = TldwCli()

    assert app.tts_service is service
    builder.assert_called_once_with(app.app_config)


@pytest.mark.asyncio
async def test_app_binding_and_close_are_explicit() -> None:
    service = FakeOwnedService()
    owner = SimpleNamespace(tts_service=service, _tts_binding_active=False)

    TldwCli._bind_tts_service(owner)
    assert await get_tts_service() is service

    await TldwCli._close_tts_service(owner)
    assert service.close_calls == 1
    with pytest.raises(RuntimeError, match="not bound"):
        await get_tts_service()


def test_application_and_stts_do_not_reach_through_to_backend_manager() -> None:
    paths = (
        Path("tldw_chatbook/app.py"),
        Path("tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py"),
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        accesses = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "backend_manager"
        ]
        assert accesses == [], f"{path} reaches through to backend_manager"
```

- [ ] **Step 2: Run the ownership tests and confirm direct access remains**

Run: `pytest Tests/TTS/test_tts_app_ownership.py -q`

Expected: tests fail because the app does not own a registry-backed service and
both syntax trees contain direct `backend_manager` attribute access.

- [ ] **Step 3: Construct and bind the service before screen work**

After `self.app_config = load_settings()` in `TldwCli.__init__()`:

```python
self.tts_service = build_default_tts_service(self.app_config)
self._tts_binding_active = False
```

Add:

```python
def _bind_tts_service(self) -> None:
    if self._tts_binding_active:
        return
    bind_tts_service(self.tts_service)
    self._tts_binding_active = True


async def _close_tts_service(self) -> None:
    try:
        await self.tts_service.close()
    finally:
        reset_tts_service_binding(expected=self.tts_service)
        self._tts_binding_active = False


async def on_mount(self) -> None:
    self._bind_tts_service()
```

The current `TldwCli` class has no `on_mount()` method, so TASK-402 adds the
single method above.

- [ ] **Step 4: Replace handler-owned progress callback mutation**

Delete the block that calls:

```python
await self._stts_service.backend_manager.get_backend(internal_model_id)
backend.set_progress_callback(progress_callback)
```

Pass the existing callback into the compatibility call:

```python
async for chunk in self._stts_service.generate_audio_stream(
    request,
    internal_model_id,
    progress_sink=progress_callback,
):
    audio_data += chunk
    chunk_count += 1
    total_size = len(audio_data)
```

Keep the existing UI scheduling inside `progress_callback`; do not move Textual
widget references into the service or bridge.

- [ ] **Step 5: Simplify shutdown to one owner**

In `TldwCli.on_unmount()`, keep handler task/file cleanup, then call:

```python
try:
    await self._close_tts_service()
    self.loguru_logger.info("TTS service cleaned up properly")
except Exception:
    self.loguru_logger.exception("Error cleaning up TTS service")
```

Remove both calls to global manager cleanup and the special Higgs scan through
`service.backend_manager._backends`. The provider-scoped bridge adapters close
their own managers through `TTSService.close()`. Also delete the
`action_quit()` block labeled “Force cleanup Higgs backends immediately”;
`on_unmount()` is the sole async TTS resource owner and already runs after quit
is requested.

- [ ] **Step 6: Run ownership, service, STTS, and startup tests**

Run: `pytest Tests/TTS/test_tts_app_ownership.py Tests/TTS/test_tts_registry_service.py Tests/TTS/test_tts_improvements.py Tests/Performance/test_app_startup_performance.py -q`

Expected: all selected tests pass and no legacy adapter materializes during app
construction.

- [ ] **Step 7: Commit application ownership**

```bash
git add tldw_chatbook/app.py tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py Tests/TTS/test_tts_app_ownership.py
git commit -m "refactor(tts): bind service lifecycle to app"
```

---

### Task 7: Export the new API, close the key-prefix leak, and document the bridge

**Files:**
- Modify: `tldw_chatbook/TTS/__init__.py:1-25`
- Modify: `tldw_chatbook/TTS/backends/openai.py:40-60`
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Create: `Tests/TTS/test_tts_logging_privacy.py`

**Interfaces:**
- Consumes: public contracts and service/binding functions from prior tasks.
- Produces: stable package exports for callers; no package export for
  `LegacyBackendHost`, `LegacyTTSAdapter`, `BackendRegistry`, or a concrete
  backend.

- [ ] **Step 1: Write the API-key logging regression**

```python
@pytest.mark.asyncio
async def test_openai_backend_never_logs_api_key_fragments(monkeypatch) -> None:
    secret = "sk-super-secret-value"
    messages: list[str] = []
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.config.load_cli_config_and_ensure_existence",
        lambda: {"api_settings": {"openai": {"api_key": secret}}},
    )

    sink_id = logger.add(
        lambda message: messages.append(str(message)), level="DEBUG"
    )
    backend = None
    try:
        backend = OpenAITTSBackend(config={})
    finally:
        if backend is not None:
            await backend.close()
        logger.remove(sink_id)

    rendered = "\n".join(messages)
    assert secret not in rendered
    assert secret[:10] not in rendered
```

- [ ] **Step 2: Run the privacy test and reproduce the disclosure**

Run: `pytest Tests/TTS/test_tts_logging_privacy.py -q`

Expected: failure because `OpenAITTSBackend` logs the key length and first ten
characters when it reads `api_settings.openai.api_key`.

- [ ] **Step 3: Remove value-bearing diagnostics and define package exports**

Replace the key diagnostic with:

```python
logger.debug(
    "OpenAITTSBackend: api_settings.openai API key is configured"
)
```

Do not log length, prefix, suffix, hash, or configuration value. Export:

```python
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    ProgressSink,
    TTSAudioResponse,
    TTSModelInfo,
    TTSProgress,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSRequest,
)
from tldw_chatbook.TTS.TTS_Generation import (
    TTSService,
    bind_tts_service,
    close_tts_resources,
    get_tts_service,
    reset_tts_service_binding,
)
```

Retain `OpenAISpeechRequest` and `NormalizationOptions` exports for callers.
Remove `TTSBackendManager` and `TTSBackendBase` from `__all__`; direct legacy
imports remain temporarily valid only at their original module paths.

- [ ] **Step 4: Document ownership and migration**

Add a “TTS adapter service” section to the module guide containing:

```markdown
The application owns one sealed `TTSAdapterRegistry` and one `TTSService`.
New callers use canonical provider IDs and `TTSService.synthesize()`. Existing
callers may temporarily use `generate_audio_stream()` with an enumerated legacy
internal model ID.

The six legacy providers are separate adapter entries. Each adapter lazily owns
one provider-scoped `TTSBackendManager`; application and UI code must not access
that manager or a concrete backend. The bridge is removed only after every
retained provider has a native adapter and all legacy internal-model callers
have migrated.
```

Link ADR-023 and the approved audio.cpp design.

- [ ] **Step 5: Run privacy and import tests**

Run: `pytest Tests/TTS/test_tts_logging_privacy.py Tests/TTS/test_tts_registry_service.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit exports, privacy, and documentation**

```bash
git add tldw_chatbook/TTS/__init__.py tldw_chatbook/TTS/backends/openai.py Docs/Development/TTS/TTS_MODULE_GUIDE.md Tests/TTS/test_tts_logging_privacy.py
git commit -m "docs(tts): publish adapter service boundary"
```

---

### Task 8: Verify compatibility and close TASK-402

**Files:**
- Modify: `backlog/tasks/task-402 - Establish-TTS-adapter-registry-authority-and-legacy-bridge.md`

**Interfaces:**
- Consumes: all TASK-402 production and test changes.
- Produces: verified task notes, checked acceptance criteria, and Done status.

- [ ] **Step 1: Run focused registry and bridge suites**

Run:

```bash
pytest \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_legacy_backend_registry.py \
  Tests/TTS/test_legacy_bridge.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_app_ownership.py \
  Tests/TTS/test_tts_logging_privacy.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run existing TTS, STTS, media-reading, and audio-service regressions**

Run:

```bash
pytest \
  Tests/TTS \
  Tests/UI/test_stts_capability_state.py \
  Tests/Audio_Services/test_local_audio_services_service.py \
  Tests/Media/test_local_media_reading_service.py -q
```

Expected: all selected tests pass; optional-backend skips remain skips.

- [ ] **Step 3: Run static and boundary checks**

Run:

```bash
python -m compileall -q tldw_chatbook/TTS
python -m mypy \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapter_registry.py \
  tldw_chatbook/TTS/legacy_catalogs.py \
  tldw_chatbook/TTS/legacy_bridge.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  tldw_chatbook/TTS/TTS_Generation.py
rg -n "\\.backend_manager|BackendRegistry|TTSBackendManager" \
  tldw_chatbook/app.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/TTS/TTS_Generation.py
git diff --check
```

Expected:

- `compileall` exits zero.
- mypy reports `Success: no issues found` for the changed service boundary.
- `rg` prints no matches.
- `git diff --check` prints nothing and exits zero.

- [ ] **Step 4: Perform the ADR and scope audit**

Confirm:

```bash
rg -n "ADR-023|023-tts-adapter" \
  "backlog/tasks/task-402 - Establish-TTS-adapter-registry-authority-and-legacy-bridge.md" \
  Docs/Development/TTS/TTS_MODULE_GUIDE.md
rg -n "audio_cpp|audiocpp|AudioCpp" \
  tldw_chatbook/TTS Tests/TTS
```

Expected: the task and guide link ADR-023. The second command finds no native
audio.cpp adapter, HTTP client, supervisor, settings, or process code in this
slice.

- [ ] **Step 5: Update Backlog implementation notes and acceptance criteria**

Use:

```bash
backlog task edit 402 --notes "Implemented the app-owned sealed TTS adapter registry, operation leases and targeted provider retirement, six provider-scoped legacy adapters, enumerated compatibility routing, operation-scoped progress delivery, explicit application binding/shutdown, and the OpenAI key-prefix logging fix. Added focused registry, bridge, lifecycle, concurrency, privacy, and compatibility coverage. ADR-023 remains the governing architecture decision; audio.cpp native transport and supervision remain in later ordered tasks."
```

After the verification evidence passes, check all eight criteria and set Done:

```bash
backlog task edit 402 \
  --check-ac 1 \
  --check-ac 2 \
  --check-ac 3 \
  --check-ac 4 \
  --check-ac 5 \
  --check-ac 6 \
  --check-ac 7 \
  --check-ac 8
backlog task edit 402 \
  --check-dod 1 \
  --check-dod 2 \
  --check-dod 3 \
  --check-dod 4 \
  --check-dod 5
backlog task edit 402 -s Done
backlog task 402 --plain
```

Expected: TASK-402 shows status `Done`, all acceptance criteria checked, the
implementation plan and notes present, and links to ADR-023 and this plan.

- [ ] **Step 6: Commit task completion metadata**

```bash
git add "backlog/tasks/task-402 - Establish-TTS-adapter-registry-authority-and-legacy-bridge.md"
git commit -m "chore(tts): complete registry bridge task"
```
