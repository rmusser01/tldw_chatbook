from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from Tests.TTS.adapter_fakes import (
    FakeAdapter,
    FakeAdapterFactory,
    provider_spec,
)
from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
    TTSAdapterRegistry,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSProviderReconfiguringError,
    TTSProviderSpec,
    TTSRegistryClosedError,
    UnknownTTSProviderError,
)


@pytest.mark.asyncio
async def test_registry_uses_exact_ids_and_materializes_once() -> None:
    factory = FakeAdapterFactory("openai")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("openai", factory),),
        aliases={},
    )

    leases = await asyncio.gather(*(registry.acquire("openai") for _ in range(20)))

    assert factory.calls == 1
    assert [item.provider_id for item in registry.descriptors()] == ["openai"]
    assert registry.aliases() == {}
    with pytest.raises(UnknownTTSProviderError):
        await registry.acquire("open")
    for lease in leases:
        await lease.release()
    await registry.close()


@pytest.mark.asyncio
async def test_registry_resolves_only_explicit_aliases() -> None:
    factory = FakeAdapterFactory("openai")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("openai", factory),),
        aliases={"oa": "openai"},
    )

    exact, aliased = await asyncio.gather(
        registry.acquire("openai"),
        registry.acquire("oa"),
    )

    assert exact.provider_id == "openai"
    assert aliased.provider_id == "openai"
    assert exact.adapter is aliased.adapter
    assert factory.calls == 1
    returned_aliases = registry.aliases()
    returned_aliases["other"] = "openai"
    assert registry.aliases() == {"oa": "openai"}
    await exact.release()
    await aliased.release()
    await registry.close()


def test_registry_rejects_duplicate_ids_and_alias_collisions() -> None:
    factory = FakeAdapterFactory("openai")
    spec = provider_spec("openai", factory)
    with pytest.raises(ValueError, match="Duplicate provider"):
        TTSAdapterRegistry(specs=(spec, spec), aliases={})
    with pytest.raises(ValueError, match="Alias"):
        TTSAdapterRegistry(specs=(spec,), aliases={"openai": "openai"})


def test_registry_rejects_invalid_alias_targets() -> None:
    factory = FakeAdapterFactory("openai")
    spec = provider_spec("openai", factory)

    with pytest.raises(ValueError, match="Alias target"):
        TTSAdapterRegistry(specs=(spec,), aliases={"missing": "kokoro"})


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

    assert (
        await registry.reconfigure_provider("openai", {"key": "second"})
        is ReconfigureResult.CHANGED
    )
    assert registry.configuration_revision("openai") == 2
    assert registry.configuration_revision("kokoro") == 1
    assert old_openai.adapter.close_calls == 0
    assert kokoro.adapter.close_calls == 0

    await old_openai.release()
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
async def test_get_catalog_materializes_lazily_and_releases_its_lease() -> None:
    factory = FakeAdapterFactory("openai")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("openai", factory, {"key": "first"}),),
        aliases={"oa": "openai"},
    )

    assert factory.calls == 0
    catalog = await registry.get_catalog("oa", refresh=True)
    adapter = factory.instances[0]

    assert catalog.provider_id == "openai"
    assert adapter.ensure_ready_calls == 1
    assert factory.calls == 1
    assert (
        await registry.reconfigure_provider("openai", {"key": "second"})
        is ReconfigureResult.CHANGED
    )
    assert adapter.close_calls == 1
    assert factory.calls == 1
    await registry.close()


@pytest.mark.asyncio
async def test_exclusive_reconfigure_blocks_until_old_lease_releases() -> None:
    factory = FakeAdapterFactory("exclusive")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("exclusive", factory, {"revision": 1}, exclusive=True),),
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

    await asyncio.wait_for(registry.close(), timeout=0.1)
    await asyncio.wait_for(registry.close(), timeout=0.1)

    assert close_order == ["openai", "kokoro"]
    with pytest.raises(TTSRegistryClosedError):
        await registry.acquire("openai")
    await openai.release()
    await kokoro.release()
    assert openai.adapter.close_calls == 1
    assert kokoro.adapter.close_calls == 1


class _BlockingCloseAdapter(FakeAdapter):
    def __init__(
        self,
        provider_id: str,
        *,
        close_started: asyncio.Event,
        allow_close: asyncio.Event,
    ) -> None:
        super().__init__(provider_id)
        self._close_started = close_started
        self._allow_close = allow_close

    async def close(self) -> None:
        self.close_calls += 1
        self._close_started.set()
        await self._allow_close.wait()


class _BlockingCloseFactory:
    def __init__(
        self,
        provider_id: str,
        *,
        close_started: asyncio.Event,
        allow_close: asyncio.Event,
    ) -> None:
        self._provider_id = provider_id
        self._close_started = close_started
        self._allow_close = allow_close
        self.calls = 0
        self.instances: list[_BlockingCloseAdapter] = []

    def __call__(self, config: Mapping[str, Any]) -> _BlockingCloseAdapter:
        del config
        self.calls += 1
        adapter = _BlockingCloseAdapter(
            self._provider_id,
            close_started=self._close_started,
            allow_close=self._allow_close,
        )
        self.instances.append(adapter)
        return adapter


@pytest.mark.asyncio
async def test_shutdown_timeout_bounds_and_rejoins_adapter_cleanup() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    factory = _BlockingCloseFactory(
        "openai",
        close_started=close_started,
        allow_close=allow_close,
    )
    spec = TTSProviderSpec(
        descriptor=provider_spec("openai", FakeAdapterFactory("unused")).descriptor,
        factory=factory,
        initial_config={},
    )
    registry = TTSAdapterRegistry(
        specs=(spec,),
        aliases={},
        shutdown_timeout_seconds=0.01,
    )
    lease = await registry.acquire("openai")
    adapter = lease.adapter
    await lease.release()

    await asyncio.wait_for(registry.close(), timeout=0.1)
    await close_started.wait()
    await asyncio.wait_for(registry.close(), timeout=0.1)

    assert adapter.close_calls == 1
    allow_close.set()
    await registry.close()
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_wait_closed_joins_zero_timeout_adapter_cleanup_once() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    factory = _BlockingCloseFactory(
        "openai",
        close_started=close_started,
        allow_close=allow_close,
    )
    spec = TTSProviderSpec(
        descriptor=provider_spec("openai", FakeAdapterFactory("unused")).descriptor,
        factory=factory,
        initial_config={},
    )
    registry = TTSAdapterRegistry(
        specs=(spec,),
        aliases={},
        shutdown_timeout_seconds=0,
    )
    lease = await registry.acquire("openai")
    adapter = lease.adapter
    await lease.release()

    await registry.close()
    await close_started.wait()
    wait_for_close = asyncio.create_task(registry.wait_closed())
    await asyncio.sleep(0)

    assert wait_for_close.done() is False
    await registry.close()
    assert adapter.close_calls == 1
    allow_close.set()
    await wait_for_close
    await registry.wait_closed()
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_shutdown_does_not_report_cleanup_complete() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    factory = _BlockingCloseFactory(
        "openai",
        close_started=close_started,
        allow_close=allow_close,
    )
    spec = TTSProviderSpec(
        descriptor=provider_spec("openai", FakeAdapterFactory("unused")).descriptor,
        factory=factory,
        initial_config={},
    )
    registry = TTSAdapterRegistry(
        specs=(spec,),
        aliases={},
        shutdown_timeout_seconds=1.0,
    )
    lease = await registry.acquire("openai")
    adapter = lease.adapter
    await lease.release()
    first_close = asyncio.create_task(registry.close())
    await close_started.wait()

    first_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_close
    second_close = asyncio.create_task(registry.close())
    await asyncio.sleep(0)

    assert second_close.done() is False
    allow_close.set()
    await second_close
    assert adapter.close_calls == 1


@pytest.mark.asyncio
async def test_shutdown_reports_known_failure_while_other_cleanup_is_pending() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    failing_factory = _FailingCloseFactory("failing")
    blocking_factory = _BlockingCloseFactory(
        "blocking",
        close_started=close_started,
        allow_close=allow_close,
    )
    failing_spec = TTSProviderSpec(
        descriptor=provider_spec("failing", FakeAdapterFactory("unused")).descriptor,
        factory=failing_factory,
        initial_config={},
    )
    blocking_spec = TTSProviderSpec(
        descriptor=provider_spec("blocking", FakeAdapterFactory("unused")).descriptor,
        factory=blocking_factory,
        initial_config={},
    )
    registry = TTSAdapterRegistry(
        specs=(failing_spec, blocking_spec),
        aliases={},
        shutdown_timeout_seconds=0.01,
    )
    failing = await registry.acquire("failing")
    blocking = await registry.acquire("blocking")
    await failing.release()
    await blocking.release()

    first_error: RuntimeError | None = None
    try:
        await registry.close()
    except RuntimeError as error:
        first_error = error
    await close_started.wait()
    allow_close.set()
    second_error: RuntimeError | None = None
    try:
        await registry.close()
    except RuntimeError as error:
        second_error = error
    await registry.close()

    assert str(first_error) == "adapter close failed"
    assert str(second_error) == "adapter close failed"
    assert failing.adapter.close_calls == 1
    assert blocking.adapter.close_calls == 1


class _FailingCloseAdapter(FakeAdapter):
    async def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("adapter close failed")


class _FailingCloseFactory:
    def __init__(self, provider_id: str) -> None:
        self._provider_id = provider_id
        self.calls = 0
        self.instances: list[_FailingCloseAdapter] = []

    def __call__(self, config: Mapping[str, Any]) -> _FailingCloseAdapter:
        del config
        self.calls += 1
        adapter = _FailingCloseAdapter(self._provider_id)
        self.instances.append(adapter)
        return adapter


@pytest.mark.asyncio
async def test_wait_closed_reports_delayed_zero_timeout_cleanup_failure() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class DelayedFailingCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await allow_close.wait()
            raise RuntimeError("delayed adapter close failed")

    adapter = DelayedFailingCloseAdapter("openai")
    spec = TTSProviderSpec(
        descriptor=provider_spec("openai", FakeAdapterFactory("unused")).descriptor,
        factory=lambda _config: adapter,
        initial_config={},
    )
    registry = TTSAdapterRegistry(
        specs=(spec,),
        aliases={},
        shutdown_timeout_seconds=0,
    )
    lease = await registry.acquire("openai")
    await lease.release()

    await registry.close()
    await close_started.wait()
    wait_for_close = asyncio.create_task(registry.wait_closed())
    await asyncio.sleep(0)

    assert wait_for_close.done() is False
    allow_close.set()
    with pytest.raises(RuntimeError, match="delayed adapter close failed"):
        await wait_for_close
    with pytest.raises(RuntimeError, match="delayed adapter close failed"):
        await registry.wait_closed()
    assert adapter.close_calls == 1


@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.asyncio
async def test_close_waits_for_in_flight_reconfigure_cleanup(
    exclusive: bool,
) -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    factory = _BlockingCloseFactory(
        "openai",
        close_started=close_started,
        allow_close=allow_close,
    )
    spec = TTSProviderSpec(
        descriptor=provider_spec("openai", FakeAdapterFactory("unused")).descriptor,
        factory=factory,
        initial_config={"key": "first"},
        exclusive_reconfigure=exclusive,
    )
    registry = TTSAdapterRegistry(specs=(spec,), aliases={})
    lease = await registry.acquire("openai")
    adapter = lease.adapter
    await lease.release()
    reconfigure = asyncio.create_task(
        registry.reconfigure_provider("openai", {"key": "second"})
    )
    await close_started.wait()

    close = asyncio.create_task(registry.close())
    await asyncio.sleep(0)

    assert close.done() is False
    allow_close.set()
    assert await reconfigure is ReconfigureResult.CHANGED
    await close
    assert adapter.close_calls == 1
    with pytest.raises(TTSRegistryClosedError):
        await registry.acquire("openai")


@pytest.mark.asyncio
async def test_cancelled_exclusive_cleanup_keeps_admission_sealed() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    factory = _BlockingCloseFactory(
        "exclusive",
        close_started=close_started,
        allow_close=allow_close,
    )
    spec = TTSProviderSpec(
        descriptor=provider_spec("exclusive", FakeAdapterFactory("unused")).descriptor,
        factory=factory,
        initial_config={"revision": 1},
        exclusive_reconfigure=True,
    )
    registry = TTSAdapterRegistry(specs=(spec,), aliases={})
    lease = await registry.acquire("exclusive")
    await lease.release()
    reconfigure = asyncio.create_task(
        registry.reconfigure_provider("exclusive", {"revision": 2})
    )
    await close_started.wait()

    reconfigure.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reconfigure

    retry = asyncio.create_task(
        registry.reconfigure_provider("exclusive", {"revision": 2})
    )
    await asyncio.sleep(0)
    retry_completed_before_cleanup = retry.done()
    replacement_created_before_cleanup = factory.calls > 1
    unexpected_lease = None
    try:
        unexpected_lease = await registry.acquire("exclusive")
        admission_blocked = False
    except TTSProviderReconfiguringError:
        admission_blocked = True

    allow_close.set()
    assert await retry is ReconfigureResult.CHANGED
    replacement = unexpected_lease or await registry.acquire("exclusive")
    await replacement.release()
    await registry.close()

    assert retry_completed_before_cleanup is False
    assert replacement_created_before_cleanup is False
    assert admission_blocked is True
    assert factory.calls == 2
    assert factory.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_failed_exclusive_cleanup_keeps_admission_sealed() -> None:
    factory = _FailingCloseFactory("exclusive")
    spec = TTSProviderSpec(
        descriptor=provider_spec("exclusive", FakeAdapterFactory("unused")).descriptor,
        factory=factory,
        initial_config={"revision": 1},
        exclusive_reconfigure=True,
    )
    registry = TTSAdapterRegistry(specs=(spec,), aliases={})
    lease = await registry.acquire("exclusive")
    await lease.release()

    with pytest.raises(RuntimeError, match="adapter close failed"):
        await registry.reconfigure_provider("exclusive", {"revision": 2})

    retry_error: RuntimeError | None = None
    try:
        await registry.reconfigure_provider("exclusive", {"revision": 2})
    except RuntimeError as error:
        retry_error = error
    unexpected_lease = None
    try:
        unexpected_lease = await registry.acquire("exclusive")
        admission_blocked = False
    except TTSProviderReconfiguringError:
        admission_blocked = True
    if unexpected_lease is not None:
        await unexpected_lease.release()
    with pytest.raises(RuntimeError, match="adapter close failed"):
        await registry.close()
    await registry.close()

    assert str(retry_error) == "adapter close failed"
    assert admission_blocked is True
    assert factory.calls == 1


@pytest.mark.asyncio
async def test_reconfigure_is_rejected_after_shutdown_begins() -> None:
    factory = FakeAdapterFactory("openai")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("openai", factory, {"key": "first"}),),
        aliases={},
        shutdown_timeout_seconds=0.1,
    )
    lease = await registry.acquire("openai")
    close = asyncio.create_task(registry.close())
    await asyncio.sleep(0)

    with pytest.raises(TTSRegistryClosedError):
        await registry.reconfigure_provider("openai", {"key": "second"})

    await lease.release()
    await close
