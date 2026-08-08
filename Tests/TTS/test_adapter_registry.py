from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import pytest

from Tests.TTS.adapter_fakes import (
    FakeAdapter,
    FakeAdapterFactory,
    provider_spec,
)
from tldw_chatbook.TTS import adapter_registry as adapter_registry_module
from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
    TTSAdapterRegistry,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSConfigurationRevisionError,
    TTSProviderReconfiguringError,
    TTSProviderSpec,
    TTSProviderUnavailableError,
    TTSRegistryClosedError,
    UnknownTTSProviderError,
)
from tldw_chatbook.TTS.legacy_bridge import LEGACY_PROVIDER_IDS, legacy_provider_specs


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
async def test_cancelled_release_finishes_retired_adapter_cleanup() -> None:
    factory = FakeAdapterFactory("openai")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("openai", factory, {"revision": 1}),),
        aliases={},
        shutdown_timeout_seconds=0,
    )
    lease = await registry.acquire("openai")
    old_adapter = lease.adapter
    await registry.reconfigure_provider("openai", {"revision": 2})
    slot = registry._slots["openai"]
    await slot.lock.acquire()
    release = asyncio.create_task(lease.release())
    try:
        await asyncio.sleep(0)
        release.cancel()
        await asyncio.sleep(0)
        returned_while_registry_release_was_blocked = release.done()
    finally:
        slot.lock.release()

    with pytest.raises(asyncio.CancelledError):
        await release
    await lease.release()

    assert returned_while_registry_release_was_blocked is False
    assert old_adapter.close_calls == 1
    await registry.close()
    await registry.wait_closed()


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
    assert adapter.ensure_ready_calls == 0
    assert factory.calls == 1
    assert (
        await registry.reconfigure_provider("openai", {"key": "second"})
        is ReconfigureResult.CHANGED
    )
    assert adapter.close_calls == 1
    assert factory.calls == 1
    await registry.close()


@pytest.mark.asyncio
async def test_get_voices_materializes_lazily_and_releases_its_lease() -> None:
    voices_started = asyncio.Event()
    allow_voices = asyncio.Event()
    instances: list[FakeAdapter] = []

    class BlockingVoiceAdapter(FakeAdapter):
        async def get_voices(
            self,
            model_id: str,
            refresh: bool = False,
        ) -> tuple[str, ...]:
            self.get_voices_calls += 1
            self.get_voices_requests.append((model_id, refresh))
            voices_started.set()
            await allow_voices.wait()
            return ("voice-a", "voice-b")

    def factory(config: Mapping[str, Any]) -> BlockingVoiceAdapter:
        del config
        adapter = BlockingVoiceAdapter("openai")
        instances.append(adapter)
        return adapter

    registry = TTSAdapterRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=provider_spec(
                    "openai",
                    FakeAdapterFactory("unused"),
                ).descriptor,
                factory=factory,
                initial_config={"key": "first"},
            ),
        ),
        aliases={"oa": "openai"},
    )

    assert instances == []
    voice_lookup = asyncio.create_task(registry.get_voices("oa", "model", refresh=True))
    await voices_started.wait()
    adapter = instances[0]
    result = await registry.reconfigure_provider("openai", {"key": "second"})

    assert result is ReconfigureResult.CHANGED
    assert adapter.ensure_ready_calls == 0
    assert adapter.get_voices_requests == [("model", True)]
    assert adapter.close_calls == 0

    allow_voices.set()
    assert await voice_lookup == ("voice-a", "voice-b")
    assert adapter.close_calls == 1
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
async def test_exclusive_handoff_ticket_retains_old_lease_and_replaces_lazily() -> None:
    events: list[str] = []
    adapters: list[FakeAdapter] = []

    class OrderedAdapter(FakeAdapter):
        async def close(self) -> None:
            await super().close()
            events.append("old-closed")

    def factory(config: Mapping[str, Any]) -> FakeAdapter:
        events.append(f"factory-{config['revision']}")
        adapter = OrderedAdapter("audio_cpp")
        adapters.append(adapter)
        return adapter

    spec = TTSProviderSpec(
        descriptor=provider_spec(
            "audio_cpp",
            FakeAdapterFactory("unused"),
        ).descriptor,
        factory=factory,
        initial_config={"revision": 1},
        exclusive_reconfigure=True,
    )
    registry = TTSAdapterRegistry(specs=(spec,), aliases={})
    old_lease = await registry.acquire("audio_cpp")

    ticket_type = getattr(
        adapter_registry_module,
        "TTSReconfigurationTicket",
        None,
    )
    assert ticket_type is not None, "generation-aware handoff ticket is missing"
    ticket = await registry.begin_reconfigure_provider(
        "audio_cpp",
        {"revision": 2},
        generation=2,
    )
    await asyncio.sleep(0)

    assert isinstance(ticket, ticket_type)
    assert ticket.provider_id == "audio_cpp"
    assert ticket.generation == 2
    assert ticket.completion.done() is False
    with pytest.raises(TTSProviderReconfiguringError):
        await registry.acquire("audio_cpp")
    assert events == ["factory-1"]
    assert adapters[0].close_calls == 0

    await old_lease.release()
    assert await ticket.completion is ReconfigureResult.CHANGED
    assert registry.configuration_revision("audio_cpp") == 2
    assert events == ["factory-1", "old-closed"]

    replacement = await registry.acquire("audio_cpp", expected_revision=2)
    assert replacement.adapter is adapters[1]
    assert events == ["factory-1", "old-closed", "factory-2"]
    await replacement.release()
    await registry.close()
    await registry.wait_closed()


@pytest.mark.asyncio
async def test_exclusive_handoff_applies_only_latest_pending_generation() -> None:
    configs: list[dict[str, Any]] = []

    def factory(config: Mapping[str, Any]) -> FakeAdapter:
        configs.append(deepcopy(dict(config)))
        return FakeAdapter("audio_cpp")

    spec = TTSProviderSpec(
        descriptor=provider_spec(
            "audio_cpp",
            FakeAdapterFactory("unused"),
        ).descriptor,
        factory=factory,
        initial_config={"revision": 1},
        exclusive_reconfigure=True,
    )
    registry = TTSAdapterRegistry(specs=(spec,), aliases={})
    old_lease = await registry.acquire("audio_cpp")

    generation_two = await registry.begin_reconfigure_provider(
        "audio_cpp",
        {"revision": 2},
        generation=2,
    )
    generation_three = await registry.begin_reconfigure_provider(
        "audio_cpp",
        {"revision": 3},
        generation=3,
    )
    await asyncio.sleep(0)

    assert registry.configuration_revision("audio_cpp") == 1
    assert configs == [{"revision": 1}]
    assert generation_two.completion is not generation_three.completion

    await old_lease.release()
    assert await generation_two.completion is ReconfigureResult.SUPERSEDED
    assert await generation_three.completion is ReconfigureResult.CHANGED
    assert registry.configuration_revision("audio_cpp") == 2
    assert configs == [{"revision": 1}]

    replacement = await registry.acquire("audio_cpp", expected_revision=2)
    assert configs == [{"revision": 1}, {"revision": 3}]
    assert old_lease.adapter.close_calls == 1
    await replacement.release()
    await registry.close()
    await registry.wait_closed()


@pytest.mark.asyncio
async def test_new_generation_recovers_one_sealed_pending_exclusive_handoff() -> None:
    configs: list[dict[str, Any]] = []

    def factory(config: Mapping[str, Any]) -> FakeAdapter:
        configs.append(deepcopy(dict(config)))
        return FakeAdapter("audio_cpp")

    spec = TTSProviderSpec(
        descriptor=provider_spec(
            "audio_cpp",
            FakeAdapterFactory("unused"),
        ).descriptor,
        factory=factory,
        initial_config={"revision": 1},
        exclusive_reconfigure=True,
    )
    registry = TTSAdapterRegistry(specs=(spec,), aliases={})
    old_lease = await registry.acquire("audio_cpp")
    generation_two = await registry.begin_reconfigure_provider(
        "audio_cpp",
        {"revision": 2},
        generation=2,
    )
    await asyncio.sleep(0)

    await registry.seal_provider_unavailable("audio_cpp")
    with pytest.raises(TTSProviderUnavailableError):
        await registry.acquire("audio_cpp")

    generation_three = await registry.begin_reconfigure_provider(
        "audio_cpp",
        {"revision": 3},
        generation=3,
    )
    await old_lease.release()

    assert await generation_two.completion is ReconfigureResult.SUPERSEDED
    assert await generation_three.completion is ReconfigureResult.CHANGED
    assert registry.configuration_revision("audio_cpp") == 2
    assert old_lease.adapter.close_calls == 1
    assert configs == [{"revision": 1}]

    replacement = await registry.acquire("audio_cpp")
    assert configs == [{"revision": 1}, {"revision": 3}]
    await replacement.release()
    await registry.close()
    await registry.wait_closed()


@pytest.mark.asyncio
async def test_failed_retained_handoff_seals_exclusive_provider_unavailable() -> None:
    factory = _FailingCloseFactory("audio_cpp")
    spec = TTSProviderSpec(
        descriptor=provider_spec(
            "audio_cpp",
            FakeAdapterFactory("unused"),
        ).descriptor,
        factory=factory,
        initial_config={"revision": 1},
        exclusive_reconfigure=True,
    )
    registry = TTSAdapterRegistry(specs=(spec,), aliases={})
    lease = await registry.acquire("audio_cpp")
    await lease.release()

    ticket = await registry.begin_reconfigure_provider(
        "audio_cpp",
        {"revision": 2},
        generation=2,
    )
    with pytest.raises(RuntimeError, match="adapter close failed"):
        await asyncio.shield(ticket.completion)

    with pytest.raises(TTSProviderUnavailableError):
        await registry.acquire("audio_cpp")
    assert registry.configuration_revision("audio_cpp") == 1
    assert factory.calls == 1
    with pytest.raises(RuntimeError, match="adapter close failed"):
        await registry.close()
    await registry.close()


@pytest.mark.asyncio
async def test_revision_checked_acquire_rejects_stale_selection_before_factory() -> (
    None
):
    original_value = "http://private-revision-one.invalid"
    replacement_value = "PRIVATE_REVISION_TWO_CREDENTIAL"
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                factory,
                {"origin": original_value},
                exclusive=True,
            ),
        ),
        aliases={},
    )
    selected_revision = registry.configuration_revision("audio_cpp")

    assert (
        await registry.reconfigure_provider(
            "audio_cpp",
            {"credential": replacement_value},
        )
        is ReconfigureResult.CHANGED
    )
    assert registry.configuration_revision("audio_cpp") == 2

    try:
        await registry.acquire(
            "audio_cpp",
            expected_revision=selected_revision,
        )
    except BaseException as error:
        assert isinstance(error, TTSConfigurationRevisionError)
        assert str(error) == "TTS provider configuration changed: audio_cpp"
        assert error.__cause__ is None
        assert error.__context__ is None
        assert original_value not in repr(error)
        assert replacement_value not in repr(error)
    else:
        raise AssertionError("stale registry selection was admitted")

    assert factory.calls == 0
    await registry.close()


@pytest.mark.asyncio
async def test_sealed_provider_is_unavailable_until_reviewed_reconfiguration() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                factory,
                {"revision": 1},
                exclusive=True,
            ),
        ),
        aliases={},
    )

    await registry.seal_provider_unavailable("audio_cpp")
    with pytest.raises(TTSProviderUnavailableError) as unavailable:
        await registry.acquire("audio_cpp", expected_revision=0)

    assert str(unavailable.value) == "TTS provider is unavailable: audio_cpp"
    assert factory.calls == 0
    assert (
        await registry.reconfigure_provider("audio_cpp", {"revision": 2})
        is ReconfigureResult.CHANGED
    )
    assert registry.configuration_revision("audio_cpp") == 2

    lease = await registry.acquire("audio_cpp", expected_revision=2)
    assert factory.calls == 1
    await lease.release()
    await registry.close()


@pytest.mark.asyncio
async def test_reconfiguring_unavailable_and_revision_errors_are_distinct() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                factory,
                {"revision": 1},
                exclusive=True,
            ),
        ),
        aliases={},
    )
    await registry.reconfigure_provider("audio_cpp", {"revision": 2})
    with pytest.raises(TTSConfigurationRevisionError) as revision:
        await registry.acquire("audio_cpp", expected_revision=1)

    await registry.seal_provider_unavailable("audio_cpp")
    with pytest.raises(TTSProviderUnavailableError) as unavailable:
        await registry.acquire("audio_cpp", expected_revision=1)

    await registry.reconfigure_provider("audio_cpp", {"revision": 3})
    lease = await registry.acquire("audio_cpp", expected_revision=3)
    reconfigure = asyncio.create_task(
        registry.reconfigure_provider("audio_cpp", {"revision": 4})
    )
    await asyncio.sleep(0)
    with pytest.raises(TTSProviderReconfiguringError) as reconfiguring:
        await registry.acquire("audio_cpp", expected_revision=2)

    assert type(revision.value) is TTSConfigurationRevisionError
    assert type(unavailable.value) is TTSProviderUnavailableError
    assert type(reconfiguring.value) is TTSProviderReconfiguringError
    assert str(revision.value) == "TTS provider configuration changed: audio_cpp"
    assert str(unavailable.value) == "TTS provider is unavailable: audio_cpp"
    assert str(reconfiguring.value) == "TTS provider is reconfiguring: audio_cpp"

    await lease.release()
    assert await reconfigure is ReconfigureResult.CHANGED
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
    with pytest.raises(TTSProviderUnavailableError):
        await registry.acquire("exclusive")

    await registry.seal_provider_unavailable("exclusive")
    with pytest.raises(TTSProviderUnavailableError) as unavailable:
        await registry.acquire("exclusive")
    with pytest.raises(RuntimeError, match="adapter close failed"):
        await registry.close()
    await registry.close()

    assert str(retry_error) == "adapter close failed"
    assert str(unavailable.value) == "TTS provider is unavailable: exclusive"
    assert unavailable.value.__cause__ is None
    assert unavailable.value.__context__ is None
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


@pytest.mark.asyncio
async def test_registry_serves_configuration_revision_for_legacy_providers() -> None:
    """Characterize the REAL adapter registry, not a test fake.

    Slice 1's ``create_from_artifact`` legacy acceptance path (Task 2c) rests
    on the assumption that every legacy provider registered through
    ``legacy_provider_specs`` exposes a ``configuration_revision``. This pins
    that assumption against the production registry construction rather than
    ``FakeAdapterFactory``.
    """
    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs({}),
        aliases={},
    )

    for provider_id in LEGACY_PROVIDER_IDS:
        revision = registry.configuration_revision(provider_id)
        assert type(revision) is int and revision >= 0

    await registry.close()


@pytest.mark.asyncio
async def test_stage_keeps_applied_config_revision_and_active_adapter() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                factory,
                {"mode": "external", "nested": {"value": 1}},
                exclusive=True,
            ),
        ),
        aliases={},
    )
    lease = await registry.acquire("audio_cpp")
    adapter = lease.adapter
    await lease.release()

    result = await registry.stage_provider_configuration(
        "audio_cpp",
        {"mode": "managed", "nested": {"value": 2}},
        generation=1,
    )
    snapshot = await registry.provider_configuration_snapshot("audio_cpp")

    assert result is ReconfigureResult.CHANGED
    assert snapshot.revision == 1
    assert snapshot.applied_generation == 0
    assert snapshot.applied_config == {
        "mode": "external",
        "nested": {"value": 1},
    }
    assert snapshot.staged_generation == 1
    assert snapshot.staged_config == {
        "mode": "managed",
        "nested": {"value": 2},
    }
    assert factory.calls == 1
    assert adapter.close_calls == 0
    await registry.close()


@pytest.mark.asyncio
async def test_newer_stage_supersedes_older_without_starting_handoff() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", factory, {"revision": 0}, exclusive=True),),
        aliases={},
    )

    first = await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 1}, generation=1
    )
    second = await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 2}, generation=2
    )
    superseded = await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 3}, generation=1
    )
    snapshot = await registry.provider_configuration_snapshot("audio_cpp")

    assert first is ReconfigureResult.CHANGED
    assert second is ReconfigureResult.CHANGED
    assert superseded is ReconfigureResult.SUPERSEDED
    assert snapshot.staged_generation == 2
    assert snapshot.staged_config == {"revision": 2}
    assert snapshot.revision == 1
    assert factory.calls == 0
    await registry.close()


@pytest.mark.asyncio
async def test_equal_config_advances_generation_without_restart_required() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                factory,
                {"mode": "external", "options": {"timeout": 10}},
                exclusive=True,
            ),
        ),
        aliases={},
    )

    result = await registry.stage_provider_configuration(
        "audio_cpp",
        {"mode": "external", "options": {"timeout": 10}},
        generation=4,
    )
    snapshot = await registry.provider_configuration_snapshot("audio_cpp")

    assert result is ReconfigureResult.UNCHANGED
    assert snapshot.revision == 1
    assert snapshot.applied_generation == 4
    assert snapshot.staged_generation is None
    assert snapshot.staged_config is None
    assert factory.calls == 0
    await registry.close()


@pytest.mark.asyncio
async def test_reverting_to_applied_config_clears_an_older_stage() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", factory, {"revision": 0}, exclusive=True),),
        aliases={},
    )
    await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 1}, generation=1
    )

    result = await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 0}, generation=2
    )
    snapshot = await registry.provider_configuration_snapshot("audio_cpp")

    assert result is ReconfigureResult.UNCHANGED
    assert snapshot.applied_generation == 2
    assert snapshot.applied_config == {"revision": 0}
    assert snapshot.staged_generation is None
    assert snapshot.staged_config is None
    assert factory.calls == 0
    await registry.close()


@pytest.mark.asyncio
async def test_immediate_reconfigure_clears_every_older_stage() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", factory, {"revision": 0}, exclusive=True),),
        aliases={},
    )
    await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 1}, generation=1
    )

    ticket = await registry.begin_reconfigure_provider(
        "audio_cpp", {"revision": 2}, generation=2
    )
    assert await ticket.completion is ReconfigureResult.CHANGED
    snapshot = await registry.provider_configuration_snapshot("audio_cpp")

    assert snapshot.applied_generation == 2
    assert snapshot.applied_config == {"revision": 2}
    assert snapshot.staged_generation is None
    assert snapshot.staged_config is None
    await registry.close()


@pytest.mark.asyncio
async def test_configuration_snapshot_is_deeply_immutable() -> None:
    source = {
        "nested": {"items": ["first", {"leaf": "value"}]},
        "set": {"one", "two"},
    }
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                FakeAdapterFactory("audio_cpp"),
                {"applied": {"items": [1, 2]}},
                exclusive=True,
            ),
        ),
        aliases={},
    )
    await registry.stage_provider_configuration("audio_cpp", source, generation=1)
    snapshot = await registry.provider_configuration_snapshot("audio_cpp")
    source["nested"]["items"].append("mutated")

    with pytest.raises(TypeError):
        snapshot.applied_config["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.staged_config["nested"]["new"] = "value"  # type: ignore[index,union-attr]
    with pytest.raises(AttributeError):
        snapshot.staged_config["nested"]["items"].append("value")  # type: ignore[union-attr]
    assert snapshot.staged_config == {
        "nested": {"items": ("first", {"leaf": "value"})},
        "set": frozenset({"one", "two"}),
    }
    await registry.close()


@pytest.mark.asyncio
async def test_transition_rejects_new_leases_and_drains_admitted_lease() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", factory, {"revision": 0}, exclusive=True),),
        aliases={},
    )
    lease = await registry.acquire("audio_cpp")
    await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 1}, generation=1
    )
    draining = asyncio.Event()
    action_calls = 0

    async def on_draining() -> None:
        draining.set()

    async def action() -> None:
        nonlocal action_calls
        action_calls += 1

    transition = asyncio.create_task(
        registry.run_exclusive_provider_transition(
            "audio_cpp",
            on_draining=on_draining,
            action=action,
            apply_staged=True,
        )
    )
    await draining.wait()

    with pytest.raises(TTSProviderReconfiguringError):
        await registry.acquire("audio_cpp")
    assert transition.done() is False
    assert action_calls == 0

    await lease.release()
    assert await transition is ReconfigureResult.CHANGED
    assert action_calls == 1
    assert lease.adapter.close_calls == 1
    await registry.close()


@pytest.mark.asyncio
async def test_transition_publishes_draining_before_waiting_for_admitted_lease() -> (
    None
):
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                FakeAdapterFactory("audio_cpp"),
                {"revision": 0},
                exclusive=True,
            ),
        ),
        aliases={},
    )
    lease = await registry.acquire("audio_cpp")
    observed_leases: list[int] = []
    draining = asyncio.Event()

    async def on_draining() -> None:
        observed_leases.append(registry._total_leases())
        draining.set()

    transition = asyncio.create_task(
        registry.run_exclusive_provider_transition(
            "audio_cpp",
            on_draining=on_draining,
            action=lambda: asyncio.sleep(0),
            apply_staged=False,
        )
    )
    await draining.wait()

    assert observed_leases == [1]
    assert transition.done() is False
    await lease.release()
    assert await transition is ReconfigureResult.UNCHANGED
    await registry.close()


@pytest.mark.asyncio
async def test_transition_action_runs_after_last_lease_without_holding_slot_lock() -> (
    None
):
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                FakeAdapterFactory("audio_cpp"),
                {"revision": 0},
                exclusive=True,
            ),
        ),
        aliases={},
    )
    lease = await registry.acquire("audio_cpp")
    draining = asyncio.Event()
    action_observations: list[int] = []

    async def on_draining() -> None:
        draining.set()

    async def action() -> None:
        slot = registry._slots["audio_cpp"]
        async with slot.lock:
            action_observations.append(lease.adapter.close_calls)

    transition = asyncio.create_task(
        registry.run_exclusive_provider_transition(
            "audio_cpp",
            on_draining=on_draining,
            action=action,
            apply_staged=False,
        )
    )
    await draining.wait()
    await lease.release()

    assert await transition is ReconfigureResult.UNCHANGED
    assert action_observations == [0]
    await registry.close()


@pytest.mark.asyncio
async def test_transition_promotes_only_latest_staged_config() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", factory, {"revision": 0}, exclusive=True),),
        aliases={},
    )
    lease = await registry.acquire("audio_cpp")
    await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 1}, generation=1
    )
    draining = asyncio.Event()

    async def on_draining() -> None:
        draining.set()

    transition = asyncio.create_task(
        registry.run_exclusive_provider_transition(
            "audio_cpp",
            on_draining=on_draining,
            action=lambda: asyncio.sleep(0),
            apply_staged=True,
        )
    )
    await draining.wait()
    await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 2}, generation=2
    )
    await lease.release()

    assert await transition is ReconfigureResult.CHANGED
    snapshot = await registry.provider_configuration_snapshot("audio_cpp")
    assert snapshot.applied_generation == 2
    assert snapshot.applied_config == {"revision": 2}
    assert snapshot.staged_config is None
    assert registry.configuration_revision("audio_cpp") == 2
    await registry.close()


@pytest.mark.asyncio
async def test_transition_without_stage_keeps_config_revision_and_adapter() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", factory, {"revision": 0}, exclusive=True),),
        aliases={},
    )
    first = await registry.acquire("audio_cpp")
    adapter = first.adapter
    await first.release()

    result = await registry.run_exclusive_provider_transition(
        "audio_cpp",
        on_draining=lambda: asyncio.sleep(0),
        action=lambda: asyncio.sleep(0),
        apply_staged=True,
    )
    second = await registry.acquire("audio_cpp")

    assert result is ReconfigureResult.UNCHANGED
    assert registry.configuration_revision("audio_cpp") == 1
    assert second.adapter is adapter
    assert adapter.close_calls == 0
    await second.release()
    await registry.close()


@pytest.mark.asyncio
async def test_transition_failure_seals_provider_unavailable_and_releases_waiters() -> (
    None
):
    private_detail = "SYNTHETIC_PRIVATE_ACTION_FAILURE"
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", factory, {"revision": 0}, exclusive=True),),
        aliases={},
    )
    lease = await registry.acquire("audio_cpp")
    await lease.release()
    await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 1}, generation=1
    )
    action_started = asyncio.Event()
    release_action = asyncio.Event()

    async def action() -> None:
        action_started.set()
        await release_action.wait()
        raise RuntimeError(private_detail)

    transition = asyncio.create_task(
        registry.run_exclusive_provider_transition(
            "audio_cpp",
            on_draining=lambda: asyncio.sleep(0),
            action=action,
            apply_staged=True,
        )
    )
    await action_started.wait()
    with pytest.raises(TTSProviderReconfiguringError):
        await registry.acquire("audio_cpp")
    release_action.set()

    with pytest.raises(RuntimeError, match=private_detail):
        await transition
    with pytest.raises(TTSProviderUnavailableError) as unavailable:
        await registry.acquire("audio_cpp")
    snapshot = await registry.provider_configuration_snapshot("audio_cpp")
    assert str(unavailable.value) == "TTS provider is unavailable: audio_cpp"
    assert snapshot.staged_config == {"revision": 1}
    assert registry._slots["audio_cpp"].reconfiguring is False
    await registry.close()


@pytest.mark.asyncio
async def test_action_failure_retry_clears_unavailable_after_success() -> None:
    private_detail = "SYNTHETIC_FIRST_ACTION_FAILURE"
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", factory, {"revision": 0}, exclusive=True),),
        aliases={},
    )
    lease = await registry.acquire("audio_cpp")
    old_adapter = lease.adapter
    await lease.release()
    await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 1}, generation=1
    )

    async def failing_action() -> None:
        raise RuntimeError(private_detail)

    with pytest.raises(RuntimeError, match=private_detail):
        await registry.run_exclusive_provider_transition(
            "audio_cpp",
            on_draining=lambda: asyncio.sleep(0),
            action=failing_action,
            apply_staged=True,
        )

    result = await registry.run_exclusive_provider_transition(
        "audio_cpp",
        on_draining=lambda: asyncio.sleep(0),
        action=lambda: asyncio.sleep(0),
        apply_staged=True,
    )
    replacement = await registry.acquire("audio_cpp")

    assert result is ReconfigureResult.CHANGED
    assert old_adapter.close_calls == 1
    assert replacement.adapter is not old_adapter
    assert registry.configuration_revision("audio_cpp") == 2
    await replacement.release()
    await registry.close()


@pytest.mark.asyncio
async def test_adapter_close_failure_retains_record_and_uses_fresh_retry_task() -> None:
    class RetryCloseAdapter(FakeAdapter):
        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("synthetic close failure")

    adapter = RetryCloseAdapter("audio_cpp")
    factory_calls = 0

    def factory(_config: Mapping[str, Any]) -> FakeAdapter:
        nonlocal factory_calls
        factory_calls += 1
        return adapter if factory_calls == 1 else FakeAdapter("audio_cpp")

    spec = TTSProviderSpec(
        descriptor=provider_spec("audio_cpp", FakeAdapterFactory("unused")).descriptor,
        factory=factory,
        initial_config={"revision": 0},
        exclusive_reconfigure=True,
    )
    registry = TTSAdapterRegistry(specs=(spec,), aliases={})
    lease = await registry.acquire("audio_cpp")
    await lease.release()
    await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 1}, generation=1
    )

    with pytest.raises(RuntimeError, match="synthetic close failure"):
        await registry.run_exclusive_provider_transition(
            "audio_cpp",
            on_draining=lambda: asyncio.sleep(0),
            action=lambda: asyncio.sleep(0),
            apply_staged=True,
        )
    result = await registry.run_exclusive_provider_transition(
        "audio_cpp",
        on_draining=lambda: asyncio.sleep(0),
        action=lambda: asyncio.sleep(0),
        apply_staged=True,
    )

    assert result is ReconfigureResult.CHANGED
    assert adapter.close_calls == 2
    assert registry.configuration_revision("audio_cpp") == 2
    await registry.close()


@pytest.mark.asyncio
async def test_successful_adapter_close_is_never_repeated_on_retry() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", factory, {"revision": 0}, exclusive=True),),
        aliases={},
    )
    lease = await registry.acquire("audio_cpp")
    old_adapter = lease.adapter
    await lease.release()
    await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 1}, generation=1
    )

    async def failing_action() -> None:
        raise RuntimeError("synthetic action failure")

    with pytest.raises(RuntimeError, match="synthetic action failure"):
        await registry.run_exclusive_provider_transition(
            "audio_cpp",
            on_draining=lambda: asyncio.sleep(0),
            action=failing_action,
            apply_staged=True,
        )
    assert old_adapter.close_calls == 1

    assert (
        await registry.run_exclusive_provider_transition(
            "audio_cpp",
            on_draining=lambda: asyncio.sleep(0),
            action=lambda: asyncio.sleep(0),
            apply_staged=True,
        )
        is ReconfigureResult.CHANGED
    )
    assert old_adapter.close_calls == 1
    await registry.close()


@pytest.mark.asyncio
async def test_registry_close_joins_an_in_progress_transition() -> None:
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", factory, {"revision": 0}, exclusive=True),),
        aliases={},
    )
    lease = await registry.acquire("audio_cpp")
    await lease.release()
    await registry.stage_provider_configuration(
        "audio_cpp", {"revision": 1}, generation=1
    )
    action_started = asyncio.Event()
    release_action = asyncio.Event()

    async def action() -> None:
        action_started.set()
        await release_action.wait()

    transition = asyncio.create_task(
        registry.run_exclusive_provider_transition(
            "audio_cpp",
            on_draining=lambda: asyncio.sleep(0),
            action=action,
            apply_staged=True,
        )
    )
    await action_started.wait()
    close = asyncio.create_task(registry.close())
    await asyncio.sleep(0)

    assert close.done() is False
    release_action.set()
    assert await transition is ReconfigureResult.CHANGED
    await close
    await registry.wait_closed()
    assert lease.adapter.close_calls == 1
