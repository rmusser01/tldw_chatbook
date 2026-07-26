from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tldw_chatbook.TTS._async_lifecycle import (
    join_retained_task,
    shutdown_deadline_scope,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSAdapter,
    TTSConfigurationRevisionError,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderReconfiguringError,
    TTSProviderSpec,
    TTSProviderUnavailableError,
    TTSRegistryClosedError,
    UnknownTTSProviderError,
)


class ReconfigureResult(Enum):
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class TTSReconfigurationTicket:
    """One generation-scoped view of a retained provider handoff."""

    provider_id: str
    generation: int
    completion: asyncio.Task[ReconfigureResult]


@dataclass(slots=True)
class _AdapterRecord:
    adapter: TTSAdapter
    leases: int = 0
    retired: bool = False
    close_task: asyncio.Task[None] | None = None
    close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class _ProviderSlot:
    spec: TTSProviderSpec
    config: dict[str, Any]
    revision: int = 1
    active: _AdapterRecord | None = None
    retired: list[_AdapterRecord] = field(default_factory=list)
    reconfiguring: bool = False
    unavailable: bool = False
    exclusive_record: _AdapterRecord | None = None
    pending_config: dict[str, Any] | None = None
    pending_generation: int | None = None
    applied_generation: int = 0
    highest_generation: int = 0
    sealed_generation: int = 0
    applying_generation: int | None = None
    handoff_task: asyncio.Task[ReconfigureResult] | None = None
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
        self._release_task: asyncio.Future[None] | None = None

    async def release(self) -> None:
        if self._release_task is None:
            self._release_task = asyncio.ensure_future(self._release_callback())
        await join_retained_task(self._release_task)

    async def __aenter__(self) -> TTSAdapterLease:
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
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._shutdown_deadline: float | None = None
        self._close_error_reported = False
        self._lease_changed = asyncio.Event()
        self._records_collected = False
        self._closing_records: list[_AdapterRecord] = []

    def descriptors(self) -> tuple[TTSProviderDescriptor, ...]:
        return tuple(slot.spec.descriptor for slot in self._slots.values())

    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def configuration_revision(self, provider_id: str) -> int:
        return self._slots[self._resolve_id(provider_id)].revision

    def configuration_generation(self, provider_id: str) -> int:
        """Return the latest settings generation applied to one provider."""
        return self._slots[self._resolve_id(provider_id)].applied_generation

    async def acquire(
        self,
        provider_id: str,
        *,
        expected_revision: int | None = None,
    ) -> TTSAdapterLease:
        canonical_id = self._resolve_id(provider_id)
        if self._closed:
            raise TTSRegistryClosedError("The TTS registry is closed")

        slot = self._slots[canonical_id]
        async with slot.lock:
            if self._closed:
                raise TTSRegistryClosedError("The TTS registry is closed")
            if slot.unavailable:
                raise TTSProviderUnavailableError(
                    f"TTS provider is unavailable: {canonical_id}"
                )
            if slot.reconfiguring:
                raise TTSProviderReconfiguringError(
                    f"TTS provider is reconfiguring: {canonical_id}"
                )
            if expected_revision is not None and slot.revision != expected_revision:
                raise TTSConfigurationRevisionError(
                    f"TTS provider configuration changed: {canonical_id}"
                )
            if slot.active is None:
                slot.active = _AdapterRecord(
                    adapter=slot.spec.factory(deepcopy(slot.config))
                )
            record = slot.active
            record.leases += 1

        async def release() -> None:
            await self._release(slot, record)

        return TTSAdapterLease(canonical_id, record.adapter, release)

    async def get_catalog(
        self, provider_id: str, refresh: bool = False
    ) -> TTSProviderCatalog:
        lease = await self.acquire(provider_id)
        try:
            return await lease.adapter.get_catalog(refresh=refresh)
        finally:
            await lease.release()

    async def get_voices(
        self,
        provider_id: str,
        model_id: str,
        refresh: bool = False,
    ) -> tuple[str, ...]:
        lease = await self.acquire(provider_id)
        try:
            return await lease.adapter.get_voices(model_id, refresh=refresh)
        finally:
            await lease.release()

    async def reconfigure_provider(
        self, provider_id: str, config: Mapping[str, Any]
    ) -> ReconfigureResult:
        """Apply one provider config and await its definitive retained result."""
        ticket = await self.begin_reconfigure_provider(provider_id, config)
        return await asyncio.shield(ticket.completion)

    async def begin_reconfigure_provider(
        self,
        provider_id: str,
        config: Mapping[str, Any],
        *,
        generation: int | None = None,
    ) -> TTSReconfigurationTicket:
        """Begin a retained, generation-aware provider reconfiguration."""
        canonical_id = self._resolve_id(provider_id)
        if self._closed:
            raise TTSRegistryClosedError("The TTS registry is closed")
        if not isinstance(config, Mapping):
            raise TypeError("TTS provider configuration must be a mapping")
        if generation is not None and (
            isinstance(generation, bool) or not isinstance(generation, int)
        ):
            raise TypeError("TTS reconfiguration generation must be an integer")
        if generation is not None and generation < 1:
            raise ValueError("TTS reconfiguration generation must be positive")

        slot = self._slots[canonical_id]
        new_config = deepcopy(dict(config))
        async with slot.lock:
            if self._closed:
                raise TTSRegistryClosedError("The TTS registry is closed")
            selected_generation = (
                slot.highest_generation + 1 if generation is None else generation
            )

        if slot.spec.exclusive_reconfigure:
            return await self._begin_exclusive_reconfiguration(
                canonical_id,
                slot,
                new_config,
                selected_generation,
            )
        return await self._begin_retiring_reconfiguration(
            canonical_id,
            slot,
            new_config,
            selected_generation,
        )

    async def seal_provider_unavailable(self, provider_id: str) -> None:
        """Seal a provider slot after a reviewed handoff fails.

        Args:
            provider_id: Exact provider identifier or registered alias.
        """
        canonical_id = self._resolve_id(provider_id)
        if self._closed:
            raise TTSRegistryClosedError("The TTS registry is closed")

        slot = self._slots[canonical_id]
        async with slot.lock:
            if self._closed:
                raise TTSRegistryClosedError("The TTS registry is closed")
            slot.reconfiguring = (
                slot.handoff_task is not None and not slot.handoff_task.done()
            )
            slot.unavailable = True
            slot.sealed_generation = max(
                slot.sealed_generation,
                slot.highest_generation,
            )

    async def close(self) -> None:
        """Seal admission and wait no longer than the shutdown timeout.

        Adapter cleanup continues in one retained task after a timeout. Call
        :meth:`wait_closed` to join definitive completion.
        """
        close_task = await self._ensure_close_task()
        deadline = self._shutdown_deadline
        if not close_task.done() and deadline is not None:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            await asyncio.sleep(0)
            await asyncio.wait(
                {close_task},
                timeout=remaining,
            )

        async with self._close_lock:
            if close_task.done():
                if self._close_error_reported:
                    return
                try:
                    close_task.result()
                except BaseException:
                    self._close_error_reported = True
                    raise
                return

            known_error = self._known_close_error()
            if known_error is not None:
                raise known_error

    async def wait_closed(self) -> None:
        """Wait for retained shutdown work and report its definitive result."""
        close_task = await self._ensure_close_task()
        await asyncio.shield(close_task)

    async def _ensure_close_task(self) -> asyncio.Task[None]:
        async with self._close_lock:
            if self._close_task is not None:
                return self._close_task

            self._closed = True
            self._lease_changed.set()
            for slot in self._slots.values():
                slot.lease_changed.set()

            deadline = (
                asyncio.get_running_loop().time() + self._shutdown_timeout_seconds
            )
            self._shutdown_deadline = deadline
            self._close_task = asyncio.create_task(self._complete_close(deadline))
            self._close_task.add_done_callback(self._observe_close_result)
            return self._close_task

    async def _complete_close(self, deadline: float) -> None:
        loop = asyncio.get_running_loop()
        if not self._records_collected:
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

            for slot in self._slots.values():
                async with slot.lock:
                    if slot.active is not None:
                        self._closing_records.append(slot.active)
                    self._closing_records.extend(slot.retired)
                    slot.active = None
                    slot.retired = []
            self._records_collected = True

        close_tasks = [
            await self._start_close_record(
                record,
                shutdown_deadline=deadline,
            )
            for record in self._closing_records
        ]
        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _known_close_error(self) -> BaseException | None:
        for record in self._closing_records:
            close_task = record.close_task
            if close_task is None or not close_task.done():
                continue
            try:
                close_task.result()
            except BaseException as error:
                return error
        return None

    @staticmethod
    def _observe_close_result(close_task: asyncio.Task[None]) -> None:
        try:
            close_task.exception()
        except BaseException:
            pass

    def _resolve_id(self, provider_id: str) -> str:
        canonical_id = self._aliases.get(provider_id, provider_id)
        if canonical_id not in self._slots:
            raise UnknownTTSProviderError(f"Unknown TTS provider: {provider_id}")
        return canonical_id

    async def _release(self, slot: _ProviderSlot, record: _AdapterRecord) -> None:
        close_record = False
        async with slot.lock:
            if record.leases == 0:
                return
            record.leases -= 1
            if record.leases == 0:
                slot.lease_changed.set()
                self._lease_changed.set()
                close_record = record.retired

        if close_record:
            await self._close_record(record)
            async with slot.lock:
                if record in slot.retired:
                    slot.retired.remove(record)

    async def _reconfigure_retiring(
        self, slot: _ProviderSlot, new_config: dict[str, Any]
    ) -> ReconfigureResult:
        close_record: _AdapterRecord | None = None
        if self._closed:
            raise TTSRegistryClosedError("The TTS registry is closed")
        async with slot.lock:
            if (
                slot.unavailable
                and slot.applying_generation is not None
                and slot.applying_generation <= slot.sealed_generation
            ):
                return ReconfigureResult.SUPERSEDED
            if slot.config == new_config and not slot.unavailable:
                return ReconfigureResult.UNCHANGED
            close_record = slot.active
            slot.active = None
            slot.config = new_config
            slot.revision += 1
            slot.unavailable = False
            if close_record is not None:
                close_record.retired = True
                slot.retired.append(close_record)
                if close_record.leases > 0:
                    close_record = None

        if close_record is not None:
            await self._close_record(close_record)
            async with slot.lock:
                if close_record in slot.retired:
                    slot.retired.remove(close_record)

        return ReconfigureResult.CHANGED

    async def _begin_retiring_reconfiguration(
        self,
        provider_id: str,
        slot: _ProviderSlot,
        new_config: dict[str, Any],
        generation: int,
    ) -> TTSReconfigurationTicket:
        async with slot.lock:
            if generation <= slot.highest_generation:
                return self._completed_reconfiguration_ticket(
                    provider_id,
                    generation,
                    ReconfigureResult.SUPERSEDED,
                )
            slot.highest_generation = generation

        async def apply() -> ReconfigureResult:
            async with slot.transition_lock:
                async with slot.lock:
                    if generation < slot.highest_generation or (
                        slot.unavailable and generation <= slot.sealed_generation
                    ):
                        return ReconfigureResult.SUPERSEDED
                    slot.applying_generation = generation
                try:
                    result = await self._reconfigure_retiring(slot, new_config)
                finally:
                    async with slot.lock:
                        if slot.applying_generation == generation:
                            slot.applying_generation = None
                async with slot.lock:
                    if result is not ReconfigureResult.SUPERSEDED:
                        slot.applied_generation = generation
                return result

        completion = asyncio.create_task(apply())
        completion.add_done_callback(self._observe_task_result)
        return TTSReconfigurationTicket(provider_id, generation, completion)

    async def _begin_exclusive_reconfiguration(
        self,
        provider_id: str,
        slot: _ProviderSlot,
        new_config: dict[str, Any],
        generation: int,
    ) -> TTSReconfigurationTicket:
        async with slot.lock:
            if generation <= slot.highest_generation:
                return self._completed_reconfiguration_ticket(
                    provider_id,
                    generation,
                    ReconfigureResult.SUPERSEDED,
                )
            slot.highest_generation = generation

            if slot.reconfiguring:
                slot.pending_config = new_config
                slot.pending_generation = generation
                if slot.unavailable and generation > slot.sealed_generation:
                    slot.unavailable = False
                handoff_task = slot.handoff_task
                assert handoff_task is not None
            elif slot.config == new_config and not slot.unavailable:
                slot.applied_generation = generation
                return self._completed_reconfiguration_ticket(
                    provider_id,
                    generation,
                    ReconfigureResult.UNCHANGED,
                )
            else:
                recovering_unavailable = slot.unavailable
                slot.reconfiguring = True
                slot.unavailable = False
                slot.pending_config = new_config
                slot.pending_generation = generation
                old_record = (
                    slot.exclusive_record if recovering_unavailable else slot.active
                )
                if old_record is None:
                    old_record = slot.active
                slot.exclusive_record = old_record
                handoff_task = asyncio.create_task(
                    self._complete_exclusive_handoff(slot)
                )
                handoff_task.add_done_callback(self._observe_task_result)
                slot.handoff_task = handoff_task

        async def ticket_result() -> ReconfigureResult:
            await asyncio.shield(handoff_task)
            async with slot.lock:
                if slot.applied_generation != generation:
                    return ReconfigureResult.SUPERSEDED
            return ReconfigureResult.CHANGED

        completion = asyncio.create_task(ticket_result())
        completion.add_done_callback(self._observe_task_result)
        return TTSReconfigurationTicket(provider_id, generation, completion)

    async def _complete_exclusive_handoff(
        self,
        slot: _ProviderSlot,
    ) -> ReconfigureResult:
        old_record = slot.exclusive_record
        try:
            if old_record is not None:
                while True:
                    async with slot.lock:
                        if old_record.leases == 0:
                            if slot.active is old_record:
                                slot.active = None
                            old_record.retired = True
                            if old_record not in slot.retired:
                                slot.retired.append(old_record)
                            break
                        slot.lease_changed.clear()
                    await slot.lease_changed.wait()
                await self._close_record(old_record)

            async with slot.lock:
                if slot.unavailable:
                    raise TTSProviderUnavailableError(
                        f"TTS provider is unavailable: "
                        f"{slot.spec.descriptor.provider_id}"
                    )
                pending_config = slot.pending_config
                pending_generation = slot.pending_generation
                if pending_config is None or pending_generation is None:
                    raise RuntimeError("TTS provider handoff state is unavailable")
                if old_record is not None and old_record in slot.retired:
                    slot.retired.remove(old_record)
                slot.config = pending_config
                slot.revision += 1
                slot.applied_generation = pending_generation
                slot.pending_config = None
                slot.pending_generation = None
                slot.reconfiguring = False
                slot.unavailable = False
                slot.exclusive_record = None
                slot.handoff_task = None
            return ReconfigureResult.CHANGED
        except BaseException:
            async with slot.lock:
                slot.pending_config = None
                slot.pending_generation = None
                slot.reconfiguring = False
                slot.unavailable = True
                if slot.handoff_task is asyncio.current_task():
                    slot.handoff_task = None
            raise

    @staticmethod
    def _completed_reconfiguration_ticket(
        provider_id: str,
        generation: int,
        result: ReconfigureResult,
    ) -> TTSReconfigurationTicket:
        async def completed() -> ReconfigureResult:
            return result

        completion = asyncio.create_task(completed())
        return TTSReconfigurationTicket(provider_id, generation, completion)

    @staticmethod
    def _observe_task_result(task: asyncio.Task[Any]) -> None:
        try:
            task.exception()
        except BaseException:
            pass

    async def _close_record(self, record: _AdapterRecord) -> None:
        close_task = await self._start_close_record(record)
        await asyncio.shield(close_task)

    async def _start_close_record(
        self,
        record: _AdapterRecord,
        *,
        shutdown_deadline: float | None = None,
    ) -> asyncio.Task[None]:
        async with record.close_lock:
            if record.close_task is None:
                with shutdown_deadline_scope(shutdown_deadline):
                    record.close_task = asyncio.create_task(record.adapter.close())
            return record.close_task

    def _total_leases(self) -> int:
        total = 0
        for slot in self._slots.values():
            if slot.active is not None:
                total += slot.active.leases
            total += sum(record.leases for record in slot.retired)
        return total
