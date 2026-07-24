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
    exclusive_record: _AdapterRecord | None = None
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

    async def close(self) -> None:
        """Seal admission and wait no longer than the shutdown timeout.

        Adapter cleanup continues in one retained task after a timeout. Call
        :meth:`wait_closed` to join definitive completion.
        """
        close_task = await self._ensure_close_task()
        if not close_task.done():
            await asyncio.wait(
                {close_task},
                timeout=self._shutdown_timeout_seconds,
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

            self._close_task = asyncio.create_task(self._complete_close())
            self._close_task.add_done_callback(self._observe_close_result)
            return self._close_task

    async def _complete_close(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._shutdown_timeout_seconds
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
            await self._start_close_record(record) for record in self._closing_records
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
        async with slot.transition_lock:
            if self._closed:
                raise TTSRegistryClosedError("The TTS registry is closed")
            async with slot.lock:
                if slot.config == new_config:
                    return ReconfigureResult.UNCHANGED
                close_record = slot.active
                slot.active = None
                slot.config = new_config
                slot.revision += 1
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

    async def _reconfigure_exclusive(
        self, slot: _ProviderSlot, new_config: dict[str, Any]
    ) -> ReconfigureResult:
        async with slot.transition_lock:
            if self._closed:
                raise TTSRegistryClosedError("The TTS registry is closed")
            async with slot.lock:
                if slot.reconfiguring:
                    old_record = slot.exclusive_record
                else:
                    if slot.config == new_config:
                        return ReconfigureResult.UNCHANGED
                    slot.reconfiguring = True
                    old_record = slot.active
                    slot.exclusive_record = old_record

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
                if old_record is not None and old_record in slot.retired:
                    slot.retired.remove(old_record)
                slot.config = new_config
                slot.revision += 1
                slot.reconfiguring = False
                slot.exclusive_record = None

        return ReconfigureResult.CHANGED

    async def _close_record(self, record: _AdapterRecord) -> None:
        close_task = await self._start_close_record(record)
        await asyncio.shield(close_task)

    async def _start_close_record(self, record: _AdapterRecord) -> asyncio.Task[None]:
        async with record.close_lock:
            if record.close_task is None:
                record.close_task = asyncio.create_task(record.adapter.close())
            return record.close_task

    def _total_leases(self) -> int:
        total = 0
        for slot in self._slots.values():
            if slot.active is not None:
                total += slot.active.leases
            total += sum(record.leases for record in slot.retired)
        return total
