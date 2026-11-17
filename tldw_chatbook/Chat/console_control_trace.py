"""Retain each original trace call's source across its existing retry queue."""

from __future__ import annotations


class _TraceWriteOwner:
    def __init__(self, dispatch, boundary):
        self.dispatch = dispatch
        self.boundary = boundary
        self.service = boundary.service
        self.coordinator = boundary.service._settlement
        self.database = boundary.database
        self.call_id = boundary._started.call_id
        self.worker_futures = []

    def require(self, coordinator, database, call_id):
        controls = self.dispatch.controls
        if (
            self.boundary._control_write_owner is not self
            or self.dispatch.trace_owner is not self
            or self.boundary.service is not self.service
            or self.service._settlement is not coordinator
            or self.coordinator is not coordinator
            or self.boundary.database is not database
            or self.database is not database
            or database is not controls._domain.sources.chat_database
            or self.boundary._started.call_id != call_id
            or self.call_id != call_id
            or controls._gateway is not self.dispatch.gateway
        ):
            raise RuntimeError("Console trace write owner changed")


def bind_trace_call(dispatch, boundary):
    from .console_trace_service import ConsoleTraceCallBoundary
    from .console_trace_runtime import ConsoleTraceBoundaryFactory
    from .console_runtime import _LazyTraceBoundaryFactory

    factory = dispatch.gateway._trace_call_boundary_factory
    if type(factory) is _LazyTraceBoundaryFactory:
        factory = factory._delegate

    if (
        type(boundary) is not ConsoleTraceCallBoundary
        or boundary._started is None
        or boundary.database is not dispatch.controls._domain.sources.chat_database
        or boundary._control_write_owner is not None
        or dispatch.trace_owner is not None
        or dispatch.finished
        or type(factory) is not ConsoleTraceBoundaryFactory
        or boundary._factory is not factory
        or factory.service is not boundary.service
        or factory.database is not boundary.database
    ):
        raise RuntimeError("Console trace call boundary changed")
    owner = _TraceWriteOwner(dispatch, boundary)
    dispatch.trace_owner = owner
    boundary._control_write_owner = owner
    with dispatch.controls._lock:
        dispatch.controls._trace_coordinators.add(owner.coordinator)


def run_trace_write(owner, coordinator, database, call_id, callback):
    """Observe only the original transaction and its exact native retirement."""
    from .console_native_commit import (
        _ConsoleNativeBinding,
        _ConsolePlanNativeRetirement,
        _invoke_plan_native,
    )

    if type(owner) is not _TraceWriteOwner:
        raise RuntimeError("Console trace write binding unavailable")
    owner.require(coordinator, database, call_id)
    controls = owner.dispatch.controls
    binding = _ConsoleNativeBinding(
        controls._store,
        owner,
        controls._store.persistence,
        database,
        retirement=_ConsolePlanNativeRetirement(),
        control_phase=(owner.dispatch.source, "trace"),
    )
    with controls.native_write_effect(binding):
        return _invoke_plan_native(binding, callback)


def retain_trace_worker_future(store, handoff):
    """Retain the original submission while its queue lock still binds the call."""
    from concurrent.futures import Future

    from .console_trace_settlement import ConsoleTraceSettlementHandoff

    if type(handoff) is not ConsoleTraceSettlementHandoff:
        return
    owner = handoff._prepared._control_owner
    if owner is None:
        return
    if type(owner) is not _TraceWriteOwner:
        raise RuntimeError("Console trace worker owner changed")
    owner.require(handoff._coordinator, handoff._database, handoff.call_id)
    with store._provider_trace_settlement_lock:
        future = store._provider_trace_settlement_worker_future
        queued = store._provider_trace_settlement_work.get(handoff.call_id)
        if (
            owner.dispatch.controls._store is not store
            or queued is None
            or queued[0] is not handoff
            or not store._provider_trace_settlement_worker_active
            or type(future) is not Future
        ):
            raise RuntimeError("Console trace worker submission changed")
        if future not in owner.worker_futures:
            owner.worker_futures.append(future)


async def drain_source_trace_workers(source):
    """Join the captured submissions without following replacement workers."""
    import asyncio
    from concurrent.futures import Future

    from .console_control_dispatch import _CommandSource
    from .console_control_provider import _ProviderDispatch

    if type(source) is not _CommandSource:
        raise RuntimeError("Console trace retirement source changed")
    controls = source.controls
    with controls._lock:
        dispatches = tuple(
            effect.owner
            for effect in source.effects
            if effect.kind == "provider_dispatch"
        )
    futures = set()
    for dispatch in dispatches:
        if (
            type(dispatch) is not _ProviderDispatch
            or dispatch.source is not source
            or dispatch.controls is not controls
        ):
            raise RuntimeError("Console trace retirement dispatch changed")
        owner = dispatch.trace_owner
        if owner is None:
            continue
        if type(owner) is not _TraceWriteOwner or owner.dispatch is not dispatch:
            raise RuntimeError("Console trace retirement owner changed")
        owner.require(owner.coordinator, owner.database, owner.call_id)
        with controls._store._provider_trace_settlement_lock:
            for future in owner.worker_futures:
                if type(future) is not Future:
                    raise RuntimeError("Console trace retirement future changed")
                futures.add(future)
    for future in futures:
        await asyncio.shield(asyncio.wrap_future(future))


async def drain_trace_worker(store):
    """Join the original executor submission, never its earlier active flag."""
    import asyncio

    while True:
        with store._provider_trace_settlement_lock:
            future = store._provider_trace_settlement_worker_future
        if future is not None:
            await asyncio.shield(asyncio.wrap_future(future))
        with store._provider_trace_settlement_lock:
            if future is not store._provider_trace_settlement_worker_future:
                continue
            if store._provider_trace_settlement_worker_active:
                raise RuntimeError("Console trace executor retirement is unproven")
            if store._provider_trace_settlement_work:
                raise RuntimeError("Console trace work remains owned")
            # Failed/pending records remain with the original store and bounded
            # coordinator queue. They are not successful settlements or live
            # native workers; Shutdown retains the original teardown retry.
            return


def require_trace_outcomes(controls):
    store = controls._store
    with store._provider_trace_settlement_lock:
        if (
            store._provider_trace_settlement_failed_work
            or store._provider_trace_settlements
            or store._provider_trace_settlement_owned_call_ids
        ):
            raise RuntimeError("Console trace settlement outcomes remain unresolved")
    with controls._lock:
        coordinators = tuple(controls._trace_coordinators)
    for coordinator in coordinators:
        with coordinator._queue_lock:
            if coordinator._pending or coordinator._inflight:
                raise RuntimeError("Console trace coordinator retains unresolved work")
