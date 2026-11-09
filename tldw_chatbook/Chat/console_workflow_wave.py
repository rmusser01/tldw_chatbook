"""One finite accepted native phase wave; the existing owners execute all work."""

from __future__ import annotations

import asyncio

from tldw_chatbook.Agents.workflow_wave import (
    WorkflowWave as _WorkflowWave,
    require_workflow_wave_sources as require_workflow_wave_sources,
)

from tldw_chatbook.Workspaces.change_review_finalization import ChangeReviewCompletion


async def run_workflow_wave(phase, prepared, *, _retain_waits=False):
    """Private complete-wave consumer for the existing accepted phase owner."""
    from tldw_chatbook.Agents.workflow_phase import WorkflowPhaseAuthorization

    from .console_workflow_preparation import _PreparedWorkflowReply

    if type(phase) is not WorkflowPhaseAuthorization or not prepared:
        raise PermissionError("accepted native workflow phase required")
    prepared = tuple(prepared)
    if any(type(item) is not _PreparedWorkflowReply for item in prepared):
        raise PermissionError("original native workflow preparation required")
    wave = _WorkflowWave(phase, prepared, retain_waits=_retain_waits)
    require_workflow_wave_sources(wave)
    if wave.runtime._workflow_closing:
        raise PermissionError("workflow admission is closing")
    bound = []
    try:
        for item in wave.prepared:
            auth = item.authorization
            with auth._lock:
                native = auth._native_preparation[0]
                if (
                    item._wave is not None
                    or item._closed
                    or item._closing
                    or item._launch_operation is not None
                    or item._close_operation is not None
                    or auth._native_launch
                    or auth._native_dispatch
                    or native.uncertain_start
                    or native.wait_return
                    and native.wait_return[0] is None
                ):
                    raise PermissionError("workflow preparation already claimed")
                item._wave = wave
                bound.append(item)
    except BaseException:
        # No create_task call, source, B, owner or native launch was attempted.
        for item in bound:
            with item.authorization._lock:
                if (
                    item._wave is not wave
                    or item._closed
                    or item._closing
                    or item.authorization._native_launch
                    or item._launch_operation is not None
                    or item._close_operation is not None
                    or wave.driver is not None
                    or wave.source is not None
                    or wave.reservation is not None
                    or wave.fence is not None
                ):
                    raise PermissionError("workflow unused enrollment return unproven")
                item._wave = None
        raise
    wave.runtime._workflow_waves[wave] = wave
    driver_coroutine = wave.run()
    # Once create_task is attempted an eager/failing factory may have entered.
    # The registered wave retains its original coroutine even if no Task returns.
    wave._driver_coroutine = driver_coroutine
    wave.driver = asyncio.create_task(driver_coroutine, name="native-workflow-wave")
    cancellation = None
    try:
        while not wave.driver.done():
            try:
                await asyncio.shield(wave.driver)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
                wave.stopping.set()
            except BaseException:  # noqa: BLE001 - original driver result retains failure custody
                break
        result = wave.driver.result()
        require_workflow_wave_sources(wave)
        if _retain_waits:
            for item in wave.unlaunched:
                with item.authorization._lock:
                    native = item.authorization._native_preparation[0]
                    if (
                        item._wave is not wave
                        or not wave.closed
                        or item._launch_operation is not None
                        or item.authorization._native_launch
                        or item.authorization._native_dispatch
                        or native.wait_return
                        and native.wait_return[0] is None
                    ):
                        raise PermissionError("original workflow wait unlink unproven")
                    item._wave = None
            if cancellation is not None:
                raise cancellation
            return wave
        if cancellation is not None:
            raise cancellation
        return result
    finally:
        # Unknown driver/source/worker exits retain the original registration.
        # Only the original driver's positive _release removes it.
        pass


def _workflow_close_association(item, wave):
    """Retain only the actual enrolled wave and prepared member."""
    if item._wave is not wave:
        raise PermissionError("workflow close wave changed")
    if wave is None:
        return None
    association = wave, item
    _workflow_close_current(association)
    return association


def _workflow_close_current(association):
    if association is None:
        return
    wave, item = association
    require_workflow_wave_sources(wave)
    if (
        item._wave is not wave
        or item.phase is not wave.phase
        or wave.closed
        or wave.runtime._workflow_waves.get(wave) is not wave
        or not any(member is item for member in wave.prepared)
    ):
        raise PermissionError("workflow close enrollment changed")


def _workflow_wave_release_return(wave, prepared):
    """Prove original release callouts, without certifying driver success."""
    require_workflow_wave_sources(wave)
    returned = wave._release_return
    current = (
        wave,
        wave.driver,
        wave._driver_coroutine,
        wave.source,
        wave.reservation,
        wave.fence,
        wave.baseline_return,
        wave.ending_return,
        wave.prepared,
    )
    if (
        type(returned) is not tuple
        or len(returned) != 12
        or any(a is not b for a, b in zip(returned[:9], current, strict=True))
        or not wave.closed
        or wave in wave.runtime._workflow_waves
        or wave.driver is None
        or not wave.driver.done()
        or wave.unproven_launches
        or prepared._wave is not wave
        or any(
            type(saved) is not tuple
            or len(saved) != len(actual)
            or any(a is not b for a, b in zip(saved, actual, strict=True))
            for saved, actual in (
                (returned[9], wave._snapshot_returns),
                (returned[10], wave.receipts),
            )
        )
    ):
        raise RuntimeError("original workflow release return unproven")
    members = tuple(
        item
        for item in wave.prepared
        if not wave.retain_waits or item not in wave.unlaunched
    )
    if (
        type(returned[11]) is not tuple
        or len(returned[11]) != len(members)
        or not any(item is prepared for item in members)
        or any(
            type(saved) is not tuple
            or len(saved) != 2
            or saved[0] is not item
            or saved[1] is not item._terminal_configuration_return
            for saved, item in zip(returned[11], members, strict=True)
        )
    ):
        raise RuntimeError("original workflow released members changed")
    for item in members:
        item._execution_configuration_return()
    return returned


def _workflow_wave_return(wave, saved=None):
    """Retain/recheck the actual original wave's finite positive returns."""
    from . import console_workflow_preparation as preparation

    require_workflow_wave_sources(wave)
    current = (
        wave,
        wave.driver,
        wave.baseline_return,
        wave.ending_return,
        wave.source,
        wave.reservation,
        tuple(wave.receipts),
        tuple(wave._snapshot_returns),
    )
    if saved is not None and (
        type(saved) is not tuple
        or len(saved) != len(current)
        or any(
            first is not second
            for first, second in zip(saved[:6], current[:6], strict=True)
        )
        or any(
            len(first) != len(second)
            or any(a is not b for a, b in zip(first, second, strict=True))
            for first, second in zip(saved[6:], current[6:], strict=True)
        )
    ):
        raise RuntimeError("original workflow wave return changed")
    if (
        not wave.closed
        or not wave.driver.done()
        or wave.driver.cancelled()
        or wave.driver.exception() is not None
        or wave in wave.runtime._workflow_waves
        or wave.unproven_launches
    ):
        raise RuntimeError("original workflow wave return unproven")
    for item, receipt in current[6]:
        preparation._finite_owner_return(receipt.owner, wave.runtime.execution_capacity)
        if not item._closed or item.authorization._native_launch[0] is not receipt:
            raise RuntimeError("original workflow child return changed")
    for kind, ticket, source, owner, operation, claim, retired in current[7]:
        if (
            type(ticket) is not _TICKET_CLASS
            or type(source) is not _SnapshotSource
            or ticket.source is not source
            or source is not wave.source
            or ticket.owner is not owner
            or ticket.operation is not operation
            or ticket.claim is not claim
            or ticket.retired is not retired
            or not retired.is_set()
            or not ticket._worker_returned
            or not ticket._handoff_returned
            or ticket._failed_write
            or source.workflow_wave is not wave
            or source.bridge is not wave.bridge
            or source.issuer.runtime is not wave.runtime
            or source.issuer.coordinator is not wave.coordinator
            or wave.runtime._change_review_coordinator is not wave.coordinator
            or wave.bridge._change_finalization_coordinator is not wave.coordinator
            or source.controller is not wave.runtime.chat_controller
            or source.store is not wave.runtime._chat_store
            or source.store._sessions.get(source.session.id) is not source.session
        ):
            raise RuntimeError("original workflow snapshot return changed")
        preparation._finite_owner_return(
            owner, wave.runtime.execution_capacity, operation
        )
        activity = wave.runtime.worktree_activity
        with activity._lock:
            if claim is not None and (
                claim.id in activity._claims or claim.id in activity._members
            ):
                raise RuntimeError("original workflow snapshot claim unreturned")
        completion = (
            wave.reservation._baseline_completion
            if kind == "baseline"
            else wave.reservation._completion
        )
        if completion.ticket is not ticket:
            raise RuntimeError("original workflow snapshot completion changed")
    if wave.reservation is not None:
        kinds = tuple(record[0] for record in current[7])
        if (
            kinds.count("baseline") != 1
            or wave.baseline_return.result() is not ChangeReviewCompletion.OBSERVED
        ):
            raise RuntimeError("original workflow baseline return unproven")
        if wave.anchor is not None:
            if (
                kinds.count("finalize") != 1
                or wave.ending_return.result() is not ChangeReviewCompletion.PUBLISHED
            ):
                raise RuntimeError("original workflow ending return unproven")
        elif (
            len(kinds) != 1
            or wave.ending_return.done()
            or wave.reservation.completion is not ChangeReviewCompletion.CANCELLED
        ):
            raise RuntimeError("original workflow unused window return unproven")
    return current if saved is None else saved


from tldw_chatbook.Workspaces.change_review_ownership import (
    _SnapshotSource,
    _SnapshotTicket,
)

_TICKET_CLASS = _SnapshotTicket
