"""One finite accepted native phase wave; the existing owners execute all work."""

from __future__ import annotations

import asyncio
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

from tldw_chatbook.Agents.activation import execution
from tldw_chatbook.Agents.execution_capacity import WorkOrigin
from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
from tldw_chatbook.Agents.workflow_models import WorkflowTaskResult
from tldw_chatbook.DB.base_db import run_owned_db_call
from tldw_chatbook.Workspaces.change_review_finalization import ChangeReviewCompletion
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint, WorktreeScope


@dataclass(frozen=True, slots=True)
class _WaveOpenRefusal:
    """An original explicit no-entry body disposition, never authority."""

    error: BaseException = field(repr=False)
    retain: bool
    source: object = field(default=None, repr=False)
    footprint: object = field(default=None, repr=False)


class _WorkflowWave:
    """Exact process-local membership, never reconstructed from stored IDs."""

    def __init__(self, phase, prepared, *, retain_waits=False):
        from .console_workflow_preparation import _PreparedWorkflowReply
        from .console_workflow_runs import WorkflowPhaseAuthorization

        if type(phase) is not WorkflowPhaseAuthorization or not prepared:
            raise PermissionError("accepted native workflow phase required")
        self.phase = phase
        self.prepared = tuple(prepared)
        if any(type(item) is not _PreparedWorkflowReply for item in self.prepared):
            raise PermissionError("original native workflow preparation required")
        first = self.prepared[0]
        self.runtime, self.bridge, self.service = (
            first.runtime,
            first.bridge,
            first.service,
        )
        self.coordinator = self.bridge._change_finalization_coordinator
        self.roots = tuple(
            dict.fromkeys(
                root
                for item in self.prepared
                for root in item.configuration.workspace_roots
            )
        )
        self.capture_enabled = bool(
            self.roots
            and self.coordinator is not None
            and self.bridge.change_tracking_enabled
        )
        self.loop = asyncio.get_running_loop()
        self.baseline_return = self.loop.create_future()
        self.ending_return = self.loop.create_future()
        self.source = self.publication = self.reservation = self.fence = None
        self.receipts = []
        self._snapshot_returns = []
        self.unlaunched = {}
        self.unproven_launches = set()
        self.retain_waits = retain_waits
        self.wait_footprint = None
        self.anchor = None
        self.closed = False
        self.stopping = Event()
        self.driver = None
        self._driver_coroutine = None
        self.check_prepared_members(tuple(item.authorization for item in self.prepared))

    def check_prepared_members(self, members):
        self.phase.check()
        if len(members) != len(self.prepared) or len(
            {id(item) for item in members}
        ) != len(members):
            raise PermissionError("distinct original wave members required")
        for item, member in zip(self.prepared, members):
            if (
                item.phase is not self.phase
                or item.runtime is not self.runtime
                or item.authorization is not member
                or item._closed
            ):
                raise PermissionError("original wave preparation changed")
            member.check_preparation()

    def _check_enrollment(self, item):
        with item.authorization._lock:
            if (
                item._wave is not self
                or item._closing
                or item._closed
                or self.runtime._workflow_waves.get(self) is not self
                or not any(member is item for member in self.prepared)
            ):
                raise PermissionError("original workflow enrollment changed")

    def check_snapshot_source(self, source):
        if (
            self.source is not source
            or self.closed
            or self.runtime._workflow_waves.get(self) is not self
            or source.bridge is not self.bridge
            or source.conversation_id != self.phase.attempt.conversation_id
        ):
            raise PermissionError("original workflow snapshot source changed")

    def snapshot_returned(self, reservation, kind, ticket):
        """Called after the actual operation finish and native source scopes return."""
        if (
            _workflow_wave_sources is not _WAVE_SOURCE[0]
            or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
            or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
            or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
            or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        _workflow_wave_sources(self)
        if (
            reservation is not self.reservation
            or ticket.source is not self.source
            or not ticket.retired.is_set()
        ):
            raise PermissionError("original snapshot return required")
        if kind == "discover":
            return
        self._snapshot_returns.append(
            (
                kind,
                ticket,
                ticket.source,
                ticket.owner,
                ticket.operation,
                ticket.claim,
                ticket.retired,
            )
        )
        future = self.baseline_return if kind == "baseline" else self.ending_return
        disposition = (
            reservation.baseline_completion
            if kind == "baseline"
            else reservation.completion
        )

        def deliver():
            if not future.done():
                future.set_result(disposition)

        self.loop.call_soon_threadsafe(deliver)

    def validate_discovered_roots(self, preparations):
        """Bound original discovery before the coordinator enrolls or captures B."""
        roots = tuple(
            dict.fromkeys(
                (
                    *(str(chain.canonical_root) for chain in self.source.chains),
                    *(str(item.root) for item in preparations),
                )
            )
        )
        self.service.db.workflow_runs._window_roots(roots)

    def unadmitted(self, _message):
        """Original coordinator acknowledgement that no finite worker entered."""

        def deliver():
            future = (
                self.ending_return
                if self.baseline_return.done()
                else self.baseline_return
            )
            if not future.done():
                future.set_result(ChangeReviewCompletion.UNADMITTED)

        self.loop.call_soon_threadsafe(deliver)

    def _release(self):
        if (
            _workflow_wave_sources is not _WAVE_SOURCE[0]
            or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
            or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
            or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
            or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        _workflow_wave_sources(self)
        caller = sys._getframe(1)
        if (
            self.closed
            or self.unproven_launches
            or asyncio.current_task() is not self.driver
            or self.driver.get_coro() is not self._driver_coroutine
            or self._driver_coroutine.cr_code is not _WAVE_RUN_CODE
            or caller.f_code not in {_WAVE_RUN_CODE, _WAVE_OPEN_CODE}
            or caller.f_locals.get("self") is not self
            or self.runtime._workflow_waves.get(self) is not self
        ):
            raise PermissionError("original workflow wave release required")
        for item in self.prepared:
            with item.authorization._lock:
                native = item.authorization._native_preparation[0]
                retained = self.retain_waits and item in self.unlaunched
                if (
                    item._wave is not self
                    or item._launch_operation is not None
                    or item._close_operation is not None
                    or item._closing
                    or item.authorization._native_launch == [None]
                    or native.uncertain_start
                    or native.wait_return
                    and native.wait_return[0] is None
                    or not retained
                    and not item._closed
                    or retained
                    and (
                        item._closed
                        or item.authorization._native_launch
                        or item.authorization._native_dispatch
                    )
                ):
                    raise PermissionError(
                        "original workflow wave member return unproven"
                    )
            if not retained:
                read = item._finite_reads[-1]
                item._finite_read_return(read)
                if (
                    not read.positive
                    or read.wrapper_error is not None
                    or read.body_error is not None
                ):
                    raise PermissionError("original workflow close read unproven")
        if self.reservation is not None and caller.f_code is _WAVE_RUN_CODE:
            if self.anchor is None:
                if self.reservation.completion is not ChangeReviewCompletion.CANCELLED:
                    raise PermissionError("original unused workflow window required")
            elif not self.ending_return.done() or self.ending_return.result() not in {
                ChangeReviewCompletion.PUBLISHED,
                ChangeReviewCompletion.DISCLOSED_FAILURE,
            }:
                raise PermissionError("original workflow ending return required")
        if self.fence is not None:
            self.runtime.worktree_activity._close_wave(self.fence)
        for item in self.prepared:
            if not self.retain_waits or item not in self.unlaunched:
                item._release_configuration()
        # All original callouts returned; enrollment protects them until now.
        if (
            _workflow_wave_sources is not _WAVE_SOURCE[0]
            or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
            or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
            or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
            or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        _workflow_wave_sources(self)
        if self.runtime._workflow_waves.get(self) is not self:
            raise PermissionError("original workflow wave registration changed")
        self.closed = True
        self.runtime._workflow_waves.pop(self)

    def bind_member(self, authorization, owner):
        """Use the existing committed bind_native seam, never a requested run ID."""
        if (
            self.closed
            or self.runtime._workflow_waves.get(self) is not self
            or not any(item.authorization is authorization for item in self.prepared)
        ):
            raise PermissionError("original open wave member required")
        if not self.capture_enabled:
            return
        if self.fence is None:
            raise PermissionError("original workflow ordering fence required")
        if self.baseline_return.result() is not ChangeReviewCompletion.OBSERVED:
            raise PermissionError("actual baseline return required")
        self.service.db.workflow_runs.anchor_change_window(
            self.reservation.id, authorization, owner
        )
        # The FULL transaction chooses once. Read its winner rather than racing
        # two native worker notifications to attribute the anchor in memory.
        with self.service.db.connection() as conn:
            anchor = conn.execute(
                "SELECT t.run_id FROM workflow_change_windows w JOIN workflow_task_attempts t ON t.id=w.anchor_attempt_id WHERE w.id=?",
                (self.reservation.id,),
            ).fetchone()[0]
        self.anchor = anchor

    def publish(self, publication):
        from tldw_chatbook.Workspaces.change_review_ownership import (
            native_snapshot_ticket,
        )

        ticket = native_snapshot_ticket()
        if (
            ticket is None
            or ticket.source is not self.source
            or publication.reservation_id != self.reservation.id
            or publication.run_id != self.anchor
        ):
            raise PermissionError("original workflow E publisher required")
        with owned_goal_storage(self.service):
            self.service.db.workflow_runs.publish_change_window(
                self.reservation.id,
                run_id=publication.run_id,
                records=tuple(record.__dict__ for record in publication.records),
                kind=publication.kind,
            )

    async def _open(self):
        from tldw_chatbook.Agents.agent_service import _WorkflowChildWait

        if not self.runtime.execution_capacity._workflow_children_ready(
            len(self.prepared)
        ):
            if self.runtime.execution_capacity.snapshot().closed:
                raise PermissionError("workflow runtime closed")
            wait = _WorkflowChildWait(
                "runtime_capacity", "workflow wave capacity unavailable"
            )
            return _WaveOpenRefusal(wait, self.retain_waits)
        fleets = {}
        for item in self.prepared:
            fleet = item.authorization._native_preparation[0].fleet
            fleets[fleet] = fleets.get(fleet, 0) + 1
        for fleet, count in fleets.items():
            with fleet._lock:
                if fleet._fenced:
                    raise PermissionError("workflow fleet fenced")
                full = len(fleet._live_ids) + count > fleet.max_live
            if full:
                wait = _WorkflowChildWait(
                    "fleet_capacity", "workflow wave fleet capacity unavailable"
                )
                return _WaveOpenRefusal(wait, self.retain_waits)
        footprints = tuple(
            item.authorization.physical_footprint for item in self.prepared
        )
        if any(
            first._conflicts(second)
            for i, first in enumerate(footprints)
            for second in footprints[i + 1 :]
        ):
            return _WaveOpenRefusal(
                _WorkflowChildWait(
                    "worktree_activity", "workflow wave members conflict"
                ),
                False,
            )
        if not self.capture_enabled:
            for item in self.prepared:
                self._check_enrollment(item)
            return
        roots = self.roots
        from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

        first = self.prepared[0]

        def capture():
            source = self.coordinator._native.capture(
                self.bridge,
                session_id=first.session.id,
                conversation_id=self.phase.attempt.conversation_id,
                origin=WorkOrigin.AUTOMATIC,
                roots=roots,
                workflow_wave=self,
            )
            self.source = source
            footprint = WorktreeFootprint(
                writes=tuple(
                    WorktreeScope(Path(root), True, capture_directory_chain(Path(root)))
                    for root in roots
                )
            )

            return source, footprint

        reads, read_index = first._finite_reads, len(first._finite_reads)
        captured = await first._read(capture)
        if (
            _workflow_wave_sources is not _WAVE_SOURCE[0]
            or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
            or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
            or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
            or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        _workflow_wave_sources(self)
        if first._finite_reads is not reads or len(reads) != read_index + 1:
            raise PermissionError("original workflow capture reader changed")
        read = reads[read_index]
        proof = first._finite_read_return(read)
        if (
            not read.positive
            or read.wrapper_error is not None
            or read.body_error is not None
            or read.issuing is not self.driver
            or proof[8] is not captured
            or type(captured) is not tuple
            or len(captured) != 2
            or captured[0] is not self.source
            or type(captured[0]) is not _SnapshotSource
            or captured[0].issuer is not self.coordinator._native
            or captured[0].workflow_wave is not self
            or captured[0].bridge is not self.bridge
            or dict(
                zip(
                    read.callback.__code__.co_freevars,
                    (cell.cell_contents for cell in read.callback.__closure__ or ()),
                )
            ).get("callback")
            is not capture
        ):
            raise PermissionError("original workflow capture return unproven")
        captured_source, footprint = captured
        self.wait_footprint = footprint
        self.fence = self.runtime.worktree_activity._open_wave(
            self, footprint, tuple(item.authorization for item in self.prepared)
        )
        if self.fence is None:
            # Capture can borrow an ordinary tree and its three preheld tickets.
            if (
                captured_source is not self.source
                or captured_source.ordinary_tree is not None
                or captured_source.owner is not None
                or captured_source.claim is not None
                or type(captured_source._holds_lock) is not _WAVE_LOCK_CLASS
                or self.reservation is not None
                or self.publication is not None
                or self.baseline_return.done()
                or self.ending_return.done()
                or self.receipts
            ):
                raise PermissionError("workflow captured source remains owned")
            with captured_source._holds_lock:
                if captured_source._holds or captured_source._closed.is_set():
                    raise PermissionError("workflow captured source return unproven")
            error = (
                _WorkflowChildWait("worktree_activity", "workflow wave activity busy")
                if self.retain_waits
                else PermissionError("workflow wave activity busy")
            )
            return _WaveOpenRefusal(
                error, self.retain_waits, captured_source, footprint
            )
        if self.runtime._workflow_waves.get(self) is not self:
            raise PermissionError("original workflow enrollment changed")
        for item in self.prepared:
            self._check_enrollment(item)
        with execution(self.bridge), owned_goal_storage(self.service):
            self.reservation = self.coordinator.register(
                roots, native_source=self.source, on_unadmitted=self.unadmitted
            )
        if self.reservation is None or self.reservation.admission_error:
            if self.reservation is not None:
                self.coordinator.cancel(self.reservation)
            for item in self.prepared:
                closing = item.close()
                await closing
            self._release()
            raise PermissionError("workflow window admission refused")
        from tldw_chatbook.DB.workflow_runs import _WindowUnpublished

        with execution(self.bridge), owned_goal_storage(self.service):
            publication = self.service.db.workflow_runs.open_change_window(
                self.phase, self.reservation, roots
            )
            if type(publication) is not _WindowUnpublished:
                self.publication = publication
                self.coordinator._activate_workflow_reservation(
                    self.reservation, self.source, self.publication
                )
        if type(publication) is _WindowUnpublished:
            # A private no-commit outcome is consumable only after the original
            # execution/storage scopes above have physically returned.
            self.coordinator._cancel_unpublished_workflow(
                self.reservation, self.source, publication
            )
            for item in self.prepared:
                closing = item.close(_unpublished=publication)
                await closing
            self._release()
            raise publication.error
        baseline = await asyncio.shield(self.baseline_return)
        roots = tuple(str(root) for root in self.reservation._handle.roots)
        disposition = (
            "completed"
            if baseline is ChangeReviewCompletion.OBSERVED
            else "disclosed_failure"
        )

        def record_baseline():
            with owned_goal_storage(self.service):
                self.service.db.workflow_runs.baseline_change_window(
                    self.reservation.id, roots, disposition
                )

        await run_owned_db_call(self.service.db, record_baseline)
        if baseline is not ChangeReviewCompletion.OBSERVED:
            self.coordinator.cancel(self.reservation)
            for item in self.prepared:
                closing = item.close()
                await closing
            await run_owned_db_call(
                self.service.db,
                self.service.db.workflow_runs.unused_change_window,
                self.reservation.id,
            )
            self._release()
            raise PermissionError("workflow baseline unavailable")

    async def _launch(self, item):
        from tldw_chatbook.Agents.agent_service import (
            _WF_LAUNCH_SOURCE,
            _workflow_launch_sources,
            _WorkflowLaunchReturn,
        )

        from .console_ordinary_tree import _function_record

        if (
            _workflow_wave_sources is not _WAVE_SOURCE[0]
            or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
            or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
            or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
            or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        _workflow_wave_sources(self)
        issuing = asyncio.current_task()
        invocation = sys._getframe()
        unused = []
        unused_returned = False

        def launch():
            source = (self, item, issuing, invocation, worker, launch)
            with (
                owned_goal_storage(self.service),
                self.phase.context.scope(),
                execution(self.bridge),
                item.activate_source(),
            ):
                if (
                    _workflow_launch_sources is not _WF_LAUNCH_SOURCE[0]
                    or _workflow_launch_sources.__code__ is not _WF_LAUNCH_SOURCE[1]
                    or _workflow_launch_sources.__defaults__ is not _WF_LAUNCH_SOURCE[2]
                    or _workflow_launch_sources.__kwdefaults__
                    is not _WF_LAUNCH_SOURCE[3]
                    or _workflow_launch_sources.__closure__ is not _WF_LAUNCH_SOURCE[5]
                ):
                    raise PermissionError("original workflow helper source changed")
                _workflow_launch_sources(item.native.service)
                if self.stopping.is_set():
                    result = PermissionError("workflow wave cancelled before child")
                    unused.append(result)
                else:
                    result = item.native.service._workflow_launch_result(
                        item.authorization.task,
                        authorization=item.authorization,
                        should_cancel=lambda: (
                            self.stopping.is_set() or item.should_cancel()
                        ),
                    )
                    if result.receipt is not None:
                        self.service.bind_native_receipt(
                            item.authorization, result.receipt
                        )
            # source remains an actual local association through the worker's return.
            if source[0] is not self:
                raise PermissionError("original workflow worker changed")
            return result

        worker = asyncio.to_thread(launch)
        worker_frame = worker.cr_frame  # noqa: F841 - retained by the original invocation frame
        launch_record = _function_record(launch)  # noqa: F841 - same producer checks this frame local
        try:
            result = await worker
            if (
                _workflow_wave_sources is not _WAVE_SOURCE[0]
                or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
                or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
                or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
                or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
            ):
                raise PermissionError("original workflow helper source changed")
            _workflow_wave_sources(self)
            if unused:
                if len(unused) != 1 or result is not unused[0]:
                    raise PermissionError("original unused workflow worker required")
                # A real original no-entry worker returned; no launch marker exists.
                unused_returned = True
                raise result
            if type(result) is not _WorkflowLaunchReturn:
                raise PermissionError("original workflow worker result required")
            if not result.known:
                self.unproven_launches.add(item)
                raise result.error
            item.native.service._finish_workflow_launch(result)
        except BaseException:
            if not unused_returned:
                self.unproven_launches.add(item)
            raise
        if result.receipt is None:
            if result.wait:
                self.unlaunched[item] = result.error
                return None
            raise result.error
        self.receipts.append((item, result.receipt))
        if result.error is not None:
            raise result.error
        return result.receipt

    async def _settle(self, item, receipt):
        drained = self.loop.create_future()

        def observed(proven):
            self.loop.call_soon_threadsafe(
                lambda: not drained.done() and drained.set_result(proven)
            )

        receipt.owner.on_drained(observed)
        if not await asyncio.shield(drained):
            raise PermissionError("workflow native drain unproven")
        if receipt.failure == "no_run":
            closing = item.close()
            await closing
            return
        row = await run_owned_db_call(
            self.service.db, self.service.db.get_run, receipt.run_id
        )
        from tldw_chatbook.Agents.agent_models import RUN_DONE, TERMINAL_RUN_STATUSES

        if row is None or row["status"] not in TERMINAL_RUN_STATUSES:
            raise PermissionError("workflow native terminal state missing")
        body = str(row.get("result") or "").strip()
        if not body:
            if row["status"] == RUN_DONE:
                raise PermissionError("workflow required native handoff missing")
            body = f"Native task ended with status {row['status']}; no result text was recorded."
        result = WorkflowTaskResult(
            workflow_id=self.phase.attempt.workflow_id,
            attempt_id=receipt.attempt_id,
            run_id=receipt.run_id,
            status="completed"
            if row["status"] == RUN_DONE and receipt.failure is None
            else "failed",
            summary=body,
            handoff=body,
        )
        await self.service.call(self.service.checkpoint_task, result, drained=True)
        closing = item.close()
        await closing

    async def run(self):
        """Launch a fixed eligible set; close E at its first actual quiescent gap."""
        if (
            _workflow_wave_sources is not _WAVE_SOURCE[0]
            or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
            or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
            or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
            or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        _workflow_wave_sources(self)
        opening = self._open()
        waiting = await opening
        if (
            _workflow_wave_sources is not _WAVE_SOURCE[0]
            or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
            or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
            or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
            or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        _workflow_wave_sources(self)
        if waiting is not None:
            if (
                type(waiting) is not _WaveOpenRefusal
                or opening.cr_code is not _WAVE_OPEN_CODE
                or opening.cr_frame is not None
                or asyncio.current_task() is not self.driver
                or waiting.source is not self.source
                or waiting.footprint is not self.wait_footprint
                or self.reservation is not None
                or self.publication is not None
                or self.fence is not None
                or self.baseline_return.done()
                or self.ending_return.done()
                or self.receipts
            ):
                raise PermissionError("original workflow no-entry return required")
            for item in self.prepared:
                self._check_enrollment(item)
                with item.authorization._lock:
                    native = item.authorization._native_preparation[0]
                    if (
                        item._launch_operation is not None
                        or item._close_operation is not None
                        or item.authorization._native_launch
                        or item.authorization._native_dispatch
                        or native.uncertain_start
                        or native.wait_return
                        and native.wait_return[0] is None
                    ):
                        raise PermissionError("original workflow unused member changed")
            if waiting.retain:
                self.unlaunched.update((item, waiting.error) for item in self.prepared)
            else:
                for item in self.prepared:
                    closing = item.close()
                    await closing
            self._release()
            if not waiting.retain:
                raise waiting.error
            return self.reservation
        for item in self.prepared:
            self._check_enrollment(item)
        launch_group = asyncio.gather(
            *(self._launch(item) for item in self.prepared), return_exceptions=True
        )
        launches = await launch_group
        settle_group = asyncio.gather(
            *(self._settle(item, receipt) for item, receipt in self.receipts),
            return_exceptions=True,
        )
        settled = await settle_group
        for result in settled:
            if isinstance(result, BaseException):
                raise result
        if self.unproven_launches:
            raise next(
                result for result in launches if isinstance(result, BaseException)
            )
        for item in self.prepared:
            if not any(item is admitted for admitted, _receipt in self.receipts) and (
                not self.retain_waits or item not in self.unlaunched
            ):
                closing = item.close()
                await closing
        if not self.capture_enabled:
            pass
        elif self.anchor is None:
            self.coordinator.cancel(self.reservation)
            await run_owned_db_call(
                self.service.db,
                self.service.db.workflow_runs.unused_change_window,
                self.reservation.id,
            )
        else:
            if not self.coordinator.finalize(
                self.reservation, run_id=self.anchor, kind="workflow_wave"
            ):
                raise PermissionError("workflow E admission refused")
            disposition = await asyncio.shield(self.ending_return)
            if disposition not in {
                ChangeReviewCompletion.PUBLISHED,
                ChangeReviewCompletion.DISCLOSED_FAILURE,
            }:
                raise PermissionError("workflow E required publication unresolved")
        self._release()
        for result in launches:
            if isinstance(result, BaseException):
                raise result
        if self.unlaunched and not self.retain_waits:
            raise next(iter(self.unlaunched.values()))
        return self.reservation


async def run_workflow_wave(phase, prepared, *, _retain_waits=False):
    """Private complete-wave consumer for the existing accepted phase owner."""
    if (
        _workflow_wave_sources is not _WAVE_SOURCE[0]
        or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
        or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
        or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
        or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
    ):
        raise PermissionError("original workflow helper source changed")
    _workflow_wave_sources()
    wave = _WorkflowWave(phase, prepared, retain_waits=_retain_waits)
    if (
        _workflow_wave_sources is not _WAVE_SOURCE[0]
        or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
        or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
        or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
        or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
    ):
        raise PermissionError("original workflow helper source changed")
    _workflow_wave_sources(wave)
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
        if (
            _workflow_wave_sources is not _WAVE_SOURCE[0]
            or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
            or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
            or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
            or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
        ):
            raise PermissionError("original workflow helper source changed")
        _workflow_wave_sources(wave)
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


def _workflow_launch_association(prepared, caller):
    """Observe this actual original worker, never reconstruct it from an ID."""
    if prepared is None or caller.f_code is not _WAVE_WORKER_CODE:
        raise PermissionError("original workflow launch worker required")
    source = caller.f_locals.get("source")
    if type(source) is not tuple or len(source) != 6:
        raise PermissionError("original workflow launch closure required")
    wave, item, issuing, invocation, worker, launch = source
    if item is not prepared:
        raise PermissionError("original workflow launch member changed")
    driver = wave.driver
    run_frame = driver.get_coro().cr_frame if driver is not None else None
    group = run_frame.f_locals.get("launch_group") if run_frame is not None else None
    association = (
        wave,
        item,
        issuing,
        invocation,
        worker,
        invocation.f_locals.get("worker_frame"),
        launch,
        invocation.f_locals.get("launch_record"),
        driver,
        run_frame,
        group,
        caller,
    )
    if (
        _workflow_launch_association_current is not _WAVE_ASSOCIATION_CURRENT_SOURCE[0]
        or _workflow_launch_association_current.__code__
        is not _WAVE_ASSOCIATION_CURRENT_SOURCE[1]
        or _workflow_launch_association_current.__defaults__
        is not _WAVE_ASSOCIATION_CURRENT_SOURCE[2]
        or _workflow_launch_association_current.__kwdefaults__
        is not _WAVE_ASSOCIATION_CURRENT_SOURCE[3]
        or _workflow_launch_association_current.__closure__
        is not _WAVE_ASSOCIATION_CURRENT_SOURCE[5]
    ):
        raise PermissionError("original workflow helper source changed")
    _workflow_launch_association_current(association)
    return association


def _workflow_launch_association_current(association):
    from .console_ordinary_tree import _function_unchanged

    if type(association) is not tuple or len(association) != 12:
        raise PermissionError("original workflow launch association required")
    (
        wave,
        item,
        issuing,
        invocation,
        worker,
        worker_frame,
        launch,
        record,
        driver,
        run_frame,
        group,
        caller,
    ) = association
    if (
        _workflow_wave_sources is not _WAVE_SOURCE[0]
        or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
        or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
        or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
        or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
    ):
        raise PermissionError("original workflow helper source changed")
    _workflow_wave_sources(wave)
    if (
        wave.driver is not driver
        or driver is None
        or driver.get_coro().cr_code is not _WAVE_RUN_CODE
        or driver.get_coro().cr_frame is not run_frame
        or run_frame is None
        or run_frame.f_locals.get("self") is not wave
        or run_frame.f_locals.get("launch_group") is not group
        or type(group) is not _WAVE_GATHER_CLASS
        or not any(child is issuing for child in group._children)
        or type(issuing) is not _WAVE_TASK_CLASS
        or issuing.get_coro().cr_code is not _WAVE_LAUNCH_CODE
        or issuing.get_coro().cr_frame is not invocation
        or invocation.f_locals.get("self") is not wave
        or invocation.f_locals.get("item") is not item
        or invocation.f_locals.get("issuing") is not issuing
        or invocation.f_locals.get("worker") is not worker
        or invocation.f_locals.get("worker_frame") is not worker_frame
        or invocation.f_locals.get("launch") is not launch
        or worker.cr_code is not _WAVE_TO_THREAD_CODE
        or worker_frame is None
        or worker_frame.f_code is not _WAVE_TO_THREAD_CODE
        or worker_frame.f_locals.get("func") is not launch
        or type(record) is not tuple
        or record[0] is not launch
        or not _function_unchanged(record)
        or caller.f_code is not _WAVE_WORKER_CODE
        or type(caller.f_locals.get("source")) is not tuple
        or len(caller.f_locals["source"]) != 6
        or any(
            a is not b
            for a, b in zip(
                caller.f_locals["source"],
                (wave, item, issuing, invocation, worker, launch),
            )
        )
        or item._wave is not wave
        or item.phase is not wave.phase
        or item.runtime is not wave.runtime
        or wave.runtime._workflow_waves.get(wave) is not wave
        or not any(member is item for member in wave.prepared)
    ):
        raise PermissionError("original workflow launch association changed")


def _workflow_launch_received(association, result, caller):
    if (
        _workflow_launch_association_current is not _WAVE_ASSOCIATION_CURRENT_SOURCE[0]
        or _workflow_launch_association_current.__code__
        is not _WAVE_ASSOCIATION_CURRENT_SOURCE[1]
        or _workflow_launch_association_current.__defaults__
        is not _WAVE_ASSOCIATION_CURRENT_SOURCE[2]
        or _workflow_launch_association_current.__kwdefaults__
        is not _WAVE_ASSOCIATION_CURRENT_SOURCE[3]
        or _workflow_launch_association_current.__closure__
        is not _WAVE_ASSOCIATION_CURRENT_SOURCE[5]
    ):
        raise PermissionError("original workflow helper source changed")
    _workflow_launch_association_current(association)
    wave, item, issuing, invocation, worker = association[:5]
    if (
        caller is not invocation
        or asyncio.current_task() is not issuing
        or caller.f_locals.get("result") is not result
        or worker.cr_frame is not None
        or result.entry.wave is not wave
        or result.entry.preparation is not item
        or result.entry.association is not association
    ):
        raise PermissionError("original workflow worker return required")


def _workflow_close_association(item, close_frame, caller):
    wave = item._wave
    if wave is None:
        return None
    if (
        _workflow_wave_sources is not _WAVE_SOURCE[0]
        or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
        or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
        or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
        or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
    ):
        raise PermissionError("original workflow helper source changed")
    _workflow_wave_sources(wave)
    task = asyncio.current_task()
    driver = wave.driver
    run_frame = driver.get_coro().cr_frame if driver is not None else None
    closing = caller.f_locals.get("closing")
    if (
        closing is None
        or closing.cr_frame is not close_frame
        or caller.f_locals.get("self") is not wave
        or not any(member is item for member in wave.prepared)
        or caller.f_locals.get("item") is not item
        or run_frame is None
        or run_frame.f_locals.get("self") is not wave
    ):
        raise PermissionError("original wave close invocation required")
    if caller.f_code in {_WAVE_RUN_CODE, _WAVE_OPEN_CODE}:
        if task is not driver:
            raise PermissionError("original wave close driver required")
        group = receipt = None
    elif caller.f_code is _WAVE_SETTLE_CODE:
        group = run_frame.f_locals.get("settle_group")
        receipt = caller.f_locals.get("receipt")
        if (
            type(group) is not _WAVE_GATHER_CLASS
            or not any(child is task for child in group._children)
            or task.get_coro().cr_frame is not caller
            or not any(p is item and r is receipt for p, r in wave.receipts)
        ):
            raise PermissionError("original wave settlement close required")
    else:
        raise PermissionError("original wave close caller required")
    return (
        wave,
        item,
        task,
        closing,
        close_frame,
        caller,
        driver,
        run_frame,
        group,
        receipt,
    )


def _workflow_close_current(association):
    if association is None:
        return
    (
        wave,
        item,
        task,
        closing,
        close_frame,
        caller,
        driver,
        run_frame,
        group,
        receipt,
    ) = association
    if (
        _workflow_wave_sources is not _WAVE_SOURCE[0]
        or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
        or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
        or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
        or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
    ):
        raise PermissionError("original workflow helper source changed")
    _workflow_wave_sources(wave)
    if (
        item._wave is not wave
        or wave.driver is not driver
        or driver.get_coro().cr_frame is not run_frame
        or wave.runtime._workflow_waves.get(wave) is not wave
        or closing.cr_frame is not close_frame
        or caller.f_locals.get("closing") is not closing
        or caller.f_locals.get("self") is not wave
        or group is None
        and task is not driver
        or group is not None
        and (
            run_frame.f_locals.get("settle_group") is not group
            or not any(child is task for child in group._children)
            or task.get_coro().cr_frame is not caller
            or caller.f_locals.get("receipt") is not receipt
        )
    ):
        raise PermissionError("original wave close association changed")


def _workflow_wave_sources(wave=None):
    """Check the original native producers without treating a result as authority."""
    from tldw_chatbook.Workspaces import change_review_finalization as finalization
    from tldw_chatbook.Workspaces import change_review_ownership as native

    from .console_ordinary_tree import _function_record, _function_unchanged
    from .console_workflow_preparation import _PLANNING_PREPARATION_METHODS

    if (
        _function_record is not _WAVE_RECORDER[0]
        or _function_record.__code__ is not _WAVE_RECORDER[1]
        or _function_record.__defaults__ is not _WAVE_RECORDER[2]
        or _function_record.__kwdefaults__ is not _WAVE_RECORDER[3]
        or _function_record.__closure__ is not _WAVE_RECORDER[5]
        or _function_unchanged is not _WAVE_CHECKER[0]
        or _function_unchanged.__code__ is not _WAVE_CHECKER[1]
        or _function_unchanged.__defaults__ is not _WAVE_CHECKER[2]
        or _function_unchanged.__kwdefaults__ is not _WAVE_CHECKER[3]
        or _function_unchanged.__closure__ is not _WAVE_CHECKER[5]
        or asyncio.Task is not _WAVE_TASK_CLASS
        or asyncio.tasks._GatheringFuture is not _WAVE_GATHER_CLASS
        or any(
            vars(module).get(name) is not record[0] or not _function_unchanged(record)
            for module, name, record in _WAVE_ASYNC_FUNCTIONS
        )
        or _WaveOpenRefusal is not _WAVE_OPEN_RESULT_CLASS
        or any(not _function_unchanged(record) for record in _WAVE_OPEN_RESULT_METHODS)
        or native._NativeSnapshotIssuer is not _WAVE_ISSUER_CLASS
        or _WorkflowWave is not _WAVE_CLASS
        or native._SnapshotTicket is not _TICKET_CLASS
        or native._SnapshotSource is not _SnapshotSource
        or finalization.ChangeReviewReservation is not ChangeReviewReservation
        or any(
            globals().get(name) is not record[0] or not _function_unchanged(record)
            for name, record in _WAVE_FUNCTIONS
        )
        or wave is not None
        and (
            type(wave) is not _WAVE_CLASS
            or wave.coordinator is not None
            and (
                type(wave.coordinator._native) is not _WAVE_ISSUER_CLASS
                or wave.coordinator._native.runtime is not wave.runtime
                or wave.coordinator._native.coordinator is not wave.coordinator
                or inspect.getattr_static(wave.coordinator._native, "capture", None)
                is not _WAVE_CAPTURE_METHOD[0]
            )
            or any(
                inspect.getattr_static(item, name, None) is not descriptor
                or any(not _function_unchanged(record) for record in records)
                for item in wave.prepared
                for name, descriptor, records in _PLANNING_PREPARATION_METHODS
            )
        )
        or inspect.getattr_static(_WAVE_ISSUER_CLASS, "capture", None)
        is not _WAVE_CAPTURE_METHOD[0]
        or not _function_unchanged(_WAVE_CAPTURE_METHOD)
        or any(
            inspect.getattr_static(target, name, None) is not descriptor
            or any(not _function_unchanged(record) for record in records)
            or target is _WAVE_CLASS
            and wave is not None
            and inspect.getattr_static(wave, name, None) is not descriptor
            for target, name, descriptor, records in _WAVE_METHODS
        )
    ):
        raise PermissionError("original workflow wave source changed")


def _workflow_wave_return(wave, saved=None):
    """Retain/recheck the actual original wave's finite positive returns."""
    from . import console_workflow_preparation as preparation
    from .console_ordinary_tree import _function_unchanged

    if (
        _workflow_wave_sources is not _WAVE_SOURCE[0]
        or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
        or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
        or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
        or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
    ):
        raise PermissionError("original workflow helper source changed")

    if (
        _workflow_wave_sources is not _WAVE_SOURCE[0]
        or _workflow_wave_sources.__code__ is not _WAVE_SOURCE[1]
        or _workflow_wave_sources.__defaults__ is not _WAVE_SOURCE[2]
        or _workflow_wave_sources.__kwdefaults__ is not _WAVE_SOURCE[3]
        or _workflow_wave_sources.__closure__ is not _WAVE_SOURCE[5]
    ):
        raise PermissionError("original workflow helper source changed")
    _workflow_wave_sources(wave)
    if preparation._finite_owner_return is not preparation._FINITE_OWNER_CHECK[
        0
    ] or not _function_unchanged(preparation._FINITE_OWNER_CHECK):
        raise PermissionError("original workflow wave owner source changed")
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
        or any(first is not second for first, second in zip(saved[:6], current[:6]))
        or any(
            len(first) != len(second) or any(a is not b for a, b in zip(first, second))
            for first, second in zip(saved[6:], current[6:])
        )
    ):
        raise RuntimeError("original workflow wave return changed")
    if (
        not wave.closed
        or not wave.driver.done()
        or wave.driver.cancelled()
        or wave.driver.exception() is not None
        or wave.driver.get_coro().cr_code is not _WAVE_RUN_CODE
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
            or any(
                inspect.getattr_static(instance, name, None) is not descriptor
                for target, name, descriptor, _records in _WAVE_METHODS
                for candidate, instance in (
                    (_TICKET_CLASS, ticket),
                    (_SnapshotSource, source),
                    (ChangeReviewReservation, wave.reservation),
                )
                if target is candidate
            )
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


from tldw_chatbook.Workspaces.change_review_finalization import ChangeReviewReservation
from tldw_chatbook.Workspaces.change_review_ownership import (
    _SnapshotSource,
    _SnapshotTicket,
)

from .console_ordinary_tree import _function_record as _wave_function_record

_WAVE_CLASS = _WorkflowWave
_TICKET_CLASS = _SnapshotTicket
_WAVE_RUN_CODE = _WorkflowWave.run.__code__
_WAVE_FUNCTIONS = tuple(
    (name, _wave_function_record(globals()[name]))
    for name in (
        "run_workflow_wave",
        "_workflow_wave_sources",
        "_workflow_wave_return",
        "_workflow_launch_association",
        "_workflow_launch_association_current",
        "_workflow_launch_received",
        "_workflow_close_association",
        "_workflow_close_current",
    )
)
_WAVE_METHODS = tuple(
    (
        target,
        name,
        descriptor,
        tuple(
            _wave_function_record(function)
            for function in (
                (descriptor.fget,)
                if isinstance(descriptor, property)
                else (
                    descriptor,
                    *(
                        [descriptor.__wrapped__]
                        if hasattr(descriptor, "__wrapped__")
                        else []
                    ),
                )
            )
        ),
    )
    for target in (
        _WAVE_CLASS,
        _SnapshotTicket,
        _SnapshotSource,
        ChangeReviewReservation,
    )
    for name, descriptor in vars(target).items()
    if inspect.isfunction(descriptor) or isinstance(descriptor, property)
)

_WAVE_LAUNCH_CODE = _WorkflowWave._launch.__code__
_WAVE_OPEN_CODE = _WorkflowWave._open.__code__
_WAVE_SETTLE_CODE = _WorkflowWave._settle.__code__
_WAVE_WORKER_CODE = next(
    code
    for code in _WAVE_LAUNCH_CODE.co_consts
    if isinstance(code, type(_WAVE_LAUNCH_CODE)) and code.co_name == "launch"
)
_WAVE_TO_THREAD_CODE = asyncio.to_thread.__code__
_WAVE_GATHER_CLASS = asyncio.tasks._GatheringFuture
_WAVE_TASK_CLASS = asyncio.Task
_WAVE_ASYNC_FUNCTIONS = tuple(
    (module, name, _wave_function_record(vars(module)[name]))
    for module, names in (
        (asyncio, ("to_thread", "gather")),
        (asyncio.threads, ("to_thread",)),
        (asyncio.tasks, ("gather",)),
    )
    for name in names
)

from .console_ordinary_tree import _function_unchanged as _wave_unchanged

_WAVE_CHECKER = _wave_function_record(_wave_unchanged)

_WAVE_SOURCE = dict(_WAVE_FUNCTIONS)["_workflow_wave_sources"]
_WAVE_ASSOCIATION_SOURCE = dict(_WAVE_FUNCTIONS)["_workflow_launch_association"]
_WAVE_ASSOCIATION_CURRENT_SOURCE = dict(_WAVE_FUNCTIONS)[
    "_workflow_launch_association_current"
]
_WAVE_RECEIVED_SOURCE = dict(_WAVE_FUNCTIONS)["_workflow_launch_received"]
_WAVE_CLOSE_SOURCE = dict(_WAVE_FUNCTIONS)["_workflow_close_association"]
_WAVE_CLOSE_CURRENT_SOURCE = dict(_WAVE_FUNCTIONS)["_workflow_close_current"]

_WAVE_RECORDER = _wave_function_record(_wave_function_record)

from _thread import LockType as _WAVE_LOCK_CLASS

from tldw_chatbook.Workspaces.change_review_ownership import (
    _NativeSnapshotIssuer as _WAVE_ISSUER_CLASS,
)

_WAVE_CAPTURE_METHOD = _wave_function_record(_WAVE_ISSUER_CLASS.capture)
_WAVE_OPEN_RESULT_CLASS = _WaveOpenRefusal
_WAVE_OPEN_RESULT_METHODS = tuple(
    _wave_function_record(method)
    for method in vars(_WaveOpenRefusal).values()
    if inspect.isfunction(method)
)
