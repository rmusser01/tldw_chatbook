"""One finite accepted native phase wave; the existing owners execute all work."""

from __future__ import annotations

import asyncio
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
        self._release_return = None
        self.check_prepared_members(tuple(item.authorization for item in self.prepared))

    def check_prepared_members(self, members):
        self.phase.check()
        if len(members) != len(self.prepared) or len(
            {id(item) for item in members}
        ) != len(members):
            raise PermissionError("distinct original wave members required")
        for item, member in zip(self.prepared, members, strict=True):
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

    def _release(self, *, _unused=False):
        _workflow_wave_sources(self)
        if (
            self.closed
            or self.unproven_launches
            or asyncio.current_task() is not self.driver
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
        if self.reservation is not None and not _unused:
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
                records = tuple(item._finite_reads)
                read = records[-1]
                item._finite_read_return(read)
                item._execution_reads_source()
                try:
                    returned = item._release_configuration()
                    item._configuration_received(returned)
                except BaseException as error:
                    item._configuration_transition_error = error
                    raise
                item._terminal_configuration_return = returned, read, records
                item._execution_configuration_return()
        # All original callouts returned; enrollment protects them until now.
        _workflow_wave_sources(self)
        if self.runtime._workflow_waves.get(self) is not self:
            raise PermissionError("original workflow wave registration changed")
        self.closed = True
        self.runtime._workflow_waves.pop(self)
        return (
            self,
            self.driver,
            self._driver_coroutine,
            self.source,
            self.reservation,
            self.fence,
            self.baseline_return,
            self.ending_return,
            self.prepared,
            tuple(self._snapshot_returns),
            tuple(self.receipts),
            tuple(
                (item, item._terminal_configuration_return)
                for item in self.prepared
                if not self.retain_waits or item not in self.unlaunched
            ),
        )

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
        try:
            captured = await first._read(capture)
        except _WorkflowChildWait as error:
            if first._finite_reads is not reads or len(reads) != read_index + 1:
                raise
            read = reads[read_index]
            first._finite_read_return(read)
            if (
                not read.positive
                or read.body_error is not error
                or read.wrapper_error is not None
                or read.owner is not None
                or read.claim is not None
                or read.operation is not None
                or self.source is not None
            ):
                raise
            return _WaveOpenRefusal(error, self.retain_waits)
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
                closing = item.close(_wave=self)
                await closing
            _workflow_wave_sources(self)
            self._release_return = self._release(_unused=True)
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
                closing = item.close(_unpublished=publication, _wave=self)
                await closing
            _workflow_wave_sources(self)
            self._release_return = self._release(_unused=True)
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
                closing = item.close(_wave=self)
                await closing
            await run_owned_db_call(
                self.service.db,
                self.service.db.workflow_runs.unused_change_window,
                self.reservation.id,
            )
            _workflow_wave_sources(self)
            self._release_return = self._release(_unused=True)
            raise PermissionError("workflow baseline unavailable")

    async def _launch(self, item):
        from tldw_chatbook.Agents.agent_service import _WorkflowLaunchReturn

        self._check_enrollment(item)
        unused = []
        unused_returned = False

        def launch():
            with (
                owned_goal_storage(self.service),
                self.phase.context.scope(),
                execution(self.bridge),
                item.activate_source(),
            ):
                if self.stopping.is_set() or item.should_cancel():
                    result = PermissionError("workflow native source changed")
                    unused.append(result)
                else:
                    result = item.native.service._workflow_launch_result(
                        item.authorization.task,
                        authorization=item.authorization,
                        should_cancel=lambda: (
                            self.stopping.is_set() or item.should_cancel()
                        ),
                        _wave=self,
                    )
                    if result.receipt is not None:
                        self.service.bind_native_receipt(
                            item.authorization, result.receipt
                        )
            return result

        try:
            result = await asyncio.to_thread(launch)
            _workflow_wave_sources(self)
            if unused:
                if len(unused) != 1 or result is not unused[0]:
                    raise PermissionError("workflow no-entry result changed")
                unused_returned = True
                raise result
            if not isinstance(result, _WorkflowLaunchReturn):
                raise PermissionError("native workflow launch result required")
            if not result.known:
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
            closing = item.close(_wave=self)
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
        closing = item.close(_wave=self)
        await closing

    async def run(self):
        """Launch a fixed eligible set; close E at its first actual quiescent gap."""
        _workflow_wave_sources(self)
        opening = self._open()
        waiting = await opening
        _workflow_wave_sources(self)
        if waiting is not None:
            if (
                type(waiting) is not _WaveOpenRefusal
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
                    closing = item.close(_wave=self)
                    await closing
            _workflow_wave_sources(self)
            self._release_return = self._release()
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
                closing = item.close(_wave=self)
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
        _workflow_wave_sources(self)
        self._release_return = self._release()
        for result in launches:
            if isinstance(result, BaseException):
                raise result
        if self.unlaunched and not self.retain_waits:
            raise next(iter(self.unlaunched.values()))
        return self.reservation


async def run_workflow_wave(phase, prepared, *, _retain_waits=False):
    """Private complete-wave consumer for the existing accepted phase owner."""
    _workflow_wave_sources()
    wave = _WorkflowWave(phase, prepared, retain_waits=_retain_waits)
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
    _workflow_wave_sources(wave)
    if (
        item._wave is not wave
        or item.phase is not wave.phase
        or wave.closed
        or wave.runtime._workflow_waves.get(wave) is not wave
        or not any(member is item for member in wave.prepared)
    ):
        raise PermissionError("workflow close enrollment changed")


def _workflow_wave_sources(wave=None):
    """Validate the runtime resources owned by this registered native wave."""
    if wave is None:
        return
    if (
        not isinstance(wave, _WorkflowWave)
        or wave.runtime.chat_controller is not wave.phase._controller
        or wave.phase._controller._agent_bridge is not wave.bridge
        or wave.runtime._workflow_service is not wave.service
        or any(
            item.runtime is not wave.runtime
            or item.phase is not wave.phase
            or item.service is not wave.service
            for item in wave.prepared
        )
        or wave.coordinator is not None
        and (
            wave.coordinator._native.runtime is not wave.runtime
            or wave.coordinator._native.coordinator is not wave.coordinator
        )
    ):
        raise PermissionError("workflow wave resources changed")


def _workflow_wave_release_return(wave, prepared):
    """Prove original release callouts, without certifying driver success."""
    _workflow_wave_sources(wave)
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

    _workflow_wave_sources(wave)
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

from _thread import LockType as _WAVE_LOCK_CLASS
