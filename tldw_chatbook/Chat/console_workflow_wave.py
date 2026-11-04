"""One finite accepted native phase wave; the existing owners execute all work."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event

from tldw_chatbook.Agents.activation import execution
from tldw_chatbook.Agents.execution_capacity import WorkOrigin
from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
from tldw_chatbook.Agents.workflow_models import WorkflowTaskResult
from tldw_chatbook.DB.base_db import run_owned_db_call
from tldw_chatbook.Workspaces.change_review_finalization import ChangeReviewCompletion
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint, WorktreeScope


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
        self.unlaunched = {}
        self.unproven_launches = set()
        self.retain_waits = retain_waits
        self.wait_footprint = None
        self.anchor = None
        self.closed = False
        self.stopping = Event()
        self.driver = None
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
            reservation is not self.reservation
            or ticket.source is not self.source
            or not ticket.retired.is_set()
        ):
            raise PermissionError("original snapshot return required")
        if kind == "discover":
            return
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
        if self.fence is not None:
            self.runtime.worktree_activity._close_wave(self.fence)
        self.closed = True
        self.runtime._workflow_waves.pop(self, None)
        for item in self.prepared:
            if not self.retain_waits or item not in self.unlaunched:
                item._release_configuration()

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
            if self.retain_waits:
                return wait
            raise wait
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
                if self.retain_waits:
                    return wait
                raise wait
        footprints = tuple(
            item.authorization.physical_footprint for item in self.prepared
        )
        if any(
            first._conflicts(second)
            for i, first in enumerate(footprints)
            for second in footprints[i + 1 :]
        ):
            raise _WorkflowChildWait(
                "worktree_activity", "workflow wave members conflict"
            )
        if not self.capture_enabled:
            for item in self.prepared:
                item._wave = self
            return
        roots = self.roots
        from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

        first = self.prepared[0]

        def capture():
            self.source = self.coordinator._native.capture(
                self.bridge,
                session_id=first.session.id,
                conversation_id=self.phase.attempt.conversation_id,
                origin=WorkOrigin.AUTOMATIC,
                roots=roots,
                workflow_wave=self,
            )
            return WorktreeFootprint(
                writes=tuple(
                    WorktreeScope(Path(root), True, capture_directory_chain(Path(root)))
                    for root in roots
                )
            )

        footprint = await first._read(capture)
        self.wait_footprint = footprint
        self.fence = self.runtime.worktree_activity._open_wave(
            self, footprint, tuple(item.authorization for item in self.prepared)
        )
        if self.fence is None:
            if self.retain_waits:
                return _WorkflowChildWait(
                    "worktree_activity", "workflow wave activity busy"
                )
            raise PermissionError("workflow wave activity busy")
        self.runtime._workflow_waves[self] = self
        for item in self.prepared:
            item._wave = self
        with execution(self.bridge), owned_goal_storage(self.service):
            self.reservation = self.coordinator.register(
                roots, native_source=self.source, on_unadmitted=self.unadmitted
            )
        if self.reservation is None or self.reservation.admission_error:
            if self.reservation is not None:
                self.coordinator.cancel(self.reservation)
            for item in self.prepared:
                await item.close()
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
                await item.close(_unpublished=publication)
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
                await item.close()
            await run_owned_db_call(
                self.service.db,
                self.service.db.workflow_runs.unused_change_window,
                self.reservation.id,
            )
            self._release()
            raise PermissionError("workflow baseline unavailable")

    async def _launch(self, item):
        from tldw_chatbook.Agents.agent_service import _WorkflowChildLaunchFailed

        def launch():
            if self.stopping.is_set():
                raise PermissionError("workflow wave cancelled before child")
            native = item.authorization._native_preparation[0]
            with item.authorization._lock:
                prior = native.wait_return[0] if native.wait_return else None
            wait = None
            body_error = None
            source_returned = False
            try:
                with (
                    owned_goal_storage(self.service),
                    self.phase.context.scope(),
                    execution(self.bridge),
                    item.activate_source(),
                ):
                    try:
                        try:
                            receipt = item.native.service.launch_workflow_task(
                                item.authorization.task,
                                authorization=item.authorization,
                                should_cancel=lambda: (
                                    self.stopping.is_set() or item.should_cancel()
                                ),
                            )
                        except _WorkflowChildLaunchFailed as error:
                            receipt, failure = error.receipt, error
                        except BaseException as error:
                            with item.authorization._lock:
                                candidate = (
                                    native.wait_return[0]
                                    if native.wait_return
                                    else None
                                )
                            if (
                                candidate is None
                                or candidate is prior
                                or candidate[0] is not error
                            ):
                                raise
                            # Suppress ONLY this invocation's post-inner-return wait.
                            # All outer scopes must now return normally, even if an
                            # unwind were to raise this very same exception object.
                            wait = candidate
                            receipt, failure = None, None
                        else:
                            failure = None
                        if receipt is not None:
                            self.service.bind_native_receipt(
                                item.authorization, receipt
                            )
                    except BaseException as error:
                        body_error = error
                        raise
                source_returned = True
            except BaseException as error:
                with item.authorization._lock:
                    unproven = (
                        bool(native.wait_return) and native.wait_return[0] is None
                    )
                source_returned = not unproven and wait is None and error is body_error
                raise
            finally:
                operation = item._launch_operation
                if source_returned and operation is not None:
                    operation.finish()
                    item._launch_operation = None
            if wait is not None:
                error, owner, claim = wait
                if owner is not None:
                    drained = []
                    owner.on_drained(drained.append)
                    if drained != [True]:
                        raise PermissionError("workflow capacity wait drain unproven")
                activity = self.runtime.worktree_activity
                with activity._lock:
                    if claim is not None and (
                        claim.id in activity._claims or claim.id in activity._members
                    ):
                        raise PermissionError(
                            "workflow capacity wait activity return unproven"
                        )
                if (
                    item.authorization._native_launch
                    or item.authorization._native_dispatch
                ):
                    raise PermissionError("workflow capacity wait dispatched")
                return None, error
            return receipt, failure

        try:
            receipt, error = await asyncio.to_thread(launch)
        except BaseException:
            native = item.authorization._native_preparation[0]
            with item.authorization._lock:
                if native.wait_return:
                    self.unproven_launches.add(item)
            raise
        if receipt is None:
            # Only after the actual original worker RETURN is this pending.
            self.unlaunched[item] = error
            return None
        self.receipts.append((item, receipt))
        if error is not None:
            raise error
        return receipt

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
            await item.close()
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
        await item.close()

    async def run(self):
        """Launch a fixed eligible set; close E at its first actual quiescent gap."""
        waiting = await self._open()
        if waiting is not None:
            self.unlaunched.update((item, waiting) for item in self.prepared)
            self._release()
            return self.reservation
        for item in self.prepared:
            item._wave = self
        launches = await asyncio.gather(
            *(self._launch(item) for item in self.prepared), return_exceptions=True
        )
        settled = await asyncio.gather(
            *(self._settle(item, receipt) for item, receipt in self.receipts),
            return_exceptions=True,
        )
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
                await item.close()
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
    wave = _WorkflowWave(phase, prepared, retain_waits=_retain_waits)
    if wave.runtime._workflow_closing:
        raise PermissionError("workflow admission is closing")
    wave.runtime._workflow_waves[wave] = wave
    wave.driver = asyncio.create_task(wave.run(), name="native-workflow-wave")
    cancelled = False
    try:
        while not wave.driver.done():
            try:
                await asyncio.shield(wave.driver)
            except asyncio.CancelledError:
                cancelled = True
                wave.stopping.set()
            except BaseException:  # noqa: BLE001 - original driver result retains failure custody
                break
        result = wave.driver.result()
        if cancelled:
            raise asyncio.CancelledError
        if _retain_waits:
            for item in wave.unlaunched:
                item._wave = None
            return wave
        return result
    finally:
        if (
            wave.reservation is None
            and not wave.closed
            and not wave.receipts
            and not wave.unproven_launches
        ):
            if wave.fence is not None:
                wave.runtime.worktree_activity._close_wave(wave.fence)
            wave.runtime._workflow_waves.pop(wave, None)
