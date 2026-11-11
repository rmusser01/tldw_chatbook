"""Original user-confirmed Undo custody, separate from snapshot policy."""

from __future__ import annotations

import asyncio
import copy
import json
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar, copy_context
from pathlib import Path
from threading import Event, Lock, get_ident

from tldw_chatbook.app_services import read_console_runtime
from tldw_chatbook.Agents.activation import _guard, execution
from tldw_chatbook.Agents.execution_capacity import CapacityRefused, WorkOrigin
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Workspaces.change_tracking import (
    NativeUndoIssuer as NativeUndoIssuer,
    RevertRefusedError,
)
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

_current_undo = ContextVar("native_confirmed_undo", default=None)


class _UndoActivationRefused(RevertRefusedError):
    """No effect callback entered, and every partial scope closed normally."""


def native_undo_call():
    """Return only custody active on the original worker."""
    call = _current_undo.get()
    return call if call is not None and call.thread == get_ident() else None


def capture_undo(provider):
    """Capture the original app feature source without granting confirmation."""
    issuer = getattr(provider, "_undo_native", None)
    if issuer is None:
        if getattr(provider, "_native_undo_expected", False):
            raise RevertRefusedError("original Undo source unavailable")
        return None
    with issuer.runtime._first_send_source_scope():
        return _UndoConfirmation(issuer, provider)


class _UndoConfirmation:
    def __init__(self, issuer, provider):
        from tldw_chatbook import config

        self.issuer, self.runtime, self.provider = issuer, issuer.runtime, provider
        self.app = self.runtime._app
        self.bridge = self.runtime._agent_bridge
        self.store = self.runtime._chat_store
        self.controller = self.runtime._chat_controller
        self.tracker = self.bridge._change_tracker
        self.db, self.service = provider._db, provider._service
        self.db_path, self.db_local = self.db.db_path, self.db._thread_local
        self.conversation = provider._conversation_id
        self.persistence = getattr(self.store, "persistence", None)
        self.chat_db = getattr(self.persistence, "db", None)
        self.shadow_root, self.git = self.service._data_dir, self.service._git
        self.scratch = self.runtime.scratch_spaces
        self.config_generation = config.get_runtime_config_generation()
        self.sources = self.selection = None
        self.chains, self.scratch_snapshots, self.rows = {}, {}, []
        self.prepared = self.confirmed = self.consumed = False

    def check_memory(self, *, completion=False):
        from tldw_chatbook import config

        r, p = self.runtime, self.provider
        if (
            not completion
            and (r._disposed or self.issuer.closed)
            or r._app is not self.app
            or read_console_runtime(self.app) is not r
            or r._change_revert_native is not self.issuer
            or r._agent_bridge is not self.bridge
            or self.bridge._change_revert_native is not self.issuer
            or p._undo_native is not self.issuer
            or r._chat_store is not self.store
            or self.bridge._store is not self.store
            or r._chat_controller is not self.controller
            or self.bridge._change_tracker is not self.tracker
            or self.tracker.service is not self.service
            or p._service is not self.service
            or p._db is not self.db
            or self.bridge._db is not self.db
            or r._agent_runs_db is not self.db
            or self.db.db_path != self.db_path
            or self.db._thread_local is not self.db_local
            or p._conversation_id != self.conversation
            or getattr(self.store, "persistence", None) is not self.persistence
            or getattr(self.persistence, "db", None) is not self.chat_db
            or self.service._data_dir != self.shadow_root
            or self.service._git != self.git
            or r.scratch_spaces is not self.scratch
            or config.get_runtime_config_generation() != self.config_generation
            or self.issuer.capacity is not None
            and r._execution_capacity is not self.issuer.capacity
            or self.issuer.activity is not None
            and r._worktree_activity is not self.issuer.activity
        ):
            raise RevertRefusedError("original Undo source changed")

    def check(self, *, completion=False):
        self.check_memory(completion=completion)
        if self.selection is not None and _guard.sources(self.bridge) != self.selection:
            raise RevertRefusedError("original Undo storage selection changed")
        self.check_memory(completion=completion)

    def confirm(self):
        """Called only by the original positive modal callback."""
        self.check_memory()
        if not self.prepared or self.confirmed or self.consumed:
            raise RevertRefusedError("Undo confirmation is no longer current")
        self.confirmed = True

    @contextmanager
    def activate(self, owner=None):
        entry_error = None
        with ExitStack() as stack:
            try:
                stack.enter_context(execution(sources=self.sources))
                if owner is not None:
                    stack.enter_context(owner.activate())
                for db in dict.fromkeys((self.db, self.chat_db)):
                    if db is not None:
                        stack.enter_context(operation_owned_connection(db))
                stack.enter_context(acquire_storage(self.shadow_root))
            except BaseException as exc:  # noqa: BLE001 -- retain entry error until every partial scope unwinds
                entry_error = exc
            else:
                yield stack
        # This marker is unreachable if partial scope unwinding failed.
        if entry_error is not None:
            raise _UndoActivationRefused(
                "original Undo storage unavailable"
            ) from entry_error

    def validate_row(self, row):
        self.check(completion=True)
        run = self.db.get_run(str(row.get("run_id", "")))
        current = self.db.change_snapshots_for_run_review(str(row.get("run_id", "")))
        if (
            run is None
            or run["conversation_id"] != self.conversation
            or not any(value == row for value in current)
        ):
            raise RevertRefusedError("original Undo snapshot changed")

    async def _admit(self, *, applying):
        async with self.runtime._first_send_source_scope_async():
            call = _UndoCall(self, applying=applying)
            try:
                await run_preparation_read(
                    call.admit,
                    creator=self.issuer,
                    session_id=None,
                    reads=self.runtime._preparation_reads,
                    require_current=self.check_memory,
                    source=self,
                )
                self.check_memory()
                return call
            except BaseException:
                await call.cancel_unused()
                raise

    async def prepare(self, callback, *, marshal, current):
        call = await self._admit(applying=False)
        try:
            result = await call.run_async(callback, marshal=marshal, current=current)
            self.check_memory()
            self.prepared = True
            return result
        finally:
            await call.cancel_unused()

    async def apply(self, callback, *, marshal, current, publish):
        self.check_memory()
        if not self.confirmed or self.consumed:
            raise RevertRefusedError("original Undo confirmation required")
        self.consumed = True
        call = await self._admit(applying=True)
        try:
            return await call.run_async(
                callback, marshal=marshal, current=current, publish=publish
            )
        finally:
            await call.cancel_unused()


class _UndoCall:
    def __init__(self, source, *, applying):
        self.source, self.applying = source, applying
        self.owner = self.operation = self.claim = None
        self.retired = Event()
        self._start_lock = Lock()
        self._started = self._cancelled = self._finished = False
        self.required_proven = self.host_proven = True
        self._stack = None
        self.thread = None
        self._leased = set()
        self._reverted = set()

    def admit(self):
        s, issuer = self.source, self.source.issuer
        s.check()
        if s.sources is None:
            s.selection = _guard.sources(s.bridge)
            s.sources = _guard.captured_sources(s.bridge)
        with s.activate():
            s.check()
            issuer.capacity = issuer.capacity or s.runtime.execution_capacity
            issuer.activity = issuer.activity or s.runtime.worktree_activity
            s.check()
            with issuer.activity.prelaunch(
                footprint=WorktreeFootprint(opaque=True)
            ) as claim:
                if claim is None:
                    raise RevertRefusedError(
                        "Undo is unavailable while native work is active"
                    )
                try:
                    owner = issuer.capacity.begin_execution(
                        origin=WorkOrigin.MANUAL, conversation_id=s.conversation
                    )
                    try:
                        operation = owner.reserve_tool()
                    except BaseException:
                        owner.finish_root()
                        raise
                except CapacityRefused as exc:
                    raise RevertRefusedError(
                        "Undo native capacity is unavailable"
                    ) from exc
                try:
                    with s.runtime._execution_capacity_lock:
                        s.check_memory()
                        claim.bind_owner(owner)
                        self.owner, self.operation, self.claim = owner, operation, claim
                        issuer.pending.add(self)
                        # The admitted member carries capture through native return.
                        s.runtime._first_send_sources += 1
                except BaseException:
                    operation.finish()
                    owner.finish_root()
                    raise

    def prevent_start(self):
        with self._start_lock:
            self._cancelled = True
            return not self._started

    async def cancel_unused(self):
        if self.prevent_start() and self.operation is not None:
            await run_preparation_read(
                self.finish_unused,
                creator=self.source.issuer,
                session_id=None,
                reads=self.source.runtime._preparation_reads,
                require_current=lambda: None,
            )

    def finish_unused(self):
        with self._start_lock:
            if self._started or self._finished:
                return
            self._finished = True
        try:
            with self.source.activate(self.owner):
                self.operation.finish()
        except _UndoActivationRefused:
            self.finish_unentered()
        except BaseException:  # noqa: BLE001 -- scope or finish uncertainty retains the original call
            self.retain_unknown()
        self.retire()

    def finish_unentered(self):
        """Retire metadata only after proven no-entry and positive scope unwind."""
        try:
            with self.owner.activate():
                self.operation.finish()
        except BaseException:  # noqa: BLE001 -- unknown finish return cannot retire an unused ticket
            self.retain_unknown()

    def retain_unknown(self):
        self.operation.mark_stopping()
        Event().wait()  # No original settlement retry producer exists.

    def retire(self):
        self.owner.finish_root()
        self.claim.settle(durable=True)
        with self.source.runtime._execution_capacity_lock:
            if self in self.source.issuer.pending:
                self.source.issuer.pending.remove(self)
                self.source.runtime._first_send_sources -= 1
        self.retired.set()
        self.source.runtime._wake_first_send()

    def validate_root(self, root):
        s = self.source
        s.check(completion=True)
        root = Path(root).resolve(strict=True)
        key = str(root)
        if key not in s.chains:
            if self.applying:
                raise RevertRefusedError("Undo root is outside original confirmation")
            s.chains[key] = capture_directory_chain(root)
            s.scratch_snapshots[key] = s.scratch.existing_snapshot_for_root(root)
        if capture_directory_chain(root) != s.chains[key]:
            raise RevertRefusedError("original Undo directory changed")
        snapshot = s.scratch_snapshots[key]
        if key not in self._leased:
            if snapshot is not None:
                self._stack.enter_context(s.scratch.lease(snapshot))
            self._leased.add(key)
        s.check_memory(completion=True)

    def record_preflight(self, provider, row, paths, result):
        s = self.source
        if provider is not s.provider:
            raise RevertRefusedError("original Undo provider changed")
        if not self.applying:
            s.validate_row(row)
            s.rows.append((copy.deepcopy(row), tuple(paths), copy.deepcopy(result)))

    def require_revert(self, service, db, row, paths):
        """Consume one original selection at the shared engine's first entry."""
        s = self.source
        if (
            not self.applying
            or not self.required_proven
            or (row.get("id"), tuple(paths)) in self._reverted
            or not s.confirmed
            or not s.consumed
            or service is not s.service
            or db is not s.db
            or not any(
                row == original and tuple(paths) == selected
                for original, selected, _ in s.rows
            )
        ):
            raise RevertRefusedError("original confirmed Undo operation required")
        s.check(completion=True)
        self._reverted.add((row.get("id"), tuple(paths)))

    def revalidate(self):
        from .change_revert import preflight_revert

        s = self.source
        for row, paths, original in s.rows:
            s.validate_row(row)
            self.validate_root(row["root"])
            if s.provider.run_active_for_root(str(row["root"])):
                raise RevertRefusedError(
                    "a run is active on this workspace — finish or stop the run first"
                )
            current = preflight_revert(s.service, row, paths)
            if (
                current.disk_state != original.disk_state
                or current.edited_since != original.edited_since
            ):
                raise RevertRefusedError(
                    "Files changed while the confirmation was open. Open Review and try again."
                )

    def observe_reverted(self, row, reverted):
        """Read the exact original row; update returning None proves no row count."""
        s = self.source
        try:
            s.check(completion=True)
            rows = s.db.change_snapshots_for_run_review(str(row["run_id"]))
            expected = list(
                dict.fromkeys([*json.loads(row.get("reverted") or "[]"), *reverted])
            )
            for fresh in rows:
                if fresh["id"] == row["id"]:
                    previous = dict(fresh, reverted=row.get("reverted"))
                    return (
                        previous == row
                        and json.loads(fresh.get("reverted") or "[]") == expected
                    )
        except Exception:  # noqa: BLE001 -- unknown required outcome retains custody
            return False
        return False

    async def run_async(self, callback, *, marshal, current, publish=None):
        host_context = copy_context()
        worker = asyncio.create_task(
            asyncio.to_thread(
                self.run, callback, marshal, current, publish, host_context
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            await self.cancel_unused()
            # Consume the actual callback exception without interpreting it as drain.
            worker.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
            raise

    def run(self, callback, marshal, current, publish, host_context):
        s = self.source
        try:
            with s.runtime._execution_capacity_lock, self._start_lock:
                allowed = not (
                    self._cancelled or s.runtime._disposed or s.issuer.closed
                )
                if allowed:
                    s.check_memory()
                    self._started = True
        except BaseException:
            self.prevent_start()
            self.finish_unused()
            raise
        if not allowed:
            self.finish_unused()
            raise RevertRefusedError("Undo was cancelled before execution")
        try:
            result, error = self.run_entered(
                callback, marshal, current, publish, host_context
            )
        except _UndoActivationRefused:
            self.finish_unentered()
            self.retire()
            raise
        except BaseException:  # noqa: BLE001 -- unknown scope exit retains original worker custody
            # Callback errors are handled inside. Scope/finish uncertainty is not
            # a no-effect refusal, even if the callback itself never entered.
            self.retain_unknown()
        if error is not None and publish is None:
            raise error
        return result

    def run_entered(self, callback, marshal, current, publish, host_context):
        s = self.source
        result = error = None
        with s.activate(self.owner) as stack:
            self._stack = stack
            self.thread = get_ident()
            token = _current_undo.set(self)
            try:
                try:
                    s.check(completion=True)
                    if self.applying:
                        self.revalidate()
                    result = callback()
                except BaseException as exc:  # noqa: BLE001 -- deliver the original callback outcome before settlement
                    error = exc

                def deliver():
                    try:
                        s.check_memory(completion=True)
                    except RevertRefusedError:
                        return False
                    if s.runtime._disposed or not current():
                        return False
                    if publish is not None:
                        publish(result, error)
                    return True

                self.host_proven = False
                try:
                    # Textual copies context into its loop; retain the original host's.
                    accepted = host_context.run(marshal, deliver)
                    self.host_proven = True
                    if not accepted and not self.applying:
                        error = RevertRefusedError(
                            "Undo preparation is no longer current"
                        )
                except BaseException as exc:  # noqa: BLE001 -- unknown host return retains original worker custody
                    error = error or exc
                    try:
                        s.check_memory(completion=True)
                        self.host_proven = s.runtime._disposed or not current()
                    except RevertRefusedError:
                        self.host_proven = True
                if not (self.required_proven and self.host_proven):
                    self.retain_unknown()
                try:
                    self.operation.finish()
                except BaseException:  # noqa: BLE001 -- finish return is required physical settlement
                    self.retain_unknown()
            finally:
                _current_undo.reset(token)
                self._stack = None
        self.retire()
        return result, error
