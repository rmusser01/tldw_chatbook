"""One original paired-reader hold; no replaced reader, guard, or global hook."""

import asyncio
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem
import inspect
import sqlite3
import sys
import threading
import time
from types import CodeType, CoroutineType, FunctionType, MethodType

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver


def physically_closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True
    return False


class OriginalCharacterTeardownGate(OriginalStorageUnitObserver):
    def __init__(self, screen, host, exit_function, group):
        super().__init__({}, lambda: False, lambda _name: None)
        self.screen, self.host, self.group = screen, host, group
        self.controller = screen._character_context
        self.database = screen.app_instance.chachanotes_db
        self.loop, self.creator = asyncio.get_running_loop(), threading.current_thread()
        self.exit_function = exit_function
        self.entered, self.release = threading.Event(), threading.Event()
        self.wait_seen, self.refusal_seen = threading.Event(), threading.Event()
        self.invoke_returned = threading.Event()
        self.worker = self.thread = self.connection = self.lease = None
        self.operation = self.future = self.participant = None
        self.cold_invokes = set()
        self.shutdown_rows = []
        self.closer_rows = []
        self.release_controller = None
        self.stage_rows = []

    def mark_stage(self, stage):
        assert stage in {
            "drive_release_requested",
            "controller_entered_wait",
            "paired_reader_held",
            "real_resume_issued",
            "new_session_returned",
            "switch_session_returned",
            "stock_sync_worker_issued",
            "controller_release",
        }
        if len(self.stage_rows) >= 16:
            self._error("stage_overflow", OverflowError())
            return
        self.stage_rows.append({"stage": stage, "monotonic": time.monotonic()})

    def install(self):
        from textual import worker as worker_module
        from tldw_chatbook.Backup_Recovery import participants, storage_admission
        from tldw_chatbook.DB import base_db
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
        from tldw_chatbook.UI.Console_Modules.character_context import (
            ConsoleCharacterContextController,
        )

        assert type(self.database) is CharactersRAGDB and not self.database.is_memory_db
        assert type(self.controller) is ConsoleCharacterContextController
        for name in ("_start", "_return", "_hold", "_matched_worker", "boundary_facts"):
            function = inspect.getattr_static(type(self), name)
            self._pin(function)
            self.slots.append((type(self), name, function))
        self.storage, self.worker_module = storage_admission, worker_module
        self.worker_type = worker_module.Worker
        owner = ConsoleCharacterContextController
        metadata = inspect.getattr_static(owner, "_read_database_scope_metadata")
        assert type(metadata) is staticmethod
        self.metadata_code = self._pin(metadata.__func__)
        self.pair = inspect.getattr_static(owner, "_read_database_scope_metadata_pair")
        self.pair_code = self._pin(self.pair)
        self.work_owner = (
            self.screen if self.group == "console-sync" else self.controller
        )
        work_name = (
            "_sync_native_console_chat_ui"
            if self.group == "console-sync"
            else "refresh_if_scope_changed"
        )
        self.work_function = inspect.getattr_static(type(self.work_owner), work_name)
        self.work_code = self._pin(self.work_function)
        self.slots.append((type(self.work_owner), work_name, self.work_function))
        self.resume_function = None
        self.resume_code = None
        if self.group == "console-character-context-refresh":
            character_module = sys.modules[self.work_function.__module__]
            self.resume_function = vars(character_module).get(
                "_resume_stock_character_view"
            )
            if self.resume_function is not None:
                assert type(self.resume_function) is FunctionType
                self.resume_code = self._pin(self.resume_function)
                self.slots.append(
                    (
                        character_module,
                        "_resume_stock_character_view",
                        self.resume_function,
                    )
                )
        self._pin(base_db.run_owned_db_call)
        invokes = [
            code
            for code in base_db.run_owned_db_call.__code__.co_consts
            if type(code) is CodeType and code.co_name == "invoke"
        ]
        assert len(invokes) == 1
        self.invoke_code = invokes[0]
        self.work_item_code = self._pin(_WorkItem.run)
        self.wait_code = self._pin(worker_module.Worker.wait)
        self.close_code = self._pin(participants._close_settled_core_cache)
        self.exit_code = self._pin(self.exit_function)
        # The event still belongs to the passed original parent method. A
        # deliberate supported host override is a separate source dependency.
        exit_name = self.exit_function.__name__
        exit_owners = tuple(
            owner
            for owner in type(self.host).__mro__
            if vars(owner).get(exit_name) is self.exit_function
        )
        assert len(exit_owners) == 1
        self.slots.append((exit_owners[0], exit_name, self.exit_function))
        consumer_exit = inspect.getattr_static(type(self.host), exit_name)
        assert type(consumer_exit) is FunctionType
        if consumer_exit is not self.exit_function:
            self._pin(consumer_exit)
        self.slots.append((type(self.host), exit_name, consumer_exit))
        for name in ("running", "done", "cancelled"):
            function = inspect.getattr_static(Future, name)
            self._pin(function)
            self.slots.append((Future, name, function))
        for name in (
            "get_local_authority_id",
            "get_character_conversation_search_revision",
        ):
            function = inspect.getattr_static(CharactersRAGDB, name)
            self._pin(function)
            self.slots.append((CharactersRAGDB, name, function))
        self.slots.extend(
            (
                (owner, "_read_database_scope_metadata", metadata),
                (owner, "_read_database_scope_metadata_pair", self.pair),
                (base_db, "run_owned_db_call", base_db.run_owned_db_call),
                (worker_module, "active_worker", worker_module.active_worker),
                (_WorkItem, "run", _WorkItem.run),
                (worker_module.Worker, "wait", worker_module.Worker.wait),
                (
                    participants,
                    "_close_settled_core_cache",
                    participants._close_settled_core_cache,
                ),
            )
        )
        self.codes = {
            self.invoke_code: "original_invoke",
            self.metadata_code: "original_metadata",
            self.wait_code: "original_worker_wait",
            self.exit_code: "original_exit",
            self.close_code: "original_settled_close",
        }
        for tool in range(5, 0, -1):
            if tool == self.monitor.DEBUGGER_ID:
                continue
            try:
                self.monitor.use_tool_id(tool, "character-native-teardown")
            except ValueError:
                continue
            self.tool = tool
            break
        assert self.tool is not None and self.monitor.get_events(self.tool) == 0
        for event, callback in (
            (self.monitor.events.PY_START, self._start),
            (self.monitor.events.PY_RETURN, self._return),
        ):
            previous = self.monitor.register_callback(self.tool, event, callback)
            if previous is not None:
                self.monitor.register_callback(self.tool, event, previous)
                raise AssertionError("borrowed_monitor_callback")
            self.registered[event] = callback
        for code in self.codes:
            assert self.monitor.get_local_events(self.tool, code) == 0
            self.monitor.set_local_events(
                self.tool,
                code,
                self.monitor.events.PY_START | self.monitor.events.PY_RETURN,
            )
        self.active = self.installed = True

    def _matched_worker(self):
        worker = self.worker_module.active_worker.get(None)
        if (
            type(worker) is not self.worker_type
            or worker.node is not self.screen
            or worker.group != self.group
        ):
            return None
        work = vars(worker).get("_work")
        assert type(work) is CoroutineType
        work_frame = work.cr_frame
        if work.cr_code is self.work_code:
            assert (
                work_frame is not None
                and work_frame.f_globals is self.work_function.__globals__
                and work_frame.f_locals.get("self") is self.work_owner
            )
        else:
            # Additive qualification of this view's issued stock facade wrapper.
            assert self.resume_code is not None and work.cr_code is self.resume_code
            assert (
                work_frame is not None
                and work_frame.f_globals is self.resume_function.__globals__
                and work_frame.f_locals.get("controller") is self.controller
                and work_frame.f_locals.get("screen") is self.screen
            )
        assert (
            type(worker._task) is asyncio.Task and worker._task.get_loop() is self.loop
        )
        return worker

    def _error(self, stage, error):
        if len(self.invalid) < 12:
            self.invalid.append(stage + ":" + type(error).__name__)

    def _start(self, code, offset):
        if not self.active:
            return
        try:
            frame = self._frame(code)
            if (
                code is self.invoke_code
                and frame.f_locals.get("database") is self.database
            ):
                worker = self._matched_worker()
                operation = frame.f_locals.get("operation")
                if (
                    worker is not None
                    and type(operation) is MethodType
                    and operation.__self__ is self.controller
                    and operation.__func__ is self.pair
                ):
                    assert (
                        getattr(self.database._local, "conn", None) is None
                    ), "paired callback is not a cold new owner"
                    self.cold_invokes.add(id(worker))
            elif (
                code is self.wait_code
                and frame.f_locals.get("self") is self.worker
                and self.worker is not None
            ):
                assert (
                    threading.current_thread() is self.creator
                    and asyncio.get_running_loop() is self.loop
                )
                # A wait on an already-terminal cancelled asyncio task cannot
                # establish custody of its still-running executor callback.
                if not self.worker._task.done():
                    self.wait_seen.set()
        except Exception as error:
            self._error("start", error)

    def _hold(self, frame):
        if self.entered.is_set() or frame.f_locals.get("database") is not self.database:
            return False
        parent = frame.f_back
        if (
            parent is None
            or parent.f_code is not self.pair_code
            or parent.f_locals.get("self") is not self.controller
        ):
            return False
        worker = self._matched_worker()
        if worker is None:
            return False
        assert id(worker) in self.cold_invokes
        assert (
            frame.f_globals is self.controller._read_database_scope_metadata.__globals__
        )
        actor = threading.current_thread()
        assert actor is not self.creator
        future = None
        cursor = frame
        for _ in range(16):
            if cursor is None:
                break
            if cursor.f_code is self.work_item_code:
                item = cursor.f_locals.get("self")
                assert type(item) is _WorkItem
                future = item.future
                break
            cursor = cursor.f_back
        assert type(future) is Future and future.running() and not future.done()
        storage = self.storage
        with storage._lock:
            participant = self.database._maintenance_participant
            operation = getattr(storage._operation_local, "operation", None)
            assert (
                operation in storage._operations
                and operation.participant is participant
                and operation.thread is actor
            )
            assert operation.lease in storage._live_leases
            held = [
                (conn, lease)
                for conn, lease in participant.connections.items()
                if lease.resource_thread is actor
                and lease.resource_path == self.database.db_path
            ]
            assert len(held) == 1
            connection, lease = held[0]
            assert lease in storage._live_leases and not physically_closed(connection)
            assert self.database._connection_quiescence.is_registered(connection)
        self.worker, self.thread, self.future = worker, actor, future
        self.operation, self.participant = operation, participant
        self.connection, self.lease = connection, lease
        return True

    def _return(self, code, offset, value):
        if not self.active:
            return
        hold = False
        try:
            frame = self._frame(code)
            if code is self.metadata_code:
                hold = self._hold(frame)
            elif (
                code is self.invoke_code
                and self.worker is not None
                and frame.f_locals.get("database") is self.database
            ):
                operation = frame.f_locals.get("operation")
                if (
                    type(operation) is MethodType
                    and operation.__self__ is self.controller
                    and operation.__func__ is self.pair
                    and threading.current_thread() is self.thread
                ):
                    self.invoke_returned.set()
            elif (
                code is self.exit_code
                and frame.f_locals.get("self") is self.host
                and self.entered.is_set()
            ):
                self.shutdown_rows.append(
                    self.boundary_facts() | {"release_set": self.release.is_set()}
                )
            elif (
                code is self.close_code
                and frame.f_locals.get("repository") is self.database
                and self.entered.is_set()
            ):
                assert type(value) is bool  # noqa: E721 - exact original close result.
                self.closer_rows.append(
                    self.boundary_facts()
                    | {"allowed": value, "release_set": self.release.is_set()}
                )
                if not value:
                    self.refusal_seen.set()
            del frame, value
        except Exception as error:
            self._error("return", error)
        if hold:
            # No transient frame or parent remains retained during the hold.
            self.mark_stage("paired_reader_held")
            self.entered.set()
            self.drive_release()
            if not self.release.wait(10):
                self._error("hold", TimeoutError())

    def boundary_facts(self):
        with self.storage._lock:
            return {
                "native_live": self.connection is not None
                and not physically_closed(self.connection),
                "exact_operation_counted": self.operation in self.storage._operations,
                "exact_lease_live": self.lease in self.storage._live_leases,
                "pause_active": self.storage._pause is not None,
                "worker_task_done": self.worker is not None
                and self.worker._task.done(),
            }

    def drive_release(self):
        self.mark_stage("drive_release_requested")

        def release_after_real_boundary():
            try:
                self.mark_stage("controller_entered_wait")
                assert self.entered.wait(10)
                # A corrected owner waits before release. The original owner
                # instead reaches its real safe-close refusal while still live.
                for _ in range(1000):
                    if self.wait_seen.is_set() or self.refusal_seen.is_set():
                        break
                    if self.release.wait(0.01):
                        return
                else:
                    raise AssertionError("original_wait_or_close_boundary_not_reached")
            except Exception as error:
                self._error("controller", error)
            finally:
                self.mark_stage("controller_release")
                self.release.set()

        self.release_controller = threading.Thread(
            target=release_after_real_boundary,
            name="character-native-teardown-controller",
        )
        self.release_controller.start()

    def close(self):
        self.release.set()
        if self.release_controller is not None:
            self.release_controller.join(10)
            if self.release_controller.is_alive():
                self.invalid.append("release_controller_not_retired")
        tool = self.tool
        codes = tuple(self.codes)
        result = super().close()
        retired = False
        try:
            assert self.monitor.get_tool(tool) is None
            self.monitor.use_tool_id(tool, "character-teardown-retirement-proof")
            try:
                assert self.monitor.get_events(tool) == 0
                assert all(
                    self.monitor.get_local_events(tool, code) == 0 for code in codes
                )
                for event in self.registered:
                    assert self.monitor.register_callback(tool, event, None) is None
            finally:
                self.monitor.free_tool_id(tool)
            assert self.monitor.get_tool(tool) is None
            retired = True
        except Exception as error:
            result["invalid"].append(
                "physical_monitor_retirement:" + type(error).__name__
            )
            result["complete"] = False
        result["local_zero_callbacks_none_global_zero_same_slot_reclaimed"] = retired
        # This gate deliberately retains the exact known actors/handles needed
        # for custody proofs; unlike performance observers, no frame is held.
        result["frames_arguments_results_tasks_retained"] = True
        result["python_frames_retained"] = False
        result["actual_actor_ids"] = {
            "screen": id(self.screen),
            "controller": id(self.controller),
            "database": id(self.database),
            "worker": None if self.worker is None else id(self.worker),
            "worker_task": None if self.worker is None else id(self.worker._task),
            "worker_thread": None if self.thread is None else id(self.thread),
            "native_future": None if self.future is None else id(self.future),
            "connection": None if self.connection is None else id(self.connection),
            "lease": None if self.lease is None else id(self.lease),
            "operation": None if self.operation is None else id(self.operation),
        }
        result["selected_group"] = self.group
        result["prerequisite_stages"] = list(self.stage_rows)
        result["source_hashes"] = {
            name: state[-1] for name, state in self.modules.items()
        }
        result.update(
            actual_original_pair_held=self.entered.is_set(),
            actual_original_worker_wait=self.wait_seen.is_set(),
            original_close_rows=self.closer_rows,
            original_exit_rows=self.shutdown_rows,
            original_invoke_returned=self.invoke_returned.is_set(),
            actual_native_future_done=self.future is not None and self.future.done(),
            actual_native_connection_closed=self.connection is not None
            and physically_closed(self.connection),
            actual_lease_retired=self.lease is not None
            and self.lease not in self.storage._live_leases,
            actual_operation_retired=self.operation is not None
            and self.operation not in self.storage._operations,
            original_worker_task_done=self.worker is not None
            and self.worker._task.done(),
        )
        return result
