"""Exact getter-wrapper/body revision of the acquisition-unqualified observer.

Integration: construct with app.chachanotes_db after TldwCli construction and
start BEFORE run_test. Stop in unconditional finally before the ownership
receipt, and include receipt() alongside that receipt. Unmatched exceptional
spans remain strongly retained and invalidate attribution; they are never
invented as healthy returns. This module imports no
project code and replaces no callable, source guard, timer or cleanup behavior.
"""

import inspect
import sqlite3
import sys
import threading
import time
from types import CodeType, FunctionType


class OriginalNotesWorkerCalls:
    def __init__(self, database, storage):
        notes = sys.modules["tldw_chatbook.DB.ChaChaNotes_DB"]
        base = sys.modules["tldw_chatbook.DB.base_db"]
        participants = sys.modules["tldw_chatbook.Backup_Recovery.participants"]
        cls = inspect.getattr_static(notes, "CharactersRAGDB")
        connection_cls = inspect.getattr_static(base, "_QuiescentSQLiteConnection")
        assert type(database) is cls and not database.is_memory_db
        self.database, self.storage = database, storage
        self.main = threading.current_thread()
        self.modules = tuple(
            (module.__name__, module) for module in (notes, base, participants)
        )
        getter = inspect.getattr_static(cls, "_get_thread_connection")
        decorator = inspect.getattr_static(participants, "_core_getter")
        registration = inspect.getattr_static(participants, "_register_core_connection")
        assert type(getter) is type(decorator) is type(registration) is FunctionType
        assert inspect.getattr_static(notes, "_core_getter") is decorator
        assert (
            inspect.getattr_static(notes, "_register_core_connection") is registration
        )
        assert (
            decorator.__globals__ is registration.__globals__ is participants.__dict__
        )
        # This is one known original decorator, not a general unwrap protocol.
        # Its exact inner code and closure identify the original notes body.
        wrapper_codes = tuple(
            value
            for value in decorator.__code__.co_consts
            if type(value) is CodeType and value.co_name == "accessed"
        )
        assert len(wrapper_codes) == 1 and getter.__code__ is wrapper_codes[0]
        assert getter.__globals__ is participants.__dict__
        assert getter.__code__.co_freevars == ("function",)
        assert getter.__closure__ is not None and len(getter.__closure__) == 1
        getter_cell = getter.__closure__[0]
        getter_body = getter_cell.cell_contents
        assert type(getter_body) is FunctionType
        assert inspect.getattr_static(getter, "__wrapped__") is getter_body
        assert getter_body.__globals__ is notes.__dict__
        assert (
            getter_body.__code__.co_qualname == "CharactersRAGDB._get_thread_connection"
        )
        assert getter_body.__code__.co_argcount >= 1
        assert getter_body.__code__.co_varnames[0] == "self"
        self.getter_wrapper, self.getter_body = getter, getter_body
        self.getter_closure, self.getter_cell = getter.__closure__, getter_cell
        self.getter_decorator = decorator
        self.registration = registration
        self.bindings = [
            (
                cls,
                "_get_thread_connection",
                inspect.getattr_static(cls, "_get_thread_connection"),
            ),
            (cls, "close_connection", inspect.getattr_static(cls, "close_connection")),
            (connection_cls, "close", inspect.getattr_static(connection_cls, "close")),
            (participants, "_core_getter", decorator),
            (notes, "_core_getter", decorator),
            (participants, "_register_core_connection", registration),
            (notes, "_register_core_connection", registration),
        ]
        assert all(type(function) is FunctionType for _, _, function in self.bindings)
        self.body_bindings = tuple(
            (function, function.__code__, function.__globals__)
            for function in (
                *(function for _, _, function in self.bindings),
                getter_body,
            )
        )
        self.codes = {
            getter.__code__: "_get_thread_connection_wrapper",
            getter_body.__code__: "_get_thread_connection",
            registration.__code__: "_register_core_connection",
            inspect.getattr_static(
                cls, "close_connection"
            ).__code__: "close_connection",
            inspect.getattr_static(connection_cls, "close").__code__: "close",
        }
        assert len(self.codes) == 5
        self.monitor, self.tool = sys.monitoring, None
        self.tool_name = "tldw-hidden-sql-caller-" + str(id(self))
        self.mask = self.monitor.events.PY_START | self.monitor.events.PY_RETURN
        self.lock = threading.Lock()
        self.frames, self.rows = {}, {}
        self.references = [database, storage, self.main, cls, connection_cls]
        self.installed = []
        self.overflow = 0
        self.close_events = 0
        self.active = False
        self.restoration = None
        self.callbacks = (self._start, self._return)

    @staticmethod
    def _callers(frame):
        callers = []
        current = frame.f_back
        while current is not None and len(callers) < 24:
            code = current.f_code
            callers.append(
                {
                    "code_name": code.co_qualname,
                    "source": code.co_filename,
                    "definition_line": code.co_firstlineno,
                    "executing_line": current.f_lineno,
                }
            )
            current = current.f_back
        return callers

    def _connection_row(self, connection, actor, callers, route=None):
        # The original DB getter has returned, so participant registration is
        # complete. Observe only an exact link; this snapshot grants no close.
        participant = self.database._maintenance_participant
        with self.storage._lock:
            lease = participant.connections.get(connection)
            lease_id = id(lease) if lease is not None else None
            if route is not None:
                assert lease is not None and lease in self.storage._live_leases
                assert lease.resource_thread is actor
                assert lease.resource_participant is participant
        with self.lock:
            row = self.rows.get(id(connection))
            if row is None:
                if len(self.rows) >= 128:
                    self.overflow += 1
                    return None
                row = {
                    "connection_object_id": id(connection),
                    "lease_object_id": lease_id,
                    "participant_object_id": id(participant),
                    "exact_app_database_object_id": id(self.database),
                    "thread_object_id": id(actor),
                    "thread_ident": actor.ident,
                    "thread_name": actor.name,
                    "is_initial_thread": actor is self.main,
                    "first_observed_return_at": time.monotonic(),
                    "first_acquisition_callers": callers,
                    "first_observation_route": route,
                    "getter_returns": 0,
                    "getter_wrapper_returns": 0,
                    "registration_returns": 0,
                    "close_events": [],
                }
                self.rows[id(connection)] = row
                self.references.extend((connection, participant, lease, actor))
            return row

    def _start(self, code, _offset):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        kind = self.codes[code]
        owner = frame.f_locals.get(
            "repository"
            if kind in {"_get_thread_connection_wrapper", "_register_core_connection"}
            else "self"
        )
        if kind == "_get_thread_connection_wrapper":
            if frame.f_locals.get("function") is not self.getter_body:
                return
        if kind == "close":
            with self.lock:
                if id(owner) not in self.rows:
                    return
            connection = owner
        else:
            if owner is not self.database:
                return
            connection = (
                frame.f_locals.get("connection")
                if kind == "_register_core_connection"
                else getattr(self.database._local, "conn", None)
            )
        actor = threading.current_thread()
        callers = (
            None if kind.startswith("_get_thread_connection") else self._callers(frame)
        )
        with self.lock:
            if len(self.frames) >= 256:
                self.overflow += 1
                return
            # Keep the actual frame until its actual normal return. Local
            # PY_UNWIND is unsupported on 3.12; an exceptional exit must
            # remain unmatched, and its frame id must not be reused.
            self.frames[id(frame)] = (kind, connection, actor, callers, frame)

    def _return(self, code, _offset, value):
        frame = sys._getframe(1)
        assert frame.f_code is code
        with self.lock:
            state = self.frames.pop(id(frame), None)
        if state is None:
            return
        kind, previous, actor, callers, retained_frame = state
        assert retained_frame is frame
        if (
            kind.startswith("_get_thread_connection")
            or kind == "_register_core_connection"
        ):
            assert isinstance(value, sqlite3.Connection)
            if kind == "_register_core_connection":
                assert value is previous
            callers = (
                self._callers(frame)
                if kind == "_register_core_connection" or value is not previous
                else None
            )
            row = self._connection_row(value, actor, callers, route=kind)
            if row is not None:
                count_name = {
                    "_get_thread_connection": "getter_returns",
                    "_get_thread_connection_wrapper": "getter_wrapper_returns",
                    "_register_core_connection": "registration_returns",
                }[kind]
                with self.lock:
                    row[count_name] += 1
            return
        if previous is None:
            return
        row = self._connection_row(previous, actor, None)
        if row is None:
            return
        try:
            transaction = sqlite3.Connection.in_transaction.__get__(previous)
            closed, descriptor_error = False, None
        except sqlite3.ProgrammingError as error:
            # A generic ProgrammingError is not proof of physical close.
            closed = "closed" in str(error).lower()
            transaction, descriptor_error = None, type(error).__name__
        participant = self.database._maintenance_participant
        with self.storage._lock:
            still_registered = previous in participant.connections
        with self.lock:
            if self.close_events >= 512:
                self.overflow += 1
                return
            self.close_events += 1
            row["close_events"].append(
                {
                    "original_method": kind,
                    "returned_at": time.monotonic(),
                    "actor_object_id": id(actor),
                    "on_acquisition_actor": id(actor) == row["thread_object_id"],
                    "closed_descriptor_observed": closed,
                    "connection_in_transaction": transaction,
                    "descriptor_error_type": descriptor_error,
                    "still_registered": still_registered,
                    "callers": callers,
                }
            )
            self.references.append(actor)

    def start(self):
        self.tool = next(
            (number for number in (5, 4, 3) if self.monitor.get_tool(number) is None),
            None,
        )
        assert self.tool is not None
        self.monitor.use_tool_id(self.tool, self.tool_name)
        assert self.monitor.get_events(self.tool) == 0
        for event, callback in zip(
            (
                self.monitor.events.PY_START,
                self.monitor.events.PY_RETURN,
            ),
            self.callbacks,
            strict=True,
        ):
            assert self.monitor.register_callback(self.tool, event, callback) is None
        self.active = True
        try:
            for code in self.codes:
                assert self.monitor.get_local_events(self.tool, code) == 0
                self.monitor.set_local_events(self.tool, code, self.mask)
                self.installed.append(code)
        except BaseException:
            self.stop()
            raise

    def stop(self):
        assert self.monitor.get_tool(self.tool) == self.tool_name
        assert self.monitor.get_events(self.tool) == 0
        for code in self.installed:
            assert self.monitor.get_local_events(self.tool, code) == self.mask
            self.monitor.set_local_events(self.tool, code, 0)
        for event, callback in zip(
            (
                self.monitor.events.PY_START,
                self.monitor.events.PY_RETURN,
            ),
            self.callbacks,
            strict=True,
        ):
            assert self.monitor.register_callback(self.tool, event, None) is callback
        self.active = False
        self.monitor.free_tool_id(self.tool)
        assert self.monitor.get_tool(self.tool) is None
        self.restoration = "selected_local_callbacks_removed_tool_freed_global0"

    def receipt(self):
        with self.lock:
            rows = list(self.rows.values())
            live = len(self.frames)
        return {
            "diagnostic_only": True,
            "exact_app_database_object_id": id(self.database),
            "rows": rows,
            "overflow": self.overflow,
            "live_original_frames": live,
            "unmatched_exceptional_spans_invalidate_attribution": True,
            "healthy_normal_returns_only": True,
            "restoration": self.restoration,
            "selected_original_code_count": len(self.codes),
            "global_monitor_events": 0,
            "original_bindings_and_codes_unchanged": all(
                inspect.getattr_static(owner, name) is function
                for owner, name, function in self.bindings
            )
            and all(
                function.__code__ is code and function.__globals__ is namespace
                for function, code, namespace in self.body_bindings
            )
            and self.getter_wrapper.__closure__ is self.getter_closure
            and self.getter_wrapper.__closure__[0] is self.getter_cell
            and self.getter_cell.cell_contents is self.getter_body
            and inspect.getattr_static(self.getter_wrapper, "__wrapped__")
            is self.getter_body
            and any(
                value is self.getter_wrapper.__code__
                for value in self.getter_decorator.__code__.co_consts
            )
            and all(sys.modules.get(name) is module for name, module in self.modules),
            "strong_actual_owner_references_retained": True,
            "no_callable_guard_or_cleanup_replacement": True,
            "no_sql_or_unknown_close_added": True,
        }
