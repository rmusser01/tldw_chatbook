"""Opt-in finite observer of the original factory's declared DB retirement.

No App import, callable replacement, waits, SQL, close or global owner census.
The plugin activates only for the exact original bootstrap-profile Library
delete/undo node and writes after its original teardown.
"""

import contextlib
import hashlib
import inspect
import json
import os
import sqlite3
import sys
import threading
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

import pytest


TARGET = "Tests/UI/test_library_reader_console_delete_refresh.py::test_console_delete_then_undo_rereads_the_open_library_reader"
OUTPUT_ENV = "TLDW_FACTORY_RETIREMENT_WITNESS"
_witness = None


def _shape(code):
    return (
        code.co_name,
        code.co_qualname,
        code.co_firstlineno,
        code.co_code,
        code.co_exceptiontable,
        code.co_stacksize,
        code.co_flags,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        tuple(_shape(x) if type(x) is CodeType else x for x in code.co_consts),
    )


def _nested(code, qualname):
    if code.co_qualname == qualname:
        return code
    for item in code.co_consts:
        if type(item) is CodeType:
            found = _nested(item, qualname)
            if found is not None:
                return found
    return None


class FactoryRetirementWitness:
    def __init__(self, config, item):
        self.config, self.item = config, item
        self.active = False
        self.tool = None
        self.rows, self.invalid = [], []
        self.bindings, self.hashes, self.codes = [], {}, {}
        self.retained = []  # Strong exact owner/participant/lease/Thread references.
        self.frame_last_line = {}
        self.monitor = sys.monitoring
        self.original_close_seen = False
        self.original_refusal_seen = False
        self.install_complete = False
        self.sources = {}
        self.module_states = {}

    def _pin(self, module, function):
        assert type(module) is ModuleType and type(function) is FunctionType
        assert function.__globals__ is vars(module)
        source = Path(module.__file__).resolve()
        assert Path(module.__spec__.origin).resolve() == source
        data = source.read_bytes()
        loader = module.__spec__.loader
        rewritten = getattr(loader, "_rewritten_names", {})
        if module.__name__ in rewritten:
            from _pytest.assertion import rewrite

            assert type(loader) is rewrite.AssertionRewritingHook
            assert module.__loader__ is loader and loader.config is self.config
            assert rewritten[module.__name__].resolve() == source
            _, compiled = rewrite._rewrite_test(source, self.config)
        else:
            compiled = compile(
                data, function.__code__.co_filename, "exec", dont_inherit=True
            )
        expected = _nested(compiled, function.__code__.co_qualname)
        assert expected is not None and _shape(expected) == _shape(function.__code__)
        closure = tuple(c.cell_contents for c in function.__closure__ or ())
        self.bindings.append(
            (
                module,
                function,
                function.__code__,
                function.__globals__,
                function.__defaults__,
                function.__kwdefaults__,
                closure,
            )
        )
        self.hashes[str(source)] = hashlib.sha256(data).hexdigest()
        self.sources[module.__name__] = {
            "origin": str(source),
            "sha256": self.hashes[str(source)],
        }
        self.module_states[module.__name__] = (
            module,
            module.__file__,
            module.__spec__,
            module.__spec__.origin,
            module.__loader__,
            module.__spec__.loader,
        )
        return function.__code__

    def install(self):
        factory = sys.modules.get("Tests.UI.app_factory")
        participants = sys.modules.get("tldw_chatbook.Backup_Recovery.participants")
        storage = sys.modules.get("tldw_chatbook.Backup_Recovery.storage_admission")
        test = sys.modules.get("Tests.UI.test_library_reader_console_delete_refresh")
        assert all(
            type(x) is ModuleType for x in (factory, participants, storage, test)
        )
        body = test.test_console_delete_then_undo_rereads_the_open_library_reader
        assert body is self.item.obj and body.__globals__ is vars(test)
        assert inspect.iscoroutinefunction(body) and body.__closure__ is None
        assert not hasattr(body, "__wrapped__")
        assert not getattr(body, "_private_profile_test", False)
        assert self.item.get_closest_marker("bootstrap_profile") is not None
        self.test_code = self._pin(
            test, body
        )  # Actual pytest rewrite, including asserts.
        self.build_app_function = test._build_app
        self.build_app_code = self._pin(test, self.build_app_function)
        self.test_module = test
        # pytest-asyncio legitimately replaces item.obj at Coroutine.runtest.
        # Pin that exact installed wrapper body/globals/func closure, not just
        # its mutable __wrapped__ marker, while retaining the original test.
        plugin = sys.modules.get("pytest_asyncio.plugin")
        assert type(plugin) is ModuleType
        self.execution_module = plugin
        self.execution_wrap = plugin.wrap_in_sync
        self.execution_runtest = plugin.Coroutine.runtest
        self._pin(plugin, self.execution_wrap)
        self._pin(plugin, self.execution_runtest)
        self.execution_wrapper_code = _nested(
            self.execution_wrap.__code__, "wrap_in_sync.<locals>.inner"
        )
        assert self.execution_wrapper_code is not None
        self.factory, self.participants, self.storage = factory, participants, storage
        self.retire_code = self._pin(factory, factory._retire_created_databases)
        self.retire_function = factory._retire_created_databases
        self.test_wrapper = self.test_body = body
        self.close_function = participants._close_settled_core_cache
        self.close_code = self._pin(participants, self.close_function)
        self.codes = {
            self.retire_code: "factory_retire",
            self.close_code: "settled_close",
        }
        events = self.monitor.events
        for tool in range(5, -1, -1):
            try:
                self.monitor.use_tool_id(tool, "tldw-factory-retirement-refusal")
            except ValueError:
                continue
            self.tool = tool
            break
        assert self.tool is not None
        self.monitor.register_callback(self.tool, events.PY_START, self._start)
        self.monitor.register_callback(self.tool, events.PY_RETURN, self._return)
        self.monitor.register_callback(self.tool, events.LINE, self._line)
        self.active = True
        for code in self.codes:
            self.monitor.set_local_events(
                self.tool, code, events.PY_START | events.PY_RETURN | events.LINE
            )
        assert self.monitor.get_events(self.tool) == 0
        self.install_complete = True

    def _execution_current(self):
        function = self.item.obj
        if function is self.test_body:
            return True  # Original setup may fail before Coroutine.runtest.
        if (
            type(function) is not FunctionType
            or function.__code__ is not self.execution_wrapper_code
            or function.__globals__ is not vars(self.execution_module)
            or function.__defaults__ is not None
            or function.__kwdefaults__ is not None
            or function.__code__.co_freevars != ("func",)
        ):
            return False
        cells = tuple(function.__closure__ or ())
        return (
            len(cells) == 1
            and cells[0].cell_contents is self.test_body
            and function.__wrapped__ is self.test_body
            and function._raw_test_func is self.test_body
        )

    def _frame(self, code):
        frame = sys._getframe(1)
        for _ in range(8):
            if frame is None:
                break
            if frame.f_code is code:
                return frame
            frame = frame.f_back
        self.invalid.append("selected_frame_unavailable")
        return None

    def _declared(self, frame):
        parent = frame
        for _ in range(5):
            if parent is None:
                return None
            if parent.f_code is self.retire_code:
                directory = parent.f_locals.get("directory")
                owner = frame.f_locals.get("repository", parent.f_locals.get("owner"))
                declarations = self.factory._created_databases.get(directory, ())
                assert len(declarations) <= 6
                matched = [x for x in declarations if x[0] is owner]
                if len(matched) != 1:
                    return None
                owner, expected_type, path = matched[0]
                assert type(owner) is expected_type and owner.db_path == path
                participant = getattr(owner, "_maintenance_participant", None)
                assert participant is not None and participant.repository() is owner
                assert participant.path == path
                self.retained.extend((owner, participant))
                return owner, participant, expected_type, path
            parent = parent.f_back
        return None

    def _thread(self, thread):
        if thread is None:
            return None
        self.retained.append(thread)
        return {
            "actor": id(thread),
            "ident": thread.ident,
            "current": thread is threading.current_thread(),
            "main": thread is threading.main_thread(),
            "alive": thread.is_alive(),
        }

    def _snapshot(self, frame, boundary, result=None):
        declared = self._declared(frame)
        if declared is None:
            return False
        if len(self.rows) >= 32:
            self.invalid.append("detail_cap_reached")
            return False
        owner, participant, expected_type, path = declared
        storage = self.storage
        with storage._lock:
            connections = tuple(participant.connections.items())
            operations = tuple(
                x for x in storage._operations if x.participant is participant
            )
            pending = tuple(
                x
                for x in storage._pending_acquisitions
                if getattr(x.operation, "participant", None) is participant
            )
            leases = tuple(x for x in storage._live_leases if x.resource_path == path)
            retiring = tuple(participant.retiring_threads)
            assert (
                max(
                    len(connections),
                    len(operations),
                    len(pending),
                    len(leases),
                    len(retiring),
                )
                <= 24
            )
            native = []
            for connection, lease in connections:
                self.retained.extend((connection, lease))
                try:
                    transaction = sqlite3.Connection.in_transaction.__get__(connection)
                    state = "open"
                except sqlite3.ProgrammingError:
                    transaction, state = None, "physically_closed"
                native.append(
                    {
                        "connection_actor": id(connection),
                        "lease_actor": id(lease),
                        "physical_state": state,
                        "in_transaction": transaction,
                        "lease_thread": self._thread(lease.resource_thread),
                    }
                )
            row = {
                "boundary": boundary,
                "result": result,
                "original_line": frame.f_lineno,
                "last_original_line": self.frame_last_line.get(id(frame)),
                "owner_actor": id(owner),
                "participant_actor": id(participant),
                "owner_type": expected_type.__module__
                + "."
                + expected_type.__qualname__,
                "owner_id": participant.owner_id,
                "path_basename": path.name,
                "path_matches_exact_factory_declaration": True,
                "participant_closed": participant.closed,
                "pause_present": storage._pause is not None,
                "current_thread": self._thread(threading.current_thread()),
                "connections": native,
                "operations": [
                    {
                        "actor": id(x),
                        "thread": self._thread(x.thread),
                        "task_actor": None if x.task is None else id(x.task),
                        "lease_actor": None if x.lease is None else id(x.lease),
                    }
                    for x in operations
                ],
                "pending_acquisitions": len(pending),
                "path_leases": [
                    {"actor": id(x), "thread": self._thread(x.resource_thread)}
                    for x in leases
                ],
                "retiring_threads": [self._thread(x) for x in retiring],
            }
            self.retained.extend(operations + pending + leases)
        self.rows.append(row)
        return True

    def _start(self, code, offset):
        if not self.active:
            return
        try:
            frame = self._frame(code)
            if frame is not None and code is self.close_code:
                if self._snapshot(frame, "original_settled_close_start"):
                    self.original_close_seen = True
        except Exception as exc:
            self.invalid.append("start:" + type(exc).__name__)

    def _line(self, code, line):
        if not self.active:
            return
        frame = self._frame(code)
        if frame is not None:
            self.frame_last_line[id(frame)] = line

    def _return(self, code, offset, value):
        if not self.active:
            return
        try:
            frame = self._frame(code)
            if frame is not None and code is self.close_code:
                assert type(value) is bool  # noqa: E721 - require the exact declared source owner
                if self._snapshot(frame, "original_settled_close_return", value):
                    self.original_refusal_seen |= not value
            if frame is not None:
                self.frame_last_line.pop(id(frame), None)
        except Exception as exc:
            self.invalid.append("return:" + type(exc).__name__)

    def stop(self):
        events = self.monitor.events
        hooks_retired = False
        try:
            if self.tool is not None:
                for code in self.codes:
                    self.monitor.set_local_events(self.tool, code, 0)
                for event in (events.PY_START, events.PY_RETURN, events.LINE):
                    self.monitor.register_callback(self.tool, event, None)
                assert self.monitor.get_events(self.tool) == 0
            hooks_retired = True
        finally:
            # Free the exact owned tool even when retirement validation fails;
            # source validation runs only after the monitoring hooks are gone.
            try:
                if self.tool is not None:
                    self.monitor.free_tool_id(self.tool)
            finally:
                self.active = False
        for name, state in self.module_states.items():
            module, file, spec, origin, loader, spec_loader = state
            if (
                sys.modules.get(name) is not module
                or module.__file__ != file
                or module.__spec__ is not spec
                or spec.origin != origin
                or module.__loader__ is not loader
                or spec.loader is not spec_loader
                or Path(file).resolve() != Path(origin).resolve()
            ):
                self.invalid.append("installed_module_origin_or_spec_changed:" + name)
        if self.install_complete and (
            self.factory._retire_created_databases is not self.retire_function
            or self.participants._close_settled_core_cache is not self.close_function
            or sys.modules.get("Tests.UI.test_library_reader_console_delete_refresh")
            is not self.test_module
            or self.test_module.test_console_delete_then_undo_rereads_the_open_library_reader
            is not self.test_body
            or not self._execution_current()
            or self.execution_module.wrap_in_sync is not self.execution_wrap
            or self.execution_module.Coroutine.runtest is not self.execution_runtest
            or self.test_module._build_app is not self.build_app_function
        ):
            self.invalid.append("original_callable_slot_changed")
        for (
            module,
            function,
            code,
            globals_,
            defaults,
            kwdefaults,
            closure,
        ) in self.bindings:
            if (
                function.__code__ is not code
                or function.__globals__ is not globals_
                or function.__defaults__ is not defaults
                or function.__kwdefaults__ is not kwdefaults
                or len(function.__closure__ or ()) != len(closure)
                or any(
                    c.cell_contents is not x
                    for c, x in zip(function.__closure__ or (), closure)
                )
            ):
                self.invalid.append("original_callable_changed:" + code.co_qualname)
        unchanged = all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            for path, digest in self.hashes.items()
        )
        if not unchanged:
            self.invalid.append("source_changed")
        return {
            "diagnostic_only": True,
            "exact_original_node": TARGET,
            "install_complete": self.install_complete,
            "actual_bootstrap_profile_body_qualified": self.install_complete,
            "private_profile_wrapper_used": False,
            "source_unchanged": unchanged,
            "installed_sources": self.sources,
            "invalid_evidence": self.invalid,
            "rows": self.rows,
            "original_close_seen": self.original_close_seen,
            "original_refusal_seen": self.original_refusal_seen,
            "global_events": 0,
            "hooks_retired_before_inactive": hooks_retired,
            "exact_owner_and_native_objects_retained": True,
            "limits": "Only six original factory-declared owners and exact original close frames. Native in_transaction descriptor reads add bounded diagnostic overhead; no SQL, close, added waits, deadlines, guard replacement or global thread/owner enumeration. Exceptional returns remain unobserved.",
        }


def _exact_nested(code, actual):
    if (
        code.co_qualname == actual.co_qualname
        and code.co_firstlineno == actual.co_firstlineno
    ):
        return code
    for item in code.co_consts:
        if type(item) is CodeType:
            found = _exact_nested(item, actual)
            if found is not None:
                return found
    return None


class FactoryOriginWitness(FactoryRetirementWitness):
    def __init__(self, config, item):
        super().__init__(config, item)
        self.declared, self.origins, self.phase_rows = {}, [], []
        self.origin_code_states, self.compiled_sources = {}, {}
        self.original_app = None
        self.extra_aliases = []
        self.declaration_seen = False
        self.origin_signatures, self.duplicate_origins = set(), 0

    def install(self):
        super().install()
        factory, participants = self.factory, self.participants
        self.record_code = self._pin(factory, factory._record_created_databases)
        self.register_code = self._pin(
            participants, participants._register_core_connection
        )
        self.extra_aliases.extend(
            (
                (
                    factory,
                    "_record_created_databases",
                    factory._record_created_databases,
                ),
                (
                    participants,
                    "_register_core_connection",
                    participants._register_core_connection,
                ),
            )
        )
        wrapper = participants._core_operation
        body = wrapper.__wrapped__
        assert type(wrapper) is FunctionType and type(body) is FunctionType
        assert wrapper.__globals__ is vars(contextlib)
        cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__ or ()))
        assert cells["func"].cell_contents is body
        self._pin(contextlib, wrapper)
        self.operation_code = self._pin(participants, body)
        self.operation_wrapper, self.operation_body = wrapper, body
        self.extra_aliases.append((participants, "_core_operation", wrapper))
        lifecycle = sys.modules.get("tldw_chatbook.app_lifecycle")
        assert type(lifecycle) is ModuleType
        self.phase_codes = {}
        for name in (
            "_shutdown",
            "_shutdown_app_owned_lifecycles",
            "_cancel_and_settle_workers",
            "on_unmount",
        ):
            function = getattr(lifecycle.LifecycleMixin, name)
            self.phase_codes[self._pin(lifecycle, function)] = name
            self.extra_aliases.append((lifecycle.LifecycleMixin, name, function))
        self.codes.update(
            {
                self.record_code: "ctor_declaration",
                self.register_code: "native_registration",
                self.operation_code: "issued_operation",
            }
        )
        self.codes.update(self.phase_codes)
        events = self.monitor.events
        self.monitor.register_callback(self.tool, events.PY_YIELD, self._yield)
        for code in self.codes:
            mask = events.PY_START | events.PY_RETURN | events.LINE
            if code is self.operation_code:
                mask |= events.PY_YIELD
            self.monitor.set_local_events(self.tool, code, mask)
        assert self.monitor.get_events(self.tool) == 0

    def _known_owner(self, owner):
        declaration = self.declared.get(id(owner))
        if declaration is None:
            return None
        captured, expected_type, path, participant = declaration
        assert (
            owner is captured and type(owner) is expected_type and owner.db_path == path
        )
        current = getattr(owner, "_maintenance_participant", None)
        if participant is None:
            if current is None:
                return None  # Lazy original owner has not acquired stock metadata.
            participant = current
            assert participant.repository() is owner and participant.path == path
            declaration = owner, expected_type, path, participant
            self.declared[id(owner)] = declaration
            self.retained.append(participant)
        assert current is participant
        assert participant.repository() is owner and participant.path == path
        return declaration

    def _known(self, frame):
        return self._known_owner(frame.f_locals.get("repository"))

    def _caller_source(self, frame):
        code, globals_ = frame.f_code, frame.f_globals
        state = self.origin_code_states.get(code)
        if state is None:
            assert len(self.origin_code_states) < 64
            name = globals_.get("__name__")
            module = sys.modules.get(name)
            assert type(module) is ModuleType and vars(module) is globals_
            source = Path(module.__file__).resolve()
            assert Path(module.__spec__.origin).resolve() == source
            assert Path(code.co_filename).resolve() == source
            cached = self.compiled_sources.get(name)
            if cached is None:
                assert len(self.compiled_sources) < 24
                data = source.read_bytes()
                loader = module.__spec__.loader
                rewritten = getattr(loader, "_rewritten_names", {})
                if name in rewritten:
                    from _pytest.assertion import rewrite

                    assert type(loader) is rewrite.AssertionRewritingHook
                    assert module.__loader__ is loader and loader.config is self.config
                    assert rewritten[name].resolve() == source
                    _, compiled = rewrite._rewrite_test(source, self.config)
                else:
                    compiled = compile(
                        data, code.co_filename, "exec", dont_inherit=True
                    )
                digest = hashlib.sha256(data).hexdigest()
                self.hashes[str(source)] = digest
                self.module_states[name] = (
                    module,
                    module.__file__,
                    module.__spec__,
                    module.__spec__.origin,
                    module.__loader__,
                    module.__spec__.loader,
                )
                self.sources[name] = {"origin": str(source), "sha256": digest}
                cached = compiled, digest
                self.compiled_sources[name] = cached
            expected = _exact_nested(cached[0], code)
            assert expected is not None and _shape(expected) == _shape(code)
            state = (globals_, name, cached[1])
            self.origin_code_states[code] = state
            self.retained.extend((code, globals_))
        assert state[0] is globals_
        return {
            "module": state[1],
            "qualname": code.co_qualname,
            "first_line": code.co_firstlineno,
            "line": frame.f_lineno,
            "source_sha256": state[2],
            "original_body_qualified": True,
        }

    def _origin(self, frame, boundary, *, connection=None, operation=None):
        declaration = self._known(frame)
        if declaration is None:
            return
        owner, expected_type, path, participant = declaration
        if (
            participant.owner_id != "db.workspaces"
            or threading.current_thread() is threading.main_thread()
        ):
            return  # Detailed origins belong only to the declared Workspace worker.
        parents, parent = [], frame.f_back
        for _ in range(12):
            if parent is None:
                break
            parents.append(parent)
            parent = parent.f_back
        signature = (
            boundary,
            id(owner),
            id(threading.current_thread()),
            id(connection) if connection is not None else None,
            tuple(parent.f_code for parent in parents),
        )
        if signature in self.origin_signatures:
            self.duplicate_origins += 1
            return
        if len(self.origins) >= 96:
            self.invalid.append("origin_detail_cap_reached")
            return
        callers = [self._caller_source(parent) for parent in parents]
        with self.storage._lock:
            lease = (
                participant.connections.get(connection)
                if connection is not None
                else None
            )
            if connection is not None:
                assert (
                    lease is not None
                    and lease.resource_thread is threading.current_thread()
                )
            if operation is not None:
                assert operation in self.storage._operations
                assert operation.participant is participant
                assert operation.thread is threading.current_thread()
            self.retained.extend(
                x
                for x in (owner, participant, connection, lease, operation)
                if x is not None
            )
            row = {
                "boundary": boundary,
                "owner_actor": id(owner),
                "participant_actor": id(participant),
                "owner_id": participant.owner_id,
                "path_basename": path.name,
                "connection_actor": None if connection is None else id(connection),
                "lease_actor": None if lease is None else id(lease),
                "operation_actor": None if operation is None else id(operation),
                "thread": self._thread(threading.current_thread()),
                "callers": callers,
            }
        self.origins.append(row)
        self.origin_signatures.add(signature)

    def _phase(self, frame, boundary):
        if (
            self.original_app is None
            or frame.f_locals.get("self") is not self.original_app
        ):
            return
        assert len(self.phase_rows) < 24
        with self.storage._lock:
            counts = []
            for owner, _, path, _ in tuple(self.declared.values()):
                declaration = self._known_owner(owner)
                row = {
                    "owner_actor": id(owner),
                    "path_basename": path.name,
                    "participant_present": declaration is not None,
                }
                if declaration is not None:
                    participant = declaration[3]
                    row.update(
                        owner_id=participant.owner_id,
                        connections=len(participant.connections),
                        operations=sum(
                            x.participant is participant
                            for x in self.storage._operations
                        ),
                        retiring_threads=[
                            self._thread(x) for x in participant.retiring_threads
                        ],
                    )
                counts.append(row)
        self.phase_rows.append(
            {
                "boundary": boundary,
                "phase": self.phase_codes[frame.f_code],
                "original_line": frame.f_lineno,
                "app_actor": id(self.original_app),
                "owners": counts,
            }
        )

    def _start(self, code, offset):
        super()._start(code, offset)
        if not self.active or code not in getattr(self, "phase_codes", {}):
            return
        try:
            frame = self._frame(code)
            if frame is not None:
                self._phase(frame, "original_phase_start")
        except Exception as exc:
            self.invalid.append("phase_start:" + type(exc).__name__)

    def _yield(self, code, offset, value):
        if not self.active or code is not self.operation_code:
            return
        try:
            frame = self._frame(code)
            declaration = self._known(frame) if frame is not None else None
            if (
                declaration is not None
                and declaration[3].owner_id == "db.workspaces"
                and threading.current_thread() is not threading.main_thread()
            ):
                operation = getattr(self.storage._operation_local, "operation", None)
                assert operation is not None
                self._origin(
                    frame, "original_issued_operation_yield", operation=operation
                )
        except Exception as exc:
            self.invalid.append("operation_yield:" + type(exc).__name__)

    def _return(self, code, offset, value):
        if self.active:
            try:
                frame = self._frame(code)
                if frame is not None and code is getattr(self, "record_code", None):
                    directory, app = frame.f_locals["directory"], frame.f_locals["app"]
                    # Ctor declaration must be on this original test's own
                    # synchronous _build_app chain, not fixture or foreign App work.
                    caller, ancestry = frame.f_back, set()
                    for _ in range(12):
                        if caller is None:
                            break
                        if caller.f_code in (self.test_code, self.build_app_code):
                            assert caller.f_globals is vars(self.test_module)
                            ancestry.add(caller.f_code)
                        caller = caller.f_back
                    assert ancestry == {self.test_code, self.build_app_code}
                    assert self.original_app is None and not self.declared
                    for owner, expected_type, path in self.factory._created_databases[
                        directory
                    ]:
                        assert len(self.declared) < 6
                        participant = getattr(owner, "_maintenance_participant", None)
                        assert type(owner) is expected_type and owner.db_path == path
                        if participant is not None:
                            assert (
                                participant.repository() is owner
                                and participant.path == path
                            )
                        self.declared[id(owner)] = (
                            owner,
                            expected_type,
                            path,
                            participant,
                        )
                        self.retained.extend((owner, participant))
                    self.original_app, self.declaration_seen = app, True
                    self.retained.append(app)
                elif frame is not None and code is getattr(self, "register_code", None):
                    self._origin(
                        frame, "original_native_registration_return", connection=value
                    )
                elif frame is not None and code in getattr(self, "phase_codes", {}):
                    self._phase(frame, "original_phase_return")
            except Exception as exc:
                self.invalid.append("origin_return:" + type(exc).__name__)
        super()._return(code, offset, value)

    def stop(self):
        try:
            if self.tool is not None:
                self.monitor.register_callback(
                    self.tool, self.monitor.events.PY_YIELD, None
                )
        finally:
            receipt = super().stop()
        for parent, name, function in self.extra_aliases:
            if getattr(parent, name, None) is not function:
                self.invalid.append("original_origin_alias_changed:" + name)
        if (
            hasattr(self, "operation_wrapper")
            and self.operation_wrapper.__wrapped__ is not self.operation_body
        ):
            self.invalid.append("original_operation_wrapped_body_changed")
        receipt.update(
            declaration_seen=self.declaration_seen,
            exact_declared_owner_count=len(self.declared),
            acquisition_and_operation_origins=self.origins,
            original_shutdown_phases=self.phase_rows,
            caller_original_code_count=len(self.origin_code_states),
            repeated_workspace_origin_chains_omitted=self.duplicate_origins,
            origin_limits="Only the six exact ctor declarations; lazy participant metadata is captured after it appears naturally. Detailed origins cover normal original Workspace worker registration returns and deduplicated issued-operation caller chains only. No broader owner census, exceptional-return or pre-declaration acquisition inference; original refusal rows remain authoritative for actual live state.",
        )
        return receipt


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global _witness
    if not os.environ.get(OUTPUT_ENV) or item.nodeid != TARGET:
        return
    assert _witness is None
    assert item.get_closest_marker("bootstrap_profile") is not None
    assert not getattr(item.obj, "_private_profile_test", False)
    _witness = FactoryOriginWitness(item.config, item)
    _witness.install()


def pytest_sessionfinish(session, exitstatus):
    if _witness is not None:
        receipt = _witness.stop()
        receipt["original_exitstatus"] = int(exitstatus)
        path = Path(os.environ[OUTPUT_ENV])
        assert not path.exists()
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
