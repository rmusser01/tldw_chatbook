"""Opt-in finite observer of the original factory's declared DB retirement.

No App import, callable replacement, waits, SQL, close or global owner census.
The plugin is inherited by the private child through PYTEST_PLUGINS. It activates
only for that exact selected child and writes after the original teardown.
"""

import hashlib
import json
import os
import sqlite3
import sys
import threading
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

import pytest


TARGET = "Tests/UI/test_app_keep_alive_dead_screen.py::test_a_dead_content_screen_takes_textuals_loud_exit"
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
        test = sys.modules.get("Tests.UI.test_app_keep_alive_dead_screen")
        private = sys.modules.get("Tests.private_profile")
        assert all(
            type(x) is ModuleType
            for x in (factory, participants, storage, test, private)
        )
        wrapper = test.test_a_dead_content_screen_takes_textuals_loud_exit
        assert wrapper is self.item.obj and wrapper.__globals__ is vars(private)
        cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__ or ()))
        body = cells["function"].cell_contents
        assert wrapper.__wrapped__ is body
        self._pin(private, wrapper)
        self._pin(test, body)  # Exact actual pytest-rewritten body, including asserts.
        self.factory, self.participants, self.storage = factory, participants, storage
        self.retire_code = self._pin(factory, factory._retire_created_databases)
        self.retire_function = factory._retire_created_databases
        self.test_wrapper, self.test_body = wrapper, body
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
            or getattr(
                sys.modules.get("Tests.UI.test_app_keep_alive_dead_screen"),
                "test_a_dead_content_screen_takes_textuals_loud_exit",
                None,
            )
            is not self.test_wrapper
            or self.test_wrapper.__wrapped__ is not self.test_body
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


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global _witness
    output = os.environ.get(OUTPUT_ENV)
    if (
        not output
        or item.nodeid != TARGET
        or os.environ.get("TLDW_TEST_PRIVATE_PROFILE_NODE") != TARGET
    ):
        return
    assert _witness is None
    assert getattr(item.obj, "_private_profile_test", False)
    _witness = FactoryRetirementWitness(item.config, item)
    _witness.install()


def pytest_sessionfinish(session, exitstatus):
    if _witness is not None:
        receipt = _witness.stop()
        receipt["original_exitstatus"] = int(exitstatus)
        path = Path(os.environ[OUTPUT_ENV])
        assert not path.exists()
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
