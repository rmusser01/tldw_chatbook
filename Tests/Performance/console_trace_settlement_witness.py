"""Diagnostic-only, code-local witness for the original three-turn native probe.

No production callable, guard, provider fixture, deadline or settling wait is
replaced. Only the existing test-owned Observation lifecycle is extended in the
actual opted-in private child. No application module is imported by this plugin.
"""

import ast
import hashlib
import sys
import threading
import time
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

import pytest


PROBE_NAME = "Tests.Performance.test_console_native_pause_probe"
STORE_NAME = "tldw_chatbook.Chat.console_chat_store"
COORDINATOR_NAME = "tldw_chatbook.Chat.console_trace_settlement"
TARGET_NAME = "test_native_console_pause_probe"


def _shape(code):
    """Compare immutable compiled-code structure; never arbitrary unwrap metadata."""
    assert type(code) is CodeType
    return (
        code.co_name,
        code.co_qualname,
        code.co_firstlineno,
        code.co_code,
        code.co_flags,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        tuple(
            _shape(item) if type(item) is CodeType else item for item in code.co_consts
        ),
    )


def _compiled_code(
    module, qualname, filename, *, assertion_config=None, assertion_loader=None
):
    if assertion_config is None:
        assert assertion_loader is None
        code = compile(
            Path(module.__file__).read_text(encoding="utf-8"),
            filename,
            "exec",
            dont_inherit=True,
        )
    else:
        from _pytest.assertion import rewrite

        spec = module.__spec__
        assert type(assertion_loader) is rewrite.AssertionRewritingHook
        assert module.__loader__ is spec.loader is assertion_loader
        assert assertion_loader.config is assertion_config
        source = Path(module.__file__).resolve()
        assert Path(spec.origin).resolve() == source == Path(filename).resolve()
        assert assertion_loader._rewritten_names[module.__name__].resolve() == source
        # Recompile the exact installed source through the loader's own pytest
        # rewrite pipeline. Full code structure is compared, including asserts.
        _, code = rewrite._rewrite_test(Path(filename), assertion_config)
    for name in qualname.split("."):
        code = next(
            item
            for item in code.co_consts
            if type(item) is CodeType and item.co_name == name
        )
    return code


class TraceSettlementWitness:
    tool_name = "tldw-finite-trace-settlement-witness"

    def __init__(self, probe, test_body, observation, *, assertion_config=None):
        self.assertion_config = assertion_config
        self.assertion_loader = (
            probe.__spec__.loader if assertion_config is not None else None
        )
        self.probe = probe
        self.test_body = test_body
        self.test_code = test_body.__code__
        self.observation = observation
        self.monitor = sys.monitoring
        self.tool = None
        self.active = False
        self.codes = {}
        self.handler_lines = {}
        self.masks = {}
        self.bindings = []
        self.module_hashes = {}
        self.store_type = None
        self.coordinator_type = None
        self.stores = {}
        self.coordinators = {}
        self.threads = {}
        self.call_tokens = {}
        self.live = {}
        self.rows = []
        self.handlers = []
        self.snapshots = []
        self.invalid = []
        self.callbacks = {}
        self.installed_modules = {}
        self.installed_sources_before = None
        self.terminal_line = self._terminal_line()
        self.discovery = dict(
            initial_store_loaded=type(sys.modules.get(STORE_NAME)) is ModuleType,
            initial_coordinator_loaded=type(sys.modules.get(COORDINATOR_NAME))
            is ModuleType,
            qualified_bindings=[],
            original_terminal_seen=False,
        )

    def _loaded_source_records(self):
        """Record actual already loaded child origins; never import a source."""
        names = {
            "Tests.private_profile",
            "Tests.real_profile_guard",
            "Tests.network_guard",
            "Tests.windows_private_fixture_runner",
            __name__,
        }
        names.update(
            name
            for name in tuple(sys.modules)
            if name == "tldw_profile_core" or name.startswith("tldw_profile_core.")
        )
        records = {}
        for name in sorted(names):
            module = sys.modules.get(name)
            if type(module) is not ModuleType:
                records[name] = {"loaded": False}
                continue
            origin = module.__dict__.get("__file__")
            if not isinstance(origin, str):
                records[name] = {"loaded": True, "source_origin_missing": True}
                continue
            digest = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
            self.installed_modules.setdefault(name, (module, origin, digest))
            records[name] = {"loaded": True, "origin": origin, "sha256": digest}
        return {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "isolated": sys.flags.isolated,
            "utf8_mode": sys.flags.utf8_mode,
            "modules": records,
        }

    def _terminal_line(self):
        tree = ast.parse(Path(self.probe.__file__).read_text(encoding="utf-8"))
        test = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == TARGET_NAME
        )
        expected = ast.dump(
            ast.parse('result["trace_states"] == ["complete"] * 3', mode="eval").body
        )
        matches = [
            node
            for node in ast.walk(test)
            if isinstance(node, ast.Assert) and ast.dump(node.test) == expected
        ]
        assert len(matches) == 1, "original unchanged terminal assertion was not found"
        return matches[0].lineno

    def _retain_function(self, module, klass, name, label):
        descriptor = klass.__dict__[name]
        function = (
            descriptor.__func__ if type(descriptor) is staticmethod else descriptor
        )
        assert type(function) is FunctionType
        assert function.__globals__ is module.__dict__
        assert _shape(function.__code__) == _shape(
            _compiled_code(
                module, klass.__name__ + "." + name, function.__code__.co_filename
            )
        ), "installed method differs from its defining source"
        self.bindings.append(
            (module, klass, name, descriptor, function, function.__code__)
        )
        self.codes[function.__code__] = label
        if label in {"store_run", "claimed"}:
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            owner = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == klass.__name__
            )
            method = next(
                node
                for node in owner.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            self.handler_lines[function.__code__] = {
                line
                for handler in ast.walk(method)
                if isinstance(handler, ast.ExceptHandler)
                for statement in handler.body
                for line in range(statement.lineno, statement.end_lineno + 1)
            }
        self.module_hashes[module.__name__] = hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
        self._enable(function.__code__, line=label in {"store_run", "claimed"})

    def _enable(self, code, *, line=False, only_line=False):
        events = self.monitor.events
        mask = events.LINE if only_line else events.PY_START | events.PY_RETURN
        if line:
            mask |= events.LINE
        self.monitor.set_local_events(self.tool, code, mask)
        self.masks[code] = mask

    def _bind_loaded(self):
        if self.store_type is None:
            module = sys.modules.get(STORE_NAME)
            if type(module) is ModuleType:
                self.store_type = module.__dict__["ConsoleChatStore"]
                for name, label in (
                    ("_run_provider_trace_settlement", "store_run"),
                    ("_drain_provider_trace_settlement_work", "store_drain"),
                ):
                    self._retain_function(module, self.store_type, name, label)
                self.discovery["qualified_bindings"].append(
                    dict(
                        source="store",
                        phase=self.observation.phase,
                        time=time.perf_counter(),
                    )
                )
        if self.coordinator_type is None:
            module = sys.modules.get(COORDINATOR_NAME)
            if type(module) is ModuleType:
                self.coordinator_type = module.__dict__[
                    "ConsoleTraceSettlementCoordinator"
                ]
                for name, label in (
                    ("_settle_claimed", "claimed"),
                    ("_settle_prepared", "prepared"),
                    ("_enqueue_locked", "enqueue"),
                ):
                    self._retain_function(module, self.coordinator_type, name, label)
                self.discovery["qualified_bindings"].append(
                    dict(
                        source="coordinator",
                        phase=self.observation.phase,
                        time=time.perf_counter(),
                    )
                )

    def _actor(self):
        actor = threading.current_thread()
        self.threads[id(actor)] = actor
        return actor

    def _token(self, call_id):
        call_type = type(call_id)
        if call_type is not str or not call_id:
            return None
        return self.call_tokens.setdefault(
            call_id, "call_" + str(len(self.call_tokens) + 1)
        )

    def _owners(self, frame):
        self._bind_loaded()  # Only already loaded defining modules; no lazy import.
        values = frame.f_locals
        owner = values.get("self")
        if self.store_type is not None and type(owner) is self.store_type:
            self.stores[id(owner)] = owner
        if self.coordinator_type is not None and type(owner) is self.coordinator_type:
            self.coordinators[id(owner)] = owner
        controller = values.get("controller")
        if controller is not None:
            store = getattr(controller, "__dict__", {}).get("store")
            if self.store_type is not None and type(store) is self.store_type:
                self.stores[id(store)] = store
        prepared = values.get("prepared")
        handoff = values.get("handoff")
        module = sys.modules.get(COORDINATOR_NAME)
        if type(module) is ModuleType and type(handoff) is module.__dict__.get(
            "ConsoleTraceSettlementHandoff"
        ):
            prepared = object.__getattribute__(handoff, "_prepared")
            coordinator = object.__getattribute__(handoff, "_coordinator")
            if type(coordinator) is self.coordinator_type:
                self.coordinators[id(coordinator)] = coordinator
        if type(module) is ModuleType and type(prepared) is module.__dict__.get(
            "_PreparedSettlement"
        ):
            return self._token(object.__getattribute__(prepared, "call_id"))
        return None

    def _parent(self, frame, actor):
        current = frame.f_back
        for _ in range(12):
            if current is None:
                break
            row = self.live.get((actor, id(current)))
            if row is not None:
                return row["id"]
            current = current.f_back
        return None

    def _start(self, code, offset):
        if not self.active or code not in self.codes:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        actor = self._actor()
        token = self._owners(frame)
        key = (actor, id(frame))
        if key in self.live:
            self.invalid.append("live_frame_id_collision")
            return
        row = {
            "id": len(self.rows) + 1,
            "parent_id": self._parent(frame, actor),
            "label": self.codes[code],
            "call": token,
            "actor": id(actor),
            "phase": self.observation.phase,
            "entered_at": time.perf_counter(),
            "returned_at": None,
            "result": None,
            "handled_exception_types": [],
        }
        self.rows.append(row)
        self.live[key] = row

    def _returned(self, code, offset, value):
        if not self.active or code not in self.codes:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        actor = self._actor()
        key = (actor, id(frame))
        row = self.live.pop(key, None)
        if row is None:
            self.invalid.append("return_without_original_start:" + self.codes[code])
            return
        row["returned_at"] = time.perf_counter()
        row["result"] = (
            value
            if value is True or value is False or value is None
            else type(value).__name__
        )
        module = sys.modules.get(COORDINATOR_NAME)
        if type(module) is ModuleType and type(value) is module.__dict__.get(
            "TraceCallRecord"
        ):
            state = object.__getattribute__(value, "state")
            if type(state) is module.__dict__.get("TraceCallState"):
                row["terminal_state"] = object.__getattribute__(state, "_value_")
        # A normal original synchronous parent return proves its selected child
        # has unwound. It is explicitly not a healthy child return observation.
        retired = {row["id"]}
        while True:
            children = [
                (other_key, child)
                for other_key, child in self.live.items()
                if other_key[0] is actor and child["parent_id"] in retired
            ]
            if not children:
                break
            for other_key, child in children:
                child["retired_by_original_parent_return"] = row["id"]
                self.live.pop(other_key)
                retired.add(child["id"])

    def _line(self, code, line):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        if code is self.test_code:
            self._owners(frame)
            if line == self.terminal_line:
                result = frame.f_locals["result"]
                self.snapshots.append(
                    self._snapshot("original_terminal_assertion", result)
                )
            return
        if self.codes.get(code) not in {"claimed", "store_run"}:
            return
        if line not in self.handler_lines.get(code, ()):
            return
        error = sys.exception()
        if error is None:
            return
        actor = self._actor()
        row = self.live.get((actor, id(frame)))
        if row is None:
            self.invalid.append("handled_exception_without_original_start")
            return
        category = type(error).__module__ + "." + type(error).__name__
        if category in row["handled_exception_types"]:
            return
        row["handled_exception_types"].append(category)
        event = {
            "span": row["id"],
            "call": row["call"],
            "category": category,
            "line": line,
        }
        trace = error.__traceback__
        origins = []
        for _ in range(12):
            if trace is None:
                break
            frame_code = trace.tb_frame.f_code
            if frame_code in self.codes:
                origins.append(
                    {"selected_code": self.codes[frame_code], "line": trace.tb_lineno}
                )
            trace = trace.tb_next
        event["selected_exception_origins"] = origins
        # No error message, args, response, canonical message, SQL or config values.
        self.handlers.append(event)

    def _snapshot(self, boundary, result):
        if boundary == "original_terminal_assertion":
            self._bind_loaded()
            self.discovery["original_terminal_seen"] = True
            complete = (
                self.store_type is not None
                and self.coordinator_type is not None
                and bool(self.stores)
                and bool(self.coordinators)
                and {item["source"] for item in self.discovery["qualified_bindings"]}
                == {"store", "coordinator"}
            )
            if not complete:
                self.invalid.append(
                    "original_terminal_source_or_owner_coverage_incomplete"
                )
            assert complete, "original terminal witness requires qualified stock sources and observed owners"
        stores = []
        for owner in self.stores.values():
            values = owner.__dict__
            lock = values["_provider_trace_settlement_lock"]
            if not lock.acquire(blocking=False):
                stores.append({"owner": id(owner), "busy": True})
                continue
            try:
                pending = values["_provider_trace_settlements"]
                stores.append(
                    {
                        "owner": id(owner),
                        "busy": False,
                        "pending": sum(len(value) for value in pending.values()),
                        "work": [
                            self._token(key)
                            for key in values["_provider_trace_settlement_work"]
                        ],
                        "failed": [
                            self._token(key)
                            for key in values["_provider_trace_settlement_failed_work"]
                        ],
                        "owned": [
                            self._token(key)
                            for key in values[
                                "_provider_trace_settlement_owned_call_ids"
                            ]
                        ],
                        "worker_active": values[
                            "_provider_trace_settlement_worker_active"
                        ],
                        "executor_closed": values[
                            "_provider_trace_settlement_executor_closed"
                        ],
                    }
                )
            finally:
                lock.release()
        coordinators = []
        for owner in self.coordinators.values():
            values = owner.__dict__
            lock = values["_queue_lock"]
            if not lock.acquire(blocking=False):
                coordinators.append({"owner": id(owner), "busy": True})
                continue
            try:
                coordinators.append(
                    {
                        "owner": id(owner),
                        "busy": False,
                        "pending": [self._token(key) for key in values["_pending"]],
                        "inflight": [self._token(key) for key in values["_inflight"]],
                        "dropped": values["_dropped_count"],
                    }
                )
            finally:
                lock.release()
        return {
            "boundary": boundary,
            "time": time.perf_counter(),
            "trace_states": list(result.get("trace_states", [])),
            "response_links": result.get("response_links"),
            "provider_calls": result.get("provider_calls"),
            "live_spans": sorted(row["id"] for row in self.live.values()),
            "stores": stores,
            "coordinators": coordinators,
        }

    def start(self):
        assert type(self.test_body) is FunctionType
        assert self.test_body.__globals__ is self.probe.__dict__
        assert _shape(self.test_body.__code__) == _shape(
            _compiled_code(
                self.probe,
                TARGET_NAME,
                self.test_body.__code__.co_filename,
                assertion_config=self.assertion_config,
                assertion_loader=self.assertion_loader,
            )
        )
        self.module_hashes[self.probe.__name__] = hashlib.sha256(
            Path(self.probe.__file__).read_bytes()
        ).hexdigest()
        self.installed_sources_before = self._loaded_source_records()
        self.tool = next(
            slot for slot in range(6) if self.monitor.get_tool(slot) is None
        )
        self.monitor.use_tool_id(self.tool, self.tool_name)
        self.active = True
        try:
            for event, callback in (
                (self.monitor.events.PY_START, self._start),
                (self.monitor.events.PY_RETURN, self._returned),
                (self.monitor.events.LINE, self._line),
            ):
                self.callbacks[event] = callback
                assert (
                    self.monitor.register_callback(self.tool, event, callback) is None
                )
            self._enable(self.test_code, only_line=True)
            self._bind_loaded()
            # New dev imports the store lazily. Original test LINE/selected
            # frame callbacks retry only already-loaded source qualification.
            assert self.monitor.get_events(self.tool) == 0
        except BaseException:
            self.stop({})
            raise

    def stop(self, result):
        self.snapshots.append(
            self._snapshot("original_observation_write_after_teardown", result)
        )
        assert self.active and self.monitor.get_tool(self.tool) == self.tool_name
        assert self.monitor.get_events(self.tool) == 0
        assert all(
            self.monitor.get_local_events(self.tool, code) == mask
            for code, mask in self.masks.items()
        )
        for code in self.masks:
            self.monitor.set_local_events(self.tool, code, 0)
        assert all(
            self.monitor.get_local_events(self.tool, code) == 0 for code in self.masks
        )
        for event, callback in self.callbacks.items():
            assert self.monitor.register_callback(self.tool, event, None) is callback
        self.monitor.free_tool_id(self.tool)
        assert self.monitor.get_tool(self.tool) is None
        self.active = False
        assert (
            self.test_body.__code__ is self.test_code
            and self.test_body.__globals__ is self.probe.__dict__
        )
        assert sys.modules.get(PROBE_NAME) is self.probe
        assert _shape(self.test_code) == _shape(
            _compiled_code(
                self.probe,
                TARGET_NAME,
                self.test_code.co_filename,
                assertion_config=self.assertion_config,
                assertion_loader=self.assertion_loader,
            )
        )
        assert (
            hashlib.sha256(Path(self.probe.__file__).read_bytes()).hexdigest()
            == self.module_hashes[PROBE_NAME]
        )
        for module, klass, name, descriptor, function, code in self.bindings:
            assert sys.modules.get(module.__name__) is module
            assert module.__dict__.get(klass.__name__) is klass
            assert klass.__dict__.get(name) is descriptor
            assert function.__code__ is code and function.__globals__ is module.__dict__
            assert (
                hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                == self.module_hashes[module.__name__]
            )
        installed_sources_after = self._loaded_source_records()
        for name, (module, origin, digest) in self.installed_modules.items():
            assert sys.modules.get(name) is module
            assert module.__dict__.get("__file__") == origin
            assert hashlib.sha256(Path(origin).read_bytes()).hexdigest() == digest
        return {
            "diagnostic_only": True,
            "spans": self.rows,
            "handled_exceptions": self.handlers,
            "snapshots": self.snapshots,
            "invalid_evidence": self.invalid,
            "live_spans_after_teardown": sorted(
                row["id"] for row in self.live.values()
            ),
            "original_coordinator_bound": self.coordinator_type is not None,
            "lazy_source_discovery": self.discovery,
            "original_terminal_source_and_owner_coverage_complete": self.discovery[
                "original_terminal_seen"
            ]
            and not any(
                item == "original_terminal_source_or_owner_coverage_incomplete"
                for item in self.invalid
            ),
            "source_hashes": self.module_hashes,
            "actual_child_installed_sources_before": self.installed_sources_before,
            "actual_child_installed_sources_after": installed_sources_after,
            "actual_child_installed_sources_unchanged": True,
            "global_events": 0,
            "local_code_count": len(self.masks),
            "hooks_retired_before_inactive": True,
            "actual_thread_objects_retained": True,
            "limits": "Code-local selected starts/returns and handler lines add overhead; timing is diagnostic only. "
            "Missing child returns may be retired by a normal original synchronous parent return and remain exceptional evidence. "
            "A busy census or remaining live span is explicit incomplete evidence. No added waits, SQL queries, guarded callable replacements, "
            "unwind/C events, broad task census, error messages or response/config values.",
        }


@pytest.fixture(autouse=True)
def original_trace_probe_witness(request, monkeypatch):
    if request.node.name != TARGET_NAME or request.module.__name__ != PROBE_NAME:
        yield
        return
    from Tests import private_profile

    if not private_profile.is_private_profile_child(request):
        yield  # Parent never installs an observer or overwrites child receipts.
        return
    wrapper = request.function
    assert type(wrapper) is FunctionType
    assert wrapper.__globals__ is private_profile.__dict__
    expected_wrapper = next(
        code
        for code in private_profile.private_profile_test.__code__.co_consts
        if type(code) is CodeType and code.co_name == "wrapped"
    )
    assert wrapper.__code__ is expected_wrapper
    closure = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__))
    body = closure["function"].cell_contents
    assert type(body) is FunctionType and wrapper.__wrapped__ is body
    probe = request.module
    original_install, original_write = (
        probe.Observation.install,
        probe.Observation.write,
    )
    observers = []

    def install(observation, patcher):
        original_install(observation, patcher)
        observer = TraceSettlementWitness(
            probe, body, observation, assertion_config=request.config
        )
        observers.append(observer)
        observation.trace_settlement_witness = observer
        try:
            observer.start()
        except BaseException:
            observation.stop.set()
            observation.sampler.join(
                timeout=2
            )  # Original test-owned sampler retirement.
            assert not observation.sampler.is_alive()
            raise

    def write(observation, result):
        result["diagnostic_trace_settlement"] = (
            observation.trace_settlement_witness.stop(result)
        )
        result["diagnostic_only"] = True
        original_write(observation, result)

    monkeypatch.setattr(probe.Observation, "install", install)
    monkeypatch.setattr(probe.Observation, "write", write)
    try:
        yield
    finally:
        for observer in observers:
            if observer.active:
                observer.invalid.append("original_observation_write_not_reached")
                observer.stop({})
