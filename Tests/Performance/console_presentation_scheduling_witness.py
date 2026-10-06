"""Opt-in original history/archive scheduling; scalar metadata only."""
# ruff: noqa: E721 -- exact source-qualified builtin containers; no custom accessors.

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
import weakref
from types import FunctionType, ModuleType

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.Performance.console_storage_unit_observer import _shape, _nested


OUTPUT = "TLDW_HISTORY_ARCHIVE_SCHEDULING_RECEIPT"
PROBE = "Tests.Performance.test_console_native_pause_probe"
TARGET = "Tests/Performance/test_console_native_pause_probe.py::test_native_console_pause_probe"
EXTENSION = None


class SchedulingWitness(OriginalStorageUnitObserver):
    """Reuse accepted source/retirement machinery, retaining no dynamic owners."""

    def __init__(self, observation, root):
        super().__init__({}, {}, lambda name: None)
        self.observation, self.root = observation, Path(root)
        self.rows, self.spans, self.last_keys = [], {}, {}
        self.event_counts = {}
        self.sequence = 0
        self.lines, self.overflow = {}, 0
        self.invalid_overflow = 0
        self.salt = os.urandom(32)
        self.started_at = time.monotonic()
        self.caller_codes = {}
        self.bound = False
        self.close_receipt = None

    def _pin(self, function):
        # These selected originals are undecorated bodies with literal defaults.
        # Reject an already retargeted default/closure before retaining a pin.
        assert type(function) is FunctionType and function.__closure__ is None
        tree = ast.parse(Path(function.__code__.co_filename).read_bytes())
        definition = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.lineno == function.__code__.co_firstlineno
            and node.name == function.__code__.co_name
        )
        assert not definition.decorator_list
        expected = tuple(ast.literal_eval(node) for node in definition.args.defaults)
        actual = function.__defaults__ or ()
        assert type(actual) is tuple and len(actual) == len(expected)
        assert all(
            type(value) is type(original) and value == original
            for value, original in zip(actual, expected, strict=True)
        )
        keywords = {
            argument.arg: ast.literal_eval(value)
            for argument, value in zip(
                definition.args.kwonlyargs, definition.args.kw_defaults, strict=True
            )
            if value is not None
        }
        actual_keywords = function.__kwdefaults__ or {}
        assert (
            type(actual_keywords) is dict and actual_keywords.keys() == keywords.keys()
        )
        assert all(
            type(actual_keywords[name]) is type(value)
            and actual_keywords[name] == value
            for name, value in keywords.items()
        )
        return super()._pin(function)

    def invalidate(self, stage, error):
        if len(self.invalid) < 64:
            self.invalid.append(stage + ":" + type(error).__name__)
        else:
            self.invalid_overflow += 1

    def digest(self, value, *, identity=False):
        if identity:
            data = b"object:" + str(id(value)).encode()
        elif value is None:
            data = b"none"
        elif type(value) in {str, int, bool}:
            data = type(value).__name__.encode() + b":" + str(value).encode()
        elif type(value) is tuple and len(value) <= 64:
            data = b"tuple:" + b";".join(self.digest(item).encode() for item in value)
        else:
            data = b"object:" + str(id(value)).encode()
        return hashlib.blake2b(data, key=self.salt, digest_size=12).hexdigest()

    def add(self, event, frame, **fields):
        count_key = (self.observation.phase, event)
        if count_key not in self.event_counts and len(self.event_counts) >= 1024:
            self.overflow += 1
            return
        self.event_counts[count_key] = self.event_counts.get(count_key, 0) + 1
        if event == "key_start" or (
            event == "key_return" and not fields["changed_slots"]
        ):
            return  # Counts remain exact; unchanged key calls need no repeated row.
        if len(self.rows) >= 4096:
            self.overflow += 1
            return
        self.sequence += 1
        self.rows.append(
            dict(
                sequence=self.sequence,
                event=event,
                phase=self.observation.phase,
                time=time.monotonic() - self.started_at,
                thread=self.digest(threading.current_thread(), identity=True),
                **fields,
            )
        )

    def bind_selected(self):
        if self.bound:
            return
        if any(
            sys.modules.get(name) is None
            for name in (
                "tldw_chatbook.UI.Console_Modules.agent",
                "tldw_chatbook.UI.Console_Modules.workspace",
            )
        ):
            return
        assert len(self.pins) < 16
        specs = {
            "tldw_chatbook.UI.Console_Modules.agent": (
                "ConsoleAgentController",
                {
                    "_historical_presentation_key": "key",
                    "_presentation_historical_snapshot": "presentation",
                    "_load_historical_presentation": "load",
                },
            ),
            "tldw_chatbook.UI.Console_Modules.workspace": (
                "ConsoleWorkspaceController",
                {"_refresh_console_persisted_rows_cache": "browser"},
            ),
        }
        for name, (class_name, methods) in specs.items():
            module = sys.modules.get(name)
            assert type(module) is ModuleType
            path = self.root / (name.replace(".", "/") + ".py")
            assert Path(module.__file__).resolve() == path.resolve()
            owner = inspect.getattr_static(module, class_name) if class_name else module
            if class_name:
                self.slots.append((module, class_name, owner))
            for method, label in methods.items():
                function = inspect.getattr_static(owner, method)
                assert type(function) is FunctionType and function.__globals__ is vars(
                    module
                )
                code = self._pin(function)
                self.codes[code] = label
                self.slots.append((owner, method, function))
                # Only selected original exception-handler/assignment lines.
                tree = ast.parse(path.read_bytes())
                definition = next(
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == method
                    and node.lineno == code.co_firstlineno
                )
                points = {}
                if label == "load":
                    for node in ast.walk(definition):
                        if isinstance(node, ast.ExceptHandler):
                            points[node.body[0].lineno] = "load_error_handler"
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Subscript)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "state"
                            and isinstance(target.slice, ast.Constant)
                            and target.slice.value == "value"
                            for target in node.targets
                        ):
                            points[node.lineno] = "publish_attempt"
                        if isinstance(
                            node, ast.If
                        ) and "_historical_presentation_key" in ast.unparse(node.test):
                            points[node.body[0].lineno] = "load_rejected"
                elif label == "presentation":
                    for node in ast.walk(definition):
                        if isinstance(node, ast.Assign) and isinstance(
                            node.value, ast.Dict
                        ):
                            points[node.lineno] = "new_history_state"
                        if (
                            isinstance(node, ast.Assign)
                            and isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Attribute)
                            and node.value.func.attr == "run_worker"
                        ):
                            points[node.lineno] = "schedule_attempt"
                if label == "browser":
                    for node in ast.walk(definition):
                        if (
                            isinstance(node, ast.Await)
                            and isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == "storage_call"
                        ):
                            points[node.lineno] = "archive_callsite"
                self.lines[code] = points
                self.watch(code)
        self.bound = True

    def bind_archive(self):
        if "archive" in self.codes.values():
            return
        name = "tldw_chatbook.Chat.conversation_archive_actions"
        module = sys.modules.get(name)
        assert type(module) is ModuleType
        path = self.root / (name.replace(".", "/") + ".py")
        assert Path(module.__file__).resolve() == path.resolve()
        function = inspect.getattr_static(module, "storage_call")
        code = self._pin(function)
        self.codes[code], self.lines[code] = "archive", {}
        self.slots.append((module, "storage_call", function))
        self.watch(code)

    def watch(self, code):
        mask = self.monitor.events.PY_START | self.monitor.events.PY_RETURN
        if code.co_flags & inspect.CO_COROUTINE:
            mask |= self.monitor.events.PY_YIELD | self.monitor.events.PY_RESUME
        if self.lines[code]:
            mask |= self.monitor.events.LINE
        self.monitor.set_local_events(self.tool, code, mask)

    def install(self):
        name = "tldw_chatbook.UI.Navigation.screen_registry"
        module = sys.modules.get(name)
        assert type(module) is ModuleType
        owner = inspect.getattr_static(module, "ScreenRoute")
        function = inspect.getattr_static(owner, "load_screen_class")
        assert (
            Path(module.__file__).resolve()
            == (self.root / (name.replace(".", "/") + ".py")).resolve()
        )
        code = self._pin(function)
        self.codes[code], self.lines[code] = "screen_loader", {}
        self.slots.extend(
            ((module, "ScreenRoute", owner), (owner, "load_screen_class", function))
        )
        self.tool = next(
            value for value in range(5, -1, -1) if self.monitor.get_tool(value) is None
        )
        self.monitor.use_tool_id(self.tool, "history-archive-scheduling")
        callbacks = {
            self.monitor.events.PY_START: self.start_span,
            self.monitor.events.PY_RETURN: self.return_span,
            self.monitor.events.PY_YIELD: self.yield_span,
            self.monitor.events.PY_RESUME: self.resume_span,
            self.monitor.events.LINE: self.line_span,
        }
        for event, callback in callbacks.items():
            assert self.monitor.register_callback(self.tool, event, callback) is None
            self.registered[event] = callback
        self.active = True
        for code in tuple(self.codes):
            self.watch(code)
        self.bind_selected()
        assert self.monitor.get_events(self.tool) == 0
        self.installed = True

    @staticmethod
    def caller_current(record):
        path, digest, name, module, defining, spec, origin, loader, spec_loader = record
        return (
            sys.modules.get(name) is module
            and vars(module) is defining
            and module.__spec__ is spec
            and spec.origin == origin
            and Path(origin).resolve() == path
            and module.__loader__ is loader
            and spec.loader is spec_loader
            and Path(module.__file__).resolve() == path
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        )

    def start_span(self, code, offset):
        if not self.active or code not in self.codes:
            return
        try:
            frame = self._frame(code)
            assert frame.f_globals is next(
                pin[2] for pin in self.pins if pin[1] is code
            )
            label = self.codes[code]
            thread = threading.current_thread()
            token = id(thread)
            assert token in self.threads or len(self.threads) < 64
            assert self.threads.get(token, thread) is thread
            self.threads[token] = thread
            key = (token, id(frame))
            assert key not in self.spans and len(self.spans) < 128
            self.started += 1
            self.spans[key] = label
            fields = {}
            if label == "presentation":
                owner = frame.f_locals.get("self")
                state = vars(owner).get("_console_historical_read")
                fields = dict(
                    state_before=self.digest(state, identity=True),
                    pending_before=type(state) is dict and state.get("pending") is True,  # noqa: E721 -- exact built-in metadata
                    value_before=type(state) is dict and state.get("value") is not None,
                )  # noqa: E721 -- exact built-in metadata
            if label == "archive":
                method = frame.f_locals.get("method")
                if method == "get_conversation_archive_states":
                    args = frame.f_locals.get("args")
                    ids = args[0] if type(args) is tuple and args else ()
                    assert (
                        type(ids) in {list, tuple}
                        and len(ids) <= 4096
                        and all(type(item) is str for item in ids)
                    )  # noqa: E721 -- never invoke custom metadata containers
                    identifier_hash = hashlib.blake2b(key=self.salt, digest_size=12)
                    for identifier in sorted(ids):
                        encoded = identifier.encode()
                        identifier_hash.update(len(encoded).to_bytes(8, "big"))
                        identifier_hash.update(encoded)
                    fields = dict(
                        operation=method,
                        ids_count=len(ids),
                        ids_hash=identifier_hash.hexdigest(),
                    )
                    caller = frame.f_back
                    if caller is not None:
                        path = Path(caller.f_code.co_filename).resolve()
                        assert path.is_relative_to(self.root)
                        module = sys.modules.get(caller.f_globals.get("__name__"))
                        assert (
                            type(module) is ModuleType
                            and vars(module) is caller.f_globals
                        )
                        assert Path(module.__file__).resolve() == path
                        if caller.f_code in self.caller_codes:
                            record = self.caller_codes[caller.f_code]
                            assert record[3] is module and record[4] is caller.f_globals
                            assert self.caller_current(record)
                        else:
                            assert len(self.caller_codes) < 16
                            assert Path(module.__spec__.origin).resolve() == path
                            data = path.read_bytes()
                            compiled = compile(
                                data, str(path), "exec", dont_inherit=True
                            )
                            expected = _nested(compiled, caller.f_code)
                            assert expected is not None and _shape(expected) == _shape(
                                caller.f_code
                            )
                            self.caller_codes[caller.f_code] = (
                                path,
                                hashlib.sha256(data).hexdigest(),
                                module.__name__,
                                module,
                                caller.f_globals,
                                module.__spec__,
                                module.__spec__.origin,
                                module.__loader__,
                                module.__spec__.loader,
                            )
                        fields.update(
                            caller=path.relative_to(self.root).as_posix(),
                            caller_function=caller.f_code.co_qualname,
                            caller_line=caller.f_lineno,
                        )
                else:
                    fields = dict(operation="other_storage_action")
            if label != "screen_loader":
                self.add(label + "_start", frame, **fields)
        except BaseException as error:
            self.invalidate("start", error)

    def return_span(self, code, offset, value):
        if not self.active or code not in self.codes:
            return
        try:
            frame = self._frame(code)
            assert (
                self.threads[id(threading.current_thread())]
                is threading.current_thread()
            )
            label = self.spans.pop((id(threading.current_thread()), id(frame)))
            self.returned += 1
            if label == "screen_loader":
                self.bind_selected()
                return
            fields = {}
            if label == "key":
                assert type(value) is tuple and len(value) == 11
                signature = tuple(
                    self.digest(item, identity=index in {0, 1, 2, 4})
                    for index, item in enumerate(value)
                )
                owner = frame.f_locals.get("self")
                actor = self.digest(owner, identity=True)
                assert actor in self.last_keys or len(self.last_keys) < 32
                prior = self.last_keys.get(actor)
                previous = (
                    prior[1] if prior is not None and prior[0]() is owner else None
                )
                self.last_keys[actor] = (weakref.ref(owner), signature)
                fields = dict(
                    owner_hash=actor,
                    changed_slots=[
                        index
                        for index in range(11)
                        if previous is None or previous[index] != signature[index]
                    ],
                    conversation_hash=signature[7],
                    status=value[9]
                    if value[9]
                    in {
                        "idle",
                        "setup",
                        "running",
                        "done",
                        "failed",
                        "error",
                        "stuck",
                        "cancelled",
                        "waiting",
                    }
                    else "other",
                )
            elif label in {"presentation", "load"}:
                state = frame.f_locals.get("state")
                fields = dict(
                    state_hash=self.digest(state, identity=True),
                    pending=type(state) is dict and state.get("pending") is True,  # noqa: E721 -- exact built-in metadata
                    value_present=type(state) is dict
                    and state.get("value") is not None,
                )  # noqa: E721 -- exact built-in metadata
                if label == "presentation":
                    fields["scheduled_or_inflight"] = (
                        type(state) is dict and state.get("pending") is True
                    )  # noqa: E721 -- exact built-in metadata
                    fields["reused"] = (
                        type(state) is dict and state.get("value") is not None
                    )  # noqa: E721 -- exact built-in metadata
                else:
                    owner = frame.f_locals.get("self")
                    fields["published_current"] = (
                        type(state) is dict
                        and state.get("value") is not None  # noqa: E721 -- exact built-in metadata
                        and vars(owner).get("_console_historical_read") is state
                    )
            self.add(label + "_return", frame, **fields)
        except BaseException as error:
            self.invalidate("return", error)

    def yield_span(self, code, offset, value):
        if self.active and code in self.codes:
            try:
                frame = self._frame(code)
                assert (
                    self.spans[(id(threading.current_thread()), id(frame))]
                    == self.codes[code]
                )
                self.add(self.codes[code] + "_yield", None)
            except BaseException as error:
                self.invalidate("yield", error)

    def resume_span(self, code, offset):
        if self.active and code in self.codes:
            try:
                frame = self._frame(code)
                assert (
                    self.spans[(id(threading.current_thread()), id(frame))]
                    == self.codes[code]
                )
                self.add(self.codes[code] + "_resume", None)
            except BaseException as error:
                self.invalidate("resume", error)

    def line_span(self, code, line):
        if self.active and line in self.lines.get(code, {}):
            try:
                frame = self._frame(code)
                assert (
                    self.spans[(id(threading.current_thread()), id(frame))]
                    == self.codes[code]
                )
                if self.lines[code][line] == "archive_callsite":
                    self.bind_archive()
                self.add(self.lines[code][line], None)
            except BaseException as error:
                self.invalidate("line", error)

    def close(self):
        if self.close_receipt is not None:
            return self.close_receipt
        receipt = super().close()
        try:
            current = all(
                self.caller_current(record) for record in self.caller_codes.values()
            )
        except BaseException as error:
            current = False
            receipt["invalid"].append("caller_source:" + type(error).__name__)
        receipt.update(
            diagnostic_only=True,
            rows=self.rows,
            selected_event_counts=[
                dict(phase=phase, event=event, count=count)
                for (phase, event), count in sorted(self.event_counts.items())
            ],
            overflow=self.overflow,
            invalid_overflow=self.invalid_overflow,
            selected_bound=self.bound,
            caller_source_current=current,
            unresolved_selected_spans=list(self.spans.values()),
            owners_keys_authority_bodies_retained=False,
            limits="Code-local observation adds overhead; phase labels are original shared labels, not async ownership. Exceptional unmatched spans are invalid, never synthetic returns.",
        )
        receipt["complete"] &= (
            self.bound
            and current
            and not self.spans
            and not self.overflow
            and not self.invalid_overflow
        )
        self.last_keys.clear()
        self.close_receipt = receipt
        return receipt


def pytest_runtest_setup(item):
    global EXTENSION
    if not os.environ.get(OUTPUT):
        return
    assert item.nodeid == TARGET
    if os.environ.get("TLDW_TEST_PRIVATE_PROFILE_NODE") != TARGET:
        return  # Parent only launches the original private child; no observer or output.
    assert getattr(item.obj, "_private_profile_test", False) is True
    probe = sys.modules[PROBE]
    assert type(probe) is ModuleType and EXTENSION is None

    original_install, original_write = (
        probe.Observation.install,
        probe.Observation.write,
    )

    def install(observation, monkeypatch):
        original_install(observation, monkeypatch)
        observation.history_archive_witness = SchedulingWitness(
            observation, Path(probe.__file__).resolve().parents[2]
        )
        try:
            observation.history_archive_witness.install()
        except BaseException:
            observation.history_archive_witness.close()
            observation.stop.set()
            observation.sampler.join(timeout=2)
            assert not observation.sampler.is_alive()
            raise

    def write(observation, result):
        try:
            receipt = observation.history_archive_witness.close()
            path = Path(os.environ[OUTPUT])
            assert not path.exists()
            path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        finally:
            original_write(observation, result)

    EXTENSION = probe, original_install, original_write, install, write
    probe.Observation.install, probe.Observation.write = install, write


def pytest_unconfigure(config):
    if EXTENSION is not None:
        probe, old_install, old_write, install, write = EXTENSION
        if probe.Observation.install is install:
            probe.Observation.install = old_install
        if probe.Observation.write is write:
            probe.Observation.write = old_write
