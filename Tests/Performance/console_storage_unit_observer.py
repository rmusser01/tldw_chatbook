"""Finite original-code census mechanics; no application imports or replacements."""

import hashlib
import inspect
import contextlib
import builtins
from pathlib import Path
import sys
import threading
from types import CodeType, FunctionType, ModuleType, GeneratorType, BuiltinMethodType
import weakref


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
        tuple(
            _shape(value) if type(value) is CodeType else value
            for value in code.co_consts
        ),
    )


def _nested(code, actual):
    if (
        code.co_qualname == actual.co_qualname
        and code.co_firstlineno == actual.co_firstlineno
    ):
        return code
    for value in code.co_consts:
        if type(value) is CodeType:
            selected = _nested(value, actual)
            if selected is not None:
                return selected
    return None


class OriginalStorageUnitObserver:
    """Count complete original spans, retaining only codes, metadata and Threads.

    Exceptional terminals cannot be observed locally on Python 3.12. Unresolved
    spans make the receipt invalid; they never justify accepted lower counts.
    """

    def __init__(self, counts, counting, bump):
        self.counts, self.counting, self.bump = counts, counting, bump
        self.monitor = sys.monitoring
        self.tool, self.active = None, False
        self.codes, self.pins, self.slots, self.modules = {}, [], [], {}
        self.pending, self.depth, self.threads, self.invalid = {}, {}, {}, []
        self.started = self.returned = 0
        self.installed = False
        self.registered = {}
        self.context_codes = {}
        self.closed_without_return = 0

    def _pin(self, function):
        assert type(function) is FunctionType
        module = sys.modules[function.__globals__["__name__"]]
        assert type(module) is ModuleType and function.__globals__ is vars(module)
        path = Path(module.__file__).resolve()
        assert Path(module.__spec__.origin).resolve() == path
        assert Path(function.__code__.co_filename).resolve() == path
        data = path.read_bytes()
        compiled = compile(data, str(path), "exec", dont_inherit=True)
        expected = _nested(compiled, function.__code__)
        assert expected is not None and _shape(expected) == _shape(function.__code__)
        closure = function.__closure__
        self.pins.append(
            (
                function,
                function.__code__,
                function.__globals__,
                function.__defaults__,
                function.__kwdefaults__,
                tuple((function.__kwdefaults__ or {}).items()),
                closure,
                tuple((cell, cell.cell_contents) for cell in closure or ()),
            )
        )
        self.modules[module.__name__] = (
            module,
            path,
            module.__spec__,
            module.__spec__.origin,
            module.__loader__,
            module.__spec__.loader,
            hashlib.sha256(data).hexdigest(),
        )
        return function.__code__

    def install(self, config_participants, storage, helper_class):
        assert self.tool is None
        operation = inspect.getattr_static(config_participants, "operation")
        body = inspect.getattr_static(operation, "__wrapped__")
        acquire = inspect.getattr_static(storage, "_acquire_storage")
        descriptor = inspect.getattr_static(helper_class, "start")
        assert type(descriptor) is classmethod
        start = descriptor.__func__
        self._pin(operation)  # Defining contextlib wrapper, never globally observed.
        operation_code = self._pin(body)
        cells = dict(zip(operation.__code__.co_freevars, operation.__closure__ or ()))
        assert cells["func"].cell_contents is body
        self.codes = {
            operation_code: "config_admissions",
            self._pin(acquire): "storage_admissions",
            self._pin(start): "helper_spawns",
        }
        self.slots = [
            (config_participants, "operation", operation),
            (operation, "__wrapped__", body),
            (storage, "_acquire_storage", acquire),
            (helper_class, "start", descriptor),
        ]
        self.context_owner = contextlib._GeneratorContextManager
        self.enter_function = inspect.getattr_static(self.context_owner, "__enter__")
        self.exit_function = inspect.getattr_static(self.context_owner, "__exit__")
        self.context_codes = {
            self._pin(self.enter_function),
            self._pin(self.exit_function),
        }
        self.slots.extend(
            (
                (self.context_owner, "__enter__", self.enter_function),
                (self.context_owner, "__exit__", self.exit_function),
            )
        )
        for tool in range(5, 0, -1):
            if tool == self.monitor.DEBUGGER_ID:
                continue
            try:
                self.monitor.use_tool_id(tool, "original-storage-unit-census")
            except ValueError:
                continue
            self.tool = tool
            break
        assert self.tool is not None
        for event, callback in (
            (self.monitor.events.PY_START, self._start),
            (self.monitor.events.PY_RETURN, self._return),
            (self.monitor.events.C_RAISE, self._raised),
        ):
            previous = self.monitor.register_callback(self.tool, event, callback)
            self.registered[event] = callback
            if previous is not None:
                self.monitor.register_callback(self.tool, event, previous)
                self.registered.pop(event)
                raise RuntimeError("monitoring_callback_not_unowned")
        self.active = True
        for code in self.codes:
            self.monitor.set_local_events(
                self.tool,
                code,
                self.monitor.events.PY_START | self.monitor.events.PY_RETURN,
            )
        for code in self.context_codes:
            self.monitor.set_local_events(self.tool, code, self.monitor.events.CALL)
        assert self.monitor.get_events(self.tool) == 0
        self.installed = True

    def _frame(self, code):
        frame = sys._getframe(1)
        for _ in range(6):
            if frame is None:
                break
            if frame.f_code is code:
                return frame
            frame = frame.f_back
        raise RuntimeError("selected_frame_missing")

    def _start(self, code, offset):
        if not self.active or code not in self.codes:
            return
        try:
            frame = self._frame(code)
            assert frame.f_globals is next(
                pin[2] for pin in self.pins if pin[1] is code
            )
            thread = threading.current_thread()
            token = id(thread)
            assert (
                len(self.pending) < 128
                and len(self.threads) < 64
                and self.started < 100_000
            )
            self.threads[token] = thread
            key = id(frame)
            self._reap_terminal()
            assert key not in self.pending
            unit = self.codes[code]
            generator = (
                self._config_generator(frame) if unit == "config_admissions" else None
            )
            self.pending[key] = (
                token,
                unit,
                weakref.ref(generator) if generator is not None else None,
            )
            self.started += 1
            if unit == "config_admissions":
                depth = self.depth.get(token, 0)
                self.depth[token] = depth + 1
                if depth == 0:
                    self.bump(unit)
            else:
                self.bump(unit)
        except BaseException as error:
            self.invalid.append("start:" + type(error).__name__)

    def _return(self, code, offset, value):
        if not self.active or code not in self.codes:
            return
        try:
            frame = self._frame(code)
            token, unit, reference = self.pending.pop(id(frame))
            assert self.threads[token] is threading.current_thread()
            if unit == "config_admissions":
                assert self.depth[token] > 0
                self.depth[token] -= 1
            self.returned += 1
        except BaseException as error:
            self.invalid.append("return:" + type(error).__name__)

    def _config_generator(self, frame):
        parent = frame.f_back
        for _ in range(6):
            if parent is None:
                break
            if parent.f_code is self.enter_function.__code__:
                assert parent.f_globals is self.enter_function.__globals__
                receiver = parent.f_locals.get("self")
                assert type(receiver) is self.context_owner
                generator = vars(receiver).get("gen")
                assert type(generator) is GeneratorType
                assert generator.gi_code is frame.f_code and generator.gi_frame is frame
                return generator
            parent = parent.f_back
        raise RuntimeError("original_context_generator_not_found")

    def _reap_terminal(self):
        for key, (token, unit, reference) in tuple(self.pending.items()):
            if reference is None:
                continue
            generator = reference()
            if generator is not None and generator.gi_frame is None:
                assert unit == "config_admissions" and self.depth[token] > 0
                self.depth[token] -= 1
                self.pending.pop(key)
                self.closed_without_return += 1

    def _raised(self, code, offset, callable_, arg):
        if not self.active or code not in self.context_codes:
            return
        try:
            frame = self._frame(code)
            function = (
                self.enter_function
                if code is self.enter_function.__code__
                else self.exit_function
            )
            assert frame.f_globals is function.__globals__
            receiver = frame.f_locals.get("self")
            if type(receiver) is not self.context_owner:
                return
            generator = vars(receiver).get("gen")
            if (
                type(generator) is not GeneratorType
                or generator.gi_code not in self.codes
            ):
                return
            if self.codes[generator.gi_code] != "config_admissions":
                return
            if not (
                callable_ is builtins.next
                and arg is generator
                or (
                    callable_ is GeneratorType.throw or callable_ is GeneratorType.close
                )
                and arg is generator
                or type(callable_) is BuiltinMethodType
                and callable_.__self__ is generator
                and callable_.__name__ in {"throw", "close"}
            ):
                return
            self._reap_terminal()  # The exact live generator proves terminal, never a dead weakref.
        except BaseException as error:
            self.invalid.append("exception_terminal:" + type(error).__name__)

    def close(self):
        # Retirement precedes both inactive state and any acceptance judgment.
        retirement = []
        global_events = 0
        if self.tool is not None:
            for code in (*self.codes, *self.context_codes):
                try:
                    self.monitor.set_local_events(self.tool, code, 0)
                except BaseException as error:
                    retirement.append("local:" + type(error).__name__)
            for event, expected in self.registered.items():
                try:
                    if (
                        self.monitor.register_callback(self.tool, event, None)
                        is not expected
                    ):
                        retirement.append("callback_owner_changed")
                except BaseException as error:
                    retirement.append("callback:" + type(error).__name__)
            try:
                global_events = self.monitor.get_events(self.tool)
                assert global_events == 0
            except BaseException as error:
                retirement.append("global_events:" + type(error).__name__)
            finally:
                try:
                    self.monitor.set_events(self.tool, 0)
                except BaseException as error:
                    retirement.append("global_disable:" + type(error).__name__)
                finally:
                    try:
                        self.monitor.free_tool_id(self.tool)
                    except BaseException as error:
                        retirement.append("tool:" + type(error).__name__)
        self.active = False
        self.invalid.extend(retirement)
        try:
            self._reap_terminal()
        except BaseException as error:
            self.invalid.append("terminal_reap:" + type(error).__name__)
        current = True
        try:
            for (
                function,
                code,
                defining,
                defaults,
                kwdefaults,
                items,
                closure,
                cells,
            ) in self.pins:
                current &= (
                    function.__code__ is code
                    and function.__globals__ is defining
                    and function.__defaults__ is defaults
                    and function.__kwdefaults__ is kwdefaults
                    and len(function.__kwdefaults__ or {}) == len(items)
                    and all(
                        (function.__kwdefaults__ or {}).get(key) is value
                        for key, value in items
                    )
                    and function.__closure__ is closure
                    and all(cell.cell_contents is value for cell, value in cells)
                )
            current &= all(
                inspect.getattr_static(owner, name) is value
                for owner, name, value in self.slots
            )
            for name, (
                module,
                path,
                spec,
                origin,
                loader,
                spec_loader,
                digest,
            ) in self.modules.items():
                current &= (
                    sys.modules.get(name) is module
                    and module.__spec__ is spec
                    and spec.origin == origin
                    and module.__loader__ is loader
                    and spec.loader is spec_loader
                    and Path(module.__file__).resolve() == path
                    and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                )
        except BaseException as error:
            current = False
            self.invalid.append("source:" + type(error).__name__)
        receipt = dict(
            original_source_current=bool(current),
            global_events=global_events,
            hooks_retired_before_inactive=not retirement,
            install_complete=self.installed,
            started=self.started,
            returned=self.returned,
            closed_without_return=self.closed_without_return,
            unresolved_spans=[
                (token, unit, reference is not None and reference() is not None)
                for token, unit, reference in self.pending.values()
            ],
            nonzero_depths=[value for value in self.depth.values() if value],
            invalid=self.invalid,
            frames_arguments_results_tasks_retained=False,
            replacements_installed=False,
        )
        receipt["complete"] = (
            self.installed
            and current
            and not self.pending
            and not any(self.depth.values())
            and not self.invalid
        )
        return receipt
