"""Evidence-only real reader/real Close controls; root runs serially."""

import json
import asyncio
import hashlib
import inspect
import os
import sys
import threading
from pathlib import Path
from types import CodeType, FunctionType

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_session_tab_close import (
    ProductionConsoleHarness,
    _SIZE,
    _await_tabs,
    _click,
    _mounted_console,
    _open_tab_ids,
    _ready_app,
    _session_ids,
    _settle,
    _show_tabs,
)


def _shape(code):
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


def _source_code(module, qualname):
    code = compile(
        Path(module.__file__).read_bytes(),
        str(module.__file__),
        "exec",
        dont_inherit=True,
    )
    for name in qualname.split("."):
        if name == "<locals>":
            continue
        code = next(
            item
            for item in code.co_consts
            if type(item) is CodeType and item.co_name == name
        )
    return code


class _OriginalReadGate:
    """Hold the actual installed reader at entry, before its native/lock work.

    The reader is neither replaced nor mocked. Its eventual actual native
    calls and checked source proof must be observed after release. Entry holds
    deliberately happen before config/storage locks; holding an arbitrary
    native helper while it owns a global lock would not test UI dependency.
    """

    def __init__(self, projection, store, doomed_id, phase):
        from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
        from tldw_chatbook.UI.Screens import chat_screen

        self.projection, self.store, self.doomed_id, self.phase = (
            projection,
            store,
            doomed_id,
            phase,
        )
        self.reader = projection.read_current
        assert (
            type(self.reader) is FunctionType
            and self.reader.__globals__ is spend.__dict__
        )
        self.reader_code, self.reader_globals = (
            self.reader.__code__,
            self.reader.__globals__,
        )
        self.reader_defaults, self.reader_kwdefaults = (
            self.reader.__defaults__,
            self.reader.__kwdefaults__,
        )
        factory = inspect.getattr_static(
            spend.ConsoleReadinessConfigProjection, "for_screen"
        )
        assert type(factory) is classmethod
        assert _shape(factory.__func__.__code__) == _shape(
            _source_code(spend, "ConsoleReadinessConfigProjection.for_screen")
        )
        nested = tuple(
            item
            for item in factory.__func__.__code__.co_consts
            if type(item) is CodeType and item.co_name == "read_current"
        )
        assert len(nested) == 1 and self.reader.__code__ is nested[0]
        self.reader_closure = self.reader.__closure__
        self.closures = tuple(
            (cell, cell.cell_contents) for cell in self.reader_closure
        )
        named = dict(
            zip(self.reader.__code__.co_freevars, self.reader_closure, strict=True)
        )
        assert named["projection"].cell_contents is projection
        assert named["screen"].cell_contents is projection.screen
        assert named["read_current"].cell_contents is self.reader
        if os.name == "nt":
            from tldw_chatbook.Utils import windows_files as native_module

            native_owner, native_name = native_module._Native, "open_handle"
        else:
            from tldw_chatbook.Backup_Recovery import storage_admission as native_module

            native_owner, native_name = native_module, "_observe_stamps"
        self.native = inspect.getattr_static(native_owner, native_name)
        assert (
            type(self.native) is FunctionType
            and self.native.__globals__ is native_module.__dict__
        )
        assert _shape(self.native.__code__) == _shape(
            _source_code(native_module, self.native.__qualname__)
        )
        from concurrent.futures import _base as future_module
        from concurrent.futures import thread as pool_module

        self.pool_module, self.future_module = pool_module, future_module
        self.work_type, self.future_type = pool_module._WorkItem, future_module.Future
        self.work_run = inspect.getattr_static(self.work_type, "run")
        self.future_methods = {
            name: inspect.getattr_static(self.future_type, name)
            for name in ("done", "running", "cancelled", "exception")
        }
        self.held_work_item, self.held_future = None, None
        self.future_capture = "not_entered"
        self.return_callback_seen = False
        self.future_bindings = []
        self.bindings = []
        for module, owner, name in (
            (pool_module, self.work_type, "run"),
            *((future_module, self.future_type, name) for name in self.future_methods),
        ):
            function = inspect.getattr_static(owner, name)
            assert (
                type(function) is FunctionType
                and function.__globals__ is module.__dict__
            )
            assert (
                Path(function.__code__.co_filename).resolve()
                == Path(module.__file__).resolve()
            )
            assert _shape(function.__code__) == _shape(
                _source_code(module, function.__qualname__)
            )
            binding = (
                owner,
                name,
                function,
                function,
                function.__code__,
                function.__globals__,
                function.__defaults__,
                function.__kwdefaults__,
                function.__closure__,
            )
            self.future_bindings.append(binding)
            self.bindings.append(binding)
        for module, owner, name in (
            (spend, spend.ConsoleReadinessConfigProjection, "for_screen"),
            (spend, spend.ConsoleReadinessConfigProjection, "run"),
            (spend, spend.ConsoleReadinessConfigProjection, "warm"),
            (spend, spend.ConsoleReadinessConfigProjection, "_refresh"),
            (spend, spend.ConsoleContextReadSnapshot, "warm"),
            (chat_screen, chat_screen.ChatScreen, "_sync_native_console_chat_ui"),
            (chat_screen, chat_screen.ChatScreen, "_sync_console_native_session_tabs"),
            (native_module, native_owner, native_name),
        ):
            descriptor = inspect.getattr_static(owner, name)
            function = (
                descriptor.__func__ if type(descriptor) is classmethod else descriptor
            )
            assert (
                type(function) is FunctionType
                and function.__globals__ is module.__dict__
            )
            assert _shape(function.__code__) == _shape(
                _source_code(module, function.__qualname__)
            )
            self.bindings.append(
                (
                    owner,
                    name,
                    descriptor,
                    function,
                    function.__code__,
                    function.__globals__,
                    function.__defaults__,
                    function.__kwdefaults__,
                    function.__closure__,
                )
            )
        self.modules = tuple(
            (
                module,
                Path(module.__file__),
                hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            )
            for module in (
                spend,
                chat_screen,
                native_module,
                pool_module,
                future_module,
            )
        )
        self.entered, self.release = threading.Event(), threading.Event()
        self.reader_returned = threading.Event()
        self.reader_actor = None
        self.held_frame = None
        self.main_actor = threading.current_thread()
        self.native_calls = 0
        self.completed_checked_read = False
        self.live = threading.local()
        self.references = []
        self.active, self.restored, self.tool = False, False, None
        self.monitor = sys.monitoring
        self.callbacks = {}
        self.masks = {
            self.reader_code: self.monitor.events.PY_START
            | self.monitor.events.PY_RETURN,
            self.native.__code__: self.monitor.events.PY_START,
        }

    def _start(self, code, offset):
        frame = sys._getframe(1)
        assert frame.f_code is code
        if (
            code is self.reader_code
            and frame.f_locals.get("projection") is self.projection
        ):
            self.live.frame = frame
            self.references.extend((frame, threading.current_thread()))
            should_hold = (
                self.phase == "coalesced" or self.doomed_id not in self.store._sessions
            )
            if should_hold and not self.entered.is_set():
                self.reader_actor = threading.current_thread()
                self.held_frame = frame
                self._capture_held_future(frame)
                assert self.reader_actor is not self.main_actor
                self.entered.set()
                assert self.release.wait(
                    30
                ), "test never released the original stock reader"
        elif (
            code is self.native.__code__
            and self.held_frame is not None
            and threading.current_thread() is self.reader_actor
        ):
            parent = frame.f_back
            for _ in range(64):
                if parent is self.held_frame:
                    self.native_calls += 1
                    break
                if parent is None:
                    break
                parent = parent.f_back

    def _return(self, code, offset, value):
        frame = sys._getframe(1)
        assert frame.f_code is code and code is self.reader_code
        if frame.f_locals.get("projection") is self.projection:
            if frame is self.held_frame:
                self.return_callback_seen = True
                self.completed_checked_read = (
                    "before" in frame.f_locals
                    and "after" in frame.f_locals
                    and frame.f_locals["before"] == frame.f_locals["after"]
                    and value._display_proof is not None
                )
                self.reader_returned.set()
            self.live.frame = None

    def _future_bindings_current(self):
        if sys.modules.get("concurrent.futures.thread") is not self.pool_module:
            return False
        if sys.modules.get("concurrent.futures._base") is not self.future_module:
            return False
        if vars(self.pool_module).get("_WorkItem") is not self.work_type:
            return False
        if vars(self.future_module).get("Future") is not self.future_type:
            return False
        return all(
            inspect.getattr_static(owner, name) is descriptor
            and function.__code__ is code
            and function.__globals__ is namespace
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            and function.__closure__ is closure
            for owner, name, descriptor, function, code, namespace, defaults, kwdefaults, closure in self.future_bindings
        )

    def _capture_held_future(self, frame):
        self.future_capture = "missing_original_work_item"
        if not self._future_bindings_current():
            self.future_capture = "defining_source_changed"
            return
        parent = frame.f_back
        for _ in range(8):
            if parent is None:
                break
            if (
                parent.f_code is self.work_run.__code__
                and parent.f_globals is self.pool_module.__dict__
            ):
                item = parent.f_locals.get("self")
                if type(item) is self.work_type:
                    future = vars(item).get("future")
                    if type(future) is self.future_type:
                        self.held_work_item, self.held_future = item, future
                        self.future_capture = "qualified_original_work_item"
                        return
                break
            parent = parent.f_back

    def _held_entry_wait(self):
        """Sample only the captured actual actor's live held-reader ancestry."""
        import _thread

        actor, held = self.reader_actor, self.held_frame
        if actor is None or held is None or not actor.is_alive():
            return {"qualified": False, "reason": "held_actor_not_live"}
        frame = sys._current_frames().get(actor.ident)
        chain = []
        for _ in range(64):
            if frame is None:
                break
            chain.append(frame)
            if frame is held:
                break
            frame = frame.f_back
        if not chain or chain[-1] is not held:
            return {"qualified": False, "reason": "exact_held_frame_not_current"}
        rows, sources = [], {}
        for frame in chain:
            code, namespace = frame.f_code, frame.f_globals
            name = namespace.get("__name__")
            module = sys.modules.get(name)
            matched = False
            if type(module) is type(sys) and vars(module) is namespace:
                try:
                    defining = _source_code(module, code.co_qualname)
                    matched = (
                        _shape(defining) == _shape(code)
                        and defining.co_exceptiontable == code.co_exceptiontable
                        and defining.co_stacksize == code.co_stacksize
                    )
                    source_path = Path(module.__file__)
                    sources[name] = hashlib.sha256(source_path.read_bytes()).hexdigest()
                except (
                    AttributeError,
                    OSError,
                    StopIteration,
                    SyntaxError,
                    TypeError,
                    ValueError,
                ):
                    pass
            row = {
                "module": name,
                "qualname": code.co_qualname,
                "definition_line": code.co_firstlineno,
                "current_line": frame.f_lineno,
                "instruction_offset": frame.f_lasti,
                "matches_defining_source": matched,
                "is_exact_held_reader": frame is held,
                "is_original_gate_callback": code
                in {callback.__func__.__code__ for callback in self.callbacks.values()},
            }
            if matched and name in {
                "tldw_chatbook.Backup_Recovery.config_participants",
                "tldw_chatbook.Backup_Recovery.raw_participants",
            }:
                row["builtin_RLock_metadata"] = []
                for local_name in ("lock", "source_lock"):
                    lock = frame.f_locals.get(local_name)
                    if type(lock) is _thread.RLock:
                        row["builtin_RLock_metadata"].append(
                            {
                                "local_name": local_name,
                                "object_id": id(lock),
                                "owner_count_repr": repr(lock),
                                "owned_by_observing_UI_thread": lock._is_owned(),
                            }
                        )
            rows.append(row)
        return {
            "qualified": True,
            "captured_actual_Thread_object_id": id(actor),
            "thread_ident": actor.ident,
            "observing_UI_Thread_ident": self.main_actor.ident,
            "only_exact_held_thread_ancestry_saved": True,
            "no_locals_or_values_saved_except_builtin_lock_metadata": True,
            "no_permission_or_deadlock_verdict": True,
            "defining_source_hashes": sources,
            "frames": rows,
        }

    def reader_diagnostic(self):
        # The retained frame's last line is not a claim that it is still active.
        facts = {
            "held_last_source_line": None
            if self.held_frame is None
            else self.held_frame.f_lineno,
            "held_frame_is_original": self.held_frame is not None
            and self.held_frame.f_code is self.reader_code
            and self.held_frame.f_globals is self.reader_globals,
            "entered": self.entered.is_set(),
            "released": self.release.is_set(),
            "reader_returned": self.reader_returned.is_set(),
            "return_callback_seen": self.return_callback_seen,
            "native_calls_exact_held_ancestry": self.native_calls,
            "future_capture": self.future_capture,
        }
        future, item = self.held_future, self.held_work_item
        current = self._future_bindings_current()
        current = (
            current
            and type(item) is self.work_type
            and type(future) is self.future_type
            and vars(item).get("future") is future
        )
        facts["future_source_current"] = current
        if not current:
            return facts
        # Invoke captured defining functions, never an instance override/proxy API.
        for name in ("done", "running", "cancelled"):
            if not self._future_bindings_current():
                facts["future_source_current"] = False
                return facts
            facts["future_" + name] = self.future_methods[name](future)
        if facts["future_running"]:
            facts["held_entry_wait"] = self._held_entry_wait()
        if facts["future_done"] and not facts["future_cancelled"]:
            if not self._future_bindings_current():
                facts["future_source_current"] = False
                return facts
            error = self.future_methods["exception"](future, timeout=0)
            facts["future_exception_type"] = (
                None if error is None else type(error).__name__[:80]
            )
            facts["first_original_tldw_site"] = None
            facts["first_gate_site"] = None
            if error is not None:
                traceback = error.__traceback__
                callbacks = tuple(
                    callback.__func__.__code__ for callback in self.callbacks.values()
                )
                for _ in range(32):
                    if traceback is None:
                        break
                    frame = traceback.tb_frame
                    # The exact reader is the first project frame under the
                    # original WorkItem/contextvars invocation. Do not label an
                    # unknown deeper code object as an original source site.
                    if (
                        facts["first_original_tldw_site"] is None
                        and frame.f_code is self.reader_code
                        and frame.f_globals is self.reader_globals
                    ):
                        facts["first_original_tldw_site"] = [
                            self.reader.__module__,
                            self.reader_code.co_name,
                            traceback.tb_lineno,
                        ]
                    if (
                        facts["first_gate_site"] is None
                        and frame.f_code in callbacks
                        and frame.f_locals.get("self") is self
                    ):
                        facts["first_gate_site"] = [
                            type(self).__module__,
                            frame.f_code.co_name,
                            traceback.tb_lineno,
                        ]
                    traceback = traceback.tb_next
        facts["future_source_current"] = (
            self._future_bindings_current() and vars(item).get("future") is future
        )
        return facts

    def start(self):
        self.tool = next(
            slot for slot in range(6) if self.monitor.get_tool(slot) is None
        )
        self.monitor.use_tool_id(self.tool, "tldw-original-tab-read-barrier")
        assert self.monitor.get_events(self.tool) == 0
        for event, callback in (
            (self.monitor.events.PY_START, self._start),
            (self.monitor.events.PY_RETURN, self._return),
        ):
            assert self.monitor.register_callback(self.tool, event, callback) is None
            self.callbacks[event] = callback
        self.active = True
        for code, mask in self.masks.items():
            assert self.monitor.get_local_events(self.tool, code) == 0
            self.monitor.set_local_events(self.tool, code, mask)

    def stop(self):
        assert self.monitor.get_events(self.tool) == 0
        for code, mask in self.masks.items():
            assert self.monitor.get_local_events(self.tool, code) == mask
            self.monitor.set_local_events(self.tool, code, 0)
        for event, callback in self.callbacks.items():
            assert self.monitor.register_callback(self.tool, event, None) is callback
        self.monitor.free_tool_id(self.tool)
        self.active = False
        self.restored = self.monitor.get_tool(self.tool) is None
        assert self.projection.read_current is self.reader
        assert (
            self.reader.__code__ is self.reader_code
            and self.reader.__globals__ is self.reader_globals
        )
        assert (
            self.reader.__defaults__ is self.reader_defaults
            and self.reader.__kwdefaults__ is self.reader_kwdefaults
        )
        assert self.reader.__closure__ is self.reader_closure
        assert all(cell.cell_contents is value for cell, value in self.closures)
        for (
            owner,
            name,
            descriptor,
            function,
            code,
            namespace,
            defaults,
            kwdefaults,
            closure,
        ) in self.bindings:
            assert inspect.getattr_static(owner, name) is descriptor
            assert function.__code__ is code and function.__globals__ is namespace
            assert (
                function.__defaults__ is defaults
                and function.__kwdefaults__ is kwdefaults
            )
            assert function.__closure__ is closure
        assert all(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for module, path, digest in self.modules
        )


@pytest.mark.ui
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@pytest.mark.parametrize("phase", ["after_close", "coalesced"])
@private_profile_test
async def test_actual_close_publishes_real_tabs_before_stock_read_release(
    request, phase
):
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    app = _ready_app()
    host = ProductionConsoleHarness(app)
    gate, held_sync = None, None
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        keeper = controller.store.active_session_id
        doomed = controller.new_session(title="Read barrier Close")
        controller.switch_session(keeper)
        await _show_tabs(console, pilot, {keeper, doomed.id})
        projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
        assert type(projection) is spend.ConsoleReadinessConfigProjection
        assert projection.max_age == 1.0
        assert await _settle(
            pilot,
            lambda: not projection.pending
            and not console._console_sync_in_progress
            and not getattr(console, "_console_control_bar_replay_whole_sync", False),
        )
        if phase == "after_close":
            controller.switch_session(doomed.id)
            await _show_tabs(console, pilot, {keeper, doomed.id})
            assert await projection.warm()
            assert projection.key[5] == doomed.id
            assert await _settle(
                pilot,
                lambda: not projection.pending
                and not console._console_sync_in_progress,
            )
        gate = _OriginalReadGate(projection, controller.store, doomed.id, phase)
        gate.start()
        try:
            if phase == "coalesced":
                controller.switch_session(
                    doomed.id
                )  # A real owner transition makes the stock projection cold.
                held_sync = asyncio.create_task(console._sync_native_console_chat_ui())
                assert await _settle(
                    pilot, gate.entered.is_set
                ), "original reader was never admitted"
                assert console._console_sync_in_progress
                assert doomed.id in _session_ids(controller.store)
            await _click(pilot, f"#console-close-session-tab-{doomed.id}")
            assert await _settle(
                pilot, lambda: doomed.id not in _session_ids(controller.store)
            )
            assert await _settle(
                pilot, gate.entered.is_set
            ), "no held original post-Close reader"
            assert not gate.release.is_set()
            assert gate.reader_actor is not threading.current_thread()
            await _await_tabs(console, pilot, {keeper})
            assert _open_tab_ids(console) == {keeper}
            assert (
                not gate.completed_checked_read
            ), "tab assertion happened after reader completion"
        finally:
            gate.release.set()
            try:
                reader_returned = await _settle(pilot, gate.reader_returned.is_set)
                assert reader_returned, (
                    "released original held reader never returned; "
                    + json.dumps(gate.reader_diagnostic(), sort_keys=True)
                )
                assert (
                    gate.native_calls > 0
                ), "released original reader never reached the native path boundary"
                assert (
                    gate.completed_checked_read
                ), "released original reader lacked exact checked source proof"
                if held_sync is not None:
                    await asyncio.wait_for(held_sync, 8)
            finally:
                gate.stop()
        assert gate.restored
