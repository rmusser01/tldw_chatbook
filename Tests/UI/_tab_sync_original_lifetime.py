# Exact stock owner types are required for passive attribution.
# ruff: noqa: E721
"""Owned await-chain revision; new supported-event qualification is required.

Construct after the original tab-close module and ChatScreen are imported.
Start before the three original journey nodes; stop in unconditional finally
and write receipt even when an original assertion fails. No original callable,
deadline, app configuration, worker or widget is replaced. Session IDs and
geometry are metadata; message/draft/title/config bodies are never recorded.
All selected codes use local START/RETURN; the exact original sync code also
uses supported local YIELD/RESUME. Global monitoring remains zero.
"""

import inspect
import sys
import threading
import time
from types import (
    AsyncGeneratorType,
    BuiltinFunctionType,
    CodeType,
    CoroutineType,
    FunctionType,
    GeneratorType,
)


class OriginalTabSyncLifetime:
    def __init__(self):
        screen_module = sys.modules["tldw_chatbook.UI.Screens.chat_screen"]
        surface_module = sys.modules[
            "tldw_chatbook.Widgets.Console.console_session_surface"
        ]
        spend = sys.modules["tldw_chatbook.UI.Console_Modules.console_spend_projection"]
        tests = sys.modules["Tests.UI.test_console_session_tab_close"]
        self.screen_cls = inspect.getattr_static(screen_module, "ChatScreen")
        self.surface_cls = inspect.getattr_static(
            surface_module, "ConsoleSessionSurface"
        )
        self.modules = tuple(
            (module.__name__, module)
            for module in (screen_module, surface_module, spend, tests)
        )
        self.bindings = []
        self.codes = {}

        def select(owner, name, label=None):
            function = inspect.getattr_static(owner, name)
            assert type(function) is FunctionType
            self.bindings.append((owner, name, function))
            self.codes[function.__code__] = label or name
            return function

        for name in (
            "_sync_native_console_chat_ui",
            "_sync_console_native_session_tabs",
            "_sync_console_rail_and_controls",
            "_request_console_control_bar_sync",
            "_run_coalesced_control_bar_sync",
        ):
            select(self.screen_cls, name)
        select(self.surface_cls, "sync_sessions", "surface_sync_sessions")
        select(self.surface_cls, "_session_strip_is_attached", "surface_parent_check")
        select(spend, "run_console_config_sync", "finite_config_sync")
        select(spend, "_run_checked_display_sync", "checked_display_sync")
        self.await_tabs = select(tests, "_await_tabs", "original_await_tabs")
        self.settle = select(tests, "_settle", "original_tab_wait_settle")

        # Exactly the known stock presentation decorator, with no generic
        # unwrapping protocol or replaced producer qualification.
        factory = inspect.getattr_static(spend, "console_readiness_presentation")
        decorated = select(
            self.screen_cls, "_run_console_config_sync", "config_sync_wrapper"
        )
        assert type(factory) is FunctionType
        self.bindings.append((spend, "console_readiness_presentation", factory))
        wrapper_codes = tuple(
            item
            for item in factory.__code__.co_consts
            if type(item) is CodeType and item.co_name == "wrapped"
        )
        assert len(wrapper_codes) == 1 and decorated.__code__ is wrapper_codes[0]
        assert decorated.__globals__ is spend.__dict__
        assert decorated.__code__.co_freevars == ("function",)
        assert decorated.__closure__ is not None and len(decorated.__closure__) == 1
        cell = decorated.__closure__[0]
        body = cell.cell_contents
        assert type(body) is FunctionType
        assert inspect.getattr_static(decorated, "__wrapped__") is body
        assert body.__globals__ is screen_module.__dict__
        assert body.__code__.co_qualname == "ChatScreen._run_console_config_sync"
        self.decorated, self.decorated_closure, self.decorated_cell = (
            decorated,
            decorated.__closure__,
            cell,
        )
        self.config_body, self.factory = body, factory
        self.codes[body.__code__] = "config_sync_body"
        self.bodies = tuple(
            (function, function.__code__, function.__globals__)
            for function in (*(item[2] for item in self.bindings), body)
        )
        self.lock = threading.Lock()
        self.frames, self.events = {}, []
        self.references = [self.screen_cls, self.surface_cls, body, cell]
        self.overflow = 0
        self.monitor = sys.monitoring
        self.mask = self.monitor.events.PY_START | self.monitor.events.PY_RETURN
        self.tool, self.active, self.restoration = None, False, None
        self.installed = []
        self.callbacks = (self._start, self._return, self._yield, self._resume)
        self.events_supported = (
            self.monitor.events.PY_START,
            self.monitor.events.PY_RETURN,
            self.monitor.events.PY_YIELD,
            self.monitor.events.PY_RESUME,
        )
        self.sync_code = inspect.getattr_static(
            self.screen_cls, "_sync_native_console_chat_ui"
        ).__code__
        self.sync_mask = (
            self.mask | self.monitor.events.PY_YIELD | self.monitor.events.PY_RESUME
        )
        self.sync_spans = {}
        asyncio = sys.modules["asyncio"]
        tasks = sys.modules["asyncio.tasks"]
        native_tasks = sys.modules["_asyncio"]
        self.current_task = inspect.getattr_static(tasks, "current_task")
        assert type(self.current_task) is BuiltinFunctionType
        assert inspect.getattr_static(native_tasks, "current_task") is self.current_task
        assert inspect.getattr_static(asyncio, "current_task") is self.current_task
        self.task_cls = inspect.getattr_static(native_tasks, "Task")
        assert inspect.getattr_static(asyncio, "Task") is self.task_cls
        self.task_descriptors = {
            name: inspect.getattr_static(self.task_cls, name)
            for name in ("get_coro", "get_loop", "done", "cancelled", "cancelling")
        }
        self.task_modules = tuple(
            (module.__name__, module) for module in (asyncio, tasks, native_tasks)
        )
        self.task_aliases = (
            (asyncio, "current_task", self.current_task),
            (tasks, "current_task", self.current_task),
            (native_tasks, "current_task", self.current_task),
            (asyncio, "Task", self.task_cls),
            (native_tasks, "Task", self.task_cls),
        )
        self.tool_name = "tldw-original-tab-owned-await-" + str(id(self))

    def _task_call(self, task, name):
        assert type(task) is self.task_cls
        return self.task_descriptors[name].__get__(task, self.task_cls)()

    def _capture_sync_task(self, frame, screen):
        try:
            task = self.current_task()
        except RuntimeError:
            task = None
        root = (
            self._task_call(task, "get_coro") if type(task) is self.task_cls else None
        )
        self.references.extend((task, root, frame, screen, threading.current_thread()))
        return {
            "task": task,
            "root": root,
            "frame": frame,
            "screen": screen,
            "actor": threading.current_thread(),
            "last_local_event": "PY_START",
            "last_source_line": frame.f_lineno,
        }

    def _await_chain(self, root, target_frame):
        chain, references, seen = [], [], set()
        current, matched = root, False
        while current is not None and len(chain) < 32:
            if id(current) in seen:
                chain.append({"cycle": True})
                return chain, matched, True
            seen.add(id(current))
            references.append(current)
            actual_type = type(current)
            if actual_type is CoroutineType:
                frame, child = current.cr_frame, current.cr_await
                state = (
                    "running"
                    if current.cr_running
                    else "suspended"
                    if current.cr_suspended
                    else "closed"
                    if frame is None
                    else "created"
                )
                kind = "CPython coroutine"
            elif actual_type is GeneratorType:
                frame, child = current.gi_frame, current.gi_yieldfrom
                state = (
                    "running"
                    if current.gi_running
                    else "suspended"
                    if current.gi_suspended
                    else "closed"
                    if frame is None
                    else "created"
                )
                kind = "CPython generator"
            elif actual_type is AsyncGeneratorType:
                frame, child = current.ag_frame, current.ag_await
                state = (
                    "running"
                    if current.ag_running
                    else "closed"
                    if frame is None
                    else "not_running"
                )
                kind = "CPython async generator"
            else:
                # An opaque await iterator exposes no qualified frame. Do not
                # unwrap it, inspect payloads, or invent its source/health.
                chain.append(
                    {
                        "object_id": id(current),
                        "kind": "opaque await leaf",
                        "type_module": actual_type.__module__,
                        "type_name": actual_type.__qualname__,
                    }
                )
                break
            row = {"object_id": id(current), "kind": kind, "state": state}
            if frame is None:
                row["frame"] = None
            else:
                references.append(frame)
                code = frame.f_code
                exact_match = frame is target_frame
                matched |= exact_match
                row["frame"] = {
                    "object_id": id(frame),
                    "code_object_id": id(code),
                    "code_name": code.co_qualname,
                    "source": code.co_filename,
                    "definition_line": code.co_firstlineno,
                    "current_line": frame.f_lineno,
                    "instruction_offset": frame.f_lasti,
                    "is_exact_started_original_sync_frame": exact_match,
                }
            chain.append(row)
            current = child
        self.references.extend(references)
        return chain, matched, current is not None and len(chain) >= 32

    def _deadline_sync_ownership(self, screen):
        try:
            deadline_task = self.current_task()
        except RuntimeError:
            deadline_task = None
        self.references.append(deadline_task)
        exact_deadline_task = type(deadline_task) is self.task_cls
        records = []
        with self.lock:
            states = tuple(self.frames.values())
        for kind, frame, owner, _await_frame in states:
            if kind != "_sync_native_console_chat_ui" or owner is not screen:
                continue
            span = self.sync_spans[id(frame)]
            task, root = span["task"], span["root"]
            exact_task = type(task) is self.task_cls
            record = {
                "original_frame_object_id": id(frame),
                "original_code_object_id": id(frame.f_code),
                "original_current_line": frame.f_lineno,
                "original_instruction_offset": frame.f_lasti,
                "actual_task_object_id": id(task) if task is not None else None,
                "exact_CPython_Task": exact_task,
                "last_supported_local_event": span["last_local_event"],
                "last_supported_event_source_line": span["last_source_line"],
                "same_actor_as_deadline": span["actor"] is threading.current_thread(),
            }
            if exact_task:
                still_root = self._task_call(task, "get_coro") is root
                chain, matched, overflow = self._await_chain(root, frame)
                record.update(
                    done=self._task_call(task, "done"),
                    cancelled=self._task_call(task, "cancelled"),
                    cancelling=self._task_call(task, "cancelling"),
                    root_coroutine_object_id=id(root),
                    root_still_exact=still_root,
                    exact_original_frame_in_owned_await_chain=matched,
                    owned_await_chain=chain,
                    await_chain_overflow=overflow,
                    loop_matches_deadline_task=(
                        self._task_call(task, "get_loop")
                        is self._task_call(deadline_task, "get_loop")
                        if exact_deadline_task
                        else None
                    ),
                )
                if overflow:
                    self.overflow += 1
            records.append(record)
        return {
            "deadline_actual_task_object_id": id(deadline_task)
            if deadline_task is not None
            else None,
            "deadline_exact_CPython_Task": exact_deadline_task,
            "only_strong_started_original_sync_spans_for_exact_screen": True,
            "no_global_task_or_thread_frame_enumeration": True,
            "unreturned_original_sync_spans": records,
        }

    def _yield(self, code, _offset, _value):
        self._suspension_event(code, "PY_YIELD")

    def _resume(self, code, _offset):
        self._suspension_event(code, "PY_RESUME")

    def _suspension_event(self, code, event):
        if not self.active:
            return
        frame = sys._getframe(2)
        assert frame.f_code is code and code is self.sync_code
        span = self.sync_spans.get(id(frame))
        if span is None:
            return
        assert span["frame"] is frame
        span["last_local_event"] = event
        span["last_source_line"] = frame.f_lineno

    def _screen_for(self, frame):
        current = frame
        while current is not None:
            for name in ("self", "screen", "console"):
                owner = current.f_locals.get(name)
                if type(owner) is self.screen_cls:
                    return owner
                if type(owner) is self.surface_cls:
                    parent = owner
                    for _ in range(16):
                        parent = getattr(parent, "_parent", None)
                        if type(parent) is self.screen_cls:
                            return parent
                        if parent is None:
                            break
            current = current.f_back
        return None

    def _await_frame(self, frame):
        current = frame.f_back
        while current is not None:
            if current.f_code is self.await_tabs.__code__:
                return current
            current = current.f_back
        return None

    @staticmethod
    def _flags(owner):
        if owner is None:
            return None
        return {
            "object_id": id(owner),
            "attached": bool(owner.is_attached),
            "closing": bool(owner._closing),
            "closed": bool(owner._closed),
            "pruning": bool(owner._pruning),
        }

    def _snapshot(self, screen):
        store = getattr(screen, "_console_chat_store", None)
        sessions = getattr(store, "_sessions", None)
        self.references.extend((screen, store))
        return {
            "screen_object_id": id(screen),
            "screen_flags": self._flags(screen),
            "sync_in_progress": bool(screen._console_sync_in_progress),
            "sync_requested": bool(screen._console_sync_requested),
            "whole_sync_replay": bool(
                getattr(screen, "_console_control_bar_replay_whole_sync", False)
            ),
            "control_retry_scheduled": bool(
                getattr(screen, "_console_control_bar_sync_scheduled", False)
            ),
            "maintenance_paused": bool(
                getattr(screen, "_console_sync_maintenance_paused", False)
            ),
            "store_object_id": id(store) if store is not None else None,
            "live_store_session_ids": sorted(sessions)
            if type(sessions) is dict
            else None,
        }

    def _wait_snapshot(self, await_frame):
        screen = await_frame.f_locals["console"]
        expected = await_frame.f_locals["expected"]
        assert type(screen) is self.screen_cls and type(expected) is set
        surface = screen.query_one("#console-session-surface", self.surface_cls)
        strip = surface.query_one("#console-native-tab-strip")
        children = list(strip.children)
        self.references.extend((surface, strip, *children))
        close_regions = {}
        for child in children:
            child_id = child.id or ""
            if child_id.startswith("console-close-session-tab-"):
                region = child.region
                close_regions[child_id.removeprefix("console-close-session-tab-")] = {
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height,
                }
        return {
            "expected_session_ids": sorted(expected),
            "surface_flags": self._flags(surface),
            "strip_flags": self._flags(strip),
            "actual_tab_ids": [
                child.id.removeprefix("console-session-tab-")
                for child in children
                if (child.id or "").startswith("console-session-tab-")
            ],
            "close_regions": close_regions,
        }

    def _record(self, kind, phase, frame, screen, value=None, await_frame=None):
        row = {
            "kind": kind,
            "phase": phase,
            "at": time.monotonic(),
            "source_line": frame.f_lineno,
            "returned": value if value is None or type(value) is bool else "non-bool",
            **self._snapshot(screen),
        }
        if kind in {"surface_sync_sessions", "surface_parent_check"}:
            surface = frame.f_locals["self"]
            assert type(surface) is self.surface_cls
            row["surface_flags"] = self._flags(surface)
            strip = frame.f_locals.get("tab_strip")
            row["captured_strip_flags"] = self._flags(strip)
            sessions = frame.f_locals.get("sessions")
            if type(sessions) is list:
                row["captured_session_ids"] = [session.id for session in sessions]
            self.references.extend((surface, strip))
        if kind == "finite_config_sync":
            for name in ("entered", "maintenance_paused", "displayed"):
                captured = frame.f_locals.get(name)
                row[name] = (
                    captured
                    if captured is None or type(captured) is bool
                    else "non-bool"
                )
            failure = frame.f_locals.get("failure")
            row["body_failure_type"] = (
                type(failure).__name__ if failure is not None else None
            )
        if await_frame is not None and phase == "return":
            row["original_wait_timeout"] = frame.f_locals["timeout"]
            row["original_wait_deadline"] = frame.f_locals["deadline"]
            row["deadline_sync_ownership"] = self._deadline_sync_ownership(screen)
            try:
                row["deadline_dom"] = self._wait_snapshot(await_frame)
            except Exception as error:
                row["deadline_dom_error_type"] = type(error).__name__
        with self.lock:
            if len(self.events) >= 8192:
                self.overflow += 1
                return
            self.events.append(row)

    def _start(self, code, _offset):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        kind = self.codes[code]
        if (
            kind == "config_sync_wrapper"
            and frame.f_locals.get("function") is not self.config_body
        ):
            return
        await_frame = (
            self._await_frame(frame) if kind == "original_tab_wait_settle" else None
        )
        if kind == "original_tab_wait_settle" and await_frame is None:
            return
        screen = self._screen_for(frame)
        if screen is None:
            return
        with self.lock:
            if len(self.frames) >= 256:
                self.overflow += 1
                return
            self.frames[id(frame)] = (kind, frame, screen, await_frame)
        if kind == "_sync_native_console_chat_ui":
            self.sync_spans[id(frame)] = self._capture_sync_task(frame, screen)
        self._record(kind, "start", frame, screen)

    def _return(self, code, _offset, value):
        frame = sys._getframe(1)
        assert frame.f_code is code
        with self.lock:
            state = self.frames.pop(id(frame), None)
        if state is None:
            return
        kind, actual_frame, screen, await_frame = state
        assert actual_frame is frame
        self._record(kind, "return", frame, screen, value, await_frame)

    def start(self):
        self.tool = next(
            (number for number in (5, 4, 3) if self.monitor.get_tool(number) is None),
            None,
        )
        assert self.tool is not None
        self.monitor.use_tool_id(self.tool, self.tool_name)
        assert self.monitor.get_events(self.tool) == 0
        for event, callback in zip(
            self.events_supported,
            self.callbacks,
            strict=True,
        ):
            assert self.monitor.register_callback(self.tool, event, callback) is None
        self.active = True
        try:
            for code in self.codes:
                assert self.monitor.get_local_events(self.tool, code) == 0
                self.monitor.set_local_events(
                    self.tool,
                    code,
                    self.sync_mask if code is self.sync_code else self.mask,
                )
                self.installed.append(code)
        except BaseException:
            self.stop()
            raise

    def stop(self):
        assert self.monitor.get_events(self.tool) == 0
        for code in self.installed:
            assert self.monitor.get_local_events(self.tool, code) == (
                self.sync_mask if code is self.sync_code else self.mask
            )
            self.monitor.set_local_events(self.tool, code, 0)
        for event, callback in zip(
            self.events_supported,
            self.callbacks,
            strict=True,
        ):
            assert self.monitor.register_callback(self.tool, event, None) is callback
        self.active = False
        self.monitor.free_tool_id(self.tool)
        assert self.monitor.get_tool(self.tool) is None
        self.restoration = "selected_local_callbacks_removed_tool_freed_global0"

    def receipt(self):
        return {
            "diagnostic_only": True,
            "events": list(self.events),
            "overflow": self.overflow,
            "live_original_frames": [state[0] for state in self.frames.values()],
            "exceptional_spans_retained_never_invented_as_returns": True,
            "restoration": self.restoration,
            "selected_code_count": len(self.codes),
            "global_events": 0,
            "bindings_and_bodies_unchanged": all(
                inspect.getattr_static(owner, name) is function
                for owner, name, function in self.bindings
            )
            and all(
                function.__code__ is code and function.__globals__ is namespace
                for function, code, namespace in self.bodies
            )
            and self.decorated.__closure__ is self.decorated_closure
            and self.decorated_cell.cell_contents is self.config_body
            and inspect.getattr_static(self.decorated, "__wrapped__")
            is self.config_body
            and all(sys.modules.get(name) is module for name, module in self.modules),
            "exact_task_aliases_and_metadata_descriptors_unchanged": all(
                inspect.getattr_static(owner, name) is captured
                for owner, name, captured in self.task_aliases
            )
            and all(
                inspect.getattr_static(self.task_cls, name) is descriptor
                for name, descriptor in self.task_descriptors.items()
            )
            and all(
                sys.modules.get(name) is module for name, module in self.task_modules
            ),
            "local_sync_events": ["PY_START", "PY_RETURN", "PY_YIELD", "PY_RESUME"],
            "other_selected_local_events": ["PY_START", "PY_RETURN"],
            "no_global_task_or_thread_frame_enumeration": True,
            "no_callable_guard_wait_deadline_or_cleanup_replacement": True,
            "AST_only_native_App_qualification_pending": True,
        }
