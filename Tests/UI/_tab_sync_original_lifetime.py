# Exact stock owner types are required for passive attribution.
# ruff: noqa: E721
# Exact stock owner types are required for passive attribution.
# ruff: noqa: E721
"""Shield-owned child revision; new supported-event qualification is required.

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


class _OriginalTabShieldChild:
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
        # One defining character method and its literal nested observe body.
        # No generic callable unwrapping or task enumeration is permitted.
        character = sys.modules["tldw_chatbook.UI.Console_Modules.character_context"]
        self.character_module = character
        self.character_cls = inspect.getattr_static(
            character, "ConsoleCharacterContextController"
        )
        method = inspect.getattr_static(
            self.character_cls, "refresh_presentation_if_scope_changed"
        )
        assert type(method) is FunctionType
        assert method.__globals__ is character.__dict__
        self.character_method = method
        self.character_code = method.__code__
        assert self.character_code.co_qualname == (
            "ConsoleCharacterContextController.refresh_presentation_if_scope_changed"
        )
        observe_codes = tuple(
            code
            for code in self.character_code.co_consts
            if type(code) is CodeType and code.co_name == "observe"
        )
        assert len(observe_codes) == 1
        self.character_observe_code = observe_codes[0]
        assert self.character_observe_code.co_qualname == (
            self.character_code.co_qualname + ".<locals>.observe"
        )
        self.character_namespace = character.__dict__
        self.bindings.append(
            (self.character_cls, "refresh_presentation_if_scope_changed", method)
        )
        self.modules += ((character.__name__, character),)
        self.codes[self.character_observe_code] = "character_owned_observe"
        self.observe_spans = {}
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

    def _await_chain(self, root, target_frame, *, depth=0, expected_screen=None):
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
            if frame is not None and frame.f_code is self.character_code:
                row["defining_character_owned_observation"] = (
                    self._character_owned_observation(
                        frame, depth=depth, expected_screen=expected_screen
                    )
                )
            if frame is not None and frame.f_code is getattr(
                self, "worker_async_code", None
            ):
                row["finite_database_worker_stage"] = self._worker_stage(
                    frame, expected_screen
                )
            chain.append(row)
            current = child
        self.references.extend(references)
        return chain, matched, current is not None and len(chain) >= 32

    def _capture_observe(self, frame):
        controller = frame.f_locals.get("self")
        screen = frame.f_locals.get("screen")
        task = self.current_task()
        if (
            type(controller) is not self.character_cls
            or type(screen) is not self.screen_cls
            or type(task) is not self.task_cls
        ):
            return
        root = self._task_call(task, "get_coro")
        if (
            type(root) is not CoroutineType
            or root.cr_code is not self.character_observe_code
            or root.cr_frame is not frame
        ):
            return
        if len(self.observe_spans) >= 256:
            self.overflow += 1
            return
        self.observe_spans[id(frame)] = {
            "frame": frame,
            "controller": controller,
            "screen": screen,
            "task": task,
            "root": root,
            "actor": threading.current_thread(),
            "normal_return": False,
        }
        self.references.extend((frame, controller, screen, task, root))

    def _character_owned_observation(self, frame, *, depth, expected_screen):
        record = {
            "qualified": False,
            "only_exact_defining_method_local_owned_Task": True,
        }

        def refuse(reason):
            record["refusal"] = reason
            return record

        if depth >= 4:
            self.overflow += 1
            return refuse("bounded_child_chain_depth")
        if (
            frame.f_code is not self.character_code
            or sys.modules.get(self.character_module.__name__)
            is not self.character_module
            or inspect.getattr_static(
                self.character_module, "ConsoleCharacterContextController"
            )
            is not self.character_cls
            or self.character_method.__code__ is not self.character_code
            or self.character_method.__globals__ is not self.character_namespace
            or inspect.getattr_static(
                self.character_cls, "refresh_presentation_if_scope_changed"
            )
            is not self.character_method
        ):
            return refuse("defining_method_or_body_changed")
        controller, screen, observe, task = (
            frame.f_locals.get(name) for name in ("self", "screen", "observe", "owned")
        )
        if (
            type(controller) is not self.character_cls
            or type(screen) is not self.screen_cls
        ):
            return refuse("foreign_controller_or_screen")
        if screen is not expected_screen:
            return refuse("character_frame_screen_not_owned_sync_screen")
        if (
            type(observe) is not FunctionType
            or observe.__code__ is not self.character_observe_code
            or observe.__globals__ is not self.character_namespace
            or observe.__closure__ is None
        ):
            return refuse("local_observe_not_defining_body")
        cells = dict(
            zip(observe.__code__.co_freevars, observe.__closure__, strict=True)
        )
        if (
            "self" not in cells
            or cells["self"].cell_contents is not controller
            or "screen" not in cells
            or cells["screen"].cell_contents is not screen
        ):
            return refuse("local_observe_owner_closure_changed")
        if type(task) is not self.task_cls:
            return refuse("local_owned_not_exact_CPython_Task")
        child_root = self._task_call(task, "get_coro")
        if (
            type(child_root) is not CoroutineType
            or child_root.cr_code is not self.character_observe_code
        ):
            return refuse("owned_coroutine_not_original_observe")
        candidates = [
            span
            for span in self.observe_spans.values()
            if span["task"] is task and span["root"] is child_root
        ]
        if len(candidates) != 1:
            return refuse("original_observe_start_not_covered")
        span = candidates[0]
        if span["controller"] is not controller or span["screen"] is not screen:
            return refuse("original_observe_started_for_other_owner")
        deadline_task = self.current_task()
        if type(deadline_task) is not self.task_cls:
            return refuse("deadline_not_exact_CPython_Task")
        same_actor = span["actor"] is threading.current_thread()
        same_loop = self._task_call(task, "get_loop") is self._task_call(
            deadline_task, "get_loop"
        )
        if not same_actor or not same_loop:
            return refuse("original_observe_actor_or_loop_changed")
        chain, matched, overflow = self._await_chain(
            child_root, span["frame"], depth=depth + 1, expected_screen=screen
        )
        self.references.extend(
            (observe, *cells.values(), task, child_root, deadline_task)
        )
        record.update(
            qualified=True,
            actual_owned_task_object_id=id(task),
            exact_CPython_Task=True,
            owned_root_coroutine_object_id=id(child_root),
            original_observe_frame_object_id=id(span["frame"]),
            original_observe_code_object_id=id(self.character_observe_code),
            original_observe_same_actor_as_deadline=same_actor,
            owned_loop_matches_deadline_task=same_loop,
            original_observe_normal_return=span["normal_return"],
            done=self._task_call(task, "done"),
            cancelled=self._task_call(task, "cancelled"),
            cancelling=self._task_call(task, "cancelling"),
            exact_original_observe_frame_in_owned_chain=matched,
            owned_await_chain=chain,
            await_chain_overflow=overflow,
        )
        if overflow:
            self.overflow += 1
        return record

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
                chain, matched, overflow = self._await_chain(
                    root, frame, expected_screen=screen
                )
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

    def _start(self, code, _offset, *, _observed_frame=None):
        if not self.active:
            return
        frame = sys._getframe(1) if _observed_frame is None else _observed_frame
        assert frame.f_code is code
        kind = self.codes[code]
        if code is self.character_observe_code:
            self._capture_observe(frame)
            return
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

    def _return(self, code, _offset, value, *, _observed_frame=None):
        frame = sys._getframe(1) if _observed_frame is None else _observed_frame
        assert frame.f_code is code
        if code is self.character_observe_code:
            span = self.observe_spans.get(id(frame))
            if span is not None:
                assert span["frame"] is frame
                span["normal_return"] = True
            return
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
            "live_character_observe_frames": sum(
                not span["normal_return"] for span in self.observe_spans.values()
            ),
            "character_shield_child_capture_is_local_defining_code_only": True,
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


class _OriginalTabFiniteWorker(_OriginalTabShieldChild):
    """Only the finite metadata callback owned by the retained observe Task."""

    def __init__(self):
        from dis import get_instructions
        from types import MethodType

        super().__init__()
        self.method_type = MethodType
        base = sys.modules["tldw_chatbook.DB.base_db"]
        notes = sys.modules["tldw_chatbook.DB.ChaChaNotes_DB"]
        participants = sys.modules["tldw_chatbook.Backup_Recovery.participants"]
        contextlib = sys.modules["contextlib"]
        futures = sys.modules["concurrent.futures._base"]
        self.notes_cls = inspect.getattr_static(notes, "CharactersRAGDB")
        self.modules += tuple(
            (module.__name__, module)
            for module in (base, notes, participants, contextlib, futures)
        )

        def anchor(owner, name, *, namespace, label=None):
            function = inspect.getattr_static(owner, name)
            assert type(function) is FunctionType and function.__globals__ is namespace
            self.bindings.append((owner, name, function))
            self.bodies += ((function, function.__code__, function.__globals__),)
            if label:
                self.codes[function.__code__] = label
            return function

        self.run_owned = anchor(
            base,
            "run_owned_db_call",
            namespace=base.__dict__,
            label="metadata_owned_async",
        )
        self.worker_async_code = self.run_owned.__code__
        invoke_codes = tuple(
            code
            for code in self.worker_async_code.co_consts
            if type(code) is CodeType and code.co_name == "invoke"
        )
        assert len(invoke_codes) == 1
        self.invoke_code = invoke_codes[0]
        assert set(self.invoke_code.co_freevars) == {
            "args",
            "kwargs",
            "database",
            "operation",
        }
        self.invoke_namespace = base.__dict__
        self.codes[self.invoke_code] = "metadata_owned_invoke"
        self.pair = anchor(
            self.character_cls,
            "_read_database_scope_metadata_pair",
            namespace=self.character_namespace,
            label="metadata_pair_reader",
        )
        self.pair_code = self.pair.__code__
        metadata_descriptor = inspect.getattr_static(
            self.character_cls, "_read_database_scope_metadata"
        )
        assert type(metadata_descriptor) is staticmethod
        self.metadata_body = metadata_descriptor.__func__
        assert (
            type(self.metadata_body) is FunctionType
            and self.metadata_body.__globals__ is self.character_namespace
        )
        self.bindings.append(
            (self.character_cls, "_read_database_scope_metadata", metadata_descriptor)
        )
        self.bodies += (
            (
                self.metadata_body,
                self.metadata_body.__code__,
                self.metadata_body.__globals__,
            ),
        )

        core = inspect.getattr_static(participants, "_core_operation")
        factory = anchor(contextlib, "contextmanager", namespace=contextlib.__dict__)
        helpers = tuple(
            code
            for code in factory.__code__.co_consts
            if type(code) is CodeType and code.co_name == "helper"
        )
        assert (
            len(helpers) == 1
            and type(core) is FunctionType
            and core.__code__ is helpers[0]
        )
        assert (
            core.__globals__ is contextlib.__dict__
            and core.__code__.co_freevars == ("func",)
        )
        assert core.__closure__ is not None and len(core.__closure__) == 1
        core_body = core.__closure__[0].cell_contents
        assert (
            type(core_body) is FunctionType
            and core_body.__globals__ is participants.__dict__
        )
        assert inspect.getattr_static(core, "__wrapped__") is core_body
        assert core_body.__code__.co_qualname == "_core_operation"
        self.bindings.append((participants, "_core_operation", core))
        self.bodies += (
            (core, core.__code__, core.__globals__),
            (core_body, core_body.__code__, core_body.__globals__),
        )
        self.core_wrapper, self.core_body, self.core_closure = (
            core,
            core_body,
            core.__closure__,
        )
        self.core_code = core_body.__code__
        self.codes[self.core_code] = "metadata_direct_core"
        manager_cls = inspect.getattr_static(contextlib, "_GeneratorContextManager")
        self.core_enter = anchor(
            manager_cls, "__enter__", namespace=contextlib.__dict__
        )
        self.core_yields = frozenset(
            item.offset
            for item in get_instructions(self.core_code)
            if item.opname == "YIELD_VALUE"
        )
        self.getter_codes = set()
        for name in (
            "get_local_authority_id",
            "get_character_conversation_search_revision",
        ):
            getter = anchor(self.notes_cls, name, namespace=notes.__dict__, label=name)
            self.getter_codes.add(getter.__code__)
        self.future_cls = inspect.getattr_static(futures, "Future")
        self.future_readers = {
            name: anchor(self.future_cls, name, namespace=futures.__dict__)
            for name in ("done", "running", "cancelled")
        }
        self.async_spans, self.worker_spans, self.stage_spans = {}, {}, {}
        self.worker_codes = {
            self.worker_async_code,
            self.invoke_code,
            self.pair_code,
            self.core_code,
            *self.getter_codes,
        }

    def _owned_async_start(self, frame):
        database, operation = (
            frame.f_locals.get(name) for name in ("database", "operation")
        )
        if (
            type(operation) is not self.method_type
            or operation.__func__ is not self.pair
            or type(database) is not self.notes_cls
        ):
            return
        task = self.current_task()
        matches = [
            span
            for span in self.observe_spans.values()
            if span["task"] is task
            and span["controller"] is operation.__self__
            and not span["normal_return"]
        ]
        if len(matches) != 1:
            return
        parent = matches[0]
        if len(self.async_spans) >= 256:
            self.overflow += 1
            return
        self.async_spans[id(frame)] = {
            "frame": frame,
            "database": database,
            "operation": operation,
            "args": frame.f_locals["args"],
            "kwargs": frame.f_locals["kwargs"],
            "controller": parent["controller"],
            "screen": parent["screen"],
            "task": task,
            "actor": threading.current_thread(),
            "normal_return": False,
        }

    def _qualified_invoke(self, span):
        invoke = span["frame"].f_locals.get("invoke")
        if (
            type(invoke) is not FunctionType
            or invoke.__code__ is not self.invoke_code
            or invoke.__globals__ is not self.invoke_namespace
            or invoke.__closure__ is None
        ):
            return None
        cells = dict(zip(invoke.__code__.co_freevars, invoke.__closure__, strict=True))
        if all(
            cells[name].cell_contents is span[name]
            for name in ("database", "operation", "args", "kwargs")
        ):
            return invoke
        return None

    def _invoke_start(self, frame):
        matches = [
            span
            for span in self.async_spans.values()
            if all(
                frame.f_locals.get(name) is span[name]
                for name in ("database", "operation", "args", "kwargs")
            )
            and self._qualified_invoke(span) is not None
        ]
        if len(matches) != 1:
            return
        if len(self.worker_spans) >= 256:
            self.overflow += 1
            return
        self.worker_spans[id(frame)] = {
            "frame": frame,
            "async": matches[0],
            "invoke": self._qualified_invoke(matches[0]),
            "actor": threading.current_thread(),
            "normal_return": False,
        }

    def _stage_start(self, code, frame):
        parent = frame.f_back
        if code is self.core_code:
            if parent is None or parent.f_code is not self.core_enter.__code__:
                return
            parent = parent.f_back
        elif code in self.getter_codes:
            if parent is None or parent.f_code is not self.metadata_body.__code__:
                return
            parent = parent.f_back
            if parent is None or parent.f_code is not self.pair_code:
                return
            parent = parent.f_back
        if parent is None or parent.f_code is not self.invoke_code:
            return
        worker = self.worker_spans.get(id(parent))
        if (
            worker is None
            or worker["frame"] is not parent
            or worker["actor"] is not threading.current_thread()
        ):
            return
        span = worker["async"]
        if code is self.core_code:
            valid = frame.f_locals.get("repository") is span["database"]
        elif code is self.pair_code:
            valid = (
                frame.f_locals.get("self") is span["controller"]
                and frame.f_locals.get("database") is span["database"]
            )
        else:
            valid = frame.f_locals.get("self") is span["database"]
        if not valid:
            return
        if len(self.stage_spans) >= 1024:
            self.overflow += 1
            return
        self.stage_spans[id(frame)] = {
            "frame": frame,
            "worker": worker,
            "actor": threading.current_thread(),
            "normal_return": False,
        }

    def _start(self, code, offset):
        if code not in self.worker_codes:
            return super()._start(code, offset, _observed_frame=sys._getframe(1))
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        with self.lock:
            if code is self.worker_async_code:
                self._owned_async_start(frame)
            elif code is self.invoke_code:
                self._invoke_start(frame)
            else:
                self._stage_start(code, frame)

    def _return(self, code, offset, value):
        if code not in self.worker_codes:
            return super()._return(
                code, offset, value, _observed_frame=sys._getframe(1)
            )
        frame = sys._getframe(1)
        assert frame.f_code is code
        with self.lock:
            table = (
                self.async_spans
                if code is self.worker_async_code
                else self.worker_spans
                if code is self.invoke_code
                else self.stage_spans
            )
            span = table.get(id(frame))
            if span is not None:
                assert span["frame"] is frame
                span["normal_return"] = True

    @staticmethod
    def _stage_metadata(span):
        frame, actor = span["frame"], span["actor"]
        return {
            "frame_object_id": id(frame),
            "code_object_id": id(frame.f_code),
            "code_name": frame.f_code.co_qualname,
            "source": frame.f_code.co_filename,
            "definition_line": frame.f_code.co_firstlineno,
            "current_line": frame.f_lineno,
            "instruction_offset": frame.f_lasti,
            "normal_return": span["normal_return"],
            "actual_Thread_object_id": id(actor),
            "thread_ident": actor.ident,
            "thread_alive": actor.is_alive(),
        }

    def _worker_stage(self, frame, screen):
        with self.lock:
            span = self.async_spans.get(id(frame))
            if (
                span is None
                or span["frame"] is not frame
                or span["screen"] is not screen
            ):
                return {
                    "qualified": False,
                    "refusal": "no_exact_observe_owned_async_start",
                }
            invoke = self._qualified_invoke(span)
            if invoke is None:
                return {
                    "qualified": False,
                    "refusal": "invoke_body_or_closure_not_original",
                }
            workers = [
                worker
                for worker in self.worker_spans.values()
                if worker["async"] is span and worker["invoke"] is invoke
            ]
            facts = []
            for worker in workers:
                stages = [
                    stage
                    for stage in self.stage_spans.values()
                    if stage["worker"] is worker
                ]
                rows = []
                for stage in stages:
                    row = self._stage_metadata(stage)
                    if stage["frame"].f_code is self.core_code:
                        row["at_original_core_yield"] = (
                            stage["frame"].f_lasti in self.core_yields
                        )
                    if stage["frame"].f_code is self.pair_code:
                        future = stage["frame"].f_locals.get("ambient_check")
                        row["ambient_check_present"] = future is not None
                        row["ambient_check_exact_concurrent_Future"] = (
                            type(future) is self.future_cls
                        )
                        if type(future) is self.future_cls:
                            self.references.append(future)
                            row["ambient_check_object_id"] = id(future)
                            row["ambient_check_state"] = {
                                name: function.__get__(future, self.future_cls)()
                                for name, function in self.future_readers.items()
                            }
                    rows.append(row)
                facts.append(
                    {
                        **self._stage_metadata(worker),
                        "exact_callback_stage_frames": rows,
                    }
                )
            return {
                "qualified": True,
                "only_exact_observe_Task_finite_callback": True,
                "async_frame_object_id": id(frame),
                "owned_Task_object_id": id(span["task"]),
                "controller_object_id": id(span["controller"]),
                "database_object_id": id(span["database"]),
                "bound_operation_object_id": id(span["operation"]),
                "original_invoke_function_object_id": id(invoke),
                "invoke_start_seen": bool(workers),
                "qualified_invoke_not_started": not workers,
                "matched_invoke_count": len(workers),
                "workers": facts,
                "metadata_only_sequential_deadline_observation": True,
            }

    def receipt(self):
        result = super().receipt()
        result["bindings_and_bodies_unchanged"] &= (
            self.core_wrapper.__closure__ is self.core_closure
            and self.core_closure[0].cell_contents is self.core_body
            and inspect.getattr_static(self.core_wrapper, "__wrapped__")
            is self.core_body
        )
        result.update(
            live_matched_async_frames=sum(
                not span["normal_return"] for span in self.async_spans.values()
            ),
            live_matched_invoke_frames=sum(
                not span["normal_return"] for span in self.worker_spans.values()
            ),
            live_matched_callback_stage_frames=sum(
                not span["normal_return"] for span in self.stage_spans.values()
            ),
            worker_stages_only_exact_observe_Task_callback_and_local_code=True,
            no_global_thread_frame_enumeration=True,
            no_Future_result_exception_wait_or_callback_called_by_observer=True,
        )
        return result


def _transaction_source_shape(code):
    assert type(code) is CodeType
    return (
        code.co_name, code.co_qualname, code.co_firstlineno, code.co_code,
        code.co_flags, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_names, code.co_varnames,
        code.co_freevars, code.co_cellvars,
        tuple(_transaction_source_shape(item) if type(item) is CodeType else item for item in code.co_consts),
    )


class OriginalTabSyncLifetime(_OriginalTabFiniteWorker):
    """Attribute transaction/getter stages only below the known finite worker."""

    def __init__(self):
        import ast
        import hashlib
        from pathlib import Path

        super().__init__()
        notes = sys.modules["tldw_chatbook.DB.ChaChaNotes_DB"]
        participants = sys.modules["tldw_chatbook.Backup_Recovery.participants"]
        self.transaction_manager_cls = inspect.getattr_static(notes, "TransactionContextManager")
        self.transaction_sources = {}
        compiled_sources = {}
        for module in (notes, participants):
            path = Path(module.__dict__["__file__"])
            source = path.read_bytes()
            self.transaction_sources[module.__name__] = (module, path, hashlib.sha256(source).hexdigest())
            compiled_sources[module.__name__] = compile(source, str(path), "exec", dont_inherit=True)

        def source_code(module, qualname):
            code = compiled_sources[module.__name__]
            for name in qualname.split("."):
                matches = [item for item in code.co_consts if type(item) is CodeType and item.co_name == name]
                assert len(matches) == 1
                code = matches[0]
            return code

        def anchor(owner, name, module, label=None):
            function = inspect.getattr_static(owner, name)
            assert type(function) is FunctionType and function.__globals__ is module.__dict__
            assert _transaction_source_shape(function.__code__) == _transaction_source_shape(source_code(module, function.__code__.co_qualname))
            self.bindings.append((owner, name, function))
            self.bodies += ((function, function.__code__, function.__globals__),)
            if label is not None:
                self.codes[function.__code__] = label
            return function

        self.transaction = anchor(self.notes_cls, "transaction", notes, "metadata_transaction_factory")
        self.transaction_enter = anchor(self.transaction_manager_cls, "__enter__", notes, "metadata_transaction_enter")
        self.transaction_body = anchor(self.transaction_manager_cls, "_enter_transaction", notes, "metadata_transaction_body")
        self.connection_public = anchor(self.notes_cls, "get_connection", notes, "metadata_connection_public")
        factory = anchor(participants, "_core_getter", participants)
        wrapper = inspect.getattr_static(self.notes_cls, "_get_thread_connection")
        wrappers = [item for item in factory.__code__.co_consts if type(item) is CodeType and item.co_name == "accessed"]
        assert len(wrappers) == 1 and type(wrapper) is FunctionType
        assert wrapper.__code__ is wrappers[0] and wrapper.__globals__ is participants.__dict__
        assert wrapper.__code__.co_freevars == ("function",)
        assert wrapper.__closure__ is not None and len(wrapper.__closure__) == 1
        body = wrapper.__closure__[0].cell_contents
        assert type(body) is FunctionType and body.__globals__ is notes.__dict__
        assert body.__code__.co_qualname == "CharactersRAGDB._get_thread_connection"
        assert inspect.getattr_static(wrapper, "__wrapped__") is body
        assert _transaction_source_shape(body.__code__) == _transaction_source_shape(source_code(notes, body.__code__.co_qualname))
        self.connection_wrapper, self.connection_body = wrapper, body
        self.connection_closure, self.connection_cell = wrapper.__closure__, wrapper.__closure__[0]
        self.bindings.append((self.notes_cls, "_get_thread_connection", wrapper))
        self.bodies += ((wrapper, wrapper.__code__, wrapper.__globals__), (body, body.__code__, body.__globals__))
        self.codes[wrapper.__code__] = "metadata_connection_stock_wrapper"
        self.codes[body.__code__] = "metadata_connection_stock_body"
        self.transaction_codes = {
            self.transaction.__code__, self.transaction_enter.__code__,
            self.transaction_body.__code__, self.connection_public.__code__,
            wrapper.__code__, body.__code__,
        }
        self.worker_codes.update(self.transaction_codes)
        tree = ast.parse(Path(notes.__dict__["__file__"]).read_bytes())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CharactersRAGDB")
        getter = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_thread_connection")
        self.connection_call_ranges = []
        for node in ast.walk(getter):
            if not isinstance(node, ast.Call):
                continue
            label = None
            if isinstance(node.func, ast.Name) and node.func.id == "connect_private_sqlite":
                label = "private_sqlite_connect"
            elif isinstance(node.func, ast.Name) and node.func.id == "_core_access":
                label = "repository_access_admission"
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "begin_acquisition":
                label = "connection_registry_acquisition"
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "execute" and node.args and isinstance(node.args[0], ast.Constant):
                label = {
                    "SELECT 1": "cached_connection_liveness",
                    "PRAGMA journal_mode=WAL;": "journal_mode_pragma",
                    "PRAGMA synchronous=NORMAL;": "synchronous_pragma",
                    "PRAGMA foreign_keys = ON;": "foreign_keys_pragma",
                }.get(node.args[0].value)
            if label is not None:
                self.connection_call_ranges.append((node.lineno, node.end_lineno, label))
        assert sum(label == "private_sqlite_connect" for _, _, label in self.connection_call_ranges) == 1

    def _stage_start(self, code, frame):
        if code not in self.transaction_codes:
            return super()._stage_start(code, frame)
        parent = frame.f_back
        if parent is None:
            return
        expected = (
            self.getter_codes if code in {self.transaction.__code__, self.transaction_enter.__code__}
            else {self.transaction_enter.__code__} if code is self.transaction_body.__code__
            else {self.transaction_body.__code__} if code is self.connection_public.__code__
            else {self.connection_public.__code__} if code is self.connection_wrapper.__code__
            else {self.connection_wrapper.__code__}
        )
        prior = self.stage_spans.get(id(parent))
        if prior is None or prior["frame"] is not parent or parent.f_code not in expected or prior["normal_return"]:
            return
        worker = prior["worker"]
        if worker["actor"] is not threading.current_thread() or worker["normal_return"]:
            return
        database = worker["async"]["database"]
        manager = None
        if code in {self.transaction_enter.__code__, self.transaction_body.__code__}:
            manager = frame.f_locals.get("self")
            if type(manager) is not self.transaction_manager_cls:
                return
            fields = object.__getattribute__(manager, "__dict__")
            if fields.get("db") is not database:
                return
            if code is self.transaction_body.__code__ and parent.f_locals.get("self") is not manager:
                return
        elif code is self.connection_wrapper.__code__:
            if frame.f_locals.get("repository") is not database or frame.f_locals.get("function") is not self.connection_body:
                return
        elif frame.f_locals.get("self") is not database:
            return
        if len(self.stage_spans) >= 1024:
            self.overflow += 1
            return
        self.stage_spans[id(frame)] = {
            "frame": frame, "worker": worker, "actor": threading.current_thread(),
            "normal_return": False, "transaction_manager": manager,
        }

    def _worker_stage(self, frame, screen):
        result = super()._worker_stage(frame, screen)
        if result.get("qualified"):
            with self.lock:
                for worker in result["workers"]:
                    for row in worker["exact_callback_stage_frames"]:
                        span = self.stage_spans[row["frame_object_id"]]
                        current = span["frame"]
                        if current.f_code in self.transaction_codes:
                            row["only_matched_original_transaction_getter_chain"] = True
                            manager = span.get("transaction_manager")
                            if manager is not None:
                                row["exact_transaction_manager_object_id"] = id(manager)
                            if current.f_code is self.connection_body.__code__:
                                row["original_getter_call_stages_at_current_line"] = [
                                    label for first, last, label in self.connection_call_ranges
                                    if first <= row["current_line"] <= last
                                ]
                result["transaction_getter_source_and_exact_wrapper_qualified"] = True
        return result

    def receipt(self):
        import hashlib

        result = super().receipt()
        result["bindings_and_bodies_unchanged"] &= (
            self.connection_wrapper.__closure__ is self.connection_closure
            and self.connection_closure[0] is self.connection_cell
            and self.connection_cell.cell_contents is self.connection_body
            and inspect.getattr_static(self.connection_wrapper, "__wrapped__") is self.connection_body
        )
        result["transaction_getter_defining_sources_unchanged"] = all(
            sys.modules.get(name) is module and module.__dict__.get("__file__") == str(path)
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for name, (module, path, digest) in self.transaction_sources.items()
        )
        result["transaction_getter_defining_source_hashes"] = {
            name: digest for name, (_, _, digest) in self.transaction_sources.items()
        }
        result["transaction_getter_only_known_worker_exact_database_and_parent_chain"] = True
        return result
