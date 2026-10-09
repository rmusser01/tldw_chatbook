"""Opt-in original-source postcommit timing; no callable replacement or App import."""

# ruff: noqa: E721 -- Exact defining types are diagnostic source qualifications.

import ast
import hashlib
import inspect
import json
import os
import sys
import threading
import time
import weakref
from pathlib import Path
from types import CodeType, FunctionType

import pytest

PROBE = "Tests.Performance.test_console_native_pause_probe"
TARGET = "test_native_console_pause_probe"
CONTROLLER = "tldw_chatbook.Chat.console_chat_controller"
STORE = "tldw_chatbook.Chat.console_chat_store"
EFFECTS = frozenset(
    {
        "identity_publication",
        "durable_owner_publication",
        "staged_input_clearing",
        "workspace_projection",
        "queue_acknowledgement",
        "accepted_hook",
        "prompt_history",
        "preparation_publication",
        "provider_entry",
        "checkpoint_transition",
    }
)
CONTROLLER_METHODS = (
    "_admit_capture_policy",
    "_build_durable_trace_request",
    "resume_durable_postcommit",
    "_run_durable_postcommit_effect",
    "_run_durable_db_call",
    "hook_admission_reason",
    "_stream_assistant_response_inner",
    "_compose_agent_request_providers",
    "_run_maintenance_agent_call",
    "_run_owned_chat_db_operation",
    "_run_owned_chat_worker_call",
)
STORE_METHODS = (
    "reconcile_durable_turn_settings",
    "reconcile_durable_turn_roleplay_context",
    "_project_workspace_membership_after_commit",
)
PUBLICATIONS = ("publish_durable_turn_identity", "publish_durable_turn_owners")


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
            _shape(item) if type(item) is CodeType else item for item in code.co_consts
        ),
    )


def _find(compiled, name):
    for part in name.split("."):
        compiled = next(
            item
            for item in compiled.co_consts
            if type(item) is CodeType and item.co_name == part
        )
    return compiled


class CriticalPostcommitWitness:
    """Bounded scalar lifecycle states; definition-time source references only."""

    def __init__(self, probe=None, body=None, *, assertion_config=None):
        self.probe, self.body, self.assertion_config = probe, body, assertion_config
        self.body_code = body.__code__ if body is not None else None
        self.bindings, self.originals, self.modules, self.source_hashes = {}, [], {}, {}
        self.source_origins = {}
        self.class_bindings, self.wrapper_bodies = [], {}
        self.states, self.rows, self.windows = {}, {}, {}
        self.actors = weakref.WeakKeyDictionary()
        self.actor_sequence = 0
        self.main_thread_id = threading.get_ident()
        self.gap_count, self.gap_samples, self.overflow = 0, [], 0
        self.bound_at_success, self.bind_failure = {}, None
        self.boundary_duplicates = 0
        self.tool = None
        self.tool_retired = False
        self.global_events_at_stop = None
        self.stop_validation_error = None
        self.selected = set()
        self.coverage = {"lazy_modules_bound": False, "original_stage_bound": False}

    def _source_bytes(self, module):
        path = Path(module.__file__).resolve()
        origin = Path(module.__spec__.origin).resolve()
        assert origin == path
        data = path.read_bytes()
        self.source_hashes[module.__name__] = hashlib.sha256(data).hexdigest()
        self.source_origins[module.__name__] = (path, origin)
        self.modules[module.__name__] = module
        return path, data

    def _source(self, module):
        path, data = self._source_bytes(module)
        return compile(data.decode("utf-8"), str(path), "exec", dont_inherit=True)

    def _add(
        self, module, owner, name, function, code, *, label=None, dispatch_body=None
    ):
        assert (
            type(function) is FunctionType and function.__globals__ is module.__dict__
        )
        assert (
            Path(function.__code__.co_filename).resolve()
            == Path(module.__file__).resolve()
        )
        assert _shape(function.__code__) == _shape(code)
        self.originals.append((module, owner, name, function, function.__code__))
        self.bindings[function.__code__] = (
            label or name,
            module.__dict__,
            dispatch_body,
        )
        if self.tool is not None:
            self._select(function.__code__)

    def _bind_loaded(self):
        if self.coverage["lazy_modules_bound"] or self.bind_failure is not None:
            return
        controller, store = sys.modules.get(CONTROLLER), sys.modules.get(STORE)
        if controller is None or store is None:
            return
        if "ConsoleChatController" not in vars(
            controller
        ) or "ConsoleChatStore" not in vars(store):
            return
        try:
            project_root = Path(self.probe.__file__).resolve().parents[2]
            for module in (controller, store):
                assert module.__spec__.name == module.__name__
                assert (
                    Path(module.__spec__.origin).resolve()
                    == Path(module.__file__).resolve()
                )
                assert (
                    Path(module.__file__).resolve().parent
                    == project_root / "tldw_chatbook" / "Chat"
                )
            cc, sc = self._source(controller), self._source(store)
            ctype, stype = controller.ConsoleChatController, store.ConsoleChatStore
            assert type(ctype) is type and type(stype) is type
            self.class_bindings.extend(
                (
                    (controller, "ConsoleChatController", ctype),
                    (store, "ConsoleChatStore", stype),
                )
            )
            for name in CONTROLLER_METHODS:
                function = inspect.getattr_static(ctype, name)
                self._add(
                    controller,
                    ctype,
                    name,
                    function,
                    _find(cc, "ConsoleChatController." + name),
                )
            for name in STORE_METHODS:
                function = inspect.getattr_static(stype, name)
                self._add(
                    store, stype, name, function, _find(sc, "ConsoleChatStore." + name)
                )
            factory = store._fork_session_transition
            assert type(factory) is FunctionType
            assert factory.__globals__ is store.__dict__
            assert (
                Path(factory.__code__.co_filename).resolve()
                == Path(store.__file__).resolve()
            )
            assert _shape(factory.__code__) == _shape(
                _find(sc, "_fork_session_transition")
            )
            self.originals.append(
                (store, store, "_fork_session_transition", factory, factory.__code__)
            )
            for name in PUBLICATIONS:
                wrapper = inspect.getattr_static(stype, name)
                assert (
                    type(wrapper) is FunctionType
                    and wrapper.__globals__ is store.__dict__
                )
                assert wrapper.__code__.co_freevars == ("method",)
                assert (
                    Path(wrapper.__code__.co_filename).resolve()
                    == Path(store.__file__).resolve()
                )
                body = wrapper.__closure__[0].cell_contents
                assert wrapper.__wrapped__ is body and type(body) is FunctionType
                self.wrapper_bodies[wrapper] = body
                assert _shape(wrapper.__code__) == _shape(
                    _find(sc, "_fork_session_transition.transitioned")
                )
                assert body.__globals__ is store.__dict__
                assert (
                    Path(body.__code__.co_filename).resolve()
                    == Path(store.__file__).resolve()
                )
                assert _shape(body.__code__) == _shape(
                    _find(sc, "ConsoleChatStore." + name)
                )
                self.originals.append((store, stype, name, wrapper, wrapper.__code__))
                self.originals.append((store, None, name, body, body.__code__))
                self.bindings[body.__code__] = (name, store.__dict__, None)
                # Both original publications share this wrapper code. Filter by its
                # exact original closure body, rather than measuring other methods.
                existing = self.bindings.get(wrapper.__code__)
                bodies = dict(existing[2]) if existing is not None else {}
                bodies[body] = name
                self.bindings[wrapper.__code__] = (
                    "fork_publication",
                    store.__dict__,
                    bodies,
                )
                self._select(body.__code__)
                self._select(wrapper.__code__)
            self.coverage["lazy_modules_bound"] = True
        except Exception as error:
            self.bind_failure = type(error).__name__

    def _current(self):
        if self.body is not None and (
            self.body.__code__ is not self.body_code
            or self.body.__globals__ is not self.probe.__dict__
        ):
            return False
        for module, name, owner in self.class_bindings:
            if vars(module).get(name) is not owner:
                return False
        for name, module in self.modules.items():
            if sys.modules.get(name) is not module:
                return False
            path, origin = self.source_origins[name]
            try:
                if Path(module.__file__).resolve() != path:
                    return False
                if Path(module.__spec__.origin).resolve() != origin:
                    return False
            except (AttributeError, TypeError, OSError):
                return False
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != self.source_hashes[name]
            ):
                return False
        for module, owner, name, function, code in self.originals:
            if (
                function.__globals__ is not module.__dict__
                or function.__code__ is not code
            ):
                return False
            if (
                owner is not None
                and inspect.getattr_static(owner, name) is not function
            ):
                return False
            if code.co_freevars == ("method",):
                body = function.__closure__[0].cell_contents
                if (
                    function.__wrapped__ is not body
                    or self.wrapper_bodies.get(function) is not body
                ):
                    return False
        return True

    def _select(self, code):
        if self.tool is None or code in self.selected:
            return
        mask = sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN
        if code.co_flags & inspect.CO_COROUTINE:
            mask |= sys.monitoring.events.PY_YIELD | sys.monitoring.events.PY_RESUME
        sys.monitoring.set_local_events(self.tool, code, mask)
        self.selected.add(code)

    def _frame(self, code):
        frame = sys._getframe(1)
        for _ in range(12):
            if frame.f_code is code:
                return frame
            frame = frame.f_back
            if frame is None:
                break
        return None

    def _actor(self):
        thread = threading.current_thread()
        token = self.actors.get(thread)
        if token is None:
            self.actor_sequence += 1
            token = self.actor_sequence
            self.actors[thread] = token
        return token

    def _gap(self, reason, label, phase):
        self.gap_count += 1
        if len(self.gap_samples) < 32:
            self.gap_samples.append({"reason": reason, "label": label, "phase": phase})

    def _phase(self):
        opened = [name for name, window in self.windows.items() if window[1] is None]
        return opened[-1] if opened else None

    def _stage(self, frame):
        name, status = frame.f_locals.get("name"), frame.f_locals.get("status")
        if (name, status) not in {
            ("durable_commit", "succeeded"),
            ("trace_reservation", "entered"),
        }:
            return
        observed = frame.f_locals.get("observed")
        phase = getattr(observed, "phase", None)
        if type(phase) is not str or phase not in {"send_1", "send_2", "send_3"}:
            return
        now = time.perf_counter()
        if name == "durable_commit":
            if phase in self.windows:
                self.boundary_duplicates += 1
                self._gap("duplicate_durable_success", "stage", phase)
                return
            self._bind_loaded()
            self.bound_at_success[phase] = (
                self.coverage["lazy_modules_bound"] and self._current()
            )
            self.windows[phase] = [now, None]
        elif phase in self.windows:
            if self.windows[phase][1] is not None:
                self.boundary_duplicates += 1
                self._gap("duplicate_trace_entry", "stage", phase)
                return
            self.windows[phase][1] = now

    def _relation(self, frame, actor):
        parent = frame.f_back
        for _ in range(64):
            if parent is None:
                break
            prior = self.states.get((actor, parent.f_code, id(parent)))
            if prior is not None and prior["label"] == "resume_durable_postcommit":
                return "original_resume_ancestry"
            parent = parent.f_back
        return (
            "main_unattributed"
            if threading.get_ident() == self.main_thread_id
            else "worker_unattributed"
        )

    def _event(self, kind, code):
        frame = self._frame(code)
        if frame is None:
            self._gap("event_frame_missing", code.co_name, self._phase())
            return
        if self.body is not None and code is self.body.__code__:
            self._bind_loaded()
            return
        if getattr(self, "stage_code", None) is code:
            if kind == "start":
                self._stage(frame)
            return
        binding = self.bindings.get(code)
        if binding is None or frame.f_globals is not binding[1]:
            return
        label, _globals, bodies = binding
        if bodies is not None:
            body = frame.f_locals.get("method")
            label = bodies.get(body) if type(body) is FunctionType else None
            if label is None:
                return
            label += ".fork_scope"
        actor = self._actor()
        key = (actor, code, id(frame))
        now = time.perf_counter()
        if kind == "start":
            prior = self.states.pop(key, None)
            if prior is not None:
                self._gap(
                    "new_start_proves_prior_unknown_exit",
                    prior["label"],
                    prior["phase"],
                )
            phase = self._phase()
            if phase is None:
                return
            if len(self.states) >= 256:
                self.overflow += 1
                return
            effect = (
                frame.f_locals.get("effect_name")
                if label == "_run_durable_postcommit_effect"
                else None
            )
            effect = effect if type(effect) is str and effect in EFFECTS else None
            self.states[key] = {
                "label": label,
                "phase": phase,
                "effect": effect,
                "relation": self._relation(frame, actor),
                "start": now,
                "last": now,
                "running": 0.0,
                "suspended": 0.0,
                "yielded": False,
            }
            return
        state = self.states.get(key)
        if state is None:
            return
        boundary = self.windows[state["phase"]][1]
        effective_now = now if boundary is None else min(now, boundary)
        effective_last = (
            state["last"] if boundary is None else min(state["last"], boundary)
        )
        state["suspended" if state["yielded"] else "running"] += (
            effective_now - effective_last
        )
        state["last"] = now
        if kind == "return":
            self.states.pop(key)
            row_key = (
                state["phase"],
                state["label"],
                state["effect"],
                state["relation"],
            )
            if row_key not in self.rows and len(self.rows) >= 1024:
                self.overflow += 1
                return
            row = self.rows.setdefault(
                row_key,
                {
                    "normal_returns": 0,
                    "window_overlap_wall_seconds": 0.0,
                    "inclusive_running_in_window_seconds": 0.0,
                    "suspended_in_window_seconds": 0.0,
                    "full_call_wall_seconds": 0.0,
                    "max_window_overlap_seconds": 0.0,
                    "first_start": state["start"],
                    "last_return": now,
                },
            )
            wall = effective_now - state["start"]
            row["normal_returns"] += 1
            row["window_overlap_wall_seconds"] += wall
            row["inclusive_running_in_window_seconds"] += state["running"]
            row["suspended_in_window_seconds"] += state["suspended"]
            row["full_call_wall_seconds"] += now - state["start"]
            row["max_window_overlap_seconds"] = max(
                row["max_window_overlap_seconds"], wall
            )
            row["last_return"] = now
        else:
            state["yielded"] = kind == "yield"

    def start(self):
        assert self.tool is None
        if self.probe is not None:
            from _pytest.assertion import rewrite

            loader = self.probe.__spec__.loader
            assert (
                type(loader) is rewrite.AssertionRewritingHook
                and loader.config is self.assertion_config
            )
            assert self.probe.__loader__ is loader
            path = Path(self.probe.__file__).resolve()
            assert (
                path
                == Path(self.probe.__spec__.origin).resolve()
                == Path(self.body.__code__.co_filename).resolve()
            )
            assert loader._rewritten_names[self.probe.__name__].resolve() == path
            path, data = self._source_bytes(self.probe)
            tree = ast.parse(data, filename=str(path))
            rewrite.rewrite_asserts(tree, data, str(path), self.assertion_config)
            compiled = compile(tree, str(path), "exec", dont_inherit=True)
            assert (
                type(self.body) is FunctionType
                and self.body.__globals__ is self.probe.__dict__
            )
            assert _shape(self.body.__code__) == _shape(_find(compiled, TARGET))
            self.body_code = self.body.__code__
            self.stage_code = _find(self.body.__code__, "stage")
            self.coverage["original_stage_bound"] = True
        for tool in range(5, -1, -1):
            if sys.monitoring.get_tool(tool) is None:
                sys.monitoring.use_tool_id(tool, "tldw-critical-postcommit-witness")
                self.tool = tool
                break
        assert self.tool is not None
        try:
            for event, kind in (
                (sys.monitoring.events.PY_START, "start"),
                (sys.monitoring.events.PY_RETURN, "return"),
                (sys.monitoring.events.PY_YIELD, "yield"),
                (sys.monitoring.events.PY_RESUME, "resume"),
            ):
                sys.monitoring.register_callback(
                    self.tool,
                    event,
                    lambda code, *args, kind=kind: self._event(kind, code),
                )
            for code in self.bindings:
                self._select(code)
            if self.body is not None:
                self._select(self.body.__code__)
                self._select(self.stage_code)
            self._bind_loaded()
            assert sys.monitoring.get_events(self.tool) == 0
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.tool is None:
            return
        tool = self.tool
        try:
            for code in self.selected:
                sys.monitoring.set_local_events(tool, code, 0)
            for event in (
                sys.monitoring.events.PY_START,
                sys.monitoring.events.PY_RETURN,
                sys.monitoring.events.PY_YIELD,
                sys.monitoring.events.PY_RESUME,
            ):
                sys.monitoring.register_callback(tool, event, None)
            self.global_events_at_stop = sys.monitoring.get_events(tool)
            assert self.global_events_at_stop == 0, "unexpected global monitoring mask"
        except BaseException as error:
            self.stop_validation_error = type(error).__name__
            raise
        finally:
            try:
                sys.monitoring.set_events(tool, 0)
            finally:
                sys.monitoring.free_tool_id(tool)
                self.tool = None
                self.tool_retired = True

    def write(self, path):
        assert self.tool is None
        source_current = self._current()
        open_intervals = []
        for state in self.states.values():
            end = self.windows[state["phase"]][1]
            open_intervals.append(
                {
                    "phase": state["phase"],
                    "label": state["label"],
                    "effect": state["effect"],
                    "relation": state["relation"],
                    "start": state["start"],
                    "no_supported_return_before_stop": True,
                    "start_to_trace_boundary_seconds": None
                    if end is None
                    else max(0.0, end - state["start"]),
                }
            )
        receipt = {
            "source_current": source_current,
            "source_hashes": self.source_hashes,
            "source_origins": {
                name: {"file": str(paths[0]), "spec_origin": str(paths[1])}
                for name, paths in self.source_origins.items()
            },
            "coverage": self.coverage,
            "qualified_at_durable_success": self.bound_at_success,
            "bind_failure": self.bind_failure,
            "duplicate_boundaries": self.boundary_duplicates,
            "selected_codes": len(self.selected),
            "unselected_controller_wrappers": [
                "_stream_assistant_response",
                "_run_agent_reply",
            ],
            "global_events": self.global_events_at_stop,
            "tool_retired": self.tool_retired,
            "stop_validation_error": self.stop_validation_error,
            "overflow": self.overflow,
            "unknown_exit_count": self.gap_count,
            "unknown_exit_samples": self.gap_samples,
            "windows": {
                name: {
                    "durable_success": v[0],
                    "trace_entry": v[1],
                    "wall_seconds": None if v[1] is None else v[1] - v[0],
                }
                for name, v in self.windows.items()
            },
            "rows": [
                {
                    "phase": key[0],
                    "label": key[1],
                    "effect": key[2],
                    "relation": key[3],
                    **value,
                }
                for key, value in self.rows.items()
            ],
            "no_supported_return_intervals": open_intervals,
            "limits": "Scalar states only; no frames/receiver/Task/arguments/returns retained. Local START/RETURN/YIELD/RESUME only. No exception events. Inclusive/suspended times overlap and are not CPU. Worker intervals are not assigned to a caller. Missing RETURN is an explicit unknown exit, not proof of a live await.",
        }
        Path(path).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        self.states.clear()
        return receipt


@pytest.fixture(autouse=True)
def _critical_postcommit_witness(request):
    target = os.environ.get("TLDW_POSTCOMMIT_DIAGNOSTIC_PATH")
    if not target or request.node.name != TARGET or request.module.__name__ != PROBE:
        yield
        return
    from Tests import private_profile

    if not private_profile.is_private_profile_child(request):
        yield
        return
    wrapper = request.module.test_native_console_pause_probe
    assert (
        type(wrapper) is FunctionType
        and wrapper.__globals__ is private_profile.__dict__
    )
    factory = private_profile.private_profile_test
    assert (
        type(factory) is FunctionType
        and factory.__globals__ is private_profile.__dict__
    )
    original = next(
        code
        for code in private_profile.private_profile_test.__code__.co_consts
        if type(code) is CodeType and code.co_name == "wrapped"
    )
    assert wrapper.__code__ is original
    closure = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__))
    body = closure["function"].cell_contents
    assert wrapper.__wrapped__ is body
    witness = CriticalPostcommitWitness(
        request.module, body, assertion_config=request.config
    )
    compiled = witness._source(private_profile)
    assert _shape(factory.__code__) == _shape(_find(compiled, "private_profile_test"))
    witness.originals.append(
        (
            private_profile,
            private_profile,
            "private_profile_test",
            factory,
            factory.__code__,
        )
    )
    try:
        witness.start()
        yield
    finally:
        try:
            witness.stop()
        finally:
            witness.write(target)
