"""Eight-body passive startup timing; diagnostic rows never accept budgets."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import inspect
from pathlib import Path
import sys
import threading
import time
from types import CodeType, FunctionType, ModuleType

_MEMBERS = (
    "__init__",
    "_init_notes_service",
    "_init_providers_models",
    "_init_prompts_service",
    "_init_media_db",
    "on_mount",
    "_push_initial_screen",
    "_post_mount_setup",
)
_INITIALIZERS = frozenset(_MEMBERS[1:5])


class _ScalarSpans:
    """Only bounded scalar state; frame identity is an integer, never a frame."""

    def __init__(self):
        self.active = {}
        self.rows = []
        self.gaps = []
        self.gap_counts = Counter()
        self.overflow = 0
        self.details = 0

    def gap(self, label, reason):
        self.gap_counts[reason] += 1
        if len(self.gaps) < 32:
            self.gaps.append({"label": label, "reason": reason})

    def event(self, event, key, label, now, asynchronous):
        state = self.active.get(key)
        if event == "start":
            if state is not None:
                self.gap(label, "frame_id_reused_without_return")
                del self.active[key]
            if len(self.active) >= 64:
                self.overflow += 1
                self.gap(label, "active_span_overflow")
                return
            self.active[key] = {
                "label": label,
                "thread": key[0],
                "started": now,
                "segment": now,
                "segments": [],
                "async": asynchronous,
            }
            return
        if state is None:
            self.gap(label, event + "_without_start")
            return
        if event == "resume":
            if not asynchronous or state["segment"] is not None:
                self.gap(label, "resume_without_yield")
                del self.active[key]
            else:
                state["segment"] = now
            return
        if event == "yield":
            if not asynchronous or state["segment"] is None:
                self.gap(label, "yield_without_active_segment")
                del self.active[key]
            elif self.details >= 2048:
                self.overflow += 1
                self.gap(label, "segment_overflow")
                del self.active[key]
            else:
                self.details += 1
                state["segments"].append([state["segment"], now])
                state["segment"] = None
            return
        if event != "return":
            self.gap(label, "unknown_event")
            return
        del self.active[key]
        if state["segment"] is None:
            self.gap(label, "return_without_active_segment")
            return
        if self.details >= 2048:
            self.overflow += 1
            self.gap(label, "detail_overflow")
            return
        self.details += 1
        state["segments"].append([state["segment"], now])
        if self.details >= 2048:
            self.overflow += 1
            self.gap(label, "row_overflow")
            return
        self.details += 1
        self.rows.append(
            {
                "label": label,
                "thread": state["thread"],
                "started": state["started"],
                "finished": now,
                "inclusive_elapsed": now - state["started"],
                "active_elapsed": sum(b - a for a, b in state["segments"]),
                "segments": state["segments"],
                "async": state["async"],
            }
        )

    def finish(self, now):
        for state in self.active.values():
            self.gap(state["label"], "no_observed_normal_return")
        self.active.clear()
        return {
            "spans": list(self.rows),
            "gaps": list(self.gaps),
            "gap_counts": dict(self.gap_counts),
            "overflow": self.overflow,
            "complete": not self.gap_counts and not self.overflow,
        }


def _nested(code, name):
    if code.co_qualname == name:
        return code
    for value in code.co_consts:
        if type(value) is CodeType:
            found = _nested(value, name)
            if found is not None:
                return found
    return None


def _source(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    return raw, ast.parse(text), compile(text, str(path), "exec", dont_inherit=True)


def _anchors(tree):
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TldwCli"
    )
    constructor = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    wait = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and ast.unparse(node.iter.func) == "concurrent.futures.as_completed"
    ]
    after = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "parallel_duration"
            for target in node.targets
        )
    ]
    if len(wait) != 1 or len(after) != 1 or wait[0].lineno >= after[0].lineno:
        raise ValueError("constructor_wait_anchor_ambiguous")
    push = next(
        node
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_push_initial_screen"
    )
    regions = [("constructor_parallel_join", wait[0].lineno, after[0].lineno)]
    definitions = (
        ("navigation_target", "_resolve_screen_navigation_target"),
        ("screen_owned_css", "_ensure_screen_owned_css"),
        ("screen_construct", "screen_class"),
        ("screen_push", "push_screen"),
    )
    for label, call_name in definitions:
        matches = []
        for index, statement in enumerate(push.body[:-1]):
            call = getattr(statement, "value", None)
            if isinstance(call, ast.Await):
                call = call.value
            if isinstance(call, ast.Call) and (
                (isinstance(call.func, ast.Attribute) and call.func.attr == call_name)
                or (isinstance(call.func, ast.Name) and call.func.id == call_name)
            ):
                start = (
                    call.lineno if label == "navigation_target" else statement.lineno
                )
                matches.append((start, push.body[index + 1].lineno))
        if len(matches) != 1:
            raise ValueError("push_anchor_ambiguous:" + label)
        regions.append((label, *matches[0]))
    return regions


def _initializer_summary(rows, regions):
    """Correlate exact scalar endpoints; unions and remainders are not CPU time."""
    selected = {
        label: [row for row in rows if row["label"] == label]
        for label in _INITIALIZERS | {"__init__"}
    }
    joins = [row for row in regions if row["label"] == "constructor_parallel_join"]
    if any(len(values) != 1 for values in selected.values()) or len(joins) != 1:
        return None
    constructor = selected["__init__"][0]
    workers = [selected[label][0] for label in _INITIALIZERS]
    join = joins[0]
    if any(
        not constructor["started"]
        <= row["started"]
        <= row["finished"]
        <= constructor["finished"]
        for row in [*workers, join]
    ):
        return None
    union = []
    for begin, end in sorted((row["started"], row["finished"]) for row in workers):
        if union and begin <= union[-1][1]:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([begin, end])
    uncovered = []
    previous = constructor["started"]
    for begin, end in union:
        if previous < begin:
            uncovered.append([previous, begin])
        previous = end
    if previous < constructor["finished"]:
        uncovered.append([previous, constructor["finished"]])
    return {
        "last_observed_normal_completion": max(row["finished"] for row in workers),
        "worker_body_union_intervals": union,
        "constructor_intervals_without_selected_worker_body": uncovered,
        "remainder_is_not_exclusive_cpu": True,
        "body_completion_is_not_service_success": True,
    }


class StartupTimingWitness:
    """Observe only actual loaded original bodies; never import the application."""

    def __init__(self, repo: Path):
        self.repo = repo.absolute()
        self.monitor = sys.monitoring
        self._specs = tuple(
            ("tldw_chatbook.app", "TldwCli", name, "tldw_chatbook/app.py", name)
            for name in _MEMBERS
        )
        self._caller_spec = (
            "tldw_chatbook.app_service_wiring",
            "ServiceWiringMixin",
            "_timed_init_task",
            "tldw_chatbook/app_service_wiring.py",
        )
        self._regions = None
        self.bindings = []
        self.class_slots = []
        self.module_metadata = []
        self.registered_callbacks = {}
        self.callback_pending = None
        self.sources = {}
        self.codes = {}
        self.threads = []
        self.ledger = _ScalarSpans()
        self.lock = threading.RLock()
        self.tool = None
        self.active = False
        self.issues = Counter()
        self.masks = {}
        self.callbacks = {}
        self.callback_shapes = []
        self.region_states = {}
        self.region_rows = []
        self.constructors = 0
        self.prepared = False

    def _bind(self, module_name, class_name, name, relative):
        module = sys.modules.get(module_name)
        if type(module) is not ModuleType:
            raise ValueError("selected_module_not_loaded:" + module_name)
        owner = inspect.getattr_static(module, class_name)
        self.class_slots.append((module, class_name, owner))
        metadata = vars(module)
        spec = metadata.get("__spec__")
        loader = metadata.get("__loader__")
        origin = None if spec is None else inspect.getattr_static(spec, "origin")
        spec_loader = None if spec is None else inspect.getattr_static(spec, "loader")
        filename = metadata.get("__file__")
        path = self.repo / relative
        if (
            type(filename) is not str  # noqa: E721 -- decline custom path metadata.
            or Path(filename).absolute() != path.absolute()
            or (
                spec is not None
                and (
                    type(origin) is not str  # noqa: E721 -- exact source metadata.
                    or Path(origin).absolute() != path.absolute()
                    or spec_loader is not loader
                )
            )
        ):
            raise ValueError("selected_module_source_metadata_changed")
        self.module_metadata.append(
            (module, filename, spec, origin, loader, spec_loader)
        )
        function = inspect.getattr_static(owner, name)
        if type(function) is not FunctionType or function.__globals__ is not vars(
            module
        ):
            raise ValueError("selected_function_namespace_or_slot_changed")
        raw, tree, compiled = _source(path)
        expected = _nested(compiled, class_name + "." + name)
        if (
            expected is None
            or function.__code__ != expected
            or function.__code__.co_firstlineno != expected.co_firstlineno
            or Path(function.__code__.co_filename).absolute() != path.absolute()
        ):
            raise ValueError("selected_code_differs_from_actual_source")
        self.sources[path] = hashlib.sha256(raw).hexdigest()
        binding = (
            module_name,
            module,
            owner,
            name,
            function,
            function.__code__,
            function.__globals__,
        )
        self.bindings.append(binding)
        return function, tree

    def _prepare(self):
        self.caller, _ = self._bind(*self._caller_spec)
        for module, cls, name, path, label in self._specs:
            function, tree = self._bind(module, cls, name, path)
            if function.__code__ in self.codes:
                raise ValueError("selected_code_not_unique")
            self.codes[function.__code__] = (
                label,
                function.__globals__,
                bool(function.__code__.co_flags & inspect.CO_COROUTINE),
            )
            if name == "__init__" and self._regions is None:
                self._regions = _anchors(tree)
        self._regions = tuple(self._regions or ())
        module_name, class_name, _, _, _ = self._specs[0]
        module = sys.modules[module_name]
        owner = inspect.getattr_static(module, class_name)
        caller_slot = self._caller_spec[2]
        if inspect.getattr_static(owner, caller_slot) is not self.caller:
            raise ValueError("inherited_initializer_caller_slot_changed")
        self.bindings.append(
            (
                module_name,
                module,
                owner,
                caller_slot,
                self.caller,
                self.caller.__code__,
                self.caller.__globals__,
            )
        )
        observer_module = type(self).__module__
        observer_file = Path(vars(sys.modules[observer_module])["__file__"])
        observer_relative = str(observer_file.absolute().relative_to(self.repo))
        for name in (
            "_start_event",
            "_resume_event",
            "_yield_event",
            "_return_event",
            "_line_event",
            "_record",
        ):
            self._bind(observer_module, type(self).__name__, name, observer_relative)
        self.prepared = True

    def _bindings_current(self, *, source=False):
        for module, class_name, owner in self.class_slots:
            if inspect.getattr_static(module, class_name) is not owner:
                return False
        for module, filename, spec, origin, loader, spec_loader in self.module_metadata:
            values = vars(module)
            if (
                values.get("__file__") != filename
                or values.get("__spec__") is not spec
                or values.get("__loader__") is not loader
                or (
                    spec is not None
                    and (
                        inspect.getattr_static(spec, "origin") != origin
                        or inspect.getattr_static(spec, "loader") is not spec_loader
                    )
                )
            ):
                return False
        for name, module, owner, slot, function, code, namespace in self.bindings:
            if (
                sys.modules.get(name) is not module
                or inspect.getattr_static(owner, slot) is not function
                or function.__code__ is not code
                or function.__globals__ is not namespace
            ):
                return False
        if any(
            inspect.getattr_static(type(self), name) is not function
            or function.__code__ is not code
            or function.__globals__ is not namespace
            for name, function, code, namespace in self.callback_shapes
        ):
            return False
        return not source or all(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for path, digest in self.sources.items()
        )

    def _record(self, kind, code, frame, line=None):
        if not self.active:
            self.issues["callback_after_inactive"] += 1
            return
        try:
            label, namespace, asynchronous = self.codes[code]
            if (
                frame.f_code is not code
                or frame.f_globals is not namespace
                or not self._bindings_current()
            ):
                self.issues["event_source_or_callback_drift"] += 1
                return
            caller = frame.f_back
            if label in _INITIALIZERS and (
                caller is None
                or caller.f_code is not self.caller.__code__
                or caller.f_globals is not self.caller.__globals__
            ):
                self.issues["initializer_caller_not_original"] += 1
                return
            thread = threading.current_thread()
            with self.lock:
                token = next(
                    (
                        index
                        for index, actual in enumerate(self.threads)
                        if actual is thread
                    ),
                    None,
                )
                if token is None:
                    if len(self.threads) >= 8:
                        self.issues["thread_identity_overflow"] += 1
                        return
                    token = len(self.threads)
                    self.threads.append(thread)
                key = token, id(code), id(frame)
                now = time.perf_counter()
                if kind == "line":
                    for region, begin, end in self._regions:
                        selected = (
                            label == "__init__"
                            if region == "constructor_parallel_join"
                            else label == "_push_initial_screen"
                        )
                        if not selected:
                            continue
                        region_key = (*key, region)
                        if line == begin and region_key not in self.region_states:
                            if len(self.region_states) < 64:
                                self.region_states[region_key] = now
                            else:
                                self.issues["active_region_overflow"] += 1
                        elif line == end and region_key in self.region_states:
                            started = self.region_states.pop(region_key)
                            if self.ledger.details < 2048:
                                self.ledger.details += 1
                                self.region_rows.append(
                                    {
                                        "label": region,
                                        "thread": token,
                                        "started": started,
                                        "finished": now,
                                        "inclusive_elapsed": now - started,
                                    }
                                )
                            else:
                                self.issues["region_row_overflow"] += 1
                    return
                if kind == "start" and label == "__init__":
                    self.constructors += 1
                self.ledger.event(kind, key, label, now, asynchronous)
        except Exception:
            self.issues["event_metadata_refused"] += 1

    def _start_event(self, code, offset):
        frame = sys._getframe(1)
        try:
            self._record("start", code, frame)
        finally:
            del frame

    def _resume_event(self, code, offset):
        frame = sys._getframe(1)
        try:
            self._record("resume", code, frame)
        finally:
            del frame

    def _yield_event(self, code, offset, value):
        frame = sys._getframe(1)
        try:
            self._record("yield", code, frame)
        finally:
            del frame

    def _return_event(self, code, offset, value):
        frame = sys._getframe(1)
        try:
            self._record("return", code, frame)
        finally:
            del frame

    def _line_event(self, code, line):
        frame = sys._getframe(1)
        try:
            self._record("line", code, frame, line)
        finally:
            del frame

    def start(self) -> None:
        if self.active or self.tool is not None:
            raise ValueError("witness_is_one_shot")
        self._prepare()
        self.tool = next(
            (number for number in (5, 4, 3) if self.monitor.get_tool(number) is None),
            None,
        )
        if self.tool is None:
            raise ValueError("monitoring_tools_occupied")
        self.tool_name = "startup-timing-" + str(id(self))
        self.monitor.use_tool_id(self.tool, self.tool_name)
        events = self.monitor.events
        self.callbacks = {
            events.PY_START: self._start_event,
            events.PY_RESUME: self._resume_event,
            events.PY_YIELD: self._yield_event,
            events.PY_RETURN: self._return_event,
            events.LINE: self._line_event,
        }
        for name in (
            "_start_event",
            "_resume_event",
            "_yield_event",
            "_return_event",
            "_line_event",
            "_record",
        ):
            function = inspect.getattr_static(type(self), name)
            self.callback_shapes.append(
                (name, function, function.__code__, function.__globals__)
            )
        self.active = True
        try:
            if self.monitor.get_events(self.tool) != 0:
                raise ValueError("foreign_global_monitoring_state")
            for event, callback in self.callbacks.items():
                self.callback_pending = event, callback
                previous = self.monitor.register_callback(self.tool, event, callback)
                self.callback_pending = None
                if previous is not None:
                    self.monitor.register_callback(self.tool, event, previous)
                    self.issues["foreign_callback_at_install"] += 1
                    raise ValueError("foreign_callback_at_install")
                self.registered_callbacks[event] = callback
            for code, (label, namespace, asynchronous) in self.codes.items():
                mask = events.PY_START | events.PY_RETURN
                if asynchronous:
                    mask |= events.PY_RESUME | events.PY_YIELD
                if label in ("__init__", "_push_initial_screen"):
                    mask |= events.LINE
                if self.monitor.get_local_events(self.tool, code) != 0:
                    raise ValueError("foreign_local_mask_at_install")
                self.masks[code] = mask
                self.monitor.set_local_events(self.tool, code, mask)
        except BaseException:
            self.issues["installation_failed"] += 1
            self.stop()
            raise

    def stop(self) -> dict:
        if self.tool is None:
            raise ValueError("witness_never_started")
        owned = self.monitor.get_tool(self.tool) == self.tool_name
        global_zero = owned and self.monitor.get_events(self.tool) == 0
        masks_owned = owned and all(
            self.monitor.get_local_events(self.tool, code) == mask
            or (
                self.issues.get("installation_failed")
                and self.monitor.get_local_events(self.tool, code) == 0
            )
            for code, mask in self.masks.items()
        )
        try:
            source_current = self._bindings_current(source=True)
        except Exception:
            source_current = False
            self.issues["source_check_failed_at_retirement"] += 1
        callbacks_owned = True
        if owned and global_zero and masks_owned:
            for code in self.masks:
                self.monitor.set_local_events(self.tool, code, 0)
            for event, expected in self.registered_callbacks.items():
                previous = self.monitor.register_callback(self.tool, event, None)
                if previous is not expected:
                    callbacks_owned = False
                    self.monitor.register_callback(self.tool, event, previous)
            if self.callback_pending is not None:
                event, expected = self.callback_pending
                previous = self.monitor.register_callback(self.tool, event, None)
                if previous is not None and previous is not expected:
                    callbacks_owned = False
                    self.monitor.register_callback(self.tool, event, previous)
                self.callback_pending = None
            if callbacks_owned and not self.issues.get("foreign_callback_at_install"):
                self.monitor.free_tool_id(self.tool)
        else:
            callbacks_owned = False
        cleared = owned and all(
            self.monitor.get_local_events(self.tool, code) == 0 for code in self.masks
        )
        freed = self.monitor.get_tool(self.tool) is None
        self.active = False
        for key in self.region_states:
            self.ledger.gap(key[-1], "region_endpoint_not_observed")
        self.region_states.clear()
        scalar = self.ledger.finish(time.perf_counter())
        if self.constructors != 1:
            self.issues["constructor_correlation_not_exactly_one"] += 1
        completed_labels = {row["label"] for row in scalar["spans"]}
        missing_bodies = sorted({item[-1] for item in self._specs} - completed_labels)
        missing_regions = sorted(
            {region[0] for region in self._regions}
            - {row["label"] for row in self.region_rows}
        )
        if missing_bodies:
            self.issues["expected_body_normal_completion_missing"] += len(
                missing_bodies
            )
        if missing_regions:
            self.issues["expected_region_endpoint_missing"] += len(missing_regions)
        complete = (
            scalar["complete"]
            and source_current
            and owned
            and global_zero
            and masks_owned
            and callbacks_owned
            and cleared
            and freed
            and not self.issues
        )
        summary = (
            _initializer_summary(scalar["spans"], self.region_rows)
            if source_current
            and owned
            and global_zero
            and masks_owned
            and callbacks_owned
            and cleared
            and freed
            else None
        )
        return {
            "diagnostic_only": True,
            "budget_acceptance_eligible": False,
            "budgets_pass": None,
            "complete": complete,
            **{key: value for key, value in scalar.items() if key != "complete"},
            "regions": self.region_rows,
            "initializer_summary": summary,
            "missing_bodies": missing_bodies,
            "missing_regions": missing_regions,
            "source_current": source_current,
            "source_bytes": {
                str(path.relative_to(self.repo)): digest
                for path, digest in self.sources.items()
            },
            "issues": dict(self.issues),
            "constructor_count": self.constructors,
            "thread_identity_count": len(self.threads),
            "monitoring_global_zero": global_zero,
            "monitoring_masks_owned": masks_owned,
            "monitoring_callbacks_owned": callbacks_owned,
            "monitoring_masks_cleared_while_active": cleared,
            "monitoring_tool_freed": freed,
            "coverage": "Selected original body normal completion; async inclusive wall and active segments, not exclusive CPU. Initializers overlap and are never summed; missing returns/endpoints are gaps.",
        }
