# ruff: noqa: E721 -- exact defining function/descriptor types are evidence anchors.
"""Fixed original-code pause-probe seams; never replace a production callable.

Construct after the original probe's project imports. It fills the existing
Observation rows/config counters. All attempts count at original START;
elapsed time exists only for a supported original return/first context yield.
Exceptional timing gaps stay explicit. No config/argument bodies are recorded.
"""

import collections
import inspect
import os
from pathlib import Path
import sys
import threading
import time
from types import CodeType, FunctionType, ModuleType


class PassivePauseProbeSeams:
    def __init__(self, observation, *, config_only=False):
        self.observation = observation
        self.monitor = sys.monitoring
        self.lock = threading.Lock()
        self.local = threading.local()
        self.codes, self.bindings, self.bodies, self.contexts = {}, [], [], []
        self.states, self.actors = {}, {}
        self.gaps, self.gap_counts = [], collections.Counter()
        self.modules, self.overflow = [], 0
        self.tool, self.active, self.restoration = None, False, None
        self.installed = []
        self.config_only = config_only
        self.sensitive_code = None
        self.sensitive_coverage = "not_observed"
        self.mask = self.monitor.events.PY_START | self.monitor.events.PY_RETURN
        self.context_mask = (
            self.mask | self.monitor.events.PY_YIELD | self.monitor.events.PY_RESUME
        )
        self.callbacks = (self._start, self._return, self._yield, self._resume)
        self.events = (
            self.monitor.events.PY_START,
            self.monitor.events.PY_RETURN,
            self.monitor.events.PY_YIELD,
            self.monitor.events.PY_RESUME,
        )
        contextlib = sys.modules["contextlib"]
        factory = inspect.getattr_static(contextlib, "contextmanager")
        assert (
            type(factory) is FunctionType and factory.__globals__ is contextlib.__dict__
        )
        helper_codes = tuple(
            code
            for code in factory.__code__.co_consts
            if type(code) is CodeType and code.co_name == "helper"
        )
        assert len(helper_codes) == 1
        self.bindings.append((contextlib, "contextmanager", factory))
        self.bodies.append((factory, factory.__code__, factory.__globals__))
        self.modules.append((contextlib.__name__, contextlib))

        def module(name):
            owner = sys.modules[name]
            assert type(owner) is ModuleType
            self.modules.append((name, owner))
            return owner

        def select(
            owner, name, defining, label, *, context=False, native=False, config=None
        ):
            descriptor = inspect.getattr_static(owner, name)
            function = (
                descriptor.__func__ if type(descriptor) is classmethod else descriptor
            )
            assert type(function) is FunctionType
            self.bindings.append((owner, name, descriptor))
            self.bodies.append((function, function.__code__, function.__globals__))
            if context:
                assert (
                    function.__code__ is helper_codes[0]
                    and function.__globals__ is contextlib.__dict__
                )
                assert function.__code__.co_freevars == ("func",)
                assert (
                    function.__closure__ is not None and len(function.__closure__) == 1
                )
                body = function.__closure__[0].cell_contents
                assert (
                    type(body) is FunctionType and body.__globals__ is defining.__dict__
                )
                assert inspect.getattr_static(function, "__wrapped__") is body
                self.contexts.append((function, function.__closure__, body))
                self.bodies.append((body, body.__code__, body.__globals__))
                function = body
            else:
                assert function.__globals__ is defining.__dict__
            assert os.path.normcase(
                os.path.normpath(function.__code__.co_filename)
            ) == os.path.normcase(os.path.normpath(defining.__file__))
            code = function.__code__
            assert code not in self.codes
            assert not code.co_flags & (
                inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
            )
            if not context:
                assert not code.co_flags & (
                    inspect.CO_GENERATOR
                    | inspect.CO_COROUTINE
                    | inspect.CO_ASYNC_GENERATOR
                )
            self.codes[code] = dict(
                label=label,
                context=context,
                native=native,
                config=config,
                namespace=defining.__dict__,
            )
            return code

        self.config = module("tldw_chatbook.config")
        self.hit_code = self.posture_code = None
        for name in (
            "_config_file_posture",
            "_settings_cache_hit",
            "load_settings",
            "_invalidate_config_caches",
            "set_encryption_password",
            "_set_session_encryption_password",
        ):
            code = select(
                self.config,
                name,
                self.config,
                "tldw_chatbook.config." + name,
                config=name,
            )
            if name == "_settings_cache_hit":
                self.hit_code = code
            elif name == "_config_file_posture":
                self.posture_code = code
        if config_only:
            return
        life = module("tldw_chatbook.Backup_Recovery.config_participants")
        storage = module("tldw_chatbook.Backup_Recovery.storage_admission")
        select(life, "operation", life, life.__name__ + ".operation", context=True)
        for name in (
            "_acquire_storage",
            "_scope",
            "_local_pause_requested",
            "_observe_candidates",
            "_reuse_evidence",
        ):
            select(storage, name, storage, storage.__name__ + "." + name)
        acquisition_cls = inspect.getattr_static(storage, "_Acquisition")
        self.bindings.append((storage, "_Acquisition", acquisition_cls))
        select(
            acquisition_cls,
            "initializing",
            storage,
            "_Acquisition.initializing",
            context=True,
        )
        helpers = module("tldw_chatbook.DB.private_sqlite_process")
        helper_cls = inspect.getattr_static(helpers, "HelperLease")
        self.bindings.append((helpers, "HelperLease", helper_cls))
        select(helper_cls, "start", helpers, "HelperLease.start")
        if os.name == "nt":
            windows = module("tldw_chatbook.Utils.windows_files")
            native_cls = inspect.getattr_static(windows, "_Native")
            self.bindings.append((windows, "_Native", native_cls))
            for name in ("open_handle", "security", "ntfs", "_token_sid"):
                select(native_cls, name, windows, "_Native." + name, native=True)

    def _bind_sensitive_loaded(self):
        """No forced import: qualify the actual completed module and code once."""
        if self.config_only or self.sensitive_code is not None:
            return
        name = "tldw_chatbook.Utils.sensitive_paths"
        module = sys.modules.get(name)
        if type(module) is not ModuleType:
            self.sensitive_coverage = "module_not_loaded"
            return
        if type(module.__dict__.get("_SENSITIVE_INPUT_ORIGINALS")) is not tuple:
            self.sensitive_coverage = "module_definition_not_completed"
            return
        function = inspect.getattr_static(module, "_stock_sensitive_config_bundle")
        assert (
            type(function) is FunctionType and function.__globals__ is module.__dict__
        )
        original = function.__code__
        assert os.path.normcase(
            os.path.normpath(original.co_filename)
        ) == os.path.normcase(os.path.normpath(module.__file__))
        compiled = compile(
            Path(module.__file__).read_text(encoding="utf-8"),
            original.co_filename,
            "exec",
            dont_inherit=True,
        )
        expected = next(
            code
            for code in compiled.co_consts
            if type(code) is CodeType
            and code.co_name == "_stock_sensitive_config_bundle"
        )
        assert (
            original == expected
        ), "stock-bundle code differs from its defining source"
        assert not original.co_flags & (
            inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
        )
        self.modules.append((name, module))
        self.bindings.append((module, "_stock_sensitive_config_bundle", function))
        self.bodies.append((function, original, module.__dict__))
        self.codes[original] = dict(
            label=name + "._stock_sensitive_config_bundle",
            context=False,
            native=False,
            config="_stock_sensitive_config_bundle",
            namespace=module.__dict__,
        )
        self.sensitive_code = original
        self.sensitive_coverage = (
            "defining_module_body_qualified_before_original_return"
        )
        if self.active:
            assert self.monitor.get_local_events(self.tool, original) == 0
            self.monitor.set_local_events(self.tool, original, self.mask)
            self.installed.append(original)

    def _gap(self, state, reason):
        """Keep bounded scalar evidence; never retain an exceptional body/frame."""
        with self.lock:
            self.gap_counts[(state["label"], state["phase"], reason)] += 1
            if len(self.gaps) < 32:
                self.gaps.append(
                    dict(label=state["label"], phase=state["phase"], reason=reason)
                )

    def _frame_key(self, frame):
        # Strong Thread identity prevents thread-id ABA; no selected receiver,
        # argument, frame, parent or return object is kept by this scalar state.
        actor = threading.current_thread()
        self.actors[id(actor)] = actor
        return id(actor), frame.f_code, id(frame)

    def _elapsed(self, state, label, started, ended):
        if started is None:
            return
        elapsed = max(0.0, ended - started)
        observed = self.observation
        thread = "main" if state["actor_ident"] == observed.main else "worker"
        with observed.lock:
            row = observed.rows[(state["phase"], thread, label, state["caller"])]
            row[1] += elapsed
            row[2] = max(row[2], elapsed)
            if thread == "main" and elapsed >= 0.1 and len(observed.slow) < 100:
                observed.slow.append(
                    dict(
                        phase=state["phase"],
                        seam=label,
                        seconds=elapsed,
                        stack=state["caller"],
                    )
                )

    def _cache_metadata_return(self, frame, state, value):
        observed = self.observation
        if frame.f_code is self.posture_code:
            parent = frame.f_back
            if parent is not None and parent.f_code is self.hit_code:
                with self.lock:
                    hit = self.states.get(self._frame_key(parent))
                    if hit is not None and parent.f_globals is self.config.__dict__:
                        hit["observed_posture"] = value
            return
        if frame.f_code is not self.hit_code:
            return
        before, path = state["cache_before"], state["selected_path"]
        posture = state.get("observed_posture")
        has_posture = "observed_posture" in state
        outcome = (
            "hit"
            if value is not None
            else "empty"
            if state["cache_missing"]
            else "source"
            if before[1] != path
            else "posture"
            if has_posture and posture != before[2]
            else "unobserved_or_raced"
        )
        metadata, caller = {}, ""
        if outcome != "hit":
            after = (
                id(self.config._SETTINGS_CACHE),
                self.config._SETTINGS_CACHE_SOURCE,
                self.config._SETTINGS_CACHE_POSTURE,
            )
            metadata = dict(
                source_sha256=observed._metadata_hash(before[1]),
                selected_sha256=observed._metadata_hash(path),
                expected_posture_sha256=observed._metadata_hash(before[2]),
                observed_posture_sha256=observed._metadata_hash(posture),
                changed_fields=observed._posture_differences(before[2], posture)
                if has_posture
                else [],
                cache_state_changed=before != after,
            )
            caller = state["caller"]
            self.local.last_miss = (state["phase"], outcome)
        observed.config_record(
            "_settings_cache_hit", outcome, state["phase"], metadata, caller
        )

    def _start(self, code, offset):
        if not self.active:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        spec = self.codes[code]
        observed = self.observation
        if not spec["native"]:
            self._bind_sensitive_loaded()
        state = dict(
            phase=observed.phase,
            started=time.perf_counter(),
            actor_ident=threading.get_ident(),
            caller="" if spec["native"] else observed.stack(frame.f_back),
            entered=False,
            exit_started=None,
            context=spec["context"],
            label=spec["label"],
        )
        if code is self.hit_code:
            state.update(
                cache_before=(
                    id(self.config._SETTINGS_CACHE),
                    self.config._SETTINGS_CACHE_SOURCE,
                    self.config._SETTINGS_CACHE_POSTURE,
                ),
                cache_missing=self.config._SETTINGS_CACHE is None,
                selected_path=frame.f_locals["active_config_path"],
            )
        assert frame.f_globals is spec["namespace"]
        key = self._frame_key(frame)
        with self.lock:
            previous = self.states.pop(key, None)
            if len(self.states) >= 32768:
                self.overflow += 1
            else:
                self.states[key] = state
        if previous is not None:
            # START creates an invocation; a suspended context resumes through
            # PY_RESUME instead. A new START cannot reuse a still-live frame id.
            self._gap(previous, "frame_reused_without_observed_return")
        label = spec["label"] + ".enter" if spec["context"] else spec["label"]
        observed.record(label, state["phase"], 0.0, state["caller"])
        name = spec["config"]
        if name in {
            "load_settings",
            "_invalidate_config_caches",
            "set_encryption_password",
            "_set_session_encryption_password",
        }:
            forced = (
                frame.f_locals.get("force_reload", False)
                if name == "load_settings"
                else False
            )
            outcome = (
                "forced"
                if forced is True
                else "read"
                if name == "load_settings" and forced is False
                else "non_bool_force_argument"
                if name == "load_settings"
                else "invalidate"
            )
            recent = getattr(self.local, "last_miss", None)
            observed.config_record(
                name,
                outcome,
                state["phase"],
                dict(
                    force_reload=forced if type(forced) is bool else None,
                    preceding_miss=recent[1]
                    if recent and recent[0] == state["phase"]
                    else None,
                ),
                state["caller"],
            )

    def _yield(self, code, offset, value):
        frame = sys._getframe(1)
        assert frame.f_code is code and self.codes[code]["context"]
        with self.lock:
            state = self.states.get(self._frame_key(frame))
            if state is None:
                return
            repeated = state["entered"]
            state["entered"] = True
        if repeated:
            self._gap(state, "context_yielded_more_than_once")
            return
        self._elapsed(
            state, state["label"] + ".enter", state["started"], time.perf_counter()
        )

    def _resume(self, code, offset):
        frame = sys._getframe(1)
        assert frame.f_code is code and self.codes[code]["context"]
        with self.lock:
            state = self.states.get(self._frame_key(frame))
            if state is None:
                return
            state["exit_started"] = time.perf_counter()
        self.observation.record(
            state["label"] + ".exit", state["phase"], 0.0, state["caller"]
        )

    def _return(self, code, offset, value):
        frame = sys._getframe(1)
        assert frame.f_code is code
        assert frame.f_globals is self.codes[code]["namespace"]
        with self.lock:
            state = self.states.pop(self._frame_key(frame), None)
        if state is None:
            return
        if state["context"]:
            if not state["entered"]:
                self._elapsed(
                    state,
                    state["label"] + ".enter",
                    state["started"],
                    time.perf_counter(),
                )
            elif state["exit_started"] is not None:
                self._elapsed(
                    state,
                    state["label"] + ".exit",
                    state["exit_started"],
                    time.perf_counter(),
                )
            else:
                self._gap(state, "normal_context_return_without_supported_resume")
        else:
            self._elapsed(state, state["label"], state["started"], time.perf_counter())
        if code is self.sensitive_code:
            self.observation.config_record(
                "_stock_sensitive_config_bundle",
                "custom" if value is None else "stock",
                state["phase"],
                {},
                state["caller"],
            )
        if self.codes[code]["config"] in {
            "_settings_cache_hit",
            "_config_file_posture",
        }:
            self._cache_metadata_return(frame, state, value)

    def start(self):
        self._bind_sensitive_loaded()
        assert self.bindings_current()
        self.tool = next(
            (number for number in (5, 4, 3) if self.monitor.get_tool(number) is None),
            None,
        )
        assert self.tool is not None
        self.monitor.use_tool_id(self.tool, "tldw-passive-pause-seams-" + str(id(self)))
        assert self.monitor.get_events(self.tool) == 0
        for event, callback in zip(self.events, self.callbacks, strict=True):
            assert self.monitor.register_callback(self.tool, event, callback) is None
        self.active = True
        try:
            for code, spec in tuple(self.codes.items()):
                assert self.monitor.get_local_events(self.tool, code) == 0
                self.monitor.set_local_events(
                    self.tool, code, self.context_mask if spec["context"] else self.mask
                )
                self.installed.append(code)
        except BaseException:
            self.stop()
            raise

    def bindings_current(self):
        return (
            all(sys.modules.get(name) is module for name, module in self.modules)
            and all(
                inspect.getattr_static(owner, name) is descriptor
                for owner, name, descriptor in self.bindings
            )
            and all(
                function.__code__ is code and function.__globals__ is namespace
                for function, code, namespace in self.bodies
            )
            and all(
                function.__closure__ is closure
                and closure[0].cell_contents is body
                and inspect.getattr_static(function, "__wrapped__") is body
                for function, closure, body in self.contexts
            )
        )

    def stop(self):
        assert self.monitor.get_events(self.tool) == 0
        for code in self.installed:
            expected = self.context_mask if self.codes[code]["context"] else self.mask
            assert self.monitor.get_local_events(self.tool, code) == expected
            self.monitor.set_local_events(self.tool, code, 0)
        for event, callback in zip(self.events, self.callbacks, strict=True):
            assert self.monitor.register_callback(self.tool, event, None) is callback
        self.active = False
        self.monitor.free_tool_id(self.tool)
        assert self.monitor.get_tool(self.tool) is None
        self.restoration = "selected_local_callbacks_removed_tool_freed_global0"
        # Unmatched scalar entries are explicit unknown exits. No receiver,
        # argument, selected frame or parent frame can be retained by them.
        with self.lock:
            unmatched = list(self.states.values())
            self.states.clear()
            self.actors.clear()
        for state in unmatched:
            self._gap(state, "scalar_without_supported_original_return")

    def receipt(self):
        return dict(
            diagnostic_only=True,
            selected_code_count=len(self.codes),
            global_events=0,
            overflow=self.overflow,
            restoration=self.restoration,
            original_bindings_and_bodies_unchanged=self.bindings_current(),
            all_attempt_counts_from_original_START=True,
            elapsed_only_from_supported_return_or_first_context_yield=True,
            exceptional_timing_gaps_not_invented_as_returns=list(self.gaps),
            exceptional_timing_gap_counts=[
                dict(label=label, phase=phase, reason=reason, count=count)
                for (label, phase, reason), count in self.gap_counts.items()
            ],
            gap_sample_limit=32,
            sensitive_bundle_return_coverage=self.sensitive_coverage,
            sensitive_bundle_return_records_only_stock_or_custom=True,
            all_selected_states_keep_no_frame_receiver_argument_or_return_objects=True,
            all_returns_match_exact_thread_code_and_frame_id=True,
            frame_reuse_records_unknown_exit_never_healthy_return=True,
            no_production_callable_or_alias_replacement=True,
            no_global_monitoring_or_new_task_thread_frame_enumeration=True,
            native_source_and_whole_probe_qualification_pending=True,
        )
