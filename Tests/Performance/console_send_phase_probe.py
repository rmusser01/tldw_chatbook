"""Optional scalar elapsed spans; overlapping intervals are not additive costs."""

import asyncio
import inspect
import sys
import threading
import time


class SendPhaseProbe:
    """Observe original bodies without retaining frames, arguments or results."""

    LIMIT = 512
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

    def __init__(self, controller, gateway, phase):
        self.phase = phase
        self.selected = {}
        self.bindings = []
        for owner, names in (
            (
                controller,
                (
                    "_run_durable_postcommit_effect",
                    "_build_durable_trace_request",
                    "_admit_capture_policy",
                    "hook_admission_reason",
                    "_stream_assistant_response",
                    "_apply_conversation_memory_preflight",
                    "_compose_agent_request_providers",
                    "_run_agent_reply",
                    "_run_direct_provider_reply",
                    "_personal_context_service",
                    "_personal_context_builder",
                ),
            ),
            (gateway, ("prepare_chat_request",)),
        ):
            for name in names:
                descriptor = inspect.getattr_static(type(owner), name)
                function = inspect.unwrap(descriptor)
                code = function.__code__
                assert code not in self.selected
                self.selected[code] = (id(owner), type(owner).__name__ + "." + name)
                self.bindings.append((type(owner), name, descriptor, function, code))
        self.monitor = sys.monitoring
        self.tool = None
        self.registered = []
        self.touched = []
        self.active = False
        self.lock = threading.Lock()
        self.pending = {}
        self.spans = []
        self.overflow = self.collisions = 0

    @staticmethod
    def _key(code, frame):
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return (
            code,
            id(frame),
            threading.get_ident(),
            id(task) if task is not None else 0,
        )

    def _start(self, code, _offset):
        if not self.active or code not in self.selected:
            return
        started = time.perf_counter()
        frame = sys._getframe(1)
        owner, label = self.selected[code]
        phase = self.phase()
        if id(frame.f_locals.get("self")) != owner or phase not in {
            "send_1",
            "send_2",
            "send_3",
        }:
            return
        key = self._key(code, frame)
        effect = None
        if code.co_name == "_run_durable_postcommit_effect":
            candidate = frame.f_locals.get("effect_name")
            effect = (
                candidate
                if type(candidate) is str and candidate in self.EFFECTS  # noqa: E721 -- retain builtin scalars only
                else "other"
            )
        with self.lock:
            if len(self.spans) >= self.LIMIT:
                self.overflow += 1
                return
            if key in self.pending:
                self.collisions += 1
            index = len(self.spans)
            self.pending[key] = index
            self.spans.append(
                dict(
                    span_id=index,
                    phase=phase,
                    function=label,
                    effect=effect,
                    frame=key[1],
                    thread=key[2],
                    task=key[3],
                    started=started,
                    ended=None,
                    seconds=None,
                    outcome="unfinished",
                )
            )

    def _finish(self, code, frame, outcome):
        if not self.active or code not in self.selected:
            return
        ended = time.perf_counter()
        key = self._key(code, frame)
        with self.lock:
            index = self.pending.pop(key, None)
            if index is None:
                # Calls begun outside a Send window are intentionally ignored.
                return
            span = self.spans[index]
            span.update(ended=ended, seconds=ended - span["started"], outcome=outcome)

    def _returned(self, code, _offset, _value):
        self._finish(code, sys._getframe(1), "returned")

    def _unwound(self, code, _offset, _error):
        # PY_UNWIND is global-only in Python 3.12. Filter before touching a frame.
        if code in self.selected:
            self._finish(code, sys._getframe(1), "unwound")

    def start(self):
        monitor = self.monitor
        tool = next(value for value in range(6) if monitor.get_tool(value) is None)
        monitor.use_tool_id(tool, "console-send-scalar-phases")
        self.tool = tool
        try:
            for event, callback in (
                (monitor.events.PY_START, self._start),
                (monitor.events.PY_RETURN, self._returned),
                (monitor.events.PY_UNWIND, self._unwound),
            ):
                previous = monitor.register_callback(tool, event, callback)
                self.registered.append((event, previous))
                assert previous is None
            for code in self.selected:
                self.touched.append(code)
                monitor.set_local_events(
                    tool, code, monitor.events.PY_START | monitor.events.PY_RETURN
                )
            monitor.set_events(tool, monitor.events.PY_UNWIND)
            self.active = True
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.tool is None:
            return
        tool, self.tool = self.tool, None
        self.active = False
        failures = []
        # Attempt every reset even if setup failed partway through registration.
        for reset in (
            lambda: self.monitor.set_events(tool, 0),
            *(
                lambda code=code: self.monitor.set_local_events(tool, code, 0)
                for code in self.touched
            ),
            *(
                lambda event=event, previous=previous: self.monitor.register_callback(
                    tool, event, previous
                )
                for event, previous in self.registered
            ),
            lambda: self.monitor.free_tool_id(tool),
        ):
            try:
                reset()
            except BaseException as error:
                failures.append(type(error).__name__)
        assert not failures, failures

    def report(self):
        with self.lock:
            return dict(
                diagnostic_only=True,
                timing_acceptance=False,
                elapsed_including_awaits=True,
                nested_spans_are_not_additive=True,
                original_body_excludes_decorator_overhead=True,
                global_event="PY_UNWIND (exact selected-code filter)",
                selected_code_count=len(self.selected),
                span_limit=self.LIMIT,
                overflow=self.overflow,
                pairing_collisions=self.collisions,
                unfinished_spans=sum(span["ended"] is None for span in self.spans),
                monitoring_retired=self.tool is None and not self.active,
                original_bindings_current=all(
                    inspect.getattr_static(owner, name) is descriptor
                    and inspect.unwrap(descriptor) is function
                    and function.__code__ is code
                    for owner, name, descriptor, function, code in self.bindings
                ),
                spans=list(self.spans),
            )
