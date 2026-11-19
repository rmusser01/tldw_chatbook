"""Opt-in scalar metadata attribution for one existing native hook read."""

import hashlib
import inspect
from pathlib import Path
import sys
import threading
import time


class RawParentMetadataProbe:
    """Observe original synchronous bodies without replacing their callbacks."""

    SLOT_LIMIT = 128
    TOOL_NAME = "raw-parent-metadata-probe"

    def __init__(self):
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        windows = sys.modules["tldw_chatbook.Utils.windows_files"]
        assert type(raw.os) is windows.WindowsOS
        self.modules = (raw, windows)
        self.classes = (windows.WindowsOS, windows._Native)
        self.source_hashes = self._source_hashes()
        self.bindings = []
        self.selected = {}
        for owner, name, label in (
            (raw, "_check_parent_pins", "parent_check"),
            (windows.WindowsOS, "stat", "stat"),
            (windows.WindowsOS, "fstat", "fstat"),
            (windows.WindowsOS, "_stat_handle", "stat_handle"),
            (windows._Native, "security", "security"),
            (windows._Native, "open_handle", "open_handle"),
        ):
            function = inspect.getattr_static(owner, name)
            code = function.__code__
            module = sys.modules[function.__module__]
            assert function.__globals__ is vars(module)
            assert Path(code.co_filename).resolve() == Path(module.__file__).resolve()
            self.bindings.append((owner, name, function, code, module))
            assert code not in self.selected
            self.selected[code] = label
        keys = ["parent_check"] + [
            f"{branch}.{label}"
            for branch in ("parent_named", "parent_retained", "parent_other", "outside")
            for label in ("stat", "fstat", "stat_handle", "security", "open_handle")
        ]
        self.rows = {
            key: dict(
                starts=0, returns=0, unwinds=0, wall_seconds=0.0, thread_cpu_seconds=0.0
            )
            for key in keys
        }
        self.monitor = sys.monitoring
        self.thread = threading.get_ident()
        self.tool = None
        self.active = False
        self.started = False
        self.retired = False
        self.slots = {}
        self.stack = []
        self.overflow = self.unmatched = self.exit_order_mismatches = 0
        self.unwind_callbacks = self.unselected_unwinds = self.other_thread_unwinds = 0
        self.local_codes = []
        self.registered = []
        self.retirement_errors = []
        self.callbacks = (
            (self.monitor.events.PY_START, self._start),
            (self.monitor.events.PY_RETURN, self._return),
            (self.monitor.events.PY_UNWIND, self._unwind),
        )
        assert self.defining_bodies_current()

    def _source_hashes(self):
        return {
            module.__name__: hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in self.modules
        }

    def defining_bodies_current(self):
        raw, windows = self.modules
        return (
            type(raw.os) is self.classes[0]
            and windows.WindowsOS is self.classes[0]
            and windows._Native is self.classes[1]
            and all(
                sys.modules.get(module.__name__) is module
                and inspect.getattr_static(owner, name) is function
                and function.__code__ is code
                and function.__globals__ is vars(module)
                for owner, name, function, code, module in self.bindings
            )
            and self._source_hashes() == self.source_hashes
        )

    def _start(self, code, offset):
        if not self.active or threading.get_ident() != self.thread:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        key = id(frame)
        if len(self.slots) >= self.SLOT_LIMIT:
            self.overflow += 1
            return
        label = self.selected[code]
        branch = "outside"
        # Only scalar frame identities and labels survive a callback. All six
        # selected bodies are synchronous, so their starts/exits form a stack.
        parent_index = next(
            (
                index
                for index in range(len(self.stack) - 1, -1, -1)
                if self.slots[self.stack[index]][0] == "parent_check"
            ),
            None,
        )
        if parent_index is not None:
            branch = "parent_other"
            for identity in reversed(self.stack[parent_index + 1 :]):
                enclosing = self.slots[identity][0]
                if enclosing in ("stat", "fstat"):
                    branch = (
                        "parent_named" if enclosing == "stat" else "parent_retained"
                    )
                    break
            if label in ("stat", "fstat"):
                branch = "parent_named" if label == "stat" else "parent_retained"
        row_key = "parent_check" if label == "parent_check" else f"{branch}.{label}"
        self.rows[row_key]["starts"] += 1
        self.slots[key] = (label, row_key, time.perf_counter(), time.thread_time())
        self.stack.append(key)

    def _finish(self, key, outcome):
        wall, cpu = time.perf_counter(), time.thread_time()
        started = self.slots.pop(key, None)
        if started is None:
            self.unmatched += 1
            return
        if not self.stack or self.stack[-1] != key:
            self.exit_order_mismatches += 1
            self.stack.remove(key)
        else:
            self.stack.pop()
        _, row_key, wall_start, cpu_start = started
        row = self.rows[row_key]
        row[outcome] += 1
        row["wall_seconds"] += wall - wall_start
        row["thread_cpu_seconds"] += cpu - cpu_start

    def _return(self, code, offset, value):
        if self.active and threading.get_ident() == self.thread:
            frame = sys._getframe(1)
            assert frame.f_code is code
            self._finish(id(frame), "returns")

    def _unwind(self, code, offset, exception):
        # PY_UNWIND is global-only on Python 3.12. Ignore every other code.
        if not self.active:
            return
        self.unwind_callbacks += 1
        if code not in self.selected:
            self.unselected_unwinds += 1
            return
        if threading.get_ident() != self.thread:
            self.other_thread_unwinds += 1
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        self._finish(id(frame), "unwinds")

    def __enter__(self):
        assert not self.started
        self.tool = next(
            (tool for tool in range(5, -1, -1) if self.monitor.get_tool(tool) is None),
            None,
        )
        assert self.tool is not None
        self.monitor.use_tool_id(self.tool, self.TOOL_NAME)
        try:
            assert self.monitor.get_events(self.tool) == 0
            for event, callback in self.callbacks:
                previous = self.monitor.register_callback(self.tool, event, callback)
                self.registered.append((event, callback))
                assert previous is None
            mask = self.monitor.events.PY_START | self.monitor.events.PY_RETURN
            for code in self.selected:
                assert self.monitor.get_local_events(self.tool, code) == 0
                self.monitor.set_local_events(self.tool, code, mask)
                self.local_codes.append(code)
            self.monitor.set_events(self.tool, self.monitor.events.PY_UNWIND)
            self.started = self.active = True
            return self
        except BaseException:
            self._stop()
            raise

    def _stop(self):
        self.active = False
        if self.monitor.get_tool(self.tool) != self.TOOL_NAME:
            self.retirement_errors.append("monitor_owner_changed")
            return
        # Cleanup also covers partial registration/setup failure. Keep attempting
        # every owned retirement step if an earlier one fails.
        actions = [(self.monitor.set_events, (self.tool, 0))]
        actions.extend(
            (self.monitor.set_local_events, (self.tool, code, 0))
            for code in self.local_codes
        )
        expected_callbacks = dict(self.registered)
        actions.extend(
            (self.monitor.register_callback, (self.tool, event, None))
            for event, _ in self.registered
        )
        actions.append((self.monitor.free_tool_id, (self.tool,)))
        for callback, arguments in actions:
            try:
                removed = callback(*arguments)
                if (
                    callback == self.monitor.register_callback
                    and removed is not expected_callbacks[arguments[1]]
                ):
                    self.retirement_errors.append("monitor_callback_changed")
            except BaseException as error:
                self.retirement_errors.append(type(error).__name__)
        self.retired = (
            not self.retirement_errors
            and self.monitor.get_tool(self.tool) is None
            and self.monitor.get_events(self.tool) == 0
            and all(
                self.monitor.get_local_events(self.tool, code) == 0
                for code in self.local_codes
            )
        )

    def __exit__(self, exc_type, exc, traceback):
        self._stop()
        assert self.retired, self.retirement_errors

    def receipt(self):
        return dict(
            diagnostic_only=True,
            timing_acceptance=False,
            measured_scope="one existing warm hook read; current thread only",
            timing_limit="inclusive and overlapping; includes monitoring/profile overhead; do not sum nested rows",
            native_custody="original test assertions and runner receipt remain required",
            source_sha256=self.source_hashes,
            selected_defining_bodies_and_source_files_current=self.defining_bodies_current(),
            binding_scope="selected defining bodies and source files; not every installed dispatch binding",
            selected_code_count=len(self.selected),
            aggregate_key_count=len(self.rows),
            active_slot_limit=self.SLOT_LIMIT,
            overflow=self.overflow,
            unmatched=self.unmatched,
            exit_order_mismatches=self.exit_order_mismatches,
            unfinished=len(self.slots),
            monitoring_retired=self.retired,
            retirement_errors=self.retirement_errors,
            global_event_mask_during_read=self.monitor.events.PY_UNWIND,
            global_event_mask_after_read=(
                self.monitor.get_events(self.tool) if self.tool is not None else None
            ),
            unwind_callbacks=self.unwind_callbacks,
            unselected_unwind_callbacks=self.unselected_unwinds,
            other_thread_selected_unwinds=self.other_thread_unwinds,
            exception_accounting="code-filtered global PY_UNWIND; local PY_START/PY_RETURN",
            no_frame_path_argument_result_or_native_handle_retained=True,
            rows=self.rows,
        )
