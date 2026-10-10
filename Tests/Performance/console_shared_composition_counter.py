"""Opt-in original-code composition counters; no production call replacement."""

import contextvars
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading

import pytest


class CompositionCounter:
    """Observe only selected original bodies and bounded scalar outcomes."""

    def __init__(self):
        from tldw_chatbook.Chat import console_chat_controller as controller
        from tldw_chatbook.Agents import mcp_tool_provider as provider
        from tldw_chatbook.MCP import console_snapshot as snapshot
        from tldw_chatbook.MCP import console_tool_preparation as preparation
        from tldw_chatbook.MCP import permission_store

        self.monitor = sys.monitoring
        self.current_operation = contextvars.ContextVar(
            "diagnostic_composition", default=()
        )
        self.operations = []
        self.selected = {}
        self.bindings = []
        self.files = {}
        self.overflow = 0
        self.tool = None
        self.active = False
        self.main_thread = threading.get_ident()
        self.callbacks = (self.on_start, self.on_return)
        self.events = (self.monitor.events.PY_START, self.monitor.events.PY_RETURN)
        self.mask = self.events[0] | self.events[1]
        targets = (
            (
                controller.ConsoleChatController,
                "_compose_agent_request_providers",
                "composition",
            ),
            (
                controller.ConsoleChatController,
                "_compose_shared_tool_providers",
                "shared",
            ),
            (preparation, "prepare_console_tools", "prepare"),
            (preparation, "_shared_normalization_eligible", "normalization_eligible"),
            (controller, "_stock_console_tool_composition_current", "stock_controller"),
            (provider, "_controller_factory_current", "stock_factory"),
            (provider, "_console_preparation_pipeline_current", "preparation_pipeline"),
            (preparation, "_resolver_current", "stock_resolver"),
            (snapshot, "standard_console_catalog_sources", "stock_catalog_sources"),
            (snapshot, "standard_console_sources", "stock_sources"),
            (permission_store.MCPPermissionStore, "load", "permission_load"),
        )
        for owner, name, label in targets:
            descriptor = inspect.getattr_static(owner, name)
            function = inspect.unwrap(descriptor)
            module = sys.modules[function.__module__]
            assert function.__globals__ is vars(module)
            path = Path(module.__file__).resolve()
            assert Path(function.__code__.co_filename).resolve() == path
            self.bindings.append(
                (owner, name, descriptor, function, function.__code__, module)
            )
            self.files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert function.__code__ not in self.selected
            self.selected[function.__code__] = label

    def _row(self):
        stack = self.current_operation.get()
        if not stack:
            return None
        row = self.operations[stack[-1]]
        # A detached task may inherit diagnostic context but cannot extend the
        # original composition's observation boundary after its real return.
        return row if not row["completed"] else None

    def _event(self, row, **values):
        if len(row["events"]) < 128:
            row["events"].append(values)
        else:
            self.overflow += 1

    @staticmethod
    def _line(code, offset):
        return next(
            (line for start, end, line in code.co_lines() if start <= offset < end),
            None,
        )

    def on_start(self, code, offset):
        label = self.selected[code]
        if label == "composition":
            if len(self.operations) >= 24:
                self.overflow += 1
                return
            frame = sys._getframe(1)
            assert frame.f_code is code
            context = frame.f_locals.get("turn_context")
            maximum = context.mcp_tool_maximum if context is not None else None
            row = dict(
                operation=len(self.operations),
                publish_counts=bool(frame.f_locals.get("publish_mcp_counts", True)),
                maximum_kind=type(maximum).__name__,
                maximum_count=len(maximum) if maximum is not None else None,
                starts={},
                returns={},
                events=[],
                completed=False,
            )
            self.operations.append(row)
            self.current_operation.set(
                (*self.current_operation.get(), row["operation"])
            )
        row = self._row()
        if row is None:
            return
        row["starts"][label] = row["starts"].get(label, 0) + 1
        if label == "permission_load":
            self._event(
                row,
                kind="permission_load_start",
                worker=threading.get_ident() != self.main_thread,
            )
        elif label == "prepare":
            frame = sys._getframe(1)
            assert frame.f_code is code
            self._event(
                row,
                kind="prepare_start",
                include_mcp_catalog=bool(frame.f_locals["include_mcp_catalog"]),
                need_local_switch=bool(frame.f_locals["need_local_switch"]),
            )

    def on_return(self, code, offset, value):
        row = self._row()
        if row is None:
            return
        label = self.selected[code]
        row["returns"][label] = row["returns"].get(label, 0) + 1
        if label in ("prepare", "shared"):
            event = dict(
                kind=label + "_return",
                returned_none=value is None,
                source_line=self._line(code, offset),
            )
            if label == "prepare" and value is not None:
                event.update(
                    tool_count=len(value.tools), kill_switch=bool(value.kill_switch)
                )
            self._event(row, **event)
        elif label not in ("permission_load", "composition") and value is False:
            self._event(
                row,
                kind="qualifier_false",
                helper=label,
                source_line=self._line(code, offset),
            )
        if label == "composition":
            row["completed"] = True
            self.current_operation.set(self.current_operation.get()[:-1])

    def bindings_current(self):
        return all(
            inspect.getattr_static(owner, name) is descriptor
            and inspect.unwrap(descriptor) is function
            and function.__code__ is code
            and function.__globals__ is vars(module)
            and sys.modules.get(module.__name__) is module
            for owner, name, descriptor, function, code, module in self.bindings
        ) and all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            for path, digest in self.files.items()
        )

    def start(self):
        assert self.bindings_current()
        self.tool = next(
            (tool for tool in range(5, -1, -1) if self.monitor.get_tool(tool) is None),
            None,
        )
        assert self.tool is not None
        self.monitor.use_tool_id(self.tool, "console-composition-counts")
        assert self.monitor.get_events(self.tool) == 0
        for event, callback in zip(self.events, self.callbacks, strict=True):
            assert self.monitor.register_callback(self.tool, event, callback) is None
        for code in self.selected:
            assert self.monitor.get_local_events(self.tool, code) == 0
            self.monitor.set_local_events(self.tool, code, self.mask)
        self.active = True

    def stop(self):
        assert self.active and self.monitor.get_events(self.tool) == 0
        for code in self.selected:
            assert self.monitor.get_local_events(self.tool, code) == self.mask
            self.monitor.set_local_events(self.tool, code, 0)
        for event, callback in zip(self.events, self.callbacks, strict=True):
            assert self.monitor.register_callback(self.tool, event, None) is callback
        self.monitor.free_tool_id(self.tool)
        self.active = False
        assert self.monitor.get_tool(self.tool) is None
        assert all(
            self.monitor.get_local_events(self.tool, code) == 0
            for code in self.selected
        )

    def receipt(self):
        return dict(
            diagnostic_only=True,
            timing_acceptance=False,
            original_bindings_and_sources_current=self.bindings_current(),
            source_sha256=self.files,
            selected_code_count=len(self.selected),
            global_events=0,
            monitoring_retired=not self.active
            and self.monitor.get_tool(self.tool) is None,
            overflow=self.overflow,
            operation_limit=24,
            event_limit_per_operation=128,
            operations=self.operations,
            context_propagated_to_workers=True,
            no_frame_receiver_argument_or_return_objects_retained=True,
            native_custody_and_app_completion="owned runner and original diagnostic receipts required",
        )


@pytest.fixture(autouse=True)
def original_composition_counter(request):
    requested = os.environ.get("TLDW_COMPOSITION_COUNTER_RESULT")
    if not requested:
        yield
        return
    from Tests.private_profile import is_private_profile_child

    if not is_private_profile_child(request):
        yield
        return
    target = Path(requested).resolve()
    root = Path(os.environ["RUNNER_TEMP"]).resolve()
    assert target.is_relative_to(root)
    observer = CompositionCounter()
    observer.start()
    try:
        yield
    finally:
        observer.stop()
        target.write_text(json.dumps(observer.receipt(), indent=2), encoding="utf-8")
