"""Observation-only timing hooks for the synthetic Windows UAT."""
from collections import defaultdict
import ctypes
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import threading
import time

import requests

BASE = "def clamp_minutes(minutes):\n    return min(minutes, 60)\n"
FIX = "def clamp_minutes(minutes):\n    return max(0, min(minutes, 60))\n"
DIFF = " def clamp_minutes(minutes):\n-    return min(minutes, 60)\n+    return max(0, min(minutes, 60))\n"


def git(project, *args):
    return subprocess.check_output(
        ["git", "-C", str(project), *args], encoding="utf-8", errors="strict",
    )


def seed_readable(path, data):
    """Use equal NTFS timestamps for the synthetic LF fixture, before Git init."""
    from tldw_chatbook.Utils.windows_files import _BasicInfo, _native

    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    path.write_bytes(data)
    native = _native()
    file_write_attributes = 0x100
    share_read_write_delete = 7
    open_existing = 3
    handle = native.kernel.CreateFileW(
        str(path), file_write_attributes, share_read_write_delete, None, open_existing, 0, None,
    )
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        stamp = 116444736000000000 + time.time_ns() // 100
        info = _BasicInfo(stamp, stamp, stamp, stamp, 0)
        native.check(native.kernel.SetFileInformationByHandle(handle, 0, ctypes.byref(info), ctypes.sizeof(info)))
    finally:
        native.kernel.CloseHandle(handle)
    before = path.stat(follow_symlinks=False)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        opened = os.fstat(descriptor)
        assert os.read(descriptor, len(data) + 1) == data
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        assert all(len({getattr(item, key) for item in (before, opened, after, current)}) == 1 for key in fields)
    finally:
        os.close(descriptor)


class DevModel:
    answer = "Fixed main.py. Git diff confirmed the one-line change."
    names = ("find_tools", "load_tools", "fs_read", "fs_edit", "git_diff")

    def __init__(self, model, alias):
        self.model = model
        self.alias = alias
        self.count = 0

    def reply(self, payload):
        from Tests.uat_project_chat_support import disclosed_schemas, tool_results

        i = self.count
        self.count += 1
        assert i <= len(self.names)
        schemas = disclosed_schemas(payload)
        assert payload.get("tools")
        if i:
            name, content = tool_results(payload)[-1]
            assert name == self.names[i - 1] and not content.startswith("ERROR:"), content
            witness = ("fs_edit", "git_diff", "return min(minutes, 60)",
                       "made 1 replacement", "+    return max(0, min(minutes, 60))")
            assert witness[i - 1] in content, content
        if i == len(self.names):
            return {"role": "assistant", "content": self.answer}
        name = self.names[i]
        assert name in schemas, list(schemas)
        args = {"path": "main.py", "root_alias": self.alias}
        if name == "find_tools":
            args = {"query": "fs_read fs_edit git_diff"}
        elif name == "load_tools":
            args = {"ids": list(self.names[2:])}
        elif name == "fs_edit":
            args.update(old_string="return min(minutes, 60)",
                        new_string="return max(0, min(minutes, 60))")
        assert set(args) <= set(schemas[name]["parameters"]["properties"])
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"uat-dev-{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}


class Trace:
    def __init__(self, artifacts, monkeypatch):
        self.artifacts = artifacts
        progress = os.environ.get("TLDW_TEST_LATENCY_PROGRESS")
        self.progress = Path(progress) if progress else artifacts / "events.jsonl"
        self.mp = monkeypatch
        self.start = time.perf_counter()
        self.events = []
        self.turn = "setup"
        self.screen = None
        self.development = None
        self.feedback = set()
        self.approvals = defaultdict(list)
        self.totals = defaultdict(lambda: [0, 0.0, 0.0])
        self.scope = "pre_send"

    def mark(self, event, *, stamp=None, **data):
        if stamp is None:
            stamp = time.perf_counter()
        row = dict(t=round(stamp - self.start, 6), turn=self.turn, event=event, **data)
        if event in {"http_send", "reply_completed"}:
            names = {"_acquire_storage", "_scope", "_reuse_evidence_hit"}
            counts = dict.fromkeys(sorted(names), 0)
            for (turn, _, function, _), values in tuple(self.totals.items()):
                if turn == self.turn and function in counts:
                    counts[function] += values[0]
            row["admission_counts"] = counts
        self.events.append(row)
        # Live progress can be inspected without logging prompts, paths, or bodies.
        with self.progress.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")

    def watch(self, console, turn):
        self.screen = console
        self.turn = turn
        self.scope = "pre_send"
        self.turn_started = False

    def install(self):
        from Tests.UI.test_console_configuration_worker_feedback import _update_contains
        from textual.app import App
        from textual.pilot import Pilot
        from textual.widgets import Button
        from tldw_chatbook.Agents.agent_models import AgentStep
        from tldw_chatbook.Backup_Recovery import storage_admission
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
        from tldw_chatbook.MCP.permission_store import MCPPermissionStore
        from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor, _parse_worker_output

        button_press = Button.press
        wait_for_screen = Pilot._wait_for_screen
        display = App._display
        agent_step = AgentStep.__init__
        send = requests.Session.send
        write = socketserver._SocketWriter.write
        status = "#console-status-text"
        watched_buttons = {
            "console-send-message": "send", "console-project-notice-proceed": "disclosure_click",
        }

        async def wait_on_slow_windows(pilot, timeout=180.0):
            return await wait_for_screen(pilot, timeout)

        if os.name == "nt":
            self.mp.setattr(Pilot, "_wait_for_screen", wait_on_slow_windows)

        def press(button):
            if self.screen is not None:
                label = watched_buttons.get(button.id)
                if label:
                    if label == "send":
                        self.turn_started = True
                    self.scope = label
                    self.mark(label)
                elif button.has_class("approval-row-fast-approve"):
                    self.mark("approval_click")
            return button_press(button)

        def paint(app, screen, renderable):
            result = display(app, screen, renderable)
            when = time.perf_counter()
            if self.screen is None or not self.turn_started or renderable is None or app.screen is not screen:
                return result
            if screen is self.screen and self.turn not in self.feedback:
                widget = next(iter(screen.query(status)), None)
                if widget is not None and _update_contains(renderable, widget.region, "Sending"):
                    self.feedback.add(self.turn)
                    self.mark("painted_sending", stamp=when)
            elif screen is not self.screen:
                button = next(iter(screen.query("#console-project-notice-proceed")), None)
                if button is not None and "disclosure" not in self.feedback and _update_contains(
                    renderable, button.region, "Proceed"
                ):
                    self.feedback.add("disclosure")
                    self.mark("painted_disclosure", stamp=when)
            if screen is self.screen:
                for row in screen.query(".approval-row"):
                    button = next(iter(row.query(".approval-row-fast-approve")), None)
                    if button is None:
                        continue
                    key = self.turn, row.id or id(row)
                    if key in self.feedback:
                        continue
                    if any(
                        _update_contains(renderable, button.region, label)
                        for label in ("Allow once", "Once")
                    ):
                        self.feedback.add(key)
                        self.mark("painted_approval", stamp=when)
            return result

        steps = {
            "model_request_started", "model_response_completed", "tool_proposed",
            "approval_requested", "approval_approved", "tool_execution_started",
            "tool_succeeded", "tool_failed", "agent_run_created", "agent_run_started",
            "agent_run_completed",
        }

        def init_step(step, *args, **kwargs):
            agent_step(step, *args, **kwargs)
            if self.screen is not None and step.kind in steps:
                self.scope = step.kind + (":" + step.tool_name if step.tool_name else "")
                self.mark(step.kind, tool=step.tool_name, call=step.call_id)

        def outbound(client, request, **kwargs):
            if request.method == "POST" and request.url.startswith("http://127.0.0.1:") and request.url.endswith("/v1/chat/completions"):
                self.mark("http_send")
            return send(client, request, **kwargs)

        def response(writer, buffer):
            result = write(writer, buffer)
            if buffer.endswith(b"data: [DONE]\n\n") or b'"object": "chat.completion"' in buffer:
                self.mark("http_response_sent")
            return result

        publish = ConsoleChatStore._publish_message_completed

        def complete(store, session, message):
            item = next((m for m in store.messages_for_session(session) if m.id == message), None)
            if self.screen is not None and item is not None and item.role == "assistant":
                self.mark("reply_completed", persisted=bool(item.persisted_message_id))
            return publish(store, session, message)

        self.mp.setattr(Button, "press", press)
        self.mp.setattr(App, "_display", paint)
        self.mp.setattr(AgentStep, "__init__", init_step)
        self.mp.setattr(requests.Session, "send", outbound)
        self.mp.setattr(socketserver._SocketWriter, "write", response)
        self.mp.setattr(ConsoleChatStore, "_publish_message_completed", complete)
        targets = {
            MCPPermissionStore.load.__wrapped__: False,
            MCPPermissionStore.read_snapshot_strict.__wrapped__: False,
            storage_admission._acquire_storage: False,
            storage_admission._scope: False,
            storage_admission._reuse_evidence: False,
            WorkspaceToolExecutor.execute: True,
            WorkspaceToolExecutor._build_request: True,
            _parse_worker_output: True,
        }
        from tldw_chatbook.Chat import console_tool_surface_preview
        for name in ("preflight_tool_surface", "owned_preflight_tool_surface"):
            observe = getattr(console_tool_surface_preview, name, None)
            if observe is not None:
                targets[observe] = True
        self.monitor(targets)

    def monitor(self, targets):
        monitoring = sys.monitoring
        tool = next(i for i in range(6) if monitoring.get_tool(i) is None)
        monitoring.use_tool_id(tool, "workspace-latency")
        self.monitor_id = tool
        codes = {f.__code__: (f.__qualname__, boundary) for f, boundary in targets.items()}
        self.monitor_codes = codes
        starts = {}

        def begin(code, _offset):
            stamp = time.perf_counter()
            name, boundary = codes[code]
            frame = sys._getframe(1)
            starts[id(frame)] = stamp, self.turn, self.scope
            if boundary:
                self.mark(name + "_begin", stamp=stamp)

        def end(code, _offset, value):
            stamp = time.perf_counter()
            recorded = starts.pop(id(sys._getframe(1)), None)
            if recorded is None:
                return
            start, turn, phase = recorded
            name, boundary = codes[code]
            group = self.totals[turn, phase, name, threading.current_thread().name]
            elapsed = stamp - start
            group[0] += 1
            group[1] += elapsed
            group[2] = max(group[2], elapsed)
            if name == "_reuse_evidence" and value is not None:
                self.totals[turn, phase, "_reuse_evidence_hit", threading.current_thread().name][0] += 1
            if boundary:
                extra = {"worker_ms": getattr(value, "elapsed_ms", None)} if name == "_parse_worker_output" else {}
                self.mark(name + "_end", stamp=stamp, **extra)

        monitoring.register_callback(tool, monitoring.events.PY_START, begin)
        monitoring.register_callback(tool, monitoring.events.PY_RETURN, end)
        for code in codes:
            monitoring.set_local_events(tool, code, monitoring.events.PY_START | monitoring.events.PY_RETURN)

    def attach_model(self, model):
        reply = model.reply
        def wrapped(payload):
            self.mark("http_request_received")
            try:
                return reply(payload) if self.development is None else self.development.reply(payload)
            finally:
                self.mark("script_reply_ready")
        self.mp.setattr(model, "reply", wrapped)

    async def wait_with_approvals(self, pilot, console, controller, session, server, final, click, *, main=None):
        deadline = time.perf_counter() + 3500
        answered = set()
        while time.perf_counter() < deadline:
            assert not server.errors, server.errors
            result = final()
            if result is not None:
                return result
            rows = list(console.query(".approval-row"))
            if rows and all(row.is_mounted for row in rows):
                pending = controller._head_round_payload(controller._parked_approval_payloads, session.id)
                if pending is not None:
                    [call] = pending["calls"]
                    call_id = call["call_id"]
                    name = call["tool_name"]
                    assert name in ("find_tools", "load_tools", "fs_list", "fs_read", "fs_edit", "git_diff")
                    if call_id not in answered:
                        answered.add(call_id)
                        self.approvals[self.turn].append(name)
                        self.mark("approval_observed", tool=name)
                        if name == "fs_edit":
                            assert main is not None and main.read_bytes() == BASE.encode()
                        await click(pilot, ".approval-row-fast-approve", server)
            await pilot.pause(0.05)
        raise AssertionError("timed out waiting for completed reply or tool approval")

    def finish(self):
        if hasattr(self, "monitor_id"):
            monitoring = sys.monitoring
            for code in self.monitor_codes:
                monitoring.set_local_events(self.monitor_id, code, 0)
            monitoring.register_callback(self.monitor_id, monitoring.events.PY_START, None)
            monitoring.register_callback(self.monitor_id, monitoring.events.PY_RETURN, None)
            monitoring.free_tool_id(self.monitor_id)
        report = [
            dict(turn=turn, phase=phase, function=name, thread=thread, count=count,
                 total_s=round(total, 6), max_s=round(maximum, 6))
            for (turn, phase, name, thread), (count, total, maximum) in sorted(self.totals.items())
        ]
        (self.artifacts / "costs.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

