"""Prepared real native TldwCli child; no fake transport/paint/input facts.

No approval verdict command exists. Actual gestures come only from authorized
owned-window OS input in the outer observer. Snapshot is semantic ONLY. A real
independently reviewed capture must prove actual card/action/feedback pixels.
"""

from __future__ import annotations
import argparse
import asyncio
import faulthandler
from ctypes import wintypes as W
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import qa_process_guard as guard


def write_json(path, value):
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class StageWatchdog:
    """One diagnostic timer; never interprets its dump as a stalled owner."""

    def __init__(self, directory, fail):
        self.directory = Path(directory)
        self.fail = fail
        self.stream = None
        self.used = False
        self.armed = False
        self.observer_failure = False
        self.events = []

    def _fail(self):
        self.observer_failure = True
        try:
            self.fail()
        except BaseException:
            pass

    def _record(self, event, reason, started):
        try:
            self.events.append(
                {
                    "event": event,
                    "reason": reason,
                    "operation_start_ns": started,
                    "timestamp_ns": time.perf_counter_ns(),
                }
            )
            write_json(
                self.directory / "native-stage-watchdog.json",
                {
                    "events": self.events,
                    "observer_failure": self.observer_failure,
                    "one_shot_seconds": 20,
                    "repeat": False,
                    "dump_is_not_stall_or_cause_proof": True,
                    "qualified": False,
                    "product_samples": 0,
                },
            )
        except BaseException:
            self._fail()

    def arm(self):
        if self.used:
            return
        self.used = True
        started = time.perf_counter_ns()
        try:
            self.stream = (self.directory / "native-stage-stall-stacks.txt").open(
                "x", encoding="utf-8"
            )
            faulthandler.dump_traceback_later(20, file=self.stream, repeat=False)
            self.armed = True
            self._record("armed", "before_original_fixture_stage", started)
        except BaseException:
            self._fail()
            if not self.armed and self.stream is not None:
                try:
                    self.stream.close()
                except BaseException:
                    self._fail()

    def cancel(self, reason):
        if not self.armed:
            return
        started = time.perf_counter_ns()
        try:
            faulthandler.cancel_dump_traceback_later()
        except BaseException:
            self._fail()
            return  # Keep its descriptor open while cancellation is unproven.
        self.armed = False
        try:
            self.stream.close()
        except BaseException:
            self._fail()
        self._record("cancelled", reason, started)

    def watch(self, task):
        try:
            if task is None:
                self.cancel("no_original_task")
                return
            task.add_done_callback(lambda _task: self.cancel("original_task_done"))
        except BaseException:
            self._fail()
            self.cancel("task_observation_failed")


def main():
    write_json(
        guard.RUN / "native-preflight-stdio.json",
        {
            "pid": os.getpid(),
            "launcher_parent_pid": os.getppid(),
            "stdin_tty": sys.__stdin__.isatty(),
            "stdout_tty": sys.__stdout__.isatty(),
            "stderr_tty": sys.__stderr__.isatty(),
            "isolated": bool(sys.flags.isolated),
            "before_native_gates": True,
            "before_product_imports": True,
            "qualified": False,
            "product_samples": 0,
        },
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-native-product", action="store_true")
    parser.add_argument("--columns", required=True, type=int)
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--shape", choices=("single", "batch", "raw-deny", "large"))
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument(
        "--diagnostic-stack", action=argparse.BooleanOptionalAction, default=None
    )
    arguments = parser.parse_args()
    from native_series_method import bind_diagnostic_stack

    diagnostic_stack_enabled = bind_diagnostic_stack(
        guard.SELECTION, arguments.diagnostic_stack
    )
    if not arguments.allow_native_product:
        parser.error(
            "Explicit alternative-method authorization and root freeze/lease required"
        )
    if os.name != "nt" or not sys.flags.isolated:
        parser.error("Private isolated Windows interpreter required")
    if len(arguments.nonce) != 32 or any(
        value not in "0123456789abcdef" for value in arguments.nonce
    ):
        parser.error("Internally minted nonce required")
    if bool(arguments.shape) == bool(arguments.negative_controls):
        parser.error("Choose ONE shape process or a dedicated negative-control process")

    def startup_phase(stage):
        with (guard.RUN / "native-startup-phases.jsonl").open(
            "a", encoding="utf-8"
        ) as output:
            output.write(
                json.dumps(
                    {
                        "stage": stage,
                        "timestamp_ns": time.perf_counter_ns(),
                        "pid": os.getpid(),
                    }
                )
                + "\n"
            )

    startup_phase("freeze_verify_before")
    frozen = guard.verify_frozen()  # BEFORE any product import.
    from native_series_method import authorized_round_limit

    round_limit = authorized_round_limit(guard.SELECTION, guard.RUN)
    startup_phase("freeze_verify_after")
    from native_window_api import (
        win32,
        geometry,
        console_state,
        resize_owned_console,
        clock_bounds,
    )

    user, kernel, _ = win32()
    hwnd = kernel.GetConsoleWindow()
    if not hwnd or not user.IsWindowVisible(hwnd) or user.IsIconic(hwnd):
        raise RuntimeError(
            "No own visible native console; ConPTY is not physical presentation"
        )
    if not sys.__stdin__.isatty() or not sys.__stdout__.isatty():
        raise RuntimeError("Actual native console handles required")
    if geometry(user, hwnd)["class"] != "ConsoleWindowClass":
        raise RuntimeError("Only verified classic conhost mapping supported")
    kernel.SetConsoleTitleW.argtypes = [W.LPCWSTR]
    kernel.SetConsoleTitleW.restype = W.BOOL
    if not kernel.SetConsoleTitleW("PRIVATE QA CONSOLE " + arguments.nonce):
        raise RuntimeError("Own console nonce title refused")
    resized = resize_owned_console(
        user, kernel, hwnd, arguments.columns, arguments.rows
    )
    write_json(
        guard.RUN / "native-pre-app-geometry.json",
        {
            "native": resized,
            "stdio": {
                "stdin": sys.__stdin__.isatty(),
                "stdout": sys.__stdout__.isatty(),
                "stderr": sys.__stderr__.isatty(),
            },
            "coordinate_qualified": False,
            "qualified": False,
            "product_samples": 0,
        },
    )
    write_json(
        guard.RUN / "native-bootstrap-window.json",
        {
            "pid": os.getpid(),
            "launcher_parent_pid": os.getppid(),
            "hwnd": hwnd,
            "nonce": arguments.nonce,
            "revision": frozen["revision"],
            "native": resized,
            "pre_product_imports": True,
            "qualified": False,
            "product_samples": 0,
        },
    )
    startup_phase("fixture_import_before")
    from approval_fixture_workload_v7 import ScriptedProvider, ApprovalFixtureDriver

    startup_phase("fixture_import_after")
    provider = ScriptedProvider().start()
    guard.allow_loopback_port(
        provider.server.server_port, "scripted_provider", owner_pid=os.getpid()
    )
    os.environ["TLDW_CANVAS_ENABLED"] = (
        "false"  # Supported terminal-only condition, same both arms.
    )
    from tldw_chatbook.Backup_Recovery import bootstrap

    startup_phase("startup_admission_before")
    admission = bootstrap.startup_permission(
        bootstrap.effective_config_path(), bootstrap.default_bootstrap_root()
    )
    startup_phase("startup_admission_after")
    if not admission[0]:
        raise RuntimeError("Actual startup admission refused:" + str(admission))
    from tldw_chatbook.config import save_settings_to_cli_config, get_cli_setting

    settings = provider.offline_settings()
    settings["canvas"] = {"enabled": False}
    if not save_settings_to_cli_config(settings):
        raise RuntimeError("Canonical private config writer refused")
    for section, values in settings.items():
        for key, expected in values.items():
            if get_cli_setting(section, key, default=None) != expected:
                raise RuntimeError(
                    "Canonical private config reread mismatch:" + section + "." + key
                )
    from loguru import logger

    logger.remove()
    logger.add(guard.RUN / "native-product.log", level="INFO")
    from textual.driver import Driver
    from textual import events

    startup_phase("app_import_before")
    from tldw_chatbook.app import TldwCli

    startup_phase("app_import_after")
    # Match the shipped native entry before WindowsDriver owns console input.
    from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol

    startup_phase("image_protocol_warmup_before")
    warm_up_image_protocol()
    startup_phase("image_protocol_warmup_after")
    startup_phase("app_constructor_before")
    app = TldwCli()
    startup_phase("app_constructor_after")
    from native_rehearsal_observation import EventSink, CardEvents

    sink = EventSink(guard.RUN)
    stage_watchdog = StageWatchdog(guard.RUN, sink.fail)
    fixture = ApprovalFixtureDriver(
        app, guard.PROFILE / "fixture", emit=sink.emit, provider=provider
    )
    card_events = None
    original_input = Driver.process_message
    input_spans = []

    def process_message(driver, message):
        started = time.perf_counter_ns()
        current = fixture.current
        owning = getattr(driver, "_app", None) is app
        interesting = isinstance(
            message, (events.Key, events.MouseDown, events.MouseUp)
        )
        if owning and interesting and current:
            fixture._record(
                "app_input_prequeue",
                current["correlation"],
                timestamp_ns=started,
                kind=type(message).__name__,
            )
        answer = original_input(driver, message)  # Original prequeue/offset/queue body.
        if owning and interesting and current:
            input_spans.append([started, time.perf_counter_ns()])
        return answer

    # WindowsDriver passes its bound process_message into the Win32 input loop
    # at start, so observation MUST be installed before app.run, not afterwards.
    Driver.process_message = process_message
    control_token = arguments.nonce
    stage_count = 0
    state = {"ready": False, "errors": [], "negative_results": None}
    common_observer = None

    def observer_native_state():
        value = console_state(user, kernel, hwnd)
        value["request_summary_cell_region"] = None
        value["request_summary_selector"] = None
        # Select an ACTUAL owned rendered widget. The old card has row args,
        # not the new feature's request-summary ID; no geometry/paint is made up.
        for selector in (
            "#approval-request-summary",
            ".approval-row-args",
            ".approval-row-full-command",
            "#approval-batch-rows",
        ):
            try:
                summaries = list(app.screen.query(selector))
                if len(summaries) != 1:
                    continue
                summary = summaries[0]
                visible = app.screen._compositor.visible_widgets
                bounds, clip = visible[summary]
                if (
                    bounds.width > 0
                    and bounds.height > 0
                    and bounds.intersection(clip) == bounds
                ):
                    value["request_summary_cell_region"] = [
                        bounds.x,
                        bounds.y,
                        bounds.width,
                        bounds.height,
                    ]
                    value["request_summary_selector"] = selector
                    break
            except Exception:
                continue
        return value

    def capture_state():
        full = fixture.snapshot()
        keep = (
            "visible_send_lifecycle",
            "card_identity",
            "card_identity_source",
            "feature_presentation_revision",
            "phase",
            "card_display",
            "card_attached",
            "information_ready",
            "submitted",
            "card_region",
            "controls",
            "feedback",
            "theme",
            "size",
        )
        view = {key: full.get(key) for key in keep}
        view["current"] = {
            key: (full.get("current") or {}).get(key)
            for key in ("correlation", "run_id", "shape")
        }
        view["current"]["session_id"] = (
            fixture.session.id if fixture.session is not None else None
        )
        rail = app.screen.query_one("#console-inspector-rail-body")
        visible = app.screen._compositor.visible_widgets
        view["inspect_open"] = bool(
            rail.display
            and rail in visible
            and visible[rail][0].width > 0
            and visible[rail][0].height > 0
        )
        native = observer_native_state()
        native = {
            key: native[key]
            for key in (
                "font",
                "columns",
                "rows",
                "geometry",
                "request_summary_cell_region",
                "request_summary_selector",
            )
            if key in native
        }
        title = app.screen.query_one("#approval-title")
        if title in visible:
            bounds, clip = visible[title]
            if (
                bounds.width > 0
                and bounds.height > 0
                and bounds.intersection(clip) == bounds
            ):
                native["feedback_title_cell_region"] = [
                    bounds.x,
                    bounds.y,
                    bounds.width,
                    bounds.height,
                ]
        settlement = {
            "owner_terminal": False,
            "session_matches": False,
            "card_retired": False,
            "header_terminal": False,
            "send_terminal": False,
            "stop_hidden": False,
            "run_segment_absent": False,
            "title_generation_matches": False,
            "rois": {},
        }
        if fixture.current and fixture.session is not None:
            current = fixture.current
            controller = fixture.controller
            task = full.get("submission_task") or {}
            submitted = task.get("result") or {}
            settlement["owner_terminal"] = (
                full.get("run_status") == "completed"
                and task.get("done") is True
                and submitted.get("accepted") is True
                and submitted.get("terminal_status") == "completed"
            )
            settlement["session_matches"] = (
                controller.store.active_session_id == fixture.session.id
            )
            settlement["card_retired"] = (
                view.get("card_display") is False and view.get("card_identity") is None
            )
            widgets = {
                key: app.screen.query_one(selector)
                for key, selector in {
                    "generation": "#console-transcript-title",
                    "header": "#console-workbench-header #workbench-header-status",
                    "send": "#console-send-message",
                    "provider": "#console-status-text",
                }.items()
            }
            stop = app.screen.query_one("#console-stop-generation")
            settlement["stop_hidden"] = not stop.display and stop not in visible
            header = widgets["header"].content
            header = getattr(header, "plain", header)
            send = widgets["send"].label
            send = getattr(send, "plain", send)
            settlement["header_terminal"] = (
                type(header) is str  # noqa: E721 - Exact builtin labels only.
                and header in ("Ready", "Ready · not tested")
            )
            settlement["send_terminal"] = (
                type(send) is str  # noqa: E721 - Exact boundary types reject bool/subclasses.
                and send == "Send"
                and widgets["send"].disabled is True
            )
            title = widgets["generation"].content
            title = getattr(title, "plain", title)
            expected = "Conversation | QA-" + current["correlation"][:12]
            settlement["title_generation_matches"] = (
                type(title) is str and title == expected  # noqa: E721 - Exact boundary types reject bool/subclasses.
            )
            line = widgets["provider"]
            spans = line.segment_spans
            settlement["run_segment_absent"] = not any(
                name == "run" for name, _start, _end in spans
            )
            for name, widget in widgets.items():
                if widget not in visible:
                    continue
                bounds, clip = visible[widget]
                if (
                    bounds.width <= 0
                    or bounds.height != 1
                    or (name != "provider" and bounds.intersection(clip) != bounds)
                ):
                    continue
                region = [bounds.x, bounds.y, bounds.width, bounds.height]
                content = widget.content_region
                if name == "generation":
                    if (
                        not settlement["title_generation_matches"]
                        or content.height != 1
                        or content.width < 30
                        or content.intersection(clip) != content
                    ):
                        continue
                    region = [content.x + 18, content.y, 12, 1]
                elif name == "provider":
                    if content.height != 1:
                        continue
                    provider = [
                        (start, end)
                        for label, start, end in spans
                        if label == "provider"
                    ]
                    scroll = app.screen.query_one("#console-status-chip-scroll")
                    if len(provider) != 1 or scroll.scroll_x != 0:
                        continue
                    start, end = provider[0]
                    if start < 0 or end <= start or end > content.width:
                        continue
                    provider_region = type(content)(
                        content.x + start, content.y, end - start, 1
                    )
                    if any(
                        provider_region.intersection(area) != provider_region
                        for area in (content, bounds, clip)
                    ):
                        continue
                    region = [content.x + start, content.y, end - start, 1]
                settlement["rois"][name] = region
        answer = {
            "snapshot": view,
            "native": native,
            "settlement": settlement,
            "server_ns": time.perf_counter_ns(),
            "observer_health": sink.health(),
            "rehearsal_scope": "feature_single_120x40_closed_dark",
        }
        return answer

    async def command(value):
        nonlocal stage_count
        kind = value["command"]
        if kind == "capture-state":
            return capture_state()
        if kind == "snapshot":
            pass
        elif kind == "configure":
            await fixture.configure(
                theme=value["theme"], inspect_open=value["inspect_open"]
            )
        elif kind == "stage":
            if arguments.negative_controls or value["shape"] != arguments.shape:
                raise RuntimeError("One selected shape per metric process")
            if (
                arguments.shape != "single"
                or arguments.columns != 120
                or arguments.rows != 40
                or stage_count >= round_limit
            ):
                raise RuntimeError(
                    "Fixed validation1 or authorized qualification41 bound"
                )
            stage_count += 1
            correlation = fixture.new_correlation()
            if diagnostic_stack_enabled:
                stage_watchdog.arm()
            try:
                await fixture.stage(arguments.shape, correlation)
            finally:
                if diagnostic_stack_enabled:
                    stage_watchdog.watch(fixture.worker)
        elif kind == "collect":
            stage_watchdog.cancel("collect")
            collected = await fixture.collect()
            if card_events is not None:
                card_events.schedule()
            return {
                "snapshot": collected,
                "common_activity": common_observer.snapshot()
                if common_observer is not None
                else None,
                "native": observer_native_state(),
                "server_ns": time.perf_counter_ns(),
                "observer_health": sink.health(),
            }
        elif kind == "trace-negatives":
            if not arguments.negative_controls:
                raise RuntimeError("Dedicated fresh negative-control process required")
            state["negative_results"] = await fixture.trace_admission_negatives()
        elif kind == "hide":
            user.ShowWindow(hwnd, 0)
        elif kind == "show":
            user.ShowWindow(hwnd, 5)
        elif kind == "exit":
            stage_watchdog.cancel("exit")
            await fixture.close()  # Original runtime/controller drains first.
            if common_observer is not None:
                state["common_activity_exit"] = common_observer.snapshot()
                common_observer.close()  # Restore only after actual close completed.
            app.exit()
            return {"exiting": True}
        else:
            raise ValueError("Unsupported bounded QA command")
        return {
            "snapshot": fixture.snapshot(),
            "common_activity": common_observer.snapshot()
            if common_observer is not None
            else None,
            "native": observer_native_state(),
            "server_ns": time.perf_counter_ns(),
            "negative_results": state["negative_results"],
        }

    def diagnostic_status():
        modules = {}
        for name, module in tuple(sys.modules.items()):
            if not name.startswith(guard.PRODUCT_PREFIXES):
                continue
            filename = getattr(module, "__file__", None)
            if not filename:
                continue
            source = Path(filename).resolve()
            within = source.is_relative_to(guard.REPO)
            relative = source.relative_to(guard.REPO).as_posix() if within else None
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            modules[name] = {
                "path": str(source),
                "within_selected_root": within,
                "sha256": digest,
                "matches_frozen_source": within
                and frozen["sha256"].get(relative) == digest,
            }
        value = {
            "pid": os.getpid(),
            "nonce": control_token,
            "ui_ready": getattr(app, "_ui_ready", None),
            "driver": type(getattr(app, "_driver", None)).__name__,
            "state": state,
            "modules": modules,
            "guard_file_sha256": hashlib.sha256(
                Path(guard.real_profile_guard.__file__).read_bytes()
            ).hexdigest(),
            "network": guard.NETWORK,
            "profile_refusals": len(guard.real_profile_guard._violations),
            "qualified": False,
            "product_samples": 0,
        }
        write_json(guard.RUN / "native-diagnostic-status.json", value)
        return value

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer " + control_token:
                self.send_error(403)
                return
            value = (
                diagnostic_status()
                if self.path == "/diagnostic-status"
                else {
                    "pid": os.getpid(),
                    "nonce": control_token,
                    "server_ns": time.perf_counter_ns(),
                }
            )
            raw = json.dumps(value).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + control_token:
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("QA command size bound")
                request = json.loads(self.rfile.read(size))
                answer = app.call_from_thread(command, request)
                status = 200
            except BaseException as exc:
                answer = {
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "server_ns": time.perf_counter_ns(),
                }
                status = 500
                state["errors"].append(
                    {
                        "type": type(exc).__name__,
                        "reason": str(exc),
                        "command": request.get("command")
                        if isinstance(locals().get("request"), dict)
                        else None,
                    }
                )
            raw = json.dumps(answer).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    control = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    control.daemon_threads = True
    thread = threading.Thread(
        target=control.serve_forever, name="private-native-control", daemon=True
    )
    thread.start()
    guard.allow_loopback_port(
        control.server_port, "native_control", owner_pid=os.getpid()
    )

    def actual_driver_proof():
        from types import CodeType
        from textual.drivers.windows_driver import WindowsDriver
        from tldw_chatbook.Utils.text_selection_crash_guard import TextualAppGuards

        actual = type(app._driver)
        factory = TextualAppGuards.get_driver_class
        nested = next(
            code
            for code in factory.__code__.co_consts
            if isinstance(code, CodeType) and code.co_name == "WindowsResizeDriver"
        )
        expected = next(
            code
            for code in nested.co_consts
            if isinstance(code, CodeType) and code.co_name == "start_application_mode"
        )
        stock_subclass = (
            actual.__module__ == TextualAppGuards.__module__
            and actual.__qualname__
            == factory.__qualname__ + ".<locals>.WindowsResizeDriver"
            and actual.__bases__ == (WindowsDriver,)
            and actual.start_application_mode.__code__ is expected
        )
        valid = (
            isinstance(app._driver, WindowsDriver)
            and (actual is WindowsDriver or stock_subclass)
            and not app.is_headless
        )
        source = guard.REPO / "tldw_chatbook/Utils/text_selection_crash_guard.py"
        return {
            "valid_stock_native_driver": valid,
            "class": actual.__name__,
            "module": actual.__module__,
            "qualname": actual.__qualname__,
            "mro": [
                item.__module__ + "." + item.__qualname__ for item in actual.__mro__
            ],
            "stock_factory_code_identity": stock_subclass,
            "factory_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "headless": app.is_headless,
        }

    async def prepare():
        nonlocal common_observer, card_events
        startup_phase("actual_first_refresh_callback")
        try:
            native = console_state(user, kernel, hwnd)
            driver_proof = actual_driver_proof()
            write_json(
                guard.RUN / "native-window.json",
                {
                    "pid": os.getpid(),
                    "launcher_parent_pid": os.getppid(),
                    "hwnd": hwnd,
                    "nonce": arguments.nonce,
                    "native": native,
                    "control_port": control.server_port,
                    "provider_port": provider.server.server_port,
                    "driver": type(app._driver).__name__,
                    "headless": app.is_headless,
                    "admission": admission,
                    "revision": frozen["revision"],
                    "clock": clock_bounds(kernel),
                    "shape": arguments.shape,
                    "negative_controls": arguments.negative_controls,
                    "attach_complete": False,
                    "driver_source_proof": driver_proof,
                    "qualified": False,
                    "product_samples": 0,
                },
            )
            if not driver_proof["valid_stock_native_driver"]:
                raise RuntimeError(
                    "Actual production native driver MRO/source identity refused"
                )
            exact = native["geometry"]["client"] == [
                native["columns"] * native["font"][0],
                native["rows"] * native["font"][1],
            ]
            write_json(
                guard.RUN / "native-post-driver-geometry.json",
                {
                    "native": native,
                    "driver": type(app._driver).__name__,
                    "after_actual_first_refresh": True,
                    "exact_cell_map": exact,
                    "stdio": {
                        "stdin": sys.__stdin__.isatty(),
                        "stdout": sys.__stdout__.isatty(),
                        "stderr": sys.__stderr__.isatty(),
                    },
                    "qualified": False,
                    "product_samples": 0,
                },
            )
            if not exact:
                raise RuntimeError(
                    "Actual WindowsDriver first-refresh cell-to-pixel mapping mismatch; no coordinate qualification"
                )
            await fixture.attach()
            from common_activity_observer import CommonActivityObserver

            common_observer = CommonActivityObserver(
                fixture
            )  # No feature/ToolActivity imports.
            rail_screen = app.screen
            await fixture.configure(theme="textual-dark", inspect_open=False)
            rail_workers = tuple(
                worker
                for worker in app.workers
                if worker.node is rail_screen
                and worker.name == "_save_console_rail_preferences"
            )
            for worker in rail_workers:
                await worker.wait()
            card = app.screen.query_one("#chat-approval-card")
            card_events = CardEvents(card, fixture, sink, capture_state)
            original_resolve = fixture.controller.resolve_pending_approval

            def observe_host_return(*args, **kwargs):
                answer = original_resolve(*args, **kwargs)
                try:
                    if fixture.current:
                        fixture._record(
                            "host_resolve_return",
                            fixture.current["correlation"],
                            host_returned=True,
                        )
                except BaseException:
                    sink.fail()
                return answer

            fixture.controller.resolve_pending_approval = observe_host_return
            if not actual_driver_proof()["valid_stock_native_driver"]:
                raise RuntimeError(
                    "Actual production native driver MRO/source identity changed"
                )
            native = console_state(user, kernel, hwnd)
            if (
                native["columns"] != arguments.columns
                or native["rows"] != arguments.rows
                or [app.size.width, app.size.height]
                != [arguments.columns, arguments.rows]
            ):
                raise RuntimeError("App/physical grid mismatch")
            state["ready"] = True
            write_json(
                guard.RUN / "native-ready.json",
                {
                    "pid": os.getpid(),
                    "launcher_parent_pid": os.getppid(),
                    "hwnd": hwnd,
                    "nonce": arguments.nonce,
                    "native": native,
                    "control_port": control.server_port,
                    "provider_port": provider.server.server_port,
                    "driver": type(app._driver).__name__,
                    "headless": app.is_headless,
                    "admission": admission,
                    "revision": frozen["revision"],
                    "clock": clock_bounds(kernel),
                    "shape": arguments.shape,
                    "negative_controls": arguments.negative_controls,
                    "qualified": False,
                    "product_samples": 0,
                },
            )
            app.set_timer(600, app.exit)
        except BaseException as exc:
            write_json(
                guard.RUN / "native-setup-error.json",
                {
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "qualified": False,
                    "product_samples": 0,
                },
            )
            # Keep the owned window until the observer captures the typed failure.
            deadline = time.monotonic() + 5
            while (
                not (guard.RUN / "native-failure-capture-ack.json").exists()
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            app.exit()

    app.call_after_refresh(
        lambda: app.run_worker(prepare(), exclusive=True, group="qa-native-prepare")
    )
    startup_phase("app_run_before")
    try:
        app.run()
    finally:
        stage_watchdog.cancel("app_finally")
        startup_phase("app_run_after")
        Driver.process_message = original_input
        control.shutdown()
        control.server_close()
        thread.join(timeout=3)
        if not fixture._closed:
            # Failures are retained; no fabricated verdict or forced release.
            # The actual runtime disposal is still app.run's normal teardown.
            provider.close()
        try:
            if card_events is not None:
                card_events.close()
            sink.close()
        except BaseException as observation_error:
            state["observer_cleanup_error_type"] = type(observation_error).__name__
            try:
                sink.fail()
            except BaseException:
                pass
        write_json(
            guard.RUN / "native-child-exit.json",
            {
                "pid": os.getpid(),
                "ready": state["ready"],
                "errors": state["errors"],
                "app_return_code": app.return_code,
                "app_exception_type": type(app._exception).__name__
                if app._exception
                else None,
                "stages": fixture.events,
                "observations": fixture._owner_observations,
                "input_observer_spans_ns": input_spans,
                "negative_results": state["negative_results"],
                "diagnostic_stack": diagnostic_stack_enabled,
                "common_activity_after_actual_close": state.get("common_activity_exit"),
                "observer_health": sink.health(),
                "stage_count": stage_count,
                "scripted_turns": provider.scripted_turns,
                "chat_post_entries": provider.chat_post_entries,
                "clock_after": clock_bounds(kernel),
                "qualified": False,
                "product_samples": 0,
            },
        )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        import traceback

        write_json(
            guard.RUN / "native-early-error.json",
            {
                "type": type(error).__name__,
                "reason": str(error),
                "traceback": traceback.format_exc(),
                "pid": os.getpid(),
                "stdin_tty": sys.__stdin__.isatty(),
                "stdout_tty": sys.__stdout__.isatty(),
                "stderr_tty": sys.__stderr__.isatty(),
                "qualified": False,
                "product_samples": 0,
            },
        )
        try:
            from native_window_api import win32, geometry

            user, kernel, _ = win32()
            hwnd = kernel.GetConsoleWindow()
            nonce = sys.argv[sys.argv.index("--nonce") + 1]
            if hwnd and user.IsWindowVisible(hwnd):
                write_json(
                    guard.RUN / "native-early-window.json",
                    {
                        "pid": os.getpid(),
                        "launcher_parent_pid": os.getppid(),
                        "hwnd": hwnd,
                        "nonce": nonce,
                        "geometry": geometry(user, hwnd),
                        "qualified": False,
                        "product_samples": 0,
                    },
                )
        except BaseException as capture_setup:
            write_json(
                guard.RUN / "native-early-window-error.json",
                {
                    "type": type(capture_setup).__name__,
                    "reason": str(capture_setup),
                    "qualified": False,
                    "product_samples": 0,
                },
            )
        print(
            "PRIVATE QA native startup refused: " + type(error).__name__,
            file=sys.__stderr__,
            flush=True,
        )
        deadline = time.monotonic() + 5
        while (
            not (guard.RUN / "native-early-capture-ack.json").exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        raise
