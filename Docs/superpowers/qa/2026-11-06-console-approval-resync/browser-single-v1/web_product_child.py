"""Frozen-source real TldwCli served child; fixture HTTP only, no gateway overrides."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from qa_process_guard import RUN, REPO, allow_loopback_port, verify_frozen

FROZEN = verify_frozen()
SELECTION = json.loads((RUN / "source-selection.json").read_text(encoding="utf-8-sig"))
for name, expected in SELECTION["fixture_hashes"].items():
    assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == expected, name
from browser_series_method import authorized_rounds

if SELECTION.get("method_scope") != "browser_feature_single_120x40_closed_dark":
    raise RuntimeError("Explicit_browser_single_scope_required")
PLANNED_ROUNDS = authorized_rounds(SELECTION, RUN)
from tldw_chatbook.Backup_Recovery import bootstrap

admission = bootstrap.startup_permission(
    bootstrap.effective_config_path(), bootstrap.default_bootstrap_root()
)
assert admission[0], admission
from approval_fixture_workload_v7 import ScriptedProvider, ApprovalFixtureDriver

provider = ScriptedProvider()
allow_loopback_port(
    provider.server.server_port, "scripted_provider", owner_pid=os.getpid()
)
provider.start()
from tldw_chatbook.config import save_settings_to_cli_config

settings = provider.offline_settings()
settings["canvas"] = {"enabled": False}
os.environ["TLDW_CANVAS_ENABLED"] = "false"
assert save_settings_to_cli_config(settings), "Canonical owned provider config refused"
from loguru import logger

logger.remove()
logger.add(RUN / "product-child.log", level="INFO")
from textual.driver import Driver
from textual import events
from tldw_chatbook.app import TldwCli

TRANSPORT = []
LOCK = threading.Lock()
ACTIVE_CORRELATION = ""
STAGE_COUNT = 0
COMMON_OBSERVER = None
app = TldwCli()
from native_rehearsal_observation import EventSink, CardEvents

sink = EventSink(RUN)
CARD_EVENTS = None
ORIGIN_OBSERVER = None
ORIGIN_OBSERVER_INITIALIZATION_ERROR = None
try:
    from origin_observer import OriginObservation
    from textual.app import App
    from textual.drivers.web_driver import WebDriver

    ORIGIN_OBSERVER = OriginObservation(RUN / "child-origin-observations.json")
    ORIGIN_OBSERVER.bind_app(app, App.exit)
except BaseException as observer_error:
    ORIGIN_OBSERVER_INITIALIZATION_ERROR = type(observer_error).__name__
WRITE_OBSERVER = None
WRITE_OBSERVER_INITIALIZATION_ERROR = None
try:
    from web_write_observer import WriteObservation

    WRITE_OBSERVER = WriteObservation(RUN / "web-write-observations.json")
except BaseException as observer_error:
    WRITE_OBSERVER_INITIALIZATION_ERROR = type(observer_error).__name__
original_build_driver = app._build_driver


def build_driver(*args, **kwargs):
    actual = original_build_driver(*args, **kwargs)  # Original constructor/result.
    original_input = actual.process_message
    if getattr(original_input, "__self__", None) is not actual:
        raise RuntimeError("Actual_owned_input_method_missing")

    def process_message(message):
        entered = time.perf_counter_ns()
        try:
            current = driver.current
            if (
                getattr(actual, "_app", None) is app
                and current
                and isinstance(message, (events.MouseDown, events.MouseUp, events.Key))
            ):
                driver._record(
                    "app_input_prequeue",
                    current["correlation"],
                    timestamp_ns=entered,
                    kind=type(message).__name__,
                )
        except BaseException:
            try:
                sink.fail()
            except BaseException:
                pass
        return original_input(
            message
        )  # Original exact owned message/offset/prequeue body once.

    actual.process_message = process_message  # Only this actual WebDriver instance.
    if WRITE_OBSERVER is not None:
        WRITE_OBSERVER.bind(actual, app)
    if ORIGIN_OBSERVER is not None:
        ORIGIN_OBSERVER.bind_meta(actual, app, WebDriver)
    return actual


app._build_driver = build_driver  # Only this owned app instance, before firstwrite.
driver = ApprovalFixtureDriver(app, RUN / "fixture", emit=sink.emit, provider=provider)
nonce = secrets.token_hex(24)


def capture_state():
    full = driver.snapshot()
    keep = (
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
        driver.session.id if driver.session is not None else None
    )
    rail = app.screen.query_one("#console-inspector-rail-body")
    visible = app.screen._compositor.visible_widgets
    view["inspect_open"] = bool(
        rail.display
        and rail in visible
        and visible[rail][0].width > 0
        and visible[rail][0].height > 0
    )
    native = {}  # Browser owns pixel mapping; these are only live cell regions.
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
    if driver.current and driver.session is not None:
        current = driver.current
        controller = driver.controller
        task = full.get("submission_task") or {}
        submitted = task.get("result") or {}
        settlement["owner_terminal"] = (
            full.get("run_status") == "completed"
            and task.get("done") is True
            and submitted.get("accepted") is True
            and submitted.get("terminal_status") == "completed"
        )
        settlement["session_matches"] = (
            controller.store.active_session_id == driver.session.id
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
                    (start, end) for label, start, end in spans if label == "provider"
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
        "rehearsal_scope": "browser_feature_single_120x40_closed_dark",
    }
    return answer


async def command(value):
    global ACTIVE_CORRELATION, COMMON_OBSERVER, CARD_EVENTS, STAGE_COUNT
    kind = value["command"]
    extra = {}
    if kind == "attach":
        await driver.attach()
        from common_activity_observer import CommonActivityObserver

        COMMON_OBSERVER = CommonActivityObserver(driver)
        if SELECTION["mode"] != "negative":
            CARD_EVENTS = CardEvents(
                app.screen.query_one("#chat-approval-card"), driver, sink, capture_state
            )
    elif kind == "stage":
        if (
            SELECTION["mode"] == "negative"
            or value.get("shape") != "single"
            or STAGE_COUNT >= PLANNED_ROUNDS
        ):
            raise RuntimeError("Fixed_browser_validation1_or_authorized41_stage_bound")
        STAGE_COUNT += 1  # A failed/invalid planned round is never replaced.
        ACTIVE_CORRELATION = driver.new_correlation()
        await driver.stage(value["shape"], ACTIVE_CORRELATION)
    elif kind == "collect":
        extra = await driver.collect()
    elif kind == "configure":
        rail_screen = app.screen
        await driver.configure(theme=value["theme"], inspect_open=value["inspect_open"])
        rail_workers = tuple(
            worker
            for worker in app.workers
            if worker.node is rail_screen
            and worker.name == "_save_console_rail_preferences"
        )
        for worker in rail_workers:
            await worker.wait()
    elif kind == "trace_negatives":
        extra = await driver.trace_admission_negatives()
    elif kind == "snapshot":
        pass
    elif kind == "exit":
        await driver.close()
        if COMMON_OBSERVER is not None:
            COMMON_OBSERVER.close()
        if CARD_EVENTS is not None:
            CARD_EVENTS.close()
        app.exit()
        return {"exiting": True}
    else:
        raise ValueError("Unsupported_QA_command")
    snapshot = driver.snapshot()
    with LOCK:
        stages = [
            dict(row) for row in TRANSPORT if row["correlation"] == ACTIVE_CORRELATION
        ]
    stages.extend(snapshot.get("stages", []))
    stages.sort(key=lambda item: item["timestamp_ns"])
    return {
        "snapshot": snapshot,
        "stages": stages,
        "result": extra,
        "size": list(app.size),
        "driver": type(app._driver).__name__,
        "server_ns": time.perf_counter_ns(),
        "common_activity": COMMON_OBSERVER.snapshot()
        if COMMON_OBSERVER is not None
        else None,
        "browser_state": capture_state(),
        "observer_health": sink.health(),
    }


ATTACH_JOB = {"status": "not_started"}
ATTACH_DONE = threading.Event()
MUTATION_LOCK = threading.Lock()


def attach_status():
    with LOCK:
        answer = dict(ATTACH_JOB)
    path = RUN / "attach-stages.jsonl"
    phases = []
    if path.exists():
        data = path.read_bytes()
        if len(data) > 65536:
            raise RuntimeError("Attach_phase_evidence_bound_exceeded")
        # The writer may be between write and newline; retain complete rows only.
        phases = [
            json.loads(line)
            for line in data.splitlines(keepends=True)
            if line.endswith(b"\n")
        ]
    return {**answer, "phases": phases, "server_ns": time.perf_counter_ns()}


def attach_worker():
    try:
        with MUTATION_LOCK:
            answer = app.call_from_thread(command, {"command": "attach"})
        update = {"status": "completed", "result": answer}
    except BaseException as error:
        import traceback

        update = {
            "status": "failed",
            "error_type": type(error).__name__,
            "code_frames": [
                {"file": item.filename, "function": item.name, "line": item.lineno}
                for item in traceback.extract_tb(error.__traceback__)
            ],
        }
    with LOCK:
        ATTACH_JOB.update(update, finished_ns=time.perf_counter_ns())
        receipt = dict(ATTACH_JOB)
    (RUN / "attach-job.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    ATTACH_DONE.set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/qa/attach-status":
            if self.headers.get("X-QA-Control") != nonce:
                self.reply(403, {"error_type": "ControlNonceMismatch"})
                return
            self.reply(200, attach_status())
            return
        self.reply(
            200,
            {"server_ns": time.perf_counter_ns(), "pid": os.getpid(), "nonce": nonce},
        )

    def do_POST(self):
        if self.headers.get("X-QA-Control") != nonce:
            self.reply(403, {"error_type": "ControlNonceMismatch"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            assert 0 < length <= 4096
            value = json.loads(self.rfile.read(length))
            if value.get("command") == "attach_start":
                with LOCK:
                    if ATTACH_JOB["status"] != "not_started":
                        self.reply(409, {"error_type": "AttachAlreadyStarted"})
                        return
                    ATTACH_JOB.update(
                        status="running", started_ns=time.perf_counter_ns()
                    )
                self.reply(202, {"status": "running"})
                self.wfile.flush()
                threading.Thread(
                    target=attach_worker, name="qa-attach-control", daemon=True
                ).start()
                return
            kind = value.get("command")
            if kind == "attach":
                self.reply(409, {"error_type": "UseAsyncAttachJob"})
                return
            with LOCK:
                attaching = ATTACH_JOB["status"] == "running"
            if attaching and kind == "exit":
                if not ATTACH_DONE.wait(timeout=60):
                    self.reply(409, {"error_type": "AttachDrainCleanupBoundExceeded"})
                    return
            elif attaching and kind != "snapshot":
                self.reply(409, {"error_type": "AttachMutationStillRunning"})
                return
            if kind == "snapshot":
                result = app.call_from_thread(command, value)
            else:
                if not MUTATION_LOCK.acquire(blocking=False):
                    self.reply(409, {"error_type": "ConcurrentQAMutationRefused"})
                    return
                try:
                    result = app.call_from_thread(command, value)
                finally:
                    MUTATION_LOCK.release()
            self.reply(200, result)
        except Exception as error:
            import traceback

            frames = [
                {"file": item.filename, "function": item.name, "line": item.lineno}
                for item in traceback.extract_tb(error.__traceback__)
            ]
            self.reply(500, {"error_type": type(error).__name__, "code_frames": frames})


control = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
control.daemon_threads = True
allow_loopback_port(control.server_port, "native_control", owner_pid=os.getpid())
thread = threading.Thread(target=control.serve_forever, daemon=True)
thread.start()
(RUN / "product-control.json").write_text(
    json.dumps(
        {
            "port": control.server_port,
            "provider_port": provider.server.server_port,
            "pid": os.getpid(),
            "nonce": nonce,
            "source_root": str(REPO),
            "admission": admission,
            "fixture_sha256": hashlib.sha256(
                (RUN / "approval_fixture_workload_v7.py").read_bytes()
            ).hexdigest(),
        }
    ),
    encoding="utf-8",
)
try:
    app.run()
finally:
    # Observe only the actual app-owned exception. Metadata failure never
    # changes original app return/exception or control cleanup semantics.
    exception_diagnostic = {}
    try:
        actual_error = app._exception
        if actual_error is not None:
            import traceback

            errno_value = getattr(actual_error, "errno", None)
            winerror_value = getattr(actual_error, "winerror", None)
            exception_diagnostic = {
                "errno": errno_value if type(errno_value) is int else None,
                "winerror": winerror_value if type(winerror_value) is int else None,
                "code_frames": [
                    {
                        "file": frame.filename,
                        "function": frame.name,
                        "line": frame.lineno,
                    }
                    for frame in traceback.extract_tb(actual_error.__traceback__)
                ],
            }
    except BaseException as observer_error:
        exception_diagnostic = {"observer_error_type": type(observer_error).__name__}
    write_observation = None
    try:
        if WRITE_OBSERVER is not None:
            WRITE_OBSERVER.persist()
            write_observation = WRITE_OBSERVER.snapshot()
    except BaseException as observer_error:
        write_observation = {"observer_error_type": type(observer_error).__name__}
    origin_observation = None
    try:
        if ORIGIN_OBSERVER is not None:
            ORIGIN_OBSERVER.persist()
            origin_observation = ORIGIN_OBSERVER.snapshot()
    except BaseException as observer_error:
        origin_observation = {"observer_error_type": type(observer_error).__name__}
    sink.close()
    control.shutdown()
    control.server_close()
    (RUN / "product-child-exit.json").write_text(
        json.dumps(
            {
                "return_code": app.return_code,
                "exception_type": type(app._exception).__name__
                if app._exception
                else None,
                "exception_diagnostic": exception_diagnostic,
                "write_observation": write_observation,
                "origin_observation": origin_observation,
                "origin_observer_initialization_error_type": ORIGIN_OBSERVER_INITIALIZATION_ERROR,
                "write_observer_initialization_error_type": WRITE_OBSERVER_INITIALIZATION_ERROR,
                "transport_stages": TRANSPORT,
                "fixture_stages": driver.events,
                "provider_post_entries": provider.chat_post_entries,
                "provider_scripted_turns": provider.scripted_turns,
                "observer_health": sink.health(),
                "planned_rounds": PLANNED_ROUNDS,
                "actual_stage_attempts": STAGE_COUNT,
            }
        ),
        encoding="utf-8",
    )
