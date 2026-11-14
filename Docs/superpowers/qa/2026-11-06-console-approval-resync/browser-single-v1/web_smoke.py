"""Bounded frozen-X served smoke/negative run; zero qualified distributions."""

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from qa_process_guard import RUN, REPO, allow_loopback_port, verify_frozen

FROZEN = verify_frozen()
SELECTION = json.loads((RUN / "source-selection.json").read_text(encoding="utf-8-sig"))
for name, expected in SELECTION["fixture_hashes"].items():
    assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == expected, name
from browser_series_method import (
    authorized_rounds,
    sample_metrics,
    select_observed_frame,
    terminal_pair,
    summarize,
    validate_clock_series,
    validate_capture_limits,
    validate_capture_series,
)

if SELECTION.get("method_scope") != "browser_feature_single_120x40_closed_dark":
    raise RuntimeError("Explicit_browser_single_scope_required")
PLANNED_ROUNDS = authorized_rounds(SELECTION, RUN)
os.environ["TLDW_CANVAS_ENABLED"] = "false"
from tldw_chatbook.Backup_Recovery import bootstrap

admission = bootstrap.startup_permission(
    bootstrap.effective_config_path(), bootstrap.default_bootstrap_root()
)
assert admission[0], admission
from tldw_chatbook.config import save_settings_to_cli_config

assert save_settings_to_cli_config(
    {
        "general": {"default_tab": "chat"},
        "first_run": {"setup_started": True, "setup_completed": True},
        "splash_screen": {"enabled": False},
        "model_catalog": {"auto_refresh_enabled": False},
        "permission_summary": {"enabled": False},
        "canvas": {"enabled": False},
        "console": {
            "agent_runtime": True,
            "local_tools_enabled": True,
            "raw_cli_permitted": True,
        },
    }
)


def is_eligible_presented_frame(frame):
    """Only the analyzer's unique, complete actual-presentation join is eligible."""
    return (
        frame.get("presented_us") is not None
        and frame.get("rejection", "missing_rejection_field") is None
        and frame.get("identity_matches") == 1
        and frame.get("eligible_matches") == 1
    )


def eligible_presented_frames(frames):
    eligible = [frame for frame in frames if is_eligible_presented_frame(frame)]
    if not eligible:
        raise RuntimeError("Actual_card_has_no_unique_presented_frame_join")
    return eligible


def select_reviewed_presented_frame(frames, reviewed):
    matches = [
        frame
        for frame in frames
        if frame.get("image") == reviewed.get("presented_image")
        and frame.get("image_sha256") == reviewed.get("image_sha256")
    ]
    if len(matches) != 1 or not is_eligible_presented_frame(matches[0]):
        raise RuntimeError("Actual_selected_frame_has_no_unique_presented_join")
    return matches[0]


async def main():
    from aiohttp import web, ClientSession
    from playwright.async_api import async_playwright
    import psutil
    from tldw_chatbook.Web_Server.serve import create_server
    from browser_trace_capture import BrowserTraceCapture
    from browser_presented_frames import presented_frames
    from browser_viewed_gate import ViewedBrowserGate
    from native_rehearsal_observation import EventReader
    from origin_observer import OriginObservation

    origin = OriginObservation(RUN / "parent-origin-observations.json")
    import platform
    import importlib.metadata

    counts = {"proxy_denials": 0, "page_route_denials": 0}
    capture_clocks = []
    trace_paths = []
    result = {
        "qualified": False,
        "product_samples": 0,
        "revision": FROZEN["revision"],
        "mode": SELECTION["mode"],
        "canvas_enabled": False,
        "bytecode_writes_disabled": bool(sys.dont_write_bytecode),
        "startup_admission": admission,
        "runs": [],
        "network_counts": counts,
    }
    result["environment"] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine_architecture": platform.machine(),
        "logical_CPUs": psutil.cpu_count(),
        "physical_CPUs": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "python_version": platform.python_version(),
        "textual_version": importlib.metadata.version("textual"),
        "playwright_version": importlib.metadata.version("playwright"),
        "load_assertions_from_Root": SELECTION.get("method_authorization", {}).get(
            "load_conditions"
        )
        if SELECTION.get("method_authorization")
        else None,
    }
    result["system_CPU_times_before"] = psutil.cpu_times()._asdict()

    async def deny_proxy(reader, writer):
        counts["proxy_denials"] += 1
        try:
            await asyncio.wait_for(reader.read(8192), 2)
            writer.write(
                b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    proxy = await asyncio.start_server(deny_proxy, "127.0.0.1", 0)
    proxy_port = proxy.sockets[0].getsockname()[1]
    allow_loopback_port(proxy_port, "browser_deny_proxy", owner_pid=os.getpid())
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    allow_loopback_port(port, "served_site", owner_pid=os.getpid())
    server = create_server(host="127.0.0.1", port=port)
    interpreter = RUN / "qa-venv/Scripts/python.exe"
    server.command = subprocess.list2cmdline(
        [str(interpreter), "-I", str(RUN / "web_product_child.py")]
    )
    application = await server._make_app()
    parent_close_observer = None
    try:
        from parent_close_observer import ParentCloseObservation

        parent_close_observer = ParentCloseObservation(
            RUN / "parent-close-observations.json"
        )
        parent_close_observer.bind_server(server)  # Only this original server instance.
    except BaseException as observer_error:
        result["parent_close_observer_initialization_error_type"] = type(
            observer_error
        ).__name__
    session = ClientSession()
    child = None
    sampling_active = False
    sampling_started = None
    active_control_record = None

    async def child_info():
        nonlocal child
        if child is not None:
            return child
        path = RUN / "product-control.json"
        with origin.span("control_wait"):
            async with asyncio.timeout(90):
                while not path.is_file():
                    await asyncio.sleep(0.05)
        with origin.span("control_read_parse"):
            value = json.loads(path.read_text(encoding="utf-8"))
        with origin.span("source_assert"):
            assert value["source_root"] == str(REPO)
        with origin.span("process_object"):
            process = psutil.Process(value["pid"])
        with origin.span("exe_resolve_assert"):
            assert Path(process.exe()).resolve() in {
                interpreter.resolve(),
                (Path(sys.base_prefix) / "python.exe").resolve(),
            }
        with origin.span("parents_assert"):
            assert os.getpid() in {parent.pid for parent in process.parents()}
        with origin.span("net_connections"):
            listeners = {
                connection.laddr.port
                for connection in process.net_connections(kind="tcp")
                if connection.status == "LISTEN" and connection.laddr.ip == "127.0.0.1"
            }
        with origin.span("listener_assert"):
            assert value["port"] in listeners and value["provider_port"] in listeners
        with origin.span("guard_read_parse"):
            guard_rows = [
                json.loads(line)
                for line in (RUN / f"guard-events-{value['pid']}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            starts = [
                row for row in guard_rows if row["event"] == "protected_process_started"
            ]
        with origin.span("guard_start_assert"):
            assert (
                len(starts) == 1
                and starts[0]["guard_before_redirection"]
                and starts[0]["source_root"] == str(REPO)
            )
        with origin.span("guard_executable_assert"):
            assert (
                Path(starts[0]["executable"]).resolve() == interpreter.resolve()
                and starts[0]["isolated_mode"]
            )
        with origin.span("guard_prefix_assert"):
            assert Path(starts[0]["prefix"]).resolve() == (RUN / "qa-venv").resolve()
        with origin.span("allow_control_port"):
            allow_loopback_port(value["port"], "child_control", owner_pid=value["pid"])
        with origin.span("cache_identity"):
            child = value
        with origin.span("identity_receipt_write"):
            (RUN / "child-identity-receipt.json").write_text(
                json.dumps(
                    {
                        "pid": value["pid"],
                        "descendant_verified": True,
                        "interpreter_verified": True,
                        "owned_listeners_verified": True,
                        "guard_before_redirection": True,
                        "source_root": str(REPO),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return child

    async def command_original(kind, **values):
        value = await child_info()
        endpoint = f"http://127.0.0.1:{value['port']}"
        remaining = (
            60
            if sampling_started is None
            else min(60, 600 - (time.monotonic() - sampling_started))
        )
        if remaining <= 0:
            raise TimeoutError("Actual_browser_sampling600s_bound")
        async with session.post(
            endpoint,
            json={"command": kind, **values},
            headers={"X-QA-Control": value["nonce"]},
            timeout=remaining,
        ) as response:
            answer = await response.json()
            assert response.status == (202 if kind == "attach_start" else 200), answer
            if (
                sampling_started is not None
                and time.monotonic() - sampling_started >= 600
            ):
                raise TimeoutError("Actual_late_response_sampling600s_bound")
            if sampling_active and kind != "exit":
                if (
                    answer["snapshot"]["size"] != [120, 40]
                    or answer["snapshot"]["theme"] != "textual-dark"
                    or answer["browser_state"]["snapshot"].get("inspect_open")
                    is not False
                ):
                    raise RuntimeError("Actual_browser_sample_condition_changed")
            return answer

    async def command(kind, **values):
        control_started = time.perf_counter_ns()
        try:
            return await command_original(kind, **values)
        finally:
            try:
                record = (
                    active_control_record
                    if active_control_record is not None
                    else result
                )
                spans = record.setdefault("control_spans_ns", [])
                if len(spans) < (32 if active_control_record is not None else 16):
                    spans.append(
                        {
                            "command": kind,
                            "elapsed_ns": time.perf_counter_ns() - control_started,
                        }
                    )
                else:
                    record["control_span_overflow"] = True
            except BaseException:
                try:
                    result["control_span_observer_failure"] = True
                except BaseException:
                    pass

    async def attach():
        from passive_attach import observe_attach

        value = await child_info()  # Original owned-child checks before originalPOST.
        if any(
            (RUN / name).exists() for name in ("attach-job.json", "attach-stages.jsonl")
        ):
            raise RuntimeError("Actual_attach_records_preexist_original_request")
        request_before_ns = time.perf_counter_ns()
        await command("attach_start")  # Exactly one original authenticated POST.
        request_after_ns = time.perf_counter_ns()
        return await observe_attach(
            RUN, value, request_before_ns, request_after_ns, result
        )

    async def clock(request):
        value = await child_info()
        async with session.get(
            f"http://127.0.0.1:{value['port']}", timeout=10
        ) as response:
            answer = await response.json()
        assert answer["pid"] == value["pid"] and answer["nonce"] == value["nonce"]
        return web.json_response(
            {"server_ns": answer["server_ns"]}, headers={"Cache-Control": "no-store"}
        )

    application.router.add_get("/qa/clock", clock)
    runner = web.AppRunner(application)
    await runner.setup()
    await web.SockSite(runner, listener).start()
    browser = None
    playwright = None
    viewed = None
    capture = None
    phase = "setup"
    exited = False
    try:
        playwright = await async_playwright().start()
        executable = Path(
            "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
        )
        flags = [
            "--enable-automation",
            f"--proxy-server=http://127.0.0.1:{proxy_port}",
            f"--proxy-bypass-list=<-loopback>;127.0.0.1:{port}",
            "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1, EXCLUDE [::1]",
            "--disable-quic",
            "--disable-background-networking",
            "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
        ]
        result["browser_executable_sha256"] = hashlib.sha256(
            executable.read_bytes()
        ).hexdigest()
        result["browser_flags"] = flags
        browser = await playwright.chromium.launch(
            executable_path=str(executable), headless=False, args=flags
        )
        result["actual_browser_version"] = browser.version
        if SELECTION["method_mode"] == "qualification":
            authorization = SELECTION["method_authorization"]
            reference = authorization.get("browser_capability")
            if type(reference) is not dict:
                raise RuntimeError("Reviewed_browser_capability_reference_missing")
            path = Path(reference["path"])
            raw = path.read_bytes()
            if (
                len(raw) > 1024 * 1024
                or hashlib.sha256(raw).hexdigest() != reference["sha256"]
            ):
                raise RuntimeError("Reviewed_browser_capability_changed")
            capability = json.loads(raw)
            if (
                capability.get("actual_browser_version") != browser.version
                or capability.get("browser_executable_sha256")
                != result["browser_executable_sha256"]
                or capability.get("root_observed_image_acknowledgment", {}).get(
                    "observed"
                )
                is not True
            ):
                raise RuntimeError(
                    "Actual_browser_software_not_the_reviewed_capability"
                )
            if authorization.get("method_scope") != SELECTION["method_scope"]:
                raise RuntimeError("Browser_qualification_method_scope_missing")

        context = await browser.new_context(
            viewport={"width": 1200, "height": 800}, service_workers="block"
        )

        async def route(route):
            target = urlsplit(route.request.url)
            if target.hostname == "127.0.0.1" and target.port == port:
                await route.continue_()
            else:
                counts["page_route_denials"] += 1
                await route.abort()

        await context.route("**/*", route)
        page = await context.new_page()
        await origin.bind_browser(browser, context, page, port)
        phase = "page_navigation"
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded")
        await page.bring_to_front()
        phase = "os_identity_bind"
        viewed = await ViewedBrowserGate.bind(browser, page, RUN, executable)
        result["browser_identity"] = dict(viewed.identity)
        result["activation_attempts"] = []
        phase = "os_activation"
        result["activation"] = await viewed.activate_owned(
            result["activation_attempts"]
        )
        phase = "initial_owned_window_check"
        result["viewed_diagnostics"] = [
            await viewed.check("browser-open", capture=True)
        ]
        phase = "first_byte_wait"
        await page.wait_for_function(
            "document.body.classList.contains('-first-byte')", timeout=90000
        )
        result["viewed_diagnostics"].append(
            await viewed.check("first-byte-before-attach", capture=True)
        )
        await page.screenshot(path=str(RUN / "first-byte-before-attach.png"))
        phase = "attach_job"
        result["attach"] = await attach()
        await page.screenshot(path=str(RUN / "full-app-ready.png"))
        if SELECTION["mode"] == "negative":
            phase = "trace_negatives"
            result["trace_negatives"] = await command("trace_negatives")
            proof = result["trace_negatives"]["result"]
            controls = proof["trace_admission_negatives"]
            expected = {
                "temporary_capture_on": {"TemporaryCaptureRequiresSave"},
                "durable_capture_on_missing_provenance": {
                    "TraceCallPersistenceError",
                    "TraceProvenanceAlignmentError",
                },
            }
            assert len(controls) == 2 and {row["control"] for row in controls} == set(
                expected
            )
            assert (
                proof["adapter_turns_consumed"] == 0
                and proof["chat_post_entry_delta"] == 0
            )
            for row in controls:
                assert row["actual_exception"] in expected[row["control"]]
                assert (
                    row["scripted_turns_consumed"] == 0
                    and row["chat_post_entry_delta"] == 0
                )
        else:
            cdp = await context.new_cdp_session(page)
            capture = BrowserTraceCapture(page, cdp)
            event_reader = EventReader(RUN / "rehearsal-events.jsonl")
            all_stage_rows = []

            async def retain_frames(calibration, label, directory, write_images=True):
                frames, diagnostics = presented_frames(
                    capture.events, page_pid=calibration["page_pid"]
                )
                outputs = []
                for index, frame in enumerate(frames):
                    frame = dict(frame)
                    path = directory / (label + f"-frame-{index}.jpg")
                    snapshot = frame.pop("snapshot")
                    if write_images:
                        raw = base64.b64decode(snapshot)
                        if len(raw) > 1024 * 1024:
                            raise RuntimeError("Actual_JPEG_1MiB_cap")
                        path.write_bytes(raw)
                    outputs.append(
                        {
                            **frame,
                            "image": path.name,
                            "image_sha256": hashlib.sha256(
                                path.read_bytes()
                            ).hexdigest()
                            if write_images
                            else None,
                            "image_decoded": write_images,
                        }
                    )
                return outputs, diagnostics

            # Genuine browser fit/resize traffic before target samples; no server-size injection.
            await command("configure", theme="textual-dark", inspect_open=False)
            result["resize_conditions"] = []
            for attempt in range(4):
                fitted = await command("snapshot")
                result["resize_conditions"].append(
                    {
                        "attempt": attempt,
                        "size": fitted["size"],
                        "viewport": dict(page.viewport_size),
                    }
                )
                if fitted["size"] == [120, 40]:
                    break
                rect = await page.locator(".xterm-screen").evaluate(
                    "(element)=>{const r=element.getBoundingClientRect();return {width:r.width,height:r.height}}"
                )
                columns, rows = fitted["size"]
                viewport = page.viewport_size
                await page.set_viewport_size(
                    {
                        "width": round(
                            viewport["width"]
                            + (120 - columns) * rect["width"] / columns
                        ),
                        "height": round(
                            viewport["height"] + (40 - rows) * rect["height"] / rows
                        ),
                    }
                )
                result["activation_attempts"].append(
                    {"controlled_browser_resize": True, "attempt": attempt}
                )
                await viewed.activate_owned(result["activation_attempts"])
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("Actual_browser_terminal_grid_not120x40")
            if (
                fitted["snapshot"]["theme"] != "textual-dark"
                or fitted["browser_state"]["snapshot"].get("inspect_open") is not False
            ):
                raise RuntimeError("Actual_browser_method_condition_mismatch")
            sampling_active = True
            sampling_started = time.monotonic()
            async with asyncio.timeout(600):
                result["planned_rounds"] = PLANNED_ROUNDS
                prefixes = set()
                for round_index in range(PLANNED_ROUNDS):
                    active_control_record = {
                        "round_index": round_index,
                        "control_spans_ns": [],
                    }
                    result.setdefault("round_control_spans", []).append(
                        active_control_record
                    )
                    decision = "approve"
                    round_run = RUN / f"round-{round_index:03d}"
                    round_run.mkdir()
                    phase = f"round{round_index}_stage"
                    await page.evaluate(
                        """() => {window.__qaViewLoss=[];const note=()=>{if(window.__qaViewLoss.length<16)window.__qaViewLoss.push({visible:document.visibilityState==='visible',focused:document.hasFocus(),ms:performance.now()});};window.__qaViewNote=note;document.addEventListener('visibilitychange',note);window.addEventListener('blur',note);}"""
                    )
                    await capture.start(calibration_samples=20 if round_index == 0 else 1)
                    started = progress = time.monotonic()
                    last_phase = None
                    progress_rows = []
                    staged = await command("stage", shape="single")
                    result["last_staged"] = staged
                    if staged["snapshot"]["current"].get("protocol_error"):
                        raise RuntimeError("Actual_staged_workload_protocol_error")
                    if time.monotonic() - started >= 60:
                        raise TimeoutError("Actual_late_stage_reply_phase60")
                    expected_correlation = staged["snapshot"]["current"]["correlation"]
                    result["actual_preflight_progress"] = progress_rows
                    while True:
                        now = time.monotonic()
                        if now - started >= 300 or now - progress >= 60:
                            raise TimeoutError(
                                "Actual_preflight_phase60_or_total300_bound_exceeded"
                            )
                        updates = event_reader.poll()
                        all_stage_rows.extend(updates)
                        observed = time.monotonic()
                        if observed - started >= 300 or observed - progress >= 60:
                            raise TimeoutError(
                                "Actual_late_preflight_file_read_before_phase_reset"
                            )
                        candidate = None
                        for row in updates:
                            if row.get("correlation") != expected_correlation:
                                continue
                            if row.get("stage") == "geometry_after_refresh":
                                state_view = row.get("state", {})
                                view = state_view.get("snapshot", {})
                                key = (
                                    "geometry",
                                    view.get("card_identity"),
                                    view.get("phase"),
                                    view.get("card_display"),
                                    view.get("information_ready"),
                                    view.get("submitted"),
                                )
                                controls = [
                                    control
                                    for control in view.get("controls", [])
                                    if (control["id"] or "").startswith(
                                        "approval-fast-" + decision + "-"
                                    )
                                    and control["fully_visible"]
                                    and not control["disabled"]
                                    and control["display"]
                                ]
                                region = view.get("card_region")
                                if (
                                    view.get("card_display")
                                    and view.get("card_identity")
                                    and region
                                    and region["fully_visible"]
                                    and len(controls) == 1
                                ):
                                    candidate = state_view
                            else:
                                key = (
                                    row.get("stage"),
                                    row.get("ordinal"),
                                    row.get("boundary"),
                                    row.get("outcome"),
                                )
                            if key != last_phase:
                                last_phase = key
                                progress = observed
                                if len(progress_rows) < 64:
                                    progress_rows.append(
                                        {
                                            "observed_ns": time.perf_counter_ns(),
                                            "actual_stage": row.get("stage"),
                                            "server_ns": row.get("timestamp_ns"),
                                        }
                                    )
                        if candidate is not None:
                            remaining = min(
                                300 - (time.monotonic() - started),
                                60 - (time.monotonic() - progress),
                            )
                            if remaining <= 0:
                                raise TimeoutError(
                                    "Actual_preflight_bound_before_confirmation"
                                )
                            state = await asyncio.wait_for(command("snapshot"), remaining)
                            observed = time.monotonic()
                            if observed - started >= 300 or observed - progress >= 60:
                                raise TimeoutError("Actual_late_preflight_confirmation")
                            snapshot = state["snapshot"]
                            result["last_wait_state"] = state
                            task = snapshot.get("submission_task") or {}
                            if task.get("done") and (
                                task.get("exception_type")
                                or task.get("cancelled")
                                or (
                                    task.get("result") is not None
                                    and task["result"].get("accepted") is False
                                )
                            ):
                                raise RuntimeError("ActualSubmitRaisedOrRefused")
                            candidates = [
                                control
                                for control in snapshot["controls"]
                                if (control["id"] or "").startswith(
                                    "approval-fast-" + decision + "-"
                                )
                                and control["fully_visible"]
                                and not control["disabled"]
                                and control["display"]
                            ]
                            region = snapshot.get("card_region")
                            if snapshot["current"]["correlation"] != expected_correlation:
                                raise RuntimeError("Actual_preflight_generation_changed")
                            if (
                                snapshot.get("card_display")
                                and snapshot.get("card_identity")
                                and region
                                and region["fully_visible"]
                                and len(candidates) == 1
                            ):
                                break
                        await asyncio.sleep(0.05)
                    correlation = snapshot["current"]["correlation"]
                    identity = snapshot["card_identity"]
                    control = candidates[0]
                    current = snapshot["current"]
                    if current.get("protocol_error"):
                        raise RuntimeError("Actual_workload_protocol_error")
                    if current.get("process_cold_first_use") is not (
                        round_index == 0
                    ) or current.get("first_shape_use") is not (round_index == 0):
                        raise RuntimeError("Actual_firstuse_generation_mismatch")
                    if correlation[:12] in prefixes:
                        raise RuntimeError("Actual_nonce_prefix_collision_no_replacement")
                    prefixes.add(correlation[:12])
                    phase = decision + "_actual_card_presented"
                    result["viewed_diagnostics"].append(
                        await viewed.check(
                            f"round{round_index}-actionable", capture=round_index == 0
                        )
                    )
                    image = round_run / (decision + "-actionable.png")
                    await page.screenshot(path=str(image))
                    rect = await page.locator(".xterm-screen").evaluate(
                        "(element)=>{const r=element.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}}"
                    )
                    width, height = state["size"]
                    cells = region["region"]
                    clip = {
                        "x": rect["x"] + cells[0] * rect["width"] / width,
                        "y": rect["y"] + cells[1] * rect["height"] / height,
                        "width": cells[2] * rect["width"] / width,
                        "height": cells[3] * rect["height"] / height,
                    }
                    card_image = await page.screenshot(clip=clip)
                    (round_run / (decision + "-card-candidate.png")).write_bytes(card_image)
                    card_hash = hashlib.sha256(card_image).hexdigest()
                    card_trace = round_run / (decision + "-card-trace.json")
                    calibration = await capture.stop(card_trace, calibration_samples=1)
                    calibration.update(round_index=round_index, role="actionable")
                    capture_clocks.append(calibration)
                    trace_paths.append(card_trace)
                    validate_capture_limits(card_trace)
                    frames, diagnostics = await retain_frames(
                        calibration,
                        decision + "-card",
                        round_run,
                        write_images=SELECTION["method_mode"] == "validation",
                    )
                    evidence = {
                        "round_index": round_index,
                        "status": "pending_review",
                        "first_use": round_index == 0,
                        "decision": decision,
                        "correlation": correlation,
                        "card_identity": identity,
                        "control_id": control["id"],
                        "png_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "card_png_sha256": card_hash,
                        "file": image.name,
                        "card_file": decision + "-card-candidate.png",
                        "frames": frames,
                        "frame_diagnostics": diagnostics,
                        "clock_calibration": calibration,
                        "semantic_actionable": state,
                        "observed": False,
                        "qualified": False,
                        "product_samples": 0,
                    }
                    result["runs"].append(evidence)
                    eligible_frames = eligible_presented_frames(frames)
                    selected_frame = None
                    if SELECTION["method_mode"] == "validation":
                        request = {
                            key: evidence[key]
                            for key in (
                                "decision",
                                "correlation",
                                "card_identity",
                                "control_id",
                                "png_sha256",
                                "card_png_sha256",
                                "file",
                                "card_file",
                                "frames",
                            )
                        }
                        request["frames"] = eligible_frames
                        request["review_required"] = (
                            "Verify full32 nonce; add full_nonce_verified=true. Root independently reads actual uniquely joined presented JPEG: correct visible QA session/target nonce, intended enabled action; then returns matching identities and selected presented_image/image_sha256 with observed,visible_generation_verified,enabled_action_verified true and reviewer. No input before classification."
                        )
                        request_path = round_run / (decision + "-card-review-required.json")
                        request_path.write_text(
                            json.dumps(request, indent=2), encoding="utf-8"
                        )
                        print(
                            json.dumps(
                                {
                                    "actual_card_png": str(image),
                                    "review_request": str(request_path),
                                }
                            ),
                            flush=True,
                        )
                        phase = decision + "_independent_card_review"
                        review_path = round_run / (decision + "-card-review.json")
                        deadline = time.monotonic() + 45
                        while not review_path.exists():
                            if time.monotonic() > deadline:
                                raise TimeoutError(
                                    "No_independent_presented_card_classification_input_withheld"
                                )
                            await asyncio.sleep(0.05)
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Actual_card_review_arrived_after_original45s"
                            )
                        reviewed = json.loads(review_path.read_text(encoding="utf-8"))
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Actual_card_review_read_crossed_original45s"
                            )
                        for key in (
                            "decision",
                            "correlation",
                            "card_identity",
                            "control_id",
                            "png_sha256",
                            "card_png_sha256",
                        ):
                            if reviewed.get(key) != request[key]:
                                raise RuntimeError("Stale_or_mismatched_actual_card_review")
                        selected_frame = select_reviewed_presented_frame(frames, reviewed)
                        if any(
                            reviewed.get(key) is not True
                            for key in (
                                "observed",
                                "visible_generation_verified",
                                "enabled_action_verified",
                                "full_nonce_verified",
                            )
                        ) or not reviewed.get("reviewer"):
                            raise RuntimeError(
                                "Actual_presented_frame_classification_missing"
                            )
                        evidence["selected_presented_frame"] = selected_frame
                    await capture.start(calibration_samples=1)
                    if active_control_record.get("control_span_overflow") or result.get(
                        "control_span_observer_failure"
                    ):
                        raise RuntimeError("Actual_control_observation_budget_or_failure")
                    live = await command("snapshot")
                    live_view = live["snapshot"]
                    current_control = next(
                        (
                            item
                            for item in live_view["controls"]
                            if item["id"] == control["id"]
                        ),
                        None,
                    )
                    if (
                        live_view.get("card_identity") != identity
                        or live_view["current"]["correlation"] != correlation
                        or current_control != control
                        or live_view.get("card_region") != region
                        or live["size"] != state["size"]
                    ):
                        raise RuntimeError("Actual_card_or_action_changed_after_review")
                    result["viewed_diagnostics"].append(
                        await viewed.check(decision + "-before-input")
                    )
                    if (
                        hashlib.sha256(await page.screenshot(clip=clip)).hexdigest()
                        != card_hash
                    ):
                        raise RuntimeError(
                            "Actual_card_pixels_changed_after_review_input_withheld"
                        )
                    if SELECTION["method_mode"] == "validation":
                        evidence["review"] = reviewed
                        evidence["observed"] = True
                    else:
                        evidence["observed"] = False
                        evidence["human_content_classification_pending"] = True
                    live_rect = await page.locator(".xterm-screen").evaluate(
                        "(element)=>{const r=element.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}}"
                    )
                    if live_rect != rect:
                        raise RuntimeError("Actual_terminal_geometry_changed_after_review")
                    target = control["region"]
                    x = rect["x"] + (target[0] + target[2] / 2) * rect["width"] / width
                    y = rect["y"] + (target[1] + target[3] / 2) * rect["height"] / height
                    await page.evaluate(
                        """token=>{window.__qaInput=[];document.addEventListener('pointerdown',event=>{window.__qaInput.push({timestamp_ms:event.timeStamp,handler_ms:performance.now(),trusted:event.isTrusted});performance.mark('qa-input-'+token);},{once:true,capture:true});}""",
                        correlation,
                    )
                    phase = decision + "_actual_input"
                    result["viewed_diagnostics"].append(
                        await viewed.check(decision + "-immediate-before-input")
                    )
                    await page.mouse.click(x, y)
                    collected = await command("collect")
                    result["viewed_diagnostics"].append(
                        await viewed.check(
                            f"round{round_index}-feedback", capture=round_index == 0
                        )
                    )
                    await page.screenshot(
                        path=str(round_run / (decision + "-feedback.png"))
                    )
                    # Actual visible completion is separate from accepted model state.
                    terminal_started = time.monotonic()
                    terminal_states = []
                    required = (
                        "owner_terminal",
                        "session_matches",
                        "card_retired",
                        "header_terminal",
                        "send_terminal",
                        "stop_hidden",
                        "run_segment_absent",
                        "title_generation_matches",
                    )
                    for observation in range(21):
                        remaining = 2 - (time.monotonic() - terminal_started)
                        if remaining <= 0:
                            raise TimeoutError(
                                "Actual_browser_terminal_observation_censored"
                            )
                        terminal = await asyncio.wait_for(command("snapshot"), remaining)
                        if time.monotonic() - terminal_started >= 2:
                            raise TimeoutError("Actual_late_terminal_response_censored")
                        if all(
                            terminal["browser_state"]["settlement"].get(k) is True
                            for k in required
                        ):
                            break
                        if observation == 20:
                            raise TimeoutError("Actual_browser_terminal_UI_not_settled")
                        await asyncio.sleep(
                            min(0.05, max(0, 2 - (time.monotonic() - terminal_started)))
                        )
                    ready_server_ns = terminal["server_ns"]
                    for index in range(2):
                        before = terminal if index == 0 else await command("snapshot")
                        if not all(
                            before["browser_state"]["settlement"].get(k) is True
                            for k in required
                        ):
                            raise RuntimeError("Actual_terminal_owner_changed")
                        await page.screenshot(path=str(round_run / f"terminal-{index}.png"))
                        after = await command("snapshot")
                        if not all(
                            after["browser_state"]["settlement"].get(k) is True
                            for k in required
                        ):
                            raise RuntimeError("Actual_terminal_owner_changed")
                        terminal_states.append(
                            {
                                "before": before["browser_state"],
                                "after": after["browser_state"],
                            }
                        )
                    feedback_trace = round_run / (decision + "-feedback-trace.json")
                    calibration = await capture.stop(
                        feedback_trace,
                        calibration_samples=20 if round_index == PLANNED_ROUNDS - 1 else 1,
                    )
                    calibration.update(round_index=round_index, role="feedback")
                    capture_clocks.append(calibration)
                    trace_paths.append(feedback_trace)
                    validate_capture_limits(feedback_trace)
                    validate_capture_series(
                        [card_trace, feedback_trace]
                    )  # Combined64 roundcap before anotheractorround.
                    frames, diagnostics = await retain_frames(
                        calibration,
                        decision + "-feedback",
                        round_run,
                        write_images=SELECTION["method_mode"] == "validation",
                    )
                    view_losses = await page.evaluate(
                        """() => {document.removeEventListener('visibilitychange',window.__qaViewNote);window.removeEventListener('blur',window.__qaViewNote);return window.__qaViewLoss;}"""
                    )
                    if view_losses:
                        raise RuntimeError(
                            "Actual_browser_visibility_or_focus_changed_during_round"
                        )
                    for _ in range(20):
                        all_stage_rows.extend(event_reader.poll())
                        if (
                            event_reader.rows >= after["observer_health"]["rows"]
                            and not event_reader.partial
                        ):
                            break
                        await asyncio.sleep(0.01)
                    else:
                        raise RuntimeError("Actual_stage_spool_incomplete")
                    if any(
                        after["observer_health"].get(k) is not False
                        for k in ("loss", "write_failure", "observer_failure")
                    ):
                        raise RuntimeError("Actual_browser_observer_loss_or_failure")
                    browser_input = await page.evaluate("window.__qaInput")
                    if len(browser_input) != 1:
                        raise RuntimeError("Actual_browser_input_not_unique")
                    rows = [
                        row
                        for row in all_stage_rows
                        if row.get("correlation") == correlation
                    ]
                    evidence.update(
                        status="captured_pending_offline_classification",
                        collected=collected,
                        feedback_clock_calibration=calibration,
                        feedback_frames=frames,
                        feedback_frame_diagnostics=diagnostics,
                        browser_input=browser_input,
                        owner_stages=rows,
                        terminal_states=terminal_states,
                        ready_server_ns=ready_server_ns,
                        actionable_trace=card_trace.name,
                        feedback_trace=feedback_trace.name,
                        actionable_trace_sha256=hashlib.sha256(
                            card_trace.read_bytes()
                        ).hexdigest(),
                        feedback_trace_sha256=hashlib.sha256(
                            feedback_trace.read_bytes()
                        ).hexdigest(),
                    )
                    if SELECTION["method_mode"] == "validation":
                        request = {
                            "round_index": round_index,
                            "correlation": correlation,
                            "card_identity": identity,
                            "eligible_frames": eligible_presented_frames(frames),
                            "terminal_states": terminal_states,
                            "ready_server_ns": ready_server_ns,
                            "required_review": "Root independently views actual JPEGs; feedback={observed,reviewer,correlation,state:feedback,full_nonce_verified,truthful_current_generation_verified,visible_feedback_state:applying/received/accepted/running/completed,image,image_sha256}; terminal=[two ordered distinct actualpresented Ready/disabledSend/hiddenStop/noRun/currentQA12 frames, each observed/reviewer/correlation/state:terminal/visible_nonce_prefix/terminal_UI_verified/image/image_sha256]. Neverreuseglyph/hash-only references or metadata as paint.",
                        }
                        review_request = (
                            round_run / "feedback-terminal-review-required.json"
                        )
                        review_request.write_text(
                            json.dumps(request, indent=2), encoding="utf-8"
                        )
                        print(
                            json.dumps(
                                {"feedback_terminal_review_request": str(review_request)}
                            ),
                            flush=True,
                        )
                        review_path = round_run / "feedback-terminal-review.json"
                        deadline = time.monotonic() + 45
                        while not review_path.exists():
                            if time.monotonic() > deadline:
                                raise TimeoutError(
                                    "No_independent_feedback_terminal_frame_classification"
                                )
                            await asyncio.sleep(0.05)
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Actual_feedback_review_arrived_after_original45s"
                            )
                        reviews = json.loads(review_path.read_text(encoding="utf-8"))
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Actual_feedback_review_read_crossed_original45s"
                            )
                        if reviews.get("correlation") != correlation:
                            raise RuntimeError("Stale_browser_feedback_terminal_review")
                        if reviews["feedback"].get(
                            "truthful_current_generation_verified"
                        ) is not True or reviews["feedback"].get(
                            "visible_feedback_state"
                        ) not in (
                            "applying",
                            "received",
                            "accepted",
                            "running",
                            "completed",
                        ):
                            raise RuntimeError("Actual_truthful_feedback_not_classified")
                        feedback = select_observed_frame(
                            round_run,
                            frames,
                            reviews["feedback"],
                            correlation=correlation,
                            state="feedback",
                        )
                        settled = terminal_pair(
                            round_run,
                            frames,
                            reviews["terminal"],
                            correlation=correlation,
                            ready_server_ns=ready_server_ns,
                            calibration=calibration,
                        )
                        browser_input = await page.evaluate("window.__qaInput")
                        if len(browser_input) != 1:
                            raise RuntimeError("Actual_browser_input_not_unique")
                        rows = [
                            row
                            for row in all_stage_rows
                            if row.get("correlation") == correlation
                        ]
                        metrics = sample_metrics(
                            rows,
                            correlation,
                            selected_frame,
                            feedback,
                            evidence["clock_calibration"],
                            calibration,
                            browser_input[0],
                        )
                        evidence.update(
                            collected=collected,
                            feedback_clock_calibration=calibration,
                            feedback_frames=frames,
                            feedback_frame_diagnostics=diagnostics,
                            browser_input=browser_input,
                            owner_stages=rows,
                            terminal_states=terminal_states,
                            terminal_presented=settled,
                            reviews=reviews,
                            metrics=metrics,
                        )
                    for path in round_run.glob("*-trace.json"):
                        if path.stat().st_size > 64 * 1024 * 1024:
                            raise RuntimeError(
                                "Actual_raw_trace_persisted_size_cap_exceeded"
                            )
                    PNGs = list(round_run.glob("*.png"))
                    if (
                        len(PNGs) > 6
                        or sum(path.stat().st_size for path in PNGs) > 512 * 1024
                    ):
                        raise RuntimeError("Actual_round_PNG_capture_budget_exceeded")
                    if SELECTION["method_mode"] == "validation":
                        evidence["status"] = "complete"
                    (round_run / "round-result.json").write_text(
                        json.dumps(evidence, indent=2), encoding="utf-8"
                    )
                if (
                    result.get("control_span_overflow")
                    or result.get("control_span_observer_failure")
                    or any(
                        r.get("control_span_overflow")
                        for r in result.get("round_control_spans", [])
                    )
                ):
                    raise RuntimeError("Actual_control_observation_budget_or_failure")
                result["clock_series"] = validate_clock_series(capture_clocks)
                result["capture_series"] = validate_capture_series(trace_paths)
                result["capture_clock_segments"] = capture_clocks
                result["trace_paths"] = [str(path.relative_to(RUN)) for path in trace_paths]
                result["series_summary"] = summarize(result["runs"], PLANNED_ROUNDS)
                if time.monotonic() - sampling_started >= 600:
                    raise TimeoutError("Actual_sampling_safety600_after_reply")
        active_control_record = None
        await command("exit")
        exited = True
        await browser.close()
        browser = None
        result["completed"] = True
    except BaseException as error:
        origin.record("main_error_before_cleanup", error=error)
        result["completed"] = False
        if (
            "round_index" in locals()
            and result["runs"]
            and result["runs"][-1].get("round_index") == round_index
        ):
            result["runs"][-1]["status"] = "failed"
            result["runs"][-1]["failure_type"] = type(error).__name__
        result["series_summary"] = summarize(result["runs"], PLANNED_ROUNDS)
        result["error_type"] = type(error).__name__
        result["failure_phase"] = phase
        if isinstance(error, RuntimeError) and str(error).startswith(
            ("Owned_", "Actual", "Stale_", "No_independent_")
        ):
            result["gate_error_code"] = str(error)
        if viewed is not None:
            try:
                result["failure_os_state"] = viewed.os_state()
            except Exception as state_error:
                result["failure_os_state_error_type"] = type(state_error).__name__
        import traceback

        result["error_code_frames"] = [
            {"file": item.filename, "function": item.name, "line": item.lineno}
            for item in traceback.extract_tb(error.__traceback__)
        ]
        (RUN / "failure-before-cleanup.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        # Retain an in-flight failed segment without probing a stalled child clock.
        # This uncalibrated/possibly partial artifact NEVER qualifies a sample.
        if (
            capture is not None
            and capture.complete is not None
            and not capture.complete.done()
        ):
            failed_status = None
            try:
                async with asyncio.timeout(30):
                    await capture.cdp.send("Tracing.end")
                    failed_status = await capture.complete
            except BaseException as trace_error:
                result["failed_trace_end_error_type"] = type(trace_error).__name__
            try:
                raw = {
                    "traceEvents": capture.events,
                    "tracing_complete": failed_status,
                    "page_mark_brackets": capture.mark_brackets,
                    "server_http_brackets": capture.http_brackets,
                    "partial_failed_segment": True,
                    "clock_qualified": False,
                    "original_failure_phase": phase,
                }
                path = RUN / "failed-partial-trace.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                result["failed_partial_trace"] = {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "calibrated": False,
                    "may_be_incomplete": True,
                }
            except BaseException as trace_error:
                result["failed_trace_persistence_error_type"] = type(
                    trace_error
                ).__name__
        if browser is not None:
            try:
                await page.screenshot(path=str(RUN / "failed-state.png"), timeout=5000)
            except Exception:
                pass
        raise
    finally:
        sampling_active = False
        sampling_started = None
        active_control_record = None
        cleanup_started = time.perf_counter_ns()
        origin.record("cleanup_start")

        async def graceful_cleanup():
            nonlocal child
            if child is None:
                try:
                    await asyncio.wait_for(child_info(), 45)
                except Exception as error:
                    result["cleanup_child_identity_error_type"] = type(error).__name__
            if child is not None and not exited:
                try:
                    await command("exit")
                    result["cleanup_command_completed"] = True
                except Exception as error:
                    result["cleanup_error_type"] = type(error).__name__
            if browser is not None:
                await browser.close()
            await runner.cleanup()
            await session.close()
            proxy.close()
            await proxy.wait_closed()
            if playwright is not None:
                await playwright.stop()

        try:
            await asyncio.wait_for(graceful_cleanup(), 60)
            result["closed_browser_and_servers"] = True
            result["forced_cleanup"] = False
        except Exception as error:
            result["cleanup_bound_or_error_type"] = type(error).__name__
            result["forced_cleanup"] = True
            forced = []
            owner = psutil.Process(os.getpid())
            for process in reversed(owner.children(recursive=True)):
                try:
                    args = " ".join(process.cmdline()).replace("\\", "/").lower()
                    executable = Path(process.exe()).resolve()
                    run_owned = RUN.as_posix().lower() in args
                    browser_owned = (
                        viewed is not None
                        and (
                            process.pid == viewed.pid
                            or viewed.pid in {p.pid for p in process.parents()}
                        )
                        and executable
                        == Path(
                            "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
                        ).resolve()
                    )
                    driver_owned = (
                        process.ppid() == os.getpid()
                        and executable
                        == Path(
                            str(
                                Path(__import__("playwright").__file__).resolve().parent
                                / "driver/node.exe"
                            )
                        ).resolve()
                    )
                    if not (run_owned or browser_owned or driver_owned):
                        continue
                    # Recheck identity immediately before stopping only this owned PID.
                    created = process.create_time()
                    pid = process.pid
                    target = psutil.Process(pid)
                    if target.create_time() != created or os.getpid() not in {
                        p.pid for p in target.parents()
                    }:
                        continue
                    target.terminate()
                    forced.append(
                        {
                            "pid": pid,
                            "creation_time": created,
                            "owned_run_argument": run_owned,
                            "verified_browser_descendant": browser_owned,
                            "owned_playwright_driver": driver_owned,
                        }
                    )
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error as stop_error:
                    forced.append(
                        {
                            "pid": process.pid,
                            "stop_error_type": type(stop_error).__name__,
                        }
                    )
            result["forced_processes"] = forced
            targets = []
            for row in forced:
                try:
                    targets.append(psutil.Process(row["pid"]))
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(targets, timeout=5)
            result["forced_processes_still_alive"] = [p.pid for p in alive]
            listener.close()
            proxy.close()
            try:
                await asyncio.wait_for(session.close(), 2)
            except Exception:
                pass
            try:
                if playwright is not None:
                    await asyncio.wait_for(playwright.stop(), 5)
            except Exception:
                pass
            result["closed_browser_and_servers"] = not alive
        result["cleanup_elapsed_ns"] = time.perf_counter_ns() - cleanup_started
        try:
            result["system_CPU_times_after"] = psutil.cpu_times()._asdict()
        except BaseException as metadata_error:
            result["system_CPU_times_after_error_type"] = type(metadata_error).__name__
        try:
            if parent_close_observer is not None:
                parent_close_observer.persist()
                result["parent_close_observation"] = parent_close_observer.snapshot()
        except BaseException as observer_error:
            result["parent_close_observer_exit_error_type"] = type(
                observer_error
            ).__name__
        try:
            origin.persist()
            result["origin_observation"] = origin.snapshot()
        except BaseException as observer_error:
            result["origin_observer_exit_error_type"] = type(observer_error).__name__
        result["capture_clock_segments"] = capture_clocks
        result["trace_paths"] = [str(path.relative_to(RUN)) for path in trace_paths]
        result["partial_capture_clocks_are_not_qualified"] = not result.get(
            "completed", False
        )
        (RUN / "smoke-result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    asyncio.run(main())
