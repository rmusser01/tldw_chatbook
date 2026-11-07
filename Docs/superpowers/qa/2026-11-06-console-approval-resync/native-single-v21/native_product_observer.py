"""First-smoke native presentation observer, never an automatic budget pass.

Actual owned native Console only. One verified bounded BitBlt capture per image.
Only this fresh synthetic private app's client pixels may be persisted. Capture
brackets bound observation; no atomic DWM frame/physical scanout claim. Actual
card-generation/action pixels require independent visible-pixel review before
an OS click, and actual feedback frames require review before counting samples.
This first-smoke artifact ALWAYS reports qualified=false/product_samples=0.
"""

from __future__ import annotations
import argparse
import ctypes as C
from ctypes import wintypes as W
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import sys
import time

import qa_process_guard as guard
from native_window_api import win32, geometry, clock_bounds, qpc
from owned_roi_capture import capture_owned_roi


def write_json(path, value):
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class NativeConsoleProcess:
    """Real CreateProcessW via installed PyWin32, with NO inherited pipes."""

    def __init__(self, arguments):
        import win32api
        import win32event
        import win32process

        self.api, self.event, self.process_api = win32api, win32event, win32process
        startup = (
            win32process.STARTUPINFO()
        )  # dwFlags0: Windows supplies CONIN$/CONOUT$.
        self.handle, thread, self.pid, thread_id = win32process.CreateProcess(
            arguments[0],
            subprocess.list2cmdline(arguments),
            None,
            None,
            False,
            subprocess.CREATE_NEW_CONSOLE,
            None,
            str(guard.REPO),
            startup,
        )
        win32api.CloseHandle(thread)
        self.returncode = None
        guard.record(
            "actual_native_process_created",
            child_pid=self.pid,
            thread_id=thread_id,
            inherit_handles=False,
            new_console=True,
        )

    def poll(self):
        if self.event.WaitForSingleObject(self.handle, 0) == 0:
            self.returncode = self.process_api.GetExitCodeProcess(self.handle)
        return self.returncode

    def wait(self, timeout):
        if self.event.WaitForSingleObject(self.handle, int(timeout * 1000)) == 258:
            raise subprocess.TimeoutExpired("owned-native-console", timeout)
        self.returncode = self.process_api.GetExitCodeProcess(self.handle)
        return self.returncode

    def terminate(self):
        self.process_api.TerminateProcess(self.handle, 97)

    def close(self):
        self.api.CloseHandle(self.handle)


def same_eligible_action_button(expected: dict | None, actual: dict | None) -> bool:
    """Keep all original fields exact; permit only fully containing clip changes."""
    if type(expected) is not dict or type(actual) is not dict:  # noqa: E721 - Exact boundary types reject bool/subclasses.
        return False
    required = {
        "id",
        "disabled",
        "display",
        "focused",
        "region",
        "clip",
        "fully_visible",
    }
    if not required.issubset(expected) or expected.keys() != actual.keys():
        return False
    for button in (expected, actual):
        if (
            button["disabled"] is not False
            or button["display"] is not True
            or button["fully_visible"] is not True
        ):
            return False
    if any(
        type(expected[key]) is not type(actual[key]) or expected[key] != actual[key]  # noqa: E721 - Exact boundary types reject bool/subclasses.
        for key in expected
        if key != "clip"
    ):
        return False
    region = expected["region"]
    clips = (expected["clip"], actual["clip"])
    for rect in (region, actual["region"], *clips):
        if (
            type(rect) is not list  # noqa: E721 - Exact boundary types reject bool/subclasses.
            or len(rect) != 4
            or any(type(value) is not int for value in rect)  # noqa: E721 - Exact boundary types reject bool/subclasses.
            or rect[2] <= 0
            or rect[3] <= 0
        ):
            return False
    x, y, width, height = region
    return all(
        clip[0] <= x
        and clip[1] <= y
        and x + width <= clip[0] + clip[2]
        and y + height <= clip[1] + clip[3]
        for clip in clips
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-native-observer", action="store_true")
    arguments = parser.parse_args()
    if not arguments.allow_native_observer:
        parser.error(
            "User alternate-method authorization + root display lease/freeze required"
        )
    selected = guard.SELECTION
    run = guard.RUN
    if (
        selected["shape"] != "single"
        or selected["negative_controls"]
        or (selected["columns"], selected["rows"]) != (120, 40)
    ):
        raise RuntimeError("Only current-feature single120x40 rehearsal supported")
    from native_rehearsal_observation import EventReader, classify_pixels, check_scope
    from native_series_method import (
        LIMITS,
        authorized_round_limit,
        classify_feedback,
        classify_terminal,
        settlement_pair,
        planned_outcomes,
        input_components,
        qpc_perf_interval,
        nearest_rank_p95,
    )

    round_limit = authorized_round_limit(selected, run)
    refs = json.loads((run / "exact-pixel-refs.json").read_text(encoding="utf-8"))
    for relative, digest in refs["source_pins"].items():
        if hashlib.sha256((run / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Pinned root reference changed")
    if round_limit == 41:
        terminal_review = json.loads(
            Path(selected["method_authorization"]["terminal_review"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        refs["terminal_patches"] = terminal_review["patches"]
    for relative, digest in refs["ui_source_pins"].items():
        if hashlib.sha256((guard.REPO / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Pinned current-feature UI/reference owner changed")
    reader = EventReader(run / "rehearsal-events.jsonl")
    event_rows = []
    latest_state = None
    if (run / "native-smoke-result.json").exists() or (
        run / "native-ready.json"
    ).exists():
        raise RuntimeError(
            "Fresh unique native run required; evidence never overwritten"
        )
    frozen = guard.verify_frozen()
    user, kernel, gdi = win32()
    before_clock = clock_bounds(kernel)
    nonce = selected["run_nonce"]
    child = [
        sys.executable,
        "-I",
        str(run / "native_product_child.py"),
        "--allow-native-product",
        "--nonce",
        nonce,
        "--columns",
        str(selected["columns"]),
        "--rows",
        str(selected["rows"]),
    ]
    child += (
        ["--negative-controls"]
        if selected["negative_controls"]
        else ["--shape", selected["shape"]]
    )
    child += [
        "--diagnostic-stack"
        if selected["diagnostic_stack"]
        else "--no-diagnostic-stack"
    ]

    result = {
        "qualified": False,
        "product_samples": 0,
        "revision": frozen["revision"],
        "scope": "actual fullapp smoke/controls only; independent pixel classification required",
        "clock_before": before_clock,
        "captures": [],
        "input": None,
        "errors": [],
        "conditions": selected["conditions"],
        "diagnostic_stack": selected["diagnostic_stack"],
    }
    process = None
    ready = None
    sequence = 0
    active_round = None
    active_dir = run
    PNG_process_bytes = 0
    seen_prefixes = set()
    result.update(
        method_mode=selected.get("method_mode", "validation"),
        runtime_round_limit=round_limit,
        planned_qualification_rounds=41,
        rounds=[],
    )

    def control(command=None, _diagnostic=False, **values):
        control_started = time.perf_counter_ns()
        connection = http.client.HTTPConnection(
            "127.0.0.1", ready["control_port"], timeout=40
        )
        headers = {
            "Authorization": "Bearer " + nonce,
            "Content-Type": "application/json",
        }
        try:
            connection.request(
                "GET" if command is None else "POST",
                "/diagnostic-status" if _diagnostic else "/",
                None if command is None else json.dumps({"command": command, **values}),
                headers,
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            if response.status != 200:
                raise RuntimeError(
                    "Actual native control refused:" + json.dumps(payload)
                )
            return payload
        finally:
            connection.close()
            record = active_round if active_round is not None else result
            spans = record.setdefault("control_spans_ns", [])
            if len(spans) < (
                LIMITS["control_spans_round"]
                if active_round is not None
                else LIMITS["control_spans_setup"]
            ):
                spans.append(
                    {
                        "command": command or "status",
                        "elapsed_ns": time.perf_counter_ns() - control_started,
                    }
                )
            else:
                record["control_span_overflow"] = True

    def poll_events():
        nonlocal latest_state
        for row in reader.poll():
            event_rows.append(row)
            if row.get("stage") == "geometry_after_refresh":
                latest_state = row["state"]
        return latest_state

    def eligible_scope(state):
        check_scope(state)

    try:
        process = NativeConsoleProcess(child)
        result["launcher_pid"] = process.pid
        startup_started = time.monotonic()
        startup_phase_at = startup_started
        last_startup_phase = None
        initial_view_attempted = False
        result["whole_startup_start_ns"] = time.perf_counter_ns()
        while not (run / "native-window.json").exists():
            phase_path = run / "native-startup-phases.jsonl"
            if phase_path.exists():
                phases = []
                for line in phase_path.read_text(encoding="utf-8").splitlines():
                    try:
                        phases.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                observed = phases[-1]["stage"] if phases else None
                if observed != last_startup_phase:
                    last_startup_phase = observed
                    startup_phase_at = time.monotonic()
                result["startup_phases"] = phases
            if (
                not initial_view_attempted
                and (run / "native-bootstrap-window.json").exists()
            ):
                initial_view_attempted = True
                descriptor = json.loads(
                    (run / "native-bootstrap-window.json").read_text(encoding="utf-8")
                )
                if descriptor["revision"] != frozen["revision"]:
                    raise RuntimeError("Pre-App source revision mismatch")
                try:
                    from native_bootstrap import retain_initial_owned_view

                    result["initial_owned_view"] = retain_initial_owned_view(
                        user, kernel, gdi, process, descriptor, run, nonce
                    )
                    print(
                        json.dumps(
                            {
                                "pre_product_diagnostic_png": str(
                                    run / result["initial_owned_view"]["file"]
                                ),
                                "qualification": False,
                            }
                        ),
                        flush=True,
                    )
                except BaseException as exc:
                    result["initial_owned_view_failure"] = {
                        "type": type(exc).__name__,
                        "reason": str(exc),
                    }
                    raise
            if (run / "native-early-error.json").exists():
                result["early_preflight_error"] = json.loads(
                    (run / "native-early-error.json").read_text(encoding="utf-8")
                )
                if (run / "native-early-window.json").exists():
                    early = json.loads(
                        (run / "native-early-window.json").read_text(encoding="utf-8")
                    )
                    if early["nonce"] != nonce or not (
                        early["pid"] == process.pid
                        or early["launcher_parent_pid"] == process.pid
                    ):
                        raise RuntimeError("Early owned window process/nonce mismatch")
                    hwnd = early["hwnd"]
                    expected = early["geometry"]
                    if expected["class"] != "ConsoleWindowClass":
                        raise RuntimeError("Early owned native class unsupported")

                    def validate_early():
                        if (
                            process.poll() is not None
                            or not user.IsWindowVisible(hwnd)
                            or user.IsIconic(hwnd)
                            or user.GetForegroundWindow() != hwnd
                            or geometry(user, hwnd) != expected
                        ):
                            raise RuntimeError(
                                "Early owned native visibility/owner/geometry changed"
                            )

                    image = capture_owned_roi(
                        user,
                        kernel,
                        gdi,
                        hwnd,
                        expected,
                        (0, 0, *expected["client"]),
                        validate_early,
                    )
                    from PIL import Image

                    path = run / "owned-early-native-refusal.png"
                    Image.frombytes(
                        "RGB",
                        (image["width"], image["height"]),
                        image["pixels"],
                        "raw",
                        "BGRX",
                    ).save(path)
                    result["early_refusal_capture"] = {
                        key: value for key, value in image.items() if key != "pixels"
                    }
                    result["early_refusal_capture"]["file"] = path.name
                    print(
                        json.dumps(
                            {
                                "early_refusal_png": str(path),
                                "typed_exception": result["early_preflight_error"][
                                    "type"
                                ],
                            }
                        ),
                        flush=True,
                    )
                    write_json(
                        run / "native-early-capture-ack.json",
                        {"nonce": nonce, "file": path.name},
                    )
                raise RuntimeError(
                    "Actual early native preflight refusal: "
                    + result["early_preflight_error"]["type"]
                )
            if process.poll() is not None:
                raise RuntimeError("Actual native app exited before ready")
            if (run / "native-setup-error.json").exists():
                raise RuntimeError(
                    "Actual native setup refusal:"
                    + str(run / "native-setup-error.json")
                )
            if (
                time.monotonic() - startup_phase_at > 60
                or time.monotonic() - startup_started > 300
            ):
                raise RuntimeError(
                    "Actual native startup exceeded finite60s actual phase/300s total diagnostic bound"
                )
            time.sleep(0.02)
        result["whole_startup_end_ns"] = time.perf_counter_ns()
        result["whole_startup_elapsed_ns"] = (
            result["whole_startup_end_ns"] - result["whole_startup_start_ns"]
        )
        ready = json.loads((run / "native-window.json").read_text(encoding="utf-8"))
        if (
            not (
                ready["pid"] == process.pid
                or ready["launcher_parent_pid"] == process.pid
            )
            or ready["nonce"] != nonce
        ):
            raise RuntimeError("Owned launcher/direct-child/nonce mismatch")
        if ready["revision"] != frozen["revision"]:
            raise RuntimeError("Native source identity mismatch")
        result["ready"] = ready
        guard.allow_loopback_port(
            ready["control_port"], "child_control", owner_pid=ready["pid"]
        )
        hello = control()
        if hello["pid"] != ready["pid"] or hello["nonce"] != nonce:
            raise RuntimeError("Owned control PID/nonce mismatch")
        hwnd = ready["hwnd"]
        expected = ready["native"]["geometry"]
        font = ready["native"]["font"]
        if expected["class"] != "ConsoleWindowClass":
            raise RuntimeError("Owned native Console class unavailable")
        exact_cell_map = expected["client"] == [
            selected["columns"] * font[0],
            selected["rows"] * font[1],
        ]
        initial_expected = dict(expected)
        bootstrap = {
            "initial_expected": initial_expected,
            "states": [],
            "activation": [],
            "measured": False,
        }
        result["window_bootstrap"] = bootstrap

        def window_state():
            foreground = int(user.GetForegroundWindow() or 0)
            foreground_pid = W.DWORD()
            if foreground:
                user.GetWindowThreadProcessId(foreground, C.byref(foreground_pid))
            actual = geometry(user, hwnd)
            fields = {key: actual.get(key) == expected.get(key) for key in expected}
            return {
                "observed_ns": time.perf_counter_ns(),
                "hwnd": hwnd,
                "launcher_alive": process.poll() is None,
                "visible": bool(user.IsWindowVisible(hwnd)),
                "iconic": bool(user.IsIconic(hwnd)),
                "foreground_hwnd": foreground,
                "foreground_pid": foreground_pid.value,
                "foreground_matches": foreground == hwnd,
                "expected_geometry": dict(expected),
                "actual_geometry": actual,
                "field_matches": fields,
            }

        def remember(state):
            if len(bootstrap["states"]) < 256:
                bootstrap["states"].append(state)
            write_json(run / "native-window-bootstrap.json", bootstrap)

        def own_for_activation():
            state = window_state()
            remember(state)
            if (
                not state["launcher_alive"]
                or not state["visible"]
                or state["iconic"]
                or any(
                    not state["field_matches"][key]
                    for key in ("host_pid", "class", "dpi")
                )
            ):
                raise RuntimeError("Own console bootstrap identity/visibility refused")
            return state

        # Actual startup may move/activate before the observation interval. This
        # bounded bootstrap has no product samples and never resets a measured
        # expectation. The exact native grid map is still mandatory afterward.
        state = own_for_activation()
        if not state["foreground_matches"]:
            user.SetForegroundWindow.argtypes = [W.HWND]
            user.SetForegroundWindow.restype = W.BOOL
            bootstrap["activation"].append(
                {
                    "set_foreground_returned": bool(user.SetForegroundWindow(hwnd)),
                    "observed_ns": time.perf_counter_ns(),
                }
            )
            time.sleep(0.1)
            state = own_for_activation()
            if not state["foreground_matches"]:
                from native_bootstrap import activate_owned_caption

                bootstrap["activation"].append(
                    activate_owned_caption(
                        user,
                        hwnd,
                        own_for_activation,
                        run / "native-caption-activation.jsonl",
                    )
                )
        stable_started = None
        last_geometry = None
        settle_deadline = time.monotonic() + 10
        while time.monotonic() < settle_deadline:
            state = own_for_activation()
            actual = state["actual_geometry"]
            mapped = actual["client"] == [
                selected["columns"] * font[0],
                selected["rows"] * font[1],
            ]
            if state["foreground_matches"] and mapped:
                if actual == last_geometry:
                    if time.monotonic() - stable_started >= 0.5:
                        break
                else:
                    last_geometry = actual
                    stable_started = time.monotonic()
            else:
                last_geometry = None
                stable_started = None
            time.sleep(0.05)
        else:
            raise RuntimeError(
                "Own console bootstrap foreground/exact-map did not settle within10s"
            )
        expected = dict(state["actual_geometry"])
        exact_cell_map = True
        bootstrap["settled_expected"] = dict(expected)
        bootstrap["settled_ns"] = time.perf_counter_ns()
        remember(state)

        def validate():
            state = window_state()
            bad = [key for key, value in state["field_matches"].items() if not value]
            if not state["launcher_alive"]:
                bad.append("launcher_alive")
            if not state["visible"]:
                bad.append("visible")
            if state["iconic"]:
                bad.append("iconic")
            if not state["foreground_matches"]:
                bad.append("foreground")
            if bad:
                state["failed_fields"] = bad
                result["last_window_validation_failure"] = state
                write_json(run / "native-window-validation-failure.json", state)
                raise RuntimeError("Owned native state changed: " + ",".join(bad))

        def persist_image(metadata, image, label):
            nonlocal PNG_process_bytes
            record = active_round if active_round is not None else result
            failure = label.startswith("failure")
            count = record.get("persisted_PNGs", 0)
            if count >= LIMITS["PNGs_round"] + (1 if failure else 0):
                raise RuntimeError("Perround persisted PNG bound; missing retained")
            filename = f"owned-{len(record.get('captures',[])):03d}-{label}.png"
            path = active_dir / filename
            if path.exists():
                raise RuntimeError("Owned evidence would be overwritten")
            started = time.perf_counter_ns()
            image.save(path)
            size = path.stat().st_size
            record["PNG_bytes"] = record.get("PNG_bytes", 0) + size
            PNG_process_bytes += size
            metadata.update(
                file=path.relative_to(run).as_posix(),
                png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                persist_encode_ns=time.perf_counter_ns() - started,
            )
            record["persisted_PNGs"] = count + 1
            if (
                record["PNG_bytes"] > LIMITS["PNG_bytes_round"]
                or PNG_process_bytes > LIMITS["PNG_bytes_process"]
            ):
                raise RuntimeError(
                    "Declared perround/process PNG byte bound; invalid retained"
                )

        def capture(
            label,
            rect,
            before=None,
            comparison_regions=None,
            semantic_poll=True,
            persist=True,
        ):
            nonlocal sequence
            if sequence >= 64:
                raise RuntimeError("Rehearsal capture capacity exceeded; unqualified")
            record = active_round if active_round is not None else result
            capture_started = time.perf_counter_ns()
            image = capture_owned_roi(user, kernel, gdi, hwnd, expected, rect, validate)
            after = (
                control("capture-state")
                if label.startswith(("card", "settlement"))
                else poll_events()
                if semantic_poll
                else None
            )
            if label.startswith("card") and before["snapshot"].get(
                "card_identity"
            ) != after["snapshot"].get("card_identity"):
                raise RuntimeError("Actual card generation changed during capture")
            comparisons = {}
            for name, sub in (comparison_regions or {}).items():
                x, y, width, height = sub
                x -= rect[0]
                y -= rect[1]
                if (
                    min(x, y) < 0
                    or min(width, height) <= 0
                    or x + width > image["width"]
                    or y + height > image["height"]
                ):
                    raise RuntimeError(
                        "Actual stable comparison region outside own card"
                    )
                pixels = b"".join(
                    image["pixels"][
                        ((y + row) * image["width"] + x) * 4 : (
                            (y + row) * image["width"] + x + width
                        )
                        * 4
                    ]
                    for row in range(height)
                )
                comparisons[name] = hashlib.sha256(pixels).hexdigest()
            # Retain only after all owner/geometry/occlusion postchecks pass.
            # Images contain this owned synthetic profile only, never desktop.
            from PIL import Image

            sequence += 1
            actual_image = Image.frombytes(
                "RGB", (image["width"], image["height"]), image["pixels"], "raw", "BGRX"
            )
            automatic = None
            classification_started = time.perf_counter_ns()
            if label.startswith("card"):
                try:
                    eligible_scope(before)
                    eligible_scope(after)
                    if any(
                        value["snapshot"].get("information_ready") is not True
                        or value["snapshot"].get("submitted")
                        or value["snapshot"].get("card_display") is not True
                        for value in (before, after)
                    ):
                        raise RuntimeError("Actual committing controls not ready")
                    action = comparison_regions["action"]
                    local_action = [
                        action[0] - rect[0],
                        action[1] - rect[1],
                        action[2],
                        action[3],
                    ]
                    automatic = classify_pixels(
                        actual_image.convert("RGBA"),
                        before["snapshot"]["current"]["correlation"],
                        local_action,
                        refs,
                    )
                except Exception as observation_error:
                    automatic = {
                        "nonce_exact": False,
                        "action_enabled": False,
                        "reason": "actual_scope_or_eligibility_refusal",
                        "error_type": type(observation_error).__name__,
                    }
            metadata = {key: value for key, value in image.items() if key != "pixels"}
            feedback = (
                classify_feedback(
                    actual_image.convert("RGBA"), before, expected_round, refs
                )
                if label.startswith("feedback") and before is not None
                else None
            )
            terminal = (
                classify_terminal(
                    actual_image.convert("RGBA"), before, expected_round, refs
                )
                if label.startswith("settlement") and before is not None
                else None
            )
            if feedback is not None:
                later = (
                    classify_feedback(
                        actual_image.convert("RGBA"), after, expected_round, refs
                    )
                    if after is not None
                    else {}
                )
                if later.get("accepted_pixels") is not True:
                    feedback["accepted_pixels"] = False
                    feedback["after_state_refused"] = True
            if terminal is not None:
                later = (
                    classify_terminal(
                        actual_image.convert("RGBA"), after, expected_round, refs
                    )
                    if after is not None
                    else {}
                )
                if later.get("candidate_pixels") is not True or later.get(
                    "patches"
                ) != terminal.get("patches"):
                    terminal["candidate_pixels"] = False
                    terminal["settled_pixels"] = False
                    terminal["after_state_refused"] = True
            metadata.update(
                pre_persist_capture_total_ns=time.perf_counter_ns() - capture_started,
                automatic_pixel_classification=automatic,
                classification_compute_ns=time.perf_counter_ns()
                - classification_started,
                comparison_pixel_sha256=comparisons,
                comparison_client_pixel_regions=comparison_regions,
                automatic_feedback_classification=feedback,
                automatic_terminal_classification=terminal,
                semantic_before=before,
                semantic_after=after,
                independent_pixel_review=False,
            )
            record.setdefault("captures", []).append(metadata)
            if persist:
                persist_image(metadata, actual_image, label)
            if automatic is not None and not (
                automatic["nonce_exact"] and automatic["action_enabled"]
            ):
                raise RuntimeError(
                    "Actual new-capture exact nonce/action classifier refused; input withheld"
                )
            return metadata if persist else (metadata, actual_image)

        full = (0, 0, *expected["client"])
        startup = capture("startup", full, semantic_poll=False)
        write_json(
            run / "native-startup-image-ready.json",
            {
                "file": startup["file"],
                "png_sha256": startup["png_sha256"],
                "pid": ready["pid"],
                "hwnd": hwnd,
                "nonce": nonce,
                "qualified": False,
                "product_samples": 0,
            },
        )
        print(
            json.dumps(
                {
                    "startup_png": str(run / startup["file"]),
                    "startup_receipt": str(run / "native-startup-image-ready.json"),
                }
            ),
            flush=True,
        )
        if not ready.get("driver_source_proof", {}).get("valid_stock_native_driver"):
            write_json(
                run / "native-failure-capture-ack.json",
                {"nonce": nonce, "file": startup["file"]},
            )
            raise RuntimeError(
                "Actual driver MRO/source identity refused; owned startup diagnostic PNG retained"
            )
        if not exact_cell_map:
            try:
                result["diagnostic_status"] = control(_diagnostic=True)
            except BaseException as exc:
                result["diagnostic_status_error"] = type(exc).__name__
            write_json(
                run / "native-failure-capture-ack.json",
                {"nonce": nonce, "file": startup["file"]},
            )
            raise RuntimeError(
                "Actual post-WindowsDriver mapping mismatch; owned startup PNG retained, no coordinate action"
            )
        attach_started = time.monotonic()
        last_phase_at = attach_started
        last_phase = None
        phase_path = run / "product-profile/attach-stages.jsonl"
        while not (run / "native-ready.json").exists():
            if phase_path.exists():
                lines = phase_path.read_text(encoding="utf-8").splitlines()
                phases = []
                for line in lines:
                    try:
                        phases.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Only an incomplete trailing append.
                observed = phases[-1]["stage"] if phases else None
                if observed != last_phase:
                    last_phase = observed
                    last_phase_at = time.monotonic()
                result["attach_phases"] = phases
            if (
                (run / "native-setup-error.json").exists()
                or time.monotonic() - last_phase_at > 60
                or time.monotonic() - attach_started > 300
            ):
                failure = capture("attach-failure", full, semantic_poll=False)
                try:
                    result["diagnostic_status"] = control(_diagnostic=True)
                except BaseException as exc:
                    result["diagnostic_status_error"] = type(exc).__name__
                write_json(
                    run / "native-failure-capture-ack.json",
                    {"nonce": nonce, "file": failure["file"]},
                )
                raise RuntimeError(
                    "Actual native fixture attach failed/stalled; owned PNG/status/stacks retained"
                )
            if process.poll() is not None:
                raise RuntimeError("Actual native app exited during attach")
            time.sleep(0.02)
        result["attach_ready"] = json.loads(
            (run / "native-ready.json").read_text(encoding="utf-8")
        )

        # ONE named pre-target bootstrap point; never retry activation during
        # a request/measurement/card decision. Keep the original expected map.
        def own_title_hash():
            user.GetWindowTextW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
            user.GetWindowTextW.restype = C.c_int
            title = C.create_unicode_buffer(1024)
            length = user.GetWindowTextW(hwnd, title, len(title))
            if length == len(title) - 1:
                raise RuntimeError("Own title observation truncated")
            return hashlib.sha256(title.value.encode("utf-8")).hexdigest()

        before_post_attach = window_state()
        title_before = own_title_hash()
        post_attach = {
            "phase": "post_attach_before_target_submit",
            "before": before_post_attach,
            "own_title_sha256_before": title_before,
            "normal_caption_activation_attempts": 0,
            "expected_geometry_unchanged": dict(expected),
            "no_target_yet": True,
        }
        result["post_attach_activation"] = post_attach

        def persist_post_attach():
            write_json(run / "native-post-attach-activation.json", post_attach)

        def own_post_attach():
            state = window_state()
            post_attach["last_observed"] = state
            persist_post_attach()
            if (
                not state["launcher_alive"]
                or not state["visible"]
                or state["iconic"]
                or not all(state["field_matches"].values())
                or own_title_hash() != title_before
            ):
                raise RuntimeError(
                    "Post-attach own process/title/class/geometry/DPI changed; no activation"
                )
            return state

        own_post_attach()
        if not before_post_attach["foreground_matches"]:
            post_attach["normal_caption_activation_attempts"] = 1
            persist_post_attach()
            from native_bootstrap import activate_owned_caption

            post_attach["caption"] = activate_owned_caption(
                user, hwnd, own_post_attach, run / "native-post-attach-caption.jsonl"
            )
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if own_post_attach()["foreground_matches"]:
                    break
                time.sleep(0.05)
        post_attach["after"] = own_post_attach()
        post_attach["own_title_sha256_after"] = own_title_hash()
        persist_post_attach()
        validate()  # Original ALL-fields/foreground invariant remains mandatory.
        capture("post-attach-startup", full, semantic_poll=False)
        baseline = control("capture-state")
        capture("baseline", full, baseline)
        expected_round = None

        def await_observed_review(request, stem):
            deadline = time.monotonic() + 45
            request = {**request, "review_deadline_monotonic": deadline}
            required = active_dir / (stem + "-required.json")
            reviewed = active_dir / (stem + ".json")
            write_json(required, request)
            print(
                json.dumps(
                    {
                        "actual_image": request["source_png"],
                        "pixel_review_required": str(required),
                        "stage": request["stage"],
                        "review_deadline_monotonic": deadline,
                    }
                ),
                flush=True,
            )
            while not reviewed.exists():
                if time.monotonic() > deadline or process.poll() is not None:
                    raise RuntimeError(
                        "Independent first-live reference review unavailable within45s"
                    )
                time.sleep(0.05)
            actual = json.loads(reviewed.read_text(encoding="utf-8"))
            if (
                time.monotonic() > deadline
                or any(actual.get(key) != value for key, value in request.items())
                or actual.get("observed") is not True
                or not actual.get("reviewer")
            ):
                raise RuntimeError(
                    "Stale/mismatched first-live reference review refused"
                )
            result.setdefault("first_live_reference_reviews", []).append(
                {
                    "required": required.relative_to(run).as_posix(),
                    "reviewed": reviewed.relative_to(run).as_posix(),
                    "review": actual,
                }
            )
            return actual

        def run_round(round_index, result):
            nonlocal expected_round
            preparation_started_ns = time.perf_counter_ns()
            event_start = len(event_rows)
            stage = control("stage", shape=selected["shape"])
            result["stage"] = stage
            expected_correlation = stage["snapshot"]["current"]["correlation"]
            if expected_correlation[:12] in seen_prefixes:
                raise RuntimeError(
                    "Current visible QA12 prefix collision; no replacement"
                )
            seen_prefixes.add(expected_correlation[:12])
            result["whole_preparation_start_ns"] = preparation_started_ns
            wait_started = time.monotonic()
            candidate = None
            last_source_ns = time.perf_counter_ns()
            result["actual_preflight_progress"] = []
            while time.monotonic() - wait_started < 300:
                snapshot = poll_events()
                # Check PRIOR progress immediately after a possibly slow poll;
                # a newly returned stage must not erase an expired owner phase.
                received_at = time.monotonic()
                received_ns = time.perf_counter_ns()
                if received_at - wait_started >= 300:
                    raise RuntimeError(
                        "Actual preflight returned after original finite300s bound"
                    )
                if received_ns - last_source_ns > 60_000_000_000:
                    raise RuntimeError(
                        "Actual preflight poll returned after prior finite60s owner-phase bound"
                    )
                real = [
                    row
                    for row in event_rows[event_start:]
                    if row.get("correlation") == expected_correlation
                    and row.get("stage") != "geometry_after_refresh"
                ]
                if any(
                    type(row.get("timestamp_ns")) is not int  # noqa: E721 - Exact boundary types reject bool/subclasses.
                    or row["timestamp_ns"] < 0
                    or row["timestamp_ns"] > received_ns
                    for row in real
                ):
                    raise RuntimeError(
                        "Invalid or future actual stage timestamp; no deadline extension"
                    )
                if real:
                    last_source_ns = max(row["timestamp_ns"] for row in real)
                if time.perf_counter_ns() - last_source_ns > 60_000_000_000:
                    raise RuntimeError(
                        "Actual recorded preflight stage exceeded original finite60s bound"
                    )
                if snapshot is not None:
                    view = snapshot["snapshot"]
                    result["last_action_wait_snapshot"] = snapshot
                    buttons = [
                        item
                        for item in view["controls"]
                        if item["id"].startswith("approval-fast-approve-")
                        and not item["disabled"]
                        and item["display"]
                        and item["fully_visible"]
                    ]
                    region = view.get("card_region")
                    if (
                        view["current"]["correlation"] == expected_correlation
                        and view.get("card_display")
                        and view.get("card_identity")
                        and view.get("information_ready") is True
                        and region
                        and region["fully_visible"]
                        and len(buttons) == 1
                    ):
                        eligible_scope(snapshot)
                        if (
                            time.monotonic() - wait_started >= 300
                            or time.perf_counter_ns() - last_source_ns > 60_000_000_000
                        ):
                            raise RuntimeError(
                                "Actual candidate validation exceeded original preflight bounds"
                            )
                        candidate = (snapshot, buttons[0])
                        break
                time.sleep(0.01)
            result["whole_preparation_end_ns"] = time.perf_counter_ns()
            result["whole_preparation_elapsed_ns"] = (
                result["whole_preparation_end_ns"] - preparation_started_ns
            )
            if candidate is None:
                raise RuntimeError(
                    "Actual preflight exceeded finite300s diagnostic bound; no local approval target inferred"
                )
            semantic, button = candidate
            view = semantic["snapshot"]
            correlation = view["current"]["correlation"]
            identity = view["card_identity"]
            expected_round = {
                "correlation": correlation,
                "card_identity": identity,
                "run_id": view["current"]["run_id"],
                "session_id": view["current"]["session_id"],
            }
            cells = view["card_region"]["region"]
            rect = tuple(value * font[index % 2] for index, value in enumerate(cells))
            summary = semantic["native"].get("request_summary_cell_region")
            if summary is None:
                raise RuntimeError(
                    "Actual visible request-summary pixel region unavailable"
                )
            comparison_regions = {
                name: tuple(
                    value * font[index % 2] for index, value in enumerate(cells)
                )
                for name, cells in [
                    ("request_summary", summary),
                    ("action", button["region"]),
                ]
            }
            card = capture("card-candidate", rect, semantic, comparison_regions)
            deadline = time.monotonic() + 45
            result["pixel_review_deadline_monotonic"] = deadline
            result["pixel_reviews"] = []
            for review_generation in range(2):
                # The ONE original deadline covers both independent reviews.
                # A fresh captured image never grants more time or authority.
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Original45s pixel review deadline expired; input withheld"
                    )
                suffix = "" if review_generation == 0 else "-refresh1"
                required_path = active_dir / (
                    "native-card-review-required" + suffix + ".json"
                )
                reviewed_path = active_dir / ("native-card-review" + suffix + ".json")
                review_request = {
                    "stage": "card_actionable",
                    "correlation": correlation,
                    "card_identity": identity,
                    "control_id": button["id"],
                    "png_sha256": card["png_sha256"],
                    "pixel_sha256": card["pixel_sha256"],
                    "file": card["file"],
                    "review_generation": review_generation,
                    "review_request_id": selected["run_nonce"]
                    + ":round:"
                    + str(round_index)
                    + ":card:"
                    + str(review_generation),
                    "review_deadline_monotonic": deadline,
                    "review_required": "Independently inspect this exact actual owned image: unique fixture suffix/generation, intended action label and enabled action pixels. Write matching "
                    + reviewed_path.name
                    + " with observed=true and reviewer identity. Fresh review is required even when only the highlight changed; original45s deadline is NOT reset.",
                    "qualified": False,
                    "product_samples": 0,
                }
                if round_limit == 1:
                    write_json(required_path, review_request)
                    print(
                        json.dumps(
                            {
                                "actual_card_png": str(run / card["file"]),
                                "pixel_review_required": str(required_path),
                                "review_generation": review_generation,
                                "review_deadline_monotonic": deadline,
                            }
                        ),
                        flush=True,
                    )
                    while not reviewed_path.exists():
                        if time.monotonic() > deadline:
                            raise RuntimeError(
                                "No independent actual-card pixel classification within original45s; input withheld"
                            )
                        if process.poll() is not None:
                            raise RuntimeError("Owned app retired before pixel review")
                        time.sleep(0.05)
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            "Original45s pixel review deadline expired; input withheld"
                        )
                    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
                    for key in (
                        "stage",
                        "correlation",
                        "card_identity",
                        "control_id",
                        "png_sha256",
                        "pixel_sha256",
                        "review_generation",
                        "review_request_id",
                    ):
                        if reviewed.get(key) != review_request[key]:
                            raise RuntimeError("Stale/mismatched pixel review rejected")
                    if reviewed.get("observed") is not True or not reviewed.get(
                        "reviewer"
                    ):
                        raise RuntimeError(
                            "Missing independent visible-pixel classification"
                        )
                else:
                    # Explicitly prequalified classifier record, never a forged Root review file.
                    reviewed = {
                        **review_request,
                        "observed": True,
                        "reviewer": "Root-qualified exact-newRGBA method",
                        "automatic_method_record": True,
                    }
                    if (
                        not card["automatic_pixel_classification"]["nonce_exact"]
                        or not card["automatic_pixel_classification"]["action_enabled"]
                    ):
                        raise RuntimeError(
                            "Qualified card pixel method refused; no input"
                        )
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            "Original45s card deadline expired before automatic review"
                        )
                result["pixel_reviews"].append(
                    {
                        "required_file": required_path.name,
                        "review_file": reviewed_path.name,
                        "request": review_request,
                        "review": reviewed,
                    }
                )
                live = control("capture-state")
                eligible_scope(live)
                live_view = live["snapshot"]
                current_button = next(
                    (
                        item
                        for item in live_view["controls"]
                        if item["id"] == button["id"]
                    ),
                    None,
                )
                if (
                    live_view.get("card_identity") != identity
                    or live_view["current"]["correlation"] != correlation
                    or not same_eligible_action_button(button, current_button)
                ):
                    raise RuntimeError("Card/action changed after pixel review")
                preclick = capture(
                    "card-before-input"
                    if review_generation == 0
                    else "card-before-input-refresh1",
                    rect,
                    live,
                    comparison_regions,
                )
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Original45s pixel review deadline expired; input withheld"
                    )
                if (
                    preclick["comparison_pixel_sha256"]
                    == card["comparison_pixel_sha256"]
                ):
                    break
                if review_generation != 0:
                    raise RuntimeError(
                        "Action pixels changed a second time after fresh review; no click"
                    )
                if (
                    preclick["comparison_pixel_sha256"]["request_summary"]
                    != card["comparison_pixel_sha256"]["request_summary"]
                ):
                    raise RuntimeError(
                        "Actual request-summary pixels changed after reviewed frame; no fresh review or click"
                    )
                # A changed action image is ONLY a new review candidate. Its
                # existing saved PNG and both original review files survive.
                after_view = preclick["semantic_after"]["snapshot"]
                after_button = next(
                    (
                        item
                        for item in after_view["controls"]
                        if item["id"] == button["id"]
                    ),
                    None,
                )
                if (
                    after_view.get("card_identity") != identity
                    or after_view["current"]["correlation"] != correlation
                    or not same_eligible_action_button(button, after_button)
                ):
                    raise RuntimeError(
                        "Card/action semantics changed during fresh candidate capture; no click"
                    )
                card = preclick
            validate()
            x, y, width, height = button["region"]
            point = W.POINT(
                expected["origin"][0] + int((x + width / 2) * font[0]),
                expected["origin"][1] + int((y + height / 2) * font[1]),
            )
            if user.GetAncestor(user.WindowFromPoint(point), 2) != hwnd:
                raise RuntimeError("Owned action point occluded")

            class MOUSEINPUT(C.Structure):
                _fields_ = [
                    ("dx", W.LONG),
                    ("dy", W.LONG),
                    ("mouseData", W.DWORD),
                    ("dwFlags", W.DWORD),
                    ("time", W.DWORD),
                    ("dwExtraInfo", C.c_size_t),
                ]

            class KEYBDINPUT(C.Structure):
                _fields_ = [
                    ("wVk", W.WORD),
                    ("wScan", W.WORD),
                    ("dwFlags", W.DWORD),
                    ("time", W.DWORD),
                    ("dwExtraInfo", C.c_size_t),
                ]

            class HARDWAREINPUT(C.Structure):
                _fields_ = [("uMsg", W.DWORD), ("wParamL", W.WORD), ("wParamH", W.WORD)]

            class UNION(C.Union):
                _fields_ = [
                    ("mi", MOUSEINPUT),
                    ("ki", KEYBDINPUT),
                    ("hi", HARDWAREINPUT),
                ]

            class INPUT(C.Structure):
                _anonymous_ = ("union",)
                _fields_ = [("type", W.DWORD), ("union", UNION)]

            user.SetCursorPos.argtypes = [C.c_int, C.c_int]
            user.SetCursorPos.restype = W.BOOL
            user.SendInput.argtypes = [W.UINT, C.POINTER(INPUT), C.c_int]
            user.SendInput.restype = W.UINT
            transport_before = qpc(kernel)
            if not user.SetCursorPos(point.x, point.y):
                raise RuntimeError("Own action OS pointer placement refused")
            validate()
            inputs = (INPUT * 2)()
            inputs[0].type = inputs[1].type = 0
            inputs[0].mi.dwFlags = 2
            inputs[1].mi.dwFlags = 4
            sent = user.SendInput(2, inputs, C.sizeof(INPUT))
            transport_after = qpc(kernel)
            result["input"] = {
                "kind": "actual Win32 mouse left down/up",
                "point": [point.x, point.y],
                "qpc_before": transport_before,
                "qpc_after": transport_after,
                "sent": sent,
                "card_identity": identity,
                "correlation": correlation,
                "review": reviewed,
                "no_direct_resolver_call": True,
            }
            if sent != 2:
                raise RuntimeError(
                    "SendInput did not admit exact owned gesture; UIPI or input capability failure"
                )
            accepted = None
            for _ in range(20):
                snapshot = poll_events() or live
                feedback, image = capture(
                    "feedback-candidate", full, snapshot, persist=False
                )
                classified = feedback["automatic_feedback_classification"]
                if classified and classified["accepted_pixels"]:
                    persist_image(feedback, image, "feedback-accepted")
                    accepted = feedback
                    break
                time.sleep(0.01)
            if accepted is None:
                raise RuntimeError(
                    "Actual accepted feedback not observed within finite20captures; invalid retained"
                )
            result["first_accepted_feedback"] = accepted
            if round_limit == 1:
                review = {
                    "stage": "feedback_accepted",
                    "correlation": correlation,
                    "card_identity": identity,
                    "source_png": str(run / accepted["file"]),
                    "png_sha256": accepted["png_sha256"],
                    "pixel_sha256": accepted["pixel_sha256"],
                    "title_RGBA_sha256": accepted["automatic_feedback_classification"][
                        "title_RGBA_sha256"
                    ],
                    "automatic_feedback_accepted": True,
                    "qualified": False,
                    "product_samples": 0,
                }
                await_observed_review(review, "native-feedback-review")
            result["collected"] = control("collect")
            first = None
            pair = None
            settle_started = time.monotonic()
            state = control("capture-state")
            for observation in range(LIMITS["settlement_observations"]):
                if time.monotonic() - settle_started > LIMITS["settlement_seconds"]:
                    break
                eligible_scope(state)
                if time.monotonic() - settle_started > LIMITS["settlement_seconds"]:
                    break
                frame, image = capture(
                    "settlement-candidate", full, state, persist=False
                )
                if time.monotonic() - settle_started > LIMITS["settlement_seconds"]:
                    break
                state = frame["semantic_after"]
                facts = frame["automatic_terminal_classification"]
                if facts and facts["candidate_pixels"]:
                    facts = {
                        **facts,
                        "qpc_start": frame["qpc_start"],
                        "qpc_end": frame["qpc_end"],
                    }
                    if first is not None and settlement_pair(first[0], facts):
                        pair = (first, (facts, frame, image))
                        break
                    first = (facts, frame, image)
                time.sleep(0.05)
            result["settlement_observational_cap"] = {
                "observations": LIMITS["settlement_observations"],
                "seconds": LIMITS["settlement_seconds"],
                "actual_elapsed_ns": int((time.monotonic() - settle_started) * 1e9),
                "missing_is_not_product_inconsistency": True,
            }
            if pair is None:
                raise RuntimeError(
                    "Two fresh current-generation settled ROI candidates missing within observational cap; no terminal defect inferred"
                )
            for index, (_facts, frame, image) in enumerate(pair):
                persist_image(frame, image, "settled-" + str(index + 1))
            if round_limit == 1:
                last = pair[1][1]
                patches = {
                    name: {
                        "RGBA_sha256": value,
                        "pixel_size": pair[1][0]["patch_pixel_sizes"][name],
                    }
                    for name, value in pair[1][0]["patches"].items()
                }
                review = {
                    "stage": "terminal_settlement",
                    "correlation": correlation,
                    "visible_generation_prefix": correlation[:12],
                    "run_id": expected_round["run_id"],
                    "session_id": expected_round["session_id"],
                    "source_png": str(run / last["file"]),
                    "png_sha256": last["png_sha256"],
                    "source_pair": [
                        {
                            "source_png": str(run / item[1]["file"]),
                            "png_sha256": item[1]["png_sha256"],
                            "qpc_start": item[1]["qpc_start"],
                            "qpc_end": item[1]["qpc_end"],
                        }
                        for item in pair
                    ],
                    "patches": patches,
                    "qualified": False,
                    "product_samples": 0,
                }
                observed = await_observed_review(review, "native-settlement-review")
                refs["terminal_patches"] = observed["patches"]
            if not all(
                classify_terminal(
                    image, frame["semantic_before"], expected_round, refs
                )["settled_pixels"]
                for _facts, frame, image in pair
            ):
                raise RuntimeError("Root-qualified terminal patch comparison refused")
            result["settled_pair"] = [frame for _facts, frame, _image in pair]
            result["rehearsal_events"] = [
                row
                for row in event_rows[event_start:]
                if row.get("correlation") == correlation
            ]
            result["final_observer_health"] = state["observer_health"]
            result["observer_condition"] = (
                "Passive stage spool plus bounded transition/preinput/postcollect ROI checks; all observation cost retained."
            )
            result["status"] = "valid"

        for round_index in range(round_limit):
            active_round = {
                "round_index": round_index,
                "temperature": "process_cold" if round_index == 0 else "warm",
                "status": "incomplete",
                "captures": [],
            }
            result["rounds"].append(active_round)
            active_dir = run / f"round-{round_index:03d}"
            active_dir.mkdir()
            sequence = 0
            try:
                run_round(round_index, active_round)
            except BaseException:
                active_round["status"] = "invalid"
                raise
            finally:
                result["round_outcomes"] = planned_outcomes(
                    result["rounds"], round_limit
                )
        active_round = None
        active_dir = run
        control("exit")
    except BaseException as exc:
        if "whole_startup_start_ns" in result and "whole_startup_end_ns" not in result:
            result["whole_startup_end_ns"] = time.perf_counter_ns()
            result["whole_startup_elapsed_ns"] = (
                result["whole_startup_end_ns"] - result["whole_startup_start_ns"]
            )
        if "whole_preparation_start_ns" in result:
            result["whole_preparation_end_ns"] = time.perf_counter_ns()
            result["whole_preparation_elapsed_ns"] = (
                result["whole_preparation_end_ns"]
                - result["whole_preparation_start_ns"]
            )
        result["errors"].append({"type": type(exc).__name__, "reason": str(exc)})
        # Before ORIGINAL runtime cleanup, retain actual task/owner outcome and
        # strictly owned diagnostic pixels. This is evidence, never a verdict.
        if (
            ready is not None
            and process is not None
            and process.poll() is None
            and not result.get("early_preflight_error")
        ):
            try:
                result["failure_snapshot_before_cleanup"] = control("snapshot")
            except BaseException as detail:
                result["failure_snapshot_error_type"] = type(detail).__name__
            try:
                if callable(locals().get("capture")):
                    result["failure_capture_before_cleanup"] = capture(
                        "failure-before-cleanup", full, semantic_poll=False
                    )
            except BaseException as detail:
                result["failure_capture_error_type"] = type(detail).__name__
            try:
                result["failure_status_before_cleanup"] = control(_diagnostic=True)
            except BaseException as detail:
                result["failure_status_error_type"] = type(detail).__name__
    finally:
        if ready is not None and process is not None and process.poll() is None:
            try:
                if result.get("early_preflight_error"):
                    raise RuntimeError(
                        "Product app was not started; no runtime exit command"
                    )
                if (
                    result.get("diagnostic_status")
                    and not (run / "native-ready.json").exists()
                ):
                    raise RuntimeError(
                        "UI thread stalled; original exit request unavailable"
                    )
                control("exit")
            except BaseException as exc:
                result["errors"].append(
                    {
                        "type": type(exc).__name__,
                        "during": "original runtime exit request",
                    }
                )
        if process is not None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                actual = ready.get("pid") if ready else None
                if (
                    actual
                    and actual != process.pid
                    and ready.get("launcher_parent_pid") == process.pid
                ):
                    kernel.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
                    kernel.OpenProcess.restype = W.HANDLE
                    kernel.TerminateProcess.argtypes = [W.HANDLE, W.UINT]
                    kernel.TerminateProcess.restype = W.BOOL
                    kernel.CloseHandle.argtypes = [W.HANDLE]
                    kernel.CloseHandle.restype = W.BOOL
                    handle = kernel.OpenProcess(1, False, actual)
                    if handle:
                        result["forced_verified_actual_child_exit"] = bool(
                            kernel.TerminateProcess(handle, 97)
                        )
                        kernel.CloseHandle(handle)
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)
                result["forced_own_child_cleanup"] = True
            result["launcher_exit_code"] = process.returncode
            process.close()
        try:
            for _ in range(165):
                previous_offset = reader.offset
                poll_events()
                if reader.offset == previous_offset:
                    break
            if reader.path.exists() and (
                reader.offset < reader.path.stat().st_size or reader.partial
            ):
                result["errors"].append({"type": "ObserverTailIncomplete"})
        except BaseException as observation_error:
            result["errors"].append(
                {
                    "type": "ObserverReadError",
                    "error_type": type(observation_error).__name__,
                }
            )
        result["rehearsal_events"] = event_rows
        stage_times = {}
        for row in event_rows:
            if row.get("stage") in (
                "request_complete",
                "local_checks_complete",
                "app_input_prequeue",
                "card_commit_enter",
                "card_commit_return",
                "resolver_entered",
                "host_resolve_return",
                "worker_released",
            ):
                stage_times.setdefault(row["stage"], row["timestamp_ns"])
        result["recorded_stage_times_ns"] = stage_times
        result["request_complete_to_checks_ns"] = (
            (stage_times["local_checks_complete"] - stage_times["request_complete"])
            if all(
                key in stage_times
                for key in ("request_complete", "local_checks_complete")
            )
            else None
        )
        result["checks_to_paint_evidence"] = (
            "Actual new-card capture QPC brackets + classification; no automatic latency/budget pass in rehearsal"
        )
        result["last_visible_output_to_card"] = {
            "status": "missing",
            "reason": "No independent prior presented output frame; scripted_output_start is not paint",
        }
        result["event_reader_IO"] = {
            "polls": reader.polls,
            "bytes": reader.offset,
            "max_poll_ns": reader.poll_max_ns,
        }
        result["data_limits"] = LIMITS
        result["PNG_process_bytes"] = PNG_process_bytes
        result["round_outcomes"] = planned_outcomes(result["rounds"], round_limit)
        health_path = run / "rehearsal-observer-health.json"
        if health_path.exists():
            try:
                result["exit_observer_health"] = json.loads(
                    health_path.read_text(encoding="utf-8")
                )
                if any(
                    result["exit_observer_health"].get(key) is not False
                    for key in ("loss", "write_failure", "observer_failure")
                ):
                    result["errors"].append(
                        {
                            "type": "ObserverLoss",
                            "reason": "Observation health failure; always unqualified",
                        }
                    )
            except BaseException as observation_error:
                result["errors"].append(
                    {
                        "type": "ObserverHealthReadError",
                        "error_type": type(observation_error).__name__,
                    }
                )
        else:
            result["observer_health_exit_missing"] = True
        result["clock_after"] = clock_bounds(kernel)
        for record in result["rounds"]:
            stages = record.get("rehearsal_events", [])
            times = {}
            for row in stages:
                times.setdefault(row.get("stage"), row.get("timestamp_ns"))
            record["request_complete_to_checks_ns"] = (
                (times["local_checks_complete"] - times["request_complete"])
                if all(
                    key in times
                    for key in ("request_complete", "local_checks_complete")
                )
                else None
            )
            card = next(
                (
                    cap
                    for cap in record.get("captures", [])
                    if (cap.get("automatic_pixel_classification") or {}).get(
                        "action_enabled"
                    )
                ),
                None,
            )
            feedback = record.get("first_accepted_feedback")
            if card and feedback:
                card_interval = qpc_perf_interval(
                    card["qpc_end"], [result["clock_before"], result["clock_after"]]
                )
                feedback_interval = qpc_perf_interval(
                    feedback["qpc_end"], [result["clock_before"], result["clock_after"]]
                )
                record["checks_to_actionable_capture_upper_ns"] = (
                    card_interval[1] - times["local_checks_complete"]
                    if "local_checks_complete" in times
                    else None
                )
                rows = [
                    row for row in stages if row.get("stage") == "app_input_prequeue"
                ]
                record["input_components"] = input_components(
                    rows, times.get("card_commit_enter"), feedback_interval[1]
                )
                record["feedback_capture_perf_interval_ns"] = feedback_interval
            record["last_visible_output_to_card"] = {
                "status": "missing",
                "reason": "No independent prior presented-output frame; fixture scripted text is not model inference or paint.",
            }
        result["qualification_runtime_held"] = round_limit == 1
        result["product_samples"] = 0
        result["observed_metric_round_candidates"] = (
            len([row for row in result["rounds"] if row.get("status") == "valid"])
            if round_limit == 41
            else 0
        )
        result["observed_warm_round_candidates"] = (
            len([row for row in result["rounds"][1:] if row.get("status") == "valid"])
            if round_limit == 41
            else 0
        )
        if (
            round_limit == 41
            and len(result["rounds"]) == 41
            and all(row.get("status") == "valid" for row in result["rounds"])
            and not result["errors"]
        ):
            warm = result["rounds"][1:]
            checks = [row.get("checks_to_actionable_capture_upper_ns") for row in warm]
            feedback = [
                (row.get("input_components") or {}).get(
                    "earliest_app_input_to_feedback_ns"
                )
                for row in warm
            ]
            if all(type(n) is int and n >= 0 for n in checks + feedback):  # noqa: E721 - Exact boundary types reject bool/subclasses.
                result["warm_distributions"] = {
                    "count": 40,
                    "checks_to_actionable_p95_upper_ns": nearest_rank_p95(checks),
                    "earliest_Down_to_feedback_p95_upper_ns": nearest_rank_p95(
                        feedback
                    ),
                    "p95_method": "nearest rank",
                    "cold_separate": True,
                    "no_observer_cost_subtraction": True,
                }
        result["qualified"] = (
            False  # Full source/worker/process closure review remains mandatory.
        )
        result["source_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        write_json(run / "native-smoke-result.json", result)
    print(
        json.dumps(
            {
                "receipt": str(run / "native-smoke-result.json"),
                "qualified": False,
                "product_samples": 0,
                "errors": result["errors"],
            },
            indent=2,
        )
    )
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
