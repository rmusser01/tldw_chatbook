"""QA-only own-console bootstrap activation; never a measured approval action."""

import ctypes as C
from ctypes import wintypes as W
import time


def activate_owned_caption(user, hwnd, own, evidence_path):
    """One normal verified HTCAPTION click on the authenticated own console."""
    import json

    attempt = time.perf_counter_ns()

    def record(phase, **facts):
        row = {
            "phase": phase,
            "attempt_ns": attempt,
            "observed_ns": time.perf_counter_ns(),
            "hwnd": int(hwnd),
            "approval_action": False,
            **facts,
        }
        with evidence_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")

    own()
    for name, restype, args in [
        (
            "SetWindowPos",
            W.BOOL,
            [W.HWND, W.HWND, C.c_int, C.c_int, C.c_int, C.c_int, W.UINT],
        ),
        ("GetWindowRect", W.BOOL, [W.HWND, C.POINTER(W.RECT)]),
        ("GetTopWindow", W.HWND, [W.HWND]),
        ("GetWindow", W.HWND, [W.HWND, W.UINT]),
        ("GetSystemMetrics", C.c_int, [C.c_int]),
        ("GetCursorPos", W.BOOL, [C.POINTER(W.POINT)]),
        ("GetAsyncKeyState", W.SHORT, [C.c_int]),
        (
            "SendMessageTimeoutW",
            W.LPARAM,
            [W.HWND, W.UINT, W.WPARAM, W.LPARAM, W.UINT, W.UINT, C.POINTER(C.c_size_t)],
        ),
    ]:
        function = getattr(user, name)
        function.restype = restype
        function.argtypes = args
    # Ordinary own-window Z-order only; never topmost or an input-policy bypass.
    if not user.SetWindowPos(hwnd, W.HWND(0), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010):
        raise RuntimeError("Own console normal raise refused")
    bounds = W.RECT()
    if not user.GetWindowRect(hwnd, C.byref(bounds)):
        raise RuntimeError("Own caption bounds unavailable")
    vx, vy, vw, vh = [user.GetSystemMetrics(index) for index in (76, 77, 78, 79)]
    if vw <= 1 or vh <= 1:
        raise RuntimeError("Virtual screen unavailable")

    def caption(x, y):
        own()
        if (
            not (vx <= x < vx + vw and vy <= y < vy + vh)
            or user.GetAncestor(user.WindowFromPoint(W.POINT(x, y)), 2) != hwnd
        ):
            return False
        other = user.GetTopWindow(None)
        for _ in range(1024):
            if not other:
                return False
            if other == hwnd:
                break
            if user.IsWindowVisible(other) and not user.IsIconic(other):
                rect = W.RECT()
                if not user.GetWindowRect(other, C.byref(rect)):
                    return False
                if rect.left <= x < rect.right and rect.top <= y < rect.bottom:
                    return False
            other = user.GetWindow(other, 2)
        else:
            return False
        hit = C.c_size_t()
        packed = ((y & 0xFFFF) << 16) | (x & 0xFFFF)
        return (
            bool(
                user.SendMessageTimeoutW(
                    hwnd, 0x0084, 0, packed, 0x0002 | 0x0020, 100, C.byref(hit)
                )
            )
            and hit.value == 2
        )

    point = None
    tested = 0
    deadline = time.monotonic() + 2
    for dy in (12, 18, 24, 30, 36, 42, 48, 54):
        for fraction in (0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9):
            if time.monotonic() > deadline:
                break
            x = bounds.left + int((bounds.right - bounds.left) * fraction)
            y = bounds.top + dy
            tested += 1
            if caption(x, y):
                point = (x, y)
                break
        if point is not None or time.monotonic() > deadline:
            break
    if point is None:
        raise RuntimeError("No verified own HTCAPTION point")
    if any(
        user.GetAsyncKeyState(key) & 0x8000
        for key in (0x10, 0x11, 0x12, 0x5B, 0x5C, 1, 2, 4)
    ):
        raise RuntimeError("Own caption input state busy")

    class MOUSE(C.Structure):
        _fields_ = [
            ("dx", W.LONG),
            ("dy", W.LONG),
            ("data", W.DWORD),
            ("flags", W.DWORD),
            ("time", W.DWORD),
            ("extra", C.c_size_t),
        ]

    class PAYLOAD(C.Union):
        _fields_ = [("mouse", MOUSE)]

    class INPUT(C.Structure):
        _fields_ = [("type", W.DWORD), ("payload", PAYLOAD)]

    if C.sizeof(INPUT) != (40 if C.sizeof(C.c_void_p) == 8 else 28):
        raise RuntimeError("Native INPUT layout mismatch")
    user.SendInput.argtypes = [W.UINT, C.POINTER(INPUT), C.c_int]
    user.SendInput.restype = W.UINT

    def event(flags, x=0, y=0):
        value = INPUT()
        value.type = 0
        value.payload.mouse = MOUSE(x, y, 0, flags, 0, 0)
        return value

    x, y = point
    if not caption(x, y):
        raise RuntimeError("Own caption changed before pointer move")
    absolute = [round((x - vx) * 65535 / (vw - 1)), round((y - vy) * 65535 / (vh - 1))]
    record(
        "move_prepared",
        intended_screen_point=[x, y],
        encoded_absolute=absolute,
        virtual_screen=[vx, vy, vw, vh],
    )
    move = event(0x0001 | 0x8000 | 0x4000, *absolute)
    move_before = time.perf_counter_ns()
    inserted = user.SendInput(1, C.byref(move), C.sizeof(INPUT))
    move_after = time.perf_counter_ns()
    record(
        "move_queued",
        intended_screen_point=[x, y],
        events_inserted=inserted,
        before_ns=move_before,
        after_ns=move_after,
    )
    if inserted != 1:
        raise RuntimeError("Own caption pointer move refused")
    actual = W.POINT()
    settle_deadline = time.monotonic() + 0.5
    polls = 0
    # SendInput queues OS input; it does not prove cursor movement completed.
    # Poll only before bootstrap activation, using the UNCHANGED exact own
    # HTCAPTION/windowpoint/higher-Z/owner predicate before any click pair.
    while True:
        observed = bool(user.GetCursorPos(C.byref(actual)))
        polls += 1
        verified = observed and caption(actual.x, actual.y)
        record(
            "cursor_observed",
            intended_screen_point=[x, y],
            actual_screen_point=[actual.x, actual.y] if observed else None,
            get_cursor_succeeded=observed,
            own_HTCAPTION_verified=verified,
            poll=polls,
        )
        if verified:
            break
        if time.monotonic() >= settle_deadline:
            raise RuntimeError(
                "Actual cursor not verified own caption within bounded500ms"
            )
        time.sleep(0.01)
    # A queued MOVE/poll is not the immediate pre-click owner proof. Re-read
    # the ACTUAL cursor, rerun the SAME own HTCAPTION/point/occlusion predicate,
    # and refuse newly held input immediately before submitting the pair.
    if not user.GetCursorPos(C.byref(actual)) or not caption(actual.x, actual.y):
        raise RuntimeError("Immediate own caption cursor recheck refused")
    if any(
        user.GetAsyncKeyState(key) & 0x8000
        for key in (0x10, 0x11, 0x12, 0x5B, 0x5C, 1, 2, 4)
    ):
        raise RuntimeError("Immediate own caption input state busy")
    pair = (INPUT * 2)(event(2), event(4))
    start = time.perf_counter_ns()
    sent = user.SendInput(2, pair, C.sizeof(INPUT))
    end = time.perf_counter_ns()
    release_inserted = None
    if sent == 1:
        release = event(4)
        release_inserted = user.SendInput(1, C.byref(release), C.sizeof(INPUT))
    record(
        "caption_click_queued",
        intended_screen_point=[x, y],
        actual_screen_point=[actual.x, actual.y],
        immediate_own_caption_and_held_input_rechecked=True,
        events_inserted=sent,
        partial_release_events_inserted=release_inserted,
        before_ns=start,
        after_ns=end,
    )
    if sent == 1 and release_inserted != 1:
        raise RuntimeError("Own caption partial input release unproven")
    if sent != 2:
        raise RuntimeError("Own caption input partial/refused")
    return {
        "screen_point": [actual.x, actual.y],
        "HTCAPTION": True,
        "candidate_count": tested,
        "events_inserted": sent,
        "before_ns": start,
        "after_ns": end,
        "approval_action": False,
    }


def retain_initial_owned_view(user, kernel, gdi, process, descriptor, run, nonce):
    """Unqualified pre-product diagnostic pixels only, after strict own checks."""
    import hashlib
    import json
    from native_window_api import geometry
    from owned_roi_capture import capture_owned_roi

    hwnd = descriptor["hwnd"]
    identity = descriptor["native"]["geometry"]
    if descriptor["nonce"] != nonce or not (
        descriptor["pid"] == process.pid
        or descriptor["launcher_parent_pid"] == process.pid
    ):
        raise RuntimeError("Pre-App own process/nonce mismatch")
    rows = []
    activation = []

    def save():
        (run / "native-initial-window-states.json").write_text(
            json.dumps(
                {
                    "states": rows,
                    "activation": activation,
                    "qualified": False,
                    "product_samples": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def own():
        foreground = int(user.GetForegroundWindow() or 0)
        foreground_pid = W.DWORD()
        if foreground:
            user.GetWindowThreadProcessId(foreground, C.byref(foreground_pid))
        actual = geometry(user, hwnd)
        row = {
            "observed_ns": time.perf_counter_ns(),
            "actual_geometry": actual,
            "expected_identity": identity,
            "launcher_alive": process.poll() is None,
            "visible": bool(user.IsWindowVisible(hwnd)),
            "iconic": bool(user.IsIconic(hwnd)),
            "foreground_hwnd": foreground,
            "foreground_pid": foreground_pid.value,
            "foreground_matches": foreground == hwnd,
        }
        if len(rows) < 256:
            rows.append(row)
        save()
        if (
            not row["launcher_alive"]
            or not row["visible"]
            or row["iconic"]
            or any(actual[key] != identity[key] for key in ("host_pid", "class", "dpi"))
        ):
            raise RuntimeError("Pre-App own console identity/visibility refused")
        return row

    row = own()
    if not row["foreground_matches"]:
        user.SetForegroundWindow.argtypes = [W.HWND]
        user.SetForegroundWindow.restype = W.BOOL
        activation.append(
            {
                "set_foreground_returned": bool(user.SetForegroundWindow(hwnd)),
                "timestamp_ns": time.perf_counter_ns(),
            }
        )
        time.sleep(0.1)
        row = own()
        if not row["foreground_matches"]:
            activation.append(
                activate_owned_caption(
                    user, hwnd, own, run / "native-caption-activation.jsonl"
                )
            )
    deadline = time.monotonic() + 10
    last = None
    stable = None
    while time.monotonic() < deadline:
        row = own()
        if row["foreground_matches"]:
            if row["actual_geometry"] == last:
                if time.monotonic() - stable >= 0.5:
                    break
            else:
                last = row["actual_geometry"]
                stable = time.monotonic()
        else:
            last = None
            stable = None
        time.sleep(0.05)
    else:
        raise RuntimeError("Pre-App own console did not settle within10s")
    expected = row["actual_geometry"]

    def validate():
        live = own()
        if not live["foreground_matches"] or live["actual_geometry"] != expected:
            raise RuntimeError("Pre-App own capture state changed")

    image = capture_owned_roi(
        user, kernel, gdi, hwnd, expected, (0, 0, *expected["client"]), validate
    )
    from PIL import Image

    path = run / "owned-pre-app-diagnostic.png"
    Image.frombytes(
        "RGB", (image["width"], image["height"]), image["pixels"], "raw", "BGRX"
    ).save(path)
    save()
    return {
        **{key: value for key, value in image.items() if key != "pixels"},
        "file": path.name,
        "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "pre-product bootstrap only, no card/driver/paint qualification",
        "qualified": False,
        "product_samples": 0,
    }
