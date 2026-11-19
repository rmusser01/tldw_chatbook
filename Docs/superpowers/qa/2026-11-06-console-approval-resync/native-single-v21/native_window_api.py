"""Native window/clock API definitions from reviewed synthetic capability. No launch."""

import ctypes as C
from ctypes import wintypes as W
import statistics
import time


class COORD(C.Structure):
    _fields_ = [("X", W.SHORT), ("Y", W.SHORT)]


class SMALL_RECT(C.Structure):
    _fields_ = [
        ("Left", W.SHORT),
        ("Top", W.SHORT),
        ("Right", W.SHORT),
        ("Bottom", W.SHORT),
    ]


class INFO(C.Structure):
    _fields_ = [
        ("size", COORD),
        ("cursor", COORD),
        ("attributes", W.WORD),
        ("window", SMALL_RECT),
        ("maximum", COORD),
    ]


class FONT(C.Structure):
    _fields_ = [
        ("cbSize", W.ULONG),
        ("index", W.DWORD),
        ("size", COORD),
        ("family", W.UINT),
        ("weight", W.UINT),
        ("face", W.WCHAR * 32),
    ]


def win32():
    user, kernel, gdi = (
        C.WinDLL("user32", use_last_error=True),
        C.WinDLL("kernel32", use_last_error=True),
        C.WinDLL("gdi32", use_last_error=True),
    )
    for name, restype, arguments in [
        ("GetForegroundWindow", W.HWND, []),
        ("GetAncestor", W.HWND, [W.HWND, W.UINT]),
        ("WindowFromPoint", W.HWND, [W.POINT]),
        ("IsWindowVisible", W.BOOL, [W.HWND]),
        ("IsIconic", W.BOOL, [W.HWND]),
        ("GetClientRect", W.BOOL, [W.HWND, C.POINTER(W.RECT)]),
        ("ClientToScreen", W.BOOL, [W.HWND, C.POINTER(W.POINT)]),
        ("GetWindowThreadProcessId", W.DWORD, [W.HWND, C.POINTER(W.DWORD)]),
        ("GetClassNameW", C.c_int, [W.HWND, W.LPWSTR, C.c_int]),
        ("GetDC", W.HDC, [W.HWND]),
        ("ReleaseDC", C.c_int, [W.HWND, W.HDC]),
        ("ShowWindow", W.BOOL, [W.HWND, C.c_int]),
    ]:
        function = getattr(user, name)
        function.restype = restype
        function.argtypes = arguments
    kernel.GetConsoleWindow.restype = W.HWND
    kernel.GetStdHandle.restype = W.HANDLE
    kernel.GetStdHandle.argtypes = [W.DWORD]
    kernel.GetConsoleScreenBufferInfo.argtypes = [W.HANDLE, C.POINTER(INFO)]
    kernel.GetCurrentConsoleFontEx.argtypes = [W.HANDLE, W.BOOL, C.POINTER(FONT)]
    kernel.QueryPerformanceCounter.argtypes = [C.POINTER(C.c_longlong)]
    kernel.QueryPerformanceFrequency.argtypes = [C.POINTER(C.c_longlong)]
    gdi.GetPixel.argtypes = [W.HDC, C.c_int, C.c_int]
    gdi.GetPixel.restype = W.DWORD
    # Observer/fixture thread only; never changes a system DPI setting.
    user.SetThreadDpiAwarenessContext.argtypes = [W.HANDLE]
    user.SetThreadDpiAwarenessContext.restype = W.HANDLE
    if not user.SetThreadDpiAwarenessContext(W.HANDLE(-4)):
        raise RuntimeError("Per-monitor-v2 observer DPI context unavailable")
    return user, kernel, gdi


def qpc(kernel):
    value = C.c_longlong()
    if not kernel.QueryPerformanceCounter(C.byref(value)):
        raise RuntimeError("QPC unavailable")
    return value.value


def clock_bounds(kernel):
    frequency = C.c_longlong()
    if not kernel.QueryPerformanceFrequency(C.byref(frequency)):
        raise RuntimeError("QPC frequency unavailable")
    samples = []
    for _ in range(64):
        start = qpc(kernel)
        python = time.perf_counter_ns()
        end = qpc(kernel)
        samples.append([start, python, end])
    spans = [(end - start) * 1e9 / frequency.value for start, _, end in samples]
    return {
        "qpc_frequency": frequency.value,
        "brackets": samples,
        "median_bracket_ns": statistics.median(spans),
        "max_bracket_ns": max(spans),
    }


def geometry(user, hwnd):
    rect, origin, pid, name = (
        W.RECT(),
        W.POINT(0, 0),
        W.DWORD(),
        C.create_unicode_buffer(128),
    )
    if not user.GetClientRect(hwnd, C.byref(rect)) or not user.ClientToScreen(
        hwnd, C.byref(origin)
    ):
        raise RuntimeError("Client-to-screen mapping unavailable")
    if not user.GetWindowThreadProcessId(hwnd, C.byref(pid)):
        raise RuntimeError("HWND owner unavailable")
    user.GetClassNameW(hwnd, name, len(name))
    user.GetDpiForWindow.argtypes = [W.HWND]
    user.GetDpiForWindow.restype = W.UINT
    dpi = user.GetDpiForWindow(hwnd)
    if not dpi:
        raise RuntimeError("Native window DPI unavailable")
    return {
        "client": [rect.right, rect.bottom],
        "origin": [origin.x, origin.y],
        "host_pid": pid.value,
        "class": name.value,
        "dpi": dpi,
    }


def console_state(user, kernel, hwnd):
    info, font = INFO(), FONT()
    font.cbSize = C.sizeof(font)
    handle = kernel.GetStdHandle(-11 & 0xFFFFFFFF)
    if not kernel.GetConsoleScreenBufferInfo(
        handle, C.byref(info)
    ) or not kernel.GetCurrentConsoleFontEx(handle, False, C.byref(font)):
        raise RuntimeError("Native console grid unavailable")
    return {
        "geometry": geometry(user, hwnd),
        "columns": info.window.Right - info.window.Left + 1,
        "rows": info.window.Bottom - info.window.Top + 1,
        "font": [font.size.X, font.size.Y],
    }


def resize_owned_console(user, kernel, hwnd, columns, rows):
    if (columns, rows) not in {(80, 24), (120, 40), (170, 48)}:
        raise ValueError("Controlled physical size required")
    handle = kernel.GetStdHandle(-11 & 0xFFFFFFFF)
    kernel.SetConsoleScreenBufferSize.argtypes = [W.HANDLE, COORD]
    kernel.SetConsoleScreenBufferSize.restype = W.BOOL
    kernel.SetConsoleWindowInfo.argtypes = [W.HANDLE, W.BOOL, C.POINTER(SMALL_RECT)]
    kernel.SetConsoleWindowInfo.restype = W.BOOL
    info = INFO()
    if not kernel.GetConsoleScreenBufferInfo(handle, C.byref(info)):
        raise RuntimeError("Own console pre-resize geometry unavailable")
    if not kernel.SetConsoleScreenBufferSize(
        handle, COORD(max(columns, info.size.X), max(rows, info.size.Y))
    ):
        raise RuntimeError("Own console grow buffer refused")
    rect = SMALL_RECT(0, 0, columns - 1, rows - 1)
    if not kernel.SetConsoleWindowInfo(handle, True, C.byref(rect)):
        raise RuntimeError("Own console window resize refused")
    if not kernel.SetConsoleScreenBufferSize(handle, COORD(columns, rows)):
        raise RuntimeError("Own console final buffer resize refused")
    state = console_state(user, kernel, hwnd)
    if (state["columns"], state["rows"]) != (columns, rows):
        raise RuntimeError("Own physical console size mismatch")
    # Pre-application buffer/window resize only. Exact pixel qualification is
    # mandatory AFTER the actual WindowsDriver enables application/alt mode.
    state["pre_app_client_map_exact"] = state["geometry"]["client"] == [
        columns * state["font"][0],
        rows * state["font"][1],
    ]
    return state
