"""One bounded, verified owned-window desktop ROI. No input or image export.

The returned pixels are an observation during a bracketed capture interval,
not proof of atomic DWM presentation or physical scanout. Callers must join
actual card/action pixels to independently bound generation/state, not use a
marker or driver flush alone. Any foreground/identity/geometry/occlusion loss
rejects the observation. A concurrent desktop change can race the pre/post
checks; callers must discard rather than persist unidentified pixels.
"""

import ctypes as C
from ctypes import wintypes as W
import hashlib

SRCCOPY = 0x00CC0020
MAX_PIXELS = 2_000_000


class HEADER(C.Structure):
    _fields_ = [
        ("size", W.DWORD),
        ("width", W.LONG),
        ("height", W.LONG),
        ("planes", W.WORD),
        ("bits", W.WORD),
        ("compression", W.DWORD),
        ("image_size", W.DWORD),
        ("x_ppm", W.LONG),
        ("y_ppm", W.LONG),
        ("used", W.DWORD),
        ("important", W.DWORD),
    ]


class BITMAPINFO(C.Structure):
    _fields_ = [("header", HEADER), ("colors", W.DWORD * 1)]


def capture_owned_roi(user, kernel, gdi, hwnd, expected, rect, validate):
    """Capture client-local (x,y,width,height) after exact owned-ROI checks.

    validate is the existing nonce/process/native-window/geometry lease check.
    No raw pixels are written here. All GDI objects and the DC are released.
    """
    x, y, width, height = rect
    if min(x, y) < 0 or min(width, height) <= 0 or width * height > MAX_PIXELS:
        raise RuntimeError("Invalid bounded diagnostic ROI")
    if x + width > expected["client"][0] or y + height > expected["client"][1]:
        raise RuntimeError("Diagnostic ROI exceeds owned client")
    for name, restype, args in [
        ("GetTopWindow", W.HWND, [W.HWND]),
        ("GetWindow", W.HWND, [W.HWND, W.UINT]),
        ("GetWindowRect", W.BOOL, [W.HWND, C.POINTER(W.RECT)]),
    ]:
        function = getattr(user, name)
        function.restype = restype
        function.argtypes = args
    for name, restype, args in [
        ("CreateCompatibleDC", W.HDC, [W.HDC]),
        ("CreateCompatibleBitmap", W.HBITMAP, [W.HDC, C.c_int, C.c_int]),
        ("SelectObject", W.HANDLE, [W.HDC, W.HANDLE]),
        ("DeleteObject", W.BOOL, [W.HANDLE]),
        ("DeleteDC", W.BOOL, [W.HDC]),
        (
            "BitBlt",
            W.BOOL,
            [
                W.HDC,
                C.c_int,
                C.c_int,
                C.c_int,
                C.c_int,
                W.HDC,
                C.c_int,
                C.c_int,
                W.DWORD,
            ],
        ),
        (
            "GetDIBits",
            C.c_int,
            [
                W.HDC,
                W.HBITMAP,
                W.UINT,
                W.UINT,
                C.c_void_p,
                C.POINTER(BITMAPINFO),
                W.UINT,
            ],
        ),
    ]:
        function = getattr(gdi, name)
        function.restype = restype
        function.argtypes = args

    def tick():
        value = C.c_longlong()
        if not kernel.QueryPerformanceCounter(C.byref(value)):
            raise RuntimeError("QPC unavailable")
        return value.value

    screen_x, screen_y = expected["origin"][0] + x, expected["origin"][1] + y

    def preflight():
        validate()
        # Reject every visible higher-Z window whose rectangle touches the ROI,
        # including a tiny overlay between diagnostic marker sample points.
        window = user.GetTopWindow(None)
        for _ in range(1024):
            if not window:
                raise RuntimeError("Owned HWND missing from bounded Z-order walk")
            if window == hwnd:
                return
            if user.IsWindowVisible(window) and not user.IsIconic(window):
                bounds = W.RECT()
                if not user.GetWindowRect(window, C.byref(bounds)):
                    raise RuntimeError("Occlusion geometry unavailable")
                if (
                    bounds.left < screen_x + width
                    and bounds.right > screen_x
                    and bounds.top < screen_y + height
                    and bounds.bottom > screen_y
                ):
                    raise RuntimeError(
                        "Owned diagnostic ROI is occluded; no pixel copy"
                    )
            window = user.GetWindow(window, 2)
        raise RuntimeError("Window Z-order validation exceeded bound")

    started = tick()
    preflight()
    validated = tick()
    source = memory = bitmap = previous = None
    pixels = None
    try:
        source = user.GetDC(None)
        if not source:
            raise RuntimeError("Screen DC unavailable")
        memory = gdi.CreateCompatibleDC(source)
        if not memory:
            raise RuntimeError("Compatible DC unavailable")
        bitmap = gdi.CreateCompatibleBitmap(source, width, height)
        if not bitmap:
            raise RuntimeError("Compatible bitmap unavailable")
        previous = gdi.SelectObject(memory, bitmap)
        if not previous or previous == W.HANDLE(-1).value:
            raise RuntimeError("Bitmap selection failed")
        before_blit = tick()
        if not gdi.BitBlt(
            memory, 0, 0, width, height, source, screen_x, screen_y, SRCCOPY
        ):
            raise RuntimeError("BitBlt failed: " + str(C.get_last_error()))
        after_blit = tick()
        # GetDIBits requires the bitmap NOT to be selected into a DC.
        if not gdi.SelectObject(memory, previous):
            raise RuntimeError("Bitmap deselection failed")
        previous = None
        info = BITMAPINFO()
        info.header.size = C.sizeof(HEADER)
        info.header.width = width
        info.header.height = -height
        info.header.planes = 1
        info.header.bits = 32
        info.header.compression = 0
        info.header.image_size = width * height * 4
        buffer = C.create_string_buffer(info.header.image_size)
        if gdi.GetDIBits(source, bitmap, 0, height, buffer, C.byref(info), 0) != height:
            raise RuntimeError("GetDIBits failed or incomplete")
        pixels = buffer.raw
        read_back = tick()
    finally:
        if previous and memory:
            gdi.SelectObject(memory, previous)
        if bitmap:
            gdi.DeleteObject(bitmap)
        if memory:
            gdi.DeleteDC(memory)
        if source:
            user.ReleaseDC(None, source)
    preflight()
    finished = tick()
    return {
        "pixels": pixels,
        "width": width,
        "height": height,
        "qpc_start": started,
        "qpc_validated": validated,
        "qpc_before_blit": before_blit,
        "qpc_after_blit": after_blit,
        "qpc_read_back": read_back,
        "qpc_end": finished,
        "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
        "owned_client_roi": list(rect),
        "atomic_presentation_proven": False,
        "real_card_join_proven": False,
    }
