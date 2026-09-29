"""Binary frames for the SSH session worker (stdlib-only; bundled).

``u32 length | u32 request_id | u8 kind | body``. Shared by the host
``serve`` loop and the laptop ``RemoteSessionWorker``; both reject an
oversize frame from its header, before buffering its body.
"""

from __future__ import annotations

import json
import struct

HELLO, REQUEST, CANCEL = 1, 2, 3
LINE, STATUS, BUSY = 16, 17, 18
HEADER = struct.Struct(">IIB")
_U32_MAX = 2**32 - 1

#: STATUS exit code for a request the host could not start (fork or pipe
#: refused, e.g. EAGAIN at the process limit): sysexits' EX_OSERR. The
#: request never ran; the session and every other request carry on.
HOST_SPAWN_FAILED = 71


class FrameError(ValueError):
    """A frame violated the size cap or the header format."""


def encode_frame(kind: int, request_id: int, body: bytes) -> bytes:
    """Serialize one frame.

    Args:
        kind: One of the kind constants.
        request_id: Per-session request id (0 for session-level frames).
        body: Frame payload.

    Returns:
        Header plus body.

    Raises:
        ValueError: If the id or body length does not fit a u32.
    """
    if not 0 <= request_id <= _U32_MAX or len(body) > _U32_MAX:
        raise ValueError("frame field out of range")
    return HEADER.pack(len(body), request_id, kind) + body


class FrameReader:
    """Incremental frame parser with a per-frame body cap."""

    def __init__(self, max_body: int) -> None:
        self._max = max_body
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        """Consume bytes and return every complete frame.

        Raises:
            FrameError: When a header announces a body over the cap.
        """
        self._buf += data
        frames: list[tuple[int, int, bytes]] = []
        while len(self._buf) >= HEADER.size:
            length, request_id, kind = HEADER.unpack_from(self._buf)
            if length > self._max:
                raise FrameError(f"frame body {length} exceeds cap {self._max}")
            end = HEADER.size + length
            if len(self._buf) < end:
                break
            frames.append((kind, request_id, bytes(self._buf[HEADER.size:end])))
            del self._buf[:end]
        return frames


def encode_status(exit_code: int | None, signal_no: int | None) -> bytes:
    """Encode a child's termination for a STATUS frame."""
    return json.dumps({"exit": exit_code, "signal": signal_no}).encode()


def decode_status(body: bytes) -> tuple[int | None, int | None]:
    """Decode a STATUS body into ``(exit_code, signal_no)``."""
    payload = json.loads(body)
    return payload["exit"], payload["signal"]
