"""Machine-local ffplay predecessor. Empty is the only legacy representation.

The 20 bytes are big-endian magic/version ALP1, uint32 PID, uint64 Windows
creation FILETIME, then uint32 CRC-32 of the preceding sixteen bytes. The
checksum detects damage. Permission comes only from the private pinned holder.
"""

import struct
import zlib


_BODY = struct.Struct(">4sIQ")
RECEIPT_SIZE = _BODY.size + 4


def encode_predecessor(pid: int, created: int) -> bytes:
    if (
        type(pid) is not int
        or not 0 < pid < 2**32
        or (type(created) is not int or not 0 < created < 2**64)
    ):
        raise ValueError("invalid_player_predecessor")
    body = _BODY.pack(b"ALP1", pid, created)
    return body + zlib.crc32(body).to_bytes(4, "big")


def decode_predecessor(data: bytes) -> tuple[int, int] | None:
    if data == b"":
        return None
    if type(data) is not bytes or len(data) != RECEIPT_SIZE:
        raise ValueError("invalid_player_predecessor")
    _, pid, created = _BODY.unpack_from(data)
    if data != encode_predecessor(pid, created):
        raise ValueError("invalid_player_predecessor")
    return pid, created
