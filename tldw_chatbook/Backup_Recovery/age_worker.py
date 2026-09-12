"""Isolated, pipe-only age v1 single-scrypt backup encryption worker.

Executed as ``python -I <fixed package file> info|encrypt|decrypt``. Transform
stdin starts with a four-byte big-endian password length and raw password bytes.
The remaining stdin and all stdout are binary container streams. This module
imports no application configuration. Format: https://c2sp.org/age (age v1).
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import sys
from typing import BinaryIO

try:
    from Cryptodome.Cipher import ChaCha20_Poly1305
    from Cryptodome.Hash import HMAC, SHA256
    from Cryptodome.Protocol.KDF import HKDF, scrypt
    from Cryptodome.Random import get_random_bytes
except (ImportError, OSError):
    # A missing/broken native dependency must not print import paths or diagnostics.
    _DEPENDENCIES_AVAILABLE = False
else:
    _DEPENDENCIES_AVAILABLE = True

MAX_HEADER = 64 * 1024
MAX_PASSWORD = 4096
MAX_CONTAINER = 2 * 1024**4
CHUNK_SIZE = 64 * 1024
WORK_FACTOR = 18
_MAGIC = b"age-encryption.org/v1\n"
_SCRYPT_LABEL = b"age-encryption.org/v1/scrypt"


class WorkerError(Exception):
    """A fixed, sanitized protocol, format, limit, or authentication failure."""


class BudgetReader:
    """Count actual container input, allowing one byte to detect overflow."""

    def __init__(self, stream: BinaryIO, limit: int) -> None:
        self.stream = stream
        self.remaining = limit

    def read(self, size: int) -> bytes:
        """Read bounded bytes; probe EOF even when the budget is exhausted."""
        data = self.stream.read(min(size, self.remaining + 1))
        if data is None:
            raise WorkerError("transform_failed")
        if len(data) > self.remaining:
            raise WorkerError("byte_limit")
        self.remaining -= len(data)
        return data


class BudgetWriter:
    """Bound output independently and complete every short blocking write."""

    def __init__(self, stream: BinaryIO, limit: int) -> None:
        self.stream = stream
        self.remaining = limit

    def write(self, data: bytes) -> None:
        """Write a complete block or fail without spinning on a stalled pipe."""
        if len(data) > self.remaining:
            raise WorkerError("byte_limit")
        pending = memoryview(data)
        while pending:
            count = self.stream.write(pending)
            if count is None or count <= 0 or count > len(pending):
                raise WorkerError("transform_failed")
            self.remaining -= count
            pending = pending[count:]


def _read_up_to(stream: BinaryIO | BudgetReader, size: int) -> bytes:
    """Fill one bounded block across short reads, stopping only at EOF."""
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if block is None:
            raise WorkerError("transform_failed")
        if not block:
            break
        result.extend(block)
    return bytes(result)


def _read_exact(stream: BinaryIO | BudgetReader, size: int, code: str) -> bytes:
    """Read fixed-length protocol data or raise the supplied fixed error."""
    result = _read_up_to(stream, size)
    if len(result) != size:
        raise WorkerError(code)
    return result


def _base64(data: bytes) -> bytes:
    """Encode canonical unpadded standard base64."""
    return base64.b64encode(data).rstrip(b"=")


def _decode_base64(value: bytes, size: int) -> bytes:
    """Reject padding, invalid alphabets, nonzero pad bits and wrong lengths."""
    if len(value) != (size * 8 + 5) // 6:
        raise WorkerError("invalid_header")
    try:
        decoded = base64.b64decode(value + b"=" * (-len(value) % 4), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise WorkerError("invalid_header") from exc
    if len(decoded) != size or _base64(decoded) != value:
        raise WorkerError("invalid_header")
    return decoded


def _read_header(stream: BudgetReader) -> tuple[bytes, bytes, int, bytes, bytes]:
    """Admit the whole canonical single-scrypt header before allocating a KDF."""
    header = bytearray()

    def line() -> bytes:
        start = len(header)
        while len(header) < MAX_HEADER:
            byte = _read_exact(stream, 1, "invalid_header")
            header.extend(byte)
            if byte == b"\n":
                return bytes(header[start:-1])
        raise WorkerError("invalid_header")

    if line() + b"\n" != _MAGIC:
        raise WorkerError("invalid_header")
    fields = line().split(b" ")
    if len(fields) != 4 or fields[:2] != [b"->", b"scrypt"]:
        raise WorkerError("invalid_header")
    salt = _decode_base64(fields[2], 16)
    # A tiny whitelist rejects signs, leading zeroes and giant decimal values.
    if fields[3] not in tuple(str(n).encode("ascii") for n in range(1, 19)):
        raise WorkerError("invalid_header")
    factor = int(fields[3])
    wrapped_key = _decode_base64(line(), 32)
    footer = line()
    if not footer.startswith(b"--- "):
        raise WorkerError("invalid_header")
    mac = _decode_base64(footer[4:], 32)
    authenticated = bytes(header[: -(len(footer) + 1)]) + b"---"
    return authenticated, salt, factor, wrapped_key, mac


def _derive_key(file_key: bytes, salt: bytes, context: bytes) -> bytes:
    """Use the standard HKDF-SHA256 primitive with the age domain separator."""
    return HKDF(file_key, 32, salt, SHA256, context=context)


def _wrap_key(password: bytes, salt: bytes, factor: int) -> bytes:
    """Run the admitted age scrypt parameters (256 MiB at write factor 18)."""
    return scrypt(password, _SCRYPT_LABEL + salt, 32, N=1 << factor, r=8, p=1)


def _seal(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
    """Encrypt and append the standard ChaCha20-Poly1305 authentication tag."""
    cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return ciphertext + tag


def _open(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    """Release one block only after its authentication tag verifies."""
    if len(ciphertext) < 16:
        raise WorkerError("transform_failed")
    try:
        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
        return cipher.decrypt_and_verify(ciphertext[:-16], ciphertext[-16:])
    except ValueError as exc:
        raise WorkerError("transform_failed") from exc


def _encrypt(stream: BudgetReader, output: BudgetWriter, password: bytes) -> None:
    """Emit a canonical header and bounded authenticated STREAM chunks."""
    file_key = get_random_bytes(16)
    salt = get_random_bytes(16)
    wrapped = _seal(_wrap_key(password, salt, WORK_FACTOR), bytes(12), file_key)
    header = (
        _MAGIC + b"-> scrypt " + _base64(salt) + b" 18\n" + _base64(wrapped) + b"\n---"
    )
    mac = HMAC.new(_derive_key(file_key, b"", b"header"), header, SHA256).digest()
    output.write(header + b" " + _base64(mac) + b"\n")
    payload_nonce = get_random_bytes(16)
    output.write(payload_nonce)
    key = _derive_key(file_key, payload_nonce, b"payload")
    block = _read_up_to(stream, CHUNK_SIZE)
    counter = 0
    while True:
        following = _read_up_to(stream, 1) if len(block) == CHUNK_SIZE else b""
        final = not following
        nonce = counter.to_bytes(11, "big") + bytes([final])
        output.write(_seal(key, nonce, block))
        if final:
            return
        block = following + _read_up_to(stream, CHUNK_SIZE - 1)
        counter += 1


def _decrypt(stream: BudgetReader, output: BudgetWriter, password: bytes) -> None:
    """Authenticate the header and require an authenticated final payload chunk."""
    header, salt, factor, wrapped, mac = _read_header(stream)
    file_key = _open(_wrap_key(password, salt, factor), bytes(12), wrapped)
    try:
        HMAC.new(_derive_key(file_key, b"", b"header"), header, SHA256).verify(mac)
    except ValueError as exc:
        raise WorkerError("transform_failed") from exc
    payload_nonce = _read_exact(stream, 16, "transform_failed")
    key = _derive_key(file_key, payload_nonce, b"payload")
    block_size = CHUNK_SIZE + 16
    block = _read_up_to(stream, block_size)
    counter = 0
    while True:
        following = _read_up_to(stream, 1) if len(block) == block_size else b""
        final = not following
        if final and counter and len(block) == 16:
            # Only an entirely empty payload may have an empty final chunk.
            raise WorkerError("transform_failed")
        nonce = counter.to_bytes(11, "big") + bytes([final])
        output.write(_open(key, nonce, block))
        if final:
            return
        block = following + _read_up_to(stream, block_size - 1)
        counter += 1


def run(mode: str, stream: BinaryIO, output: BinaryIO) -> None:
    """Run one private protocol operation; never read credentials for info."""
    if not _DEPENDENCIES_AVAILABLE:
        raise WorkerError("transform_failed")
    if mode == "info":
        info = {"protocol": 2, "implementation": "python", "format": "age-v1"}
        BudgetWriter(output, MAX_CONTAINER).write(
            json.dumps(info).encode("ascii") + b"\n"
        )
        return
    if mode not in ("encrypt", "decrypt"):
        raise WorkerError("protocol_error")
    size = int.from_bytes(_read_exact(stream, 4, "protocol_error"), "big")
    if not 1 <= size <= MAX_PASSWORD:
        raise WorkerError("protocol_error")
    password = _read_exact(stream, size, "protocol_error")
    operation = _decrypt if mode == "decrypt" else _encrypt
    operation(
        BudgetReader(stream, MAX_CONTAINER),
        BudgetWriter(output, MAX_CONTAINER),
        password,
    )


def main() -> int:
    """Expose only fixed failure codes, including unexpected dependency/I/O errors."""
    try:
        if len(sys.argv) != 2:
            raise WorkerError("protocol_error")
        # Unbuffered writes cannot be retried by interpreter shutdown after EPIPE.
        # Borrow stdout without closing its process-owned descriptor.
        with io.FileIO(sys.stdout.fileno(), "wb", closefd=False) as output:
            run(sys.argv[1], sys.stdin.buffer, output)
    except WorkerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - protocol boundary must sanitize every failure.
        print("transform_failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
