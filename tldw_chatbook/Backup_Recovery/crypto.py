"""Package-owned age v1 transport; no PATH lookup or plaintext fallback.

Output is published without replacement only after authenticated EOF and a clean
helper exit. Callers must supply an owner-private staging directory for decrypt.
"""

import hashlib
import io
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Lock, Thread
from typing import BinaryIO

from tldw_chatbook.Utils.path_validation import validate_path_simple

_MAX_CONTAINER = 2 * 1024**4
_BUFFER = 64 * 1024
_JOBS = Lock()
_WORKER_INFO = {"protocol": 2, "implementation": "python", "format": "age-v1"}
# Update this anchor in the same reviewed change whenever age_worker.py changes.
_WORKER_SHA256 = "451a3dc158c3c1e8ee599f384a23046eb9d446ca94e8c5e04b8ffe9c6e068c05"


class CryptoError(RuntimeError):
    """A fixed nonsecret error code; never wrap child or filesystem diagnostics."""


def _package_resource_root() -> Path:
    return Path(__file__).parent


def _child_environment() -> dict[str, str]:
    # Do not forward credentials or ambient application/agent settings.
    return (
        {"SYSTEMROOT": os.environ["SYSTEMROOT"]} if "SYSTEMROOT" in os.environ else {}
    )


def _pipes(
    worker_path: Path,
    mode: str,
    source: BinaryIO,
    output: BinaryIO,
    *,
    password: bytes | None,
    cancel: Event,
    limit: int,
    deadline: float | None = None,
    input_limit: int | None = None,
    output_limit: int | None = None,
    space_check: Callable[[int], None] | None = None,
    expected_input_sha256: str | None = None,
) -> None:
    """Pump all three pipes concurrently, retaining no input or diagnostics."""
    failed = Event()
    input_digest = hashlib.sha256()
    process = subprocess.Popen(
        [sys.executable, "-I", str(worker_path), mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_child_environment(),
        bufsize=0,
    )

    def pump() -> None:
        try:
            with process.stdin as pipe:
                if password is not None:
                    # Raw pipes may perform short writes; drain the complete prefix.
                    header = struct.pack(">I", len(password)) + password
                    _write_all(pipe, header)
                total = 0
                while data := source.read(_BUFFER):
                    total += len(data)
                    if total > (input_limit if input_limit is not None else limit):
                        failed.set()
                        return
                    _write_all(pipe, data)
                    input_digest.update(data)
        except (OSError, ValueError):
            failed.set()

    def drain() -> None:
        try:
            with process.stdout as pipe:
                total = 0
                while data := pipe.read(_BUFFER):
                    total += len(data)
                    if total > (output_limit if output_limit is not None else limit):
                        failed.set()
                        return
                    if space_check is not None:
                        space_check(len(data))
                    _write_all(output, data)
        except (OSError, ValueError):
            failed.set()

    def errors() -> None:
        try:
            with process.stderr as pipe:
                total = 0
                while data := pipe.read(256):
                    total += len(data)
                    if total > 256:
                        failed.set()
                        return
                    # Any stderr is a failure; never retain or expose its body.
                    failed.set()
        except (OSError, ValueError):
            failed.set()

    workers = [Thread(target=f, name="backup-age-pipe") for f in (pump, drain, errors)]
    try:
        for worker in workers:
            worker.start()
        while process.poll() is None:
            if (
                cancel.is_set()
                or failed.is_set()
                or (deadline and time.monotonic() > deadline)
            ):
                process.kill()
                break
            cancel.wait(0.02)
        process.wait()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        for worker in workers:
            if worker.ident is not None:
                worker.join()
        for pipe in (process.stdin, process.stdout, process.stderr):
            pipe.close()
    if cancel.is_set():
        raise CryptoError("cancelled")
    if (
        failed.is_set()
        or process.returncode != 0
        or (deadline and time.monotonic() > deadline)
    ):
        raise CryptoError("transform_failed")
    if (
        expected_input_sha256 is not None
        and input_digest.hexdigest() != expected_input_sha256
    ):
        raise CryptoError("input_changed")


def _write_all(output: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = output.write(view)
        if not count:
            raise OSError("write_failed")
        view = view[count:]


def _open_regular(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("not_regular")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _qualified_helper() -> Path:
    root = _package_resource_root()
    worker_path = root / "age_worker.py"
    try:
        if worker_path.is_symlink():
            raise CryptoError("helper_unavailable")
        with _open_regular(worker_path) as stream:
            metadata = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or not 0 < metadata.st_size <= 1024**2
            ):
                raise CryptoError("helper_unavailable")
            hasher = hashlib.sha256()
            total = 0
            while block := stream.read(_BUFFER):
                total += len(block)
                if total > 1024**2:
                    raise CryptoError("helper_unavailable")
                hasher.update(block)
            digest = hasher.hexdigest()
        if digest != _WORKER_SHA256:
            raise CryptoError("helper_integrity_mismatch")
        output = io.BytesIO()
        _pipes(
            worker_path,
            "info",
            io.BytesIO(),
            output,
            password=None,
            cancel=Event(),
            limit=1024,
            deadline=time.monotonic() + 5,
        )
        info = json.loads(output.getvalue())
        if type(info) is not dict or info != _WORKER_INFO:
            raise CryptoError("helper_unavailable")
        return worker_path
    except CryptoError as error:
        if str(error) == "helper_integrity_mismatch":
            raise
        raise CryptoError("helper_unavailable") from None
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        raise CryptoError("helper_unavailable") from None


def helper_capability() -> tuple[bool, str]:
    """Check worker integrity, dependencies, and protocol before password entry."""
    try:
        _qualified_helper()
    except CryptoError as error:
        return False, str(error)
    return True, "available"


def transform(
    source: Path,
    target: Path,
    *,
    password: bytes,
    decrypt: bool,
    cancel: Event,
    input_limit: int = _MAX_CONTAINER,
    output_limit: int = _MAX_CONTAINER,
    space_check: Callable[[int], None] | None = None,
    expected_input_sha256: str | None = None,
) -> None:
    """Stream an age transform to a new private file, or raise a fixed code.

    The process-wide job lock bounds KDF concurrency to one per application.
    The helper independently enforces both 2 TiB stream budgets.
    """
    if any(
        type(value) is not int or not 0 < value <= _MAX_CONTAINER
        for value in (input_limit, output_limit)
    ):
        raise CryptoError("invalid_budget")
    if type(password) is not bytes or not 1 <= len(password) <= 4096:
        raise CryptoError("invalid_password")
    while not _JOBS.acquire(timeout=0.05):
        if cancel.is_set():
            raise CryptoError("cancelled")
    temporary = None
    try:
        if cancel.is_set():
            raise CryptoError("cancelled")
        source = validate_path_simple(source, probe_existing=False)
        target = validate_path_simple(target, probe_existing=False)
        worker_path = _qualified_helper()
        if target.exists() or target.is_symlink():
            raise CryptoError("target_exists")
        # Never read device/FIFO/link input, which could outlive child cleanup.
        if source.is_symlink() or not source.is_file():
            raise CryptoError("invalid_source")
        with _open_regular(source) as input_stream:
            descriptor, name = tempfile.mkstemp(
                prefix=".backup-age-", dir=target.parent
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as output:
                _pipes(
                    worker_path,
                    "decrypt" if decrypt else "encrypt",
                    input_stream,
                    output,
                    password=password,
                    cancel=cancel,
                    limit=_MAX_CONTAINER,
                    input_limit=input_limit,
                    output_limit=output_limit,
                    space_check=space_check,
                    expected_input_sha256=expected_input_sha256,
                )
                output.flush()
                os.fsync(output.fileno())
            if cancel.is_set():
                raise CryptoError("cancelled")
            # Atomic no-clobber publication: an existing/racing target survives.
            os.link(temporary, target)
    except (OSError, ValueError):
        raise CryptoError("transform_failed") from None
    finally:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError:
            raise CryptoError("cleanup_failed") from None
        finally:
            _JOBS.release()
