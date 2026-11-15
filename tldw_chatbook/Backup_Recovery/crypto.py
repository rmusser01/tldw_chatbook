"""Package-owned age v1 transport; no PATH lookup or plaintext fallback.

Output is published without replacement only after authenticated EOF and a clean
helper exit. Callers must supply an owner-private staging directory for decrypt.
"""

import hashlib
import io
import json
import stat
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Event, Lock, Thread
from typing import BinaryIO
from uuid import uuid4

from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Utils.platform_files import os

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


class _CryptoJob:
    """The actual child and pipe threads remain owned through uncertain cleanup."""

    def __init__(self, custody):
        self.scope = custody.scope
        self.process = None
        self.workers = []
        self.closed_pipes = []
        self.close_failed = False
        self.scope.resources.append(self)

    def close_pipe(self, pipe):
        if pipe in self.closed_pipes:
            return
        # Record an attempted close before invoking it; never retry ambiguity.
        self.closed_pipes.append(pipe)
        try:
            pipe.close()
            if not pipe.closed:
                raise OSError("crypto_pipe_not_closed")
        except BaseException:  # noqa: BLE001 - interrupted native retirement must quarantine.
            self.close_failed = True
            raise CryptoError("crypto_resources_not_retired") from None

    def retire(self):
        from .bootstrap import RecoveryRequired

        if self.close_failed:
            raise RecoveryRequired("capture_resources_not_retired")
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait()
                for worker in self.workers:
                    if worker.ident is not None:
                        worker.join()
                    if worker.is_alive():
                        raise OSError("crypto_thread_not_retired")
                for pipe in (
                    self.process.stdin,
                    self.process.stdout,
                    self.process.stderr,
                ):
                    self.close_pipe(pipe)
                if self.process.returncode is None:
                    raise OSError("crypto_child_not_retired")
        except BaseException:  # noqa: BLE001 - interrupted native retirement must quarantine.
            self.close_failed = True
            raise RecoveryRequired("capture_resources_not_retired") from None
        self.scope.resources.remove(self)


@contextmanager
def _pipe_scope(pipe, job):
    try:
        yield pipe
    finally:
        if job is None:
            pipe.close()
        else:
            job.close_pipe(pipe)


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
    _custody=None,
) -> None:
    """Pump all three pipes concurrently, retaining no input or diagnostics."""
    job = None if _custody is None else _CryptoJob(_custody)
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

    if job is not None:
        job.process = process

    def pump() -> None:
        try:
            with _pipe_scope(process.stdin, job) as pipe:
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
        except (OSError, ValueError, CryptoError):
            failed.set()

    def drain() -> None:
        try:
            with _pipe_scope(process.stdout, job) as pipe:
                total = 0
                while data := pipe.read(_BUFFER):
                    total += len(data)
                    if total > (output_limit if output_limit is not None else limit):
                        failed.set()
                        return
                    if space_check is not None:
                        space_check(len(data))
                    _write_all(output, data)
        except (OSError, ValueError, CryptoError):
            failed.set()

    def errors() -> None:
        try:
            with _pipe_scope(process.stderr, job) as pipe:
                total = 0
                while data := pipe.read(256):
                    total += len(data)
                    if total > 256:
                        failed.set()
                        return
                    # Any stderr is a failure; never retain or expose its body.
                    failed.set()
        except (OSError, ValueError, CryptoError):
            failed.set()

    workers = [Thread(target=f, name="backup-age-pipe") for f in (pump, drain, errors)]
    if job is not None:
        job.workers = workers
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
        if job is not None:
            job.retire()
        else:
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


def _open_regular(path: Path, *, _custody=None) -> BinaryIO:
    if _custody is not None:
        from .archive_reader import _regular

        return _regular(path, _custody=_custody)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("not_regular")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _qualified_helper(*, _custody=None) -> Path:
    custody = {} if _custody is None else {"_custody": _custody}
    root = _package_resource_root()
    worker_path = root / "age_worker.py"
    try:
        if worker_path.is_symlink():
            raise CryptoError("helper_unavailable")
        with _open_regular(worker_path, **custody) as stream:
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
            **custody,
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


class _CryptoOutput:
    """Retain the exact output parent until its name and close are qualified."""

    def __init__(self, target, custody):
        self.target, self.custody = target, custody
        self.stack, self.parent, self.identity = ExitStack(), None, None
        self.close_failed = False
        custody.scope.resources.append(self)

    def open(self):
        from .native_files import pinned_directory

        self.parent = self.stack.enter_context(
            pinned_directory(self.target.parent, _custody=self.custody)
        )
        info = os.fstat(self.parent)
        self.identity = (info.st_dev, info.st_ino)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise CryptoError("output_parent_not_private")

    def check(self):
        from .native_files import pinned_directory

        if self.close_failed or self.custody.close_failed:
            raise CryptoError("output_parent_not_retired")
        with pinned_directory(self.target.parent, _custody=self.custody) as named:
            for fd in (self.parent, named):
                info = os.fstat(fd)
                if self.identity != (info.st_dev, info.st_ino):
                    raise CryptoError("output_parent_changed")

    def retire(self):
        if self.close_failed or self.custody.close_failed:
            raise CryptoError("output_parent_not_retired")
        if self.identity is not None:
            self.check()
        # A changed name leaves the actual pin live; an attempted close is final.
        self.close_failed = True
        self.stack.close()
        self.custody.scope.resources.remove(self)


def _transform_owned(source, target, worker_path, *, _custody, **options):
    from .native_files import _create_private_file_at

    if source.parent != target.parent:
        raise CryptoError("output_parent_changed")
    owner = _CryptoOutput(target, _custody)
    owner.open()
    temporary = ".backup-age-" + uuid4().hex
    with _open_regular(source, _custody=_custody) as input_stream:
        owner.check()
        with _create_private_file_at(owner.parent, temporary, _custody=_custody) as fd:
            with os.fdopen(fd, "wb", closefd=False) as output:
                _pipes(
                    worker_path,
                    source=input_stream,
                    output=output,
                    _custody=_custody,
                    **options,
                )
                output.flush()
                os.fsync(fd)
            if options["cancel"].is_set():
                raise CryptoError("cancelled")
            owner.check()
            before = os.fstat(fd)
            named = os.stat(temporary, dir_fd=owner.parent, follow_symlinks=False)
            if (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
                raise CryptoError("output_changed")
            os.link(
                temporary,
                target.name,
                src_dir_fd=owner.parent,
                dst_dir_fd=owner.parent,
                follow_symlinks=False,
            )
            named = os.stat(target.name, dir_fd=owner.parent, follow_symlinks=False)
            if (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
                raise CryptoError("output_changed")
            owner.check()


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
    _custody=None,
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
    custody = {} if _custody is None else {"_custody": _custody}
    try:
        if cancel.is_set():
            raise CryptoError("cancelled")
        source = validate_path_simple(source, probe_existing=False)
        target = validate_path_simple(target, probe_existing=False)
        worker_path = _qualified_helper(**custody)
        if target.exists() or target.is_symlink():
            raise CryptoError("target_exists")
        # Never read device/FIFO/link input, which could outlive child cleanup.
        if source.is_symlink() or not source.is_file():
            raise CryptoError("invalid_source")
        if _custody is not None:
            return _transform_owned(
                source,
                target,
                worker_path,
                _custody=_custody,
                mode="decrypt" if decrypt else "encrypt",
                password=password,
                cancel=cancel,
                limit=_MAX_CONTAINER,
                input_limit=input_limit,
                output_limit=output_limit,
                space_check=space_check,
                expected_input_sha256=expected_input_sha256,
            )
        with _open_regular(source) as input_stream:
            descriptor, name = tempfile.mkstemp(
                prefix=".backup-age-", dir=target.parent
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb", closefd=True) as output:
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
                    **custody,
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
            if temporary is not None and _custody is None:
                temporary.unlink(missing_ok=True)
        except OSError:
            raise CryptoError("cleanup_failed") from None
        finally:
            _JOBS.release()
