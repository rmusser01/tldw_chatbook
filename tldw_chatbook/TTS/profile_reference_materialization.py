"""Private, operation-scoped filesystem owners for clone references."""

from __future__ import annotations

import asyncio
import os
import re
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from secrets import token_hex
from typing import Final, Literal, TypeVar, cast

try:  # pragma: no cover - exercised by the explicit platform seam.
    import fcntl
except ImportError:  # pragma: no cover - Windows is intentionally unsupported here.
    fcntl = None  # type: ignore[assignment]

from tldw_chatbook.TTS.profile_reference_types import (
    CanonicalTTSCloneReference,
    TTSCloneReference,
)
from tldw_chatbook.Utils.private_paths import secure_private_directory

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_POSIX_SUPPORTED = (
    os.name == "posix"
    and fcntl is not None
    and bool(_NOFOLLOW)
    and bool(_DIRECTORY)
    and hasattr(os, "geteuid")
)
_DIRECTORY_FLAGS = os.O_RDONLY | _DIRECTORY | _CLOEXEC | _NOFOLLOW
_FILE_FLAGS = os.O_RDWR | _CLOEXEC | _NOFOLLOW
_OWNER_DIRECTORY_MODE = 0o700
_OWNER_FILE_MODE = 0o600
_OWNER_PATTERN: Final = re.compile(r"clone-v1-[0-9a-f]{32}\Z")
_ASSET_PATTERN: Final = re.compile(r"asset-[0-9a-f]{32}\.wav\Z")
_HANDLE_TOKEN = object()
_T = TypeVar("_T")
_ORIGINAL_OPEN = os.open


class TTSCloneMaterializationError(RuntimeError):
    """A bounded clone-reference filesystem failure."""

    __slots__ = ("code",)

    def __init__(
        self,
        code: Literal["closed", "unavailable", "cleanup_failed", "unsupported"],
    ) -> None:
        messages = {
            "closed": "Clone reference materialization is closed.",
            "unavailable": "Clone reference materialization is unavailable.",
            "cleanup_failed": "Clone reference cleanup could not be completed safely.",
            "unsupported": "Clone reference materialization is unsupported on this platform.",
        }
        super().__init__(messages[code])
        self.code = code


@dataclass(eq=False)
class _CloneOpenOutcome:
    descriptor: int | None = None
    rejected: bool = False


def _open_clone_descriptor(*args, _outcome, **kwargs):
    primitive = os.open
    try:
        fd = primitive(*args, **kwargs)
    except OSError:
        if primitive is _ORIGINAL_OPEN:
            _outcome.rejected = True
        raise
    _outcome.descriptor = fd
    return fd


class _CloneNativeOperation:
    """Concrete native outcomes; retained leases confer no IO permission."""

    def __init__(self, materializer):
        self.materializer = materializer
        self.root = materializer._root
        self.paths = {self.root}
        self.parents = {}
        self.pid = os.getpid()
        self.thread = None
        self.leases = []
        self.descriptors = {}
        self.pending = set()
        self.failed_closes = set()
        self.uncertain = False
        self.residue = False
        self.record = None
        self.errors = []

    def admit(self):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        self.check()
        self.materializer._check_source()
        if self.root != self.materializer._root:
            raise TTSCloneMaterializationError("unavailable")
        lease = storage.acquire_storage(self.root)
        lease.native_owner = self
        self.leases.append(lease)

    def check(self):
        if (
            self.pid != os.getpid()
            or self.thread is not threading.current_thread()
            or self not in self.materializer._native_operations
        ):
            raise TTSCloneMaterializationError("unavailable")

    def check_root(self, root):
        self.check()
        if root != self.root:
            raise TTSCloneMaterializationError("unavailable")

    def select(self, parent, name):
        self.check()
        directory = self.parents.get(parent)
        if directory is None or not (
            directory == self.root
            and _OWNER_PATTERN.fullmatch(name)
            or directory.parent == self.root
            and _OWNER_PATTERN.fullmatch(directory.name)
            and (name == "owner.lock" or _ASSET_PATTERN.fullmatch(name))
        ):
            raise TTSCloneMaterializationError("unavailable")
        self.paths.add(directory / name)

    def check_record(self, record, *, cleanup=False):
        self.check()
        if cleanup:
            valid = record is self.record and record.native is self
        else:
            valid = (
                record.native in self.materializer._native_operations
                and any(
                    handle._record is record
                    for handle in self.materializer._active.values()
                )
                and record.native.record is record
            )
        if not valid or record.asset_path.parent.parent != self.root:
            raise TTSCloneMaterializationError("unavailable")
        if not cleanup:
            self.parents[record.owner_fd] = record.asset_path.parent
            self.paths.add(record.asset_path)

    def open(self, *args, **kwargs):
        self.check()
        if "dir_fd" in kwargs:
            parent = self.parents.get(kwargs["dir_fd"])
            if parent is None:
                raise TTSCloneMaterializationError("unavailable")
            selected = parent / args[0]
        else:
            selected = Path(args[0])
        if selected not in self.paths:
            raise TTSCloneMaterializationError("unavailable")
        self.admit()
        outcome = _CloneOpenOutcome()
        self.pending.add(outcome)
        try:
            return _open_clone_descriptor(*args, _outcome=outcome, **kwargs)
        finally:
            if outcome.descriptor is not None:
                self.descriptors[outcome.descriptor] = None
                if args[1] & _DIRECTORY:
                    self.parents[outcome.descriptor] = selected
                self.pending.discard(outcome)
            elif outcome.rejected:
                self.pending.discard(outcome)

    def open_directory(self, *args, **kwargs):
        from tldw_chatbook.Utils import private_paths

        self.admit()
        outcome = private_paths._NativeOpenOutcome()
        attempt = object()
        self.pending.add(attempt)
        try:
            return private_paths._native_open(*args, _outcome=outcome, **kwargs)
        finally:
            if outcome.descriptor is not None:
                self.descriptors[outcome.descriptor] = private_paths._native_close
                self.pending.discard(attempt)
            elif outcome.rejected:
                self.pending.discard(attempt)

    def close(self, fd):
        self.check()
        if fd in self.failed_closes:
            raise TTSCloneMaterializationError("cleanup_failed")
        if fd not in self.descriptors:
            raise TTSCloneMaterializationError("cleanup_failed")
        try:
            # Preserve each helper's actual native attribution.
            (self.descriptors[fd] or os.close)(fd)
        except BaseException as error:
            self.failed_closes.add(fd)
            self.errors.append(error)
            self.uncertain = True
            raise
        del self.descriptors[fd]
        self.parents.pop(fd, None)

    def finish(self):
        self.check()
        for fd in tuple(self.descriptors):
            if fd not in self.failed_closes:
                try:
                    self.close(fd)
                except BaseException:  # noqa: BLE001, S110 - error retained on native owner
                    pass  # Actual error and uncertain ownership retained above.
        if self.pending or self.descriptors or self.uncertain or self.residue:
            return False
        while self.leases:
            try:
                self.leases[-1].close()
            except BaseException as error:  # noqa: BLE001 - preserve native control flow
                self.uncertain = True
                self.errors.append(error)
                return False
            self.leases.pop()
        self.materializer._native_operations.discard(self)
        return True


def _checked_clone_native(native):
    if type(native) is not _CloneNativeOperation:
        raise TTSCloneMaterializationError("unavailable")
    native.check()
    return native


def _clone_select(native, parent, name):
    if native is not None:
        _checked_clone_native(native).select(parent, name)


def _clone_open(native, *args, **kwargs):
    return (
        os.open(*args, **kwargs)
        if native is None
        else _checked_clone_native(native).open(*args, **kwargs)
    )


def _clone_close(native, fd):
    return os.close(fd) if native is None else _checked_clone_native(native).close(fd)


@dataclass(slots=True)
class _MaterializationRecord:
    owner_name: str
    asset_name: str
    asset_path: Path
    root_fd: int
    owner_fd: int
    lock_fd: int
    root_identity: tuple[int, int]
    owner_identity: tuple[int, int]
    asset_identity: tuple[int, int]
    lock_identity: tuple[int, int]
    native: _CloneNativeOperation | None = None


class TTSCloneReferenceMaterialization:
    """An opaque live owner of one materialized clone reference."""

    __slots__ = ("_materializer", "_record", "_reference_text", "_token")

    def __init__(
        self,
        token: object,
        materializer: TTSCloneReferenceMaterializer,
        record: _MaterializationRecord,
        reference_text: str,
    ) -> None:
        if token is not _HANDLE_TOKEN:
            raise TypeError("Clone materializations cannot be constructed directly")
        self._materializer = materializer
        self._record = record
        self._reference_text = reference_text
        self._token = object()

    @property
    def voice_ref(self) -> Path:
        """Return the opaque WAV path while this owner remains live."""
        return self._record.asset_path

    @property
    def reference_text(self) -> str:
        """Return the exact bounded transcript for the admitted operation."""
        return self._reference_text

    async def aclose(self) -> None:
        """Remove the exact owned materialization, preserving substitutions."""
        await self._materializer._close_handle(self)

    async def validated_voice_ref(self) -> Path:
        """Return the path only while every retained filesystem identity matches."""
        return await self._materializer._validated_path(self)

    def _is_live_owner(self) -> bool:
        return self._materializer.owns(self)

    def __repr__(self) -> str:
        return "TTSCloneReferenceMaterialization(<private>)"


class TTSCloneReferenceMaterializer:
    """Lazily create and definitively reap POSIX clone-reference owners."""

    def __init__(self, runtime_root: Path) -> None:
        self._root = Path(runtime_root)
        self._configured_source = None
        self._closed = False
        self._maintenance_admission_closed = False
        self._owner_loop = None
        self._native_operations: set[_CloneNativeOperation] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._sweep_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._sweep_complete = False
        self._root_identity: tuple[int, int] | None = None
        self._worker_tasks: set[asyncio.Task[object]] = set()
        self._materialize_calls: set[asyncio.Task[object]] = set()
        self._handle_calls: set[asyncio.Task[object]] = set()
        self._active: dict[object, TTSCloneReferenceMaterialization] = {}

    async def materialize(
        self,
        reference: TTSCloneReference | CanonicalTTSCloneReference,
    ) -> TTSCloneReferenceMaterialization:
        """Publish one private owner from an exact stored reference.

        Args:
            reference: Exact canonical reference to materialize privately.

        Returns:
            A live opaque owner for the operation-scoped reference asset.

        Raises:
            TTSCloneMaterializationError: If the platform, root, reference,
                publication, or materializer lifecycle is unavailable.
            BaseException: A caller control-flow signal after retained worker
                cleanup reaches a terminal state.
        """
        self._check_loop()
        call = cast(asyncio.Task[object] | None, asyncio.current_task())
        if call is not None:
            self._materialize_calls.add(call)
        record: _MaterializationRecord | None = None
        try:
            self._check_source()
            if self._closed or self._maintenance_admission_closed:
                raise TTSCloneMaterializationError("closed")
            if not _POSIX_SUPPORTED:
                raise TTSCloneMaterializationError("unsupported")
            if type(reference) not in (TTSCloneReference, CanonicalTTSCloneReference):
                raise TTSCloneMaterializationError("unavailable")

            await self._ensure_swept()
            if self._closed or self._maintenance_admission_closed:
                raise TTSCloneMaterializationError("closed")

            worker = self._start_worker(
                _create_materialization_sync,
                self._root,
                self._root_identity,
                reference.wav_bytes,
            )
            cancelled, result, failure = await _await_retained_result(worker)
            self._worker_tasks.discard(worker)
            if failure is not None:
                if _is_control_flow(failure):
                    raise failure
                raise TTSCloneMaterializationError("unavailable") from None
            record = cast(_MaterializationRecord, result)

            if (
                cancelled is not None
                or self._closed
                or self._maintenance_admission_closed
            ):
                cleanup = self._start_worker(_cleanup_materialization_sync, record)
                _, _, cleanup_failure = await _await_retained_result(cleanup)
                self._worker_tasks.discard(cleanup)
                if cleanup_failure is not None:
                    raise cancelled or TTSCloneMaterializationError("cleanup_failed")
                if cancelled is not None:
                    raise cancelled
                raise TTSCloneMaterializationError("closed")

            try:
                self._check_source()
            except TTSCloneMaterializationError:
                cleanup = self._start_worker(_cleanup_materialization_sync, record)
                await _await_retained_result(cleanup)
                self._worker_tasks.discard(cleanup)
                raise

            handle = TTSCloneReferenceMaterialization(
                _HANDLE_TOKEN,
                self,
                record,
                reference.reference_text,
            )
            self._active[handle._token] = handle
            return handle
        finally:
            if call is not None:
                self._materialize_calls.discard(call)

    async def _ensure_swept(self) -> None:
        async with self._sweep_lock:
            if self._sweep_complete:
                return
            worker = self._start_worker(_prepare_runtime_root_sync, self._root)
            cancelled, prepared_root, failure = await _await_retained_result(worker)
            self._worker_tasks.discard(worker)
            if cancelled is not None:
                raise cancelled
            if failure is not None:
                if _is_control_flow(failure):
                    raise failure
                raise TTSCloneMaterializationError("unavailable") from None
            if self._closed or self._maintenance_admission_closed:
                raise TTSCloneMaterializationError("closed")
            self._root, self._root_identity = cast(
                tuple[Path, tuple[int, int]], prepared_root
            )
            self._sweep_complete = True

    def owns(self, handle: object) -> bool:
        """Return whether *handle* is an exact live owner from this instance."""
        return (
            type(handle) is TTSCloneReferenceMaterialization
            and self._active.get(handle._token) is handle
        )

    def seal(self) -> None:
        """Reject new materializations without beginning destructive cleanup."""
        self._closed = True

    async def close(self) -> None:
        """Seal creation and retain cleanup until all owned work terminates."""
        self._check_loop()
        if self._close_task is None:
            self.seal()
            self._close_task = asyncio.create_task(self._complete_close())
        cancelled, _, failure = await _await_retained_result(self._close_task)
        if cancelled is not None:
            raise cancelled
        if failure is not None:
            if _is_control_flow(failure):
                raise failure
            raise TTSCloneMaterializationError("cleanup_failed") from None

    async def _complete_close(self) -> None:
        current = asyncio.current_task()
        calls = [task for task in self._materialize_calls if task is not current]
        if calls:
            await asyncio.gather(
                *(asyncio.shield(task) for task in calls), return_exceptions=True
            )
        handle_calls = [task for task in self._handle_calls if task is not current]
        if handle_calls:
            await asyncio.gather(
                *(asyncio.shield(task) for task in handle_calls),
                return_exceptions=True,
            )

        failed = False
        control_flow: BaseException | None = None
        async with self._cleanup_lock:
            for handle in tuple(self._active.values()):
                record = handle._record
                worker = self._start_worker(_cleanup_materialization_sync, record)
                _, _, failure = await _await_retained_result(worker)
                self._worker_tasks.discard(worker)
                if failure is None:
                    self._active.pop(handle._token, None)
                else:
                    failed = True
                    if _is_control_flow(failure) and control_flow is None:
                        control_flow = failure
        if control_flow is not None:
            raise control_flow
        if failed or self._native_operations:
            raise TTSCloneMaterializationError("cleanup_failed")

    async def _close_handle(self, handle: TTSCloneReferenceMaterialization) -> None:
        self._check_loop()
        call = cast(asyncio.Task[object] | None, asyncio.current_task())
        if call is not None:
            self._handle_calls.add(call)
        try:
            async with self._cleanup_lock:
                if not self.owns(handle):
                    return
                worker = self._start_worker(
                    _cleanup_materialization_sync, handle._record
                )
                cancelled, _, failure = await _await_retained_result(worker)
                self._worker_tasks.discard(worker)
                if failure is None:
                    self._active.pop(handle._token, None)
                if cancelled is not None:
                    raise cancelled
                if failure is not None:
                    if _is_control_flow(failure):
                        raise failure
                    raise TTSCloneMaterializationError("cleanup_failed") from None
        finally:
            if call is not None:
                self._handle_calls.discard(call)

    async def _validated_path(
        self,
        handle: TTSCloneReferenceMaterialization,
    ) -> Path:
        self._check_loop()
        call = cast(asyncio.Task[object] | None, asyncio.current_task())
        if call is not None:
            self._handle_calls.add(call)
        try:
            if self._closed or self._maintenance_admission_closed:
                raise TTSCloneMaterializationError("closed")
            async with self._cleanup_lock:
                if not self.owns(handle):
                    raise TTSCloneMaterializationError("unavailable")
                worker = self._start_worker(
                    _validate_materialization_sync,
                    handle._record,
                )
                cancelled, result, failure = await _await_retained_result(worker)
                self._worker_tasks.discard(worker)
                if cancelled is not None:
                    raise cancelled
                if failure is not None:
                    if _is_control_flow(failure):
                        raise failure
                    raise TTSCloneMaterializationError("unavailable") from None
                return cast(Path, result)
        finally:
            if call is not None:
                self._handle_calls.discard(call)

    def _check_source(self):
        from .profile_source import check_materializer_source

        check_materializer_source(self)

    def _check_loop(self):
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
        if self._owner_loop is not loop:
            raise TTSCloneMaterializationError("unavailable")

    def _maintenance_close_admission(self):
        self._check_loop()
        self._maintenance_admission_closed = True

    async def _maintenance_drain(self, deadline: float) -> bool:
        self._check_loop()
        self._maintenance_close_admission()
        while self._materialize_calls or self._handle_calls or self._worker_tasks:
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(min(0.01, max(0, deadline - time.monotonic())))
        # Maintenance never destroys an asset still owned by a response/consumer.
        return not self._active and not self._native_operations

    async def _maintenance_resume(self):
        self._check_loop()
        self._check_source()
        if self._closed:
            raise TTSCloneMaterializationError("closed")
        self._maintenance_admission_closed = False

    def _start_worker(
        self,
        function: Callable[..., object],
        *args: object,
    ) -> asyncio.Task[object]:
        self._check_loop()
        # Reserve before to_thread is queued, including cancelled/late creators.
        cleanup = function is _cleanup_materialization_sync
        native = args[0].native if cleanup else _CloneNativeOperation(self)
        assert native is not None
        self._native_operations.add(native)
        task = asyncio.create_task(
            asyncio.to_thread(self._run_native_worker, native, function, args, cleanup)
        )
        self._worker_tasks.add(task)
        return task

    def _run_native_worker(self, native, function, args, cleanup):
        native.thread = threading.current_thread()
        result = None
        failure = None
        try:
            if not cleanup:
                # Two independent holds retain exclusion if release itself fails.
                native.admit()
                native.admit()
            result = function(*args, _native=native)
            return result
        except BaseException as error:
            failure = error
            raise
        finally:
            if failure is not None and not cleanup and native.record is not None:
                try:
                    _cleanup_materialization_sync(native.record, _native=native)
                except BaseException as error:  # noqa: BLE001 - original body stays primary
                    native.errors.append(error)
                    # Keep the exact unpublished record and its native holds.
                    cleanup = True
            if not isinstance(result, _MaterializationRecord) and not (
                cleanup and failure is not None and not native.uncertain
            ):
                prior_errors = len(native.errors)
                retired = native.finish()
                if failure is None:
                    for error in native.errors[prior_errors:]:
                        if not isinstance(error, Exception):
                            raise error
                if failure is None and not retired:
                    raise TTSCloneMaterializationError("cleanup_failed")


async def _await_retained_result(
    task: asyncio.Task[_T],
) -> tuple[asyncio.CancelledError | None, _T | None, BaseException | None]:
    cancellation: asyncio.CancelledError | None = None
    waiter = asyncio.current_task()
    cancellation_requests = waiter.cancelling() if waiter is not None else 0
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            next_requests = waiter.cancelling() if waiter is not None else 0
            if next_requests > cancellation_requests:
                cancellation = cancellation or error
                cancellation_requests = next_requests
        except BaseException:
            if not task.done():
                raise
            break
    try:
        return cancellation, task.result(), None
    except BaseException as error:  # noqa: BLE001 - return original task control/error
        return cancellation, None, error


def _owned_by_effective_user(info: os.stat_result) -> bool:
    return info.st_uid == os.geteuid()


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _is_control_flow(error: BaseException) -> bool:
    return isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit))


def _private_directory(info: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(info.st_mode)
        and _owned_by_effective_user(info)
        and stat.S_IMODE(info.st_mode) == _OWNER_DIRECTORY_MODE
    )


def _private_regular_file(info: os.stat_result) -> bool:
    return _owned_regular_file(info) and stat.S_IMODE(info.st_mode) == _OWNER_FILE_MODE


def _owned_regular_file(info: os.stat_result) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and _owned_by_effective_user(info)
        and info.st_nlink == 1
    )


def _private_open_lock(info: os.stat_result) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and _owned_by_effective_user(info)
        and info.st_nlink in (0, 1)
        and stat.S_IMODE(info.st_mode) == _OWNER_FILE_MODE
    )


def _lock_exclusive_nonblocking(descriptor: int) -> None:
    lock_module = fcntl
    if lock_module is None:
        raise OSError("POSIX file locking is unavailable")
    lock_module.flock(descriptor, lock_module.LOCK_EX | lock_module.LOCK_NB)


def _lock_root_exclusive(descriptor: int) -> None:
    lock_module = fcntl
    if lock_module is None:
        raise OSError("POSIX file locking is unavailable")
    lock_module.flock(descriptor, lock_module.LOCK_EX)


def _unlock(descriptor: int) -> None:
    lock_module = fcntl
    if lock_module is None:
        raise OSError("POSIX file locking is unavailable")
    lock_module.flock(descriptor, lock_module.LOCK_UN)


def _create_materialization_sync(
    root: Path,
    expected_root_identity: tuple[int, int] | None,
    wav_bytes: bytes,
    *,
    _native=None,
) -> _MaterializationRecord:
    if _native is not None:
        _checked_clone_native(_native).check_root(root)
    root_fd = _clone_open(_native, root, _DIRECTORY_FLAGS)
    owner_fd = -1
    lock_fd = -1
    owner_name = ""
    owner_created = False
    root_locked = False
    asset_name = ""
    owner_identity = lock_identity = asset_identity = None
    try:
        root_info = os.fstat(root_fd)
        if (
            expected_root_identity is None
            or _identity(root_info) != expected_root_identity
            or not _private_directory(root_info)
        ):
            raise OSError("unsafe runtime root")
        _lock_root_exclusive(root_fd)
        root_locked = True
        for _ in range(16):
            owner_name = f"clone-v1-{token_hex(16)}"
            _clone_select(_native, root_fd, owner_name)
            try:
                os.mkdir(owner_name, _OWNER_DIRECTORY_MODE, dir_fd=root_fd)
                owner_created = True
                break
            except FileExistsError:
                continue
        else:
            raise OSError("could not allocate owner")

        owner_fd = _clone_open(_native, owner_name, _DIRECTORY_FLAGS, dir_fd=root_fd)
        owner_identity = _identity(os.fstat(owner_fd))
        os.fchmod(owner_fd, _OWNER_DIRECTORY_MODE)
        owner_info = os.fstat(owner_fd)
        named_owner_info = os.stat(
            owner_name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        if not _private_directory(owner_info) or _identity(owner_info) != _identity(
            named_owner_info
        ):
            raise OSError("unsafe owner")

        _clone_select(_native, owner_fd, "owner.lock")
        lock_fd = _clone_open(
            _native,
            "owner.lock",
            _FILE_FLAGS | os.O_CREAT | os.O_EXCL,
            _OWNER_FILE_MODE,
            dir_fd=owner_fd,
        )
        lock_identity = _identity(os.fstat(lock_fd))
        os.fchmod(lock_fd, _OWNER_FILE_MODE)
        _lock_exclusive_nonblocking(lock_fd)
        lock_info = os.fstat(lock_fd)
        if not _private_regular_file(lock_info):
            raise OSError("unsafe lock")

        asset_name = f"asset-{token_hex(16)}.wav"
        _clone_select(_native, owner_fd, asset_name)
        asset_fd = _clone_open(
            _native,
            asset_name,
            _FILE_FLAGS | os.O_CREAT | os.O_EXCL,
            _OWNER_FILE_MODE,
            dir_fd=owner_fd,
        )
        try:
            asset_identity = _identity(os.fstat(asset_fd))
            os.fchmod(asset_fd, _OWNER_FILE_MODE)
            view = memoryview(wav_bytes)
            while view:
                written = os.write(asset_fd, view)
                if written <= 0:
                    raise OSError("short materialization write")
                view = view[written:]
            os.fsync(asset_fd)
            asset_info = os.fstat(asset_fd)
            named_asset_info = os.stat(
                asset_name,
                dir_fd=owner_fd,
                follow_symlinks=False,
            )
            if not _private_regular_file(asset_info) or _identity(
                asset_info
            ) != _identity(named_asset_info):
                raise OSError("unsafe asset")
        finally:
            _clone_close(_native, asset_fd)
        os.fsync(owner_fd)
        os.fsync(root_fd)
        _unlock(root_fd)
        root_locked = False

        record = _MaterializationRecord(
            owner_name=owner_name,
            asset_name=asset_name,
            asset_path=root / owner_name / asset_name,
            root_fd=root_fd,
            owner_fd=owner_fd,
            lock_fd=lock_fd,
            root_identity=_identity(root_info),
            owner_identity=_identity(owner_info),
            asset_identity=_identity(asset_info),
            lock_identity=_identity(lock_info),
            native=_native,
        )
        if _native is not None:
            _native.record = record
        return record
    except BaseException:
        # The failed creator still owns only its original entries. A renamed
        # owner or substituted leaf is evidence to preserve, never a new target.
        removed = not owner_created
        if owner_identity is not None and _same_entry(
            root_fd, owner_name, owner_identity
        ):
            for name, identity in (
                (asset_name, asset_identity),
                ("owner.lock", lock_identity),
            ):
                if (
                    name
                    and identity is not None
                    and _same_entry(owner_fd, name, identity)
                ):
                    try:
                        os.unlink(name, dir_fd=owner_fd)
                    except OSError:
                        pass
            if _same_entry(root_fd, owner_name, owner_identity):
                try:
                    os.rmdir(owner_name, dir_fd=root_fd)
                    removed = True
                except OSError:
                    pass
        if not removed and _native is not None:
            _native.residue = True
        if root_locked:
            try:
                _unlock(root_fd)
            except OSError:
                pass
        for fd in (lock_fd, owner_fd, root_fd):
            if fd >= 0:
                try:
                    _clone_close(_native, fd)
                except BaseException:  # noqa: BLE001, S110 - preserve body and native owner
                    pass  # Keep the original body signal; native owns the error.
        raise


def _same_entry(parent_fd, name, identity):
    try:
        return (
            _identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))
            == identity
        )
    except OSError:
        return False


def _validate_materialization_sync(
    record: _MaterializationRecord, *, _native=None
) -> Path:
    if _native is not None:
        _checked_clone_native(_native).check_record(record)
    root_info = os.fstat(record.root_fd)
    named_root_info = os.stat(
        record.asset_path.parent.parent,
        follow_symlinks=False,
    )
    if (
        _identity(root_info) != record.root_identity
        or _identity(named_root_info) != record.root_identity
        or not _private_directory(root_info)
        or not _private_directory(named_root_info)
    ):
        raise OSError("owned materialization changed")

    owner_info = os.fstat(record.owner_fd)
    named_owner_info = os.stat(
        record.owner_name,
        dir_fd=record.root_fd,
        follow_symlinks=False,
    )
    if (
        _identity(owner_info) != record.owner_identity
        or _identity(named_owner_info) != record.owner_identity
        or not _private_directory(owner_info)
        or not _private_directory(named_owner_info)
    ):
        raise OSError("owned materialization changed")

    lock_info = os.fstat(record.lock_fd)
    named_lock_info = os.stat(
        "owner.lock",
        dir_fd=record.owner_fd,
        follow_symlinks=False,
    )
    if (
        _identity(lock_info) != record.lock_identity
        or _identity(named_lock_info) != record.lock_identity
        or not _private_open_lock(lock_info)
        or not _private_regular_file(named_lock_info)
    ):
        raise OSError("owned materialization changed")

    asset_fd = _clone_open(
        _native, record.asset_name, _FILE_FLAGS, dir_fd=record.owner_fd
    )
    try:
        asset_info = os.fstat(asset_fd)
        named_asset_info = os.stat(
            record.asset_name,
            dir_fd=record.owner_fd,
            follow_symlinks=False,
        )
        if (
            _identity(asset_info) != record.asset_identity
            or _identity(named_asset_info) != record.asset_identity
            or not _private_regular_file(asset_info)
            or not _private_regular_file(named_asset_info)
        ):
            raise OSError("owned materialization changed")
    finally:
        _clone_close(_native, asset_fd)
    return record.asset_path


def _prepare_runtime_root_sync(
    root: Path, *, _native=None
) -> tuple[Path, tuple[int, int]]:
    if _native is not None:
        _checked_clone_native(_native).check_root(root)
    observers = (
        {}
        if _native is None
        else {
            "_open": _native.open_directory,
            "_close": _native.close,
        }
    )
    result = secure_private_directory(
        root, create=True, application_owned=True, **observers
    )
    selected = result.lexical_path
    root_fd = _clone_open(_native, selected, _DIRECTORY_FLAGS)
    try:
        root_info = os.fstat(root_fd)
        named_info = os.stat(selected, follow_symlinks=False)
        if not _private_directory(root_info) or _identity(root_info) != _identity(
            named_info
        ):
            raise OSError("unsafe runtime root")
        _lock_root_exclusive(root_fd)
        try:
            _sweep_orphans(root_fd, _native=_native)
        finally:
            _unlock(root_fd)
        return selected, _identity(root_info)
    finally:
        _clone_close(_native, root_fd)


def _sweep_orphans(root_fd: int, *, _native=None) -> None:
    if _native is not None:
        native = _checked_clone_native(_native)
        if native.parents.get(root_fd) != native.root:
            raise TTSCloneMaterializationError("unavailable")
    for owner_name in os.listdir(root_fd):
        if _OWNER_PATTERN.fullmatch(owner_name) is None:
            continue
        _clone_select(_native, root_fd, owner_name)
        owner_fd = -1
        lock_fd = -1
        try:
            owner_fd = _clone_open(
                _native, owner_name, _DIRECTORY_FLAGS, dir_fd=root_fd
            )
            owner_info = os.fstat(owner_fd)
            named_info = os.stat(owner_name, dir_fd=root_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(owner_info.st_mode)
                or not _owned_by_effective_user(owner_info)
                or _identity(owner_info) != _identity(named_info)
            ):
                continue
            if stat.S_IMODE(owner_info.st_mode) != _OWNER_DIRECTORY_MODE:
                os.fchmod(owner_fd, _OWNER_DIRECTORY_MODE)
                owner_info = os.fstat(owner_fd)
                if not _private_directory(owner_info):
                    continue

            entries = os.listdir(owner_fd)
            if not entries:
                os.rmdir(owner_name, dir_fd=root_fd)
                os.fsync(root_fd)
                continue
            assets = [name for name in entries if _ASSET_PATTERN.fullmatch(name)]
            if (
                "owner.lock" not in entries
                or len(assets) > 1
                or set(entries) != {"owner.lock", *assets}
            ):
                continue
            named_lock_info = os.stat(
                "owner.lock", dir_fd=owner_fd, follow_symlinks=False
            )
            if not _owned_regular_file(named_lock_info):
                continue
            _clone_select(_native, owner_fd, "owner.lock")
            lock_fd = _clone_open(_native, "owner.lock", _FILE_FLAGS, dir_fd=owner_fd)
            lock_info = os.fstat(lock_fd)
            if not _owned_regular_file(lock_info) or _identity(lock_info) != _identity(
                named_lock_info
            ):
                continue
            if stat.S_IMODE(lock_info.st_mode) != _OWNER_FILE_MODE:
                os.fchmod(lock_fd, _OWNER_FILE_MODE)
                lock_info = os.fstat(lock_fd)
                if not _private_regular_file(lock_info):
                    continue
            try:
                _lock_exclusive_nonblocking(lock_fd)
            except BlockingIOError:
                continue

            if assets:
                asset_name = assets[0]
                named_asset_info = os.stat(
                    asset_name, dir_fd=owner_fd, follow_symlinks=False
                )
                if not _owned_regular_file(named_asset_info):
                    continue
                _clone_select(_native, owner_fd, asset_name)
                asset_fd = _clone_open(
                    _native, asset_name, _FILE_FLAGS, dir_fd=owner_fd
                )
                try:
                    asset_info = os.fstat(asset_fd)
                    if not _owned_regular_file(asset_info) or _identity(
                        asset_info
                    ) != _identity(named_asset_info):
                        continue
                    if stat.S_IMODE(asset_info.st_mode) != _OWNER_FILE_MODE:
                        os.fchmod(asset_fd, _OWNER_FILE_MODE)
                        if not _private_regular_file(os.fstat(asset_fd)):
                            continue
                finally:
                    _clone_close(_native, asset_fd)
                os.unlink(asset_name, dir_fd=owner_fd)
            os.unlink("owner.lock", dir_fd=owner_fd)
            os.fsync(owner_fd)
            os.rmdir(owner_name, dir_fd=root_fd)
            os.fsync(root_fd)
        except OSError:
            continue
        finally:
            if lock_fd >= 0:
                _clone_close(_native, lock_fd)
            if owner_fd >= 0:
                _clone_close(_native, owner_fd)


def _cleanup_materialization_sync(
    record: _MaterializationRecord, *, _native=None
) -> None:
    if _native is not None:
        _checked_clone_native(_native).check_record(record, cleanup=True)
    root_info = os.fstat(record.root_fd)
    if _identity(root_info) != record.root_identity or not _private_directory(
        root_info
    ):
        raise OSError("owned materialization changed")

    try:
        named_owner_info = os.stat(
            record.owner_name,
            dir_fd=record.root_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        os.fsync(record.root_fd)
        _abandon_record(record)
        return

    owner_info = os.fstat(record.owner_fd)
    if (
        _identity(owner_info) != record.owner_identity
        or _identity(named_owner_info) != record.owner_identity
        or not _private_directory(owner_info)
    ):
        raise OSError("owned materialization changed")

    try:
        named_asset_info = os.stat(
            record.asset_name,
            dir_fd=record.owner_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        named_asset_info = None
    if named_asset_info is not None:
        if _identity(
            named_asset_info
        ) != record.asset_identity or not _private_regular_file(named_asset_info):
            raise OSError("owned materialization changed")
        os.unlink(record.asset_name, dir_fd=record.owner_fd)

    lock_info = os.fstat(record.lock_fd)
    if _identity(lock_info) != record.lock_identity or not _private_open_lock(
        lock_info
    ):
        raise OSError("owned materialization changed")
    try:
        named_lock_info = os.stat(
            "owner.lock",
            dir_fd=record.owner_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        named_lock_info = None
    if named_lock_info is not None:
        if _identity(
            named_lock_info
        ) != record.lock_identity or not _private_regular_file(named_lock_info):
            raise OSError("owned materialization changed")
        os.unlink("owner.lock", dir_fd=record.owner_fd)

    os.fsync(record.owner_fd)
    os.rmdir(record.owner_name, dir_fd=record.root_fd)
    os.fsync(record.root_fd)
    _abandon_record(record)


def _abandon_record(record: _MaterializationRecord) -> None:
    failure = None
    for descriptor in (record.lock_fd, record.owner_fd, record.root_fd):
        try:
            _clone_close(record.native, descriptor)
        except BaseException as error:  # noqa: BLE001 - close independent owners then rethrow
            failure = failure or error
    if failure is not None:
        raise failure


__all__ = [
    "TTSCloneMaterializationError",
    "TTSCloneReferenceMaterialization",
    "TTSCloneReferenceMaterializer",
]
