"""Private, operation-scoped filesystem owners for clone references."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import stat
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

from .windows_artifact_fs import (
    OS_WINDOWS_ARTIFACT_FILESYSTEM,
    WindowsArtifactFilesystem,
    WindowsFileIdentity,
    WindowsPinnedHandle,
    take_windows_artifact_cleanup_owner,
    windows_audio_cpp_platform_supported,
)

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
_windows_artifact_filesystem: WindowsArtifactFilesystem | None = (
    OS_WINDOWS_ARTIFACT_FILESYSTEM if windows_audio_cpp_platform_supported() else None
)


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


@dataclass(slots=True)
class _WindowsMaterializationRecord:
    filesystem: WindowsArtifactFilesystem
    owner_name: str
    asset_name: str
    asset_path: Path
    root_handle: WindowsPinnedHandle
    owner_handle: WindowsPinnedHandle
    lock_handle: WindowsPinnedHandle
    asset_handle: WindowsPinnedHandle
    root_identity: WindowsFileIdentity
    owner_identity: WindowsFileIdentity
    asset_identity: WindowsFileIdentity
    lock_identity: WindowsFileIdentity
    asset_size: int
    asset_sha256: str
    cleanup_started: bool = False
    asset_removed: bool = False
    lock_released: bool = False
    lock_removed: bool = False
    owner_removed: bool = False
    root_closed: bool = False


_OwnedMaterializationRecord = _MaterializationRecord | _WindowsMaterializationRecord


@dataclass(slots=True)
class _WindowsPartialCleanup:
    root_handle: WindowsPinnedHandle | None
    owner_handle: WindowsPinnedHandle | None
    lock_handle: WindowsPinnedHandle | None
    asset_handle: WindowsPinnedHandle | None
    extras: list[WindowsPinnedHandle]
    lock_released: bool = False

    def cleanup(self) -> None:
        """Retire only the exact partially-created objects, retaining failures."""

        retained_extras: list[WindowsPinnedHandle] = []
        for handle in self.extras:
            try:
                handle.close()
            except Exception:
                retained_extras.append(handle)
        self.extras = retained_extras
        if self.asset_handle is not None:
            self.asset_handle.delete_exact()
            self.asset_handle.close()
            self.asset_handle = None
        if self.lock_handle is not None:
            if not self.lock_released:
                self.lock_handle.unlock()
                self.lock_released = True
            self.lock_handle.delete_exact()
            self.lock_handle.close()
            self.lock_handle = None
        if self.owner_handle is not None:
            self.owner_handle.delete_exact()
            self.owner_handle.close()
            self.owner_handle = None
        if self.root_handle is not None:
            self.root_handle.close()
            self.root_handle = None
        if self.extras:
            raise OSError("partial cleanup incomplete")


class _WindowsPartialCleanupFailure(RuntimeError):
    __slots__ = ("cleanup", "primary")

    def __init__(
        self,
        cleanup: _WindowsPartialCleanup,
        primary: BaseException,
    ) -> None:
        super().__init__("Windows clone cleanup is incomplete")
        self.cleanup = cleanup
        self.primary = primary


class TTSCloneReferenceMaterialization:
    """An opaque live owner of one materialized clone reference."""

    __slots__ = ("_materializer", "_record", "_reference_text", "_token")

    def __init__(
        self,
        token: object,
        materializer: TTSCloneReferenceMaterializer,
        record: _OwnedMaterializationRecord,
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
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._sweep_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._sweep_complete = False
        self._root_identity: tuple[int, int] | WindowsFileIdentity | None = None
        self._worker_tasks: set[asyncio.Task[object]] = set()
        self._materialize_calls: set[asyncio.Task[object]] = set()
        self._handle_calls: set[asyncio.Task[object]] = set()
        self._active: dict[object, TTSCloneReferenceMaterialization] = {}
        self._pending_windows_cleanup: list[_WindowsPartialCleanup] = []

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
        call = cast(asyncio.Task[object] | None, asyncio.current_task())
        if call is not None:
            self._materialize_calls.add(call)
        record: _OwnedMaterializationRecord | None = None
        try:
            if self._closed:
                raise TTSCloneMaterializationError("closed")
            if not _POSIX_SUPPORTED and _windows_artifact_filesystem is None:
                raise TTSCloneMaterializationError("unsupported")
            if type(reference) not in (TTSCloneReference, CanonicalTTSCloneReference):
                raise TTSCloneMaterializationError("unavailable")

            await self._drain_pending_windows_cleanup()
            await self._ensure_swept()
            if self._closed:
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
                if isinstance(failure, _WindowsPartialCleanupFailure):
                    self._pending_windows_cleanup.append(failure.cleanup)
                    if _is_control_flow(failure.primary):
                        raise failure.primary
                    raise TTSCloneMaterializationError("cleanup_failed") from None
                if _is_control_flow(failure):
                    raise failure
                raise TTSCloneMaterializationError("unavailable") from None
            record = cast(_OwnedMaterializationRecord, result)

            handle = TTSCloneReferenceMaterialization(
                _HANDLE_TOKEN,
                self,
                record,
                reference.reference_text,
            )
            self._active[handle._token] = handle

            if cancelled is not None or self._closed:
                cleanup = self._start_worker(_cleanup_materialization_sync, record)
                _, _, cleanup_failure = await _await_retained_result(cleanup)
                self._worker_tasks.discard(cleanup)
                if cleanup_failure is None:
                    self._active.pop(handle._token, None)
                if cancelled is not None:
                    raise cancelled
                if cleanup_failure is not None and _is_control_flow(cleanup_failure):
                    raise cleanup_failure
                raise TTSCloneMaterializationError("closed")

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
                if isinstance(failure, _WindowsPartialCleanupFailure):
                    self._pending_windows_cleanup.append(failure.cleanup)
                    if _is_control_flow(failure.primary):
                        raise failure.primary
                    raise TTSCloneMaterializationError("cleanup_failed") from None
                cleanup_owner = take_windows_artifact_cleanup_owner(failure)
                if cleanup_owner is not None:
                    self._pending_windows_cleanup.append(
                        _WindowsPartialCleanup(
                            root_handle=None,
                            owner_handle=None,
                            lock_handle=None,
                            asset_handle=None,
                            extras=[cleanup_owner],
                            lock_released=True,
                        )
                    )
                    raise TTSCloneMaterializationError("cleanup_failed") from None
                if _is_control_flow(failure):
                    raise failure
                raise TTSCloneMaterializationError("unavailable") from None
            if self._closed:
                raise TTSCloneMaterializationError("closed")
            self._root, self._root_identity = cast(
                tuple[Path, tuple[int, int] | WindowsFileIdentity], prepared_root
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
        if self._close_task is None:
            self.seal()
            self._close_task = asyncio.create_task(self._complete_close())
        close_task = self._close_task
        cancelled, _, failure = await _await_retained_result(close_task)
        if cancelled is not None:
            raise cancelled
        if failure is not None:
            if self._close_task is close_task:
                self._close_task = None
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
            for cleanup in tuple(self._pending_windows_cleanup):
                worker = self._start_worker(cleanup.cleanup)
                _, _, failure = await _await_retained_result(worker)
                self._worker_tasks.discard(worker)
                if failure is None:
                    self._pending_windows_cleanup.remove(cleanup)
                else:
                    failed = True
                    if _is_control_flow(failure) and control_flow is None:
                        control_flow = failure
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
        if failed:
            raise TTSCloneMaterializationError("cleanup_failed")

    async def _drain_pending_windows_cleanup(self) -> None:
        async with self._cleanup_lock:
            failed = False
            for cleanup in tuple(self._pending_windows_cleanup):
                worker = self._start_worker(cleanup.cleanup)
                cancelled, _, failure = await _await_retained_result(worker)
                self._worker_tasks.discard(worker)
                if failure is None:
                    self._pending_windows_cleanup.remove(cleanup)
                else:
                    failed = True
                    if _is_control_flow(failure):
                        raise failure
                if cancelled is not None:
                    raise cancelled
            if failed:
                raise TTSCloneMaterializationError("cleanup_failed")

    async def _close_handle(self, handle: TTSCloneReferenceMaterialization) -> None:
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
        call = cast(asyncio.Task[object] | None, asyncio.current_task())
        if call is not None:
            self._handle_calls.add(call)
        try:
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

    def _start_worker(
        self,
        function: Callable[..., object],
        *args: object,
    ) -> asyncio.Task[object]:
        task: asyncio.Task[object] = asyncio.create_task(
            asyncio.to_thread(function, *args)
        )
        self._worker_tasks.add(task)
        return task


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
    except BaseException as error:
        return cancellation, None, error


def _owned_by_effective_user(info: os.stat_result) -> bool:
    return info.st_uid == os.geteuid()


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _is_control_flow(error: BaseException) -> bool:
    return not isinstance(error, Exception)


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
    expected_root_identity: tuple[int, int] | WindowsFileIdentity | None,
    wav_bytes: bytes,
) -> _OwnedMaterializationRecord:
    if _windows_artifact_filesystem is not None:
        return _create_windows_materialization_sync(
            _windows_artifact_filesystem,
            root,
            cast(WindowsFileIdentity | None, expected_root_identity),
            wav_bytes,
        )
    return _create_posix_materialization_sync(
        root,
        cast(tuple[int, int] | None, expected_root_identity),
        wav_bytes,
    )


def _create_posix_materialization_sync(
    root: Path,
    expected_root_identity: tuple[int, int] | None,
    wav_bytes: bytes,
) -> _MaterializationRecord:
    root_fd = os.open(root, _DIRECTORY_FLAGS)
    owner_fd = -1
    lock_fd = -1
    owner_name = ""
    root_locked = False
    asset_name = ""
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
            try:
                os.mkdir(owner_name, _OWNER_DIRECTORY_MODE, dir_fd=root_fd)
                break
            except FileExistsError:
                continue
        else:
            raise OSError("could not allocate owner")

        owner_fd = os.open(owner_name, _DIRECTORY_FLAGS, dir_fd=root_fd)
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

        lock_fd = os.open(
            "owner.lock",
            _FILE_FLAGS | os.O_CREAT | os.O_EXCL,
            _OWNER_FILE_MODE,
            dir_fd=owner_fd,
        )
        os.fchmod(lock_fd, _OWNER_FILE_MODE)
        _lock_exclusive_nonblocking(lock_fd)
        lock_info = os.fstat(lock_fd)
        if not _private_regular_file(lock_info):
            raise OSError("unsafe lock")

        asset_name = f"asset-{token_hex(16)}.wav"
        asset_fd = os.open(
            asset_name,
            _FILE_FLAGS | os.O_CREAT | os.O_EXCL,
            _OWNER_FILE_MODE,
            dir_fd=owner_fd,
        )
        try:
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
            os.close(asset_fd)
        os.fsync(owner_fd)
        os.fsync(root_fd)
        _unlock(root_fd)
        root_locked = False

        return _MaterializationRecord(
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
        )
    except BaseException:
        if owner_fd >= 0:
            if asset_name:
                try:
                    os.unlink(asset_name, dir_fd=owner_fd)
                except OSError:
                    pass
            try:
                os.unlink("owner.lock", dir_fd=owner_fd)
            except OSError:
                pass
            os.close(owner_fd)
        if lock_fd >= 0:
            os.close(lock_fd)
        if owner_name:
            try:
                os.rmdir(owner_name, dir_fd=root_fd)
            except OSError:
                pass
        if root_locked:
            try:
                _unlock(root_fd)
            except OSError:
                pass
        os.close(root_fd)
        raise


def _validate_materialization_sync(record: _OwnedMaterializationRecord) -> Path:
    if isinstance(record, _WindowsMaterializationRecord):
        return _validate_windows_materialization_sync(record.filesystem, record)
    return _validate_posix_materialization_sync(record)


def _validate_posix_materialization_sync(record: _MaterializationRecord) -> Path:
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

    asset_fd = os.open(record.asset_name, _FILE_FLAGS, dir_fd=record.owner_fd)
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
        os.close(asset_fd)
    return record.asset_path


def _prepare_runtime_root_sync(
    root: Path,
) -> tuple[Path, tuple[int, int] | WindowsFileIdentity]:
    if _windows_artifact_filesystem is not None:
        return _prepare_windows_runtime_root_sync(_windows_artifact_filesystem, root)
    return _prepare_posix_runtime_root_sync(root)


def _prepare_posix_runtime_root_sync(root: Path) -> tuple[Path, tuple[int, int]]:
    result = secure_private_directory(root, create=True, application_owned=True)
    selected = result.lexical_path
    root_fd = os.open(selected, _DIRECTORY_FLAGS)
    try:
        root_info = os.fstat(root_fd)
        named_info = os.stat(selected, follow_symlinks=False)
        if not _private_directory(root_info) or _identity(root_info) != _identity(
            named_info
        ):
            raise OSError("unsafe runtime root")
        _lock_root_exclusive(root_fd)
        try:
            _sweep_orphans(root_fd)
        finally:
            _unlock(root_fd)
        return selected, _identity(root_info)
    finally:
        os.close(root_fd)


def _sweep_orphans(root_fd: int) -> None:
    for owner_name in os.listdir(root_fd):
        if _OWNER_PATTERN.fullmatch(owner_name) is None:
            continue
        owner_fd = -1
        lock_fd = -1
        try:
            owner_fd = os.open(owner_name, _DIRECTORY_FLAGS, dir_fd=root_fd)
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
            lock_fd = os.open("owner.lock", _FILE_FLAGS, dir_fd=owner_fd)
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
                asset_fd = os.open(asset_name, _FILE_FLAGS, dir_fd=owner_fd)
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
                    os.close(asset_fd)
                os.unlink(asset_name, dir_fd=owner_fd)
            os.unlink("owner.lock", dir_fd=owner_fd)
            os.fsync(owner_fd)
            os.rmdir(owner_name, dir_fd=root_fd)
            os.fsync(root_fd)
        except OSError:
            continue
        finally:
            if lock_fd >= 0:
                os.close(lock_fd)
            if owner_fd >= 0:
                os.close(owner_fd)


def _prepare_windows_runtime_root_sync(
    filesystem: WindowsArtifactFilesystem,
    root: Path,
) -> tuple[Path, WindowsFileIdentity]:
    root.parent.mkdir(parents=True, exist_ok=True)
    root_handle = (
        filesystem.protect_private_directory(root)
        if root.exists()
        else filesystem.create_private_directory(root)
    )
    try:
        if (
            root_handle.identity.kind != "directory"
            or root_handle.identity.reparse_tag
            or not root_handle.verify_private_acl()
        ):
            raise OSError("unsafe runtime root")
        _sweep_windows_orphans(filesystem, root, root_handle.identity)
        return root, root_handle.identity
    finally:
        root_handle.close()


def _sweep_windows_orphans(
    filesystem: WindowsArtifactFilesystem,
    root: Path,
    root_identity: WindowsFileIdentity,
) -> None:
    """Remove only recognized, unlocked, completely pinned orphan owners."""

    for owner_path in tuple(root.iterdir()):
        if _OWNER_PATTERN.fullmatch(owner_path.name) is None:
            continue
        owner_handle: WindowsPinnedHandle | None = None
        lock_handle: WindowsPinnedHandle | None = None
        asset_handle: WindowsPinnedHandle | None = None
        locked = False
        try:
            root_check = filesystem.pin_directory_no_reparse(root)
            try:
                if root_check.identity != root_identity:
                    continue
            finally:
                root_check.close()
            owner_handle = filesystem.pin_directory_no_reparse(owner_path)
            if not owner_handle.verify_private_acl():
                continue
            entries = tuple(owner_path.iterdir())
            names = {entry.name for entry in entries}
            assets = tuple(
                entry for entry in entries if _ASSET_PATTERN.fullmatch(entry.name)
            )
            if not entries:
                owner_handle.delete_exact()
                owner_handle.close()
                owner_handle = None
                continue
            if (
                "owner.lock" not in names
                or len(assets) > 1
                or names != {"owner.lock", *(entry.name for entry in assets)}
            ):
                continue
            lock_handle = filesystem.open_file_no_reparse(
                owner_path / "owner.lock", writable=True
            )
            if not lock_handle.verify_private_acl():
                continue
            try:
                lock_handle.lock_exclusive_nonblocking()
                locked = True
            except Exception as error:
                if getattr(error, "code", None) == "busy":
                    continue
                raise
            if assets:
                asset_handle = filesystem.open_file_no_reparse(assets[0])
                if not asset_handle.verify_private_acl():
                    continue
                asset_handle.delete_exact()
                asset_handle.close()
                asset_handle = None
            lock_handle.unlock()
            locked = False
            lock_handle.delete_exact()
            lock_handle.close()
            lock_handle = None
            owner_handle.delete_exact()
            owner_handle.close()
            owner_handle = None
        except Exception:
            continue
        finally:
            if locked and lock_handle is not None:
                try:
                    lock_handle.unlock()
                except Exception:
                    pass
            for handle in (asset_handle, lock_handle, owner_handle):
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass


def _create_windows_materialization_sync(
    filesystem: WindowsArtifactFilesystem,
    root: Path,
    expected_root_identity: WindowsFileIdentity | None,
    wav_bytes: bytes,
) -> _WindowsMaterializationRecord:
    root_handle = filesystem.pin_directory_no_reparse(root)
    owner_handle: WindowsPinnedHandle | None = None
    lock_handle: WindowsPinnedHandle | None = None
    asset_handle: WindowsPinnedHandle | None = None
    extras: list[WindowsPinnedHandle] = []
    lock_acquired = False
    owner_name = ""
    asset_name = ""
    try:
        if (
            expected_root_identity is None
            or root_handle.identity != expected_root_identity
            or not root_handle.verify_private_acl()
        ):
            raise OSError("unsafe runtime root")
        for _ in range(16):
            owner_name = f"clone-v1-{token_hex(16)}"
            try:
                owner_handle = filesystem.create_private_directory(root / owner_name)
                break
            except Exception as error:
                if getattr(error, "code", None) != "unavailable":
                    raise
        if owner_handle is None:
            raise OSError("could not allocate owner")
        if not owner_handle.verify_private_acl():
            raise OSError("unsafe owner")

        lock_handle = filesystem.create_private_file(
            root / owner_name / "owner.lock",
            b"",
        )
        if not lock_handle.verify_private_acl():
            raise OSError("unsafe lock")
        lock_handle.lock_exclusive_nonblocking()
        lock_acquired = True

        asset_name = f"asset-{token_hex(16)}.wav"
        asset_handle = filesystem.create_private_file(
            root / owner_name / asset_name,
            wav_bytes,
            read_only=True,
        )
        if not asset_handle.verify_private_acl():
            raise OSError("unsafe asset")
        if {path.name for path in (root / owner_name).iterdir()} != {
            "owner.lock",
            asset_name,
        }:
            raise OSError("unsafe owner entries")
        return _WindowsMaterializationRecord(
            filesystem=filesystem,
            owner_name=owner_name,
            asset_name=asset_name,
            asset_path=root / owner_name / asset_name,
            root_handle=root_handle,
            owner_handle=owner_handle,
            lock_handle=lock_handle,
            asset_handle=asset_handle,
            root_identity=root_handle.identity,
            owner_identity=owner_handle.identity,
            lock_identity=lock_handle.identity,
            asset_identity=asset_handle.identity,
            asset_size=len(wav_bytes),
            asset_sha256=hashlib.sha256(wav_bytes).hexdigest(),
        )
    except BaseException as error:
        attached = take_windows_artifact_cleanup_owner(error)
        if attached is not None:
            extras.append(attached)
        cleanup = _WindowsPartialCleanup(
            root_handle=root_handle,
            owner_handle=owner_handle,
            lock_handle=lock_handle,
            asset_handle=asset_handle,
            extras=extras,
            lock_released=not lock_acquired,
        )
        try:
            cleanup.cleanup()
        except BaseException as cleanup_error:
            if _is_control_flow(cleanup_error):
                raise _WindowsPartialCleanupFailure(cleanup, cleanup_error) from None
            raise _WindowsPartialCleanupFailure(cleanup, error) from None
        raise error


def _open_matching_windows_handle(
    filesystem: WindowsArtifactFilesystem,
    path: Path,
    expected: WindowsFileIdentity,
) -> None:
    handle = (
        filesystem.pin_directory_no_reparse(path)
        if expected.kind == "directory"
        else filesystem.open_file_no_reparse(path)
    )
    try:
        if handle.identity != expected or not handle.verify_private_acl():
            raise OSError("owned materialization changed")
    finally:
        handle.close()


def _validate_windows_materialization_sync(
    filesystem: WindowsArtifactFilesystem,
    record: _WindowsMaterializationRecord,
) -> Path:
    if record.cleanup_started:
        raise OSError("owned materialization is closing")
    for handle, expected in (
        (record.root_handle, record.root_identity),
        (record.owner_handle, record.owner_identity),
        (record.lock_handle, record.lock_identity),
        (record.asset_handle, record.asset_identity),
    ):
        if handle.identity != expected or not handle.verify_private_acl():
            raise OSError("owned materialization changed")
    _open_matching_windows_handle(
        filesystem, record.asset_path.parent.parent, record.root_identity
    )
    _open_matching_windows_handle(
        filesystem, record.asset_path.parent, record.owner_identity
    )
    _open_matching_windows_handle(
        filesystem, record.asset_path.parent / "owner.lock", record.lock_identity
    )
    _open_matching_windows_handle(filesystem, record.asset_path, record.asset_identity)
    if {path.name for path in record.asset_path.parent.iterdir()} != {
        "owner.lock",
        record.asset_name,
    }:
        raise OSError("owned materialization changed")
    data = record.asset_handle.read(record.asset_size + 1)
    if (
        len(data) != record.asset_size
        or hashlib.sha256(data).hexdigest() != record.asset_sha256
    ):
        raise OSError("owned materialization changed")
    return record.asset_path


def _cleanup_windows_materialization_sync(
    record: _WindowsMaterializationRecord,
) -> None:
    if not record.cleanup_started:
        _validate_windows_materialization_sync(record.filesystem, record)
        record.cleanup_started = True
    if not record.asset_removed:
        record.asset_handle.delete_exact()
        record.asset_handle.close()
        record.asset_removed = True
    if not record.lock_released:
        record.lock_handle.unlock()
        record.lock_released = True
    if not record.lock_removed:
        record.lock_handle.delete_exact()
        record.lock_handle.close()
        record.lock_removed = True
    if not record.owner_removed:
        record.owner_handle.delete_exact()
        record.owner_handle.close()
        record.owner_removed = True
    if not record.root_closed:
        record.root_handle.close()
        record.root_closed = True


def _cleanup_materialization_sync(record: _OwnedMaterializationRecord) -> None:
    if isinstance(record, _WindowsMaterializationRecord):
        _cleanup_windows_materialization_sync(record)
        return
    _cleanup_posix_materialization_sync(record)


def _cleanup_posix_materialization_sync(record: _MaterializationRecord) -> None:
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


def _abandon_record(record: _OwnedMaterializationRecord) -> None:
    if isinstance(record, _WindowsMaterializationRecord):
        return
    for descriptor in (record.lock_fd, record.owner_fd, record.root_fd):
        try:
            os.close(descriptor)
        except OSError:
            pass


__all__ = [
    "TTSCloneMaterializationError",
    "TTSCloneReferenceMaterialization",
    "TTSCloneReferenceMaterializer",
]
