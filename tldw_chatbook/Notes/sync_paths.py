"""Descriptor-anchored filesystem boundary for legacy Notes sync."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY_FLAGS = os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC
_FILE_READ_FLAGS = os.O_RDONLY | _NOFOLLOW | _CLOEXEC
_FILE_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_RENAME_NOREPLACE = 4 if sys.platform == "darwin" else 1
_RENAME_EXCHANGE = 2
_DEFAULT_MAX_SYNC_FILE_BYTES = 10 * 1024 * 1024
_MAX_XATTR_NAMES_BYTES = 64 * 1024
_MAX_XATTR_COUNT = 128
_MAX_XATTR_VALUE_BYTES = 256 * 1024
_MAX_XATTR_TOTAL_BYTES = 1024 * 1024


def _descriptor_guards_available() -> bool:
    """Return whether the runtime exposes the required no-follow APIs."""

    required_dir_fd = {os.open, os.stat, os.mkdir, os.unlink, os.rename}
    return (
        os.name == "posix"
        and bool(_NOFOLLOW)
        and bool(_DIRECTORY)
        and required_dir_fd.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
        and os.listdir in os.supports_fd
    )


@dataclass(frozen=True)
class SyncPathIssue:
    """A bounded per-entry containment diagnostic."""

    relative_path: Path
    reason: str


@dataclass(frozen=True)
class SafeSyncFile:
    """A file read through the pinned sync-root descriptor."""

    absolute_path: Path
    relative_path: Path
    content: str
    mtime: float
    extension: str


@dataclass(frozen=True, slots=True, repr=False)
class SafeSyncFileIdentity:
    """Stable identity admitted by the descriptor-anchored POSIX adapter."""

    device: int
    inode: int
    link_count: int

    def __repr__(self) -> str:
        return "SafeSyncFileIdentity(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class SafeSyncBytes:
    """Exact file bytes plus the reviewed descriptor state."""

    relative_path: Path
    content: bytes
    identity: SafeSyncFileIdentity
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    owner_user: int
    owner_group: int
    flags: int
    extended_attributes: tuple[tuple[str, bytes], ...]
    has_extended_acl: bool

    def __repr__(self) -> str:
        return "SafeSyncBytes(<private>)"


class SyncPathError(OSError):
    """Raised when a legacy sync path cannot be verified safely."""

    def __init__(self, reason: str, relative_path: Path | str = Path(".")):
        self.reason = reason
        self.relative_path = Path(relative_path)
        super().__init__(f"{reason}: {self.relative_path}")


class SyncPathPartialError(SyncPathError):
    """A mutation crossed commit and needs durable attention."""

    def __init__(
        self,
        reason: str,
        cleanup_leaf: str | None = None,
        cleanup_identity: SafeSyncFileIdentity | None = None,
    ):
        self.cleanup_leaf = cleanup_leaf
        self.cleanup_identity = cleanup_identity
        super().__init__(reason)


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _is_reparse(entry_stat: os.stat_result) -> bool:
    attributes = getattr(entry_stat, "st_file_attributes", 0)
    return bool(attributes & _REPARSE_ATTRIBUTE)


def _xattr_call(
    function_name: str,
    descriptor: int,
    name: bytes | None = None,
    *,
    maximum: int,
) -> bytes:
    """Read one descriptor xattr value/list without pathname races."""

    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, function_name)
    if function_name == "flistxattr":
        arguments = (
            (descriptor, None, 0, 0)
            if sys.platform == "darwin"
            else (descriptor, None, 0)
        )
    elif sys.platform == "darwin":
        arguments = (descriptor, name, None, 0, 0, 0)
    else:
        arguments = (descriptor, name, None, 0)
    size = function(*arguments)
    if size < 0:
        error = ctypes.get_errno()
        if error in {errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}:
            return b""
        raise OSError(error, os.strerror(error))
    if size == 0:
        return b""
    if size > maximum:
        raise OSError(errno.EFBIG, "extended metadata exceeds its bound")
    buffer = ctypes.create_string_buffer(size)
    if function_name == "flistxattr":
        arguments = (
            (descriptor, buffer, size, 0)
            if sys.platform == "darwin"
            else (descriptor, buffer, size)
        )
    elif sys.platform == "darwin":
        arguments = (descriptor, name, buffer, size, 0, 0)
    else:
        arguments = (descriptor, name, buffer, size)
    read = function(*arguments)
    if read < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return bytes(buffer.raw[:read])


def _read_extended_attributes(descriptor: int) -> tuple[tuple[str, bytes], ...]:
    names = _xattr_call(
        "flistxattr",
        descriptor,
        maximum=_MAX_XATTR_NAMES_BYTES,
    )
    selected_names = sorted(filter(None, names.split(b"\0")))
    if len(selected_names) > _MAX_XATTR_COUNT:
        raise OSError(errno.EFBIG, "too many extended metadata entries")
    attributes: list[tuple[str, bytes]] = []
    total = 0
    for name in selected_names:
        value = _xattr_call(
            "fgetxattr",
            descriptor,
            name,
            maximum=_MAX_XATTR_VALUE_BYTES,
        )
        total += len(name) + len(value)
        if total > _MAX_XATTR_TOTAL_BYTES:
            raise OSError(errno.EFBIG, "extended metadata exceeds its bound")
        attributes.append((name.decode("utf-8", errors="surrogateescape"), value))
    return tuple(attributes)


def _write_extended_attributes(
    descriptor: int,
    attributes: tuple[tuple[str, bytes], ...],
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.fsetxattr
    for name, value in attributes:
        encoded_name = name.encode("utf-8", errors="surrogateescape")
        buffer = ctypes.create_string_buffer(value) if value else None
        if sys.platform == "darwin":
            result = function(descriptor, encoded_name, buffer, len(value), 0, 0)
        else:
            result = function(descriptor, encoded_name, buffer, len(value), 0)
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))


def _has_extended_acl(descriptor: int) -> bool:
    if sys.platform != "darwin":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    acl_get_fd_np = libc.acl_get_fd_np
    acl_get_fd_np.argtypes = (ctypes.c_int, ctypes.c_int)
    acl_get_fd_np.restype = ctypes.c_void_p
    acl_get_entry = libc.acl_get_entry
    acl_get_entry.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    )
    acl_get_entry.restype = ctypes.c_int
    acl_free = libc.acl_free
    acl_free.argtypes = (ctypes.c_void_p,)
    acl_free.restype = ctypes.c_int
    acl = acl_get_fd_np(descriptor, 0x00000100)
    if not acl:
        return False
    try:
        entry = ctypes.c_void_p()
        return acl_get_entry(acl, 0, ctypes.byref(entry)) == 0
    finally:
        acl_free(acl)


def _rename_with_flags(
    source_fd: int,
    source: str,
    destination_fd: int,
    destination: str,
    flags: int,
) -> None:
    """Use the native no-clobber/exchange rename primitive or fail closed."""

    libc = ctypes.CDLL(None, use_errno=True)
    function_name = "renameatx_np" if sys.platform == "darwin" else "renameat2"
    function = getattr(libc, function_name, None)
    if function is None:
        raise OSError(errno.ENOTSUP, "guarded rename is unavailable")
    result = function(
        source_fd,
        os.fsencode(source),
        destination_fd,
        os.fsencode(destination),
        flags,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def guarded_rename_available(*, platform: str | None = None) -> bool:
    """Return whether this runtime exposes the required native rename primitive."""

    selected = sys.platform if platform is None else platform
    if selected not in {"darwin", "linux"}:
        return False
    name = "renameatx_np" if selected == "darwin" else "renameat2"
    return getattr(ctypes.CDLL(None), name, None) is not None


class PinnedSyncRoot:
    """Pin a selected Notes root and contain all descendant operations."""

    def __init__(self, selected_root: Path | str):
        self.lexical_root = Path(selected_root)
        try:
            lexical = self.lexical_root.lstat()
            self.canonical_root = self.lexical_root.resolve(strict=True)
        except OSError:
            raise SyncPathError("root_unavailable") from None
        if stat.S_ISLNK(lexical.st_mode) or _is_reparse(lexical):
            raise SyncPathError("root_link_or_reparse")
        if not self.canonical_root.is_dir():
            raise SyncPathError("root_not_directory")
        self._root_fd: int | None = None
        self._root_stat: os.stat_result | None = None
        self._supported = _descriptor_guards_available()

    def __enter__(self) -> "PinnedSyncRoot":
        if not self._supported:
            return self
        root_fd = os.open(self.canonical_root, _DIRECTORY_FLAGS)
        try:
            opened = os.fstat(root_fd)
            entry = os.stat(self.canonical_root, follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or _is_reparse(entry)
                or not _same_identity(opened, entry)
            ):
                raise SyncPathError("root_identity_changed")
            self._root_fd = root_fd
            self._root_stat = opened
            return self
        except Exception:
            os.close(root_fd)
            raise

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None

    def _require_supported(self) -> tuple[int, os.stat_result]:
        if not self._supported:
            raise SyncPathError("unsupported_platform")
        if self._root_fd is None or self._root_stat is None:
            raise RuntimeError("PinnedSyncRoot must be entered before use")
        return self._root_fd, self._root_stat

    def _same_device(self, entry_stat: os.stat_result) -> bool:
        """Return whether an entry remains on the pinned root device."""

        _, root_stat = self._require_supported()
        return entry_stat.st_dev == root_stat.st_dev

    @staticmethod
    def _validate_relative(relative_path: Path | str) -> Path:
        selected = Path(relative_path)
        if (
            selected.is_absolute()
            or not selected.parts
            or any(part in {"", ".", ".."} for part in selected.parts)
        ):
            raise SyncPathError("invalid_relative_path", selected)
        return selected

    @classmethod
    def validate_relative(cls, relative_path: Path | str) -> Path:
        """Validate and normalize a stored legacy relative path."""

        return cls._validate_relative(relative_path)

    def _verify_root_path_identity(self) -> None:
        _, root_stat = self._require_supported()
        try:
            current = os.stat(self.canonical_root, follow_symlinks=False)
        except OSError as exc:
            raise SyncPathError("root_identity_changed") from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or _is_reparse(current)
            or not _same_identity(root_stat, current)
        ):
            raise SyncPathError("root_identity_changed")

    def _verified_child_directory(
        self,
        parent_fd: int,
        component: str,
        relative_path: Path,
        *,
        create: bool,
    ) -> int:
        try:
            entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if not create:
                raise SyncPathError("missing_parent", relative_path) from None
            created = False
            try:
                os.mkdir(component, mode=0o700, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                pass
            entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
        else:
            created = False

        if stat.S_ISLNK(entry.st_mode) or _is_reparse(entry):
            raise SyncPathError("link_or_reparse", relative_path)
        if not stat.S_ISDIR(entry.st_mode):
            raise SyncPathError("non_directory_parent", relative_path)
        if not self._same_device(entry):
            raise SyncPathError("cross_device", relative_path)

        try:
            child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise SyncPathError("link_or_reparse", relative_path) from exc
        opened = os.fstat(child_fd)
        if not _same_identity(entry, opened):
            os.close(child_fd)
            raise SyncPathError("parent_identity_changed", relative_path)
        if created:
            os.fchmod(child_fd, 0o700)
        return child_fd

    def _open_parent(self, relative_path: Path, *, create: bool) -> int:
        root_fd, _ = self._require_supported()
        current_fd = os.dup(root_fd)
        try:
            traversed = Path()
            for component in relative_path.parts[:-1]:
                traversed /= component
                child_fd = self._verified_child_directory(
                    current_fd,
                    component,
                    traversed,
                    create=create,
                )
                os.close(current_fd)
                current_fd = child_fd
            return current_fd
        except Exception:
            os.close(current_fd)
            raise

    def _read_file(
        self,
        parent_fd: int,
        leaf: str,
        relative_path: Path,
        entry: os.stat_result,
    ) -> SafeSyncFile:
        if entry.st_nlink != 1:
            raise SyncPathError("multiple_links", relative_path)
        if not self._same_device(entry):
            raise SyncPathError("cross_device", relative_path)
        try:
            file_fd = os.open(leaf, _FILE_READ_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise SyncPathError("link_or_reparse", relative_path) from exc
        try:
            opened = os.fstat(file_fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or not self._same_device(opened)
                or not _same_identity(entry, opened)
            ):
                raise SyncPathError("target_identity_changed", relative_path)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            content = b"".join(chunks).decode("utf-8")
            current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if not _same_identity(opened, current):
                raise SyncPathError("target_identity_changed", relative_path)
            self._verify_parent_identity(relative_path, parent_fd)
            return SafeSyncFile(
                absolute_path=self.canonical_root / relative_path,
                relative_path=relative_path,
                content=content,
                mtime=opened.st_mtime,
                extension=relative_path.suffix.lower(),
            )
        finally:
            os.close(file_fd)

    @staticmethod
    def _descriptor_state(entry: os.stat_result) -> tuple[int, ...]:
        return (
            entry.st_dev,
            entry.st_ino,
            entry.st_mode,
            entry.st_nlink,
            entry.st_size,
            entry.st_mtime_ns,
            entry.st_ctime_ns,
        )

    @staticmethod
    def _byte_result(
        relative_path: Path,
        content: bytes,
        entry: os.stat_result,
        extended_attributes: tuple[tuple[str, bytes], ...] = (),
        has_extended_acl: bool = False,
    ) -> SafeSyncBytes:
        return SafeSyncBytes(
            relative_path=relative_path,
            content=content,
            identity=SafeSyncFileIdentity(
                device=entry.st_dev,
                inode=entry.st_ino,
                link_count=entry.st_nlink,
            ),
            mode=stat.S_IMODE(entry.st_mode),
            size=entry.st_size,
            mtime_ns=entry.st_mtime_ns,
            ctime_ns=entry.st_ctime_ns,
            owner_user=entry.st_uid,
            owner_group=entry.st_gid,
            flags=getattr(entry, "st_flags", 0),
            extended_attributes=extended_attributes,
            has_extended_acl=has_extended_acl,
        )

    def _after_read(self, _relative_path: Path) -> None:
        """Test seam after bytes are read but before state is accepted."""

    def _read_bytes(
        self,
        parent_fd: int,
        leaf: str,
        relative_path: Path,
        entry: os.stat_result,
        max_bytes: int = _DEFAULT_MAX_SYNC_FILE_BYTES,
    ) -> SafeSyncBytes:
        if entry.st_nlink != 1:
            raise SyncPathError("multiple_links", relative_path)
        if not self._same_device(entry):
            raise SyncPathError("cross_device", relative_path)
        try:
            file_fd = os.open(leaf, _FILE_READ_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise SyncPathError("link_or_reparse", relative_path) from exc
        try:
            opened = os.fstat(file_fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or not self._same_device(opened)
                or not _same_identity(entry, opened)
            ):
                raise SyncPathError("target_identity_changed", relative_path)
            try:
                before_attributes = _read_extended_attributes(file_fd)
                before_acl = _has_extended_acl(file_fd)
            except OSError:
                raise SyncPathError("unsupported_metadata", relative_path) from None
            chunks: list[bytes] = []
            total = 0
            while chunk := os.read(file_fd, min(1024 * 1024, max_bytes + 1)):
                total += len(chunk)
                if total > max_bytes:
                    raise SyncPathError("max_file_bytes_exceeded", relative_path)
                chunks.append(chunk)
            self._after_read(relative_path)
            after = os.fstat(file_fd)
            try:
                after_attributes = _read_extended_attributes(file_fd)
                after_acl = _has_extended_acl(file_fd)
            except OSError:
                raise SyncPathError("unsupported_metadata", relative_path) from None
            if (
                self._descriptor_state(opened) != self._descriptor_state(after)
                or before_attributes != after_attributes
                or before_acl != after_acl
            ):
                raise SyncPathError("target_changed_during_read", relative_path)
            current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if self._descriptor_state(after) != self._descriptor_state(current):
                raise SyncPathError("target_identity_changed", relative_path)
            self._verify_parent_identity(relative_path, parent_fd)
            return self._byte_result(
                relative_path,
                b"".join(chunks),
                after,
                after_attributes,
                after_acl,
            )
        finally:
            os.close(file_fd)

    def _verify_parent_identity(
        self,
        relative_path: Path,
        parent_fd: int,
    ) -> None:
        """Verify an opened parent is still reachable from the pinned root."""

        fresh_parent_fd = self._open_parent(relative_path, create=False)
        try:
            if not _same_identity(
                os.fstat(parent_fd),
                os.fstat(fresh_parent_fd),
            ):
                raise SyncPathError("parent_identity_changed", relative_path)
        finally:
            os.close(fresh_parent_fd)

    def scan(
        self,
        extensions: Iterable[str],
    ) -> tuple[dict[Path, SafeSyncFile], list[SyncPathIssue]]:
        """Scan verified descendants without following aliases."""

        if not self._supported:
            return {}, [SyncPathIssue(Path("."), "unsupported_platform")]
        root_fd, _ = self._require_supported()
        self._verify_root_path_identity()
        selected_extensions = {extension.lower() for extension in extensions}
        files: dict[Path, SafeSyncFile] = {}
        issues: list[SyncPathIssue] = []

        def walk(directory_fd: int, relative_directory: Path) -> None:
            for name in sorted(os.listdir(directory_fd)):
                relative_path = relative_directory / name
                try:
                    entry = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if stat.S_ISLNK(entry.st_mode) or _is_reparse(entry):
                        raise SyncPathError("link_or_reparse", relative_path)
                    if not self._same_device(entry):
                        raise SyncPathError("cross_device", relative_path)
                    if stat.S_ISDIR(entry.st_mode):
                        child_fd = self._verified_child_directory(
                            directory_fd,
                            name,
                            relative_path,
                            create=False,
                        )
                        try:
                            walk(child_fd, relative_path)
                        finally:
                            os.close(child_fd)
                        continue
                    if not stat.S_ISREG(entry.st_mode):
                        raise SyncPathError("non_regular", relative_path)
                    if relative_path.suffix.lower() not in selected_extensions:
                        continue
                    files[relative_path] = self._read_file(
                        directory_fd,
                        name,
                        relative_path,
                        entry,
                    )
                except SyncPathError as exc:
                    issues.append(SyncPathIssue(relative_path, exc.reason))
                except (OSError, UnicodeError):
                    issues.append(SyncPathIssue(relative_path, "operation_failed"))

        scan_fd = os.dup(root_fd)
        try:
            walk(scan_fd, Path())
        finally:
            os.close(scan_fd)
        return files, issues

    def read_file(self, relative_path: Path | str) -> SafeSyncFile:
        """Read one verified regular file beneath the pinned root."""

        selected = self._validate_relative(relative_path)
        self._require_supported()
        self._verify_root_path_identity()
        try:
            parent_fd = self._open_parent(selected, create=False)
        except SyncPathError:
            raise
        except OSError:
            raise SyncPathError("operation_failed", selected) from None
        try:
            entry = os.stat(
                selected.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(entry.st_mode) or _is_reparse(entry):
                raise SyncPathError("link_or_reparse", selected)
            if not stat.S_ISREG(entry.st_mode):
                raise SyncPathError("non_regular", selected)
            return self._read_file(parent_fd, selected.name, selected, entry)
        except FileNotFoundError:
            raise SyncPathError("missing_target", selected) from None
        except SyncPathError:
            raise
        except OSError:
            raise SyncPathError("operation_failed", selected) from None
        finally:
            os.close(parent_fd)

    def read_bytes(
        self,
        relative_path: Path | str,
        *,
        max_bytes: int = _DEFAULT_MAX_SYNC_FILE_BYTES,
    ) -> SafeSyncBytes:
        """Read exact bytes while proving descriptor state stayed unchanged."""

        selected = self._validate_relative(relative_path)
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        self._require_supported()
        self._verify_root_path_identity()
        try:
            parent_fd = self._open_parent(selected, create=False)
        except SyncPathError:
            raise
        except OSError:
            raise SyncPathError("operation_failed", selected) from None
        try:
            entry = os.stat(
                selected.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(entry.st_mode) or _is_reparse(entry):
                raise SyncPathError("link_or_reparse", selected)
            if not stat.S_ISREG(entry.st_mode):
                raise SyncPathError("non_regular", selected)
            return self._read_bytes(
                parent_fd,
                selected.name,
                selected,
                entry,
                max_bytes,
            )
        except FileNotFoundError:
            raise SyncPathError("missing_target", selected) from None
        except SyncPathError:
            raise
        except OSError:
            raise SyncPathError("operation_failed", selected) from None
        finally:
            os.close(parent_fd)

    def cleanup_private_file(
        self,
        relative_path: Path | str,
        *,
        expected_identity: SafeSyncFileIdentity,
    ) -> None:
        """Idempotently remove one guarded replacement staging file."""

        if type(expected_identity) is not SafeSyncFileIdentity:
            raise TypeError("expected_identity must be a SafeSyncFileIdentity.")
        selected = self._validate_relative(relative_path)
        prefix, separator, token = selected.name.rpartition(".tmp-")
        if (
            separator != ".tmp-"
            or not prefix.startswith(".")
            or len(prefix) == 1
            or len(token) != 32
            or any(character not in "0123456789abcdef" for character in token)
        ):
            raise SyncPathError("invalid_cleanup_authority", selected)
        self._require_supported()
        self._verify_root_path_identity()
        try:
            parent_fd = self._open_parent(selected, create=False)
        except SyncPathError:
            raise
        except OSError:
            raise SyncPathError("operation_failed", selected) from None
        try:
            try:
                entry = os.stat(
                    selected.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return
            if (
                not stat.S_ISREG(entry.st_mode)
                or stat.S_ISLNK(entry.st_mode)
                or _is_reparse(entry)
                or entry.st_nlink != 1
                or not self._same_device(entry)
            ):
                raise SyncPathError("invalid_cleanup_authority", selected)
            if (
                entry.st_dev,
                entry.st_ino,
                entry.st_nlink,
            ) != (
                expected_identity.device,
                expected_identity.inode,
                expected_identity.link_count,
            ):
                raise SyncPathError("target_identity_changed", selected)
            self._verify_parent_identity(selected, parent_fd)
            current = os.stat(
                selected.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if self._descriptor_state(entry) != self._descriptor_state(current):
                raise SyncPathError("target_identity_changed", selected)
            self._before_cleanup_rename(selected)
            quarantine_leaf: str | None = None
            for _ in range(32):
                candidate = f".{selected.name}.cleanup-{uuid.uuid4().hex}"
                try:
                    _rename_with_flags(
                        parent_fd,
                        selected.name,
                        parent_fd,
                        candidate,
                        _RENAME_NOREPLACE,
                    )
                    quarantine_leaf = candidate
                    break
                except FileExistsError:
                    continue
                except OSError:
                    raise SyncPathError(
                        "guarded_rename_unavailable", selected
                    ) from None
            if quarantine_leaf is None:
                raise SyncPathError("temporary_name_exhausted", selected)
            quarantined = os.stat(
                quarantine_leaf,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                quarantined.st_dev,
                quarantined.st_ino,
                quarantined.st_nlink,
            ) != (
                expected_identity.device,
                expected_identity.inode,
                expected_identity.link_count,
            ):
                try:
                    _rename_with_flags(
                        parent_fd,
                        quarantine_leaf,
                        parent_fd,
                        selected.name,
                        _RENAME_NOREPLACE,
                    )
                    os.fsync(parent_fd)
                except OSError:
                    raise SyncPathPartialError(
                        "cleanup_rollback_failed",
                        quarantine_leaf,
                        SafeSyncFileIdentity(
                            device=quarantined.st_dev,
                            inode=quarantined.st_ino,
                            link_count=quarantined.st_nlink,
                        ),
                    ) from None
                raise SyncPathError("target_identity_changed", selected)
            try:
                os.unlink(quarantine_leaf, dir_fd=parent_fd)
            except OSError:
                try:
                    _rename_with_flags(
                        parent_fd,
                        quarantine_leaf,
                        parent_fd,
                        selected.name,
                        _RENAME_NOREPLACE,
                    )
                    os.fsync(parent_fd)
                except OSError:
                    raise SyncPathPartialError(
                        "cleanup_rollback_failed",
                        quarantine_leaf,
                        expected_identity,
                    ) from None
                raise SyncPathError("operation_failed", selected) from None
            try:
                os.fsync(parent_fd)
            except OSError:
                raise SyncPathPartialError("cleanup_commit_unverified") from None
        except SyncPathError:
            raise
        except OSError:
            raise SyncPathError("operation_failed", selected) from None
        finally:
            os.close(parent_fd)

    def _before_cleanup_rename(self, _relative_path: Path) -> None:
        """Test seam immediately before private cleanup quarantine."""

    def _before_replace(self, _relative_path: Path) -> None:
        """Test seam immediately before identity rechecks and replacement."""

    def _before_create(self, _relative_path: Path) -> None:
        """Test seam immediately before the exclusive create in ``create_new_text``."""

    def _before_commit(self, _relative_path: Path) -> None:
        """Test seam at the guarded native-rename boundary."""

    def _after_displaced_verification(self, _relative_path: Path) -> None:
        """Test seam before the exchanged replacement is accepted."""

    def _existing_target(
        self,
        parent_fd: int,
        leaf: str,
        relative_path: Path,
    ) -> os.stat_result | None:
        try:
            entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(entry.st_mode) or _is_reparse(entry):
            raise SyncPathError("link_or_reparse", relative_path)
        if not stat.S_ISREG(entry.st_mode):
            raise SyncPathError("non_regular", relative_path)
        if entry.st_nlink != 1:
            raise SyncPathError("multiple_links", relative_path)
        if not self._same_device(entry):
            raise SyncPathError("cross_device", relative_path)
        return entry

    @staticmethod
    def _write_all(file_fd: int, content: bytes) -> None:
        remaining = memoryview(content)
        while remaining:
            written = os.write(file_fd, remaining)
            if written <= 0:
                raise OSError("short write")
            remaining = remaining[written:]

    def create_new_text(
        self,
        relative_path: Path | str,
        content: str,
    ) -> SafeSyncFile:
        """Create a NEW file beneath the pinned root; never replace one.

        The counterpart to ``write_text`` for content that must not collide
        with content already at the target. ``write_text`` writes a temporary
        file and renames it over the destination, which is atomic but
        **replacing** -- correct for a synced note (the file IS the note), and
        wrong for a preserved conflict copy, where the thing already at that
        name is another run's saved copy of user text.

        The name is claimed with ``O_CREAT | O_EXCL`` (plus the usual
        ``O_NOFOLLOW``), so the claim and the check are one syscall: two
        concurrent runs cannot both decide a name is free. The loser gets
        ``FileExistsError`` and is expected to pick another name rather than
        overwrite. Anything that goes wrong after the create unlinks the file
        this call made, so a caller never sees a half-written file it was told
        had failed.

        Args:
            relative_path: Destination relative to the pinned root.
            content: Text to write, encoded UTF-8.

        Returns:
            The created file, described the same way ``write_text`` describes
            a replaced one.

        Raises:
            FileExistsError: If the name is already taken. This is the signal
                to try another name -- it is NOT a boundary violation.
            SyncPathError: If the path, its parent, or the resulting file
                fails the pinned-root checks.
            OSError: Propagated from the create or the write.
        """
        selected = self._validate_relative(relative_path)
        self._require_supported()
        self._verify_root_path_identity()
        parent_fd = self._open_parent(selected, create=False)
        created = False
        try:
            self._before_create(selected)
            file_fd = os.open(
                selected.name,
                _FILE_WRITE_FLAGS,
                0o600,
                dir_fd=parent_fd,
            )
            created = True
            try:
                self._write_all(file_fd, content.encode("utf-8"))
                os.fchmod(file_fd, 0o600)
                os.fsync(file_fd)
                opened = os.fstat(file_fd)
            finally:
                os.close(file_fd)

            self._verify_root_path_identity()
            self._verify_parent_identity(selected, parent_fd)
            current = os.stat(selected.name, dir_fd=parent_fd, follow_symlinks=False)
            if not _same_identity(opened, current):
                raise SyncPathError("target_identity_changed", selected)
            if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                raise SyncPathError("multiple_links", selected)
            if not self._same_device(current):
                raise SyncPathError("cross_device", selected)
            return SafeSyncFile(
                absolute_path=self.canonical_root / selected,
                relative_path=selected,
                content=content,
                mtime=current.st_mtime,
                extension=selected.suffix.lower(),
            )
        except Exception:
            # Only ever our own file: ``created`` is False when the create
            # itself failed, so a name someone else holds is never unlinked.
            if created:
                try:
                    os.unlink(selected.name, dir_fd=parent_fd)
                except OSError:
                    pass
            raise
        finally:
            os.close(parent_fd)

    def write_text(
        self,
        relative_path: Path | str,
        content: str,
    ) -> SafeSyncFile:
        """Atomically write beneath the pinned root, preserving prior mode."""

        selected = self._validate_relative(relative_path)
        self._require_supported()
        self._verify_root_path_identity()
        try:
            parent_fd = self._open_parent(selected, create=True)
        except SyncPathError:
            raise
        except OSError:
            raise SyncPathError("operation_failed", selected) from None
        temporary_leaf: str | None = None
        try:
            parent_identity = os.fstat(parent_fd)
            existing = self._existing_target(
                parent_fd,
                selected.name,
                selected,
            )
            selected_mode = (
                stat.S_IMODE(existing.st_mode) if existing is not None else 0o600
            )
            for _ in range(32):
                candidate = f".{selected.name}.tmp-{uuid.uuid4().hex}"
                try:
                    temporary_fd = os.open(
                        candidate,
                        _FILE_WRITE_FLAGS,
                        selected_mode,
                        dir_fd=parent_fd,
                    )
                    temporary_leaf = candidate
                    break
                except FileExistsError:
                    continue
            else:
                raise SyncPathError("temporary_name_exhausted", selected)

            try:
                self._write_all(temporary_fd, content.encode("utf-8"))
                os.fchmod(temporary_fd, selected_mode)
                os.fsync(temporary_fd)
            finally:
                os.close(temporary_fd)

            self._before_replace(selected)
            self._verify_root_path_identity()
            self._verify_parent_identity(selected, parent_fd)
            if not _same_identity(parent_identity, os.fstat(parent_fd)):
                raise SyncPathError("parent_identity_changed", selected)

            current = self._existing_target(parent_fd, selected.name, selected)
            if (existing is None) != (current is None) or (
                existing is not None
                and current is not None
                and not _same_identity(existing, current)
            ):
                raise SyncPathError("target_identity_changed", selected)

            os.rename(
                temporary_leaf,
                selected.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temporary_leaf = None
            final = self._existing_target(parent_fd, selected.name, selected)
            if final is None or stat.S_IMODE(final.st_mode) != selected_mode:
                raise SyncPathError("replacement_postcondition_failed", selected)
            return SafeSyncFile(
                absolute_path=self.canonical_root / selected,
                relative_path=selected,
                content=content,
                mtime=final.st_mtime,
                extension=selected.suffix.lower(),
            )
        finally:
            if temporary_leaf is not None:
                try:
                    os.unlink(temporary_leaf, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            os.close(parent_fd)

    @staticmethod
    def _matches_reviewed_state(
        entry: os.stat_result | None,
        expected: SafeSyncBytes | None,
    ) -> bool:
        if entry is None or expected is None:
            return entry is None and expected is None
        return (
            entry.st_dev == expected.identity.device
            and entry.st_ino == expected.identity.inode
            and entry.st_nlink == expected.identity.link_count
            and stat.S_IMODE(entry.st_mode) == expected.mode
            and entry.st_size == expected.size
            and entry.st_mtime_ns == expected.mtime_ns
            and entry.st_ctime_ns == expected.ctime_ns
        )

    @staticmethod
    def _matches_displaced_state(
        observed: SafeSyncBytes | None,
        expected: SafeSyncBytes,
    ) -> bool:
        """Compare exchange-displaced state, whose rename legitimately changes ctime."""

        return observed is not None and (
            observed.content == expected.content
            and observed.identity == expected.identity
            and observed.mode == expected.mode
            and observed.size == expected.size
            and observed.mtime_ns == expected.mtime_ns
            and observed.owner_user == expected.owner_user
            and observed.owner_group == expected.owner_group
            and observed.flags == expected.flags
            and observed.extended_attributes == expected.extended_attributes
            and observed.has_extended_acl == expected.has_extended_acl
        )

    def replace_bytes(
        self,
        relative_path: Path | str,
        content: bytes,
        *,
        expected: SafeSyncBytes | None,
        mode: int,
    ) -> SafeSyncBytes:
        """Atomically replace bytes only if the reviewed state still matches."""

        selected = self._validate_relative(relative_path)
        if type(content) is not bytes:
            raise TypeError("content must be bytes.")
        if type(mode) is not int or not 0 <= mode <= 0o7777:
            raise ValueError("mode must be a supported file mode.")
        if expected is not None and expected.relative_path != selected:
            raise SyncPathError("expected_path_mismatch", selected)
        self._require_supported()
        self._verify_root_path_identity()
        try:
            parent_fd = self._open_parent(selected, create=True)
        except SyncPathError:
            raise
        except OSError:
            raise SyncPathError("operation_failed", selected) from None
        temporary_leaf: str | None = None
        committed = False
        try:
            parent_identity = os.fstat(parent_fd)
            existing = self._existing_target(parent_fd, selected.name, selected)
            if not self._matches_reviewed_state(existing, expected):
                raise SyncPathError("target_identity_changed", selected)
            for _ in range(32):
                candidate = f".{selected.name}.tmp-{uuid.uuid4().hex}"
                try:
                    temporary_fd = os.open(
                        candidate,
                        _FILE_WRITE_FLAGS,
                        mode,
                        dir_fd=parent_fd,
                    )
                    temporary_leaf = candidate
                    break
                except FileExistsError:
                    continue
            else:
                raise SyncPathError("temporary_name_exhausted", selected)
            try:
                try:
                    self._write_all(temporary_fd, content)
                    os.fchmod(temporary_fd, mode)
                    try:
                        _write_extended_attributes(
                            temporary_fd,
                            (() if expected is None else expected.extended_attributes),
                        )
                    except OSError:
                        raise SyncPathError("unsupported_metadata", selected) from None
                    os.fsync(temporary_fd)
                finally:
                    os.close(temporary_fd)
            except SyncPathError:
                raise
            except OSError:
                raise SyncPathError("operation_failed", selected) from None

            self._before_replace(selected)
            self._verify_root_path_identity()
            self._verify_parent_identity(selected, parent_fd)
            if not _same_identity(parent_identity, os.fstat(parent_fd)):
                raise SyncPathError("parent_identity_changed", selected)
            current = self._existing_target(parent_fd, selected.name, selected)
            if not self._matches_reviewed_state(current, expected):
                raise SyncPathError("target_identity_changed", selected)
            self._before_commit(selected)
            try:
                if expected is None:
                    _rename_with_flags(
                        parent_fd,
                        temporary_leaf,
                        parent_fd,
                        selected.name,
                        _RENAME_NOREPLACE,
                    )
                    committed = True
                    temporary_leaf = None
                    final = self._existing_target(parent_fd, selected.name, selected)
                    if final is None or stat.S_IMODE(final.st_mode) != mode:
                        raise SyncPathError(
                            "replacement_postcondition_failed",
                            selected,
                        )
                    try:
                        os.fsync(parent_fd)
                    except OSError:
                        raise SyncPathPartialError(
                            "replacement_commit_unverified",
                            selected.as_posix(),
                        ) from None
                else:
                    _rename_with_flags(
                        parent_fd,
                        temporary_leaf,
                        parent_fd,
                        selected.name,
                        _RENAME_EXCHANGE,
                    )
                    committed = True
                    try:
                        displaced = self._existing_target(
                            parent_fd,
                            temporary_leaf,
                            selected,
                        )
                        displaced_snapshot = (
                            None
                            if displaced is None
                            else self._read_bytes(
                                parent_fd,
                                temporary_leaf,
                                selected,
                                displaced,
                                max(expected.size, 1),
                            )
                        )
                        if not self._matches_displaced_state(
                            displaced_snapshot,
                            expected,
                        ):
                            raise SyncPathError("target_identity_changed", selected)
                        self._after_displaced_verification(selected)
                        final = self._existing_target(
                            parent_fd,
                            selected.name,
                            selected,
                        )
                        final_snapshot = (
                            None
                            if final is None
                            else self._read_bytes(
                                parent_fd,
                                selected.name,
                                selected,
                                final,
                                max(len(content), 1),
                            )
                        )
                        expected_attributes = expected.extended_attributes
                        if (
                            final_snapshot is None
                            or final_snapshot.content != content
                            or final_snapshot.mode != mode
                            or final_snapshot.extended_attributes != expected_attributes
                            or final_snapshot.has_extended_acl
                        ):
                            preserved_leaf = temporary_leaf
                            try:
                                _rename_with_flags(
                                    parent_fd,
                                    temporary_leaf,
                                    parent_fd,
                                    selected.name,
                                    _RENAME_EXCHANGE,
                                )
                                committed = False
                            except OSError:
                                temporary_leaf = None
                                raise SyncPathPartialError(
                                    "replacement_rollback_failed",
                                    preserved_leaf,
                                ) from None
                            try:
                                os.fsync(parent_fd)
                            except OSError:
                                temporary_leaf = None
                                raise SyncPathPartialError(
                                    "replacement_rollback_unverified",
                                    preserved_leaf,
                                ) from None
                            temporary_leaf = None
                            raise SyncPathPartialError(
                                "replacement_raced_after_exchange",
                                preserved_leaf,
                            )
                        try:
                            os.fsync(parent_fd)
                        except OSError:
                            raise SyncPathError(
                                "replacement_commit_unverified",
                                selected,
                            ) from None
                    except BaseException as failure:
                        if isinstance(failure, SyncPathPartialError):
                            raise
                        try:
                            _rename_with_flags(
                                parent_fd,
                                temporary_leaf,
                                parent_fd,
                                selected.name,
                                _RENAME_EXCHANGE,
                            )
                            committed = False
                        except OSError:
                            preserved_leaf = temporary_leaf
                            temporary_leaf = None
                            raise SyncPathPartialError(
                                "replacement_rollback_failed",
                                preserved_leaf,
                            ) from None
                        try:
                            os.fsync(parent_fd)
                        except OSError:
                            preserved_leaf = temporary_leaf
                            temporary_leaf = None
                            raise SyncPathPartialError(
                                "replacement_rollback_unverified",
                                preserved_leaf,
                            ) from None
                        raise
                    else:
                        displaced_leaf = temporary_leaf
                        temporary_leaf = None
                        try:
                            os.unlink(displaced_leaf, dir_fd=parent_fd)
                        except OSError:
                            raise SyncPathPartialError(
                                "replacement_cleanup_pending",
                                displaced_leaf,
                                None
                                if displaced_snapshot is None
                                else displaced_snapshot.identity,
                            ) from None
            except SyncPathError:
                raise
            except OSError as exc:
                if committed:
                    raise SyncPathPartialError(
                        "replacement_commit_unverified",
                        selected.as_posix(),
                    ) from None
                if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise SyncPathError("target_identity_changed", selected) from None
                raise SyncPathError("guarded_rename_unavailable", selected) from None
            final = self._existing_target(parent_fd, selected.name, selected)
            if final is None or stat.S_IMODE(final.st_mode) != mode:
                raise SyncPathPartialError(
                    "replacement_postcondition_failed",
                    selected.as_posix(),
                )
            try:
                result = self._read_bytes(
                    parent_fd,
                    selected.name,
                    selected,
                    final,
                    max(len(content), 1),
                )
            except (OSError, SyncPathError):
                raise SyncPathPartialError(
                    "replacement_postcondition_failed",
                    selected.as_posix(),
                ) from None
            if result.content != content:
                raise SyncPathPartialError(
                    "replacement_postcondition_failed",
                    selected.as_posix(),
                )
            return result
        except SyncPathPartialError:
            raise
        except SyncPathError:
            if committed:
                raise SyncPathPartialError(
                    "replacement_postcondition_failed",
                    selected.as_posix(),
                ) from None
            raise
        except OSError:
            raise SyncPathError("operation_failed", selected) from None
        finally:
            if temporary_leaf is not None:
                try:
                    os.unlink(temporary_leaf, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            os.close(parent_fd)

    def move_file(
        self,
        source_path: Path | str,
        destination_path: Path | str,
        *,
        expected: SafeSyncBytes,
    ) -> SafeSyncBytes:
        """Rename one reviewed regular file within the pinned root."""

        source = self._validate_relative(source_path)
        destination = self._validate_relative(destination_path)
        if source == destination:
            raise SyncPathError("same_destination", destination)
        if expected.relative_path != source:
            raise SyncPathError("expected_path_mismatch", source)
        self._require_supported()
        self._verify_root_path_identity()
        try:
            source_parent = self._open_parent(source, create=False)
        except SyncPathError:
            raise
        except OSError:
            raise SyncPathError("operation_failed", source) from None
        try:
            destination_parent = self._open_parent(destination, create=True)
        except SyncPathError:
            os.close(source_parent)
            raise
        except OSError:
            os.close(source_parent)
            raise SyncPathError("operation_failed", destination) from None
        except BaseException:
            os.close(source_parent)
            raise
        committed = False
        try:
            source_entry = self._existing_target(source_parent, source.name, source)
            if not self._matches_reviewed_state(source_entry, expected):
                raise SyncPathError("target_identity_changed", source)
            if (
                self._existing_target(
                    destination_parent,
                    destination.name,
                    destination,
                )
                is not None
            ):
                raise SyncPathError("destination_exists", destination)
            self._verify_root_path_identity()
            self._verify_parent_identity(source, source_parent)
            self._verify_parent_identity(destination, destination_parent)
            source_entry = self._existing_target(source_parent, source.name, source)
            if not self._matches_reviewed_state(source_entry, expected):
                raise SyncPathError("target_identity_changed", source)
            if (
                self._existing_target(
                    destination_parent,
                    destination.name,
                    destination,
                )
                is not None
            ):
                raise SyncPathError("destination_exists", destination)
            self._before_commit(destination)
            try:
                _rename_with_flags(
                    source_parent,
                    source.name,
                    destination_parent,
                    destination.name,
                    _RENAME_NOREPLACE,
                )
                committed = True
            except OSError as exc:
                if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise SyncPathError("destination_exists", destination) from None
                raise SyncPathError("guarded_rename_unavailable", destination) from None
            try:
                os.fsync(source_parent)
                if not _same_identity(
                    os.fstat(source_parent),
                    os.fstat(destination_parent),
                ):
                    os.fsync(destination_parent)
            except OSError:
                raise SyncPathPartialError(
                    "move_commit_unverified",
                    destination.as_posix(),
                ) from None
            final = self._existing_target(
                destination_parent,
                destination.name,
                destination,
            )
            if (
                final is None
                or final.st_dev != expected.identity.device
                or final.st_ino != expected.identity.inode
                or final.st_nlink != 1
                or final.st_size != expected.size
                or stat.S_IMODE(final.st_mode) != expected.mode
            ):
                raise SyncPathPartialError(
                    "move_postcondition_failed",
                    destination.as_posix(),
                )
            try:
                moved = self._read_bytes(
                    destination_parent,
                    destination.name,
                    destination,
                    final,
                    max(expected.size, 1),
                )
            except (OSError, SyncPathError):
                raise SyncPathPartialError(
                    "move_postcondition_failed",
                    destination.as_posix(),
                ) from None
            if not self._matches_displaced_state(moved, expected):
                raise SyncPathPartialError(
                    "move_postcondition_failed",
                    destination.as_posix(),
                )
            return moved
        except SyncPathPartialError:
            raise
        except SyncPathError:
            if committed:
                raise SyncPathPartialError(
                    "move_postcondition_failed",
                    destination.as_posix(),
                ) from None
            raise
        except OSError:
            if committed:
                raise SyncPathPartialError(
                    "move_commit_unverified",
                    destination.as_posix(),
                ) from None
            raise SyncPathError("operation_failed", destination) from None
        finally:
            os.close(destination_parent)
            os.close(source_parent)
