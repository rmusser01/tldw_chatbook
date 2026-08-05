"""Descriptor-anchored filesystem boundary for legacy Notes sync."""

from __future__ import annotations

import os
import stat
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


class SyncPathError(OSError):
    """Raised when a legacy sync path cannot be verified safely."""

    def __init__(self, reason: str, relative_path: Path | str = Path(".")):
        self.reason = reason
        self.relative_path = Path(relative_path)
        super().__init__(f"{reason}: {self.relative_path}")


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _is_reparse(entry_stat: os.stat_result) -> bool:
    attributes = getattr(entry_stat, "st_file_attributes", 0)
    return bool(attributes & _REPARSE_ATTRIBUTE)


class PinnedSyncRoot:
    """Pin a selected Notes root and contain all descendant operations."""

    def __init__(self, selected_root: Path | str):
        self.lexical_root = Path(selected_root)
        self.canonical_root = self.lexical_root.resolve(strict=True)
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
        parent_fd = self._open_parent(selected, create=False)
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
        finally:
            os.close(parent_fd)

    def _before_replace(self, _relative_path: Path) -> None:
        """Test seam immediately before identity rechecks and replacement."""

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

    def write_text(
        self,
        relative_path: Path | str,
        content: str,
    ) -> SafeSyncFile:
        """Atomically write beneath the pinned root, preserving prior mode."""

        selected = self._validate_relative(relative_path)
        self._require_supported()
        self._verify_root_path_identity()
        parent_fd = self._open_parent(selected, create=True)
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
