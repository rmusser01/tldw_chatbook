"""POSIX-only, captured-destination publication for Tool Pack archives."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Tool_Packs.export import ToolPackExportSnapshot, write_tool_pack_archive


@dataclass(frozen=True, slots=True)
class ToolPackPublicationPrimitives:
    """The POSIX primitives required to safely publish an archive."""

    posix: bool
    nofollow: bool
    directory: bool
    directory_descriptors: bool
    nofollow_stat: bool

    @classmethod
    def current(cls) -> "ToolPackPublicationPrimitives":
        return cls(
            posix=os.name == "posix",
            nofollow=getattr(os, "O_NOFOLLOW", 0) > 0,
            directory=getattr(os, "O_DIRECTORY", 0) > 0,
            directory_descriptors=all(
                operation in os.supports_dir_fd
                for operation in (os.open, os.stat, os.unlink, os.rename)
            ),
            nofollow_stat=os.stat in os.supports_follow_symlinks,
        )

    def supported(self) -> bool:
        return all(
            (
                self.posix,
                self.nofollow,
                self.directory,
                self.directory_descriptors,
                self.nofollow_stat,
            )
        )


@dataclass(frozen=True, slots=True)
class CapturedToolPackDestination:
    """One picker-confirmed path and its exact parent/target identities."""

    path: Path = field(repr=False)
    parent_identity: tuple[int, int]
    target_identity: tuple[int, int] | None
    _overwrite_token: str | None = field(repr=False, compare=False)
    _target_digest: str | None = field(repr=False, compare=False)

    @classmethod
    def capture(cls, path: Path) -> "CapturedToolPackDestination":
        """Capture an absent or regular `.tldw-tool-pack` destination."""
        try:
            if (
                not isinstance(path, Path)
                or not path.is_absolute()
                or path.name in {"", ".", ".."}
                or path.suffix.lower() != ".tldw-tool-pack"
            ):
                raise ValueError
            parent = path.parent
            if parent.resolve(strict=True) != parent:
                raise ValueError
            parent_stat = os.lstat(parent)
            if not stat.S_ISDIR(parent_stat.st_mode):
                raise ValueError
            try:
                target = os.lstat(path)
            except FileNotFoundError:
                target_identity = None
                target_digest = None
                token = None
            else:
                if not stat.S_ISREG(target.st_mode):
                    raise ValueError
                target_identity = _identity(target)
                target_digest = _digest_path(path, target_identity)
                token = secrets.token_urlsafe(32)
            return cls(
                path=path,
                parent_identity=(parent_stat.st_dev, parent_stat.st_ino),
                target_identity=target_identity,
                _overwrite_token=token,
                _target_digest=target_digest,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            raise ToolPackError("export", "destination_invalid") from None

    @property
    def overwrite_token(self) -> str | None:
        """Return the opaque confirmation value for this exact existing target."""
        return self._overwrite_token


@dataclass(frozen=True, slots=True)
class ToolPackPublicationResult:
    """The publication outcome without destination details."""

    archive_sha256: str
    committed: bool
    durability_uncertain: bool


def publish_tool_pack(
    snapshot: ToolPackExportSnapshot,
    destination: CapturedToolPackDestination,
    *,
    overwrite: bool = False,
    overwrite_token: str | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    phase_hook: Callable[[str], None] | None = None,
    primitives: ToolPackPublicationPrimitives | None = None,
) -> ToolPackPublicationResult:
    """Atomically publish a deterministic archive to one captured destination."""
    if (
        type(snapshot) is not ToolPackExportSnapshot
        or type(destination) is not CapturedToolPackDestination
        or type(overwrite) is not bool
        or not callable(cancelled)
        or (phase_hook is not None and not callable(phase_hook))
        or (primitives is not None and type(primitives) is not ToolPackPublicationPrimitives)
    ):
        raise ToolPackError("export", "publication_failed")
    active_primitives = primitives or ToolPackPublicationPrimitives.current()
    if not active_primitives.supported():
        raise ToolPackError("export", "publication_unsupported")
    if destination.target_identity is not None and (
        not overwrite or overwrite_token != destination.overwrite_token
    ):
        raise ToolPackError("export", "destination_changed")

    parent_fd = -1
    temporary_name: str | None = None
    temporary_identity: tuple[int, int] | None = None
    archive_sha256 = ""
    replacement_may_have_occurred = False
    try:
        parent_fd = _open_parent(destination)
        temporary_name = f".{destination.path.name}.{secrets.token_hex(16)}.tmp"
        temporary_fd = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            with os.fdopen(temporary_fd, "w+b", closefd=False) as sink:
                archive_sha256 = write_tool_pack_archive(snapshot, sink)
                sink.flush()
                os.fsync(temporary_fd)
        finally:
            temporary_identity = _identity(os.fstat(temporary_fd))
            os.close(temporary_fd)
        if phase_hook is not None:
            phase_hook("archive_fsynced")
        _validate_parent_and_target(parent_fd, destination)
        _validate_owned_temporary(parent_fd, temporary_name, temporary_identity)
        try:
            is_cancelled = cancelled()
        except Exception:
            is_cancelled = True
        if is_cancelled is not False:
            raise ToolPackError("export", "cancelled")
        if phase_hook is not None:
            phase_hook("before_replace")
        _validate_parent_and_target(parent_fd, destination)
        _validate_owned_temporary(parent_fd, temporary_name, temporary_identity)
        replacement_may_have_occurred = True
        os.replace(
            temporary_name,
            destination.path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
        return ToolPackPublicationResult(archive_sha256, True, False)
    except ToolPackError:
        raise
    except Exception:
        if replacement_may_have_occurred:
            return _reconcile_after_replace(
                parent_fd, destination, archive_sha256, temporary_identity
            )
        raise ToolPackError("export", "publication_failed") from None
    finally:
        try:
            if (
                temporary_name is not None
                and temporary_identity is not None
                and parent_fd >= 0
            ):
                _remove_owned_temporary(parent_fd, temporary_name, temporary_identity)
        finally:
            if parent_fd >= 0:
                try:
                    os.close(parent_fd)
                except OSError:
                    pass


def _open_parent(destination: CapturedToolPackDestination) -> int:
    parent_fd = -1
    try:
        parent_fd = os.open(
            destination.path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        opened = os.fstat(parent_fd)
        named = os.lstat(destination.path.parent)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _identity(opened) != destination.parent_identity
            or _identity(named) != destination.parent_identity
        ):
            raise ValueError
        return parent_fd
    except (OSError, TypeError, ValueError):
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        raise ToolPackError("export", "destination_changed") from None


def _validate_parent_and_target(
    parent_fd: int, destination: CapturedToolPackDestination
) -> None:
    try:
        named_parent = os.lstat(destination.path.parent)
        opened_parent = os.fstat(parent_fd)
        if (
            _identity(named_parent) != destination.parent_identity
            or _identity(opened_parent) != destination.parent_identity
        ):
            raise ValueError
        try:
            target = os.stat(destination.path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            current_identity = None
        else:
            if not stat.S_ISREG(target.st_mode):
                raise ValueError
            current_identity = _identity(target)
        if current_identity != destination.target_identity:
            raise ValueError
    except (OSError, TypeError, ValueError):
        raise ToolPackError("export", "destination_changed") from None


def _validate_owned_temporary(
    parent_fd: int, name: str, identity: tuple[int, int]
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or _identity(current) != identity:
            raise ValueError
    except (OSError, TypeError, ValueError):
        raise ToolPackError("export", "publication_failed") from None


def _remove_owned_temporary(parent_fd: int, name: str, identity: tuple[int, int]) -> None:
    try:
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(current.st_mode) or _identity(current) != identity:
            raise ValueError
        os.unlink(name, dir_fd=parent_fd)
    except (TypeError, ValueError):
        raise ToolPackError("export", "publication_failed") from None
    except OSError:
        raise ToolPackError("export", "publication_failed") from None


def _reconcile_after_replace(
    parent_fd: int,
    destination: CapturedToolPackDestination,
    archive_sha256: str,
    replacement_identity: tuple[int, int] | None,
) -> ToolPackPublicationResult:
    try:
        named_parent = os.lstat(destination.path.parent)
        opened_parent = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(named_parent.st_mode)
            or _identity(named_parent) != destination.parent_identity
            or _identity(opened_parent) != destination.parent_identity
        ):
            raise ValueError
        current = os.stat(
            destination.path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        current_identity = None
        current_digest = None
    except (OSError, TypeError, ValueError):
        raise ToolPackError("export", "durability_uncertain") from None
    else:
        if not stat.S_ISREG(current.st_mode):
            raise ToolPackError("export", "durability_uncertain") from None
        current_identity = _identity(current)
        try:
            current_digest = _digest_descriptor_relative(
                parent_fd, destination.path.name, current_identity
            )
        except (OSError, TypeError, ValueError):
            raise ToolPackError("export", "durability_uncertain") from None
    if (
        replacement_identity is not None
        and current_identity == replacement_identity
        and current_digest == archive_sha256
    ):
        return ToolPackPublicationResult(archive_sha256, True, True)
    if (
        current_identity == destination.target_identity
        and current_digest == destination._target_digest
    ):
        raise ToolPackError("export", "publication_failed")
    raise ToolPackError("export", "durability_uncertain")


def _digest_path(path: Path, expected_identity: tuple[int, int]) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or _identity(metadata) != expected_identity:
            raise ValueError
        digest = hashlib.sha256()
        while block := os.read(descriptor, 64 * 1024):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _digest_descriptor_relative(
    parent_fd: int, name: str, expected_identity: tuple[int, int]
) -> str:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or _identity(metadata) != expected_identity:
            raise ValueError
        digest = hashlib.sha256()
        while block := os.read(descriptor, 64 * 1024):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


__all__ = [
    "CapturedToolPackDestination",
    "ToolPackPublicationPrimitives",
    "ToolPackPublicationResult",
    "publish_tool_pack",
]
