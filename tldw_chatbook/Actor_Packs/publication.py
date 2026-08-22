"""Pinned same-directory publication for deterministic Actor Pack archives."""

from __future__ import annotations

import errno
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .export import (
    ActorPackExportResult,
    ActorPackExportSnapshot,
    write_actor_pack_archive,
)


class ActorPackPublicationError(ValueError):
    """One stable, path-free publication failure."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ActorPackDestinationContract:
    """Exact user-confirmed destination identity, hidden from public repr."""

    parent_identity: tuple[int, int]
    existing_identity: tuple[int, ...] | None
    destination: Path = field(repr=False)


def capture_actor_pack_destination(destination: Path) -> ActorPackDestinationContract:
    """Capture an absent or exact regular destination without following links."""

    try:
        if (
            not isinstance(destination, Path)
            or not destination.is_absolute()
            or destination.name in {"", ".", ".."}
            or destination.suffix.lower() != ".tldw-actor-pack"
        ):
            raise ValueError
        parent = destination.parent
        if parent.resolve(strict=True) != parent:
            raise ValueError
        parent_stat = os.lstat(parent)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError
        try:
            existing = os.lstat(destination)
        except FileNotFoundError:
            existing_identity = None
        else:
            if not stat.S_ISREG(existing.st_mode):
                raise ValueError
            existing_identity = _stat_identity(existing)
        return ActorPackDestinationContract(
            parent_identity=(parent_stat.st_dev, parent_stat.st_ino),
            existing_identity=existing_identity,
            destination=destination,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ActorPackPublicationError(
            "actor_pack_export_destination_invalid"
        ) from None


def publish_actor_pack(
    snapshot: ActorPackExportSnapshot,
    destination: ActorPackDestinationContract,
    *,
    authority_guard: Callable[[], bool] = lambda: True,
    cancelled: Callable[[], bool] = lambda: False,
    phase_hook: Callable[[str], None] | None = None,
) -> ActorPackExportResult:
    """Write, attest, and atomically replace one exact destination."""

    if (
        type(destination) is not ActorPackDestinationContract
        or not callable(authority_guard)
        or not callable(cancelled)
        or (phase_hook is not None and not callable(phase_hook))
    ):
        raise ActorPackPublicationError("actor_pack_export_publication_invalid")
    if not _secure_publication_supported():
        raise ActorPackPublicationError("actor_pack_export_publication_unsupported")

    parent_fd = -1
    temporary_name: str | None = None
    temporary_identity: tuple[int, ...] | None = None
    committed = False
    try:
        parent_fd = _open_parent(destination)
        temporary_name = f".{destination.destination.name}.{secrets.token_hex(16)}.tmp"
        temporary_fd = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            with os.fdopen(temporary_fd, "w+b", closefd=False) as stream:
                archive_sha256 = write_actor_pack_archive(snapshot, stream)
                stream.flush()
                os.fsync(temporary_fd)
        finally:
            temporary_identity = _stat_identity(os.fstat(temporary_fd))
            os.close(temporary_fd)
        if phase_hook is not None:
            phase_hook("archive_fsynced")
        _validate_parent_and_destination(parent_fd, destination)
        _validate_owned_temporary(parent_fd, temporary_name, temporary_identity)
        try:
            authority_current = authority_guard()
        except Exception:
            authority_current = False
        if authority_current is not True:
            raise ActorPackPublicationError("actor_pack_export_authority_changed")
        try:
            is_cancelled = cancelled()
        except Exception:
            is_cancelled = True
        if is_cancelled is not False:
            raise ActorPackPublicationError("actor_pack_export_cancelled")
        _validate_parent_and_destination(parent_fd, destination)
        _validate_owned_temporary(parent_fd, temporary_name, temporary_identity)
        os.replace(
            temporary_name,
            destination.destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        committed = True
        temporary_name = None
        try:
            durability = _fsync_parent(parent_fd)
        except OSError:
            durability = "actor_pack_export_durability_uncertain"
        return ActorPackExportResult(
            archive_sha256=archive_sha256,
            committed=True,
            durability=durability,
        )
    except ActorPackPublicationError:
        raise
    except Exception:
        if committed:
            raise ActorPackPublicationError(
                "actor_pack_export_durability_uncertain"
            ) from None
        raise ActorPackPublicationError(
            "actor_pack_export_publication_failed"
        ) from None
    finally:
        try:
            if (
                not committed
                and parent_fd >= 0
                and temporary_name is not None
                and temporary_identity is not None
            ):
                _remove_owned_temporary(parent_fd, temporary_name, temporary_identity)
        finally:
            if parent_fd >= 0:
                os.close(parent_fd)


def _secure_publication_supported() -> bool:
    return (
        os.name == "posix"
        and getattr(os, "O_NOFOLLOW", 0) > 0
        and getattr(os, "O_DIRECTORY", 0) > 0
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and os.rename in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
    )


def _open_parent(contract: ActorPackDestinationContract) -> int:
    try:
        parent_fd = os.open(
            contract.destination.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        opened = os.fstat(parent_fd)
        named = os.lstat(contract.destination.parent)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != contract.parent_identity
            or (named.st_dev, named.st_ino) != contract.parent_identity
        ):
            raise ValueError
        return parent_fd
    except (OSError, TypeError, ValueError):
        raise ActorPackPublicationError(
            "actor_pack_export_destination_changed"
        ) from None


def _validate_parent_and_destination(
    parent_fd: int, contract: ActorPackDestinationContract
) -> None:
    try:
        named_parent = os.lstat(contract.destination.parent)
        opened_parent = os.fstat(parent_fd)
        if (named_parent.st_dev, named_parent.st_ino) != contract.parent_identity or (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ) != contract.parent_identity:
            raise ValueError
        try:
            current = os.stat(
                contract.destination.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            current_identity = None
        else:
            current_identity = _stat_identity(current)
        if current_identity != contract.existing_identity:
            raise ValueError
    except (OSError, TypeError, ValueError):
        raise ActorPackPublicationError(
            "actor_pack_export_destination_changed"
        ) from None


def _validate_owned_temporary(
    parent_fd: int, name: str, identity: tuple[int, ...]
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or _stat_identity(current) != identity:
            raise ValueError
    except (OSError, TypeError, ValueError):
        raise ActorPackPublicationError("actor_pack_export_temporary_changed") from None


def _remove_owned_temporary(
    parent_fd: int, name: str, identity: tuple[int, ...]
) -> None:
    try:
        _validate_owned_temporary(parent_fd, name, identity)
        os.unlink(name, dir_fd=parent_fd)
    except ActorPackPublicationError:
        raise ActorPackPublicationError("actor_pack_export_cleanup_ambiguous") from None
    except OSError:
        raise ActorPackPublicationError("actor_pack_export_cleanup_failed") from None


def _fsync_parent(parent_fd: int) -> str:
    try:
        os.fsync(parent_fd)
        return "durable"
    except OSError as exc:
        unsupported = {errno.EINVAL, errno.ENOTSUP}
        if hasattr(errno, "EOPNOTSUPP"):
            unsupported.add(errno.EOPNOTSUPP)
        if exc.errno in unsupported:
            return "unsupported"
        raise


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
