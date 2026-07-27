"""Verified local file access and atomic result publication."""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
import stat
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping

from pydantic import BaseModel

from scripts.stt_eval.schema import canonical_json


DEFAULT_CHUNK_SIZE = 1024 * 1024


def _descriptor_flags() -> int:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
    ):
        raise RuntimeError(
            "platform lacks descriptor-anchored no-follow file operations"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _lexical_absolute(path: Path) -> Path:
    path = Path(path)
    if ".." in path.parts:
        raise ValueError(f"path traversal is not allowed: {path}")
    return path if path.is_absolute() else Path.cwd() / path


def _raise_unsafe_component(path: Path, error: OSError) -> None:
    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
        raise ValueError(
            f"path traverses a symlink or non-directory component: {path}"
        ) from error
    raise error


def _open_directory_no_symlinks(path: Path) -> int:
    absolute = _lexical_absolute(path)
    flags = _descriptor_flags()
    descriptor = os.open(absolute.anchor, flags)
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                _raise_unsafe_component(current, error)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_file_no_symlinks(path: Path) -> int:
    absolute = _lexical_absolute(path)
    if absolute.name in {"", ".", ".."}:
        raise ValueError(f"artifact path must name a regular file: {path}")

    parent_descriptor = _open_directory_no_symlinks(absolute.parent)
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW
        try:
            descriptor = os.open(
                absolute.name,
                flags,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            _raise_unsafe_component(absolute, error)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise ValueError(f"artifact path is not a regular file: {path}")
        return descriptor
    finally:
        os.close(parent_descriptor)


def _contained_relative_parts(relative_path: str | Path) -> tuple[str, ...]:
    raw = os.fspath(relative_path)
    if not raw or raw == "." or "\\" in raw:
        raise ValueError("path must be a contained relative path")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise ValueError("path must be a contained relative path")
    return posix.parts


def _open_regular_file_under_root(
    root_descriptor: int,
    relative_path: str | Path,
) -> int:
    parts = _contained_relative_parts(relative_path)
    current_descriptor = os.dup(root_descriptor)
    try:
        for part in parts[:-1]:
            try:
                child = os.open(
                    part,
                    _descriptor_flags(),
                    dir_fd=current_descriptor,
                )
            except OSError as error:
                _raise_unsafe_component(Path(relative_path), error)
            os.close(current_descriptor)
            current_descriptor = child

        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=current_descriptor,
            )
        except OSError as error:
            _raise_unsafe_component(Path(relative_path), error)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise ValueError(f"artifact path is not a regular file: {relative_path}")
        return descriptor
    finally:
        os.close(current_descriptor)


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("expected SHA-256 must be 64 lowercase hex characters")


def stream_file_identity(
    path: Path,
    *,
    root: Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[int, str]:
    """Return size/hash, anchoring a relative path under ``root`` when given."""

    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")

    path = Path(path)
    try:
        if root is None:
            descriptor = _open_regular_file_no_symlinks(path)
        else:
            root_descriptor = _open_directory_no_symlinks(root)
            try:
                descriptor = _open_regular_file_under_root(
                    root_descriptor,
                    path,
                )
            finally:
                os.close(root_descriptor)
    except FileNotFoundError:
        raise FileNotFoundError(f"artifact file does not exist: {path}") from None

    try:
        digest = hashlib.sha256()
        byte_count = 0
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with stream:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                byte_count += len(chunk)
                digest.update(chunk)
        return byte_count, digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_file(
    path: Path,
    *,
    root: Path | None = None,
    expected_size: int,
    expected_sha256: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Verify a file, anchoring a relative path under ``root`` when given."""

    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise ValueError("expected size must be a positive integer")
    _validate_sha256(expected_sha256)
    actual_size, actual_sha256 = stream_file_identity(
        path,
        root=root,
        chunk_size=chunk_size,
    )
    if actual_size != expected_size:
        raise ValueError(
            f"artifact size mismatch: expected {expected_size}, got {actual_size}"
        )
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def resolve_contained_path(root: Path, relative_path: str | Path) -> Path:
    """Check a contained path; use ``verify_file(root=...)`` before reading."""

    parts = _contained_relative_parts(relative_path)

    root = Path(root)
    try:
        root_descriptor = _open_directory_no_symlinks(root)
    except FileNotFoundError:
        raise ValueError(f"declared root does not exist: {root}") from None

    try:
        current_descriptor = root_descriptor
        root_descriptor = -1
        for index, part in enumerate(parts):
            is_last = index == len(parts) - 1
            if is_last:
                try:
                    metadata = os.stat(
                        part,
                        dir_fd=current_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    break
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError(
                        f"contained path traverses a symlink: {relative_path}"
                    )
                break
            try:
                child = os.open(
                    part,
                    _descriptor_flags(),
                    dir_fd=current_descriptor,
                )
            except FileNotFoundError:
                break
            except OSError as error:
                _raise_unsafe_component(Path(relative_path), error)
            os.close(current_descriptor)
            current_descriptor = child
    finally:
        if root_descriptor >= 0:
            os.close(root_descriptor)
        if "current_descriptor" in locals():
            os.close(current_descriptor)

    absolute_root = _lexical_absolute(root)
    candidate = absolute_root.joinpath(*parts)
    return candidate


def _create_staging_file(
    parent_descriptor: int,
    destination_name: str,
) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    for _ in range(100):
        staging_name = f".{destination_name}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                staging_name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            continue
        return descriptor, staging_name
    raise FileExistsError("unable to allocate a unique staging filename")


def _atomic_write(destination: Path, chunks: Iterable[bytes]) -> None:
    if os.rename not in os.supports_dir_fd or os.unlink not in os.supports_dir_fd:
        raise RuntimeError(
            "platform lacks descriptor-anchored atomic publication operations"
        )
    destination = _lexical_absolute(Path(destination))
    parent_descriptor = _open_directory_no_symlinks(destination.parent)
    descriptor = -1
    staging_name: str | None = None
    try:
        descriptor, staging_name = _create_staging_file(
            parent_descriptor,
            destination.name,
        )
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            for chunk in chunks:
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            staging_name,
            destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        staging_name = None
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if staging_name is not None:
            try:
                os.unlink(staging_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def atomic_write_json(
    destination: Path,
    value: BaseModel | Mapping[str, object],
) -> None:
    """Atomically publish one canonical JSON document."""

    _atomic_write(destination, (canonical_json(value),))


def atomic_write_jsonl(
    destination: Path,
    records: Iterable[BaseModel | Mapping[str, object]],
) -> None:
    """Atomically publish canonical JSON Lines records."""

    def encoded_records() -> Iterable[bytes]:
        for record in records:
            yield canonical_json(record)
            yield b"\n"

    _atomic_write(destination, encoded_records())


__all__ = [
    "atomic_write_json",
    "atomic_write_jsonl",
    "resolve_contained_path",
    "stream_file_identity",
    "verify_file",
]
