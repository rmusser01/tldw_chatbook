"""Bounded package capture and descriptor-anchored snapshot materialization.

Inspection can retain safe siblings after a path error. Acquisition cannot:
materialization rejects any incomplete capture before creating its destination.
No package file is executed, and links are never copied as links.
"""

import hashlib
import json
import math
import os
import re
import shutil
import stat
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from tldw_chatbook.Utils.path_validation import validate_canonical_directory

from .models import Diagnostic, PackageInspection

MAX_DOCUMENT_BYTES = 256 * 1024
MAX_JSON_DEPTH = 32
MAX_PACKAGE_BYTES = 100 * 1024 * 1024
MAX_FILES = 10_000
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PATH_DEPTH = 32
CHUNK_BYTES = 64 * 1024
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.IGNORECASE)


class PackageFileError(ValueError):
    """A safe, metadata-only package failure code."""


def validate_relative_member(value: str) -> PurePosixPath:
    """Reject traversal and path spellings that cannot travel across platforms."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise PackageFileError("package_path_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PackageFileError("package_path_invalid")
    for part in path.parts:
        if (
            part.endswith((" ", "."))
            or any(ord(c) < 32 for c in part)
            or any(c in '<>:"|?*' for c in part)
            or _RESERVED.match(part)
        ):
            raise PackageFileError("package_path_invalid")
    if len(path.parts) > MAX_PATH_DEPTH:
        raise PackageFileError("package_depth_limit")
    return path


def collision_key(value: str) -> str:
    """Destination-independent conservative collision identity."""
    return unicodedata.normalize("NFC", value).casefold()


def canonical_json(value: object) -> str:
    """Serialize validated JSON deterministically without whitespace variance."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PackageFileError("json_duplicate_key")
        result[key] = value
    return result


def parse_document(data: bytes) -> dict:
    """Parse bounded JSON with duplicate-key and depth rejection."""
    if len(data) > MAX_DOCUMENT_BYTES:
        raise PackageFileError("document_bytes_limit")
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                PackageFileError("json_nonfinite")
            ),
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, PackageFileError):
            raise
        raise PackageFileError("document_invalid") from None
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise PackageFileError("document_depth_limit")
        if isinstance(node, dict):
            stack.extend((child, depth + 1) for child in node.values())
            stack.extend((key, depth + 1) for key in node)
        elif isinstance(node, list):
            stack.extend((child, depth + 1) for child in node)
        elif isinstance(node, str):
            try:
                node.encode("utf-8")
            except UnicodeEncodeError:
                raise PackageFileError("json_unicode_invalid") from None
        elif isinstance(node, float) and not math.isfinite(node):
            raise PackageFileError("json_nonfinite")
    if not isinstance(value, dict):
        raise PackageFileError("document_object_required")
    return value


@dataclass(frozen=True)
class CapturedFile:
    data: bytes
    executable_mode: int


@dataclass
class PackageCapture:
    root: Path
    files: dict[str, CapturedFile]
    directories: set[str]
    diagnostics: list[Diagnostic]
    errors: list[Diagnostic]
    digest: str | None = None
    source_digest: str | None = None
    link_targets: dict[str, str] = field(default_factory=dict)

    def read(self, value: str) -> bytes:
        key = validate_relative_member(value).as_posix()
        if key not in self.files:
            raise PackageFileError("package_file_unavailable")
        return self.files[key].data

    def document(self, value: str) -> dict:
        return parse_document(self.read(value))


def _identity(info):
    return info.st_dev, info.st_ino


def _unchanged(info):
    return (
        _identity(info),
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_mode,
    )


def _check_reparse(info):
    if getattr(info, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
    ):
        raise PackageFileError("package_reparse_unsupported")


def _require_descriptor_support():
    if (
        os.name != "posix"
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
    ):
        raise PackageFileError("platform_capture_unqualified")


@contextmanager
def _directory_fd(path: Path):
    """Pin every ancestor without following links, including the final directory."""
    _require_descriptor_support()
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _member_fd(root_fd: int, member: PurePosixPath):
    descriptor = os.dup(root_fd)
    try:
        for part in member.parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        file_fd = os.open(
            member.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
        )
        try:
            yield file_fd
        finally:
            os.close(file_fd)
    finally:
        os.close(descriptor)


def capture_package(root: Path) -> PackageCapture:
    """Capture bounded regular files; report unsafe members without reading them.

    Raises:
        PackageFileError: For root failures, exhaustion, or concurrent changes.
    """
    try:
        root = validate_canonical_directory(root)
        capture = PackageCapture(root, {}, set(), [], [])
        total = 0
        count = 0
        directory_count = 0
        seen = set()
        with _directory_fd(root) as root_fd:
            root_identity = _identity(os.fstat(root_fd))

            def account_entry(is_directory: bool):
                nonlocal count, directory_count
                if is_directory:
                    directory_count += 1
                    if directory_count > MAX_FILES * MAX_PATH_DEPTH:
                        raise PackageFileError("package_directory_count_limit")
                else:
                    # An entry with unreadable metadata consumes the file budget.
                    count += 1
                    if count > MAX_FILES:
                        raise PackageFileError("package_file_count_limit")

            def retain_diagnostic(diagnostic: Diagnostic, *, error: bool = False):
                if len(capture.errors) + len(capture.diagnostics) >= MAX_FILES:
                    raise PackageFileError("package_diagnostics_limit")
                (capture.errors if error else capture.diagnostics).append(diagnostic)

            def walk(directory_fd: int, prefix: PurePosixPath):
                nonlocal total
                before = os.fstat(directory_fd)
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        member = prefix / entry.name
                        path = member.as_posix()
                        try:
                            try:
                                info = entry.stat(follow_symlinks=False)
                            except OSError:
                                account_entry(False)
                                raise
                            account_entry(stat.S_ISDIR(info.st_mode))
                            validate_relative_member(path)
                            key = collision_key(path)
                            if key in seen:
                                raise PackageFileError("package_path_collision")
                            seen.add(key)
                            _check_reparse(info)
                            if stat.S_ISDIR(info.st_mode):
                                child_fd = os.open(
                                    entry.name,
                                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=directory_fd,
                                )
                                try:
                                    if _identity(os.fstat(child_fd)) != _identity(info):
                                        raise PackageFileError("package_changed")
                                    capture.directories.add(path)
                                    walk(child_fd, member)
                                finally:
                                    os.close(child_fd)
                                continue
                            target = member
                            if stat.S_ISLNK(info.st_mode):
                                resolved = (root / path).resolve(strict=True)
                                if not resolved.is_relative_to(root):
                                    raise PackageFileError("package_path_escape")
                                target = validate_relative_member(
                                    resolved.relative_to(root).as_posix()
                                )
                            elif not stat.S_ISREG(info.st_mode):
                                raise PackageFileError("package_special_file")
                            with _member_fd(root_fd, target) as file_fd:
                                start = os.fstat(file_fd)
                                _check_reparse(start)
                                if not stat.S_ISREG(start.st_mode):
                                    raise PackageFileError("package_link_target_type")
                                if start.st_size > MAX_FILE_BYTES:
                                    raise PackageFileError("package_file_bytes_limit")
                                data = bytearray()
                                while block := os.read(
                                    file_fd,
                                    min(
                                        CHUNK_BYTES,
                                        MAX_FILE_BYTES - len(data) + 1,
                                        MAX_PACKAGE_BYTES - total + 1,
                                    ),
                                ):
                                    total += len(block)
                                    data.extend(block)
                                    if len(data) > MAX_FILE_BYTES:
                                        raise PackageFileError(
                                            "package_file_bytes_limit"
                                        )
                                    if total > MAX_PACKAGE_BYTES:
                                        raise PackageFileError("package_bytes_limit")
                                if _unchanged(start) != _unchanged(os.fstat(file_fd)):
                                    raise PackageFileError("package_changed")
                            if _unchanged(info) != _unchanged(
                                os.stat(
                                    entry.name,
                                    dir_fd=directory_fd,
                                    follow_symlinks=False,
                                )
                            ):
                                raise PackageFileError("package_changed")
                            capture.files[path] = CapturedFile(
                                bytes(data), stat.S_IMODE(start.st_mode) & 0o111
                            )
                            if stat.S_ISLNK(info.st_mode):
                                capture.link_targets[path] = target.as_posix()
                                retain_diagnostic(
                                    Diagnostic(
                                        code="internal_link_materialized", path=path
                                    )
                                )
                        except (OSError, ValueError, RuntimeError) as exc:
                            code = (
                                str(exc)
                                if isinstance(exc, PackageFileError)
                                else "package_file_unavailable"
                            )
                            if "limit" in code or code == "package_changed":
                                raise PackageFileError(code) from None
                            retain_diagnostic(
                                Diagnostic(code=code, path=path), error=True
                            )
                if _unchanged(before) != _unchanged(os.fstat(directory_fd)):
                    raise PackageFileError("package_changed")

            walk(root_fd, PurePosixPath())
            if _identity(root.stat()) != root_identity:
                raise PackageFileError("package_changed")
        if not capture.errors:
            digest = hashlib.sha256()
            for name, file in sorted(capture.files.items()):
                header = canonical_json(
                    [name, file.executable_mode, len(file.data)]
                ).encode()
                digest.update(len(header).to_bytes(8, "big"))
                digest.update(header)
                digest.update(file.data)
            capture.digest = digest.hexdigest()
            capture.source_digest = hashlib.sha256(
                canonical_json(
                    {"content": capture.digest, "links": capture.link_targets}
                ).encode()
            ).hexdigest()
        return capture
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, PackageFileError):
            raise
        raise PackageFileError("package_root_invalid") from None


@contextmanager
def _output_parent(root_fd: int, path: str):
    """Pin an output member's parent without following substituted directories."""
    member = validate_relative_member(path)
    descriptor = os.dup(root_fd)
    try:
        for part in member.parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        yield descriptor, member.name
    finally:
        os.close(descriptor)


def materialize_package(
    source: Path, destination: Path, *, dialect: str | None = None
) -> PackageInspection:
    """Validate and copy into an absent child of a canonical directory.

    Existing destinations are never replaced. POSIX descriptor support is
    required; other platforms fail closed pending qualification. Content and
    effective digests bind the copied files; source_digest also binds link
    provenance, which is retained after internal links become regular files.
    """
    from .inspection import inspect_capture

    capture = capture_package(source)
    if capture.errors:
        raise PackageFileError(capture.errors[0].code)
    inspection = inspect_capture(capture, dialect=dialect)
    if inspection.rejected:
        raise PackageFileError("package_inspection_rejected")
    try:
        parent = validate_canonical_directory(destination.parent)
        validate_relative_member(destination.name)
        if destination.is_relative_to(capture.root) or capture.root.is_relative_to(
            destination
        ):
            raise PackageFileError("destination_overlaps_source")
        with _directory_fd(parent) as parent_fd:
            parent_identity = _identity(os.fstat(parent_fd))
            try:
                os.mkdir(destination.name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                raise PackageFileError("destination_exists") from None
            destination_fd = os.open(
                destination.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            destination_id = _identity(os.fstat(destination_fd))
            try:
                for directory in sorted(
                    capture.directories, key=lambda value: (value.count("/"), value)
                ):
                    with _output_parent(destination_fd, directory) as (fd, name):
                        os.mkdir(name, mode=0o700, dir_fd=fd)
                for path, file in capture.files.items():
                    with _output_parent(destination_fd, path) as (fd, name):
                        output = os.open(
                            name,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o600,
                            dir_fd=fd,
                        )
                        with os.fdopen(output, "wb") as stream:
                            stream.write(file.data)
                            stream.flush()
                            os.fchmod(stream.fileno(), 0o400 | file.executable_mode)
                if (
                    _identity(
                        os.stat(
                            destination.name, dir_fd=parent_fd, follow_symlinks=False
                        )
                    )
                    != destination_id
                    or _identity(parent.stat()) != parent_identity
                ):
                    raise PackageFileError("destination_changed")
                copied = capture_package(destination)
                if copied.digest != capture.digest or copied.errors:
                    raise PackageFileError("destination_changed")
                return inspection.model_copy(
                    update={"materialized_identity": str(destination)}
                )
            except BaseException:
                # rmtree uses descriptor-relative, symlink-resistant deletion;
                # only remove the entry still owned by this operation.
                try:
                    if (
                        _identity(
                            os.stat(
                                destination.name,
                                dir_fd=parent_fd,
                                follow_symlinks=False,
                            )
                        )
                        == destination_id
                    ):
                        shutil.rmtree(destination.name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                raise
            finally:
                os.close(destination_fd)
    except OSError:
        raise PackageFileError("destination_unavailable") from None
