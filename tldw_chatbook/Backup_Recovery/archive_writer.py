"""Verify and publish a new artifact from a completed private capture.

CaptureResult.inventory is the final ORIGINAL-source inventory, including actual
used recovery control roots as intentionally excluded authority items. Only an
explicit recovery.output exclusion permits publication inside a selected external
source. Payloads are staged under root at their strict manifest payload names.

The returned digest identifies published bytes (ciphertext for encrypted output).
Reader.verify_sealed instead consumes acquire's internal decrypted ZIP artifact.
"""

import hashlib
import json
import os
import shutil
import stat
import zipfile
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from uuid import uuid4

from . import archive_reader as reader
from .archive_models import SealedArchive
from .bootstrap import default_bootstrap_root
from .capture import CaptureResult
from .crypto import transform
from .limits import ArchiveLimits
from .native_files import (
    create_private_directory,
    create_private_file,
    pinned_directory,
    publish_new,
)

_BUFFER = 1024 * 1024
_TEXT = frozenset(
    {
        ".txt",
        ".md",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".csv",
        ".xml",
        ".html",
        ".py",
        ".js",
        ".css",
        ".sql",
    }
)


def _overlaps(path: Path, root: Path) -> bool:
    return path == root or root in path.parents or path in root.parents


def _output(capture: CaptureResult, destination: Path):
    if not destination.is_absolute() or ".." in destination.parts:
        raise ValueError("absolute_output_required")
    with pinned_directory(destination.parent) as parent:
        try:
            os.stat(destination.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("output_exists")
        identity = os.fstat(parent)
    protected = [capture.root, default_bootstrap_root()]
    protected.extend(
        item.path
        for item in capture.inventory.items
        if item.path is not None
        and item.owner.startswith("recovery.")
        and item.owner != "recovery.output"
    )
    if any(_overlaps(destination, path.resolve()) for path in protected):
        raise ValueError("output_overlap")
    excluded = [
        item.path
        for item in capture.inventory.items
        if item.owner == "recovery.output"
        and item.status == "intentionally_excluded"
        and item.path is not None
    ]
    for item in capture.inventory.items:
        if item.path is None or item.owner.startswith("recovery."):
            continue
        source = item.path.resolve()
        if _overlaps(destination, source):
            permitted = item.owner.startswith("external.") and any(
                root.resolve() in destination.parents for root in excluded
            )
            if not permitted:
                raise ValueError("output_overlap")
    return identity.st_dev, identity.st_ino


@contextmanager
def _payload(path: Path):
    with reader._regular(path) as stream:
        info = os.fstat(stream.fileno())
        if info.st_uid != os.geteuid() or info.st_nlink != 1 or info.st_mode & 0o077:
            raise ValueError("capture_not_private")
        yield stream


def _package(capture, doc, destination, cancel, limits, stored=()):
    """Stream only declared immutable payloads; never archive a directory walk."""
    with create_private_file(destination) as descriptor:
        with os.fdopen(os.dup(descriptor), "w+b") as output:
            with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
                info = zipfile.ZipInfo("manifest.json")
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                archive.writestr(info, capture.manifest_bytes)
                for payload in doc.files:
                    reader._check(cancel)
                    info = zipfile.ZipInfo(payload.payload)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    info.file_size = payload.size
                    info.compress_type = (
                        zipfile.ZIP_DEFLATED
                        if Path(payload.relative_path).suffix.lower() in _TEXT
                        and payload.payload not in stored
                        else zipfile.ZIP_STORED
                    )
                    path = capture.root / payload.payload
                    with (
                        _payload(path) as source,
                        archive.open(info, "w", force_zip64=True) as member,
                    ):
                        before = reader._identity(os.fstat(source.fileno()))
                        if before[2] != payload.size:
                            raise ValueError("capture_changed")
                        digest, count = hashlib.sha256(), 0
                        while chunk := source.read(_BUFFER):
                            reader._check(cancel)
                            count += len(chunk)
                            if count > payload.size:
                                raise ValueError("capture_changed")
                            reader._space(destination.parent, len(chunk))
                            member.write(chunk)
                            digest.update(chunk)
                            if output.tell() > limits.input_bytes:
                                raise ValueError("input_limit")
                        if (
                            count != payload.size
                            or digest.hexdigest() != payload.sha256
                            or before != reader._identity(os.fstat(source.fileno()))
                            or before
                            != reader._identity(path.stat(follow_symlinks=False))
                        ):
                            raise ValueError("capture_changed")
                high = {
                    item.filename
                    for item in archive.infolist()
                    if item.file_size > max(1, item.compress_size) * 100
                }
            output.flush()
            if output.tell() > limits.input_bytes:
                raise ValueError("input_limit")
        return high


def write_archive(
    capture: CaptureResult, destination: Path, *, password: bytes | None, cancel: Event
) -> SealedArchive:
    """Package, actually read/verify, then durably publish without replacement."""
    destination = Path(destination)
    parent_identity = _output(capture, destination)
    suffix = ".tldw-backup.zip" + (".age" if password is not None else "")
    if not destination.name.endswith(suffix):
        raise ValueError("invalid_output_suffix")
    reader._check(cancel)
    limits = ArchiveLimits()
    if len(capture.manifest_bytes) > limits.manifest_bytes:
        raise ValueError("manifest_limit")
    doc = reader._manifest(capture.manifest_bytes, limits, password is not None)
    if doc.consistency == "coherent" and not capture.inventory.complete:
        raise ValueError("incomplete_capture")
    expanded = len(capture.manifest_bytes) + sum(item.size for item in doc.files)
    if expanded > limits.expanded_bytes or any(
        item.size > limits.member_bytes for item in doc.files
    ):
        raise ValueError("expanded_limit")
    with pinned_directory(capture.root) as fd:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("capture_not_private")
    reader._space(destination.parent, expanded * (5 if password is not None else 3))
    operation = destination.parent / (".backup-write-" + uuid4().hex)
    create_private_directory(operation)
    try:
        output = operation / "archive.zip"
        high = _package(capture, doc, output, cancel, limits)
        if high:
            # Self-produced highly repetitive text needs no imported compression
            # exception: store it and remain within the reader's normal limits.
            replacement = operation / "stored.zip"
            _package(capture, doc, replacement, cancel, limits, stored=high)
            output.unlink()
            output = replacement
        if password is not None:
            encrypted = operation / "archive.age"
            transform(
                output,
                encrypted,
                password=password,
                decrypt=False,
                cancel=cancel,
                input_limit=limits.input_bytes,
                output_limit=limits.input_bytes,
                space_check=lambda count: reader._space(operation, count),
            )
            output = encrypted
        digest = reader._hash(output, cancel)
        verified = reader.acquire(
            output, operation / "verify", limits, password, cancel
        )
        if reader._hash(output, cancel) != digest:
            raise ValueError("output_changed")
        canonical = json.dumps(
            doc.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        if verified.manifest_bytes != canonical:
            raise ValueError("manifest_changed")
        if _output(capture, destination) != parent_identity:
            raise ValueError("output_parent_changed")
        reader._check(cancel)
        publish_new(output, destination)
        # A late cancellation never retracts or conceals durable publication.
        return SealedArchive(destination, digest, verified.manifest_bytes)
    finally:
        # The destination is never cleaned up, even if native rename succeeded
        # and its subsequent durability barrier reported an ambiguous failure.
        shutil.rmtree(operation)
