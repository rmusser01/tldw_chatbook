"""Bounded inspection of completed private copies, never live restore paths.

Acquired paths carry local byte identities. Consumers verify the sealed digest immediately
before using it; owner-only permissions are not a filesystem immutability promise.
"""

import hashlib
import json
import os
import shutil
import stat
import struct
import unicodedata
import uuid
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from threading import Event

from pydantic import ValidationError

from .archive_models import ArchiveManifest, EncryptedSource, SealedArchive
from .limits import ArchiveLimits
from .native_files import (
    create_private_directory,
    create_private_file,
    pinned_directory,
)

_BUFFER = 64 * 1024
_SPACE_MARGIN = 1024 * 1024


class CompressionReviewRequired(ValueError):
    """Declared expansion needs local review; retry requires the same digest."""

    def __init__(self, digest: str, expanded_bytes: int):
        super().__init__("compression_review_required")
        self.digest = digest
        self.expanded_bytes = expanded_bytes


def _check(cancel: Event) -> None:
    if cancel.is_set():
        raise InterruptedError("cancelled")


def _space(root: Path, required: int) -> None:
    if shutil.disk_usage(root).free < required + _SPACE_MARGIN:
        raise ValueError("insufficient_space")


@contextmanager
def _regular(path: Path):
    with pinned_directory(path.parent) as parent:
        fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("invalid_source")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                yield stream
        finally:
            os.close(fd)


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _hash(path: Path, cancel: Event) -> str:
    digest = hashlib.sha256()
    with _regular(path) as stream:
        before = _identity(os.fstat(stream.fileno()))
        while chunk := stream.read(_BUFFER):
            _check(cancel)
            digest.update(chunk)
        if before != _identity(os.fstat(stream.fileno())):
            raise ValueError("sealed_changed")
    return digest.hexdigest()


def _path(value: str, limits: ArchiveLimits, *, root=False) -> str:
    if root and value == "":
        return ""
    if (
        not value
        or len(value.encode("utf-8")) > limits.path_bytes
        or "\\" in value
        or ":" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or any(p in ("", ".", "..") or p.endswith((" ", ".")) for p in value.split("/"))
    ):
        raise ValueError("unsafe_path")
    return unicodedata.normalize("NFC", value).casefold()


def _unique(values):
    values = tuple(values)
    if len(set(values)) != len(values):
        raise ValueError("duplicate_identifier")


def _manifest(data: bytes, limits: ArchiveLimits, encrypted: bool) -> ArchiveManifest:
    def pairs(items):
        _unique(k for k, _ in items)
        return dict(items)

    # Reject duplicate JSON keys before strict schema parsing.
    json.loads(data, object_pairs_hook=pairs)
    doc = ArchiveManifest.model_validate_json(data)
    _unique(doc.profile_ids)
    _unique(o.owner_id for o in doc.owners)
    _unique(g.group_id for g in doc.dependency_groups)
    _unique(e.logical_id for e in doc.exclusions)
    _unique(r.logical_id for r in doc.relocations)
    records = (*doc.directories, *doc.files)
    _unique(r.logical_id for r in records)
    if len(records) > limits.members:
        raise ValueError("member_limit")
    directories = {d.logical_id: d for d in doc.directories}
    if any(
        directory.synthetic and directory.parent_id is not None
        for directory in doc.directories
    ):
        raise ValueError("invalid_synthetic_root")
    owners = {o.owner_id for o in doc.owners}
    names = set()
    for record in records:
        root = directories.get(record.root_id)
        if (
            root is None
            or root.parent_id is not None
            or root.root_id != root.logical_id
            or root.relative_path != ""
        ):
            raise ValueError("invalid_root")
        key = (record.root_id, _path(record.relative_path, limits, root=record is root))
        if key in names:
            raise ValueError("path_collision")
        names.add(key)
        if record is root:
            continue
        parent = directories.get(record.parent_id)
        if parent is None or parent.root_id != record.root_id:
            raise ValueError("invalid_parent")
        expected = str(PurePosixPath(record.relative_path).parent)
        if parent.relative_path != ("" if expected == "." else expected):
            raise ValueError("invalid_parent")
    for payload in doc.files:
        if payload.owner_id not in owners:
            raise ValueError("unknown_owner")
        _path(payload.payload, limits)
        if not payload.payload.startswith("payload/"):
            raise ValueError("invalid_payload_name")
    ids = {r.logical_id for r in records}
    grouped = []
    for group in doc.dependency_groups:
        _unique(group.members)
        if not set(group.members) <= ids:
            raise ValueError("invalid_dependency_group")
        if doc.consistency == "coherent" and not group.complete:
            raise ValueError("partial_group")
        grouped.extend(group.members)
    if not {p.logical_id for p in doc.files} <= set(grouped):
        raise ValueError("ungrouped_payload")
    if any(r.logical_id not in ids for r in doc.relocations):
        raise ValueError("invalid_relocation")
    if not encrypted and (doc.relocations or doc.credential_policy != "exclude"):
        raise ValueError("encryption_required")
    _producer_inventory(doc, limits)
    return doc


def _producer_inventory(doc: ArchiveManifest, limits: ArchiveLimits) -> None:
    """Validate explicit coverage without treating imported ownership as proof."""
    if not doc.producer_inventory:
        return  # Legacy archives do not gain inferred producer metadata.
    if len(doc.producer_inventory) > limits.members:
        raise ValueError("member_limit")
    _unique(item.logical_id for item in doc.producer_inventory)
    items = {item.logical_id: item for item in doc.producer_inventory}
    files = {item.logical_id: item for item in doc.files}
    directories = {item.logical_id: item for item in doc.directories}
    exclusions = {item.logical_id: item for item in doc.exclusions}
    owners = {owner.owner_id for owner in doc.owners}
    if set(items) != files.keys() | directories.keys() | exclusions.keys():
        raise ValueError("producer_coverage_mismatch")
    shared = {}
    for item in items.values():
        if doc.consistency == "coherent" and item.status in {
            "unsupported",
            "unavailable",
            "missing_required",
        }:
            raise ValueError("producer_coverage_incomplete")
        if item.owner_id not in owners:
            raise ValueError("unknown_producer_owner")
        _unique(item.dependencies)
        if not set(item.dependencies) <= items.keys():
            raise ValueError("unknown_producer_dependency")
        if item.logical_id in files:
            payload = files[item.logical_id]
            if item.status != "included" or item.owner_id != payload.owner_id:
                raise ValueError("producer_payload_mismatch")
        elif item.logical_id in directories:
            if item.status != "included_directory":
                raise ValueError("producer_directory_mismatch")
        elif (
            item.status in {"included", "included_directory"}
            or exclusions[item.logical_id].reason != item.status
        ):
            raise ValueError("producer_exclusion_mismatch")
        if item.shared_group and item.logical_id in files.keys() | directories.keys():
            shared.setdefault(item.shared_group, []).append(item)
    if exclusions.keys() & (files.keys() | directories.keys()):
        raise ValueError("producer_exclusion_mismatch")
    for group in shared.values():
        if len({item.status for item in group}) != 1:
            raise ValueError("shared_kind_mismatch")
        if (
            group[0].status == "included"
            and len(
                {
                    (files[item.logical_id].size, files[item.logical_id].sha256)
                    for item in group
                }
            )
            != 1
        ):
            raise ValueError("shared_payload_mismatch")


def _central_preflight(stream, limits):
    """Bound count/metadata before ZipFile allocates its central-directory list."""
    stream.seek(0, 2)
    size = stream.tell()
    if size < 22:
        raise ValueError("invalid_zip")
    stream.seek(size - 22)
    end = struct.unpack("<4s4H2IH", stream.read(22))
    if end[0] != b"PK\x05\x06" or end[1] or end[2] or end[3] != end[4] or end[7]:
        raise ValueError("invalid_zip_end")
    count, central_size, offset = end[4:7]
    end_offset = size - 22
    stream.seek(max(0, size - 42))
    has_zip64 = stream.read(4) == b"PK\x06\x07"
    if (
        has_zip64
        or count == 65535
        or central_size == 0xFFFFFFFF
        or offset == 0xFFFFFFFF
    ):
        stream.seek(size - 42)
        locator = struct.unpack("<4sIQI", stream.read(20))
        if locator[0] != b"PK\x06\x07" or locator[1] or locator[3] != 1:
            raise ValueError("invalid_zip64")
        stream.seek(locator[2])
        z64 = struct.unpack("<4sQ2H2I4Q", stream.read(56))
        if (
            z64[0] != b"PK\x06\x06"
            or z64[1] != 44
            or z64[4]
            or z64[5]
            or z64[6] != z64[7]
            or locator[2] + 56 != size - 42
        ):
            raise ValueError("invalid_zip64")
        count, central_size, offset = z64[7:10]
        end_offset = locator[2]
    if count > limits.members:
        raise ValueError("member_limit")
    if offset + central_size != end_offset:
        raise ValueError("invalid_zip_ranges")
    stream.seek(offset)
    for _ in range(count):
        header = stream.read(46)
        if len(header) != 46 or header[:4] != b"PK\x01\x02":
            raise ValueError("invalid_central_header")
        name_len, extra_len, comment_len = struct.unpack_from("<3H", header, 28)
        if (
            name_len > limits.path_bytes
            or extra_len > 32
            or comment_len
            or struct.unpack_from("<H", header, 34)[0]
        ):
            raise ValueError("header_limit")
        stream.seek(name_len + extra_len, 1)
        if stream.tell() > end_offset:
            raise ValueError("invalid_zip_ranges")
    if stream.tell() != end_offset:
        raise ValueError("invalid_zip_ranges")
    return offset


def _local_ranges(stream, infos, central):
    offsets = {}
    cursor = 0
    for info in sorted(infos, key=lambda i: i.header_offset):
        if info.header_offset != cursor:
            raise ValueError("overlapping_zip_structures")
        stream.seek(cursor)
        raw = stream.read(30)
        if len(raw) != 30 or raw[:4] != b"PK\x03\x04":
            raise ValueError("invalid_local_header")
        (
            _,
            version,
            flags,
            method,
            _,
            _,
            crc,
            compressed,
            expanded,
            name_len,
            extra_len,
        ) = struct.unpack("<4s5H3I2H", raw)
        if flags != info.flag_bits or method != info.compress_type or extra_len > 32:
            raise ValueError("header_mismatch")
        name = stream.read(name_len)
        if name != info.filename.encode("utf-8" if flags & 2048 else "cp437"):
            raise ValueError("header_mismatch")
        extra = stream.read(extra_len)
        if expanded == 0xFFFFFFFF or compressed == 0xFFFFFFFF:
            if len(extra) < 20 or struct.unpack_from("<HH", extra) != (1, 16):
                raise ValueError("invalid_zip64")
            expanded, compressed = struct.unpack_from("<QQ", extra, 4)
        if (crc, compressed, expanded) != (
            info.CRC,
            info.compress_size,
            info.file_size,
        ):
            raise ValueError("header_mismatch")
        offsets[info.filename] = cursor + 30 + name_len + extra_len
        cursor += 30 + name_len + extra_len + compressed
        if cursor > central:
            raise ValueError("overlapping_zip_structures")
    if cursor != central:
        raise ValueError("unexplained_zip_structure")
    return offsets


def _member_chunks(stream, info, offset, cancel):
    """Validate actual stored/deflated bytes without ZipExtFile size truncation.

    Decompression emits at most one bounded chunk, with one extra output byte
    permitted only to detect a false declared size before yielding that chunk.
    """
    stream.seek(offset)
    remaining = info.compress_size
    expanded = 0
    crc = 0
    decoder = zlib.decompressobj(-15) if info.compress_type == 8 else None
    while remaining:
        _check(cancel)
        compressed = stream.read(min(_BUFFER, remaining))
        if not compressed:
            raise ValueError("truncated_member")
        remaining -= len(compressed)
        pending = compressed
        while True:
            _check(cancel)
            limit = min(_BUFFER, info.file_size - expanded + 1)
            chunk = decoder.decompress(pending, limit) if decoder else pending
            expanded += len(chunk)
            if expanded > info.file_size:
                raise ValueError("member_size_mismatch")
            crc = zlib.crc32(chunk, crc)
            if chunk:
                yield chunk
            if decoder is None:
                break
            if decoder.eof:
                if decoder.unused_data or decoder.unconsumed_tail or remaining:
                    raise ValueError("trailing_compressed_bytes")
                break
            pending = decoder.unconsumed_tail
            if not pending and len(chunk) < limit:
                break
    if decoder is not None and not decoder.eof:
        raise ValueError("truncated_deflate_stream")
    if expanded != info.file_size or crc != info.CRC:
        raise ValueError("member_integrity_mismatch")


def _inspect(
    path: Path, limits: ArchiveLimits, encrypted: bool, cancel: Event, digest: str
) -> bytes:
    with _regular(path) as stream:
        central = _central_preflight(stream, limits)
        with zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            names = set()
            total = 0
            high_compression = False
            for info in infos:
                _check(cancel)
                key = _path(info.filename, limits)
                mode = info.external_attr >> 16
                if key in names:
                    raise ValueError("path_collision")
                names.add(key)
                if (
                    info.is_dir()
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                    or info.external_attr & 0x10
                    or info.compress_type not in (0, 8)
                    or info.flag_bits & ~2048
                ):
                    raise ValueError("unsupported_zip_member")
                if info.file_size > limits.member_bytes:
                    raise ValueError("member_limit")
                total += info.file_size
                if total > limits.expanded_bytes:
                    raise ValueError("expanded_limit")
                if info.file_size > max(1, info.compress_size) * 100:
                    high_compression = True
            if high_compression and (
                limits.reviewed_digest,
                limits.reviewed_expanded_bytes,
            ) != (digest, total):
                raise CompressionReviewRequired(digest, total)
            offsets = _local_ranges(stream, infos, central)
            if "manifest.json" not in archive.namelist():
                raise ValueError("manifest_missing")
            info = archive.getinfo("manifest.json")
            if info.file_size > limits.manifest_bytes:
                raise ValueError("manifest_limit")
            _space(path.parent, total)
            data = b"".join(
                _member_chunks(stream, info, offsets[info.filename], cancel)
            )
            doc = _manifest(data, limits, encrypted)
            if len(infos) + len(doc.directories) > limits.members:
                raise ValueError("member_limit")
            payloads = {p.payload: p for p in doc.files}
            if len(payloads) != len(doc.files) or set(archive.namelist()) != {
                "manifest.json",
                *payloads,
            }:
                raise ValueError("unexplained_entry")
            streamed = len(data)
            for name, payload in payloads.items():
                if archive.getinfo(name).file_size != payload.size:
                    raise ValueError("payload_size_mismatch")
                digest = hashlib.sha256()
                count = 0
                for chunk in _member_chunks(
                    stream, archive.getinfo(name), offsets[name], cancel
                ):
                    count += len(chunk)
                    streamed += len(chunk)
                    if count > limits.member_bytes or streamed > limits.expanded_bytes:
                        raise ValueError("expanded_limit")
                    _space(path.parent, total)
                    digest.update(chunk)
                if count != payload.size or digest.hexdigest() != payload.sha256:
                    raise ValueError("payload_digest_mismatch")
            return json.dumps(
                doc.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()


def acquire(
    source: Path,
    work_root: Path,
    limits: ArchiveLimits,
    password: bytes | None,
    cancel: Event,
) -> SealedArchive:
    """Copy, authenticate, and inspect into newly allocated private staging."""
    _check(cancel)
    operation = None
    completed = False
    try:
        with _regular(source) as incoming:
            before = _identity(os.fstat(incoming.fileno()))
            if before[2] > limits.input_bytes:
                raise ValueError("input_limit")
            prefix = incoming.read(20)
            incoming.seek(0)
            encrypted = prefix.startswith(b"age-encryption.org/")
            _space(
                work_root if work_root.exists() else work_root.parent,
                before[2]
                + (min(before[2], limits.decrypted_bytes) if encrypted else 0),
            )
            if not work_root.exists():
                create_private_directory(work_root)
            with pinned_directory(work_root) as fd:
                info = os.fstat(fd)
                if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                    raise ValueError("work_root_not_private")
            operation = work_root / uuid.uuid4().hex
            create_private_directory(operation)
            copied = operation / "input"
            total = 0
            with create_private_file(copied) as fd:
                while chunk := incoming.read(_BUFFER):
                    _check(cancel)
                    total += len(chunk)
                    if total > limits.input_bytes:
                        raise ValueError("input_limit")
                    _space(operation, len(chunk))
                    view = memoryview(chunk)
                    while view:
                        written = os.write(fd, view)
                        if not written:
                            raise OSError("write_failed")
                        view = view[written:]
            if before != _identity(os.fstat(incoming.fileno())) or before != _identity(
                source.stat(follow_symlinks=False)
            ):
                raise ValueError("source_changed")
        path = copied
        encrypted_digest = _hash(copied, cancel) if encrypted else None
        copied_identity = _identity(copied.stat(follow_symlinks=False))
        if encrypted:
            from .crypto import transform

            path = operation / "decrypted.zip"
            transform(
                copied,
                path,
                password=password,
                decrypt=True,
                cancel=cancel,
                input_limit=limits.input_bytes,
                output_limit=limits.decrypted_bytes,
                space_check=lambda count: _space(operation, count),
                expected_input_sha256=encrypted_digest,
            )
        digest = _hash(path, cancel)
        manifest_bytes = _inspect(path, limits, encrypted, cancel, digest)
        if digest != _hash(path, cancel):
            raise ValueError("sealed_changed")
        _check(cancel)
        provenance = None
        if encrypted:
            if (
                copied_identity != _identity(copied.stat(follow_symlinks=False))
                or _hash(copied, cancel) != encrypted_digest
            ):
                raise ValueError("encrypted_source_changed")
            provenance = EncryptedSource(
                copied,
                encrypted_digest,
                copied_identity,
                digest,
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
        sealed = SealedArchive(path, digest, manifest_bytes, provenance)
        completed = True
        return sealed
    except (
        zipfile.BadZipFile,
        UnicodeError,
        struct.error,
        RecursionError,
        ValidationError,
        json.JSONDecodeError,
        zlib.error,
    ):
        raise ValueError("invalid_archive") from None
    except InterruptedError:
        raise
    except OSError:
        raise ValueError("archive_io_failed") from None
    finally:
        # Successful return owns the operation; every incomplete stage is removed.
        if operation is not None and not completed:
            shutil.rmtree(operation)


def verify_sealed(
    sealed: SealedArchive, cancel: Event | None = None
) -> ArchiveManifest:
    """Invalidate a preview if its completed bytes changed; never reopen source."""
    if _hash(sealed.path, cancel or Event()) != sealed.digest:
        raise ValueError("sealed_changed")
    with _regular(sealed.path) as stream, zipfile.ZipFile(stream) as archive:
        info = archive.getinfo("manifest.json")
        if info.file_size > max(16 * 1024**2, len(sealed.manifest_bytes)):
            raise ValueError("sealed_changed")
        doc = ArchiveManifest.model_validate_json(archive.read(info))
    canonical = json.dumps(
        doc.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    if (
        canonical != sealed.manifest_bytes
        or _hash(sealed.path, cancel or Event()) != sealed.digest
    ):
        raise ValueError("sealed_changed")
    return doc


def retain_encrypted(
    sealed: SealedArchive, destination: Path, cancel: Event
) -> EncryptedSource:
    """Retain the exact ciphertext authenticated by acquisition, without a password.

    Metadata is local acquisition output, never inferred from a sibling filename.
    A writer receipt or plaintext acquisition carries no such binding.
    """
    source = sealed.encrypted_source
    if source is None:
        raise ValueError("encrypted_acquisition_required")
    verify_sealed(sealed, cancel)
    if (
        source.plaintext_digest != sealed.digest
        or source.manifest_digest != hashlib.sha256(sealed.manifest_bytes).hexdigest()
    ):
        raise ValueError("encrypted_source_changed")
    with (
        pinned_directory(source.path.parent) as parent,
        _regular(source.path) as incoming,
    ):
        info = os.fstat(incoming.fileno())

        def check():
            named = os.stat(source.path.name, dir_fd=parent, follow_symlinks=False)
            if (
                source.identity != _identity(os.fstat(incoming.fileno()))
                or source.identity != _identity(named)
                or named.st_nlink != 1
                or named.st_uid != os.geteuid()
                or named.st_mode & 0o077
            ):
                raise ValueError("encrypted_source_changed")

        check()
        _space(destination.parent, info.st_size)
        digest = hashlib.sha256()
        with create_private_file(destination) as fd:
            while chunk := incoming.read(_BUFFER):
                _check(cancel)
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    count = os.write(fd, view)
                    if not count:
                        raise OSError("write_failed")
                    view = view[count:]
            check()
            if digest.hexdigest() != source.digest:
                raise ValueError("encrypted_source_changed")
        verify_sealed(sealed, cancel)
        if _hash(destination, cancel) != source.digest:
            raise ValueError("retained_ciphertext_changed")
        return EncryptedSource(
            destination,
            source.digest,
            _identity(destination.stat(follow_symlinks=False)),
            sealed.digest,
            source.manifest_digest,
        )
