"""Side-effect-free admission and review of portable Tool Pack archives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import stat
import struct
import unicodedata
import zipfile

from tldw_chatbook.MCP.permission_store import PermissionStoreSnapshot
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventoryRegistry,
    PermissionInventorySnapshot,
    capture_v1_inventory,
)
from tldw_chatbook.Tool_Packs.contracts import (
    MAX_MANIFEST_BYTES,
    MAX_PROFILE_BYTES,
    PROFILE_PATH,
    TOOL_PACK_SCHEMA,
    PortableFallback,
    PortableToolRule,
    ToolPackDocument,
    ToolPackError,
    ToolPackManifest,
    canonical_json_bytes,
    strict_json_object,
)


_MAX_ARCHIVE_BYTES = 5 * 1024 * 1024
_MAX_MAPPINGS = 256
_READ_CHUNK = 64 * 1024
_DESTINATION_WORDS = re.compile(r"[^a-z0-9._-]+")
_EXPECTED_MEMBERS = ("tool-pack.json", PROFILE_PATH)


@dataclass(frozen=True, slots=True)
class ServerMapping:
    """One explicit external MCP source-to-destination server mapping."""

    source_server_key: str
    destination_server_key: str


@dataclass(frozen=True, slots=True)
class MappedToolRule:
    """One source rule matched to an exact destination contract."""

    source_rule: PortableToolRule
    destination_identity: tuple[str, str, str]
    destination_contract_sha256: str
    destination_connected: bool

    @property
    def state(self) -> str:
        return self.source_rule.state

    @property
    def authority(self) -> str:
        return self.destination_identity[0]

    @property
    def server_key(self) -> str:
        return self.destination_identity[1]

    @property
    def tool_name(self) -> str:
        return self.destination_identity[2]

    @property
    def rule(self) -> PortableToolRule:
        """Return the immutable portable source rule."""
        return self.source_rule


@dataclass(frozen=True, slots=True)
class ToolPackImportReview:
    """Immutable, process-local evidence for a later import commit."""

    archive_path: Path
    archive_sha256: str
    manifest_sha256: str
    payload_sha256: str
    destination_id: str
    store_generation: str
    inventory_digest: str
    mappings: tuple[ServerMapping, ...]
    expires_at: datetime
    matched: tuple[MappedToolRule, ...]
    changed: tuple[PortableToolRule, ...]
    missing: tuple[PortableToolRule, ...]
    pending_denies: tuple[PortableToolRule, ...]
    omitted_allow_ask: tuple[PortableToolRule, ...]
    content_digest: str = ""
    display_name: str = ""
    producer: tuple[str, str] = ()
    fallbacks: tuple[PortableFallback, ...] = ()


class ToolPackImportService:
    """Inspect a pack without mutation.

    The reference dependency is deliberately read-only. A callback must report
    active, archived, and dangling profile ids; a service receives
    ``include_archived=True`` explicitly.
    """

    def __init__(
        self,
        permission_store: object,
        inventory: PermissionInventoryRegistry,
        reference_checker: object,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(reference_checker) and not callable(
            getattr(reference_checker, "references_profile", None)
        ):
            raise ToolPackError("import", "destination_referenced")
        self._permission_store = permission_store
        self._inventory = inventory
        self._reference_checker = reference_checker
        self._now = now or (lambda: datetime.now(timezone.utc))

    def inspect_archive(
        self,
        archive_path: Path,
        *,
        destination_id: str,
        mappings: Sequence[ServerMapping] = (),
    ) -> ToolPackImportReview:
        """Admit an archive and return an inert, expiring review.

        Args:
            archive_path: Exact ``.tldw-tool-pack`` file selected by the user.
            destination_id: Proposed unbound profile id; normalized without suffixes.
            mappings: Explicit one-to-one external MCP server mappings.

        Returns:
            Deeply immutable review evidence valid for exactly fifteen minutes.

        Raises:
            ToolPackError: If archive, store, inventory, reference, or mapping
                admission fails. Errors contain no filesystem paths.
        """
        path = Path(archive_path)
        if path.suffix != ".tldw-tool-pack":
            raise ToolPackError("import", "archive_invalid")
        archive_bytes = _read_regular_archive(path)
        document, manifest_bytes, payload_bytes = _read_document(archive_bytes)
        normalized_id = _normalize_destination_id(destination_id)

        try:
            store = self._permission_store.read_snapshot_strict()
        except Exception:
            raise ToolPackError("import", "store_invalid") from None
        if (
            type(store) is not PermissionStoreSnapshot
            or not isinstance(store.payload, Mapping)
            or type(store.generation) is not str
            or not store.generation
        ):
            raise ToolPackError("import", "store_invalid")
        profiles = store.payload.get("profiles")
        if not isinstance(profiles, Mapping):
            raise ToolPackError("import", "store_invalid")
        folded = normalized_id.casefold()
        if any(type(key) is not str for key in profiles):
            raise ToolPackError("import", "store_invalid")
        if any(key.casefold() == folded for key in profiles):
            raise ToolPackError("import", "destination_referenced")
        try:
            referenced = self._references_profile(normalized_id)
        except Exception:
            raise ToolPackError("import", "destination_referenced") from None
        if type(referenced) is not bool or referenced:
            raise ToolPackError("import", "destination_referenced")

        try:
            inventory = capture_v1_inventory(self._inventory)
        except Exception:
            raise ToolPackError("import", "inventory_invalid") from None
        if type(inventory) is not PermissionInventorySnapshot:
            raise ToolPackError("import", "inventory_invalid")

        try:
            if len(mappings) > _MAX_MAPPINGS:
                raise ToolPackError("import", "too_large")
            mapping_tuple = tuple(mappings)
        except ToolPackError:
            raise
        except Exception:
            raise ToolPackError("import", "mapping_invalid") from None
        mapped_fallbacks, mapped, changed, missing, pending, omitted = _classify(
            document, inventory, mapping_tuple
        )
        now = self._now()
        if type(now) is not datetime or now.tzinfo is None:
            raise ToolPackError("import", "review_stale")
        return ToolPackImportReview(
            archive_path=path,
            archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
            content_digest=document.manifest.content_digest,
            display_name=document.manifest.display_name,
            producer=(
                document.manifest.producer_name,
                document.manifest.producer_version,
            ),
            fallbacks=mapped_fallbacks,
            destination_id=normalized_id,
            store_generation=store.generation,
            inventory_digest=inventory.digest,
            mappings=mapping_tuple,
            expires_at=now + timedelta(minutes=15),
            matched=mapped,
            changed=changed,
            missing=missing,
            pending_denies=pending,
            omitted_allow_ask=omitted,
        )

    def _references_profile(self, profile_id: str) -> bool:
        if callable(self._reference_checker):
            return self._reference_checker(profile_id)  # type: ignore[no-any-return]
        return self._reference_checker.references_profile(  # type: ignore[union-attr,no-any-return]
            profile_id, include_archived=True
        )


def _normalize_destination_id(value: object) -> str:
    if type(value) is not str:
        raise ToolPackError("import", "mapping_invalid")
    normalized = unicodedata.normalize("NFC", value).casefold().strip()
    normalized = _DESTINATION_WORDS.sub("-", normalized).strip(".-_")
    if (
        not normalized
        or normalized == "default"
        or normalized.startswith("ws-")
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", normalized) is None
    ):
        raise ToolPackError("import", "mapping_invalid")
    return normalized


def _read_regular_archive(path: Path) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if type(no_follow) is not int:
        raise ToolPackError("import", "archive_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    try:
        descriptor = os.open(path, flags)
    except (OSError, TypeError, ValueError):
        raise ToolPackError("import", "archive_invalid") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_ARCHIVE_BYTES:
            category = (
                "too_large"
                if before.st_size > _MAX_ARCHIVE_BYTES
                else "archive_invalid"
            )
            raise ToolPackError("import", category)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor, min(_READ_CHUNK, _MAX_ARCHIVE_BYTES + 1 - total)
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_ARCHIVE_BYTES:
                raise ToolPackError("import", "too_large")
        after = os.fstat(descriptor)
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError:
            raise ToolPackError("import", "archive_invalid") from None

        def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if identity(before) != identity(after) or identity(after) != identity(current):
            raise ToolPackError("import", "archive_invalid")
        return b"".join(chunks)
    except ToolPackError:
        raise
    except MemoryError:
        raise ToolPackError("import", "too_large") from None
    except OSError:
        raise ToolPackError("import", "archive_invalid") from None
    finally:
        os.close(descriptor)


def _read_document(archive_bytes: bytes) -> tuple[ToolPackDocument, bytes, bytes]:
    _validate_zip_envelope(archive_bytes)
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes), mode="r") as archive:
            infos = archive.infolist()
            if (
                archive.comment
                or tuple(info.filename for info in infos) != _EXPECTED_MEMBERS
            ):
                raise ToolPackError("import", "archive_invalid")
            if any(not _canonical_member(info) for info in infos):
                raise ToolPackError("import", "archive_invalid")
            if (
                infos[0].file_size > MAX_MANIFEST_BYTES
                or infos[1].file_size > MAX_PROFILE_BYTES
            ):
                raise ToolPackError("import", "too_large")
            manifest_bytes = archive.read(infos[0])
            manifest_raw = strict_json_object(
                manifest_bytes,
                category="manifest_invalid",
                max_bytes=MAX_MANIFEST_BYTES,
            )
            if canonical_json_bytes(manifest_raw) != manifest_bytes:
                raise ToolPackError("import", "manifest_invalid")
            manifest = ToolPackManifest.from_dict(manifest_raw)
            payload_bytes = archive.read(infos[1])
            if (
                len(payload_bytes) != manifest.payload_size
                or hashlib.sha256(payload_bytes).hexdigest() != manifest.payload_sha256
            ):
                raise ToolPackError("import", "manifest_invalid")
            content_preimage = (
                TOOL_PACK_SCHEMA.encode("ascii")
                + b"\0"
                + canonical_json_bytes(manifest.to_dict(include_content_digest=False))
                + b"\0"
                + payload_bytes
            )
            if hashlib.sha256(content_preimage).hexdigest() != manifest.content_digest:
                raise ToolPackError("import", "manifest_invalid")
            payload_raw = strict_json_object(
                payload_bytes,
                category="payload_invalid",
                max_bytes=MAX_PROFILE_BYTES,
            )
            document = ToolPackDocument.from_dicts(
                manifest_raw, payload_raw, profile_bytes=payload_bytes
            )
            return document, manifest_bytes, payload_bytes
    except ToolPackError:
        raise
    except MemoryError:
        raise ToolPackError("import", "too_large") from None
    except (EOFError, OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        raise ToolPackError("import", "archive_invalid") from None


def _canonical_member(info: zipfile.ZipInfo) -> bool:
    return (
        info.date_time == (1980, 1, 1, 0, 0, 0)
        and info.compress_type == zipfile.ZIP_STORED
        and info.create_system == 3
        and info.create_version == 20
        and info.extract_version == 20
        and info.flag_bits == 0
        and info.external_attr == 0o100644 << 16
        and info.internal_attr == 0
        and getattr(info, "volume", 0) == 0
        and info.extra == b""
        and info.comment == b""
        and not info.is_dir()
    )


def _validate_zip_envelope(raw: bytes) -> None:
    """Validate the exact two-member, descriptor-free canonical ZIP encoding."""
    try:
        if len(raw) < 22:
            raise ValueError
        eocd_offset = len(raw) - 22
        (
            signature,
            disk,
            central_disk,
            disk_count,
            total_count,
            central_size,
            central_offset,
            comment_size,
        ) = struct.unpack_from("<4s4H2LH", raw, eocd_offset)
        if (
            signature != b"PK\x05\x06"
            or disk != 0
            or central_disk != 0
            or disk_count != 2
            or total_count != 2
            or comment_size != 0
            or central_offset + central_size != eocd_offset
        ):
            raise ValueError

        local_records: list[tuple[int, int, int, int]] = []
        cursor = 0
        for expected_name in _EXPECTED_MEMBERS:
            (
                local_signature,
                extract_version,
                flags,
                compression,
                dos_time,
                dos_date,
                crc,
                compressed_size,
                file_size,
                name_size,
                extra_size,
            ) = struct.unpack_from("<4s5H3L2H", raw, cursor)
            name_start = cursor + 30
            name_end = name_start + name_size
            data_start = name_end + extra_size
            data_end = data_start + compressed_size
            if (
                local_signature != b"PK\x03\x04"
                or extract_version != 20
                or flags != 0
                or compression != 0
                or dos_time != 0
                or dos_date != 33
                or raw[name_start:name_end] != expected_name.encode("ascii")
                or extra_size != 0
                or compressed_size != file_size
                or data_end > central_offset
            ):
                raise ValueError
            limit = (
                MAX_MANIFEST_BYTES
                if expected_name == "tool-pack.json"
                else MAX_PROFILE_BYTES
            )
            if file_size > limit:
                raise ToolPackError("import", "too_large")
            local_records.append((cursor, crc, compressed_size, file_size))
            cursor = data_end
        if cursor != central_offset:
            raise ValueError

        cursor = central_offset
        for expected_name, local in zip(_EXPECTED_MEMBERS, local_records, strict=True):
            (
                central_signature,
                create_version,
                extract_version,
                flags,
                compression,
                dos_time,
                dos_date,
                crc,
                compressed_size,
                file_size,
                name_size,
                extra_size,
                member_comment_size,
                disk_start,
                internal_attr,
                external_attr,
                local_offset,
            ) = struct.unpack_from("<4s6H3L5H2L", raw, cursor)
            name_start = cursor + 46
            name_end = name_start + name_size
            cursor = name_end + extra_size + member_comment_size
            if (
                central_signature != b"PK\x01\x02"
                or create_version != (3 << 8) | 20
                or extract_version != 20
                or flags != 0
                or compression != 0
                or dos_time != 0
                or dos_date != 33
                or raw[name_start:name_end] != expected_name.encode("ascii")
                or extra_size != 0
                or member_comment_size != 0
                or disk_start != 0
                or internal_attr != 0
                or external_attr != 0o100644 << 16
                or (local_offset, crc, compressed_size, file_size) != local
            ):
                raise ValueError
        if cursor != eocd_offset:
            raise ValueError
    except ToolPackError:
        raise
    except (IndexError, TypeError, ValueError, struct.error):
        raise ToolPackError("import", "archive_invalid") from None


def _classify(
    document: ToolPackDocument,
    inventory: PermissionInventorySnapshot,
    mappings: tuple[ServerMapping, ...],
) -> tuple[
    tuple[PortableFallback, ...],
    tuple[MappedToolRule, ...],
    tuple[PortableToolRule, ...],
    tuple[PortableToolRule, ...],
    tuple[PortableToolRule, ...],
    tuple[PortableToolRule, ...],
]:
    source_servers = {
        (rule.authority, rule.server_key) for rule in document.profile.tools
    } | {(item.authority, item.server_key) for item in document.profile.fallbacks}
    destination_servers = set(inventory.namespaces)
    mapping_by_source: dict[str, str] = {}
    source_folded: set[str] = set()
    destination_folded: set[str] = set()
    for mapping in mappings:
        if type(mapping) is not ServerMapping:
            raise ToolPackError("import", "mapping_invalid")
        source = mapping.source_server_key
        destination_server = mapping.destination_server_key
        if (
            not _external_mcp_server(source)
            or not _external_mcp_server(destination_server)
            or ("mcp", source) not in source_servers
            or ("mcp", destination_server) not in destination_servers
            or source == destination_server
            or source.casefold() in source_folded
            or destination_server.casefold() in destination_folded
        ):
            raise ToolPackError("import", "mapping_invalid")
        source_folded.add(source.casefold())
        destination_folded.add(destination_server.casefold())
        mapping_by_source[source] = destination_server

    resulting: set[tuple[str, str, str]] = set()
    resulting_folded: set[tuple[str, str, str]] = set()
    for rule in document.profile.tools:
        server_key = mapping_by_source.get(rule.server_key, rule.server_key)
        identity = (rule.authority, server_key, rule.tool_name)
        folded_identity = tuple(part.casefold() for part in identity)
        if identity in resulting or folded_identity in resulting_folded:
            raise ToolPackError("import", "identity_duplicate")
        resulting.add(identity)
        resulting_folded.add(folded_identity)

    fallback_resulting: set[tuple[str, str]] = set()
    fallback_folded: set[tuple[str, str]] = set()
    mapped_fallbacks: list[PortableFallback] = []
    for fallback in document.profile.fallbacks:
        server_key = mapping_by_source.get(fallback.server_key, fallback.server_key)
        identity = (fallback.authority, server_key)
        folded_identity = tuple(part.casefold() for part in identity)
        if identity in fallback_resulting or folded_identity in fallback_folded:
            raise ToolPackError("import", "identity_duplicate")
        fallback_resulting.add(identity)
        fallback_folded.add(folded_identity)
        mapped_fallbacks.append(
            fallback
            if server_key == fallback.server_key
            else PortableFallback(fallback.authority, server_key, fallback.state)
        )

    destination = {item.identity: item for item in inventory.tools}
    matched: list[MappedToolRule] = []
    changed: list[PortableToolRule] = []
    missing: list[PortableToolRule] = []
    pending: list[PortableToolRule] = []
    omitted: list[PortableToolRule] = []
    for rule in document.profile.tools:
        destination_server = mapping_by_source.get(rule.server_key, rule.server_key)
        identity = (rule.authority, destination_server, rule.tool_name)
        item = destination.get(identity)
        if item is not None and rule.contract_sha256 == item.contract_sha256:
            matched.append(
                MappedToolRule(
                    rule,
                    identity,
                    item.contract_sha256,
                    bool(item.tool.executable),
                )
            )
            continue
        mapped_rule = (
            rule
            if destination_server == rule.server_key
            else PortableToolRule(
                rule.authority,
                destination_server,
                rule.tool_name,
                rule.state,
                rule.contract_sha256,
            )
        )
        (changed if item is not None else missing).append(mapped_rule)
        (pending if rule.state == "deny" else omitted).append(mapped_rule)
    return (
        tuple(
            sorted(mapped_fallbacks, key=lambda item: (item.authority, item.server_key))
        ),
        tuple(matched),
        tuple(changed),
        tuple(missing),
        tuple(pending),
        tuple(omitted),
    )


def _external_mcp_server(value: object) -> bool:
    if (
        type(value) is not str
        or not value
        or unicodedata.normalize("NFC", value) != value
    ):
        return False
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        return False
    folded = value.casefold()
    return (
        len(encoded) <= 512
        and folded
        not in {"*", "agent:builtin", "local:__local__", "local:__virtual_cli__"}
        and not folded.startswith("builtin:")
    )


__all__ = [
    "MappedToolRule",
    "ServerMapping",
    "ToolPackImportReview",
    "ToolPackImportService",
]
