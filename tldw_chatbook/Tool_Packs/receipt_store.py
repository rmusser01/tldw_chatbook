"""Private, bounded evidence receipts for portable Tool Pack lifecycle work."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import threading
from typing import AbstractSet
import unicodedata

from tldw_chatbook.Tool_Packs.contracts import (
    ToolPackError,
    canonical_json_bytes,
    strict_json_object,
)


RECEIPT_SCHEMA = "tldw.tool-pack-receipt/v1"
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_RECEIPT_STORE_BYTES = 32 * 1024 * 1024
MAX_RECONCILE_ENTRIES = 4096
ORPHAN_GRACE = timedelta(hours=24)

_RECEIPT_ID_RE = re.compile(r"tp-[0-9a-f]{32}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_IMPORT_KEYS = frozenset(
    {
        "schema",
        "kind",
        "profile_id",
        "pack_digest",
        "archive_digest",
        "producer",
        "imported_at",
        "reviewed_mappings",
        "matched",
        "changed",
        "missing",
        "pending_deny",
        "omitted",
    }
)
_TOMBSTONE_KEYS = frozenset(
    {
        "schema",
        "kind",
        "profile_id",
        "pack_digest",
        "removed_at",
        "prior_receipt_digest",
    }
)
_PRODUCER_KEYS = frozenset({"name", "version"})
_MAPPING_KEYS = frozenset({"source_server_key", "destination_server_key"})
_IDENTITY_KEYS = frozenset({"authority", "server_key", "tool_name"})


def _fail(category: str) -> ToolPackError:
    return ToolPackError("import", category)


def _exact_dict(value: object, keys: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise _fail("payload_invalid")
    return value


def _text(value: object, *, max_bytes: int = 512) -> str:
    if (
        type(value) is not str
        or not value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise _fail("payload_invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise _fail("payload_invalid") from None
    if len(encoded) > max_bytes:
        raise _fail("payload_invalid")
    return value


def _digest(value: object) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise _fail("payload_invalid")
    return value


def _profile_id(value: object) -> str:
    identifier = _text(value, max_bytes=128)
    if (
        not _PROFILE_ID_RE.fullmatch(identifier)
        or identifier == "default"
        or identifier.startswith("ws-")
    ):
        raise _fail("payload_invalid")
    return identifier


def _timestamp(value: object) -> str:
    text = _text(value, max_bytes=64)
    if not text.endswith("Z"):
        raise _fail("payload_invalid")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        raise _fail("payload_invalid") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise _fail("payload_invalid")
    return text


def _identity(raw: object) -> tuple[str, str, str]:
    value = _exact_dict(raw, _IDENTITY_KEYS)
    authority = value["authority"]
    if type(authority) is not str or authority not in {"mcp", "builtin"}:
        raise _fail("payload_invalid")
    server_key = _text(value["server_key"])
    tool_name = _text(value["tool_name"])
    if (authority == "builtin") != (server_key == "agent:builtin"):
        raise _fail("payload_invalid")
    return authority, server_key, tool_name


def _identity_list(raw: object) -> tuple[tuple[str, str, str], ...]:
    if type(raw) is not list:
        raise _fail("payload_invalid")
    values = tuple(_identity(item) for item in raw)
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise _fail("payload_invalid")
    return values


def _mapping_list(raw: object) -> tuple[tuple[str, str], ...]:
    if type(raw) is not list or len(raw) > 256:
        raise _fail("payload_invalid")
    values: list[tuple[str, str]] = []
    for item in raw:
        value = _exact_dict(item, _MAPPING_KEYS)
        values.append(
            (
                _text(value["source_server_key"]),
                _text(value["destination_server_key"]),
            )
        )
    result = tuple(values)
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise _fail("payload_invalid")
    return result


@dataclass(frozen=True, slots=True)
class ToolPackReceipt:
    """One immutable privacy-safe import or compact-tombstone receipt."""

    schema: str
    kind: str
    profile_id: str
    pack_digest: str
    archive_digest: str | None = None
    producer: tuple[str, str] | None = None
    imported_at: str | None = None
    reviewed_mappings: tuple[tuple[str, str], ...] = ()
    matched: tuple[tuple[str, str, str], ...] = ()
    changed: tuple[tuple[str, str, str], ...] = ()
    missing: tuple[tuple[str, str, str], ...] = ()
    pending_deny: tuple[tuple[str, str, str], ...] = ()
    omitted: tuple[tuple[str, str, str], ...] = ()
    removed_at: str | None = None
    prior_receipt_digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise _fail("payload_invalid")
        _profile_id(self.profile_id)
        _digest(self.pack_digest)
        if self.kind == "import":
            if (
                self.archive_digest is None
                or self.producer is None
                or self.imported_at is None
                or self.removed_at is not None
                or self.prior_receipt_digest is not None
                or type(self.reviewed_mappings) is not tuple
            ):
                raise _fail("payload_invalid")
            _digest(self.archive_digest)
            if type(self.producer) is not tuple or len(self.producer) != 2:
                raise _fail("payload_invalid")
            _text(self.producer[0], max_bytes=128)
            _text(self.producer[1], max_bytes=128)
            _timestamp(self.imported_at)
            if len(self.reviewed_mappings) > 256:
                raise _fail("too_large")
            checked_mappings: list[tuple[str, str]] = []
            for mapping in self.reviewed_mappings:
                if type(mapping) is not tuple or len(mapping) != 2:
                    raise _fail("payload_invalid")
                checked_mappings.append((_text(mapping[0]), _text(mapping[1])))
            if tuple(checked_mappings) != tuple(sorted(checked_mappings)) or len(
                checked_mappings
            ) != len(set(checked_mappings)):
                raise _fail("payload_invalid")
            groups = {
                "matched": self.matched,
                "changed": self.changed,
                "missing": self.missing,
                "pending_deny": self.pending_deny,
                "omitted": self.omitted,
            }
            memberships: dict[tuple[str, str, str], set[str]] = {}
            folded_identities: dict[tuple[str, str, str], tuple[str, str, str]] = {}
            for group_name, group in groups.items():
                if (
                    type(group) is not tuple
                    or group != tuple(sorted(group))
                    or len(group) != len(set(group))
                ):
                    raise _fail("payload_invalid")
                for identity in group:
                    if type(identity) is not tuple or len(identity) != 3:
                        raise _fail("payload_invalid")
                    _identity(
                        {
                            "authority": identity[0],
                            "server_key": identity[1],
                            "tool_name": identity[2],
                        }
                    )
                    folded = tuple(part.casefold() for part in identity)
                    prior = folded_identities.get(folded)
                    if prior is not None and prior != identity:
                        raise _fail("payload_invalid")
                    folded_identities[folded] = identity
                    memberships.setdefault(identity, set()).add(group_name)
            if len(memberships) > 2_000:
                raise _fail("too_large")
            diagnostics = {"changed", "missing"}
            actions = {"pending_deny", "omitted"}
            for groups_for_identity in memberships.values():
                if len(groups_for_identity) == 1:
                    continue
                if not (
                    len(groups_for_identity) == 2
                    and len(groups_for_identity & diagnostics) == 1
                    and len(groups_for_identity & actions) == 1
                ):
                    raise _fail("payload_invalid")
        elif self.kind == "compact_tombstone":
            if (
                self.archive_digest is not None
                or self.producer is not None
                or self.imported_at is not None
                or self.reviewed_mappings
                or self.matched
                or self.changed
                or self.missing
                or self.pending_deny
                or self.omitted
                or self.removed_at is None
                or self.prior_receipt_digest is None
            ):
                raise _fail("payload_invalid")
            _timestamp(self.removed_at)
            _digest(self.prior_receipt_digest)
        else:
            raise _fail("payload_invalid")

    @classmethod
    def from_dict(cls, raw: object) -> ToolPackReceipt:
        if type(raw) is not dict:
            raise _fail("payload_invalid")
        kind = raw.get("kind")
        if kind == "import":
            value = _exact_dict(raw, _IMPORT_KEYS)
            producer_raw = _exact_dict(value["producer"], _PRODUCER_KEYS)
            groups = [
                _identity_list(value[name])
                for name in ("matched", "changed", "missing", "pending_deny", "omitted")
            ]
            return cls(
                schema=value["schema"],  # type: ignore[arg-type]
                kind="import",
                profile_id=_profile_id(value["profile_id"]),
                pack_digest=_digest(value["pack_digest"]),
                archive_digest=_digest(value["archive_digest"]),
                producer=(
                    _text(producer_raw["name"], max_bytes=128),
                    _text(producer_raw["version"], max_bytes=128),
                ),
                imported_at=_timestamp(value["imported_at"]),
                reviewed_mappings=_mapping_list(value["reviewed_mappings"]),
                matched=groups[0],
                changed=groups[1],
                missing=groups[2],
                pending_deny=groups[3],
                omitted=groups[4],
            )
        if kind == "compact_tombstone":
            value = _exact_dict(raw, _TOMBSTONE_KEYS)
            return cls(
                schema=value["schema"],  # type: ignore[arg-type]
                kind="compact_tombstone",
                profile_id=_profile_id(value["profile_id"]),
                pack_digest=_digest(value["pack_digest"]),
                removed_at=_timestamp(value["removed_at"]),
                prior_receipt_digest=_digest(value["prior_receipt_digest"]),
            )
        raise _fail("payload_invalid")

    def to_dict(self) -> dict[str, object]:
        if self.kind == "compact_tombstone":
            return {
                "schema": self.schema,
                "kind": self.kind,
                "profile_id": self.profile_id,
                "pack_digest": self.pack_digest,
                "removed_at": self.removed_at,
                "prior_receipt_digest": self.prior_receipt_digest,
            }
        assert self.archive_digest is not None
        assert self.producer is not None
        assert self.imported_at is not None

        def identities(
            values: tuple[tuple[str, str, str], ...],
        ) -> list[dict[str, str]]:
            return [
                {"authority": authority, "server_key": server, "tool_name": tool}
                for authority, server, tool in values
            ]

        return {
            "schema": self.schema,
            "kind": self.kind,
            "profile_id": self.profile_id,
            "pack_digest": self.pack_digest,
            "archive_digest": self.archive_digest,
            "producer": {"name": self.producer[0], "version": self.producer[1]},
            "imported_at": self.imported_at,
            "reviewed_mappings": [
                {"source_server_key": source, "destination_server_key": destination}
                for source, destination in self.reviewed_mappings
            ],
            "matched": identities(self.matched),
            "changed": identities(self.changed),
            "missing": identities(self.missing),
            "pending_deny": identities(self.pending_deny),
            "omitted": identities(self.omitted),
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class ReceiptHandle:
    receipt_id: str
    digest: str
    path: Path
    size: int


@dataclass(frozen=True, slots=True)
class VerifiedToolPackReceipt:
    receipt: ToolPackReceipt
    digest: str
    handle: ReceiptHandle


@dataclass(slots=True)
class _RootState:
    lock: threading.RLock
    max_total_bytes: int
    reserved_bytes: int = 0


_ROOT_STATES_LOCK = threading.Lock()
_ROOT_STATES: dict[Path, _RootState] = {}


def _root_state(root: Path, max_total_bytes: int) -> _RootState:
    with _ROOT_STATES_LOCK:
        # ponytail: process-lifetime registry; weak cleanup only matters for apps
        # constructing unbounded distinct receipt roots in one process.
        state = _ROOT_STATES.setdefault(
            root,
            _RootState(threading.RLock(), max_total_bytes),
        )
        with state.lock:
            state.max_total_bytes = min(state.max_total_bytes, max_total_bytes)
        return state


class ReceiptReservation(AbstractContextManager["ReceiptReservation"]):
    """One idempotently releasable receipt-capacity reservation."""

    def __init__(self, store: ToolPackReceiptStore, projected_bytes: int) -> None:
        self._store = store
        self._projected_bytes = projected_bytes
        self._active = True
        self._committed = False

    def commit(self, data: bytes) -> ReceiptHandle:
        if not self._active or self._committed:
            raise _fail("activation_failed")
        handle = self._store._commit(self, data)
        self._committed = True
        self.release()
        return handle

    def release(self) -> None:
        with self._store._state.lock:
            self._release_locked()

    def _release_locked(self) -> None:
        if not self._active:
            return
        self._store._state.reserved_bytes -= self._projected_bytes
        self._active = False

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
        return None


class ToolPackReceiptStore:
    """Private local receipt storage; receipts never grant policy authority."""

    def __init__(
        self,
        root: Path,
        *,
        max_receipt_bytes: int = MAX_RECEIPT_BYTES,
        max_total_bytes: int = MAX_RECEIPT_STORE_BYTES,
        max_reconcile_entries: int = MAX_RECONCILE_ENTRIES,
        _fault: Callable[[str], None] | None = None,
        _id_source: Callable[[], bytes] | None = None,
    ) -> None:
        if (
            not isinstance(root, Path)
            or type(max_receipt_bytes) is not int
            or type(max_total_bytes) is not int
            or type(max_reconcile_entries) is not int
            or max_receipt_bytes <= 0
            or max_total_bytes <= 0
            or max_reconcile_entries <= 0
        ):
            raise _fail("payload_invalid")
        self.root = Path(os.path.abspath(root.expanduser()))
        self.max_receipt_bytes = max_receipt_bytes
        self.max_total_bytes = max_total_bytes
        self.max_reconcile_entries = max_reconcile_entries
        self._fault = _fault
        self._id_source = _id_source or (lambda: secrets.token_bytes(16))
        self._ensure_root()
        self._state = _root_state(
            self.root.resolve(strict=True),
            max_total_bytes,
        )

    def _ensure_root(self) -> None:
        try:
            if self.root.exists() or self.root.is_symlink():
                info = self.root.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise _fail("activation_failed")
            else:
                self.root.mkdir(parents=True, mode=0o700)
            self.root.chmod(0o700)
            self._fsync_directory(self.root)
            self._fsync_directory(self.root.parent)
        except ToolPackError:
            raise
        except OSError:
            raise _fail("activation_failed") from None

    def reserve(self, projected_bytes: int) -> ReceiptReservation:
        if (
            type(projected_bytes) is not int
            or projected_bytes <= 0
            or projected_bytes > self.max_receipt_bytes
        ):
            raise _fail("capacity_exceeded")
        with self._state.lock:
            committed = self._committed_bytes_locked()
            if (
                committed + self._state.reserved_bytes + projected_bytes
                > self._state.max_total_bytes
            ):
                raise _fail("capacity_exceeded")
            self._state.reserved_bytes += projected_bytes
            return ReceiptReservation(self, projected_bytes)

    def exists(self, receipt_id: str) -> bool:
        path = self._path(receipt_id)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return False
        except OSError:
            raise _fail("payload_invalid") from None
        return stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)

    def read(self, receipt_id: str, *, expected_digest: str) -> VerifiedToolPackReceipt:
        path = self._path(receipt_id)
        digest = _digest(expected_digest)
        try:
            info = path.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise _fail("payload_invalid")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or stat.S_IMODE(opened.st_mode) != 0o600
                ):
                    raise _fail("payload_invalid")
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    data = stream.read(self.max_receipt_bytes + 1)
            finally:
                os.close(descriptor)
        except ToolPackError:
            raise
        except OSError:
            raise _fail("payload_invalid") from None
        if (
            len(data) > self.max_receipt_bytes
            or hashlib.sha256(data).hexdigest() != digest
        ):
            raise _fail("payload_invalid")
        raw = strict_json_object(
            data,
            category="payload_invalid",
            max_bytes=self.max_receipt_bytes,
        )
        receipt = ToolPackReceipt.from_dict(raw)
        if receipt.to_bytes() != data:
            raise _fail("payload_invalid")
        handle = ReceiptHandle(receipt_id, digest, path, len(data))
        return VerifiedToolPackReceipt(receipt, digest, handle)

    def reconcile_orphans(
        self,
        referenced_ids: AbstractSet[str],
        live_ids: AbstractSet[str],
        *,
        now: datetime,
    ) -> tuple[str, ...]:
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
            raise _fail("payload_invalid")
        protected = set(referenced_ids) | set(live_ids)
        cutoff = (now - ORPHAN_GRACE).timestamp()
        removed: list[str] = []
        with self._state.lock:
            try:
                entries: list[Path] = []
                with os.scandir(self.root) as scan:
                    for entry in scan:
                        if len(entries) >= self.max_reconcile_entries:
                            raise _fail("capacity_exceeded")
                        entries.append(Path(entry.path))
                entries.sort(key=lambda path: path.name)
            except ToolPackError:
                raise
            except OSError:
                raise _fail("activation_failed") from None
            for path in entries:
                name = path.name
                if name in protected or not _RECEIPT_ID_RE.fullmatch(name):
                    continue
                try:
                    info = path.lstat()
                    if (
                        stat.S_ISLNK(info.st_mode)
                        or not stat.S_ISREG(info.st_mode)
                        or info.st_mtime > cutoff
                    ):
                        continue
                    path.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
                removed.append(name)
        return tuple(removed)

    def write_compact_tombstone(
        self, source: ReceiptHandle, *, profile_id: str
    ) -> ReceiptHandle:
        if type(source) is not ReceiptHandle or source.path != self._path(
            source.receipt_id
        ):
            raise ToolPackError("remove", "non_removable")
        verified = self.read(source.receipt_id, expected_digest=source.digest)
        if (
            verified.receipt.kind != "import"
            or verified.receipt.profile_id != profile_id
        ):
            raise ToolPackError("remove", "non_removable")
        removed_at = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        compact = ToolPackReceipt(
            schema=RECEIPT_SCHEMA,
            kind="compact_tombstone",
            profile_id=_profile_id(profile_id),
            pack_digest=verified.receipt.pack_digest,
            removed_at=removed_at,
            prior_receipt_digest=source.digest,
        )
        data = compact.to_bytes()
        with self.reserve(len(data)) as reservation:
            return reservation.commit(data)

    def _path(self, receipt_id: str) -> Path:
        if type(receipt_id) is not str or not _RECEIPT_ID_RE.fullmatch(receipt_id):
            raise _fail("payload_invalid")
        return self.root / receipt_id

    def _committed_bytes_locked(self) -> int:
        total = 0
        try:
            entries = self.root.iterdir()
            for path in entries:
                if not _RECEIPT_ID_RE.fullmatch(path.name):
                    continue
                info = path.lstat()
                if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    total += info.st_size
        except OSError:
            raise _fail("capacity_exceeded") from None
        return total

    def _commit(self, reservation: ReceiptReservation, data: bytes) -> ReceiptHandle:
        if type(data) is not bytes or len(data) > reservation._projected_bytes:
            raise _fail("capacity_exceeded")
        if not data or len(data) > self.max_receipt_bytes:
            raise _fail("capacity_exceeded")
        raw = strict_json_object(
            data,
            category="payload_invalid",
            max_bytes=self.max_receipt_bytes,
        )
        receipt = ToolPackReceipt.from_dict(raw)
        if receipt.to_bytes() != data:
            raise _fail("payload_invalid")

        with self._state.lock:
            if not reservation._active:
                raise _fail("activation_failed")
            receipt_id = self._new_receipt_id_locked()
            target = self.root / receipt_id
            temporary: Path | None = None
            replaced = False
            try:
                descriptor, temp_name = tempfile.mkstemp(
                    prefix=f".{receipt_id}.", suffix=".tmp", dir=self.root
                )
                temporary = Path(temp_name)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        descriptor = -1
                        os.fchmod(stream.fileno(), 0o600)
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                if self._fault is not None:
                    self._fault("before_replace")
                os.replace(temporary, target)
                replaced = True
                temporary = None
                if self._fault is not None:
                    self._fault("after_replace")
                self._fsync_directory(self.root)
            except Exception as error:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                if isinstance(error, ToolPackError):
                    raise
                raise _fail(
                    "activation_uncertain" if replaced else "activation_failed"
                ) from None
            digest = hashlib.sha256(data).hexdigest()
            reservation._release_locked()
            return ReceiptHandle(receipt_id, digest, target, len(data))

    def _new_receipt_id_locked(self) -> str:
        for _ in range(64):
            try:
                value = self._id_source()
            except Exception:
                raise _fail("activation_failed") from None
            if type(value) is not bytes or len(value) != 16:
                continue
            receipt_id = "tp-" + value.hex()
            if (
                not (self.root / receipt_id).exists()
                and not (self.root / receipt_id).is_symlink()
            ):
                return receipt_id
        raise _fail("activation_failed")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = -1
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            os.fsync(descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)


__all__ = [
    "ReceiptHandle",
    "ReceiptReservation",
    "ToolPackReceipt",
    "ToolPackReceiptStore",
    "VerifiedToolPackReceipt",
]
