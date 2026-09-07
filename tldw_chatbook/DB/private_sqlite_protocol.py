"""Bounded, source-free wire types for the private SQLite file helper."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import asdict, dataclass, fields
from typing import BinaryIO

VERSION = 1
MAX_BODY_BYTES = 65536
MAX_NESTING = 8
OPERATIONS = frozenset({"prepare", "pin_source", "recheck_source", "close"})
PRIVATE_STATUSES = frozenset(
    {
        "created_private",
        "hardened_private",
        "already_private",
        "unsafe_parent",
        "wrong_owner",
        "link_or_non_regular",
        "operation_failed",
        "trusted_directory",
        "unverified_platform",
    }
)
ARTIFACT_STATUSES = frozenset(
    {
        "created_private",
        "hardened_private",
        "already_private",
        "unverified_platform",
        "preserved_source_mode",
        "absent",
    }
)
PRIVATE_REASONS = frozenset(
    {
        "missing_sqlite_artifact",
        "non_regular_sqlite_artifact",
        "unsafe_sqlite_artifact",
        "private_sqlite_identity_changed",
        "private_sqlite_postcondition_failed",
        "optional_sqlite_generation_churn",
        "required_posix_guards_unavailable",
        "private_sqlite_source_identity_changed",
        "invalid_absolute_path",
        "untrusted_directory_owner",
        "shared_writable_parent",
        "missing_parent",
        "non_directory_parent",
        "missing_leaf_in_shared_sticky_parent",
        "symlink_hop_limit_exceeded",
        "missing_directory",
        "non_directory_component",
        "trusted_directory_postcondition_failed",
        "shared_sticky_directory",
        "native_acl_not_verified",
        "OSError",
        "PermissionError",
        "FileNotFoundError",
        "FileExistsError",
        "NotADirectoryError",
        "IsADirectoryError",
        "BlockingIOError",
        "InterruptedError",
        "operation_failed",
    }
)
HELPER_STATUSES = frozenset({"helper_unavailable", "protocol_error", "timeout"})


class ProtocolError(ValueError):
    """Reject invalid transport input without including any input bytes."""

    def __init__(self) -> None:
        super().__init__("private_sqlite_protocol_error")


@dataclass(frozen=True, repr=False)
class FileIdentity:
    """Exact integer metadata; inode equality deliberately excludes mutable facts."""

    dev: int
    ino: int
    mode: int
    uid: int
    gid: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int

    def __post_init__(self) -> None:
        if any(type(getattr(self, field.name)) is not int for field in fields(self)):
            raise ProtocolError()

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        return cls(
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def same_inode(self, other: FileIdentity) -> bool:
        return (self.dev, self.ino) == (other.dev, other.ino)

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_payload(cls, value: object) -> FileIdentity:
        if type(value) is not dict or set(value) != {
            field.name for field in fields(cls)
        }:
            raise ProtocolError()
        return cls(**value)


@dataclass(frozen=True, repr=False)
class PrepareRequest:
    """Parent-admitted file preparation options, never executable policy."""

    path: str
    writable: bool
    create_if_missing: bool
    preserve_source_mode: bool

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path or "\x00" in self.path:
            raise ProtocolError()
        if any(
            type(value) is not bool
            for value in (
                self.writable,
                self.create_if_missing,
                self.preserve_source_mode,
            )
        ):
            raise ProtocolError()
        if self.create_if_missing and not self.writable:
            raise ProtocolError()
        if self.preserve_source_mode and (self.writable or self.create_if_missing):
            raise ProtocolError()


@dataclass(frozen=True, repr=False)
class PrepareResult:
    """Main identity and the fixed main/WAL/SHM/journal privacy cohort."""

    main_identity: FileIdentity
    artifacts: tuple[str, str, str, str]

    def to_payload(self) -> dict[str, object]:
        return {
            "main_identity": self.main_identity.to_payload(),
            "artifacts": list(self.artifacts),
        }

    @classmethod
    def from_payload(cls, value: object) -> PrepareResult:
        if type(value) is not dict or set(value) != {"main_identity", "artifacts"}:
            raise ProtocolError()
        cohort = value["artifacts"]
        if (
            type(cohort) is not list
            or len(cohort) != 4
            or any(
                type(status) is not str or status not in ARTIFACT_STATUSES
                for status in cohort
            )
            or cohort[0] == "absent"
        ):
            raise ProtocolError()
        return cls(FileIdentity.from_payload(value["main_identity"]), tuple(cohort))


def validate_payload(payload: object) -> dict[str, object]:
    """Validate the closed request or response direction before either is used."""
    if type(payload) is not dict:
        raise ProtocolError()
    if type(payload.get("version")) is not int or payload["version"] != VERSION:
        raise ProtocolError()
    operation = payload.get("operation")
    if type(operation) is not str or operation not in OPERATIONS:
        raise ProtocolError()
    base = {"version", "operation"}
    if "status" not in payload:
        if operation in {"prepare", "pin_source"}:
            if set(payload) != base | {
                "path",
                "writable",
                "create_if_missing",
                "preserve_source_mode",
            }:
                raise ProtocolError()
            request = PrepareRequest(
                **{
                    key: payload[key]
                    for key in (
                        "path",
                        "writable",
                        "create_if_missing",
                        "preserve_source_mode",
                    )
                }
            )
            if operation == "pin_source" and (
                request.writable or request.create_if_missing
            ):
                raise ProtocolError()
        elif set(payload) != base:
            raise ProtocolError()
        return payload
    status = payload["status"]
    base.add("status")
    if status == "ok":
        if operation == "close":
            if set(payload) != base:
                raise ProtocolError()
        else:
            if set(payload) != base | {"result"}:
                raise ProtocolError()
            PrepareResult.from_payload(payload["result"])
    elif type(status) is str and status == "private_path_error":
        if set(payload) != base | {"privacy_status", "reason"}:
            raise ProtocolError()
        if (
            type(payload["privacy_status"]) is not str
            or payload["privacy_status"] not in PRIVATE_STATUSES
            or type(payload["reason"]) is not str
            or payload["reason"] not in PRIVATE_REASONS
        ):
            raise ProtocolError()
    elif type(status) is str and status in HELPER_STATUSES:
        if set(payload) != base:
            raise ProtocolError()
    else:
        raise ProtocolError()
    return payload


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError()
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ProtocolError()


def _check_nesting(body: bytes) -> None:
    depth = 0
    quoted = escaped = False
    for byte in body:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > MAX_NESTING:
                raise ProtocolError()
        elif byte in (93, 125):
            depth -= 1


def encode_frame(payload: dict[str, object]) -> bytes:
    """Encode one closed payload, capped at 64 KiB of JSON."""
    try:
        validate_payload(payload)
        body = json.dumps(
            payload, allow_nan=False, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        if not 0 < len(body) <= MAX_BODY_BYTES:
            raise ProtocolError()
        return struct.pack("!I", len(body)) + body
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ProtocolError() from None


def decode_frame(frame: bytes) -> dict[str, object]:
    """Decode exactly one frame, refusing trailing data and ambiguous JSON."""
    try:
        if type(frame) is not bytes or len(frame) < 4:
            raise ProtocolError()
        length = struct.unpack("!I", frame[:4])[0]
        if not 0 < length <= MAX_BODY_BYTES or len(frame) != length + 4:
            raise ProtocolError()
        body = frame[4:]
        _check_nesting(body)
        return validate_payload(
            json.loads(
                body.decode("utf-8"),
                object_pairs_hook=_object_pairs,
                parse_constant=_reject_constant,
            )
        )
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ProtocolError() from None


def read_frame(stream: BinaryIO) -> dict[str, object] | None:
    """Read a bounded frame; check its length before reading/allocating the body.

    The caller owns pipe deadlines. None denotes clean EOF between frames only.
    """
    header = bytearray()
    while len(header) < 4:
        part = stream.read(4 - len(header))
        if not part:
            if not header:
                return None
            raise ProtocolError()
        header.extend(part)
    length = struct.unpack("!I", header)[0]
    if not 0 < length <= MAX_BODY_BYTES:
        raise ProtocolError()
    body = bytearray()
    while len(body) < length:
        part = stream.read(length - len(body))
        if not part:
            raise ProtocolError()
        body.extend(part)
    return decode_frame(bytes(header + body))
