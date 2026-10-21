"""Closed, length-prefixed v1 protocol used only by the project index worker.

Locators and bodies are pipe data. No dependency on the ordinary tool protocol.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import struct
from typing import BinaryIO

from tldw_chatbook.Utils.filesystem_identity import DirectoryIdentity
from tldw_chatbook.Workspaces.project_index.models import (
    MAX_FILE_BYTES,
    MAX_QUEUED_BYTES,
    ProjectIndexPolicy,
    ScanEntry,
    relative_path,
)

VERSION = 1
MAX_FRAME_BYTES = 2 * 1024**2
FRAME_HEADER_BYTES = struct.calcsize("!I")
MAX_PAYLOAD_BYTES = MAX_FRAME_BYTES - FRAME_HEADER_BYTES
TEXT_PART_BYTES = 64 * 1024
# Retained page plus two full scratch bodies and the sender's distinct UTF-8
# slice: parent join/validation can overlap the final worker flush. The same
# reserve also covers the worker decoding the next file while the parent joins.
SOURCE_BUFFER_HEADROOM = 2 * MAX_FILE_BYTES + TEXT_PART_BYTES
PAGE_TEXT_BYTES = MAX_QUEUED_BYTES - SOURCE_BUFFER_HEADROOM
_FIELDS = {
    "Init": {"root", "identities", "policy"},
    "Directory": {"directory", "continuation", "revision", "exclusions", "ack"},
    "Capture": {"path", "revision", "exclusions", "ack"},
    "Close": set(),
    "Started": {"revision", "ack"},
    "Entries": {"entries"},
    "File": {"entry", "sha256", "text", "part", "final"},
    "DirectoryDone": {
        "directory",
        "identity",
        "complete",
        "more",
        "issues",
        "reads",
        "queued",
    },
    "Error": {"code"},
}


class ScanProtocolError(ValueError):
    """Invalid or oversized index frame (messages contain no input data)."""


def frame(kind: str, **fields) -> dict:
    return {"version": VERSION, "type": kind, **fields}


def _integer(value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScanProtocolError("invalid_integer")


def _string(value, limit=32768):
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise ScanProtocolError("invalid_string")


def _path(value, root=False):
    _string(value)
    if relative_path(value, root=root) != value:
        raise ScanProtocolError("noncanonical_path")


def _entry(value):
    if not isinstance(value, dict) or set(value) != {
        "relative_path",
        "kind",
        "identity",
        "mtime_ns",
        "size_bytes",
    }:
        raise ScanProtocolError("invalid_entry")
    _path(value["relative_path"])
    _string(value["identity"], 128)
    _integer(value["mtime_ns"])
    _integer(value["size_bytes"])
    ScanEntry(**value)


def validate_frame(value: dict) -> dict:
    """Validate exact fields and bounded primitive types on both pipe ends."""
    try:
        if (
            not isinstance(value, dict)
            or value.get("version") != VERSION
            or not isinstance(value.get("version"), int)
            or isinstance(value.get("version"), bool)
        ):
            raise ScanProtocolError("invalid_version")
        kind = value.get("type")
        if (
            not isinstance(kind, str)
            or kind not in _FIELDS
            or set(value) != _FIELDS[kind] | {"version", "type"}
        ):
            raise ScanProtocolError("invalid_fields")
        if kind == "Init":
            _string(value["root"])
            identities = value["identities"]
            if not isinstance(identities, list) or not 1 <= len(identities) <= 1024:
                raise ScanProtocolError("invalid_identities")
            for item in identities:
                if not isinstance(item, dict) or set(item) != {
                    "device",
                    "inode",
                    "mode",
                    "reparse",
                }:
                    raise ScanProtocolError("invalid_identity")
                for field in ("device", "inode", "mode"):
                    _integer(item[field])
                if not isinstance(item["reparse"], bool) or item["reparse"]:
                    raise ScanProtocolError("invalid_identity")
                DirectoryIdentity(**item)
            policy = value["policy"]
            if not isinstance(policy, dict) or set(policy) != {
                "include_prefixes",
                "text_suffixes",
            }:
                raise ScanProtocolError("invalid_policy")
            for items in policy.values():
                if not isinstance(items, list):
                    raise ScanProtocolError("invalid_policy")
                for item in items:
                    _string(item)
            ProjectIndexPolicy(
                tuple(policy["include_prefixes"]), tuple(policy["text_suffixes"])
            )
        if kind in {"Directory", "Capture", "Started"}:
            _integer(value["revision"], -1 if kind == "Started" else 0)
            _integer(value["ack"])
        if kind in {"Directory", "Capture"}:
            _path(
                value["directory"] if kind == "Directory" else value["path"],
                root=kind == "Directory",
            )
            if kind == "Directory" and not isinstance(value["continuation"], bool):
                raise ScanProtocolError("invalid_continuation")
            if not isinstance(value["exclusions"], list):
                raise ScanProtocolError("invalid_exclusions")
            for exclusion in value["exclusions"]:
                if (
                    not isinstance(exclusion, list)
                    or len(exclusion) != 2
                    or exclusion[0]
                    not in {"subtree", "file", "name", "direct_children"}
                ):
                    raise ScanProtocolError("invalid_exclusion")
                _path(exclusion[1], root=exclusion[0] in {"subtree", "direct_children"})
                if exclusion[0] == "name" and "/" in exclusion[1]:
                    raise ScanProtocolError("invalid_name")
        if kind == "Entries":
            if not isinstance(value["entries"], list) or len(value["entries"]) > 256:
                raise ScanProtocolError("invalid_entries")
            for entry in value["entries"]:
                _entry(entry)
        if kind == "File":
            _entry(value["entry"])
            if (
                value["entry"]["kind"] != "file"
                or value["entry"]["size_bytes"] > 1024**2
            ):
                raise ScanProtocolError("invalid_file")
            _string(value["sha256"], 64)
            if len(value["sha256"]) != 64 or any(
                c not in "0123456789abcdef" for c in value["sha256"]
            ):
                raise ScanProtocolError("invalid_digest")
            if (
                not isinstance(value["text"], str)
                or len(value["text"].encode("utf-8")) > TEXT_PART_BYTES
            ):
                raise ScanProtocolError("invalid_text_part")
            _integer(value["part"])
            if value["part"] > 64 or not isinstance(value["final"], bool):
                raise ScanProtocolError("invalid_part")
        if kind == "DirectoryDone":
            _path(value["directory"], root=True)
            _string(value["identity"], 128)
            if (
                not isinstance(value["complete"], bool)
                or not isinstance(value["more"], bool)
                or (value["complete"] and value["more"])
            ):
                raise ScanProtocolError("invalid_completion")
            if not isinstance(value["issues"], list) or len(value["issues"]) > 16:
                raise ScanProtocolError("invalid_issues")
            for issue in value["issues"]:
                _string(issue, 64)
            for name in ("reads", "queued"):
                _integer(value[name])
            if value["reads"] > 2 or value["queued"] > 4 * 1024**2:
                raise ScanProtocolError("resource_limit")
        if kind == "Error":
            _string(value["code"], 64)
        return value
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise ScanProtocolError("invalid_frame") from error


def encode_frame(value: dict) -> bytes:
    """Encode one validated frame, including a four-byte network-order length."""
    validate_frame(value)
    raw = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise ScanProtocolError("frame_too_large")
    return struct.pack("!I", len(raw)) + raw


def write_frame(stream: BinaryIO, value: dict) -> int:
    """Write and flush one bounded frame; return total wire size for accounting."""
    raw = encode_frame(value)
    stream.write(raw)
    stream.flush()
    return len(raw)


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    result = bytearray()
    while len(result) < count:
        data = stream.read(count - len(result))
        if not data:
            raise ScanProtocolError("unexpected_eof")
        result.extend(data)
    return bytes(result)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ScanProtocolError("duplicate_key")
        result[key] = value
    return result


def read_frame(stream: BinaryIO) -> dict:
    """Reject an oversized header before allocating or reading its payload."""
    count = struct.unpack("!I", _read_exact(stream, FRAME_HEADER_BYTES))[0]
    if not 0 < count <= MAX_PAYLOAD_BYTES:
        raise ScanProtocolError("frame_too_large")
    try:
        value = json.loads(_read_exact(stream, count), object_pairs_hook=_unique_pairs)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ScanProtocolError("invalid_json") from error
    return validate_frame(value)


def authority_fields(authority) -> dict:
    """Serialize the complete code-issued exclusion floor without root bodies."""
    return {
        "revision": authority.authority_revision,
        "exclusions": [list(e) for e in authority.exclusions],
    }


def initialization(authority, policy: ProjectIndexPolicy) -> dict:
    return frame(
        "Init",
        root=str(authority.root.path),
        identities=[asdict(i) for i in authority.chain.identities],
        policy={
            "include_prefixes": list(policy.include_prefixes),
            "text_suffixes": list(policy.text_suffixes),
        },
    )
