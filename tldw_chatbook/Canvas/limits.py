"""Shared, pure validation helpers for the Canvas V1 runtime boundary."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

MAX_WIRE_INTEGER = (1 << 53) - 1
MAX_CANVASES_PER_CONVERSATION = 10
MAX_REVISIONS_PER_CANVAS = 100
MAX_DURABLE_SOURCE_BYTES_PER_CONVERSATION = 50 * 1024 * 1024
MAX_DURABLE_SOURCE_BYTES_PER_REVISION = 512 * 1024
MAX_CANVAS_TITLE_BYTES = 4 * 1024
MAX_CANVAS_ORIGIN_TURN_ID_BYTES = 256
_DATA_MIME_PATTERN = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class CanvasLimitError(ValueError):
    """Raised when a Canvas value exceeds a closed runtime boundary."""


@dataclass(frozen=True, slots=True)
class CanvasLimits:
    """Non-negotiable Canvas V1 resource ceilings."""

    html_bytes: int = 512 * 1024
    asset_bytes: int = 1024 * 1024
    aggregate_asset_bytes: int = 4 * 1024 * 1024
    dom_nodes: int = 5_000
    css_rules: int = 2_000
    script_bytes: int = 256 * 1024
    runtime_memory_bytes: int = 32 * 1024 * 1024
    stack_bytes: int = 512 * 1024
    startup_milliseconds: int = 250
    event_milliseconds: int = 50
    patches_per_event: int = 1_000
    submit_payload_bytes: int = 16 * 1024
    json_depth: int = 16
    download_payload_bytes: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        for field_name, value in (
            ("html_bytes", self.html_bytes),
            ("asset_bytes", self.asset_bytes),
            ("aggregate_asset_bytes", self.aggregate_asset_bytes),
            ("dom_nodes", self.dom_nodes),
            ("css_rules", self.css_rules),
            ("script_bytes", self.script_bytes),
            ("runtime_memory_bytes", self.runtime_memory_bytes),
            ("stack_bytes", self.stack_bytes),
            ("startup_milliseconds", self.startup_milliseconds),
            ("event_milliseconds", self.event_milliseconds),
            ("patches_per_event", self.patches_per_event),
            ("submit_payload_bytes", self.submit_payload_bytes),
            ("json_depth", self.json_depth),
            ("download_payload_bytes", self.download_payload_bytes),
        ):
            _validate_non_negative_integer(value, field_name=field_name)
            if value > MAX_WIRE_INTEGER:
                raise CanvasLimitError(f"{field_name} exceeds the supported integer range")
            if value == 0:
                raise CanvasLimitError(f"{field_name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class CanvasRepositoryLimits:
    """Central durable Canvas ceilings, injectable for boundary tests."""

    max_canvases_per_conversation: int = MAX_CANVASES_PER_CONVERSATION
    max_revisions_per_canvas: int = MAX_REVISIONS_PER_CANVAS
    max_source_bytes_per_conversation: int = (
        MAX_DURABLE_SOURCE_BYTES_PER_CONVERSATION
    )
    max_source_bytes_per_revision: int = MAX_DURABLE_SOURCE_BYTES_PER_REVISION
    max_title_bytes: int = MAX_CANVAS_TITLE_BYTES
    max_origin_turn_id_bytes: int = MAX_CANVAS_ORIGIN_TURN_ID_BYTES

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_canvases_per_conversation", self.max_canvases_per_conversation),
            ("max_revisions_per_canvas", self.max_revisions_per_canvas),
            (
                "max_source_bytes_per_conversation",
                self.max_source_bytes_per_conversation,
            ),
            ("max_source_bytes_per_revision", self.max_source_bytes_per_revision),
            ("max_title_bytes", self.max_title_bytes),
            ("max_origin_turn_id_bytes", self.max_origin_turn_id_bytes),
        ):
            _validate_non_negative_integer(value, field_name=field_name)
            if value == 0:
                raise CanvasLimitError(f"{field_name} must be greater than zero")
            if value > MAX_WIRE_INTEGER:
                raise CanvasLimitError(f"{field_name} exceeds the supported integer range")


@dataclass(frozen=True, slots=True)
class DecodedDataUrl:
    """A data URL after its transport encoding has been removed."""

    mime_type: str
    data: bytes


def utf8_byte_length(value: str) -> int:
    """Return the strict UTF-8 byte length of *value* without coercion."""
    if not isinstance(value, str):
        raise CanvasLimitError("value must be a string")
    try:
        return len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise CanvasLimitError("value must contain valid Unicode") from exc


def validate_utf8_text(value: str, *, limit: int, field_name: str) -> int:
    """Validate a strict UTF-8 string and return its encoded byte length."""
    byte_count = utf8_byte_length(value)
    _validate_non_negative_integer(limit, field_name=f"{field_name} limit")
    if limit > MAX_WIRE_INTEGER:
        raise CanvasLimitError(f"{field_name} limit exceeds the supported integer range")
    if byte_count > limit:
        raise CanvasLimitError(f"{field_name} exceeds {limit} UTF-8 bytes")
    return byte_count


def validate_utf8_text_parts(
    values: Iterable[str], *, limit: int, field_name: str
) -> int:
    """Validate the aggregate strict UTF-8 size of untrusted text values."""
    _validate_non_negative_integer(limit, field_name=f"{field_name} limit")
    if limit > MAX_WIRE_INTEGER:
        raise CanvasLimitError(f"{field_name} limit exceeds the supported integer range")

    total = 0
    for value in values:
        total += utf8_byte_length(value)
        if total > MAX_WIRE_INTEGER:
            raise CanvasLimitError(f"{field_name} exceeds the supported integer range")
        if total > limit:
            raise CanvasLimitError(f"{field_name} exceeds {limit} UTF-8 bytes")
    return total


def sha256_utf8(value: str) -> str:
    """Return the lowercase SHA-256 identity for one strict UTF-8 text value."""
    if not isinstance(value, str):
        raise CanvasLimitError("value must be a string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise CanvasLimitError("value must contain valid Unicode") from exc
    return hashlib.sha256(encoded).hexdigest()


def verify_sha256_utf8(value: str, digest: str) -> bool:
    """Return whether *digest* is the exact lowercase SHA-256 of UTF-8 *value*."""
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CanvasLimitError("SHA-256 digest must be 64 lowercase hexadecimal characters")
    return hmac.compare_digest(sha256_utf8(value), digest)


def validate_count(count: int, *, limit: int, field_name: str) -> int:
    """Validate an untrusted count, accepting the ceiling and rejecting overflow."""
    _validate_non_negative_integer(limit, field_name=f"{field_name} limit")
    if limit > MAX_WIRE_INTEGER:
        raise CanvasLimitError(f"{field_name} limit exceeds the supported integer range")
    _validate_non_negative_integer(count, field_name=field_name)
    if count > MAX_WIRE_INTEGER:
        raise CanvasLimitError(f"{field_name} exceeds the supported integer range")
    if count > limit:
        raise CanvasLimitError(f"{field_name} exceeds {limit}")
    return count


def validate_unique_identifiers(
    identifiers: Sequence[str], *, field_name: str
) -> tuple[str, ...]:
    """Return opaque identifiers after rejecting empty, malformed, or duplicate IDs."""
    if isinstance(identifiers, (str, bytes)) or not isinstance(identifiers, Sequence):
        raise CanvasLimitError(f"{field_name} must be a sequence")

    checked: list[str] = []
    seen: set[str] = set()
    for identifier in identifiers:
        validate_opaque_identifier(identifier, field_name=field_name.removesuffix("s"))
        if identifier in seen:
            raise CanvasLimitError(f"{field_name} contains a duplicate identifier")
        seen.add(identifier)
        checked.append(identifier)
    return tuple(checked)


def validate_opaque_identifier(identifier: str, *, field_name: str = "identifier") -> str:
    """Validate an application-issued opaque identifier without interpreting it."""
    if not isinstance(identifier, str) or not identifier:
        raise CanvasLimitError(f"{field_name} must be a non-empty string")
    validate_utf8_text(identifier, limit=256, field_name=field_name)
    return identifier


def decode_data_url(value: str, *, field_name: str) -> DecodedDataUrl:
    """Decode a base64 ``data:`` URL while rejecting permissive browser syntax."""
    if not isinstance(value, str) or not value.startswith("data:"):
        raise CanvasLimitError(f"{field_name} must be a data URL")
    try:
        header, encoded_payload = value[5:].split(",", maxsplit=1)
    except ValueError as exc:
        raise CanvasLimitError(f"{field_name} data URL is missing a payload") from exc

    parts = header.split(";")
    mime_type = parts[0].lower()
    if not _DATA_MIME_PATTERN.fullmatch(mime_type):
        raise CanvasLimitError(f"{field_name} data URL has an invalid MIME type")
    if parts[1:] != ["base64"]:
        raise CanvasLimitError(f"{field_name} data URL has an unsupported data URL parameter")
    try:
        payload = base64.b64decode(encoded_payload.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise CanvasLimitError(f"{field_name} data URL must contain valid base64") from exc
    return DecodedDataUrl(mime_type=mime_type, data=payload)


def validate_asset_payloads(
    assets: Sequence[DecodedDataUrl], *, per_asset_limit: int, aggregate_limit: int
) -> int:
    """Validate decoded asset and aggregate byte ceilings and return their total."""
    if isinstance(assets, (str, bytes)) or not isinstance(assets, Sequence):
        raise CanvasLimitError("assets must be a sequence")

    total = 0
    for asset in assets:
        if not isinstance(asset, DecodedDataUrl):
            raise CanvasLimitError("asset must be a decoded data URL")
        asset_size = len(asset.data)
        _validate_non_negative_integer(per_asset_limit, field_name="asset limit")
        if asset_size > per_asset_limit:
            raise CanvasLimitError(f"asset exceeds {per_asset_limit} decoded bytes")
        total += asset_size
        _validate_non_negative_integer(aggregate_limit, field_name="aggregate assets limit")
        if total > aggregate_limit:
            raise CanvasLimitError(
                f"aggregate assets exceed {aggregate_limit} decoded bytes"
            )
    return total


def json_depth(value: JsonValue) -> int:
    """Return a validated JSON value's structural depth without recursive descent."""
    return _validate_json_value(value, field_name="JSON value", max_depth=None)


def validate_json_value(value: JsonValue, *, max_depth: int, field_name: str) -> int:
    """Validate JSON compatibility and structural depth without calling user code."""
    _validate_non_negative_integer(max_depth, field_name=f"{field_name} maximum JSON depth")
    return _validate_json_value(value, field_name=field_name, max_depth=max_depth)


def _validate_json_value(value: object, *, field_name: str, max_depth: int | None) -> int:
    max_found_depth = 0
    active_container_ids: set[int] = set()
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]

    while stack:
        current, depth, exiting = stack.pop()
        if exiting:
            active_container_ids.remove(id(current))
            continue
        if max_depth is not None and depth > max_depth:
            raise CanvasLimitError(f"{field_name} exceeds JSON depth {max_depth}")
        max_found_depth = max(max_found_depth, depth)

        if current is None or isinstance(current, bool):
            continue
        if isinstance(current, str):
            utf8_byte_length(current)
            continue
        if isinstance(current, int):
            if abs(current) > MAX_WIRE_INTEGER:
                raise CanvasLimitError(f"{field_name} integer exceeds the supported range")
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise CanvasLimitError(f"{field_name} numbers must be finite")
            continue
        if not isinstance(current, (Mapping, list, tuple)):
            raise CanvasLimitError(f"{field_name} must be JSON-compatible")

        container_id = id(current)
        if container_id in active_container_ids:
            raise CanvasLimitError(f"{field_name} must not contain a cycle")
        active_container_ids.add(container_id)
        stack.append((current, depth, True))

        if isinstance(current, Mapping):
            for key, child in current.items():
                if not isinstance(key, str):
                    raise CanvasLimitError(f"{field_name} object keys must be strings")
                utf8_byte_length(key)
                stack.append((child, depth + 1, False))
        else:
            for child in current:
                stack.append((child, depth + 1, False))

    return max_found_depth


def _validate_non_negative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CanvasLimitError(f"{field_name} must be a non-negative integer")
    return value
