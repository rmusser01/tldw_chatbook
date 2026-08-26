"""Bounded, canonical storage for Console thinking evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast


THINKING_ENVELOPE_VERSION = 1
MAX_THINKING_BLOCKS = 32
MAX_THINKING_TEXT_BYTES = 256 * 1024
MAX_THINKING_ENVELOPE_BYTES = 1024 * 1024
MAX_THINKING_PROVENANCE_BYTES = 200
MAX_THINKING_BLOCK_ID_BYTES = 128

ThinkingVisibility = Literal["displayable", "proprietary"]
ThinkingStatus = Literal["complete", "stopped", "failed"]
ThinkingHistoryPolicy = Literal["auto", "include", "exclude"]

_INVALID_MESSAGE = "Invalid thinking data."
_MALFORMED_WARNING = "Thinking data could not be read."
_UNSUPPORTED_WARNING = "Thinking data version is unsupported."
_BLOCK_KEYS = frozenset(
    {
        "block_id",
        "round_ordinal",
        "provider",
        "model",
        "protocol",
        "source_format",
        "status",
        "visibility",
    }
)
_DISPLAYABLE_KEYS = _BLOCK_KEYS | {"text"}
_TOP_LEVEL_KEYS = frozenset({"version", "blocks"})
_CANONICAL_ENCODER = json.JSONEncoder(
    ensure_ascii=False,
    separators=(",", ":"),
    allow_nan=False,
)


class ThinkingEnvelopeValidationError(ValueError):
    """Raised when a thinking envelope is not canonical V1 data."""


class _InvalidThinking(Exception):
    pass


def _fail() -> None:
    raise _InvalidThinking


def _bounded_string(value: object, maximum: int, *, nonblank: bool = True) -> str:
    if type(value) is not str:
        _fail()
    text = cast(str, value)
    if (nonblank and not text.strip()) or len(text.encode("utf-8")) > maximum:
        _fail()
    return text


def _validate_shared_fields(
    block_id: object,
    round_ordinal: object,
    provider: object,
    model: object,
    protocol: object,
    source_format: object,
    status: object,
) -> None:
    _bounded_string(block_id, MAX_THINKING_BLOCK_ID_BYTES)
    if type(round_ordinal) is not int or cast(int, round_ordinal) < 0:
        _fail()
    for value in (provider, model, protocol, source_format):
        _bounded_string(value, MAX_THINKING_PROVENANCE_BYTES)
    if status not in {"complete", "stopped", "failed"}:
        _fail()


@dataclass(frozen=True, slots=True)
class DisplayableThinkingBlock:
    """A displayable reasoning block with exact, bounded text."""

    block_id: str
    round_ordinal: int
    provider: str
    model: str
    protocol: str
    source_format: str
    status: ThinkingStatus
    text: str = field(repr=False)
    visibility: Literal["displayable"] = field(default="displayable", init=False)

    def __post_init__(self) -> None:
        try:
            _validate_shared_fields(
                self.block_id,
                self.round_ordinal,
                self.provider,
                self.model,
                self.protocol,
                self.source_format,
                self.status,
            )
            _bounded_string(self.text, MAX_THINKING_TEXT_BYTES, nonblank=False)
            if not self.text:
                _fail()
        except _InvalidThinking:
            raise ThinkingEnvelopeValidationError(_INVALID_MESSAGE) from None


@dataclass(frozen=True, slots=True)
class ProprietaryThinkingBlock:
    """Content-free evidence that a provider produced proprietary thinking."""

    block_id: str
    round_ordinal: int
    provider: str
    model: str
    protocol: str
    source_format: str
    status: ThinkingStatus
    visibility: Literal["proprietary"] = field(default="proprietary", init=False)

    def __post_init__(self) -> None:
        try:
            _validate_shared_fields(
                self.block_id,
                self.round_ordinal,
                self.provider,
                self.model,
                self.protocol,
                self.source_format,
                self.status,
            )
        except _InvalidThinking:
            raise ThinkingEnvelopeValidationError(_INVALID_MESSAGE) from None


ThinkingBlock = DisplayableThinkingBlock | ProprietaryThinkingBlock


@dataclass(frozen=True, slots=True)
class ThinkingEnvelope:
    """One ordered collection of terminal thinking blocks."""

    blocks: tuple[ThinkingBlock, ...] = field(repr=False)

    def __post_init__(self) -> None:
        try:
            if type(self.blocks) is not tuple or len(self.blocks) > MAX_THINKING_BLOCKS:
                _fail()
            block_ids: set[str] = set()
            prior_ordinal = -1
            for block in self.blocks:
                if not isinstance(
                    block, (DisplayableThinkingBlock, ProprietaryThinkingBlock)
                ):
                    _fail()
                if block.block_id in block_ids or block.round_ordinal <= prior_ordinal:
                    _fail()
                block_ids.add(block.block_id)
                prior_ordinal = block.round_ordinal
        except _InvalidThinking:
            raise ThinkingEnvelopeValidationError(_INVALID_MESSAGE) from None


@dataclass(frozen=True, slots=True)
class ThinkingEnvelopeRead:
    """Safe durable-hydration result that never exposes opaque content in repr."""

    envelope: ThinkingEnvelope | None = field(default=None, repr=False)
    opaque_json: str | None = field(default=None, repr=False)
    warning: str | None = None

    @property
    def generation_actions_enabled(self) -> bool:
        """Whether actions that could replace durable thinking remain safe."""
        return self.opaque_json is None


def _strict_json_loads(value: str) -> object:
    def reject_constant(_value: str) -> None:
        _fail()

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                _fail()
            result[key] = item
        return result

    return json.loads(
        value, parse_constant=reject_constant, object_pairs_hook=unique_object
    )


def _exact_mapping(value: object, keys: frozenset[str]) -> Mapping[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != keys:
        _fail()
    return cast(Mapping[str, object], value)


def _parse_block(value: object) -> ThinkingBlock:
    if type(value) is not dict:
        _fail()
    raw = cast(Mapping[str, object], value)
    visibility = raw.get("visibility")
    expected_keys = _DISPLAYABLE_KEYS if visibility == "displayable" else _BLOCK_KEYS
    item = _exact_mapping(value, expected_keys)
    shared = {
        "block_id": _bounded_string(item["block_id"], MAX_THINKING_BLOCK_ID_BYTES),
        "round_ordinal": item["round_ordinal"],
        "provider": _bounded_string(item["provider"], MAX_THINKING_PROVENANCE_BYTES),
        "model": _bounded_string(item["model"], MAX_THINKING_PROVENANCE_BYTES),
        "protocol": _bounded_string(item["protocol"], MAX_THINKING_PROVENANCE_BYTES),
        "source_format": _bounded_string(
            item["source_format"], MAX_THINKING_PROVENANCE_BYTES
        ),
        "status": item["status"],
    }
    if visibility == "displayable":
        text = _bounded_string(item["text"], MAX_THINKING_TEXT_BYTES, nonblank=False)
        if not text:
            _fail()
        return DisplayableThinkingBlock(text=text, **shared)  # type: ignore[arg-type]
    if visibility == "proprietary":
        return ProprietaryThinkingBlock(**shared)  # type: ignore[arg-type]
    _fail()


def _envelope_value(envelope: ThinkingEnvelope) -> dict[str, object]:
    blocks: list[dict[str, object]] = []
    for block in envelope.blocks:
        value: dict[str, object] = {
            "block_id": block.block_id,
            "round_ordinal": block.round_ordinal,
            "provider": block.provider,
            "model": block.model,
            "protocol": block.protocol,
            "source_format": block.source_format,
            "status": block.status,
            "visibility": block.visibility,
        }
        if isinstance(block, DisplayableThinkingBlock):
            value["text"] = block.text
        blocks.append(value)
    return {"version": THINKING_ENVELOPE_VERSION, "blocks": blocks}


def _validate_canonical_size(envelope: ThinkingEnvelope) -> None:
    total = 0
    for chunk in _CANONICAL_ENCODER.iterencode(_envelope_value(envelope)):
        total += len(chunk.encode("utf-8"))
        if total > MAX_THINKING_ENVELOPE_BYTES:
            _fail()


def _parse_value(value: object) -> ThinkingEnvelope:
    if type(value) is not str:
        _fail()
    raw = cast(str, value)
    if len(raw.encode("utf-8")) > MAX_THINKING_ENVELOPE_BYTES:
        _fail()
    item = _exact_mapping(_strict_json_loads(raw), _TOP_LEVEL_KEYS)
    if type(item["version"]) is not int or item["version"] != THINKING_ENVELOPE_VERSION:
        _fail()
    if type(item["blocks"]) is not list:
        _fail()
    blocks = tuple(_parse_block(block) for block in cast(list[object], item["blocks"]))
    envelope = ThinkingEnvelope(blocks=blocks)
    _validate_canonical_size(envelope)
    return envelope


def parse_thinking_blocks_json(value: object) -> ThinkingEnvelope:
    """Strictly parse one canonical V1 JSON envelope."""
    try:
        return _parse_value(value)
    except Exception:
        raise ThinkingEnvelopeValidationError(_INVALID_MESSAGE) from None


def read_thinking_blocks_json(value: object) -> ThinkingEnvelopeRead:
    """Read durable data, preserving an unsupported version as opaque JSON."""
    if value is None:
        return ThinkingEnvelopeRead()
    if type(value) is not str:
        return ThinkingEnvelopeRead(warning=_MALFORMED_WARNING)
    raw = cast(str, value)
    try:
        decoded = _strict_json_loads(raw)
        if (
            type(decoded) is dict
            and type(decoded.get("version")) is int
            and decoded["version"] != THINKING_ENVELOPE_VERSION
        ):
            return ThinkingEnvelopeRead(opaque_json=raw, warning=_UNSUPPORTED_WARNING)
        return ThinkingEnvelopeRead(envelope=_parse_value(raw))
    except Exception:
        return ThinkingEnvelopeRead(warning=_MALFORMED_WARNING)


def dump_thinking_blocks_json(envelope: ThinkingEnvelope | None) -> str | None:
    """Serialize a valid envelope using deterministic canonical JSON."""
    if envelope is None:
        return None
    try:
        validated = _parse_value(_CANONICAL_ENCODER.encode(_envelope_value(envelope)))
        return _CANONICAL_ENCODER.encode(_envelope_value(validated))
    except Exception:
        raise ThinkingEnvelopeValidationError(_INVALID_MESSAGE) from None


def normalize_thinking_history_policy(value: object) -> ThinkingHistoryPolicy:
    """Normalize nullable durable preference values without raising."""
    if value is None or value == "":
        return "auto"
    if type(value) is str and value in {"auto", "include", "exclude"}:
        return cast(ThinkingHistoryPolicy, value)
    return "auto"
