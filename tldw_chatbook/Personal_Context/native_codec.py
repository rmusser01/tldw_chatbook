"""Bounded native profile decoding; validation alone never grants V2 use."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ValidationError
from tldw_profile_core import enums, models, payloads
from tldw_profile_core.canonical import canonical_bytes, parse_json_integer

NativeProfileKind = Literal["manifest", "scope", "record", "proposal"]
_MAX_BYTES = 262144
_MAX_NODES = 4096
_MAX_DEPTH = 20
_V1_MODELS = {
    "manifest": models.ProfileManifest,
    "scope": models.ProfileScope,
    "record": models.ProfileRecord,
    "proposal": models.ProfileProposal,
}
_KNOWN_MODELS = frozenset(
    (
        *_V1_MODELS.values(),
        models.SemanticKey,
        models.ProfileControls,
        models.ProfileProvenance,
        payloads.IdentityPayload,
        payloads.PreferencePayload,
        payloads.RelationshipPayload,
        payloads.CorrectionPayload,
        payloads.ConstraintPayload,
        payloads.GoalPayload,
        payloads.ConventionPayload,
        payloads.WorkingContextPayload,
        payloads.LegacyUnclassifiedPayload,
    )
)
_KNOWN_ENUMS = frozenset(
    (
        models.ActorType,
        models.ProvenanceSource,
        enums.RecordKind,
        enums.RecordState,
        enums.ScopeKind,
        enums.SyncMode,
        enums.AgentVisibility,
        enums.ProposalOperation,
        enums.ProposalState,
    )
)


class NativeProfileDecodeError(ValueError):
    """Reject unsupported input without echoing profile content."""

    def __init__(self) -> None:
        super().__init__("personal_context_payload_invalid")


class NativeV1ProfileCorruptionError(NativeProfileDecodeError):
    """Identify known V1 data damage without exposing its validation input."""


def _known_v1_corruption(error: ValidationError) -> bool:
    """Keep ambiguous shapes, selectors and privacy vocabulary unavailable."""
    errors = error.errors(include_input=False, include_context=False, include_url=False)
    known_data_errors = {
        "value_error",
        "string_type",
        "string_too_short",
        "string_too_long",
        "int_type",
        "float_type",
        "bool_type",
        "finite_number",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "datetime_type",
        "datetime_parsing",
        "none_required",
        "literal_error",
    }

    def recognized(item):
        error_type, location = item["type"], item["loc"]
        if "schema_version" in location:
            return False
        provenance = location[1:] if location[:1] == ("proposed_record",) else location
        if error_type == "string_pattern_mismatch":
            return (
                len(provenance) == 3
                and provenance[:2] == ("provenance", "source_hashes")
                and type(provenance[2]) is int
            )
        if error_type == "too_long":
            return provenance in (
                ("provenance", "source_hashes"),
                ("provenance", "source_references"),
            )
        return error_type in known_data_errors and (
            error_type != "literal_error" or location[-1:] == ("polarity",)
        )

    return bool(errors) and all(recognized(item) for item in errors)


@dataclass(frozen=True, slots=True)
class DecodedNativeProfileObject:
    """Validated data, never a runtime admission token."""

    kind: NativeProfileKind
    schema_version: int
    value: BaseModel = field(repr=False)
    canonical: bytes = field(repr=False)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise NativeProfileDecodeError()
        result[key] = value
    return result


def _reject_nonfinite(_value):
    raise NativeProfileDecodeError()


def _detach(value: object) -> Any:
    nodes = 0

    def walk(item: object, depth: int):
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise NativeProfileDecodeError()
        cls = type(item)
        if any(cls is model for model in _KNOWN_MODELS):
            state = object.__getattribute__(item, "__dict__")
            extra = object.__getattribute__(item, "__pydantic_extra__")
            if type(state) is not dict or extra is not None:
                raise NativeProfileDecodeError()
            if any(type(k) is not str or k not in cls.model_fields for k in state):
                raise NativeProfileDecodeError()
            return walk(state, depth + 1)
        if any(cls is enum for enum in _KNOWN_ENUMS):
            return walk(item.value, depth + 1)
        if cls is datetime:
            if type(item.tzinfo) is not timezone and type(item.tzinfo) is not ZoneInfo:
                raise NativeProfileDecodeError()
            return item
        if item is None or any(cls is scalar for scalar in (bool, int, str)):
            return item
        if cls is float and isfinite(item):
            return item
        if cls is dict:
            if any(type(k) is not str for k in item):
                raise NativeProfileDecodeError()
            return {k: walk(v, depth + 1) for k, v in item.items()}
        if cls is list or cls is tuple:
            return [walk(v, depth + 1) for v in item]
        raise NativeProfileDecodeError()

    return walk(value, 0)


def _kind_model(kind):
    if type(kind) is not str or kind not in _V1_MODELS:
        raise NativeProfileDecodeError()
    return _V1_MODELS[kind]


def _decode(kind, raw):
    v1_model = _kind_model(kind)
    if type(raw) is not bytes and type(raw) is not str:
        raise NativeProfileDecodeError()
    encoded = raw if type(raw) is bytes else raw.encode("utf-8", errors="strict")
    if len(encoded) > _MAX_BYTES:
        raise NativeProfileDecodeError()
    text = encoded.decode("utf-8", errors="strict")
    body = _detach(
        json.loads(
            text, object_pairs_hook=_unique_pairs, parse_constant=_reject_nonfinite
        )
    )
    if type(body) is not dict:
        raise NativeProfileDecodeError()
    version = parse_json_integer(body.get("schema_version", 1))
    if version == 1:
        try:
            validated = v1_model.model_validate(body)
        except ValidationError as error:
            if _known_v1_corruption(error):
                raise NativeV1ProfileCorruptionError() from None
            raise
        canonical = canonical_bytes(validated)
    elif version == 2:
        from tldw_profile_core.v2_contract import canonical_v2_bytes, validate_v2_json
        from tldw_profile_core.v2_models import (
            ProfileManifestV2,
            ProfileProposalV2,
            ProfileRecordV2,
        )

        expected = {
            "manifest": ProfileManifestV2,
            "record": ProfileRecordV2,
            "proposal": ProfileProposalV2,
        }.get(kind)
        if expected is None:
            raise NativeProfileDecodeError()
        validated = validate_v2_json(encoded)
        if type(validated) is not expected:
            raise NativeProfileDecodeError()
        canonical = canonical_v2_bytes(validated)
    else:
        raise NativeProfileDecodeError()
    return DecodedNativeProfileObject(kind, version, validated, canonical)


def decode_native_profile(
    kind: NativeProfileKind, raw: bytes | str
) -> DecodedNativeProfileObject:
    """Validate exact raw native input with fixed bounds and version dispatch.

    Args:
        kind: Expected canonical object kind.
        raw: Exact UTF-8 bytes or a built-in string.

    Returns:
        Fresh validated data and its canonical bytes; no authorization.

    Raises:
        NativeProfileDecodeError: Malformed, unsupported or unbounded input.
    """
    try:
        return _decode(kind, raw)
    except NativeV1ProfileCorruptionError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError, ImportError):
        raise NativeProfileDecodeError() from None


def native_v1_bytes(
    kind: NativeProfileKind, value: BaseModel | Mapping[str, Any]
) -> bytes:
    """Freshly validate complete exact V1 state before native encryption.

    Args:
        kind: Expected canonical kind.
        value: Exact known V1 models or plain built-in JSON state.

    Returns:
        Published V1 canonical bytes.

    Raises:
        NativeProfileDecodeError: Unsupported models/types/fields/version/semantics.
    """
    try:
        model = _kind_model(kind)
        if type(value) is not model and type(value) is not dict:
            raise NativeProfileDecodeError()
        body = _detach(value)
        if (
            type(body) is not dict
            or parse_json_integer(body.get("schema_version", 1)) != 1
        ):
            raise NativeProfileDecodeError()
        result = canonical_bytes(model.model_validate(body))
        if len(result) > _MAX_BYTES:
            raise NativeProfileDecodeError()
        return result
    except (ValueError, TypeError, OverflowError, RecursionError, ImportError):
        raise NativeProfileDecodeError() from None
