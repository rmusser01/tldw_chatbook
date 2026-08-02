"""Truthful capability contracts for local and server Prompt sources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .prompt_artifact_codec import decode_prompt_artifact


LOCAL_COMPILED_LANE_LIMIT = 20_000
LOCAL_DEFINITION_LIMIT = 256_000
LOCAL_REQUEST_LIMIT = 512_000


@dataclass(frozen=True)
class CanonicalJSONByteMeasurement:
    """Normalized description of the server's canonical JSON byte algorithm."""

    name: Literal["canonical_json_utf8_v1"]
    encoding: Literal["utf-8"]
    ensure_ascii: Literal[False]
    sort_keys: Literal[True]
    separators: tuple[Literal[","], Literal[":"]]


CANONICAL_JSON_UTF8_V1 = CanonicalJSONByteMeasurement(
    name="canonical_json_utf8_v1",
    encoding="utf-8",
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)


@dataclass(frozen=True)
class PromptSourceCapabilities:
    """Immutable, source-normalized Prompt capabilities and save limits."""

    backend: Literal["local", "server"]
    structured_kinds: frozenset[tuple[int, str]]
    artifact_types: frozenset[str]
    search: bool
    conditional_update: bool
    compiled_lane_limit: int
    definition_limit: int
    request_limit: int
    json_byte_measurement: CanonicalJSONByteMeasurement | None


class PromptCapabilityError(ValueError):
    """A selected Prompt source cannot honestly provide a requested capability."""

    def __init__(self, backend: str, capability: str) -> None:
        self.backend = backend
        self.capability = capability
        super().__init__(f"{backend} prompt source does not support {capability}.")


def canonical_json_utf8_size(value: Any) -> int:
    """Measure a decoded JSON value using ``canonical_json_utf8_v1``."""
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return len(serialized.encode("utf-8"))


def local_prompt_capabilities() -> PromptSourceCapabilities:
    """Return capabilities guaranteed by the in-process local Prompt stack."""
    return PromptSourceCapabilities(
        backend="local",
        structured_kinds=frozenset({(2, "block_prompt"), (2, "block_recipe")}),
        artifact_types=frozenset({"prompt", "recipe"}),
        search=True,
        conditional_update=True,
        compiled_lane_limit=LOCAL_COMPILED_LANE_LIMIT,
        definition_limit=LOCAL_DEFINITION_LIMIT,
        request_limit=LOCAL_REQUEST_LIMIT,
        json_byte_measurement=CANONICAL_JSON_UTF8_V1,
    )


def _plain_mapping(value: Any) -> Mapping[str, Any] | None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return value if isinstance(value, Mapping) else None


def _legacy_server_capabilities() -> PromptSourceCapabilities:
    return PromptSourceCapabilities(
        backend="server",
        structured_kinds=frozenset(),
        artifact_types=frozenset({"prompt"}),
        search=False,
        conditional_update=False,
        compiled_lane_limit=LOCAL_COMPILED_LANE_LIMIT,
        definition_limit=LOCAL_DEFINITION_LIMIT,
        request_limit=LOCAL_REQUEST_LIMIT,
        json_byte_measurement=None,
    )


def _normalize_structured_kinds(value: Any) -> frozenset[tuple[int, str]] | None:
    if not isinstance(value, list):
        return None
    normalized: set[tuple[int, str]] = set()
    for item in value:
        item_mapping = _plain_mapping(item)
        if item_mapping is None:
            return None
        version = item_mapping.get("schema_version")
        kind = item_mapping.get("kind")
        if type(version) is not int or not isinstance(kind, str) or not kind:
            return None
        normalized.add((version, kind))
    return frozenset(normalized)


def _normalize_artifact_types(value: Any) -> frozenset[str] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    normalized = frozenset(value)
    if not normalized.issubset({"prompt", "recipe"}):
        return None
    return normalized


def _normalize_measurement(value: Any) -> CanonicalJSONByteMeasurement | None:
    descriptor = _plain_mapping(value)
    if descriptor is None:
        return None
    separators = descriptor.get("separators")
    is_exact = (
        set(descriptor) == {
            "name",
            "encoding",
            "ensure_ascii",
            "sort_keys",
            "separators",
        }
        and type(descriptor.get("name")) is str
        and descriptor["name"] == "canonical_json_utf8_v1"
        and type(descriptor.get("encoding")) is str
        and descriptor["encoding"] == "utf-8"
        and type(descriptor.get("ensure_ascii")) is bool
        and descriptor["ensure_ascii"] is False
        and type(descriptor.get("sort_keys")) is bool
        and descriptor["sort_keys"] is True
        and type(separators) is list
        and len(separators) == 2
        and all(type(item) is str for item in separators)
        and separators == [",", ":"]
    )
    return CANONICAL_JSON_UTF8_V1 if is_exact else None


def _smaller_positive_limit(value: Any, fallback: int) -> int:
    if type(value) is not int or value <= 0:
        return fallback
    return min(value, fallback)


def normalize_server_prompt_capabilities(health: Any) -> PromptSourceCapabilities:
    """Normalize modern server health, failing closed on absent core metadata."""
    health_mapping = _plain_mapping(health)
    if health_mapping is None:
        return _legacy_server_capabilities()
    capabilities = _plain_mapping(health_mapping.get("capabilities"))
    if capabilities is None:
        return _legacy_server_capabilities()

    structured_kinds = _normalize_structured_kinds(
        capabilities.get("structured_kinds")
    )
    artifact_types = _normalize_artifact_types(capabilities.get("artifact_types"))
    if structured_kinds is None or artifact_types is None:
        return _legacy_server_capabilities()

    size_limits = _plain_mapping(capabilities.get("size_limits")) or {}
    measurement = _normalize_measurement(size_limits.get("json_byte_measurement"))
    return PromptSourceCapabilities(
        backend="server",
        structured_kinds=structured_kinds,
        artifact_types=artifact_types,
        search=capabilities.get("search") is True,
        # The current authenticated client update contract has no expected_version.
        conditional_update=False,
        compiled_lane_limit=_smaller_positive_limit(
            size_limits.get("compiled_lane_characters"), LOCAL_COMPILED_LANE_LIMIT
        ),
        definition_limit=_smaller_positive_limit(
            size_limits.get("definition_utf8_bytes"), LOCAL_DEFINITION_LIMIT
        ),
        request_limit=_smaller_positive_limit(
            size_limits.get("request_utf8_bytes"), LOCAL_REQUEST_LIMIT
        ),
        json_byte_measurement=measurement,
    )


def validate_console_artifact_payload(
    payload: dict[str, Any], capabilities: PromptSourceCapabilities
) -> dict[str, Any]:
    """Validate one Console block-v2 save without truncating or inferring support."""
    if payload.get("prompt_format") != "structured":
        raise PromptCapabilityError(capabilities.backend, "valid Console block artifact")
    decoded = decode_prompt_artifact(payload)
    raw_definition = decoded.raw_definition
    kind = raw_definition.get("kind") if raw_definition is not None else None
    version = payload.get("prompt_schema_version")
    pair = (version if type(version) is int else None, kind)
    if pair not in capabilities.structured_kinds:
        raise PromptCapabilityError(
            capabilities.backend, f"structured kind {pair!r}"
        )
    if payload.get("artifact_type", "prompt") not in capabilities.artifact_types:
        raise PromptCapabilityError(
            capabilities.backend,
            f"artifact type {payload.get('artifact_type', 'prompt')!r}",
        )
    if (
        decoded.state != "supported_v2"
        or raw_definition is None
        or decoded.definition is None
    ):
        raise PromptCapabilityError(capabilities.backend, "valid Console block artifact")
    if capabilities.json_byte_measurement != CANONICAL_JSON_UTF8_V1:
        raise PromptCapabilityError(
            capabilities.backend, "canonical JSON byte measurement"
        )

    for field, text in (
        ("system_prompt", decoded.compiled_system),
        ("user_prompt", decoded.compiled_user),
    ):
        if len(text) > capabilities.compiled_lane_limit:
            raise ValueError(
                f"{field} exceeds {capabilities.compiled_lane_limit} characters."
            )

    normalized_definition = {
        "schema_version": decoded.definition.schema_version,
        "kind": decoded.definition.kind,
        "lanes": [
            {
                "id": lane.id,
                "blocks": [
                    {
                        **{
                            "id": block.id,
                            "title": block.title,
                            "syntax": block.syntax,
                            "content": block.content,
                        },
                        **(
                            {"xml_tag": block.xml_tag}
                            if block.xml_tag is not None
                            else {}
                        ),
                        **(
                            {"mapping_hint": block.mapping_hint}
                            if block.mapping_hint is not None
                            else {}
                        ),
                    }
                    for block in lane.blocks
                ],
            }
            for lane in decoded.definition.lanes
        ],
    }
    normalized_payload = dict(payload)
    normalized_payload.update(
        {
            "artifact_type": decoded.artifact_type,
            "prompt_format": "structured",
            "prompt_schema_version": decoded.definition.schema_version,
            "prompt_definition": normalized_definition,
            "system_prompt": decoded.compiled_system,
            "user_prompt": decoded.compiled_user,
        }
    )
    definition_size = canonical_json_utf8_size(normalized_payload["prompt_definition"])
    if definition_size > capabilities.definition_limit:
        raise ValueError(
            "prompt_definition exceeds "
            f"{capabilities.definition_limit} UTF-8 bytes."
        )
    return normalized_payload


def validate_prompt_request_size(
    payload: Mapping[str, Any], capabilities: PromptSourceCapabilities
) -> None:
    """Reject an exact outgoing save mapping that exceeds the source limit."""
    request_size = canonical_json_utf8_size(payload)
    if request_size > capabilities.request_limit:
        raise ValueError(f"request exceeds {capabilities.request_limit} UTF-8 bytes.")
