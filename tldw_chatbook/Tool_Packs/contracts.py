"""Pure contracts for the portable ``tldw.tool-pack/v1`` format.

The module owns no archive or filesystem behavior.  It provides bounded JSON
admission, immutable schema objects, and the destination-independent tool
fingerprint shared by later export and import stages.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from tldw_chatbook.MCP.hub_tool_catalog import HubTool


TOOL_PACK_SCHEMA = "tldw.tool-pack/v1"
TOOL_PROFILE_SCHEMA = "tldw.tool-profile/v1"
PROFILE_PATH = "profile/profile.json"

MAX_MANIFEST_BYTES = 256 * 1024
MAX_PROFILE_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 12
MAX_JSON_NODES = 50_000
MAX_JSON_STRING_BYTES = MAX_PROFILE_BYTES
MAX_TOOLS = 2_000
MAX_SERVERS = 256
MAX_FALLBACKS = 257
MAX_IDENTITY_BYTES = 512
MAX_PRODUCER_BYTES = 128
MAX_DISPLAY_CODEPOINTS = 200

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PORTABLE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_ERROR_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
Authority = Literal["mcp", "builtin"]
ToolState = Literal["allow", "ask", "deny"]
FallbackState = Literal["ask", "deny"]


def _admit_nfc_text(value: str, *, max_bytes: int, allow_empty: bool = False) -> str:
    if (not allow_empty and not value) or unicodedata.normalize("NFC", value) != value:
        raise ValueError("invalid text")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("invalid text")
    try:
        if len(value.encode("utf-8")) > max_bytes:
            raise ValueError("text too large")
    except UnicodeError:
        raise ValueError("invalid text") from None
    return value


class _StrictArchiveModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class _PortableFallbackModel(_StrictArchiveModel):
    authority: Authority
    server_key: str
    state: FallbackState

    @field_validator("server_key")
    @classmethod
    def _validate_server_key(cls, value: str) -> str:
        return _admit_nfc_text(value, max_bytes=MAX_IDENTITY_BYTES)

    @model_validator(mode="after")
    def _validate_authority_pair(self) -> "_PortableFallbackModel":
        if (self.authority == "builtin") != (self.server_key == "agent:builtin"):
            raise ValueError("invalid authority pair")
        return self


class _PortableToolRuleModel(_StrictArchiveModel):
    authority: Authority
    server_key: str
    tool_name: str
    state: ToolState
    contract_sha256: str | None = None

    @field_validator("server_key", "tool_name")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        return _admit_nfc_text(value, max_bytes=MAX_IDENTITY_BYTES)

    @field_validator("contract_sha256")
    @classmethod
    def _validate_contract_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_RE.fullmatch(value) is None:
            raise ValueError("invalid digest")
        return value

    @model_validator(mode="after")
    def _validate_rule_relationships(self) -> "_PortableToolRuleModel":
        supplied = "contract_sha256" in self.model_fields_set
        if self.server_key == "*" or (
            (self.authority == "builtin") != (self.server_key == "agent:builtin")
        ):
            raise ValueError("invalid authority pair")
        if self.state == "deny":
            if supplied and self.contract_sha256 is None:
                raise ValueError("explicit null digest")
        elif not supplied or self.contract_sha256 is None:
            raise ValueError("missing digest")
        return self


class _ToolProfilePayloadModel(_StrictArchiveModel):
    schema_value: Literal[TOOL_PROFILE_SCHEMA] = Field(alias="schema")
    fallbacks: list[_PortableFallbackModel] = Field(max_length=MAX_FALLBACKS)
    tools: list[_PortableToolRuleModel] = Field(max_length=MAX_TOOLS)


class _ToolPackProducerModel(_StrictArchiveModel):
    name: str
    version: str

    @field_validator("name", "version")
    @classmethod
    def _validate_producer_text(cls, value: str) -> str:
        return _admit_nfc_text(value, max_bytes=MAX_PRODUCER_BYTES)


class _ToolPackProfileModel(_StrictArchiveModel):
    suggested_id: str
    display_name: str
    payload: Literal[PROFILE_PATH]

    @field_validator("suggested_id")
    @classmethod
    def _validate_suggested_id(cls, value: str) -> str:
        value = _admit_nfc_text(value, max_bytes=128)
        if (
            _PORTABLE_ID_RE.fullmatch(value) is None
            or value == "default"
            or value.startswith("ws-")
        ):
            raise ValueError("invalid portable id")
        return value

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str) -> str:
        value = _admit_nfc_text(value, max_bytes=MAX_PROFILE_BYTES)
        if not value.strip() or len(value) > MAX_DISPLAY_CODEPOINTS:
            raise ValueError("invalid display name")
        return value


class _ToolPackFileModel(_StrictArchiveModel):
    path: Literal[PROFILE_PATH]
    size: int = Field(gt=0, le=MAX_PROFILE_BYTES)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("invalid digest")
        return value


class _ToolPackManifestModel(_StrictArchiveModel):
    schema_value: Literal[TOOL_PACK_SCHEMA] = Field(alias="schema")
    producer: _ToolPackProducerModel
    required_features: list[str]
    profile: _ToolPackProfileModel
    files: list[_ToolPackFileModel] = Field(min_length=1, max_length=1)
    content_digest: str

    @field_validator("content_digest")
    @classmethod
    def _validate_content_digest(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("invalid digest")
        return value


def _admit_fallback_model(
    raw: object, *, operation: str, category: str
) -> _PortableFallbackModel:
    if type(raw) is not dict:
        raise _error(operation, category)
    try:
        return _PortableFallbackModel.model_validate(raw)
    except ValidationError:
        raise _error(operation, category) from None


def _admit_rule_model(
    raw: object, *, operation: str, category: str
) -> _PortableToolRuleModel:
    if type(raw) is not dict:
        raise _error(operation, category)
    try:
        return _PortableToolRuleModel.model_validate(raw)
    except ValidationError:
        raise _error(operation, category) from None


def _admit_profile_model(
    raw: object, *, operation: str, category: str
) -> _ToolProfilePayloadModel:
    if type(raw) is not dict:
        raise _error(operation, category)
    if raw.get("schema") != TOOL_PROFILE_SCHEMA:
        raise _error(operation, "schema_unsupported")
    try:
        return _ToolProfilePayloadModel.model_validate(raw)
    except ValidationError as error:
        if any(item["type"] == "too_long" for item in error.errors()):
            raise _error(operation, "too_large") from None
        raise _error(operation, category) from None


def _admit_manifest_model(
    raw: object, *, operation: str, category: str
) -> _ToolPackManifestModel:
    if type(raw) is not dict:
        raise _error(operation, category)
    if raw.get("schema") != TOOL_PACK_SCHEMA:
        raise _error(operation, "schema_unsupported")
    required = raw.get("required_features")
    if type(required) is list and required:
        raise _error(operation, "feature_unsupported")
    try:
        return _ToolPackManifestModel.model_validate(raw)
    except ValidationError:
        raise _error(operation, category) from None


_ERROR_CATEGORIES = {
    "export": frozenset(
        {
            "profile_unavailable",
            "profile_invalid",
            "store_invalid",
            "inventory_incomplete",
            "too_large",
            "destination_invalid",
            "destination_changed",
            "cancelled",
            "publication_unsupported",
            "publication_failed",
            "durability_uncertain",
        }
    ),
    "import": frozenset(
        {
            "archive_invalid",
            "schema_unsupported",
            "feature_unsupported",
            "manifest_invalid",
            "inventory_invalid",
            "payload_invalid",
            "identity_duplicate",
            "mapping_invalid",
            "too_large",
            "review_stale",
            "capacity_exceeded",
            "store_invalid",
            "store_changed",
            "destination_referenced",
            "activation_failed",
            "activation_uncertain",
        }
    ),
    "bind": frozenset(
        {
            "confirmation_required",
            "confirmation_stale",
            "confirmation_expired",
            "confirmation_invalid",
            "lifecycle_invalid",
            "binding_uncertain",
        }
    ),
    "remove": frozenset(
        {"referenced", "in_use", "non_removable", "stale", "outcome_uncertain"}
    ),
}
_VALIDATION_FALLBACK = {
    "export": "profile_invalid",
    "import": "payload_invalid",
    "bind": "confirmation_invalid",
    "remove": "non_removable",
}


class ToolPackError(ValueError):
    """Path-free Tool Pack failure with a stable public error code."""

    def __init__(self, operation: str, category: str) -> None:
        if (
            type(operation) is not str
            or not _ERROR_TOKEN_RE.fullmatch(operation)
            or operation not in _ERROR_CATEGORIES
        ):
            operation = "contract"
        if operation == "contract":
            operation = "import"
        if (
            type(category) is not str
            or not _ERROR_TOKEN_RE.fullmatch(category)
            or category not in _ERROR_CATEGORIES[operation]
        ):
            category = _VALIDATION_FALLBACK[operation]
        self.operation = operation
        self.category = category
        super().__init__(f"tool_pack.{operation}.{category}")


def _error(operation: str, category: str) -> ToolPackError:
    return ToolPackError(operation, category)


def _nfc_text(
    value: object,
    *,
    operation: str,
    category: str,
    max_bytes: int,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise _error(operation, category)
    if unicodedata.normalize("NFC", value) != value or _contains_surrogate(value):
        raise _error(operation, category)
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise _error(operation, category) from None
    if len(encoded) > max_bytes:
        raise _error(operation, category)
    return value


def _identity(value: object, *, operation: str, category: str) -> str:
    return _nfc_text(
        value,
        operation=operation,
        category=category,
        max_bytes=MAX_IDENTITY_BYTES,
    )


def _authority(value: object, *, operation: str, category: str) -> Authority:
    if type(value) is not str or value not in {"mcp", "builtin"}:
        raise _error(operation, category)
    return value  # type: ignore[return-value]


def _state(value: object, *, operation: str, category: str) -> ToolState:
    if type(value) is not str or value not in {"allow", "ask", "deny"}:
        raise _error(operation, category)
    return value  # type: ignore[return-value]


def _sha256(
    value: object, *, operation: str, category: str, optional: bool = False
) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise _error(operation, category)
    return value


def _validate_server_authority(
    authority: Authority,
    server_key: str,
    *,
    fallback: bool,
    operation: str,
    category: str,
) -> None:
    if authority == "builtin" and server_key != "agent:builtin":
        raise _error(operation, category)
    if authority == "mcp" and server_key == "agent:builtin":
        raise _error(operation, category)
    if not fallback and server_key == "*":
        raise _error(operation, category)


@dataclass(frozen=True, slots=True)
class PortableFallback:
    """One safe future-tool fallback in a portable profile."""

    authority: Authority
    server_key: str
    state: FallbackState

    def __post_init__(self) -> None:
        _validate_fallback_fields(
            self.authority,
            self.server_key,
            self.state,
            operation="import",
            category="payload_invalid",
        )

    @classmethod
    def from_dict(
        cls,
        raw: object,
        *,
        operation: str = "import",
        category: str = "payload_invalid",
    ) -> PortableFallback:
        model = _admit_fallback_model(raw, operation=operation, category=category)
        authority, server_key, state = _validate_fallback_fields(
            model.authority,
            model.server_key,
            model.state,
            operation=operation,
            category=category,
        )
        return cls(authority, server_key, state)

    def to_dict(self) -> dict[str, object]:
        return {
            "authority": self.authority,
            "server_key": self.server_key,
            "state": self.state,
        }


def _validate_fallback_fields(
    raw_authority: object,
    raw_server_key: object,
    raw_state: object,
    *,
    operation: str,
    category: str,
) -> tuple[Authority, str, FallbackState]:
    authority = _authority(raw_authority, operation=operation, category=category)
    server_key = _identity(raw_server_key, operation=operation, category=category)
    if type(raw_state) is not str or raw_state not in {"ask", "deny"}:
        raise _error(operation, category)
    _validate_server_authority(
        authority,
        server_key,
        fallback=True,
        operation=operation,
        category=category,
    )
    return authority, server_key, raw_state  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PortableToolRule:
    """One exact permission identity and its reviewed portable contract."""

    authority: Authority
    server_key: str
    tool_name: str
    state: ToolState
    contract_sha256: str | None

    def __post_init__(self) -> None:
        _validate_rule_fields(
            self.authority,
            self.server_key,
            self.tool_name,
            self.state,
            self.contract_sha256,
            fingerprint_present=self.contract_sha256 is not None,
            operation="import",
            category="payload_invalid",
        )

    @classmethod
    def from_dict(
        cls,
        raw: object,
        *,
        operation: str = "import",
        category: str = "payload_invalid",
    ) -> PortableToolRule:
        model = _admit_rule_model(raw, operation=operation, category=category)
        fingerprint_present = "contract_sha256" in model.model_fields_set
        fields = _validate_rule_fields(
            model.authority,
            model.server_key,
            model.tool_name,
            model.state,
            model.contract_sha256,
            fingerprint_present=fingerprint_present,
            operation=operation,
            category=category,
        )
        return cls(*fields)

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "authority": self.authority,
            "server_key": self.server_key,
            "tool_name": self.tool_name,
            "state": self.state,
        }
        if self.contract_sha256 is not None:
            value["contract_sha256"] = self.contract_sha256
        return value


def _validate_rule_fields(
    raw_authority: object,
    raw_server_key: object,
    raw_tool_name: object,
    raw_state: object,
    raw_contract_sha256: object,
    *,
    fingerprint_present: bool = False,
    operation: str,
    category: str,
) -> tuple[Authority, str, str, ToolState, str | None]:
    authority = _authority(raw_authority, operation=operation, category=category)
    server_key = _identity(raw_server_key, operation=operation, category=category)
    tool_name = _identity(raw_tool_name, operation=operation, category=category)
    state = _state(raw_state, operation=operation, category=category)
    if raw_contract_sha256 is None:
        if state != "deny" or fingerprint_present:
            raise _error(operation, category)
        contract_sha256 = None
    else:
        contract_sha256 = _sha256(
            raw_contract_sha256,
            operation=operation,
            category=category,
        )
    if state != "deny" and (not fingerprint_present or contract_sha256 is None):
        raise _error(operation, category)
    _validate_server_authority(
        authority,
        server_key,
        fallback=False,
        operation=operation,
        category=category,
    )
    return authority, server_key, tool_name, state, contract_sha256


@dataclass(frozen=True, slots=True)
class ToolProfilePayload:
    """One complete flattened portable policy payload."""

    schema: str
    fallbacks: tuple[PortableFallback, ...]
    tools: tuple[PortableToolRule, ...]

    def __post_init__(self) -> None:
        _validate_profile_values(
            self.schema,
            self.fallbacks,
            self.tools,
            operation="import",
            category="payload_invalid",
        )

    @classmethod
    def from_dict(
        cls,
        raw: object,
        *,
        operation: str = "import",
        category: str = "payload_invalid",
    ) -> ToolProfilePayload:
        model = _admit_profile_model(raw, operation=operation, category=category)
        fallbacks = tuple(
            PortableFallback(item.authority, item.server_key, item.state)
            for item in model.fallbacks
        )
        tools = tuple(
            PortableToolRule(
                item.authority,
                item.server_key,
                item.tool_name,
                item.state,
                item.contract_sha256,
            )
            for item in model.tools
        )
        _validate_profile_values(
            TOOL_PROFILE_SCHEMA,
            fallbacks,
            tools,
            operation=operation,
            category=category,
        )
        return cls(TOOL_PROFILE_SCHEMA, fallbacks, tools)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "fallbacks": [item.to_dict() for item in self.fallbacks],
            "tools": [item.to_dict() for item in self.tools],
        }


def _validate_profile_values(
    schema: object,
    fallbacks: object,
    tools: object,
    *,
    operation: str,
    category: str,
) -> None:
    if (
        schema != TOOL_PROFILE_SCHEMA
        or type(fallbacks) is not tuple
        or type(tools) is not tuple
        or len(fallbacks) > MAX_FALLBACKS
        or len(tools) > MAX_TOOLS
        or any(type(item) is not PortableFallback for item in fallbacks)
        or any(type(item) is not PortableToolRule for item in tools)
    ):
        raise _error(operation, category)

    fallback_ids = [(item.authority, item.server_key) for item in fallbacks]
    tool_ids = [(item.authority, item.server_key, item.tool_name) for item in tools]
    if _has_exact_or_casefold_collision(
        fallback_ids
    ) or _has_exact_or_casefold_collision(tool_ids):
        raise _error(operation, "identity_duplicate")

    server_spellings: dict[tuple[str, str], tuple[str, str]] = {}
    for authority, server_key in fallback_ids + [item[:2] for item in tool_ids]:
        folded = (authority.casefold(), server_key.casefold())
        previous = server_spellings.setdefault(folded, (authority, server_key))
        if previous != (authority, server_key):
            raise _error(operation, "identity_duplicate")

    if fallback_ids != sorted(fallback_ids) or tool_ids != sorted(tool_ids):
        raise _error(operation, category)
    fallback_set = set(fallback_ids)
    if ("mcp", "*") not in fallback_set or (
        "builtin",
        "agent:builtin",
    ) not in fallback_set:
        raise _error(operation, category)
    if any(
        (authority, server_key) not in fallback_set
        for authority, server_key, _ in tool_ids
    ):
        raise _error(operation, category)
    distinct_servers = fallback_set - {("mcp", "*")}
    if len(distinct_servers) > MAX_SERVERS:
        raise _error(operation, "too_large")


def _has_exact_or_casefold_collision(identities: list[tuple[str, ...]]) -> bool:
    seen_exact: set[tuple[str, ...]] = set()
    seen_folded: set[tuple[str, ...]] = set()
    for identity in identities:
        folded = tuple(part.casefold() for part in identity)
        if identity in seen_exact or folded in seen_folded:
            return True
        seen_exact.add(identity)
        seen_folded.add(folded)
    return False


@dataclass(frozen=True, slots=True)
class ToolPackManifest:
    """Validated manifest fields flattened into an immutable value object."""

    schema: str
    producer_name: str
    producer_version: str
    required_features: tuple[str, ...]
    suggested_id: str
    display_name: str
    payload_path: str
    payload_size: int
    payload_sha256: str
    content_digest: str

    def __post_init__(self) -> None:
        _validate_manifest_fields(
            self.schema,
            self.producer_name,
            self.producer_version,
            self.required_features,
            self.suggested_id,
            self.display_name,
            self.payload_path,
            self.payload_size,
            self.payload_sha256,
            self.content_digest,
            operation="import",
            category="manifest_invalid",
        )

    @classmethod
    def from_dict(
        cls,
        raw: object,
        *,
        operation: str = "import",
        category: str = "manifest_invalid",
    ) -> ToolPackManifest:
        model = _admit_manifest_model(raw, operation=operation, category=category)
        try:
            encoded_size = len(
                canonical_json_bytes(model.model_dump(mode="python", by_alias=True))
            )
        except ToolPackError:
            raise _error(operation, category) from None
        if encoded_size > MAX_MANIFEST_BYTES:
            raise _error(operation, "too_large")

        producer_name = _nfc_text(
            model.producer.name,
            operation=operation,
            category=category,
            max_bytes=MAX_PRODUCER_BYTES,
        )
        producer_version = _nfc_text(
            model.producer.version,
            operation=operation,
            category=category,
            max_bytes=MAX_PRODUCER_BYTES,
        )
        required = model.required_features
        if required:
            raise _error(operation, "feature_unsupported")

        suggested_id = _portable_id(
            model.profile.suggested_id, operation=operation, category=category
        )
        display_name = _nfc_text(
            model.profile.display_name,
            operation=operation,
            category=category,
            max_bytes=MAX_PROFILE_BYTES,
        )
        if not display_name.strip() or len(display_name) > MAX_DISPLAY_CODEPOINTS:
            raise _error(operation, category)
        if model.profile.payload != PROFILE_PATH:
            raise _error(operation, category)

        file_entry = model.files[0]
        payload_size = file_entry.size
        if (
            file_entry.path != PROFILE_PATH
            or type(payload_size) is not int
            or not 0 < payload_size <= MAX_PROFILE_BYTES
        ):
            raise _error(operation, category)
        payload_sha256 = _sha256(
            file_entry.sha256, operation=operation, category=category
        )
        content_digest = _sha256(
            model.content_digest, operation=operation, category=category
        )
        assert payload_sha256 is not None and content_digest is not None
        return cls(
            TOOL_PACK_SCHEMA,
            producer_name,
            producer_version,
            (),
            suggested_id,
            display_name,
            PROFILE_PATH,
            payload_size,
            payload_sha256,
            content_digest,
        )

    def to_dict(self, *, include_content_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": self.schema,
            "producer": {"name": self.producer_name, "version": self.producer_version},
            "required_features": list(self.required_features),
            "profile": {
                "suggested_id": self.suggested_id,
                "display_name": self.display_name,
                "payload": self.payload_path,
            },
            "files": [
                {
                    "path": self.payload_path,
                    "size": self.payload_size,
                    "sha256": self.payload_sha256,
                }
            ],
        }
        if include_content_digest:
            value["content_digest"] = self.content_digest
        return value


def _validate_manifest_fields(
    schema: object,
    producer_name: object,
    producer_version: object,
    required_features: object,
    suggested_id: object,
    display_name: object,
    payload_path: object,
    payload_size: object,
    payload_sha256: object,
    content_digest: object,
    *,
    operation: str,
    category: str,
) -> None:
    if schema != TOOL_PACK_SCHEMA:
        raise _error(operation, "schema_unsupported")
    _nfc_text(
        producer_name,
        operation=operation,
        category=category,
        max_bytes=MAX_PRODUCER_BYTES,
    )
    _nfc_text(
        producer_version,
        operation=operation,
        category=category,
        max_bytes=MAX_PRODUCER_BYTES,
    )
    if type(required_features) is not tuple or any(
        type(item) is not str for item in required_features
    ):
        raise _error(operation, category)
    if required_features:
        raise _error(operation, "feature_unsupported")
    _portable_id(suggested_id, operation=operation, category=category)
    checked_display = _nfc_text(
        display_name,
        operation=operation,
        category=category,
        max_bytes=MAX_PROFILE_BYTES,
    )
    if not checked_display.strip() or len(checked_display) > MAX_DISPLAY_CODEPOINTS:
        raise _error(operation, category)
    if payload_path != PROFILE_PATH:
        raise _error(operation, category)
    if type(payload_size) is not int or not 0 < payload_size <= MAX_PROFILE_BYTES:
        raise _error(operation, category)
    _sha256(payload_sha256, operation=operation, category=category)
    _sha256(content_digest, operation=operation, category=category)


def _portable_id(value: object, *, operation: str, category: str) -> str:
    identifier = _nfc_text(
        value,
        operation=operation,
        category=category,
        max_bytes=128,
    )
    if (
        not _PORTABLE_ID_RE.fullmatch(identifier)
        or identifier == "default"
        or identifier.startswith("ws-")
    ):
        raise _error(operation, category)
    return identifier


@dataclass(frozen=True, slots=True)
class ToolPackDocument:
    """A manifest and its exact, verified profile payload."""

    manifest: ToolPackManifest
    profile: ToolProfilePayload

    def __post_init__(self) -> None:
        if (
            type(self.manifest) is not ToolPackManifest
            or type(self.profile) is not ToolProfilePayload
        ):
            raise _error("import", "payload_invalid")
        _validate_document_relationships(
            self.manifest, self.profile, operation="import"
        )

    @classmethod
    def from_dicts(
        cls,
        manifest_raw: object,
        profile_raw: object,
        *,
        profile_bytes: bytes,
        operation: str = "import",
    ) -> ToolPackDocument:
        if type(profile_bytes) is not bytes:
            raise _error(operation, "payload_invalid")
        if len(profile_bytes) > MAX_PROFILE_BYTES:
            raise _error(operation, "too_large")
        manifest = ToolPackManifest.from_dict(manifest_raw, operation=operation)
        profile = ToolProfilePayload.from_dict(profile_raw, operation=operation)
        try:
            expected_bytes = canonical_json_bytes(profile.to_dict())
        except ToolPackError:
            raise _error(operation, "payload_invalid") from None
        if profile_bytes != expected_bytes:
            raise _error(operation, "payload_invalid")
        _validate_document_relationships(manifest, profile, operation=operation)
        return cls(manifest, profile)


def _validate_document_relationships(
    manifest: ToolPackManifest,
    profile: ToolProfilePayload,
    *,
    operation: str,
) -> None:
    profile_bytes = canonical_json_bytes(profile.to_dict(), operation=operation)
    if (
        manifest.payload_size != len(profile_bytes)
        or manifest.payload_sha256 != hashlib.sha256(profile_bytes).hexdigest()
    ):
        raise _error(operation, "manifest_invalid")
    preimage = (
        TOOL_PACK_SCHEMA.encode("ascii")
        + b"\0"
        + canonical_json_bytes(
            manifest.to_dict(include_content_digest=False), operation=operation
        )
        + b"\0"
        + profile_bytes
    )
    if manifest.content_digest != hashlib.sha256(preimage).hexdigest():
        raise _error(operation, "manifest_invalid")


def canonical_json_bytes(value: object, *, operation: str = "import") -> bytes:
    """Return deterministic, NFC-normalized, LF-terminated canonical JSON."""

    try:
        normalized = _normalize_json_tree(value)
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (encoded + "\n").encode("utf-8")
    except ToolPackError:
        raise _error(operation, "payload_invalid") from None
    except (
        MemoryError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        raise _error(operation, "payload_invalid") from None


def strict_json_object(
    data: bytes,
    *,
    category: str,
    max_bytes: int,
    operation: str = "import",
) -> dict[str, Any]:
    """Decode one bounded strict UTF-8 JSON object.

    Duplicate keys and non-finite constants are rejected during parsing.  Every
    other implementation exception is collapsed to the caller-supplied stable
    category; byte admission has the distinct public ``too_large`` category.
    """

    if type(data) is not bytes or type(max_bytes) is not int or max_bytes < 0:
        raise _error(operation, category)
    if len(data) > max_bytes:
        raise _error(operation, "too_large")

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise ValueError("non-finite number")

    try:
        text = data.decode("utf-8", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
        if type(parsed) is not dict:
            raise ValueError("root is not an object")
        _inspect_json_tree(parsed, require_nfc=True)
        return parsed
    except ToolPackError:
        raise _error(operation, category) from None
    except (
        MemoryError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        raise _error(operation, category) from None


def _normalize_json_tree(value: object) -> object:
    count = 0
    active: set[int] = set()

    def visit(item: object, depth: int) -> object:
        nonlocal count
        count += 1
        if depth > MAX_JSON_DEPTH or count > MAX_JSON_NODES:
            raise _error("contract", "payload_invalid")
        if item is None or type(item) in {bool, int}:
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise _error("contract", "payload_invalid")
            return item
        if type(item) is str:
            return _normalized_string(item)
        if type(item) not in {list, dict}:
            raise _error("contract", "payload_invalid")
        marker = id(item)
        if marker in active:
            raise _error("contract", "payload_invalid")
        active.add(marker)
        try:
            if type(item) is list:
                return [visit(child, depth + 1) for child in item]
            normalized: dict[str, object] = {}
            for key, child in item.items():
                if type(key) is not str:
                    raise _error("contract", "payload_invalid")
                normalized_key = _normalized_string(key)
                if normalized_key in normalized:
                    raise _error("contract", "payload_invalid")
                normalized[normalized_key] = visit(child, depth + 1)
            return normalized
        finally:
            active.remove(marker)

    return visit(value, 1)


def _inspect_json_tree(value: object, *, require_nfc: bool) -> None:
    count = 0
    active: set[int] = set()

    def visit(item: object, depth: int) -> None:
        nonlocal count
        count += 1
        if depth > MAX_JSON_DEPTH or count > MAX_JSON_NODES:
            raise ValueError("JSON tree exceeds bounds")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("non-finite number")
            return
        if type(item) is str:
            _checked_string(item, require_nfc=require_nfc)
            return
        if type(item) not in {list, dict}:
            raise ValueError("unsupported JSON value")
        marker = id(item)
        if marker in active:
            raise ValueError("recursive JSON tree")
        active.add(marker)
        try:
            if type(item) is list:
                for child in item:
                    visit(child, depth + 1)
                return
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("non-string key")
                _checked_string(key, require_nfc=require_nfc)
                visit(child, depth + 1)
        finally:
            active.remove(marker)

    visit(value, 1)


def _checked_string(value: str, *, require_nfc: bool) -> None:
    if _contains_surrogate(value):
        raise ValueError("surrogate")
    if require_nfc and unicodedata.normalize("NFC", value) != value:
        raise ValueError("non-NFC string")
    if len(value.encode("utf-8")) > MAX_JSON_STRING_BYTES:
        raise ValueError("string exceeds bound")


def _normalized_string(value: str) -> str:
    if _contains_surrogate(value):
        raise _error("contract", "payload_invalid")
    normalized = unicodedata.normalize("NFC", value)
    try:
        if len(normalized.encode("utf-8")) > MAX_JSON_STRING_BYTES:
            raise _error("contract", "payload_invalid")
    except UnicodeError:
        raise _error("contract", "payload_invalid") from None
    return normalized


def _contains_surrogate(value: str) -> bool:
    return any("\ud800" <= character <= "\udfff" for character in value)


def portable_contract_sha256(
    tool: HubTool,
    *,
    risk_tags: Iterable[str] | None = None,
    operation: str = "import",
) -> str:
    """Hash the destination-independent portable tool contract preimage."""

    try:
        tool_name = _identity(
            tool.name, operation="contract", category="payload_invalid"
        )
        if type(tool.description) is not str:
            raise _error("contract", "payload_invalid")
        description = unicodedata.normalize(
            "NFC", tool.description.replace("\r\n", "\n").replace("\r", "\n")
        )
        source_tags = tool.tags if risk_tags is None else risk_tags
        normalized_tags: set[str] = set()
        for tag in source_tags:
            if type(tag) is not str:
                raise _error("contract", "payload_invalid")
            normalized = unicodedata.normalize("NFC", tag)
            _identity(normalized, operation="contract", category="payload_invalid")
            normalized_tags.add(normalized)
        preimage = {
            "tool_name": tool_name,
            "description": description,
            "input_schema": tool.input_schema,
            "policy_risk_tags": sorted(normalized_tags),
        }
        return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()
    except ToolPackError:
        raise _error(operation, "payload_invalid") from None
    except (
        MemoryError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        raise _error(operation, "payload_invalid") from None


def validate_tool_pack_manifest(
    raw: object, *, operation: str = "import"
) -> ToolPackManifest:
    """Validate and materialize one exact manifest object."""

    return ToolPackManifest.from_dict(raw, operation=operation)


def validate_tool_profile_payload(
    raw: object, *, operation: str = "import"
) -> ToolProfilePayload:
    """Validate and materialize one exact profile payload object."""

    return ToolProfilePayload.from_dict(raw, operation=operation)


def validate_tool_pack_document(
    manifest_raw: object,
    profile_raw: object,
    *,
    profile_bytes: bytes,
    operation: str = "import",
) -> ToolPackDocument:
    """Validate manifest/profile relationships and exact payload bytes."""

    return ToolPackDocument.from_dicts(
        manifest_raw, profile_raw, profile_bytes=profile_bytes, operation=operation
    )


__all__ = [
    "MAX_FALLBACKS",
    "MAX_JSON_DEPTH",
    "MAX_JSON_NODES",
    "MAX_MANIFEST_BYTES",
    "MAX_PROFILE_BYTES",
    "MAX_SERVERS",
    "MAX_TOOLS",
    "PortableFallback",
    "PortableToolRule",
    "ToolPackDocument",
    "ToolPackError",
    "ToolPackManifest",
    "ToolProfilePayload",
    "canonical_json_bytes",
    "portable_contract_sha256",
    "strict_json_object",
    "validate_tool_pack_document",
    "validate_tool_pack_manifest",
    "validate_tool_profile_payload",
]
