"""Data-only citation source locator, policy, and authorization contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
import json
import math
import re
from types import MappingProxyType
from typing import Annotated, Any, Literal, Mapping, TypeVar
import unicodedata

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from .citation_trace_models import (
    EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX,
    EvidenceStorageMode,
)


LOCATOR_ENVELOPE_JSON_BYTES_MAX = 16 * 1024
SOURCE_OBSERVATION_JSON_BYTES_MAX = 8 * 1024
SOURCE_OBSERVATION_ERROR_CHARACTERS_MAX = 256
SOURCE_OBSERVATION_REQUEST_GENERATION_MAX = (1 << 63) - 1
AUTHORITY_IDS_PER_READ_AUTHORIZATION_MAX = 32
CURRENT_AUTHORITY_LOOKUP_TTL_MAX = timedelta(minutes=5)
INERT_LOCATOR_JSON_DEPTH_MAX = 32
INERT_LOCATOR_JSON_CONTAINERS_MAX = 256
INERT_LOCATOR_JSON_ITEMS_MAX = 2048
INERT_LOCATOR_JSON_KEY_UTF8_BYTES_MAX = 256
_DRIVE_PATH = re.compile(r"^[A-Za-z]:")
_SAFE_ERROR_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")


def _bounded_identifier(value: str) -> str:
    if not value:
        raise ValueError("identifier must not be empty")
    byte_count = len(value.encode("utf-8"))
    if byte_count > EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX:
        raise ValueError(
            f"identifier exceeds {EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX} UTF-8 bytes"
        )
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("identifier must not contain control characters")
    return value


def _source_root_identifier(value: str) -> str:
    value = _bounded_identifier(value)
    if (
        value.startswith(("~", ".", "/", "\\"))
        or _DRIVE_PATH.match(value)
        or any(separator in value for separator in ("/", "\\", ":"))
    ):
        raise ValueError("source_root_id must be an opaque identifier, not a path")
    return value


def _safe_relative_path(value: str) -> str:
    if not value:
        raise ValueError("relative_path must not be empty")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("relative_path must not contain control characters")
    if (
        value.startswith(("/", "\\", "~"))
        or _DRIVE_PATH.match(value)
        or "\\" in value
        or ":" in value
    ):
        raise ValueError("relative_path must be a safe POSIX relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("relative_path contains unsafe path semantics")
    return value


BoundedIdentifier = Annotated[str, AfterValidator(_bounded_identifier)]
SourceRootIdentifier = Annotated[str, AfterValidator(_source_root_identifier)]
SafeRelativePath = Annotated[str, AfterValidator(_safe_relative_path)]
CandidateJson = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
]


def _sanitized_error_code(value: str) -> str:
    if _SAFE_ERROR_CODE.fullmatch(value) is None:
        raise ValueError("error_code must be a sanitized status code")
    return value


SanitizedObservationErrorCode = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=SOURCE_OBSERVATION_ERROR_CHARACTERS_MAX,
    ),
    AfterValidator(_sanitized_error_code),
]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
    )


_ModelT = TypeVar("_ModelT", bound=_StrictFrozenModel)


class CanonicalSourceKind(str, Enum):
    """Allowlisted canonical source kinds represented by v1 locators."""

    MEDIA_DB = "media_db"
    NOTES = "notes"
    CHAT_HISTORY = "chat_history"
    CHARACTER_CARDS = "character_cards"
    WEB_CONTENT = "web_content"
    PROMPTS = "prompts"
    WORLD_BOOKS = "world_books"
    DICTIONARIES = "dictionaries"
    KANBAN = "kanban"
    SQL = "sql"
    CLAIMS = "claims"


class AuthorityScope(str, Enum):
    """Trusted scope that owns a source locator or read authorization."""

    LOCAL_PROFILE = "local_profile"
    AUTHENTICATED_TENANT = "authenticated_tenant"


class LocatorBindingState(str, Enum):
    """Whether locator-shaped data has current native authority."""

    NATIVE = "native"
    INERT_IMPORTED = "inert_imported"
    INERT_LEGACY = "inert_legacy"


class SourceCapability(str, Enum):
    """Independent actions gated by both policy and request authorization."""

    VIEW_SNAPSHOT = "view_snapshot"
    VIEW_SOURCE_IDENTITY = "view_source_identity"
    RESOLVE_CURRENT = "resolve_current"
    OPEN_NATIVE = "open_native"
    OPEN_EXTERNAL = "open_external"
    COMPARE = "compare"
    REFRESH_OBSERVATION = "refresh_observation"
    EXPORT = "export"


class CitationSourceAvailability(str, Enum):
    """Current reachability, independent of permission and content state."""

    AVAILABLE = "available"
    MISSING = "missing"
    OFFLINE = "offline"
    ERROR = "error"
    UNKNOWN = "unknown"


class CitationSourcePermission(str, Enum):
    """Current authorization, independent of source reachability."""

    ALLOWED = "allowed"
    DENIED = "denied"
    AUTHENTICATION_REQUIRED = "authentication_required"
    REVOKED = "revoked"
    UNKNOWN = "unknown"


class CitationContentState(str, Enum):
    """Comparison of the current item with the submitted snapshot."""

    UNCHANGED = "unchanged"
    CHANGED = "changed"
    UNKNOWN = "unknown"


class CitationLocationState(str, Enum):
    """Current location result without embedding a governed locator."""

    UNCHANGED = "unchanged"
    RELOCATED = "relocated"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"
    UNKNOWN = "unknown"


def validate_source_observation_json_size(serialized: str) -> str:
    """Return bounded UTF-8 observation JSON or reject it unchanged."""

    byte_count = len(serialized.encode("utf-8"))
    if byte_count > SOURCE_OBSERVATION_JSON_BYTES_MAX:
        raise ValueError(
            "source observation JSON exceeds "
            f"{SOURCE_OBSERVATION_JSON_BYTES_MAX} UTF-8 bytes"
        )
    return serialized


class CitationSourceObservation(_StrictFrozenModel):
    """Latest bounded, non-governed status for one exact evidence resolver."""

    schema_version: Literal[1] = 1
    resolver_kind: CanonicalSourceKind
    resolver_version: BoundedIdentifier
    availability: CitationSourceAvailability
    permission: CitationSourcePermission
    content_state: CitationContentState
    location_state: CitationLocationState
    capabilities: tuple[SourceCapability, ...] = Field(
        default=(),
        max_length=len(SourceCapability),
    )
    observed_at: datetime
    request_generation: int = Field(
        ge=0,
        le=SOURCE_OBSERVATION_REQUEST_GENERATION_MAX,
    )
    request_nonce: BoundedIdentifier
    error_code: SanitizedObservationErrorCode | None = None

    @model_validator(mode="after")
    def _validate_observation(self) -> "CitationSourceObservation":
        try:
            offset = self.observed_at.utcoffset()
        except (OverflowError, TypeError, ValueError):
            offset = None
        if offset is None:
            raise ValueError("observed_at must be timezone-aware")
        try:
            normalized_observed_at = self.observed_at.astimezone(UTC)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("observed_at must be timezone-aware") from None
        object.__setattr__(self, "observed_at", normalized_observed_at)
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("source observation capabilities must be unique")

        definitive_actions = {
            SourceCapability.OPEN_NATIVE,
            SourceCapability.OPEN_EXTERNAL,
            SourceCapability.COMPARE,
        }
        if (
            self.availability is not CitationSourceAvailability.AVAILABLE
            or self.permission is not CitationSourcePermission.ALLOWED
        ) and definitive_actions.intersection(self.capabilities):
            raise ValueError(
                "unavailable or unauthorized observations cannot open or compare"
            )
        if self.location_state is CitationLocationState.RELOCATED and (
            self.availability is not CitationSourceAvailability.AVAILABLE
            or self.permission is not CitationSourcePermission.ALLOWED
        ):
            raise ValueError(
                "relocated observations require an available authorized resolution"
            )
        if self.location_state is CitationLocationState.AMBIGUOUS and any(
            capability in self.capabilities
            for capability in (
                SourceCapability.OPEN_NATIVE,
                SourceCapability.OPEN_EXTERNAL,
                SourceCapability.COMPARE,
            )
        ):
            raise ValueError(
                "ambiguous observations cannot grant definitive open or compare actions"
            )
        validate_source_observation_json_size(
            _canonical_json(self.model_dump(mode="json"))
        )
        return self


class SourceProducer(str, Enum):
    """Producer family recorded by the versioned inventory."""

    LOCAL = "local"
    PINNED_SERVER = "pinned_server"
    DERIVED = "derived"


class SnapshotOnlyCondition(str, Enum):
    """Static condition that removes current-source capabilities."""

    NEVER = "never"
    ALWAYS = "always"
    MISSING_DURABLE_IDENTITY = "missing_durable_identity"
    MISSING_AUTHORIZED_PARENT = "missing_authorized_parent"


class InventoryIdentityField(str, Enum):
    ITEM_ID = "item_id"


class InventoryAuthorityField(str, Enum):
    AUTHORITY_ID = "authority_id"
    PROFILE_ID = "profile_id"
    AUTHENTICATED_TENANT_ID = "authenticated_tenant_id"
    GOVERNANCE_SCOPE_ID = "governance_scope_id"


_SHARED_SOURCE_KINDS = frozenset(
    {
        CanonicalSourceKind.MEDIA_DB,
        CanonicalSourceKind.NOTES,
        CanonicalSourceKind.CHAT_HISTORY,
    }
)


class SourceCapabilityPolicy(_StrictFrozenModel):
    """Seal/current policy with storage and independently reducible actions."""

    schema_version: Literal[1] = 1
    storage_mode: EvidenceStorageMode
    view_snapshot: bool = False
    view_source_identity: bool = False
    resolve_current: bool = False
    open_native: bool = False
    open_external: bool = False
    compare: bool = False
    refresh_observation: bool = False
    export: bool = False

    def permits(self, capability: SourceCapability) -> bool:
        """Return the explicitly selected value for one capability."""

        return bool(getattr(self, capability.value))


class _LocatorPayloadV1(_StrictFrozenModel):
    """Common bounded identity shared by the static payload union."""

    schema_version: Literal[1] = 1
    source_kind: CanonicalSourceKind
    item_id: BoundedIdentifier


class MediaLocatorPayloadV1(_LocatorPayloadV1):
    """Media item with optional chunk, page, section, or time lineage."""

    source_kind: Literal[CanonicalSourceKind.MEDIA_DB] = CanonicalSourceKind.MEDIA_DB
    chunk_id: BoundedIdentifier | None = None
    page_number: int | None = Field(default=None, ge=1)
    section_ordinal: int | None = Field(default=None, ge=0)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_time_range(self) -> "MediaLocatorPayloadV1":
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("end_seconds must not precede start_seconds")
        return self


class NoteLocatorPayloadV1(_LocatorPayloadV1):
    """Note item with optional local file and indexed-chunk lineage."""

    source_kind: Literal[CanonicalSourceKind.NOTES] = CanonicalSourceKind.NOTES
    source_root_id: SourceRootIdentifier | None = None
    relative_path: SafeRelativePath | None = None
    chunk_id: BoundedIdentifier | None = None
    section_ordinal: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_file_binding(self) -> "NoteLocatorPayloadV1":
        if (self.source_root_id is None) != (self.relative_path is None):
            raise ValueError("source_root_id and relative_path must appear together")
        return self


class ChatHistoryLocatorPayloadV1(_LocatorPayloadV1):
    """Conversation item with optional message and indexed-chunk lineage."""

    source_kind: Literal[CanonicalSourceKind.CHAT_HISTORY] = (
        CanonicalSourceKind.CHAT_HISTORY
    )
    chunk_id: BoundedIdentifier | None = None
    message_id: BoundedIdentifier | None = None


class CharacterCardLocatorPayloadV1(_LocatorPayloadV1):
    """Character-card item identity."""

    source_kind: Literal[CanonicalSourceKind.CHARACTER_CARDS] = (
        CanonicalSourceKind.CHARACTER_CARDS
    )


class WebContentLocatorPayloadV1(_LocatorPayloadV1):
    """Indexed web item identity without an executable URL."""

    source_kind: Literal[CanonicalSourceKind.WEB_CONTENT] = (
        CanonicalSourceKind.WEB_CONTENT
    )
    chunk_id: BoundedIdentifier | None = None
    page_number: int | None = Field(default=None, ge=1)
    section_ordinal: int | None = Field(default=None, ge=0)


class PromptLocatorPayloadV1(_LocatorPayloadV1):
    """Prompt item identity."""

    source_kind: Literal[CanonicalSourceKind.PROMPTS] = CanonicalSourceKind.PROMPTS


class WorldBookLocatorPayloadV1(_LocatorPayloadV1):
    """World-book item with optional entry and chunk lineage."""

    source_kind: Literal[CanonicalSourceKind.WORLD_BOOKS] = (
        CanonicalSourceKind.WORLD_BOOKS
    )
    chunk_id: BoundedIdentifier | None = None
    entry_id: BoundedIdentifier | None = None
    section_ordinal: int | None = Field(default=None, ge=0)


class DictionaryLocatorPayloadV1(_LocatorPayloadV1):
    """Dictionary item with optional entry and chunk lineage."""

    source_kind: Literal[CanonicalSourceKind.DICTIONARIES] = (
        CanonicalSourceKind.DICTIONARIES
    )
    chunk_id: BoundedIdentifier | None = None
    entry_id: BoundedIdentifier | None = None
    section_ordinal: int | None = Field(default=None, ge=0)


class KanbanLocatorPayloadV1(_LocatorPayloadV1):
    """Kanban item identity."""

    source_kind: Literal[CanonicalSourceKind.KANBAN] = CanonicalSourceKind.KANBAN


class SQLLocatorPayloadV1(_LocatorPayloadV1):
    """Structured SQL result identity with no replay or path fields."""

    source_kind: Literal[CanonicalSourceKind.SQL] = CanonicalSourceKind.SQL


class ClaimLocatorPayloadV1(_LocatorPayloadV1):
    """Claim identity with inert parent media/chunk lineage."""

    source_kind: Literal[CanonicalSourceKind.CLAIMS] = CanonicalSourceKind.CLAIMS
    parent_media_id: BoundedIdentifier | None = None
    parent_chunk_id: BoundedIdentifier | None = None

    @model_validator(mode="after")
    def _validate_parent_lineage(self) -> "ClaimLocatorPayloadV1":
        if (self.parent_media_id is None) != (self.parent_chunk_id is None):
            raise ValueError("parent_media_id and parent_chunk_id must appear together")
        return self


SourceLocatorPayloadV1 = Annotated[
    MediaLocatorPayloadV1
    | NoteLocatorPayloadV1
    | ChatHistoryLocatorPayloadV1
    | CharacterCardLocatorPayloadV1
    | WebContentLocatorPayloadV1
    | PromptLocatorPayloadV1
    | WorldBookLocatorPayloadV1
    | DictionaryLocatorPayloadV1
    | KanbanLocatorPayloadV1
    | SQLLocatorPayloadV1
    | ClaimLocatorPayloadV1,
    Field(discriminator="source_kind"),
]


class SourceLocatorEnvelope(_StrictFrozenModel):
    """Bounded native locator selecting only a source kind and payload version."""

    schema_version: Literal[1] = 1
    binding_state: Literal[LocatorBindingState.NATIVE] = LocatorBindingState.NATIVE
    source_kind: CanonicalSourceKind
    authority_scope: AuthorityScope
    authority_id: BoundedIdentifier
    governance_scope_id: BoundedIdentifier
    profile_id: BoundedIdentifier | None = None
    authenticated_tenant_id: BoundedIdentifier | None = None
    resolver_payload_version: Literal[1] = 1
    resolver_payload: SourceLocatorPayloadV1

    @model_validator(mode="after")
    def _validate_envelope(self) -> "SourceLocatorEnvelope":
        if self.authority_scope is AuthorityScope.LOCAL_PROFILE:
            if (
                self.profile_id is None
                or self.authenticated_tenant_id is not None
                or self.governance_scope_id != self.profile_id
            ):
                raise ValueError(
                    "local locator governance_scope_id must match profile_id"
                )
        elif (
            self.authenticated_tenant_id is None
            or self.profile_id is not None
            or self.governance_scope_id != self.authenticated_tenant_id
        ):
            raise ValueError(
                "server locator governance_scope_id must match authenticated_tenant_id"
            )

        if (
            self.authority_scope is AuthorityScope.LOCAL_PROFILE
            and self.source_kind not in _SHARED_SOURCE_KINDS
        ):
            raise ValueError("source kind does not match locator authority scope")
        if (
            self.authority_scope is AuthorityScope.AUTHENTICATED_TENANT
            and isinstance(self.resolver_payload, NoteLocatorPayloadV1)
            and self.resolver_payload.source_root_id is not None
        ):
            raise ValueError("server note locators cannot carry local file paths")
        if self.resolver_payload.source_kind is not self.source_kind:
            raise ValueError("payload source kind must match locator source kind")

        byte_count = len(_canonical_json(self.model_dump(mode="json")).encode("utf-8"))
        if byte_count > LOCATOR_ENVELOPE_JSON_BYTES_MAX:
            raise ValueError(
                f"locator envelope exceeds {LOCATOR_ENVELOPE_JSON_BYTES_MAX} UTF-8 bytes"
            )
        return self


class SourceInventoryEntry(_StrictFrozenModel):
    """One immutable source-kind classification in the reviewed v1 inventory."""

    schema_version: Literal[1] = 1
    source_kind: CanonicalSourceKind
    producer: SourceProducer
    authority_scope: AuthorityScope
    required_identity_fields: tuple[InventoryIdentityField, ...] = Field(
        min_length=1, max_length=8
    )
    required_authority_fields: tuple[InventoryAuthorityField, ...] = Field(
        min_length=1, max_length=8
    )
    locator_version: Literal[1] = 1
    default_policy: SourceCapabilityPolicy
    snapshot_only_condition: SnapshotOnlyCondition
    authoritative_parent_required_for_native_open: bool
    allowed_parent_kinds: tuple[CanonicalSourceKind, ...] = Field(
        default=(), max_length=8
    )

    @model_validator(mode="after")
    def _validate_inventory_entry(self) -> "SourceInventoryEntry":
        if len(set(self.required_identity_fields)) != len(
            self.required_identity_fields
        ):
            raise ValueError("required identity fields must be unique")
        if len(set(self.required_authority_fields)) != len(
            self.required_authority_fields
        ):
            raise ValueError("required authority fields must be unique")
        if self.authoritative_parent_required_for_native_open != bool(
            self.allowed_parent_kinds
        ):
            raise ValueError("authoritative parent requirement is inconsistent")
        if self.snapshot_only_condition is SnapshotOnlyCondition.ALWAYS and any(
            self.default_policy.permits(capability)
            for capability in (
                SourceCapability.RESOLVE_CURRENT,
                SourceCapability.OPEN_NATIVE,
                SourceCapability.OPEN_EXTERNAL,
                SourceCapability.COMPARE,
                SourceCapability.REFRESH_OBSERVATION,
            )
        ):
            raise ValueError("always snapshot-only sources cannot resolve or open")
        return self


class CitationReadAuthorization(_StrictFrozenModel):
    """Trusted request-scoped permission to hydrate or act on governed data."""

    schema_version: Literal[1] = 1
    authority_scope: AuthorityScope
    profile_id: BoundedIdentifier | None = None
    authenticated_tenant_id: BoundedIdentifier | None = None
    governance_scope_id: BoundedIdentifier
    allowlisted_authority_ids: tuple[BoundedIdentifier, ...] = Field(
        min_length=1,
        max_length=AUTHORITY_IDS_PER_READ_AUTHORIZATION_MAX,
    )
    view_snapshot: bool = False
    view_source_identity: bool = False
    resolve_current: bool = False
    open_native: bool = False
    open_external: bool = False
    compare: bool = False
    refresh_observation: bool = False
    export: bool = False

    @model_validator(mode="after")
    def _validate_authorization_scope(self) -> "CitationReadAuthorization":
        if (self.profile_id is None) == (self.authenticated_tenant_id is None):
            raise ValueError(
                "authorization requires exactly one profile or authenticated tenant"
            )
        if self.authority_scope is AuthorityScope.LOCAL_PROFILE:
            scope_id = self.profile_id
            if self.authenticated_tenant_id is not None:
                raise ValueError("local authorization cannot carry a tenant")
        else:
            scope_id = self.authenticated_tenant_id
            if self.profile_id is not None:
                raise ValueError("server authorization cannot carry a profile")
        if self.governance_scope_id != scope_id:
            raise ValueError(
                "governance_scope_id must match the authenticated caller scope"
            )
        if len(set(self.allowlisted_authority_ids)) != len(
            self.allowlisted_authority_ids
        ):
            raise ValueError("allowlisted authority IDs must be unique")
        return self

    def permits(self, capability: SourceCapability) -> bool:
        """Return the request permission for one independent capability."""

        return bool(getattr(self, capability.value))


class InertLocatorCandidate(_StrictFrozenModel):
    """Bounded imported/legacy data with no native resolution authority."""

    schema_version: Literal[1] = 1
    candidate_id: BoundedIdentifier
    binding_state: LocatorBindingState
    candidate_json: CandidateJson

    @model_validator(mode="after")
    def _validate_inert_candidate(self) -> "InertLocatorCandidate":
        if self.binding_state not in {
            LocatorBindingState.INERT_IMPORTED,
            LocatorBindingState.INERT_LEGACY,
        }:
            raise ValueError("locator candidate must remain inert")
        byte_count = len(self.candidate_json.encode("utf-8"))
        if byte_count > LOCATOR_ENVELOPE_JSON_BYTES_MAX:
            raise ValueError(
                f"locator candidate exceeds {LOCATOR_ENVELOPE_JSON_BYTES_MAX} UTF-8 bytes"
            )
        return self


class CurrentAuthorityLocatorLookup(_StrictFrozenModel):
    """Short-lived native result returned by a fresh trusted authority lookup."""

    schema_version: Literal[1] = 1
    lookup_id: BoundedIdentifier
    candidate_id: BoundedIdentifier
    authority_id: BoundedIdentifier
    governance_scope_id: BoundedIdentifier
    profile_id: BoundedIdentifier | None = None
    authenticated_tenant_id: BoundedIdentifier | None = None
    native_locator: SourceLocatorEnvelope
    observed_at: datetime
    valid_until: datetime

    @model_validator(mode="after")
    def _validate_lookup(self) -> "CurrentAuthorityLocatorLookup":
        if self.observed_at.tzinfo is None or self.valid_until.tzinfo is None:
            raise ValueError("lookup timestamps must be timezone-aware")
        if self.valid_until <= self.observed_at:
            raise ValueError("fresh lookup must have a bounded validity window")
        if self.valid_until - self.observed_at > CURRENT_AUTHORITY_LOOKUP_TTL_MAX:
            raise ValueError("current-authority lookup freshness window is too large")
        locator = self.native_locator
        if (
            self.authority_id != locator.authority_id
            or self.governance_scope_id != locator.governance_scope_id
            or self.profile_id != locator.profile_id
            or self.authenticated_tenant_id != locator.authenticated_tenant_id
        ):
            raise ValueError("lookup authority and scope must match native locator")
        return self


class RebindAction(str, Enum):
    """Explicit user/trusted-boundary decision for an inert candidate."""

    APPROVE = "approve"
    REJECT = "reject"


class RebindDecision(_StrictFrozenModel):
    """Explicit decision bound to one candidate and one fresh lookup."""

    schema_version: Literal[1] = 1
    candidate_id: BoundedIdentifier
    lookup_id: BoundedIdentifier
    action: RebindAction
    decided_at: datetime

    @model_validator(mode="after")
    def _validate_decision_time(self) -> "RebindDecision":
        if self.decided_at.tzinfo is None:
            raise ValueError("decision timestamp must be timezone-aware")
        return self


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _preflight_inert_locator_json(value: Any) -> None:
    """Reject hostile JSON trees before canonical serialization allocates."""

    structural_bytes = 0
    container_count = 0
    item_count = 0
    active_containers: set[int] = set()

    def reject(reason: str) -> None:
        raise ValueError(f"locator candidate preflight rejected {reason}")

    def add_bytes(byte_count: int) -> None:
        nonlocal structural_bytes
        if byte_count > LOCATOR_ENVELOPE_JSON_BYTES_MAX - structural_bytes:
            reject("oversized JSON")
        structural_bytes += byte_count

    def add_string(text: str, *, key: bool = False) -> None:
        if len(text) > LOCATOR_ENVELOPE_JSON_BYTES_MAX - structural_bytes:
            reject("oversized string")
        byte_count = len(text.encode("utf-8"))
        if key and byte_count > INERT_LOCATOR_JSON_KEY_UTF8_BYTES_MAX:
            reject("oversized key")
        add_bytes(byte_count + 2)

    def walk(node: Any, depth: int) -> None:
        nonlocal container_count, item_count
        if depth > INERT_LOCATOR_JSON_DEPTH_MAX:
            reject("excessive depth")
        if node is None:
            add_bytes(4)
            return
        if type(node) is bool:
            add_bytes(4 if node else 5)
            return
        if type(node) is str:
            add_string(node)
            return
        if type(node) is int:
            lower_digits = (
                1 if -10 < node < 10 else ((node.bit_length() - 1) * 3) // 10 + 1
            )
            add_bytes(lower_digits + (1 if node < 0 else 0))
            return
        if type(node) is float:
            if not math.isfinite(node):
                reject("non-finite number")
            add_bytes(1)
            return
        if type(node) not in {dict, list, tuple}:
            reject("non-JSON value")

        container_id = id(node)
        if container_id in active_containers:
            reject("cyclic container")
        container_count += 1
        if container_count > INERT_LOCATOR_JSON_CONTAINERS_MAX:
            reject("too many containers")
        child_count = len(node)
        item_count += child_count
        if item_count > INERT_LOCATOR_JSON_ITEMS_MAX:
            reject("too many items")
        add_bytes(2 + max(0, child_count - 1))

        active_containers.add(container_id)
        try:
            if type(node) is dict:
                for key, child in node.items():
                    if type(key) is not str:
                        reject("non-string key")
                    add_string(key, key=True)
                    add_bytes(1)
                    walk(child, depth + 1)
            else:
                for child in node:
                    walk(child, depth + 1)
        finally:
            active_containers.remove(container_id)

    walk(value, 0)


def canonical_locator_json(locator: SourceLocatorEnvelope) -> str:
    """Serialize a revalidated native locator deterministically."""

    revalidated = _revalidate_model(SourceLocatorEnvelope, locator)
    return _canonical_json(revalidated.model_dump(mode="json"))


def parse_inert_locator_candidate(
    raw_candidate: Any,
    *,
    candidate_id: str,
    binding_state: LocatorBindingState,
) -> InertLocatorCandidate:
    """Canonicalize untrusted locator-shaped data without granting authority."""

    if binding_state not in {
        LocatorBindingState.INERT_IMPORTED,
        LocatorBindingState.INERT_LEGACY,
    }:
        raise ValueError("native locator data cannot be created by inert parsing")
    _preflight_inert_locator_json(raw_candidate)
    try:
        candidate_json = _canonical_json(raw_candidate)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("locator candidate must be bounded JSON data") from exc
    if len(candidate_json.encode("utf-8")) > LOCATOR_ENVELOPE_JSON_BYTES_MAX:
        raise ValueError(
            f"locator candidate exceeds {LOCATOR_ENVELOPE_JSON_BYTES_MAX} UTF-8 bytes"
        )
    return InertLocatorCandidate(
        candidate_id=candidate_id,
        binding_state=binding_state,
        candidate_json=candidate_json,
    )


def _revalidate_model(model_type: type[_ModelT], value: Any) -> _ModelT:
    if not isinstance(value, model_type):
        raise TypeError(f"expected {model_type.__name__}")
    return model_type.model_validate(dict(value.__dict__))


def _inventory_entry(
    source_kind: CanonicalSourceKind,
    authority_scope: AuthorityScope,
) -> SourceInventoryEntry:
    try:
        return SOURCE_INVENTORY_BY_SCOPE_V1[(source_kind, authority_scope)]
    except KeyError as exc:
        raise ValueError("source kind and authority scope are not registered") from exc


def validate_native_locator(
    locator: SourceLocatorEnvelope,
    authorization: CitationReadAuthorization,
    policy: SourceCapabilityPolicy,
    *,
    required_capability: SourceCapability = SourceCapability.VIEW_SNAPSHOT,
    parent_locator: SourceLocatorEnvelope | None = None,
    parent_policy: SourceCapabilityPolicy | None = None,
) -> SourceLocatorEnvelope:
    """Fail closed unless scope, policy, inventory, and request capability agree."""

    native = _revalidate_model(SourceLocatorEnvelope, locator)
    read_auth = _revalidate_model(CitationReadAuthorization, authorization)
    current_policy = _revalidate_model(SourceCapabilityPolicy, policy)
    entry = _inventory_entry(native.source_kind, native.authority_scope)

    if native.authority_id not in read_auth.allowlisted_authority_ids:
        raise ValueError("locator authority is not allowlisted")
    if (
        native.authority_scope is not read_auth.authority_scope
        or native.governance_scope_id != read_auth.governance_scope_id
        or native.profile_id != read_auth.profile_id
        or native.authenticated_tenant_id != read_auth.authenticated_tenant_id
    ):
        raise ValueError("locator and read authorization scope do not match")
    if entry.authority_scope is not native.authority_scope:
        raise ValueError("locator scope is not valid for its source inventory entry")

    for capability in SourceCapability:
        if current_policy.permits(capability) and not entry.default_policy.permits(
            capability
        ):
            raise ValueError(
                f"{capability.value} is unsupported for {native.source_kind.value}"
            )
    if not current_policy.permits(required_capability):
        raise ValueError(f"policy denies {required_capability.value}")
    if not read_auth.permits(required_capability):
        raise ValueError(f"authorization denies {required_capability.value}")

    if (
        isinstance(native.resolver_payload, ClaimLocatorPayloadV1)
        and required_capability
        in {
            SourceCapability.RESOLVE_CURRENT,
            SourceCapability.OPEN_NATIVE,
            SourceCapability.COMPARE,
            SourceCapability.REFRESH_OBSERVATION,
        }
        and (
            CanonicalSourceKind.MEDIA_DB not in entry.allowed_parent_kinds
            or native.resolver_payload.parent_media_id is None
            or native.resolver_payload.parent_chunk_id is None
        )
    ):
        raise ValueError("claims capability requires authorized parent lineage")
    if isinstance(
        native.resolver_payload, ClaimLocatorPayloadV1
    ) and required_capability in {
        SourceCapability.RESOLVE_CURRENT,
        SourceCapability.OPEN_NATIVE,
        SourceCapability.COMPARE,
        SourceCapability.REFRESH_OBSERVATION,
    }:
        if parent_locator is None or parent_policy is None:
            raise ValueError("claims capability requires separately validated parent")
        parent = validate_native_locator(
            parent_locator,
            read_auth,
            parent_policy,
            required_capability=required_capability,
        )
        if (
            not isinstance(parent.resolver_payload, MediaLocatorPayloadV1)
            or parent.authority_scope is not native.authority_scope
            or parent.authority_id != native.authority_id
            or parent.governance_scope_id != native.governance_scope_id
            or parent.profile_id != native.profile_id
            or parent.authenticated_tenant_id != native.authenticated_tenant_id
            or parent.resolver_payload.item_id
            != native.resolver_payload.parent_media_id
            or parent.resolver_payload.chunk_id
            != native.resolver_payload.parent_chunk_id
        ):
            raise ValueError("claims parent lineage does not match validated parent")
    return native


def rebind_inert_locator(
    candidate: InertLocatorCandidate,
    lookup: CurrentAuthorityLocatorLookup,
    decision: RebindDecision,
    authorization: CitationReadAuthorization,
    *,
    now: datetime,
) -> SourceLocatorEnvelope:
    """Create a new native envelope from an approved, fresh authority lookup."""

    inert = _revalidate_model(InertLocatorCandidate, candidate)
    current = _revalidate_model(CurrentAuthorityLocatorLookup, lookup)
    approved = _revalidate_model(RebindDecision, decision)
    if now.tzinfo is None:
        raise ValueError("rebind time must be timezone-aware")
    if approved.action is not RebindAction.APPROVE:
        raise ValueError("rebind decision was not approved")
    if (
        current.candidate_id != inert.candidate_id
        or approved.candidate_id != inert.candidate_id
        or approved.lookup_id != current.lookup_id
    ):
        raise ValueError("rebind candidate or lookup binding does not match")
    if not current.observed_at <= approved.decided_at <= now <= current.valid_until:
        raise ValueError("current-authority lookup is not fresh for this decision")

    native = validate_native_locator(
        current.native_locator,
        authorization,
        _inventory_entry(
            current.native_locator.source_kind,
            current.native_locator.authority_scope,
        ).default_policy,
        required_capability=SourceCapability.RESOLVE_CURRENT,
    )
    return SourceLocatorEnvelope.model_validate(native.model_dump())


def _default_policy(
    storage_mode: EvidenceStorageMode,
    *,
    external: bool = False,
    snapshot_only: bool = False,
) -> SourceCapabilityPolicy:
    return SourceCapabilityPolicy(
        storage_mode=storage_mode,
        view_snapshot=True,
        view_source_identity=True,
        resolve_current=not snapshot_only,
        open_native=not snapshot_only,
        open_external=external and not snapshot_only,
        compare=not snapshot_only,
        refresh_observation=not snapshot_only,
        export=True,
    )


def _inventory(
    source_kind: CanonicalSourceKind,
    producer: SourceProducer,
    authority_scope: AuthorityScope,
    default_policy: SourceCapabilityPolicy,
    snapshot_only_condition: SnapshotOnlyCondition,
    *,
    parent_kinds: tuple[CanonicalSourceKind, ...] = (),
) -> SourceInventoryEntry:
    authority_fields = (
        (
            InventoryAuthorityField.AUTHORITY_ID,
            InventoryAuthorityField.PROFILE_ID,
            InventoryAuthorityField.GOVERNANCE_SCOPE_ID,
        )
        if authority_scope is AuthorityScope.LOCAL_PROFILE
        else (
            InventoryAuthorityField.AUTHORITY_ID,
            InventoryAuthorityField.AUTHENTICATED_TENANT_ID,
            InventoryAuthorityField.GOVERNANCE_SCOPE_ID,
        )
    )
    return SourceInventoryEntry(
        source_kind=source_kind,
        producer=producer,
        authority_scope=authority_scope,
        required_identity_fields=(InventoryIdentityField.ITEM_ID,),
        required_authority_fields=authority_fields,
        default_policy=default_policy,
        snapshot_only_condition=snapshot_only_condition,
        authoritative_parent_required_for_native_open=bool(parent_kinds),
        allowed_parent_kinds=parent_kinds,
    )


RUNTIME_SOURCE_KIND_TO_CANONICAL_V1: Mapping[str, str] = MappingProxyType(
    {
        "media": CanonicalSourceKind.MEDIA_DB.value,
        "note": CanonicalSourceKind.NOTES.value,
        "conversation": CanonicalSourceKind.CHAT_HISTORY.value,
    }
)

_LOCAL_POLICY = _default_policy(EvidenceStorageMode.EMBEDDED)
_SERVER_POLICY = _default_policy(EvidenceStorageMode.SERVER_REFERENCE)
_WEB_POLICY = _default_policy(EvidenceStorageMode.SERVER_REFERENCE, external=True)
_SQL_POLICY = _default_policy(
    EvidenceStorageMode.SERVER_REFERENCE,
    snapshot_only=True,
)

SOURCE_INVENTORY_V1: tuple[SourceInventoryEntry, ...] = (
    _inventory(
        CanonicalSourceKind.MEDIA_DB,
        SourceProducer.LOCAL,
        AuthorityScope.LOCAL_PROFILE,
        _LOCAL_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.NOTES,
        SourceProducer.LOCAL,
        AuthorityScope.LOCAL_PROFILE,
        _LOCAL_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.CHAT_HISTORY,
        SourceProducer.LOCAL,
        AuthorityScope.LOCAL_PROFILE,
        _LOCAL_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.MEDIA_DB,
        SourceProducer.PINNED_SERVER,
        AuthorityScope.AUTHENTICATED_TENANT,
        _SERVER_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.NOTES,
        SourceProducer.PINNED_SERVER,
        AuthorityScope.AUTHENTICATED_TENANT,
        _SERVER_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.CHAT_HISTORY,
        SourceProducer.PINNED_SERVER,
        AuthorityScope.AUTHENTICATED_TENANT,
        _SERVER_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.CHARACTER_CARDS,
        SourceProducer.PINNED_SERVER,
        AuthorityScope.AUTHENTICATED_TENANT,
        _SERVER_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.WEB_CONTENT,
        SourceProducer.PINNED_SERVER,
        AuthorityScope.AUTHENTICATED_TENANT,
        _WEB_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.PROMPTS,
        SourceProducer.PINNED_SERVER,
        AuthorityScope.AUTHENTICATED_TENANT,
        _SERVER_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.WORLD_BOOKS,
        SourceProducer.PINNED_SERVER,
        AuthorityScope.AUTHENTICATED_TENANT,
        _SERVER_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.DICTIONARIES,
        SourceProducer.PINNED_SERVER,
        AuthorityScope.AUTHENTICATED_TENANT,
        _SERVER_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.KANBAN,
        SourceProducer.PINNED_SERVER,
        AuthorityScope.AUTHENTICATED_TENANT,
        _SERVER_POLICY,
        SnapshotOnlyCondition.MISSING_DURABLE_IDENTITY,
    ),
    _inventory(
        CanonicalSourceKind.SQL,
        SourceProducer.DERIVED,
        AuthorityScope.AUTHENTICATED_TENANT,
        _SQL_POLICY,
        SnapshotOnlyCondition.ALWAYS,
    ),
    _inventory(
        CanonicalSourceKind.CLAIMS,
        SourceProducer.DERIVED,
        AuthorityScope.AUTHENTICATED_TENANT,
        _SERVER_POLICY,
        SnapshotOnlyCondition.MISSING_AUTHORIZED_PARENT,
        parent_kinds=(CanonicalSourceKind.MEDIA_DB,),
    ),
)

SOURCE_INVENTORY_BY_SCOPE_V1: Mapping[
    tuple[CanonicalSourceKind, AuthorityScope],
    SourceInventoryEntry,
] = MappingProxyType(
    {(entry.source_kind, entry.authority_scope): entry for entry in SOURCE_INVENTORY_V1}
)
if len(SOURCE_INVENTORY_BY_SCOPE_V1) != len(SOURCE_INVENTORY_V1):
    raise RuntimeError("duplicate citation source inventory scope")
