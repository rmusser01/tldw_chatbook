"""Data-only citation source locator, policy, and authorization contracts."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
import json
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
AUTHORITY_IDS_PER_READ_AUTHORIZATION_MAX = 32
CURRENT_AUTHORITY_LOOKUP_TTL_MAX = timedelta(minutes=5)
_DRIVE_PATH = re.compile(r"^[A-Za-z]:")


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


_LOCATION_HINTS_BY_SOURCE_KIND = MappingProxyType(
    {
        CanonicalSourceKind.MEDIA_DB: frozenset(
            {
                "chunk_id",
                "page_number",
                "section_ordinal",
                "start_seconds",
                "end_seconds",
            }
        ),
        CanonicalSourceKind.NOTES: frozenset(
            {"source_root_id", "relative_path", "chunk_id", "section_ordinal"}
        ),
        CanonicalSourceKind.CHAT_HISTORY: frozenset({"chunk_id", "message_id"}),
        CanonicalSourceKind.CHARACTER_CARDS: frozenset(),
        CanonicalSourceKind.WEB_CONTENT: frozenset(
            {"chunk_id", "page_number", "section_ordinal"}
        ),
        CanonicalSourceKind.PROMPTS: frozenset(),
        CanonicalSourceKind.WORLD_BOOKS: frozenset(
            {"chunk_id", "entry_id", "section_ordinal"}
        ),
        CanonicalSourceKind.DICTIONARIES: frozenset(
            {"chunk_id", "entry_id", "section_ordinal"}
        ),
        CanonicalSourceKind.KANBAN: frozenset(),
        CanonicalSourceKind.SQL: frozenset(),
        CanonicalSourceKind.CLAIMS: frozenset(
            {"chunk_id", "parent_source_kind", "parent_item_id"}
        ),
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


class SourceLocatorPayloadV1(_StrictFrozenModel):
    """Typed resolver-owned identity and location hints; never executable data."""

    schema_version: Literal[1] = 1
    item_id: BoundedIdentifier
    source_root_id: SourceRootIdentifier | None = None
    relative_path: SafeRelativePath | None = None
    chunk_id: BoundedIdentifier | None = None
    message_id: BoundedIdentifier | None = None
    entry_id: BoundedIdentifier | None = None
    parent_source_kind: CanonicalSourceKind | None = None
    parent_item_id: BoundedIdentifier | None = None
    page_number: int | None = Field(default=None, ge=1)
    section_ordinal: int | None = Field(default=None, ge=0)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_location_shape(self) -> "SourceLocatorPayloadV1":
        if (self.source_root_id is None) != (self.relative_path is None):
            raise ValueError("source_root_id and relative_path must appear together")
        if (self.parent_source_kind is None) != (self.parent_item_id is None):
            raise ValueError(
                "parent_source_kind and parent_item_id must appear together"
            )
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("end_seconds must not precede start_seconds")
        return self


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

        local_kinds = {
            CanonicalSourceKind.MEDIA_DB,
            CanonicalSourceKind.NOTES,
            CanonicalSourceKind.CHAT_HISTORY,
        }
        if (self.source_kind in local_kinds) != (
            self.authority_scope is AuthorityScope.LOCAL_PROFILE
        ):
            raise ValueError("source kind does not match locator authority scope")
        if (
            self.resolver_payload.source_root_id is not None
            and self.source_kind is not CanonicalSourceKind.NOTES
        ):
            raise ValueError("file-backed location hints are valid only for notes")
        if (
            self.resolver_payload.parent_source_kind is not None
            and self.source_kind is not CanonicalSourceKind.CLAIMS
        ):
            raise ValueError("parent lineage is valid only for claims")
        populated_hints = set(self.resolver_payload.model_dump(exclude_none=True)) - {
            "schema_version",
            "item_id",
        }
        unsupported_hints = (
            populated_hints - _LOCATION_HINTS_BY_SOURCE_KIND[self.source_kind]
        )
        if unsupported_hints:
            raise ValueError(
                f"unsupported location hint for {self.source_kind.value}: "
                f"{sorted(unsupported_hints)}"
            )

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
    try:
        candidate_json = _canonical_json(raw_candidate)
    except (RecursionError, TypeError, ValueError) as exc:
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


def _inventory_entry(source_kind: CanonicalSourceKind) -> SourceInventoryEntry:
    return next(
        entry for entry in SOURCE_INVENTORY_V1 if entry.source_kind is source_kind
    )


def validate_native_locator(
    locator: SourceLocatorEnvelope,
    authorization: CitationReadAuthorization,
    policy: SourceCapabilityPolicy,
    *,
    required_capability: SourceCapability = SourceCapability.VIEW_SNAPSHOT,
) -> SourceLocatorEnvelope:
    """Fail closed unless scope, policy, inventory, and request capability agree."""

    native = _revalidate_model(SourceLocatorEnvelope, locator)
    read_auth = _revalidate_model(CitationReadAuthorization, authorization)
    current_policy = _revalidate_model(SourceCapabilityPolicy, policy)
    entry = _inventory_entry(native.source_kind)

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
        native.source_kind is CanonicalSourceKind.CLAIMS
        and required_capability
        in {
            SourceCapability.RESOLVE_CURRENT,
            SourceCapability.OPEN_NATIVE,
            SourceCapability.COMPARE,
            SourceCapability.REFRESH_OBSERVATION,
        }
        and (
            native.resolver_payload.parent_source_kind not in entry.allowed_parent_kinds
            or native.resolver_payload.parent_item_id is None
        )
    ):
        raise ValueError("claims capability requires authorized parent lineage")
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
        _inventory_entry(current.native_locator.source_kind).default_policy,
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
