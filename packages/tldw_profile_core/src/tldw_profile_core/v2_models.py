"""Inactive canonical V2 data assertions; never native admission or authority."""

import json
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Annotated, Any, Literal, Self
from unicodedata import category
from zoneinfo import ZoneInfo

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    ValidationInfo,
    model_validator,
)

from . import v2_meaning
from .canonical import (
    I_JSON_MAX_INTEGER,
    Confidence,
    JsonInteger,
    canonical_bytes,
    parse_json_integer,
)
from .evidence_binding import (
    CaptureTime,
    Digest,
    Identity,
    OwnerVersionEvidenceBinding,
    owner_version_evidence_binding_digest,
)
from .models import SemanticKey
from .payloads import FrozenModel, ProfilePayload, reject_secret_material
from .v2_meaning import (
    ClaimMeaningV2,
    TemporalRelationV2,
    TemporalValidityV2,
    claim_meaning_digest,
)

MANIFEST_SEMANTICS = (
    "approval-v2",
    "disclosure-ceiling-v2",
    "evidence-binding-v1",
    "metadata-retirement-v2",
    "profile-relations-v2",
    "temporal-validity-v2",
    "typed-claim-v2",
)
OBJECT_SCHEMAS = (
    ("manifest", (2,)),
    ("proposal", (1, 2)),
    ("record", (1, 2)),
    ("scope", (1,)),
)
MAX_CLAIM_BYTES = MAX_MANIFEST_BYTES = 16 * 1024
MAX_RECORD_BYTES = 64 * 1024
MAX_PROPOSAL_BYTES = 96 * 1024
Purpose = Literal["conversation", "embedding", "interview", "summary"]
Actor = Literal["user", "agent", "system"]
ClaimBasis = Literal[
    "direct_user_assertion", "inference", "imported_assertion", "legacy_unknown"
]
RecordKind = Literal[
    "identity",
    "preference",
    "relationship",
    "correction",
    "constraint",
    "goal",
    "convention",
    "working_context",
    "legacy_unclassified",
]


def _version_two(value: object) -> int:
    if parse_json_integer(value) != 2:
        raise ValueError("version must be numeric two")
    return 2


VersionTwo = Annotated[Literal[2], BeforeValidator(_version_two)]
Counter = Annotated[JsonInteger, Field(ge=0, le=I_JSON_MAX_INTEGER)]


def _method(value: str) -> str:
    if len(value) > 64 or len(value.encode("utf-8")) > 256 or not value.strip():
        raise ValueError("method exceeds bounds or is blank")
    if any(category(c) in {"Cc", "Cf"} for c in value):
        raise ValueError("method contains forbidden characters")
    return reject_secret_material(value)


Method = Annotated[Identity, AfterValidator(_method)]


def _closed_keys(value: object, model: type[FrozenModel]) -> None:
    if type(value) is dict and any(
        type(key) is not str or key not in model.model_fields for key in value
    ):
        raise ValueError("V2 object contains fields outside its exact shape")


def _closed_composed_children(value: dict[str, object]) -> None:
    """Protect unchanged composed leaf models from per-call extra overrides."""
    for field, choices in _COMPOSED_CHILDREN.items():
        child = value.get(field)
        children = child if type(child) is tuple else (child,)
        for item in children:
            if type(item) is not dict:
                continue
            tag = item.get("kind", value.get("kind") if field == "payload" else None)
            model = choices.get(tag) if type(choices) is dict else choices
            if model is not None:
                _closed_keys(item, model)


def _snapshot(value: object) -> object:
    """Detach complete known raw state without traversing user callbacks."""
    remaining = 4096

    def visit(item: object, depth: int) -> object:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 20:
            raise ValueError("V2 input exceeds structural bounds")
        kind = type(item)
        if any(kind is model for model in _KNOWN_MODELS):
            if object.__getattribute__(item, "__pydantic_extra__") is not None:
                raise ValueError("V2 model contains extra state")
            state = object.__getattribute__(item, "__dict__")
            if type(state) is not dict:
                raise ValueError("V2 model has unsupported state dictionary")
            _closed_keys(state, kind)
            return visit(state, depth + 1)
        if kind is dict:
            if len(item) > 32 or any(
                type(k) is not str or k not in _KNOWN_KEYS for k in item
            ):
                raise ValueError("V2 object contains unsupported fields")
            if "kind" in item and (
                type(item["kind"]) is not str
                or item["kind"]
                not in v2_meaning._KIND_TAGS
                | {"deny", "on_device_only", "reviewed_destinations"}
            ):
                raise ValueError("V2 object has unsupported discriminator")
            if "no_expiry" in item and type(item["no_expiry"]) is not bool:
                raise ValueError("no_expiry must be an exact built-in boolean")
            detached = {k: visit(v, depth + 1) for k, v in item.items()}
            _closed_composed_children(detached)
            return detached
        if kind is list or kind is tuple:
            if len(item) > 32:
                raise ValueError("V2 array exceeds bounds")
            return tuple(visit(v, depth + 1) for v in item)
        if kind is datetime:
            if item.tzinfo is not None and not any(
                type(item.tzinfo) is zone for zone in (timezone, ZoneInfo)
            ):
                raise ValueError("V2 time has unsupported timezone")
            return item
        if kind is str:
            if len(item) > MAX_PROPOSAL_BYTES:
                raise ValueError("V2 string exceeds bounds")
            item.encode("utf-8", errors="strict")
            return item
        if any(kind is scalar for scalar in (int, bool, type(None))):
            return item
        if kind is float and isfinite(item):
            return item
        raise ValueError("V2 value has unsupported type")

    return visit(value, 0)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate V2 JSON member")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError("nonfinite V2 JSON constant")


def _decode_json(raw: str | bytes | bytearray) -> object:
    if not any(type(raw) is kind for kind in (str, bytes, bytearray)):
        raise TypeError("V2 JSON requires exact built-in text or bytes")
    try:
        text = raw if type(raw) is str else raw.decode("utf-8", errors="strict")
        return json.loads(
            text, object_pairs_hook=_unique_pairs, parse_constant=_nonfinite
        )
    except (ValueError, RecursionError):
        raise ValueError("invalid V2 profile JSON") from None


class _V2Model(FrozenModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _safe_input(cls, value: object, info: ValidationInfo) -> object:
        if info.mode == "json":
            raise TypeError("V2 JSON requires duplicate-aware model_validate_json")
        snapshot = _snapshot(value)
        _closed_keys(snapshot, cls)
        return snapshot

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: Any
    ) -> Self:
        """Validate duplicate-aware raw JSON; this does not grant native authority."""
        return cls.model_validate(_decode_json(json_data), **kwargs)


def _sorted_unique(values: tuple[str, ...]) -> None:
    if values != tuple(sorted(set(values))):
        raise ValueError("identities must be unique and sorted")


def _size(value: FrozenModel, ceiling: int) -> None:
    if len(canonical_bytes(value)) > ceiling:
        raise ValueError("V2 canonical byte limit exceeded")


class DenyDisclosureV2(_V2Model):
    kind: Literal["deny"] = Field(repr=False)


class OnDeviceDisclosureV2(_V2Model):
    kind: Literal["on_device_only"] = Field(repr=False)
    purposes: tuple[Purpose, ...] = Field(min_length=1, max_length=4, repr=False)

    @model_validator(mode="after")
    def _purposes(self) -> Self:
        _sorted_unique(self.purposes)
        return self


class DisclosureAudienceV2(_V2Model):
    audience_handle: Identity = Field(repr=False)
    purposes: tuple[Purpose, ...] = Field(min_length=1, max_length=4, repr=False)

    @model_validator(mode="after")
    def _audience(self) -> Self:
        if "*" in self.audience_handle:
            raise ValueError("wildcard audience is forbidden")
        _sorted_unique(self.purposes)
        return self


class ReviewedDisclosureV2(_V2Model):
    kind: Literal["reviewed_destinations"] = Field(repr=False)
    audiences: tuple[DisclosureAudienceV2, ...] = Field(
        min_length=1, max_length=16, repr=False
    )

    @model_validator(mode="after")
    def _audiences(self) -> Self:
        _sorted_unique(tuple(a.audience_handle for a in self.audiences))
        return self


ModelDisclosureV2 = Annotated[
    DenyDisclosureV2 | OnDeviceDisclosureV2 | ReviewedDisclosureV2,
    Field(discriminator="kind"),
]


class ProfileControlsV2(_V2Model):
    sync_mode: Literal["device_only", "syncable"] = Field(repr=False)
    agent_visibility: Literal["agent_visible", "user_only"] = Field(repr=False)
    model_disclosure: ModelDisclosureV2 = Field(
        default_factory=lambda: DenyDisclosureV2(kind="deny"), repr=False
    )


class ProfileProvenanceV2(_V2Model):
    source: Literal["manual", "agent", "import", "migration", "privacy_retirement"] = (
        Field(repr=False)
    )
    actor: Actor = Field(repr=False)
    reason_code: Identity = Field(repr=False)
    derived_from_record_id: Identity | None = Field(repr=False)

    @model_validator(mode="after")
    def _privacy_origin(self) -> Self:
        if self.source == "privacy_retirement" and (
            self.actor != "system"
            or self.reason_code != "privacy_retirement"
            or self.derived_from_record_id is not None
        ):
            raise ValueError("privacy successor provenance mismatch")
        return self


class AttributionOriginV2(_V2Model):
    actor: Actor = Field(repr=False)
    actor_id: Identity | None = Field(repr=False)
    method: Method = Field(repr=False)


class SupportAssessmentV2(_V2Model):
    binding_id: Identity = Field(repr=False)
    binding_digest: Digest = Field(repr=False)
    claim_digest: Digest = Field(repr=False)
    status: Literal["not_assessed", "supports", "contradicts", "insufficient"] = Field(
        repr=False
    )
    origin: AttributionOriginV2 | None = Field(repr=False)
    assessed_at: CaptureTime | None = Field(repr=False)

    @model_validator(mode="after")
    def _assessed(self) -> Self:
        unassessed = self.status == "not_assessed"
        if unassessed and (self.origin is not None or self.assessed_at is not None):
            raise ValueError("unassessed support must lack attribution")
        if not unassessed and (self.origin is None or self.assessed_at is None):
            raise ValueError("assessed support requires attribution")
        return self


class ApprovalReceiptV2(_V2Model):
    action: Literal["authored", "accepted", "accepted_edited"] = Field(repr=False)
    claim_digest: Digest = Field(repr=False)
    record_version_id: Identity = Field(repr=False)
    acted_at: CaptureTime = Field(repr=False)


class ConfidenceEstimateV2(_V2Model):
    value: Confidence = Field(repr=False)
    origin: AttributionOriginV2 = Field(repr=False)
    assessed_at: CaptureTime = Field(repr=False)
    claim_digest: Digest = Field(repr=False)


class SalienceHintV2(_V2Model):
    priority: Literal["low", "normal", "high"] = Field(repr=False)
    origin: AttributionOriginV2 = Field(repr=False)
    chosen_at: CaptureTime = Field(repr=False)
    record_version_id: Identity = Field(repr=False)


class ProfileClaimV2(_V2Model):
    claim_basis: ClaimBasis = Field(repr=False)
    temporal_validity: TemporalValidityV2 = Field(repr=False)
    relations: tuple[TemporalRelationV2, ...] = Field(max_length=4, repr=False)
    claim_digest: Digest = Field(repr=False)
    evidence_bindings: tuple[OwnerVersionEvidenceBinding, ...] = Field(
        max_length=8, repr=False
    )
    support_assessments: tuple[SupportAssessmentV2, ...] = Field(
        max_length=8, repr=False
    )
    approval_receipt: ApprovalReceiptV2 | None = Field(repr=False)
    confidence_estimate: ConfidenceEstimateV2 | None = Field(repr=False)
    salience_hint: SalienceHintV2 | None = Field(repr=False)
    hold_reason: Literal["legacy_review", "privacy_review"] | None = Field(repr=False)

    @model_validator(mode="after")
    def _claim(self) -> Self:
        _sorted_unique(tuple(b.binding_id for b in self.evidence_bindings))
        _sorted_unique(tuple(a.binding_id for a in self.support_assessments))
        bindings = {b.binding_id: b for b in self.evidence_bindings}
        for assessment in self.support_assessments:
            binding = bindings.get(assessment.binding_id)
            if (
                binding is None
                or assessment.binding_digest
                != owner_version_evidence_binding_digest(binding)
                or assessment.claim_digest != self.claim_digest
            ):
                raise ValueError("support must bind exact evidence and claim")
            if (
                assessment.assessed_at is not None
                and assessment.assessed_at < binding.captured_at
            ):
                raise ValueError("assessment precedes capture")
            if (
                assessment.status in {"supports", "contradicts"}
                and binding.span_start == binding.span_end
            ):
                raise ValueError("empty span cannot support or contradict")
        validity = self.temporal_validity
        if (
            validity.kind == "interval"
            and validity.basis.kind == "bound_source"
            and any(bid not in bindings for bid in validity.basis.binding_ids)
        ):
            raise ValueError("validity basis must name current binding")
        if self.claim_basis == "legacy_unknown" and (
            validity.kind != "unknown"
            or self.relations
            or self.evidence_bindings
            or self.support_assessments
            or self.approval_receipt is not None
            or self.confidence_estimate is not None
            or self.salience_hint is not None
            or self.hold_reason is None
        ):
            raise ValueError("legacy claim must remain unknown and held")
        _size(self, MAX_CLAIM_BYTES)
        return self


class ObjectSchemaRequirementV2(_V2Model):
    object_kind: Literal["manifest", "proposal", "record", "scope"] = Field(repr=False)
    versions: tuple[Counter, ...] = Field(min_length=1, max_length=2, repr=False)


class ProfileManifestV2(_V2Model):
    schema_version: VersionTwo = Field(repr=False)
    profile_id: Identity = Field(repr=False)
    revision: Counter = Field(repr=False)
    purge_generation: Counter = Field(repr=False)
    created_at: CaptureTime = Field(repr=False)
    updated_at: CaptureTime = Field(repr=False)
    current_version_id: Identity = Field(repr=False)
    required_object_schemas: tuple[ObjectSchemaRequirementV2, ...] = Field(
        min_length=4, max_length=4, repr=False
    )
    required_context_semantics: tuple[Identity, ...] = Field(
        min_length=7, max_length=7, repr=False
    )
    evidence_retirement_epoch: Counter = Field(repr=False)

    @model_validator(mode="after")
    def _manifest(self) -> Self:
        if self.created_at > self.updated_at:
            raise ValueError("manifest times are unordered")
        if (
            tuple((r.object_kind, r.versions) for r in self.required_object_schemas)
            != OBJECT_SCHEMAS
            or self.required_context_semantics != MANIFEST_SEMANTICS
        ):
            raise ValueError("manifest requirements are unsupported")
        _size(self, MAX_MANIFEST_BYTES)
        return self


class ProfileRecordV2(_V2Model):
    schema_version: VersionTwo = Field(repr=False)
    profile_id: Identity = Field(repr=False)
    record_id: Identity = Field(repr=False)
    scope_id: Identity = Field(repr=False)
    kind: RecordKind = Field(repr=False)
    payload: ProfilePayload | None = Field(repr=False)
    semantic_key: SemanticKey | None = Field(repr=False)
    state: Literal["active", "archived", "deleted"] = Field(repr=False)
    controls: ProfileControlsV2 = Field(repr=False)
    provenance: ProfileProvenanceV2 | None = Field(repr=False)
    claim: ProfileClaimV2 | None = Field(repr=False)
    version_id: Identity = Field(repr=False)
    parent_version_id: Identity | None = Field(repr=False)
    created_at: CaptureTime = Field(repr=False)
    updated_at: CaptureTime = Field(repr=False)
    expires_at: CaptureTime | None = Field(repr=False)
    no_expiry: StrictBool = Field(repr=False)

    @model_validator(mode="before")
    @classmethod
    def _safe_input(cls, value: object, info: ValidationInfo) -> object:
        snapshot = super()._safe_input(value, info)
        if type(snapshot) is dict:
            payload = snapshot.get("payload")
            if type(payload) is dict and "kind" not in payload and "kind" in snapshot:
                snapshot["payload"] = payload | {"kind": snapshot["kind"]}
        return snapshot

    @model_validator(mode="after")
    def _record(self) -> Self:
        if self.created_at > self.updated_at:
            raise ValueError("record times are unordered")
        if self.expires_at is not None and self.expires_at <= self.updated_at:
            raise ValueError("record expiry must follow update")
        if self.state == "deleted":
            if (
                any(
                    v is not None
                    for v in (
                        self.payload,
                        self.semantic_key,
                        self.provenance,
                        self.claim,
                        self.expires_at,
                    )
                )
                or self.no_expiry
                or self.controls.model_disclosure.kind != "deny"
            ):
                raise ValueError("deleted record must be content-free and denied")
            _size(self, MAX_RECORD_BYTES)
            return self
        if self.payload is None or self.provenance is None or self.claim is None:
            raise ValueError("non-deleted record requires payload provenance and claim")
        if self.kind == "working_context":
            if (self.expires_at is None) == (not self.no_expiry):
                raise ValueError("working context requires one expiry decision")
        elif self.expires_at is not None or self.no_expiry:
            raise ValueError("expiry applies only to working context")
        c = self.claim
        projection = ClaimMeaningV2.model_validate(
            {
                "projection": "profile-claim-v2",
                "profile_id": self.profile_id,
                "record_id": self.record_id,
                "scope_id": self.scope_id,
                "kind": self.kind,
                "payload": self.payload,
                "claim_basis": c.claim_basis,
                "temporal_validity": c.temporal_validity,
                "relations": c.relations,
            }
        )
        if c.claim_digest != claim_meaning_digest(projection):
            raise ValueError("claim digest does not match meaning")
        for edge in c.relations:
            if (
                edge.target_record_id == self.record_id
                and edge.target_version_id == self.version_id
            ):
                raise ValueError("relation targets current self")
        if any(b.captured_at > self.updated_at for b in c.evidence_bindings) or any(
            a.assessed_at is not None and a.assessed_at > self.updated_at
            for a in c.support_assessments
        ):
            raise ValueError("evidence attribution exceeds record update")
        if c.approval_receipt is not None:
            a = c.approval_receipt
            if (
                a.claim_digest != c.claim_digest
                or a.record_version_id != self.version_id
                or not self.created_at <= a.acted_at <= self.updated_at
            ):
                raise ValueError("approval must bind current claim version and time")
        if c.confidence_estimate is not None:
            a = c.confidence_estimate
            if (
                a.claim_digest != c.claim_digest
                or not self.created_at <= a.assessed_at <= self.updated_at
            ):
                raise ValueError("confidence must bind current claim and time")
        if c.salience_hint is not None:
            a = c.salience_hint
            if (
                a.record_version_id != self.version_id
                or not self.created_at <= a.chosen_at <= self.updated_at
            ):
                raise ValueError("salience must bind current version and time")
        privacy = self.provenance.source == "privacy_retirement"
        if privacy and (
            c.hold_reason != "privacy_review"
            or c.approval_receipt is not None
            or c.confidence_estimate is not None
            or c.salience_hint is not None
            or c.support_assessments
        ):
            raise ValueError("privacy successor cannot inherit attribution")
        if c.claim_basis == "legacy_unknown" and (
            self.controls.model_disclosure.kind != "deny"
            or (c.hold_reason == "privacy_review" and not privacy)
        ):
            raise ValueError("legacy record requires denied disclosure and review")
        _size(self, MAX_RECORD_BYTES)
        return self


class ProfileProposalV2(_V2Model):
    schema_version: VersionTwo = Field(repr=False)
    proposal_id: Identity = Field(repr=False)
    profile_id: Identity = Field(repr=False)
    scope_id: Identity = Field(repr=False)
    operation: Literal["create", "update", "archive", "promote"] = Field(repr=False)
    target_record_id: Identity | None = Field(repr=False)
    base_version_id: Identity | None = Field(repr=False)
    proposed_record: ProfileRecordV2 | None = Field(repr=False)
    provenance: ProfileProvenanceV2 | None = Field(repr=False)
    state: Literal["pending", "accepted", "rejected", "superseded", "expired"] = Field(
        repr=False
    )
    created_at: CaptureTime = Field(repr=False)
    expires_at: CaptureTime = Field(repr=False)

    @model_validator(mode="after")
    def _proposal(self) -> Self:
        if self.created_at >= self.expires_at:
            raise ValueError("proposal times are unordered")
        pending = self.state == "pending"
        if pending:
            try:
                expected_expiry = self.created_at + timedelta(days=90)
            except OverflowError:
                raise ValueError(
                    "pending proposal time exceeds portable bounds"
                ) from None
            if self.expires_at != expected_expiry or self.provenance is None:
                raise ValueError(
                    "pending proposal requires provenance and exact expiry"
                )
        elif self.proposed_record is not None or self.provenance is not None:
            raise ValueError("resolved proposal must be content-free")
        has_target = (
            self.target_record_id is not None and self.base_version_id is not None
        )
        if self.operation == "create":
            if self.target_record_id is not None or self.base_version_id is not None:
                raise ValueError("create proposal cannot have target or base")
        elif not has_target:
            raise ValueError("proposal requires exact target and base")
        needs_record = pending and self.operation in {"create", "update"}
        if (self.proposed_record is not None) != needs_record:
            raise ValueError("proposal content shape mismatch")
        record = self.proposed_record
        if record is not None:
            if (
                record.state != "active"
                or record.profile_id != self.profile_id
                or record.scope_id != self.scope_id
                or record.claim is None
                or record.claim.approval_receipt is not None
                or record.updated_at > self.created_at
            ):
                raise ValueError("pending candidate identity time or approval mismatch")
            if self.operation == "create":
                if record.parent_version_id is not None:
                    raise ValueError("create candidate cannot have parent")
            elif (
                record.record_id != self.target_record_id
                or record.parent_version_id != self.base_version_id
            ):
                raise ValueError("update candidate target or base mismatch")
        _size(self, MAX_PROPOSAL_BYTES)
        return self


_KNOWN_MODELS = v2_meaning._KNOWN_MODELS | frozenset(
    {
        SemanticKey,
        OwnerVersionEvidenceBinding,
        DenyDisclosureV2,
        OnDeviceDisclosureV2,
        DisclosureAudienceV2,
        ReviewedDisclosureV2,
        ProfileControlsV2,
        ProfileProvenanceV2,
        AttributionOriginV2,
        SupportAssessmentV2,
        ApprovalReceiptV2,
        ConfidenceEstimateV2,
        SalienceHintV2,
        ProfileClaimV2,
        ObjectSchemaRequirementV2,
        ProfileManifestV2,
        ProfileRecordV2,
        ProfileProposalV2,
    }
)
_KNOWN_KEYS = frozenset(k for model in _KNOWN_MODELS for k in model.model_fields)


# Exact wire shapes for leaf models whose original APIs/configuration stay V1.
_COMPOSED_CHILDREN = {
    "payload": {
        model.model_fields["kind"].default: model
        for model in v2_meaning._KNOWN_MODELS
        if "schema_version" in model.model_fields and "kind" in model.model_fields
    },
    "semantic_key": SemanticKey,
    "evidence_bindings": OwnerVersionEvidenceBinding,
    "temporal_validity": {
        "unknown": v2_meaning.UnknownValidityV2,
        "standing": v2_meaning.StandingValidityV2,
        "interval": v2_meaning.IntervalValidityV2,
    },
    "basis": {
        "user_reviewed": v2_meaning.UserReviewedBasisV2,
        "bound_source": v2_meaning.BoundSourceBasisV2,
    },
    "relations": {
        "correction_of": v2_meaning.CorrectionOfV2,
        "change_from": v2_meaning.ChangeFromV2,
        "supersedes": v2_meaning.SupersedesV2,
        "workspace_exception_to": v2_meaning.WorkspaceExceptionToV2,
    },
    "effect": {
        "all_target_validity": v2_meaning.AllTargetValidityV2,
        "overlap": v2_meaning.OverlapEffectV2,
        "replace_from": v2_meaning.ReplaceFromEffectV2,
    },
}
