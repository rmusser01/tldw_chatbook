"""Inactive V2 meaning identity; no record admission, source access or approval."""

import json
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
from typing import Annotated, Any, Literal, Self
from zoneinfo import ZoneInfo

from pydantic import ConfigDict, Field, ValidationInfo, model_validator

from .canonical import canonical_bytes
from .evidence_binding import CaptureTime, Identity
from .payloads import (
    ConstraintPayload,
    ConventionPayload,
    CorrectionPayload,
    FrozenModel,
    GoalPayload,
    IdentityPayload,
    LegacyUnclassifiedPayload,
    PreferencePayload,
    ProfilePayload,
    RelationshipPayload,
    WorkingContextPayload,
)

MAX_PAYLOAD_UTF8_BYTES = 16 * 1024
_MAX_SNAPSHOT_DEPTH = 8
_MAX_SNAPSHOT_NODES = 256
_MAX_CONTAINER_MEMBERS = 32
_CLAIM_BASES = frozenset(
    {"direct_user_assertion", "inference", "imported_assertion", "legacy_unknown"}
)
_KIND_TAGS = frozenset(
    {
        "identity",
        "preference",
        "relationship",
        "correction",
        "constraint",
        "goal",
        "convention",
        "working_context",
        "legacy_unclassified",
        "user_reviewed",
        "bound_source",
        "unknown",
        "standing",
        "interval",
        "all_target_validity",
        "overlap",
        "replace_from",
        "correction_of",
        "change_from",
        "supersedes",
        "workspace_exception_to",
    }
)


def _meaning_snapshot(value: object) -> object:
    """Copy only known exact values, retaining fields and rejecting extra state."""
    remaining = _MAX_SNAPSHOT_NODES

    def visit(item: object, depth: int) -> object:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > _MAX_SNAPSHOT_DEPTH:
            raise ValueError("meaning input exceeds structural bounds")
        kind = type(item)
        if any(kind is model for model in _KNOWN_MODELS):
            if object.__getattribute__(item, "__pydantic_extra__") is not None:
                raise ValueError("meaning model has unexpected extra state")
            state = object.__getattribute__(item, "__dict__")
            if type(state) is not dict:
                raise ValueError("meaning model has an unsupported state dictionary")
            return visit(state, depth + 1)
        if kind is dict:
            if len(item) > _MAX_CONTAINER_MEMBERS:
                raise ValueError("meaning input exceeds structural bounds")
            if any(type(key) is not str or key not in _KNOWN_KEYS for key in item):
                raise ValueError("meaning input has unknown fields")
            if "kind" in item and (
                type(item["kind"]) is not str or item["kind"] not in _KIND_TAGS
            ):
                raise ValueError("meaning input has an unsupported kind")
            if "projection" in item and (
                type(item["projection"]) is not str
                or item["projection"] != "profile-claim-v2"
            ):
                raise ValueError("meaning input has an unsupported projection")
            if "claim_basis" in item and (
                type(item["claim_basis"]) is not str
                or item["claim_basis"] not in _CLAIM_BASES
            ):
                raise ValueError("meaning input has an unsupported basis")
            return {key: visit(child, depth + 1) for key, child in item.items()}
        if kind is list or kind is tuple:
            if len(item) > _MAX_CONTAINER_MEMBERS:
                raise ValueError("meaning input exceeds structural bounds")
            return tuple(visit(child, depth + 1) for child in item)
        if kind is datetime:
            if item.tzinfo is not None and not any(
                type(item.tzinfo) is zone for zone in (timezone, ZoneInfo)
            ):
                raise ValueError("meaning time has an unsupported timezone type")
            return item
        if any(kind is scalar for scalar in (str, int, bool, type(None))):
            return item
        if kind is float and isfinite(item):
            return item
        raise ValueError("meaning input has an unsupported value type")

    return visit(value, 0)


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate meaning JSON member")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError("nonfinite meaning JSON constant")


class _MeaningModel(FrozenModel):
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
            raise TypeError("meaning JSON requires duplicate-aware model_validate_json")
        return _meaning_snapshot(value)

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: Any
    ) -> Self:
        """Validate JSON without silently accepting duplicate members.

        Args:
            json_data: Exact built-in JSON text/UTF-8 bytes.
            **kwargs: Pydantic validation options forwarded to model_validate.

        Returns:
            A freshly validated component instance.

        Raises:
            TypeError: Input is not an exact supported JSON scalar.
            ValueError: JSON is malformed, duplicate-bearing or invalid data.
        """
        if not any(type(json_data) is scalar for scalar in (str, bytes, bytearray)):
            raise TypeError("meaning JSON requires built-in text or bytes")
        try:
            text = (
                json_data
                if type(json_data) is str
                else json_data.decode("utf-8", errors="strict")
            )
            value = json.loads(
                text,
                object_pairs_hook=_unique_json_pairs,
                parse_constant=_reject_json_constant,
            )
        except (ValueError, RecursionError):
            raise ValueError("invalid meaning JSON") from None
        return cls.model_validate(value, **kwargs)


def _validate_interval(start: datetime | None, end: datetime | None) -> None:
    if start is None and end is None:
        raise ValueError("interval requires a bound")
    if start is not None and end is not None and start >= end:
        raise ValueError("interval bounds must be ordered")


class UserReviewedBasisV2(_MeaningModel):
    """Reported review basis, not a runtime approval receipt."""

    kind: Literal["user_reviewed"] = Field(repr=False)


class BoundSourceBasisV2(_MeaningModel):
    """Inert source-binding identities, not source permissions."""

    kind: Literal["bound_source"] = Field(repr=False)
    binding_ids: tuple[Identity, ...] = Field(min_length=1, max_length=8, repr=False)

    @model_validator(mode="after")
    def _ordered_ids(self) -> Self:
        if self.binding_ids != tuple(sorted(set(self.binding_ids))):
            raise ValueError("basis identities must be unique and sorted")
        return self


class UnknownValidityV2(_MeaningModel):
    """No dates are known or silently inferred."""

    kind: Literal["unknown"] = Field(repr=False)


class StandingValidityV2(_MeaningModel):
    """Explicit asserted standing validity; authority remains external."""

    kind: Literal["standing"] = Field(repr=False)
    basis: UserReviewedBasisV2 = Field(repr=False)


ValidityBasisV2 = Annotated[
    UserReviewedBasisV2 | BoundSourceBasisV2, Field(discriminator="kind")
]


class IntervalValidityV2(_MeaningModel):
    """An explicitly based half-open effective interval."""

    kind: Literal["interval"] = Field(repr=False)
    valid_from: CaptureTime | None = Field(repr=False)
    valid_until: CaptureTime | None = Field(repr=False)
    basis: ValidityBasisV2 = Field(repr=False)

    @model_validator(mode="after")
    def _ordered_interval(self) -> Self:
        _validate_interval(self.valid_from, self.valid_until)
        return self


TemporalValidityV2 = Annotated[
    UnknownValidityV2 | StandingValidityV2 | IntervalValidityV2,
    Field(discriminator="kind"),
]


class AllTargetValidityV2(_MeaningModel):
    """Asserted effect on a target; native admission checks its validity."""

    kind: Literal["all_target_validity"] = Field(repr=False)


class OverlapEffectV2(_MeaningModel):
    """Explicit interval; actual authorized overlap is runtime-owned."""

    kind: Literal["overlap"] = Field(repr=False)
    valid_from: CaptureTime | None = Field(repr=False)
    valid_until: CaptureTime | None = Field(repr=False)

    @model_validator(mode="after")
    def _ordered_interval(self) -> Self:
        _validate_interval(self.valid_from, self.valid_until)
        return self


class ReplaceFromEffectV2(_MeaningModel):
    """Replacement boundary, without claiming historical falsity."""

    kind: Literal["replace_from"] = Field(repr=False)
    replace_from: CaptureTime = Field(repr=False)


class _TargetEdgeV2(_MeaningModel):
    edge_id: Identity = Field(repr=False)
    target_record_id: Identity = Field(repr=False)
    target_version_id: Identity = Field(repr=False)


class CorrectionOfV2(_TargetEdgeV2):
    """Exact target assertion and correction effect."""

    kind: Literal["correction_of"] = Field(repr=False)
    effect: Annotated[
        AllTargetValidityV2 | OverlapEffectV2, Field(discriminator="kind")
    ] = Field(repr=False)


class ChangeFromV2(_TargetEdgeV2):
    """Exact change transition; historical projection is runtime-owned."""

    kind: Literal["change_from"] = Field(repr=False)
    transition_at: CaptureTime = Field(repr=False)


class SupersedesV2(_TargetEdgeV2):
    """Exact replacement with independent reason/effect semantics."""

    kind: Literal["supersedes"] = Field(repr=False)
    effect: Annotated[
        AllTargetValidityV2 | ReplaceFromEffectV2, Field(discriminator="kind")
    ] = Field(repr=False)
    reason_code: Identity = Field(repr=False)


class WorkspaceExceptionToV2(_TargetEdgeV2):
    """Asserted workspace exception, never a target-read capability."""

    kind: Literal["workspace_exception_to"] = Field(repr=False)
    workspace_scope_id: Identity = Field(repr=False)


TemporalRelationV2 = Annotated[
    CorrectionOfV2 | ChangeFromV2 | SupersedesV2 | WorkspaceExceptionToV2,
    Field(discriminator="kind"),
]


class ClaimMeaningV2(_MeaningModel):
    """Closed meaning projection, not a canonical profile record."""

    projection: Literal["profile-claim-v2"] = Field(repr=False)
    profile_id: Identity = Field(repr=False)
    record_id: Identity = Field(repr=False)
    scope_id: Identity = Field(repr=False)
    kind: Literal[
        "identity",
        "preference",
        "relationship",
        "correction",
        "constraint",
        "goal",
        "convention",
        "working_context",
        "legacy_unclassified",
    ] = Field(repr=False)
    payload: ProfilePayload = Field(repr=False)
    claim_basis: Literal[
        "direct_user_assertion", "inference", "imported_assertion", "legacy_unknown"
    ] = Field(repr=False)
    temporal_validity: TemporalValidityV2 = Field(repr=False)
    relations: tuple[TemporalRelationV2, ...] = Field(max_length=4, repr=False)

    @model_validator(mode="before")
    @classmethod
    def _safe_input(cls, value: object, info: ValidationInfo) -> object:
        snapshot = super()._safe_input(value, info)
        if type(snapshot) is dict and "kind" in snapshot:
            payload = snapshot.get("payload")
            if type(payload) is dict and "kind" not in payload:
                snapshot["payload"] = payload | {"kind": snapshot["kind"]}
        return snapshot

    @model_validator(mode="after")
    def _consistent_meaning(self) -> Self:
        if self.payload.kind != self.kind:
            raise ValueError("payload kind mismatch")
        try:
            payload_size = len(canonical_bytes(self.payload))
        except ValueError:
            raise ValueError("payload is not valid canonical text") from None
        if payload_size > MAX_PAYLOAD_UTF8_BYTES:
            raise ValueError("payload exceeds canonical byte limit")
        ids = tuple(edge.edge_id for edge in self.relations)
        targets = tuple(
            (edge.target_record_id, edge.target_version_id) for edge in self.relations
        )
        if ids != tuple(sorted(set(ids))) or len(set(targets)) != len(targets):
            raise ValueError("relations must be unique and sorted")
        changes = tuple(edge for edge in self.relations if edge.kind == "change_from")
        if len(changes) > 1:
            raise ValueError("only one change transition is allowed")
        for edge in changes:
            if (
                self.temporal_validity.kind != "interval"
                or self.temporal_validity.valid_from != edge.transition_at
            ):
                raise ValueError("change requires exact interval start")
        for edge in self.relations:
            if (
                edge.kind == "workspace_exception_to"
                and edge.workspace_scope_id != self.scope_id
            ):
                raise ValueError("workspace exception scope mismatch")
        if self.claim_basis == "legacy_unknown" and (
            self.temporal_validity.kind != "unknown" or self.relations
        ):
            raise ValueError("legacy meaning remains unknown")
        return self


_KNOWN_MODELS = frozenset(
    {
        UserReviewedBasisV2,
        BoundSourceBasisV2,
        UnknownValidityV2,
        StandingValidityV2,
        IntervalValidityV2,
        AllTargetValidityV2,
        OverlapEffectV2,
        ReplaceFromEffectV2,
        CorrectionOfV2,
        ChangeFromV2,
        SupersedesV2,
        WorkspaceExceptionToV2,
        ClaimMeaningV2,
        IdentityPayload,
        PreferencePayload,
        RelationshipPayload,
        CorrectionPayload,
        ConstraintPayload,
        GoalPayload,
        ConventionPayload,
        WorkingContextPayload,
        LegacyUnclassifiedPayload,
    }
)
_KNOWN_KEYS = frozenset(key for model in _KNOWN_MODELS for key in model.model_fields)


def claim_meaning_digest(projection: ClaimMeaningV2) -> str:
    """Hash a freshly validated snapshot of every included meaning field.

    Args:
        projection: Exact component instance; assertions remain untrusted.

    Returns:
        Lowercase SHA-256 of the validated projection's canonical UTF-8 bytes.

    Raises:
        TypeError: The argument is not the exact component class.
        ValueError: Complete or nested field values contain unsafe state.
        pydantic.ValidationError: Snapshot values violate the component contract.
    """
    if type(projection) is not ClaimMeaningV2:
        raise TypeError("projection must be an exact ClaimMeaningV2 instance")
    validated = ClaimMeaningV2.model_validate(_meaning_snapshot(projection))
    return sha256(canonical_bytes(validated)).hexdigest()
