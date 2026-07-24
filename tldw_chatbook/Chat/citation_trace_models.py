"""Pure, bounded contracts for immutable citation traces and governed payloads."""

from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Mapping, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)


IMMUTABLE_AGGREGATE_JSON_BYTES_MAX = 256 * 1024
SNAPSHOT_TEXT_UTF8_BYTES_MAX = 64 * 1024
GOVERNED_PAYLOAD_UTF8_BYTES_MAX = 4 * 1024 * 1024
PROMPT_EVIDENCE_SETS_MAX = 8
EVIDENCE_ENTRIES_PER_PROMPT_MAX = 64
ANSWER_ATTEMPTS_MAX = 8
CITATION_OCCURRENCES_MAX = 512
RETRIEVAL_CANDIDATES_PER_RUN_MAX = 200
EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX = 256
ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX = 1024 * 1024

TIMING_SUMMARIES_MAX = 16
POLICY_CAPABILITIES_MAX = 16
SHORT_CODE_CHARACTERS_MAX = 256
MARKER_CHARACTERS_MAX = 32
GOVERNED_DESCRIPTOR_JSON_BYTES_MAX = 16 * 1024

_CHATBOOK_MARKER = re.compile(r"\[S([1-9][0-9]*)\]\Z")
_LEGACY_NUMERIC_MARKER = re.compile(r"\[([1-9][0-9]*)\]\Z")


def _bounded_utf8(value: str, *, field_name: str, limit: int) -> str:
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    byte_count = len(value.encode("utf-8"))
    if byte_count > limit:
        raise ValueError(
            f"{field_name} exceeds {limit} UTF-8 bytes ({byte_count} provided)"
        )
    return value


def _opaque_identifier(value: str) -> str:
    return _bounded_utf8(
        value,
        field_name="opaque identifier",
        limit=EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX,
    )


OpaqueIdentifier = Annotated[str, AfterValidator(_opaque_identifier)]
ShortCode = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=SHORT_CODE_CHARACTERS_MAX,
    ),
]
MarkerText = Annotated[
    str,
    StringConstraints(strict=True, min_length=3, max_length=MARKER_CHARACTERS_MAX),
]


class TraceOrigin(str, Enum):
    """Origin that authored or inferred a citation trace."""

    LOCAL = "local"
    SERVER = "server"
    IMPORTED = "imported"
    LEGACY_INFERRED = "legacy_inferred"


class TraceLifecycle(str, Enum):
    """Persistable citation trace lifecycle."""

    SEALED = "sealed"


class CitationCompleteness(str, Enum):
    """Seal-time completeness of selected-answer provenance."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    REDACTED = "redacted"
    UNAVAILABLE = "unavailable"


class EvidenceStorageMode(str, Enum):
    """Governed evidence storage policy selected at seal time."""

    EMBEDDED = "embedded"
    SERVER_REFERENCE = "server_reference"
    EPHEMERAL = "ephemeral"
    REDACTED = "redacted"


class MarkerNamespace(str, Enum):
    """Versioned answer marker grammar."""

    CHATBOOK_S_V1 = "chatbook_s_v1"
    LEGACY_NUMERIC_V1 = "legacy_numeric_v1"


class OffsetBasis(str, Enum):
    """Coordinate system used by answer and claim spans."""

    UNICODE_CODEPOINT_V1 = "unicode_codepoint_v1"


class ClaimSupport(str, Enum):
    """Semantic support assessment independent of structural validity."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"
    NOT_CHECKED = "not_checked"


class StructuralValidationState(str, Enum):
    """Structural state of one parsed citation occurrence."""

    VALID = "valid"
    UNKNOWN_MARKER = "unknown_marker"
    INVALID_SPAN = "invalid_span"


class AnswerAttemptKind(str, Enum):
    """Bounded generation attempt category."""

    INITIAL = "initial"
    CITATION_REPAIR = "citation_repair"
    PIPELINE_RERUN = "pipeline_rerun"
    LEGACY_INFERRED = "legacy_inferred"


class PolicyCapability(str, Enum):
    """Seal-time policy capability retained in the immutable trace."""

    VIEW_SNAPSHOT = "view_snapshot"
    VIEW_SOURCE_IDENTITY = "view_source_identity"
    RESOLVE_CURRENT_SOURCE = "resolve_current_source"
    OPEN_NATIVE = "open_native"
    OPEN_EXTERNAL = "open_external"
    COMPARE_CURRENT_SOURCE = "compare_current_source"
    EXPORT_SNAPSHOT = "export_snapshot"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class TimingSummary(_StrictFrozenModel):
    """Small non-sensitive duration summary."""

    name: ShortCode
    milliseconds: float = Field(ge=0, le=86_400_000)


class StructuralTrustSummary(_StrictFrozenModel):
    """Bounded aggregate structural counts."""

    valid_occurrences: int = Field(default=0, ge=0, le=CITATION_OCCURRENCES_MAX)
    unknown_occurrences: int = Field(default=0, ge=0, le=CITATION_OCCURRENCES_MAX)
    invalid_spans: int = Field(default=0, ge=0, le=CITATION_OCCURRENCES_MAX)


class SemanticTrustSummary(_StrictFrozenModel):
    """Bounded aggregate semantic-support counts."""

    supported_claims: int = Field(default=0, ge=0, le=CITATION_OCCURRENCES_MAX)
    unsupported_claims: int = Field(default=0, ge=0, le=CITATION_OCCURRENCES_MAX)
    insufficient_claims: int = Field(default=0, ge=0, le=CITATION_OCCURRENCES_MAX)
    unknown_claims: int = Field(default=0, ge=0, le=CITATION_OCCURRENCES_MAX)
    not_checked_claims: int = Field(default=0, ge=0, le=CITATION_OCCURRENCES_MAX)


class EvidenceRun(_StrictFrozenModel):
    """Immutable relationship metadata for one retrieval execution."""

    schema_version: Literal[1] = 1
    run_id: OpaqueIdentifier
    request_id: OpaqueIdentifier
    run_ordinal: int = Field(ge=1)
    stage: ShortCode
    payload_ref: OpaqueIdentifier
    timings: tuple[TimingSummary, ...] = Field(
        default=(), max_length=TIMING_SUMMARIES_MAX
    )
    started_at: datetime
    ended_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_time_range(self) -> "EvidenceRun":
        _require_aware_datetime(self.started_at, "started_at")
        if self.ended_at is not None:
            _require_aware_datetime(self.ended_at, "ended_at")
            if self.ended_at < self.started_at:
                raise ValueError("ended_at must not precede started_at")
        _require_unique(self.timings, lambda item: item.name, "timing name")
        return self


class PromptEvidenceEntry(_StrictFrozenModel):
    """One exact prompt-boundary evidence relationship."""

    schema_version: Literal[1] = 1
    evidence_ordinal: int = Field(ge=1)
    marker_ordinal: int = Field(ge=1)
    run_id: OpaqueIdentifier
    snapshot_payload_ref: OpaqueIdentifier
    storage_mode: EvidenceStorageMode


class PromptEvidenceSet(_StrictFrozenModel):
    """Exact evidence submitted to one provider request."""

    schema_version: Literal[1] = 1
    prompt_set_id: OpaqueIdentifier
    prompt_set_ordinal: int = Field(ge=1)
    marker_namespace: MarkerNamespace
    entries: tuple[PromptEvidenceEntry, ...] = Field(
        default=(), max_length=EVIDENCE_ENTRIES_PER_PROMPT_MAX
    )
    created_at: datetime

    @model_validator(mode="after")
    def _validate_entries(self) -> "PromptEvidenceSet":
        _require_aware_datetime(self.created_at, "created_at")
        _require_unique(
            self.entries, lambda item: item.evidence_ordinal, "evidence_ordinal"
        )
        _require_unique(
            self.entries, lambda item: item.marker_ordinal, "marker_ordinal"
        )
        return self


class CitationOccurrence(_StrictFrozenModel):
    """One parsed marker span in the exact unrendered answer."""

    schema_version: Literal[1] = 1
    occurrence_id: OpaqueIdentifier
    occurrence_ordinal: int = Field(ge=1)
    raw_marker: MarkerText
    marker_namespace: MarkerNamespace
    evidence_ordinal: int | None = Field(default=None, ge=1)
    marker_start: int = Field(ge=0)
    marker_end: int = Field(ge=1)
    claim_start: int | None = Field(default=None, ge=0)
    claim_end: int | None = Field(default=None, ge=0)
    offset_basis: OffsetBasis = OffsetBasis.UNICODE_CODEPOINT_V1
    structural_state: StructuralValidationState
    claim_support: ClaimSupport = ClaimSupport.NOT_CHECKED

    @model_validator(mode="after")
    def _validate_marker(self) -> "CitationOccurrence":
        matcher = (
            _CHATBOOK_MARKER
            if self.marker_namespace is MarkerNamespace.CHATBOOK_S_V1
            else _LEGACY_NUMERIC_MARKER
        )
        namespace = self.marker_namespace.value
        if matcher.fullmatch(self.raw_marker) is None:
            raise ValueError(f"{namespace} marker does not match its grammar")
        if self.marker_end - self.marker_start != len(self.raw_marker):
            raise ValueError("marker span length must equal raw_marker length")
        if (self.claim_start is None) != (self.claim_end is None):
            raise ValueError("claim_start and claim_end must be supplied together")
        if (
            self.claim_start is not None
            and self.claim_end is not None
            and self.claim_end < self.claim_start
        ):
            raise ValueError("claim_end must not precede claim_start")
        if self.evidence_ordinal is None:
            if self.structural_state is not StructuralValidationState.UNKNOWN_MARKER:
                raise ValueError(
                    "unknown markers require structural_state=unknown_marker"
                )
        elif self.structural_state is StructuralValidationState.UNKNOWN_MARKER:
            raise ValueError("known evidence cannot use unknown_marker state")
        return self

    @property
    def marker_ordinal(self) -> int:
        """Return the positive ordinal encoded in ``raw_marker``."""

        matcher = (
            _CHATBOOK_MARKER
            if self.marker_namespace is MarkerNamespace.CHATBOOK_S_V1
            else _LEGACY_NUMERIC_MARKER
        )
        match = matcher.fullmatch(self.raw_marker)
        if match is None:  # guarded by validation
            raise ValueError("invalid marker")
        return int(match.group(1))


class AnswerAttempt(_StrictFrozenModel):
    """Immutable metadata for one generation or repair attempt."""

    schema_version: Literal[1] = 1
    attempt_id: OpaqueIdentifier
    attempt_ordinal: int = Field(ge=1)
    kind: AnswerAttemptKind
    prompt_evidence_set_id: OpaqueIdentifier
    answer_payload_ref: OpaqueIdentifier | None = None
    occurrences: tuple[CitationOccurrence, ...] = Field(
        default=(), max_length=CITATION_OCCURRENCES_MAX
    )
    structural_summary: StructuralTrustSummary = Field(
        default_factory=StructuralTrustSummary
    )
    semantic_summary: SemanticTrustSummary = Field(default_factory=SemanticTrustSummary)
    repair_reason_code: ShortCode | None = None
    created_at: datetime

    @model_validator(mode="after")
    def _validate_occurrences(self) -> "AnswerAttempt":
        _require_aware_datetime(self.created_at, "created_at")
        _require_unique(
            self.occurrences,
            lambda item: item.occurrence_id,
            "occurrence_id",
        )
        _require_unique(
            self.occurrences,
            lambda item: item.occurrence_ordinal,
            "occurrence_ordinal",
        )
        _require_unique(
            self.occurrences,
            lambda item: (item.marker_start, item.marker_end),
            "marker span",
        )
        return self


class CitationTrace(_StrictFrozenModel):
    """Sealed immutable answer-level citation provenance aggregate."""

    schema_version: Literal[1] = 1
    trace_id: OpaqueIdentifier
    request_id: OpaqueIdentifier
    generation_id: OpaqueIdentifier
    origin: TraceOrigin
    lifecycle: TraceLifecycle
    completeness_at_seal: CitationCompleteness
    evidence_runs: tuple[EvidenceRun, ...] = Field(min_length=1)
    prompt_evidence_sets: tuple[PromptEvidenceSet, ...] = Field(
        min_length=1, max_length=PROMPT_EVIDENCE_SETS_MAX
    )
    answer_attempts: tuple[AnswerAttempt, ...] = Field(
        min_length=1, max_length=ANSWER_ATTEMPTS_MAX
    )
    selected_attempt_id: OpaqueIdentifier
    structural_trust: StructuralTrustSummary = Field(
        default_factory=StructuralTrustSummary
    )
    semantic_trust: SemanticTrustSummary = Field(default_factory=SemanticTrustSummary)
    policy_capabilities: tuple[PolicyCapability, ...] = Field(
        default=(), max_length=POLICY_CAPABILITIES_MAX
    )
    policy_version: OpaqueIdentifier
    created_at: datetime
    sealed_at: datetime

    @model_validator(mode="after")
    def _validate_graph(self) -> "CitationTrace":
        if self.lifecycle is not TraceLifecycle.SEALED:
            raise ValueError("citation trace lifecycle must be sealed")
        _require_aware_datetime(self.created_at, "created_at")
        _require_aware_datetime(self.sealed_at, "sealed_at")
        if self.sealed_at < self.created_at:
            raise ValueError("sealed_at must not precede created_at")

        _require_unique(self.evidence_runs, lambda item: item.run_id, "run_id")
        _require_unique(
            self.evidence_runs, lambda item: item.run_ordinal, "run_ordinal"
        )
        _require_unique(
            self.prompt_evidence_sets,
            lambda item: item.prompt_set_id,
            "prompt_set_id",
        )
        _require_unique(
            self.prompt_evidence_sets,
            lambda item: item.prompt_set_ordinal,
            "prompt_set_ordinal",
        )
        _require_unique(
            self.answer_attempts, lambda item: item.attempt_id, "attempt_id"
        )
        _require_unique(
            self.answer_attempts,
            lambda item: item.attempt_ordinal,
            "attempt_ordinal",
        )
        _require_unique(
            self.policy_capabilities, lambda item: item, "policy capability"
        )

        runs = {run.run_id for run in self.evidence_runs}
        if any(run.request_id != self.request_id for run in self.evidence_runs):
            raise ValueError("evidence run request_id must match trace request_id")
        prompt_sets = {
            prompt_set.prompt_set_id: prompt_set
            for prompt_set in self.prompt_evidence_sets
        }
        attempts = {attempt.attempt_id: attempt for attempt in self.answer_attempts}
        if self.selected_attempt_id not in attempts:
            raise ValueError("selected_attempt_id must identify one answer attempt")

        for prompt_set in self.prompt_evidence_sets:
            for entry in prompt_set.entries:
                if entry.run_id not in runs:
                    raise ValueError(
                        f"prompt entry references unknown evidence run {entry.run_id!r}"
                    )
        for attempt in self.answer_attempts:
            prompt_set = prompt_sets.get(attempt.prompt_evidence_set_id)
            if prompt_set is None:
                raise ValueError(
                    "answer attempt references unknown prompt evidence set "
                    f"{attempt.prompt_evidence_set_id!r}"
                )
            entries = {entry.evidence_ordinal: entry for entry in prompt_set.entries}
            for occurrence in attempt.occurrences:
                if occurrence.marker_namespace is not prompt_set.marker_namespace:
                    raise ValueError(
                        "citation occurrence marker namespace differs from prompt set"
                    )
                if occurrence.evidence_ordinal is None:
                    continue
                entry = entries.get(occurrence.evidence_ordinal)
                if entry is None:
                    raise ValueError(
                        "citation occurrence references unknown evidence ordinal"
                    )
                if occurrence.marker_ordinal != entry.marker_ordinal:
                    raise ValueError(
                        "citation occurrence marker does not resolve to evidence"
                    )

        validate_aggregate_json_bytes(_canonical_json_bytes(self))
        return self


class RetrievalCandidatePayload(_StrictFrozenModel):
    """Governed metadata for one retained retrieval candidate."""

    schema_version: Literal[1] = 1
    candidate_id: OpaqueIdentifier
    rank: int = Field(ge=1, le=RETRIEVAL_CANDIDATES_PER_RUN_MAX)
    source_identity: dict[str, JsonValue] = Field(default_factory=dict)
    title: str | None = None
    locator: dict[str, JsonValue] = Field(default_factory=dict)
    lineage: dict[str, JsonValue] = Field(default_factory=dict)
    score_kind: ShortCode | None = None
    score: float | None = None

    @model_validator(mode="after")
    def _validate_descriptors(self) -> "RetrievalCandidatePayload":
        _validate_governed_descriptors(
            self.source_identity,
            self.locator,
            self.lineage,
        )
        return self


class EvidenceRunPayload(_StrictFrozenModel):
    """Governed query, authority, and candidate details for one run."""

    schema_version: Literal[1] = 1
    payload_id: OpaqueIdentifier
    run_id: OpaqueIdentifier
    raw_query: str | None = None
    query_fingerprint: OpaqueIdentifier | None = None
    authority_id: OpaqueIdentifier | None = None
    retrieval_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    candidates: tuple[RetrievalCandidatePayload, ...] = Field(
        default=(), max_length=RETRIEVAL_CANDIDATES_PER_RUN_MAX
    )

    @model_validator(mode="after")
    def _validate_payload(self) -> "EvidenceRunPayload":
        _validate_governed_descriptors(self.retrieval_metadata)
        _require_unique(self.candidates, lambda item: item.candidate_id, "candidate_id")
        _require_unique(self.candidates, lambda item: item.rank, "candidate rank")
        return self


class EvidenceSnapshotPayload(_StrictFrozenModel):
    """Governed exact evidence and source descriptor."""

    schema_version: Literal[1] = 1
    payload_id: OpaqueIdentifier
    storage_mode: EvidenceStorageMode
    snapshot_text: str | None = None
    server_reference: OpaqueIdentifier | None = None
    title: str | None = None
    source_identity: dict[str, JsonValue] = Field(default_factory=dict)
    locator: dict[str, JsonValue] = Field(default_factory=dict)
    lineage: dict[str, JsonValue] = Field(default_factory=dict)
    transformations: tuple[ShortCode, ...] = Field(default=(), max_length=32)
    content_hash: OpaqueIdentifier | None = None
    comparison_hash: OpaqueIdentifier | None = None

    @field_validator("snapshot_text")
    @classmethod
    def _validate_snapshot_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        byte_count = len(value.encode("utf-8"))
        if byte_count > SNAPSHOT_TEXT_UTF8_BYTES_MAX:
            raise ValueError(
                f"snapshot_text exceeds {SNAPSHOT_TEXT_UTF8_BYTES_MAX} UTF-8 bytes"
            )
        return value

    @model_validator(mode="after")
    def _validate_payload(self) -> "EvidenceSnapshotPayload":
        if (
            self.storage_mode is EvidenceStorageMode.EMBEDDED
            and self.snapshot_text is None
        ):
            raise ValueError("embedded snapshot requires snapshot_text")
        if (
            self.storage_mode is EvidenceStorageMode.SERVER_REFERENCE
            and self.server_reference is None
        ):
            raise ValueError("server_reference storage requires server_reference")
        if self.storage_mode is EvidenceStorageMode.REDACTED and (
            self.snapshot_text is not None or self.server_reference is not None
        ):
            raise ValueError("redacted snapshot cannot retain text or reference")
        _validate_governed_descriptors(
            self.source_identity,
            self.locator,
            self.lineage,
        )
        return self


class AnswerAttemptPayload(_StrictFrozenModel):
    """Governed exact answer body for one generation attempt."""

    schema_version: Literal[1] = 1
    payload_id: OpaqueIdentifier
    attempt_id: OpaqueIdentifier
    answer_body: str | None = None
    body_integrity_hmac: OpaqueIdentifier | None = None

    @field_validator("answer_body")
    @classmethod
    def _validate_answer_body(cls, value: str | None) -> str | None:
        if value is None:
            return None
        byte_count = len(value.encode("utf-8"))
        if byte_count > ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX:
            raise ValueError(
                f"answer_body exceeds {ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX} UTF-8 bytes"
            )
        return value


GovernedPayload = EvidenceRunPayload | EvidenceSnapshotPayload | AnswerAttemptPayload


class SealedCitationWrite(_StrictFrozenModel):
    """One sealed trace plus the exact governed payload graph it references."""

    schema_version: Literal[1] = 1
    trace: CitationTrace
    evidence_run_payloads: tuple[EvidenceRunPayload, ...] = ()
    evidence_snapshot_payloads: tuple[EvidenceSnapshotPayload, ...] = ()
    answer_attempt_payloads: tuple[AnswerAttemptPayload, ...] = ()

    @property
    def governed_payload_bytes(self) -> int:
        """Return canonical serialized bytes for every governed payload."""

        return sum(
            len(_canonical_json_bytes(payload))
            for payload in self._all_governed_payloads()
        )

    def _all_governed_payloads(self) -> tuple[GovernedPayload, ...]:
        return (
            *self.evidence_run_payloads,
            *self.evidence_snapshot_payloads,
            *self.answer_attempt_payloads,
        )

    @model_validator(mode="after")
    def _validate_complete_graph(self) -> "SealedCitationWrite":
        if self.trace.lifecycle is not TraceLifecycle.SEALED:
            raise ValueError("SealedCitationWrite requires a sealed trace")

        _reject_duplicate_payloads(self.evidence_run_payloads, "governed run payload")
        _reject_duplicate_payloads(
            self.evidence_snapshot_payloads, "governed snapshot payload"
        )
        _reject_duplicate_payloads(
            self.answer_attempt_payloads, "governed answer payload"
        )
        _require_unique(
            self._all_governed_payloads(),
            lambda item: item.payload_id,
            "governed payload_id",
        )

        run_payloads = {
            payload.payload_id: payload for payload in self.evidence_run_payloads
        }
        snapshot_payloads = {
            payload.payload_id: payload for payload in self.evidence_snapshot_payloads
        }
        answer_payloads = {
            payload.payload_id: payload for payload in self.answer_attempt_payloads
        }
        required_run_payloads = {run.payload_ref for run in self.trace.evidence_runs}
        required_snapshot_payloads = {
            entry.snapshot_payload_ref
            for prompt_set in self.trace.prompt_evidence_sets
            for entry in prompt_set.entries
        }
        required_answer_payloads = {
            attempt.answer_payload_ref
            for attempt in self.trace.answer_attempts
            if attempt.answer_payload_ref is not None
        }
        _require_exact_payload_refs(
            required_run_payloads,
            set(run_payloads),
            "governed run payload",
        )
        _require_exact_payload_refs(
            required_snapshot_payloads,
            set(snapshot_payloads),
            "governed snapshot payload",
        )
        _require_exact_payload_refs(
            required_answer_payloads,
            set(answer_payloads),
            "governed answer payload",
        )

        for run in self.trace.evidence_runs:
            if run_payloads[run.payload_ref].run_id != run.run_id:
                raise ValueError("governed run payload belongs to another run")
        for prompt_set in self.trace.prompt_evidence_sets:
            for entry in prompt_set.entries:
                payload = snapshot_payloads[entry.snapshot_payload_ref]
                if payload.storage_mode is not entry.storage_mode:
                    raise ValueError(
                        "snapshot payload storage mode differs from prompt entry"
                    )
        for attempt in self.trace.answer_attempts:
            if attempt.answer_payload_ref is None:
                if attempt.occurrences:
                    raise ValueError(
                        "attempt occurrences require a governed answer payload"
                    )
                continue
            payload = answer_payloads[attempt.answer_payload_ref]
            if payload.attempt_id != attempt.attempt_id:
                raise ValueError("governed answer payload belongs to another attempt")
            _validate_answer_offsets(attempt, payload)

        reduced = reduce_selected_attempt_completeness(self.trace, snapshot_payloads)
        if self.trace.completeness_at_seal is not reduced:
            raise ValueError(
                "completeness_at_seal does not match selected attempt reduction"
            )
        if self.governed_payload_bytes > GOVERNED_PAYLOAD_UTF8_BYTES_MAX:
            raise ValueError(
                "governed payload exceeds "
                f"{GOVERNED_PAYLOAD_UTF8_BYTES_MAX} UTF-8 bytes"
            )
        return self


def reduce_selected_attempt_completeness(
    trace: CitationTrace,
    payload_index: Mapping[str, EvidenceSnapshotPayload],
) -> CitationCompleteness:
    """Reduce only the selected attempt and its exact prompt evidence set.

    Args:
        trace: Sealed immutable trace.
        payload_index: Governed snapshot payloads keyed by opaque payload ID.

    Returns:
        Deterministic worst-state completeness for the selected final set.
    """

    attempts = {attempt.attempt_id: attempt for attempt in trace.answer_attempts}
    selected = attempts.get(trace.selected_attempt_id)
    if selected is None:
        return CitationCompleteness.UNAVAILABLE
    prompt_sets = {
        prompt_set.prompt_set_id: prompt_set
        for prompt_set in trace.prompt_evidence_sets
    }
    prompt_set = prompt_sets.get(selected.prompt_evidence_set_id)
    if prompt_set is None or not prompt_set.entries:
        return CitationCompleteness.UNAVAILABLE

    worst = CitationCompleteness.COMPLETE
    precedence = {
        CitationCompleteness.COMPLETE: 0,
        CitationCompleteness.PARTIAL: 1,
        CitationCompleteness.REDACTED: 2,
        CitationCompleteness.UNAVAILABLE: 3,
    }
    for entry in prompt_set.entries:
        payload = payload_index.get(entry.snapshot_payload_ref)
        if payload is None or payload.storage_mode is not entry.storage_mode:
            return CitationCompleteness.UNAVAILABLE
        current = {
            EvidenceStorageMode.EMBEDDED: CitationCompleteness.COMPLETE,
            EvidenceStorageMode.SERVER_REFERENCE: CitationCompleteness.COMPLETE,
            EvidenceStorageMode.EPHEMERAL: CitationCompleteness.PARTIAL,
            EvidenceStorageMode.REDACTED: CitationCompleteness.REDACTED,
        }[entry.storage_mode]
        if precedence[current] > precedence[worst]:
            worst = current
    return worst


def validate_aggregate_json_bytes(payload: bytes | str) -> int:
    """Accept aggregate JSON at the v1 byte limit and reject larger input."""

    encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
    byte_count = len(encoded)
    if byte_count > IMMUTABLE_AGGREGATE_JSON_BYTES_MAX:
        raise ValueError(
            "immutable aggregate JSON exceeds "
            f"{IMMUTABLE_AGGREGATE_JSON_BYTES_MAX} bytes"
        )
    return byte_count


def _canonical_json_bytes(value: BaseModel | Mapping[str, Any]) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_governed_descriptors(*values: Mapping[str, JsonValue]) -> None:
    for value in values:
        if len(_canonical_json_bytes(value)) > GOVERNED_DESCRIPTOR_JSON_BYTES_MAX:
            raise ValueError(
                "governed descriptor exceeds "
                f"{GOVERNED_DESCRIPTOR_JSON_BYTES_MAX} bytes"
            )


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


ItemT = TypeVar("ItemT")


def _require_unique(
    values: tuple[ItemT, ...],
    key: Any,
    field_name: str,
) -> None:
    observed: set[Any] = set()
    for value in values:
        identity = key(value)
        if identity in observed:
            raise ValueError(f"{field_name} values must be unique")
        observed.add(identity)


def _reject_duplicate_payloads(
    payloads: tuple[GovernedPayload, ...],
    label: str,
) -> None:
    identities = [payload.payload_id for payload in payloads]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate {label}")


def _require_exact_payload_refs(
    required: set[str],
    supplied: set[str],
    label: str,
) -> None:
    missing = required - supplied
    if missing:
        raise ValueError(f"missing {label}: {sorted(missing)!r}")
    extraneous = supplied - required
    if extraneous:
        raise ValueError(f"extraneous {label}: {sorted(extraneous)!r}")


def _validate_answer_offsets(
    attempt: AnswerAttempt,
    payload: AnswerAttemptPayload,
) -> None:
    body = payload.answer_body
    if body is None and attempt.occurrences:
        raise ValueError("answer offsets require a retained answer body")
    if body is None:
        return
    for occurrence in attempt.occurrences:
        if (
            occurrence.marker_end > len(body)
            or body[occurrence.marker_start : occurrence.marker_end]
            != occurrence.raw_marker
        ):
            raise ValueError("citation occurrence answer offsets do not match body")
        if occurrence.claim_end is not None and occurrence.claim_end > len(body):
            raise ValueError("citation occurrence claim offsets exceed answer body")


__all__ = [
    "ANSWER_ATTEMPT_BODY_UTF8_BYTES_MAX",
    "ANSWER_ATTEMPTS_MAX",
    "CITATION_OCCURRENCES_MAX",
    "EVIDENCE_ENTRIES_PER_PROMPT_MAX",
    "EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX",
    "GOVERNED_PAYLOAD_UTF8_BYTES_MAX",
    "IMMUTABLE_AGGREGATE_JSON_BYTES_MAX",
    "PROMPT_EVIDENCE_SETS_MAX",
    "RETRIEVAL_CANDIDATES_PER_RUN_MAX",
    "SNAPSHOT_TEXT_UTF8_BYTES_MAX",
    "AnswerAttempt",
    "AnswerAttemptKind",
    "AnswerAttemptPayload",
    "CitationCompleteness",
    "CitationOccurrence",
    "CitationTrace",
    "ClaimSupport",
    "EvidenceRun",
    "EvidenceRunPayload",
    "EvidenceSnapshotPayload",
    "EvidenceStorageMode",
    "MarkerNamespace",
    "OffsetBasis",
    "PolicyCapability",
    "PromptEvidenceEntry",
    "PromptEvidenceSet",
    "RetrievalCandidatePayload",
    "SealedCitationWrite",
    "SemanticTrustSummary",
    "StructuralTrustSummary",
    "StructuralValidationState",
    "TimingSummary",
    "TraceLifecycle",
    "TraceOrigin",
    "reduce_selected_attempt_completeness",
    "validate_aggregate_json_bytes",
]
