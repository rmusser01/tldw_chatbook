"""Strict, immutable review wire records and admission arithmetic (ADR-281).

References are lookup keys, never authority. Source membership, caller roles,
permissions, terminal receipts and total quotas belong to the owning services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Annotated, Literal, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from tldw_chatbook.Utils.input_validation import validate_text_input

MIB = 1024 * 1024
RECORD_MAX_BYTES = 8 * 1024
SUBMISSION_MAX_RECORDS = 8
GUIDE_MAX_BYTES = 12 * 1024
SOURCE_MAX_BYTES = 64 * MIB
REPORTS_MAX_BYTES = 16 * MIB
PARTICIPANT_MAX_BYTES = MIB
REVIEW_MAX_BYTES = 80 * MIB
REVIEW_MAX_FILES = 2048
TAB_MAX_BYTES = 160 * MIB
TAB_MAX_CONTEXTS = 4
VERIFICATION_MAX_UNITS = 8
VERIFICATION_MAX_CLAIM_TOKENS = 4096
EVIDENCE_PAGE_MAX_TOKENS = 1024


class ReviewRefused(ValueError):
    """Stable feature refusal; preserve upstream authority/routing reasons."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ReviewWireModel(BaseModel):
    """Reject coercion and extra fields; freeze records after admission."""

    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, revalidate_instances="always"
    )


def _wire_tuple(value: object) -> object:
    # JSON arrays become immutable internally without coercing their elements.
    return tuple(value) if isinstance(value, list) else value


T = TypeVar("T")
WireTuple = Annotated[tuple[T, ...], BeforeValidator(_wire_tuple)]
NonemptyText = Annotated[str, Field(min_length=1, pattern=r"\S")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
CommitId = Annotated[str, Field(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")]


def _relative_path(value: str) -> str:
    if (
        not value
        or value.startswith(("/", "\\"))
        or "\\" in value
        or "\x00" in value
        or ".." in value.split("/")
        or ":" in value.split("/")[0]
    ):
        raise ValueError("expected a relative binding path")
    return value


RelativePath = Annotated[
    str,
    BeforeValidator(
        lambda value: _relative_path(value) if isinstance(value, str) else value
    ),
]


class VersionedRecord(ReviewWireModel):
    schema_version: Literal[1]

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported schema version")
        return value


class SourceRange(ReviewWireModel):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered_lines(self) -> SourceRange:
        if self.end_line < self.start_line:
            raise ValueError("end_line precedes start_line")
        return self


class SourceLocation(SourceRange):
    source_ref: NonemptyText


class SourceEvidence(SourceLocation):
    kind: Literal["source"]
    digest: Digest


class CheckEvidence(ReviewWireModel):
    kind: Literal["check"]
    receipt_id: NonemptyText


EvidenceRef = Annotated[SourceEvidence | CheckEvidence, Field(discriminator="kind")]


class FindingRecord(VersionedRecord):
    local_key: NonemptyText
    title: NonemptyText
    locations: Annotated[WireTuple[SourceLocation], Field(min_length=1)]
    trigger: NonemptyText
    impact: NonemptyText
    evidence: WireTuple[EvidenceRef]
    severity: Literal["critical", "high", "medium", "low"]
    uncertainty: str


class CoverageGap(ReviewWireModel):
    reason: NonemptyText
    source_ref: NonemptyText | None = None


class CoverageRecord(VersionedRecord):
    inspected_refs: WireTuple[NonemptyText]
    gaps: WireTuple[CoverageGap]
    empty_findings: bool


class DispositionRecord(VersionedRecord):
    verification_unit_id: NonemptyText
    verdict: Literal["confirmed", "refuted", "unresolved"]
    reason: NonemptyText
    evidence: WireTuple[EvidenceRef]
    duplicate_unit_id: NonemptyText | None = None
    next_check: NonemptyText | None = None

    @model_validator(mode="after")
    def unresolved_next_check(self) -> DispositionRecord:
        if self.verdict == "unresolved" and self.next_check is None:
            raise ValueError("unresolved dispositions require a settling next_check")
        return self


class CheckProposal(VersionedRecord):
    argv: Annotated[WireTuple[NonemptyText], Field(min_length=1)]
    intent: NonemptyText
    verification_unit_ids: Annotated[WireTuple[NonemptyText], Field(min_length=1)]


class CheckActionRecord(VersionedRecord):
    proposal_id: NonemptyText
    action: Literal["claim", "decline"]
    reason: NonemptyText | None = None


class CheckReceipt(VersionedRecord):
    receipt_id: NonemptyText
    proposal_id: NonemptyText
    review_id: NonemptyText
    verifier_attempt_id: NonemptyText
    target_digest: Digest
    outcome: Literal[
        "executed", "declined", "unavailable", "cancelled", "expired", "uncertain"
    ]
    evidence: str
    native_execution_receipt_id: NonemptyText | None = None

    @model_validator(mode="after")
    def execution_identity(self) -> CheckReceipt:
        if (
            self.outcome == "executed" and self.native_execution_receipt_id is None
        ) or (
            self.native_execution_receipt_id is not None
            and self.outcome not in {"executed", "uncertain"}
        ):
            raise ValueError(
                "native execution receipt required for executed checks and permitted for uncertain execution"
            )
        return self


class ReviewerChoice(ReviewWireModel):
    preset_id: NonemptyText
    focus: str = ""

    @field_validator("focus")
    @classmethod
    def valid_focus(cls, value: str) -> str:
        if not validate_text_input(value):
            raise ValueError("invalid review focus")
        return value


class ReviewDestinationOverride(ReviewWireModel):
    provider: NonemptyText | None = None
    model: NonemptyText | None = None

    @model_validator(mode="after")
    def supplied_destination(self) -> ReviewDestinationOverride:
        if self.provider is None and self.model is None:
            raise ValueError("destination override is empty")
        return self


class ReviewRunChoice(ReviewerChoice):
    destination: ReviewDestinationOverride | None = None


class ReviewVerifierChoice(ReviewRunChoice):
    focus: Literal[""] = ""


class ReviewTeam(ReviewWireModel):
    reviewers: WireTuple[ReviewerChoice] = ()
    verifier_preset_id: NonemptyText | None = None

    @field_validator("verifier_preset_id", mode="before")
    @classmethod
    def unset_verifier(cls, value: object) -> object:
        return None if value == "" else value


ReviewRole = Literal["primary", "reviewer", "verifier"]


class ReviewRunBinding(ReviewWireModel):
    review_id: NonemptyText
    slot_id: NonemptyText
    attempt_id: NonemptyText
    role: ReviewRole
    target_digest: Digest
    source_epoch: int = Field(ge=0)
    check_feedback_mode: Literal["primary_handoff", "unavailable_inline"] | None = None

    @model_validator(mode="after")
    def verifier_feedback(self) -> ReviewRunBinding:
        if (self.role == "verifier") != (self.check_feedback_mode is not None):
            raise ValueError("feedback mode required only for verifier bindings")
        return self


class TargetScope(ReviewWireModel):
    binding_ref: NonemptyText
    scope: RelativePath = "."
    context_paths: WireTuple[RelativePath] = ()


class CommitRangeTarget(TargetScope):
    kind: Literal["commit_range"]
    base_commit: CommitId
    head_commit: CommitId
    comparison: Literal["direct", "merge_base"]


class PullRequestTarget(TargetScope):
    kind: Literal["pr"]
    pr_ref: NonemptyText
    base_commit: CommitId
    head_commit: CommitId
    comparison: Literal["direct", "merge_base"]


class PatchTarget(TargetScope):
    kind: Literal["patch"]
    patch_ref: NonemptyText
    preimage_commit: CommitId | None = None


class WorkingTreeTarget(TargetScope):
    kind: Literal["working_tree"]
    staged: bool
    unstaged: bool
    untracked_paths: WireTuple[RelativePath] = ()


ReviewTarget = Annotated[
    PullRequestTarget | CommitRangeTarget | PatchTarget | WorkingTreeTarget,
    Field(discriminator="kind"),
]
SourceSide = Literal["base", "head", "staged", "worktree"]


class ListSourceQuery(ReviewWireModel):
    op: Literal["list"]
    relative_dir: RelativePath
    side: SourceSide


class SearchSourceQuery(ReviewWireModel):
    op: Literal["search"]
    text: NonemptyText
    relative_dir: RelativePath
    side: SourceSide


class ResolveSourceQuery(ReviewWireModel):
    op: Literal["resolve"]
    relative_path: RelativePath
    side: SourceSide


SourceQuery = Annotated[
    ListSourceQuery | SearchSourceQuery | ResolveSourceQuery, Field(discriminator="op")
]


@dataclass(frozen=True)
class ReviewLimits:
    """Internal cap values; smaller limits can be injected by owning-store tests."""

    source_max_bytes: int = SOURCE_MAX_BYTES
    reports_max_bytes: int = REPORTS_MAX_BYTES
    participant_max_bytes: int = PARTICIPANT_MAX_BYTES
    review_max_bytes: int = REVIEW_MAX_BYTES
    review_max_files: int = REVIEW_MAX_FILES
    tab_max_bytes: int = TAB_MAX_BYTES
    tab_max_contexts: int = TAB_MAX_CONTEXTS

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or not 1 <= value <= field.default:
                raise ValueError(
                    "review storage limits must be positive and no larger than production caps"
                )


def canonical_record_bytes(record: dict) -> bytes:
    """Encode every wire field canonically, using actual UTF-8 bytes."""
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def review_result_char_limit(max_tool_result_chars: int) -> int:
    """Keep the native continuation ceiling even for unlimited runtime results."""
    return min(max_tool_result_chars, 16000) if max_tool_result_chars > 0 else 16000


_RECORD_KINDS = {
    "findings": FindingRecord,
    "coverage": CoverageRecord,
    "reviewer_complete": CoverageRecord,
    "dispositions": DispositionRecord,
    "check_proposals": CheckProposal,
    "check_actions": CheckActionRecord,
}
ReviewRecord = (
    FindingRecord
    | CoverageRecord
    | DispositionRecord
    | CheckProposal
    | CheckActionRecord
)


def validate_record_chunk(kind: str, records: list[dict]) -> tuple[ReviewRecord, ...]:
    """Validate the entire bounded chunk before returning any admitted record.

    Raises:
        ReviewRefused: For unknown kinds, invalid counts, bytes or schemas.
    """
    if not isinstance(records, list):
        raise ReviewRefused("invalid_record_chunk")
    if kind == "verifier_complete":
        if records:
            raise ReviewRefused("completion_records_not_empty")
        return ()
    model = _RECORD_KINDS.get(kind) if isinstance(kind, str) else None
    if model is None:
        raise ReviewRefused("unknown_submission_kind")
    if not 1 <= len(records) <= SUBMISSION_MAX_RECORDS:
        raise ReviewRefused("invalid_record_count")
    if (
        kind in ("coverage", "reviewer_complete", "check_proposals", "check_actions")
        and len(records) != 1
    ):
        raise ReviewRefused("single_record_required")
    parsed = []
    local_keys = set()
    try:
        for record in records:
            if not isinstance(record, dict):
                raise ReviewRefused("invalid_record")
            if len(canonical_record_bytes(record)) > RECORD_MAX_BYTES:
                raise ReviewRefused("record_too_large")
            item = model.model_validate(record)
            if isinstance(item, FindingRecord):
                if item.local_key in local_keys:
                    raise ReviewRefused("duplicate_local_key")
                local_keys.add(item.local_key)
            parsed.append(item)
    except (ValidationError, TypeError, ValueError, UnicodeError) as exc:
        if isinstance(exc, ReviewRefused):
            raise
        refusal = ReviewRefused("invalid_record")
        if isinstance(exc, ValidationError):
            refusal.validation_errors = tuple(
                {
                    "field": ("records", len(parsed), *error["loc"]),
                    "type": error["type"],
                }
                for error in exc.errors(include_input=False, include_url=False)
            )
        raise refusal from exc
    return tuple(parsed)


def minimum_continuations(reviewers: int, verification: bool, remaining: int) -> int:
    """Count additional two-child turns without changing the native budget.

    Raises:
        ReviewRefused: For malformed counts or fewer than two reviewers.
    """
    if (
        type(reviewers) is not int
        or reviewers < 2
        or type(remaining) is not int
        or remaining < 0
        or type(verification) is not bool
    ):
        raise ReviewRefused("invalid_review_admission_counts")
    additional_children = max(reviewers + int(verification) - remaining, 0)
    return (additional_children + 1) // 2


def parse_review_team(value: object) -> ReviewTeam:
    """Parse saved defaults without resolving, replacing or dropping preset IDs.

    Raises:
        ReviewRefused: For malformed saved choices, including run overrides.
    """
    try:
        return ReviewTeam.model_validate(value)
    except ValidationError as exc:
        raise ReviewRefused("invalid_review_team") from exc


@dataclass(frozen=True)
class ReviewBoundaryReceipt:
    """Body-free host identity; only a live issuer can establish validity."""

    boundary_id: str
    review_id: str
    revision: int
    slot_id: str
    slot_attempt_id: str
    stage: str
    predecessor_run_ids: tuple[str, ...]
    predecessor_settlement_ids: tuple[str, ...]
    grant_id: str
    source_epoch: int
    chain_id: str
    owner_id: str
    session_id: str
    conversation_id: str
    target_fingerprint: str
    team_fingerprint: str
    override_fingerprint: str
