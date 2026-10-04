"""Closed helper definitions and immutable host-owned response snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

MAX_DEFINITION_BYTES = 8192
MAX_TITLE_CHARACTERS = 120
MAX_APPLICABILITY_BYTES = 1024
MAX_CRITERIA_BYTES = 2048
MAX_FEEDBACK_BYTES = 2048
MAX_FIXTURE_BYTES = 8192
MAX_LITERALS = 16
MAX_HEADINGS = 8
MAX_LITERAL_BYTES = 256
MAX_HELPER_OUTPUT_TOKENS = 4096
MAX_EFFECTIVE_RULES = 16
MAX_HELPER_SECONDS = 30
MAX_ASSESSMENT_SECONDS = 30
MAX_LEARNING_SECONDS = 120
MAX_LEARNING_ATTEMPTS = 3
MAX_APP_HELPERS = 4
MAX_CHAT_HELPERS = 1
MAX_NATIVE_FEEDBACK_BYTES = 4096
MAX_COMBINED_FEEDBACK_BYTES = 8192
MAX_NATIVE_CORRECTIONS = 2
MAX_SHARED_CONTINUATIONS = 3
MAX_CHAIN_SECONDS = 120
RULE_SCHEMA_VERSION = 1
EVALUATOR_PROTOCOL_VERSION = 1


def canonical_json(value: object) -> str:
    """Encode exact private content without writing it to diagnostics."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest_payload(value: object) -> str:
    """Return an identity for exact-content and stale-source checks."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _bounded_text(value: str | None, cap: int, name: str) -> None:
    if value is None or not value.strip():
        raise ValueError(f"{name}_empty")
    if len(value.encode("utf-8")) > cap:
        raise ValueError(f"{name}_too_large")


class _ClosedDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class RuleApplicability(_ClosedDefinition):
    """Exactly one applicability condition."""

    kind: Literal["always", "semantic"]
    criteria: str | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        if self.kind == "semantic":
            _bounded_text(self.criteria, MAX_APPLICABILITY_BYTES, "applicability")
        elif self.criteria is not None:
            raise ValueError("irrelevant_applicability_criteria")
        return self


class RuleDetector(_ClosedDefinition):
    """One native text predicate, with no executable alternatives."""

    kind: Literal["include", "exclude", "headings", "semantic"]
    criteria: str | None = Field(default=None, repr=False)
    literals: tuple[str, ...] = Field(default=(), repr=False)
    case_sensitive: StrictBool

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        if self.kind == "semantic":
            if self.literals:
                raise ValueError("irrelevant_semantic_literals")
            _bounded_text(self.criteria, MAX_CRITERIA_BYTES, "criteria")
            return self
        if self.criteria is not None:
            raise ValueError("irrelevant_deterministic_criteria")
        cap = MAX_HEADINGS if self.kind == "headings" else MAX_LITERALS
        if not 1 <= len(self.literals) <= cap:
            raise ValueError("invalid_literal_count")
        for literal in self.literals:
            _bounded_text(literal, MAX_LITERAL_BYTES, "literal")
        return self


class RuleCandidate(_ClosedDefinition):
    """The only definition accepted from learning helpers."""

    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARACTERS, repr=False)
    applicability: RuleApplicability = Field(repr=False)
    detector: RuleDetector = Field(repr=False)
    feedback: str = Field(repr=False)

    @model_validator(mode="after")
    def validate_caps(self) -> Self:
        if not self.title.strip():
            raise ValueError("title_empty")
        _bounded_text(self.feedback, MAX_FEEDBACK_BYTES, "feedback")
        if (
            len(canonical_json(self.model_dump(mode="json")).encode("utf-8"))
            > MAX_DEFINITION_BYTES
        ):
            raise ValueError("definition_too_large")
        return self

    def candidate_digest(self) -> str:
        """Identity of the complete reviewed definition."""
        return digest_payload(self.model_dump(mode="json"))

    def detector_digest(self) -> str:
        """Predicate identity independent of feedback and display title."""
        return digest_payload(
            {
                "applicability": self.applicability.model_dump(mode="json"),
                "detector": self.detector.model_dump(mode="json"),
            }
        )


@dataclass(frozen=True, slots=True)
class RuleScope:
    kind: Literal["chat", "workspace", "global"]
    owner_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"chat", "workspace", "global"} or not self.owner_id:
            raise ValueError("invalid_rule_scope")


@dataclass(frozen=True, slots=True)
class RuleSource:
    profile_id: str
    session_id: str
    conversation_id: str | None
    branch_id: str
    message_id: str
    message_version: int
    parent_turn_id: str
    operation_id: str
    settlement_id: str
    effective_digest: str
    authority_epoch: int


@dataclass(frozen=True, slots=True)
class RuleRevision:
    rule_id: str
    revision: int
    candidate: RuleCandidate = field(repr=False)
    schema_version: int
    origin: RuleSource
    created_at: str


@dataclass(frozen=True, slots=True)
class RuleBinding:
    scope: RuleScope
    rule_id: str
    revision: int | None
    state: Literal["enabled", "disabled", "excluded"]
    binding_revision: int

    def __post_init__(self) -> None:
        if self.state not in {"enabled", "disabled", "excluded"}:
            raise ValueError("invalid_binding_state")
        if self.state != "excluded" and (self.revision is None or self.revision < 1):
            raise ValueError("missing_binding_revision")
        if self.binding_revision < 1 or not self.rule_id:
            raise ValueError("invalid_binding_identity")


@dataclass(frozen=True, slots=True)
class RuleEvidence:
    ref: str
    state: Literal["settled", "not_started", "uncertain"]
    outcome: str = field(repr=False)
    work_revision: str | None
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class RuleInput:
    request_text: str = field(repr=False)
    response_text: str = field(repr=False)
    evidence: tuple[RuleEvidence, ...] = field(repr=False)
    evidence_complete: bool
    work_revision: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def evidence_digest(self) -> str:
        """Bind assessments to the frozen input, including incompleteness."""
        return digest_payload(asdict(self))


@dataclass(frozen=True, slots=True)
class RuleCheck:
    rule_id: str
    revision: int
    applicability: Literal["applicable", "inapplicable", "unknown"]
    verdict: Literal["pass", "violation", "couldnt_verify"] | None
    reason: str = field(repr=False)
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))


@dataclass(frozen=True, slots=True)
class RuleAssessment:
    assessment_id: str
    source: RuleSource
    checks: tuple[RuleCheck, ...]
    evidence_complete: bool
    evidence_digest: str
    outcome: Literal["pass", "violation", "couldnt_verify", "no_applicable_rules"]
    state: Literal["pending", "completed", "cancelled", "stale"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))


@dataclass(frozen=True, slots=True)
class RuleCaseResult:
    case_id: str
    case_type: Literal[
        "recorded_violation",
        "synthetic_violation",
        "synthetic_correction",
        "synthetic_acceptable",
    ]
    check: RuleCheck


@dataclass(frozen=True, slots=True)
class RuleValidation:
    candidate_digest: str
    detector_digest: str
    protocol_version: int
    provider: str
    model: str
    case_results: tuple[RuleCaseResult, ...]
    source: RuleSource
    fixture_ids: tuple[str, ...]
    tested_at: str
    original_input_digest: str | None = None
    fixture_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_results", tuple(self.case_results))
        object.__setattr__(self, "fixture_ids", tuple(self.fixture_ids))


@dataclass(frozen=True, slots=True)
class RuleLearningResult:
    state: Literal["active", "inactive", "cancelled", "stale"]
    rule: RuleRevision | None = field(repr=False)
    validation: RuleValidation | None
    fixtures: Mapping[str, RuleInput] = field(repr=False)
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixtures", MappingProxyType(dict(self.fixtures)))
