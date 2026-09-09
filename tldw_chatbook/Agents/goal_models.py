"""Immutable private goal launch data. Display content never grants authority."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from tldw_chatbook.Agents.agent_models import RunOutcome, RunTerminationReason
from tldw_chatbook.Agents.automatic_work_budget import (
    AutomaticWorkLimits,
    AutomaticWorkSnapshot,
)
from tldw_chatbook.Skills_Interop.skill_script_runner import ScriptRunResult

Finite = Annotated[int, Field(ge=0, lt=2**63)]
Identity = Annotated[str, StringConstraints(min_length=1, max_length=256)]


class GoalModel(BaseModel):
    model_config = ConfigDict(
        strict=True, frozen=True, extra="forbid", arbitrary_types_allowed=True
    )


class GoalPolicy(GoalModel):
    iterations: Finite = 3
    model_calls: Finite = 32
    budget_tokens: Finite = 500_000
    output_tokens: Finite = 8192
    wall_seconds: Finite = 900
    iteration_model_turns: Finite = 8
    iteration_steps: Finite = 64
    iteration_wall_seconds: Finite = 240
    payload_bytes: Finite = 128 * 1024 * 1024

    @property
    def admission_enabled(self) -> bool:
        return all(value > 0 for value in self.model_dump().values())

    def chain_limits(self) -> AutomaticWorkLimits:
        """Reuse shared accounting with no goal-created children."""
        return AutomaticWorkLimits(
            generations=self.iterations,
            child_launches=0,
            model_calls=self.model_calls,
            budget_tokens=self.budget_tokens,
            output_tokens=self.output_tokens,
            wall_seconds=self.wall_seconds,
        )


class GoalProviderRef(GoalModel):
    provider: Identity
    model: Identity
    config_ref: Identity
    authority_ref: Identity
    endpoint_ref: Annotated[str, StringConstraints(min_length=1, max_length=2048)]


class GoalBindingRef(GoalModel):
    workspace_id: Identity
    binding_id: Identity
    locator: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    access: Literal["ro", "rw"]

    @field_validator("locator")
    @classmethod
    def absolute_locator(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or str(path) != value:
            raise ValueError("binding locator must be a canonical absolute path")
        return value


class GoalMCPBinding(GoalModel):
    """Non-secret immutable process and tool-definition references."""

    tool_id: Identity
    server_key: Identity
    tool_name: Identity
    profile_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    definition_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class GoalToolScope(GoalModel):
    catalog_tools: tuple[Identity, ...] = Field(default=(), max_length=128)
    runtime_tools: tuple[Identity, ...] = Field(default=(), max_length=128)
    mcp_bindings: tuple[GoalMCPBinding, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def scoped_bindings(self):
        if any(
            binding.tool_id not in self.catalog_tools for binding in self.mcp_bindings
        ):
            raise ValueError("MCP binding is outside catalog scope")
        if len({binding.tool_id for binding in self.mcp_bindings}) != len(
            self.mcp_bindings
        ):
            raise ValueError("duplicate MCP binding")
        return self


class VerificationSpec(GoalModel):
    id: Identity
    executor_tool_id: Identity
    verifier_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    verifier_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    skill_trust_ref: Identity | None = None
    arguments: tuple[Annotated[str, StringConstraints(max_length=4096)], ...] = Field(
        default=(), max_length=64
    )
    input_paths: tuple[
        Annotated[str, StringConstraints(min_length=1, max_length=4096)], ...
    ] = Field(max_length=128, min_length=1)
    expected_exit_code: Annotated[int, Field(ge=0, le=255)] = 0
    require_complete_output: bool = True

    @property
    def invocation_key(self) -> tuple:
        """Canonical executor identity plus the complete launch-bound invocation."""
        executor = self.executor_tool_id
        if executor == "runtime:run_skill_script":
            executor = "run_skill_script"
        return (
            executor,
            self.verifier_path,
            self.verifier_sha256,
            self.arguments,
            self.skill_trust_ref,
        )

    @field_validator("verifier_path")
    @classmethod
    def absolute_verifier(cls, value: str) -> str:
        return GoalBindingRef.absolute_locator(value)

    @field_validator("input_paths")
    @classmethod
    def relative_inputs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            PurePosixPath(p).is_absolute()
            or ".." in PurePosixPath(p).parts
            or "\x00" in p
            for p in values
        ):
            raise ValueError(
                "verification inputs must stay within the selected binding"
            )
        return values


class GoalRequest(GoalModel):
    objective: str
    criteria: str
    provider: GoalProviderRef
    binding: GoalBindingRef
    source_bindings: tuple[GoalBindingRef, ...] = Field(default=(), max_length=32)
    tool_scope: GoalToolScope
    verifiers: tuple[VerificationSpec, ...] = Field(default=(), max_length=32)
    policy: GoalPolicy = Field(default_factory=GoalPolicy)
    human_review_required: bool = True

    @field_validator("objective", "criteria")
    @classmethod
    def bounded_text(cls, value: str) -> str:
        if not value.strip() or len(value.encode("utf-8")) > 8192:
            raise ValueError("objective and criteria require 1..8192 UTF-8 bytes")
        return value

    @model_validator(mode="after")
    def check_verifiers(self) -> GoalRequest:
        if not self.human_review_required and not self.verifiers:
            raise ValueError("objective-only completion requires a verifier")
        tools = self.tool_scope.catalog_tools + self.tool_scope.runtime_tools
        roots = [
            PurePosixPath(b.locator)
            for b in (self.binding, *self.source_bindings)
            if b.access == "rw"
        ]
        for verifier in self.verifiers:
            if verifier.executor_tool_id not in tools:
                raise ValueError("verifier executor is outside goal tool scope")
            if any(
                PurePosixPath(verifier.verifier_path).is_relative_to(root)
                for root in roots
            ):
                raise ValueError("trusted verifier must be outside editable bindings")
        if len({v.id for v in self.verifiers}) != len(self.verifiers):
            raise ValueError("duplicate verifier identity")
        if len(self.canonical_json().encode("utf-8")) > 128 * 1024:
            raise ValueError("launch request exceeds 128 KiB")
        return self

    def validate_verifier_invocations(self) -> None:
        """Reject ambiguous execution without making historical JSON unreadable."""
        keys = [verifier.invocation_key for verifier in self.verifiers]
        if len(set(keys)) != len(keys):
            raise ValueError("ambiguous_verifier_invocation")

    def select_script_verifier(
        self,
        path: str,
        sha256: str | None,
        arguments: tuple[str, ...],
        trust_ref: str,
        *,
        verifier_id: str | None = None,
    ) -> VerificationSpec | None:
        """Resolve one exact script invocation; IDs cannot disambiguate duplicates."""
        key = ("run_skill_script", path, sha256, arguments, trust_ref)
        matches = [v for v in self.verifiers if v.invocation_key == key]
        if len(matches) > 1:
            raise ValueError("ambiguous_verifier_invocation")
        if not matches or (verifier_id is not None and matches[0].id != verifier_id):
            return None
        return matches[0]

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


EvidenceIdentity = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,128}$")]


class IterationReport(GoalModel):
    summary: str = ""
    learnings: tuple[str, ...] = Field(default=(), max_length=8)
    next_action: str = ""
    candidate_draft: str = ""
    evidence_ids: tuple[EvidenceIdentity, ...] = Field(default=(), max_length=32)
    completion_recommended: bool = False

    @model_validator(mode="after")
    def bounded_payload(self):
        if (
            len(self.candidate_draft.encode("utf-8")) > 32768
            or len(self.model_dump_json().encode("utf-8")) > 65536
        ):
            raise ValueError("goal report exceeds private payload limit")
        return self


# One persisted/public shape. Legacy field names are intentionally rejected.
GoalReport = IterationReport


class GoalCriterion(GoalModel):
    id: Identity
    verifier_id: str | None = None
    human: bool = False


class GoalCheck(GoalModel):
    criterion_id: str
    satisfied: bool = False
    evidence_id: str | None = None
    reason: str = "unavailable"


class GoalDecision(GoalModel):
    action: Literal[
        "continue", "pause", "awaiting_result_review", "recovery_required", "completed"
    ]
    reason: Identity
    checks: tuple[GoalCheck, ...] = Field(default=(), max_length=33)
    evidence_errors: tuple[EvidenceIdentity, ...] = Field(default=(), max_length=32)
    no_progress_count: int = 0
    failed_count: int = 0
    source_digests: tuple[str, ...] = Field(default=(), max_length=1)
    checkpoint_id: str | None = None
    artifact_digest: str | None = None
    draft_digest: str = ""


class GoalCheckpoint(GoalModel):
    id: str = ""
    artifact_digest: str = ""
    ordinal: int
    report: IterationReport
    decision: GoalDecision
    report_error: str | None = None


class GoalEvidence(GoalModel):
    """Private runtime observation. Never parsed from a model report."""

    id: EvidenceIdentity
    goal_id: str
    attempt_id: str
    run_id: str
    verifier_id: str
    source_digest: str
    checked_manifest: str | None
    verifier_sha256: str
    passed: bool
    fresh: bool
    reason: str
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class GoalScriptInvocation:
    id: str
    goal_id: str
    attempt_id: str
    ordinal: int
    run_id: str
    verifier_path: str
    verifier_sha256: str
    skill_name: str
    skill_trust_ref: str
    arguments: tuple[str, ...]
    before_manifest: str | None = None
    verifier_id: str | None = None


@dataclass(frozen=True)
class GoalScriptEvidence:
    invocation: GoalScriptInvocation
    result: ScriptRunResult
    after_manifest: str | None = None


@dataclass(frozen=True)
class GoalToolObservation:
    id: str
    goal_id: str
    attempt_id: str
    run_id: str
    tool: str
    arguments_digest: str
    content: str
    complete: bool


@dataclass(frozen=True)
class GoalIterationResult:
    goal_id: str
    ordinal: int
    attempt_id: str | None
    native_run_id: str | None
    outcome: RunOutcome | None
    termination_reason: RunTerminationReason
    tool_records: tuple[GoalScriptEvidence, ...] = ()
    reason_code: str | None = None
    observations: tuple[GoalToolObservation, ...] = ()

    def __post_init__(self):
        if isinstance(self.termination_reason, RunTerminationReason):
            return
        code = self.termination_reason
        try:
            reason = RunTerminationReason(code)
        except ValueError:
            reason = RunTerminationReason.PREFLIGHT_REFUSED
        object.__setattr__(self, "termination_reason", reason)
        object.__setattr__(self, "reason_code", code)


class GoalProvisioning(GoalModel):
    goal_id: Identity
    launch_id: Identity
    payload_hash: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    conversation_id: Identity
    workspace_id: Identity


class GoalSnapshot(GoalModel):
    id: Identity
    launch_id: Identity
    payload_hash: str
    conversation_id: Identity
    chain_id: Identity
    revision: int
    status: Literal[
        "starting",
        "ready",
        "paused",
        "awaiting_result_review",
        "recovery_required",
        "completed",
        "removed",
    ]
    iteration_count: int
    pause_reason: str | None
    request: GoalRequest | None
    checkpoints: tuple[GoalCheckpoint, ...] = ()
    reports: tuple[GoalReport, ...] = ()
    accounting: AutomaticWorkSnapshot

    @property
    def policy(self) -> GoalPolicy:
        return self.request.policy

    @property
    def provisioning(self) -> GoalProvisioning:
        return GoalProvisioning(
            goal_id=self.id,
            launch_id=self.launch_id,
            payload_hash=self.payload_hash,
            conversation_id=self.conversation_id,
            workspace_id=self.request.binding.workspace_id,
        )
