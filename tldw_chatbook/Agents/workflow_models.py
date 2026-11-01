"""Bounded structural workflow values. None are execution capabilities (ADR-285)."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .automatic_work_budget import AutomaticWorkLimits, AutomaticWorkSnapshot
from .goal_models import GoalBindingRef, GoalProviderRef

RECORD_BYTES = 128 * 1024
WORKFLOW_BYTES = 4 * 1024 * 1024
TOTAL_BYTES = 128 * 1024 * 1024
Identity = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
Finite = Annotated[int, Field(ge=0, lt=2**63)]
Positive = Annotated[int, Field(gt=0, lt=2**63)]
WorkflowStatus = Literal[
    "awaiting_plan",
    "awaiting_approval",
    "running",
    "paused",
    "stopped",
    "completed",
    "closed",
    "recovery_required",
    "removed",
    "awaiting_resume",
]


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or "\\" in value
        or "\x00" in value
        or len(value.encode()) > 4096
    ):
        raise ValueError("path must be canonical and binding-relative")
    return value


class WorkflowModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs) -> Self:
        if (
            len(json_data.encode("utf-8") if isinstance(json_data, str) else json_data)
            > RECORD_BYTES
        ):
            raise ValueError("workflow record exceeds 128 KiB")
        json.loads(
            json_data,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("nonfinite JSON")
            ),
        )
        return super().model_validate_json(json_data, **kwargs)

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @model_validator(mode="after")
    def bounded_record(self) -> Self:
        if len(self.canonical_json().encode()) > RECORD_BYTES:
            raise ValueError("workflow record exceeds 128 KiB")
        return self


class WorkflowRequest(WorkflowModel):
    conversation_id: Identity
    objective: Annotated[str, StringConstraints(min_length=1)]
    binding: GoalBindingRef
    targets: tuple[GoalProviderRef, ...] = Field(min_length=1, max_length=16)
    policy: AutomaticWorkLimits = Field(
        default_factory=lambda: AutomaticWorkLimits(generations=8, child_launches=12)
    )

    @model_validator(mode="after")
    def request_contract(self) -> Self:
        if not self.objective.strip() or len(self.objective.encode()) > 8192:
            raise ValueError("objective requires 1..8192 UTF-8 bytes")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate target")
        return self


class WorkflowFootprint(WorkflowModel):
    path: str
    access: Literal["read", "write"]
    subtree: bool = False

    _path = field_validator("path")(_relative_path)


class WorkflowCheck(WorkflowModel):
    id: Identity
    status: Literal["passed", "failed", "skipped", "stale", "missing"]
    verifier: Identity
    arguments: tuple[str, ...] = Field(default=(), max_length=64)
    input_manifest: str | None = None
    output_manifest: str | None = None
    exit_code: Annotated[int, Field(ge=0, le=255)] | None = None
    reason: Identity

    @model_validator(mode="after")
    def explicit_evidence(self) -> Self:
        if self.status == "passed" and (
            self.exit_code != 0 or not self.input_manifest or not self.output_manifest
        ):
            raise ValueError("passed check requires process and manifest evidence")
        return self


class WorkflowTaskSpec(WorkflowModel):
    id: Identity
    role: Identity
    instructions: Annotated[str, StringConstraints(min_length=1)]
    dependencies: tuple[Identity, ...] = Field(default=(), max_length=12)
    inputs: tuple[Identity, ...] = Field(default=(), max_length=64)
    outputs: tuple[Identity, ...] = Field(default=(), max_length=64)
    footprint: tuple[WorkflowFootprint, ...] = Field(max_length=64)
    allowed_tools: tuple[Identity, ...] = Field(default=(), max_length=128)
    execution_mode: Literal["scoped", "exclusive"] = "scoped"
    target: GoalProviderRef
    expected_checks: tuple[Identity, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def unique_members(self) -> Self:
        for values in (
            self.dependencies,
            self.inputs,
            self.outputs,
            self.footprint,
            self.allowed_tools,
            self.expected_checks,
        ):
            if len(set(values)) != len(values):
                raise ValueError("duplicate task member")
        return self


class WorkflowPhaseSpec(WorkflowModel):
    id: Identity
    tasks: tuple[WorkflowTaskSpec, ...] = Field(min_length=1, max_length=12)


class WorkflowPlan(WorkflowModel):
    request: WorkflowRequest
    root_identity: Identity
    roles: tuple[Identity, ...] = Field(min_length=1, max_length=32)
    tools: tuple[Identity, ...] = Field(default=(), max_length=128)
    deliverables: tuple[Identity, ...] = Field(min_length=1, max_length=64)
    phases: tuple[WorkflowPhaseSpec, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def ordered_contract(self) -> Self:
        phases, seen = set(), {}
        if sum(len(p.tasks) for p in self.phases) > 12:
            raise ValueError("plan exceeds twelve task definitions")
        for values in (self.roles, self.tools, self.deliverables):
            if len(set(values)) != len(values):
                raise ValueError("duplicate plan member")
        for phase in self.phases:
            if phase.id in phases:
                raise ValueError("duplicate phase identity")
            phases.add(phase.id)
            for task in phase.tasks:
                if task.id in seen:
                    raise ValueError("duplicate task identity")
                if any(dep not in seen for dep in task.dependencies):
                    raise ValueError("dependencies must reference earlier tasks")
                if (
                    task.role not in self.roles
                    or task.target not in self.request.targets
                    or not set(task.allowed_tools) <= set(self.tools)
                ):
                    raise ValueError("unavailable role, target or tool")
                if self.request.binding.access == "ro" and any(
                    s.access == "write" for s in task.footprint
                ):
                    raise ValueError("write outside captured binding authority")
                for ref in task.inputs:
                    if ":" in ref:
                        source, output = ref.split(":", 1)
                        if (
                            source not in task.dependencies
                            or output not in seen[source].outputs
                        ):
                            raise ValueError("invalid input reference")
                    else:
                        _relative_path(ref)
                        if not any(
                            ref == scope.path
                            or scope.subtree
                            and PurePosixPath(ref).is_relative_to(scope.path)
                            for scope in task.footprint
                        ):
                            raise ValueError("input outside declared footprint")
                for path in task.outputs:
                    _relative_path(path)
                    if not any(
                        scope.access == "write"
                        and (
                            path == scope.path
                            or scope.subtree
                            and PurePosixPath(path).is_relative_to(scope.path)
                        )
                        for scope in task.footprint
                    ):
                        raise ValueError("output outside declared write footprint")
                seen[task.id] = task
        return self


class WorkflowDecision(WorkflowModel):
    id: Identity
    workflow_id: Identity
    plan_revision: Positive
    control_revision: Positive
    phase_id: Identity
    action: Literal["approve", "revise", "pause", "stop", "complete", "resume"]
    reason: Identity
    result_refs: tuple[Identity, ...] = Field(default=(), max_length=12)


class WorkflowTaskAttempt(WorkflowModel):
    id: Identity
    workflow_id: Identity
    plan_revision: Positive
    phase_id: Identity
    task_id: Identity
    ordinal: Positive
    run_id: Identity | None = None
    native_attempt_id: Identity | None = None
    status: Literal[
        "prepared", "accepted", "completed", "failed", "uncertain", "closed"
    ] = "prepared"
    retirement: Literal["pending", "drained", "unknown"] = "pending"


class WorkflowTaskResult(WorkflowModel):
    workflow_id: Identity
    attempt_id: Identity
    run_id: Identity
    status: Literal["completed", "failed", "uncertain"]
    summary: str
    handoff: str
    checks: tuple[WorkflowCheck, ...] = Field(default=(), max_length=32)


class WorkflowPayloadReservation(WorkflowModel):
    attempt_id: Identity
    workflow_id: Identity
    bytes: Annotated[int, Field(gt=0, le=RECORD_BYTES)]
    state: Literal["reserved", "published", "removed"]


class WorkflowSnapshot(BaseModel):
    """Private history aggregate; individual records, not the aggregate, have 128 KiB bounds."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    id: Identity
    launch_id: Identity
    chain_id: Identity
    request: WorkflowRequest
    control_revision: Positive
    progress_cursor: Finite
    status: WorkflowStatus
    plans: tuple[WorkflowPlan, ...]
    results: tuple[WorkflowTaskResult, ...]
    accounting: AutomaticWorkSnapshot


class WorkflowPhaseAttempt(WorkflowModel):
    """Durable reference only; accepted state is never a live capability."""

    id: Identity
    workflow_id: Identity
    revision_id: Positive | None
    phase_id: Identity | None
    purpose: Literal["planning", "execution"]
    control_revision: Positive
    command_id: Identity
    chain_id: Identity
    conversation_id: Identity
    session_id: Identity
    owner_id: Identity
    state: Literal["prepared", "accepted", "completed", "aborted", "review_required"]


class WorkflowTaskClaim(WorkflowTaskAttempt):
    """Exact native claim projection; duplicate reads do not authorize a launch."""

    reservation_id: Identity
    launch_run_id: Identity
    phase_attempt_id: Identity
    chain_id: Identity
    conversation_id: Identity
    session_id: Identity
    owner_id: Identity
