"""Immutable private goal launch data. Display content never grants authority."""

from __future__ import annotations

import json
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

from tldw_chatbook.Agents.automatic_work_budget import (
    AutomaticWorkLimits,
    AutomaticWorkSnapshot,
)

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

    @field_validator("objective", "criteria")
    @classmethod
    def bounded_text(cls, value: str) -> str:
        if not value.strip() or len(value.encode("utf-8")) > 8192:
            raise ValueError("objective and criteria require 1..8192 UTF-8 bytes")
        return value

    @model_validator(mode="after")
    def check_verifiers(self) -> GoalRequest:
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

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


class GoalReport(GoalModel):
    summary: str = ""
    learnings: tuple[str, ...] = Field(default=(), max_length=8)
    next_action: str = ""
    draft: str = ""
    evidence_refs: tuple[Identity, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def bounded_payload(self) -> GoalReport:
        if (
            len(self.draft.encode("utf-8")) > 32768
            or len(self.model_dump_json().encode("utf-8")) > 65536
        ):
            raise ValueError("goal report exceeds private payload limit")
        return self


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
    status: Literal["starting", "ready", "paused"]
    iteration_count: int
    pause_reason: str | None
    request: GoalRequest
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
