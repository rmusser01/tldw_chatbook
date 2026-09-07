"""Pydantic domain models for the Scheduling module."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskStatus(str, Enum):
    """Unified task status for reminders, watchlist jobs, and automations."""

    WAITING = "waiting"
    RUNNING = "running"
    PAUSED = "paused"
    NEEDS_ATTENTION = "needs_attention"
    BLOCKED = "blocked"
    DISABLED = "disabled"
    ARCHIVED = "archived"
    COMPLETED = "completed"
    FOUND_RESULTS = "found_results"
    MISSED = "missed"
    TIMED_OUT = "timed_out"
    CONFLICT = "conflict"


class ScheduleKind(str, Enum):
    """Reminder schedule kind."""

    ONE_TIME = "one_time"
    RECURRING = "recurring"


class Lifecycle(str, Enum):
    """Automation definition lifecycle state."""

    CONFIGURED = "configured"
    PAUSED = "paused"
    ARCHIVED = "archived"
    DISABLED = "disabled"


class Health(str, Enum):
    """Automation definition health state."""

    READY = "ready"
    EXECUTION_UNAVAILABLE = "execution_unavailable"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    NEEDS_ATTENTION = "needs_attention"
    PERMISSION_REQUIRED = "permission_required"


class AutomationFamily(str, Enum):
    """Automation definition family."""

    RECURRING_QUESTION = "recurring_question"
    AGENT_TASK = "agent_task"


class RunStatus(str, Enum):
    """Automation run status — server literals plus local ``timed_out``.

    The server's normalized API folds timeouts into ``failed`` and keeps
    the truth in ``run_summary.legacy_status``; locally ``timed_out`` is
    first-class (TASK-18939 vocabulary, spec §4.1).
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class RunOutcome(str, Enum):
    """Automation run outcome, orthogonal to status."""

    FINDING = "finding"
    NO_MATCH = "no_match"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    NONE = "none"


class ReviewState(str, Enum):
    """Result review state."""

    UNREAD = "unread"
    READ = "read"
    DISMISSED = "dismissed"


class AutomationRun(BaseModel):
    """One local automation execution (server ``RunRow`` shape, spec §4.1)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    server_id: str | None = None
    owner_id: str = "local"
    definition_id: str
    definition_version: int = 1
    trigger_reason: str
    status: RunStatus = RunStatus.QUEUED
    outcome: RunOutcome = RunOutcome.NONE
    schedule_slot: str | None = None
    scope_snapshot: dict[str, Any] = Field(default_factory=dict)
    finding_policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    rag_request_snapshot: dict[str, Any] = Field(default_factory=dict)
    run_summary: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    failure_reason: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @field_validator(
        "scope_snapshot",
        "finding_policy_snapshot",
        "rag_request_snapshot",
        "run_summary",
        "evidence_summary",
        mode="before",
    )
    @classmethod
    def _none_dict_to_default(cls, value: Any) -> Any:
        """A DB row created without these kwargs stores NULL; coerce it."""
        return {} if value is None else value


class AutomationResult(BaseModel):
    """One automation result (server ``ResultRow`` shape, spec §4.2)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    server_id: str | None = None
    owner_id: str = "local"
    definition_id: str
    run_id: str
    kind: str
    title: str
    summary: str
    answer: Any | None = None
    answer_mode: str = "none"
    confidence: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    dedupe_key: str
    visibility_destination: dict[str, Any] = Field(default_factory=dict)
    review_state: ReviewState = ReviewState.UNREAD
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    review_note: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    @field_validator("confidence", "visibility_destination", mode="before")
    @classmethod
    def _none_dict_to_default(cls, value: Any) -> Any:
        """A DB row created without these kwargs stores NULL; coerce it."""
        return {} if value is None else value

    @field_validator("source_refs", mode="before")
    @classmethod
    def _none_list_to_default(cls, value: Any) -> Any:
        """A DB row created without ``source_refs`` stores NULL; coerce it."""
        return [] if value is None else value


class ScheduledTask(BaseModel):
    """Lightweight read-only task projection used for lists and watchlist jobs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    type: str
    status: TaskStatus
    schedule_summary: str | None = None
    next_run_at: datetime | None = None
    owner_id: str = "local"
    source: str | None = None


class ReminderTask(BaseModel):
    """Local or synced reminder task."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    server_id: str | None = None
    owner_id: str = "local"
    title: str
    body: str | None = None
    schedule_kind: ScheduleKind
    run_at: datetime | None = None
    cron: str | None = None
    timezone: str | None = None
    enabled: bool = True
    last_status: TaskStatus = TaskStatus.WAITING
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    missed_at: datetime | None = None
    #: Occurrences that elapsed undispatched before the last (late) dispatch.
    #: Client-local accounting only (task-18937): never pushed to the server
    #: and not expected in server responses.
    missed_count: int = 0
    #: Per-task handler execution timeout in seconds (task-18939). None
    #: means use the global ``[scheduling] handler_timeout_seconds``.
    timeout_seconds: float | None = None
    #: Transfer state machine marker (schedules-handoff spec §6). NULL =
    #: not in transfer. Values arrive in a later PR; the column lands with
    #: schema v4 so readers must tolerate it from day one.
    transfer_state: str | None = None
    link_type: str | None = None
    link_id: str | None = None
    link_url: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    sync_version: int = 0

    @model_validator(mode="after")
    def _validate_schedule_fields(self) -> "ReminderTask":
        if self.schedule_kind == ScheduleKind.ONE_TIME:
            if self.run_at is None:
                raise ValueError("run_at is required for one_time schedules")
        elif self.schedule_kind == ScheduleKind.RECURRING:
            if self.cron is None:
                raise ValueError("cron is required for recurring schedules")
            if self.timezone is None:
                raise ValueError("timezone is required for recurring schedules")
        return self


class AutomationDefinition(BaseModel):
    """Automation definition with lifecycle, health, and policy fields."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    server_id: str | None = None
    owner_id: str = "local"
    family: AutomationFamily
    name: str
    description: str | None = None
    lifecycle: Lifecycle = Lifecycle.CONFIGURED
    health: Health = Health.EXECUTION_UNAVAILABLE
    schedule: dict[str, Any] | None = None
    input: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    visibility_policy: dict[str, Any] | None = None
    notification_policy: dict[str, Any] | None = None
    approval_policy: dict[str, Any] | None = None
    version: int = 1
    preview_id: str | None = None
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    archived_at: datetime | None = None
    disabled_lock_kind: str | None = None
    disabled_reason: str | None = None
    resolution_state: str = "open"
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    resolved_result_id: str | None = None
    finding_policy: dict[str, Any] = Field(
        default_factory=lambda: {"preset": "balanced_findings"}
    )
    retention_policy: dict[str, Any] = Field(
        default_factory=lambda: {"mode": "default"}
    )
    next_run_at: datetime | None = None
    transfer_state: str | None = None


class PreviewStatus(str, Enum):
    """Automation preview validation state."""

    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    CONSUMED = "consumed"


class AutomationPreview(BaseModel):
    """Automation preview before committing to a definition."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    owner_id: str = "local"
    mode: str | None = None
    family: AutomationFamily
    definition_id: str | None = None
    definition_version: int | None = None
    status: PreviewStatus = PreviewStatus.VALID
    payload_hash: str | None = None
    normalized_config: dict[str, Any] | None = None
    risk_class: str | None = None
    validation_errors: list[dict[str, Any]] | None = None
    warnings: list[dict[str, Any]] | None = None
    visibility_policy: dict[str, Any] | None = None
    schedule_preview: dict[str, Any] | None = None
    redaction_policy: dict[str, Any] | None = None
    expires_at: datetime | None = None
    created_by: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    consumed_at: datetime | None = None
    created_definition_id: str | None = None


class AutomationAuditEvent(BaseModel):
    """Audit event recording changes to an automation definition."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    definition_id: str
    owner_id: str = "local"
    event_type: str
    actor: str
    summary: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    request_id: str | None = None
    idempotency_key: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
