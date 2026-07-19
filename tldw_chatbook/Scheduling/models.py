"""Pydantic domain models for the Scheduling module."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    validation_errors: list[str] | None = None
    warnings: list[str] | None = None
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
