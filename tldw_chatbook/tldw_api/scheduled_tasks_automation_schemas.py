"""Schemas for the server's Scheduled Tasks automation control plane.

Mirrors ``tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py``
(the ``/api/v1/scheduled-tasks/definitions`` surface) for the subset the
client consumes in ADR-077 phase 1: listing server-owned definitions and
triggering a manual run. Enums are typed as ``str`` and nested policies as
``dict[str, Any]`` deliberately -- the server owns those vocabularies and
the client must not break when a new lifecycle/health value ships.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ScheduledTaskAutomationDefinition(BaseModel):
    """One server-side automation definition row."""

    id: str
    owner_id: str | None = None
    version: int = Field(default=1, ge=1)
    family: str
    name: str
    description: str | None = None
    lifecycle: str = "configured"
    health: str = "ready"
    disabled_lock_kind: str | None = None
    disabled_reason: str | None = None
    schedule: dict[str, Any] = Field(default_factory=dict)
    input: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    visibility_policy: dict[str, Any] = Field(default_factory=dict)
    notification_policy: dict[str, Any] = Field(default_factory=dict)
    approval_policy: dict[str, Any] = Field(default_factory=dict)
    preview_id: str | None = None
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    archived_at: datetime | None = None
    resolution_state: str = "open"


class ScheduledTaskAutomationDefinitionList(BaseModel):
    """Paginated list of server-side automation definitions."""

    items: list[ScheduledTaskAutomationDefinition] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)
    has_more: bool = False
    next_offset: int | None = Field(default=None, ge=0)


class ScheduledTaskAutomationRunNowResponse(BaseModel):
    """Result of a manual run trigger (TASK-13022).

    The ``run_slot_utc`` reference lets the caller correlate the trigger
    with the eventual result notification and run row.
    """

    definition_id: str
    run_slot_utc: str
    job_id: int | str | None = None
    deduped: bool = False
