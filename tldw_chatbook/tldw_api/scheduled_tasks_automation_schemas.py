"""Schemas for the server's Scheduled Tasks automation control plane.

Mirrors ``tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py``
(the ``/api/v1/scheduled-tasks`` surface) for the subset the client
consumes: listing/running/auditing server-owned definitions (ADR-077 phase
1), results review (spec §4.2), and previewing/creating/updating
definitions (spec §5.1). Enums are typed as ``str`` and nested policies as
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


class ScheduledTaskPreviewCreateRequest(BaseModel):
    """Request body for ``POST /api/v1/scheduled-tasks/previews``.

    Mirrors the server's ``ScheduledTaskPreviewCreateRequest`` (see
    ``Tests/Scheduling/fixtures/server_responses/automation_endpoints.md``).
    ``visibility_policy`` is nullable (not just default-``{}``) -- the local
    preview port's ``_normalize_visibility_policy`` (``automation_preview.py``)
    treats it as ``Any`` (string/dict/``None``), and the Task 1 fixture's
    request payloads send it as an explicit ``null``.
    """

    mode: str = "create"
    family: str
    definition_id: str | None = None
    definition_version: int | None = Field(default=None, ge=1)
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    input: dict[str, Any] = Field(default_factory=dict)
    schedule: dict[str, Any] = Field(default_factory=dict)
    visibility_policy: dict[str, Any] | None = None
    notification_policy: dict[str, Any] = Field(default_factory=dict)
    approval_policy: dict[str, Any] = Field(default_factory=dict)


class ScheduledTaskPreview(BaseModel):
    """One server preview row (``ScheduledTaskPreviewResponse``).

    Datetimes are ``str | None`` here, matching this file's
    ``ScheduledTaskResult`` style -- previews are consumed as opaque ISO
    strings, never date-arithmetic'd client-side.

    ``id``, ``owner_id``, ``payload_hash``, ``risk_class``, ``expires_at``,
    ``created_by``, ``created_at``, ``consumed_at`` and
    ``created_definition_id`` are only meaningful after a real server round
    trip -- they default to ``None`` so this model also validates the Task 1
    preview fixture (``automation_preview_response.json``), which -- being a
    pure local preview with no server round trip -- omits them (see
    ``automation_preview.py``'s module docstring).
    """

    id: str | None = None
    owner_id: str | None = None
    mode: str
    family: str
    definition_id: str | None = None
    definition_version: int | None = Field(default=None, ge=1)
    status: str
    payload_hash: str | None = None
    normalized_config: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    risk_class: str | None = None
    visibility_policy: dict[str, Any] = Field(default_factory=dict)
    schedule_preview: dict[str, Any] = Field(default_factory=dict)
    redaction_policy: dict[str, Any] = Field(default_factory=dict)
    expires_at: str | None = None
    created_by: str | None = None
    created_at: str | None = None
    consumed_at: str | None = None
    created_definition_id: str | None = None


class ScheduledTaskDefinitionCreateRequest(BaseModel):
    """Request body for ``POST /api/v1/scheduled-tasks/definitions``."""

    preview_id: str
    initial_lifecycle: str = "configured"


class ScheduledTaskDefinitionUpdateRequest(BaseModel):
    """Request body for ``PATCH /api/v1/scheduled-tasks/definitions/{id}``."""

    preview_id: str


class ScheduledTaskAutomationRunNowResponse(BaseModel):
    """Result of a manual run trigger (TASK-13022).

    The ``run_slot_utc`` reference lets the caller correlate the trigger
    with the eventual result notification and run row.
    """

    definition_id: str
    run_slot_utc: str
    job_id: int | str | None = None
    deduped: bool = False


class ScheduledTaskResult(BaseModel):
    """One server-side scheduled-task result row (spec §4.2 / server
    ``ScheduledTaskResultResponse``, ``/api/v1/scheduled-tasks/results``).

    Datetimes are ``str | None`` here (not ``datetime``) to match the
    reminder/notification schemas' style elsewhere in this package --
    results are consumed as opaque ISO strings, never date-arithmetic'd
    client-side.
    """

    id: str
    owner_id: str | None = None
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
    review_state: str = "unread"
    reviewed_at: str | None = None
    reviewed_by: str | None = None
    review_note: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ScheduledTaskResultList(BaseModel):
    """Paginated list of server-side scheduled-task results."""

    items: list[ScheduledTaskResult] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)
    has_more: bool = False
    next_offset: int | None = Field(default=None, ge=0)


class ScheduledTaskAuditEvent(BaseModel):
    """One durable audit event from a definition's execution trail.

    The server writes ``run_{status}`` events from the agent-task consumer
    (plus lifecycle/authoring events from the control plane); ``after``
    carries the run reference (``run_id``/``status``) for correlation with
    run rows and result notifications.
    """

    id: str
    definition_id: str
    event_type: str
    actor: str | None = None
    summary: str | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    created_at: datetime | None = None
    request_id: str | None = None
    idempotency_key: str | None = None


class ScheduledTaskAuditList(BaseModel):
    """Paginated audit trail for one automation definition."""

    items: list[ScheduledTaskAuditEvent] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1)
    offset: int = Field(default=0, ge=0)
    has_more: bool = False
    next_offset: int | None = Field(default=None, ge=0)
