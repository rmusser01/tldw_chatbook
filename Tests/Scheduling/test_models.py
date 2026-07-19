"""Tests for Scheduling Pydantic models."""

import pytest


def test_task_status_values():
    """TaskStatus enum values must match the spec."""
    from tldw_chatbook.Scheduling.models import TaskStatus

    assert TaskStatus.WAITING.value == "waiting"


def test_reminder_task_defaults():
    """ReminderTask default values match the spec."""
    from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind, TaskStatus

    task = ReminderTask(title="Test reminder", schedule_kind=ScheduleKind.ONE_TIME)

    assert task.title == "Test reminder"
    assert task.schedule_kind is ScheduleKind.ONE_TIME
    assert task.owner_id == "local"
    assert task.enabled is True
    assert task.last_status is TaskStatus.WAITING
    assert task.sync_version == 0
    assert task.server_id is None
    assert task.body is None
    assert task.run_at is None
    assert task.cron is None
    assert task.timezone is None
    assert task.next_run_at is None
    assert task.last_run_at is None
    assert task.missed_at is None
    assert task.link_type is None
    assert task.link_id is None
    assert task.link_url is None
    assert task.updated_at is None
    assert task.id
    assert task.created_at


def test_automation_definition_defaults():
    """AutomationDefinition default values match the spec."""
    from tldw_chatbook.Scheduling.models import (
        AutomationDefinition,
        AutomationFamily,
        Health,
        Lifecycle,
    )

    definition = AutomationDefinition(
        name="Daily digest",
        family=AutomationFamily.RECURRING_QUESTION,
    )

    assert definition.name == "Daily digest"
    assert definition.family is AutomationFamily.RECURRING_QUESTION
    assert definition.owner_id == "local"
    assert definition.lifecycle is Lifecycle.CONFIGURED
    assert definition.health is Health.EXECUTION_UNAVAILABLE
    assert definition.version == 1
    assert definition.schedule is None
    assert definition.input is None
    assert definition.config is None
    assert definition.visibility_policy is None
    assert definition.notification_policy is None
    assert definition.approval_policy is None
    assert definition.server_id is None
    assert definition.description is None
    assert definition.preview_id is None
    assert definition.created_by is None
    assert definition.updated_by is None
    assert definition.updated_at is None
    assert definition.archived_at is None
    assert definition.id
    assert definition.created_at


def test_automation_preview_defaults():
    """AutomationPreview default values match the spec."""
    from tldw_chatbook.Scheduling.models import (
        AutomationFamily,
        AutomationPreview,
        PreviewStatus,
    )

    preview = AutomationPreview(family=AutomationFamily.AGENT_TASK)

    assert preview.family is AutomationFamily.AGENT_TASK
    assert preview.status is PreviewStatus.VALID
    assert preview.owner_id == "local"
    assert preview.mode is None
    assert preview.definition_id is None
    assert preview.definition_version is None
    assert preview.payload_hash is None
    assert preview.normalized_config is None
    assert preview.validation_errors is None
    assert preview.warnings is None
    assert preview.visibility_policy is None
    assert preview.schedule_preview is None
    assert preview.redaction_policy is None
    assert preview.expires_at is None
    assert preview.created_by is None
    assert preview.consumed_at is None
    assert preview.created_definition_id is None
    assert preview.id
    assert preview.created_at


def test_automation_audit_event_defaults():
    """AutomationAuditEvent default values match the spec."""
    from tldw_chatbook.Scheduling.models import AutomationAuditEvent

    event = AutomationAuditEvent(
        definition_id="def-123",
        event_type="created",
        actor="user@example.com",
        summary="Definition created",
    )

    assert event.definition_id == "def-123"
    assert event.event_type == "created"
    assert event.actor == "user@example.com"
    assert event.summary == "Definition created"
    assert event.owner_id == "local"
    assert event.before is None
    assert event.after is None
    assert event.request_id is None
    assert event.idempotency_key is None
    assert event.id
    assert event.created_at
