"""Tests for Scheduling Pydantic models."""

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from tldw_chatbook.Scheduling.models import (
    AutomationAuditEvent,
    AutomationDefinition,
    AutomationFamily,
    AutomationPreview,
    AutomationResult,
    AutomationRun,
    Health,
    Lifecycle,
    PreviewStatus,
    ReminderTask,
    ReviewState,
    RunOutcome,
    RunStatus,
    ScheduleKind,
    TaskStatus,
)


def test_task_status_values() -> None:
    """TaskStatus enum values must match the spec."""
    assert TaskStatus.WAITING.value == "waiting"


def test_reminder_task_defaults() -> None:
    """ReminderTask default values match the spec."""
    run_at = datetime.now(timezone.utc)
    task = ReminderTask(
        title="Test reminder",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=run_at,
    )

    assert task.title == "Test reminder"
    assert task.schedule_kind is ScheduleKind.ONE_TIME
    assert task.run_at == run_at
    assert task.owner_id == "local"
    assert task.enabled is True
    assert task.last_status is TaskStatus.WAITING
    assert task.sync_version == 0
    assert task.server_id is None
    assert task.body is None
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
    assert task.created_at.tzinfo is not None


def test_reminder_task_recurring_defaults() -> None:
    """Recurring ReminderTask defaults require cron and timezone."""
    task = ReminderTask(
        title="Daily reminder",
        schedule_kind=ScheduleKind.RECURRING,
        cron="0 9 * * *",
        timezone="UTC",
    )

    assert task.cron == "0 9 * * *"
    assert task.timezone == "UTC"


def test_automation_definition_defaults() -> None:
    """AutomationDefinition default values match the spec."""
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
    assert definition.created_at.tzinfo is not None


def test_automation_preview_defaults() -> None:
    """AutomationPreview default values match the spec."""
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
    assert preview.created_at.tzinfo is not None


def test_automation_audit_event_defaults() -> None:
    """AutomationAuditEvent default values match the spec."""
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
    assert event.created_at.tzinfo is not None


def test_reminder_task_missing_required_fields() -> None:
    """ReminderTask requires title and schedule_kind."""
    with pytest.raises(ValidationError):
        ReminderTask()

    with pytest.raises(ValidationError):
        ReminderTask(title="Missing kind", schedule_kind=None)  # type: ignore[arg-type]


def test_automation_definition_missing_required_fields() -> None:
    """AutomationDefinition requires family and name."""
    with pytest.raises(ValidationError):
        AutomationDefinition()

    with pytest.raises(ValidationError):
        AutomationDefinition(name="No family")


def test_invalid_enum_values_raise() -> None:
    """Invalid enum values are rejected."""
    with pytest.raises(ValidationError):
        ReminderTask(
            title="Bad status",
            schedule_kind=ScheduleKind.ONE_TIME,
            run_at=datetime.now(timezone.utc),
            last_status="not_a_status",  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError):
        AutomationDefinition(
            name="Bad family",
            family="not_a_family",  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError):
        AutomationPreview(
            family=AutomationFamily.AGENT_TASK,
            status="not_a_status",  # type: ignore[arg-type]
        )


def test_reminder_task_one_time_requires_run_at() -> None:
    """One-time reminders require run_at."""
    with pytest.raises(ValidationError):
        ReminderTask(
            title="No run_at",
            schedule_kind=ScheduleKind.ONE_TIME,
        )


def test_reminder_task_recurring_requires_cron_and_timezone() -> None:
    """Recurring reminders require cron and timezone."""
    with pytest.raises(ValidationError):
        ReminderTask(
            title="No cron",
            schedule_kind=ScheduleKind.RECURRING,
            timezone="UTC",
        )

    with pytest.raises(ValidationError):
        ReminderTask(
            title="No timezone",
            schedule_kind=ScheduleKind.RECURRING,
            cron="0 9 * * *",
        )


def test_extra_fields_forbidden() -> None:
    """All models reject unknown fields."""
    with pytest.raises(ValidationError):
        ReminderTask(
            title="Extra",
            schedule_kind=ScheduleKind.ONE_TIME,
            run_at=datetime.now(timezone.utc),
            unknown_field="nope",
        )

    with pytest.raises(ValidationError):
        AutomationDefinition(
            name="Extra",
            family=AutomationFamily.AGENT_TASK,
            unknown_field="nope",
        )

    with pytest.raises(ValidationError):
        AutomationPreview(
            family=AutomationFamily.AGENT_TASK,
            unknown_field="nope",
        )

    with pytest.raises(ValidationError):
        AutomationAuditEvent(
            definition_id="def-123",
            event_type="created",
            actor="user",
            summary="Summary",
            unknown_field="nope",
        )


def test_run_status_matches_server_vocabulary_plus_timed_out():
    assert {s.value for s in RunStatus} == {
        "queued", "running", "completed", "failed",
        "skipped", "cancelled", "timed_out",
    }


def test_run_outcome_and_review_state_match_server_literals():
    assert {o.value for o in RunOutcome} == {
        "finding", "no_match", "partial", "degraded", "none",
    }
    assert {r.value for r in ReviewState} == {"unread", "read", "dismissed"}


def test_automation_run_defaults():
    run = AutomationRun(
        owner_id="local", definition_id="d1",
        definition_version=1, trigger_reason="scheduled",
    )
    assert run.status == RunStatus.QUEUED
    assert run.outcome == RunOutcome.NONE
    assert run.schedule_slot is None


def test_automation_result_defaults_to_unread():
    result = AutomationResult(
        owner_id="local", definition_id="d1", run_id="r1",
        kind="finding", title="t", summary="s", dedupe_key="k",
    )
    assert result.review_state == ReviewState.UNREAD
    assert result.answer_mode == "none"


def test_definition_gains_parity_fields_with_defaults():
    d = AutomationDefinition(family=AutomationFamily.RECURRING_QUESTION, name="n")
    assert d.resolution_state == "open"
    assert d.finding_policy == {"preset": "balanced_findings"}
    assert d.retention_policy == {"mode": "default"}
    assert d.next_run_at is None and d.transfer_state is None


def test_automation_run_tolerates_null_json_columns():
    """A DB row created without snapshot kwargs stores NULL; must not raise."""
    run = AutomationRun(
        owner_id="local", definition_id="d1",
        definition_version=1, trigger_reason="scheduled",
        scope_snapshot=None, finding_policy_snapshot=None,
        rag_request_snapshot=None, run_summary=None, evidence_summary=None,
    )
    assert run.scope_snapshot == {}
    assert run.finding_policy_snapshot == {}
    assert run.rag_request_snapshot == {}
    assert run.run_summary == {}
    assert run.evidence_summary == {}


def test_automation_result_tolerates_null_json_columns():
    """A DB row created without snapshot kwargs stores NULL; must not raise."""
    result = AutomationResult(
        owner_id="local", definition_id="d1", run_id="r1",
        kind="finding", title="t", summary="s", dedupe_key="k",
        confidence=None, source_refs=None, visibility_destination=None,
    )
    assert result.confidence == {}
    assert result.source_refs == []
    assert result.visibility_destination == {}


def test_preview_error_lists_hold_server_shaped_dicts():
    p = AutomationPreview(
        family=AutomationFamily.RECURRING_QUESTION,
        validation_errors=[{"field": "config.scope", "code": "scope_empty",
                            "message": "Scope must include at least one readable searchable source."}],
        warnings=[{"code": "source_unavailable", "source": "chats"}],
    )
    assert p.validation_errors[0]["code"] == "scope_empty"
    assert p.risk_class is None
