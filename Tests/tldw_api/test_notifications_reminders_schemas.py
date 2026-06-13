import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api import (
    NotificationPreferencesUpdateRequest,
    NotificationResponse,
    NotificationsListResponse,
    ReminderTaskCreateRequest,
    ReminderTaskResponse,
    ReminderTaskUpdateRequest,
)


def test_reminder_task_create_request_requires_schedule_specific_fields():
    one_time = ReminderTaskCreateRequest(
        title="Follow up",
        schedule_kind="one_time",
        run_at="2026-04-24T12:00:00Z",
    )
    recurring = ReminderTaskCreateRequest(
        title="Daily review",
        schedule_kind="recurring",
        cron="0 9 * * *",
        timezone="America/Los_Angeles",
    )

    assert one_time.model_dump(exclude_none=True, mode="json") == {
        "title": "Follow up",
        "schedule_kind": "one_time",
        "run_at": "2026-04-24T12:00:00Z",
        "enabled": True,
    }
    assert recurring.cron == "0 9 * * *"

    with pytest.raises(ValidationError):
        ReminderTaskCreateRequest(title="Broken", schedule_kind="one_time")

    with pytest.raises(ValidationError):
        ReminderTaskCreateRequest(title="Broken", schedule_kind="recurring", cron="0 9 * * *")


def test_notification_and_reminder_responses_match_server_payload_shape():
    task = ReminderTaskResponse.model_validate(
        {
            "id": "task-1",
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "title": "Follow up",
            "schedule_kind": "one_time",
            "run_at": "2026-04-24T12:00:00Z",
            "enabled": True,
            "created_at": "2026-04-24T11:00:00Z",
            "updated_at": "2026-04-24T11:00:00Z",
        }
    )
    listed = NotificationsListResponse.model_validate(
        {
            "items": [
                {
                    "id": 7,
                    "user_id": "user-1",
                    "kind": "reminder_due",
                    "title": "Due",
                    "message": "Time to follow up",
                    "severity": "info",
                    "created_at": "2026-04-24T12:00:00Z",
                    "snooze_until": "2026-04-24T12:30:00Z",
                }
            ],
            "total": 1,
        }
    )

    assert task.id == "task-1"
    assert isinstance(listed.items[0], NotificationResponse)
    assert listed.items[0].snooze_until == "2026-04-24T12:30:00Z"


def test_notification_update_requests_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        ReminderTaskUpdateRequest(unknown_field=True)

    with pytest.raises(ValidationError):
        NotificationPreferencesUpdateRequest(push_enabled=True)
