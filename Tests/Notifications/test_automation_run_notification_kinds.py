"""ADR-077 phase-1 result pass-back: automation-run notifications must parse.

The server's agent-task consumer writes one user notification per terminal
run status (``automation_run_succeeded``/``_failed``/``_timed_out``/
``_skipped``) with the definition name as title and the result summary as
message (task-18940 slice 3). The client inbox list parse is the one place
that validates ``kind`` against a closed Literal -- one unknown kind fails
the WHOLE list response -- so these tests pin that the pass-back channel
parses end to end.
"""

from tldw_chatbook.tldw_api.notifications_reminders_schemas import (
    NotificationKind,
    NotificationsListResponse,
    NotificationResponse,
)

_AUTOMATION_KINDS = (
    "automation_run_succeeded",
    "automation_run_failed",
    "automation_run_timed_out",
    "automation_run_skipped",
)


def _inbox_item(kind: str) -> dict:
    return {
        "id": 7,
        "user_id": "1",
        "kind": kind,
        "title": "Morning brief",
        "message": "Ran at 12:00 UTC.",
        "severity": "info",
        "source_domain": "scheduled_tasks",
        "source_job_type": "agent_task_run",
        "source_job_id": "42",
        "dedupe_key": "automation_run:def-1:run-1",
        "created_at": "2026-08-29T12:00:05Z",
    }


def test_every_automation_run_kind_is_in_the_notification_kind_literal():
    # Literal args are introspectable via __args__.
    allowed = set(NotificationKind.__args__)
    for kind in _AUTOMATION_KINDS:
        assert kind in allowed


def test_inbox_list_parses_a_feed_containing_automation_run_results():
    items = [_inbox_item(kind) for kind in _AUTOMATION_KINDS]
    response = NotificationsListResponse.model_validate(
        {"items": items, "total": len(items)}
    )
    assert response.total == 4
    assert [item.kind for item in response.items] == list(_AUTOMATION_KINDS)
    # The pass-back payload fields survive the parse: definition name as
    # title, result summary as message, source ids for correlation.
    first = response.items[0]
    assert first.title == "Morning brief"
    assert first.source_domain == "scheduled_tasks"
    assert first.source_job_type == "agent_task_run"


def test_single_notification_response_parses_an_automation_run_result():
    item = NotificationResponse.model_validate(_inbox_item("automation_run_timed_out"))
    assert item.kind == "automation_run_timed_out"
    assert item.dedupe_key == "automation_run:def-1:run-1"


def test_inbox_list_still_rejects_unknown_kinds():
    # The Literal is the vocabulary gate -- this is the failure mode the
    # comment on NotificationKind warns about (server adds a kind, client
    # feed breaks wholesale). Pydantic must keep rejecting truly unknown
    # kinds so the failure is loud, not silent.
    import pytest

    with pytest.raises(Exception):  # noqa: B017 - ValidationError subclasses differ by version
        NotificationResponse.model_validate(_inbox_item("automation_run_novel"))
