"""Unit + fixture-parity tests for the pure local automation preview service."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_chatbook.Scheduling.automation_preview import preview_automation_definition
from tldw_chatbook.Scheduling.models import AutomationFamily, PreviewStatus

_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "server_responses" / "automation_preview_response.json"
)


def _valid_payload() -> dict:
    return {
        "mode": "create",
        "family": "recurring_question",
        "name": "Daily stand-up summary",
        "description": "Ask the team.",
        "config": {
            "scope": {"mode": "all_searchable_library"},
            "finding_policy": {"preset": "balanced_findings"},
            "retention_policy": {"mode": "default"},
            "generation_mode": "optional",
        },
        "input": {"question": "What did you work on yesterday?"},
        "schedule": {"kind": "daily", "time_of_day": "09:00", "timezone": "UTC"},
    }


# --- direct unit tests -------------------------------------------------------


def test_valid_payload_yields_valid_status_and_no_errors():
    result = preview_automation_definition(_valid_payload(), now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result.status == PreviewStatus.VALID
    assert result.validation_errors == []
    assert result.family == AutomationFamily.RECURRING_QUESTION


def test_invalid_schedule_kind_yields_invalid_status_with_one_error():
    payload = _valid_payload()
    payload["schedule"] = {"kind": "monthly", "day_of_month": 1}
    result = preview_automation_definition(payload, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result.status == PreviewStatus.INVALID
    assert result.validation_errors == [
        {
            "field": "schedule.kind",
            "code": "unsupported",
            "message": "Unsupported schedule kind: monthly",
        }
    ]
    assert result.schedule_preview["next_occurrences"] == []


def test_schedule_preview_has_up_to_three_next_occurrences():
    result = preview_automation_definition(_valid_payload(), now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result.schedule_preview["next_occurrences"] == [
        "2026-01-01T09:00:00+00:00",
        "2026-01-02T09:00:00+00:00",
        "2026-01-03T09:00:00+00:00",
    ]
    # And the base normalized-schedule keys are still present (server parity).
    assert result.schedule_preview["kind"] == "daily"
    assert result.schedule_preview["time_of_day"] == "09:00"


def test_one_time_schedule_yields_single_occurrence():
    payload = _valid_payload()
    payload["schedule"] = {"kind": "one_time", "run_at": "2026-06-01T12:00:00+00:00"}
    result = preview_automation_definition(payload, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result.schedule_preview["next_occurrences"] == ["2026-06-01T12:00:00+00:00"]


def test_visibility_policy_defaults_to_findings_only_for_recurring_question():
    result = preview_automation_definition(_valid_payload(), now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result.visibility_policy == {"mode": "findings_only"}
    assert result.normalized_config["visibility_policy"] == "findings_only"


def test_normalized_config_carries_family_and_notification_approval_policies():
    payload = _valid_payload()
    payload["notification_policy"] = {"on_success": "silent"}
    payload["approval_policy"] = {"mode": "auto"}
    result = preview_automation_definition(payload, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result.normalized_config["family"] == "recurring_question"
    assert result.normalized_config["notification_policy"] == {"on_success": "silent"}
    assert result.normalized_config["approval_policy"] == {"mode": "auto"}


def test_update_mode_requires_definition_id_and_version():
    payload = _valid_payload()
    payload["mode"] = "update"
    result = preview_automation_definition(payload, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result.status == PreviewStatus.INVALID
    codes = {(err["field"], err["code"]) for err in result.validation_errors}
    assert ("definition_id", "required_for_update") in codes
    assert ("definition_version", "required_for_update") in codes


def test_update_mode_accepts_definition_id_and_version():
    payload = _valid_payload()
    payload["mode"] = "update"
    payload["definition_id"] = "def_123"
    payload["definition_version"] = 2
    result = preview_automation_definition(payload, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result.status == PreviewStatus.VALID
    assert result.definition_id == "def_123"
    assert result.definition_version == 2


def test_create_mode_rejects_definition_id_and_version():
    payload = _valid_payload()
    payload["definition_id"] = "def_123"
    payload["definition_version"] = 1
    result = preview_automation_definition(payload, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    codes = {(err["field"], err["code"]) for err in result.validation_errors}
    assert ("definition_id", "not_allowed_for_create") in codes
    assert ("definition_version", "not_allowed_for_create") in codes


def test_agent_task_family_yields_single_unsupported_error():
    payload = _valid_payload()
    payload["family"] = "agent_task"
    result = preview_automation_definition(payload, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result.status == PreviewStatus.INVALID
    assert result.validation_errors == [
        {
            "field": "family",
            "code": "unsupported",
            "message": "Unsupported automation family: agent_task",
        }
    ]


def test_unrecognized_family_raises_value_error():
    payload = _valid_payload()
    payload["family"] = "bogus_family"
    with pytest.raises(ValueError):
        preview_automation_definition(payload, now=datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_now_defaults_to_current_time_when_omitted():
    # Just exercises the default-`now` path -- no crash, occurrences computed.
    result = preview_automation_definition(_valid_payload())
    assert len(result.schedule_preview["next_occurrences"]) == 3


# --- fixture parity -----------------------------------------------------------


def _load_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text())


@pytest.mark.parametrize(
    "case_name",
    ["valid_recurring_question_create", "invalid_recurring_question_bad_schedule_kind"],
)
def test_fixture_parity(case_name: str):
    fixture = _load_fixture()[case_name]
    now = datetime.fromisoformat(fixture["now"])
    result = preview_automation_definition(fixture["request"], now=now)
    expected = fixture["response"]

    assert result.status.value == expected["status"]
    assert result.validation_errors == expected["validation_errors"]
    assert result.schedule_preview == expected["schedule_preview"]
    assert result.warnings == expected["warnings"]
    assert result.visibility_policy == expected["visibility_policy"]
    assert result.redaction_policy == expected["redaction_policy"]
    assert result.normalized_config == expected["normalized_config"]
    assert result.mode == expected["mode"]
    assert result.family.value == expected["family"]
    assert result.definition_id == expected["definition_id"]
    assert result.definition_version == expected["definition_version"]
