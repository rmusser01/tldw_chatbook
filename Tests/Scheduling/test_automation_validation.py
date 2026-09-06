"""Direct unit tests for the ported automation-authoring validators.

One accept + one reject per distinct code path the server has, per
`automation_validation.py`'s port-parity docstring.
"""

from tldw_chatbook.Scheduling.automation_validation import (
    FINDING_POLICY_PRESETS,
    GENERATION_MODES,
    RETENTION_POLICY_MODES,
    field_error,
    normalize_finding_policy,
    normalize_retention_policy,
    validate_recurring_question_config,
    validate_schedule,
)


# --- field_error -----------------------------------------------------------


def test_field_error_shape():
    assert field_error("x.y", "required", "X is required.") == {
        "field": "x.y",
        "code": "required",
        "message": "X is required.",
    }


# --- validate_schedule -------------------------------------------------------


def test_validate_schedule_accepts_supported_kind():
    normalized, errors, warnings = validate_schedule({"kind": "daily", "time_of_day": "09:00"})
    assert normalized == {"kind": "daily", "time_of_day": "09:00"}
    assert errors == []
    assert warnings == []


def test_validate_schedule_rejects_non_dict():
    normalized, errors, warnings = validate_schedule("not-a-dict")
    assert normalized == {}
    assert errors == [
        {"field": "schedule", "code": "invalid_type", "message": "Schedule must be an object."}
    ]


def test_validate_schedule_rejects_missing_kind():
    normalized, errors, warnings = validate_schedule({})
    assert normalized == {}
    assert errors == [
        {"field": "schedule.kind", "code": "required", "message": "Schedule kind is required."}
    ]


def test_validate_schedule_rejects_unsupported_kind():
    normalized, errors, warnings = validate_schedule({"kind": "monthly"})
    assert normalized == {"kind": "monthly"}
    assert errors == [
        {
            "field": "schedule.kind",
            "code": "unsupported",
            "message": "Unsupported schedule kind: monthly",
        }
    ]


# --- normalize_finding_policy ------------------------------------------------


def test_normalize_finding_policy_accepts_supported_preset():
    errors = []
    result = normalize_finding_policy({"preset": "high_confidence_only"}, errors)
    assert result == {"preset": "high_confidence_only"}
    assert errors == []
    assert "high_confidence_only" in FINDING_POLICY_PRESETS


def test_normalize_finding_policy_defaults_when_missing():
    errors = []
    result = normalize_finding_policy(None, errors)
    assert result == {"preset": "balanced_findings"}
    assert errors == []


def test_normalize_finding_policy_rejects_unsupported_preset():
    errors = []
    result = normalize_finding_policy({"preset": "bogus"}, errors)
    # Still returns a normalized dict (server keeps the bad value visible).
    assert result == {"preset": "bogus"}
    assert errors == [
        {
            "field": "config.finding_policy.preset",
            "code": "unsupported",
            "message": "Unsupported finding policy preset: bogus",
        }
    ]


# --- normalize_retention_policy -----------------------------------------------


def test_normalize_retention_policy_accepts_supported_mode():
    errors = []
    result = normalize_retention_policy({"mode": "custom"}, errors)
    assert result == {"mode": "custom"}
    assert errors == []
    assert "custom" in RETENTION_POLICY_MODES


def test_normalize_retention_policy_defaults_when_missing():
    errors = []
    result = normalize_retention_policy(None, errors)
    assert result == {"mode": "default"}
    assert errors == []


def test_normalize_retention_policy_rejects_unsupported_mode():
    errors = []
    result = normalize_retention_policy({"mode": "bogus"}, errors)
    assert result == {"mode": "bogus"}
    assert errors == [
        {
            "field": "config.retention_policy.mode",
            "code": "unsupported",
            "message": "Unsupported retention policy mode: bogus",
        }
    ]


# --- validate_recurring_question_config ---------------------------------------


def _valid_config() -> dict:
    return {
        "name": "Daily stand-up",
        "description": "Ask the team.",
        "config": {
            "scope": {"mode": "all_searchable_library"},
            "finding_policy": {"preset": "balanced_findings"},
            "retention_policy": {"mode": "default"},
            "generation_mode": "optional",
        },
        "input": {"question": "What did you work on?"},
    }


def test_validate_recurring_question_config_accepts_valid_config():
    normalized, errors, warnings = validate_recurring_question_config(_valid_config())
    assert errors == []
    assert warnings == []
    assert normalized["name"] == "Daily stand-up"
    assert normalized["input"] == {"question": "What did you work on?"}
    assert normalized["config"] == {
        "scope": {"mode": "all_searchable_library", "resolved_sources": ["media_db", "notes", "chats"]},
        "finding_policy": {"preset": "balanced_findings"},
        "retention_policy": {"mode": "default"},
        "generation_mode": "optional",
    }


def test_validate_recurring_question_config_rejects_missing_name():
    config = _valid_config()
    config["name"] = ""
    normalized, errors, warnings = validate_recurring_question_config(config)
    assert {"field": "name", "code": "required", "message": "Name is required."} in errors


def test_validate_recurring_question_config_rejects_missing_question():
    config = _valid_config()
    config["input"] = {}
    normalized, errors, warnings = validate_recurring_question_config(config)
    assert {
        "field": "input.question",
        "code": "required",
        "message": "Question is required.",
    } in errors


def test_validate_recurring_question_config_rejects_unsupported_generation_mode():
    config = _valid_config()
    config["config"]["generation_mode"] = "bogus"
    normalized, errors, warnings = validate_recurring_question_config(config)
    assert {
        "field": "config.generation_mode",
        "code": "unsupported",
        "message": "Unsupported generation mode: bogus",
    } in errors
    assert "bogus" not in GENERATION_MODES


def test_validate_recurring_question_config_defaults_generation_mode_when_absent():
    config = _valid_config()
    del config["config"]["generation_mode"]
    normalized, errors, warnings = validate_recurring_question_config(config)
    assert normalized["config"]["generation_mode"] == "optional"
    assert errors == []


def test_validate_recurring_question_config_folds_in_scope_errors():
    config = _valid_config()
    config["config"]["scope"] = {"mode": "bogus_mode"}
    normalized, errors, warnings = validate_recurring_question_config(config)
    assert {
        "field": "config.scope.mode",
        "code": "unsupported",
        "message": "Unsupported scope mode: bogus_mode",
    } in errors


def test_validate_recurring_question_config_reduces_scope_warnings_to_codes():
    config = _valid_config()
    config["config"]["scope"] = {"mode": "sources", "sources": ["not_a_real_source"]}
    normalized, errors, warnings = validate_recurring_question_config(config)
    # The scope warning dict is `{"code": "source_unavailable", "source": ...}`;
    # only the "code" string survives into this function's own warnings list
    # (a real server behavior, ported as-is -- see the docstring).
    assert warnings == ["source_unavailable"]


# --- task-31414: mode="update" must not backfill unset config keys -----------


def _config_missing_the_trio() -> dict:
    """A config whose ``config`` sub-dict never carried the trio -- the
    genuinely-absent shape a stored row can have (task-31414's scenario),
    not the create/edit modal's always-explicit payload."""
    return {
        "name": "Daily stand-up",
        "config": {},
        "input": {"question": "What did you work on?"},
    }


def test_validate_recurring_question_config_create_mode_still_backfills_unset_keys():
    """AC2: a create needs concrete config -- explicit `mode="create"`
    (the default) backfills exactly as before this task."""
    normalized, errors, warnings = validate_recurring_question_config(
        _config_missing_the_trio(), mode="create"
    )
    assert errors == []
    assert normalized["config"] == {
        "scope": {"mode": "all_searchable_library", "resolved_sources": ["media_db", "notes", "chats"]},
        "finding_policy": {"preset": "balanced_findings"},
        "retention_policy": {"mode": "default"},
        "generation_mode": "optional",
    }


def test_validate_recurring_question_config_update_mode_leaves_unset_keys_absent():
    """AC1: an edit payload that never carried scope/finding_policy/
    retention_policy/generation_mode must not have them invented."""
    normalized, errors, warnings = validate_recurring_question_config(
        _config_missing_the_trio(), mode="update"
    )
    assert errors == []
    assert normalized["config"] == {}


def test_validate_recurring_question_config_update_mode_only_backfills_supplied_key():
    """A key the edit payload DID carry is still normalized; a sibling
    key it didn't carry stays absent -- editing one field must not touch
    the others (task-31414's core claim)."""
    config = _config_missing_the_trio()
    config["config"]["generation_mode"] = "required"
    normalized, errors, warnings = validate_recurring_question_config(config, mode="update")
    assert errors == []
    assert normalized["config"] == {"generation_mode": "required"}


def test_validate_recurring_question_config_update_mode_still_rejects_invalid_supplied_value():
    """AC5: validation strictness is unchanged -- a supplied-but-invalid
    value is rejected identically in update mode."""
    config = _config_missing_the_trio()
    config["config"]["generation_mode"] = "bogus"
    normalized, errors, warnings = validate_recurring_question_config(config, mode="update")
    assert {
        "field": "config.generation_mode",
        "code": "unsupported",
        "message": "Unsupported generation mode: bogus",
    } in errors
    # Still absent, not silently dropped or defaulted away from the bad value.
    assert normalized["config"]["generation_mode"] == "bogus"
