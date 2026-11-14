"""Selection failures give bounded, actionable recovery instructions."""

import pytest

from tldw_chatbook.Backup_Recovery.recovery_service import issue_code, issue_message


@pytest.mark.parametrize("error,code", [
    ("invalid_backup_groups", "invalid_backup_groups"),
    ("data_group_destination_missing", "data_group_destination_missing"),
    ("archive_group_unavailable", "archive_group_unavailable"),
    ("archive_group_dependency_unavailable", "archive_group_dependency_unavailable"),
    ("required_target_group_unavailable:retrieval", "required_target_group_unavailable"),
    ("preserved_group_path_changed:conversations", "preserved_group_path_changed"),
    ("selected_absence_mapping_required", "selected_absence_mapping_required"),
    ("selected_absence_publication_conflict", "selected_absence_publication_conflict"),
])
def test_group_failure_explains_how_to_review_selection(error, code):
    result = issue_code(ValueError(error), kind="restore")
    assert result == code
    assert "group" in issue_message(result).lower()
    assert "backup_operation_failed" not in result


def test_arbitrary_error_text_is_not_rendered_as_a_group_name():
    error = "required_target_group_unavailable:/private/profile-sensitive-name"
    result = issue_code(ValueError(error), kind="restore")
    assert "/private/profile-sensitive-name" not in result + issue_message(result)


def test_capture_outside_scope_refusal_names_an_action_not_a_generic_failure():
    """TASK-34100.16.1 AC#4: this refusal reaches the user with its cause.

    ``capture_source_outside_scope`` is a ``RecoveryRequired`` (a
    ``RuntimeError``), so it used to fall through to the generic
    ``backup_operation_failed``. It means a selected configuration or data
    file lies outside the roots the backup's maintenance session holds.
    """
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    result = issue_code(RecoveryRequired("capture_source_outside_scope"), kind="capture")

    assert result == "capture_source_outside_scope"
    message = issue_message(result).lower()
    # Review v5r2 (R2-3): state only what is known. The refusal has five raise
    # sites and no reproduction, so no cause (a moved folder) and no restart
    # remedy is claimed; it names the path kind and the verified manual route.
    assert "configuration or data file" in message
    assert "outside the storage" in message
    assert "config.toml and the data folder" in message
    for unverified in ("quit", "reopen", "moved a folder", "settings change"):
        assert unverified not in message, unverified
    # Any other bounded RecoveryRequired code keeps the generic fallback.
    assert issue_code(RecoveryRequired("maintenance_roots_required")) == (
        "backup_operation_failed"
    )
