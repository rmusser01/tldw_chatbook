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
