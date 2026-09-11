"""Workspace name validation retains the established display-name policy."""

import pytest

from tldw_chatbook.Utils.input_validation import validate_workspace_name
from tldw_chatbook.Workspaces.models import WorkspaceRecord


@pytest.mark.parametrize("name", ["", " \t\n", None, 123, True, b"name", [], {}])
def test_workspace_name_rejects_blank_or_nontext(name):
    with pytest.raises(ValueError, match="non-blank text"):
        validate_workspace_name(name)


@pytest.mark.parametrize(
    "name", [" Project α / 東京 🧪 ", "x" * 4096, "Project\nNotes", "One\x00Two"]
)
def test_workspace_name_matches_existing_record_validation(name):
    assert (
        validate_workspace_name(name)
        == WorkspaceRecord(workspace_id="test", name=name).name
    )
