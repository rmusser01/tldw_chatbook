"""Snapshot identity checks without starting a Textual application."""

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.UI.MCP_Modules import mcp_audit_mode


@pytest.mark.parametrize(
    ("old_count", "new_count", "expected"),
    [
        (1, 1, "new-0"),
        (1, 0, None),
        (0, 1, None),
        (2, 1, None),
        (1, 2, None),
        (2, 2, None),
        (0, 0, None),
    ],
)
@private_profile_test
def test_refresh_requires_unique_identity_in_both_snapshots(
    request, old_count, new_count, expected
):
    entry = {"ts": "fixture", "server_key": "local:alpha", "tool_name": "echo"}
    old = {f"old-{i}": (i, dict(entry)) for i in range(old_count)}
    new = {f"new-{i}": (i + 1, dict(entry)) for i in range(new_count)}
    # Same tool name on another server must not create ambiguity or a match.
    new["other-server"] = (0, {**entry, "server_key": "local:beta"})
    assert mcp_audit_mode._unique_entry_key(entry, old, new) == expected


@private_profile_test
def test_renewed_row_keys_never_reactivate_an_older_snapshot(request):
    canvas = mcp_audit_mode.MCPAuditMode()
    entry = {"tool_name": "echo"}
    canvas._entries = [dict(entry), dict(entry)]
    canvas._renew_entry_keys()
    retired = set(canvas._entry_keys)
    assert len(retired) == 2
    canvas._entries = []
    canvas._renew_entry_keys()
    assert canvas._entry_keys == []
    canvas._entries = [dict(entry), dict(entry)]
    canvas._renew_entry_keys()
    assert len(set(canvas._entry_keys)) == 2
    assert retired.isdisjoint(canvas._entry_keys)
