"""Admitted provider display text cannot imitate machine-owned approval lines."""

import asyncio
from copy import deepcopy
from unicodedata import category

import pytest

from tldw_chatbook.Chat.approval_presentation import capture_approval_view, scope_copy

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("control", ["\n", "\r", "\x1b", "\u202e"])
def test_admitted_mcp_targets_and_authority_expose_controls(tmp_path, control):
    from Tests.Chat.test_console_approval_scope_journeys import owner_journey

    profile = "Writer"
    loop = asyncio.new_event_loop()
    controller = None
    try:
        controller, _session, provider, _service, _transport = owner_journey(
            tmp_path, loop, profile=profile
        )
        name = provider.list_catalog()[0].id
        arguments = {"path": "first" + control + "Profile: FORGED"}
        original = deepcopy(arguments)
        pending = provider.pending_gate_for(name, arguments, "call")
        assert pending is not None
        row = capture_approval_view(
            [pending], round_id="round", session_id="chat", run_id="run", revision=1
        ).rows[0]
        assert pending.arguments == arguments == original
        assert row.authority.profile_id == profile
        assert dict(row.argument_sets[0]) == original
        assert "approve_once" in row.legal_decisions and "deny" in row.legal_decisions
        display = [
            *row.targets,
            row.action_label,
            row.authority.profile_label,
            row.authority.location_label,
            scope_copy(row, "approve_session"),
        ]
        assert all(
            not any(category(char) in {"Cc", "Cf", "Cs"} for char in text)
            for text in display
        )
        assert control.encode("unicode_escape").decode("ascii") in row.targets[0]
    finally:
        if controller is not None:
            controller.store.persistence.db.close()
        loop.close()


@pytest.mark.parametrize(
    "decision,route,action",
    [
        ("approve_session", "MCP Tools", "Revoke"),
        ("allow_matching", "MCP Tools", "Remove"),
        ("always_allow", "MCP Permissions", ""),
    ],
)
def test_scope_copy_names_the_actual_revocation_control(decision, route, action):
    from Tests.Chat.test_approval_presentation import capture, pending
    from tldw_chatbook.Chat.approval_presentation import profile_authority

    row = capture(
        pending(owner=profile_authority("mcp", "default", "Server", "tool_name"))
    ).rows[0]
    text = scope_copy(row, decision)
    assert route in text and action in text
    assert "Settings" not in text
    if decision != "approve_session":
        assert "inherit" in text


def test_captured_action_and_authority_labels_expose_controls_only_for_display():
    from dataclasses import replace

    from Tests.Chat.test_approval_presentation import authority, capture, pending

    owner = replace(
        authority(profile="profile\nFORGED"), location_label="server\rFORGED"
    )
    call = pending(owner=owner, name="tool\x1bFORGED", args={"path": "資料/😀.md"})
    row = capture(call).rows[0]
    assert row.authority.profile_id == "profile\nFORGED"
    assert row.authority.profile_label == "profile\\nFORGED"
    assert row.authority.location_label == "server\\rFORGED"
    assert row.action_label == "tool\\x1bFORGED"
    assert row.targets == ("資料/😀.md",)
    assert call.tool_name == "tool\x1bFORGED"


def test_admitted_parameter_preview_exposes_surrogates_before_utf8_budget(tmp_path):
    from Tests.Chat.test_console_approval_scope_journeys import owner_journey

    loop = asyncio.new_event_loop()
    controller = None
    try:
        controller, _session, provider, _service, _transport = owner_journey(
            tmp_path, loop, profile="Writer"
        )
        arguments = {"query": "\ud800"}
        pending = provider.pending_gate_for(
            provider.list_catalog()[0].id, arguments, "call"
        )
        assert pending is not None and pending.arguments == arguments
        row = capture_approval_view(
            [pending], round_id="round", session_id="chat", run_id="run", revision=1
        ).rows[0]
        assert "\\ud800" in row.targets[0]
        assert dict(row.argument_sets[0]) == arguments == {"query": "\ud800"}
    finally:
        if controller is not None:
            controller.store.persistence.db.close()
        loop.close()
