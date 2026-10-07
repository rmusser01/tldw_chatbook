"""Captured approval facts never confer permission."""

from dataclasses import replace

from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall


def api():
    from tldw_chatbook.Chat import approval_presentation

    return approval_presentation


def authority(kind="mcp", profile="default", stamp="tool_name"):
    return api().ApprovalAuthority(
        kind,
        profile,
        "Default" if profile == "default" else profile,
        "Captured binding",
        "profile",
        stamp,
        "Settings: tool permissions",
        profile == "default",
    )


def pending(
    *, key="a", args=None, owner=None, options=None, name="read_file", review=False
):
    return MCPPendingCall(
        llm_name=name,
        tool_name=name,
        server_key="external:x",
        server_label="External",
        arguments=args or {"path": "secret.txt"},
        reason="ask",
        call_id=key,
        options=options
        or (
            "approve_once",
            "approve_session",
            "allow_matching",
            "always_allow",
            "deny",
        ),
        presentation_authority=owner,
        requires_individual_review=review,
    )


def capture(*calls):
    return api().capture_approval_view(
        calls, round_id="r", session_id="s", run_id="run", revision=3
    )


def test_raw_shell_scope_is_chat_and_disarm():
    owner = api().ApprovalAuthority(
        "raw_shell",
        "chat-a",
        "Console chat",
        "Captured directory",
        "console_chat",
        "call",
        "Disarm or exit",
    )
    view = capture(
        pending(
            owner=owner,
            options=("approve_once", "approve_session", "deny"),
            review=True,
        )
    )
    copy = api().scope_copy(view.rows[0], "approve_session")
    assert "this Console chat" in copy and "Disarm" in copy and "exits" in copy
    assert not view.bulk_once and view.rows[0].requires_review


def test_default_persistent_scope_discloses_inheritance():
    row = capture(pending(owner=authority())).rows[0]
    copy = api().scope_copy(row, "always_allow")
    assert "Default" in copy and "inherit" in copy and "named profiles" in copy.lower()


def test_native_same_tool_mixed_scopes_withhold_matching():
    calls = (
        pending(key="a", owner=authority()),
        pending(key="b", owner=authority(), args={"path": "b"}),
    )
    original = tuple((c.call_id, dict(c.arguments), c.options) for c in calls)
    view = capture(*calls)
    assert [r.verdict_key for r in view.rows] == ["a", "b"]
    assert all(
        "allow_matching" not in r.legal_decisions and r.withheld_scope_copy
        for r in view.rows
    )
    assert original == tuple((c.call_id, dict(c.arguments), c.options) for c in calls)


def test_shared_verdict_counts_all_calls_and_argument_sets():
    calls = [
        pending(key="", owner=authority(stamp="shared_group"), args={"path": str(i)})
        for i in range(80)
    ]
    view = capture(*calls)
    row = view.rows[0]
    assert view.call_count == row.call_count == 80
    assert row.verdict_key == "read_file"
    assert len(row.argument_sets) == len(row.targets) == 80
    assert "allow_matching" in row.legal_decisions
    assert "80" in api().scope_copy(row, "allow_matching")
    assert "argument_sets" not in repr(row)
    assert row == replace(row, argument_sets=({"different": True},))


def test_unknown_mcp_action_does_not_infer_effects():
    row = capture(
        pending(owner=authority(), name="write_file", args={"path": "a", "text": "b"})
    ).rows[0]
    assert row.action_label == "write_file"
    assert "write" not in " ".join(row.targets).lower()
    assert row.argument_sets == ({"path": "a", "text": "b"},)


def test_policy_reason_does_not_promise_always_ask():
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        format_approval_reason,
    )

    assert (
        format_approval_reason({"reason": "risk_floored", "effects": ["mutates_local"]})
        == "High risk: current policy requires approval for this call."
    )


def test_capture_uses_run_authority_after_active_chat_changes():
    owner = authority("local", "run-profile", "call")
    call = pending(owner=owner)
    view = capture(call)
    active_profile = "other-profile"
    assert active_profile != view.rows[0].authority.profile_id == "run-profile"
    assert view.run_id == "run" and view.revision == 3


def test_missing_authority_never_invents_profile_grant():
    row = capture(pending()).rows[0]
    assert row.authority.grant_domain == "none" and row.authority.profile_id is None
    assert row.legal_decisions == ("approve_once", "deny")


def test_choices_intersect_producer_and_shared_group_options():
    view = capture(
        pending(key="", owner=authority(), options=("approve_once", "deny")),
        pending(key="", owner=authority(), options=("approve_session", "deny")),
    )
    assert view.rows[0].legal_decisions == ("deny",)
    assert not view.bulk_once and view.bulk_deny


def test_capture_does_not_share_mutable_argument_bodies():
    args = {"path": "a", "nested": {"exact": [1, 2]}}
    call = pending(args=args, owner=authority())
    view = capture(call)
    args["nested"]["exact"].append(3)
    assert view.rows[0].argument_sets[0]["nested"] == {"exact": [1, 2]}


def test_provider_import_does_not_load_presentation_module():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    code = """
import importlib.util
import sys
from pathlib import Path
repo = Path.cwd()
assert Path(importlib.util.find_spec('Tests').origin).is_relative_to(repo)
assert Path(importlib.util.find_spec('tldw_chatbook').origin).is_relative_to(repo)
from Tests import real_profile_guard
real_profile_guard.install()
import tldw_chatbook.Agents.mcp_tool_provider
import tldw_chatbook.Agents.local_tool_provider
import tldw_chatbook.Agents.virtual_cli_provider
assert 'tldw_chatbook.Chat.approval_presentation' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_virtual_cli_targets_preserve_complete_captured_argv():
    row = capture(
        pending(
            owner=authority("virtual_cli", "run-profile", "call"),
            name="cat",
            args={"command": "cat", "argv": ["a.txt", "file with spaces.txt"]},
        )
    ).rows[0]
    assert "a.txt" in row.targets[0]
    assert "file with spaces.txt" in row.targets[0]


def test_snapshot_retains_original_inputs_separately_from_safe_display_summary():
    original = {
        "character": {"name": "Ada", "private_note": "original-private-body"},
        "tags": ["one", "two"],
    }
    call = pending(
        owner=authority("local", "captured-profile", "call"),
        args={"summary": "Save Ada"},
    )
    call = replace(call, captured_arguments=original)
    row = capture(call).rows[0]
    assert row.argument_sets == (original,)
    assert "original-private-body" not in " ".join(row.targets)
    assert call.arguments == {"summary": "Save Ada"}
    assert "captured_arguments" not in repr(call)
    assert call == replace(call, captured_arguments={"different": True})
    original["character"]["name"] = "changed"
    assert row.argument_sets[0]["character"]["name"] == "Ada"


def test_empty_original_inputs_do_not_fall_back_to_display_summary():
    call = replace(
        pending(args={"summary": "No original inputs"}), captured_arguments={}
    )
    assert capture(call).rows[0].argument_sets == ({},)


def test_display_targets_redact_keys_and_secret_shapes_without_changing_originals():
    secret = "sk-test-" + "A" * 32
    args = {
        "api_key": "private-key-value",
        "note": secret,
        "nested": {"password": "nested-secret"},
    }
    row = capture(pending(owner=authority(), args=args)).rows[0]
    text = " ".join(row.targets)
    for value in ("private-key-value", secret, "nested-secret"):
        assert value not in text
    assert "***" in text
    assert row.argument_sets == (args,)


def test_complete_long_identifying_targets_remain_distinguishable():
    prefix = "/" + "directory/" * 40
    rows = capture(
        pending(key="a", owner=authority(), args={"path": prefix + "one.md"}),
        pending(key="b", owner=authority(), args={"path": prefix + "two.md"}),
    ).rows
    assert rows[0].targets == (prefix + "one.md",)
    assert rows[1].targets == (prefix + "two.md",)


def test_parameter_excerpt_marks_omissions_without_expanding_original_body():
    args = {"content": "body" * 500, "api_key": "private-key-value"}
    row = capture(pending(owner=authority(), args=args)).rows[0]
    assert row.targets[0].startswith("Parameters preview:")
    assert "parameters omitted" in row.targets[0]
    assert len(row.targets[0].encode("utf-8")) < 320
    assert row.argument_sets == (args,)


def test_url_and_virtual_command_targets_are_complete_and_display_redacted():
    url = "https://example.test/" + "directory/" * 40 + "one.md"
    assert capture(pending(owner=authority(), args={"url": url})).rows[0].targets == (
        url,
    )
    args = {
        "command": "cat",
        "argv": ["--api-key", "private-key-value", "/" + "directory/" * 40 + "one.md"],
    }
    row = capture(pending(owner=authority("virtual_cli"), args=args)).rows[0]
    assert "private-key-value" not in row.targets[0]
    assert args["argv"][-1] in row.targets[0]
    assert "***" in row.targets[0]
    assert row.argument_sets == (args,)
