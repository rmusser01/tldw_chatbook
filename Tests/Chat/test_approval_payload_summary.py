"""ADR-090: approval payload marshals rationale/description/summary."""

from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.console_chat_controller import _build_approval_payload


def _row(**overrides):
    base = dict(
        llm_name="fs_write", server_key="agent:builtin", tool_name="fs_write",
        server_label="Built-in", arguments={"path": "a.txt"}, reason="ask",
        options=("approve_once", "deny"), rationale="Saving the config",
        description="Writes a file",
    )
    base.update(overrides)
    return MCPPendingCall(**base)


def test_payload_carries_row_context_and_summary_slot():
    payload = _build_approval_payload("r1", "s1", "run-1", [_row()], 30.0, 123.5)
    row = payload["calls"][0]
    assert row["rationale"] == "Saving the config"
    assert row["description"] == "Writes a file"
    assert payload["summary"] is None
    assert payload["round_id"] == "r1"
    assert payload["run_id"] == "run-1"
    assert payload["timeout_seconds"] == 30.0
    assert payload["deadline_monotonic"] == 123.5


def test_payload_carries_the_dev_row_contract():
    # The extraction must keep every key dev's inline payload emitted --
    # not just the ADR-090 additions -- or remounts would silently drop
    # effects/execution identity from the card's rows.
    payload = _build_approval_payload("r1", "s1", "run-1", [_row()], 30.0, 123.5)
    row = payload["calls"][0]
    assert row["effects"] == []
    assert row["execution_policy"] == "bounded_abandonable"
    assert row["call_id"] == ""
    assert row["full_command"] == ""
    assert row["warning"] == ""
    assert row["scope_notice"] == ""


def test_payload_defaults_empty_context_without_excuse():
    payload = _build_approval_payload(
        "r2", "s1", "run-2", [_row(rationale="", description="")], 0.0, None
    )
    assert payload["calls"][0]["rationale"] == ""
    assert payload["summary"] is None


def test_payload_captures_semantic_revision_without_changing_calls():
    row = _row()
    payload = _build_approval_payload("r", "s", "run", [row], 0, None, revision=7)
    assert payload["presentation_revision"] == payload["view"].revision == 7
    assert payload["view"].rows[0].argument_sets == (row.arguments,)
    payload["summary"] = "advisory only"
    assert payload["presentation_revision"] == 7
    assert payload["calls"][0]["options"] == list(row.options)


def test_settled_finishing_projection_keeps_legacy_changed_call_guard():
    from Tests.Chat.console_interrupt_test_bindings import make_interrupt_host
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    controller = object.__new__(ConsoleChatController)
    controller._interrupt_host = make_interrupt_host(controller)
    payload = _build_approval_payload(
        "r", "s", "run", [_row(call_id="a"), _row(call_id="b")], 0, None
    )
    payload["phase"] = "finishing"
    controller._approval_state_lock = controller._interrupt_host.lock
    controller._parked_approval_payloads = controller._interrupt_host.payloads[
        "approval"
    ]
    controller._parked_approval_payloads["r"] = payload
    controller.set_pending_approval = None
    controller._remount_head = lambda *args: None
    controller.complete_definitive_tool("run", "a", "fs_write")
    assert [row["call_id"] for row in payload["calls"]] == ["b"]
    assert "view" not in payload and "presentation_revision" not in payload
