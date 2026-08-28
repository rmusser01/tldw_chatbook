from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    build_virtual_cli_review_hook,
)
from tldw_chatbook.Chat.console_agent_bridge import _compose_run_registry_and_allowed
from tldw_chatbook.MCP.permission_store import EffectiveToolState


ASK = EffectiveToolState(state="ask", origin="global_default")
ALLOW = EffectiveToolState(state="allow", origin="tool_override")


def _provider(root: Path) -> VirtualCliProvider:
    return VirtualCliProvider(
        workspace_root=root,
        resolve_state=lambda _hub: ASK,
    )


def test_review_hook_keeps_same_model_tool_calls_independent(tmp_path):
    (tmp_path / "a.txt").write_text("allowed", encoding="utf-8")
    (tmp_path / "b.txt").write_text("blocked", encoding="utf-8")
    provider = _provider(tmp_path)
    calls = [
        ToolCall(
            "virtual_cli",
            {"command": "cat", "argv": ["a.txt"]},
            "call-a",
        ),
        ToolCall(
            "virtual_cli",
            {"command": "cat", "argv": ["b.txt"]},
            "call-b",
        ),
    ]

    def request(rows):
        assert [(row.tool_name, row.call_id) for row in rows] == [
            ("cat", "call-a"),
            ("cat", "call-b"),
        ]
        return {"call-a": "approve_once", "call-b": "deny"}

    hook = build_virtual_cli_review_hook(provider, request)
    assert hook(calls, "run-1") == {"call-a": "proceed", "call-b": "proceed"}

    with use_run_id("run-1"), use_tool_call_id("call-a"):
        allowed = provider.invoke("virtual_cli", calls[0].args)
    with use_run_id("run-1"), use_tool_call_id("call-b"):
        blocked = provider.invoke("virtual_cli", calls[1].args)

    assert allowed.ok and "allowed" in allowed.content
    assert not blocked.ok and blocked.outcome == "blocked"


def test_review_hook_clears_stale_stamps_before_a_raising_round(tmp_path):
    (tmp_path / "a.txt").write_text("must not run", encoding="utf-8")
    provider = _provider(tmp_path)
    call = ToolCall(
        "virtual_cli",
        {"command": "cat", "argv": ["a.txt"]},
        "call-a",
    )
    pending = provider.pending_gate_for(call)
    assert pending is not None
    provider.apply_batch_decisions(
        "run-1", {"call-a": "approve_once"}, [pending]
    )

    def raise_during_review(_rows):
        raise RuntimeError("approval bridge unavailable")

    hook = build_virtual_cli_review_hook(provider, raise_during_review)
    with pytest.raises(RuntimeError, match="approval bridge unavailable"):
        hook([call], "run-1")

    with use_run_id("run-1"), use_tool_call_id("call-a"):
        result = provider.invoke("virtual_cli", call.args)
    assert not result.ok and result.outcome == "blocked"


def test_stamp_scope_hides_parent_verdicts_and_restores_them(tmp_path):
    provider = _provider(tmp_path)
    call = ToolCall(
        "virtual_cli",
        {"command": "cat", "argv": ["a.txt"]},
        "call-a",
    )
    pending = provider.pending_gate_for(call)
    assert pending is not None
    provider.apply_batch_decisions(
        "run-1", {"call-a": "approve_once"}, [pending]
    )

    with use_tool_call_id("call-a"):
        with provider.stamp_scope("run-1"):
            assert provider._pop_stamp("run-1", "cat") is None
        assert provider._pop_stamp("run-1", "cat") == "approve_once"


@pytest.mark.parametrize(
    ("local_enabled", "kill_switch", "expected"),
    ((True, False, True), (False, False, False), (True, True, False)),
)
def test_controller_composition_honors_local_master_and_kill_switch(
    tmp_path, local_enabled, kill_switch, expected
):
    service = SimpleNamespace(
        get_kill_switch=lambda: kill_switch,
        gate_tool_test=lambda _hub: ALLOW,
        approve_for_session=lambda *_args: None,
        set_tool_state=lambda *_args, **_kwargs: None,
        record_tool_decision=lambda *_args, **_kwargs: None,
        is_session_approved=lambda *_args: False,
    )
    controller = object.__new__(ConsoleChatController)
    controller.app = SimpleNamespace(unified_mcp_service=service)
    turn_context = SimpleNamespace(
        tool_configuration={"local_tools_enabled": local_enabled},
        scratch_space=None,
    )

    provider, hook = controller._compose_virtual_cli_provider(
        session_id="session-1",
        turn_context=turn_context,
        project_root=tmp_path,
    )

    assert (provider is not None) is expected
    assert (hook is not None) is expected


def test_run_registry_advertises_the_one_virtual_cli_model_tool(tmp_path):
    provider = VirtualCliProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
    )

    registry, allowed, _builtin_names, local_names = (
        _compose_run_registry_and_allowed(
            {},
            virtual_cli_provider=provider,
        )
    )

    assert "virtual_cli" in allowed
    assert "virtual_cli" in local_names
    assert registry.load_schema("virtual_cli:virtual_cli").name == "virtual_cli"
