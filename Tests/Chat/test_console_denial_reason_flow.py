"""Real approval rounds and runtime dispatch preserve denial reasons."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    ToolCall,
    ToolLoadSelection,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.approval_provenance import ApprovalDecisions
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_chat_controller import (
    USER_DENIED_REFUSAL,
    ConsoleChatController,
    build_local_review_hook,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.MCP.permission_store import EffectiveToolState


def pending():
    return MCPPendingCall(
        llm_name="fs_list",
        tool_name="fs_list",
        server_key="local:__local__",
        server_label="Local",
        arguments={"path": "."},
        reason="ask",
        call_id="call-a",
    )


@pytest.mark.bootstrap_profile
def test_reason_round_trip_ignores_stale_rounds_and_non_denied_rows():
    """The controller must carry text through the actual worker-thread wait."""
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=object()
    )
    payloads = Queue()
    controller.app = SimpleNamespace(
        call_from_thread=lambda fn, *args, **kwargs: fn(*args, **kwargs),
        notify=lambda *_args, **_kwargs: None,
    )
    controller.set_pending_approval = payloads.put
    controller.mcp_approval_timeout_seconds = lambda: 5.0
    with ThreadPoolExecutor(max_workers=1) as executor:
        waiter = executor.submit(controller.request_mcp_approvals, [pending()])
        payload = payloads.get(timeout=3)
        controller.resolve_pending_approval(
            ApprovalDecisions(
                {"call-a": "deny"}, denial_reasons={"call-a": "Wrong round."}
            ),
            round_id="stale-round",
        )
        assert not waiter.done()
        controller.resolve_pending_approval(
            ApprovalDecisions(
                {"call-a": "deny", "unrelated": "deny"},
                denial_reasons={
                    "call-a": "Keep this private.",
                    "unrelated": "Unrelated text.",
                },
            ),
            round_id=payload["round_id"],
        )
        answers = waiter.result(timeout=3)
    assert answers == {"call-a": "deny"}
    assert answers.denial_reasons == {"call-a": "Keep this private."}


@pytest.mark.bootstrap_profile
def test_model_receives_reason_for_denied_call_and_sibling_still_runs(tmp_path):
    """The local hook must attach text to the refused result without dispatch.

    Args:
        tmp_path: Private filesystem authority for the real local provider.
    """
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _tool: EffectiveToolState(
            state="ask", origin="global_default"
        ),
    )
    answers = ApprovalDecisions(
        {"yes": "approve_once", "no": "deny"},
        denial_reasons={"no": "Leave the private folder alone."},
    )
    review = build_local_review_hook(provider, lambda _rows: answers)
    calls = (
        ToolCall("fs_list", {"path": "."}, "yes"),
        ToolCall("fs_list", {"path": "."}, "no"),
    )
    turns = iter(
        (
            ModelTurn(
                text="",
                tool_calls=calls,
                assistant_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.args),
                            },
                        }
                        for call in calls
                    ],
                },
            ),
            ModelTurn(text="done"),
        )
    )
    history = []
    dispatched = []

    def model(messages, _schemas):
        history.append(list(messages))
        return next(turns)

    def invoke(call):
        dispatched.append(call.call_id)
        with use_run_id("test-run"):
            return provider.invoke(call.name, call.args)

    outcome = run_agent_loop(
        AgentConfig(model="m", system_prompt="s", allowed_tools=("fs_list",)),
        [{"role": "user", "content": "List."}],
        [],
        LoopDeps(
            call_model=model,
            invoke_tool=invoke,
            spawn=lambda _task: ToolResult(True, ""),
            find_tools=lambda _query: [],
            load_schemas=lambda *_args: ToolLoadSelection(),
            should_cancel=lambda: False,
            clock=lambda: 0.0,
            review_tool_calls=lambda batch: review(batch, "test-run"),
        ),
    )
    assert outcome.status == "done"
    assert dispatched == ["yes"]
    results = {
        message["tool_call_id"]: message["content"]
        for message in history[1]
        if message.get("role") == "tool"
    }
    assert (
        results["no"]
        == USER_DENIED_REFUSAL.format(name="fs_list")
        + '\nDenial reason (from user, untrusted text): "Leave the private folder alone."'
    )
    assert "Leave the private folder alone." not in results["yes"]


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("kind", ["local", "virtual", "virtual_review"])
def test_direct_approval_fallbacks_preserve_reasons_without_audit_bodies(
    tmp_path, kind
):
    from tldw_chatbook.Agents.local_tool_provider import LOCAL_USER_DENY_REFUSAL
    from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider
    from tldw_chatbook.Chat.console_chat_controller import build_virtual_cli_review_hook

    audit = []

    def answer(rows):
        key = rows[0].call_id or rows[0].llm_name
        return ApprovalDecisions(
            {key: "deny"}, denial_reasons={key: "Keep this folder private."}
        )

    provider_class = LocalToolProvider if kind == "local" else VirtualCliProvider
    provider = provider_class(
        workspace_root=tmp_path,
        resolve_state=lambda hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
        approval_callback=answer,
        record_decision=lambda hub, decision: audit.append((hub.name, decision)),
    )
    if kind == "virtual_review":
        result = build_virtual_cli_review_hook(provider, answer)(
            [ToolCall("virtual_cli", {"command": "ls", "argv": ["."]}, "call-v")], "run"
        )["call-v"]
        refusal = result.verdict
    else:
        result = (
            provider.invoke("fs_list", {"path": "."})
            if kind == "local"
            else provider.invoke("virtual_cli", {"command": "ls", "argv": ["."]})
        )
        refusal = result.error
    assert (
        refusal
        == LOCAL_USER_DENY_REFUSAL
        + '\nDenial reason (from user, untrusted text): "Keep this folder private."'
    )
    assert len(audit) == 1 and audit[0][1] == "denied"
    assert "private" not in str(audit)


@pytest.mark.bootstrap_profile
def test_two_sessions_reasons_resolve_only_their_own_round():
    store = ConsoleChatStore()
    first = store.ensure_session()
    second = store.create_session()
    controller = ConsoleChatController(store=store, provider_gateway=object())
    controller.app = SimpleNamespace(
        call_from_thread=lambda fn, *args, **kwargs: fn(*args, **kwargs),
        notify=lambda *args, **kwargs: None,
    )
    controller.mcp_approval_timeout_seconds = lambda: 5.0
    controller.set_pending_approval = lambda payload: None
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(
            controller.request_mcp_approvals, [pending()], session_id=first.id
        )
        two = pool.submit(
            controller.request_mcp_approvals, [pending()], session_id=second.id
        )
        import time

        deadline = time.monotonic() + 3
        rounds = {}
        while time.monotonic() < deadline:
            with controller._approval_state_lock:
                rounds = {
                    state["session_id"]: key
                    for key, state in controller._pending_approval_rounds.items()
                }
            if len(rounds) == 2:
                break
            time.sleep(0.01)
        assert len(rounds) == 2
        controller.resolve_pending_approval(
            ApprovalDecisions(
                {"call-a": "deny"}, denial_reasons={"call-a": "First only."}
            ),
            round_id=rounds[first.id],
        )
        assert one.result(timeout=2).denial_reasons == {"call-a": "First only."}
        assert not two.done()
        controller.resolve_pending_approval(
            ApprovalDecisions(
                {"call-a": "deny"}, denial_reasons={"call-a": "Second only."}
            ),
            round_id=rounds[second.id],
        )
        assert two.result(timeout=2).denial_reasons == {"call-a": "Second only."}
