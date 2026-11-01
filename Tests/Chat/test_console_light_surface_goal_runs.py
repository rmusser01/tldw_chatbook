"""TASK-34100.5.3 review round 3 (R3-1): a Console goal run is not plain chat.

The owner's "Light surface by default" ruling covers plain Console chat. A
goal iteration is automatic work its launch scoped: the reviewed launch picks
the catalog tools and the runtime tools (``run_skill_script`` by default),
and the handoff tells the model which verifier invocations to run. Before
this fix, ``light_surface_for`` never looked at the goal context, so with
Full agent tools off (the default) every goal iteration got the light plan:
no ``run_skill_script`` closure, no discovery, and the plain prompt plus the
"Answer directly" hint. A goal with a command-backed check could then never
record its evidence -- ``run_skill_script`` is the only path to
``goal.begin_script``.

These tests drive a REAL goal iteration (``ConsoleGoalCoordinator
.dispatch_once`` -> controller -> ``ConsoleAgentBridge.run_reply``) with a
real trusted skill script on a bridge that reads the production light-surface
switch, which is off on this isolated profile.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_cli_verification import trusted_skill
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture
from tldw_chatbook.Agents.goal_models import VerificationSpec
from tldw_chatbook.Agents.plain_chat_tool_surface import (
    FULL_AGENT_TOOLS_GATE_KEY,
    light_surface_configured,
)
from tldw_chatbook.config import save_setting_to_cli_config

pytestmark = pytest.mark.bootstrap_profile

stores = _stores_fixture

LIGHT_HINT = "Answer directly; call a tool only when the request needs one."
#: The opening of the agent operating prompt
#: (``agents.console_agent_operating``) a goal iteration has always carried.
AGENT_PROMPT = "You are a capable assistant with optional tools."


@pytest.fixture(autouse=True)
def _full_agent_tools_off():
    """A fresh profile: Full agent tools off, so plain chat is light."""
    assert save_setting_to_cli_config("tools", FULL_AGENT_TOOLS_GATE_KEY, False)
    yield
    assert save_setting_to_cli_config("tools", FULL_AGENT_TOOLS_GATE_KEY, False)


def _script_call_provider():
    """The model runs the launch's verifier once, then reports."""
    count = 0

    def provider(**kwargs):
        nonlocal count
        count += 1
        message = {"content": "finished"}
        if count == 1:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-check",
                        "type": "function",
                        "function": {
                            "name": "run_skill_script",
                            "arguments": json.dumps(
                                {
                                    "skill_name": "verifier",
                                    "script_path": "scripts/check.py",
                                    "args": [],
                                }
                            ),
                        },
                    }
                ],
            }
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    return provider


def _tool_names(call) -> set[str]:
    return {tool["function"]["name"] for tool in call.get("tools") or ()}


@pytest.mark.asyncio
async def test_a_goal_iteration_keeps_its_launch_scope_on_a_light_profile(
    stores, monkeypatch, tmp_path
):
    """With Full agent tools off, a goal iteration still discloses
    ``run_skill_script``, runs the launch's verifier through
    ``goal.begin_script`` (the evidence record exists, with the script's real
    exit code), and carries the agent operating prompt, not the light hint.
    A plain send on the same bridge is still light (the control)."""
    scope, path, trust = trusted_skill(
        tmp_path, "print('checked')\nraise SystemExit(3)\n"
    )
    runs, persistence, registry, req = stores
    verifier = VerificationSpec(
        id="check",
        executor_tool_id="run_skill_script",
        verifier_path=str(path),
        verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        skill_trust_ref=trust.current_fingerprint_digest("verifier"),
        input_paths=("fixture.txt",),
    )
    goal_stores = (
        runs,
        persistence,
        registry,
        req.model_copy(update={"verifiers": (verifier,)}),
    )
    goal, _store, session, controller, coordinator, gateway, calls = build_goal_rig(
        goal_stores, monkeypatch, _script_call_provider()
    )
    bridge = controller._agent_bridge
    bridge._skills_service = scope
    # The production reader ConsoleRuntime hands every bridge.
    bridge._light_tool_surface = light_surface_configured
    controller.set_pending_skill_script = lambda *args, **kwargs: None
    try:
        # Control: outside the goal, this bridge gives plain chat the light
        # surface -- so what follows is the goal context's doing.
        assert bridge.light_surface_for((), session_id=session.id)

        result = await coordinator.dispatch_once(goal.id)

        assert result.outcome is not None, result
        assert "run_skill_script" in _tool_names(calls[0]), sorted(
            _tool_names(calls[0])
        )
        assert len(result.tool_records) == 1, result
        evidence = result.tool_records[0]
        assert evidence.result.exit_code == 3
        assert evidence.invocation.verifier_path == str(path)
        assert evidence.result.stdout == "checked\n"
        assert AGENT_PROMPT in str(calls[0])
        assert LIGHT_HINT not in str(calls[0])
        # The goal's own narrowing still applies: no sub-agents in a goal.
        assert "spawn_subagent" not in _tool_names(calls[0])
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_a_goal_iteration_without_a_check_still_gets_the_agent_prompt(
    stores, monkeypatch
):
    """The launch's runtime scope, not the light plan, governs a goal
    iteration even when the model runs no check: the request carries the
    agent operating prompt and none of the light surface's one-liners."""
    goal, _store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    bridge = controller._agent_bridge
    bridge._light_tool_surface = light_surface_configured
    try:
        await coordinator.dispatch_once(goal.id)
        outgoing = str(calls[0])
        assert AGENT_PROMPT in outgoing
        assert LIGHT_HINT not in outgoing
    finally:
        await coordinator.shutdown()
        await gateway.aclose()
