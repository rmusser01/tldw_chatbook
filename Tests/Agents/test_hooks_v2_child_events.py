"""Child hooks at actual inline/fleet admission and durable settlement."""

import asyncio
import json

import pytest

from Tests.Agents.test_fleet_runtime import (
    FLEET_CFG,
    fence,
    make_fleet_service,
    make_inline_service,
)
from Tests.Agents.test_hooks_v2_execution import command
from tldw_chatbook.Agents.activation import worker_guard
from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME, WAIT_AGENTS_TOOL_NAME
from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.mark.asyncio
@pytest.mark.parametrize("fleet", [False, True])
async def test_child_narrowing_and_post_context_use_real_owner(
    tmp_path, monkeypatch, fleet
):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="h4")
    observed = tmp_path / "start.json"
    start = command(
        "import json,sys;from pathlib import Path;event=json.load(sys.stdin);"
        f"Path({str(observed)!r}).write_text(json.dumps(event));"
        'print(json.dumps({"version":2,"decision":"pass","child_limits":{"tool_ids":[],"budget_caps":{"max_model_turns":3,"max_total_tokens":5000}}}))',
        name="SubagentStart",
        effects=["child_limits"],
        id="child-start",
    )
    post = command(
        'print(\'{"version":2,"decision":"pass","context":[{"text":"child settled instruction","lifetime":"turn"}]}\')',
        name="SubagentStop",
        effects=["context"],
        required=True,
        id="child-stop",
    )
    engine = HookEngine((start, post), lambda *_: True, HookBudgetOwner())
    child = [fence("find_tools", {"query": "calculator"}), "child result"]
    parent = [fence(SPAWN_TOOL_NAME, {"task": "child"})]
    if fleet:
        service, chat, _ = make_fleet_service(
            db, parent + [fence(WAIT_AGENTS_TOOL_NAME, {}), "done"], {"child": child}
        )
    else:
        service, chat = make_inline_service(db, parent + child + ["done"], monkeypatch)
    service._hooks_v2_engine = engine
    service._hooks_v2_session_id = "session"
    service._hooks_v2_turn_id = "turn"
    try:
        _, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.final_text == "done"
        assert db.count_subagent_runs("c") == 1
        event = json.loads(observed.read_text())
        assert event["data"]["model"] == FLEET_CFG.model
        assert event["data"]["budget_caps"]["max_subagents"] == 0
        assert event["data"]["tool_ids"]
        assert "child settled instruction" in str(chat.calls[-1]["messages_payload"])
        child_row = next(
            row for row in db.list_runs("c") if row["agent_kind"] == "subagent"
        )
        assert child_row["status"] == "done"
        assert child_row["budget"]["max_model_turns"] == 3
        assert child_row["budget"]["max_total_tokens"] == 5000
        assert child_row["budget"]["max_subagents"] == 0
        tool_steps = [
            step for step in child_row["steps"] if step.get("kind") == "tool_result"
        ]
        assert tool_steps and tool_steps[0]["tool_outcome"] != "success", child_row[
            "steps"
        ]
    finally:
        await engine.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fleet", [False, True])
async def test_child_refusal_creates_no_child_but_spawn_tool_settles(
    tmp_path, monkeypatch, fleet
):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="h4-deny")
    hook = command("raise SystemExit(1)", name="SubagentStart", effects=["deny"])
    engine = HookEngine((hook,), lambda *_: True, HookBudgetOwner())
    replies = [fence(SPAWN_TOOL_NAME, {"task": "child"}), "done"]
    if fleet:
        service, chat, _ = make_fleet_service(db, replies)
    else:
        service, chat = make_inline_service(db, replies, monkeypatch)
    service._hooks_v2_engine = engine
    service._hooks_v2_session_id = "session"
    service._hooks_v2_turn_id = "turn"
    try:
        _, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.final_text == "done"
        assert db.count_subagent_runs("c") == 0
        assert "initialization refused" in str(chat.calls)
    finally:
        await engine.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fleet", [False, True])
async def test_required_child_stop_holds_actual_parent_after_child_is_durable(
    tmp_path, monkeypatch, fleet
):
    db = AgentRunsDB(tmp_path / "race.db", client_id="h4-race")
    entered = tmp_path / "child-post-entered"
    release = tmp_path / "child-post-release"
    hook = command(
        "from pathlib import Path;import time;"
        f"Path({str(entered)!r}).write_text('entered');"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(0.01)'!r});"
        'print(\'{"version":2,"decision":"pass","context":[{"text":"parent release","lifetime":"turn"}]}\')',
        name="SubagentStop",
        required=True,
        effects=["context"],
    )
    engine = HookEngine((hook,), lambda *_: True, HookBudgetOwner())
    replies = [fence(SPAWN_TOOL_NAME, {"task": "child"})]
    if fleet:
        service, chat, _ = make_fleet_service(
            db,
            replies + [fence(WAIT_AGENTS_TOOL_NAME, {}), "done"],
            {"child": ["child result"]},
        )
    else:
        service, chat = make_inline_service(
            db, replies + ["child result", "done"], monkeypatch
        )
    service._hooks_v2_engine = engine
    service._hooks_v2_session_id = "session"
    service._hooks_v2_turn_id = "turn"
    pending = asyncio.create_task(
        asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
    )
    try:
        for _ in range(2000):
            if entered.exists():
                break
            await asyncio.sleep(0.005)
        assert entered.exists(), (
            pending.result() if pending.done() else None,
            chat.calls,
        )
        child = next(
            row for row in db.list_runs("c") if row["agent_kind"] == "subagent"
        )
        assert child["status"] == "done"
        assert not pending.done()
        release.write_text("release")
        _, outcome = await pending
        assert outcome.final_text == "done"
        assert "parent release" in str(chat.calls[-1]["messages_payload"])
    finally:
        release.write_text("release")
        await asyncio.gather(pending, return_exceptions=True)
        await engine.close()
        db.close()


@pytest.mark.asyncio
async def test_late_child_settlement_cannot_attach_to_a_new_parent(tmp_path):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle

    db = AgentRunsDB(tmp_path / "late.db", client_id="h4-late")
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    child = db.create_run(
        conversation_id="c", agent_kind="subagent", parent_run_id=parent
    )
    db.set_status(parent, "done", "parent done")
    db.set_status(child, "done", "child done")
    hook = command(
        'print(\'{"version":2,"decision":"pass","context":[{"text":"late child context","lifetime":"turn"}]}\')',
        name="SubagentStop",
        effects=["context"],
    )
    engine = HookEngine((hook,), lambda *_: True, HookBudgetOwner())
    lifecycle = HookSessionLifecycle(engine, "session")
    lifecycle.checkpoints.bind_owner(parent, lifecycle.scope_id)
    lifecycle.close_scope(parent)
    current = lifecycle.open_scope()
    try:
        future = lifecycle.install(
            lifecycle.event(
                "SubagentStop",
                run_id=parent,
                data={"child_run_id": child, "status": "done"},
            ),
            parent,
        )
        assert future is None
        assert lifecycle.diagnostics["late_context"] == 1
        await lifecycle.wait(current)
        assert not lifecycle.context.blocks(current, "model")
        assert not engine.processes.records
    finally:
        lifecycle.seal()
        await engine.close()
        db.close()
