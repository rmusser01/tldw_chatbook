"""Live scoped messaging, provider context, and metadata boundaries (ADR-136)."""

from __future__ import annotations

import dataclasses
import json
import threading
import time

import pytest

from Tests.Agents.test_agent_service import FleetChat, fence
from Tests.Agents.test_fleet_runtime import FLEET_CFG
from tldw_chatbook.Agents.agent_models import ModelTurn, ToolCall, ToolResult
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.fleet_messages import MessageStore
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

REPORT = "report_to_supervisor"
READ = "read_agent_messages"
SECRET = "private finding: format is JSONL"


def test_child_reports_primary_collects_and_explicitly_relays(tmp_path, monkeypatch):
    from Tests.Agents.conftest import pin_agent_settings

    pin_agent_settings(monkeypatch, run_log_enabled=False)
    from tldw_chatbook.Agents import agent_service, run_log

    monkeypatch.setattr(run_log, "_setting", agent_service._setting)
    inbox = MessageStore().open_inbox("c")
    fleet = FleetCoordinator(max_live=3, clock=time.monotonic, message_inbox=inbox)
    reported, relayed = threading.Event(), threading.Event()

    def child_a_after_report():
        reported.set()
        return fence(READ, {})

    def primary_collect():
        assert reported.wait(5)
        return fence(READ, {})

    def primary_relay():
        handle = next(h for h in fleet.snapshot() if h.task == "B")
        return fence("send_to_agent", {"id": handle.handle_id, "message": SECRET})

    def primary_wait():
        relayed.set()
        return fence("wait_agents", {})

    def child_b_first():
        assert relayed.wait(5)
        return fence("calculator", {"expression": "1+1"})

    chat = FleetChat(
        [
            fence("spawn_subagent", {"task": "A"}),
            fence("spawn_subagent", {"task": "B"}),
            primary_collect,
            primary_relay,
            primary_wait,
            "done",
        ],
        {
            "A": [
                fence(REPORT, {"message": SECRET}),
                child_a_after_report,
                fence("send_to_agent", {"id": "B", "message": "forged"}),
                "A done",
            ],
            "B": [child_b_first, "B done"],
        },
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    service = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=fleet
    )
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[], config=FLEET_CFG, api_endpoint="llama_cpp"
    )
    assert outcome.status == "done", outcome.steps
    assert not service.run_log_writer.is_active
    assert READ in chat.parent_calls[0]["messages_payload"][0]["content"]
    child_prompt = chat.child_calls["A"][0]["messages_payload"][0]["content"]
    assert REPORT in child_prompt and READ not in child_prompt
    receipt = chat.child_calls["A"][1]["messages_payload"][-1]["content"]
    assert "queued" in receipt and SECRET not in receipt
    collected = chat.parent_calls[3]["messages_payload"][-1]["content"]
    assert SECRET in collected and '"collected"' in collected
    # A late-starting child may drain steering before its first model call.
    # Either legal boundary must deliver it exactly once by the second call.
    received = [
        row.get("content") for row in chat.child_calls["B"][1]["messages_payload"]
    ]
    assert received.count("[Steering from supervisor] " + SECRET) == 1
    assert "ERROR" in chat.child_calls["A"][2]["messages_payload"][-1]["content"]
    assert "ERROR" in chat.child_calls["A"][3]["messages_payload"][-1]["content"]
    assert SECRET not in json.dumps(
        [s for s in db.get_run(run_id)["steps"] if s.get("tool_name") == READ]
    )
    assert SECRET not in json.dumps(
        [dataclasses.asdict(s) for s in outcome.steps if s.tool_name == READ]
    )
    child = next(h for h in fleet.snapshot() if h.task == "A")
    assert SECRET not in json.dumps(db.get_run(child.run_id)["steps"])
    from tldw_chatbook.Chat.console_agent_bridge import format_agent_step_marker

    assert SECRET not in repr(
        [
            format_agent_step_marker(step)
            for step in outcome.steps
            if step.tool_name == READ
        ]
    )
    assert not inbox.snapshot()
    # The primary reader is released after completion.
    inbox.reader("next", chain_id=None, automatic=False).close()


@pytest.mark.parametrize("name", [REPORT, READ])
def test_absent_capability_never_falls_through_to_catalog(name):
    invoked = []
    replies = iter(
        [ModelTurn(text="", tool_calls=(ToolCall(name, {}),)), ModelTurn(text="done")]
    )
    deps = LoopDeps(
        call_model=lambda *a: next(replies),
        invoke_tool=lambda c: invoked.append(c) or ToolResult(True),
        spawn=lambda *a: ToolResult(True),
        find_tools=lambda q: [],
        load_schemas=lambda ids: [],
        should_cancel=lambda: False,
        clock=lambda: 0,
    )
    outcome = run_agent_loop(FLEET_CFG, [], [], deps)
    assert outcome.status == "done", outcome.steps
    assert invoked == []
    assert "unavailable" in next(
        s.result for s in outcome.steps if s.kind == "tool_result"
    )


@pytest.mark.parametrize("productive", [True, False])
def test_reader_progress_controls_cycle_detector(productive):
    from tldw_chatbook.Agents.fleet_message_tools import MessageToolResult

    replies = iter(
        [ModelTurn(text="", tool_calls=(ToolCall(READ, {}),)) for _ in range(5)]
        + [ModelTurn(text="done")]
    )
    deps = LoopDeps(
        call_model=lambda *a: next(replies),
        invoke_tool=lambda c: ToolResult(False),
        spawn=lambda *a: ToolResult(True),
        find_tools=lambda q: [],
        load_schemas=lambda ids: [],
        should_cancel=lambda: False,
        clock=lambda: 0,
        read_agent_messages=lambda args: MessageToolResult(
            True, content="{}", collected_count=int(productive)
        ),
    )
    outcome = run_agent_loop(FLEET_CFG, [], [], deps)
    assert outcome.status == ("done" if productive else "stuck")


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"message": "x", "target": "sibling"},
        {"message": 1},
        {"message": ""},
        {"message": "x\x1b"},
    ],
)
def test_report_exact_arguments_refuse_without_queue_mutation(args):
    from tldw_chatbook.Agents.fleet_message_tools import report
    from tldw_chatbook.Agents.fleet_messages import MessageIdentity

    inbox = MessageStore().open_inbox("c")
    sender = inbox.sender(MessageIdentity("h", "run", "parent", None, "child"))
    result = report(sender, args)
    assert not result.ok and result.error == "invalid_message"
    assert inbox.snapshot() == ()


def test_reader_exact_arguments_and_small_cap_preserve_whole_report():
    from tldw_chatbook.Agents.fleet_message_tools import collect
    from tldw_chatbook.Agents.fleet_messages import MessageIdentity

    inbox = MessageStore().open_inbox("c")
    inbox.sender(MessageIdentity("h", "run", "parent", None, "child")).send(SECRET)
    reader = inbox.reader("primary", chain_id=None, automatic=False)
    assert collect(reader, {"chain": "forged"}, 8000).error == "invalid_message"
    assert collect(reader, {}, 50).error == "result_limit_too_small"
    assert len(inbox.snapshot()) == 1
    assert SECRET in collect(reader, {}, 0).content


@pytest.mark.parametrize("trusted", [False, True])
def test_mixed_repeated_reader_calls_cannot_claim_progress_in_text(trusted):
    from tldw_chatbook.Agents.fleet_message_tools import MessageToolResult

    steps = [name for _ in range(5) for name in (READ, "calculator")]
    replies = iter(
        [ModelTurn(text="", tool_calls=(ToolCall(name, {}),)) for name in steps]
        + [ModelTurn(text="done")]
    )
    deps = LoopDeps(
        call_model=lambda *a: next(replies),
        invoke_tool=lambda c: ToolResult(True, "4"),
        spawn=lambda *a: ToolResult(True),
        find_tools=lambda q: [],
        load_schemas=lambda ids: [],
        should_cancel=lambda: False,
        clock=lambda: 0,
        read_agent_messages=lambda args: (
            MessageToolResult(True, "{}", collected_count=1)
            if trusted
            else ToolResult(True, '{"collected_count":99}')
        ),
    )
    outcome = run_agent_loop(FLEET_CFG, [], [], deps)
    assert outcome.status == ("done" if trusted else "stuck")


@pytest.mark.parametrize("existing", [False, True])
def test_toolless_primary_only_reads_an_existing_inbox(tmp_path, existing):
    from tldw_chatbook.Agents.fleet_messages import MessageIdentity

    registry = ToolCatalogRegistry()
    store = MessageStore()
    fleet = None
    if existing:
        inbox = store.open_inbox("c")
        inbox.sender(MessageIdentity("h", "run", "parent", None, "child")).send(SECRET)
        fleet = FleetCoordinator(max_live=3, clock=time.monotonic, message_inbox=inbox)
    chat = FleetChat([fence(READ, {}), "done"] if existing else ["done"])
    service = AgentService(
        db=AgentRunsDB(tmp_path / "runs.db", client_id="test"),
        registry=registry,
        chat_call=chat,
        fleet_coordinator=fleet,
    )
    config = dataclasses.replace(
        FLEET_CFG,
        allowed_tools=(),
        budget=dataclasses.replace(FLEET_CFG.budget, max_subagents=0),
    )
    _, outcome = service.run_turn(
        conversation_id="c", messages=[], config=config, api_endpoint="llama_cpp"
    )
    assert outcome.status == "done", outcome.steps
    assert (READ in chat.parent_calls[0]["messages_payload"][0]["content"]) is existing
    if existing:
        assert SECRET in chat.parent_calls[1]["messages_payload"][-1]["content"]
    else:
        assert store.get_inbox("c") is None


def test_service_reader_rechecks_cancel_and_exact_execution_owner(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Agents import agent_service
    from tldw_chatbook.Agents.agent_models import RunOutcome
    from tldw_chatbook.Agents.fleet_messages import MessageIdentity

    inbox = MessageStore().open_inbox("c")
    inbox.sender(MessageIdentity("h", "run", "parent", None, "child")).send(SECRET)
    fleet = FleetCoordinator(max_live=3, clock=time.monotonic, message_inbox=inbox)
    cancelled = False
    saved = []
    service = AgentService(
        db=AgentRunsDB(tmp_path / "runs.db", client_id="test"),
        registry=ToolCatalogRegistry(),
        fleet_coordinator=fleet,
    )

    def loop(config, messages, active, deps, **kwargs):
        nonlocal cancelled
        saved.append(deps.read_agent_messages)
        cancelled = True
        assert deps.read_agent_messages({}).error == "unavailable"
        cancelled = False
        service.runtime_capacity.close()
        assert deps.read_agent_messages({}).error == "unavailable"
        assert len(inbox.snapshot()) == 1
        return RunOutcome(status="done", steps=[])

    monkeypatch.setattr(agent_service, "run_agent_loop", loop)
    service.run_turn(
        conversation_id="c",
        messages=[],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
        should_cancel=lambda: cancelled,
    )
    assert saved[0]({}).error == "unavailable"
    inbox.reader("later", chain_id=None, automatic=False).close()


def test_automatic_reader_only_collects_its_chain_and_rechecks_kill_switch(
    tmp_path, monkeypatch
):
    from Tests.Agents.test_automatic_child_scope import accepted_context
    from tldw_chatbook.Agents import agent_service, automatic_work_runtime
    from tldw_chatbook.Agents.agent_models import RunOutcome
    from tldw_chatbook.Agents.fleet_messages import MessageIdentity

    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    context = accepted_context(db)
    inbox = MessageStore().open_inbox("conversation")
    for index, chain in enumerate((None, "foreign", context.chain_id)):
        inbox.sender(
            MessageIdentity(str(index), f"child-{index}", "parent", chain, "child")
        ).send(str(index))
    fleet = FleetCoordinator(max_live=3, clock=time.monotonic, message_inbox=inbox)
    with context.scope():
        service = AgentService(
            db=db, registry=ToolCatalogRegistry(), fleet_coordinator=fleet
        )

    def loop(config, messages, active, deps, **kwargs):
        result = deps.read_agent_messages({})
        assert [m["body"] for m in json.loads(result.content)["messages"]] == ["2"]
        assert [m.body for m in inbox.snapshot()] == ["0", "1"]
        original_setting = automatic_work_runtime._setting
        monkeypatch.setattr(
            automatic_work_runtime,
            "_setting",
            lambda key, default: (
                False if key == "autowake_enabled" else original_setting(key, default)
            ),
        )
        assert deps.read_agent_messages({}).error == "unavailable"
        return RunOutcome(status="done", steps=[])

    monkeypatch.setattr(agent_service, "run_agent_loop", loop)
    _, outcome = service.run_turn(
        conversation_id="conversation",
        messages=[],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == "done", outcome.steps


def test_message_runtime_schemas_are_exact_and_catalog_cannot_disclose_forged_names():
    from tldw_chatbook.Agents.agent_models import (
        RUNTIME_TOOL_NAMES,
        ToolCatalogEntry,
    )
    from tldw_chatbook.Agents.fleet_message_tools import (
        READ_AGENT_MESSAGES_SCHEMA,
        REPORT_TO_SUPERVISOR_SCHEMA,
    )
    from tldw_chatbook.Agents.tool_catalog import probe_initial_catalog

    class ForgedProvider:
        def list_catalog(self):
            return [
                ToolCatalogEntry("forged:" + name, name, "forged", "forged")
                for name in (READ, REPORT)
            ]

        def load_schema(self, tool_id):
            pytest.fail("runtime identity cannot load a catalog schema")

    registry = ToolCatalogRegistry()
    registry.register_provider(ForgedProvider())
    assert probe_initial_catalog(registry, (READ, REPORT), 1000, lambda schemas: 1) == ()
    assert {READ, REPORT} <= RUNTIME_TOOL_NAMES
    assert REPORT_TO_SUPERVISOR_SCHEMA.parameters == {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    }
    assert READ_AGENT_MESSAGES_SCHEMA.parameters == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


def test_first_request_plan_matches_live_native_fleet_disclosure(tmp_path, monkeypatch):
    from Tests.Agents.conftest import pin_agent_settings
    from tldw_chatbook.Agents.agent_models import ToolCatalogEntry, ToolSchema
    from tldw_chatbook.Agents.agent_service import build_first_request_schema_plan
    from tldw_chatbook.Agents.native_tools import schemas_to_openai_tools

    pin_agent_settings(monkeypatch, run_log_enabled=False)
    from tldw_chatbook.Agents import agent_service, run_log

    monkeypatch.setattr(run_log, "_setting", agent_service._setting)

    class Catalog:
        def list_catalog(self):
            return [
                ToolCatalogEntry(f"test:t{i}", f"t{i}", "tool", "test")
                for i in range(30)
            ]

        def load_schema(self, tool_id):
            return ToolSchema(
                tool_id, tool_id.split(":")[-1], "tool", {"type": "object"}
            )

    registry = ToolCatalogRegistry()
    registry.register_provider(Catalog())
    config = dataclasses.replace(
        FLEET_CFG, native_tools=True, allowed_tools=tuple(f"t{i}" for i in range(30))
    )
    plan = build_first_request_schema_plan(
        registry,
        config.allowed_tools,
        config,
        "llama_cpp",
        [{"role": "user", "content": "go"}],
        skill_file_enabled=False,
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        run_log_active=False,
        fleet_active=True,
        progress_available=True,
    )
    chat = FleetChat(["done"])
    fleet = FleetCoordinator(
        max_live=3, clock=time.monotonic, message_inbox=MessageStore().open_inbox("c")
    )
    service = AgentService(
        db=AgentRunsDB(tmp_path / "runs.db", client_id="test"),
        registry=registry,
        chat_call=chat,
        fleet_coordinator=fleet,
    )
    _, outcome = service.run_turn(
        conversation_id="c", messages=[], config=config, api_endpoint="groq"
    )
    assert outcome.status == "done"
    assert chat.parent_calls[0]["tools"] == schemas_to_openai_tools(
        list(plan.runtime_schemas + plan.active_schemas)
    )


def test_core_imports_keep_progress_runtime_lazy_and_public_schemas_available():
    import subprocess
    import sys

    probe = """
import sys
from tldw_chatbook.Agents import agent_runtime, agent_service, fleet_coordinator, tool_catalog
assert 'tldw_chatbook.Agents.fleet_message_tools' not in sys.modules
assert 'tldw_chatbook.Agents.fleet_messages' not in sys.modules
assert 'tldw_chatbook.Agents.automatic_work_budget' not in sys.modules
assert 'tldw_chatbook.Agents.automatic_work_runtime' not in sys.modules
assert 'tldw_chatbook.Agents.execution_capacity' not in sys.modules
# Name-based anti-forgery checks do not require loading the queue or schemas.
assert 'read_agent_messages' in tool_catalog.MESSAGE_TOOL_NAMES
from tldw_chatbook.Agents.tool_catalog import READ_AGENT_MESSAGES_SCHEMA, REPORT_TO_SUPERVISOR_SCHEMA
assert READ_AGENT_MESSAGES_SCHEMA.name == 'read_agent_messages'
assert REPORT_TO_SUPERVISOR_SCHEMA.name == 'report_to_supervisor'
assert 'tldw_chatbook.Agents.fleet_messages' not in sys.modules
from tldw_chatbook.Agents.agent_models import WorkOrigin
from tldw_chatbook.Agents.execution_capacity import WorkOrigin as LegacyWorkOrigin
assert LegacyWorkOrigin is WorkOrigin
assert agent_service.WorkOrigin is WorkOrigin
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
