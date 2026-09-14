# Tests/Agents/test_agent_routing_integration.py
"""Spawn routing integration (ADR-147, TASK-32477 Task 6): the Task 5 pure
resolver wired into ``AgentService``'s spawn path.

A spawned child can now run on a DIFFERENT provider/model than its parent
(a routed preset, a configured sub-agent default, or — when the operator
opts in — an ad-hoc override), carrying its own resolved base_url and
sampling params, with the resolved target snapshotted onto the child run
row. A ``RoutingError`` surfaces as a loud spawn-tool error
(``SpawnAdmissionRefusal``) BEFORE any budget/fleet work: no child run row,
no slot consumed.

Service construction mirrors ``Tests/Agents/test_fleet_runtime.py``'s
``make_fleet_service``: an explicit ``FleetCoordinator`` (the fleet opt-in),
a ``FleetChat`` addressed script as the capturing ``chat_call`` stub (its
``calls`` record every call's kwargs), and run rows in a tmp-path
``AgentRunsDB``. Two seams are pinned per test:

* ``AgentService(app_config=...)`` — the resolver's app-config input, an
  explicit dict so no test reads the developer's live config.toml.
* ``agent_service.load_agents_routing_config`` — monkeypatched to a fixed
  ``AgentsRoutingConfig`` (autouse default below) for the same reason: the
  real function reads the live ``[agents]`` TOML/env tier.
"""

import json
import time

import pytest

from Tests.Agents.conftest import join_fleet_children
from Tests.Agents.test_agent_service import FleetChat, fence
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    SPAWN_TOOL_NAME,
    WAIT_AGENTS_TOOL_NAME,
    AgentConfig,
    AgentDefinition,
    RunBudget,
)
from tldw_chatbook.Agents.agent_routing import AgentsRoutingConfig
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

#: The resolver's app-config fixture (same shape as test_agent_routing.py's):
#: one keyless-family custom endpoint, per-entry params, and a chat_default
#: the preset's own params must beat.
APP_CFG = {
    "custom_endpoints": {
        "qwen-local": {
            "display_name": "Qwen Local",
            "family": "llama_cpp",
            "base_url": "http://127.0.0.1:8080",
            "params": {"top_k": 40},
        }
    },
    "chat_defaults": {"temperature": 0.9},
    "api_settings": {"llama_cpp": {"model": "llama-3-8b"}},
}

#: APP_CFG without ``chat_defaults``: the inherit-path child's params then
#: come from the resolver's own function fallbacks (temperature 0.7 /
#: top_p 0.95), which is what makes "never the parent's params" observable.
BARE_APP_CFG = {"api_settings": {"llama_cpp": {"model": "llama-3-8b"}}}

IMPLEMENTER = AgentDefinition(
    name="implementer",
    instructions="Implement.",
    provider="custom-ep:qwen-local",
    model="qwen3.8-27b",
    params=(("temperature", 0.2),),
)

CFG = AgentConfig(
    model="parent-model",
    system_prompt="You are helpful.",
    allowed_tools=("calculator", SPAWN_TOOL_NAME),
    budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=4),
)


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


@pytest.fixture(autouse=True)
def _default_routing(monkeypatch):
    """Pin the ``[agents]`` routing keys to their shipped defaults.

    ``load_agents_routing_config`` reads live config via
    ``Agents.run_log._setting``; without this pin a developer's own
    config.toml (e.g. a configured ``subagent_default_provider``) would
    silently re-route these tests' plain spawns.
    """
    monkeypatch.setattr(
        agent_service, "load_agents_routing_config", AgentsRoutingConfig
    )


def _pin_routing(monkeypatch, config: AgentsRoutingConfig):
    monkeypatch.setattr(
        agent_service, "load_agents_routing_config", lambda: config
    )


def _make_service(db, parent_replies, child_replies=None, *, app_config):
    """An AgentService wired for the fleet, capturing every chat_call's
    kwargs, with the resolver's app-config injected (the production default
    falls back to ``config.load_settings()``; tests always inject)."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = FleetChat(parent_replies, child_replies)
    coordinator = FleetCoordinator(max_live=3, clock=time.monotonic)
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        fleet_coordinator=coordinator,
        app_config=app_config,
    )
    return service, chat


def _child_row(db, conversation_id="c"):
    rows = db.list_runs(conversation_id, include_superseded=True)
    return next(row for row in rows if row["agent_kind"] == "subagent")


def test_routed_preset_child_uses_own_provider_and_params(db):
    db.create_agent_definition(IMPLEMENTER)
    service, chat = _make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "implement it", "agent": "implementer"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "combined answer",
        ],
        {"implement it": ["implemented"]},
        app_config=APP_CFG,
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    assert outcome.status == RUN_DONE

    child_calls = [
        c for c in chat.calls if c.get("api_endpoint") == "custom-ep:qwen-local"
    ]
    assert child_calls, "child never called the routed provider"
    assert child_calls[0]["model"] == "qwen3.8-27b"
    # The preset's params (temperature 0.2) beat chat_defaults (0.9); the
    # registry entry's params (top_k 40) ride along; base_url comes from
    # the endpoint entry.
    assert child_calls[0]["temp"] == 0.2
    assert child_calls[0]["topk"] == 40
    assert child_calls[0]["api_base_url"] == "http://127.0.0.1:8080"
    # The parent's own calls stay on the parent endpoint and gain nothing.
    assert chat.parent_calls
    assert all(
        c["api_endpoint"] == "llama_cpp" and "temp" not in c
        for c in chat.parent_calls
    )


def test_override_refusal_returns_tool_error_and_spawns_nothing(db):
    # spawn_override_enabled=false (the pinned default); the parent emits a
    # spawn call carrying an ad-hoc provider arg.
    service, _chat = _make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "x", "provider": "llama_cpp"}),
            "final answer",
        ],
        app_config=APP_CFG,
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    spawn_results = [
        step["result"]
        for step in db.get_run(run_id)["steps"]
        if step["kind"] == "tool_result" and step["tool_name"] == SPAWN_TOOL_NAME
    ]
    assert len(spawn_results) == 1
    assert "[override_disabled]" in spawn_results[0]
    assert db.count_subagent_runs("c") == 0


def test_refusal_consumes_no_fleet_slot(db, monkeypatch):
    # max_subagents=1: had the refused spawn consumed the budget, the later
    # legal spawn would be refused with "sub-agent budget exhausted".
    # A NAMED refusal keeps the loop's redundant secondary counter honest
    # too (the unnamed path's unconditional increment is pre-existing,
    # deliberate, and unreachable at cap 1 only for admitted spawns).
    _pin_routing(monkeypatch, AgentsRoutingConfig())  # overrides disabled
    db.create_agent_definition(IMPLEMENTER)
    cfg = AgentConfig(
        model="parent-model",
        system_prompt="You are helpful.",
        allowed_tools=("calculator", SPAWN_TOOL_NAME),
        budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=1),
    )
    service, chat = _make_service(
        db,
        [
            fence(
                SPAWN_TOOL_NAME,
                {"task": "t1", "agent": "implementer", "provider": "llama_cpp"},
            ),
            fence(SPAWN_TOOL_NAME, {"task": "t2", "agent": "implementer"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"t2": ["child answer"]},
        app_config=APP_CFG,
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    assert outcome.status == RUN_DONE
    spawn_results = [
        step["result"]
        for step in db.get_run(run_id)["steps"]
        if step["kind"] == "tool_result" and step["tool_name"] == SPAWN_TOOL_NAME
    ]
    # One refusal, one admission — in that order — and the admitted child
    # produced the run row the refused one never got.
    assert len(spawn_results) == 2
    assert "[override_disabled]" in spawn_results[0]
    assert db.count_subagent_runs("c") == 1
    assert "child answer" in str(chat.calls[-1]["messages_payload"])


def test_run_row_carries_resolved_snapshot(db):
    db.create_agent_definition(IMPLEMENTER)
    service, _chat = _make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "implement it", "agent": "implementer"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"implement it": ["implemented"]},
        app_config=APP_CFG,
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    assert outcome.status == RUN_DONE

    row = _child_row(db)
    assert row["resolved_provider"] == "custom-ep:qwen-local"
    assert row["resolved_model"] == "qwen3.8-27b"
    assert row["resolved_base_url"] == "http://127.0.0.1:8080"
    params = json.loads(row["resolved_params_json"])
    assert params["temperature"] == 0.2
    assert params["top_k"] == 40


def test_plain_spawn_snapshot_matches_parent(db):
    # No routing configured anywhere: the child inherits the parent's
    # provider/model and the snapshot still records them (source=inherit
    # lives inside the resolver). The parent here carries its OWN sampling
    # params (temperature 0.99) — the child must NOT inherit them; its
    # params come from the resolver's stack for the inherited provider.
    parent_cfg = AgentConfig(
        model="parent-model",
        system_prompt="You are helpful.",
        allowed_tools=("calculator", SPAWN_TOOL_NAME),
        budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=4),
        sampling_params=(("temperature", 0.99),),
    )
    service, chat = _make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "plain child"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"plain child": ["plain answer"]},
        app_config=BARE_APP_CFG,
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=parent_cfg,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    assert outcome.status == RUN_DONE

    # The parent's calls carried its own params; the child's did not.
    assert chat.parent_calls[0]["temp"] == 0.99
    child_calls = chat.child_calls["plain child"]
    assert child_calls[0]["api_endpoint"] == "llama_cpp"
    assert child_calls[0]["model"] == "parent-model"
    assert child_calls[0]["temp"] == 0.7  # resolver fallback, not the parent's
    assert child_calls[0].get("api_base_url") is None

    row = _child_row(db)
    assert row["resolved_provider"] == "llama_cpp"
    assert row["resolved_model"] == "parent-model"
    assert row["resolved_base_url"] is None
    assert json.loads(row["resolved_params_json"]) == {
        "temperature": 0.7,
        "top_p": 0.95,
    }
    # The PRIMARY row carries no snapshot — only spawn-resolved children do.
    parent_row = db.get_run(run_id)
    assert parent_row["resolved_provider"] is None
    assert parent_row["resolved_params_json"] is None


def test_inheriting_child_of_custom_ep_parent_snapshots_endpoint_not_family(db):
    """qodo PR-2651 High: spawn resolution keys inheritance off the parent's
    RAW selection identity, not the flattened execution key. The parent here
    executes through the llama_cpp family (``api_endpoint``) but was selected
    as ``custom-ep:qwen-local`` (``parent_raw_provider``): the inheriting
    child resolves to the endpoint itself — snapshotting its registry
    base_url and entry params — and its calls ride the slug, so the audit
    row (and every continuation copied from it) names the endpoint that
    actually handled the calls."""
    service, chat = _make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "say hi"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"say hi": ["hi back"]},
        app_config=APP_CFG,
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
        parent_raw_provider="custom-ep:qwen-local",
    )
    join_fleet_children(service)
    assert outcome.status == RUN_DONE

    child_calls = chat.child_calls["say hi"]
    assert child_calls[0]["api_endpoint"] == "custom-ep:qwen-local"
    assert child_calls[0]["model"] == "parent-model"  # inherited parent model
    assert child_calls[0]["api_base_url"] == "http://127.0.0.1:8080"
    assert child_calls[0]["topk"] == 40  # registry entry params ride along
    assert child_calls[0]["temp"] == 0.9  # chat_defaults layer (never preset)

    row = _child_row(db)
    assert row["resolved_provider"] == "custom-ep:qwen-local"
    assert row["resolved_model"] == "parent-model"
    assert row["resolved_base_url"] == "http://127.0.0.1:8080"
    params = json.loads(row["resolved_params_json"])
    assert params["top_k"] == 40
    assert params["temperature"] == 0.9


def test_spawn_without_raw_identity_keeps_legacy_family_inheritance(db):
    """The fallback is unchanged: no ``parent_raw_provider`` (every pre-ADR-147
    caller and every plain-provider selection) keys inheritance off
    ``api_endpoint`` exactly as before — family target, no registry URL."""
    service, chat = _make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "say hi"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"say hi": ["hi back"]},
        app_config=APP_CFG,
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    assert outcome.status == RUN_DONE

    child_calls = chat.child_calls["say hi"]
    assert child_calls[0]["api_endpoint"] == "llama_cpp"
    assert child_calls[0].get("api_base_url") is None

    row = _child_row(db)
    assert row["resolved_provider"] == "llama_cpp"
    assert row["resolved_base_url"] is None
