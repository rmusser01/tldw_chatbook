"""Explicit preset fallback contracts (ADR-200)."""

import dataclasses
import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    AgentDefinition,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolResult,
    definition_fingerprint,
    definition_from_row,
    validate_agent_definition,
)
from tldw_chatbook.Agents.agent_routing import resolve_preset_fallback_targets
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.fallback_chain import FallbackCandidate, FallbackRuntime
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatModelUnavailableError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile

PAIRS = (("openai", "alternate"), ("custom-ep:backup", "local-model"))
PRESET = AgentDefinition(name="worker", instructions="Work.", fallback_models=PAIRS)
APP = {
    "custom_endpoints": {
        "backup": {
            "display_name": "Backup",
            "family": "llama_cpp",
            "base_url": "http://127.0.0.1:8888",
            "params": {"top_k": 7},
        }
    },
    "api_settings": {"openai": {"model_defaults": {"alternate": {"temperature": 0.1}}}},
    "chat_defaults": {"temperature": 0.8},
}


def test_authoring_round_trip_and_fingerprint(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    try:
        key = db.create_agent_definition(PRESET)
        assert (
            definition_from_row(db.get_agent_definition(key)).fallback_models == PAIRS
        )
        assert definition_fingerprint(PRESET) != definition_fingerprint(
            dataclasses.replace(PRESET, fallback_models=())
        )
        db.update_agent_definition(key, dataclasses.replace(PRESET, fallback_models=()))
        assert db.get_agent_definition(key)["fallback_models"] == []
    finally:
        db.close()


@pytest.mark.parametrize(
    "pairs",
    [
        (("openai", ""),),
        (("https://attacker", "m"),),
        (("openai", "m"), ("openai", "m")),
        (("openai", "m", "url"),),
        (("openai", "m"),) * 9,
        (("openai", "a\nb"),),
    ],
)
def test_invalid_fallback_authoring_is_refused(pairs):
    assert validate_agent_definition(dataclasses.replace(PRESET, fallback_models=pairs))


def test_targets_freeze_own_url_params_and_same_provider_model():
    targets = resolve_preset_fallback_targets(
        APP,
        PRESET,
        primary_provider="openai",
        primary_model="primary",
        readiness=lambda _cfg, _provider: None,
    )
    assert [(c.provider, c.model) for c in targets] == list(PAIRS)
    assert targets[0].target.base_url == "https://api.openai.com/v1"
    assert dict(targets[0].target.params)["temperature"] == 0.1
    assert targets[1].target.base_url == "http://127.0.0.1:8888"
    assert dict(targets[1].target.params)["top_k"] == 7
    assert not targets[1].native


def test_unavailable_candidate_retains_visible_skip():
    candidates = resolve_preset_fallback_targets(
        APP,
        PRESET,
        primary_provider="openai",
        primary_model="primary",
        readiness=lambda _cfg, _provider: "not configured",
    )
    assert len(candidates) == 2
    assert all(not c.ready and c.skip_reason for c in candidates)


def _run(primary, *, cancel=lambda: False, select=None, budget=None):
    events = []
    candidate = FallbackCandidate("openai", True, True, model="alternate", index=1)

    def build(selected):
        events.append((selected.provider, selected.model))
        return lambda *args: ModelTurn(text="rescued")

    runtime = FallbackRuntime((candidate,), build, pre_tool_only=True, select=select)
    deps = LoopDeps(
        call_model=primary,
        invoke_tool=lambda call: ToolResult(ok=False, error="refused"),
        spawn=lambda task: ToolResult(ok=False),
        find_tools=lambda query: [],
        load_schemas=lambda *args: None,
        should_cancel=cancel,
        clock=lambda: 0,
        fallback=runtime,
        sleep=lambda _: None,
    )
    cfg = AgentConfig(
        model="primary",
        provider="openai",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=budget or RunBudget(max_model_retries=0),
    )
    return run_agent_loop(cfg, [{"role": "user", "content": "hi"}], [], deps), events


@pytest.mark.parametrize(
    "error",
    [
        ChatRateLimitError(),
        ChatProviderError(status_code=503),
        TimeoutError(),
        ChatModelUnavailableError(provider="openai"),
    ],
)
def test_explicit_retryable_pre_tool_failure_selects_alternate(error):
    def primary(*args):
        raise error

    outcome, events = _run(primary)
    assert outcome.final_text == "rescued"
    assert events == [("openai", "alternate")]


@pytest.mark.parametrize(
    "error",
    [
        ChatBadRequestError(status_code=400),
        ChatBadRequestError(status_code=404),
        ChatAuthenticationError(),
        ChatProviderError(status_code=403),
        ValueError("model unavailable"),
    ],
)
def test_other_errors_never_authorize_preset_fallback(error):
    def primary(*args):
        raise error

    with pytest.raises(type(error)):
        _run(primary)


def test_proposed_refused_tool_batch_closes_fallback():
    attempts = 0

    def primary(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ModelTurn(tool_calls=[ToolCall(name="calculator", args={})])
        raise ChatRateLimitError()

    with pytest.raises(ChatRateLimitError):
        _run(primary)


def test_active_target_is_frozen_without_overwriting_original_audit(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    try:
        targets = [
            {
                "provider": "openai",
                "model": "primary",
                "base_url": None,
                "params_json": "{}",
            },
            {
                "provider": "custom-ep:backup",
                "model": "alternate",
                "base_url": "http://127.0.0.1:8888",
                "params_json": '{"top_k": 7}',
            },
        ]
        key = db.create_run(
            conversation_id="c",
            agent_kind="subagent",
            resolved_provider="openai",
            resolved_model="primary",
            fallback_targets_json=json.dumps(targets),
        )
        db.set_run_active_fallback(key, 1)
        assert db.get_run_resolved_target(key) == targets[1]
        with db.connection() as conn:
            assert conn.execute(
                "SELECT resolved_provider, resolved_model FROM agent_runs WHERE id=?",
                (key,),
            ).fetchone()[:] == ("openai", "primary")
        with pytest.raises(ValueError):
            db.set_run_active_fallback(key, 2)
    finally:
        db.close()


@pytest.mark.allow_network
@pytest.mark.parametrize("status", [400, 404])
@pytest.mark.parametrize(
    "code, expected",
    [
        ("model_not_found", ChatModelUnavailableError),
        ("not_found", ChatBadRequestError),
        ("invalid_api_key", ChatBadRequestError),
    ],
)
def test_real_provider_http_machine_code_classification(code, expected, status):
    from Tests.LLM_Calls.test_hosted_chat import (
        _scripted_hosted_server,
        _transport_config,
    )
    from tldw_chatbook.LLM_Calls.hosted_chat import owned_json_post

    with _scripted_hosted_server(
        [
            {
                "status": status,
                "body": json.dumps(
                    {
                        "error": {
                            "code": code,
                            "message": "do not classify this model unavailable prose",
                        }
                    }
                ).encode(),
            }
        ]
    ) as (server, base_url):
        with pytest.raises(expected):
            owned_json_post(
                config=_transport_config(base_url),
                route="chat/completions",
                payload={"model": "primary", "messages": []},
                streaming=False,
            )
        assert len(server.requests) == 1


def test_service_fallback_persists_before_dispatch_and_owns_request_fields(
    tmp_path, monkeypatch
):
    from Tests.Agents.conftest import join_fleet_children
    from Tests.Agents.test_agent_service import FleetChat, fence
    from tldw_chatbook.Agents import agent_service
    from tldw_chatbook.Agents.agent_routing import AgentsRoutingConfig
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )

    monkeypatch.setattr(
        agent_service, "load_agents_routing_config", AgentsRoutingConfig
    )
    db = AgentRunsDB(tmp_path / "service.db", client_id="test")
    definition = dataclasses.replace(
        PRESET,
        provider="custom-ep:first",
        model="primary",
        params=(),
        fallback_models=(("custom-ep:second", "alternate"),),
    )
    cfg = {
        "custom_endpoints": {
            "first": {
                "display_name": "First",
                "family": "openai_compatible",
                "base_url": "http://127.0.0.1:8001",
                "params": {"temperature": 0.9, "seed": 17},
            },
            "second": {
                "display_name": "Second",
                "family": "llama_cpp",
                "base_url": "http://127.0.0.1:8002",
                "params": {"temperature": 0.1, "top_k": 7},
            },
        }
    }
    db.create_agent_definition(definition)
    calls = []
    observed_targets = []

    def observe_target(run_id, provider, model):
        observed_targets.append((run_id, provider, model))
        raise OSError("observer failed")

    def primary_unavailable():
        raise ChatModelUnavailableError(provider="custom-hosted")

    def assert_selected():
        child = next(
            row for row in db.list_runs("c") if row["agent_kind"] == "subagent"
        )
        state = db.get_run_fallback_state(child["id"])
        assert state["active_index"] == 1
        assert state["original_target"]["model"] == "primary"
        assert state["active_target"]["model"] == "alternate"
        assert observed_targets == [(child["id"], "custom-ep:second", "alternate")]
        handle = next(
            handle
            for handle in service._fleet.snapshot()
            if handle.run_id == child["id"]
        )
        assert (handle.resolved_provider, handle.resolved_model) == (
            "custom-ep:second",
            "alternate",
        )
        return "rescued"

    chat = FleetChat(
        [fence("spawn_subagent", {"task": "child", "agent": "worker"}), "done"],
        {"child": [primary_unavailable, assert_selected]},
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(
        db, registry, chat_call=chat, app_config=cfg, on_resolved_target=observe_target
    )
    config = AgentConfig(
        model="parent",
        provider="llama_cpp",
        system_prompt="s",
        allowed_tools=("calculator", "spawn_subagent"),
        budget=RunBudget(max_steps=20, max_subagents=1, max_model_retries=0),
    )
    try:
        _, outcome = service.run_turn(
            conversation_id="c",
            config=config,
            api_endpoint="llama_cpp",
            messages=[{"role": "user", "content": "go"}],
        )
        assert outcome.status == "done"
        join_fleet_children(service)
        child = next(
            row for row in db.list_runs("c") if row["agent_kind"] == "subagent"
        )
        assert not any(step["kind"] == "capture_failed" for step in child["steps"])
        calls = chat.child_calls["child"]
        assert calls[0]["api_endpoint"] == "custom-ep:first"
        assert calls[0]["model"] == "primary"
        assert calls[0]["api_base_url"] == "http://127.0.0.1:8001"
        assert "tools" in calls[0]
        assert calls[1]["api_endpoint"] == "custom-ep:second"
        assert calls[1]["model"] == "alternate"
        assert calls[1]["api_base_url"] == "http://127.0.0.1:8002"
        assert calls[1]["temp"] == 0.1 and calls[1]["topk"] == 7
        assert "seed" not in calls[1] and "tools" not in calls[1]
        assert (
            "Tool calls" in calls[1]["messages_payload"][0]["content"]
            or "tool_call" in calls[1]["messages_payload"][0]["content"]
        )
    finally:
        join_fleet_children(service)
        db.close()


@pytest.mark.allow_network
@pytest.mark.parametrize(
    "automatic_call_cap,same_provider", [(None, False), (None, True), (1, False)]
)
def test_real_adapter_fallback_uses_own_url_model_and_sampling(
    tmp_path, monkeypatch, automatic_call_cap, same_provider
):
    import asyncio

    from Tests.Chat.test_console_agent_bridge import (
        _routing_parent_resolution,
        _streaming_adapter,
    )
    from Tests.LLM_Calls.test_hosted_chat import _scripted_hosted_server
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    response = b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"rescued"},"finish_reason":null}]}\n\ndata: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    primary_scripts = [{"status": 404, "body": b'{"error":{"code":"model_not_found"}}'}]
    if same_provider:
        primary_scripts.append({"body": response, "content_type": "text/event-stream"})
    fallback_slug = "first" if same_provider else "second"
    with (
        _scripted_hosted_server(primary_scripts) as (first, first_url),
        _scripted_hosted_server(
            [{"body": response, "content_type": "text/event-stream"}]
        ) as (second, second_url),
    ):
        app = {
            "custom_endpoints": {
                "first": {
                    "display_name": "First",
                    "family": "openai_compatible",
                    "base_url": first_url,
                    "params": {"temperature": 0.9, "seed": 99},
                },
                "second": {
                    "display_name": "Second",
                    "family": "openai_compatible",
                    "base_url": second_url,
                    "params": {"temperature": 0.1},
                },
            }
        }
        definition = dataclasses.replace(
            PRESET, fallback_models=((f"custom-ep:{fallback_slug}", "alternate"),)
        )
        targets = resolve_preset_fallback_targets(
            app,
            definition,
            primary_provider="custom-ep:first",
            primary_model="primary",
            readiness=lambda *args: None,
        )
        for entry in app["custom_endpoints"].values():
            entry["api_key_env"] = "PRESET_TEST_KEY"
        gateway = ConsoleProviderGateway(
            config_provider=lambda: app, environ={"PRESET_TEST_KEY": "local-test-key"}
        )
        resolved_keys = []
        resolve = gateway.resolve_for_send

        async def observed_resolve(selection):
            resolved = await resolve(selection)
            resolved_keys.append(resolved.execution_key)
            return resolved

        monkeypatch.setattr(gateway, "resolve_for_send", observed_resolve)
        registry = ToolCatalogRegistry()
        registry.register_provider(BuiltinToolProvider())
        db = AgentRunsDB(tmp_path / "adapter.db", client_id="test")
        cfg = AgentConfig(
            model="primary",
            provider="custom-ep:first",
            execution_provider=targets[0].target.execution_provider,
            system_prompt="You are a sub-agent",
            allowed_tools=("calculator",),
            base_url=first_url,
            sampling_params=(("temperature", 0.9), ("seed", 99)),
            budget=RunBudget(max_model_retries=0),
        )
        from tldw_chatbook.Agents.agent_service import get_internal_prompt

        cfg = dataclasses.replace(
            cfg, system_prompt=get_internal_prompt("agents.subagent_system")
        )
        try:
            with _streaming_adapter(_routing_parent_resolution(), gateway) as adapter:
                service = AgentService(
                    db, registry, chat_call=adapter.chat_call, app_config=app
                )
                primary = service._make_call_model(
                    cfg,
                    cfg.provider,
                    [
                        registry.load_schema(entry.id)
                        for entry in registry.list_catalog()
                    ],
                )

                def build(candidate):
                    target = candidate.target
                    c = dataclasses.replace(
                        cfg,
                        provider=target.provider,
                        model=target.model,
                        execution_provider=target.execution_provider,
                        base_url=target.base_url,
                        sampling_params=target.params,
                        reasoning_replay=None,
                    )
                    return service._make_call_model(
                        c,
                        target.provider,
                        [
                            registry.load_schema(entry.id)
                            for entry in registry.list_catalog()
                        ],
                    )

                def edit_registry(candidate):
                    from tldw_chatbook.Chat.custom_endpoint_registry import entry_for

                    app["custom_endpoints"][fallback_slug].update(
                        family="ollama",
                        base_url="http://127.0.0.1:1",
                        params={"temperature": 0.5, "seed": 11},
                    )
                    assert (
                        entry_for(app, f"custom-ep:{fallback_slug}").family == "ollama"
                    )

                deps = LoopDeps(
                    call_model=primary,
                    invoke_tool=lambda call: ToolResult(ok=True),
                    spawn=lambda task: ToolResult(ok=False),
                    find_tools=lambda query: [],
                    load_schemas=lambda *args: None,
                    should_cancel=lambda: False,
                    clock=lambda: 0,
                    fallback=FallbackRuntime(
                        targets, build, pre_tool_only=True, select=edit_registry
                    ),
                )
                if automatic_call_cap is not None:
                    from Tests.Agents.test_automatic_child_scope import accepted_context
                    from tldw_chatbook.Agents.automatic_work_budget import (
                        AutomaticWorkRefused,
                    )

                    context = accepted_context(db, model_calls=automatic_call_cap)
                    with context.scope(), pytest.raises(AutomaticWorkRefused):
                        run_agent_loop(
                            cfg, [{"role": "user", "content": "go"}], [], deps
                        )
                    snapshot = db.automatic_work.snapshot(context.chain_id)
                    assert snapshot.used["model_call"] == 1
                    assert snapshot.available["model_call"] == 0
                    assert snapshot.reserved["tokens"] > 0
                    assert snapshot.status == "review_required"
                    assert len(first.requests) == 1 and second.requests == []
                    return
                outcome = run_agent_loop(
                    cfg, [{"role": "user", "content": "go"}], [], deps
                )
                assert outcome.final_text == "rescued"
            assert app["custom_endpoints"][fallback_slug]["family"] == "ollama"
            assert resolved_keys == ["custom-hosted", "custom-hosted"]
            assert len(first.requests) == (2 if same_provider else 1)
            assert len(second.requests) == (0 if same_provider else 1)
            before = json.loads(first.requests[0]["body"])
            after = json.loads(
                (first.requests[1] if same_provider else second.requests[0])["body"]
            )
            assert before["model"] == "primary" and before["temperature"] == 0.9
            assert after["model"] == "alternate"
            assert after["temperature"] == (0.9 if same_provider else 0.1)
            if same_provider:
                assert after["seed"] == 99
            else:
                assert "seed" not in after
            assert after.get("tools")
        finally:
            asyncio.run(gateway.aclose())
            db.close()


def test_selection_write_failure_stops_before_candidate_dispatch():
    def primary(*args):
        raise ChatRateLimitError()

    def fail_selection(candidate):
        raise OSError("selection persistence failed")

    with pytest.raises(OSError, match="selection persistence"):
        _run(primary, select=fail_selection)


def test_cancellation_during_primary_failure_stops_same_child():
    cancelled = False

    def primary(*args):
        nonlocal cancelled
        cancelled = True
        raise ChatRateLimitError()

    outcome, events = _run(primary, cancel=lambda: cancelled)
    assert outcome.status == "cancelled" and events == []


def test_fallback_attempt_spends_existing_model_budget():
    def primary(*args):
        raise ChatRateLimitError()

    outcome, _events = _run(
        primary, budget=RunBudget(max_model_turns=1, max_model_retries=0)
    )
    # The existing tools-free wrap-up may run once; exhaustion never becomes success.
    assert outcome.status == "stuck"
    assert any("model-turn budget exhausted" in step.summary for step in outcome.steps)


@pytest.mark.asyncio
async def test_settings_fallback_editor_round_trip_and_invalid_authoring(tmp_path):
    from textual.widgets import ListView, Static, TextArea

    from Tests.Widgets.test_settings_agents_panel_routing import (
        PanelHarness,
        _fill_valid_preset_form,
        _make_panel,
    )

    db = AgentRunsDB(tmp_path / "settings.db", client_id="test")
    try:
        panel = _make_panel(db)
        async with PanelHarness(panel).run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            _fill_valid_preset_form(panel)
            panel.query_one(
                "#agents-fallback-models-area", TextArea
            ).text = "openai/alternate\ncustom-ep:qwen-local/local-model"
            await panel._save()
            assert definition_from_row(
                db.list_agent_definitions()[0]
            ).fallback_models == (
                ("openai", "alternate"),
                ("custom-ep:qwen-local", "local-model"),
            )
            panel._clear_form()
            item = panel.query_one("#agents-definition-list", ListView).children[0]
            panel.on_list_view_selected(
                ListView.Selected(
                    panel.query_one("#agents-definition-list", ListView), item, 0
                )
            )
            assert (
                panel.query_one("#agents-fallback-models-area", TextArea).text
                == "openai/alternate\ncustom-ep:qwen-local/local-model"
            )
            panel.query_one(
                "#agents-fallback-models-area", TextArea
            ).text = "https://attacker/model"
            await panel._save()
            assert "known provider" in str(
                panel.query_one("#agents-status", Static).render()
            )
            assert definition_from_row(db.list_agent_definitions()[0]).fallback_models[
                0
            ] == ("openai", "alternate")
    finally:
        db.close()


def test_retained_child_uses_active_frozen_target_without_reopening_chain(
    tmp_path, monkeypatch
):
    import time

    from Tests.Agents.conftest import join_fleet_children
    from Tests.Agents.test_agent_service import FleetChat, fence
    from tldw_chatbook.Agents import agent_service
    from tldw_chatbook.Agents.agent_routing import AgentsRoutingConfig
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    monkeypatch.setattr(
        agent_service, "load_agents_routing_config", AgentsRoutingConfig
    )
    db = AgentRunsDB(tmp_path / "continuation.db", client_id="test")
    definition = AgentDefinition(
        name="worker",
        instructions="Work.",
        provider="llama_cpp",
        model="primary",
        fallback_models=(("custom-ep:backup", "alternate"),),
    )
    key = db.create_agent_definition(definition)
    cfg = {
        "custom_endpoints": {
            "backup": {
                "display_name": "Backup",
                "family": "llama_cpp",
                "base_url": "http://127.0.0.1:8002",
                "params": {"top_k": 7},
            }
        }
    }
    fleet = FleetCoordinator(max_live=3, clock=time.monotonic)
    finished_targets = []
    finish = fleet.finish

    def observe_finish(handle_id, *args, **kwargs):
        handle = next(
            handle for handle in fleet.snapshot() if handle.handle_id == handle_id
        )
        finished_targets.append((handle.resolved_provider, handle.resolved_model))
        return finish(handle_id, *args, **kwargs)

    monkeypatch.setattr(fleet, "finish", observe_finish)

    def fail():
        raise ChatModelUnavailableError(provider="llama_cpp")

    def resume():
        db.update_agent_definition(
            key,
            dataclasses.replace(
                definition, model="edited", fallback_models=(("openai", "edited"),)
            ),
        )
        cfg["custom_endpoints"]["backup"]["base_url"] = "http://127.0.0.1:9999"
        return fence(
            "send_to_agent",
            {"id": fleet.snapshot()[0].handle_id, "message": "continue"},
        )

    def fail_again():
        raise ChatRateLimitError()

    chat = FleetChat(
        [
            fence("spawn_subagent", {"task": "child", "agent": "worker"}),
            fence("wait_agents", {}),
            resume,
            fence("wait_agents", {}),
            "done",
        ],
        {"child": [fail, "rescued", fail_again]},
    )
    service = AgentService(
        db,
        ToolCatalogRegistry(),
        chat_call=chat,
        fleet_coordinator=fleet,
        app_config=cfg,
    )
    config = AgentConfig(
        model="parent",
        provider="llama_cpp",
        base_url="http://127.0.0.1:8001",
        system_prompt="s",
        allowed_tools=("spawn_subagent",),
        budget=RunBudget(
            max_steps=40, max_model_turns=40, max_subagents=2, max_model_retries=0
        ),
    )
    try:
        _, outcome = service.run_turn(
            conversation_id="c",
            config=config,
            api_endpoint="llama_cpp",
            messages=[{"role": "user", "content": "go"}],
        )
        join_fleet_children(service)
        assert outcome.status == "done"
        calls = chat.child_calls["child"]
        assert len(calls) == 3
        assert calls[0]["api_base_url"] == "http://127.0.0.1:8001"
        assert calls[2]["api_endpoint"] == "custom-ep:backup"
        assert calls[2]["model"] == "alternate"
        assert calls[2]["api_base_url"] == "http://127.0.0.1:8002"
        assert calls[2]["topk"] == 7
        rows = [row for row in db.list_runs("c") if row["agent_kind"] == "subagent"]
        assert len(rows) == 2
        assert all(
            db.get_run_fallback_state(row["id"])["active_index"] == 1 for row in rows
        )
        assert any(row["status"] == "error" for row in rows)
        assert finished_targets == [("custom-ep:backup", "alternate")] * 2
        assert all(
            (row["resolved_provider"], row["resolved_model"])
            == ("llama_cpp", "primary")
            for row in rows
        )
    finally:
        join_fleet_children(service)
        db.close()


def test_broken_readiness_probe_is_a_visible_frozen_skip():
    def broken(*args):
        raise RuntimeError("do not expose probe body")

    candidates = resolve_preset_fallback_targets(
        APP,
        PRESET,
        primary_provider="openai",
        primary_model="primary",
        readiness=broken,
    )
    assert all(not candidate.ready for candidate in candidates)
    assert all(
        candidate.skip_reason == "readiness check failed (RuntimeError)"
        for candidate in candidates
    )
    assert candidates[1].target.base_url == "http://127.0.0.1:8888"


def test_v21_migration_keeps_saved_targets_and_fallback_disabled(tmp_path):
    import sqlite3
    from threading import Event

    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.recovery_operations import (
        _AGENT_RUNS_SCHEMA,
        recovery_adapters,
    )

    path = tmp_path / "v21.db"
    schema = next(catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 21)
    with sqlite3.connect(path) as connection:
        for sql in sorted(schema, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                connection.execute(sql)
        connection.execute("INSERT INTO schema_version VALUES (21)")
        connection.execute(
            "INSERT INTO agent_definitions(id,name,provider,model,created_at,updated_at) VALUES ('p','Kept','openai','primary','then','then')"
        )
        connection.execute(
            "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,created_at,updated_at,resolved_provider,resolved_model) VALUES ('r','c','subagent','done','then','then','openai','primary')"
        )
    path.chmod(0o600)
    owner = next(
        owner for owner in recovery_adapters() if owner.owner_id == "db.agent_runs"
    )
    assert validate_candidate(owner, path, Event(), migrate=True) == ()
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone() == (AgentRunsDB._CURRENT_SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT provider,model,fallback_models_json FROM agent_definitions"
        ).fetchone() == ("openai", "primary", "[]")
        assert connection.execute(
            "SELECT resolved_provider,resolved_model,fallback_targets_json,active_fallback_index FROM agent_runs"
        ).fetchone() == ("openai", "primary", None, 0)


@pytest.mark.asyncio
async def test_deleted_frozen_registry_target_refuses_before_family_credentials_or_probe():
    import httpx

    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderSelection,
    )

    app = {
        "custom_endpoints": {
            "backup": {
                "display_name": "Backup",
                "family": "llama_cpp",
                "base_url": "http://127.0.0.1:8002",
            }
        },
        "api_settings": {"llama_cpp": {"api_key": "family-canary"}},
    }
    candidate = resolve_preset_fallback_targets(
        app,
        dataclasses.replace(
            PRESET, fallback_models=(("custom-ep:backup", "alternate"),)
        ),
        primary_provider="openai",
        primary_model="primary",
        readiness=lambda *_args: None,
    )[0]
    del app["custom_endpoints"]["backup"]
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "ok", "data": [{"id": "alternate"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = ConsoleProviderGateway(
            http_client=client, config_provider=lambda: app, environ={}
        )
        target = candidate.target
        resolved = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider=target.provider,
                explicit_model=target.model,
                base_url=target.base_url,
                base_url_is_pinned=True,
                execution_provider=target.execution_provider,
            )
        )
    assert requests == []
    assert not resolved.ready
    assert resolved.api_key is None
    assert "no longer configured" in resolved.visible_copy


@pytest.mark.allow_network
def test_real_builtin_adapter_freezes_configured_url_for_fallback_and_continuation(
    tmp_path,
):
    import asyncio

    from Tests.Chat.test_console_agent_bridge import (
        _routing_parent_resolution,
        _streaming_adapter,
    )
    from Tests.LLM_Calls.test_hosted_chat import _scripted_hosted_server
    from tldw_chatbook.Agents.agent_service import AgentService, get_internal_prompt
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    def script(text):
        body = (
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"'
            + text
            + '"},"finish_reason":null}]}\n\n'
            + 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            + "data: [DONE]\n\n"
        ).encode()
        return {"body": body, "content_type": "text/event-stream"}

    with (
        _scripted_hosted_server([script("frozen"), script("frozen")]) as (
            original,
            original_url,
        ),
        _scripted_hosted_server([script("changed"), script("changed")]) as (
            edited,
            edited_url,
        ),
    ):
        app = {"api_settings": {"custom": {"api_url": original_url}}}
        candidates = resolve_preset_fallback_targets(
            app,
            dataclasses.replace(PRESET, fallback_models=(("custom", "alternate"),)),
            primary_provider="openai",
            primary_model="primary",
            readiness=lambda *_args: None,
        )
        target = candidates[0].target
        snapshot = {
            "provider": target.provider,
            "model": target.model,
            "base_url": target.base_url,
            "params_json": json.dumps(dict(target.params)),
            "execution_provider": target.execution_provider,
        }
        db = AgentRunsDB(tmp_path / "builtin.db", client_id="test")
        key = db.create_run(
            conversation_id="c",
            agent_kind="subagent",
            resolved_provider="openai",
            resolved_model="primary",
            fallback_targets_json=json.dumps(
                [
                    {
                        "provider": "openai",
                        "model": "primary",
                        "base_url": None,
                        "params_json": None,
                    },
                    snapshot,
                ]
            ),
        )
        app["api_settings"]["custom"]["api_url"] = edited_url
        gateway = ConsoleProviderGateway(config_provider=lambda: app, environ={})
        config = AgentConfig(
            model="primary",
            provider="openai",
            system_prompt=get_internal_prompt("agents.subagent_system"),
            budget=RunBudget(max_model_retries=0),
        )
        try:
            with _streaming_adapter(_routing_parent_resolution(), gateway) as adapter:
                service = AgentService(
                    db,
                    ToolCatalogRegistry(),
                    chat_call=adapter.chat_call,
                    app_config=app,
                )

                def build(candidate):
                    active = candidate.target
                    child = dataclasses.replace(
                        config,
                        provider=active.provider,
                        model=active.model,
                        base_url=active.base_url,
                        execution_provider=active.execution_provider,
                        sampling_params=active.params,
                    )
                    return service._make_call_model(child, child.provider, [])

                def fail(*_args):
                    raise ChatRateLimitError()

                deps = LoopDeps(
                    call_model=fail,
                    invoke_tool=lambda _call: ToolResult(ok=False),
                    spawn=lambda _task: ToolResult(ok=False),
                    find_tools=lambda _q: [],
                    load_schemas=lambda *_args: None,
                    should_cancel=lambda: False,
                    clock=lambda: 0,
                    fallback=FallbackRuntime(
                        candidates,
                        build,
                        pre_tool_only=True,
                        select=lambda candidate: db.set_run_active_fallback(
                            key, candidate.index
                        ),
                    ),
                )
                outcome = run_agent_loop(
                    config, [{"role": "user", "content": "go"}], [], deps
                )
                assert outcome.final_text == "frozen"
                saved = db.get_run_fallback_state(key)["active_target"]
                resumed = dataclasses.replace(
                    config,
                    provider=saved["provider"],
                    model=saved["model"],
                    base_url=saved["base_url"],
                    execution_provider=saved["execution_provider"],
                    sampling_params=tuple(json.loads(saved["params_json"]).items()),
                )
                turn = service._make_call_model(resumed, resumed.provider, [])(
                    [{"role": "user", "content": "continue"}], [], None
                )
                assert turn.text == "frozen"
            assert len(original.requests) == 2
            assert edited.requests == []
            assert all(
                json.loads(request["body"])["model"] == "alternate"
                for request in original.requests
            )
            assert target.base_url.startswith(original_url)
        finally:
            asyncio.run(gateway.aclose())
            db.close()


@pytest.mark.parametrize(
    "provider,settings,expected",
    [
        ("openai", {}, "https://api.openai.com/v1"),
        (
            "openai",
            {"api_base": "http://127.0.0.1:8000/v1"},
            "http://127.0.0.1:8000/v1",
        ),
        ("together", {}, "https://api.together.xyz/v1"),
        ("qwencloud", {}, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        (
            "qwencloud",
            {"api_base_url": "http://127.0.0.1:8000/v1/chat/completions"},
            "http://127.0.0.1:8000/v1",
        ),
        ("llama_cpp", {}, "http://127.0.0.1:9099"),
    ],
)
def test_builtin_candidate_snapshot_uses_current_endpoint_defaults(
    provider, settings, expected
):
    candidates = resolve_preset_fallback_targets(
        {"api_settings": {provider: settings}},
        dataclasses.replace(PRESET, fallback_models=((provider, "alternate"),)),
        primary_provider="openai",
        primary_model="primary",
        readiness=lambda *_args: None,
    )
    assert candidates[0].ready
    assert candidates[0].target.base_url == expected
