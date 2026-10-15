"""Native component semantics through the owned plugin runtime."""


def test_empty_agent_tool_list_never_inherits():
    from tldw_chatbook.Plugins.agent_presets import resolve_agent_tools

    eligible = frozenset({"mcp:repository:read"})
    assert resolve_agent_tools((), eligible) == frozenset()
    assert resolve_agent_tools("inherit", eligible) == eligible


def test_native_inventory_positive_control(native_package):
    from tldw_chatbook.Plugins.inspection import inspect_package

    inspection = inspect_package(native_package())
    assert not inspection.rejected
    assert inspection.inventory["skill:review"].kind == "skill"


import pytest

from Tests.Plugins.test_owned_mcp_tools import (
    shared_connection_case as _shared_connection_case,
)

shared_connection_case = _shared_connection_case


async def install_components(service, package, selection):
    review = await service.review_install(
        package, selection=selection, workspace_id="workspace-a"
    )
    await service.commit(review, review.operation_id)
    trust = await service.review_trust(review.installation_id)
    await service.commit(trust, trust.operation_id)
    enable = await service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await service.commit(enable, enable.operation_id)
    return review


@pytest.mark.asyncio
async def test_native_components_share_snapshot_and_keep_unrelated_ready(
    native_console, native_package
):
    package = native_package(
        extension={
            "version": 1,
            "commands": ["command.md"],
            "rules": ["rule.md"],
            "agents": ["agent.md"],
            "requires": {"command:review": ["hook:missing"]},
        }
    )
    (package / "command.md").write_text(
        "---\nname: review\ndescription: Review\narguments: [query]\n---\nReview carefully."
    )
    (package / "rule.md").write_text("---\nname: style\nmode: always\n---\nBe precise.")
    (package / "agent.md").write_text(
        "---\nname: reviewer\ndescription: Review\ntools: []\n---\nInspect only."
    )
    rig = native_console
    review = await install_components(
        rig.service,
        package,
        ("skill:review", "command:review", "rule:style", "agent:reviewer"),
    )
    snapshot = await rig.service.capture_mcp_snapshot(
        review.installation_id, "workspace-a", "pending:native"
    )
    assert set(snapshot.selection) == {"skill:review", "rule:style", "agent:reviewer"}
    maximum = rig.service.capture_maximum("workspace-a")
    assert {row["plugin_component_id"] for row in maximum["native_components"]} == {
        "rule:style",
        "agent:reviewer",
    }
    assert "Be precise" not in repr(maximum)


@pytest.mark.parametrize(
    "restriction", ["empty", "inherit", "canonical", "unavailable", "display"]
)
def test_native_agent_projection_keeps_empty_tools_and_user_authority(
    native_package, restriction
):
    from tldw_chatbook.Agents.agent_models import PluginContextText
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )
    from tldw_chatbook.Plugins.admission import RunPluginSnapshot
    from tldw_chatbook.Plugins.agent_presets import project_agent
    from tldw_chatbook.Plugins.components import PluginComponents
    from tldw_chatbook.Plugins.inspection import inspect_package

    package = native_package(extension={"version": 1, "agents": ["agent.md"]})
    (package / "agent.md").write_text(
        "---\nname: reviewer\ndescription: Review\ntools: []\n---\nInspect only."
    )
    inspection = inspect_package(package).model_copy(
        update={"materialized_identity": str(package)}
    )
    snapshot = RunPluginSnapshot(
        "installation",
        inspection.effective_digest,
        None,
        "r",
        ("agent:reviewer",),
        "[]",
        (),
        (("agent:reviewer", ()),),
        inspection,
        "reviewer",
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    eligible = frozenset(entry.name for entry in registry.list_catalog())
    assert "calculator" in eligible
    import json

    component = PluginComponents.project(snapshot)[0]
    canonical = next(
        entry.id for entry in registry.list_catalog() if entry.name == "calculator"
    )
    raw = json.loads(component.definition_json)
    raw["tools"] = {
        "empty": [],
        "inherit": "inherit",
        "canonical": [canonical],
        "unavailable": [canonical],
        "display": ["calculator"],
    }[restriction]
    component = component.model_copy(update={"definition_json": json.dumps(raw)})
    if restriction in {"unavailable", "display"}:
        with pytest.raises(ValueError, match="unavailable"):
            project_agent(
                component,
                registry,
                frozenset() if restriction == "unavailable" else eligible,
                "llama_cpp",
            )
        return
    definition = project_agent(component, registry, eligible, "llama_cpp")
    assert definition.instructions == ""
    assert isinstance(definition.plugin_context, PluginContextText)
    assert "Inspect only" in definition.plugin_context
    assert frozenset(definition.tool_allowlist) == (
        {
            "empty": frozenset(),
            "inherit": eligible,
            "canonical": frozenset({"calculator"}),
        }[restriction]
    )
    assert definition.inherit_tools is False


def test_model_mapping_recovers_only_exact_current_owner(plugin_stack, native_package):
    from Tests.Plugins.test_recovery import lose_registry
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.Plugins.inspection import inspect_package

    stack = plugin_stack
    settings = {
        "api_settings": {
            "llama_cpp": {"api_url": "http://127.0.0.1:8080/v1/chat/completions"}
        }
    }
    models = {"llama_cpp": ["m"]}
    owner = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: models,
        settings_loader=lambda: settings,
        environ={},
    )
    stack.call(lambda: setattr(stack.coordinator, "model_mapping_owner", owner))
    package = native_package(extension={"version": 1, "agents": ["agent.md"]})
    (package / "agent.md").write_text(
        "---\nname: analyst\ndescription: Analyze\ntools: inherit\nmodel: preferred\n---\nInspect carefully."
    )
    review = stack.call(
        lambda: stack.coordinator.review(
            inspect_package(package), selection=("agent:analyst",), workspace_id=None
        )
    )
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    with pytest.raises(ValueError, match="canonical"):
        stack.call(
            lambda: stack.coordinator.review_configuration(
                review.installation_id, models={"agent:analyst": "m"}
            )
        )
    mapped = stack.call(
        lambda: stack.coordinator.review_configuration(
            review.installation_id, models={"agent:analyst": "llama_cpp/m"}
        )
    )
    stack.call(lambda: stack.coordinator.commit(mapped, mapped.operation_id))
    expected = stack.call(stack.coordinator.published_snapshot)["mappings"]
    assert expected[0]["kind"] == "model"
    lose_registry(stack)
    stack.call(lambda: setattr(stack.coordinator, "model_mapping_owner", owner))
    stack.call(stack.coordinator.recover)
    assert stack.call(stack.coordinator.published_snapshot)["mappings"] == expected
    models["llama_cpp"] = ["different"]
    lose_registry(stack)
    stack.call(lambda: setattr(stack.coordinator, "model_mapping_owner", owner))
    outcomes = stack.call(stack.coordinator.recover)
    assert any(row.phase == "recovery_required" for row in outcomes)
    with pytest.raises(PermissionError):
        stack.call(stack.coordinator.published_snapshot)


@pytest.mark.asyncio
async def test_owned_command_revoke_before_publication_preserves_other_scope(
    native_console, native_package, monkeypatch
):
    import asyncio
    import json
    import threading

    from Tests.Agents.test_hooks_v2_execution import event
    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Plugins.components import NativeHookProcessOwner, NativeHookSet

    rig = native_console
    package = native_package(extension={"version": 1, "hooks": "hooks.json"})
    code = "import json,sys,time;from pathlib import Path;e=json.load(sys.stdin);root=Path('${PLUGIN_DATA}');(root/e['runtime_session_id']).write_text('running');exec(\"while not (root/'release').exists(): time.sleep(.01)\");print('{\"version\":2,\"decision\":\"pass\"}')"
    (package / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "work",
                        "type": "command",
                        "event": "PreToolUse",
                        "effects": ["deny"],
                        "argv": child_argv(code),
                        "timeout_seconds": 20,
                    }
                ],
            }
        )
    )
    install = await install_components(rig.service, package, ("hook:work",))
    enable = await rig.service.review_activation(
        install.installation_id, workspace_id="workspace-b", intent="enabled"
    )
    await rig.service.commit(enable, enable.operation_id)
    review = await rig.service.review_data_creation(
        install.installation_id, workspace_id=None
    )
    root = await rig.service.create_data(review, review.operation_id)
    owners, engines = {}, {}
    for scope in ("a", "b"):
        snapshot = await rig.service.capture_mcp_snapshot(
            install.installation_id, "workspace-" + scope, "pending-" + scope
        )
        owners[scope] = NativeHookSet(rig.service, (snapshot,))
        engines[scope] = HookEngine(
            owners[scope].definitions,
            owners[scope].current,
            HookBudgetOwner(),
            process_owner_for=owners[scope].process_owner,
        )
    entered, release_publish = threading.Event(), threading.Event()
    original = NativeHookProcessOwner.publish_process

    def delayed(self, token, provenance):
        if self.snapshot.workspace_id == "workspace-a":
            entered.set()
            assert release_publish.wait(10)
        return original(self, token, provenance)

    monkeypatch.setattr(NativeHookProcessOwner, "publish_process", delayed)
    a = asyncio.create_task(engines["a"].fire_async(event(runtime="a")))
    b = asyncio.create_task(engines["b"].fire_async(event(runtime="b")))
    try:
        for _ in range(500):
            if entered.is_set() and (root.path / "b").exists():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set() and (root.path / "b").exists()
        records = rig.service.live_runs()
        assert {row.workspace_id for row in records} == {"workspace-a", "workspace-b"}
        assert await rig.service._call(
            lambda: rig.service._coordinator.root_usage.dirty
        )
        disable = await rig.service.review_activation(
            install.installation_id, workspace_id="workspace-a", intent="disabled"
        )
        pending = asyncio.create_task(rig.service.commit(disable, disable.operation_id))
        for _ in range(100):
            if any(
                job.cancel.is_set() for job in engines["a"].processes.records.values()
            ):
                break
            await asyncio.sleep(0.01)
        assert any(
            job.cancel.is_set() for job in engines["a"].processes.records.values()
        )
        assert not b.done()
        assert all(
            not job.cancel.is_set() for job in engines["b"].processes.records.values()
        )
        assert not (
            root.path / "a"
        ).exists(), "payload cannot reach the child before publication"
        release_publish.set()
        assert not (await asyncio.wait_for(a, 10)).succeeded
        await asyncio.wait_for(pending, 10)
        assert {row.workspace_id for row in rig.service.live_runs()} == {"workspace-b"}
        (root.path / "release").touch()
        assert (await asyncio.wait_for(b, 10)).succeeded
        assert not rig.service.live_runs()
        assert not engines["a"].processes.records and not engines["b"].processes.records
    finally:
        release_publish.set()
        (root.path / "release").touch()
        await asyncio.gather(*(engine.close() for engine in engines.values()))
        await asyncio.gather(a, b, return_exceptions=True)


@pytest.mark.asyncio
async def test_native_agent_empty_tools_survives_real_spawn_and_resume(
    native_console, native_package, tmp_path, monkeypatch
):
    import asyncio

    from Tests.Agents.conftest import pin_agent_settings
    from Tests.Agents.test_agent_service import fence
    from Tests.Agents.test_fleet_continuation import (
        RESUME_CFG,
        _await_retained,
        _finished_child,
    )
    from Tests.Agents.test_fleet_runtime import make_fleet_service
    from tldw_chatbook.Agents.agent_models import (
        SEND_TO_AGENT_TOOL_NAME,
        SPAWN_TOOL_NAME,
        WAIT_AGENTS_TOOL_NAME,
    )
    from tldw_chatbook.Agents.agent_service import build_first_request_schema_plan
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Plugins.agent_presets import project_agent
    from tldw_chatbook.Plugins.components import PluginComponents

    pin_agent_settings(monkeypatch, max_live_subagents=3)
    rig = native_console
    package = native_package(extension={"version": 1, "agents": ["agent.md"]})
    (package / "agent.md").write_text(
        "---\nname: reviewer\ndescription: Review\ntools: []\n---\nOnly the attributed native instruction."
    )
    install = await install_components(rig.service, package, ("agent:reviewer",))
    snapshot = await rig.service.capture_mcp_snapshot(
        install.installation_id, "workspace-a", "native-agent-test"
    )
    db = AgentRunsDB(tmp_path / "native-runs.sqlite", client_id="native-agent")
    holder = {}
    service, chat, fleet = make_fleet_service(db, [], {})
    definition = project_agent(
        PluginComponents.project(snapshot)[0],
        service.registry,
        frozenset(RESUME_CFG.allowed_tools),
        "llama_cpp",
    )
    task_key = str(definition.plugin_context)
    # Use the committed fleet transport double and the real retained-child path.
    service, chat, fleet = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "Inspect it.", "agent": definition.name}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "first answer",
            lambda: fence(
                SEND_TO_AGENT_TOOL_NAME, {"id": holder["id"], "message": "Continue."}
            ),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "second answer",
        ],
        {
            task_key: [
                fence("calculator", {"expression": "1+1"}),
                "child answer",
                "resumed answer",
            ]
        },
    )

    def run():
        plan = build_first_request_schema_plan(
            service.registry,
            RESUME_CFG.allowed_tools,
            RESUME_CFG,
            "llama_cpp",
            [{"role": "user", "content": "go"}],
            skill_file_enabled=False,
            install_skill_enabled=False,
            run_skill_script_enabled=False,
            run_log_active=False,
            agent_definitions=(definition,),
            fleet_active=True,
            fleet_max_live=3,
        )
        return service.run_turn(
            conversation_id="native-conversation",
            messages=[{"role": "user", "content": "go"}],
            config=RESUME_CFG,
            api_endpoint="llama_cpp",
            first_request_schema_plan=plan,
        )

    try:
        _, first = await asyncio.to_thread(run)
        assert first.status == "done"
        child = _finished_child(fleet)
        holder["id"] = child.handle_id
        await asyncio.to_thread(_await_retained, fleet, child.handle_id)
        _, second = await asyncio.to_thread(run)
        assert second.status == "done"
        calls = chat.child_calls[task_key]
        assert len(calls) == 3
        assert "not permitted" in str(
            calls[1]["messages_payload"]
        ) or "not allowed" in str(calls[1]["messages_payload"])
        for call in calls:
            rows = call["messages_payload"]
            assert "Only the attributed native instruction." not in rows[0]["content"]
            assert (
                sum(
                    "Only the attributed native instruction." in str(row.get("content"))
                    for row in rows
                    if row["role"] == "user"
                )
                == 1
            )
            assert all(
                tool.get("function", {}).get("name") != "calculator"
                for tool in call.get("tools", ())
            )
        children = [
            row
            for row in db.list_runs("native-conversation", include_superseded=True)
            if row["agent_kind"] == "subagent"
        ]
        assert len(children) == 2 and any(
            row["resumed_from_run_id"] for row in children
        )
        assert all(row["definition_fingerprint"] for row in children)
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_connection_case", ["native_tools"], indirect=True)
async def test_exact_component_tool_mapping_uses_parent_rights_and_recovers(
    shared_connection_case,
):
    import json

    from tldw_chatbook.Plugins.agent_presets import project_agent
    from tldw_chatbook.Plugins.components import PluginComponents

    case = shared_connection_case
    kwargs = {
        "connections": {"mcp:shared": case.profile_id},
        "tools": {"mcp:shared": ("echo",)},
        "component_tools": {"agent:reader": {"search": "mcp:shared::echo"}},
    }
    review = await case.service.review_configuration(case.installation, **kwargs)
    assert (await case.service.commit(review, review.operation_id)).committed
    snapshot = await case.service.capture_mcp_snapshot(
        case.installation, "a", "mapped-agent"
    )
    component = next(
        row for row in PluginComponents.project(snapshot) if row.kind == "agent"
    )
    eligible = frozenset(row.name for row in case.registries["a"].list_catalog())
    definition = project_agent(component, case.registries["a"], eligible, "llama_cpp")
    assert frozenset(definition.tool_allowlist) == eligible
    with pytest.raises(ValueError, match="unavailable"):
        project_agent(component, case.registries["a"], frozenset(), "llama_cpp")
    receipts = await case.service._call(case.service._coordinator.recover)
    assert not any(row.phase == "recovery_required" for row in receipts)
    pending_review = await case.service.review_configuration(
        case.installation, **kwargs
    )
    original = case.local.store.get_discovery_snapshot(case.profile_id)
    changed = json.loads(json.dumps(original))
    changed["tools"][0]["description"] = "changed constraint target"
    case.local.store.save_discovery_snapshot(case.profile_id, changed)
    try:
        with pytest.raises((ValueError, PermissionError)):
            await case.service.commit(pending_review, pending_review.operation_id)
        receipts = await case.service._call(case.service._coordinator.recover)
        assert any(row.phase == "recovery_required" for row in receipts)
    finally:
        case.local.store.save_discovery_snapshot(case.profile_id, original)
        await case.service._call(case.service._coordinator.recover)


def test_standalone_agent_shape_and_fingerprint_remain_unchanged():
    import dataclasses
    import hashlib
    import json

    from tldw_chatbook.Agents.agent_models import (
        AgentDefinition,
        definition_fingerprint,
    )

    definition = AgentDefinition("reader", "Read", "Read carefully")
    assert dataclasses.asdict(definition) == {
        "name": "reader",
        "description": "Read",
        "instructions": "Read carefully",
        "tool_allowlist": (),
        "model": "",
        "enabled": True,
    }
    original = json.dumps(
        {"instructions": "Read carefully", "tool_allowlist": [], "model": ""},
        sort_keys=True,
    )
    assert (
        definition_fingerprint(definition)
        == hashlib.sha256(original.encode()).hexdigest()[:16]
    )


@pytest.mark.asyncio
async def test_owned_hook_registration_failure_settles_unlaunched_reservation(
    native_console, native_package, monkeypatch
):
    import json

    from Tests.Agents.test_hooks_v2_execution import event
    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Plugins.components import NativeHookSet

    rig = native_console
    package = native_package(extension={"version": 1, "hooks": "hooks.json"})
    (package / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "work",
                        "type": "command",
                        "event": "PreToolUse",
                        "effects": ["deny"],
                        "argv": child_argv(
                            'print(\'{"version":2,"decision":"pass"}\')'
                        ),
                    }
                ],
            }
        )
    )
    install = await install_components(rig.service, package, ("hook:work",))
    snapshot = await rig.service.capture_mcp_snapshot(
        install.installation_id, "workspace-a", "registration-failure"
    )
    owner = NativeHookSet(rig.service, (snapshot,))
    engine = HookEngine(
        owner.definitions,
        owner.current,
        HookBudgetOwner(),
        process_owner_for=owner.process_owner,
    )
    usage = rig.service._coordinator.root_usage

    def refuse(*_args):
        raise PermissionError("registration failed")

    monkeypatch.setattr(usage, "retain_cancel", refuse)
    try:
        outcome = await engine.fire_async(event())
        assert not outcome.succeeded
        assert not rig.service.live_runs() and not engine.processes.records
        assert not await rig.service._call(lambda: usage.pending)
        assert all(
            row["state"] == "settled"
            for row in await rig.service._call(
                lambda: rig.service._coordinator.owner.list_processes(
                    limit=10, offset=0
                )
            )
        )
    finally:
        await engine.close()
