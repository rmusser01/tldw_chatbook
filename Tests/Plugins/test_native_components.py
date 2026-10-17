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
            host_environment=owners[scope].host_environment,
            command_launch=owners[scope].command_launch,
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
        host_environment=owner.host_environment,
        command_launch=owner.command_launch,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_connection_case", ["native_tools"], indirect=True)
async def test_configuration_recovery_scopes_identical_tool_ids_to_installation_revision(
    shared_connection_case, native_package
):
    import asyncio
    import json
    import shutil

    from tldw_chatbook.MCP.credential_bindings import CredentialError
    from tldw_chatbook.Plugins.admission import PluginUnavailable
    from tldw_chatbook.Plugins.components import validate_mapping
    from tldw_chatbook.Plugins.mcp_provider import PluginMCPProvider
    from tldw_chatbook.Plugins.recovery import retained_inspections

    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok

    async def configure(installation, profile_id):
        review = await case.service.review_configuration(
            installation,
            connections={"mcp:shared": profile_id},
            tools={"mcp:shared": ("echo",)},
            component_tools={"agent:reader": {"search": "mcp:shared::echo"}},
        )
        assert (await case.service.commit(review, review.operation_id)).committed

    await configure(case.installation, case.profile_id)
    assert not any(
        row.phase == "recovery_required"
        for row in await case.service._call(case.service._coordinator.recover)
    )
    sibling = native_package()
    shutil.copytree(case.package, sibling, dirs_exist_ok=True)
    manifest = json.loads((sibling / "plugin.json").read_text())
    manifest["name"] = "sibling-review-helper"
    (sibling / "plugin.json").write_text(json.dumps(manifest))
    review = await case.service.review_install(
        sibling,
        selection=("skill:review", "mcp:shared", "agent:reader"),
        workspace_id="a",
    )
    assert (await case.service.commit(review, review.operation_id)).committed
    second = review.installation_id
    trust = await case.service.review_trust(second)
    assert (await case.service.commit(trust, trust.operation_id)).committed
    active = await case.service.review_activation(
        second, workspace_id="a", intent="enabled"
    )
    assert (await case.service.commit(active, active.operation_id)).committed
    data_review = await case.service.review_data_creation(second, workspace_id=None)
    data_root = await case.service.create_data(data_review, data_review.operation_id)
    snapshot = await case.service.capture_mcp_snapshot(second, "a", "sibling-discovery")
    profile = case.local.save_owned_profile(
        installation_id=second,
        inspection=snapshot.inspection,
        component_id="mcp:shared",
        data_root=data_root,
        session_isolation="request_independent",
        protocol_version="2025-03-26",
    )
    connection = await case.service.review_configuration(
        second, connections={"mcp:shared": profile["profile_id"]}
    )
    assert (await case.service.commit(connection, connection.operation_id)).committed
    provider = PluginMCPProvider(
        plugin_service=case.service,
        ownership=case.ownership,
        snapshot=await case.service.capture_mcp_snapshot(
            second, "a", "sibling-discovery"
        ),
        service=case.unified,
        main_loop=asyncio.get_running_loop(),
    )
    assert (await provider.connect("mcp:shared"))["tools"][0]["name"] == "echo"
    await configure(second, profile["profile_id"])
    authority = await case.service._call(case.service._coordinator.published_snapshot)
    mappings = authority["mappings"]
    targets = [row for row in mappings if row["kind"] == "tool"]
    assert len(targets) == 2 and targets[0]["mapping_id"] == targets[1]["mapping_id"]
    assert len({row["installation_id"] for row in targets}) == 2
    assert len({row["revision_digest"] for row in targets}) == 2
    assert not any(
        row.phase == "recovery_required"
        for row in await case.service._call(case.service._coordinator.recover)
    )
    assert (await case.service._call(case.service._coordinator.published_snapshot))[
        "mappings"
    ] == mappings
    inspections = retained_inspections(authority)
    constraints = {
        row["installation_id"]: row
        for row in mappings
        if row["kind"] == "configuration"
    }

    async def validate(installation, candidates):
        constraint = constraints[installation]
        await case.service._call(
            lambda: validate_mapping(
                case.service._coordinator,
                constraint,
                inspections[(installation, constraint["revision_digest"])],
                candidates,
            )
        )

    own = next(row for row in targets if row["installation_id"] == case.installation)
    foreign = next(row for row in targets if row["installation_id"] == second)
    with pytest.raises(PluginUnavailable, match="plugin_tool_mapping_changed"):
        await validate(case.installation, [foreign])
    with pytest.raises(PluginUnavailable, match="plugin_tool_mapping_changed"):
        await validate(
            case.installation, [{**own, "revision_digest": foreign["revision_digest"]}]
        )
    original = case.local.store.get_discovery_snapshot(case.profile_id)
    changed = json.loads(json.dumps(original))
    changed["tools"][0]["description"] = "changed A only"
    case.local.store.save_discovery_snapshot(case.profile_id, changed)
    try:
        with pytest.raises(CredentialError, match="credential_mapping_changed"):
            await validate(case.installation, mappings)
        await validate(second, mappings)
        assert any(
            row.phase == "recovery_required"
            for row in await case.service._call(case.service._coordinator.recover)
        )
    finally:
        case.local.store.save_discovery_snapshot(case.profile_id, original)
        assert not any(
            row.phase == "recovery_required"
            for row in await case.service._call(case.service._coordinator.recover)
        )


@pytest.mark.asyncio
async def test_combined_hooks_keep_distinct_installation_revision_process_owners(
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

    service = native_console.service
    snapshots, roots = [], {}
    code = "import json,sys,time;from pathlib import Path;e=json.load(sys.stdin);root=Path('${PLUGIN_DATA}');(root/e['runtime_session_id']).write_text('running');exec(\"while not (root/'release').exists(): time.sleep(.01)\");print('{\"version\":2,\"decision\":\"pass\"}')"
    for name in ("a", "b"):
        package = native_package(extension={"version": 1, "hooks": "hooks.json"})
        manifest = json.loads((package / "plugin.json").read_text())
        manifest["name"] = "owned-hook-" + name
        (package / "plugin.json").write_text(json.dumps(manifest))
        (package / "hooks.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "hooks": [
                        {
                            "id": "work",
                            "type": "command",
                            "event": "PreToolUse",
                            "effects": [],
                            "argv": child_argv(code),
                            "timeout_seconds": 20,
                        }
                    ],
                }
            )
        )
        installed = await install_components(service, package, ("hook:work",))
        review = await service.review_data_creation(
            installed.installation_id, workspace_id=None
        )
        roots[installed.installation_id] = await service.create_data(
            review, review.operation_id
        )
        snapshots.append(
            await service.capture_mcp_snapshot(
                installed.installation_id, "workspace-a", "pending-combined"
            )
        )
    a, b = snapshots
    assert (
        a.installation_id != b.installation_id
        and a.revision_digest != b.revision_digest
    )
    owner = NativeHookSet(service, snapshots)
    assert len({handler.id for handler in owner.definitions}) == 2
    engine = HookEngine(
        owner.definitions,
        owner.current,
        HookBudgetOwner(),
        process_owner_for=owner.process_owner,
        host_environment=owner.host_environment,
        command_launch=owner.command_launch,
    )
    entered, release_publish = threading.Event(), threading.Event()
    publish = NativeHookProcessOwner.publish_process

    def delayed(self, token, provenance):
        if self.snapshot.installation_id == a.installation_id:
            entered.set()
            assert release_publish.wait(10)
        return publish(self, token, provenance)

    monkeypatch.setattr(NativeHookProcessOwner, "publish_process", delayed)
    shared_event = event(runtime="combined")
    assert engine.notify(shared_event)
    try:
        for _ in range(500):
            if (
                entered.is_set()
                and (roots[b.installation_id].path / "combined").exists()
            ):
                break
            await asyncio.sleep(0.01)
        assert (
            entered.is_set() and (roots[b.installation_id].path / "combined").exists()
        )
        jobs = {
            job.owner.snapshot.installation_id: job
            for job in engine.processes.records.values()
        }
        assert set(jobs) == {a.installation_id, b.installation_id}
        original_owners = {key: job.owner for key, job in jobs.items()}
        assert {
            (row.installation_id, row.revision_digest) for row in service.live_runs()
        } == {
            (snapshot.installation_id, snapshot.revision_digest)
            for snapshot in snapshots
        }
        process_rows = await service._call(
            lambda: service._coordinator.owner.list_processes(limit=20, offset=0)
        )
        active = {
            row["installation_id"]: row
            for row in process_rows
            if row["state"] != "settled"
        }
        assert set(active) == set(jobs)
        assert {row["operation_id"] for row in active.values()} == {
            shared_event.event_id
        }
        for snapshot in snapshots:
            row = active[snapshot.installation_id]
            assert row["revision_digest"] == snapshot.revision_digest
            assert row["token"] == jobs[snapshot.installation_id].token
            blockers = await service._call(
                lambda snapshot=snapshot: service._coordinator.root_usage.blockers(
                    (roots[snapshot.installation_id],)
                )
            )
            assert any(
                blocker.get("owner_token") == row["token"] for blocker in blockers
            )
        assert await service._call(lambda: service._coordinator.root_usage.dirty)
        disabled = await service.review_activation(
            a.installation_id, workspace_id="workspace-a", intent="disabled"
        )
        committed = asyncio.create_task(service.commit(disabled, disabled.operation_id))
        for _ in range(100):
            if jobs[a.installation_id].cancel.is_set():
                break
            await asyncio.sleep(0.01)
        assert jobs[a.installation_id].cancel.is_set()
        assert (
            not jobs[b.installation_id].cancel.is_set()
            and not jobs[b.installation_id].task.done()
        )
        assert not (roots[a.installation_id].path / "combined").exists()
        release_publish.set()
        assert (await asyncio.wait_for(jobs[a.installation_id].task, 10)).failure
        assert (await asyncio.wait_for(committed, 10)).committed
        assert {row.installation_id for row in service.live_runs()} == {
            b.installation_id
        }
        assert not await service._call(
            lambda: service._coordinator.root_usage.blockers(
                (roots[a.installation_id],)
            )
        )
        assert await service._call(
            lambda: service._coordinator.root_usage.blockers(
                (roots[b.installation_id],)
            )
        )
        (roots[b.installation_id].path / "release").touch()
        result = await asyncio.wait_for(jobs[b.installation_id].task, 10)
        assert result.failure is None and result.result is not None
        for _ in range(100):
            if not engine.cleanup_pending:
                break
            await asyncio.sleep(0.01)
        assert not engine.cleanup_pending and not service.live_runs()
        assert all(job.owner is original_owners[key] for key, job in jobs.items())
        assert all(
            row["state"] == "settled"
            for row in await service._call(
                lambda: service._coordinator.owner.list_processes(limit=20, offset=0)
            )
        )
        assert not await service._call(
            lambda: service._coordinator.root_usage.blockers(tuple(roots.values()))
        )
    finally:
        release_publish.set()
        for root in roots.values():
            (root.path / "release").touch()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_native_hook_path_uses_host_lookup_and_explicit_package_control(
    native_console, native_package, tmp_path, monkeypatch, explicit
):
    import json
    import os
    import sys

    from Tests.Agents.test_hooks_v2_execution import event
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Plugins.components import NativeHookSet

    package = native_package(extension={"version": 1, "hooks": "hooks.json"})
    host = tmp_path / "host-bin"
    host.mkdir()
    (package / "bin").mkdir()
    marker = tmp_path / "which-executable"
    for directory, label in ((host, "host"), (package / "bin", "package")):
        program = directory / "owned-probe"
        program.write_text(
            f"#!{sys.executable}\nimport os,json\nfrom pathlib import Path\nPath({str(marker)!r}).write_text(json.dumps([{label!r},os.environ['PATH'],os.environ['CUSTOM']]))\nprint('{{\"version\":2,\"decision\":\"pass\"}}')\n"
        )
        program.chmod(0o700)
    monkeypatch.setenv("PATH", str(host) + os.pathsep + os.environ.get("PATH", ""))
    (package / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "path",
                        "event": "PreToolUse",
                        "type": "command",
                        "effects": ["deny"],
                        "argv": ["./bin/owned-probe" if explicit else "owned-probe"],
                        "env": {"PATH": "${PLUGIN_ROOT}/bin", "CUSTOM": "preserved"},
                    }
                ],
            }
        )
    )
    installed = await install_components(
        native_console.service, package, ("hook:path",)
    )
    snapshot = await native_console.service.capture_mcp_snapshot(
        installed.installation_id, "workspace-a", "path-run"
    )
    native = NativeHookSet(native_console.service, (snapshot,))
    engine = HookEngine(
        native.definitions,
        native.current,
        HookBudgetOwner(),
        process_owner_for=native.process_owner,
        host_environment=native.host_environment,
        command_launch=native.command_launch,
    )
    try:
        assert (await engine.fire_async(event())).succeeded
        assert json.loads(marker.read_text()) == [
            "package" if explicit else "host",
            snapshot.inspection.materialized_identity + "/bin",
            "preserved",
        ]
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path_kind", ["relative", "dot", "empty-segment"])
async def test_native_hook_relative_host_path_is_anchored(
    native_console, native_package, tmp_path, monkeypatch, path_kind
):
    import json
    import os
    import sys

    from Tests.Agents.test_hooks_v2_execution import event
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Plugins.components import NativeHookSet

    package = native_package(extension={"version": 1, "hooks": "hooks.json"})
    host_cwd = tmp_path / "host-cwd"
    relative = (
        "host-bin/owned-path-probe" if path_kind == "relative" else "owned-path-probe"
    )
    for program, label in (
        (host_cwd / relative, "host"),
        (package / relative, "package"),
        (package / "bin/owned-path-probe", "package"),
    ):
        program.parent.mkdir(parents=True, exist_ok=True)
        program.write_text(
            f"#!{sys.executable}\nimport os,json\nfrom pathlib import Path\n"
            f"with Path({str(tmp_path / (label + '-executions'))!r}).open('a') as marker:\n"
            " marker.write(json.dumps([__file__,os.environ['PATH'],os.environ['CUSTOM']])+'\\n')\n"
            'print(\'{"version":2,"decision":"pass"}\')\n'
        )
        program.chmod(0o700)
    monkeypatch.chdir(host_cwd)
    monkeypatch.setenv(
        "PATH",
        {
            "relative": "host-bin",
            "dot": ".",
            "empty-segment": os.pathsep + str(tmp_path / "missing-bin"),
        }[path_kind],
    )
    (package / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "path",
                        "event": "PreToolUse",
                        "type": "command",
                        "effects": ["deny"],
                        "argv": ["owned-path-probe"],
                        "env": {"PATH": "${PLUGIN_ROOT}/bin", "CUSTOM": "preserved"},
                    }
                ],
            }
        )
    )
    installed = await install_components(
        native_console.service, package, ("hook:path",)
    )
    snapshot = await native_console.service.capture_mcp_snapshot(
        installed.installation_id, "workspace-a", "relative-path-run"
    )
    native = NativeHookSet(native_console.service, (snapshot,))
    engine = HookEngine(
        native.definitions,
        native.current,
        HookBudgetOwner(),
        process_owner_for=native.process_owner,
        host_environment=native.host_environment,
        command_launch=native.command_launch,
    )
    try:
        assert (await engine.fire_async(event())).succeeded
        marker = tmp_path / "host-executions"
        assert marker.exists() and not (tmp_path / "package-executions").exists()
        before = marker.read_text()
        assert json.loads(before) == [
            str(host_cwd / relative),
            snapshot.inspection.materialized_identity + "/bin",
            "preserved",
        ]
        for missing in (True, False):
            with monkeypatch.context() as context:
                if missing:
                    context.delenv("PATH")
                else:
                    context.setenv("PATH", "")
                result = await engine.fire_async(event())
                assert not result.allowed and not result.succeeded
                assert marker.read_text() == before
                assert not (tmp_path / "package-executions").exists()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_unscoped_hook_retains_real_scratch_lease_until_child_terminal(
    native_console, native_package, tmp_path
):
    import asyncio
    import json

    from Tests.Agents.test_hooks_v2_execution import event
    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
    from tldw_chatbook.Plugins.components import NativeHookSet

    package = native_package(extension={"version": 1, "hooks": "hooks.json"})
    observed, release = tmp_path / "scratch-observed", tmp_path / "scratch-release"
    code = f'import os,time;from pathlib import Path;Path({str(observed)!r}).write_text(os.environ[\'CHATBOOK_WORKSPACE_DATA\']);exec("while not Path({str(release)!r}).exists(): time.sleep(.01)");print(\'{{"version":2,"decision":"pass"}}\')'
    (package / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "scratch",
                        "event": "PreToolUse",
                        "type": "command",
                        "effects": ["deny"],
                        "argv": child_argv(code),
                        "timeout_seconds": 20,
                    }
                ],
            }
        )
    )
    service = native_console.service
    installed = await install_components(service, package, ("hook:scratch",))
    activation = await service.review_activation(
        installed.installation_id, workspace_id=None, intent="enabled"
    )
    assert (await service.commit(activation, activation.operation_id)).committed
    snapshot = await service.capture_mcp_snapshot(
        installed.installation_id, None, "scratch-run"
    )
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    scratch = manager.snapshot("actual-session")
    native = NativeHookSet(
        service, (snapshot,), scratch_manager=manager, scratch_snapshot=scratch
    )
    engine = HookEngine(
        native.definitions,
        native.current,
        HookBudgetOwner(),
        process_owner_for=native.process_owner,
        host_environment=native.host_environment,
        command_launch=native.command_launch,
    )
    task = asyncio.create_task(engine.fire_async(event()))
    try:
        for _ in range(500):
            if observed.exists():
                break
            await asyncio.sleep(0.01)
        assert observed.read_text() == str(scratch.root)
        job = next(iter(engine.processes.records.values()))
        assert job.capture.transport.get_returncode() is None
        assert not json.loads(snapshot.data_roots_json)
        manager.close("actual-session")
        assert not manager.is_live(scratch) and scratch.root.exists()
        assert not native.current(native.definitions[0], event(), "accept")
        assert not (await engine.fire_async(event(runtime="other"))).allowed
        assert len(engine.processes.records) == 1 and not task.done()
        release.touch()
        outcome = await asyncio.wait_for(task, 10)
        assert not outcome.allowed and not outcome.accepted
        assert job.capture.transport.get_returncode() is not None
        assert await asyncio.to_thread(manager.wait_for_cleanup, 5)
        assert not scratch.root.exists() and not service.live_runs()
    finally:
        release.touch()
        await engine.close()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(manager.dispose)


@pytest.mark.asyncio
async def test_native_hook_shared_and_workspace_roots_are_exact_and_distinct(
    native_console, native_package, tmp_path, monkeypatch
):
    import json

    from Tests.Agents.test_hooks_v2_execution import event
    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Plugins.components import NativeHookSet

    service = native_console.service
    package = native_package(extension={"version": 1, "hooks": "hooks.json"})
    marker = tmp_path / "root-projection"
    code = f"import json,os,sys;from pathlib import Path;assert sys.argv[-2:]==[os.environ['PLUGIN_DATA'],os.environ['CHATBOOK_WORKSPACE_DATA']];Path({str(marker)!r}).write_text(json.dumps([os.environ['PLUGIN_ROOT'],*sys.argv[-2:]]));print('{{\"version\":2,\"decision\":\"pass\"}}')"
    (package / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "roots",
                        "event": "PreToolUse",
                        "type": "command",
                        "effects": ["deny"],
                        "argv": [
                            *child_argv(code),
                            "${PLUGIN_DATA}",
                            "${CHATBOOK_WORKSPACE_DATA}",
                        ],
                    }
                ],
            }
        )
    )
    installed = await install_components(service, package, ("hook:roots",))
    enabled = await service.review_activation(
        installed.installation_id, workspace_id="workspace-b", intent="enabled"
    )
    assert (await service.commit(enabled, enabled.operation_id)).committed
    roots = {}
    for scope in (None, "workspace-a", "workspace-b"):
        review = await service.review_data_creation(
            installed.installation_id, workspace_id=scope
        )
        roots[scope] = await service.create_data(review, review.operation_id)
    monkeypatch.setenv("CHATBOOK_WORKSPACE_DATA", str(tmp_path / "ambient-fake"))
    for scope in ("workspace-a", "workspace-b"):
        snapshot = await service.capture_mcp_snapshot(
            installed.installation_id, scope, "roots-" + scope
        )
        assert {row["root_id"] for row in json.loads(snapshot.data_roots_json)} == {
            roots[None].root_id,
            roots[scope].root_id,
        }
        native = NativeHookSet(service, (snapshot,))
        engine = HookEngine(
            native.definitions,
            native.current,
            HookBudgetOwner(),
            process_owner_for=native.process_owner,
            host_environment=native.host_environment,
            command_launch=native.command_launch,
        )
        try:
            assert (await engine.fire_async(event())).succeeded
            assert json.loads(marker.read_text()) == [
                snapshot.inspection.materialized_identity,
                str(roots[None].path),
                str(roots[scope].path),
            ]
        finally:
            await engine.close()
    assert roots["workspace-a"].path != roots["workspace-b"].path
    # Exact workspace grants cannot be substituted by the shared root.
    deletion = await service.review_data_deletion((roots["workspace-b"],))
    assert (
        await service.delete_data((roots["workspace-b"],), deletion.operation_id)
    ).committed
    snapshot = await service.capture_mcp_snapshot(
        installed.installation_id, "workspace-b", "missing-root"
    )
    native = NativeHookSet(service, (snapshot,))
    engine = HookEngine(
        native.definitions,
        native.current,
        HookBudgetOwner(),
        process_owner_for=native.process_owner,
        host_environment=native.host_environment,
        command_launch=native.command_launch,
    )
    marker.unlink()
    try:
        assert not (await engine.fire_async(event())).allowed
        assert not marker.exists() and not engine.processes.records
    finally:
        await engine.close()
