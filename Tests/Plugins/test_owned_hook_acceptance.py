"""Companion hook boundaries through installed, reviewed plugin registrations."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.hooks_v2_process_support import child_argv
from Tests.Plugins.test_native_components import install_components

_REQUIRES_COMMAND_HOOKS = pytest.mark.skipif(
    "sys.platform == 'win32'",
    reason="Windows command hooks refuse with unsupported_platform; command effects are unexercised.",
)


def _command(identity, event, code="pass", **values):
    return {
        "id": identity,
        "event": event,
        "type": "command",
        "effects": [],
        "argv": child_argv(code),
        **values,
    }


def _package(native_package, handlers, *, requires=None, commands=()):
    package = native_package(
        extension={
            "version": 1,
            "hooks": "hooks.json",
            "requires": requires or {},
            "commands": [name + ".md" for name in commands],
        }
    )
    (package / "hooks.json").write_text(json.dumps({"version": 2, "hooks": handlers}))
    for name in commands:
        (package / (name + ".md")).write_text(
            f"---\nname: {name}\ndescription: Controlled command\n---\n{name} body."
        )
    return package


async def _wait(check):
    for _ in range(1000):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("owned boundary did not settle")


@asynccontextmanager
async def _runtime(rig):
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    runtime = ConsoleRuntime(app=rig.controller.app)
    rig.controller.app.console_runtime = runtime
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    try:
        yield runtime
    finally:
        await runtime.dispose()


async def _native_engine(service, installation, run_id, budget=None, *, enabled=True):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Plugins.components import NativeHookSet

    snapshot = await service.capture_mcp_snapshot(installation, "workspace-a", run_id)
    native = NativeHookSet(service, (snapshot,))
    assert native.definitions
    assert all(
        handler.id.startswith(f"plugin:{installation}:hook:")
        for handler in native.definitions
    )
    engine = HookEngine(
        native.definitions,
        native.current,
        budget or HookBudgetOwner(),
        enabled=enabled,
        process_owner_for=native.process_owner,
        host_environment=native.host_environment,
        dependency_required=lambda handler, event: handler.id in native.requirements,
    )
    engine.native_hooks = native
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("empty_success", marks=_REQUIRES_COMMAND_HOOKS),
        pytest.param("valid_context", marks=_REQUIRES_COMMAND_HOOKS),
        pytest.param("empty_required_context", marks=_REQUIRES_COMMAND_HOOKS),
        pytest.param("whitespace_context", marks=_REQUIRES_COMMAND_HOOKS),
        pytest.param("oversized_context", marks=_REQUIRES_COMMAND_HOOKS),
        pytest.param("optional_failure", marks=_REQUIRES_COMMAND_HOOKS),
        pytest.param("dependency_failure", marks=_REQUIRES_COMMAND_HOOKS),
        "master_off",
        pytest.param("disabled_required", marks=_REQUIRES_COMMAND_HOOKS),
        "invalid_required",
    ],
)
async def test_installed_required_effect_admission(
    native_console, native_package, tmp_path, mode
):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event
    from tldw_chatbook.Plugins.inspection import inspect_package

    rig = native_console
    context = {
        "valid_context": "Installed bounded context.",
        "whitespace_context": "  ",
        "oversized_context": "x" * 4097,
    }.get(mode)
    required_context = mode in {
        "valid_context",
        "empty_required_context",
        "whitespace_context",
        "oversized_context",
        "master_off",
    }
    output = {"version": 2, "decision": "pass"}
    if context is not None:
        output["context"] = [{"text": context, "lifetime": "runtime"}]
    failure = mode in {"optional_failure", "dependency_failure"}
    hook = _command(
        "check",
        "UserPromptSubmit" if failure else "SessionStart",
        (
            "raise SystemExit(1)"
            if failure
            else (
                "pass"
                if mode
                in {
                    "empty_success",
                    "empty_required_context",
                    "master_off",
                    "disabled_required",
                }
                else f"print({json.dumps(output)!r})"
            )
        ),
        required=not failure,
        require_context=required_context,
        effects=["context"] if required_context else [],
    )
    requires = (
        {"command:dependent": ["hook:check"]} if mode == "dependency_failure" else {}
    )
    if mode == "invalid_required":
        hook["require_context"] = "invalid"
    stop_file = tmp_path / "root-stop"
    stop = _command(
        "stop", "Stop", f"from pathlib import Path;Path({str(stop_file)!r}).touch()"
    )
    package = _package(
        native_package,
        [hook, stop],
        requires=requires,
        commands=("dependent", "independent"),
    )
    if mode == "invalid_required":
        inspection = inspect_package(package)
        assert inspection.inventory["command:dependent"].activation_blockers
        rejected = await install_components(
            rig.service, package, ("command:dependent",)
        )
        from tldw_chatbook.Plugins.admission import PluginUnavailable

        with pytest.raises(PluginUnavailable, match="plugin_constraints_blocked"):
            await rig.service.capture_mcp_snapshot(
                rejected.installation_id, "workspace-a", "invalid-required"
            )
        assert not rig.service.capture_maximum("workspace-a")["native_components"]
        control = await rig.install()
        async with _runtime(rig):
            sent = await rig.controller.submit_draft(
                f"${control.alias}:review", session_id=rig.session.id
            )
            assert sent.accepted and rig.gateway.payloads
        return
    installed = await install_components(
        rig.service,
        package,
        ("hook:check", "hook:stop", "command:dependent", "command:independent"),
    )
    if mode == "disabled_required":
        engine = await _native_engine(
            rig.service, installed.installation_id, "disabled", enabled=False
        )
        try:
            result = await engine.fire_async(
                lifecycle_event(
                    "SessionStart",
                    "disabled",
                    data={"reason": "startup"},
                    workspace_id="workspace-a",
                )
            )
            assert not result.allowed and result.failures[0].code == "disabled"
            assert not engine.processes.records
        finally:
            await engine.close()
        control = await _native_engine(
            rig.service, installed.installation_id, "enabled"
        )
        try:
            assert (
                await control.fire_async(
                    lifecycle_event(
                        "SessionStart",
                        "enabled",
                        data={"reason": "startup"},
                        workspace_id="workspace-a",
                    )
                )
            ).succeeded
        finally:
            await control.close()
        return
    if mode == "master_off":
        rig.controller.app.app_config = {"hooks": {"enabled": False}}
    accepted = []
    rig.controller.on_submission_accepted = lambda: accepted.append(rig.session.id)
    async with _runtime(rig) as runtime:
        sent = await rig.controller.submit_draft(
            (
                f"/{installed.alias}:dependent"
                if mode == "dependency_failure"
                else "hello"
            ),
            session_id=rig.session.id,
        )
        expected = mode in {"empty_success", "valid_context", "optional_failure"}
        assert sent.accepted is expected, sent
        assert bool(rig.gateway.payloads) is expected
        assert bool(accepted) is expected
        # Direct non-agent submit proves admission/provider behavior; root Stop is
        # exercised by the separate public runtime-turn initialization cases.
        assert not stop_file.exists()
        if expected:
            assert runtime._hooks_v2_lifecycles[rig.session.id].live
        engine = runtime.get_hooks_v2(rig.session.id)
        assert (
            engine.native_hooks.snapshots[0].installation_id
            == installed.installation_id
        )
        if expected and context:
            assert context in str(rig.gateway.payloads)
        if not expected:
            assert "Installed bounded context." not in str(rig.gateway.payloads)
        if mode == "dependency_failure":
            independent = await rig.controller.submit_draft(
                f"/{installed.alias}:independent", session_id=rig.session.id
            )
            assert independent.accepted and "independent body." in str(
                rig.gateway.payloads
            )
    assert not rig.service.live_runs()


@_REQUIRES_COMMAND_HOOKS
@pytest.mark.asyncio
async def test_installed_hook_aggregate_admission_and_teardown(
    native_console, native_package, tmp_path
):
    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_hooks_v2_tool_pipeline import CFG
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    entered, release = tmp_path / "entered", tmp_path / "release"
    entered.mkdir()
    code = (
        "import json,sys,time;from pathlib import Path;e=json.load(sys.stdin);"
        f"Path({str(entered)!r},e['runtime_session_id']+'-'+e['event_id']).touch();"
        f"exec({f'while not Path({str(release)!r},e["runtime_session_id"]).exists(): time.sleep(0.01)'!r})"
    )
    release.mkdir()
    guard_code = (
        "import json,sys;e=json.load(sys.stdin);"
        "print(json.dumps({'version':2,'decision':'deny' if "
        "e['data']['tool_args']['expression']=='1' else 'pass'}))"
    )
    package = _package(
        native_package,
        [
            _command("hold", "SessionStart", code, required=True, timeout_seconds=60),
            _command("observe", "Stop", code, timeout_seconds=60),
            _command(
                "guard", "PreToolUse", guard_code, required=True, effects=["deny"]
            ),
        ],
    )
    rig = native_console
    installed = await install_components(
        rig.service, package, ("hook:hold", "hook:observe", "hook:guard")
    )
    budget = HookBudgetOwner()
    engines = {
        name: await _native_engine(rig.service, installed.installation_id, name, budget)
        for name in (
            "busy-a",
            "busy-b",
            "quiet-c",
            "quiet-d",
            "observe-a",
            "observe-b",
            "guard",
        )
    }
    pending = []
    db = AgentRunsDB(tmp_path / "aggregate-runs.db", client_id="installed-aggregate")

    def event(name, runtime):
        return lifecycle_event(
            name,
            runtime,
            data=(
                {"reason": "startup"}
                if name == "SessionStart"
                else {"status": "completed"}
            ),
            workspace_id="workspace-a",
        )

    async def guard(expression):
        registry = ToolCatalogRegistry()
        registry.register_provider(BuiltinToolProvider())
        chat = ScriptedChat([fence("calculator", {"expression": expression}), "done"])
        service = AgentService(
            db,
            registry,
            chat_call=chat,
            hooks_v2_engine=engines["guard"],
            hooks_v2_session_id="guard",
            hooks_v2_turn_id="guard-" + expression,
        )
        run, outcome = await asyncio.to_thread(
            service.run_turn,
            conversation_id="guard-" + expression,
            messages=[{"role": "user", "content": "go"}],
            config=CFG,
            api_endpoint="test",
        )
        return db.get_run(run), outcome

    try:
        # The first two runtimes occupy all execution slots; quiet runtimes queue.
        for runtime in ("busy-a", "busy-b", "quiet-c", "quiet-d"):
            pending.extend(
                asyncio.create_task(
                    engines[runtime].fire_async(event("SessionStart", runtime))
                )
                for _ in range(16)
            )
            await _wait(
                lambda runtime=runtime: budget.snapshot(runtime)["tickets"] == 16
            )
        await _wait(lambda: len(list(entered.iterdir())) == 8)
        assert budget.snapshot()["tickets"] == 64
        assert budget.snapshot()["execution"] == 8
        assert (
            budget.snapshot("busy-a")["execution"]
            == budget.snapshot("busy-b")["execution"]
            == 4
        )
        for runtime in ("busy-a", "fresh"):
            refused = await engines["busy-a"].fire_async(event("SessionStart", runtime))
            assert not refused.allowed
            assert [row.code for row in refused.failures] == ["delivery_capacity"]
        (release / "busy-a").touch()
        await _wait(
            lambda: any(p.name.startswith("quiet-c-") for p in entered.iterdir())
        )
        await engines["busy-b"].close()
        assert not engines["busy-b"].processes.records
        assert any(not task.done() for task in pending)
        assert engines["quiet-c"].processes.records
        for runtime in ("quiet-c", "quiet-d"):
            (release / runtime).touch()
        await asyncio.gather(*pending)
        assert budget.snapshot()["tickets"] == budget.snapshot()["execution"] == 0
        assert all(not engine.processes.records for engine in engines.values())

        for runtime in ("observe-a", "observe-b"):
            assert all(
                engines[runtime].notify(event("Stop", runtime)) for _ in range(64)
            )
        assert budget.snapshot()["observations"] == 128
        await _wait(lambda: budget.snapshot()["workers"] == 8)
        await _wait(
            lambda: len([p for p in entered.iterdir() if p.name.startswith("observe-")])
            == 8
        )
        for runtime in ("observe-a", "observe-b"):
            assert budget.snapshot(runtime)["workers"] == 4
            assert all(
                engines[runtime].notify(event("Stop", runtime)) for _ in range(4)
            )
        assert budget.snapshot()["observations"] == 128
        assert not engines["observe-a"].notify(event("Stop", "overflow"))
        blocked, _ = await guard("1")
        assert not any(
            step.get("tool_outcome") == "success"
            for step in blocked["steps"]
            if step.get("kind") == "tool_result"
        )
        passed, _ = await guard("2")
        assert any(
            step.get("tool_outcome") == "success"
            for step in passed["steps"]
            if step.get("kind") == "tool_result"
        )
        assert budget.snapshot()["observations"] == 128
    finally:
        for runtime in engines:
            (release / runtime).touch()
        await asyncio.gather(*(engine.close() for engine in engines.values()))
        await asyncio.gather(*pending, return_exceptions=True)
        db.close()
    assert not any(budget.snapshot().values())
    assert all(not engine.processes.records for engine in engines.values())
    assert not rig.service.live_runs()


@asynccontextmanager
async def _owned_peer(
    native_package,
    tmp_path,
    monkeypatch,
    handlers,
    *,
    transport="stdio",
    requires=None,
    schema=None,
):
    """One reviewed owned profile over the existing inert transport fixtures."""
    import os
    import sys

    from Tests.Agents.test_hooks_v2_mcp_execution import http_hook_service
    from Tests.MCP.test_typed_tool_results import controlled_stdio_client
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.MCP.connection_ownership import ConnectionOwnership
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.mcp_provider import PluginMCPProvider
    from tldw_chatbook.Plugins.recovery import retained_inspections
    from tldw_chatbook.Plugins.service import PluginService

    @asynccontextmanager
    async def transport_case():
        if transport == "http":
            async with http_hook_service(tmp_path, monkeypatch) as (
                remote,
                client,
                requests,
            ):
                profile = remote.local_service.store.get_profile("fixture")
                yield client, {"type": "streamable-http", "url": profile.url}, requests
        else:
            async with controlled_stdio_client(tmp_path, tool_schema=schema) as client:
                original_peer = (tmp_path / "peer.py").read_text()
                capability = '"capabilities": {}'
                assert original_peer.count(capability) == 1
                owned_peer = tmp_path / "owned-peer.py"
                owned_peer.write_text(
                    original_peer.replace(
                        capability, '"capabilities": {"tools": {}}', 1
                    )
                )
                assert (tmp_path / "peer.py").read_text() == original_peer
                monkeypatch.setenv(
                    "PATH",
                    str(Path(sys.executable).parent)
                    + os.pathsep
                    + os.environ.get("PATH", ""),
                )
                yield client, {
                    "type": "stdio",
                    "command": Path(sys.executable).name,
                    "args": [
                        "-I",
                        str(owned_peer),
                        json.dumps(schema or {"type": "object"}),
                    ],
                }, None

    async with transport_case() as (client, definition, requests):
        local = LocalMCPControlService(
            store=LocalMCPStore(tmp_path / "owned-mcp.json"),
            client=client,
            manifest_provider=dict,
        )
        service = PluginService(
            tmp_path / "owned-profile",
            workspace_lookup=lambda _: SimpleNamespace(archived=False),
            marker_store_factory=lambda _: FilePluginMarkerStore(
                tmp_path / "owned-marker"
            ),
            accept_reduced_protection=True,
            mcp_mapping_owner=local,
        )
        await service.bootstrap("test passphrase")
        ownership = ConnectionOwnership(plugin_service=service, local_service=local)
        unified = UnifiedMCPControlPlaneService(
            local_service=local,
            server_service=None,
            target_store=None,
            context_store=None,
        )
        package = _package(native_package, handlers, requires=requires)
        (package / "mcp.json").write_text(
            json.dumps(
                {
                    "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                    "mcpServers": {"peer": definition},
                }
            )
        )
        try:
            installed = await install_components(
                service,
                package,
                ("mcp:peer", *["hook:" + row["id"] for row in handlers]),
            )
            inspection = await service._call(
                lambda: retained_inspections(service._coordinator.published_snapshot())[
                    (installed.installation_id, installed.inspection.effective_digest)
                ]
            )
            profile = local.save_owned_profile(
                installation_id=installed.installation_id,
                inspection=inspection,
                component_id="mcp:peer",
                session_isolation="request_independent",
                protocol_version="2025-03-26",
                development_loopback=transport == "http",
            )
            profile_id = profile["profile_id"]
            configured = await service.review_configuration(
                installed.installation_id, connections={"mcp:peer": profile_id}
            )
            assert (await service.commit(configured, configured.operation_id)).committed
            snapshot = await service.capture_mcp_snapshot(
                installed.installation_id, "workspace-a", "discovery"
            )
            discovery = PluginMCPProvider(
                plugin_service=service,
                ownership=ownership,
                snapshot=snapshot,
                service=unified,
                main_loop=asyncio.get_running_loop(),
            )
            assert (await discovery.connect("mcp:peer"))["tools"][0][
                "name"
            ] == "fixture"
            for connection in tuple(ownership.connections.values()):
                for owner in tuple(connection.owners):
                    await ownership.detach(connection.connection_id, owner)
            configured = await service.review_configuration(
                installed.installation_id,
                connections={"mcp:peer": profile_id},
                tools={"mcp:peer": ("fixture",)},
            )
            assert (await service.commit(configured, configured.operation_id)).committed
            approvals = []

            def approve(calls):
                approvals.extend(calls)
                return {call.llm_name: "approve_once" for call in calls}

            snapshot = await service.capture_mcp_snapshot(
                installed.installation_id, "workspace-a", "owned-case"
            )
            provider = PluginMCPProvider(
                plugin_service=service,
                ownership=ownership,
                snapshot=snapshot,
                service=unified,
                main_loop=asyncio.get_running_loop(),
                approval_callback=approve,
            )
            await provider.compose_catalog()
            registry = ToolCatalogRegistry()
            registry.register_provider(provider)
            assert len(provider.list_catalog()) == 1
            yield SimpleNamespace(
                service=service,
                installed=installed,
                ownership=ownership,
                local=local,
                unified=unified,
                provider=provider,
                registry=registry,
                client=client,
                profile_id=profile_id,
                approvals=approvals,
                requests=requests,
                name=provider.list_catalog()[0].name,
                snapshot=snapshot,
            )
        finally:
            for connection in tuple(ownership.connections.values()):
                for owner in tuple(connection.owners):
                    await ownership.detach(connection.connection_id, owner)
            await service.aclose()


def _mcp_hook(payload, *, event="SessionStart", require_context=False):
    return {
        "id": "initial",
        "event": event,
        "type": "mcp_tool",
        "server": "peer",
        "tool": "fixture",
        "required": True,
        "require_context": require_context,
        "effects": ["context"] if require_context else [],
        "input": {"payload": payload},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize(
    "mode",
    [
        "structured",
        "text",
        "mirror",
        "empty",
        "is_error_pass",
        "conflicting_mirror",
        "invalid_structured",
        "empty_required_context",
    ],
)
async def test_installed_mcp_hook_normalization(
    native_package, tmp_path, monkeypatch, transport, mode
):
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext
    from tldw_chatbook.Agents.mcp_tool_provider import capture_mcp_result
    from tldw_chatbook.Plugins.components import tool_hook_requirements

    body = {"version": 2, "decision": "pass"}
    payload = {"content": []}
    if mode in {
        "structured",
        "mirror",
        "is_error_pass",
        "conflicting_mirror",
        "invalid_structured",
    }:
        payload["structuredContent"] = body
    if mode in {"text", "mirror", "conflicting_mirror"}:
        payload["content"] = [
            {
                "type": "text",
                "text": json.dumps(
                    {**body, "decision": "deny"}
                    if mode == "conflicting_mirror"
                    else body
                ),
            }
        ]
    if mode == "is_error_pass":
        payload["isError"] = True
    if mode == "invalid_structured":
        payload["structuredContent"] = {"version": 77}
    async with _owned_peer(
        native_package,
        tmp_path,
        monkeypatch,
        [_mcp_hook(payload, require_context=mode == "empty_required_context")],
        transport=transport,
        requires={"hook:initial": ["mcp:peer"]},
    ) as case:
        await case.provider.connect("mcp:peer")
        before = len(case.unified.execution_log.read_recent())

        def ordinary():
            with capture_mcp_result(case.provider) as capture:
                result = case.registry.invoke_by_name(
                    case.name,
                    {
                        "payload": {
                            "content": [
                                {"type": "text", "text": "ordinary owned control"}
                            ]
                        }
                    },
                )
                raw = capture.consume(result)
                assert result.ok and raw.duplicate_keys_checked
                assert raw.content[0]["text"] == "ordinary owned control"

        await asyncio.to_thread(ordinary)
        assert len(case.unified.execution_log.read_recent()) == before + 1
        engine = await _native_engine(
            case.service, case.installed.installation_id, "owned-case"
        )
        lifecycle = HookSessionLifecycle(
            engine, "owned-case", render_context=engine.native_hooks.render_context
        )
        event = lifecycle.event("SessionStart", data={"reason": "startup"})
        current = lambda: all(
            engine.native_hooks.current(handler, event, "accept")
            for handler in engine.definitions
        )
        engine.mcp_executor.bind_context(
            MCPHookContext(
                case.registry,
                lifecycle,
                lifecycle.scope_id,
                "owned-case",
                "owned-case",
                frozenset({case.name}),
                current,
                lambda definition: tool_hook_requirements(case.registry, definition),
                workspace_id="workspace-a",
            )
        )
        token = lifecycle.reserve(event)
        try:
            result = await lifecycle.initialize(token)
            assert result.succeeded is (
                mode in {"structured", "text", "mirror", "empty"}
            ), result
            assert len(case.unified.execution_log.read_recent()) == before + 2
            assert case.approvals
            assert (
                case.provider.snapshot.installation_id == case.installed.installation_id
            )
            assert case.ownership.is_connected(
                case.profile_id, "owned-case:workspace-a"
            )
        finally:
            lifecycle.cancel(token)
            await engine.close()
        assert not engine.processes.records


def _post_barrier(entered, release, mode):
    return (
        "import json,sys,time;from pathlib import Path;e=json.load(sys.stdin);"
        f"Path({str(entered)!r}).write_text(json.dumps(e));"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(0.01)'!r});"
        + (
            "raise SystemExit(1)"
            if mode == "failure"
            else "print(json.dumps({'version':2,'decision':'pass','context':[{'text':'installed post context','lifetime':'turn'}]}))"
        )
    )


@_REQUIRES_COMMAND_HOOKS
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "failure", "stale"])
async def test_installed_post_tool_checkpoint(
    native_console, native_package, tmp_path, mode
):
    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_hooks_v2_tool_pipeline import CFG
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    entered, release = tmp_path / "post-entered", tmp_path / "post-release"
    package = _package(
        native_package,
        [
            _command(
                "post",
                "PostToolUse",
                _post_barrier(entered, release, mode),
                required=True,
                effects=["context"],
                timeout_seconds=30,
            )
        ],
    )
    installed = await install_components(
        native_console.service, package, ("hook:post",)
    )
    engine = await _native_engine(
        native_console.service, installed.installation_id, "tool-session"
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    db = AgentRunsDB(tmp_path / "post-tool.db", client_id="owned-post")
    chat = ScriptedChat([fence("calculator", {"expression": "6*7"}), "done"])
    terminal, published = [], []
    service = AgentService(
        db,
        registry,
        chat_call=chat,
        hooks_v2_engine=engine,
        hooks_v2_session_id="tool-session",
        hooks_v2_turn_id="tool-turn",
        hooks_v2_render_context=engine.native_hooks.render_context,
        on_run_terminal=lambda *row: terminal.append(row),
        on_step=lambda step, *_: (
            published.append(step) if step.kind == "tool_result" else None
        ),
    )
    pending = asyncio.create_task(
        asyncio.to_thread(
            service.run_turn,
            conversation_id="tool",
            messages=[{"role": "user", "content": "go"}],
            config=CFG,
            api_endpoint="test",
        )
    )
    try:
        await _wait(entered.exists)
        await _wait(lambda: bool(published))
        event = json.loads(entered.read_text())
        assert event["event"] == "PostToolUse" and event["turn_id"] == "tool-turn"
        durable = db.get_run(event["run_id"])
        assert any(
            row.get("kind") == "tool_result" and row.get("tool_outcome") == "success"
            for row in durable["steps"]
        )
        assert len(chat.calls) == 1 and not terminal and not pending.done()
        if mode == "stale":
            disabled = await native_console.service.review_activation(
                installed.installation_id, workspace_id="workspace-a", intent="disabled"
            )
            assert (
                await native_console.service.commit(disabled, disabled.operation_id)
            ).committed
        release.touch()
        await pending
        assert bool(terminal)
        assert (len(chat.calls) == 2) is (mode == "success")
        assert ("installed post context" in str(chat.calls[-1])) is (mode == "success")
        assert any(
            row.get("kind") == "tool_result" and row.get("tool_outcome") == "success"
            for row in db.get_run(event["run_id"])["steps"]
        )
    finally:
        release.touch()
        await asyncio.gather(pending, return_exceptions=True)
        await engine.close()
        db.close()
    assert not engine.processes.records and not native_console.service.live_runs()


@_REQUIRES_COMMAND_HOOKS
@pytest.mark.asyncio
@pytest.mark.parametrize("fleet", [False, True], ids=["inline", "fleet"])
@pytest.mark.parametrize("mode", ["success", "failure", "stale"])
async def test_installed_post_child_checkpoint(
    native_console, native_package, tmp_path, monkeypatch, fleet, mode
):
    from Tests.Agents.test_fleet_runtime import (
        FLEET_CFG,
        fence,
        make_fleet_service,
        make_inline_service,
    )
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME, WAIT_AGENTS_TOOL_NAME
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    entered, release = tmp_path / "child-entered", tmp_path / "child-release"
    package = _package(
        native_package,
        [
            _command(
                "post",
                "SubagentStop",
                _post_barrier(entered, release, mode),
                required=True,
                effects=["context"],
                timeout_seconds=30,
            )
        ],
    )
    installed = await install_components(
        native_console.service, package, ("hook:post",)
    )
    engine = await _native_engine(
        native_console.service, installed.installation_id, "child-session"
    )
    db = AgentRunsDB(tmp_path / "post-child.db", client_id="owned-child")
    replies = [fence(SPAWN_TOOL_NAME, {"task": "child"})]
    if fleet:
        service, chat, _ = make_fleet_service(
            db,
            replies + [fence(WAIT_AGENTS_TOOL_NAME, {}), "done"],
            {"child": ["child result"]},
            allow_unconsumed=mode != "success",
        )
    else:
        service, chat = make_inline_service(
            db, replies + ["child result", "done"], monkeypatch
        )
    service._hooks_v2_engine = engine
    service._hooks_v2_session_id = "child-session"
    service._hooks_v2_turn_id = "child-turn"
    service._hooks_v2_render_context = engine.native_hooks.render_context
    pending = asyncio.create_task(
        asyncio.to_thread(
            service.run_turn,
            conversation_id="child",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
    )
    try:
        await _wait(entered.exists)
        event = json.loads(entered.read_text())
        child = next(
            row for row in db.list_runs("child") if row["agent_kind"] == "subagent"
        )
        assert child["status"] == "done"
        parent = db.get_run(child["parent_run_id"])
        assert parent["status"] not in {"done", "error", "cancelled"}
        assert event["event"] == "SubagentStop" and event["run_id"] == parent["id"]
        assert not pending.done()
        old_calls = len(chat.calls)
        next_scope = None
        if mode == "stale":
            lifecycle = service._hooks_v2_lifecycle
            lifecycle.close_scope(parent["id"])
            next_scope = lifecycle.open_scope()
        release.touch()
        await pending
        assert ("installed post context" in str(chat.calls[-1])) is (mode == "success")
        if mode != "success":
            assert len(chat.calls) == old_calls
        if next_scope is not None:
            assert not lifecycle.context.blocks(next_scope, "model")
        assert db.get_run(child["id"])["status"] == "done"
    finally:
        release.touch()
        await asyncio.gather(pending, return_exceptions=True)
        await engine.close()
        db.close()
    assert not engine.processes.records and not native_console.service.live_runs()


async def _installed_compaction_case(
    native_console,
    native_package,
    tmp_path,
    mode,
    *,
    standalone=False,
    automatic=False,
    owned_case=None,
    monkeypatch=None,
):
    from Tests.Chat.test_console_context_compaction import (
        _real_selection_controller,
        _resolution,
    )
    from Tests.console_provider_doubles import with_destination
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
        ContextBudgetMode,
        ContextCompactionMode,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    entered, release = tmp_path / "compact-entered", tmp_path / "compact-release"
    initializing = mode in {"initializer_failure", "cancelled_initialization"}
    handlers = (
        []
        if mode == "no_hooks"
        else [
            _command(
                "post",
                "SessionStart" if initializing else "PostCompact",
                (
                    "raise SystemExit(1)"
                    if mode == "initializer_failure"
                    else _post_barrier(entered, release, mode)
                ),
                required=True,
                effects=[] if initializing else ["context"],
                timeout_seconds=30,
            )
        ]
    )
    plugin_service = owned_case.service if owned_case else native_console.service
    if owned_case is None:
        package = _package(native_package, handlers)
        installed = await install_components(
            plugin_service,
            package,
            ("skill:review",) if mode == "no_hooks" else ("hook:post",),
        )
    else:
        installed = owned_case.installed
    global_enable = await plugin_service.review_activation(
        installed.installation_id, workspace_id=None, intent="enabled"
    )
    assert (
        await plugin_service.commit(global_enable, global_enable.operation_id)
    ).committed
    db, repository, controller, store, session, conversation, _ = (
        _real_selection_controller(tmp_path, "owned-compact")
    )
    controller.app = native_console.controller.app
    controller._skills_service = native_console.skills
    if owned_case is not None:
        from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
        from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

        local = LocalSkillsService(
            store_dir=tmp_path / "compact-skills",
            plugin_service=plugin_service,
            allow_untrusted_without_trust_service=True,
        )
        controller._skills_service = SkillsScopeService(local_service=local)
        controller.app.local_skills_service = local
        controller.app.skills_scope_service = controller._skills_service
        controller.app.local_mcp_control_service = owned_case.local
        controller.app.unified_mcp_service = owned_case.unified
        loop = asyncio.get_running_loop()
        controller.app.call_from_thread = (
            lambda callback, *args: loop.call_soon_threadsafe(callback, *args)
        )
        compose = controller._compose_native_mcp_providers

        async def connect_owned(*args, **kwargs):
            providers = await compose(*args, **kwargs)
            for provider in providers.providers:
                if hasattr(provider, "snapshot"):
                    await provider.connect("mcp:peer")
            return providers

        monkeypatch.setattr(controller, "_compose_native_mcp_providers", connect_owned)
    controller._agent_runtime_enabled = False
    runtime = ConsoleRuntime(app=controller.app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    payloads = []

    async def resolve(_selection):
        return with_destination(_resolution())

    async def stream(_resolution, messages, **_kwargs):
        payloads.append(getattr(messages, "messages_payload", messages))
        yield "answer"

    controller.provider_gateway.resolve_for_send = resolve
    controller.provider_gateway.stream_chat = stream
    for _ in range(2):
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="question " + "x " * 450,
            persist=True,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer " + "y " * 450,
            persist=True,
        )
    target = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="current request",
        persist=True,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="current answer",
        persist=True,
    )
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=1800,
            compaction_mode=(
                ContextCompactionMode.AUTOMATIC
                if standalone or automatic
                else ContextCompactionMode.OFF
            ),
            summary_max_tokens=100,
        ),
    )
    maximum = controller.resolve_turn_configuration_snapshot(
        session.id
    ).skill_context_maximum
    assert any(
        row["plugin_installation_id"] == installed.installation_id
        for row in maximum["native_components"] + maximum["available_skills"]
    ), maximum
    fences = []
    captured_configurations = []
    original_fence = controller._manual_runtime_fence

    def observe_fence(**kwargs):
        captured_configurations.append(kwargs["configuration"])
        value = original_fence(**kwargs)
        fences.append(value)
        return value

    controller._manual_runtime_fence = observe_fence
    if mode in {"disabled", "selection_changed"}:
        original_auxiliary = controller.provider_gateway.complete_auxiliary

        async def authority_change(*args, **kwargs):
            result = await original_auxiliary(*args, **kwargs)
            if mode == "disabled":
                change = await native_console.service.review_activation(
                    installed.installation_id, workspace_id=None, intent="disabled"
                )
            else:
                change = await native_console.service.review_selection(
                    installed.installation_id, selection=("skill:review",)
                )
            assert (
                await native_console.service.commit(change, change.operation_id)
            ).committed
            return result

        controller.provider_gateway.complete_auxiliary = authority_change
    automatic_observations = []
    if automatic:
        import tldw_chatbook.Chat.console_chat_controller as controller_module

        decide = controller_module.decide_compaction
        hooks_for_compaction = controller._hooks_for_compaction

        def observe_decision(*args, **kwargs):
            result = decide(*args, **kwargs)
            automatic_observations.append(("decision", result, args, kwargs))
            return result

        async def observe_hooks(*args, **kwargs):
            reservation = controller._hooks_v2_submissions.get(asyncio.current_task())
            automatic_observations.append(("hooks", reservation))
            result = await hooks_for_compaction(*args, **kwargs)
            assert reservation == (None, None, session.id)
            assert (
                controller._hooks_v2_submissions[asyncio.current_task()] is reservation
            )
            assert result is None
            return result

        monkeypatch.setattr(controller_module, "decide_compaction", observe_decision)
        monkeypatch.setattr(controller, "_hooks_for_compaction", observe_hooks)
    prior_state = controller.run_state_for(session.id)
    if standalone:
        controller._set_run_state(prior_state, session_id=session.id)
    if mode == "nonce":
        release.touch()
    compact = asyncio.create_task(
        controller.compact_context_now(session.id)
        if standalone
        else (
            controller.submit_draft("automatic next", session_id=session.id)
            if automatic
            else controller.summarize_up_to(target.id)
        )
    )

    def accepted_result(value):
        return value[0] if standalone else value.accepted

    def assert_released():
        assert compact not in getattr(controller, "_hooks_v2_submissions", {})
        owner = runtime._hooks_v2_lifecycles.get(session.id)
        assert owner is None or owner.turn_scope is None
        if standalone:
            assert controller.run_state_for(session.id) is prior_state
            assert controller.run_state_for(session.id).is_send_allowed

    next_input = None
    try:
        if mode == "nonce":
            await compact
        await _wait(
            lambda: (
                controller.has_pending_approval_round(session.id)
                if owned_case
                else entered.exists()
            )
            or compact.done()
        )
        if initializing:
            if mode == "cancelled_initialization":
                assert entered.exists() and not compact.done()
                assert controller._hooks_v2_submissions[compact][0] is not None
                compact.cancel()
            result = await compact
            assert not accepted_result(result)
            assert_released()
            snapshots = controller._durable_context_snapshots(session.id)
            assert (
                repository.load_applicable_branch_memory(
                    conversation, frozenset(row.message_id for row in snapshots)
                ).memory
                is None
            )
            return
        if mode in {"disabled", "selection_changed"}:
            assert compact.done() and not accepted_result(compact.result())
            snapshots = controller._durable_context_snapshots(session.id)
            assert (
                repository.load_applicable_branch_memory(
                    conversation, frozenset(row.message_id for row in snapshots)
                ).memory
                is None
            )
            assert not entered.exists()
            return
        if mode == "nonce":
            assert len(fences) >= 2
            # Observe original caller arguments separately from its comparison copies.
            first, last = captured_configurations[0], captured_configurations[-1]
            assert (
                first.skill_context_maximum["plugin_run_id"]
                != last.skill_context_maximum["plugin_run_id"]
            )
            from dataclasses import replace

            assert fences[0] is not None and fences[-1] is not None
            without_nonce = lambda value: replace(
                value,
                skill_context_maximum={
                    key: item
                    for key, item in value.skill_context_maximum.items()
                    if key != "plugin_run_id"
                },
            )
            assert without_nonce(first) == without_nonce(last)
            assert fences[0][:12] == fences[-1][:12]
            assert fences[0][13:] == fences[-1][13:]
            assert fences[0] == fences[-1]
            assert compact.done() and accepted_result(compact.result())
            return
        if compact.done():
            differences = []
            if len(fences) > 1 and fences[0] is not None and fences[-1] is not None:
                differences = [
                    (index, left, right)
                    for index, (left, right) in enumerate(zip(fences[0], fences[-1]))
                    if left != right
                ]
            from dataclasses import fields

            detail = {}
            if differences:
                for field in fields(fences[0][12]):
                    left, right = getattr(fences[0][12], field.name), getattr(
                        fences[-1][12], field.name
                    )
                    if left != right:
                        if field.name == "skill_context_maximum":
                            detail[field.name] = {
                                key: (left.get(key), right.get(key))
                                for key in set(left) | set(right)
                                if left.get(key) != right.get(key)
                            }
                        else:
                            detail[field.name] = (left, right)
            print("MANUAL_COMPACTION_FENCE_DIFFERENCES", detail)
            assert accepted_result(compact.result()), (compact.result(), differences)
        if automatic:
            print(
                "AUTOMATIC_COMPACTION_OBSERVATIONS",
                automatic_observations,
                compact.result(),
            )
        snapshots = controller._durable_context_snapshots(session.id)
        state = repository.load_applicable_branch_memory(
            conversation, frozenset(row.message_id for row in snapshots)
        )
        assert (
            state.memory is not None
            and "installed post context" not in state.memory.summary_text
        )
        if mode == "no_hooks":
            assert compact.done() and accepted_result(compact.result())
            assert_released()
            assert runtime.get_hooks_v2(session.id) is None
            if automatic:
                assert len(automatic_observations) == 2
                assert automatic_observations[0][1].value == "automatic"
                assert payloads and controller.provider_gateway.calls == 1
                assert controller.run_state_for(session.id).is_send_allowed
            return
        engine = runtime.get_hooks_v2(session.id)
        assert (
            engine is not None and engine.native_hooks is not None
        ), "manual compaction committed SQLite memory but omitted captured installed hooks"
        assert (
            controller.has_pending_approval_round(session.id)
            if owned_case
            else entered.exists()
        ), "installed PostCompact did not reach its actual barrier"
        assert (
            engine.native_hooks.snapshots[0].installation_id
            == installed.installation_id
        )
        lifecycle = runtime._hooks_v2_lifecycles[session.id]
        reservation = controller._hooks_v2_submissions[compact]
        assert reservation == (lifecycle, lifecycle.turn_scope, session.id)
        assert not compact.done()
        if mode == "cancelled_post":
            compact.cancel()
            result = await compact
            assert not accepted_result(result)
            assert_released()
            return
        receiver = lifecycle.open_scope()
        next_input = asyncio.create_task(lifecycle.wait(receiver))
        await asyncio.sleep(0)
        assert not next_input.done() and not payloads and not compact.done()
        if mode == "stale":
            assert repository.deactivate_memory(
                state.memory.memory_id,
                expected_revision=state.memory.revision,
                reset_at="2026-10-16T00:00:00Z",
            )
        if owned_case is not None:
            assert not owned_case.unified.execution_log.read_recent()
            with controller._approval_state_lock:
                rounds = tuple(
                    (identity, value)
                    for identity, value in controller._pending_approval_rounds.items()
                    if value["session_id"] == session.id
                )
            assert len(rounds) == 1
            identity, approval = rounds[0]
            assert len(approval["calls"]) == 1
            assert approval["calls"][0].llm_name.startswith("plugin_mcp_")
            assert lifecycle.engine.mcp_executor is not None
            controller.resolve_pending_approval(
                {call.llm_name: "approve_once" for call in approval["calls"]},
                round_id=identity,
            )
        release.touch()
        result = await compact
        if owned_case is not None:
            audit = owned_case.unified.execution_log.read_recent()
            assert len(audit) == 1 and audit[0]["decision"] == "approved"
            assert audit[0]["error_category"] is None
        assert_released()
        if mode == "success":
            await next_input
            assert accepted_result(result)
            assert lifecycle._handoff is not None
            assert lifecycle._handoff[0] == reservation[1]
            if standalone and owned_case is None:
                post_witness = (entered.stat().st_mtime_ns, entered.read_bytes())
            assert (
                await controller.submit_draft("next", session_id=session.id)
            ).accepted
            assert "installed post context" in str(payloads)
            if standalone:
                assert lifecycle._handoff is None
                assert "installed post context" in str(payloads[-1])
                first_payload_count = len(payloads)
                assert (
                    await controller.submit_draft("following", session_id=session.id)
                ).accepted
                assert len(payloads) == first_payload_count + 1
                assert "installed post context" not in str(payloads[-1])
                assert lifecycle._handoff is None
                if owned_case is None:
                    assert (
                        entered.stat().st_mtime_ns,
                        entered.read_bytes(),
                    ) == post_witness
                else:
                    assert owned_case.unified.execution_log.read_recent() == audit
        elif mode == "failure":
            from tldw_chatbook.Agents.hooks_v2.checkpoints import HookCheckpointError

            with pytest.raises(HookCheckpointError):
                await next_input
            assert not payloads
            assert (
                repository.load_applicable_branch_memory(
                    conversation, frozenset(row.message_id for row in snapshots)
                ).memory
                is not None
            )
        else:
            from tldw_chatbook.Agents.hooks_v2.checkpoints import HookCheckpointError

            with pytest.raises(HookCheckpointError):
                await next_input
            assert not (
                await controller.submit_draft("next", session_id=session.id)
            ).accepted
            assert not payloads
            assert lifecycle.diagnostics["late_context"] >= 1
    finally:
        release.touch()
        await asyncio.gather(
            compact,
            *([next_input] if next_input is not None else []),
            return_exceptions=True,
        )
        await runtime.dispose()
        await controller.shutdown()
        with db.quiesce_connections(timeout_seconds=2.0):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["approved", "disconnected", "denied", "cancelled", "cyclic"]
)
async def test_installed_provisional_mcp_initialization(
    native_console, native_package, tmp_path, monkeypatch, mode
):
    from uuid import uuid4

    from tldw_chatbook.Agents.run_hooks import HookOutcome
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    rig = native_console
    accepted, terminal, stops, cards = [], [], [], []

    class LegacyObserver:
        async def fire_async(self, *_args, **_kwargs):
            return HookOutcome()

        def notify(self, event, **kwargs):
            if event == "Stop":
                stops.append(kwargs)

    recorder = LegacyObserver()
    rig.controller._ensure_run_hooks = lambda: recorder
    rig.controller.on_submission_accepted = lambda: accepted.append(rig.session.id)
    rig.controller.set_pending_approval = lambda payload: (
        cards.append(payload) if payload else None
    )
    loop = asyncio.get_running_loop()
    rig.controller.app.call_from_thread = (
        lambda callback, *args: loop.call_soon_threadsafe(callback, *args)
    )
    payload = {
        "content": [],
        "structuredContent": {
            "version": 2,
            "decision": "pass",
            "context": [
                {"text": "installed initializer context", "lifetime": "runtime"}
            ],
        },
    }

    @asynccontextmanager
    async def setup():
        if mode == "cyclic":
            launch_marker = tmp_path / "cyclic-peer-launched"
            peer_argv = child_argv(
                f"from pathlib import Path;Path({str(launch_marker)!r}).write_text('launched')"
            )
            assert not launch_marker.exists()
            package = _package(
                native_package,
                [_mcp_hook(payload, require_context=True)],
                requires={
                    "hook:initial": ["mcp:peer"],
                    "mcp:peer": ["hook:initial"],
                    "command:dependent": ["hook:initial"],
                },
                commands=("dependent", "independent"),
            )
            (package / "mcp.json").write_text(
                json.dumps(
                    {
                        "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                        "mcpServers": {
                            "peer": {
                                "type": "stdio",
                                "command": peer_argv[0],
                                "args": peer_argv[1:],
                            }
                        },
                    }
                )
            )
            installed = await install_components(
                rig.service,
                package,
                (
                    "mcp:peer",
                    "hook:initial",
                    "command:dependent",
                    "command:independent",
                ),
            )
            assert (
                "dependency_cycle"
                in installed.inspection.inventory["hook:initial"].activation_blockers
            )
            assert (
                "dependency_cycle"
                in installed.inspection.inventory["mcp:peer"].activation_blockers
            )
            yield SimpleNamespace(
                service=rig.service, installed=installed, launch_marker=launch_marker
            ), f"/{installed.alias}:dependent"
        else:
            async with _owned_peer(
                native_package,
                tmp_path,
                monkeypatch,
                [_mcp_hook(payload, require_context=True)],
                requires={"hook:initial": ["mcp:peer"]},
            ) as case:
                local = LocalSkillsService(
                    store_dir=tmp_path / "owned-skills",
                    plugin_service=case.service,
                    allow_untrusted_without_trust_service=True,
                )
                rig.controller._skills_service = SkillsScopeService(local_service=local)
                rig.controller.app.local_skills_service = local
                rig.controller.app.skills_scope_service = rig.controller._skills_service
                rig.controller.app.local_mcp_control_service = case.local
                rig.controller.app.unified_mcp_service = case.unified
                compose = rig.controller._compose_native_mcp_providers

                async def explicit_connection(*args, **kwargs):
                    providers = await compose(*args, **kwargs)
                    if mode != "disconnected":
                        for provider in providers.providers:
                            if hasattr(provider, "snapshot"):
                                await provider.connect("mcp:peer")
                    return providers

                monkeypatch.setattr(
                    rig.controller, "_compose_native_mcp_providers", explicit_connection
                )
                yield case, "hello"

    async with setup() as (case, draft):
        if mode == "cyclic":
            from tldw_chatbook.Plugins.admission import PluginUnavailable

            snapshot = await case.service.capture_mcp_snapshot(
                case.installed.installation_id, "workspace-a", "cycle-exclusion"
            )
            assert set(snapshot.selection) == {"command:independent"}
            maximum = case.service.capture_maximum("workspace-a")
            assert not maximum["available_skills"]
            assert {
                row["plugin_component_id"] for row in maximum["native_components"]
            } == {"command:independent"}
            admitted = await case.service.admit(maximum, "cycle-exclusion")
            independent = admitted["native_components"][0]
            blocked = dict(
                independent,
                record_id=f"plugin:{case.installed.installation_id}:command:dependent",
                name=f"{case.installed.alias}:dependent",
            )
            with pytest.raises(
                PluginUnavailable, match="plugin_component_not_admitted"
            ):
                await case.service.render_native(blocked)
            body = await case.service.render_native(independent)
            assert "independent body." in str(body)
            assert (
                not accepted and not terminal and not stops and not rig.gateway.payloads
            )
            assert not case.service.live_runs()
            assert not case.launch_marker.exists()
            return
        async with _runtime(rig) as runtime:
            request = ConsoleTurnCustodyRequest(
                turn_id=str(uuid4()),
                session_id=rig.session.id,
                draft=draft,
                configuration=rig.controller.resolve_turn_configuration_snapshot(
                    rig.session.id
                ),
            )
            turn = runtime.accept_turn(
                request,
                terminal_callback=terminal.append,
                recover_before_acceptance=False,
            )
            waiter = asyncio.create_task(runtime.wait_for_turn(turn))
            await asyncio.sleep(0)
            try:
                if mode != "cyclic":
                    await _wait(
                        lambda: rig.controller.has_pending_approval_round(
                            rig.session.id
                        )
                        or waiter.done()
                    )
                    with rig.controller._approval_state_lock:
                        rounds = tuple(
                            (identity, state)
                            for identity, state in rig.controller._pending_approval_rounds.items()
                            if state["session_id"] == rig.session.id
                        )
                    assert rounds and not waiter.done(), (
                        waiter.result() if waiter.done() else None
                    )
                    assert not accepted and not stops and not rig.gateway.payloads
                    engine = runtime.get_hooks_v2(rig.session.id)
                    assert (
                        engine.native_hooks.snapshots[0].installation_id
                        == case.installed.installation_id
                    )
                    lifecycle = engine.lifecycle_owner
                    assert not lifecycle.live and lifecycle._reservations
                    owned = tuple(case.service.live_runs())
                    if mode == "disconnected":
                        assert not owned and not case.ownership.connections
                    else:
                        assert owned and all(
                            row.installation_id == case.installed.installation_id
                            and row.workspace_id == "workspace-a"
                            and not row.completed.is_set()
                            for row in owned
                        )
                    assert not case.unified.execution_log.read_recent()
                    if mode == "cancelled":
                        waiter.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await waiter
                    else:
                        round_id, state = rounds[0]
                        rig.controller.resolve_pending_approval(
                            {
                                row.llm_name: (
                                    "approve_once"
                                    if mode in {"approved", "disconnected"}
                                    else "deny"
                                )
                                for row in state["calls"]
                            },
                            round_id=round_id,
                        )
                        result = await waiter
                        assert result.accepted is (mode == "approved"), result
                else:
                    result = await waiter
                    assert not result.accepted
                await _wait(lambda: bool(terminal))
                assert len(accepted) == len(stops) == (1 if mode == "approved" else 0)
                assert bool(rig.gateway.payloads) is (mode == "approved")
                assert (
                    "installed initializer context" in str(rig.gateway.payloads)
                ) is (mode == "approved")
                assert not runtime._turn_custody
                if mode != "cyclic":
                    audit = case.unified.execution_log.read_recent()
                    if mode == "denied":
                        assert len(audit) == 1 and audit[0]["decision"] == "denied"
                        assert audit[0]["error_category"] == "denied"
                    elif mode == "disconnected":
                        assert len(audit) == 1 and audit[0]["decision"] == "approved"
                        assert audit[0]["error_category"] == "execution_failed"
                    else:
                        assert len(audit) == (1 if mode == "approved" else 0), audit
                    if mode == "disconnected":
                        assert not case.ownership.connections
            finally:
                if not waiter.done():
                    waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)


@_REQUIRES_COMMAND_HOOKS
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "failure", "stale"])
async def test_installed_post_compaction_checkpoint(
    native_console, native_package, tmp_path, mode
):
    await _installed_compaction_case(native_console, native_package, tmp_path, mode)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("nonce", marks=_REQUIRES_COMMAND_HOOKS),
        "disabled",
        "selection_changed",
    ],
)
async def test_installed_compaction_authority_fence(
    native_console, native_package, tmp_path, mode
):
    await _installed_compaction_case(native_console, native_package, tmp_path, mode)


@_REQUIRES_COMMAND_HOOKS
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["allowed", "final_guard_denied", "invalid_schema", "changed_after_review"]
)
async def test_installed_final_arguments_reach_normal_permission_and_dispatch(
    native_package, tmp_path, monkeypatch, mode
):
    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_hooks_v2_tool_pipeline import CFG
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Chat.console_chat_controller import build_mcp_review_hook
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Plugins.components import tool_hook_requirements

    order = tmp_path / "transform-order"
    payload = {"content": [{"type": "text", "text": "B payload"}]}

    def authored(label, result):
        return (
            "import json,sys;from pathlib import Path;e=json.load(sys.stdin);"
            f"Path({str(order)!r}).open('a').write(json.dumps([{label!r},e['data']['tool_args']])+'\\n');"
            f"print({json.dumps(result)!r})"
        )

    handlers = [
        _command(
            "A",
            "PreToolUse",
            authored(
                "A",
                {
                    "version": 2,
                    "decision": "pass",
                    "updated_input": {"stage": "A", "payload": payload},
                },
            ),
            effects=["updated_input"],
            match={"provider": ["plugin_mcp"]},
        ),
        _command(
            "B",
            "PreToolUse",
            authored(
                "B",
                {
                    "version": 2,
                    "decision": "pass",
                    "updated_input": {
                        "stage": 4 if mode == "invalid_schema" else "B",
                        "payload": payload,
                    },
                },
            ),
            effects=["updated_input"],
            match={"provider": ["plugin_mcp"]},
        ),
        _command(
            "C",
            "PreToolUse",
            authored(
                "C",
                {
                    "version": 2,
                    "decision": "deny" if mode == "final_guard_denied" else "pass",
                },
            ),
            required=True,
            effects=["deny"],
            match={"provider": ["plugin_mcp"]},
        ),
        _command(
            "context",
            "PreToolUse",
            authored(
                "context",
                {
                    "version": 2,
                    "decision": "pass",
                    "context": [
                        {"text": "final candidate context", "lifetime": "turn"}
                    ],
                },
            ),
            effects=["context"],
            match={"provider": ["plugin_mcp"]},
        ),
    ]
    schema = {
        "type": "object",
        "properties": {"stage": {"type": "string"}, "payload": {"type": "object"}},
        "required": ["stage", "payload"],
        "additionalProperties": False,
    }
    async with _owned_peer(
        native_package, tmp_path, monkeypatch, handlers, schema=schema
    ) as case:
        await case.provider.connect("mcp:peer")
        engine = await _native_engine(
            case.service, case.installed.installation_id, "owned-case"
        )
        reviews, invoked = [], []

        def approvals(calls):
            reviews.extend(json.loads(json.dumps(call.arguments)) for call in calls)
            return {call.llm_name: "approve_once" for call in calls}

        normal = build_mcp_review_hook(case.provider, approvals)

        def review(calls, _run_id):
            result = normal(calls, _run_id)
            if mode == "changed_after_review" and any(
                call.name == case.name for call in calls
            ):
                calls[0].args["stage"] = "changed"
            return result

        original_invoke = case.registry.invoke_by_name

        def observe(name, args, **kwargs):
            if name == case.name:
                invoked.append(json.loads(json.dumps(args)))
            return original_invoke(name, args, **kwargs)

        monkeypatch.setattr(case.registry, "invoke_by_name", observe)
        db = AgentRunsDB(tmp_path / "final-args.db", client_id="installed-final")
        chat = ScriptedChat(
            [
                fence("load_tools", {"ids": [case.name]}),
                fence(case.name, {"stage": "original", "payload": payload}),
                "done",
            ]
        )
        service = AgentService(
            db,
            case.registry,
            chat_call=chat,
            review_tool_calls=review,
            hooks_v2_engine=engine,
            hooks_v2_session_id="owned-case",
            hooks_v2_turn_id="final-turn",
            hooks_v2_render_context=engine.native_hooks.render_context,
            hooks_v2_tool_requirements=lambda definition: tool_hook_requirements(
                case.registry, definition
            ),
        )
        before = len(case.unified.execution_log.read_recent())
        try:
            result = await asyncio.to_thread(
                service.run_turn,
                conversation_id="final",
                messages=[{"role": "user", "content": "go"}],
                config=replace(CFG, allowed_tools=(case.name,)),
                api_endpoint="test",
            )
            calls = [json.loads(line) for line in order.read_text().splitlines()]
            assert (
                [row[0] for row in calls].count("A")
                == [row[0] for row in calls].count("B")
                == 1
            )
            assert calls[0][1]["stage"] == "original" and calls[1][1]["stage"] == "A"
            if mode != "invalid_schema":
                assert next(row for row in calls if row[0] == "C")[1]["stage"] == "B"
            if mode in {"allowed", "changed_after_review"}:
                assert reviews == [{"stage": "B", "payload": payload}]
                assert (
                    next(row for row in calls if row[0] == "context")[1]["stage"] == "B"
                )
            else:
                assert not reviews
            assert invoked == (
                [{"stage": "B", "payload": payload}] if mode == "allowed" else []
            )
            assert len(case.unified.execution_log.read_recent()) == before + (
                mode == "allowed"
            )
            assert any(
                step.get("tool_outcome") == "success"
                and step.get("tool_name") == case.name
                for step in db.get_run(result[0])["steps"]
                if step.get("kind") == "tool_result"
            ) is (mode == "allowed")
        finally:
            await engine.close()
            db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "no_hooks",
        pytest.param("initializer_failure", marks=_REQUIRES_COMMAND_HOOKS),
        pytest.param("cancelled_initialization", marks=_REQUIRES_COMMAND_HOOKS),
        pytest.param("cancelled_post", marks=_REQUIRES_COMMAND_HOOKS),
    ],
)
async def test_installed_compaction_operation_cleanup(
    native_console, native_package, tmp_path, mode
):
    await _installed_compaction_case(native_console, native_package, tmp_path, mode)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [pytest.param("installed", marks=_REQUIRES_COMMAND_HOOKS), "no_hooks"]
)
async def test_installed_standalone_compact_now_releases_admission(
    native_console, native_package, tmp_path, mode
):
    await _installed_compaction_case(
        native_console,
        native_package,
        tmp_path,
        "success" if mode == "installed" else mode,
        standalone=True,
    )


@pytest.mark.asyncio
async def test_installed_owned_mcp_post_compaction(
    native_console, native_package, tmp_path, monkeypatch
):
    payload = {
        "content": [],
        "structuredContent": {
            "version": 2,
            "decision": "pass",
            "context": [{"text": "installed post context", "lifetime": "turn"}],
        },
    }
    handler = dict(_mcp_hook(payload, require_context=True), event="PostCompact")
    async with _owned_peer(
        native_package,
        tmp_path,
        monkeypatch,
        [handler],
        requires={"hook:initial": ["mcp:peer"]},
    ) as case:
        await _installed_compaction_case(
            native_console,
            native_package,
            tmp_path,
            "success",
            standalone=True,
            owned_case=case,
            monkeypatch=monkeypatch,
        )


@pytest.mark.asyncio
async def test_runtime_no_hook_automatic_compaction_preserves_submission(
    native_console, native_package, tmp_path, monkeypatch
):
    await _installed_compaction_case(
        native_console,
        native_package,
        tmp_path,
        "no_hooks",
        automatic=True,
        monkeypatch=monkeypatch,
    )


@pytest.mark.asyncio
async def test_installed_windows_command_refuses_before_root_access_or_launch(
    native_console, native_package, tmp_path, monkeypatch
):
    from tldw_chatbook.Agents.hooks_v2 import command_executor
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event
    from tldw_chatbook.Plugins.components import NativeHookProcessOwner

    launched = tmp_path / "unsupported-command-launched"
    package = _package(
        native_package,
        [
            _command(
                "unsupported",
                "SessionStart",
                f"from pathlib import Path;Path({str(launched)!r}).touch()",
                required=True,
            )
        ],
    )
    installed = await install_components(
        native_console.service, package, ("hook:unsupported",)
    )
    engine = await _native_engine(
        native_console.service, installed.installation_id, "windows-refusal"
    )
    reservations = []
    reserve = NativeHookProcessOwner.reserve_launch

    def observe_reserve(owner, event):
        reservations.append(event.event_id)
        return reserve(owner, event)

    assert not launched.exists()
    try:
        with monkeypatch.context() as patch:
            # Exercise only the executor's platform guard, not shared sys/platform state.
            patch.setattr(command_executor, "sys", SimpleNamespace(platform="win32"))
            patch.setattr(NativeHookProcessOwner, "reserve_launch", observe_reserve)
            result = await engine.fire_async(
                lifecycle_event(
                    "SessionStart",
                    "windows-refusal",
                    data={"reason": "startup"},
                    workspace_id="workspace-a",
                )
            )
        assert not result.allowed
        assert [failure.code for failure in result.failures] == ["unsupported_platform"]
        assert not reservations and not launched.exists()
        assert not engine.processes.records
        assert not any(engine.budget_owner.snapshot().values())
        assert not native_console.service.live_runs()
    finally:
        await engine.close()
