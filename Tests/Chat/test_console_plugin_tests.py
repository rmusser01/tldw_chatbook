"""Explicit tests require the real Console owner and retain exact reviewed inputs."""

import pytest

from Tests.Plugins.conftest import native_console as _native_console
from Tests.Plugins.conftest import native_package as _native_package
from Tests.private_profile import private_profile_test

native_console = _native_console
native_package = _native_package


@pytest.mark.asyncio
async def test_review_requires_existing_session_and_current_component(native_console):
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    rig = native_console
    runtime = ConsoleRuntime(app=rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    installed = await rig.install()
    with pytest.raises(ValueError, match="existing_console_session_required"):
        await runtime.prepare_plugin_test(
            "missing", installed.installation_id, "skill:review", kind="hook"
        )
    with pytest.raises(ValueError, match="test_component_kind_mismatch"):
        await runtime.prepare_plugin_test(
            rig.session.id, installed.installation_id, "skill:review", kind="hook"
        )
    assert not rig.gateway.payloads
    assert rig.controller.in_flight_run_count() == 0


@private_profile_test
@pytest.mark.asyncio
async def test_reviewed_real_hook_runs_without_a_provider_or_conversation_effect(
    request, native_console, native_package, tmp_path
):
    import json
    import sys
    from dataclasses import replace

    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    rig = native_console
    package = native_package()
    ns = "io.github.rmusser01.chatbook"
    folder = package / ns
    folder.mkdir()
    marker = tmp_path / "hook-executed"

    def output_code(lifetime, label):
        return (
            "import json,sys;event=json.load(sys.stdin);"
            f"text=('TEST_{label}' if event['origin']=='host_plugin_test' else 'LIVE_{label}');"
            f"print(json.dumps({{'version':2,'decision':'pass','context':[{{'text':text,'lifetime':{lifetime!r}}}]}}))"
        )

    (folder / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "check",
                        "event": "UserPromptSubmit",
                        "type": "command",
                        "effects": ["context"],
                        "argv": [
                            sys.executable,
                            "-c",
                            f"from pathlib import Path;Path({str(marker)!r}).write_text('ran');"
                            + output_code("turn", "HANDLER"),
                        ],
                    },
                    {
                        "id": "start",
                        "event": "SessionStart",
                        "type": "command",
                        "effects": ["context"],
                        "argv": [sys.executable, "-c", output_code("runtime", "START")],
                    },
                ],
            }
        )
    )
    manifest = json.loads((package / "plugin.json").read_text())
    manifest["extensions"][ns]["hooks"] = f"./{ns}/hooks.json"
    (package / "plugin.json").write_text(json.dumps(manifest))
    install = await rig.service.review_install(
        package,
        selection=("hook:check", "hook:start"),
        workspace_id="workspace-a",
        enable_in_workspace=True,
    )
    await rig.service.commit(install, install.operation_id)
    trust = await rig.service.review_trust(install.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    runtime = ConsoleRuntime(app=rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    before = tuple(rig.store.messages_for_session(rig.session.id))
    try:
        review = await runtime.prepare_plugin_test(
            rig.session.id,
            install.installation_id,
            "hook:check",
            kind="hook",
            data={"prompt": "synthetic test"},
        )
        assert not marker.exists()
        with pytest.raises(ValueError, match="review_unavailable"):
            await runtime.execute_plugin_test(
                replace(review, component_id="hook:other")
            )
        outcome = await runtime.execute_plugin_test(review)
        assert outcome.succeeded, outcome
        assert marker.read_text() == "ran"
        assert not rig.gateway.payloads
        assert tuple(rig.store.messages_for_session(rig.session.id)) == before
        assert rig.controller.in_flight_run_count() == 0
        assert await runtime.execute_plugin_test(review) is outcome
        assert runtime.get_hooks_v2(rig.session.id) is None
        assert rig.session.id not in runtime._hooks_v2_lifecycles
        ordinary = await rig.controller.submit_draft(
            "ordinary follow-up", session_id=rig.session.id
        )
        assert ordinary.accepted
        assert rig.gateway.payloads
        sent = str(rig.gateway.payloads[-1])
        assert "TEST_START" not in sent and "TEST_HANDLER" not in sent
        assert "LIVE_START" in sent and "LIVE_HANDLER" in sent
        existing_engine = runtime.get_hooks_v2(rig.session.id)
        existing_lifecycle = runtime._hooks_v2_lifecycles[rig.session.id]
        second = await runtime.prepare_plugin_test(
            rig.session.id,
            install.installation_id,
            "hook:check",
            kind="hook",
            data={"prompt": "second synthetic input"},
        )
        assert (await runtime.execute_plugin_test(second)).succeeded
        assert runtime.get_hooks_v2(rig.session.id) is existing_engine
        assert runtime._hooks_v2_lifecycles[rig.session.id] is existing_lifecycle
        assert (
            await rig.controller.submit_draft(
                "another ordinary follow-up", session_id=rig.session.id
            )
        ).accepted
        sent = str(rig.gateway.payloads[-1])
        assert "TEST_START" not in sent and "TEST_HANDLER" not in sent
        assert "LIVE_START" in sent and "LIVE_HANDLER" in sent
    finally:
        await runtime.close_hooks_v2()


@private_profile_test
@pytest.mark.asyncio
async def test_configuration_save_then_reviewed_connection_discovery_grants_no_tools(
    request, native_console, native_package, tmp_path, monkeypatch
):
    import json
    import os
    from pathlib import Path
    from uuid import uuid4

    from Tests.hooks_v2_process_support import child_argv
    from Tests.Plugins.test_owned_mcp_tools import SERVER
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )
    from tldw_chatbook.UI.Plugins_Modules.operations import PluginOperations
    from tldw_chatbook.UI.Plugins_Modules.review import PluginConnectionDraft

    rig = native_console
    client = MCPClient()
    local = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "mcp.json"),
        client=client,
        manifest_provider=dict,
    )
    rig.service._mcp_mapping_owner = local
    await rig.service._call(
        lambda: setattr(rig.service._coordinator, "mcp_mapping_owner", local)
    )
    rig.controller.app.unified_mcp_service = UnifiedMCPControlPlaneService(
        local_service=local, server_service=None, target_store=None, context_store=None
    )
    argv = child_argv(SERVER)
    monkeypatch.setenv(
        "PATH", str(Path(argv[0]).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    package = native_package()
    (package / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {
                    "shared": {
                        "type": "stdio",
                        "command": Path(argv[0]).name,
                        "args": argv[1:],
                        "env": {"CALL_LOG": str(tmp_path / "calls")},
                    }
                },
            }
        )
    )
    install = await rig.service.review_install(
        package,
        selection=("mcp:shared",),
        workspace_id="workspace-a",
        enable_in_workspace=True,
    )
    await rig.service.commit(install, install.operation_id)
    trust = await rig.service.review_trust(install.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    details = await rig.service.management_details(
        install.installation_id, workspace_id="workspace-a"
    )
    operations = PluginOperations(rig.service)
    draft = PluginConnectionDraft(
        uuid4().hex,
        install.installation_id,
        "workspace-a",
        "mcp:shared",
        details["inspection"],
        protocol_version="2025-03-26",
    )
    saved = await operations.save_profile(draft, local)
    assert not client.sessions
    assert not (tmp_path / "launch.json").exists()
    mapped = await rig.service.review_configuration(
        install.installation_id, connections={"mcp:shared": saved["profile_id"]}
    )
    await rig.service.commit(mapped, mapped.operation_id)
    runtime = ConsoleRuntime(app=rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    before = tuple(rig.store.messages_for_session(rig.session.id))
    try:
        review = await runtime.prepare_plugin_test(
            rig.session.id, install.installation_id, "mcp:shared", kind="connection"
        )
        assert not client.sessions
        outcome = await runtime.execute_plugin_test(review)
        assert outcome["discovery"]["tools"][0]["name"] == "echo"
        assert outcome["cleanup_pending"] == ()
        assert not client.sessions
        assert not rig.gateway.payloads
        assert tuple(rig.store.messages_for_session(rig.session.id)) == before
        details = await rig.service.management_details(
            install.installation_id, workspace_id="workspace-a"
        )
        assert not any(row["kind"] == "tool" for row in details["mappings"])
        # Discovery only enables a NEW exact mapping review, never permission.
        tools = await rig.service.review_configuration(
            install.installation_id,
            connections={"mcp:shared": saved["profile_id"]},
            tools={"mcp:shared": ("echo",)},
        )
        assert tools.kind == "configure"
        assert not rig.gateway.payloads
    finally:
        await runtime.close_hooks_v2()
        await client.disconnect_all()


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("busy", ["received", "running"])
async def test_explicit_admission_refuses_real_occupied_slot_without_consuming_it(
    request, native_console, busy
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    rig = native_console
    runtime = ConsoleRuntime(app=rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    configuration = await rig.controller.capture_turn_configuration_snapshot(
        rig.session.id
    )
    claim = None
    if busy == "received":
        claim = rig.store.claim_received_turn(rig.session.id, "ordinary-received-turn")
        assert claim is not None
    else:
        rig.controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Ordinary preparation"),
            session_id=rig.session.id,
        )
    try:
        with pytest.raises(
            RuntimeError, match="console_(received_turn_pending|session_busy)"
        ):
            async with rig.controller.plugin_test_admission(configuration):
                pytest.fail("occupied session admitted an explicit plugin test")
        assert not getattr(rig.controller, "_hooks_v2_submissions", {})
        if claim is not None:
            assert rig.store.received_turn_for_session(rig.session.id) is claim
            assert not claim.sealed
        assert not rig.gateway.payloads
    finally:
        if claim is not None:
            rig.store.release_received_turn(claim)
        rig.controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STOPPED, "Control released"),
            session_id=rig.session.id,
        )
        await runtime.close_hooks_v2()


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_stock_app_test_admission_uses_actual_async_capture_and_session():
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    app._cli_no_splash = True
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        runtime = app.console_runtime
        controller = runtime._chat_controller
        assert controller is not None
        store = runtime.chat_store
        session = next(
            row for row in store.sessions() if row.id == store.active_session_id
        )
        assert controller._turn_context_provider is None
        configuration = await controller.capture_turn_configuration_snapshot(session.id)
        assert configuration.session_id == session.id
        assert next(row for row in store.sessions() if row.id == session.id) is session
        before = tuple(store.messages_for_session(session.id))
        async with controller.plugin_test_admission(configuration):
            import asyncio

            assert (
                controller._hooks_v2_submissions[asyncio.current_task()][2]
                == session.id
            )
        assert tuple(store.messages_for_session(session.id)) == before
        assert not controller._hooks_v2_submissions
        assert controller.in_flight_run_count() == 0
