"""Owned MCP tools use actual scoped authority and the ordinary permission seam."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.hooks_v2_process_support import child_argv

SERVER = r"""
import json, os, sys, threading, time
from pathlib import Path
output = threading.Lock()
def reply(message):
    method = message.get("method")
    if "id" not in message:
        return
    if method == "initialize":
        base = Path(os.environ["CALL_LOG"]).parent
        (base / "launch.json").write_text(json.dumps({"argv": sys.argv, "cwd": os.getcwd(), "root": os.environ.get("PLUGIN_ROOT"), "data": os.environ.get("PLUGIN_DATA"), "portable": os.environ.get("PORTABLE")}))
        if (base / "init-block").exists():
            (base / "init-started").touch()
            while not (base / "init-release").exists():
                time.sleep(.01)
        result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "owned-test", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "Controlled echo", "inputSchema": {"type": "object", "properties": {"hold": {"type": "string"}, "started": {"type": "string"}}}}]}
    elif method in ("resources/list", "prompts/list"):
        result = {method.split("/")[0]: []}
    elif method == "tools/call":
        args = message["params"].get("arguments", {})
        with output:
            with open(os.environ["CALL_LOG"], "a") as log:
                log.write(json.dumps(args) + "\n")
        if args.get("hold"):
            Path(args["started"]).touch()
            while not Path(args["hold"]).exists():
                time.sleep(.01)
        result = {"content": [{"type": "text", "text": "ok"}]}
    else:
        result = {}
    with output:
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
for line in sys.stdin:
    threading.Thread(target=reply, args=(json.loads(line),), daemon=True).start()
"""


@pytest.fixture
async def shared_connection_case(native_package, tmp_path, monkeypatch, request):
    import os
    from pathlib import Path

    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.connection_ownership import ConnectionOwnership
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.mcp_provider import PluginMCPProvider
    from tldw_chatbook.Plugins.recovery import retained_inspections
    from tldw_chatbook.Plugins.revocation import RevocationTarget
    from tldw_chatbook.Plugins.service import PluginService

    client = MCPClient()
    local = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "mcp.json"),
        client=client,
        manifest_provider=dict,
    )
    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: SimpleNamespace(archived=False),
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
        mcp_mapping_owner=local,
    )
    await service.bootstrap("test passphrase")
    ownership = ConnectionOwnership(plugin_service=service, local_service=local)
    argv = child_argv(SERVER)
    monkeypatch.setenv(
        "PATH", str(Path(argv[0]).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    package = native_package()
    definition = {
        "type": "stdio",
        "command": Path(argv[0]).name,
        "args": argv[1:],
        "env": {"CALL_LOG": str(tmp_path / "calls")},
    }
    fixture_mode = getattr(request, "param", "request_independent")
    if fixture_mode in {"portable", "portable-literals"}:
        definition["args"] += ["", " padded ", "${PLUGIN_ROOT}", "${PLUGIN_DATA}"]
        definition["env"]["PORTABLE"] = "${PLUGIN_ROOT}/sub space"
        definition["cwd"] = "${PLUGIN_DATA}"
        if fixture_mode == "portable-literals":
            definition["args"] += ["${UNRECOGNIZED}"]
    (package / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {"shared": definition},
            }
        )
    )
    review = await service.review_install(
        package, selection=("skill:review", "mcp:shared"), workspace_id="a"
    )
    assert (await service.commit(review, review.operation_id)).committed
    trust = await service.review_trust(review.installation_id)
    assert (await service.commit(trust, trust.operation_id)).committed
    if fixture_mode == "default":
        active = await service.review_activation(
            review.installation_id, workspace_id=None, intent="enabled"
        )
        assert (await service.commit(active, active.operation_id)).committed
    for workspace in ("a", "b"):
        active = await service.review_activation(
            review.installation_id,
            workspace_id=workspace,
            intent=(
                "inherit"
                if fixture_mode == "default" and workspace == "a"
                else "enabled"
            ),
        )
        assert (await service.commit(active, active.operation_id)).committed
    root_review = await service.review_data_creation(
        review.installation_id, workspace_id=None
    )
    data_root = await service.create_data(root_review, root_review.operation_id)
    inspection = await service._call(
        lambda: retained_inspections(service._coordinator.published_snapshot())[
            (review.installation_id, review.inspection.effective_digest)
        ]
    )
    profile = local.save_owned_profile(
        installation_id=review.installation_id,
        inspection=inspection,
        component_id="mcp:shared",
        data_root=data_root,
        session_isolation=(
            "separate" if fixture_mode == "separate" else "request_independent"
        ),
        protocol_version="2025-03-26",
    )
    profile_id = profile["profile_id"]
    assert not client.sessions, "configuration save must not launch"
    assert not ownership.is_connected(profile_id)
    configured = await service.review_configuration(
        review.installation_id, connections={"mcp:shared": profile_id}
    )
    assert (await service.commit(configured, configured.operation_id)).committed
    unified = UnifiedMCPControlPlaneService(
        local_service=local, server_service=None, target_store=None, context_store=None
    )
    discovery_snapshot = await service.capture_mcp_snapshot(
        review.installation_id, "a", "discovery"
    )
    discovery = PluginMCPProvider(
        plugin_service=service,
        ownership=ownership,
        snapshot=discovery_snapshot,
        service=unified,
        main_loop=asyncio.get_running_loop(),
    )
    assert (await discovery.connect("mcp:shared"))["tools"][0]["name"] == "echo"
    await discovery.compose_catalog()
    assert discovery.list_catalog() == [], "discovery alone must not advertise tools"
    for connection in tuple(ownership.connections.values()):
        for owner in tuple(connection.owners):
            await ownership.detach(connection.connection_id, owner)
    configured = await service.review_configuration(
        review.installation_id,
        connections={"mcp:shared": profile_id},
        tools={"mcp:shared": ("echo",)},
    )
    assert (await service.commit(configured, configured.operation_id)).committed
    snapshots, providers, registries = {}, {}, {}
    for workspace in ("a", "b"):
        snapshots[workspace] = await service.capture_mcp_snapshot(
            review.installation_id, workspace, "run-" + workspace
        )
        provider = PluginMCPProvider(
            plugin_service=service,
            ownership=ownership,
            snapshot=snapshots[workspace],
            service=unified,
            main_loop=asyncio.get_running_loop(),
            approval_callback=lambda calls: {
                call.llm_name: "approve_once" for call in calls
            },
        )
        await provider.compose_catalog()
        providers[workspace] = provider
        registry = ToolCatalogRegistry()
        registry.register_provider(provider)
        registries[workspace] = registry
    release, started = tmp_path / "release", tmp_path / "started"
    pending = None

    async def invoke(workspace, arguments=None):
        provider = providers[workspace]
        return await asyncio.to_thread(
            registries[workspace].invoke_by_name,
            provider.list_catalog()[0].name,
            arguments or {},
        )

    async def disable_a():
        return await service.disable(
            RevocationTarget(review.installation_id, "a", False)
        )

    try:
        assert (await invoke("a")).ok
        assert (await invoke("b")).ok
        # The controlled original request outlives the test's explicit revocation.
        # Crypto-backed admission is not part of the native deadline under test.
        for session in client.sessions.values():
            session.request_timeout_seconds = 60.0
        pending = asyncio.create_task(
            invoke("a", {"hold": str(release), "started": str(started)})
        )
        for _ in range(500):
            if started.exists():
                break
            await asyncio.sleep(0.01)
        assert started.exists(), "controlled A request did not reach the real server"
        yield SimpleNamespace(
            disable_a=disable_a,
            invoke=invoke,
            pending=pending,
            release=release,
            client=client,
            service=service,
            ownership=ownership,
            installation=review.installation_id,
            snapshots=snapshots,
            data_root=data_root,
            local=local,
            profile_id=profile_id,
            providers=providers,
            registries=registries,
            calls=tmp_path / "calls",
            package=package,
            inspection=inspection,
            unified=unified,
        )
    finally:
        release.touch()
        if pending is not None:
            await pending
        for connection in tuple(ownership.connections.values()):
            for owner in tuple(connection.owners):
                await ownership.detach(connection.connection_id, owner)
        await asyncio.gather(
            *(
                connection.close
                for connection in ownership.connections.values()
                if connection.close is not None
            )
        )
        for _ in range(100):
            if not any(
                operation.cleanup_tasks
                for operation in service.fences.operations.values()
            ):
                break
            await asyncio.sleep(0.01)
        await client.disconnect_all()
        await service.aclose()


@pytest.mark.asyncio
async def test_detaching_a_keeps_b_transport_alive(shared_connection_case):
    case = shared_connection_case
    await case.disable_a()
    assert len(case.client.sessions) == 1
    rows = await case.service._call(
        lambda: case.service._coordinator.owner.list_processes(limit=50, offset=0)
    )
    uncertain = [
        row
        for row in rows
        if row["workspace_id"] == "a"
        and row["state"] == "published"
        and (row.get("provenance") or {}).get("outcome") == "uncertain"
    ]
    assert uncertain, "A's unknown remote outcome must retain durable request custody"
    blockers = await case.service._call(
        lambda: case.service._coordinator.root_usage.blockers((case.data_root,))
    )
    assert blockers, "shared process/request root grants must remain held"
    assert (await case.invoke("b")).ok
    with pytest.raises(PermissionError):
        await case.service.check_mcp_snapshot(case.snapshots["a"], "mcp:shared")
    case.release.touch()
    late = await case.pending
    assert not late.ok, (
        "revoked A result must be rejected at the owned provider boundary"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_connection_case", ["separate"], indirect=True)
async def test_unqualified_scopes_use_separate_actual_processes(shared_connection_case):
    case = shared_connection_case
    assert len(case.client.sessions) == 2
    pids = {session.process.pid for session in case.client.sessions.values()}
    assert len(pids) == 2
    await case.disable_a()
    assert (await case.invoke("b")).ok


@pytest.mark.asyncio
async def test_owned_service_guards_and_normal_permission_denial(
    shared_connection_case,
):
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    case = shared_connection_case
    profile = case.local.store.get_profile(case.profile_id)
    with pytest.raises(PermissionError, match="plugin_owned_profile"):
        case.local.save_external_profile(profile)
    with pytest.raises(PermissionError, match="plugin_owned_profile"):
        case.local.delete_external_profile(case.profile_id)
    with pytest.raises(PermissionError, match="plugin_connection_authority_required"):
        await case.local.connect_profile(case.profile_id)
    standalone = MCPToolProvider(
        service=case.unified, main_loop=asyncio.get_running_loop()
    )
    await standalone.compose_catalog()
    assert standalone.list_catalog() == []
    before = case.calls.read_text()
    case.unified.set_tool_state("local:" + case.profile_id, "echo", "deny")
    denied = await case.invoke("b")
    assert not denied.ok and denied.dispatch_state == "not_started"
    assert case.calls.read_text() == before
    case.unified.set_tool_state("local:" + case.profile_id, "echo", "ask")
    assert (await case.invoke("b")).ok


@pytest.mark.asyncio
async def test_permission_change_during_approval_refuses_final_dispatch(
    shared_connection_case,
):
    case = shared_connection_case
    before = case.calls.read_text()

    def approve_after_revocation(calls):
        case.unified.set_tool_state("local:" + case.profile_id, "echo", "deny")
        return {call.llm_name: "approve_once" for call in calls}

    case.providers["b"]._approval_callback = approve_after_revocation
    result = await case.invoke("b")
    assert not result.ok and result.dispatch_state == "not_started"
    assert case.calls.read_text() == before


@pytest.mark.asyncio
async def test_owned_test_connection_keeps_other_live_scope(shared_connection_case):
    from tldw_chatbook.MCP.connection_ownership import (
        OwnedMCPInvocation,
        owned_invocation,
    )

    case = shared_connection_case
    snapshot = await case.service.capture_mcp_snapshot(
        case.installation, "b", "test-only"
    )
    token = owned_invocation.set(
        OwnedMCPInvocation(case.ownership, snapshot, "mcp:shared")
    )
    try:
        assert (await case.local.test_external_profile(case.profile_id))["tools"] == 1
    finally:
        owned_invocation.reset(token)
    assert all(
        "test-only:b" not in connection.owners
        for connection in case.ownership.connections.values()
    )
    assert (await case.invoke("b")).ok


@pytest.mark.asyncio
async def test_exact_tool_mapping_recovers_and_changed_definition_refuses(
    shared_connection_case,
):
    case = shared_connection_case
    receipts = await case.service._call(case.service._coordinator.recover)
    assert not any(item.phase == "recovery_required" for item in receipts)
    snapshot = case.local.store.get_discovery_snapshot(case.profile_id)
    changed = json.loads(json.dumps(snapshot))
    changed["tools"][0]["description"] = "Changed authority"
    case.local.store.save_discovery_snapshot(case.profile_id, changed)
    try:
        before = case.calls.read_text()
        refused = await case.invoke("b")
        assert not refused.ok and refused.dispatch_state == "not_started"
        assert case.calls.read_text() == before
        receipts = await case.service._call(case.service._coordinator.recover)
        assert any(item.phase == "recovery_required" for item in receipts)
    finally:
        case.local.store.save_discovery_snapshot(case.profile_id, snapshot)
        await case.service._call(case.service._coordinator.recover)


@pytest.mark.asyncio
async def test_public_revision_apply_closes_idle_process_after_requests_settle(
    shared_connection_case,
):
    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok
    original = next(iter(case.client.sessions.values())).process
    skill = case.package / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nRevision for idle MCP drain.\n")
    review = await case.service.review_revision(case.installation, case.package)
    receipt = await asyncio.wait_for(
        case.service.apply_revision(review, review.operation_id), 5
    )
    assert receipt.committed
    assert original.returncode is not None
    assert not case.client.sessions


@pytest.mark.asyncio
async def test_pending_launch_is_owned_before_scope_disable(shared_connection_case):
    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok
    for connection in tuple(case.ownership.connections.values()):
        for owner in tuple(connection.owners):
            await case.ownership.detach(connection.connection_id, owner)
    base = case.calls.parent
    (base / "init-block").touch()
    launch = asyncio.create_task(case.providers["a"].connect("mcp:shared"))
    try:
        for _ in range(500):
            if (base / "init-started").exists():
                break
            await asyncio.sleep(0.01)
        assert (base / "init-started").exists()
        child = next(iter(case.client._pending_connections.values())).process
        blockers = await case.service._call(
            lambda: case.service._coordinator.root_usage.blockers((case.data_root,))
        )
        assert blockers and child.returncode is None
        receipt = await case.disable_a()
        assert receipt.cleanup_pending
        (base / "init-block").unlink()
        assert (await case.invoke("b")).ok
    finally:
        (base / "init-release").touch()
        with pytest.raises(PermissionError):
            await launch
        for _ in range(500):
            if child.returncode is not None:
                break
            await asyncio.sleep(0.01)
        assert child.returncode is not None


@pytest.mark.asyncio
async def test_global_disable_stops_all_actual_children(shared_connection_case):
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    case = shared_connection_case
    processes = [session.process for session in case.client.sessions.values()]
    request = case.service.begin_disable(
        RevocationTarget(case.installation, None, True)
    )
    await case.service.finish_revocation(request)
    for _ in range(500):
        if all(process.returncode is not None for process in processes):
            break
        await asyncio.sleep(0.01)
    assert all(process.returncode is not None for process in processes)
    assert not (await case.invoke("b")).ok
    assert not (await case.pending).ok


@asynccontextmanager
async def http_owned_case(
    tmp_path,
    native_package,
    url,
    *,
    headers=None,
    credentials=None,
    binding=None,
    second=False,
    dependent=False,
):
    from tldw_chatbook.MCP.client import MCPClient
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

    client = MCPClient()
    local = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "http.json"),
        client=client,
        manifest_provider=dict,
        credential_service=credentials,
    )
    service = PluginService(
        tmp_path / "http-profile",
        workspace_lookup=lambda _: SimpleNamespace(archived=False),
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "http-marker"),
        accept_reduced_protection=True,
        mcp_mapping_owner=local,
    )
    await service.bootstrap("test passphrase")
    ownership = ConnectionOwnership(plugin_service=service, local_service=local)
    try:
        package = native_package(
            requires={"mcp:other": ["mcp:remote"]} if dependent else None
        )
        definitions = {
            "remote": {"type": "streamable-http", "url": url, "headers": headers or {}}
        }
        if second:
            definitions["other"] = {"type": "streamable-http", "url": url}
        components = tuple("mcp:" + name for name in definitions)
        if second:
            definitions["unused"] = {"type": "streamable-http", "url": url}
        (package / "mcp.json").write_text(
            json.dumps(
                {
                    "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                    "mcpServers": definitions,
                }
            )
        )
        review = await service.review_install(
            package, selection=("skill:review", *components), workspace_id="a"
        )
        await service.commit(review, review.operation_id)
        trust = await service.review_trust(review.installation_id)
        await service.commit(trust, trust.operation_id)
        for workspace in ("a", "b"):
            active = await service.review_activation(
                review.installation_id, workspace_id=workspace, intent="enabled"
            )
            await service.commit(active, active.operation_id)
        inspection = await service._call(
            lambda: retained_inspections(service._coordinator.published_snapshot())[
                (review.installation_id, review.inspection.effective_digest)
            ]
        )
        profile_ids = {}
        for component in components:
            profile = local.save_owned_profile(
                installation_id=review.installation_id,
                inspection=inspection,
                component_id=component,
                session_isolation="request_independent",
                development_loopback=True,
                credential_reference=binding.reference_id if binding else None,
                credential_generation=binding.authority_generation if binding else None,
            )
            profile_ids[component] = profile["profile_id"]
        assert not client.sessions
        configured = await service.review_configuration(
            review.installation_id, connections=profile_ids
        )
        await service.commit(configured, configured.operation_id)
        unified = UnifiedMCPControlPlaneService(
            local_service=local,
            server_service=None,
            target_store=None,
            context_store=None,
        )
        snapshot = await service.capture_mcp_snapshot(
            review.installation_id, "a", "discover-http"
        )
        provider = PluginMCPProvider(
            plugin_service=service,
            ownership=ownership,
            snapshot=snapshot,
            service=unified,
            main_loop=asyncio.get_running_loop(),
        )
        for component in components:
            await provider.connect(component)
        for connection in tuple(ownership.connections.values()):
            for owner in tuple(connection.owners):
                await ownership.detach(connection.connection_id, owner)
        configured = await service.review_configuration(
            review.installation_id,
            connections=profile_ids,
            tools={component: ("echo",) for component in components},
        )
        await service.commit(configured, configured.operation_id)
        providers = {}
        for workspace in ("a", "b"):
            snapshot = await service.capture_mcp_snapshot(
                review.installation_id, workspace, "http-" + workspace
            )
            provider = PluginMCPProvider(
                plugin_service=service,
                ownership=ownership,
                snapshot=snapshot,
                service=unified,
                main_loop=asyncio.get_running_loop(),
                approval_callback=lambda calls: {
                    call.llm_name: "approve_once" for call in calls
                },
            )
            await provider.compose_catalog()
            providers[workspace] = provider

        async def invoke(workspace):
            provider = providers[workspace]
            return await asyncio.to_thread(
                provider.invoke, provider.list_catalog()[0].id, {}
            )

        yield SimpleNamespace(
            client=client,
            local=local,
            service=service,
            ownership=ownership,
            providers=providers,
            invoke=invoke,
            installation=review.installation_id,
            unified=unified,
            profile_ids=profile_ids,
        )
    finally:
        for connection in tuple(ownership.connections.values()):
            for owner in tuple(connection.owners):
                await ownership.detach(connection.connection_id, owner)
        await client.disconnect_all()
        # The fixture owner observes the controlled peer's completed handlers.
        # This is explicit test cleanup, not production proof from pool closure.
        for connection in tuple(ownership.connections.values()):
            for request in connection.requests.values():
                if request.outcome not in {"settled", "not_started"}:
                    await ownership._settle(request.token)
                    request.outcome = "settled"
        await service.aclose()


@pytest.mark.asyncio
async def test_http_lost_response_retains_a_after_pool_close_and_b_survives(
    tmp_path, native_package, monkeypatch
):
    from Tests.MCP.test_streamable_http import peer
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    class Fault:
        value = None

        def __eq__(self, other):
            return self.value == other

        def __hash__(self):
            return hash(self.value)

    fault = Fault()
    async with peer(monkeypatch, fault=fault) as (  # noqa: SIM117
        url,
        _messages,
        calls,
    ):
        async with http_owned_case(tmp_path, native_package, url) as case:
            assert (await case.invoke("a")).ok
            assert (await case.invoke("b")).ok
            fault.value = "lost"
            result = await case.invoke("a")
            assert not result.ok and result.dispatch_state == "uncertain"
            fault.value = None
            await case.service.disable(RevocationTarget(case.installation, "a", False))
            assert (await case.invoke("b")).ok
            assert calls["count"] == 4, "uncertain invocation must never replay"
            connection = next(
                value for value in case.ownership.connections.values() if value.owners
            )
            session = connection.session
            await case.ownership.detach(connection.connection_id, "http-b:b")
            assert session._cleanup_complete
            rows = await case.service._call(
                lambda: case.service._coordinator.owner.list_processes(
                    limit=50, offset=0
                )
            )
            assert any(
                row["workspace_id"] == "a"
                and row["state"] == "published"
                and (row.get("provenance") or {}).get("outcome") == "uncertain"
                for row in rows
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_connection_case", ["portable"], indirect=True)
async def test_portable_launch_preserves_argv_env_cwd_and_host_variables(
    shared_connection_case,
):
    case = shared_connection_case
    launch = json.loads((case.calls.parent / "launch.json").read_text())
    assert launch["argv"][-4:] == [
        "",
        " padded ",
        case.inspection.materialized_identity,
        str(case.data_root.path),
    ]
    assert launch["root"] == case.inspection.materialized_identity
    assert launch["data"] == launch["cwd"] == str(case.data_root.path)
    assert launch["portable"] == case.inspection.materialized_identity + "/sub space"
    assert (await case.invoke("b")).ok


@pytest.mark.asyncio
async def test_stored_configuration_cannot_masquerade_as_retained_definition(
    shared_connection_case,
):
    from dataclasses import replace

    case = shared_connection_case
    original = case.local.store.get_profile(case.profile_id)
    changed = replace(original, args=original.args + ("unreviewed-argument",))
    case.local.store.save_profile(changed)
    try:
        with pytest.raises(Exception, match="mapping_invalid"):
            await case.service.review_configuration(
                case.installation, connections={"mcp:shared": case.profile_id}
            )
    finally:
        case.local.store.save_profile(original)


@pytest.mark.asyncio
async def test_data_delete_waits_for_actual_idle_connection_close(
    shared_connection_case,
):
    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok
    process = next(iter(case.client.sessions.values())).process
    review = await case.service.review_data_deletion((case.data_root,))
    deletion = asyncio.create_task(
        case.service.delete_data((case.data_root,), review.operation_id)
    )
    await asyncio.sleep(0.05)
    assert (
        not deletion.done()
        and case.data_root.path.exists()
        and process.returncode is None
    )
    await case.service.cancel_data_work(review.operation_id)
    # Observe the retained authenticated deletion separately from native exit.
    # Its worker must finish real crypto/storage publication; timeout is no proof.
    receipt = await asyncio.wait_for(asyncio.shield(deletion), 30)
    assert receipt.phase == "complete"
    assert process.returncode is not None
    assert not case.data_root.path.exists()
    assert not (await case.invoke("b")).ok


@pytest.mark.asyncio
async def test_owned_public_headers_and_host_precedence(
    tmp_path, native_package, monkeypatch
):
    from Tests.MCP.test_credential_bindings import MemoryCredentialBackend
    from Tests.MCP.test_streamable_http import peer
    from tldw_chatbook.MCP.credential_bindings import (
        CredentialBindingService,
        endpoint_origin,
    )

    async with peer(monkeypatch) as (url, messages, calls):
        credentials = CredentialBindingService(MemoryCredentialBackend())
        binding = credentials.create_opaque(
            endpoint_origin=endpoint_origin(url),
            headers={"X-Public": "fresh-credential"},
        )
        headers = {
            "X-Public": "package-public",
            "X-Literal": "${UNEXPANDED}",
            "hOsT": "wrong.invalid",
            "mCp-MeThOd": "wrong",
            "Content-Length": "1",
            "Accept-Encoding": "gzip",
            "Connection": "upgrade",
            "Proxy-Test": "forbidden",
        }
        async with http_owned_case(
            tmp_path,
            native_package,
            url,
            headers=headers,
            credentials=credentials,
            binding=binding,
        ) as case:
            assert (await case.invoke("a")).ok
            sent = messages[-1][1]
            assert sent["x-public"] == "fresh-credential"
            assert sent["x-literal"] == "${UNEXPANDED}"
            assert sent["mcp-method"] == "tools/call"
            assert sent["host"] != "wrong.invalid"
            assert sent["content-length"] != "1"
            assert sent["accept-encoding"] == "identity"
            assert sent.get("connection") != "upgrade"
            assert "proxy-test" not in sent
            assert "package-public" not in repr(case.local.get_external_servers())
            assert "fresh-credential" not in case.local.store.path.read_text()
            assert calls["count"] == 1
            credentials.set_opaque(
                binding.reference_id,
                endpoint_origin=endpoint_origin(url),
                headers={"x-public": "replaced-credential"},
            )
            result = await case.invoke("b")
            assert not result.ok and result.dispatch_state == "not_started"
            assert calls["count"] == 1
            receipts = await case.service._call(case.service._coordinator.recover)
            assert any(item.phase == "recovery_required" for item in receipts)


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_connection_case", ["default"], indirect=True)
async def test_default_disable_revokes_inheritor_but_preserves_explicit_workspace(
    shared_connection_case,
):
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    case = shared_connection_case
    await case.service.disable(
        RevocationTarget(case.installation, None, False, global_default=True)
    )
    assert (await case.invoke("b")).ok
    case.release.touch()
    assert not (await case.pending).ok


@asynccontextmanager
async def literal_header_peer(monkeypatch):
    """Modern MCP peer recording raw octets at one owned loopback endpoint."""
    import _socket
    import socket

    messages, tasks = [], set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            headers = dict(
                line.split(b":", 1) for line in raw.split(b"\r\n")[1:] if b":" in line
            )
            headers = {
                key.lower(): value.lstrip(b" ") for key, value in headers.items()
            }
            body = await reader.readexactly(int(headers.get(b"content-length", b"0")))
            message = json.loads(body)
            messages.append((headers, message))
            method = message["method"]
            if method == "server/discover":
                result = {
                    "supportedVersions": ["2026-07-28"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "literal", "version": "1"},
                }
            elif method == "tools/list":
                result = {
                    "tools": [{"name": "echo", "inputSchema": {"type": "object"}}]
                }
            elif method == "tools/call":
                result = {"content": [{"type": "text", "text": "ok"}]}
            else:
                result = {}
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": message["id"], "result": result}
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    previous = socket.socket.connect

    def connect(sock, address):
        if sock.family == socket.AF_INET and address == ("127.0.0.1", port):
            return _socket.socket.connect(sock, address)
        return previous(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    try:
        yield f"http://127.0.0.1:{port}/rpc", messages
    finally:
        server.close()
        await server.wait_closed()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_owned_literal_header_octets_and_explicit_unicode_refusal(
    tmp_path, native_package, monkeypatch
):
    async with literal_header_peer(monkeypatch) as (url, messages):
        async with http_owned_case(
            tmp_path,
            native_package,
            url,
            headers={"X-Latin": "café", "X-Empty": "", "X-Tab": "a\tb"},
        ) as case:
            assert (await case.invoke("a")).ok
            assert messages[-1][0][b"x-latin"] == b"caf\xe9"
            assert messages[-1][0][b"x-empty"] == b""
            assert messages[-1][0][b"x-tab"] == b"a\tb"
        count = len(messages)
        with pytest.raises(RuntimeError, match="mcp_package_header_wire_unsupported"):
            async with http_owned_case(
                tmp_path / "wide", native_package, url, headers={"X-Wide": "snowman-☃"}
            ):
                pytest.fail("wider Unicode must refuse before network")
        assert len(messages) == count


@pytest.mark.asyncio
async def test_owned_retained_http_cleanup_cannot_be_replaced(
    tmp_path, native_package, monkeypatch
):
    from Tests.MCP.test_streamable_http import peer

    async with peer(monkeypatch) as (  # noqa: SIM117
        url,
        messages,
        calls,
    ):
        async with http_owned_case(tmp_path, native_package, url) as case:
            assert (await case.invoke("a")).ok
            assert (await case.invoke("b")).ok
            connection = next(
                value for value in case.ownership.connections.values() if value.owners
            )
            session = connection.session
            release, entered = asyncio.Event(), asyncio.Event()
            real_close = session._http.aclose

            async def close_later():
                entered.set()
                await release.wait()
                await real_close()

            monkeypatch.setattr(session._http, "aclose", close_later)
            try:
                await case.ownership.detach(connection.connection_id, "http-a:a")
                closing = asyncio.create_task(
                    case.ownership.detach(connection.connection_id, "http-b:b")
                )
                await asyncio.wait_for(entered.wait(), 2)
                await asyncio.wait_for(closing, 7)
                assert not session._cleanup_complete and not connection.leases_settled
                assert not case.ownership.is_connected(connection.profile.profile_id)
                assert not next(
                    row
                    for row in case.local.get_external_servers()
                    if row["profile_id"] == connection.profile.profile_id
                )["is_connected"]
                before = len(messages)
                refused = await case.invoke("b")
                assert not refused.ok and refused.dispatch_state == "not_started"
                assert len(messages) == before
                assert case.client.sessions[connection.connection_id] is session
            finally:
                release.set()
                await asyncio.wait_for(
                    case.ownership.detach(connection.connection_id, "http-b:b"), 2
                )
            assert session._cleanup_complete and connection.leases_settled
            assert (await case.invoke("b")).ok
            assert calls["count"] == 3


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "stdio", "command": "python", "env": {"PLUGIN_ROOT": "/override"}},
        {"type": "stdio", "command": "python", "args": ["${PLUGIN_DATA}"]},
        {
            "type": "streamable-http",
            "url": "https://example.invalid/rpc",
            "headers": {"X-Dup": "one", "x-dup": "two"},
        },
        {
            "type": "streamable-http",
            "url": "https://example.invalid/rpc",
            "headers": {"X-Bad": "line\nbreak"},
        },
    ],
)
@pytest.mark.asyncio
async def test_bad_configuration_refuses_without_launch_and_preserves_valid_sibling(
    native_package, tmp_path, bad
):
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.recovery import retained_inspections
    from tldw_chatbook.Plugins.service import PluginService

    package = native_package()
    (package / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {
                    "bad": bad,
                    "valid": {
                        "type": "streamable-http",
                        "url": "https://example.invalid/rpc",
                    },
                },
            }
        )
    )
    local = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "save.json"), manifest_provider=dict
    )
    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: SimpleNamespace(archived=False),
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
        mcp_mapping_owner=local,
    )
    await service.bootstrap("test passphrase")
    try:
        review = await service.review_install(
            package, selection=("mcp:valid",), workspace_id="a"
        )
        assert (await service.commit(review, review.operation_id)).committed
        inspection = await service._call(
            lambda: retained_inspections(service._coordinator.published_snapshot())[
                (review.installation_id, review.inspection.effective_digest)
            ]
        )
        with pytest.raises((ValueError, PermissionError)):
            local.save_owned_profile(
                installation_id=review.installation_id,
                inspection=inspection,
                component_id="mcp:bad",
            )
        saved = local.save_owned_profile(
            installation_id=review.installation_id,
            inspection=inspection,
            component_id="mcp:valid",
        )
        assert saved["url"] == "https://example.invalid/rpc"
        assert local.client is None
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("dependent", [False, True])
async def test_independent_component_can_capture_after_sibling_definition_changes(
    tmp_path, native_package, monkeypatch, dependent
):
    from Tests.MCP.test_streamable_http import peer
    from tldw_chatbook.Plugins.mcp_provider import PluginMCPProvider

    async with peer(monkeypatch) as (url, messages, calls):  # noqa: SIM117
        async with http_owned_case(
            tmp_path, native_package, url, second=True, dependent=dependent
        ) as case:
            provider = case.providers["a"]
            assert len(provider.list_catalog()) == 2
            for tool in provider.list_catalog():
                assert (await asyncio.to_thread(provider.invoke, tool.id, {})).ok
            before = len(messages)
            for ceiling in ((), ("missing",), ("mcp:unused",), "mcp:other"):
                with pytest.raises(PermissionError, match="ceiling_invalid"):
                    await case.service.capture_mcp_snapshot(
                        case.installation, "a", "invalid", component_ceiling=ceiling
                    )
            narrow = await case.service.capture_mcp_snapshot(
                case.installation, "a", "only-b", component_ceiling=("mcp:other",)
            )
            assert narrow.selection == (
                ("mcp:other", "mcp:remote") if dependent else ("mcp:other",)
            )
            assert len(messages) == before, "capture must not launch or discover"
            sibling = PluginMCPProvider(
                plugin_service=case.service,
                ownership=case.ownership,
                snapshot=narrow,
                service=case.unified,
                main_loop=asyncio.get_running_loop(),
                approval_callback=lambda pending: {
                    item.llm_name: "approve_once" for item in pending
                },
            )
            await sibling.compose_catalog()
            assert len(sibling.list_catalog()) == (2 if dependent else 1)
            original = case.local.store.get_discovery_snapshot(
                case.profile_ids["mcp:remote"]
            )
            changed = json.loads(json.dumps(original))
            changed["tools"][0]["description"] = "changed"
            case.local.store.save_discovery_snapshot(
                case.profile_ids["mcp:remote"], changed
            )
            try:
                assert not (await case.invoke("a")).ok
                if dependent:
                    with pytest.raises(PermissionError, match="component_unavailable"):
                        await case.service.capture_mcp_snapshot(
                            case.installation,
                            "a",
                            "dependent",
                            component_ceiling=("mcp:other",),
                        )
                    assert not (
                        await asyncio.to_thread(
                            sibling.invoke, sibling.list_catalog()[0].id, {}
                        )
                    ).ok
                    return
                assert (
                    await asyncio.to_thread(
                        sibling.invoke, sibling.list_catalog()[0].id, {}
                    )
                ).ok
                fresh = await case.service.capture_mcp_snapshot(
                    case.installation,
                    "a",
                    "fresh-independent",
                    component_ceiling=("mcp:other",),
                )
                assert fresh.selection == ("mcp:other",)
                assert all(
                    row["component_id"] == "mcp:other"
                    for row in json.loads(fresh.mappings_json)
                )
                assert calls["count"] == 3
                await case.service._call(
                    lambda: setattr(
                        case.service._coordinator, "mcp_mapping_owner", None
                    )
                )
                try:
                    assert not (
                        await asyncio.to_thread(
                            sibling.invoke, sibling.list_catalog()[0].id, {}
                        )
                    ).ok
                    assert calls["count"] == 3
                finally:
                    await case.service._call(
                        lambda: setattr(
                            case.service._coordinator, "mcp_mapping_owner", case.local
                        )
                    )
                receipts = await case.service._call(case.service._coordinator.recover)
                assert any(item.phase == "recovery_required" for item in receipts)
            finally:
                case.local.store.save_discovery_snapshot(
                    case.profile_ids["mcp:remote"], original
                )
                await case.service._call(case.service._coordinator.recover)


@pytest.mark.asyncio
async def test_permission_profile_callback_failure_is_bounded_before_dispatch(
    shared_connection_case,
):
    case = shared_connection_case
    before = case.calls.read_text()
    provider = case.providers["b"]
    original = provider._profile_id

    def unavailable():
        raise RuntimeError("private-callback-sentinel")

    provider._profile_id = unavailable
    try:
        result = await case.invoke("b")
        assert not result.ok and result.dispatch_state == "not_started"
        assert "private-callback-sentinel" not in repr(result)
        assert case.calls.read_text() == before
    finally:
        provider._profile_id = original


@pytest.mark.asyncio
async def test_known_preclient_launch_denial_settles_reserved_custody(
    shared_connection_case, monkeypatch
):
    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok
    assert next(iter(case.client.sessions.values())).process.returncode is None
    for connection in tuple(case.ownership.connections.values()):
        for owner in tuple(connection.owners):
            await case.ownership.detach(connection.connection_id, owner)
    assert not case.client.sessions
    governance_calls = 0
    entries = 0
    allowed = case.local._require_allowed
    real_connect = case.client.connect_profile

    def refuse_at_launch(action):
        nonlocal governance_calls
        if action == "mcp.external_profiles.launch.local":
            governance_calls += 1
            if governance_calls == 2:
                raise PermissionError("controlled_preclient_denial")
        return allowed(action)

    async def observe_entry(*args, **kwargs):
        nonlocal entries
        entries += 1
        return await real_connect(*args, **kwargs)

    monkeypatch.setattr(case.local, "_require_allowed", refuse_at_launch)
    monkeypatch.setattr(case.client, "connect_profile", observe_entry)
    try:
        with pytest.raises(PermissionError, match="controlled_preclient_denial"):
            await case.providers["a"].connect("mcp:shared")
        assert governance_calls == 2 and entries == 0
        assert not case.client.sessions and not case.client._pending_connections
        assert not await case.service._call(
            lambda: case.service._coordinator.root_usage.blockers((case.data_root,))
        )
    finally:
        # RED-only cleanup uses the controlled proof that MCPClient was never
        # entered. This is not a transport/session-flag inference.
        if entries == 0:
            for connection in tuple(case.ownership.connections.values()):
                for token in connection.leases.values():
                    await case.ownership._settle(token)


@pytest.mark.asyncio
async def test_owned_connected_metadata_observes_actual_scoped_sessions(
    shared_connection_case,
):
    case = shared_connection_case
    record = next(
        row
        for row in case.local.get_external_servers()
        if row["profile_id"] == case.profile_id
    )
    assert record["is_connected"] is True
    await case.disable_a()
    assert case.ownership.is_connected(case.profile_id, "run-b:b")
    assert not case.ownership.is_connected(case.profile_id, "run-a:a")
    assert (await case.invoke("b")).ok

    assert not case.ownership.is_connected(case.profile_id, "unattached:b")
    case.release.touch()
    assert not (await case.pending).ok
    for connection in tuple(case.ownership.connections.values()):
        for owner in tuple(connection.owners):
            await case.ownership.detach(connection.connection_id, owner)
    assert not case.ownership.is_connected(case.profile_id)
    assert not next(
        row
        for row in case.local.get_external_servers()
        if row["profile_id"] == case.profile_id
    )["is_connected"]


@pytest.mark.asyncio
async def test_exited_stdio_session_is_reaped_before_fresh_scope_dispatch(
    shared_connection_case,
):
    case = shared_connection_case
    process = next(iter(case.client.sessions.values())).process
    assert process.returncode is None
    process.kill()
    await asyncio.wait_for(process.wait(), 2)
    assert not (await case.pending).ok
    assert not case.ownership.is_connected(case.profile_id)
    assert (await case.invoke("b")).ok
    assert all(
        session.process is not process for session in case.client.sessions.values()
    )
    assert len(case.calls.read_text().splitlines()) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["persona", "kill_switch"])
@pytest.mark.parametrize("phase", ["before_dispatch", "after_dispatch", "advertise"])
async def test_owned_authority_reader_failure_never_removes_required_confirmation(
    shared_connection_case, monkeypatch, tmp_path, reader, phase
):
    from tldw_chatbook.Agents.persona_policy import parse_persona_policy_from_rules

    case = shared_connection_case
    provider = case.providers["b"]
    policy = parse_persona_policy_from_rules(
        [
            {
                "rule_kind": "mcp_tool",
                "rule_name": "echo",
                "allowed": True,
                "require_confirmation": True,
            }
        ]
    )
    assert policy.rules and policy.rules[0]["require_confirmation"]
    monkeypatch.setattr(provider, "_persona_policy_provider", lambda: policy)
    tool, _ = next(iter(provider._entry_by_llm_name.values()))
    case.unified.set_tool_state("local:" + case.profile_id, "echo", "allow", tool=tool)
    approvals = []

    def approve(calls):
        approvals.extend(calls)
        return {call.llm_name: "approve_once" for call in calls}

    monkeypatch.setattr(provider, "_approval_callback", approve)
    before_control = len(case.calls.read_text().splitlines())
    assert (await case.invoke("b")).ok
    assert len(approvals) == 1, "real persona must floor persisted allow to ask"
    assert len(case.calls.read_text().splitlines()) == before_control + 1
    before = case.calls.read_text()
    original_tokens = await case.service._call(
        lambda: case.service._coordinator.owner.unsettled_tokens(case.installation)
    )
    release, started = tmp_path / "authority-release", tmp_path / "authority-started"
    pending = None

    def unavailable():
        raise RuntimeError("private-authority-reader-sentinel")

    try:
        if phase == "after_dispatch":
            pending = asyncio.create_task(
                case.invoke("b", {"hold": str(release), "started": str(started)})
            )
            for _ in range(500):
                if started.exists():
                    break
                await asyncio.sleep(0.01)
            assert started.exists(), "B must actually dispatch before authority loss"
        target, attribute = (
            (provider, "_persona_policy_provider")
            if reader == "persona"
            else (case.unified, "get_kill_switch")
        )
        with monkeypatch.context() as fault:
            fault.setattr(target, attribute, unavailable)
            if phase == "advertise":
                try:
                    await provider.compose_catalog()
                except Exception as exc:  # noqa: BLE001
                    assert "private-authority-reader-sentinel" not in str(exc)
                assert provider.list_catalog() == [], (
                    "unavailable authority must clear advertising"
                )
                assert case.calls.read_text() == before
            else:
                release.touch()
                result = (
                    await pending if pending is not None else await case.invoke("b")
                )
                assert not result.ok
                assert result.dispatch_state == (
                    "settled" if phase == "after_dispatch" else "not_started"
                )
                assert "private-authority-reader-sentinel" not in repr(result)
                assert len(case.calls.read_text().splitlines()) == (
                    len(before.splitlines()) + (phase == "after_dispatch")
                )
                tokens = await case.service._call(
                    lambda: case.service._coordinator.owner.unsettled_tokens(
                        case.installation
                    )
                )
                assert tokens == original_tokens, (
                    "failed authority must not leak request grants"
                )
    finally:
        release.touch()
        if pending is not None:
            await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint", ["connect", "reserve", "publish"])
async def test_owned_automatic_authority_is_rechecked_after_awaited_setup(
    shared_connection_case, monkeypatch, tmp_path, checkpoint
):
    from Tests.Chat.test_automatic_provider_budget import context_for
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    case = shared_connection_case
    database = AgentRunsDB(tmp_path / "automatic.sqlite", client_id="owned-automatic")
    context = context_for(database)
    reached, resume = asyncio.Event(), asyncio.Event()
    pending = None
    try:
        with context.scope():
            before_control = len(case.calls.read_text().splitlines())
            control = await case.invoke("b")
            assert control.ok and control.dispatch_state == "settled"
            assert len(case.calls.read_text().splitlines()) == before_control + 1
        before = case.calls.read_text()
        original_tokens = await case.service._call(
            lambda: case.service._coordinator.owner.unsettled_tokens(case.installation)
        )
        method = {"connect": "connect", "reserve": "_reserve", "publish": "_publish"}[
            checkpoint
        ]
        original = getattr(case.ownership, method)

        async def pause_after_real_setup(*args, **kwargs):
            value = await original(*args, **kwargs)
            is_request = (
                checkpoint == "connect"
                or (checkpoint == "reserve" and kwargs.get("request") is not None)
                or (checkpoint == "publish" and "mcp_request_id" in args[1])
            )
            if is_request:
                reached.set()
                await resume.wait()
            return value

        monkeypatch.setattr(case.ownership, method, pause_after_real_setup)
        with context.scope():
            pending = asyncio.create_task(case.invoke("b"))
        await asyncio.wait_for(reached.wait(), 5)
        database.automatic_work.pause(context.chain_id, "controlled_stop")
        with pytest.raises(AutomaticWorkRefused):
            context.check()
        resume.set()
        result = await asyncio.wait_for(pending, 5)
        assert not result.ok and result.dispatch_state == "not_started"
        assert case.calls.read_text() == before, (
            "stopped automatic chain must not reach peer"
        )
        tokens = await case.service._call(
            lambda: case.service._coordinator.owner.unsettled_tokens(case.installation)
        )
        assert tokens == original_tokens, (
            "not-started request must settle its exact grant"
        )
        assert (
            database.automatic_work.snapshot(context.chain_id).pause_reason
            == "controlled_stop"
        )
    finally:
        resume.set()
        if pending is not None:
            await pending
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["persona", "kill_switch"])
async def test_late_authority_failure_preserves_actual_uncertain_dispatch(
    tmp_path, native_package, monkeypatch, reader
):
    from Tests.MCP.test_streamable_http import peer
    from tldw_chatbook.Agents.persona_policy import parse_persona_policy_from_rules

    class Fault:
        value = None

        def __eq__(self, other):
            return self.value == other

        def __hash__(self):
            return hash(self.value)

    fault = Fault()
    async with peer(monkeypatch, fault=fault) as (  # noqa: SIM117
        url,
        _messages,
        calls,
    ):
        async with http_owned_case(tmp_path, native_package, url) as case:
            provider = case.providers["b"]
            policy = parse_persona_policy_from_rules(
                [
                    {
                        "rule_kind": "mcp_tool",
                        "rule_name": "echo",
                        "allowed": True,
                        "require_confirmation": True,
                    }
                ]
            )
            tool, _ = next(iter(provider._entry_by_llm_name.values()))
            case.unified.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
            monkeypatch.setattr(provider, "_persona_policy_provider", lambda: policy)
            approvals = []

            def approve(pending):
                approvals.extend(pending)
                return {call.llm_name: "approve_once" for call in pending}

            monkeypatch.setattr(provider, "_approval_callback", approve)
            assert (await case.invoke("b")).ok
            assert len(approvals) == 1 and calls["count"] == 1
            original = case.client.call_tool_result

            def unavailable():
                raise RuntimeError("private-late-reader-sentinel")

            async def fail_reader_after_real_call(*args, **kwargs):
                result = await original(*args, **kwargs)
                assert result.dispatch_state == "uncertain"
                if reader == "persona":
                    monkeypatch.setattr(
                        provider, "_persona_policy_provider", unavailable
                    )
                else:
                    monkeypatch.setattr(case.unified, "get_kill_switch", unavailable)
                return result

            monkeypatch.setattr(
                case.client, "call_tool_result", fail_reader_after_real_call
            )
            fault.value = "lost"
            result = await case.invoke("b")
            assert not result.ok and result.dispatch_state == "uncertain"
            assert "private-late-reader-sentinel" not in repr(result)
            assert calls["count"] == 2, "lost call must not replay"
            rows = await case.service._call(
                lambda: case.service._coordinator.owner.list_processes(
                    limit=50, offset=0
                )
            )
            assert any(
                row["workspace_id"] == "b"
                and row["state"] == "published"
                and (row.get("provenance") or {}).get("outcome") == "uncertain"
                for row in rows
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_connection_case", ["portable-literals"], indirect=True)
async def test_portable_unknown_placeholder_stays_literal_on_actual_argv(
    shared_connection_case,
):
    case = shared_connection_case
    launch = json.loads((case.calls.parent / "launch.json").read_text())
    assert launch["argv"][-1] == "${UNRECOGNIZED}"
    assert launch["data"] == str(case.data_root.path)
    assert (await case.invoke("b")).ok


@pytest.mark.asyncio
async def test_shared_stdio_deadline_retains_a_custody_without_stopping_b(
    shared_connection_case, tmp_path
):
    case = shared_connection_case
    session = next(iter(case.client.sessions.values()))
    process = session.process
    case.release.touch()
    assert (await case.pending).ok
    release, started = tmp_path / "deadline-release", tmp_path / "deadline-started"
    pending = None
    session.request_timeout_seconds = 0.03
    try:
        pending = asyncio.create_task(
            case.invoke("a", {"hold": str(release), "started": str(started)})
        )
        async with asyncio.timeout(20):
            while not started.exists():
                await asyncio.sleep(0.001)
        await asyncio.sleep(0.2)
        assert process.returncode is None, (
            "A timeout must not kill B's shared native transport"
        )
        assert not pending.done(), (
            "native A custody must remain until a terminal response or child exit"
        )
        session.request_timeout_seconds = 10.0
        await case.disable_a()
        assert (await case.invoke("b")).ok
        assert next(iter(case.client.sessions.values())) is session
        assert session._producer_lifetime.calls, (
            "accepted A native calls must retain source admission"
        )
        release.touch()
        result = await asyncio.wait_for(pending, 3)
        assert not result.ok, (
            "A's late terminal response must not regain revoked acceptance"
        )
        assert len(case.calls.read_text().splitlines()) == 5, (
            "a timed out call must never replay"
        )
    finally:
        session.request_timeout_seconds = 10.0
        release.touch()
        if pending is not None:
            await pending
