"""The original model operation covers real local-peer SSH request retirement."""

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest
import toml

from Tests.Chat.test_native_v2_command_events import until
from Tests.Chat.test_native_v2_mcp_events import (
    event_case as _event_case,
)
from Tests.Chat.test_native_v2_mcp_events import (
    hook_file as _hook_file,
)
from Tests.Chat.test_native_v2_mcp_events import (
    mcp_case as _mcp_case,
)
from Tests.Chat.test_native_v2_mcp_events import (
    native_console as _native_console,
)
from Tests.Chat.test_native_v2_mcp_events import (
    native_package as _native_package,
)
from Tests.private_profile import private_profile_test
from Tests.Tools.test_remote_executor_ssh import FakeSsh
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified

event_case, hook_file, mcp_case = _event_case, _hook_file, _mcp_case
native_console, native_package = _native_console, _native_package

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
    pytest.mark.skipif(
        not native_workflow_qualified(), reason="native POSIX platform required"
    ),
]


def assert_original_finishes_returned(state):
    assert state.finishes
    for operation, admission, was_live in state.finishes:
        if admission is None:
            assert operation in state.deferred_returns
        else:
            assert was_live


@pytest.fixture
async def ssh_console(mcp_case, hook_file, monkeypatch, tmp_path, request):
    from Tests.Agents.test_agent_service import fence
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook import config
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Tools import remote_workspace_transport as transport

    case, rig = mcp_case, mcp_case.rig
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Tools import workspace_file_roots

    monkeypatch.setattr(
        workspace_file_roots, "_default_registry_instance", rig.registry
    )
    rig.store.set_session_project_instruction_state(
        rig.session.id, ProjectInstructionControlState.legacy_disabled()
    )
    raw = toml.loads(hook_file.read_text())
    raw["console_ssh"] = {"session_worker": False, "bundle_cache": False}
    raw.setdefault("console", {})["local_tools_enabled"] = True
    hook_file.write_text(toml.dumps(raw))
    config.refresh_runtime_config_from_cli_config()
    fake = FakeSsh(tmp_path, monkeypatch)
    manager = transport.SshMasterManager(ssh_bin=str(fake.bin), enabled=False)
    monkeypatch.setattr(transport, "_MASTER_MANAGER", manager)
    workspace = tmp_path / "remote-workspace"
    workspace.mkdir()
    (workspace / "alpha.txt").write_text("original remote body\n")
    host = getattr(request, "param", {}).get("host", "fake-host")
    monkeypatch.setenv("FAKE_SSH_G_HOST", host)
    if getattr(request, "param", {}).get("local", False):
        binding = rig.registry.add_folder_binding(rig.session.workspace_id, workspace)
    else:
        binding = rig.registry.add_ssh_binding(
            rig.session.workspace_id,
            f"ssh://{host}{workspace}",
            python_interpreter=sys.executable,
            ssh_bin=str(fake.bin),
            allow_write=getattr(request, "param", {}).get("write", False),
        )
    case.service.approve_for_session("local:__local__", "fs_read")
    case.service.approve_for_session("local:__local__", "fs_edit")
    rig.store.persistence.db.close()
    rig.store.persistence = ChatPersistenceService(
        CharactersRAGDB(tmp_path / "console-chat.sqlite", "ssh-custody"),
        workspace_registry=rig.registry,
    )

    state = SimpleNamespace(
        before_model=None,
        before_native_return=None,
        reads=1,
        operations=[],
        finishes=[],
        native_returns=[],
        deferred_returns={},
        edit=False,
        virtual=getattr(request, "param", {}).get("virtual", False),
    )

    case.service.approve_for_session("local:__virtual_cli__", "cat")

    from tldw_chatbook.Agents.activation import _guard
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

    reserve_tool = ExecutionOwner.reserve_tool

    def observe_operation(owner):
        operation = reserve_tool(owner)
        state.operations.append(operation)
        finish = operation.finish

        def observe_return():
            from tldw_chatbook.Chat.console_ssh_custody import current_ssh_call

            native = current_ssh_call()
            admission = _guard.context.get()
            if admission is None:
                # Deferred workers finish after original SSH/DB/source exits.
                # Observe that exact runner and its original operation, rather
                # than acquiring new admission merely to test finalization.
                from tldw_chatbook.Agents.agent_service import _worker_database_state
                from tldw_chatbook.Chat.console_ordinary_tree import (
                    current_ordinary_tree,
                )

                frame = sys._getframe(1)
                while frame is not None and frame.f_code.co_name != "_runner":
                    frame = frame.f_back
                assert frame is not None and frame.f_locals["operation"] is operation
                assert frame.f_locals["retained_source"] is None
                assert current_ssh_call() is None and current_ordinary_tree() is None
                native = frame.f_locals["ssh_call"]
                if native is not None and native.operation is None:
                    assert not frame.f_locals["box"].get("source_attempted")
                    assert (
                        not native.requests
                        and not native.master_launches
                        and not native.session_starts
                    )
                    native = None  # Original pre-source refusal never activated SSH.
                if native is not None:
                    assert native.operation is operation and not native.finished
                    assert not native.uncertain
                    assert all(
                        r.proven
                        for r in (
                            *native.requests,
                            *native.master_launches,
                            *native.session_starts,
                        )
                    )
                dbs = tuple(
                    db
                    for db in (database, rig.store.persistence.db, rig.registry.db)
                    if db is not None
                )
                slots, operations, pending, leases = _worker_database_state(dbs)()
                assert all(
                    local and participant and conn is None and not held and not retiring
                    for local, participant, conn, held, retiring in slots
                )
                assert not operations and not pending and not leases
                state.deferred_returns[operation] = native
            else:
                assert admission.live  # Unchanged default activate contract.
            if native is not None and state.before_native_return is not None:
                state.before_native_return(native)
            assert not any(existing is operation for existing, _, _ in state.finishes)
            finish()
            state.native_returns.append(native)
            state.finishes.append(
                (operation, admission, admission.live if admission else False)
            )

        operation.finish = observe_return
        return operation

    monkeypatch.setattr(ExecutionOwner, "reserve_tool", observe_operation)

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            self.payloads.append(messages)
            if state.before_model is not None:
                await state.before_model(len(self.payloads))
            if len(self.payloads) == 1:
                yield fence(
                    "load_tools",
                    {
                        "ids": ["local:fs_read", "local:fs_edit"]
                        if state.edit
                        else ["virtual_cli:virtual_cli"]
                        if state.virtual
                        else ["local:fs_read"]
                    },
                )
            elif len(self.payloads) <= state.reads + 1:
                yield fence(
                    "virtual_cli" if state.virtual else "fs_read",
                    {
                        "command": "cat",
                        "argv": ["alpha.txt"],
                        "root_alias": getattr(
                            state, "virtual_alias", binding.binding_id
                        ),
                    }
                    if state.virtual
                    else {"path": "alpha.txt", "root_alias": binding.binding_id},
                )
            elif state.edit and len(self.payloads) == state.reads + 2:
                yield fence(
                    "fs_edit",
                    {
                        "path": "alpha.txt",
                        "root_alias": binding.binding_id,
                        "old_string": "original remote body",
                        "new_string": "edited remote body",
                    },
                )
            else:
                yield "read complete"

    gateway = Gateway()
    database = AgentRunsDB(tmp_path / "console-runs.sqlite", client_id="ssh-custody")
    bridge = ConsoleAgentBridge(
        agent_runs_db=database,
        store=rig.store,
        provider_gateway=gateway,
        skills_service=(
            rig.skills if getattr(request, "param", {}).get("skills", True) else None
        ),
        runtime_capacity=case.runtime.execution_capacity,
        get_hooks_v2=case.runtime.get_hooks_v2,
    )
    case.runtime.set_agent_bridge(bridge)
    rig.controller._agent_bridge = bridge
    rig.controller._agent_runtime_enabled = True
    rig.controller.provider_gateway = gateway
    state.case, state.rig, state.gateway = case, rig, gateway
    state.fake, state.manager, state.database = fake, manager, database
    state.binding = binding
    state.config_file = hook_file
    state.root = tmp_path

    def submit(**kwargs):
        task = asyncio.create_task(
            rig.controller.submit_draft(
                "read the remote file", session_id=rig.session.id, **kwargs
            )
        )
        case.tasks.append(task)
        return task

    state.submit = submit
    try:
        yield state
    finally:
        database.close()


async def _submit_local_declarations(h):
    from dataclasses import replace

    import tldw_chatbook.MCP.server  # noqa: F401 - load the original manifest source
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    local = LocalMCPControlService(store=LocalMCPStore())
    service = UnifiedMCPControlPlaneService(
        local_service=local, server_service=None, target_store=None, context_store=None
    )
    h.rig.controller.app.unified_mcp_service = h.case.service = service
    service.set_server_default("builtin:tldw_chatbook", "deny")
    service.approve_for_session("local:__local__", "fs_read")
    service.approve_for_session("local:__virtual_cli__", "cat")
    configuration = await h.rig.controller.capture_turn_configuration_snapshot(
        h.rig.session.id
    )
    assert (
        configuration.mcp_definition_complete and not configuration.mcp_capture_pending
    )
    assert (
        not configuration.mcp_definition_maximum
        and not configuration.mcp_definition_sources
    )
    tools = dict(configuration.tool_configuration)
    tools["agent_run_budget_maximum"] = replace(
        tools["agent_run_budget_maximum"], max_subagents=0
    )
    assert (
        await h.submit(configuration=replace(configuration, tool_configuration=tools))
    ).accepted
    assert not service._producer_lifetime.calls and not local._producer_lifetime.calls
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0


@private_profile_test
@pytest.mark.parametrize(
    "ssh_console,opaque,custom_factory",
    [
        ({"skills": False}, False, False),
        ({"skills": True}, True, False),
        ({"skills": False}, True, True),
    ],
    indirect=["ssh_console"],
)
async def test_stock_local_declarations_keep_private_effects_bounded(
    request,
    ssh_console,
    opaque,
    custom_factory,
    monkeypatch,
):
    from dataclasses import replace

    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    h = ssh_console
    binding = h.rig.registry.list_runtime_bindings(h.rig.session.workspace_id)[0]
    h.rig.registry.remove_runtime_binding(binding.binding_id)
    h.reads = 0
    trees = []

    async def observe(_count):
        trees.append(current_ordinary_tree())

    h.before_model = observe
    if not opaque:
        from tldw_chatbook import config

        raw = toml.loads(h.config_file.read_text())
        raw.setdefault("tools", {})["read_file_enabled"] = True
        h.config_file.write_text(toml.dumps(raw))
        config.refresh_runtime_config_from_cli_config()
    constructions = []
    if custom_factory:
        from tldw_chatbook.Agents import local_tool_provider as local

        original = local._default_specs

        def custom(*args, **kwargs):
            tree = current_ordinary_tree()
            constructions.append(tree)
            return [
                replace(spec, worktree_effect="none")
                for spec in original(*args, **kwargs)
            ]

        monkeypatch.setattr(local, "_default_specs", custom)
    await _submit_local_declarations(h)
    assert trees and all(tree is trees[0] for tree in trees)
    assert trees[0].footprint.opaque is opaque, (
        "maximum missed the actual skills/runtime gate"
    )
    if not opaque:
        assert trees[0].builtin_maximum["read_file"] == "read"
        assert any(
            "read_file" in tools for tools in trees[0].builtin_providers.values()
        )
    if custom_factory:
        assert constructions and all(tree is trees[0] for tree in constructions)
        assert all(tree.footprint.opaque for tree in constructions), (
            "custom factory ran before opaque admission"
        )


@private_profile_test
@pytest.mark.parametrize(
    "ssh_console", [{"skills": False, "virtual": True}], indirect=True
)
@pytest.mark.parametrize("custom_factory", [False, True])
async def test_original_local_virtual_source_and_roots_are_in_initial_maximum(
    request, ssh_console, monkeypatch, custom_factory
):
    from tldw_chatbook.Agents import virtual_cli_provider as virtual
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    h, seen, constructions = ssh_console, [], []
    remote = h.rig.registry.list_runtime_bindings(h.rig.session.workspace_id)[0]
    h.rig.registry.remove_runtime_binding(remote.binding_id)
    local = h.root / "virtual-workspace"
    local.mkdir()
    (local / "alpha.txt").write_text("original local virtual body\n")
    binding = h.rig.registry.add_folder_binding(h.rig.session.workspace_id, local)
    h.virtual_alias = binding.binding_id
    original = virtual.VirtualCliProvider
    if custom_factory:

        def custom(**kwargs):
            tree = current_ordinary_tree()
            constructions.append((tree, tree.footprint.opaque))
            return original(**kwargs)

        monkeypatch.setattr(virtual, "VirtualCliProvider", custom)

    async def observe(_count):
        seen.append(current_ordinary_tree())

    h.before_model = observe
    await _submit_local_declarations(h)
    assert len(h.gateway.payloads) == 3 and "original local virtual body" in str(
        h.gateway.payloads[-1]
    )
    assert all(tree.footprint.opaque is custom_factory for tree in seen)
    assert seen[0].virtual_providers
    if custom_factory:
        assert constructions and all(
            tree is seen[0] and opaque for tree, opaque in constructions
        )


@pytest.mark.parametrize("descriptor_replaced", [False, True])
async def test_original_startup_ssh_has_operation_before_reader_enqueue(
    ssh_console, monkeypatch, descriptor_replaced
):
    from tldw_chatbook.Chat.console_chat_controller import (
        list_project_instruction_bindings,
    )
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )

    h = ssh_console
    selected = list_project_instruction_bindings(h.rig.session, h.rig.registry)[0]
    h.rig.store.set_session_project_instruction_state(
        h.rig.session.id,
        ProjectInstructionControlState(
            project_instructions_enabled=True,
            working_folder_binding_id=selected.binding.binding_id,
            working_folder_locator_fingerprint=selected.locator_fingerprint,
        ),
    )
    h.rig.controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    (h.root / "remote-workspace" / "AGENTS.md").write_text(
        "Original remote instructions.\n"
    )
    release, entered = h.root / "startup-release", h.root / "startup-io"
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'exec sh -c "$remote_cmd"',
            'sh -c "$remote_cmd"\nrc=$?\n'
            '(touch "$SSH_IO_ENTERED"; while [ ! -f "$SSH_IO_RELEASE" ]; do sleep .05; done) >&2 &\n'
            'exit "$rc"',
        )
    )
    monkeypatch.setenv("SSH_IO_ENTERED", str(entered))
    monkeypatch.setenv("SSH_IO_RELEASE", str(release))
    descriptor_reads = []
    if descriptor_replaced:
        from tldw_chatbook.Agents.project_instruction_resolver import (
            RemoteInstructionIO,
        )
        from tldw_chatbook.Chat import console_chat_controller as host

        original_reader = host._remote_instruction_io_for_selection

        def replacement(*args, **kwargs):
            io = original_reader(*args, **kwargs)

            def forbidden_descriptor(value):
                descriptor_reads.append(value)
                return value._executor

            monkeypatch.setattr(
                RemoteInstructionIO, "executor", property(forbidden_descriptor)
            )
            return io

        monkeypatch.setattr(host, "_remote_instruction_io_for_selection", replacement)
        h.reads = 0
    task = h.submit()
    try:
        await until(lambda: entered.exists() or task.done())
        if descriptor_replaced:
            assert (await task).accepted
            assert not descriptor_reads, (
                "source check invoked a replacement IO descriptor"
            )
            assert not entered.exists(), "changed IO source reached SSH"
            return
        assert entered.exists(), (
            task.result() if task.done() else "original startup did not run"
        )
        assert not h.gateway.payloads, "probe was reached only by a model call"
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1, (
            "original remote startup reader ran without its physical operation"
        )
    finally:
        release.touch()
        await task


@pytest.mark.parametrize(
    "ssh_console,dispose",
    [
        ({"host": "ordinary-probe-live"}, False),
        ({"host": "ordinary-probe-dispose"}, True),
    ],
    indirect=["ssh_console"],
)
async def test_original_recovery_probe_retains_tree_after_turn_return(
    ssh_console, monkeypatch, dispose
):
    from tldw_chatbook.Chat.console_chat_controller import (
        _app_remote_binding_status_cache,
    )
    from tldw_chatbook.Tools.remote_workspace_transport import TransportFailureKind
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    h = ssh_console
    binding = h.rig.registry.list_runtime_bindings(h.rig.session.workspace_id)[0]
    cache = _app_remote_binding_status_cache()
    cache.record_transport_failure(
        binding.binding_id, TransportFailureKind.UNREACHABLE, "controlled probe"
    )
    release, entered = h.root / "probe-release", h.root / "probe-io"
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'exec sh -c "$remote_cmd"',
            'sh -c "$remote_cmd"\nrc=$?\n'
            '(touch "$SSH_IO_ENTERED"; while [ ! -f "$SSH_IO_RELEASE" ]; do sleep .05; done) >&2 &\n'
            'exit "$rc"',
        )
    )
    monkeypatch.setenv("SSH_IO_ENTERED", str(entered))
    monkeypatch.setenv("SSH_IO_RELEASE", str(release))
    h.reads = 0
    task = h.submit()
    disposal = None
    try:
        await until(lambda: entered.exists() or task.done())
        assert entered.exists(), "original degraded-binding probe did not dispatch"
        result = await task
        assert result.accepted
        snapshot = h.case.runtime.execution_capacity.snapshot()
        assert snapshot.tool_workers == 1
        assert len(snapshot.executions) == 1 and snapshot.executions[0].root_finished
        with h.case.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None, "probe return obligation lost original activity claim"
        if dispose:
            disposal = asyncio.create_task(h.case.runtime.dispose())
            await until(lambda: h.case.runtime._disposed)
            assert not disposal.done(), "disposal passed the original live probe"
    finally:
        release.touch()
        await task
        await until(lambda: not h.case.runtime.execution_capacity.snapshot().executions)
        if disposal is not None:
            await disposal
    direct = [
        call
        for call in h.native_returns
        if call is not None and call.source.authority is None
    ]
    assert len(direct) == 1
    assert (
        direct[0].finished
        and direct[0].requests
        and all(r.proven for r in direct[0].requests)
    )
    assert all(
        live
        for operation, admission, live in h.finishes
        if operation is direct[0].operation
    )
    with h.case.runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as claim:
        assert claim is not None, "proven probe return did not retire original claim"


async def test_original_model_operation_waits_for_real_ssh_stderr_return(
    ssh_console, monkeypatch
):
    h = ssh_console
    release, entered = h.root / "release-io", h.root / "io-live"
    # The actual remote bundle exits; a real OS child retains its stderr pipe.
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'exec sh -c "$remote_cmd"',
            'sh -c "$remote_cmd"\nrc=$?\n'
            '(touch "$SSH_IO_ENTERED"; while [ ! -f "$SSH_IO_RELEASE" ]; do sleep .05; done) >&2 &\n'
            'exit "$rc"',
        )
    )
    monkeypatch.setenv("SSH_IO_ENTERED", str(entered))
    monkeypatch.setenv("SSH_IO_RELEASE", str(release))
    task = h.submit()
    try:
        await until(lambda: entered.exists() or task.done())
        assert entered.exists(), [p[-2:] for p in h.gateway.payloads]
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1
        await asyncio.sleep(5)
        assert any(
            t.name.startswith("ssh-call-stderr-") and t.is_alive()
            for t in threading.enumerate()
        )
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1, (
            "original fs_read operation retired while actual SSH stderr IO was live"
        )
    finally:
        release.touch()
        await task
        await until(
            lambda: (
                not any(
                    t.name.startswith("ssh-call-stderr-") and t.is_alive()
                    for t in threading.enumerate()
                )
            )
        )
    assert len(h.gateway.payloads) == 3
    assert "original remote body" in str(h.gateway.payloads[2])
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert_original_finishes_returned(h)


@pytest.mark.parametrize("ssh_console", [{"virtual": True}], indirect=True)
async def test_original_virtual_model_operation_waits_for_real_ssh_stderr_return(
    ssh_console, monkeypatch
):
    await test_original_model_operation_waits_for_real_ssh_stderr_return(
        ssh_console, monkeypatch
    )
    calls = [call for call in ssh_console.native_returns if call is not None]
    assert calls and all(call.name == "virtual_cli" for call in calls)
    assert all(request.proven for call in calls for request in call.requests)


@pytest.mark.parametrize("ssh_console", [{"virtual": True}], indirect=True)
@pytest.mark.parametrize(
    "replacement", ["registry", "parser", "provider_descriptor", "callback"]
)
async def test_changed_original_virtual_source_refuses_after_enqueue(
    ssh_console, monkeypatch, replacement
):
    from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider
    from tldw_chatbook.Tools import virtual_cli_impls

    h, invoked = ssh_console, []
    start = threading.Thread.start

    def altered(*args, **kwargs):
        import traceback

        invoked.append(
            [(frame.name, frame.lineno) for frame in traceback.extract_stack(limit=12)]
        )
        raise AssertionError("replacement virtual source ran")

    def replace_before_start(thread):
        if thread.name == "tool-virtual_cli":
            (service,) = h.rig.controller._agent_bridge._fleet_services.values()
            provider = service.registry.resolve_owner_for_name("virtual_cli")[1]
            if replacement == "provider_descriptor":
                monkeypatch.setattr(VirtualCliProvider, "invoke", property(altered))
            elif replacement == "parser":
                monkeypatch.setattr(virtual_cli_impls, "parse_request", altered)
            elif replacement == "callback":
                monkeypatch.setattr(provider, "_resolve_state", altered)
            else:
                registry = next(
                    value
                    for alias, value in provider._registries_by_alias.items()
                    if alias.startswith("ssh-")
                )
                monkeypatch.setattr(registry, "execute", altered)
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", replace_before_start)
    assert (await h.submit()).accepted
    assert len(h.gateway.payloads) == 3
    assert not invoked and not h.fake.call_invocations()
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert_original_finishes_returned(h)


@pytest.mark.parametrize(
    "ssh_console", [{"virtual": True, "local": True}], indirect=True
)
async def test_original_local_virtual_callback_refuses_before_review(
    ssh_console, monkeypatch
):
    await test_changed_original_virtual_source_refuses_after_enqueue(
        ssh_console, monkeypatch, "callback"
    )


@pytest.mark.parametrize(
    "ssh_console", [{"virtual": True, "local": True}], indirect=True
)
async def test_original_local_tool_finish_return_retains_source_through_disposal(
    ssh_console, monkeypatch
):
    from tldw_chatbook.Agents.activation import _guard
    from tldw_chatbook.Agents.execution_capacity import CapacityRefused, ExecutionOwner
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    h = ssh_console
    entered, release = threading.Event(), threading.Event()
    observations = []
    reserve = ExecutionOwner.reserve_tool

    def reserve_observed(owner):
        operation = reserve(owner)
        finish = operation.finish

        def held_finish():
            finish()
            if threading.current_thread().name == "tool-virtual_cli":
                frame = sys._getframe(1)
                while frame is not None and frame.f_code.co_name != "_runner":
                    frame = frame.f_back
                assert frame is not None and frame.f_locals["operation"] is operation
                tree = frame.f_locals["ordinary"]
                token = frame.f_locals["finalization_host"]
                assert current_ordinary_tree() is None and _guard.context.get() is None
                assert frame.f_locals["retained_source"] is None
                assert token in tree.primary_hosts
                observations.append((tree, token))
                entered.set()
                assert release.wait(20)

        operation.finish = held_finish
        return operation

    monkeypatch.setattr(ExecutionOwner, "reserve_tool", reserve_observed)
    task = h.submit()
    disposal = None
    grace_returned = asyncio.Event()
    bounded_wait = h.case.runtime._bounded_wait

    async def observe_grace(tasks, *, timeout_seconds):
        result = await bounded_wait(tasks, timeout_seconds=timeout_seconds)
        if timeout_seconds == 0:
            grace_returned.set()
        return result

    monkeypatch.setattr(h.case.runtime, "_bounded_wait", observe_grace)
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set(), h.gateway.payloads
        tree, token = observations[0]
        assert token in tree.primary_hosts
        with h.case.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
        disposal = asyncio.create_task(h.case.runtime.dispose(timeout_seconds=0))
        await until(lambda: h.case.runtime._disposed)
        await until(lambda: task.done())
        await asyncio.wait_for(grace_returned.wait(), 5)
        done, _pending = await asyncio.wait({disposal}, timeout=3)
        assert not done, "disposal passed original operation.finish RETURN"
        assert token in tree.primary_hosts and tree.scratch_lease is not None
        assert not tree.retired.done()
        with pytest.raises(
            (CapacityRefused, RuntimeError),
            match="ordinary_turn_closed|original agent storage source changed",
        ):
            tree.borrow(h.case.runtime)
        with pytest.raises(CapacityRefused):
            tree.owner.reserve_tool()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if disposal is not None:
            await disposal
    from tldw_chatbook.Chat.console_ordinary_tree import _json

    details = []
    for member in tree.members.values():
        row = h.database.get_run(member.run_id) if member.run_id else None
        stored = {step["index"]: _json(step) for step in row["steps"]} if row else {}
        details.append(
            {
                "expected_terminal": member.terminal,
                "actual_terminal": (
                    row["status"],
                    row.get("result"),
                    row.get("budget_tokens"),
                )
                if row
                else None,
                "expected_steps": list(member.steps),
                "actual_steps": list(stored),
                "different_steps": [
                    index
                    for index, value in member.steps.items()
                    if stored.get(index) != value
                ],
                "durable": member.durable,
            }
        )
    assert all(row["durable"] and not row["different_steps"] for row in details), (
        details
    )
    assert tree.retired.done() and tree not in h.case.runtime._ordinary_turns
    assert token not in tree.primary_hosts
    with h.case.runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as claim:
        assert claim is not None, {
            "scopes": tree.source_scopes,
            "sealed": tree.sealed,
            "settled": tree.settled,
            "members": [vars(member) for member in tree.members.values()],
        }


async def test_two_model_requests_return_while_original_shared_connection_stays_healthy(
    ssh_console, monkeypatch
):
    from tldw_chatbook import config
    from tldw_chatbook.Tools import remote_session_registry as sessions

    h = ssh_console
    raw = toml.loads(h.config_file.read_text())
    raw["console_ssh"]["session_worker"] = True
    h.config_file.write_text(toml.dumps(raw))
    config.refresh_runtime_config_from_cli_config()
    registry = sessions.RemoteSessionRegistry()
    monkeypatch.setattr(sessions, "_REGISTRY", registry)
    entered, release = threading.Event(), threading.Event()
    h.reads = 2

    async def hold(index):
        if index == 4:
            entered.set()
            await asyncio.to_thread(release.wait, 10)

    h.before_model = hold
    task = h.submit()
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set(), [p[-2:] for p in h.gateway.payloads]
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
        assert len(registry._sessions) == 1
        worker = next(iter(registry._sessions.values()))
        assert worker.alive and worker._reader_thread.is_alive()
        assert worker._proc.poll() is None
        assert worker._native_acceptance.registry is registry
        assert len(h.fake.call_invocations()) == 1
        for option in ("ControlMaster=no", "ControlPath=none", "ControlPersist=no"):
            assert option in h.fake.call_invocations()[0]
        assert_original_finishes_returned(h)
        assert "original remote body" in str(h.gateway.payloads[2])
        assert "original remote body" in str(h.gateway.payloads[3])
    finally:
        release.set()
        await task
        registry.close_all()


async def test_never_started_destination_helper_retires_original_operation(ssh_console):
    h = ssh_console

    async def remove_peer(index):
        if index == 2:
            h.fake.bin.unlink()

    h.before_model = remove_peer
    assert (await h.submit()).accepted
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert not h.fake.call_invocations()
    assert "ssh -G could not re-resolve" in str(h.gateway.payloads[-1])
    assert_original_finishes_returned(h)


@pytest.mark.parametrize(
    "replacement", ["provider", "handler", "authority", "executor"]
)
async def test_changed_original_provider_callback_refuses_after_enqueue(
    ssh_console, monkeypatch, replacement
):
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider

    h, invoked = ssh_console, []
    start = threading.Thread.start

    def altered(*args, **kwargs):
        invoked.append(True)
        raise AssertionError("replacement provider ran")

    def replace_before_start(thread):
        if thread.name == "tool-fs_read":
            from dataclasses import replace

            (service,) = h.rig.controller._agent_bridge._fleet_services.values()
            provider = service.registry.resolve_owner_for_name("fs_read")[1]
            authority = next(
                a
                for a in provider._admitted_roots.values()
                if a.alias.startswith("ssh-")
            )
            if replacement == "provider":
                monkeypatch.setattr(LocalToolProvider, "_invoke_detailed", altered)
            elif replacement == "handler":
                spec = provider._path_specs_by_alias[authority.alias]["fs_read"]
                provider._path_specs_by_alias[authority.alias]["fs_read"] = replace(
                    spec, handler=altered
                )
            elif replacement == "authority":
                provider._admitted_roots[authority.alias] = replace(
                    authority, guard=lambda _: True
                )
            else:
                monkeypatch.setattr(
                    authority.workspace_executor._inner, "execute", altered
                )
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", replace_before_start)
    assert (await h.submit()).accepted
    assert not invoked and not h.fake.call_invocations()
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    refusal = {
        "provider": "ordinary local callable maximum changed",
        "handler": "ordinary local handler source changed",
    }.get(replacement, "original SSH")
    assert refusal in str(h.gateway.payloads[-1])


@pytest.mark.parametrize(
    "replacement,shared",
    [
        pytest.param(replacement, shared, id=f"{replacement}-{kind}")
        for replacement in (
            "manager",
            "manager_config",
            "control_directory",
            "_bundle_payload",
            "_remote_command",
        )
        for shared, kind in ((False, "one_shot"), (True, "session"))
    ]
    + [pytest.param("manager_descriptor", True, id="manager_descriptor-session")],
)
async def test_changed_receiving_ssh_source_refuses_after_enqueue(
    ssh_console, monkeypatch, shared, replacement
):
    from tldw_chatbook.Tools import remote_session_worker as session
    from tldw_chatbook.Tools import remote_workspace_transport as transport

    h, invoked = ssh_console, []
    registry = enable_sessions(h, monkeypatch) if shared else None
    peer_entered = h.root / "replacement-peer-entered"
    replacement_peer = h.root / "replacement-ssh"
    replacement_peer.write_text(
        h.fake.bin.read_text().replace("\n", '\ntouch "$SSH_REPLACEMENT_ENTERED"\n', 1)
    )
    replacement_peer.chmod(0o700)
    monkeypatch.setenv("SSH_REPLACEMENT_ENTERED", str(peer_entered))
    start = threading.Thread.start
    original_builders = []

    def altered(*args, **kwargs):
        invoked.append(True)
        return original_builders[-1](*args, **kwargs)

    def replace_before_start(thread):
        if thread.name == "tool-fs_read":
            (service,) = h.rig.controller._agent_bridge._fleet_services.values()
            provider = service.registry.resolve_owner_for_name("fs_read")[1]
            authority = next(
                a
                for a in provider._admitted_roots.values()
                if a.alias.startswith("ssh-")
            )
            if replacement == "manager":
                monkeypatch.setattr(
                    authority.workspace_executor._inner._ssh.transport,
                    "_manager",
                    transport.SshMasterManager(
                        ssh_bin=str(replacement_peer), enabled=False
                    ),
                )
            elif replacement == "manager_config":
                authority.workspace_executor._inner._ssh.masters._control_persist = (
                    "changed"
                )
            elif replacement == "control_directory":
                authority.workspace_executor._inner._ssh.masters._control_dir = (
                    h.root / "changed"
                )
            elif replacement == "manager_descriptor":
                manager = transport.SshMasterManager(
                    ssh_bin=str(replacement_peer), enabled=False
                )

                class ChangedManagerDescriptor:
                    def __get__(self, instance, owner=None):
                        invoked.append("class" if instance is None else "instance")
                        return (
                            transport._NATIVE_TRANSPORT_MANAGER
                            if instance is None
                            else manager
                        )

                monkeypatch.setattr(
                    transport.RemoteWorkspaceTransport,
                    "master_manager",
                    ChangedManagerDescriptor(),
                )
            else:
                original_builders.append(
                    getattr(session if shared else transport, replacement)
                )
                monkeypatch.setattr(
                    session if shared else transport, replacement, altered
                )
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", replace_before_start)
    try:
        assert (await h.submit()).accepted
        assert not invoked and not peer_entered.exists(), {
            "callback_entries": invoked,
            "replacement_peer_entered": peer_entered.exists(),
        }
        assert not h.fake.call_invocations()
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
        assert "original SSH" in str(h.gateway.payloads[-1])
        assert_original_finishes_returned(h)
    finally:
        if registry is not None:
            registry.close_all()


def install_control_peer(h, monkeypatch, *, checks=True):
    """Actual Unix control server: check succeeds only after a socket exchange."""
    peer = h.root / "control-peer.py"
    peer.write_text("""import os, socket, subprocess, sys
from pathlib import Path
mode, *args = sys.argv[1:]
if mode == "serve":
    path = args[0]
    server = socket.socket(socket.AF_UNIX)
    server.bind(path)
    server.listen()
    Path(path + ".pid").write_text(str(os.getpid()))
    print("READY", flush=True)
    while True:
        conn, _ = server.accept()
        with conn:
            data = conn.recv(20)
            conn.sendall(b"OK")
        if data == b"exit": break
    server.close()
    Path(path).unlink()
else:
    path = next(x.split("=", 1)[1] for x in args if x.startswith("ControlPath=")).replace("%C", "peer-socket")
    if mode == "launch":
        proc = subprocess.Popen([sys.executable, __file__, "serve", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
        assert proc.stdout.readline() == b"READY\\n"
        proc.stdout.close()
        if os.environ.get("SSH_PEER_LAUNCH_FAIL"): sys.exit(255)
    else:
        if mode == "check" and os.environ.get("SSH_PEER_HIDE_CHECK"): sys.exit(1)
        try:
            with socket.socket(socket.AF_UNIX) as conn:
                conn.connect(path)
                conn.sendall(mode.encode())
                assert conn.recv(20) == b"OK"
        except OSError: sys.exit(1)
""")
    script = (
        h.fake.bin.read_text()
        .replace(
            "check) exit 1 ;;",
            'check) exec "$SSH_CONTROL_PYTHON" "$SSH_CONTROL_PEER" check "$@" ;;',
        )
        .replace(
            "exit)  exit 0 ;;",
            'exit) exec "$SSH_CONTROL_PYTHON" "$SSH_CONTROL_PEER" exit "$@" ;;',
        )
    )
    start = script.index('for arg in "$@"; do\n  if [ "$arg" = "-MNf" ]')
    end = script.index("# -- per-call:", start)
    script = (
        script[:start]
        + """for arg in "$@"; do
  if [ "$arg" = "-MNf" ]; then
    exec "$SSH_CONTROL_PYTHON" "$SSH_CONTROL_PEER" launch "$@"
  fi
done

"""
        + script[end:]
    )
    h.fake.bin.write_text(script)
    monkeypatch.setenv("SSH_CONTROL_PYTHON", sys.executable)
    monkeypatch.setenv("SSH_CONTROL_PEER", str(peer))
    if not checks:
        monkeypatch.setenv("SSH_PEER_HIDE_CHECK", "1")
    h.manager._enabled = True
    # Keep this test-owned Unix server off the process-shared short-path
    # fallback. Its fixed peer-socket name must identify only this fixture.
    import tempfile
    from pathlib import Path

    h.control_state = tempfile.TemporaryDirectory(prefix="t8-cm-", dir="/tmp")
    h.manager._state_dir = Path(h.control_state.name)
    return peer


async def test_native_direct_read_avoids_failed_optional_master(
    ssh_console, monkeypatch
):
    """Real model/worker return; the old ambiguous launcher stays child-isolated."""
    import json
    import os
    import subprocess

    from tldw_chatbook.Chat.console_ssh_custody import current_ssh_call

    h = ssh_console
    if not os.environ.get("TASK2_SSH_DIRECT_CHILD"):
        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_native_direct_read_avoids_failed_optional_master",
                "-s",
                "--basetemp=" + str(h.root / "direct-child"),
            ],
            env=dict(os.environ, TASK2_SSH_DIRECT_CHILD="1"),
            capture_output=True,
            text=True,
            timeout=40,
        )
        (h.root / "direct-child.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        return

    h.manager._enabled = True
    h.manager._state_dir = h.fake.state_dir
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'if [ "$arg" = "-MNf" ]; then',
            'if [ "$arg" = "-MNf" ]; then\n    exit 255',
        )
    )
    waiting = []
    wait = threading.Event.wait

    def observe_retention(event, timeout=None):
        native = current_ssh_call()
        if native is not None and timeout is None and native.operation._stopping:
            waiting.append(native)
        return wait(event, timeout)

    monkeypatch.setattr(threading.Event, "wait", observe_retention)
    task = h.submit()
    await until(lambda: task.done() or bool(waiting))
    if waiting:
        native = waiting[0]
        print(
            json.dumps(
                {
                    "retained": True,
                    "launch_returncodes": [
                        a.process.returncode for a in native.master_launches
                    ],
                    "requests_proven": [r.proven for r in native.requests],
                }
            ),
            flush=True,
        )
        os._exit(17)  # Bound the old unknown path; termination is not drain proof.
    assert (await task).accepted
    calls = [call for call in h.native_returns if call is not None]
    assert calls and all(call.finished and not call.uncertain for call in calls)
    assert all(not call.master_launches for call in calls)
    assert all(r.local_returned and r.proven for call in calls for r in call.requests)
    assert "original remote body" in str(h.gateway.payloads[-1])
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert_original_finishes_returned(h)
    assert all(
        admission is None or not admission.live for _, admission, _ in h.finishes
    )
    assert h.fake.count("-MNf") == 0
    assert not h.manager._native_checks
    for argv in h.fake.call_invocations():
        assert "ControlMaster=no" in argv
        assert "ControlPath=none" in argv
        assert "ControlPersist=no" in argv


async def test_native_master_check_timeout_retires_helper_before_direct_read(
    ssh_console, monkeypatch
):
    import os

    from tldw_chatbook.Tools import remote_workspace_transport as transport

    h = ssh_console
    helper_pid = h.root / "check-helper.pid"
    monkeypatch.setenv("SSH_CHECK_PID", str(helper_pid))
    monkeypatch.setattr(transport, "_CHECK_TIMEOUT_SECONDS", 0.1)
    h.manager._enabled = True
    h.manager._state_dir = h.fake.state_dir
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            "check) exit 1 ;;", 'check) echo $$ > "$SSH_CHECK_PID"; exec sleep 30 ;;'
        )
    )
    assert (await h.submit()).accepted
    assert helper_pid.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(helper_pid.read_text()), 0)
    calls = [call for call in h.native_returns if call is not None]
    assert calls and all(call.finished and not call.uncertain for call in calls)
    assert all(not call.master_launches for call in calls)
    assert all(r.proven for call in calls for r in call.requests)
    assert "original remote body" in str(h.gateway.payloads[-1])
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert h.fake.count("-MNf") == 0
    assert not h.manager._native_checks


async def test_original_manager_reuses_verified_existing_master_without_closing_it(
    ssh_console, monkeypatch
):
    import socket
    from dataclasses import replace
    from pathlib import PurePosixPath

    from tldw_chatbook.Tools.remote_binding_locator import parse_remote_locator

    h = ssh_console
    install_control_peer(h, monkeypatch)
    # Genuine nonnative setup, with a different root on the same registered host.
    prewarm = replace(
        parse_remote_locator(h.binding.locator), path=PurePosixPath("/prewarm")
    )
    await asyncio.to_thread(h.manager.ensure_master, prewarm)
    assert h.fake.count("-MNf") == 1
    entered, release = threading.Event(), threading.Event()
    h.reads = 2

    async def hold(index):
        if index == 4:
            entered.set()
            await asyncio.to_thread(release.wait, 10)

    h.before_model = hold
    task = h.submit()
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set(), [p[-2:] for p in h.gateway.payloads]
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
        (acceptance,) = h.manager._native_acceptances.values()
        assert acceptance.manager is h.manager
        assert acceptance.launcher is None
        calls = [call for call in h.native_returns if call is not None]
        assert calls and all(
            call.finished and not call.master_launches for call in calls
        )
        assert calls[-1].master_acceptance is acceptance
        assert acceptance.locator is calls[-1].source.cfg.loc
        assert acceptance.locator.path != prewarm.path
        assert all(r.proven for call in calls for r in call.requests)
        assert not h.manager._native_checks
        with socket.socket(socket.AF_UNIX) as conn:
            conn.connect(str(acceptance.control_dir / "peer-socket"))
            conn.sendall(b"check")
            assert conn.recv(20) == b"OK"
        assert h.fake.count("-MNf") == 1
        assert h.fake.count("check") >= 2
        assert h.fake.count("exit") == 0
        for argv in h.fake.call_invocations():
            assert "ControlMaster=no" in argv and "ControlPersist=no" in argv
            assert f"ControlPath={acceptance.control_dir.as_posix()}/%C" in argv
        assert "original remote body" in str(h.gateway.payloads[3])
        assert_original_finishes_returned(h)
    finally:
        release.set()
        await task
        h.manager.close_all()


@pytest.mark.parametrize(
    "selection",
    [
        "previous_call",
        "replaced_record",
        "other_locator",
        "other_manager",
        "closed",
        "restart",
        "stale_socket",
    ],
)
async def test_native_client_options_require_current_exact_acceptance(
    ssh_console, monkeypatch, selection
):
    from dataclasses import replace
    from pathlib import PurePosixPath

    from tldw_chatbook.Tools.remote_binding_locator import parse_remote_locator
    from tldw_chatbook.Tools.remote_workspace_transport import (
        SshMasterManager,
        _host_key,
    )

    h = ssh_console
    install_control_peer(h, monkeypatch)
    await asyncio.to_thread(
        h.manager.ensure_master, parse_remote_locator(h.binding.locator)
    )
    h.reads = 2
    accepted, checked_options = [], []

    def inspect_current_call(native):
        if not native.requests:
            return
        manager, loc = h.manager, native.source.cfg.loc
        if not accepted:
            assert (
                native.master_acceptance is manager._native_acceptances[_host_key(loc)]
            )
            accepted.append(native.master_acceptance)
            if selection == "stale_socket":
                monkeypatch.setenv("SSH_PEER_HIDE_CHECK", "1")
            return
        if selection == "previous_call":
            native.master_acceptance = accepted[0]
        elif selection == "replaced_record":
            with manager._registry_lock:
                manager._native_acceptances[_host_key(loc)] = accepted[0]
        elif selection == "other_locator":
            loc = replace(loc, path=PurePosixPath("/other-root"))
        elif selection == "other_manager":
            manager = SshMasterManager(
                ssh_bin=str(h.fake.bin), state_dir=h.fake.state_dir
            )
        elif selection == "closed":
            manager.close_all()
        elif selection == "restart":
            before = h.fake.invocations()
            assert manager.restart_if_dead(loc) is False
            assert h.fake.invocations() == before
            assert (manager._control_dir / "peer-socket").exists()
            assert not manager._native_checks
        elif selection == "stale_socket":
            assert native.master_acceptance is None
            assert (manager._control_dir / "peer-socket").exists()
        options = manager.client_options(loc)
        checked_options.append(options)
        assert "ControlMaster=no" in options
        assert "ControlPath=none" in options
        assert "ControlPersist=no" in options

    h.before_native_return = inspect_current_call
    try:
        assert (await h.submit()).accepted
        assert checked_options
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
        assert "original remote body" in str(h.gateway.payloads[-1])
        assert not h.manager._native_checks
        assert h.fake.count("-MNf") == 1
    finally:
        h.manager.close_all()


async def test_source_change_after_positive_master_check_discards_only_candidate(
    ssh_console, monkeypatch
):
    from tldw_chatbook.Notes.git_process_containment import ProcessTreeController
    from tldw_chatbook.Tools.remote_binding_locator import parse_remote_locator

    h = ssh_console
    install_control_peer(h, monkeypatch)
    await asyncio.to_thread(
        h.manager.ensure_master, parse_remote_locator(h.binding.locator)
    )
    refusals = []

    def check_after_real_read(native):
        if not native.requests:
            return
        close = ProcessTreeController.close

        def change_after_helper_return(controller, tree):
            close(controller, tree)
            h.manager._control_persist = "changed-after-helper-return"

        with monkeypatch.context() as mutation:
            mutation.setattr(ProcessTreeController, "close", change_after_helper_return)
            with pytest.raises(
                RuntimeError, match="original SSH executor source changed"
            ):
                h.manager.ensure_master(native.source.cfg.loc)
        refusals.append(True)
        assert not h.manager._native_checks
        assert native.master_acceptance is None
        assert not native.uncertain and all(r.proven for r in native.requests)

    h.before_native_return = check_after_real_read
    try:
        assert (await h.submit()).accepted
        assert refusals
        assert h.fake.count("-MNf") == 1
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    finally:
        h.manager.close_all()


async def test_destination_helper_waits_for_its_original_process_group(
    ssh_console, monkeypatch
):
    h = ssh_console
    entered, release = h.root / "helper-live", h.root / "helper-release"
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'if [ "$arg" = "-G" ]; then',
            'if [ "$arg" = "-G" ]; then\n'
            '    (touch "$SSH_HELPER_ENTERED"; while [ ! -f "$SSH_HELPER_RELEASE" ]; do sleep .05; done) >/dev/null 2>&1 &',
        )
    )
    monkeypatch.setenv("SSH_HELPER_ENTERED", str(entered))
    monkeypatch.setenv("SSH_HELPER_RELEASE", str(release))
    task = h.submit()
    try:
        await until(lambda: entered.exists() or task.done())
        assert entered.exists()
        await asyncio.sleep(0.2)
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1
        assert not h.fake.call_invocations(), "request overtook original ssh -G group"
    finally:
        release.touch()
        await task
    assert "original remote body" in str(h.gateway.payloads[-1])
    assert_original_finishes_returned(h)


@pytest.mark.parametrize(
    "mode",
    [
        "cleanup_false",
        "virtual_cleanup_false",
        "no_terminal",
        "entered_ambiguous_master",
        "admission_one_shot",
        "admission_session",
    ],
)
async def test_unknown_peer_proof_keeps_original_model_worker_and_storage(
    ssh_console, monkeypatch, mode
):
    import json
    import os
    import subprocess
    from pathlib import Path

    h = ssh_console
    if os.environ.get("TASK8_SSH_UNPROVEN_MODE") != mode:
        receipt = h.root / "retention.json"
        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_unknown_peer_proof_keeps_original_model_worker_and_storage[{mode}]",
                "--basetemp=" + str(h.root / "child"),
            ],
            env=dict(
                os.environ, TASK8_SSH_UNPROVEN_MODE=mode, TASK8_SSH_RECEIPT=str(receipt)
            ),
            capture_output=True,
            text=True,
            timeout=40,
        )
        (h.root / "retention-child.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(receipt.read_text()) == {
            "mode": mode,
            "held": True,
            "storage_live": True,
        }
        return

    h.virtual = mode == "virtual_cleanup_false"
    if mode == "entered_ambiguous_master":
        install_control_peer(h, monkeypatch, checks=False)
        monkeypatch.setenv("SSH_PEER_LAUNCH_FAIL", "1")
        from tldw_chatbook.Chat.console_ssh_custody import (
            accept_master,
            current_ssh_call,
        )
        from tldw_chatbook.Tools.remote_workspace_transport import _host_key

        start = threading.Thread.start

        def enter_adversarial_launcher(thread):
            native = current_ssh_call()
            if (
                thread.name.startswith("ssh-call-stderr-")
                and native is not None
                and not native.master_launches
            ):
                # Explicit lower-level entered-launch adversary, not native ensure.
                loc = native.source.cfg.loc
                directory = h.manager.control_path_for(loc)
                h.manager._spawn_master(loc, _host_key(loc), directory)
                accept_master(h.manager, loc, directory)
            return start(thread)

        monkeypatch.setattr(threading.Thread, "start", enter_adversarial_launcher)
    elif mode == "admission_session":
        enable_sessions(h, monkeypatch)
        relay = h.root / "admission-session-relay.py"
        relay.write_text("""import json, os, struct, subprocess, sys, threading
child = subprocess.Popen(["sh", "-c", sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def upload():
    try:
        while data := os.read(0, 65536):
            child.stdin.write(data)
            child.stdin.flush()
    except BrokenPipeError: pass
    finally: child.stdin.close()
threading.Thread(target=upload, daemon=True).start()
while True:
    line = child.stdout.readline()
    if not line: sys.exit(child.wait())
    sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
    if b"READY " in line: break
header, admitted = struct.Struct(">IIB"), {}
while raw := child.stdout.read(header.size):
    length, request, kind = header.unpack(raw)
    body = child.stdout.read(length)
    if kind == 16:
        frame = json.loads(body)
        if frame["outcome"] == "admitted": admitted[request] = body
        elif request in admitted:
            body = admitted[request]
            raw = header.pack(len(body), request, kind)
    sys.stdout.buffer.write(raw + body); sys.stdout.buffer.flush()
sys.exit(child.wait())
""")
        h.fake.bin.write_text(
            h.fake.bin.read_text().replace(
                'exec sh -c "$remote_cmd"',
                'exec "$SSH_RELAY_PYTHON" "$SSH_TERMINAL_RELAY" "$remote_cmd"',
            )
        )
        monkeypatch.setenv("SSH_RELAY_PYTHON", sys.executable)
        monkeypatch.setenv("SSH_TERMINAL_RELAY", str(relay))
    else:
        relay = h.root / "terminal-relay.py"
        relay.write_text("""import json, os, sys
admitted = None
for line in sys.stdin.buffer:
    start = line.find(b"{")
    if start < 0:
        sys.stdout.buffer.write(line)
        continue
    frame = json.loads(line[start:])
    if frame["outcome"] == "admitted": admitted = frame
    if frame["outcome"] != "admitted":
        if os.environ["TASK8_SSH_UNPROVEN_MODE"] == "no_terminal": continue
        if os.environ["TASK8_SSH_UNPROVEN_MODE"] == "admission_one_shot":
            if admitted is not None: frame = admitted
        else: frame["cleanup_proven"] = False
    sys.stdout.buffer.write(line[:start] + json.dumps(frame).encode() + b"\\n")
    sys.stdout.buffer.flush()
""")
        h.fake.bin.write_text(
            h.fake.bin.read_text().replace(
                'exec sh -c "$remote_cmd"',
                'sh -c "$remote_cmd" | "$SSH_RELAY_PYTHON" "$SSH_TERMINAL_RELAY"',
            )
        )
        monkeypatch.setenv("SSH_RELAY_PYTHON", sys.executable)
        monkeypatch.setenv("SSH_TERMINAL_RELAY", str(relay))

    from tldw_chatbook.Agents.activation import _guard
    from tldw_chatbook.Chat.console_ssh_custody import current_ssh_call

    waiting = []
    wait = threading.Event.wait

    def observe_retention(event, timeout=None):
        native = current_ssh_call()
        if native is not None and timeout is None and native.operation._stopping:
            waiting.append((native, _guard.context.get(), threading.current_thread()))
        return wait(event, timeout)

    monkeypatch.setattr(threading.Event, "wait", observe_retention)
    task = h.submit()
    await until(lambda: bool(waiting) or task.done())
    assert waiting, [
        (r.operation_id, r.local_returned, r.terminal.outcome, r.proven)
        for call in h.native_returns
        if call is not None
        for r in call.requests
        if r.terminal is not None
    ]
    native, storage, worker = waiting[0]
    assert worker.is_alive() and native.operation.owner._cleanup_unproven
    assert storage.live and not native.finished
    assert all(operation is not native.operation for operation, _, _ in h.finishes)
    assert ("db.agent_runs", h.database.db_path) in storage.sources
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1
    assert native.requests and all(r.local_returned for r in native.requests)
    if mode in {"cleanup_false", "virtual_cleanup_false"}:
        assert all(
            r.terminal is not None and r.terminal.cleanup_proven is False
            for r in native.requests
        )
    elif mode == "no_terminal":
        assert native.requests[0].terminal is None
    elif mode in {"admission_one_shot", "admission_session"}:
        assert [json.loads(r.wire)["operation"] for r in native.requests] == [
            "ping",
            "fs_read",
        ]
        assert native.requests[0].proven
        assert native.requests[1].terminal is None
    else:
        assert native.master_launches and not native.master_launches[0].proven
        attempt = native.master_launches[0]
        assert attempt.returned and attempt.process.returncode == 255
        assert not attempt.accepted and not attempt.never_started
        import socket

        with socket.socket(socket.AF_UNIX) as conn:
            conn.connect(str(h.manager._control_dir / "peer-socket"))
            conn.sendall(b"check")
            assert conn.recv(20) == b"OK"
        assert not h.manager._native_acceptances
        # Actual test-owned infrastructure shutdown is not native proof/release.
        h.manager.close_all()
    Path(os.environ["TASK8_SSH_RECEIPT"]).write_text(
        json.dumps({"mode": mode, "held": True, "storage_live": True})
    )
    os._exit(
        0
    )  # End only this deliberately retained test process; never reset custody.


def enable_sessions(h, monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Tools import remote_session_registry as sessions

    raw = toml.loads(h.config_file.read_text())
    raw["console_ssh"]["session_worker"] = True
    h.config_file.write_text(toml.dumps(raw))
    config.refresh_runtime_config_from_cli_config()
    registry = sessions.RemoteSessionRegistry()
    monkeypatch.setattr(sessions, "_REGISTRY", registry)
    return registry


async def test_lost_cancel_keeps_actual_pending_terminal_until_reader_status(
    ssh_console, monkeypatch
):
    from tldw_chatbook.Chat.console_ssh_custody import current_ssh_call

    h = ssh_console
    registry = enable_sessions(h, monkeypatch)
    relay = h.root / "session-relay.py"
    entered, release = h.root / "terminal-waiting", h.root / "release-terminal"
    # Transparent transport peer: the original bundled session/child executes.
    # Delay only request 3's actual terminal/STATUS bytes (ping/read/read).
    relay.write_text("""import os, struct, subprocess, sys, threading, time
from pathlib import Path
child = subprocess.Popen(["sh", "-c", sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def upload():
    try:
        while data := os.read(0, 65536):
            child.stdin.write(data)
            child.stdin.flush()
    except BrokenPipeError: pass
    finally: child.stdin.close()
threading.Thread(target=upload, daemon=True).start()
while True:
    line = child.stdout.readline()
    if not line: sys.exit(child.wait())
    sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
    if b"READY " in line: break
header = struct.Struct(">IIB")
while raw := child.stdout.read(header.size):
    length, request, kind = header.unpack(raw)
    body = child.stdout.read(length)
    if request == 3 and kind == 16 and b'"outcome":"admitted"' not in body and b'"outcome": "admitted"' not in body:
        Path(os.environ["SSH_RELAY_ENTERED"]).touch()
        while not Path(os.environ["SSH_RELAY_RELEASE"]).exists(): time.sleep(.01)
    sys.stdout.buffer.write(raw + body); sys.stdout.buffer.flush()
sys.exit(child.wait())
""")
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'exec sh -c "$remote_cmd"',
            'exec "$SSH_RELAY_PYTHON" "$SSH_SESSION_RELAY" "$remote_cmd"',
        )
    )
    monkeypatch.setenv("SSH_RELAY_PYTHON", sys.executable)
    monkeypatch.setenv("SSH_SESSION_RELAY", str(relay))
    monkeypatch.setenv("SSH_RELAY_ENTERED", str(entered))
    monkeypatch.setenv("SSH_RELAY_RELEASE", str(release))
    h.reads = 2

    # Configure the actual original executor before its first model enqueue.
    async def budget(index):
        if index == 2:
            bridge = h.rig.controller._agent_bridge
            (service,) = bridge._fleet_services.values()
            provider = service.registry.resolve_owner_for_name("fs_read")[1]
            authority = next(
                a
                for a in provider._admitted_roots.values()
                if a.alias.startswith("ssh-")
            )
            executor = authority.workspace_executor._inner
            executor._budget_seconds = 1
            executor._ssh.transport._grace_seconds = 0.1

    h.before_model = budget
    waiting = []
    wait = threading.Event.wait

    def observe_lost_cancel(event, timeout=None):
        native = current_ssh_call()
        if (
            native is not None
            and timeout is None
            and worker is not None
            and any(event is pending.done for pending in worker._pending.values())
        ):
            waiting.append(native)
        return wait(event, timeout)

    monkeypatch.setattr(threading.Event, "wait", observe_lost_cancel)
    task = h.submit()
    worker = None
    locked = False
    try:
        await until(lambda: entered.exists() or task.done())
        assert entered.exists(), [p[-2:] for p in h.gateway.payloads]
        (worker,) = registry._sessions.values()
        locked = worker._write_lock.acquire(blocking=False)
        assert locked
        await until(lambda: bool(waiting))
        assert 3 in worker._pending, (
            "lost CANCEL dropped the original request observation"
        )
        assert worker._pending[3].status is None
        assert waiting[0].request.issued
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1
    finally:
        if locked:
            worker._write_lock.release()
        release.touch()
        await task
        registry.close_all()
    assert worker._next_id == 4, "a possibly issued request was retried"
    assert len(h.fake.call_invocations()) == 1
    assert "original remote body" in str(h.gateway.payloads[-1])
    assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    assert_original_finishes_returned(h)


async def test_unpublished_session_keeps_original_io_until_close_returns(
    ssh_console, monkeypatch
):
    from tldw_chatbook.Chat.console_ssh_custody import current_ssh_call

    h = ssh_console
    registry = enable_sessions(h, monkeypatch)
    entered, release = h.root / "unpublished-io", h.root / "release-unpublished"
    h.fake.bin.write_text(
        h.fake.bin.read_text().replace(
            'exec sh -c "$remote_cmd"',
            'sh -c "$remote_cmd"\nrc=$?\n'
            '(touch "$SSH_IO_ENTERED"; while [ ! -f "$SSH_IO_RELEASE" ]; do sleep .05; done) >&2 &\n'
            'exit "$rc"',
        )
    )
    monkeypatch.setenv("SSH_IO_ENTERED", str(entered))
    monkeypatch.setenv("SSH_IO_RELEASE", str(release))
    unpublished = []
    joining = threading.Event()
    start, join = threading.Thread.start, threading.Thread.join

    def observe_original_join(thread, timeout=None):
        if (
            timeout is None
            and thread.name == "ssh-session-stderr"
            and current_ssh_call() is not None
        ):
            joining.set()
        return join(thread, timeout)

    monkeypatch.setattr(threading.Thread, "join", observe_original_join)

    def close_before_publication(thread):
        native = current_ssh_call()
        if thread.name == "ssh-session-reader" and native is not None:
            attempt = native.session_starts[-1]
            unpublished.append(attempt)
            registry.close_key(attempt.key[0])
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", close_before_publication)
    task = h.submit()
    try:
        await until(lambda: entered.exists() or task.done())
        assert entered.exists() and unpublished
        await until(joining.is_set)
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 1
        assert not unpublished[0].proven
        assert unpublished[0].worker._native_acceptance is None
        assert unpublished[0].worker._stderr_thread.is_alive()
        assert not registry._sessions
    finally:
        release.touch()
        await task
        registry.close_all()
    assert unpublished[0].proven
    assert unpublished[0].worker._proc.poll() is not None
    assert not unpublished[0].worker._stderr_thread.is_alive()
    assert "original remote body" in str(h.gateway.payloads[-1])
    assert_original_finishes_returned(h)


@pytest.mark.parametrize("ssh_console", [{"write": True}], indirect=True)
async def test_original_edit_operation_covers_actual_cas_read_before_write(ssh_console):
    import json

    h = ssh_console
    h.edit = True
    assert (await h.submit()).accepted
    assert (
        h.root / "remote-workspace" / "alpha.txt"
    ).read_text() == "edited remote body\n"
    edit = next(
        call for call in h.native_returns if call is not None and call.name == "fs_edit"
    )
    assert [json.loads(request.wire)["operation"] for request in edit.requests] == [
        "fs_read",
        "fs_edit",
    ]
    assert all(request.proven for request in edit.requests)
    assert (
        edit.finished and h.case.runtime.execution_capacity.snapshot().tool_workers == 0
    )
    assert_original_finishes_returned(h)


@pytest.mark.parametrize("ssh_console", [{"skills": False}], indirect=True)
@pytest.mark.parametrize("complete", [False, True])
async def test_caller_snapshot_cannot_prove_mcp_footprint_is_bounded(
    ssh_console, complete
):
    from dataclasses import replace

    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

    h = ssh_console
    binding = h.rig.registry.list_runtime_bindings(h.rig.session.workspace_id)[0]
    h.rig.registry.remove_runtime_binding(binding.binding_id)
    h.case.service.set_tool_state("local:fixture", "fixture", "deny", tool=h.case.tool)
    h.reads = 0
    trees = []

    async def observe(_count):
        trees.append(current_ordinary_tree())

    h.before_model = observe
    configuration = await h.rig.controller.capture_turn_configuration_snapshot(
        h.rig.session.id
    )
    tools = dict(configuration.tool_configuration)
    tools["agent_run_budget_maximum"] = replace(
        tools["agent_run_budget_maximum"], max_subagents=0
    )
    configuration = replace(
        configuration, tool_configuration=tools, mcp_definition_complete=complete
    )
    assert not configuration.mcp_definition_maximum
    if complete:
        from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

        with pytest.raises(RuntimeError) as refusal:
            await h.submit(configuration=configuration)
        assert (
            str(refusal.value)
            == "Original MCP capture source is unavailable; send again."
        )
        assert not trees and not h.gateway.payloads and not h.fake.call_invocations()
        assert h.case.runtime.execution_capacity.snapshot().tool_workers == 0
        with h.case.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is not None
        return
    assert (await h.submit(configuration=configuration)).accepted
    assert trees and all(tree.footprint.opaque for tree in trees)
