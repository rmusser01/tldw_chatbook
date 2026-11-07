"""Run-bound workflow restrictions at real dispatch seams (ADR-285)."""

import dataclasses
import json
from contextlib import ExitStack
from threading import Event
from types import SimpleNamespace

import pytest

from Tests.Agents.test_local_tool_provider import _agent_authority, make_provider
from Tests.Agents.workflow_support import make_workflow_request, plan_data
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.execution_capacity import WorkOrigin
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def task_host(tmp_path, monkeypatch):
    from tldw_chatbook.Agents import automatic_work_runtime, run_log
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_workflow_runs import (
        _accept_workflow_phase,
        native_workflow_qualified,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    if not native_workflow_qualified():
        pytest.skip("native workflow platform qualification required")
    setting = lambda key, default: True if key == "workflow_runs_enabled" else default
    monkeypatch.setattr(run_log, "_setting", setting)
    monkeypatch.setattr(automatic_work_runtime, "_setting", setting)
    owners = []
    stack = ExitStack()

    def make(
        *,
        mode="scoped",
        tools=("fs_read", "fs_write"),
        readonly=False,
        configure=None,
        sibling=False,
        plan_edit=None,
        exclusions=(),
        shared_db=None,
    ):
        from tldw_chatbook.Agents.workflow_task_scope import _prepare_workflow_task

        root = tmp_path / str(len(owners))
        root.mkdir(exist_ok=True)
        (root / "source.py").write_text("source\n")
        (root / "outside.py").write_text("private\n")
        if shared_db is None:
            db = AgentRunsDB(tmp_path / f"runs-{len(owners)}.db")
            stack.callback(db.close)
        else:
            db = shared_db
        db.automatic_work.recover(current_owner_id="owner")
        snapshot = db.workflow_runs.create(
            make_workflow_request(root),
            launch_id="start" if shared_db is None else f"start-{len(owners)}",
        )
        data = plan_data(root)
        data["tools"] = list(tools)
        data["phases"][0]["tasks"][0].update(
            allowed_tools=list(tools), execution_mode=mode
        )
        if plan_edit is not None:
            plan_edit(data, root)
        if sibling:
            second = dict(data["phases"][0]["tasks"][0])
            second.update(
                id="task-2",
                footprint=[{"path": "outside.py", "access": "write", "subtree": False}],
                outputs=["outside.py"],
            )
            data["phases"][0]["tasks"].append(second)
        snapshot = db.workflow_runs.save_plan(
            snapshot.id,
            WorkflowPlan.model_validate_json(json.dumps(data)),
            expected_control_revision=1,
        )
        attempt = db.automatic_work.prepare_workflow_phase(
            snapshot.id,
            revision_id=1,
            phase_id="phase-1",
            owner_id="owner",
            session_id="session",
            command_id="approve" if shared_db is None else f"approve-{len(owners)}",
        )
        session = SimpleNamespace(
            id="session", persisted_conversation_id=snapshot.request.conversation_id
        )
        runtime = ConsoleRuntime(None)
        activity = runtime.worktree_activity
        capacity = runtime.execution_capacity
        controller = SimpleNamespace(
            _disposed=False,
            _shutdown_requested=Event(),
            _agent_bridge=SimpleNamespace(runs_db=db),
            _fleet_wake=SimpleNamespace(runtime_owner_id="owner"),
            store=SimpleNamespace(sessions=lambda: [session]),
            app=SimpleNamespace(console_runtime=runtime),
        )
        runtime.set_agent_bridge(controller._agent_bridge)
        runtime.set_chat_controller(controller)
        phase = _accept_workflow_phase(controller, attempt.id, session_id="session")
        live = {"allowed": True}
        binding = dataclasses.replace(
            _agent_authority(root, "selected"),
            workspace_id="w",
            binding_id="b",
            allow_write=not readonly,
            guard=lambda write: live["allowed"] and (not write or not readonly),
            exclusions_provider=lambda: tuple(root / path for path in exclusions),
        )
        provider = make_provider(root=root, admitted_roots=(binding,), allow_write=True)
        registry = ToolCatalogRegistry()
        registry.register_provider(provider)
        if configure is not None:
            configure(registry, root, binding)
        authorization = _prepare_workflow_task(
            phase,
            task_id="task-1",
            registry=registry,
            binding=binding,
        )
        owner = capacity.begin_execution(
            origin=WorkOrigin.AUTOMATIC,
            conversation_id=snapshot.request.conversation_id,
            child=True,
        )
        owners.append(owner)
        claim = activity.try_acquire(
            owner=owner, footprint=authorization.physical_footprint
        )
        assert claim is not None
        host = SimpleNamespace(
            root=root,
            db=db,
            phase=phase,
            binding=binding,
            live=live,
            provider=provider,
            registry=registry,
            authorization=authorization,
            owner=owner,
            claim=claim,
            activity=activity,
            controller=controller,
        )

        def bind():
            run_id = db.create_run(
                conversation_id=snapshot.request.conversation_id,
                agent_kind="subagent",
                run_id=authorization.claim.launch_run_id,
                work_chain_id=snapshot.chain_id,
            )
            db.automatic_work.commit(
                authorization.claim.reservation_id, owner_id="owner"
            )
            owner.bind_run(run_id)
            authorization.bind_native(owner=owner, activity_claim=claim)
            return run_id

        host.bind = bind
        if sibling:
            second = _prepare_workflow_task(
                phase, task_id="task-2", registry=registry, binding=binding
            )
            second_owner = capacity.begin_execution(
                origin=WorkOrigin.AUTOMATIC,
                conversation_id=snapshot.request.conversation_id,
                child=True,
            )
            owners.append(second_owner)
            second_claim = activity.try_acquire(
                owner=second_owner, footprint=second.physical_footprint
            )
            db.create_run(
                conversation_id=snapshot.request.conversation_id,
                agent_kind="subagent",
                run_id=second.claim.launch_run_id,
                work_chain_id=snapshot.chain_id,
            )
            db.automatic_work.commit(second.claim.reservation_id, owner_id="owner")
            second_owner.bind_run(second.claim.launch_run_id)
            second.bind_native(owner=second_owner, activity_claim=second_claim)
            host.sibling = SimpleNamespace(
                authorization=second, owner=second_owner, registry=registry
            )
        return host

    yield make
    for owner in owners:
        owner.finish_root()
    stack.close()


def invoke(host, name, args, *, run_id=None):
    auth = host.authorization
    with (
        auth.phase.context.scope(),
        auth.scope(),
        host.owner.activate(),
        use_run_id(run_id or auth.claim.launch_run_id),
    ):
        return host.registry.invoke_by_name(name, args)


def test_prepared_claim_cannot_invoke_before_native_binding(task_host):
    host = task_host()
    result = invoke(host, "fs_read", {"path": "source.py"})
    assert not result.ok
    assert result.dispatch_state == "not_started"


@pytest.mark.parametrize("path", ["outside.py", "../outside.py", "/tmp/outside.py"])
def test_outside_path_never_reaches_actual_local_handler(task_host, path):
    host = task_host()
    host.bind()
    result = invoke(
        host, "fs_write", {"path": path, "content": "changed", "expected_absent": True}
    )
    assert not result.ok
    assert result.dispatch_state == "not_started"
    assert (host.root / "outside.py").read_text() == "private\n"


def test_wrong_alias_and_forged_task_ids_never_select_authority(task_host):
    host = task_host()
    host.bind()
    result = invoke(
        host,
        "fs_read",
        {
            "path": "source.py",
            "root_alias": "sibling",
            "task_id": host.authorization.claim.task_id,
        },
    )
    assert not result.ok
    result = invoke(host, "fs_read", {"path": "source.py"}, run_id="forged-native-run")
    assert not result.ok


def test_revocation_after_permission_wait_prevents_actual_effect(task_host):
    host = task_host()
    host.bind()
    original = host.provider._verdict_for

    def waited(*args, **kwargs):
        verdict = original(*args, **kwargs)
        host.live["allowed"] = False
        return verdict

    host.provider._verdict_for = waited
    result = invoke(host, "fs_read", {"path": "source.py"})
    assert not result.ok
    assert result.dispatch_state == "not_started"


def test_actual_read_with_live_native_custody_succeeds(task_host):
    host = task_host()
    host.bind()
    result = invoke(host, "fs_read", {"path": "source.py"})
    assert result.ok, result.error
    assert "source" in result.content


@pytest.mark.parametrize(
    "name,args",
    [
        ("fs_read", {"path": "outside.py"}),
        ("fs_list", {"path": "."}),
        ("fs_glob", {"pattern": "*.py"}),
        ("fs_grep", {"pattern": "private"}),
        (
            "fs_edit",
            {"path": "outside.py", "old_string": "private", "new_string": "changed"},
        ),
        (
            "fs_patch",
            {
                "diff": "--- a/outside.py\n+++ b/outside.py\n@@ -1 +1 @@\n-private\n+changed\n"
            },
        ),
        ("fs_mkdir", {"path": "outside-dir"}),
        ("git_status", {}),
        ("git_diff", {}),
        ("git_log", {}),
        ("git_blame", {"path": "outside.py"}),
        ("git_branches", {}),
    ],
)
def test_every_local_path_route_refuses_before_native_dispatch(task_host, name, args):
    host = task_host(tools=(name,))
    host.bind()
    result = invoke(host, name, args)
    assert not result.ok
    assert result.dispatch_state == "not_started"
    assert "scope" in result.error
    assert (host.root / "outside.py").read_text() == "private\n"
    assert not (host.root / "outside-dir").exists()


def test_forged_activity_claim_id_is_not_native_custody(task_host):
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeActivityClaim

    host = task_host(mode="exclusive")
    host.owner.bind_run(host.authorization.claim.launch_run_id)
    forged = WorktreeActivityClaim(host.claim.id, host.activity)
    with pytest.raises((ValueError, PermissionError), match="claim|custody|native"):
        host.authorization.bind_native(owner=host.owner, activity_claim=forged)


@pytest.mark.parametrize(
    "name,args",
    [
        ("spawn_subagent", {"task": "escape", "task_id": "task-1"}),
        ("new_chat", {"message": "escape", "task_id": "task-1"}),
        ("fork_chat", {"message": "escape", "task_id": "task-1"}),
        ("run_skill_script", {"skill": "x", "script": "x.py", "args": []}),
    ],
)
def test_runtime_effect_callback_cannot_escape_task(task_host, name, args):
    from Tests.Agents.test_agent_runtime import make_deps
    from tldw_chatbook.Agents.agent_models import AgentConfig, ModelTurn, ToolResult
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop

    host = task_host(tools=("fs_read", name))
    host.bind()
    effects = []
    callback = lambda *a, **kw: (
        effects.append((a, kw)) or ToolResult(ok=True, content="effect")
    )
    deps = make_deps(
        [
            ModelTurn(tool_calls=(ToolCall(name=name, args=args),)),
            ModelTurn(text="done"),
        ],
        spawn=callback,
        new_chat=callback,
        fork_chat=callback,
    )
    deps.run_skill_script = callback
    with (
        host.phase.context.scope(),
        host.authorization.scope(),
        host.owner.activate(),
        use_run_id(host.owner.run_id),
    ):
        run_agent_loop(
            AgentConfig(model="m", system_prompt="s", allowed_tools=(name,)),
            [],
            [],
            deps,
        )
    assert effects == []


def test_progressive_discovery_cannot_advertise_unplanned_tools(task_host):
    host = task_host(tools=("fs_read", "find_tools", "load_tools"))
    host.bind()
    with (
        host.phase.context.scope(),
        host.authorization.scope(),
        host.owner.activate(),
        use_run_id(host.owner.run_id),
    ):
        assert "fs_write" not in {entry.name for entry in host.registry.list_catalog()}
        assert "fs_write" not in {entry.name for entry in host.registry.find("write")}
        with pytest.raises(KeyError):
            host.registry.load_schema("local:fs_write")


def test_preapproved_session_shell_cannot_execute_scoped_task(task_host):
    from Tests.Agents.test_raw_shell_tool_provider import _provider, _RuntimeProbe

    runtime = _RuntimeProbe()
    runtime.model_session_granted = lambda session: True
    host = task_host(
        tools=("fs_read", "shell_exec"),
        configure=lambda registry, root, binding: registry.register_provider(
            _provider(root, runtime=runtime)
        ),
    )
    host.bind()
    result = invoke(host, "shell_exec", {"command": "pwd"})
    assert not result.ok
    assert runtime.execute_calls == []


def test_shared_provider_does_not_use_sibling_authorization(task_host):
    host = task_host()
    host.bind()
    with (
        host.phase.context.scope(),
        host.authorization.scope(),
        host.owner.activate(),
        use_run_id("sibling"),
    ):
        result = host.provider.invoke("local:fs_read", {"path": "source.py"})
    assert not result.ok
    assert invoke(host, "fs_read", {"path": "source.py"}).ok


def test_virtual_cli_read_respects_same_footprint(task_host):
    from Tests.Agents.test_virtual_cli_provider import ALLOW
    from Tests.Agents.test_virtual_cli_provider import make_provider as virtual_provider

    host = task_host(
        tools=("fs_read", "virtual_cli"),
        configure=lambda registry, root, binding: registry.register_provider(
            virtual_provider(root, state=ALLOW, admitted_roots=(binding,))
        ),
    )
    host.bind()
    result = invoke(host, "virtual_cli", {"command": "cat", "argv": ["outside.py"]})
    assert not result.ok
    assert result.dispatch_state == "not_started"


def test_native_service_consumes_same_reservation_and_binds_actual_run(task_host):
    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
    from tldw_chatbook.Agents.agent_service import AgentService

    host = task_host()
    chat = ScriptedChat([fence("fs_read", {"path": "source.py"}), "done"])
    with host.phase.context.scope():
        service = AgentService(
            db=host.db,
            registry=host.registry,
            chat_call=chat,
            runtime_capacity=host.owner.capacity,
        )
        run_id, outcome = service._run_owned(
            execution_owner=host.owner,
            automatic_child_reservation=host.authorization.claim.reservation_id,
            workflow_task_authorization=host.authorization,
            workflow_activity_claim=host.claim,
            requested_run_id=host.authorization.claim.launch_run_id,
            conversation_id=host.authorization.claim.conversation_id,
            messages=[{"role": "user", "content": "read source"}],
            config=AgentConfig(
                model="test-model",
                system_prompt="test",
                allowed_tools=("fs_read",),
                budget=RunBudget(max_tool_call_seconds=5),
            ),
            api_endpoint="openai",
            should_cancel=lambda: False,
            agent_kind="subagent",
            task="read",
            parent_run_id=None,
        )
    assert run_id == host.authorization.claim.launch_run_id
    assert outcome.final_text == "done"
    assert len(host.db.list_runs(host.authorization.claim.conversation_id)) == 1
    assert any("source" in str(call.get("messages_payload")) for call in chat.calls[1:])


def test_private_issuer_rejects_forged_phase_and_duplicate_preparation(task_host):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.workflow_task_scope import _prepare_workflow_task

    host = task_host()
    for phase in (
        host.phase.attempt.id,
        SimpleNamespace(context=host.phase.context),
        host.phase,
    ):
        with pytest.raises(AutomaticWorkRefused):
            _prepare_workflow_task(
                phase, task_id="task-1", registry=host.registry, binding=host.binding
            )


def test_readonly_binding_never_becomes_workflow_write_authority(task_host):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    with pytest.raises(AutomaticWorkRefused, match="readonly"):
        task_host(readonly=True)


def test_effectful_hook_activation_revokes_dispatch_without_running_hook(task_host):
    host = task_host(mode="exclusive")
    host.bind()
    host.controller.app.console_runtime._run_hooks_engine = object()
    assert not invoke(host, "fs_read", {"path": "source.py"}).ok


def test_activity_claim_query_rejects_wrong_footprint_and_drained_owner(task_host):
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    host = task_host()
    with pytest.raises(ValueError, match="claim"):
        host.activity.validate_claim(
            host.claim, owner=host.owner, footprint=WorktreeFootprint()
        )
    host.bind()
    host.owner.finish_root()
    with pytest.raises(ValueError, match="claim"):
        host.activity.validate_claim(
            host.claim,
            owner=host.owner,
            footprint=host.authorization.physical_footprint,
        )


@pytest.mark.parametrize("mode,allowed", [("scoped", False), ("exclusive", True)])
def test_unknown_effect_mcp_requires_actual_exclusive_claim(task_host, mode, allowed):
    import asyncio
    import threading

    from Tests.Agents.test_mcp_tool_provider import (
        FakeMCPService,
        _catalog_record,
        _compose,
        _tool_dict,
    )
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("effect")])],
        default_state=EffectiveToolState(state="allow", origin="tool_override"),
    )
    provider = MCPToolProvider(service=service, main_loop=loop)
    _compose(provider)
    name = provider.list_catalog()[0].name
    try:
        host = task_host(
            mode=mode,
            tools=("fs_read", name),
            configure=lambda registry, root, binding: registry.register_provider(
                provider
            ),
        )
        host.bind()
        result = invoke(host, name, {})
        assert result.ok is allowed, result.error
        assert bool(service.execute_calls) is allowed
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        loop.close()


def test_alternate_builtin_writer_cannot_bypass_atomic_task_writer(task_host):
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
    from tldw_chatbook.Tools.file_operation_tools import WriteFileTool

    provider = BuiltinToolProvider()
    provider._tools["write_file"] = WriteFileTool()
    host = task_host(
        tools=("fs_read", "write_file"),
        configure=lambda registry, root, binding: registry.register_provider(provider),
    )
    host.bind()
    result = invoke(
        host,
        "write_file",
        {"file_path": str(host.root / "source.py"), "content": "bad"},
    )
    assert not result.ok
    assert result.dispatch_state == "not_started"
    assert (host.root / "source.py").read_text() == "source\n"


def test_skill_provider_cannot_create_unregistered_child(task_host):
    from Tests.Agents.test_agent_runtime import make_deps
    from tldw_chatbook.Agents.agent_models import AgentConfig, ModelTurn, ToolResult
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop
    from tldw_chatbook.Agents.tool_catalog import SkillToolProvider

    provider = SkillToolProvider([{"name": "escape", "description": "escape"}])
    name = provider.list_catalog()[0].name
    host = task_host(
        tools=("fs_read", name),
        configure=lambda registry, root, binding: registry.register_provider(provider),
    )
    host.bind()
    effects = []
    deps = make_deps(
        [
            ModelTurn(tool_calls=(ToolCall(name, {"args": "spawn"}),)),
            ModelTurn(text="done"),
        ],
        invoke=lambda call: effects.append(call) or ToolResult(ok=True),
    )
    with (
        host.phase.context.scope(),
        host.authorization.scope(),
        host.owner.activate(),
        use_run_id(host.owner.run_id),
    ):
        run_agent_loop(
            AgentConfig(model="m", system_prompt="s", allowed_tools=(name,)),
            [],
            [provider.load_schema(provider.list_catalog()[0].id)],
            deps,
        )
    assert effects == []


def test_promotion_early_route_cannot_widen_task_footprint(task_host, monkeypatch):
    host = task_host()
    host.bind()
    effects = []
    monkeypatch.setattr(
        host.provider, "_invoke_promotion", lambda *a, **kw: effects.append(a)
    )
    result = invoke(
        host,
        "fs_write",
        {
            "path": "outside.py",
            "content": "bad",
            "repository_instruction_promotion": {
                "task_id": host.authorization.claim.task_id
            },
        },
    )
    assert not result.ok
    assert effects == []


def test_phase_stop_during_permission_wait_prevents_handler(task_host):
    host = task_host()
    host.bind()
    original = host.provider._verdict_for

    def waited(*args, **kwargs):
        verdict = original(*args, **kwargs)
        host.controller._disposed = True
        return verdict

    host.provider._verdict_for = waited
    result = invoke(host, "fs_read", {"path": "source.py"})
    assert not result.ok
    assert result.dispatch_state == "not_started"


def test_workflow_read_then_write_uses_existing_observation_writer(task_host):
    host = task_host()
    host.bind()
    assert invoke(host, "fs_read", {"path": "source.py"}).ok
    result = invoke(host, "fs_write", {"path": "source.py", "content": "updated\n"})
    assert result.ok, result.error
    assert (host.root / "source.py").read_text() == "updated\n"


def test_model_task_id_cannot_widen_path_even_for_same_actual_run(task_host):
    host = task_host()
    host.bind()
    result = invoke(
        host,
        "fs_read",
        {
            "path": "outside.py",
            "workflow_id": host.authorization.claim.workflow_id,
            "task_id": host.authorization.claim.task_id,
            "run_id": host.owner.run_id,
        },
    )
    assert not result.ok
    assert result.dispatch_state == "not_started"


def test_existing_permission_deny_still_wins_in_exclusive_mode(task_host):
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    host = task_host(mode="exclusive")
    host.bind()
    host.provider._resolve_state = lambda hub: EffectiveToolState(
        state="deny", origin="tool_override"
    )
    assert not invoke(host, "fs_read", {"path": "source.py"}).ok


def test_two_real_siblings_share_provider_without_overwriting_scope(task_host):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    host = task_host(sibling=True)
    host.bind()
    barrier = Barrier(2)

    def read(h, own, other):
        barrier.wait(2)
        return invoke(h, "fs_read", {"path": own}), invoke(
            h, "fs_read", {"path": other}
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(read, host, "source.py", "outside.py")
        second = pool.submit(read, host.sibling, "outside.py", "source.py")
        for own, other in (first.result(5), second.result(5)):
            assert own.ok, own.error
            assert not other.ok


def test_nested_or_same_path_alias_never_replaces_captured_binding(task_host):
    host = task_host()
    host.bind()
    nested = host.root / "nested"
    nested.mkdir()
    (nested / "source.py").write_text("nested")
    for root in (host.root, nested):
        other = dataclasses.replace(
            host.binding, root=root, alias="other", binding_id="other"
        )
        host.provider._admitted_roots["other"] = other
        result = invoke(host, "fs_read", {"path": "source.py", "root_alias": "other"})
        assert not result.ok
    host.provider._admitted_roots.pop("other")
    host.provider.admit_run_workspace_root(host.owner.run_id, other)
    assert not invoke(
        host, "fs_read", {"path": "source.py", "root_alias": host.binding.alias}
    ).ok


@pytest.mark.parametrize("mode", ["scoped", "exclusive"])
def test_script_subprocess_refuses_without_owned_cleanup_contract(
    task_host, monkeypatch, mode
):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Skills_Interop import skill_script_runner as runner

    host = task_host(mode=mode, tools=("fs_read", "run_skill_script"))
    host.bind()
    effects = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: effects.append(a))
    with (
        host.phase.context.scope(),
        host.authorization.scope(),
        host.owner.activate(),
        use_run_id(host.owner.run_id),
        pytest.raises(AutomaticWorkRefused),
    ):
        runner.run_script_subprocess(
            ["/bin/echo", "no"], cwd=host.root, limits=runner.ScriptRunLimits()
        )
    assert effects == []


def test_phase_without_task_cannot_fall_back_to_ordinary_script_authority(
    task_host, monkeypatch
):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Skills_Interop import skill_script_runner as runner

    host = task_host(mode="exclusive", tools=("fs_read", "run_skill_script"))
    effects = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: effects.append(a))
    with host.phase.context.scope(), pytest.raises(AutomaticWorkRefused):
        runner.run_script_subprocess(
            ["/bin/echo", "no"], cwd=host.root, limits=runner.ScriptRunLimits()
        )
    assert effects == []
    assert not host.authorization.permits_name("run_skill_script")


def test_native_viewless_runtime_identity_survives_detach_but_not_replacement(
    task_host,
):
    host = task_host()
    host.bind()
    del host.controller.app
    assert invoke(host, "fs_read", {"path": "source.py"}).ok
    host.controller._hooks_v2_runtime.set_chat_controller(None)
    assert not invoke(host, "fs_read", {"path": "source.py"}).ok


def test_native_binding_cannot_repeat_or_create_second_row(task_host):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    host = task_host()
    host.bind()
    with pytest.raises(AutomaticWorkRefused, match="already_bound"):
        host.authorization.bind_native(owner=host.owner, activity_claim=host.claim)
    assert len(host.db.list_runs(host.authorization.claim.conversation_id)) == 1


def _prepare_native_binding(host, *, row="matching"):
    claim = host.authorization.claim
    if row != "missing":
        host.db.create_run(
            conversation_id=(
                "other-conversation" if row == "conversation" else claim.conversation_id
            ),
            agent_kind="subagent",
            run_id=claim.launch_run_id,
            work_chain_id=None if row in {"chain", "conversation"} else claim.chain_id,
        )
    host.db.automatic_work.commit(claim.reservation_id, owner_id=claim.owner_id)
    host.owner.bind_run(claim.launch_run_id)
    host.authorization.check_preparation()
    host.activity.validate_claim(
        host.claim, owner=host.owner, footprint=host.authorization.physical_footprint
    )


def _assert_unbound_and_no_dispatch(host, monkeypatch):
    assert host.authorization._binding_state == []
    with host.db.connection() as conn:
        row = conn.execute(
            "SELECT run_id FROM workflow_task_attempts WHERE id=?",
            (host.authorization.claim.id,),
        ).fetchone()
    assert row[0] is None
    effects = []
    with monkeypatch.context() as patch:
        patch.setattr(host.provider, "invoke", lambda *a, **kw: effects.append(a))
        result = invoke(host, "fs_write", {"path": "source.py", "content": "bad"})
    assert not result.ok
    assert result.dispatch_state == "not_started"
    assert effects == []
    assert (host.root / "source.py").read_text() == "source\n"


@pytest.mark.parametrize("row", ["missing", "conversation", "chain"])
def test_native_binding_reaches_durable_lineage_fence_under_claimed_run_id(
    task_host, monkeypatch, row
):
    host = task_host()
    _prepare_native_binding(host, row=row)
    ledger = host.phase._ledger
    original = ledger.bind_workflow_task_run
    reached = []

    def observed(*args, **kwargs):
        reached.append(kwargs["run_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(ledger, "bind_workflow_task_run", observed)
    with pytest.raises(ValueError, match="^invalid_workflow_run_ownership$"):
        host.authorization.bind_native(owner=host.owner, activity_claim=host.claim)
    assert reached == [host.authorization.claim.launch_run_id]
    _assert_unbound_and_no_dispatch(host, monkeypatch)


def test_native_binding_rejects_another_live_member_of_same_claim(
    task_host, monkeypatch
):
    host = task_host()
    _prepare_native_binding(host)
    other = host.owner.capacity.begin_execution(
        origin=WorkOrigin.AUTOMATIC,
        conversation_id=host.owner.conversation_id,
        child=True,
    )
    try:
        other.bind_run(host.owner.run_id)
        member = host.claim.attach_child(other)
        assert member.id != host.claim.id
        assert other.is_admitted and host.owner.is_admitted
        with pytest.raises(
            ValueError, match="^live original activity claim custody required$"
        ):
            host.authorization.bind_native(owner=other, activity_claim=host.claim)
        _assert_unbound_and_no_dispatch(host, monkeypatch)
        host.authorization.bind_native(owner=host.owner, activity_claim=host.claim)
        assert invoke(host, "fs_read", {"path": "source.py"}).ok
    finally:
        other.finish_root()


@pytest.mark.parametrize("foreign", ["claim", "capacity"])
def test_native_binding_rejects_other_runtime_valid_custody(
    task_host, monkeypatch, foreign
):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    host = task_host()
    _prepare_native_binding(host)
    runtime = ConsoleRuntime(None)
    other = runtime.execution_capacity.begin_execution(
        origin=WorkOrigin.AUTOMATIC,
        conversation_id=host.owner.conversation_id,
        child=True,
    )
    try:
        other.bind_run(host.owner.run_id)
        claim = runtime.worktree_activity.try_acquire(
            owner=other, footprint=host.authorization.physical_footprint
        )
        assert claim is not None
        runtime.worktree_activity.validate_claim(
            claim, owner=other, footprint=host.authorization.physical_footprint
        )
        assert other.is_admitted and other.capacity is not host.owner.capacity
        reached = []
        original = host.activity.validate_claim

        def observed(*args, **kwargs):
            reached.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(host.activity, "validate_claim", observed)
        if foreign == "claim":
            with pytest.raises(
                ValueError, match="^live original activity claim custody required$"
            ):
                host.authorization.bind_native(owner=host.owner, activity_claim=claim)
            assert reached == [claim]
        else:
            with pytest.raises(
                AutomaticWorkRefused, match="^workflow_task_native_owner_required$"
            ):
                host.authorization.bind_native(owner=other, activity_claim=claim)
            assert reached == []
        _assert_unbound_and_no_dispatch(host, monkeypatch)
        host.authorization.bind_native(owner=host.owner, activity_claim=host.claim)
        assert invoke(host, "fs_read", {"path": "source.py"}).ok
    finally:
        other.finish_root()


def test_native_binding_matching_durable_lineage_allows_real_dispatch(task_host):
    host = task_host()
    _prepare_native_binding(host)
    host.authorization.bind_native(owner=host.owner, activity_claim=host.claim)
    result = invoke(host, "fs_read", {"path": "source.py"})
    assert result.ok
    assert "source" in result.content


def test_mcp_main_loop_wait_rechecks_task_before_remote_execution(task_host):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from Tests.Agents.test_mcp_tool_provider import (
        FakeMCPService,
        _catalog_record,
        _compose,
        _tool_dict,
    )
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    loop = asyncio.new_event_loop()
    entered, release, submitted = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("effect")])],
        default_state=EffectiveToolState(state="allow", origin="tool_override"),
    )
    provider = MCPToolProvider(service=service, main_loop=loop)
    _compose(provider)
    name = provider.list_catalog()[0].name
    host = task_host(
        mode="exclusive",
        tools=(name,),
        configure=lambda registry, root, binding: registry.register_provider(provider),
    )
    host.bind()

    def block():
        entered.set()
        release.wait(5)

    loop.call_soon_threadsafe(block)
    assert entered.wait(2)
    original = provider._execute

    def observed(*args, **kwargs):
        submitted.set()
        return original(*args, **kwargs)

    provider._execute = observed
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(invoke, host, name, {})
            assert submitted.wait(2)
            host.controller._disposed = True
            release.set()
            result = future.result(5)
        assert not result.ok
        assert result.dispatch_state == "not_started"
        assert service.execute_calls == []
    finally:
        release.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        loop.close()


def test_outside_call_cannot_trigger_lazy_instruction_reads_or_review(task_host):
    from Tests.Agents.test_agent_runtime import make_deps
    from tldw_chatbook.Agents.agent_models import (
        AgentConfig,
        ModelTurn,
    )
    from tldw_chatbook.Agents.agent_runtime import ToolBatchPreparation, run_agent_loop

    host = task_host()
    host.bind()
    discovered, reviewed = [], []
    deps = make_deps(
        [
            ModelTurn(tool_calls=(ToolCall("fs_read", {"path": "outside.py"}),)),
            ModelTurn(text="done"),
        ]
    )
    deps.prepare_tool_calls = lambda calls: (
        discovered.extend(calls) or ToolBatchPreparation("proceed")
    )
    deps.review_tool_calls = lambda calls: reviewed.extend(calls) or {}
    with (
        host.phase.context.scope(),
        host.authorization.scope(),
        host.owner.activate(),
        use_run_id(host.owner.run_id),
    ):
        run_agent_loop(
            AgentConfig(model="m", system_prompt="s", allowed_tools=("fs_read",)),
            [],
            [],
            deps,
        )
    assert discovered == []
    assert reviewed == []


def test_native_guidance_candidates_are_physical_reads_without_model_grants(task_host):
    host = task_host()
    auth = host.authorization
    candidates = {host.root / "AGENTS.md", host.root / "AGENTS.override.md"}
    assert candidates <= {scope.path for scope in auth.physical_footprint.reads}
    assert not candidates & {scope.path for scope in auth.footprint.reads}
    host.bind()
    (host.root / "AGENTS.md").write_text("untrusted guidance")
    result = invoke(host, "fs_read", {"path": "AGENTS.md"})
    assert not result.ok


def test_native_guidance_claim_protects_absent_candidate_creation(task_host):
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    host = task_host()
    auth = host.authorization
    candidate = host.root / "AGENTS.override.md"
    assert not candidate.exists()
    root = auth.physical_footprint.reads[0].root
    with host.activity.prelaunch(
        footprint=WorktreeFootprint(writes=(WorktreeScope(candidate, False, root),))
    ) as contender:
        assert contender is None


def test_native_guidance_includes_ancestors_for_exact_nested_output(task_host):
    def nested(data, root):
        (root / "nested" / "deep").mkdir(parents=True)
        task = data["phases"][0]["tasks"][0]
        task["footprint"] = [
            {"path": "nested/deep/result.py", "access": "write", "subtree": False}
        ]
        task["outputs"] = ["nested/deep/result.py"]

    host = task_host(plan_edit=nested)
    reads = {scope.path for scope in host.authorization.physical_footprint.reads}
    for directory in (host.root, host.root / "nested", host.root / "nested" / "deep"):
        assert {directory / "AGENTS.md", directory / "AGENTS.override.md"} <= reads


def test_native_guidance_excludes_hidden_candidate(task_host):
    host = task_host(exclusions=("AGENTS.override.md",))
    reads = {s.path for s in host.authorization.physical_footprint.reads}
    assert host.root / "AGENTS.override.md" not in reads
    assert host.root / "AGENTS.md" in reads


@pytest.mark.parametrize("initially_absent", [False, True])
@pytest.mark.parametrize("filename", ["AGENTS.md", "AGENTS.override.md"])
def test_native_mkdir_guidance_is_protected_on_directory_revisit(
    task_host, initially_absent, filename
):
    from Tests.Agents.test_project_instruction_runtime import _payload, _snapshot
    from tldw_chatbook.Agents.project_instruction_runtime import (
        InstructionActivationLedger,
    )
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    def directory_task(data, root):
        task = data["phases"][0]["tasks"][0]
        task["footprint"] = [{"path": "nested", "access": "write", "subtree": False}]
        task["outputs"] = ["nested"]
        if not initially_absent:
            (root / "nested").mkdir()

    host = task_host(tools=("fs_mkdir", "fs_read"), plan_edit=directory_task)
    host.bind()
    nested = host.root / "nested"
    ledger = InstructionActivationLedger(_snapshot(host.root), nested_max_bytes=100)
    payload, _ = _payload()
    calls = [ToolCall("fs_mkdir", {"path": "nested"}, "mkdir")]
    # Real native provider directory mapping and guidance preparation precede
    # actual mkdir, including when the destination has never existed.
    ledger.prepare(calls, "primary", host.registry, payload)
    result = invoke(host, "fs_mkdir", {"path": "nested"})
    assert result.ok, result
    assert nested.is_dir()
    # Simulate an external edit; runtime admission is not an OS sandbox. The
    # same dispatch must protect discovery when this directory is revisited.
    candidate = nested / filename
    candidate.write_text("nested instruction")
    delivery = ledger.prepare(calls, "primary", host.registry, payload)
    assert any("nested instruction" in row["content"] for row in delivery.rows)
    assert invoke(host, "fs_mkdir", {"path": "nested"}).ok
    assert not invoke(host, "fs_read", {"path": f"nested/{filename}"}).ok
    chain = host.authorization.footprint.writes[0].root
    with host.activity.prelaunch(
        footprint=WorktreeFootprint(writes=(WorktreeScope(candidate, False, chain),))
    ) as contender:
        assert contender is None


def test_mkdir_capable_exact_file_scope_preserves_native_file_access(task_host):
    host = task_host(tools=("fs_read", "fs_mkdir"))
    host.bind()
    assert invoke(host, "fs_read", {"path": "source.py"}).ok
    assert not invoke(host, "fs_read", {"path": "AGENTS.md"}).ok
