"""The genuine Console producer can prepare a task without a primary turn."""

import asyncio
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Agents.workflow_support import make_workflow_request, plan_data
from Tests.Chat.test_automatic_provider_budget import resolution, response
from tldw_chatbook.Agents.workflow_models import WorkflowPlan, WorkflowTaskResult
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import (
    capture_run_admitted_workspace_roots,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_goal_runs import goal_provider_ref
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_workflow_runs import _accept_workflow_phase
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.fixture
async def native_host(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_AGENTS_WORKFLOW_RUNS_ENABLED", "true")
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    db = AgentRunsDB(tmp_path / "runs.db")
    chat = CharactersRAGDB(tmp_path / "chat.db", "workflow")
    workspace_db = WorkspaceDB(tmp_path / "workspaces.db")
    registry = LocalWorkspaceRegistryService(workspace_db)
    registry.create_workspace(workspace_id="w", name="Workflow")
    root = tmp_path / "project"
    root.mkdir()
    (root / "source.py").write_text("original source\n")
    registry.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="w",
            binding_id="b",
            binding_kind="local-filesystem",
            label="Project",
            locator=str(root),
            status="ready",
            metadata={"access": "rw"},
        )
    )
    persistence = ChatPersistenceService(chat, workspace_registry=registry)
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(
        session_id="native-workflow",
        workspace_id="w",
        project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
    )
    conversation_id = store.persist_session_if_needed(session.id)
    selected = resolution(model="test-model")
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return response()

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider)
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    runtime = ConsoleRuntime(
        SimpleNamespace(
            app_config={},
            workspace_registry_service=registry,
            chachanotes_db=chat,
        )
    )
    runtime._chat_store = store
    runtime.set_agent_bridge(bridge)
    controller = runtime.ensure_chat_controller(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    service = runtime.ensure_workflow_service()
    owner_id = controller._fleet_wake.runtime_owner_id
    controller._fleet_wake.start_recovery()
    assert await controller._fleet_wake.wait_for_recovery()
    request = make_workflow_request(root).model_copy(
        update={
            "conversation_id": conversation_id,
            "targets": (goal_provider_ref(selected),),
        }
    )
    workflow = db.workflow_runs.create(request, launch_id="native-start")
    data = plan_data(root)
    data["request"] = request.model_dump(mode="json")
    data["tools"] = []
    data["phases"][0]["tasks"][0].update(
        target=request.targets[0].model_dump(mode="json"),
        allowed_tools=[],
        instructions="Only the approved task",
        outputs=[],
        expected_checks=[],
    )
    workflow = db.workflow_runs.save_plan(
        workflow.id,
        WorkflowPlan.model_validate_json(json.dumps(data)),
        expected_control_revision=workflow.control_revision,
    )
    attempt = db.automatic_work.prepare_workflow_phase(
        workflow.id,
        revision_id=1,
        phase_id="phase-1",
        owner_id=owner_id,
        session_id=session.id,
        command_id="native-approve",
    )
    phase = _accept_workflow_phase(controller, attempt.id, session_id=session.id)
    binding = capture_run_admitted_workspace_roots(session=session, registry=registry)[
        0
    ]
    host = SimpleNamespace(**locals())
    capacity = runtime.execution_capacity
    try:
        yield host
    finally:
        await runtime.dispose()
        assert capacity.snapshot().executions == ()
        await gateway.aclose()
        db.close()
        chat.close_connection()
        workspace_db.close()


async def test_task_preparation_uses_native_service_without_primary_effects(
    native_host,
):
    h = native_host
    before = tuple(h.store.messages_for_session(h.session.id))
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-1",
        binding=h.binding,
        resolution=h.selected,
    )
    try:
        assert (
            prepared.authorization.registry
            is prepared.native.first_request_plan.registry
        )
        assert prepared.native.service.registry is prepared.authorization.registry
        assert prepared.native.service.db is h.db
        assert h.calls == []
        assert h.db.list_runs(h.conversation_id) == []
        assert tuple(h.store.messages_for_session(h.session.id)) == before
        assert h.bridge.live_primary_run_id(h.conversation_id) is None
        assert h.conversation_id not in h.bridge._fleet_services
        assert h.runtime.execution_capacity.snapshot().executions == ()
        assert not h.runtime._preparation_reads
        assert [
            row["content"] for row in prepared.native.first_request_plan.messages
        ].count("Only the approved task") == 1
    finally:
        await prepared.close()


async def test_task_preparation_rejects_foreign_phase_before_effects(native_host):
    h = native_host
    with pytest.raises(PermissionError):
        await h.bridge._prepare_workflow_reply(
            phase=object(),
            task_id="task-1",
            binding=h.binding,
            resolution=h.selected,
        )
    assert h.calls == []
    assert h.db.list_runs(h.conversation_id) == []


async def test_original_native_child_and_durable_settlement_own_finalization(
    native_host,
):
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-1",
        binding=h.binding,
        resolution=h.selected,
    )
    native = prepared.native

    def launch_and_join():
        with owned_goal_storage(h.service), h.phase.context.scope():
            receipt = native.service.launch_workflow_task(
                prepared.authorization.task,
                authorization=prepared.authorization,
                should_cancel=lambda: False,
            )
            for thread in prepared.authorization._native_preparation[
                0
            ].threads.values():
                thread.join(5)
                assert not thread.is_alive()
            return receipt

    receipt = await asyncio.to_thread(launch_and_join)
    h.service.bind_native_receipt(prepared.authorization, receipt)
    assert len(h.calls) == 1, h.db.get_run(receipt.run_id)
    assert len(h.db.list_runs(h.conversation_id)) == 1
    assert h.db.get_run(receipt.run_id)["agent_kind"] == "subagent"
    with pytest.raises(ValueError, match="checkpoint_missing"):
        await prepared.close()
    assert not native.turn_lifeline.loop.is_closed()
    result = WorkflowTaskResult(
        workflow_id=h.workflow.id,
        attempt_id=receipt.attempt_id,
        run_id=receipt.run_id,
        status="completed",
        summary="Native task completed",
        handoff="Complete required handoff",
    )
    await h.service.call(h.service.checkpoint_task, result, drained=True)
    await prepared.close()
    await prepared.close()
    assert native.turn_lifeline.loop.is_closed()
    assert h.bridge.live_primary_run_id(h.conversation_id) is None
    assert not h.gateway._loop_clients


async def test_duplicate_and_detached_preparation_preserve_original_resources(
    native_host,
):
    h = native_host
    cancelled = threading.Event()
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-1",
        binding=h.binding,
        resolution=h.selected,
        should_cancel=cancelled.is_set,
    )
    original = prepared.authorization._native_preparation[0]
    try:
        with pytest.raises(PermissionError, match="already prepared"):
            await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
            )
        assert prepared.authorization._native_preparation == [original]
        assert original.service is prepared.native.service
        assert not prepared.native.turn_lifeline.loop.is_closed()
        h.runtime._agent_bridge = None
        with pytest.raises(PermissionError, match="source changed"):
            prepared.authorization.check_preparation()
        h.runtime._agent_bridge = h.bridge
        cancelled.set()
        with pytest.raises(PermissionError, match="source changed"):
            prepared.authorization.check_preparation()
        cancelled.clear()
        h.store._sessions.pop(h.session.id)
        with pytest.raises(PermissionError, match="source changed"):
            prepared.authorization.check_preparation()
        h.store._sessions[h.session.id] = h.session
    finally:
        await prepared.close()
    with pytest.raises(PermissionError, match="retired"):
        prepared.authorization.check_preparation()


@pytest.mark.parametrize("change", ["target", "capture", "history"])
async def test_unsupported_task_source_refuses_before_native_resources(
    native_host, change
):
    h = native_host
    kwargs = {
        "phase": h.phase,
        "task_id": "task-1",
        "binding": h.binding,
        "resolution": h.selected,
    }
    if change == "target":
        kwargs["resolution"] = replace(h.selected, model="other-model")
    elif change == "capture":
        h.bridge._change_tracker = object()
    else:
        kwargs["agent_messages"] = [
            {"role": "user", "content": "private parent sentinel"}
        ]
    with pytest.raises(PermissionError):
        await h.bridge._prepare_workflow_reply(**kwargs)
    h.bridge._change_tracker = None
    assert not h.phase._task_scopes
    assert not h.calls
    assert not h.db.list_runs(h.conversation_id)


async def test_cancelled_finite_read_retains_root_and_drains_before_delivery(
    native_host, monkeypatch
):
    from tldw_chatbook.Chat import console_workflow_preparation as preparation
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    h = native_host
    entered, release = threading.Event(), threading.Event()
    actual = h.service.prepare_task_inputs
    resources = []
    original_enter = preparation._PreparedWorkflowReply._enter

    def remember(owner):
        resources.append(owner)
        return original_enter(owner)

    def hold(authorization):
        entered.set()
        assert release.wait(5)
        return actual(authorization)

    monkeypatch.setattr(preparation._PreparedWorkflowReply, "_enter", remember)
    monkeypatch.setattr(h.service, "prepare_task_inputs", hold)
    task = asyncio.create_task(
        h.bridge._prepare_workflow_reply(
            phase=h.phase,
            task_id="task-1",
            binding=h.binding,
            resolution=h.selected,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert h.runtime._preparation_reads
        chain = capture_directory_chain(h.root)
        write = WorktreeFootprint(
            writes=(WorktreeScope(h.root / "source.py", False, chain),)
        )
        with h.runtime.worktree_activity.prelaunch(footprint=write) as claim:
            assert claim is None
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not h.runtime._preparation_reads
    assert not h.controller._preparation_reads
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert resources[0].native.turn_lifeline.loop.is_closed()
    with h.runtime.worktree_activity.prelaunch(footprint=write) as claim:
        assert claim is not None
    assert not h.calls


async def test_original_guidance_is_read_under_claim_and_confirmed_outside_it(
    native_host, monkeypatch
):
    from tldw_chatbook.Agents.project_instruction_resolver import (
        ProjectInstructionResolver,
    )
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    h = native_host
    (h.root / "AGENTS.md").write_text("Original native instruction sentinel\n")
    h.session.project_instruction_state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id=h.binding.binding_id,
        working_folder_locator_fingerprint=h.binding.locator_fingerprint,
    )
    chain = capture_directory_chain(h.root)
    write = WorktreeFootprint(
        writes=(WorktreeScope(h.root / "AGENTS.md", False, chain),)
    )
    reads, notices = [], []
    original = ProjectInstructionResolver.resolve_startup

    def resolve(resolver, **kwargs):
        with h.runtime.worktree_activity.prelaunch(footprint=write) as claim:
            assert claim is None
        reads.append(threading.get_ident())
        return original(resolver, **kwargs)

    def confirm(snapshot):
        with h.runtime.worktree_activity.prelaunch(footprint=write) as claim:
            assert claim is not None
        notices.append(snapshot)
        return "proceed"

    monkeypatch.setattr(ProjectInstructionResolver, "resolve_startup", resolve)
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-1",
        binding=h.binding,
        resolution=h.selected,
        confirm_project_instruction_dispatch=confirm,
    )
    try:
        captured = prepared.authorization._native_preparation[0].kwargs[
            "startup_instruction_snapshot"
        ]
        assert reads and reads[0] != threading.get_ident()
        assert notices == [captured]
        assert captured.startup_source.body == "Original native instruction sentinel\n"
        assert prepared.native.project_instruction_context._ledger is not None
        assert not h.calls
        assert not h.runtime._preparation_reads
    finally:
        await prepared.close()


async def test_required_seed_is_never_replaced_by_parent_or_silently_shrunk(
    native_host, monkeypatch
):
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    h = native_host
    h.store.append_message(
        h.session.id, role=ConsoleMessageRole.USER, content="PRIVATE PARENT SENTINEL"
    )
    actual = bridge_module.build_console_first_request_plan
    observed = []

    def bounded(**kwargs):
        plan = actual(**kwargs)
        observed.append(plan)
        return replace(plan, schemas=replace(plan.schemas, request_fits=False))

    monkeypatch.setattr(bridge_module, "build_console_first_request_plan", bounded)
    with pytest.raises(PermissionError, match="seed does not fit"):
        await h.bridge._prepare_workflow_reply(
            phase=h.phase,
            task_id="task-1",
            binding=h.binding,
            resolution=h.selected,
        )
    assert len(observed) == 1
    assert "PRIVATE PARENT SENTINEL" not in json.dumps(observed[0].messages)
    assert observed[0].messages[0] == {
        "role": "user",
        "content": "Only the approved task",
    }
    assert len(observed[0].messages) == 2
    assert not h.calls
    assert not h.db.list_runs(h.conversation_id)
    prepared = h.phase._task_scopes["task-1"]._bridge_preparation[0]
    assert prepared._closed
    assert prepared.native.turn_lifeline.loop.is_closed()


async def test_preparation_wait_has_no_task_claim_and_retries_after_actual_release(
    native_host,
):
    from tldw_chatbook.Agents.agent_service import _WorkflowChildWait
    from tldw_chatbook.Agents.execution_capacity import WorkOrigin
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    h = native_host
    chain = capture_directory_chain(h.root)
    owner = h.runtime.execution_capacity.begin_execution(
        origin=WorkOrigin.MANUAL,
        conversation_id="other-chat",
    )
    claim = h.runtime.worktree_activity.try_acquire(
        owner=owner,
        footprint=WorktreeFootprint(writes=(WorktreeScope(h.root, True, chain),)),
    )
    assert claim is not None
    try:
        with pytest.raises(_WorkflowChildWait) as wait:
            await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
            )
        assert wait.value.reason == "worktree_activity"
        assert not h.phase._task_scopes
        assert not h.calls
    finally:
        owner.finish_root()
        claim.settle(durable=True)
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-1",
        binding=h.binding,
        resolution=h.selected,
    )
    await prepared.close()


async def test_preparation_retires_sqlite_and_context_on_its_actual_worker(
    native_host, monkeypatch
):
    import sqlite3

    from tldw_chatbook.Agents.activation import _guard
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work

    h = native_host
    connections = []
    retired = []
    original = h.service.prepare_task_inputs
    close = h.db.close

    def observe_close():
        connection = getattr(h.db._thread_local, "conn", None)
        close()
        if any(connection is item for item in connections):
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
            retired.append(connection)

    def observe(auth):
        with h.db.connection() as conn:
            connections.append(conn)
            assert current_automatic_work() is h.phase.context
        return original(auth)

    monkeypatch.setattr(h.service, "prepare_task_inputs", observe)
    monkeypatch.setattr(h.db, "close", observe_close)
    before = _guard.context.get()
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-1",
        binding=h.binding,
        resolution=h.selected,
    )
    try:
        assert _guard.context.get() is before
        assert current_automatic_work() is None
        assert prepared.native.service._automatic_work is h.phase.context
        assert connections
        assert retired == connections
    finally:
        await prepared.close()
    assert _guard.context.get() is before
    assert current_automatic_work() is None


async def test_native_client_cleanup_keeps_original_resources_until_actual_drain(
    native_host, monkeypatch
):
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

    h = native_host
    entered, release = threading.Event(), threading.Event()
    clients = []
    stream = h.gateway.stream_chat
    cleanup = h.gateway.aclose_current_loop

    async def capture_client(*args, **kwargs):
        clients.append(h.gateway._active_http_client())
        async for item in stream(*args, **kwargs):
            yield item

    async def hold_cleanup():
        entered.set()
        assert await asyncio.to_thread(release.wait, 8)
        await cleanup()

    monkeypatch.setattr(h.gateway, "stream_chat", capture_client)
    monkeypatch.setattr(h.gateway, "aclose_current_loop", hold_cleanup)
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-1",
        binding=h.binding,
        resolution=h.selected,
    )

    def launch():
        with owned_goal_storage(h.service), h.phase.context.scope():
            return prepared.native.service.launch_workflow_task(
                prepared.authorization.task,
                authorization=prepared.authorization,
                should_cancel=lambda: False,
            )

    receipt = await asyncio.to_thread(launch)
    h.service.bind_native_receipt(prepared.authorization, receipt)
    drained = threading.Event()
    receipt.owner.on_drained(lambda positive: drained.set() if positive else None)
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert clients and not clients[0].is_closed
        assert not drained.is_set()
        with pytest.raises(PermissionError, match="not drained"):
            await prepared.close()
        assert not prepared.native.turn_lifeline.loop.is_closed()
    finally:
        release.set()
    assert await asyncio.to_thread(drained.wait, 5)
    assert clients[0].is_closed
    await h.service.call(
        h.service.checkpoint_task,
        WorkflowTaskResult(
            workflow_id=h.workflow.id,
            attempt_id=receipt.attempt_id,
            run_id=receipt.run_id,
            status="completed",
            summary="Native result",
            handoff="Required result",
        ),
        drained=True,
    )
    await prepared.close()
    assert not h.gateway._loop_clients


async def test_runtime_shutdown_drains_preparation_and_original_finalizer(
    native_host, monkeypatch
):
    from tldw_chatbook.Chat import console_workflow_preparation as preparation

    h = native_host
    entered, release = threading.Event(), threading.Event()
    captured = []
    actual = preparation._PreparedWorkflowReply._capture

    def hold(owner):
        captured.append(owner)
        entered.set()
        assert release.wait(5)
        return actual(owner)

    monkeypatch.setattr(preparation._PreparedWorkflowReply, "_capture", hold)
    task = asyncio.create_task(
        h.bridge._prepare_workflow_reply(
            phase=h.phase,
            task_id="task-1",
            binding=h.binding,
            resolution=h.selected,
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    shutdown = asyncio.create_task(h.runtime.dispose())
    try:
        await asyncio.sleep(0)
        assert not shutdown.done()
        assert not captured[0].native.turn_lifeline.loop.is_closed()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(shutdown, 5)
    assert captured[0]._closed
    assert captured[0].native.turn_lifeline.loop.is_closed()
    assert not h.runtime._preparation_reads
    assert not h.calls
