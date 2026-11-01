"""Workflow dispatch shares native child custody and physical lifetime."""

import time

import pytest

from Tests.Agents.test_workflow_task_scope import (
    task_host as task_host,  # noqa: PLC0414
)
from tldw_chatbook.Agents.agent_service import AgentService, _WorkflowChildWait
from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def service(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(
        db=db,
        registry=registry,
        runtime_capacity=RuntimeCapacity.from_settings(),
        fleet_coordinator=FleetCoordinator(max_live=3, clock=time.monotonic),
        chat_call=lambda **kwargs: pytest.fail("untrusted launch reached provider"),
    )
    yield service
    db.close()


def test_untrusted_host_launch_never_creates_a_child(service):
    original_rows = service.db.list_runs("conversation")
    original_capacity = service.runtime_capacity.snapshot()
    with pytest.raises(PermissionError):
        service.launch_workflow_task(
            None, authorization=None, should_cancel=lambda: False
        )
    assert service.db.list_runs("conversation") == original_rows
    assert service.runtime_capacity.snapshot() == original_capacity
    assert service._injected_fleet_coordinator.live_count() == 0


@pytest.fixture
def workflow_host(task_host):
    host = task_host(tools=("fs_read",))
    # The scope fixture proves admission independently; this launcher must acquire
    # its own original activity claim before its actual native child capacity.
    host.owner.finish_root()
    host.claim.settle(durable=True)
    return host


def _prepared_service(
    host,
    chat,
    *,
    child_scope=None,
    on_settled=None,
    snapshot=None,
    seed=None,
    writer=None,
    parent_owner=None,
):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_agent_bridge import build_console_first_request_plan

    target = host.authorization.task.target
    with host.phase.context.scope():
        plan = build_console_first_request_plan(
            shared_registry=host.registry,
            shared_allowed_tools=host.authorization.task.allowed_tools,
            context={},
            skills_present=False,
            mcp_provider=None,
            builtin_gate=None,
            local_provider=None,
            library_provider=None,
            library_authority=None,
            workspace_id=None,
            ephemeral=True,
            diff_sink=None,
            scratch_root=None,
            scratch_lease=None,
            resolution=SimpleNamespace(provider=target.provider, model=target.model),
            fallback_model=target.model,
            session_system_prompt="test",
            native_tools=False,
            turn_skill_bindings=(),
            turn_bundle_block="",
            install_skill_enabled=False,
            run_skill_script_enabled=False,
            agent_messages=seed
            if seed is not None
            else [{"role": "user", "content": host.authorization.task.instructions}],
            fleet_max_live=3,
        )
        service = AgentService(
            db=host.db,
            registry=host.registry,
            chat_call=chat,
            runtime_capacity=host.owner.capacity,
            fleet_coordinator=FleetCoordinator(max_live=3, clock=time.monotonic),
            child_model_scope=child_scope,
            on_child_settled=on_settled,
            run_log_request_plan=plan.run_log,
            run_log_writer=writer,
        )
        service._prepare_workflow_child(
            authorization=host.authorization,
            first_request_plan=plan,
            target_ref=target,
            startup_instruction_snapshot=snapshot,
            parent_owner=parent_owner,
        )
    return service, plan


def _launch(service, host, **kwargs):
    return service.launch_workflow_task(
        host.authorization.task,
        authorization=host.authorization,
        should_cancel=kwargs.get("should_cancel", lambda: False),
    )


def _join(host):
    prepared = host.authorization._native_preparation[0]
    for thread in prepared.threads.values():
        thread.join(5)
        assert not thread.is_alive()


def test_workflow_uses_original_reservation_and_real_native_tool_run(
    workflow_host, monkeypatch
):
    from Tests.Agents.test_agent_service import ScriptedChat, fence

    host = workflow_host
    chat = ScriptedChat([fence("fs_read", {"path": "source.py"}), "done"])
    service, _ = _prepared_service(host, chat)
    commits = []
    commit = host.db.automatic_work.commit

    def counted(reservation_id, **kwargs):
        commits.append(reservation_id)
        return commit(reservation_id, **kwargs)

    monkeypatch.setattr(host.db.automatic_work, "commit", counted)
    receipt = _launch(service, host)
    _join(host)
    assert commits == [host.authorization.claim.reservation_id]
    assert receipt.run_id == host.authorization.claim.launch_run_id
    assert receipt.attempt_id == host.authorization.claim.id
    assert receipt.owner.run_id == receipt.run_id
    rows = host.db.list_runs(host.authorization.claim.conversation_id)
    assert len(rows) == 1
    assert rows[0]["id"] == receipt.run_id
    assert rows[0]["status"] == "done"
    assert rows[0]["parent_run_id"] is None
    assert any(
        "source\\n" in str(call.get("messages_payload")) for call in chat.calls[1:]
    )
    assert service.runtime_capacity.snapshot().child_executions == 0
    assert host.authorization._binding_state[0][0] is receipt.owner
    assert _launch(service, host) is receipt
    assert len(host.db.list_runs(host.authorization.claim.conversation_id)) == 1


def test_concurrent_duplicate_cannot_allocate_or_clobber_running_client(
    workflow_host, monkeypatch
):
    import threading
    from contextlib import contextmanager

    from Tests.Agents.test_agent_service import ScriptedChat

    host = workflow_host
    launch_entered, release_launch = threading.Event(), threading.Event()
    model_entered, release_model = threading.Event(), threading.Event()
    counts = {"open": 0, "close": 0}
    scripted = ScriptedChat(["done"])

    @contextmanager
    def scope():
        counts["open"] += 1
        try:
            yield
        finally:
            counts["close"] += 1

    def chat(**kwargs):
        model_entered.set()
        assert release_model.wait(5)
        return scripted(**kwargs)

    service, _ = _prepared_service(host, chat, child_scope=scope)
    actual = service._launch_prepared_child

    def delayed(*args, **kwargs):
        launch_entered.set()
        assert release_launch.wait(5)
        return actual(*args, **kwargs)

    monkeypatch.setattr(service, "_launch_prepared_child", delayed)
    results = []
    launcher = threading.Thread(target=lambda: results.append(_launch(service, host)))
    launcher.start()
    try:
        assert launch_entered.wait(5)
        before = service.runtime_capacity.snapshot()
        with pytest.raises(PermissionError, match="already claimed"):
            _launch(service, host)
        assert service.runtime_capacity.snapshot() == before
        assert host.db.list_runs(host.authorization.claim.conversation_id) == []
        release_launch.set()
        launcher.join(5)
        assert not launcher.is_alive()
        receipt = results[0]
        assert model_entered.wait(5)
        before = service.runtime_capacity.snapshot()
        original = host.db.get_run(receipt.run_id)
        for _ in range(3):
            assert _launch(service, host) is receipt
        assert host.db.get_run(receipt.run_id) == original
        assert service.runtime_capacity.snapshot() == before
        assert counts == {"open": 1, "close": 0}
        assert len(host.authorization._native_preparation[0].threads) == 1
        with host.phase.context.scope():
            foreign = AgentService(
                db=host.db,
                registry=host.registry,
                runtime_capacity=service.runtime_capacity,
                chat_call=lambda **_: pytest.fail("foreign service dispatched"),
            )
        with pytest.raises(PermissionError, match="preparation required"):
            _launch(foreign, host)
        assert host.db.get_run(receipt.run_id) == original
        assert service.runtime_capacity.snapshot() == before
    finally:
        release_launch.set()
        release_model.set()
        launcher.join(5)
        _join(host)
    assert counts == {"open": 1, "close": 1}
    assert host.db.get_run(receipt.run_id)["status"] == "done"


def test_failed_native_thread_start_refunds_only_reserved_launch(
    workflow_host, monkeypatch
):
    import threading

    host = workflow_host
    service, _ = _prepared_service(host, lambda **_: pytest.fail("provider started"))
    original_start = threading.Thread.start

    def fail_child(thread):
        if thread.name.startswith("fleet-"):
            raise RuntimeError("test thread exhaustion")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_child)
    with pytest.raises(PermissionError, match="thread exhaustion"):
        _launch(service, host)
    row = host.db.get_run(host.authorization.claim.launch_run_id)
    assert row["status"] == "error"
    assert service.runtime_capacity.snapshot().child_executions == 0
    with host.db.connection() as conn:
        state = conn.execute(
            "SELECT state FROM automatic_work_reservations WHERE id=?",
            (host.authorization.claim.reservation_id,),
        ).fetchone()[0]
    assert state == "released"
    with pytest.raises(PermissionError, match="already claimed"):
        _launch(service, host)
    assert host.db.get_run(host.authorization.claim.launch_run_id) == row
    assert service._injected_fleet_coordinator.live_count() == 0


def test_child_client_setup_failure_keeps_committed_charge(workflow_host):
    from contextlib import contextmanager

    @contextmanager
    def broken_scope():
        raise RuntimeError("native client configuration refused")
        yield

    host = workflow_host
    service, _ = _prepared_service(
        host, lambda **_: pytest.fail("provider started"), child_scope=broken_scope
    )
    receipt = _launch(service, host)
    _join(host)
    assert host.db.get_run(receipt.run_id)["status"] == "error"
    assert service.runtime_capacity.snapshot().child_executions == 0
    with host.db.connection() as conn:
        state = conn.execute(
            "SELECT state FROM automatic_work_reservations WHERE id=?",
            (host.authorization.claim.reservation_id,),
        ).fetchone()[0]
    assert state == "committed"
    assert _launch(service, host) is receipt
    assert not host.authorization._binding_state
    assert host.authorization._native_preparation[0].activity_claims == [
        receipt.activity_claim
    ]
    with host.activity.prelaunch(footprint=host.authorization.footprint) as blocked:
        assert blocked is None


def test_activity_precedes_capacity_and_conflict_never_starts_worker(
    workflow_host, monkeypatch
):
    host = workflow_host
    service, _ = _prepared_service(host, lambda **_: pytest.fail("provider started"))
    with host.activity.prelaunch(footprint=host.authorization.footprint) as blocking:
        assert blocking is not None
        monkeypatch.setattr(
            service.runtime_capacity,
            "begin_execution",
            lambda **_: pytest.fail("capacity preceded activity"),
        )
        with pytest.raises(_WorkflowChildWait, match="activity busy") as wait:
            _launch(service, host)
        assert wait.value.reason == "worktree_activity"
    assert host.db.list_runs(host.authorization.claim.conversation_id) == []
    assert not host.authorization._native_preparation[0].threads


def test_original_authorization_and_frozen_preparation_cannot_be_replaced(
    workflow_host,
):
    import dataclasses

    from Tests.Agents.test_agent_service import ScriptedChat

    host = workflow_host
    service, plan = _prepared_service(host, ScriptedChat(["done"]))
    with pytest.raises(PermissionError, match="already prepared"):
        service._prepare_workflow_child(
            authorization=host.authorization,
            first_request_plan=plan,
            target_ref=host.authorization.task.target,
        )
    with pytest.raises(PermissionError, match="original prepared"):
        service.launch_workflow_task(
            host.authorization.task.model_copy(),
            authorization=host.authorization,
            should_cancel=lambda: False,
        )
    with pytest.raises(PermissionError):
        service._prepare_workflow_child(
            authorization=host.authorization,
            first_request_plan=dataclasses.replace(
                plan, config=dataclasses.replace(plan.config, model="changed")
            ),
            target_ref=host.authorization.task.target,
        )
    assert host.db.list_runs(host.authorization.claim.conversation_id) == []
    receipt = _launch(service, host)
    _join(host)
    assert host.db.get_run(receipt.run_id)["resolved_model"] == "test-model"


def test_delayed_child_uses_explicit_empty_instruction_snapshot(workflow_host):
    from Tests.Agents.test_agent_service import ScriptedChat

    host = workflow_host
    service, _ = _prepared_service(host, ScriptedChat(["done"]), snapshot=None)
    # A later turn's slot is deliberately invalid: touching it in the child's
    # native instruction-delivery path would fail rather than silently pass.
    service._startup_instruction_snapshot = object()
    receipt = _launch(service, host)
    _join(host)
    assert host.db.get_run(receipt.run_id)["status"] == "done"


def test_real_client_cleanup_holds_capacity_and_activity_after_root_finishes(
    workflow_host, monkeypatch
):
    import asyncio
    import threading
    from contextlib import contextmanager

    from Tests.Agents.test_agent_service import ScriptedChat
    from tldw_chatbook.Chat import console_agent_bridge
    from tldw_chatbook.Chat.console_agent_bridge import _ModelCallLifeline
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    host = workflow_host
    release_cleanup, cleanup_started = threading.Event(), threading.Event()
    gateway = ConsoleProviderGateway()
    clients, lifelines = [], []
    monkeypatch.setattr(console_agent_bridge, "_LOOP_THREAD_JOIN_SECONDS", 0.02)

    async def cleanup():
        cleanup_started.set()
        while not release_cleanup.is_set():
            await asyncio.sleep(0.005)
        await gateway.aclose_current_loop()

    @contextmanager
    def scope():
        lifeline = _ModelCallLifeline("workflow-client", close_current_loop=cleanup)
        lifelines.append(lifeline)
        lifeline.start()

        async def open_client():
            clients.append(gateway._active_http_client())

        asyncio.run_coroutine_threadsafe(open_client(), lifeline.loop).result(5)
        try:
            yield
        finally:
            lifeline.shutdown()

    service, _ = _prepared_service(host, ScriptedChat(["done"]), child_scope=scope)
    try:
        receipt = _launch(service, host)
        assert cleanup_started.wait(5)
        _join(host)
        assert host.db.get_run(receipt.run_id)["status"] == "done"
        snapshot = service.runtime_capacity.snapshot()
        assert snapshot.child_executions == 1
        assert snapshot.stopping_model_lifelines == 1
        assert snapshot.executions[0].root_finished
        assert not clients[0].is_closed
        assert _launch(service, host) is receipt
        with host.activity.prelaunch(footprint=host.authorization.footprint) as blocked:
            assert blocked is None
    finally:
        release_cleanup.set()
        for lifeline in lifelines:
            lifeline._thread.join(5)
            assert not lifeline._thread.is_alive()
    assert clients[0].is_closed
    assert service.runtime_capacity.snapshot().child_executions == 0
    # Required durable task-result settlement is Task7/8's owner; physical
    # completion alone must not retire that reservation either.
    with host.activity.prelaunch(footprint=host.authorization.footprint) as blocked:
        assert blocked is None


def test_host_cancel_probe_remains_live_under_survivor_setting(workflow_host):
    import threading

    from Tests.Agents.test_agent_service import ScriptedChat

    host = workflow_host
    cancelled = threading.Event()
    cancelled.set()
    service, _ = _prepared_service(host, ScriptedChat([]))
    receipt = _launch(service, host, should_cancel=cancelled.is_set)
    _join(host)
    assert host.db.get_run(receipt.run_id)["status"] == "cancelled"
    assert service.runtime_capacity.snapshot().child_executions == 0


def test_worktree_wait_preserves_original_preparation_for_later_launch(workflow_host):
    from Tests.Agents.test_agent_service import ScriptedChat

    host = workflow_host
    service, _ = _prepared_service(host, ScriptedChat(["done"]))
    with (
        host.activity.prelaunch(footprint=host.authorization.footprint),
        pytest.raises(PermissionError, match="activity busy"),
    ):
        _launch(service, host)
    receipt = _launch(service, host)
    _join(host)
    assert receipt.run_id == host.authorization.claim.launch_run_id
    assert host.db.get_run(receipt.run_id)["status"] == "done"


def test_capacity_wait_releases_unbound_activity_and_keeps_original_allowance(
    workflow_host, monkeypatch
):
    from Tests.Agents.test_agent_service import ScriptedChat
    from tldw_chatbook.Agents.execution_capacity import CapacityRefused

    host = workflow_host
    service, _ = _prepared_service(host, ScriptedChat(["done"]))
    actual = service.runtime_capacity.begin_execution

    def refuse(**kwargs):
        raise CapacityRefused("automatic_child_capacity")

    monkeypatch.setattr(service.runtime_capacity, "begin_execution", refuse)
    with pytest.raises(CapacityRefused):
        _launch(service, host)
    with host.activity.prelaunch(footprint=host.authorization.footprint) as available:
        assert available is not None
    with host.db.connection() as conn:
        assert (
            conn.execute(
                "SELECT state FROM automatic_work_reservations WHERE id=?",
                (host.authorization.claim.reservation_id,),
            ).fetchone()[0]
            == "reserved"
        )
    monkeypatch.setattr(service.runtime_capacity, "begin_execution", actual)
    receipt = _launch(service, host)
    _join(host)
    assert host.db.get_run(receipt.run_id)["status"] == "done"


def test_frozen_instruction_body_reaches_only_original_child_payload(workflow_host):
    import dataclasses
    import json

    from Tests.Agents.test_agent_service import ScriptedChat
    from Tests.Agents.test_project_instruction_runtime import _snapshot, _source

    host = workflow_host
    original = _snapshot(host.root)
    chat = ScriptedChat(["done"])
    service, _ = _prepared_service(host, chat, snapshot=original)
    service._startup_instruction_snapshot = dataclasses.replace(
        original, startup_source=_source(host.root, "AGENTS.md", "later-phase-secret")
    )
    receipt = _launch(service, host)
    _join(host)
    assert host.db.get_run(receipt.run_id)["status"] == "done"
    payload = str(chat.calls[0]["messages_payload"])
    assert "root-secret" in payload
    assert "later-phase-secret" not in payload
    assert "root-secret" not in json.dumps(host.db.get_run(receipt.run_id))
    assert "root-secret" not in repr(host.authorization._native_preparation)


def test_held_task_service_rejects_a_later_turn_without_touching_child(workflow_host):
    import threading

    from Tests.Agents.test_agent_service import ScriptedChat

    host = workflow_host
    entered, release = threading.Event(), threading.Event()
    scripted = ScriptedChat(["done"])

    def chat(**kwargs):
        entered.set()
        assert release.wait(5)
        return scripted(**kwargs)

    service, plan = _prepared_service(host, chat)
    service._on_primary_budget_settled = lambda *_: pytest.fail(
        "child settled primary budget"
    )
    try:
        receipt = _launch(service, host)
        assert entered.wait(5)
        writer = service.run_log_writer
        before = service.runtime_capacity.snapshot()
        with pytest.raises(PermissionError, match="another turn"):
            service.run_turn(
                conversation_id=host.authorization.claim.conversation_id,
                messages=[],
                config=plan.config,
                api_endpoint=plan.api_endpoint,
            )
        with pytest.raises(PermissionError, match="already prepared"):
            service._prepare_workflow_child(
                authorization=host.authorization,
                first_request_plan=plan,
                target_ref=host.authorization.task.target,
            )
        assert service.run_log_writer is writer
        assert service.runtime_capacity.snapshot() == before
        assert host.db.get_run(receipt.run_id)["status"] == "running"
    finally:
        release.set()
        _join(host)
    assert host.db.get_run(receipt.run_id)["status"] == "done"


def test_direct_native_scope_route_seals_service_and_duplicate_before_owner_cleanup(
    task_host,
):
    import threading

    from Tests.Agents.test_agent_service import ScriptedChat
    from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget

    host = task_host(tools=("fs_read",))
    entered, release = threading.Event(), threading.Event()
    script = ScriptedChat(["done"])

    def chat(**kwargs):
        entered.set()
        assert release.wait(5)
        return script(**kwargs)

    with host.phase.context.scope():
        service = AgentService(
            db=host.db,
            registry=host.registry,
            chat_call=chat,
            runtime_capacity=host.owner.capacity,
        )
    config = AgentConfig(
        model="test-model",
        system_prompt="test",
        allowed_tools=("fs_read",),
        budget=RunBudget(),
    )
    kwargs = {
        "execution_owner": host.owner,
        "automatic_child_reservation": host.authorization.claim.reservation_id,
        "workflow_task_authorization": host.authorization,
        "workflow_activity_claim": host.claim,
        "requested_run_id": host.authorization.claim.launch_run_id,
        "conversation_id": host.authorization.claim.conversation_id,
        "messages": [{"role": "user", "content": "read"}],
        "config": config,
        "api_endpoint": "openai",
        "should_cancel": lambda: False,
        "agent_kind": "subagent",
        "task": "read",
        "parent_run_id": None,
    }
    results = []
    worker = threading.Thread(
        target=lambda: results.append(service._run_owned(**kwargs))
    )
    worker.start()
    try:
        assert entered.wait(5)
        before = service.runtime_capacity.snapshot()
        with pytest.raises(PermissionError, match="already claimed"):
            service._run_owned(**kwargs)
        with pytest.raises(PermissionError, match="another turn"):
            service.run_turn(
                conversation_id=host.authorization.claim.conversation_id,
                messages=[],
                config=config,
                api_endpoint="openai",
            )
        assert service.run_log_writer is None
        assert service.runtime_capacity.snapshot() == before
        assert not before.executions[0].root_finished
    finally:
        release.set()
        worker.join(5)
        assert not worker.is_alive()
    assert results[0][1].status == "done"
    assert len(host.db.list_runs(host.authorization.claim.conversation_id)) == 1


def test_native_task_seed_detaches_nested_handoff_without_repeating_instructions(
    workflow_host,
):
    from Tests.Agents.test_agent_service import ScriptedChat

    host = workflow_host
    seed = [
        {"role": "user", "content": host.authorization.task.instructions},
        {
            "role": "user",
            "content": [{"type": "text", "text": "required-input-sentinel"}],
        },
    ]
    chat = ScriptedChat(["done"])
    service, plan = _prepared_service(host, chat, seed=seed)
    seed[1]["content"][0]["text"] = "later-mutated-input"
    plan.messages.append({"role": "user", "content": "later-phase-context"})
    with pytest.raises(PermissionError, match="already prepared"):
        service._prepare_workflow_child(
            authorization=host.authorization,
            first_request_plan=plan,
            target_ref=host.authorization.task.target,
        )
    receipt = _launch(service, host)
    _join(host)
    assert host.db.get_run(receipt.run_id)["status"] == "done"
    payload = chat.calls[0]["messages_payload"]
    assert "required-input-sentinel" in str(payload)
    assert "later-mutated-input" not in str(payload)
    assert "later-phase-context" not in str(payload)
    assert (
        sum(
            row.get("content") == host.authorization.task.instructions
            for row in payload
        )
        == 1
    )
    assert receipt.activity_claim is host.authorization._binding_state[0][1]


@pytest.mark.parametrize("failure", [None, "client", "thread"])
def test_task_owned_native_log_finalizes_once_after_settlement(
    workflow_host, monkeypatch, tmp_path, failure
):
    import json
    import threading
    from contextlib import contextmanager

    from Tests.Agents.test_agent_service import ScriptedChat
    from tldw_chatbook.Agents.run_log import MANIFEST_NAME, RunLogWriter

    host = workflow_host
    writer = RunLogWriter(root=tmp_path / "task-log")
    finalizations = []
    close = writer.close

    def final_close():
        finalizations.append(host.db.get_run(host.authorization.claim.launch_run_id))
        close()

    monkeypatch.setattr(writer, "close", final_close)

    @contextmanager
    def scope():
        if failure == "client":
            raise RuntimeError("client setup failed")
        yield

    service, _ = _prepared_service(
        host, ScriptedChat(["done"]), child_scope=scope, writer=writer
    )
    if failure == "thread":
        start = threading.Thread.start

        def refuse(thread):
            if thread.name.startswith("fleet-"):
                raise RuntimeError("thread setup failed")
            return start(thread)

        monkeypatch.setattr(threading.Thread, "start", refuse)
        with pytest.raises(PermissionError, match="thread setup failed"):
            _launch(service, host)
        with pytest.raises(PermissionError, match="already claimed"):
            _launch(service, host)
    else:
        receipt = _launch(service, host)
        _join(host)
        assert _launch(service, host) is receipt
    assert len(finalizations) == 1
    assert finalizations[0]["status"] == ("done" if failure is None else "error")
    if failure is None:
        assert writer.is_active
        manifest = json.loads((writer.log_dir / MANIFEST_NAME).read_text())
        assert manifest["run_id"] == host.authorization.claim.launch_run_id
        assert manifest["status"] == "done"
        assert manifest["record_count"] > 0
    else:
        assert not writer.is_active
        assert writer.log_dir is None


def test_task_with_genuine_parent_never_finalizes_shared_parent_writer(
    workflow_host, monkeypatch, tmp_path
):
    from Tests.Agents.test_agent_service import ScriptedChat
    from tldw_chatbook.Agents.activation import execution
    from tldw_chatbook.Agents.run_log import RunLogWriter

    host = workflow_host
    parent = host.owner.capacity.begin_execution(
        origin=host.owner.origin,
        conversation_id=host.authorization.claim.conversation_id,
        child=False,
    )
    parent.bind_run(
        host.db.create_run(
            conversation_id=host.authorization.claim.conversation_id,
            agent_kind="primary",
            work_chain_id=host.authorization.claim.chain_id,
        )
    )
    writer = RunLogWriter(root=tmp_path / "parent-log")
    try:
        service, _ = _prepared_service(
            host, ScriptedChat(["done"]), writer=writer, parent_owner=parent
        )
        with execution(service):
            writer.bind(parent.run_id)
        assert writer.is_active
        monkeypatch.setattr(
            writer, "close", lambda: pytest.fail("closed parent writer")
        )
        monkeypatch.setattr(
            writer, "write_manifest", lambda _: pytest.fail("overwrote parent manifest")
        )
        receipt = _launch(service, host)
        _join(host)
        assert host.db.get_run(receipt.run_id)["status"] == "done"
        assert writer.log_dir.name == parent.run_id
    finally:
        parent.finish_root()


def test_real_fleet_capacity_wait_preserves_original_launch_for_retry(
    workflow_host, monkeypatch
):
    from Tests.Agents.test_agent_service import ScriptedChat

    host = workflow_host
    chat = ScriptedChat(["done"])
    service, _ = _prepared_service(host, chat)
    prepared = host.authorization._native_preparation[0]
    fleet = prepared.fleet
    blockers = [fleet.reserve(task=f"occupied-{i}", agent=None) for i in range(3)]
    assert all(blockers)
    assert service.runtime_capacity.snapshot().child_executions == 0
    reservation_id = host.authorization.claim.reservation_id
    commits, owners = [], []
    commit = host.db.automatic_work.commit
    begin = service.runtime_capacity.begin_execution

    def counted_commit(reservation, **kwargs):
        commits.append(reservation)
        return commit(reservation, **kwargs)

    def counted_owner(**kwargs):
        owner = begin(**kwargs)
        owners.append(owner)
        return owner

    monkeypatch.setattr(host.db.automatic_work, "commit", counted_commit)
    monkeypatch.setattr(service.runtime_capacity, "begin_execution", counted_owner)
    with pytest.raises(_WorkflowChildWait, match="live sub-agent limit") as wait:
        _launch(service, host)
    assert wait.value.reason == "fleet_capacity"
    assert not chat.calls
    assert not commits
    assert host.db.list_runs(host.authorization.claim.conversation_id) == []
    assert service.runtime_capacity.snapshot().child_executions == 0
    assert len(owners) == 1 and owners[0].run_id is None
    assert host.authorization._native_preparation[0] is prepared
    assert not prepared.activity_claims
    assert not prepared.log_finalized
    with host.activity.prelaunch(footprint=host.authorization.footprint) as available:
        assert available is not None
    with host.db.connection() as conn:
        assert (
            conn.execute(
                "SELECT state FROM automatic_work_reservations WHERE id=?",
                (reservation_id,),
            ).fetchone()[0]
            == "reserved"
        )
    for blocker in blockers:
        fleet.finish(blocker.handle_id, "done")
    receipt = _launch(service, host)
    _join(host)
    assert _launch(service, host) is receipt
    assert receipt.run_id == host.authorization.claim.launch_run_id
    assert receipt.owner is owners[1]
    assert [owner.run_id for owner in owners] == [None, receipt.run_id]
    assert commits == [reservation_id]
    assert len(chat.calls) == 1
    assert len(host.db.list_runs(host.authorization.claim.conversation_id)) == 1
    assert host.db.get_run(receipt.run_id)["status"] == "done"
    assert service.runtime_capacity.snapshot().child_executions == 0
