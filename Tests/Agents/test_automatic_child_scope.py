"""Accepted automatic context follows actual child and tool execution owners."""

import threading
import time
from contextlib import contextmanager
from dataclasses import replace

import pytest

from Tests.Agents.conftest import join_fleet_children, pin_agent_settings
from Tests.Agents.test_agent_service import FleetChat, fence
from Tests.Agents.test_fleet_runtime import RunIdProbeProvider
from Tests.DB.test_automatic_wake_attempts import claim, survivor
from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    AgentDefinition,
    RunBudget,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_service import AgentService, _call_with_timeout
from tldw_chatbook.Agents.automatic_work_budget import (
    AutomaticWorkLimits,
    AutomaticWorkRefused,
)
from tldw_chatbook.Agents.automatic_work_runtime import (
    AutomaticWorkContext,
    current_automatic_work,
)
from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.human_input_wait import use_human_input_wait
from tldw_chatbook.Agents.run_context import current_run_id
from tldw_chatbook.Agents.local_tool_provider import (
    LocalToolProvider,
    LocalToolSpec,
    LocalToolExposure,
    LocalApprovalEffect,
)
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState


@pytest.fixture
def db(tmp_path, monkeypatch):
    pin_agent_settings(monkeypatch, max_live_subagents=3, subagents_outlive_turn=True)
    database = AgentRunsDB(tmp_path / "runs.sqlite", client_id="automatic-child")
    yield database
    database.close()


def accepted_context(db, *, chain_id=None, attempt="attempt", accepted=True, **limits):
    chain_id = chain_id or db.automatic_work.create_chain(
        "conversation",
        root_submission_id="submission",
        limits=replace(AutomaticWorkLimits(), **limits),
    )
    claim(db, chain_id, [survivor(db, chain_id)], attempt=attempt)
    context = AutomaticWorkContext(db.automatic_work, chain_id, "owner", attempt)
    if accepted:
        assert db.automatic_work.accept_wake(attempt, owner_id="owner")
        context.mark_accepted()
    return context


def make_service(db, context, replies, children=None, *, provider=None, **kwargs):
    registry = ToolCatalogRegistry()
    if provider is not None:
        registry.register_provider(provider)
    chat = FleetChat(replies, children, allow_unconsumed=True)
    if context is None:
        service = AgentService(db, registry, chat_call=chat, **kwargs)
    else:
        with context.scope():
            service = AgentService(
                db,
                registry,
                chat_call=chat,
                work_origin=WorkOrigin.AUTOMATIC,
                work_chain_id=context.chain_id,
                **kwargs,
            )
    return service, chat


def run(service, *, max_subagents=8):
    return service.run_turn(
        conversation_id="conversation",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="test",
            system_prompt="Test",
            allowed_tools=("spawn_subagent", "whoami"),
            budget=RunBudget(
                max_subagents=max_subagents,
                max_steps=60,
                max_model_turns=30,
                max_tool_call_seconds=5,
            ),
        ),
        api_endpoint="llama_cpp",
    )


def child_rows(db):
    # Synthetic source results have no task; actual launched children do.
    return [row for row in db.list_runs("conversation") if row["task"]]


class ContextProbe(RunIdProbeProvider):
    def __init__(self):
        super().__init__()
        self.contexts = []
        self.threads = []

    def invoke(self, tool_id, args):
        self.contexts.append(current_automatic_work())
        self.threads.append(threading.current_thread().name)
        return ToolResult(ok=True, content="observed")


class PausedChildToolProvider:
    def __init__(self):
        self.entered = threading.Event()
        self.human_release = threading.Event()
        self.after_human = threading.Event()
        self.release = threading.Event()
        self.sibling_entered = threading.Event()
        self.sibling_release = threading.Event()
        self.workers = []

    def list_catalog(self):
        return [
            ToolCatalogEntry("paused:tool", "paused_tool", "wait", "test"),
            ToolCatalogEntry("paused:sibling", "sibling_tool", "wait", "test"),
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name="paused_tool" if tool_id == "paused:tool" else "sibling_tool",
            description="Wait for a human decision.",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        del args
        self.workers.append(threading.current_thread())
        if tool_id == "paused:sibling":
            self.sibling_entered.set()
            assert self.sibling_release.wait(5)
            return ToolResult(ok=True, content="late sibling")
        with use_human_input_wait(current_run_id()):
            self.entered.set()
            assert self.human_release.wait(5)
        self.after_human.set()
        assert self.release.wait(5)
        return ToolResult(ok=True, content="released")


def test_captured_context_reaches_raw_child_and_tool_threads(db):
    context = accepted_context(db)
    seen = []
    tool = ContextProbe()

    def child():
        seen.append(current_automatic_work())
        return fence("whoami", {})

    service, _ = make_service(
        db,
        context,
        [fence("spawn_subagent", {"task": "child"}), fence("wait_agents", {}), "done"],
        {"child": [child, "child done"]},
        provider=tool,
    )
    assert current_automatic_work() is None
    try:
        assert run(service)[1].status == "done"
        join_fleet_children(service)
        assert seen == [context]
        assert tool.contexts == [context]
        assert tool.threads == ["tool-whoami"]
        assert child_rows(db)[0]["work_chain_id"] == context.chain_id
        assert db.automatic_work.snapshot(context.chain_id).used["child_launch"] == 1
        assert current_automatic_work() is None
    finally:
        join_fleet_children(service)


def test_prepared_context_cannot_start_service_or_tool_worker(db):
    context = accepted_context(db, accepted=False)
    service, chat = make_service(db, context, ["should not run"])
    with pytest.raises(AutomaticWorkRefused, match="acceptance_required"):
        run(service)
    assert chat.calls == []
    dispatched = []
    with context.scope():
        result = _call_with_timeout(
            lambda: (dispatched.append(True), ToolResult(ok=True))[1], 5, "side-effect"
        )
    assert not result.ok and "acceptance_required" in result.error
    assert dispatched == []


def test_three_generations_share_six_actual_launches_and_manual_is_uncharged(db):
    context = accepted_context(db)
    chain_id = context.chain_id
    for generation in range(3):
        if generation:
            context = accepted_context(
                db, chain_id=chain_id, attempt=f"attempt-{generation}"
            )
        names = [f"child-{generation}-{i}" for i in range(2)]
        replies = []
        for name in names:
            replies += [
                fence("spawn_subagent", {"task": name}),
                fence("wait_agents", {}),
            ]
        if generation == 2:
            replies += [fence("spawn_subagent", {"task": "seventh"})]
        replies += ["done"]
        scripts = {name: ["child done"] for name in names}
        if generation == 2:
            scripts["seventh"] = ["must be refused"]
        service, _ = make_service(db, context, replies, scripts)
        try:
            run(service)
            join_fleet_children(service)
        finally:
            join_fleet_children(service)
        assert db.automatic_work.complete_wake(context.attempt_id, owner_id="owner")
    snapshot = db.automatic_work.snapshot(chain_id)
    assert snapshot.used["child_launch"] == 6
    assert snapshot.reserved["child_launch"] == 0
    assert snapshot.pause_reason == "child_launch_budget"
    assert len(child_rows(db)) == 6
    manual, _ = make_service(
        db,
        None,
        [fence("spawn_subagent", {"task": "manual"}), fence("wait_agents", {}), "done"],
        {"manual": ["manual child"]},
        work_chain_id=chain_id,
    )
    try:
        assert run(manual)[1].status == "done"
        join_fleet_children(manual)
        assert len(child_rows(db)) == 7
        assert db.automatic_work.snapshot(chain_id).used["child_launch"] == 6
    finally:
        join_fleet_children(manual)


@pytest.mark.parametrize("failure_site", ["construct", "start", "review"])
def test_proven_prelaunch_failure_refunds_shared_reservation(
    db, monkeypatch, failure_site
):
    context = accepted_context(db, child_launches=1)
    failed = []
    extra = {}
    if failure_site == "review":
        pin_agent_settings(monkeypatch, max_live_subagents=1)

        @contextmanager
        def review_scope(run_id):
            if not failed:
                failed.append(True)
                raise RuntimeError("review scope unavailable")
            yield

        extra["review_state_scope"] = review_scope
    else:
        original_thread, original_start = threading.Thread, threading.Thread.start

        def construct(*args, **kwargs):
            if kwargs.get("name", "").startswith("fleet-") and not failed:
                failed.append(True)
                raise RuntimeError("thread unavailable")
            return original_thread(*args, **kwargs)

        def start(thread):
            if thread.name.startswith("fleet-") and not failed:
                failed.append(True)
                raise RuntimeError("thread unavailable")
            return original_start(thread)

        if failure_site == "construct":
            monkeypatch.setattr(threading, "Thread", construct)
        else:
            monkeypatch.setattr(threading.Thread, "start", start)
    service, _ = make_service(
        db,
        context,
        [
            fence("spawn_subagent", {"task": "failed"}),
            fence("spawn_subagent", {"task": "retry"}),
            fence("wait_agents", {}),
            "done",
        ],
        {"retry": ["child done"]},
        **extra,
    )
    try:
        run(service, max_subagents=1)
        join_fleet_children(service)
        assert failed == [True]
        rows = child_rows(db)
        assert [row["task"] for row in rows if row["status"] == "done"] == ["retry"]
        # Upstream lifecycle capture precreates the run before thread setup.
        failed_rows = [row for row in rows if row["task"] == "failed"]
        assert len(failed_rows) == (0 if failure_site == "review" else 1)
        assert all(row["status"] == "error" for row in failed_rows)
        snapshot = db.automatic_work.snapshot(context.chain_id)
        assert snapshot.used["child_launch"] == 1
        assert snapshot.reserved["child_launch"] == 0
    finally:
        join_fleet_children(service)


def test_inline_model_setup_failure_keeps_accepted_launch_charge(db, monkeypatch):
    pin_agent_settings(monkeypatch, max_live_subagents=1)
    context = accepted_context(db, child_launches=1)

    @contextmanager
    def model_scope():
        raise RuntimeError("driver partially started")
        yield  # pragma: no cover

    service, _ = make_service(
        db,
        context,
        [
            fence("spawn_subagent", {"task": "failed"}),
            fence("spawn_subagent", {"task": "retry"}),
            "done",
        ],
        inline_child_model_scope=model_scope,
    )
    run(service, max_subagents=1)
    assert child_rows(db) == []
    snapshot = db.automatic_work.snapshot(context.chain_id)
    assert snapshot.used["child_launch"] == 1
    assert snapshot.reserved["child_launch"] == 0
    assert snapshot.pause_reason == "child_launch_budget"


def test_finished_child_continuation_cannot_bypass_shared_launch_limit(db):
    context = accepted_context(db, child_launches=1)
    fleet = FleetCoordinator(max_live=3, clock=time.monotonic)

    def resume():
        handle = fleet.snapshot()[0]
        return fence("send_to_agent", {"id": handle.handle_id, "message": "continue"})

    service, _ = make_service(
        db,
        context,
        [
            fence("spawn_subagent", {"task": "child"}),
            fence("wait_agents", {}),
            resume,
            "done",
        ],
        {"child": ["child done", "should not resume"]},
        fleet_coordinator=fleet,
    )
    try:
        run(service)
        join_fleet_children(service)
        assert len(child_rows(db)) == 1
        assert child_rows(db)[0]["resumed_from_run_id"] is None
        assert db.automatic_work.snapshot(context.chain_id).used["child_launch"] == 1
        assert (
            db.automatic_work.snapshot(context.chain_id).pause_reason
            == "child_launch_budget"
        )
    finally:
        join_fleet_children(service)


def test_survivor_still_honors_its_chain_deadline_after_parent_returns(db, monkeypatch):
    wall, monotonic = [1000.0], [10.0]
    monkeypatch.setattr(db.automatic_work, "_wall_clock", lambda: wall[0])
    monkeypatch.setattr(db.automatic_work, "_monotonic_clock", lambda: monotonic[0])
    context = accepted_context(db, wall_seconds=2)
    entered, release = threading.Event(), threading.Event()
    tool = ContextProbe()

    def child():
        entered.set()
        assert release.wait(5)
        return fence("whoami", {})

    def parent_done():
        assert entered.wait(5)
        return "done"

    service, _ = make_service(
        db,
        context,
        [fence("spawn_subagent", {"task": "survivor"}), parent_done],
        {"survivor": [child, "should not continue"]},
        provider=tool,
    )
    try:
        assert run(service)[1].status == "done"
        assert child_rows(db)[0]["status"] == "running"
        wall[0], monotonic[0] = 1003.0, 13.0
        release.set()
        join_fleet_children(service)
        assert tool.contexts == []
        assert child_rows(db)[0]["status"] == "cancelled"
        assert (
            db.automatic_work.snapshot(context.chain_id).pause_reason == "wall_budget"
        )
    finally:
        release.set()
        join_fleet_children(service)


@pytest.mark.parametrize("stop_mode", ["elapsed_deadline", "explicit_cancel"])
def test_real_paused_capped_child_releases_logically_before_physical_worker(
    db, monkeypatch, stop_mode
):
    import tldw_chatbook.Agents.agent_service as service_module

    monkeypatch.setattr(service_module, "_CANCEL_POLL_SECONDS", 0.01)
    wall, monotonic = [1000.0], [10.0]
    monkeypatch.setattr(db.automatic_work, "_wall_clock", lambda: wall[0])
    monkeypatch.setattr(db.automatic_work, "_monotonic_clock", lambda: monotonic[0])
    context = accepted_context(db, wall_seconds=2)
    provider = PausedChildToolProvider()
    capacity = RuntimeCapacity()
    fleet = FleetCoordinator(max_live=1, clock=time.monotonic)
    db.create_agent_definition(
        AgentDefinition(
            name="bounded",
            description="Bounded child.",
            instructions="Call the paused tool.",
            tool_allowlist=("paused_tool",),
            max_wall_seconds=0.2,
        )
    )
    service, _chat = make_service(
        db,
        context,
        [fence("spawn_subagent", {"task": "paused", "agent": "bounded"}), "done"],
        {"paused": [fence("paused_tool", {}), "must not continue"]},
        provider=provider,
        runtime_capacity=capacity,
        fleet_coordinator=fleet,
    )
    result = []

    def run_parent():
        result.append(
            service.run_turn(
                conversation_id="conversation",
                messages=[{"role": "user", "content": "go"}],
                config=AgentConfig(
                    model="test",
                    system_prompt="Test",
                    allowed_tools=("spawn_subagent", "paused_tool"),
                    budget=RunBudget(
                        max_subagents=1,
                        max_steps=60,
                        max_model_turns=30,
                        max_tool_call_seconds=5,
                    ),
                ),
                api_endpoint="llama_cpp",
            )
        )

    parent = threading.Thread(target=run_parent)
    parent.start()
    try:
        assert provider.entered.wait(5)
        handle = fleet.snapshot()[0]
        child = child_rows(db)[0]
        assert child["budget"]["max_wall_seconds"] == 0.2
        assert handle.definition_wall_seconds == 0.2
        if stop_mode == "elapsed_deadline":
            wall[0], monotonic[0] = 1003.0, 13.0
        else:
            assert service.cancel_subagent(handle.handle_id)
        deadline = time.monotonic() + 2
        while capacity.snapshot().stopping_tool_workers != 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        while fleet.get(handle.handle_id).status == "running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        persisted = db.get_run(child["id"])
        assert fleet.get(handle.handle_id).status == "cancelled"
        assert persisted["status"] == "cancelled"
        assert capacity.snapshot().tool_workers == 1
        assert all(worker.is_alive() for worker in provider.workers)
    finally:
        provider.human_release.set()
        provider.release.set()
        provider.sibling_release.set()
        parent.join(5)
        for worker in provider.workers:
            worker.join(5)
        join_fleet_children(service)
    assert not parent.is_alive()
    assert result and result[0][1].status == "done"
    assert capacity.snapshot().executions == ()


def test_real_human_wait_pauses_only_capped_child_then_wall_boundary_stops_it(
    db, monkeypatch
):
    import tldw_chatbook.Agents.agent_service as service_module

    monkeypatch.setattr(service_module, "_CANCEL_POLL_SECONDS", 0.01)
    provider = PausedChildToolProvider()
    capacity = RuntimeCapacity(max_tool_workers=2, reserved_manual_tool_workers=0)
    fleet = FleetCoordinator(max_live=2, clock=time.monotonic)
    db.create_agent_definition(
        AgentDefinition(
            name="bounded",
            description="Bounded child.",
            instructions="Call the paused tool.",
            tool_allowlist=("paused_tool",),
            max_wall_seconds=0.15,
        )
    )
    service, _chat = make_service(
        db,
        None,
        [
            fence("spawn_subagent", {"task": "paused", "agent": "bounded"}),
            fence("spawn_subagent", {"task": "sibling"}),
            "done",
        ],
        {
            "paused": [fence("paused_tool", {}), "must not continue"],
            "sibling": [fence("sibling_tool", {}), "sibling done"],
        },
        provider=provider,
        runtime_capacity=capacity,
        fleet_coordinator=fleet,
    )
    result = []

    def run_parent():
        result.append(
            service.run_turn(
                conversation_id="conversation",
                messages=[{"role": "user", "content": "go"}],
                config=AgentConfig(
                    model="test",
                    system_prompt="Test",
                    allowed_tools=(
                        "spawn_subagent",
                        "paused_tool",
                        "sibling_tool",
                    ),
                    budget=RunBudget(
                        max_subagents=2,
                        max_steps=60,
                        max_model_turns=30,
                        max_tool_call_seconds=0.08,
                    ),
                ),
                api_endpoint="llama_cpp",
            )
        )

    parent = threading.Thread(target=run_parent)
    parent.start()
    try:
        assert provider.entered.wait(5)
        assert provider.sibling_entered.wait(5)
        time.sleep(0.12)
        assert provider.workers[0].is_alive()
        assert capacity.snapshot().stopping_tool_workers == 1
        provider.human_release.set()
        assert provider.after_human.wait(5)
        time.sleep(0.03)
        provider.release.set()
        join_fleet_children(service)
        rows = {row["task"]: row for row in child_rows(db)}
        assert rows["paused"]["budget"]["max_wall_seconds"] == 0.15
        assert rows["paused"]["status"] == "stuck"
        assert rows["sibling"]["status"] == "done"
        assert any(worker.is_alive() for worker in provider.workers)
    finally:
        provider.human_release.set()
        provider.release.set()
        provider.sibling_release.set()
        parent.join(5)
        for worker in provider.workers:
            worker.join(5)
        join_fleet_children(service)
    assert not parent.is_alive()
    assert result and result[0][1].status == "done"
    assert capacity.snapshot().executions == ()


def test_deep_search_is_refused_in_automatic_tool_thread_but_manual_is_unchanged(
    db, tmp_path
):
    context = accepted_context(db)
    calls = []
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        specs=[
            LocalToolSpec(
                name="web_deep_search",
                exposure=LocalToolExposure.CONSOLE_ONLY,
                approval_effects=(
                    LocalApprovalEffect.NETWORK,
                    LocalApprovalEffect.LLM_SPEND,
                ),
                description="deep search",
                parameters={},
                handler=lambda args: (calls.append(args), "manual result")[1],
            )
        ],
        resolve_state=lambda hub: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
    )
    with context.scope():
        result = _call_with_timeout(
            lambda: provider.invoke("web_deep_search", {}), 5, "web_deep_search"
        )
    assert not result.ok and "automatic" in result.error.lower()
    assert calls == []
    assert provider.invoke("web_deep_search", {}).content == "manual result"
    assert calls == [{}]


def test_automatic_origin_without_accepted_context_cannot_dispatch(db):
    capacity = RuntimeCapacity()
    service, chat = make_service(
        db,
        None,
        ["must not run"],
        runtime_capacity=capacity,
        work_origin=WorkOrigin.AUTOMATIC,
    )
    with pytest.raises(AutomaticWorkRefused, match="acceptance_required"):
        run(service)
    assert chat.calls == []
    assert child_rows(db) == []
    assert capacity.snapshot().executions == ()


def test_tool_wait_keeps_automatic_deadline_while_human_clock_is_paused(
    db, monkeypatch
):
    import tldw_chatbook.Agents.agent_service as service_module

    monkeypatch.setattr(service_module, "_CANCEL_POLL_SECONDS", 0.01)
    context = accepted_context(db)
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.AUTOMATIC, conversation_id="conversation"
    )
    entered, release, deadline = threading.Event(), threading.Event(), threading.Event()
    result_box = []
    workers = []

    def tool():
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(5)
        return ToolResult(ok=True, content="late result")

    def invoke():
        with context.scope():
            result_box.append(
                _call_with_timeout(
                    tool, 5, "waiting", pauses_deadline=lambda: True, owner=owner
                )
            )
        deadline.set()

    thread = threading.Thread(target=invoke)
    thread.start()
    try:
        assert entered.wait(5)
        db.automatic_work.pause(context.chain_id, "wall_budget")
        assert deadline.wait(1), "chain deadline must end a paused human wait"
        assert not result_box[0].ok and "cancelled" in result_box[0].error
        assert capacity.snapshot().tool_workers == 1
        assert capacity.snapshot().stopping_tool_workers == 1
    finally:
        release.set()
        thread.join(5)
        for worker in workers:
            worker.join(5)
        owner.finish_root()
    assert capacity.snapshot().executions == ()


def test_local_tool_rechecks_chain_after_human_approval(db, tmp_path):
    context = accepted_context(db)
    calls = []

    def approve(pending):
        db.automatic_work.pause(context.chain_id, "elapsed_budget")
        return {"write_probe": "approve_once"}

    provider = LocalToolProvider(
        workspace_root=tmp_path,
        specs=[
            LocalToolSpec(
                name="write_probe",
                exposure=LocalToolExposure.CONSOLE_ONLY,
                approval_effects=(LocalApprovalEffect.MUTATES_LOCAL,),
                description="write",
                parameters={},
                handler=lambda args: (calls.append(args), "wrote")[1],
            )
        ],
        approval_callback=approve,
    )
    with context.scope():
        result = provider.invoke("write_probe", {})
    assert not result.ok
    assert "elapsed_budget" in result.error
    assert calls == []


def test_captured_automatic_authority_keeps_automatic_capacity_origin(db):
    context = accepted_context(db)
    capacity = RuntimeCapacity()
    seen = []
    chat = FleetChat([lambda: (seen.extend(capacity.snapshot().executions), "done")[1]])
    with context.scope():
        service = AgentService(
            db, ToolCatalogRegistry(), chat_call=chat, runtime_capacity=capacity
        )
    assert run(service)[1].status == "done"
    assert [execution.origin for execution in seen] == [WorkOrigin.AUTOMATIC]
    assert db.list_runs("conversation")[0]["work_chain_id"] == context.chain_id


def test_refused_automatic_run_releases_injected_execution_owner(db):
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.AUTOMATIC, conversation_id="conversation"
    )
    service, _ = make_service(
        db,
        None,
        ["must not run"],
        runtime_capacity=capacity,
        work_origin=WorkOrigin.AUTOMATIC,
    )
    try:
        with pytest.raises(AutomaticWorkRefused, match="acceptance_required"):
            service.run_turn(
                conversation_id="conversation",
                execution_owner=owner,
                messages=[],
                config=AgentConfig(model="test", system_prompt="Test"),
                api_endpoint="llama_cpp",
            )
        assert capacity.snapshot().executions == ()
    finally:
        owner.finish_root()


@pytest.mark.parametrize("capacity_kind", ["runtime", "fleet"])
def test_occupied_child_capacity_refunds_chain_before_retry(db, capacity_kind):
    context = accepted_context(db, child_launches=1)
    capacity = RuntimeCapacity(max_child_executions=1, reserved_manual_children=0)
    fleet = FleetCoordinator(max_live=1, clock=time.monotonic)
    blocker = (
        capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id="other", child=True
        )
        if capacity_kind == "runtime"
        else fleet.reserve(task="other", agent=None)
    )

    def release():
        if capacity_kind == "runtime":
            blocker.finish_root()
        else:
            fleet.finish(blocker.handle_id, "done", result="done")

    def retry():
        assert child_rows(db) == []
        assert (
            db.automatic_work.snapshot(context.chain_id).available["child_launch"] == 1
        )
        release()
        return fence("spawn_subagent", {"task": "retry"})

    service, _ = make_service(
        db,
        context,
        [
            fence("spawn_subagent", {"task": "refused"}),
            retry,
            fence("wait_agents", {}),
            "done",
        ],
        {"retry": ["child done"]},
        runtime_capacity=capacity,
        fleet_coordinator=fleet,
    )
    try:
        assert run(service, max_subagents=1)[1].status == "done"
        join_fleet_children(service)
        assert [row["task"] for row in child_rows(db)] == ["retry"]
        assert db.automatic_work.snapshot(context.chain_id).used["child_launch"] == 1
        assert (
            db.automatic_work.snapshot(context.chain_id).reserved["child_launch"] == 0
        )
    finally:
        release()
        join_fleet_children(service)


def test_concurrent_service_launches_cannot_double_spend_chain_remainder(db):
    context = accepted_context(db, child_launches=1)
    ready = threading.Barrier(2)
    errors = []
    children = []

    def spawn():
        ready.wait(5)
        return fence("spawn_subagent", {"task": "child"})

    def child():
        children.append(threading.current_thread())
        return "child done"

    services = [
        make_service(db, context, [spawn, "done"], {"child": [child]})[0]
        for _ in range(2)
    ]

    def parent(service):
        try:
            run(service)
        except BaseException as exc:  # noqa: BLE001 -- propagate worker failures to the test
            errors.append(exc)

    parents = [threading.Thread(target=parent, args=(service,)) for service in services]
    try:
        for parent_thread in parents:
            parent_thread.start()
        for parent_thread in parents:
            parent_thread.join(5)
            assert not parent_thread.is_alive()
        for service in services:
            join_fleet_children(service)
        assert errors == []
        assert len(children) <= 1
        assert len(child_rows(db)) <= 1
        snapshot = db.automatic_work.snapshot(context.chain_id)
        assert snapshot.used["child_launch"] <= 1
        assert snapshot.reserved["child_launch"] == 0
        assert snapshot.pause_reason == "child_launch_budget"
    finally:
        for service in services:
            join_fleet_children(service)
