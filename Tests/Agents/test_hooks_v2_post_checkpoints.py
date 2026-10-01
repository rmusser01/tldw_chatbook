"""Exact owning-event and dependency checkpoint acceptance and admission."""

import asyncio
import threading

import pytest

from Tests.Agents.test_hooks_v2_execution import event
from tldw_chatbook.Agents.activation import worker_guard
from tldw_chatbook.Agents.hooks_v2.engine import HookEventOutcome, HookFailure
from tldw_chatbook.Agents.hooks_v2.models import ContextBlock, HookResult

pytestmark = pytest.mark.bootstrap_profile


def checkpoint_store(**kwargs):
    from tldw_chatbook.Agents.hooks_v2.checkpoints import HookCheckpointStore

    return HookCheckpointStore(**kwargs)


def post_event():
    original = {
        key: value for key, value in event().model_dump().items() if value is not None
    }
    original.update(event="PostToolUse", run_id="owner")
    from tldw_chatbook.Agents.hooks_v2 import parse_event

    return parse_event(original)


def success(handler="required", text="reviewed"):
    return HookEventOutcome(
        accepted=(
            (handler, HookResult(context=(ContextBlock(text=text, lifetime="turn"),))),
        )
    )


def test_context_commit_and_release_are_one_admission_operation():
    store = checkpoint_store()
    token = store.begin(post_event(), ("required",))
    with pytest.raises(RuntimeError, match="pending"):
        store.assert_next_input_allowed("owner")
    store.accept(token, success())
    store.assert_next_input_allowed("owner")
    assert (
        store.drain_context("owner")[0][1].accepted[0][1].context[0].text == "reviewed"
    )
    assert store.drain_context("owner") == ()
    with pytest.raises(RuntimeError, match="settled"):
        store.accept(token, success())


def test_explicit_failure_cannot_be_erased_by_empty_dependency_selection():
    store = checkpoint_store()
    token = store.begin(post_event(), ("required",))
    store.accept(
        token, HookEventOutcome(failures=(HookFailure("required", "disabled", True),))
    )
    with pytest.raises(RuntimeError, match="failed"):
        store.assert_next_input_allowed("owner", required_handler_ids=())
    assert store.drain_context("owner") == ()


def test_dependency_failure_fences_only_exact_dependent_use():
    store = checkpoint_store()
    token = store.begin(post_event(), (), dependency_requirements=("dependency",))
    store.assert_next_input_allowed("owner", required_handler_ids=())
    with pytest.raises(RuntimeError, match="pending"):
        store.assert_next_input_allowed("owner", required_handler_ids=("dependency",))
    store.accept(
        token,
        HookEventOutcome(
            failures=(
                HookFailure("dependency", "dependency_check_failed", False, True),
            )
        ),
    )
    store.assert_next_input_allowed("owner", required_handler_ids=())
    with pytest.raises(RuntimeError, match="failed"):
        store.assert_next_input_allowed("owner", required_handler_ids=("dependency",))
    with pytest.raises(RuntimeError, match="unknown"):
        store.assert_next_input_allowed("owner", required_handler_ids=None)


@pytest.mark.asyncio
async def test_terminal_join_waits_for_pending_dependency_but_preserves_failure_scope():
    store = checkpoint_store()
    token = store.begin(post_event(), (), dependency_requirements=("dependency",))
    entered = threading.Event()

    def settle():
        entered.set()
        store.wait("owner", terminal=True)

    task = asyncio.create_task(asyncio.to_thread(settle))
    assert await asyncio.to_thread(entered.wait, 1)
    await asyncio.sleep(0.01)
    assert not task.done()
    store.fail(token, "dependency_unavailable")
    await asyncio.wait_for(task, 1)


@pytest.mark.parametrize("cause", ["stale", "cancelled", "missing"])
def test_invalid_late_or_incomplete_result_never_releases_requirement(cause):
    current = [True]
    store = checkpoint_store(current=lambda _event: current[0])
    token = store.begin(post_event(), ("required",))
    if cause == "stale":
        current[0] = False
    elif cause == "cancelled":
        store.close_owner("owner")
    store.accept(token, HookEventOutcome() if cause == "missing" else success())
    with pytest.raises(RuntimeError):
        store.assert_next_input_allowed("owner")
    assert not store.drain_context("owner")


def test_actual_timeout_owner_distinguishes_started_uncertainty_and_settlement(
    monkeypatch,
):
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.agent_service import _call_with_timeout

    release = threading.Event()
    entered = threading.Event()
    calls = []

    def held():
        calls.append(1)
        entered.set()
        assert release.wait(30)
        return ToolResult(ok=True, content="late")

    start = threading.Thread.start

    def start_entered(worker):
        start(worker)
        if worker.name == "tool-probe":
            assert entered.wait(10)

    # The tiny timeout measures a running tool, not native worker admission.
    monkeypatch.setattr(threading.Thread, "start", start_entered)
    try:
        result = _call_with_timeout(held, 0.03, "probe")
        assert entered.is_set()
        assert getattr(result, "dispatch_state", None) == "uncertain"
        release.set()
        assert calls == [1]
        result = _call_with_timeout(
            lambda: ToolResult(ok=True, content="known"), 1, "probe"
        )
        assert result.dispatch_state == "settled"
    finally:
        release.set()


def test_service_prestart_cancellation_has_no_dispatch_provenance():
    from Tests.Agents.test_post_tool_dispatch_hook import _service
    from tldw_chatbook.Agents.agent_models import ToolCall, ToolResult

    invoked = []
    service, config = _service(lambda: invoked.append(1) or ToolResult(ok=True), None)
    invoke = service._make_invoke_tool(config, {"probe"}, lambda: True, run_id="owner")
    result = invoke(ToolCall("probe", {}, "call"))
    assert not invoked
    assert getattr(result, "dispatch_state", None) == "not_started"


def test_shared_context_carrier_counts_mixed_copy_and_multimodal_once():
    import copy
    import pickle

    from tldw_chatbook.Agents import agent_models as models

    assert hasattr(models, "HookContextOrigin"), "missing shared hook attribution"
    package = models.PluginContextOrigin("installation", "component", "revision", 3)
    hook = models.HookContextOrigin("event", "handler", 4, (package,))
    value = models.PluginContextText("text", (package,), (hook,))
    copied = pickle.loads(pickle.dumps(copy.deepcopy(value)))
    transformed = models.carry_plugin_context("prefix " + copied, copied)
    carried = models.carry_plugin_context(str(transformed), transformed)
    assert carried.checked_origins() == (package,)
    assert carried.checked_hook_origins() == (hook,)
    rows = [{"role": "user", "content": [{"type": "text", "text": carried}]}]
    checked = models.check_host_context(rows)
    assert type(checked[0]["content"][0]["text"]) is str
    assert checked[0]["content"][0]["text"] == "prefix text"
    assert (
        models.PluginContextText(
            "user hook", (), (models.HookContextOrigin("e", "h", 9),)
        ).checked_origins()
        == ()
    )


def test_shared_context_limits_whole_blocks_events_and_actual_send():
    from tldw_chatbook.Agents import agent_models as models

    assert hasattr(models, "HookContextOrigin"), "missing shared hook attribution"
    make = lambda event: models.PluginContextText(
        "x" * 4096, (), (models.HookContextOrigin(event, "h", 4096),)
    )
    rows = [{"role": "user", "content": make(str(index))} for index in range(8)]
    assert len(models.check_host_context(rows)) == 8
    with pytest.raises(ValueError, match="send_too_large"):
        models.check_host_context(rows + rows[:1])
    with pytest.raises(ValueError, match="event_too_large"):
        models.check_host_context(
            [{"role": "user", "content": make("same")} for _ in range(5)]
        )
    assert models.check_host_context(
        [{"role": "user", "content": "<untrusted-hook-context>" * 3000}]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("large", [False, True])
async def test_required_context_is_ready_before_release_and_reaches_actual_send(
    tmp_path, large
):
    import json

    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_hooks_v2_tool_pipeline import CFG, hook_command
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    body = "r" * 4000 if large else "required reviewed context"
    output = {
        "version": 2,
        "decision": "pass",
        "context": [{"text": body, "lifetime": "turn"}],
    }
    engine = HookEngine(
        (
            hook_command(
                "post",
                event="PostToolUse",
                required=True,
                require_context=True,
                effects=("context",),
                code=f"print({json.dumps(output)!r})",
            ),
        ),
        lambda *_: True,
        HookBudgetOwner(),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    db = AgentRunsDB(tmp_path / "runs.db", client_id="h3-context")
    chat = ScriptedChat([fence("calculator", {"expression": "6*7"}), "done"])
    service = AgentService(
        db,
        registry,
        chat_call=chat,
        hooks_v2_engine=engine,
        hooks_v2_session_id="session",
        hooks_v2_turn_id="turn",
    )
    try:
        _run_id, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=CFG,
            api_endpoint="test",
        )
        if large:
            assert outcome.status == "error"
            assert len(chat.calls) == 1
        else:
            assert outcome.status == "done"
            rows = chat.calls[-1]["messages_payload"]
            assert any(body in str(row.get("content", "")) for row in rows)
            assert all(type(row.get("content")) is str for row in rows)
            assert all(body not in str(row) for row in outcome.final_messages)
    finally:
        await engine.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("definitive", [False, True])
async def test_service_completion_observer_sees_both_pending_events_once(
    tmp_path, definitive
):
    from Tests.Agents.test_hooks_v2_tool_pipeline import hook_command
    from Tests.Agents.test_post_tool_dispatch_hook import _service
    from tldw_chatbook.Agents.agent_models import ToolCall, ToolResult
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import ToolHookRun

    release = threading.Event()

    def authority(_handler, _event, stage):
        if stage == "accept":
            assert release.wait(5)
        return True

    handlers = tuple(
        hook_command(name, event=name, required=True)
        for name in ("PostToolUse", "PostToolUseFailure")
    )
    engine = HookEngine(handlers, authority, HookBudgetOwner())
    observations = []

    def observe(*_args):
        with pytest.raises(RuntimeError, match="pending"):
            owner.checkpoints.assert_next_input_allowed("run")
        observations.append(
            tuple(
                item.event.event
                for item in owner.checkpoints._entries.values()
                if item.pending
            )
        )

    service, config = _service(lambda: ToolResult(ok=False, error="known"), observe)
    if definitive:
        from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy

        service.registry._providers[0].execution_policy_for = lambda _name: (
            ToolExecutionPolicy.DEFINITIVE_AFTER_START
        )
        service._on_tool_result_terminal = observe
    owner = ToolHookRun(
        engine,
        run_id="run",
        session_id="session",
        turn_id="turn",
        resolve_definition=lambda call: service.registry.snapshot_for_hook(call.name),
    )
    call = await asyncio.to_thread(owner.prepare_call, ToolCall("probe", {}, "call"))
    invoke = service._make_invoke_tool(
        config, {"probe"}, run_id="run", install_post_checkpoint=owner.install_result
    )
    try:
        result = await asyncio.to_thread(invoke, call)
        assert observations == [("PostToolUse", "PostToolUseFailure")] * (
            2 if definitive else 1
        )
        with pytest.raises(RuntimeError, match="pending"):
            owner.checkpoints.assert_next_input_allowed("run")
        owner.install_result(call, result)  # common-loop duplicate is inert
        assert len(owner.checkpoints._entries) == 2
        release.set()
        await asyncio.to_thread(owner.settle)
    finally:
        release.set()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("not_started", []),
        ("uncertain", ["PostToolUse"]),
        ("settled", ["PostToolUse", "PostToolUseFailure"]),
    ],
)
async def test_post_classification_preserves_known_failure_and_uncertainty(
    tmp_path, state, expected
):
    from Tests.Agents.test_hooks_v2_tool_pipeline import hook_command
    from Tests.Agents.test_post_tool_dispatch_hook import _service
    from tldw_chatbook.Agents.agent_models import ToolCall, ToolResult
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import ToolHookRun

    events = []

    def authority(_handler, event, stage):
        if stage == "launch":
            events.append(event.event)
            if state == "uncertain":
                assert event.data["status"] == "uncertain"
        return True

    engine = HookEngine(
        tuple(
            hook_command(name, event=name, required=True)
            for name in ("PostToolUse", "PostToolUseFailure")
        ),
        authority,
        HookBudgetOwner(),
    )
    service, _config = _service(lambda: ToolResult(ok=True), None)
    owner = ToolHookRun(
        engine,
        run_id="run",
        session_id="session",
        turn_id="turn",
        resolve_definition=lambda call: service.registry.snapshot_for_hook(call.name),
    )
    try:
        call = await asyncio.to_thread(
            owner.prepare_call, ToolCall("probe", {}, "call")
        )
        owner.install_result(
            call, ToolResult(ok=False, outcome="cancelled", dispatch_state=state)
        )
        await asyncio.to_thread(owner.settle)
        assert events == expected
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_terminal_failure_still_joins_other_pending_required_event():
    store = checkpoint_store()
    failed = store.begin(post_event(), ("explicit",))
    store.fail(failed, "failed")
    pending_event = post_event().model_copy(update={"event_id": "other-event"})
    pending = store.begin(pending_event, (), dependency_requirements=("dependency",))
    entered = threading.Event()

    def settle():
        entered.set()
        store.wait("owner", terminal=True)

    task = asyncio.create_task(asyncio.to_thread(settle))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        await asyncio.sleep(0.02)
        assert not task.done()
    finally:
        store.accept(pending, success("dependency"))
        with pytest.raises(RuntimeError, match="failed"):
            await task


@pytest.mark.asyncio
async def test_skill_completion_installs_requirements_before_service_observer(tmp_path):
    from dataclasses import replace

    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_hooks_v2_tool_pipeline import CFG, hook_command
    from Tests.Agents.test_post_tool_dispatch_hook import _Provider
    from tldw_chatbook.Agents.agent_models import ToolResult
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    engine = HookEngine(
        tuple(
            hook_command(name, event=name, required=True)
            for name in ("PostToolUse", "PostToolUseFailure")
        ),
        lambda *_: True,
        HookBudgetOwner(),
    )
    begun = []
    begin_event = engine.begin_event

    def begin(value, **kwargs):
        begun.append(value.event)
        return begin_event(value, **kwargs)

    engine.begin_event = begin
    registry = ToolCatalogRegistry()
    registry.register_provider(
        _Provider(lambda: pytest.fail("skill dispatched through provider"))
    )

    class Runner:
        def is_skill_tool(self, name):
            return name == "probe"

        def run(self, *_args):
            return ToolResult(ok=False, error="settled skill failure")

    observations = []

    def observed(*_args):
        observations.append(set(begun))

    db = AgentRunsDB(tmp_path / "runs.db", client_id="h3-skill")
    chat = ScriptedChat([fence("probe", {}), "done"])
    service = AgentService(
        db,
        registry,
        chat_call=chat,
        skill_runner=Runner(),
        post_tool_dispatch=observed,
        hooks_v2_engine=engine,
        hooks_v2_session_id="session",
        hooks_v2_turn_id="turn",
    )
    try:
        _run, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=replace(
                CFG,
                allowed_tools=("probe",),
                budget=replace(CFG.budget, max_subagents=1),
            ),
            api_endpoint="test",
        )
        assert outcome.status == "done"
        assert len(observations) == 1
        assert {"PostToolUse", "PostToolUseFailure"} <= observations[0]
    finally:
        await engine.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["budget", "success", "error", "later_spawn_budget"])
async def test_inline_skill_precheck_and_dispatched_child_budget_have_distinct_provenance(
    tmp_path, mode
):
    from dataclasses import replace

    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_hooks_v2_tool_pipeline import CFG, hook_command
    from Tests.Agents.test_skill_tool_spawn import _registry_with_code_review_skill
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME, ToolResult
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    posted = []
    invoked = []
    observed = []

    def authority(_handler, event, stage):
        if stage == "launch":
            posted.append((event.data["tool_name"], event.event))
        return True

    engine = HookEngine(
        tuple(
            hook_command(name, event=name, required=True)
            for name in ("PostToolUse", "PostToolUseFailure")
        ),
        authority,
        HookBudgetOwner(),
    )

    class Runner:
        def is_skill_tool(self, name):
            return name == "code-review"

        def run(self, _name, args, spawn):
            invoked.append(args)
            if mode == "later_spawn_budget":
                return spawn("child task", allowed_tools=("calculator",))
            return ToolResult(
                ok=mode == "success", content="returned", error="known skill error"
            )

    script = [fence("code-review", {"args": "review"}), "done"]
    if mode == "later_spawn_budget":
        script = [
            script[0],
            "child done",
            fence(SPAWN_TOOL_NAME, {"task": "second child"}),
            "done",
        ]
    chat = ScriptedChat(script)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="h3-fix1-skill")
    service = AgentService(
        db,
        _registry_with_code_review_skill(),
        chat_call=chat,
        skill_runner=Runner(),
        post_tool_dispatch=lambda call, result, *_: observed.append(
            (call.name, result.dispatch_state)
        ),
        hooks_v2_engine=engine,
        hooks_v2_session_id="session",
        hooks_v2_turn_id="turn",
    )
    try:
        _run_id, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=replace(
                CFG,
                allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
                budget=replace(CFG.budget, max_subagents=0 if mode == "budget" else 1),
            ),
            api_endpoint="test",
        )
        assert outcome.status == "done"
        assert invoked == ([] if mode == "budget" else ["review"])
        assert observed == [
            ("code-review", "not_started" if mode == "budget" else "settled")
        ]
        expected = [] if mode == "budget" else [("code-review", "PostToolUse")]
        if mode == "error":
            expected.append(("code-review", "PostToolUseFailure"))
        if mode == "later_spawn_budget":
            expected.extend(
                [
                    (SPAWN_TOOL_NAME, "PostToolUse"),
                    (SPAWN_TOOL_NAME, "PostToolUseFailure"),
                ]
            )
            assert db.count_subagent_runs("c") == 1
        assert posted == expected
    finally:
        await engine.close()
        db.close()


@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("completion", ["failure", "cleanup", "late_success"])
def test_retired_operation_keeps_pending_custody_then_only_scoped_failure_ids(
    dependent, completion
):
    import gc
    import weakref

    class OperationCapture:
        pass

    captured = OperationCapture()
    reference = weakref.ref(captured)
    store = checkpoint_store()
    store.bind_owner("session")
    store.bind_owner("operation", "session")
    store.bind_owner("next-turn", "session")
    token = store.begin(
        post_event(),
        () if dependent else ("required",),
        dependency_requirements=("required",) if dependent else (),
        owner_id="session",
        retirement_owner_id="operation",
        current=lambda _event, operation=captured: operation is not None,
    )
    del captured
    store.retire_owner("operation")
    assert token in store._entries and store._entries[token].pending
    assert reference() is not None, "pending callback custody vanished early"
    with pytest.raises(RuntimeError, match="pending"):
        store.assert_next_input_allowed("next-turn", required_handler_ids=("required",))
    if dependent:
        store.assert_next_input_allowed("next-turn", required_handler_ids=())
    if completion == "failure":
        store.fail(token, "cancelled operation")
    elif completion == "cleanup":
        store.accept(
            token, HookEventOutcome(outstanding_cleanup=("original-process-owner",))
        )
    else:
        store.accept(token, success())
    gc.collect()
    assert not store._entries
    assert reference() is None
    assert store._failures == {
        "session": (
            frozenset() if dependent else frozenset({"required"}),
            frozenset({"required"}) if dependent else frozenset(),
        )
    }
    with pytest.raises(RuntimeError, match="failed"):
        store.assert_next_input_allowed("next-turn", required_handler_ids=("required",))
    if dependent:
        store.assert_next_input_allowed("next-turn", required_handler_ids=())
    else:
        with pytest.raises(RuntimeError, match="failed"):
            store.assert_next_input_allowed("next-turn", required_handler_ids=())
    store.retire_owner("next-turn")
    store.retire_owner("session")
    assert not store._failures


@pytest.mark.parametrize("dependent", [False, True])
def test_settled_operation_failure_compacts_without_removing_the_live_gate(dependent):
    store = checkpoint_store()
    store.bind_owner("session")
    store.bind_owner("operation", "session")
    store.bind_owner("next-turn", "session")
    token = store.begin(
        post_event(),
        () if dependent else ("required",),
        dependency_requirements=("required",) if dependent else (),
        owner_id="session",
        retirement_owner_id="operation",
    )
    store.fail(token, "required operation failed")
    assert token in store._entries
    store.retire_owner("operation")
    assert not store._entries
    with pytest.raises(RuntimeError, match="failed"):
        store.assert_next_input_allowed("next-turn", required_handler_ids=("required",))
    if dependent:
        store.assert_next_input_allowed("next-turn", required_handler_ids=())
