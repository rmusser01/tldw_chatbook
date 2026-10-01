"""Hook preparation and required completion through the actual agent loop."""

import asyncio
import threading
from dataclasses import replace

import pytest

from Tests.Agents.test_agent_runtime import CALC, CFG, make_deps
from tldw_chatbook.Agents.activation import worker_guard
from tldw_chatbook.Agents.agent_models import ModelTurn, ToolCall, ToolResult
from tldw_chatbook.Agents.agent_runtime import run_agent_loop

pytestmark = pytest.mark.bootstrap_profile


class PostHookCase:
    """Deterministic real-loop boundary fixture; readiness is actual model entry."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.result_published = threading.Event()
        self.pending = threading.Event()
        self.release = threading.Event()
        self.model_entered = threading.Event()
        self.gate_entered = threading.Event()
        self.checkpoint_settled = asyncio.Event()
        self.context = ""
        self.messages = []
        self.calls = 0
        self.deps = make_deps([])
        self.deps.call_model = self.model
        self.deps.on_record = self.record
        # Assigned on the existing dependency object so RED tests behavior,
        # not constructor/import scaffolding for the new optional entries.
        self.deps.install_tool_checkpoint = self.install
        self.deps.await_hook_checkpoints = self.admit
        self.future = None

    def model(self, messages, _schemas):
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(tool_calls=(ToolCall("calculator", {}, "call-1"),))
        self.messages = list(messages)
        self.model_entered.set()
        return ModelTurn(text="finished")

    def record(self, kind, _payload):
        if kind == "tool_result":
            self.result_published.set()

    def install(self, _call, _result):
        self.pending.set()

    def admit(self):
        if not self.pending.is_set():
            return ()
        self.gate_entered.set()
        assert self.release.wait(5), "required post hook was not released"
        self.loop.call_soon_threadsafe(self.checkpoint_settled.set)
        self.pending.clear()
        return ({"role": "user", "content": self.context},)

    async def dispatch_and_hold_required_post_hook(self):
        self.future = asyncio.create_task(
            asyncio.to_thread(
                run_agent_loop,
                CFG,
                [{"role": "user", "content": "start"}],
                [CALC],
                self.deps,
            )
        )
        assert await asyncio.to_thread(self.result_published.wait, 2)
        # The next model or its admission gate is now deterministically reached.
        for _ in range(200):
            if self.model_entered.is_set() or self.gate_entered.is_set():
                return
            await asyncio.sleep(0.005)
        raise AssertionError("neither next-model gate nor model was reached")

    def tool_result_visible(self):
        return self.result_published.is_set()

    def next_input_allowed(self):
        return self.model_entered.is_set()

    def release_post_hook_with_context(self, text):
        self.context = text
        self.release.set()

    def next_input_context(self):
        return self.messages[-1]["content"]


@pytest.fixture
async def post_hook_case():
    case = PostHookCase()
    yield case
    case.release.set()
    if case.future is not None:
        await case.future


@pytest.mark.asyncio
async def test_next_model_step_cannot_overtake_required_post_hook(post_hook_case):
    case = post_hook_case
    await case.dispatch_and_hold_required_post_hook()
    assert case.tool_result_visible()
    assert not case.next_input_allowed()
    case.release_post_hook_with_context("reviewed")
    await asyncio.wait_for(case.checkpoint_settled.wait(), 2)
    await case.future
    assert case.next_input_allowed()
    assert case.next_input_context() == "reviewed"


@pytest.mark.asyncio
async def test_same_loop_success_without_pending_requirement(post_hook_case):
    case = post_hook_case
    case.deps.install_tool_checkpoint = None
    await case.dispatch_and_hold_required_post_hook()
    outcome = await case.future
    assert case.tool_result_visible()
    assert case.next_input_allowed()
    assert outcome.final_text == "finished"


def test_common_preparation_precedes_canvas_exemption_review_and_dispatch():
    from dataclasses import replace

    order = []
    calls = [ToolCall("calculator", {"expression": "old"}, "call-1")]
    deps = make_deps([ModelTurn(tool_calls=tuple(calls)), ModelTurn(text="done")])

    def prepare(call):
        order.append("prepare")
        return replace(call, args={"expression": "new"})

    deps.prepare_hook_call = prepare
    deps.guard_tool_calls = lambda batch: order.append(("guard", batch[0].args)) or {}
    deps.is_tool_call_preauthorized = lambda call: (
        order.append(("exempt", call.args)) or call.args["expression"] == "old"
    )
    deps.review_tool_calls = lambda batch: order.append(("review", batch[0].args)) or {}
    deps.validate_hook_dispatch = lambda call: order.append(("fresh", call.args))
    deps.invoke_tool = lambda call: (
        order.append(("dispatch", call.args)) or ToolResult(ok=True, content="ok")
    )
    run_agent_loop(CFG, [{"role": "user", "content": "go"}], [CALC], deps)
    assert order == [
        "prepare",
        *(
            (name, {"expression": "new"})
            for name in ("guard", "exempt", "review", "fresh", "dispatch")
        ),
    ]


def test_fresh_dispatch_refusal_cannot_reuse_mutated_approval():
    deps = make_deps(
        [
            ModelTurn(tool_calls=(ToolCall("calculator", {}, "call-1"),)),
            ModelTurn(text="done"),
        ]
    )
    dispatched = []
    deps.prepare_hook_call = lambda call: call
    deps.validate_hook_dispatch = lambda call: (_ for _ in ()).throw(
        ValueError("stale")
    )
    deps.invoke_tool = lambda call: (
        dispatched.append(call) or ToolResult(ok=True, content="ok")
    )
    run_agent_loop(CFG, [{"role": "user", "content": "go"}], [CALC], deps)
    assert not dispatched


def test_catalog_snapshot_detaches_schema_and_rejects_generation_changes():
    from Tests.Agents.test_post_tool_dispatch_hook import _Provider
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    registry = ToolCatalogRegistry()
    registry.register_provider(_Provider(lambda: ToolResult(ok=True, content="ok")))
    assert registry.resolve_name("probe") == "test:probe"
    assert hasattr(registry, "snapshot_for_hook"), "missing current catalog snapshot"
    captured = registry.snapshot_for_hook("probe")
    assert captured.tool_id == "test:probe"
    assert captured.provider == "test"
    assert registry.snapshot_for_hook("probe") == captured
    registry.reset_catalog_cache()
    assert registry.snapshot_for_hook("probe") != captured


def hook_command(handler_id, *, effects=(), code="pass", event="PreToolUse", **kwargs):
    from Tests.Agents.test_hooks_v2_execution import command

    return command(code, event, effects=list(effects), **kwargs).model_copy(
        update={"id": handler_id}
    )


@pytest.mark.asyncio
async def test_real_command_transformers_run_once_then_final_guard_and_context(
    tmp_path,
):
    from Tests.Agents.test_hooks_v2_execution import event
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import prepare_tool
    from tldw_chatbook.Agents.tool_catalog import ToolDefinitionSnapshot

    calls = tmp_path / "order.txt"

    def code(label, value):
        import json

        return f"import pathlib;pathlib.Path({str(calls)!r}).open('a').write({label!r});print({json.dumps(value)!r})"

    engine = HookEngine(
        (
            hook_command(
                "C",
                effects=("deny",),
                code="import json,sys;v=json.load(sys.stdin);assert v['data']['tool_args']['expression']=='B'",
            ),
            hook_command(
                "A",
                effects=("updated_input",),
                code=code(
                    "A",
                    {
                        "version": 2,
                        "decision": "pass",
                        "updated_input": {"expression": "A"},
                    },
                ),
            ),
            hook_command(
                "B",
                effects=("updated_input",),
                code=code(
                    "B",
                    {
                        "version": 2,
                        "decision": "pass",
                        "updated_input": {"expression": "B"},
                    },
                ),
            ),
            hook_command(
                "context",
                effects=("context",),
                code=code(
                    "D",
                    {
                        "version": 2,
                        "decision": "pass",
                        "context": [{"text": "context only", "lifetime": "turn"}],
                    },
                ),
            ),
        ),
        lambda *_: True,
        HookBudgetOwner(),
    )
    definition = ToolDefinitionSnapshot.from_schema(
        replace(CALC, id="local:test"), "local", 1, 1
    )
    try:
        prepared = await prepare_tool(event(), engine, definition=definition)
        assert prepared.final_arguments == b'{"expression":"B"}'
        assert prepared.original_arguments == b"{}"
        assert calls.read_text() == "ABD"
        assert [handler for handler, _ in prepared.outcome.accepted] == [
            "A",
            "B",
            "C",
            "context",
        ]
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_service_result_publication_and_terminal_join_use_real_hook_owner(
    tmp_path,
):
    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    entered = threading.Event()
    release = threading.Event()
    terminal = threading.Event()
    tool_result = threading.Event()

    def authority(_handler, _event, stage):
        if stage == "accept":
            entered.set()
            assert release.wait(5)
        return True

    engine = HookEngine(
        (hook_command("post", event="PostToolUse", required=True),),
        authority,
        HookBudgetOwner(),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    db = AgentRunsDB(tmp_path / "runs.db", client_id="hooks-h3")
    chat = ScriptedChat([fence("calculator", {"expression": "6*7"}), "done"])
    service = AgentService(
        db,
        registry,
        chat_call=chat,
        hooks_v2_engine=engine,
        hooks_v2_session_id="session",
        hooks_v2_turn_id="turn",
        on_run_terminal=lambda *_: terminal.set(),
        on_step=lambda step, *_: (
            tool_result.set() if step.kind == "tool_result" else None
        ),
    )
    future = asyncio.create_task(
        asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=CFG,
            api_endpoint="test",
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert await asyncio.to_thread(tool_result.wait, 2)
        assert len(chat.calls) == 1
        assert not terminal.is_set()
        release.set()
        _run_id, outcome = await asyncio.wait_for(future, 5)
        assert outcome.final_text == "done"
        assert len(chat.calls) == 2
        assert terminal.is_set()
    finally:
        release.set()
        await future
        await engine.close()
        db.close()


@pytest.mark.asyncio
async def test_final_failed_tool_joins_post_requirements_before_terminal_persistence(
    tmp_path,
):
    from dataclasses import replace

    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_post_tool_dispatch_hook import _Provider
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    # Three distinct failed calls reach the actual terminal failure threshold.
    entered = threading.Event()
    release = threading.Event()
    terminal = threading.Event()
    events = []

    def authority(_handler, event, stage):
        if (
            stage == "accept"
            and event.event == "PostToolUseFailure"
            and event.data["tool_args"]["attempt"] == 2
        ):
            entered.set()
            assert release.wait(5)
        if stage == "launch":
            events.append(event.event)
        return True

    engine = HookEngine(
        tuple(
            hook_command(name, event=name, required=True)
            for name in ("PostToolUse", "PostToolUseFailure")
        ),
        authority,
        HookBudgetOwner(),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(
        _Provider(lambda: ToolResult(ok=False, error="known failure"))
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="h3-final")
    chat = ScriptedChat([fence("probe", {"attempt": i}) for i in range(3)])
    service = AgentService(
        db,
        registry,
        chat_call=chat,
        hooks_v2_engine=engine,
        hooks_v2_session_id="session",
        hooks_v2_turn_id="turn",
        on_run_terminal=lambda *_: terminal.set(),
    )
    future = asyncio.create_task(
        asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=replace(CFG, allowed_tools=("probe",)),
            api_endpoint="test",
        )
    )
    try:
        # Six real isolated child startups may exceed three seconds cold.
        assert await asyncio.to_thread(entered.wait, 10), (
            future.result() if future.done() else events
        )
        assert len(chat.calls) == 3
        assert not terminal.is_set()
        rows = db.list_runs("c")
        assert len(rows) == 1 and rows[0]["status"] not in {"done", "stuck", "error"}
        release.set()
        run_id, outcome = await asyncio.wait_for(future, 5)
        assert outcome.status == "stuck"
        assert db.get_run(run_id)["status"] == "stuck"
        assert terminal.is_set()
        assert events == ["PostToolUse", "PostToolUseFailure"] * 3
    finally:
        release.set()
        await future
        await engine.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "invalid_replacement",
        "unavailable_operation",
        "disabled_required",
        "valid_tool_id",
    ],
)
async def test_preparation_schema_requiredness_and_matcher_controls(case):
    from dataclasses import replace

    from Tests.Agents.test_hooks_v2_execution import event
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import prepare_tool
    from tldw_chatbook.Agents.tool_catalog import ToolDefinitionSnapshot

    if case == "invalid_replacement":
        handler = hook_command(
            "guard",
            effects=("updated_input",),
            code='print(\'{"version":2,"decision":"pass","updated_input":{"expression":7}}\')',
        )
    else:
        handler = hook_command(
            "guard",
            required=True,
            match=(
                {"operation": ["read"]}
                if case == "unavailable_operation"
                else {"tool_id": ["local:test"]}
            ),
        )
    schema = replace(
        CALC,
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
    )
    engine = HookEngine(
        (handler,),
        lambda *_: True,
        HookBudgetOwner(),
        enabled=case != "disabled_required",
    )
    try:
        if case == "valid_tool_id":
            assert (
                await prepare_tool(
                    event(),
                    engine,
                    definition=ToolDefinitionSnapshot.from_schema(
                        replace(schema, id="local:test"), "local", 1, 1
                    ),
                )
            ).final_arguments == b"{}"
        else:
            with pytest.raises((ValueError, RuntimeError)):
                await prepare_tool(
                    event(),
                    engine,
                    definition=ToolDefinitionSnapshot.from_schema(
                        replace(schema, id="local:test"), "local", 1, 1
                    ),
                )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_real_run_rechecks_changed_review_arguments_and_schema_before_provider(
    tmp_path,
):
    from dataclasses import replace

    from Tests.Agents.test_post_tool_dispatch_hook import _Provider
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import ToolHookRun
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    registry = ToolCatalogRegistry()
    invoked = []
    registry.register_provider(
        _Provider(lambda: invoked.append(1) or ToolResult(ok=True, content="ok"))
    )
    engine = HookEngine((), lambda *_: True, HookBudgetOwner())
    owner = ToolHookRun(
        engine,
        run_id="run",
        session_id="session",
        turn_id="turn",
        resolve_definition=lambda call: registry.snapshot_for_hook(call.name),
    )
    deps = make_deps(
        [ModelTurn(tool_calls=(ToolCall("probe", {}, "call"),)), ModelTurn(text="done")]
    )
    deps.prepare_hook_call = owner.prepare_call
    deps.validate_hook_dispatch = owner.validate_dispatch
    deps.install_tool_checkpoint = owner.install_result
    deps.await_hook_checkpoints = owner.admit_input
    deps.invoke_tool = lambda call: registry.invoke_by_name(call.name, call.args)
    deps.review_tool_calls = lambda calls: (
        calls[0].args.update(changed="after approval") or {}
    )
    try:
        await asyncio.to_thread(
            run_agent_loop,
            replace(CFG, allowed_tools=("probe",)),
            [{"role": "user", "content": "go"}],
            [registry.load_schema("test:probe")],
            deps,
        )
        assert not invoked
        assert not owner.checkpoints.drain_context("run")
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_runtime_tool_transformation_uses_common_preparation_before_branch():
    from tldw_chatbook.Agents.agent_models import FIND_TOOLS_NAME
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import ToolHookRun
    from tldw_chatbook.Agents.tool_catalog import (
        FIND_TOOLS_SCHEMA,
        ToolDefinitionSnapshot,
    )

    engine = HookEngine(
        (
            hook_command(
                "transform",
                effects=("updated_input",),
                code='print(\'{"version":2,"decision":"pass","updated_input":{"query":"final query"}}\')',
            ),
        ),
        lambda *_: True,
        HookBudgetOwner(),
    )
    snapshot = ToolDefinitionSnapshot.from_schema(FIND_TOOLS_SCHEMA, "runtime", 0, 1)
    owner = ToolHookRun(
        engine,
        run_id="run",
        session_id="session",
        turn_id="turn",
        resolve_definition=lambda _call: snapshot,
    )
    deps = make_deps(
        [
            ModelTurn(
                tool_calls=(ToolCall(FIND_TOOLS_NAME, {"query": "original"}, "call"),)
            ),
            ModelTurn(text="done"),
        ]
    )
    deps.prepare_hook_call = owner.prepare_call
    deps.validate_hook_dispatch = owner.validate_dispatch
    queries = []
    deps.find_tools = lambda query: queries.append(query) or []
    try:
        await asyncio.to_thread(
            run_agent_loop, CFG, [{"role": "user", "content": "go"}], [CALC], deps
        )
        assert queries == ["final query"]
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_authenticated_canvas_transform_keeps_covered_exemption(tmp_path):
    import json

    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_canvas_tool_provider import SCOPE, SOURCE_SENTINEL, _provider
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    provider, coordinator, authority = _provider()
    registry = ToolCatalogRegistry()
    assert registry.register_canvas_provider(provider, authority)
    final = {"title": "Final title", "html": SOURCE_SENTINEL}
    output = {"version": 2, "decision": "pass", "updated_input": final}
    engine = HookEngine(
        (
            hook_command(
                "transform",
                effects=("updated_input",),
                code=f"print({json.dumps(output)!r})",
            ),
        ),
        lambda *_: True,
        HookBudgetOwner(),
    )
    reviews = []
    db = AgentRunsDB(tmp_path / "runs.db", client_id="h3-canvas")
    chat = ScriptedChat(
        [fence("canvas_create", {"title": "Original", "html": SOURCE_SENTINEL}), "done"]
    )
    service = AgentService(
        db,
        registry,
        chat_call=chat,
        hooks_v2_engine=engine,
        hooks_v2_session_id=SCOPE.session_id,
        hooks_v2_turn_id="turn",
        review_tool_calls=lambda calls, *_: reviews.append(calls) or {},
    )
    try:
        _run_id, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id=SCOPE.conversation_id,
            requested_run_id=SCOPE.run_id,
            messages=[{"role": "user", "content": "go"}],
            config=replace(CFG, allowed_tools=("canvas_create",)),
            api_endpoint="test",
        )
        assert outcome.status == "done"
        assert reviews == []
        assert len(coordinator.calls) == 1
        assert coordinator.calls[0][3:] == ("Final title", SOURCE_SENTINEL)
    finally:
        await engine.close()
        db.close()


@pytest.mark.asyncio
async def test_settled_fleet_child_result_does_not_overtake_parent_post_gate(tmp_path):
    from Tests.Agents.test_fleet_runtime import FLEET_CFG, fence, make_fleet_service
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME, WAIT_AGENTS_TOOL_NAME
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    entered = threading.Event()
    release = threading.Event()

    def authority(_handler, _event, stage):
        if stage == "accept":
            entered.set()
            assert release.wait(5)
        return True

    engine = HookEngine(
        (
            hook_command(
                "wait-post",
                event="PostToolUse",
                required=True,
                match={"tool_id": ["runtime:wait_agents"]},
            ),
        ),
        authority,
        HookBudgetOwner(),
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="h3-fleet")
    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "child"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "combined",
        ],
        {"child": ["child result"]},
    )
    service._hooks_v2_engine = engine
    service._hooks_v2_session_id = "session"
    service._hooks_v2_turn_id = "turn"
    task = asyncio.create_task(
        asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert coordinator.all_finished()
        assert not task.done()
        assert all(
            call["messages_payload"][-1].get("content") != "combined"
            for call in chat.calls
        )
        release.set()
        _run_id, outcome = await asyncio.wait_for(task, 5)
        assert outcome.final_text == "combined"
        assert "child result" in str(chat.calls[-1]["messages_payload"])
    finally:
        release.set()
        await task
        await engine.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference",
    ["https://schema.invalid/credential-sentinel", "file:///private/schema-sentinel"],
)
async def test_public_preparation_refuses_external_schema_without_retrieval(
    monkeypatch, reference
):
    from Tests.Agents.test_hooks_v2_execution import event
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import (
        HookPreparationError,
        prepare_tool,
    )
    from tldw_chatbook.Agents.tool_catalog import ToolDefinitionSnapshot

    retrievals = []

    def refused(request, *args, **kwargs):
        retrievals.append(request.full_url)
        raise RuntimeError("test retrieval refused")

    monkeypatch.setattr("urllib.request.urlopen", refused)
    definition = ToolDefinitionSnapshot.from_schema(
        replace(CALC, id="local:test", parameters={"$ref": reference}), "local", 1, 1
    )
    engine = HookEngine((), lambda *_: True, HookBudgetOwner())
    try:
        from referencing.exceptions import Unresolvable

        with pytest.raises((HookPreparationError, Unresolvable)) as failure:
            await prepare_tool(event(), engine, definition=definition)
        assert retrievals == []
        assert str(failure.value) == "tool candidate schema validation failed"
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        {
            "$defs": {"text": {"type": "string"}},
            "type": "object",
            "properties": {"expression": {"$ref": "#/$defs/text"}},
            "required": ["expression"],
        },
    ],
)
async def test_public_preparation_validates_local_schema_constraints(schema):
    from Tests.Agents.test_hooks_v2_execution import event
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import (
        HookPreparationError,
        prepare_tool,
    )
    from tldw_chatbook.Agents.hooks_v2.validation import parse_event
    from tldw_chatbook.Agents.tool_catalog import ToolDefinitionSnapshot

    definition = ToolDefinitionSnapshot.from_schema(
        replace(CALC, id="local:test", parameters=schema), "local", 1, 1
    )
    engine = HookEngine((), lambda *_: True, HookBudgetOwner())
    try:
        payload = {
            key: value
            for key, value in event().model_dump().items()
            if value is not None
        }
        payload["data"]["tool_args"] = {"expression": "1+1"}
        prepared = await prepare_tool(
            parse_event(payload), engine, definition=definition
        )
        assert prepared.final_arguments == b'{"expression":"1+1"}'
        payload["data"]["tool_args"] = {"expression": 2}
        with pytest.raises(
            HookPreparationError, match="^tool candidate schema validation failed$"
        ):
            await prepare_tool(parse_event(payload), engine, definition=definition)
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("denied", [False, True])
async def test_real_loop_legacy_final_guard_precedes_optional_context(tmp_path, denied):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import ToolHookRun
    from tldw_chatbook.Agents.tool_catalog import ToolDefinitionSnapshot

    order = tmp_path / "order"

    def command_code(label):
        return (
            f"from pathlib import Path;Path({str(order)!r}).open('a').write({label!r})"
        )

    engine = HookEngine(
        (
            hook_command("validator", effects=("deny",), code=command_code("V")),
            hook_command("context", effects=("context",), code=command_code("C")),
        ),
        lambda *_: True,
        HookBudgetOwner(),
    )
    definition = ToolDefinitionSnapshot.from_schema(CALC, "builtin", 0, 1)
    owner = ToolHookRun(
        engine,
        run_id="run",
        session_id="session",
        turn_id="turn",
        resolve_definition=lambda _: definition,
    )
    deps = make_deps(
        [
            ModelTurn(tool_calls=(ToolCall("calculator", {}, "call"),)),
            ModelTurn(text="done"),
        ]
    )

    def guard(calls):
        assert order.read_text() == "V"
        with order.open("a") as output:
            output.write("L")
        return {calls[0].call_id: "refused"} if denied else {}

    deps.guard_tool_calls = guard
    deps.prepare_hook_call = owner.prepare_call
    deps.accept_hook_preparation = owner.validate_dispatch
    deps.validate_hook_dispatch = lambda call: owner.validate_dispatch(
        call, accept_context=False
    )
    invoked = []
    deps.invoke_tool = lambda call: (
        invoked.append(call) or ToolResult(ok=True, content="ok")
    )
    try:
        await asyncio.to_thread(
            run_agent_loop, CFG, [{"role": "user", "content": "go"}], [CALC], deps
        )
        owner.settle()
        assert order.read_text() == ("VL" if denied else "VLC")
        assert bool(invoked) is not denied
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("call_id", ["", "duplicate"])
async def test_real_loop_keeps_distinct_candidates_with_repeated_model_call_ids(
    call_id,
):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import ToolHookRun
    from tldw_chatbook.Agents.tool_catalog import ToolDefinitionSnapshot

    observed = []

    def authority(_handler, event, stage):
        if stage == "launch":
            observed.append((event.event_id, event.data["tool_args"]["expression"]))
        return True

    engine = HookEngine(
        (hook_command("post", event="PostToolUse", required=True),),
        authority,
        HookBudgetOwner(),
    )
    definition = ToolDefinitionSnapshot.from_schema(CALC, "builtin", 0, 1)
    owner = ToolHookRun(
        engine,
        run_id="run",
        session_id="session",
        turn_id="turn",
        resolve_definition=lambda _: definition,
    )
    deps = make_deps(
        [
            ModelTurn(
                tool_calls=tuple(
                    ToolCall("calculator", {"expression": value}, call_id)
                    for value in ("1+1", "2+2")
                )
            ),
            ModelTurn(text="done"),
        ]
    )
    deps.prepare_hook_call = owner.prepare_call
    deps.accept_hook_preparation = owner.validate_dispatch
    deps.validate_hook_dispatch = lambda call: owner.validate_dispatch(
        call, accept_context=False
    )
    deps.install_tool_checkpoint = owner.install_result
    deps.await_hook_checkpoints = owner.admit_input
    invoked = []
    deps.invoke_tool = lambda call: (
        invoked.append(call.args)
        or ToolResult(ok=True, content="ok", dispatch_state="settled")
    )
    try:
        await asyncio.to_thread(
            run_agent_loop, CFG, [{"role": "user", "content": "go"}], [CALC], deps
        )
        await asyncio.to_thread(owner.settle)
        assert invoked == [{"expression": "1+1"}, {"expression": "2+2"}]
        assert {value for _event_id, value in observed} == {"1+1", "2+2"}
        assert len({event_id for event_id, _value in observed}) == 2
    finally:
        await engine.close()


@pytest.mark.parametrize("change", ["none", "arguments", "catalog"])
def test_catalog_last_dispatch_recheck_after_existing_argument_repair(
    monkeypatch, change
):
    from Tests.Agents.test_post_tool_dispatch_hook import _Provider
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    invoked = []
    registry = ToolCatalogRegistry()
    registry.register_provider(
        _Provider(lambda: invoked.append(True) or ToolResult(ok=True))
    )
    definition = registry.snapshot_for_hook("probe")
    original_repair = registry._coerce_arguments

    def repair(name, tool_id, provider, args):
        result = original_repair(name, tool_id, provider, args)
        if change == "arguments":
            result["mutated"] = True
        if change == "catalog":
            registry.reset_catalog_cache()
        return result

    monkeypatch.setattr(registry, "_coerce_arguments", repair)
    result = registry.invoke_by_name("probe", {}, expected_definition=definition)
    assert bool(invoked) == (change == "none")
    if change != "none":
        assert result.dispatch_state == "not_started"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "mode"),
    [("spawn", mode) for mode in ("empty", "budget", "forbidden", "success", "error")]
    + [
        (entry, mode)
        for entry in ("send", "message", "load")
        for mode in ("refused", "success", "error")
    ],
)
async def test_runtime_predispatch_refusal_has_no_posts_but_invocation_settles(
    entry, mode
):
    from tldw_chatbook.Agents.agent_models import ToolLoadSelection
    from tldw_chatbook.Agents.fleet_message_tools import REPORT_TO_SUPERVISOR_SCHEMA
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import ToolHookRun
    from tldw_chatbook.Agents.tool_catalog import (
        LOAD_TOOLS_SCHEMA,
        SEND_TO_AGENT_SCHEMA,
        SPAWN_TOOL_SCHEMA,
        ToolDefinitionSnapshot,
    )

    schema, arguments = {
        "spawn": (SPAWN_TOOL_SCHEMA, {"task": "" if mode == "empty" else "valid task"}),
        "send": (SEND_TO_AGENT_SCHEMA, {"id": "child", "message": "hello"}),
        "message": (REPORT_TO_SUPERVISOR_SCHEMA, {"message": "report"}),
        "load": (LOAD_TOOLS_SCHEMA, {"ids": [CALC.id]}),
    }[entry]
    invoked = []
    posted = []
    results = []
    sent = []

    def invoke(*args):
        invoked.append(args)
        return ToolResult(ok=mode == "success", content="returned", error="known error")

    def authority(_handler, event, stage):
        if stage == "launch":
            posted.append(event.event)
        return True

    engine = HookEngine(
        tuple(
            hook_command(
                name,
                event=name,
                required=True,
                match={"tool_id": [schema.id]},
                effects=("context",),
                code='print(\'{"version":2,"decision":"pass","context":[{"text":"post invocation sentinel","lifetime":"turn"}]}\')',
            )
            for name in ("PostToolUse", "PostToolUseFailure")
        ),
        authority,
        HookBudgetOwner(),
    )
    definitions = {
        item.name: ToolDefinitionSnapshot.from_schema(
            item, "runtime" if item is schema else "builtin", 0, 1
        )
        for item in (schema, CALC)
    }
    owner = ToolHookRun(
        engine,
        run_id="run",
        session_id="session",
        turn_id="turn",
        resolve_definition=lambda call: definitions[call.name],
    )
    calls = [ToolCall(schema.name, arguments, "target")]
    if entry == "load" and mode == "refused":
        calls.append(ToolCall(CALC.name, {}, "other"))
    deps = make_deps(
        [ModelTurn(tool_calls=tuple(calls)), ModelTurn(text="done")], spawn=invoke
    )
    model = deps.call_model
    deps.call_model = lambda messages, schemas: (
        sent.append(list(messages)) or model(messages, schemas)
    )
    if entry == "send" and mode != "refused":
        deps.send_to_agent = invoke
    if entry == "message" and mode != "refused":
        deps.report_to_supervisor = invoke
    if entry == "load":

        def load(*args):
            invoked.append(args)
            return (
                ToolLoadSelection(accepted=(CALC,))
                if mode == "success"
                else ToolLoadSelection(details_omitted_for_budget=True)
            )

        deps.load_schemas = load
    deps.prepare_hook_call = owner.prepare_call
    deps.accept_hook_preparation = owner.validate_dispatch
    deps.validate_hook_dispatch = lambda call: owner.validate_dispatch(
        call, accept_context=False
    )

    def install(call, result):
        if call.name == schema.name:
            results.append(result)
        owner.install_result(call, result)

    deps.install_tool_checkpoint = install
    deps.await_hook_checkpoints = owner.admit_input
    config = replace(
        CFG,
        allowed_tools=() if mode == "forbidden" else CFG.allowed_tools,
        budget=replace(CFG.budget, max_subagents=0 if mode == "budget" else 1),
    )
    try:
        outcome = await asyncio.to_thread(
            run_agent_loop,
            config,
            [{"role": "user", "content": "go"}],
            [CALC, schema],
            deps,
        )
        await asyncio.to_thread(owner.settle)
        dispatched = mode in {"success", "error"}
        assert bool(invoked) is dispatched
        assert posted == (
            []
            if not dispatched
            else ["PostToolUse"] + (["PostToolUseFailure"] if mode == "error" else [])
        )
        assert ("post invocation sentinel" in str(sent[-1])) is dispatched
        assert outcome.status == "done"
        if not dispatched:
            assert results[0].dispatch_state == "not_started"
    finally:
        await engine.close()
