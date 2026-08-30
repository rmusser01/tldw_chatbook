# Tests/Agents/test_agent_service.py
"""Service tests: scripted chat_call (no network) + real AgentRunsDB."""

import asyncio
import dataclasses
import hashlib
import json
import threading
import time

import pytest

from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import (
    FIND_TOOLS_NAME,
    LOAD_TOOLS_NAME,
    RUN_DONE,
    RUN_ERROR,
    RUN_STUCK,
    SPAWN_TOOL_NAME,
    WAIT_AGENTS_TOOL_NAME,
    AgentConfig,
    AgentDefinition,
    ContinuationEventContext,
    RunBudget,
    ToolCatalogEntry,
    ToolBatchReady,
    ToolLoadSelection,
    ToolResult,
    ToolSchema,
    definition_fingerprint,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRestoreTarget,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
from tldw_chatbook.Agents.project_instruction_runtime import (
    PROJECT_INSTRUCTION_ROW_KEY,
    InstructionActivationLedger,
    InstructionDeliveryReceipt,
    InstructionPreparation,
)
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionChainDelivery,
    InstructionOutcome,
    InstructionSnapshot,
    InstructionSource,
    StartupInstructionCandidate,
)
from tldw_chatbook.Agents.agent_service import (
    SUBAGENT_SYSTEM_PROMPT,
    AgentService,
    FirstRequestSchemaPlan,
    _call_with_timeout,
    _usage_total_tokens,
    build_first_request_schema_plan,
    catalog_schema_tokens,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    FIND_TOOLS_SCHEMA,
    LOAD_TOOLS_SCHEMA,
    ToolCatalogRegistry,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

from Tests.Agents.conftest import join_fleet_children


def fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def provider_reply(item):
    """str -> plain content reply; dict -> used as the full message."""
    if isinstance(item, dict):
        return {"choices": [{"message": item}]}
    return {"choices": [{"message": {"content": item}}]}


def native_call(name, args, call_id="c1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class ScriptedChat:
    """Returns scripted replies; records every call's kwargs.

    ONE ordered queue shared by every agent in the run tree, which is
    deterministic only while children run INLINE (see `FleetChat` below
    for the addressed variant the fleet needs).
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return provider_reply(self.replies.pop(0))


#: The child's system prompt is the sub-agent prompt (plus, for a named
#: agent, its appended instructions) followed by the rendered fence
#: protocol -- so a prefix match on its first sentence is what identifies a
#: child's provider call, the same identity contract
#: `console_agent_bridge._is_subagent` relies on.
SUBAGENT_PROMPT_PREFIX = SUBAGENT_SYSTEM_PROMPT.split(".")[0]


def child_task_of(payload):
    """The task text of the child whose provider call this payload is.

    Args:
        payload: The ``messages_payload`` handed to ``chat_api_call``.

    Returns:
        The child's task text (its first user message), or ``None`` when
        this payload belongs to the primary agent.
    """
    if not payload:
        return None
    system = payload[0]
    if system.get("role") != "system":
        return None
    if not str(system.get("content", "")).startswith(SUBAGENT_PROMPT_PREFIX):
        return None
    for message in payload[1:]:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return None


def verbatim(item):
    """``reply`` hook for scripts already holding full provider responses.

    ``provider_reply`` wraps a str/message-dict into the ``{"choices":
    [{"message": ...}]}`` envelope; several suites script that envelope
    directly, and hand this in so ``FleetChat`` passes their entries
    through untouched.
    """
    return item


class FleetChat:
    """Addressed scripted provider: one script per agent, not one queue.

    PR2a Task 6.5. `ScriptedChat` pops one shared ordered list, which stops
    being deterministic the moment children run on their own threads
    (whichever thread wins the race takes the next reply) -- and the fleet
    is ON by default from Task 6.5 on. This keeps the same shape but
    ADDRESSES replies: an ordered script for the primary agent, and a
    separate ordered script per child TASK TEXT. Every reply therefore
    goes to the agent it was written for, no matter who calls first.

    Replies may be plain strings/dicts (as ``ScriptedChat``) or zero-arg
    callables, which are invoked at call time -- that is how a test gates a
    child on an ``Event`` or a ``Barrier`` while the parent keeps running.

    ``calls`` records every call in arrival order (so it is NOT a stable
    index under concurrency -- address ``parent_calls``/``child_calls``
    instead); ``parent_calls`` records the primary agent's own calls and
    ``child_calls`` each child's, each in that agent's own order, which IS
    stable because one agent's turns are strictly sequential.

    FAILING LOUDLY (PR2a Task 6.5 review). A scripting mistake here used to
    vanish: a bare ``assert`` raised on a CHILD's thread is swallowed by
    ``AgentService``'s ``run_child`` (which catches BaseException by design,
    so a buggy child cannot strand the parent's join) and becomes
    ``status=error``, so any test not asserting the child's RESULT still
    passed -- vacuously. Demonstrated: mis-keying ``test_child_cannot_spawn``'s
    child script to a task text no child ever asks for left it green. Since
    this harness is now load-bearing for nine suites, and a future task-text
    rename is exactly the kind of change that would re-introduce it, every
    scripting fault is recorded in ``harness_errors``, re-raised on the
    parent's next call, and swept at teardown by the autouse
    ``_fleet_chat_scripts_fully_consumed`` fixture in ``Tests/conftest.py``
    -- which also fails a test that left any scripted turn UNUSED.
    """

    #: Every instance built during the current test, for the autouse sweep.
    _live_instances: list["FleetChat"] = []

    def __init__(
        self,
        parent_replies,
        child_replies=None,
        *,
        reply=provider_reply,
        allow_unconsumed=False,
    ):
        """
        Args:
            parent_replies: ordered script for the primary agent.
            child_replies: {task_text: ordered script} per child.
            reply: item -> provider response. `verbatim` for suites whose
                scripts already hold full response envelopes.
            allow_unconsumed: opt OUT of the "every scripted turn was
                used" teardown check -- and ONLY that check. Set it when a
                test deliberately strands turns: a child cancelled
                mid-flight, one wedged past the wall clock, one that
                raises instead of answering.

                KNOWN RESIDUAL, measured: for such a test a mis-keyed
                child script can still pass silently, because the child
                may be cancelled (or may explode) before it ever asks for
                a reply -- so neither signal exists to fire. There is no
                signal to recover here; the mitigation is that this flag
                is rare and deliberate (three tests in the fleet suite,
                each commented). A child that DOES get to ask still
                records a `harness_errors` entry, which is fatal
                regardless of this flag.
        """
        self.parent_replies = list(parent_replies)
        self.child_replies = {
            task: list(script) for task, script in (child_replies or {}).items()
        }
        self.calls: list[dict] = []
        self.parent_calls: list[dict] = []
        self.child_calls: dict[str, list[dict]] = {}
        self.harness_errors: list[str] = []
        self.allow_unconsumed = allow_unconsumed
        self._reply = reply
        self._lock = threading.Lock()
        FleetChat._live_instances.append(self)

    def _fault(self, message):
        """Record a scripting fault and raise it on the calling thread."""
        self.harness_errors.append(message)
        raise AssertionError(f"FleetChat scripting fault: {message}")

    def __call__(self, **kwargs):
        payload = kwargs["messages_payload"]
        task = child_task_of(payload)
        with self._lock:
            self.calls.append(kwargs)
            # Surface a fault raised earlier on a CHILD's thread, where the
            # runtime swallowed it, the first time the parent calls again.
            if self.harness_errors:
                raise AssertionError(
                    "FleetChat scripting fault on another agent's thread: "
                    + "; ".join(self.harness_errors)
                )
            if task is None:
                self.parent_calls.append(kwargs)
                if not self.parent_replies:
                    self._fault("parent script exhausted")
                item = self.parent_replies.pop(0)
            else:
                self.child_calls.setdefault(task, []).append(kwargs)
                script = self.child_replies.get(task)
                if not script:
                    self._fault(
                        f"no scripted reply left for child task {task!r}; "
                        f"scripted tasks are {sorted(self.child_replies)}"
                    )
                item = script.pop(0)
        # Called OUTSIDE the lock: a gated reply blocks here, and holding
        # the lock would serialize the very concurrency under test.
        if callable(item):
            item = item()
        return self._reply(item)

    def unconsumed(self):
        """Scripted turns nobody ever asked for, as readable strings."""
        if self.allow_unconsumed:
            return []
        leftovers = []
        if self.parent_replies:
            leftovers.append(f"parent has {len(self.parent_replies)} unused turn(s)")
        for task, script in self.child_replies.items():
            if script:
                leftovers.append(
                    f"child {task!r} has {len(script)} unused turn(s) "
                    f"(asked for {len(self.child_calls.get(task, []))})"
                )
        return leftovers


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


def make_service(db, replies):
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = ScriptedChat(replies)
    return AgentService(db=db, registry=registry, chat_call=chat), chat


def _service_with(db, chat):
    """`make_service` for a chat double built by the caller (a `FleetChat`)."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    return AgentService(db=db, registry=registry, chat_call=chat)


class _ProjectInstructionContextSpy:
    """Duck-typed Task 9 ledger spy with content-free call evidence."""

    def __init__(self, *, initial_rows: bool = True) -> None:
        self.initial_rows = initial_rows
        self.initial_calls: list[str] = []
        self.prepare_calls: list[tuple[str, tuple[str, ...]]] = []
        self.mark_calls: list[tuple[str, tuple[dict, ...]]] = []

    @staticmethod
    def _delivery(chain_id: str) -> InstructionPreparation:
        safe_chain = chain_id.replace(":", "-")
        row_key = f"{safe_chain}-row"
        receipt = InstructionDeliveryReceipt(
            receipt_id=f"receipt-{safe_chain}",
            chain_id=chain_id,
            through_revision=1,
            source_digests=(),
            outcome_keys=(),
            row_keys=(row_key,),
        )
        row = {
            "role": "user",
            "content": f"instructions for {chain_id}",
            EPHEMERAL_ORIGIN_KEY: "project_instructions",
            PROJECT_INSTRUCTION_ROW_KEY: row_key,
        }
        return InstructionPreparation("retry_with_context", (row,), receipt)

    def initial_context_for_chain(self, chain_id, payload_state):
        self.initial_calls.append(chain_id)
        if not self.initial_rows:
            return InstructionPreparation("proceed")
        return self._delivery(chain_id)

    def prepare(self, calls, chain_id, registry, payload_state):
        self.prepare_calls.append((chain_id, tuple(call.name for call in calls)))
        return InstructionPreparation("proceed")

    def mark_payload_sent(self, receipt, payload_rows):
        self.mark_calls.append(
            (receipt.chain_id, tuple(dict(row) for row in payload_rows))
        )


def _warning_ledger(tmp_path) -> InstructionActivationLedger:
    snapshot = InstructionSnapshot(
        binding_id="binding",
        binding_root=tmp_path,
        locator_fingerprint="fingerprint",
        dispatch_started_wall_ns=time.time_ns(),
        startup_source=None,
        global_outcomes=(InstructionOutcome("AGENTS.md", ".", "resolution_failed"),),
        primary_delivery=InstructionChainDelivery((), ()),
        warning_codes=("resolution_failed",),
    )
    return InstructionActivationLedger(snapshot, nested_max_bytes=0)


CFG = AgentConfig(
    model="test-model",
    system_prompt="You are helpful.",
    allowed_tools=("calculator", "get_current_datetime", SPAWN_TOOL_NAME),
)


def _schema(name: str, description: str = "compact") -> ToolSchema:
    return ToolSchema(
        id=f"fake:{name}",
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}},
    )


def test_catalog_schema_tokens_measures_one_complete_native_schema_set(monkeypatch):
    schemas = (_schema("alpha"), _schema("beta"))
    measured: list[str] = []

    def estimate(text, *_args, **_kwargs):
        measured.append(text)
        return 17

    monkeypatch.setattr(agent_service, "estimate_tokens", estimate)
    monkeypatch.setattr(
        agent_service, "provider_supports_native_tools", lambda _endpoint: True
    )

    assert (
        catalog_schema_tokens(
            schemas,
            model="m",
            api_endpoint="openai",
            native_tools=True,
        )
        == 17
    )
    assert measured == [
        json.dumps(
            agent_service.schemas_to_openai_tools(list(schemas)),
            sort_keys=True,
            separators=(",", ":"),
        )
    ]


def test_catalog_schema_tokens_measures_one_complete_fence_protocol(monkeypatch):
    schemas = (_schema("alpha"), _schema("beta"))
    measured: list[str] = []

    def estimate(text, *_args, **_kwargs):
        measured.append(text)
        return 19

    monkeypatch.setattr(agent_service, "estimate_tokens", estimate)

    assert (
        catalog_schema_tokens(
            schemas,
            model="m",
            api_endpoint="llama_cpp",
            native_tools=False,
        )
        == 19
    )
    assert measured == [agent_service.render_tool_protocol(list(schemas))]


class PlanningProvider:
    def __init__(self, count: int):
        self.schemas = tuple(_schema(f"tool_{index}") for index in range(count))

    def list_catalog(self):
        return [
            ToolCatalogEntry(schema.id, schema.name, schema.description, "fake")
            for schema in self.schemas
        ]

    def load_schema(self, tool_id):
        return next(schema for schema in self.schemas if schema.id == tool_id)

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content=tool_id)


def _planning_registry(count: int) -> tuple[ToolCatalogRegistry, tuple[str, ...]]:
    provider = PlanningProvider(count)
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    return registry, tuple(schema.name for schema in provider.schemas)


def test_first_request_plan_direct_discloses_many_compact_schemas(monkeypatch):
    registry, allowed = _planning_registry(25)
    config = AgentConfig(
        model="m",
        system_prompt="direct",
        allowed_tools=allowed,
        budget=RunBudget(max_subagents=0),
        response_reserve_tokens=100,
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a: 10_000)
    monkeypatch.setattr(agent_service, "catalog_schema_tokens", lambda *_a, **_k: 999)
    monkeypatch.setattr(agent_service, "_count_model_messages", lambda *_a, **_k: 500)

    plan = build_first_request_schema_plan(
        registry,
        allowed,
        config,
        "llama_cpp",
        [{"role": "user", "content": "go"}],
        skill_file_enabled=False,
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        run_log_active=False,
        direct_system_prompt="direct",
        discovery_system_prompt="discovery",
    )

    assert len(plan.active_schemas) == 25
    assert plan.offer_find_load is False
    assert plan.system_prompt == "direct"


def test_first_request_plan_contains_exact_named_agent_and_fleet_schemas(monkeypatch):
    registry, allowed = _planning_registry(1)
    config = AgentConfig(
        model="m",
        system_prompt="direct",
        allowed_tools=allowed,
        budget=RunBudget(max_subagents=2),
        response_reserve_tokens=100,
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a: 10_000)
    monkeypatch.setattr(agent_service, "catalog_schema_tokens", lambda *_a, **_k: 50)
    monkeypatch.setattr(agent_service, "_count_model_messages", lambda *_a, **_k: 500)

    plan = build_first_request_schema_plan(
        registry,
        allowed,
        config,
        "llama_cpp",
        [{"role": "user", "content": "go"}],
        skill_file_enabled=False,
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        run_log_active=False,
        agent_definitions=(
            AgentDefinition(
                name="researcher",
                description="Find evidence",
                instructions="Research the task.",
            ),
        ),
        fleet_active=True,
    )

    assert [schema.name for schema in plan.runtime_schemas] == [
        "spawn_subagent",
        "wait_agents",
        "check_agents",
        "send_to_agent",
    ]
    assert (
        "researcher"
        in plan.runtime_schemas[0].parameters["properties"]["agent"]["description"]
    )


def test_first_request_plan_defers_large_or_history_cramped_catalog(monkeypatch):
    registry, allowed = _planning_registry(5)
    config = AgentConfig(
        model="m",
        system_prompt="direct",
        allowed_tools=allowed,
        budget=RunBudget(max_subagents=0),
        response_reserve_tokens=100,
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a: 10_000)
    monkeypatch.setattr(agent_service, "catalog_schema_tokens", lambda *_a, **_k: 900)
    def count(messages, *_args, **_kwargs):
        system = str(messages[0].get("content", ""))
        return 9_000 if FIND_TOOLS_NAME in system else 9_901

    monkeypatch.setattr(agent_service, "_count_model_messages", count)

    plan = build_first_request_schema_plan(
        registry,
        allowed,
        config,
        "llama_cpp",
        [{"role": "user", "content": "history pressure"}],
        skill_file_enabled=False,
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        run_log_active=False,
        direct_system_prompt="direct",
        discovery_system_prompt="discovery",
    )

    assert plan.active_schemas == ()
    assert plan.offer_find_load is True
    assert plan.system_prompt == "discovery"


def test_first_request_plan_drops_discovery_tools_when_only_no_tool_request_fits(
    monkeypatch,
):
    registry, allowed = _planning_registry(1)
    config = AgentConfig(
        model="m",
        system_prompt="direct",
        allowed_tools=allowed,
        budget=RunBudget(max_subagents=0),
        response_reserve_tokens=10,
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a: 100)
    monkeypatch.setattr(agent_service, "catalog_schema_tokens", lambda *_a, **_k: 5)

    def count(messages, *_args, **_kwargs):
        rendered = str(messages[0].get("content", ""))
        return 91 if "tool_0" in rendered or FIND_TOOLS_NAME in rendered else 80

    monkeypatch.setattr(agent_service, "_count_model_messages", count)

    plan = build_first_request_schema_plan(
        registry,
        allowed,
        config,
        "llama_cpp",
        [{"role": "user", "content": "go"}],
        skill_file_enabled=False,
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        run_log_active=False,
        direct_system_prompt="direct",
        discovery_system_prompt="discovery",
    )

    assert plan.request_fits is True
    assert plan.active_schemas == ()
    assert plan.runtime_schemas == ()
    assert plan.offer_find_load is False
    assert plan.system_prompt == "direct"


def test_first_request_plan_stops_before_provider_when_even_no_tool_request_fails(
    db, monkeypatch
):
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a: 100)
    monkeypatch.setattr(agent_service, "catalog_schema_tokens", lambda *_a, **_k: 5)
    monkeypatch.setattr(agent_service, "_count_model_messages", lambda *_a, **_k: 91)
    service, chat = make_service(db, ["must not send"])
    config = dataclasses.replace(
        CFG,
        allowed_tools=("calculator",),
        response_reserve_tokens=10,
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c-unfit-first-request",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_ERROR
    assert chat.calls == []
    assert outcome.steps[0].summary == "first request exceeds model context budget"


def test_first_request_plan_counts_workspace_context_note(monkeypatch):
    """The always-sent workspace note participates in whole-request fit."""
    registry, allowed = _planning_registry(1)
    config = AgentConfig(
        model="m",
        system_prompt="direct",
        allowed_tools=allowed,
        budget=RunBudget(max_subagents=0),
        workspace_context_note="BOUND WORKSPACE AUTHORITY",
        response_reserve_tokens=100,
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a: 10_000)
    monkeypatch.setattr(agent_service, "catalog_schema_tokens", lambda *_a, **_k: 900)

    def count(messages, *_args, **_kwargs):
        system = str(messages[0].get("content", ""))
        if FIND_TOOLS_NAME in system:
            return 9_000
        return 9_901 if "BOUND WORKSPACE AUTHORITY" in system else 500

    monkeypatch.setattr(agent_service, "_count_model_messages", count)

    plan = build_first_request_schema_plan(
        registry,
        allowed,
        config,
        "llama_cpp",
        [{"role": "user", "content": "go"}],
        skill_file_enabled=False,
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        run_log_active=False,
        direct_system_prompt="direct",
        discovery_system_prompt="discovery",
    )

    assert plan.active_schemas == ()
    assert plan.offer_find_load is True


@pytest.mark.parametrize("failure", [0, RuntimeError("unknown")])
def test_first_request_plan_invalid_model_limit_fails_into_discovery(
    monkeypatch, failure
):
    registry, allowed = _planning_registry(2)

    def model_limit(*_args):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(agent_service, "get_model_token_limit", model_limit)
    config = AgentConfig(
        model="m", system_prompt="direct", allowed_tools=allowed,
        budget=RunBudget(max_subagents=0),
    )

    plan = build_first_request_schema_plan(
        registry,
        allowed,
        config,
        "llama_cpp",
        [{"role": "user", "content": "go"}],
        skill_file_enabled=False,
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        run_log_active=False,
        direct_system_prompt="direct",
        discovery_system_prompt="discovery",
    )

    assert plan.offer_find_load is True
    assert plan.active_schemas == ()


def test_initial_project_instruction_rows_are_verified_and_marked_before_provider(
    db, monkeypatch
):
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_args: 100_000)
    context = _ProjectInstructionContextSpy()
    events: list[str] = []

    def chat_call(**kwargs):
        assert context.mark_calls
        events.append("provider")
        payload = kwargs["messages_payload"]
        assert [
            row[PROJECT_INSTRUCTION_ROW_KEY]
            for row in payload
            if PROJECT_INSTRUCTION_ROW_KEY in row
        ] == ["primary-row"]
        return provider_reply("done")

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat_call,
        project_instruction_context=context,
    )
    original = [{"role": "user", "content": "go"}]
    _run_id, outcome = service.run_turn(
        conversation_id="c-project-initial",
        messages=original,
        config=CFG,
        api_endpoint="openai",
    )

    assert outcome.status == RUN_DONE
    assert context.initial_calls == ["primary"]
    assert [chain for chain, _payload in context.mark_calls] == ["primary"]
    assert events == ["provider"]
    assert original == [{"role": "user", "content": "go"}]


def test_bounding_that_drops_pending_instruction_row_stops_without_send_or_mark(
    db, monkeypatch
):
    context = _ProjectInstructionContextSpy()
    service, chat = make_service(db, ["must not send"])
    service.project_instruction_context = context

    def drop_project_rows(messages, **_kwargs):
        return [row for row in messages if PROJECT_INSTRUCTION_ROW_KEY not in row]

    monkeypatch.setattr(agent_service, "bound_history_for_send", drop_project_rows)
    _run_id, outcome = service.run_turn(
        conversation_id="c-project-unfit",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="openai",
    )

    assert outcome.status == RUN_ERROR
    assert chat.calls == []
    assert context.mark_calls == []
    assert outcome.steps[0].summary == "project instruction context could not fit"


def test_real_ledger_one_over_final_limit_stops_without_mark_or_send(
    db, monkeypatch, tmp_path
):
    ledger = _warning_ledger(tmp_path)
    service, chat = make_service(db, ["must not send"])
    service.project_instruction_context = ledger

    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_args: 100)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_args, **_kwargs: 5)

    def count_one_over(messages, *_args, **_kwargs):
        if any(row.get("role") == "system" for row in messages):
            return (
                86
                if any(PROJECT_INSTRUCTION_ROW_KEY in row for row in messages)
                else 85
            )
        return 1

    monkeypatch.setattr(agent_service, "count_tokens_messages", count_one_over)
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(),
        response_reserve_tokens=10,
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c-project-zero-headroom",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="openai",
    )

    assert outcome.status == RUN_ERROR
    assert outcome.steps[0].summary == "project instruction context could not fit"
    assert chat.calls == []
    pending = ledger.initial_context_for_chain(
        "primary",
        agent_service.InstructionChainPayloadState(
            request_builder=lambda messages, schemas: (messages, schemas),
            safe_token_allowance=lambda request, rows: 100,
            count_tokens=lambda rows: 1,
        ),
    )
    assert pending.status == "retry_with_context"


def test_real_ledger_exact_final_limit_marks_and_sends(db, monkeypatch, tmp_path):
    ledger = _warning_ledger(tmp_path)
    service, chat = make_service(db, ["done"])
    service.project_instruction_context = ledger
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_args: 100)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_args, **_kwargs: 5)

    def count_exact_fit(messages, *_args, **_kwargs):
        if any(row.get("role") == "system" for row in messages):
            return (
                85
                if any(PROJECT_INSTRUCTION_ROW_KEY in row for row in messages)
                else 84
            )
        return 1

    monkeypatch.setattr(agent_service, "count_tokens_messages", count_exact_fit)
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(),
        response_reserve_tokens=10,
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c-project-exact-fit",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="openai",
    )

    assert outcome.status == RUN_DONE
    assert len(chat.calls) == 1
    payload = chat.calls[0]["messages_payload"]
    assert sum(PROJECT_INSTRUCTION_ROW_KEY in row for row in payload) == 1
    state = agent_service.InstructionChainPayloadState(
        request_builder=lambda messages, schemas: (messages, schemas),
        safe_token_allowance=lambda request, rows: 100,
        count_tokens=lambda rows: 1,
    )
    state.capture([], (), ())
    assert ledger.initial_context_for_chain("primary", state).status == "proceed"


def test_real_ledger_multimodal_fallback_exact_fit_marks_and_sends(
    db, monkeypatch, tmp_path
):
    ledger = _warning_ledger(tmp_path)
    service, chat = make_service(db, ["done"])
    service.project_instruction_context = ledger
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_args: 100)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_args, **_kwargs: 5)
    primary_calls: list[list[dict]] = []
    fallback_calls: list[list[dict]] = []

    def primary_counter(messages, *_args, **_kwargs):
        primary_calls.append(list(messages))
        raise TypeError("multimodal content")

    def multimodal_fallback(messages, *_args, **_kwargs):
        fallback_calls.append(list(messages))
        return 85 if any(PROJECT_INSTRUCTION_ROW_KEY in row for row in messages) else 84

    monkeypatch.setattr(agent_service, "count_tokens_messages", primary_counter)
    monkeypatch.setattr(
        agent_service, "count_console_messages_tokens", multimodal_fallback
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(),
        response_reserve_tokens=10,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        }
    ]

    _run_id, outcome = service.run_turn(
        conversation_id="c-project-multimodal-exact-fit",
        messages=messages,
        config=config,
        api_endpoint="openai",
    )

    assert outcome.status == RUN_DONE
    assert len(chat.calls) == 1
    assert primary_calls and fallback_calls
    assert (
        sum(
            PROJECT_INSTRUCTION_ROW_KEY in row
            for row in chat.calls[0]["messages_payload"]
        )
        == 1
    )


@pytest.mark.parametrize(
    ("limit_value", "count_value"),
    [
        (None, 1),
        (0, 1),
        (-1, 1),
        (RuntimeError("limit"), 1),
        (100, 0),
        (100, -1),
        (100, RuntimeError("count")),
    ],
)
def test_real_ledger_unknown_or_invalid_final_budget_fails_closed(
    db, monkeypatch, tmp_path, limit_value, count_value
):
    ledger = _warning_ledger(tmp_path)
    service, chat = make_service(db, ["must not send"])
    service.project_instruction_context = ledger

    def model_limit(*_args):
        if isinstance(limit_value, Exception):
            raise limit_value
        return limit_value

    def count_invalid(messages, *_args, **_kwargs):
        if isinstance(count_value, Exception):
            raise count_value
        if any(row.get("role") == "system" for row in messages):
            return count_value
        return 1

    monkeypatch.setattr(agent_service, "get_model_token_limit", model_limit)
    monkeypatch.setattr(agent_service, "count_tokens_messages", count_invalid)
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(),
        response_reserve_tokens=10,
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c-project-invalid-final-budget",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="openai",
    )

    assert outcome.status == RUN_ERROR
    assert outcome.steps[0].summary == "project instruction context could not fit"
    assert chat.calls == []


@pytest.mark.parametrize("failing_callback", ["initial", "mark"])
def test_project_instruction_delivery_callback_failures_are_content_free(
    db, monkeypatch, capsys, failing_callback
):
    sentinel = "SECRET-BODY /private/workspace/AGENTS.md"

    class FailingContext(_ProjectInstructionContextSpy):
        def initial_context_for_chain(self, chain_id, payload_state):
            if failing_callback == "initial":
                raise RuntimeError(sentinel)
            return self._delivery(chain_id)

        def mark_payload_sent(self, receipt, payload_rows):
            if failing_callback == "mark":
                raise RuntimeError(sentinel)
            super().mark_payload_sent(receipt, payload_rows)

    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_args: 100_000)
    context = FailingContext()
    service, chat = make_service(db, ["must not send"])
    service.project_instruction_context = context

    run_id, outcome = service.run_turn(
        conversation_id=f"c-project-{failing_callback}-failure",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="openai",
    )

    captured = capsys.readouterr()
    persisted = db.get_run(run_id)
    assert outcome.status == RUN_ERROR
    assert outcome.steps[0].summary == "project_instruction_delivery_failed"
    assert chat.calls == []
    assert sentinel not in repr(outcome)
    assert sentinel not in repr(persisted)
    assert sentinel not in captured.out
    assert sentinel not in captured.err


def test_provider_failure_after_instruction_mark_does_not_undo_advance(db, monkeypatch):
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_args: 100_000)
    context = _ProjectInstructionContextSpy()

    def failing_chat(**_kwargs):
        assert [chain for chain, _payload in context.mark_calls] == ["primary"]
        raise RuntimeError("provider unavailable")

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=failing_chat,
        project_instruction_context=context,
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c-project-provider-failure",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="openai",
    )

    assert outcome.status == RUN_ERROR
    assert [chain for chain, _payload in context.mark_calls] == ["primary"]


def test_later_batch_context_retries_before_review_and_is_marked_on_exact_request(
    db, monkeypatch
):
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_args: 100_000)

    class LaterContext(_ProjectInstructionContextSpy):
        def __init__(self):
            super().__init__(initial_rows=False)
            self.issued = False

        def prepare(self, calls, chain_id, registry, payload_state):
            self.prepare_calls.append((chain_id, tuple(call.name for call in calls)))
            if self.issued:
                return InstructionPreparation("proceed")
            self.issued = True
            return self._delivery(chain_id)

    context = LaterContext()
    reviewed: list = []
    chat = ScriptedChat(
        [
            {
                "content": None,
                "tool_calls": [native_call("calculator", {"expression": "2+2"})],
            },
            "done",
        ]
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=lambda batch: reviewed.append(list(batch)) or {},
        project_instruction_context=context,
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c-project-later",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="openai",
    )

    assert outcome.status == RUN_DONE
    assert context.prepare_calls == [("primary", ("calculator",))]
    assert reviewed == []
    assert len(chat.calls) == 2
    second = chat.calls[1]["messages_payload"]
    instruction_index = next(
        index for index, row in enumerate(second) if PROJECT_INSTRUCTION_ROW_KEY in row
    )
    stub_index = next(
        index
        for index, row in enumerate(second)
        if "Deferred because project instructions were loaded"
        in str(row.get("content", ""))
    )
    assert stub_index < instruction_index
    assert [chain for chain, _payload in context.mark_calls] == ["primary"]


@pytest.mark.parametrize("use_context", [True, False])
def test_child_uses_exactly_one_root_delivery_path(
    db, monkeypatch, tmp_path, use_context
):
    original_setting = agent_service._setting
    monkeypatch.setattr(
        agent_service,
        "_setting",
        lambda key, default: (
            1
            if key == agent_service.MAX_LIVE_SUBAGENTS_KEY
            else original_setting(key, default)
        ),
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_args: 100_000)
    sentinel = "ROOT-INSTRUCTION-SENTINEL"
    raw = sentinel.encode()
    source = InstructionSource(
        canonical_path=tmp_path / "AGENTS.md",
        relative_path="AGENTS.md",
        scope=".",
        kind="standard",
        body=sentinel,
        byte_count=len(raw),
        digest=hashlib.sha256(raw).hexdigest(),
    )
    candidate = StartupInstructionCandidate(
        binding_id="binding",
        binding_root=tmp_path,
        locator_fingerprint="fingerprint",
        dispatch_started_wall_ns=time.time_ns() + 1_000_000_000,
        source=source,
        outcomes=(),
    )
    snapshot = InstructionSnapshot(
        binding_id="binding",
        binding_root=tmp_path,
        locator_fingerprint="fingerprint",
        dispatch_started_wall_ns=candidate.dispatch_started_wall_ns,
        startup_source=source,
        global_outcomes=(),
        primary_delivery=InstructionChainDelivery((source.digest,), ()),
        warning_codes=(),
    )
    context = (
        InstructionActivationLedger(snapshot, nested_max_bytes=0)
        if use_context
        else None
    )
    chat = ScriptedChat(
        [
            {
                "content": None,
                "tool_calls": [native_call(SPAWN_TOOL_NAME, {"task": "inspect"}, "s")],
            },
            "child done",
            "parent done",
        ]
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        startup_instruction_candidate=candidate,
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
        project_instruction_context=context,
    )
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(SPAWN_TOOL_NAME,),
        native_tools=True,
        response_reserve_tokens=10,
    )

    _run_id, outcome = service.run_turn(
        conversation_id=f"c-project-child-{use_context}",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="openai",
    )

    assert outcome.status == RUN_DONE
    assert len(chat.calls) == 3
    parent_first = chat.calls[0]["messages_payload"]
    child_first = chat.calls[1]["messages_payload"]
    assert sum(sentinel in str(row) for row in parent_first) == 1
    assert sum(sentinel in str(row) for row in child_first) == 1


def test_native_endpoint_sends_tools_and_suppresses_fence_protocol(db):
    service, chat = make_service(
        db,
        [
            {
                "content": None,
                "tool_calls": [native_call("calculator", {"expression": "2+2"})],
            },
            "4.",
        ],
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=CFG,
        api_endpoint="groq",
        should_cancel=lambda: False,
    )
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    first = chat.calls[0]
    names = [t["function"]["name"] for t in first["tools"]]
    assert "calculator" in names and "spawn_subagent" in names
    assert (
        "tool_call" not in first["messages_payload"][0]["content"]
    )  # no fence protocol
    # Second call's history carries the native pairing:
    second_payload = chat.calls[1]["messages_payload"]
    assistant = [m for m in second_payload if m["role"] == "assistant"][0]
    assert assistant["tool_calls"][0]["function"]["name"] == "calculator"
    tool_msg = [m for m in second_payload if m.get("role") == "tool"][0]
    assert tool_msg["tool_call_id"] == "c1" and "4" in tool_msg["content"]


def test_native_multi_call_reply_dispatches_both_tools_in_one_turn(db):
    service, chat = make_service(
        db,
        [
            {
                "content": None,
                "tool_calls": [
                    native_call("calculator", {"expression": "2+2"}, "a"),
                    native_call("get_current_datetime", {}, "b"),
                ],
            },
            "done",
        ],
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="openai",
        should_cancel=lambda: False,
    )
    assert outcome.status == RUN_DONE
    tool_results = [s for s in outcome.steps if s.kind == "tool_result"]
    assert [s.tool_name for s in tool_results] == ["calculator", "get_current_datetime"]
    assert len(chat.calls) == 2  # one batch turn + one final turn


def test_fence_fallback_unchanged_for_llama_cpp(db):
    service, chat = make_service(db, [fence("calculator", {"expression": "2+2"}), "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=CFG,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )
    assert outcome.status == RUN_DONE
    assert "tools" not in chat.calls[0]  # no tools= kwarg at all
    assert "tool_call" in chat.calls[0]["messages_payload"][0]["content"]


def test_native_kill_switch_forces_fence(db):
    cfg = dataclasses.replace(CFG, native_tools=False)
    service, chat = make_service(db, [fence("calculator", {"expression": "2+2"}), "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=cfg,
        api_endpoint="groq",
        should_cancel=lambda: False,
    )
    assert outcome.status == RUN_DONE and "tools" not in chat.calls[0]


def test_native_subagent_turns_also_carry_tools(db):
    # PR2a Task 6.5: the fleet is ON by default, so the child runs on its
    # own thread -- addressed script, and `chat.child_calls[task]` instead
    # of `chat.calls[1]`. Same call, named rather than counted.
    chat = FleetChat(
        [
            {
                "content": None,
                "tool_calls": [native_call("spawn_subagent", {"task": "say hi"}, "s1")],
            },
            "done",
        ],
        {"say hi": ["hi from child"]},  # child's (native-mode) only turn
    )
    service = _service_with(db, chat)
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="groq",
        should_cancel=lambda: False,
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    assert outcome.status == RUN_DONE
    child_call = chat.child_calls["say hi"][0]
    assert child_call["messages_payload"][0]["content"].startswith(
        SUBAGENT_SYSTEM_PROMPT
    )
    assert "tools" in child_call  # native_tools propagated to the child


def test_malformed_native_arguments_error_is_echoed_and_recoverable(db):
    bad = {
        "id": "m1",
        "type": "function",
        "function": {"name": "calculator", "arguments": "{broken"},
    }
    service, chat = make_service(
        db,
        [
            {"content": None, "tool_calls": [bad]},
            {
                "content": None,
                "tool_calls": [native_call("calculator", {"expression": "2+2"}, "m2")],
            },
            "4.",
        ],
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=CFG,
        api_endpoint="groq",
        should_cancel=lambda: False,
    )
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    retry_payload = chat.calls[1]["messages_payload"]
    tool_msgs = [m for m in retry_payload if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "m1"
    assert "ERROR" in tool_msgs[0]["content"]  # empty-args invoke fails, echoed


def test_plain_answer_persists_done_run(db):
    service, chat = make_service(db, ["Tokyo."])
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "capital of Japan?"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE and outcome.final_text == "Tokyo."
    run = db.get_run(run_id)
    assert run["status"] == "done" and run["result"] == "Tokyo."
    assert run["agent_kind"] == "primary"
    assert all(s["created_at"] for s in run["steps"])


def test_service_wires_exact_continuation_context_callback_and_resume_input(
    db, monkeypatch
):
    captured = {}
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
        state="active",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("private",),
                calls=(
                    ContinuationCall(
                        "call-1",
                        "calculator",
                        '{"expression":"2+2"}',
                        "pending",
                    ),
                ),
            ),
        ),
    )

    def persist(event):
        raise AssertionError("not called by this wiring-only test")

    def expand(actual):
        return [{"opaque": True}]

    def fake_loop(config, messages, active, deps, **kwargs):
        captured["context"] = deps.continuation_context
        captured["callback"] = deps.persist_provider_continuation
        captured["expand"] = deps.expand_provider_continuation
        captured.update(kwargs)
        return agent_service.RunOutcome(status=RUN_DONE, steps=[], final_text="done")

    monkeypatch.setattr(agent_service, "run_agent_loop", fake_loop)
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=ScriptedChat(["unused"]),
        persist_provider_continuation=persist,
        expand_provider_continuation=expand,
    )
    run_id, outcome = service.run_turn(
        conversation_id="conversation",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="openai",
        continuation_owner_message_id="assistant-owner",
        continuation_durability="ephemeral",
        restore_provider_continuation=checkpoint,
        restore_provider_target=ContinuationRestoreTarget(
            "deepseek",
            "deepseek-v4-flash",
            "responses",
            "https://api.deepseek.com/v1",
        ),
        resume_provider_continuation=True,
    )

    assert outcome.status == RUN_DONE
    assert captured["context"] == ContinuationEventContext(
        owner_message_id="assistant-owner",
        run_id=run_id,
        agent_kind="primary",
        durability="ephemeral",
    )
    assert captured["callback"] is persist
    assert captured["expand"] is expand
    assert captured["restore_provider_continuation"] is checkpoint
    assert captured["restore_provider_target"] == ContinuationRestoreTarget(
        "deepseek",
        "deepseek-v4-flash",
        "responses",
        "https://api.deepseek.com/v1",
    )
    assert captured["resume_provider_continuation"] is True


def test_system_message_carries_protocol_and_user_prompt(db):
    service, chat = make_service(db, ["hi"])
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    call = chat.calls[0]
    assert call["api_endpoint"] == "llama_cpp"
    assert call["streaming"] is False and call["model"] == "test-model"
    system = call["messages_payload"][0]
    assert system["role"] == "system"
    assert "You are helpful." in system["content"]
    assert "```tool_call" in system["content"]  # protocol rendered
    assert "calculator" in system["content"]  # direct-disclosed
    assert SPAWN_TOOL_NAME in system["content"]
    assert "find_tools" not in system["content"]  # small catalog


def test_real_tool_executes_through_gate(db):
    service, chat = make_service(
        db, [fence("calculator", {"expression": "6*7"}), "It is 42."]
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "6*7?"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE and outcome.final_text == "It is 42."
    # The tool result really came from CalculatorTool:
    followup = chat.calls[1]["messages_payload"]
    assert any(
        "42" in m["content"]
        for m in followup
        if m["role"] == "user" and "Tool result" in m["content"]
    )


def test_permission_gate_blocks_disallowed_tool(db):
    narrow = AgentConfig(
        model="m", system_prompt="s", allowed_tools=("get_current_datetime",)
    )
    service, _ = make_service(db, [fence("calculator", {"expression": "1"}), "gave up"])
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=narrow,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    run = db.get_run(run_id)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert results and "not permitted" in results[0]["result"]


def test_spawn_creates_linked_child_with_clean_context(db):
    # PR2a Task 6.5: addressed script (the child is on its own thread).
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "compute 6*7"}),  # parent turn 1
            "The sub-agent says 42.",  # parent turn 2
        ],
        {"compute 6*7": ["sub answer: 42"]},  # CHILD turn 1
    )
    service = _service_with(db, chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "delegate this"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    assert outcome.status == RUN_DONE and outcome.subagents_spawned == 1
    runs = db.list_runs("c")
    child = next(r for r in runs if r["agent_kind"] == "subagent")
    assert child["parent_run_id"] == run_id
    assert child["task"] == "compute 6*7"
    assert child["status"] == "done" and child["result"] == "sub answer: 42"
    # Clean context: the child's provider call saw ONLY its task + its own
    # system prompt — never the parent's transcript. Addressed by task text
    # rather than by `calls[1]`: same call, named rather than counted.
    child_call = chat.child_calls["compute 6*7"][0]["messages_payload"]
    assert child_call[0]["role"] == "system"
    assert SUBAGENT_SYSTEM_PROMPT.split(".")[0] in child_call[0]["content"]
    assert child_call[1] == {"role": "user", "content": "compute 6*7"}
    assert not any("delegate this" in m["content"] for m in child_call)
    assert db.count_subagent_runs("c") == 1


def test_inline_spawn_capture_failure_never_links_child_to_absent_step(db, monkeypatch):
    original_insert = db.insert_steps_at_indices

    def fail_spawn_capture(run_id, indexed_steps):
        if any(step["kind"] == "spawn" for _index, step in indexed_steps):
            raise RuntimeError("persistent spawn capture failure")
        return original_insert(run_id, indexed_steps)

    monkeypatch.setattr(db, "insert_steps_at_indices", fail_spawn_capture)
    service, _chat = make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "inspect"}),
            "child done",
            "parent done",
        ],
    )
    parent_id, outcome = service.run_turn(
        conversation_id="inline-spawn-capture",
        messages=[{"role": "user", "content": "delegate"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    assert outcome.status == RUN_DONE
    rows = db.list_runs("inline-spawn-capture", include_superseded=True)
    parent = next(row for row in rows if row["id"] == parent_id)
    child = next(row for row in rows if row["agent_kind"] == "subagent")
    diagnostic = next(step for step in parent["steps"] if step["kind"] == "capture_failed")
    diagnostic_id = f"agent-step:{parent_id}:{diagnostic['index']}"
    assert child["spawn_event_id"] == diagnostic_id
    assert next(
        step for step in child["steps"] if step["kind"] == "agent_run_reserved"
    )["parent_event_id"] == diagnostic_id


def test_spawn_propagates_workspace_note_to_child_prompt(db):
    """A non-default workspace note rides the parent config onto the child's
    config, so a spawned sub-agent -- which operates on the same workspace
    roots -- sees it too. Appended last (in call_model), so the sub-agent
    identity prefix still leads the child's system prompt."""
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "compute 6*7"}),  # parent turn 1
            "The sub-agent says 42.",  # parent turn 2
        ],
        {"compute 6*7": ["sub answer: 42"]},  # CHILD turn 1
    )
    service = _service_with(db, chat)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("calculator", "get_current_datetime", SPAWN_TOOL_NAME),
        workspace_context_note="CHILD_WS_NOTE",
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "delegate this"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)

    child_system = chat.child_calls["compute 6*7"][0]["messages_payload"][0]["content"]
    assert "CHILD_WS_NOTE" in child_system
    assert child_system.startswith(SUBAGENT_SYSTEM_PROMPT.split(".")[0])


def test_run_turn_records_assistant_message_id_on_primary_only(db):
    """The primary run records the assistant_message_id it is handed; a
    spawned sub-agent run records None (it produces no transcript reply)."""
    service, _ = make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "compute 6*7"}),  # parent turn 1
            "sub answer: 42",  # CHILD turn 1
            "The sub-agent says 42.",  # parent turn 2
        ],
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "delegate this"}],
        config=CFG,
        api_endpoint="llama_cpp",
        assistant_message_id="a1",
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    assert outcome.status == RUN_DONE and outcome.subagents_spawned == 1
    assert db.get_run(run_id)["assistant_message_id"] == "a1"
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    assert child["assistant_message_id"] is None


def test_subagent_result_is_capped(db, inline_spawns):
    """A child's oversized answer reaches the parent capped, not whole.

    THE INLINE CARRIER, verbatim. This is the original test, re-pinned on
    `max_live_subagents = 1` -- the shipped kill switch, and the path whose
    own cap (`agent_service`'s `spawn`, inline branch) is the code under
    test here.

    PR2a Task 6.5 review caught me moving this assertion to `wait_agents`
    and calling it "the same guarantee, new carrier". It was not: the
    fleet's own cap was ALREADY covered by
    `test_fleet_runtime.test_wait_agents_splits_the_history_budget_across_children`,
    so the move landed on held ground and left the inline cap with no
    coverage at all -- deleting those three lines kept Tests/Agents and
    Tests/Chat fully green. Both carriers are now pinned: this test for the
    inline branch, `test_subagent_result_is_capped_under_the_fleet` below
    for the threaded one.
    """
    long_answer = "x" * 10000
    service, _ = make_service(
        db, [fence(SPAWN_TOOL_NAME, {"task": "t"}), long_answer, "done"]
    )
    tight = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(max_subagent_result_chars=100),
    )
    _, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=tight,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    run = db.list_runs("c", include_superseded=False)
    parent = next(r for r in run if r["agent_kind"] == "primary")
    capped = [s for s in parent["steps"] if s["kind"] == "tool_result"][0]
    assert "[truncated]" in capped["result"]
    assert len(capped["result"]) < 300


def test_subagent_result_is_capped_under_the_fleet(db):
    """The same cap, on the threaded carrier: `wait_agents`.

    Companion to the inline test above (PR2a Task 6.5). With the fleet on,
    `spawn` returns a handle and the child's text reaches the supervisor
    through `wait_agents`, so that is where the cap must be observable.
    Both branches cap, and both are now pinned.
    """
    long_answer = "x" * 10000
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "t"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"t": [long_answer]},
    )
    service = _service_with(db, chat)
    tight = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(SPAWN_TOOL_NAME,),
        # max_steps raised only to fit the extra collection round the fleet
        # adds; nothing here ever asserted on the step budget.
        budget=RunBudget(max_subagent_result_chars=100, max_steps=16),
    )
    _, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=tight,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    run = db.list_runs("c", include_superseded=False)
    parent = next(r for r in run if r["agent_kind"] == "primary")
    capped = [
        s
        for s in parent["steps"]
        if s["kind"] == "tool_result" and s["tool_name"] == WAIT_AGENTS_TOOL_NAME
    ][0]
    assert "[truncated]" in capped["result"]
    assert len(capped["result"]) < 300


def test_child_cannot_spawn(db):
    # PR2a Task 6.5: addressed script (the child is on its own thread).
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "t"}),  # parent spawns
            "parent done",
        ],
        {
            "t": [
                fence(SPAWN_TOOL_NAME, {"task": "nested"}),  # CHILD tries to spawn
                "child recovered",  # child answers
            ]
        },
    )
    service = _service_with(db, chat)
    _, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c") == 1  # no grandchildren


def test_supersede_marks_old_tree_before_new_run(db, monkeypatch):
    service, _ = make_service(db, ["first answer"])
    old_id, _ = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    local_id = db.create_run(
        conversation_id="c",
        agent_kind="local_command",
        task="Local command",
    )
    original_list_runs = db.list_runs
    requested_kinds: list[str | None] = []

    def exact_kind_only(conversation_id, *args, **kwargs):
        kind = kwargs.get("agent_kind")
        requested_kinds.append(kind)
        if kind is None:
            raise AssertionError("broad query would hydrate poison local-command steps")
        return original_list_runs(conversation_id, *args, **kwargs)

    monkeypatch.setattr(db, "list_runs", exact_kind_only)
    service2, _ = make_service(db, ["second answer"])
    new_id, _ = service2.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
        supersede_run_id=old_id,
    )
    assert db.get_run(old_id)["status"] == "superseded"
    assert db.get_run(new_id)["status"] == "done"
    assert db.get_run(local_id)["status"] == "running"
    assert requested_kinds == ["primary", "subagent"]
    assert [
        step["kind"]
        for step in db.get_run(old_id)["steps"]
        if step["kind"].startswith("agent_run_")
    ] == [
        "agent_run_created",
        "agent_run_started",
        "agent_run_completed",
        "agent_run_superseded",
    ]
    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="supersede-reload")
    runs = reopened.list_runs("c", include_superseded=True)
    records = [
        record
        for turn in derive_trajectory(
            messages=[],
            usage_by_id={},
            traj_rows=[],
            variant_sets=[],
            compaction_records=[],
            agent_runs=runs,
            agent_steps=[
                {**step, "run_id": row["id"], "conversation_id": "c"}
                for row in runs
                for step in row["steps"]
            ],
        ).turns
        for record in turn.records
    ]
    assert any(
        record.run_id == old_id and record.kind == "agent_run_superseded"
        for record in records
    )
    reopened.close()


@pytest.mark.parametrize(
    "failed_kind",
    ["agent_run_created", "agent_run_started", "agent_run_completed"],
)
def test_agent_lifecycle_capture_failure_uses_actual_diagnostic_cause_after_reload(
    db, monkeypatch, failed_kind
):
    original_insert = db.insert_steps_at_indices
    original_terminal = db.set_terminal_with_step
    failed = False

    def fail_lifecycle_once(run_id, indexed_steps):
        nonlocal failed
        if not failed and any(
            step["kind"] == failed_kind for _index, step in indexed_steps
        ):
            failed = True
            raise RuntimeError("simulated lifecycle storage failure")
        return original_insert(run_id, indexed_steps)

    monkeypatch.setattr(db, "insert_steps_at_indices", fail_lifecycle_once)

    def fail_terminal_once(run_id, status, result, terminal_step):
        nonlocal failed
        if not failed and terminal_step["kind"] == failed_kind:
            failed = True
            raise RuntimeError("simulated lifecycle storage failure")
        return original_terminal(run_id, status, result, terminal_step)

    if failed_kind == "agent_run_completed":
        monkeypatch.setattr(db, "set_terminal_with_step", fail_terminal_once)
    service, _ = make_service(db, ["safe answer"])
    run_id, outcome = service.run_turn(
        conversation_id=f"capture-failure-{failed_kind}",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE
    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="lifecycle-failure-reload")
    row = reopened.get_run(run_id)
    assert row["status"] == RUN_DONE
    steps = row["steps"]
    diagnostics = [step for step in steps if step["kind"] == "capture_failed"]
    expected_diagnostics = 0 if failed_kind == "agent_run_completed" else 1
    assert len(diagnostics) == expected_diagnostics
    lifecycle_kinds = [
        step["kind"] for step in steps if step["kind"].startswith("agent_run_")
    ]
    assert len(lifecycle_kinds) == len(set(lifecycle_kinds))
    if failed_kind == "agent_run_completed":
        assert len([step for step in steps if step["kind"] == failed_kind]) == 1
    else:
        diagnostic = diagnostics[0]
        assert failed_kind not in [step["kind"] for step in steps]
        assert diagnostic["field_states"][failed_kind] == "not_observed"
    event_ids = {
        f"agent-step:{run_id}:{step['index']}" for step in steps
    } | {f"agent-run:{run_id}"}
    for step in steps:
        assert step["parent_event_id"] in event_ids
        assert step["source_event_id"] is None or step["source_event_id"] in event_ids
    if failed_kind == "agent_run_created":
        started = next(step for step in steps if step["kind"] == "agent_run_started")
        assert started["parent_event_id"] == (
            f"agent-step:{run_id}:{diagnostic['index']}"
        )
    reopened.close()


def test_runtime_observation_and_concurrent_cancellation_share_owner_sequence(
    db, monkeypatch
):
    original_insert = db.insert_steps_at_indices
    runtime_waiting = threading.Event()
    release_runtime = threading.Event()
    blocked = False

    def block_model_request_once(run_id, indexed_steps):
        nonlocal blocked
        if not blocked and any(
            step["kind"] == "model_request_started"
            for _index, step in indexed_steps
        ):
            blocked = True
            runtime_waiting.set()
            assert release_runtime.wait(5)
        return original_insert(run_id, indexed_steps)

    monkeypatch.setattr(db, "insert_steps_at_indices", block_model_request_once)
    service, _chat = make_service(db, ["safe answer"])
    result: dict[str, object] = {}

    def run():
        result["value"] = service.run_turn(
            conversation_id="owner-seq-race",
            messages=[{"role": "user", "content": "q"}],
            config=CFG,
            api_endpoint="llama_cpp",
        )

    worker = threading.Thread(target=run)
    worker.start()
    assert runtime_waiting.wait(5)
    run_id = db.list_runs("owner-seq-race", include_superseded=True)[0]["id"]
    assert db.set_status(run_id, "cancelled") is True
    service._record_terminal_lifecycle(run_id, "cancelled")
    release_runtime.set()
    worker.join(5)
    assert not worker.is_alive()

    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="owner-seq-race-reload")
    row = reopened.get_run(run_id)
    owner_sequences = [
        step["owner_seq"]
        for step in row["steps"]
        if step["owner_seq"] is not None
    ]
    assert len(owner_sequences) == len(set(owner_sequences))
    assert sorted(owner_sequences) == list(
        range(min(owner_sequences), max(owner_sequences) + 1)
    )
    inputs = {
        "messages": [],
        "usage_by_id": {},
        "traj_rows": [],
        "variant_sets": [],
        "compaction_records": [],
        "agent_runs": [row],
        "agent_steps": [
            {**step, "run_id": run_id, "conversation_id": "owner-seq-race"}
            for step in row["steps"]
        ],
    }
    projected = [
        (record.event_id, record.source_seq)
        for turn in derive_trajectory(**inputs).turns
        for record in turn.records
    ]
    assert projected == [
        (record.event_id, record.source_seq)
        for turn in derive_trajectory(**inputs).turns
        for record in turn.records
    ]
    projected_sequences = [
        source_seq for _event_id, source_seq in projected if source_seq is not None
    ]
    assert projected_sequences == sorted(projected_sequences)
    reopened.close()


def test_transient_control_capture_recovery_has_unique_owner_sequence(db, monkeypatch):
    original_insert = db.insert_steps_at_indices
    failed = False

    def fail_tool_call_once(run_id, indexed_steps):
        nonlocal failed
        if not failed and any(
            step["kind"] == "tool_call" for _index, step in indexed_steps
        ):
            failed = True
            raise RuntimeError("transient control capture failure")
        return original_insert(run_id, indexed_steps)

    monkeypatch.setattr(db, "insert_steps_at_indices", fail_tool_call_once)
    service, _chat = make_service(
        db,
        [fence("calculator", {"expression": "1+1"}), "safe answer"],
    )
    run_id, outcome = service.run_turn(
        conversation_id="control-capture-owner-seq",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="control-capture-owner-seq-reload")
    row = reopened.get_run(run_id)
    original = next(step for step in row["steps"] if step["kind"] == "tool_call")
    diagnostic = next(
        step
        for step in row["steps"]
        if step["kind"] == "capture_failed"
        and f"failed_tool_call_{original['index']}" in step["field_states"]
    )
    owner_sequences = [
        step["owner_seq"]
        for step in row["steps"]
        if step["owner_seq"] is not None
    ]
    assert diagnostic["owner_seq"] > original["owner_seq"]
    assert len(owner_sequences) == len(set(owner_sequences))
    projection_inputs = {
        "messages": [],
        "usage_by_id": {},
        "traj_rows": [],
        "variant_sets": [],
        "compaction_records": [],
        "agent_runs": [row],
        "agent_steps": [
            {**step, "run_id": run_id, "conversation_id": row["conversation_id"]}
            for step in row["steps"]
        ],
    }
    first = derive_trajectory(**projection_inputs)
    second = derive_trajectory(**projection_inputs)
    assert first == second
    reopened.close()


def test_terminal_recovery_does_not_duplicate_lifecycle_transition(db):
    service, _ = make_service(db, ["safe answer"])
    run_id, outcome = service.run_turn(
        conversation_id="terminal-recovery",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )

    service._persist(run_id, outcome)

    completed = [
        step
        for step in db.get_run(run_id)["steps"]
        if step["kind"] == "agent_run_completed"
    ]
    assert len(completed) == 1


def test_project_instruction_service_error_has_durable_causal_identity(
    db, monkeypatch
):
    def fail_before_runtime(*_args, **_kwargs):
        raise agent_service._ProjectInstructionPayloadError("delivery failed")

    monkeypatch.setattr(agent_service, "run_agent_loop", fail_before_runtime)
    service, _ = make_service(db, [])
    run_id, outcome = service.run_turn(
        conversation_id="project-error",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_ERROR

    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="project-error-reload")
    row = reopened.get_run(run_id)
    steps = row["steps"]
    records = [
        record
        for turn in derive_trajectory(
            messages=[],
            usage_by_id={},
            traj_rows=[],
            variant_sets=[],
            compaction_records=[],
            agent_runs=[row],
            agent_steps=[
                {**step, "run_id": run_id, "conversation_id": "project-error"}
                for step in steps
            ],
        ).turns
        for record in turn.records
    ]
    assert [
        record.kind
        for record in records
        if record.kind.startswith("agent_run_") or record.kind == "error"
    ] == [
        "agent_run_created",
        "agent_run_started",
        "error",
        "agent_run_failed",
    ]
    error = next(step for step in steps if step["kind"] == "error")
    started = next(step for step in steps if step["kind"] == "agent_run_started")
    failed = next(step for step in steps if step["kind"] == "agent_run_failed")
    error_event_id = f"agent-step:{run_id}:{error['index']}"
    started_event_id = f"agent-step:{run_id}:{started['index']}"
    assert error["owner_seq"] == 2
    assert error["parent_event_id"] == started_event_id
    assert error["source_event_id"] == started_event_id
    assert failed["parent_event_id"] == error_event_id
    assert failed["source_event_id"] == error_event_id
    reopened.close()


def test_service_error_capture_recovers_once_before_failed_lifecycle(
    db, monkeypatch
):
    original_insert = db.insert_steps_at_indices
    failed_once = False
    attempts = 0

    def fail_service_error_once(run_id, indexed_steps):
        nonlocal attempts, failed_once
        has_error = any(step["kind"] == "error" for _index, step in indexed_steps)
        attempts += int(has_error)
        if not failed_once and has_error:
            failed_once = True
            raise RuntimeError("simulated service error capture failure")
        return original_insert(run_id, indexed_steps)

    monkeypatch.setattr(db, "insert_steps_at_indices", fail_service_error_once)
    service, _ = make_service(
        db,
        [lambda: (_ for _ in ()).throw(RuntimeError("provider failed"))],
    )
    run_id, outcome = service.run_turn(
        conversation_id="error-capture-recovery",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_ERROR
    assert attempts == 2

    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="error-recovery-reload")
    reloaded = reopened.get_run(run_id)["steps"]
    errors = [step for step in reloaded if step["kind"] == "error"]
    failed = [step for step in reloaded if step["kind"] == "agent_run_failed"]
    assert len(errors) == len(failed) == 1
    error_event_id = f"agent-step:{run_id}:{errors[0]['index']}"
    assert failed[0]["owner_seq"] == errors[0]["owner_seq"] + 1
    assert failed[0]["parent_event_id"] == error_event_id
    assert failed[0]["source_event_id"] == error_event_id
    reopened.close()


def test_persistent_service_error_capture_is_diagnosed_without_dangling_links(
    db, monkeypatch
):
    original_insert = db.insert_steps_at_indices
    attempts = 0

    def fail_service_error(run_id, indexed_steps):
        nonlocal attempts
        if any(step["kind"] == "error" for _index, step in indexed_steps):
            attempts += 1
            raise RuntimeError("persistent service error capture failure")
        return original_insert(run_id, indexed_steps)

    monkeypatch.setattr(db, "insert_steps_at_indices", fail_service_error)
    service, _ = make_service(
        db,
        [lambda: (_ for _ in ()).throw(RuntimeError("provider failed"))],
    )
    run_id, outcome = service.run_turn(
        conversation_id="persistent-error-capture",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_ERROR
    assert attempts == 2

    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="persistent-error-reload")
    row = reopened.get_run(run_id)
    assert row["status"] == RUN_ERROR
    assert not any(step["kind"] == "error" for step in row["steps"])
    assert not any(step["kind"] == "agent_run_failed" for step in row["steps"])
    diagnostic = next(step for step in row["steps"] if step["kind"] == "capture_failed")
    event_ids = {
        f"agent-step:{run_id}:{step['index']}" for step in row["steps"]
    } | {f"agent-run:{run_id}"}
    assert diagnostic["status"] == "incomplete"
    assert diagnostic["field_states"]["payload"] == "capture_failed"
    for step in row["steps"]:
        assert step["parent_event_id"] in event_ids
        assert step["source_event_id"] is None or step["source_event_id"] in event_ids
    reopened.close()


def test_stuck_run_persists_stuck_status(db):
    replies = [fence("calculator", {"expression": "1"})] * 10
    service, _ = make_service(db, replies)
    tight = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_steps=3),
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=tight,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_STUCK
    assert db.get_run(run_id)["status"] == "stuck"


class FakeBigProvider:
    """A deterministic catalog provider used by discovery/load tests."""

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=f"fake:t{i}",
                name=f"t{i}",
                one_line_description=f"tool {i}",
                source="fake",
            )
            for i in range(19)
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id.split(":")[1],
            description="fake",
            parameters={"type": "object"},
        )

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content=f"invoked {tool_id}")


def _forced_discovery_plan(system_prompt: str = "s") -> FirstRequestSchemaPlan:
    return FirstRequestSchemaPlan(
        active_schemas=(),
        runtime_schemas=(FIND_TOOLS_SCHEMA, LOAD_TOOLS_SCHEMA),
        offer_find_load=True,
        log_active=False,
        system_prompt=system_prompt,
    )


def test_run_turn_reuses_planned_runtime_schema_set_verbatim(db):
    service, chat = make_service(db, ["done"])
    plan = _forced_discovery_plan()

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=dataclasses.replace(CFG, native_tools=True),
        api_endpoint="groq",
        first_request_schema_plan=plan,
    )

    assert outcome.status == RUN_DONE
    assert [tool["function"]["name"] for tool in chat.calls[0]["tools"]] == [
        FIND_TOOLS_NAME,
        LOAD_TOOLS_NAME,
    ]


def test_run_turn_reuses_planned_agent_roster_without_db_reread(db):
    definition = AgentDefinition(
        name="researcher",
        description="Find evidence",
        instructions="Research the task.",
    )
    plan = FirstRequestSchemaPlan(
        active_schemas=(),
        runtime_schemas=(agent_service.build_spawn_schema((definition,)),),
        offer_find_load=False,
        log_active=False,
        system_prompt="s",
        agent_definitions=(definition,),
        fleet_max_live=1,
    )
    service, _chat = make_service(db, ["done"])

    def unexpected_reread(*_args, **_kwargs):
        raise AssertionError("planned roster must not be re-read")

    db.list_agent_definitions = unexpected_reread
    _run_id, outcome = service.run_turn(
        conversation_id="c-planned-roster",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
        first_request_schema_plan=plan,
    )

    assert outcome.status == RUN_DONE
    assert service._turn_definitions == [definition]


def test_run_turn_reuses_planned_fleet_gate(monkeypatch, db):
    plan = FirstRequestSchemaPlan(
        active_schemas=(),
        runtime_schemas=(
            agent_service.WAIT_AGENTS_SCHEMA,
            agent_service.CHECK_AGENTS_SCHEMA,
            agent_service.SEND_TO_AGENT_SCHEMA,
        ),
        offer_find_load=False,
        log_active=False,
        system_prompt="s",
        agent_definitions=(),
        fleet_max_live=3,
    )
    monkeypatch.setattr(agent_service, "_setting", lambda *_args: 1)
    service, _chat = make_service(db, ["done"])

    _run_id, outcome = service.run_turn(
        conversation_id="c-planned-fleet",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
        first_request_schema_plan=plan,
    )

    assert outcome.status == RUN_DONE
    assert service._fleet is not None
    assert service._fleet.max_live == 3


def test_load_tools_accepts_more_than_old_count_cap_when_request_fits(db):
    """A fitting load is not truncated by any cumulative tool count."""
    registry = ToolCatalogRegistry()
    registry.register_provider(FakeBigProvider())
    allowed = tuple(f"t{i}" for i in range(19))
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=allowed,
        budget=RunBudget(max_steps=20),
    )
    chat = ScriptedChat(
        [
            fence(LOAD_TOOLS_NAME, {"ids": ["fake:t0", "fake:t1", "fake:t2"]}),
            fence("t2", {}),
            fence("t0", {}),
            "done",
        ]
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=config,
        api_endpoint="llama_cpp",
        first_request_schema_plan=_forced_discovery_plan(),
    )

    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert len(tool_results) == 3

    load_result, t2_result, t0_result = tool_results
    assert load_result["result"] == "loaded: t0, t1, t2"
    assert t2_result["result"] == "invoked fake:t2"
    assert t0_result["result"] == "invoked fake:t0"


def test_provider_exception_persists_error_status(db):
    def exploding_chat(**kwargs):
        raise RuntimeError("connection refused")

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(db=db, registry=registry, chat_call=exploding_chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == "error"
    run = db.get_run(run_id)
    assert run["status"] == "error"
    assert any("connection refused" in (s.get("summary") or "") for s in run["steps"])


# --- G3: reloading an already-active tool must not consume active-tool
# room or desync the gate's disclosed set from the loop's own `active`
# list. ---


def test_replacement_updates_permission_set_in_lockstep(db):
    registry = ToolCatalogRegistry()
    registry.register_provider(FakeBigProvider())
    allowed = tuple(f"t{i}" for i in range(19))
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=allowed,
        budget=RunBudget(max_steps=30),
    )
    chat = ScriptedChat(
        [
            fence(LOAD_TOOLS_NAME, {"ids": ["fake:t0"]}),
            fence("t0", {}),
            fence(LOAD_TOOLS_NAME, {"ids": ["fake:t1"]}),
            fence("t0", {}),
            fence("t1", {}),
            "done",
        ]
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=config,
        api_endpoint="llama_cpp",
        first_request_schema_plan=_forced_discovery_plan(),
    )

    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    load1, t0_before, load2, t0_after, t1_result = tool_results
    assert load1["result"] == "loaded: t0"
    assert t0_before["result"] == "invoked fake:t0"
    assert load2["result"] == "loaded: t1"
    assert "Tool not permitted: t0" in t0_after["result"]
    assert t0_after["tool_outcome"] == "blocked"
    assert t1_result["result"] == "invoked fake:t1"


# --- Q7: disclosure (initial active set, find_tools, load_tools) must
# respect config.allowed_tools; the permission gate is a backstop, not
# the only checkpoint. ---


def test_direct_disclosure_excludes_disallowed_tools(db):
    narrow = AgentConfig(
        model="m", system_prompt="s", allowed_tools=("calculator", SPAWN_TOOL_NAME)
    )
    service, chat = make_service(db, ["ok"])
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=narrow,
        api_endpoint="llama_cpp",
    )
    system = chat.calls[0]["messages_payload"][0]["content"]
    assert "calculator" in system
    assert "get_current_datetime" not in system


def test_find_and_load_tools_respect_allowed_tools(db):
    registry = ToolCatalogRegistry()
    registry.register_provider(FakeBigProvider())
    # Catalog has t0..t10; only t0 is allowed even though t1 exists.
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("t0",),
        budget=RunBudget(max_steps=20),
    )
    chat = ScriptedChat(
        [
            fence(FIND_TOOLS_NAME, {"query": "t"}),
            fence(LOAD_TOOLS_NAME, {"ids": ["fake:t0", "fake:t1"]}),
            fence("t1", {}),
            "done",
        ]
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=config,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    find_result, load_result, t1_result = tool_results
    assert "t0" in find_result["result"]
    assert "t1" not in find_result["result"]
    assert load_result["result"] == "loaded: t0; invalid tool ids: fake:t1"
    assert "Tool not permitted: t1" in t1_result["result"]


# --- task-244 AC #3/#4: load_tools must fall back to resolve_name() when
# a model echoes a bare tool NAME (as seen in a find_tools result line)
# instead of the catalog id. ---


def test_load_tools_with_bare_name_loads_via_resolve_name_fallback(db):
    """AC #4: models echo the tool NAME from a find_tools result line, not
    the catalog id. load_tools(ids=["calculator"]) must load the tool."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FakeBigProvider())  # schema cost forces discovery
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator", "get_current_datetime"),
        budget=RunBudget(max_steps=20),
    )
    chat = ScriptedChat(
        [
            fence(LOAD_TOOLS_NAME, {"ids": ["calculator"]}),
            fence("calculator", {"expression": "2+2"}),
            "4.",
        ]
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=config,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    load_result, calc_result = tool_results
    # The bare name must actually LOAD -- not just avoid the error string.
    assert load_result["result"] == "loaded: calculator"
    calc_payload = json.loads(calc_result["result"])
    assert calc_payload["result"] == 4


def test_load_tools_bare_name_still_respects_allow_list(db):
    """A resolvable bare name OUTSIDE config.allowed_tools stays refused
    with an explicit invalid-input category."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FakeBigProvider())
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_steps=20),
    )
    chat = ScriptedChat(
        [
            # "get_current_datetime" resolves via resolve_name(), but it is
            # not in config.allowed_tools -- Q7(c) must still refuse it.
            fence(LOAD_TOOLS_NAME, {"ids": ["get_current_datetime"]}),
            "done",
        ]
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=config,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    (load_result,) = tool_results
    assert load_result["result"] == "ERROR: invalid tool ids: get_current_datetime"


def test_load_tools_unresolvable_junk_reports_invalid_input(db):
    """Unresolvable ids are reported deterministically as invalid inputs."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FakeBigProvider())
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_steps=20),
    )
    chat = ScriptedChat(
        [
            fence(LOAD_TOOLS_NAME, {"ids": ["definitely-not-a-tool"]}),
            "done",
        ]
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=config,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    (load_result,) = tool_results
    assert load_result["result"] == "ERROR: invalid tool ids: definitely-not-a-tool"


def test_load_tools_bounds_invalid_only_diagnostic_that_does_not_fit(db, monkeypatch):
    huge_id = "missing:" + ("x" * 500)
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_steps=20),
    )
    chat = ScriptedChat([fence(LOAD_TOOLS_NAME, {"ids": [huge_id]}), "done"])
    service = AgentService(db=db, registry=registry, chat_call=chat)

    def result_fits(_config, _endpoint, request):
        return len(str(request.messages[-1].get("content", ""))) < 100

    monkeypatch.setattr(service, "_project_instruction_request_fits", result_fits)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        config=config,
        api_endpoint="llama_cpp",
        first_request_schema_plan=_forced_discovery_plan(),
    )

    assert outcome.status == RUN_DONE
    load_result = [
        step for step in db.get_run(run_id)["steps"] if step["kind"] == "tool_result"
    ][0]
    assert load_result["result"] == (
        "ERROR: tool selection details omitted because the request budget is exhausted"
    )
    assert huge_id not in load_result["result"]


def test_idless_native_call_gets_synthesized_id_pairing_echo_and_result(db):
    """PR #648 review: some OpenAI-compatible servers omit tool-call ids. A
    synthesized id must appear identically in the assistant echo and the
    role='tool' reply, so history pairing never splits conventions."""
    idless = {
        "type": "function",
        "function": {
            "name": "calculator",
            "arguments": json.dumps({"expression": "2+2"}),
        },
    }
    service, chat = make_service(db, [{"content": None, "tool_calls": [idless]}, "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=CFG,
        api_endpoint="groq",
        should_cancel=lambda: False,
    )
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    second_payload = chat.calls[1]["messages_payload"]
    assistant = [m for m in second_payload if m["role"] == "assistant"][0]
    tool_msg = [m for m in second_payload if m.get("role") == "tool"][0]
    assert assistant["tool_calls"][0]["id"] == "call_0"
    assert tool_msg["tool_call_id"] == "call_0"
    assert not any(
        m.get("role") == "user"
        and str(m.get("content", "")).startswith("Tool result for")
        for m in second_payload[1:]
    )


def test_load_tools_same_batch_name_and_id_aliases_load_once(db):
    """PR #655 review (Gemini): one load_tools batch naming the SAME tool
    twice — bare name plus catalog id — must disclose it exactly once, so
    the loop's active list and this gate's disclosed set stay in lockstep
    (no duplicate schema, no phantom room-slot consumption)."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FakeBigProvider())  # schema cost forces discovery
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator", "get_current_datetime"),
        budget=RunBudget(max_steps=20),
    )
    chat = ScriptedChat(
        [
            fence(LOAD_TOOLS_NAME, {"ids": ["calculator", "builtin:calculator"]}),
            fence("calculator", {"expression": "2+2"}),
            "4.",
        ]
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=config,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    run = db.get_run(run_id)
    load_result = [s for s in run["steps"] if s["kind"] == "tool_result"][0]
    # Exactly one mention: "loaded: calculator" — not "calculator, calculator".
    assert load_result["result"] == "loaded: calculator"


# --- task-245: memoize the per-run fence-protocol render so an unchanged
# active tool set is rendered once, not once per model turn. ---


def test_protocol_render_memoized_across_unchanged_turns(db, monkeypatch):
    """AC #1: three fence turns with an unchanged active set must render the
    protocol exactly once; the payload text stays byte-identical per turn."""
    import tldw_chatbook.Agents.agent_service as svc

    real_render = svc.render_tool_protocol
    calls = []

    def counting_render(schemas):
        calls.append(tuple(s.name for s in schemas))
        return real_render(schemas)

    monkeypatch.setattr(svc, "render_tool_protocol", counting_render)
    # script: two calculator fence rounds + final answer = 3 model turns,
    # active set never changes (direct-disclose catalog, no load_tools).
    service, chat = make_service(
        db,
        [
            fence("calculator", {"expression": "1+1"}),
            fence("calculator", {"expression": "2+2"}),
            "done",
        ],
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
        first_request_schema_plan=FirstRequestSchemaPlan(
            active_schemas=(service.registry.load_schema("builtin:calculator"),),
            runtime_schemas=(),
            offer_find_load=False,
            log_active=False,
            system_prompt=CFG.system_prompt,
        ),
    )
    assert outcome.status == RUN_DONE
    assert len(calls) == 1  # rendered once, reused twice
    first_system = chat.calls[0]["messages_payload"][0]["content"]
    for later in chat.calls[1:]:
        assert later["messages_payload"][0]["content"] == first_system  # byte-stable


def test_protocol_rerenders_when_load_tools_admits_new_schema(db):
    """AC #2: the cache invalidates the moment load_tools grows the active
    set — the very next turn's protocol includes the new tool."""
    # Mirror the file's existing find/load setup (FakeBigProvider's complete
    # schema payload selects discovery).
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FakeBigProvider())  # schema cost forces discovery
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator", "get_current_datetime"),
        budget=RunBudget(max_steps=20),
    )
    chat = ScriptedChat(
        [
            fence(LOAD_TOOLS_NAME, {"ids": ["calculator"]}),
            fence("calculator", {"expression": "2+2"}),
            "done",
        ]
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=config,
        api_endpoint="llama_cpp",
        first_request_schema_plan=_forced_discovery_plan(),
    )

    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    pre_load_system = chat.calls[0]["messages_payload"][0]["content"]
    post_load_system = chat.calls[1]["messages_payload"][0]["content"]
    assert "calculator" not in pre_load_system
    assert "calculator" in post_load_system


def test_protocol_rerenders_when_replacement_changes_same_named_schema(
    db, monkeypatch
):
    """Replacing a tool definition under the same name invalidates the fence."""

    class MutableSchemaProvider:
        def __init__(self):
            self.loads = 0

        def list_catalog(self):
            return [
                ToolCatalogEntry(
                    id="mutable:tool",
                    name="mutable_tool",
                    one_line_description="mutable schema",
                    source="mutable",
                )
            ]

        def load_schema(self, tool_id):
            self.loads += 1
            parameter = "first_argument" if self.loads == 1 else "second_argument"
            return ToolSchema(
                id=tool_id,
                name="mutable_tool",
                description="mutable schema",
                parameters={
                    "type": "object",
                    "properties": {parameter: {"type": "string"}},
                },
            )

        def invoke(self, tool_id, args):
            return ToolResult(ok=True, content=tool_id)

    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a: 100_000)
    monkeypatch.setattr(agent_service, "_count_model_messages", lambda *_a, **_k: 1)
    registry = ToolCatalogRegistry()
    registry.register_provider(MutableSchemaProvider())
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("mutable_tool",),
        budget=RunBudget(max_steps=20),
    )
    chat = ScriptedChat(
        [
            fence(LOAD_TOOLS_NAME, {"ids": ["mutable:tool"]}),
            fence(LOAD_TOOLS_NAME, {"ids": ["mutable:tool"]}),
            "done",
        ]
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="llama_cpp",
        first_request_schema_plan=_forced_discovery_plan(),
    )

    assert outcome.status == RUN_DONE
    first_loaded_system = chat.calls[1]["messages_payload"][0]["content"]
    replacement_system = chat.calls[2]["messages_payload"][0]["content"]
    assert "first_argument" in first_loaded_system
    assert "second_argument" in replacement_system
    assert "first_argument" not in replacement_system


def test_native_endpoint_anthropic_sends_tools_and_suppresses_fence(db):
    """task-263: anthropic is native-capable — the service passes tools= and
    suppresses the fence protocol exactly as for the OpenAI-compatible set
    (the handler converts shapes internally; live-gated 2026-07-17)."""
    service, chat = make_service(
        db,
        [
            {
                "content": None,
                "tool_calls": [
                    native_call("calculator", {"expression": "2+2"}, "toolu_1")
                ],
            },
            "4.",
        ],
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=CFG,
        api_endpoint="anthropic",
        should_cancel=lambda: False,
    )
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    first = chat.calls[0]
    assert "tools" in first
    assert "tool_call" not in first["messages_payload"][0]["content"]
    tool_msg = [
        m for m in chat.calls[1]["messages_payload"] if m.get("role") == "tool"
    ][0]
    assert tool_msg["tool_call_id"] == "toolu_1"


def test_native_endpoint_google_sends_tools_and_suppresses_fence(db):
    """task-266: google is native-capable — tools= passed, fence suppressed
    (the handler converts shapes internally; live-gated 2026-07-17)."""
    service, chat = make_service(
        db,
        [
            {
                "content": None,
                "tool_calls": [
                    native_call("calculator", {"expression": "2+2"}, "call_g1")
                ],
            },
            "4.",
        ],
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=CFG,
        api_endpoint="google",
        should_cancel=lambda: False,
    )
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    assert "tools" in chat.calls[0]
    assert "tool_call" not in chat.calls[0]["messages_payload"][0]["content"]


def test_native_endpoint_cohere_sends_tools_and_suppresses_fence(db):
    """task-267: cohere is a real NATIVE_TOOLS_PROVIDERS member (flipped
    after the 2026-07-17 live gate, Docs/superpowers/qa/cohere-native-
    2026-07/) — the service sends native tools and suppresses the fence
    protocol, mirroring the anthropic/google siblings above."""
    service, chat = make_service(
        db,
        [
            {
                "content": None,
                "tool_calls": [
                    native_call("calculator", {"expression": "2+2"}, "call_c1")
                ],
            },
            "4.",
        ],
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "2+2?"}],
        config=CFG,
        api_endpoint="cohere",
        should_cancel=lambda: False,
    )
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    first = chat.calls[0]
    assert "tools" in first
    assert "tool_call" not in first["messages_payload"][0]["content"]
    tool_msg = [
        m for m in chat.calls[1]["messages_payload"] if m.get("role") == "tool"
    ][0]
    assert tool_msg["tool_call_id"] == "call_c1"


def test_native_endpoint_with_no_schemas_omits_tools_kwarg(db):
    """Review minor m3 (task-243 final review): a native-capable endpoint
    whose run has NO disclosable schemas (empty allow-list, no sub-agents)
    must call the provider with no tools= kwarg at all — an empty tools
    list is rejected by several providers."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(),
        budget=RunBudget(max_subagents=0),
    )
    chat = ScriptedChat(["Just an answer."])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "hi"}],
        config=config,
        api_endpoint="groq",
        should_cancel=lambda: False,
    )
    assert outcome.status == RUN_DONE
    assert "tools" not in chat.calls[0]


def test_usage_total_tokens_reads_total():
    assert _usage_total_tokens({"usage": {"total_tokens": 150}}) == 150


def test_usage_total_tokens_sums_prompt_and_completion():
    assert _usage_total_tokens(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    ) == 150


def test_usage_total_tokens_none_when_absent_or_malformed():
    assert _usage_total_tokens({"choices": []}) is None
    assert _usage_total_tokens("a string") is None
    assert _usage_total_tokens({"usage": "bad"}) is None
    assert _usage_total_tokens({"usage": {"prompt_tokens": 10}}) is None
    # Malformed values must not corrupt spend accounting (Qodo review):
    assert _usage_total_tokens({"usage": {"total_tokens": True}}) is None  # bool
    assert _usage_total_tokens({"usage": {"total_tokens": 0}}) is None
    assert _usage_total_tokens({"usage": {"total_tokens": -7}}) is None
    assert _usage_total_tokens(
        {"usage": {"prompt_tokens": -5, "completion_tokens": 10}}
    ) is None
    assert _usage_total_tokens(
        {"usage": {"prompt_tokens": False, "completion_tokens": 5}}
    ) is None
    assert _usage_total_tokens(
        {"usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    ) is None
    # Valid non-negative sum still works.
    assert _usage_total_tokens(
        {"usage": {"prompt_tokens": 0, "completion_tokens": 5}}
    ) == 5


def _service_with_chat(db, chat_call):
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    return AgentService(db=db, registry=registry, chat_call=chat_call)


def test_call_model_uses_real_provider_usage(db):
    def chat(**kwargs):
        return {
            "choices": [{"message": {"content": "hello there"}}],
            "usage": {"total_tokens": 150},
        }
    service = _service_with_chat(db, chat)
    cfg = AgentConfig(model="gpt-4o", system_prompt="s", native_tools=False)
    call_model = service._make_call_model(cfg, "openai", [])
    turn = call_model([{"role": "user", "content": "hi"}], ())
    assert turn.tokens == 150


def test_call_model_estimates_when_no_usage(db):
    def chat(**kwargs):
        return {"choices": [{"message": {"content": "hello there world"}}]}
    service = _service_with_chat(db, chat)
    cfg = AgentConfig(model="gpt-4o", system_prompt="s", native_tools=False)
    call_model = service._make_call_model(cfg, "openai", [])
    turn = call_model([{"role": "user", "content": "count these tokens"}], ())
    # No provider usage -> estimate of sent payload + response text, always > 0.
    assert turn.tokens > 0


def test_call_model_estimate_strips_provider_prefix(db):
    # A provider-qualified id (openai/gpt-4o-mini) must be normalized for token
    # counting so the GPT framing overhead applies -> same estimate as the bare
    # model. (Qodo review: prefixed models otherwise undercount.)
    def chat(**kwargs):
        return {"choices": [{"message": {"content": "hello there world"}}]}
    service = _service_with_chat(db, chat)
    msgs = [{"role": "user", "content": "count these tokens please"}]
    prefixed = service._make_call_model(
        AgentConfig(model="openai/gpt-4o-mini", system_prompt="s", native_tools=False),
        "openai", [],
    )(msgs, ())
    bare = service._make_call_model(
        AgentConfig(model="gpt-4o-mini", system_prompt="s", native_tools=False),
        "openai", [],
    )(msgs, ())
    assert prefixed.tokens == bare.tokens
    assert prefixed.tokens > 0


def test_call_model_native_path_reports_provider_tokens(db):
    """Native tool-call return path (turn.tool_calls set) must also report
    real provider usage on .tokens -- the non-native tests above only cover
    the early ``if not native: return ModelTurn(...)`` branch."""

    def chat(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [native_call("calculator", {"expression": "2+2"})],
                    }
                }
            ],
            "usage": {"total_tokens": 77},
        }

    service = _service_with_chat(db, chat)
    cfg = AgentConfig(model="gpt-4o", system_prompt="s", native_tools=True)
    call_model = service._make_call_model(cfg, "openai", [])
    turn = call_model([{"role": "user", "content": "2+2?"}], ())
    assert turn.tokens == 77
    assert turn.tool_calls


@pytest.mark.parametrize("canonical_raw", ['{ "b": 2, "a": 1 }', '{"a":1,"b":2}'])
def test_service_native_turn_reaches_batch_barrier_only_with_exact_raw_arguments(
    db, canonical_raw
):
    raw_arguments = '{ "b": 2, "a": 1 }'

    def chat(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "exact",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": raw_arguments,
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"total_tokens": 1},
        }

    config = AgentConfig(model="gpt-4o", system_prompt="s", native_tools=True)
    turn = _service_with_chat(db, chat)._make_call_model(config, "openai", [])([], ())
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
        state="active",
        rounds=(
            ContinuationRound(
                "",
                ("private",),
                (ContinuationCall("exact", "calculator", canonical_raw, "pending"),),
            ),
        ),
    )
    turn = dataclasses.replace(turn, provider_continuation=checkpoint)
    events = []
    deps = LoopDeps(
        call_model=lambda messages, active: turn,
        invoke_tool=lambda call: ToolResult(ok=True, content="ok"),
        spawn=lambda task: ToolResult(ok=True),
        find_tools=lambda query: [],
        load_schemas=lambda _ids, _messages, _call: ToolLoadSelection(),
        should_cancel=lambda: len(events) >= 3,
        clock=lambda: 0.0,
        continuation_context=ContinuationEventContext(
            "owner", "run", "primary", "persistent"
        ),
        persist_provider_continuation=events.append,
    )
    outcome = run_agent_loop(config, [], [], deps)

    assert any(isinstance(event, ToolBatchReady) for event in events) is (
        canonical_raw == raw_arguments
    )
    assert outcome.status == ("cancelled" if canonical_raw == raw_arguments else "error")


# task-327 (AC#4): per-tool-call timeout, enforced entirely in this impure
# seam via a module-level helper. `agent_runtime.run_agent_loop` and
# `LoopDeps.invoke_tool`'s type are unaffected -- the wrapping happens here,
# around the builtin/custom registry.invoke_by_name path only.


def test_call_with_timeout_returns_result_when_fast():
    out = _call_with_timeout(lambda: ToolResult(ok=True, content="hi"), 5.0, "fast_tool")
    assert out.ok and out.content == "hi"


def test_call_with_timeout_trips_on_slow_call():
    def slow():
        time.sleep(2.0)
        return ToolResult(ok=True, content="late")
    t0 = time.monotonic()
    out = _call_with_timeout(slow, 0.2, "slow_tool")
    # Bounds the wrapper's own wall-clock: a future "cleanup" that added a
    # blocking worker.join() before returning (defeating the timeout) would
    # fail this assertion instead of just making the test slow.
    assert time.monotonic() - t0 < 1.0
    assert out.ok is False
    assert "timed out" in out.error and "slow_tool" in out.error


def test_call_with_timeout_wraps_exception():
    def boom():
        raise ValueError("kaboom")
    out = _call_with_timeout(boom, 5.0, "bad_tool")
    assert out.ok is False and "kaboom" in out.error


def test_call_with_timeout_wraps_base_exception():
    """_runner only caught Exception; a BaseException (asyncio.CancelledError,
    SystemExit -- both reachable in-repo, see BuiltinToolProvider.invoke's
    asyncio.run() and any tool wrapping argparse) left neither box key set,
    so `return box["result"]` raised KeyError out of invoke_tool into the
    pure loop instead of returning a failed ToolResult. Must not regress."""
    def boom():
        raise SystemExit("bye")
    out = _call_with_timeout(boom, 5.0, "exiting_tool")
    assert out.ok is False and "bye" in out.error


def test_call_with_timeout_polls_cancellation_promptly():
    """A blocking tool call must not wedge Stop for the full timeout ceiling.
    While a tool is hung, run_agent_loop's own should_cancel() checks at
    step/tool-call boundaries are unreachable -- this wait is the only place
    that can observe a Stop until the call finishes or times out, so it must
    poll should_cancel() itself rather than a single blocking join(seconds).
    """
    def slow():
        time.sleep(2.0)
        return ToolResult(ok=True, content="too late")

    t0 = time.monotonic()
    out = _call_with_timeout(slow, 5.0, "slow_tool", should_cancel=lambda: True)
    elapsed = time.monotonic() - t0
    assert out.ok is False
    assert "cancelled" in out.error and "slow_tool" in out.error
    # Well under the 5.0s timeout ceiling -- proves cancellation actually
    # short-circuits the wait instead of merely being checked after it.
    assert elapsed < 1.5


def test_make_invoke_tool_wraps_slow_custom_tool_cancellable(db, monkeypatch):
    """The should_cancel seam threaded through _make_invoke_tool (the
    production wiring in _run_one) must let a Stop during a hung tool call
    return promptly instead of waiting out max_tool_call_seconds."""
    def chat(**kwargs):  # pragma: no cover - unused by this test
        return {"choices": [{"message": {"content": "unused"}}]}

    service = _service_with_chat(db, chat)

    def slow_invoke_by_name(name, args):
        time.sleep(2.0)
        return ToolResult(ok=True, content="too late")

    monkeypatch.setattr(service.registry, "invoke_by_name", slow_invoke_by_name)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_tool_call_seconds=5.0),
    )
    # PR2a Task 5: `run_id` is keyword-REQUIRED -- it binds the dispatching
    # run for the permission gates (run_context), and a default would let a
    # caller silently lose every approval stamp instead of failing loudly.
    invoke_tool = service._make_invoke_tool(
        cfg,
        disclosed_names={"calculator"},
        should_cancel=lambda: True,
        run_id="run-1",
    )
    from tldw_chatbook.Agents.agent_models import ToolCall
    t0 = time.monotonic()
    result = invoke_tool(ToolCall(name="calculator", args={"expression": "2+2"}))
    elapsed = time.monotonic() - t0
    assert result.ok is False
    assert "cancelled" in result.error and "calculator" in result.error
    assert elapsed < 1.5


def test_make_invoke_tool_bypasses_wrapper_when_unlimited(db, monkeypatch):
    """max_tool_call_seconds=0 must skip _call_with_timeout entirely and
    call straight through to the registry -- a real closure-level test, not
    just a check of the branch condition. Without the monkeypatch below this
    test would pass identically whether or not the wrapper is used, since a
    fast call looks the same either way -- the monkeypatch makes it actually
    prove the bypass by failing loudly if the wrapper is invoked at all."""
    def chat(**kwargs):  # pragma: no cover - unused by this test
        return {"choices": [{"message": {"content": "unused"}}]}

    monkeypatch.setattr(
        agent_service,
        "_call_with_timeout",
        lambda *a, **k: pytest.fail("wrapper used on the 0 path"),
    )
    service = _service_with_chat(db, chat)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_tool_call_seconds=0),
    )
    invoke_tool = service._make_invoke_tool(
        cfg, disclosed_names={"calculator"}, run_id="run-1"
    )
    from tldw_chatbook.Agents.agent_models import ToolCall
    result = invoke_tool(ToolCall(name="calculator", args={"expression": "2+2"}))
    assert result.ok is True
    assert json.loads(result.content)["result"] == 4


def test_make_invoke_tool_binds_exact_subagent_actor_on_timeout_thread(
    db, monkeypatch
):
    """Provider capture sees child and parent identity on the actual tool thread."""
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.run_context import (
        CurrentRunActor,
        current_run_actor,
    )

    service = _service_with_chat(db, lambda **_kwargs: provider_reply("unused"))
    seen = []

    def invoke_by_name(_name, _args):
        seen.append(current_run_actor())
        return ToolResult(ok=True, content="ok")

    monkeypatch.setattr(service.registry, "invoke_by_name", invoke_by_name)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_tool_call_seconds=1.0),
    )
    actor = CurrentRunActor("subagent", "run-child", "run-parent")
    invoke_tool = service._make_invoke_tool(
        cfg,
        disclosed_names={"calculator"},
        run_id=actor.run_id,
        run_actor=actor,
    )

    assert invoke_tool(ToolCall(name="calculator", args={})).ok is True
    assert seen == [actor]
    assert current_run_actor() is None


def test_make_invoke_tool_binds_native_call_id_on_tool_thread(db, monkeypatch):
    """Providers can scope one approval verdict to one native tool call."""
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.run_context import current_tool_call_id

    service = _service_with_chat(db, lambda **_kwargs: provider_reply("unused"))
    seen = []

    def invoke_by_name(_name, _args):
        seen.append(current_tool_call_id())
        return ToolResult(ok=True, content="ok")

    monkeypatch.setattr(service.registry, "invoke_by_name", invoke_by_name)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_tool_call_seconds=1.0),
    )
    invoke_tool = service._make_invoke_tool(
        cfg,
        disclosed_names={"calculator"},
        run_id="run-1",
    )

    assert invoke_tool(ToolCall("calculator", {}, "native-call-7")).ok is True
    assert seen == ["native-call-7"]
    assert current_tool_call_id() == ""


def test_make_invoke_tool_wraps_slow_custom_tool_in_timeout(db, monkeypatch):
    """A blocking custom tool provider must not wedge the run past
    max_tool_call_seconds -- the boundary this task exists to add."""
    def chat(**kwargs):  # pragma: no cover - unused by this test
        return {"choices": [{"message": {"content": "unused"}}]}

    service = _service_with_chat(db, chat)

    def slow_invoke_by_name(name, args):
        time.sleep(2.0)
        return ToolResult(ok=True, content="too late")

    monkeypatch.setattr(service.registry, "invoke_by_name", slow_invoke_by_name)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_tool_call_seconds=0.2),
    )
    invoke_tool = service._make_invoke_tool(
        cfg, disclosed_names={"calculator"}, run_id="run-1"
    )
    from tldw_chatbook.Agents.agent_models import ToolCall
    result = invoke_tool(ToolCall(name="calculator", args={"expression": "2+2"}))
    assert result.ok is False
    assert "timed out" in result.error and "calculator" in result.error


def test_make_invoke_tool_waits_for_definitive_watchlists_mutation(
    db, tmp_path
):
    """Once an approved definitive mutation starts, neither a tiny runtime
    budget nor cancellation may return before its transaction outcome."""
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.MCP.permission_store import EffectiveToolState
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
        WatchlistBundleService,
    )
    from tldw_chatbook.Tools.watchlists_command_service import (
        WatchlistsCommandService,
    )

    subscriptions = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    bundles = WatchlistBundleService(subscriptions)
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()

    def delayed_create(**kwargs):
        entered.set()
        if not release.wait(2):
            raise AssertionError("mutation gate was never released")
        return bundles.create_with_sources(**kwargs)

    def unavailable(*_args, **_kwargs):
        return None
    commands = WatchlistsCommandService(
        runtime_source_loader=lambda: "local",
        create_sources_batch=unavailable,
        create_collection=delayed_create,
        update_collection_sources=unavailable,
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(
        LocalToolProvider(
            workspace_root=tmp_path,
            watchlists_command_service=commands,
            resolve_state=lambda _tool: EffectiveToolState(
                state="allow", origin="tool_override"
            ),
        )
    )
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=lambda **_kwargs: provider_reply("unused"),
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("watchlists_create_collection",),
        budget=RunBudget(max_tool_call_seconds=0.02),
    )
    invoke_tool = service._make_invoke_tool(
        config,
        disclosed_names={"watchlists_create_collection"},
        should_cancel=cancelled.is_set,
        run_id="run-definitive",
    )
    result_box: dict[str, ToolResult] = {}

    worker = threading.Thread(
        target=lambda: result_box.setdefault(
            "result",
            invoke_tool(
                ToolCall(
                    name="watchlists_create_collection",
                    args={"name": "Threat intel", "if_exists": "auto_suffix"},
                )
            ),
        )
    )
    worker.start()
    assert entered.wait(2), "mutation never started"
    cancelled.set()
    time.sleep(0.1)
    still_waiting = worker.is_alive()
    before_release = bundles.list_watchlists()
    release.set()
    worker.join(2)

    assert still_waiting, "the runtime returned while the mutation could still commit"
    assert before_release == []
    assert not worker.is_alive()
    result = result_box["result"]
    assert result.ok is True
    assert json.loads(result.content)["status"] == "ok"
    assert [row["name"] for row in bundles.list_watchlists()] == ["Threat intel"]


def test_make_invoke_tool_cancels_definitive_call_before_start(db, monkeypatch):
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy

    service = _service_with_chat(db, lambda **_kwargs: provider_reply("unused"))
    invoked: list[str] = []
    monkeypatch.setattr(
        service.registry,
        "execution_policy_for",
        lambda _name: ToolExecutionPolicy.DEFINITIVE_AFTER_START,
    )
    monkeypatch.setattr(
        service.registry,
        "invoke_by_name",
        lambda name, _args: invoked.append(name) or ToolResult(ok=True, content="late"),
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_tool_call_seconds=0.02),
    )
    invoke_tool = service._make_invoke_tool(
        config,
        disclosed_names={"calculator"},
        should_cancel=lambda: True,
        run_id="run-cancel-before-start",
    )

    result = invoke_tool(ToolCall(name="calculator", args={"expression": "2+2"}))

    assert result.ok is False
    assert result.outcome == "cancelled"
    assert invoked == []


def test_definitive_result_observer_receives_structured_success_and_failure_only(
    db, monkeypatch
):
    """Receipt capture observes results without changing the legacy callback."""
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy

    observed: list[tuple[str, str, str, ToolResult]] = []
    legacy: list[tuple[str, str, str]] = []
    service = AgentService(
        db=db,
        registry=ToolCatalogRegistry(),
        chat_call=lambda **_kwargs: provider_reply("unused"),
        on_tool_terminal=lambda *args: legacy.append(args),
        on_tool_result_terminal=lambda *args: observed.append(args),
    )
    monkeypatch.setattr(
        service.registry,
        "execution_policy_for",
        lambda name: (
            ToolExecutionPolicy.DEFINITIVE_AFTER_START
            if name == "watchlists_check_sources"
            else ToolExecutionPolicy.BOUNDED_ABANDONABLE
        ),
    )
    results = iter(
        (
            ToolResult(ok=True, content='{"status":"accepted"}'),
            ToolResult(ok=False, error="safe failure"),
            ToolResult(ok=True, content="ordinary"),
        )
    )
    monkeypatch.setattr(
        service.registry, "invoke_by_name", lambda _name, _args: next(results)
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("watchlists_check_sources", "calculator"),
    )
    invoke = service._make_invoke_tool(
        config,
        disclosed_names={"watchlists_check_sources", "calculator"},
        run_id="run-receipt",
    )

    success = invoke(
        ToolCall(name="watchlists_check_sources", args={}, call_id="success")
    )
    failure = invoke(
        ToolCall(name="watchlists_check_sources", args={}, call_id="failure")
    )
    invoke(ToolCall(name="calculator", args={}, call_id="ordinary"))

    assert [row[:3] for row in observed] == [
        ("run-receipt", "success", "watchlists_check_sources"),
        ("run-receipt", "failure", "watchlists_check_sources"),
    ]
    assert [row[3] for row in observed] == [success, failure]
    assert legacy == [
        ("run-receipt", "success", "watchlists_check_sources"),
        ("run-receipt", "failure", "watchlists_check_sources"),
    ]


@pytest.mark.parametrize(
    "failure_factory",
    [
        lambda: SystemExit("secret system-exit detail"),
        lambda: asyncio.CancelledError("secret cancellation detail"),
        lambda: type("FatalToolFailure", (BaseException,), {})(
            "secret base-exception detail"
        ),
    ],
    ids=("system-exit", "cancelled-error-after-start", "custom-base-exception"),
)
def test_make_invoke_tool_scrubs_every_definitive_provider_terminal(
    db, tmp_path, failure_factory
):
    """An approved definitive tool has one never-raise terminal contract.

    These failures occur after dispatch has started.  They therefore must
    not escape as process/control-flow exceptions, and a ``CancelledError``
    here must not be confused with the pre-start cooperative-cancel result.
    """
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.MCP.permission_store import EffectiveToolState
    from tldw_chatbook.Tools.watchlists_command_service import (
        WatchlistsCommandService,
    )

    def crash(**_kwargs):
        raise failure_factory()

    def unavailable(*_args, **_kwargs):
        return None

    commands = WatchlistsCommandService(
        runtime_source_loader=lambda: "local",
        create_sources_batch=unavailable,
        create_collection=crash,
        update_collection_sources=unavailable,
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(
        LocalToolProvider(
            workspace_root=tmp_path,
            watchlists_command_service=commands,
            resolve_state=lambda _tool: EffectiveToolState(
                state="allow", origin="tool_override"
            ),
        )
    )
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=lambda **_kwargs: provider_reply("unused"),
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("watchlists_create_collection",),
        budget=RunBudget(max_tool_call_seconds=0.001),
    )
    invoke_tool = service._make_invoke_tool(
        config,
        disclosed_names={"watchlists_create_collection"},
        should_cancel=lambda: False,
        run_id="run-definitive-crash",
    )

    result = invoke_tool(
        ToolCall(
            name="watchlists_create_collection",
            args={"name": "Threat intel", "if_exists": "auto_suffix"},
            call_id="call-crash",
        )
    )

    assert result.ok is False
    assert result.error == "tool call failed: watchlists_create_collection"
    assert "secret" not in result.error
    assert result.outcome != "cancelled"


def test_registry_timeout_for_reports_a_tools_own_ceiling():
    from tldw_chatbook.Tools.tool_executor import Tool

    class _Slow(Tool):
        @property
        def name(self) -> str:
            return "slow_thing"

        @property
        def description(self) -> str:
            return "d"

        @property
        def parameters(self) -> dict:
            return {"type": "object", "properties": {}}

        @property
        def timeout_seconds(self) -> float:
            return 42.0

        async def execute(self, **kwargs):
            return {}

    provider = BuiltinToolProvider()
    provider._tools["slow_thing"] = _Slow()
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)

    assert registry.timeout_for("slow_thing") == 42.0
    assert registry.timeout_for("calculator") is None
    assert registry.timeout_for("no_such_tool") is None


RESEARCHER_DEFN = AgentDefinition(
    name="researcher",
    description="Searches and summarizes.",
    instructions="Always cite sources in your result.",
    tool_allowlist=("calculator",),
)


def _seed_definition(db, defn=RESEARCHER_DEFN):
    db.create_agent_definition(defn)


def test_named_spawn_appends_instructions_and_keeps_identity_prefix(db):
    _seed_definition(db)
    # PR2a Task 6.5: addressed script (the child is on its own thread).
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "compute 6*7", "agent": "researcher"}),
            "done",
        ],
        {"compute 6*7": ["sub answer: 42"]},
    )
    service = _service_with(db, chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "delegate"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    assert outcome.status == RUN_DONE
    child_system = chat.child_calls["compute 6*7"][0]["messages_payload"][0]
    assert child_system["role"] == "system"
    # IDENTITY CONTRACT: base subagent prompt stays the PREFIX
    # (console_agent_bridge._is_subagent prefix-matches it) ...
    assert child_system["content"].startswith(SUBAGENT_SYSTEM_PROMPT.split(".")[0])
    # ... and the definition's instructions are appended after it.
    assert "Always cite sources" in child_system["content"]


def test_named_spawn_intersects_allowlist_never_grants(db):
    _seed_definition(
        db,
        AgentDefinition(
            name="narrow",
            instructions="Do the task.",
            # calculator is in the parent set; forbidden_tool is not — the
            # definition can narrow to calculator but never grant extras.
            tool_allowlist=("calculator", "forbidden_tool"),
        ),
    )
    # PR2a Task 6.5: addressed script (the child is on its own thread).
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "t", "agent": "narrow"}),
            "done",
        ],
        {"t": ["child done"]},
    )
    service = _service_with(db, chat)
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    # Channel-agnostic: disclosed schemas may ride the system prompt
    # (fence protocol) OR the tools= kwarg (native) — inspect the whole
    # provider call.
    child_call = json.dumps(chat.child_calls["t"][0], default=str)
    assert "calculator" in child_call
    assert "get_current_datetime" not in child_call  # narrowed away
    assert "forbidden_tool" not in child_call  # never granted


def test_named_spawn_model_override_same_endpoint(db):
    _seed_definition(
        db,
        AgentDefinition(
            name="cheap", instructions="Do it.", model="tiny-model"
        ),
    )
    # PR2a Task 6.5: addressed script (the child is on its own thread).
    chat = FleetChat(
        [fence(SPAWN_TOOL_NAME, {"task": "t", "agent": "cheap"}), "done"],
        {"t": ["ok"]},
    )
    service = _service_with(db, chat)
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    child_call = chat.child_calls["t"][0]
    assert child_call["model"] == "tiny-model"
    assert chat.parent_calls[0]["model"] == "test-model"
    assert child_call["api_endpoint"] == "llama_cpp"


def test_unknown_agent_refused_without_burning_budget(db):
    _seed_definition(db)
    # PR2a Task 6.5: addressed script -- both children are on their own
    # threads, so their two answers are no longer interleaved into the
    # parent's queue at fixed positions.
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "t", "agent": "nope"}),
            fence(SPAWN_TOOL_NAME, {"task": "t2", "agent": "researcher"}),
            fence(SPAWN_TOOL_NAME, {"task": "t3", "agent": "researcher"}),
            "done",
        ],
        {"t2": ["child ok"], "t3": ["child ok 2"]},
    )
    service = _service_with(db, chat)
    # 3 parent-level spawn rounds (1 refused + 2 real) + the final answer =
    # 10 parent steps; CFG's default max_steps=8 would trip stuck before
    # the model ever gets to answer -- raise it, mirroring
    # test_spawn_result_and_budget's identical fix in test_agent_runtime.py
    # (max_subagents stays at CFG's default of 2, which is what this test
    # is actually about).
    cfg = dataclasses.replace(CFG, budget=RunBudget(max_steps=20))
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    # max_subagents=2: the unknown-agent refusal must not have consumed a
    # slot, so BOTH later spawns succeed.
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c") == 2
    # The refusal itself surfaced the roster to the model -- the PARENT's
    # own second turn, addressed rather than counted (`calls[1]` may now be
    # a child's).
    refusal = chat.parent_calls[1]["messages_payload"]
    assert any(
        "unknown agent 'nope'" in str(m.get("content", "")) for m in refusal
    )


def test_named_spawn_records_audit_fields(db):
    _seed_definition(db)
    service, _ = make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "t", "agent": "researcher"}),
            "child ok",
            "done",
        ],
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    child = next(
        r for r in db.list_runs("c") if r["agent_kind"] == "subagent"
    )
    assert child["agent_definition"] == "researcher"
    assert child["definition_fingerprint"] == definition_fingerprint(
        RESEARCHER_DEFN
    )


def test_definitions_load_once_per_turn_roster_in_protocol(db):
    _seed_definition(db)
    service, chat = make_service(db, ["no tools needed"])
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "hi"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    # The spawn schema (with the roster) must reach the provider call —
    # via the system prompt (fence protocol) or the tools= kwarg (native);
    # inspect the whole call to stay channel-agnostic.
    assert "researcher" in json.dumps(chat.calls[0], default=str)


def test_no_definitions_spawn_unchanged(db):
    # Guard the identity path: with an empty definitions table the primary
    # system prompt must NOT mention an 'agent' parameter.
    service, chat = make_service(db, ["plain answer"])
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "hi"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert '"agent"' not in json.dumps(chat.calls[0], default=str)


# -- ADR-067: pausable per-call deadline ---------------------------------
#
# A blocking human prompt (approval card / skill confirm) can wait inside
# the callable handed to _call_with_timeout. The pre-ADR invariant
# (approval timeout 120s < max_tool_call_seconds 300s) bounded the
# abandoned-thread double-execution hazard by keeping the approval wait
# strictly under the wrapper's ceiling. Indefinite human waits replace
# that with a pausable clock: while a decision is pending for the run,
# the deadline re-arms each poll slice, so wall-clock counts only actual
# tool execution.


def test_call_with_timeout_pauses_deadline_while_predicate_holds():
    """While ``pauses_deadline`` polls True the deadline re-arms, so time
    spent waiting on a human decision inside ``fn`` does not consume the
    tool's execution budget. The 0.3s ceiling would abandon this 1.0s
    call without the pause."""
    def fn():
        time.sleep(1.0)
        return ToolResult(ok=True, content="worth the wait")

    pause_until = time.monotonic() + 1.2
    out = _call_with_timeout(
        fn,
        0.3,
        "paused_tool",
        pauses_deadline=lambda: time.monotonic() < pause_until,
    )
    assert out.ok is True and out.content == "worth the wait"


def test_call_with_timeout_deadline_resumes_after_pause_ends():
    """The pause re-arms the deadline, it does not remove it: once the
    predicate goes False the armed deadline applies again, so a call that
    keeps hanging AFTER its human decision still trips the ceiling
    promptly instead of riding a frozen clock forever."""
    def fn():
        time.sleep(3.0)
        return ToolResult(ok=True, content="too late")

    pause_until = time.monotonic() + 0.7
    t0 = time.monotonic()
    out = _call_with_timeout(
        fn,
        0.3,
        "resumed_tool",
        pauses_deadline=lambda: time.monotonic() < pause_until,
    )
    elapsed = time.monotonic() - t0
    assert out.ok is False and "timed out" in out.error
    assert elapsed < 2.0


def test_make_invoke_tool_pauses_deadline_during_human_input_wait(db, monkeypatch):
    """The ADR-067 production wiring: ``_make_invoke_tool`` feeds
    ``human_input_wait_active(run_id)`` into the wrapper, so a registry
    tool blocked on a human decision (marked from ANY thread via
    ``use_human_input_wait``) outlives max_tool_call_seconds."""
    def chat(**kwargs):  # pragma: no cover - unused by this test
        return {"choices": [{"message": {"content": "unused"}}]}

    service = _service_with_chat(db, chat)

    def slow_invoke_by_name(name, args):
        time.sleep(1.0)
        return ToolResult(ok=True, content="approved then ran")

    monkeypatch.setattr(service.registry, "invoke_by_name", slow_invoke_by_name)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_tool_call_seconds=0.3),
    )
    invoke_tool = service._make_invoke_tool(
        cfg, disclosed_names={"calculator"}, run_id="run-pause-1"
    )
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.human_input_wait import use_human_input_wait

    with use_human_input_wait("run-pause-1"):
        result = invoke_tool(ToolCall(name="calculator", args={"expression": "2+2"}))

    assert result.ok is True
    assert result.content == "approved then ran"
