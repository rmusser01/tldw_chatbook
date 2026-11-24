"""TASK-33620.1.12: the emergency stop holds an in-flight run's tool calls.

Owner decision, 2026-11-23: "while the emergency stop is active, an
in-flight agent run's tool calls (shell and others) are refused with the
stop reason; nothing new executes until the stop is cleared."

Before this, the stop only held NEW work (sends, runs, schedules). A run
already going kept calling tools: the agent's own ``shell_exec`` could raise
an approval card during a stop and run the command once approved.

The hold sits in the agent loop, the one place every tool call of every run
passes on its way to execution, whatever provider owns the tool: once before
a batch is reviewed (so no approval card is raised) and once before each call
executes (so an approval granted after the stop landed, or a later call in
the same batch, is still refused). Each test below drives a REAL provider
through ``AgentService.run_turn`` and counts the provider's own dispatch, not
just the result text (lessons-testing-evidence: "count the dispatch").

The stop is the real one: ``set_emergency_stop`` on
``default_emergency_stop_path()``, the file the app's own gates read.
``AgentService`` reads config on its run path, so these tests keep the
collection-time profile (``bootstrap_profile``); that profile's data directory
is shared by every such test in the process, so the ``stop_path`` fixture
removes the stop after each test whatever happened.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    SPAWN_TOOL_NAME,
    STEP_SPAWN,
    AgentConfig,
    FirstRequestSchemaPlan,
    ModelTurn,
    ToolCall,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT, AgentService
from tldw_chatbook.Agents.local_tool_provider import (
    LocalApprovalEffect,
    LocalToolExposure,
    LocalToolProvider,
    LocalToolSpec,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_controller import (
    build_local_review_hook,
    build_mcp_review_hook,
    build_raw_shell_review_hook,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.raw_cli_executor import RawCliRequest, RawCliResult
from tldw_chatbook.emergency_stop import (
    EMERGENCY_STOP_CLEAR_COMMAND,
    clear_emergency_stop,
    default_emergency_stop_path,
    set_emergency_stop,
)

from Tests.Agents.conftest import pin_max_live_subagents

# The real default stop path: these tests prove the production lookup, so they
# opt out of Tests/conftest.py's per-test redirect.
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.real_emergency_stop_path]

STOP_PREFIX = "tool call blocked: the emergency stop is active"
UNAVAILABLE = (
    "tool call blocked: the emergency stop's state couldn't be read, so tools "
    "are held."
)
ASK = EffectiveToolState(state="ask", origin="global_default")


# --- harness -----------------------------------------------------------------


def _fence(name: str, args: dict) -> str:
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _native_call(name: str, args: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class _Chat:
    """Scripted provider for the primary agent; records every call.

    A sub-agent's call (its system prompt starts with the sub-agent prompt)
    is answered and recorded separately, so a spawn that should have been
    refused shows up as a child call instead of eating the parent's script.
    """

    _CHILD_PREFIX = SUBAGENT_SYSTEM_PROMPT.split(".")[0]

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.child_calls: list[dict] = []

    def __call__(self, **kwargs) -> dict:
        payload = kwargs.get("messages_payload") or []
        system = str(payload[0].get("content", "")) if payload else ""
        if system.startswith(self._CHILD_PREFIX):
            self.child_calls.append(kwargs)
            return {"choices": [{"message": {"content": "child answer"}}]}
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        message = item if isinstance(item, dict) else {"content": item}
        return {"choices": [{"message": message}]}


def _plan(*schemas: ToolSchema) -> FirstRequestSchemaPlan:
    return FirstRequestSchemaPlan(
        active_schemas=tuple(schemas),
        runtime_schemas=(),
        offer_find_load=False,
        log_active=False,
        system_prompt="You are helpful.",
    )


def _config(*allowed: str, native: bool = False) -> AgentConfig:
    return AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=tuple(allowed),
        native_tools=native,
    )


def _run(service, config, plan, *, endpoint="llama_cpp", conversation="estop"):
    return service.run_turn(
        conversation_id=conversation,
        messages=[{"role": "user", "content": "go"}],
        config=config,
        first_request_schema_plan=plan,
        api_endpoint=endpoint,
        should_cancel=lambda: False,
    )


def _results(outcome) -> list:
    return [step for step in outcome.steps if step.kind == "tool_result"]


def _model_saw(chat: _Chat, text: str) -> bool:
    """Whether a later provider request carried ``text`` back to the model."""
    return any(
        text in str(message.get("content", ""))
        for call in chat.calls[1:]
        for message in call["messages_payload"]
    )


def _remove_stop(path: Path) -> None:
    if path.is_dir():
        path.rmdir()  # The unreadable-state case makes it an empty directory.
    else:
        path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def stop_path():
    """The real stop file, resolved the way the app resolves it.

    Removed after the test: a stop left in the shared bootstrap profile
    would hold every later agent tool call in the process.
    """
    path = default_emergency_stop_path()
    assert not path.exists(), f"an earlier test left the emergency stop at {path}"
    try:
        yield path
    finally:
        _remove_stop(path)


@pytest.fixture()
def db(tmp_path) -> AgentRunsDB:
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


# --- the shell tool (the review's finding) -----------------------------------


class _ShellRuntime:
    """The raw-CLI runtime the shell tool drives; records every execution."""

    def __init__(self) -> None:
        self.permitted = True
        self.armed = True
        self.grants: set[str] = set()
        self.execute_calls: list[RawCliRequest] = []
        self.model_authority_revoker = None

    def grant_model_session(self, console_session_id: str) -> None:
        self.grants.add(console_session_id)

    def model_session_granted(self, console_session_id: str) -> bool:
        return console_session_id in self.grants

    def set_model_authority_revoker(self, callback) -> None:
        self.model_authority_revoker = callback

    def execute(self, request: RawCliRequest, on_event, **_kwargs) -> RawCliResult:
        self.execute_calls.append(request)
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell="/bin/bash",
            initial_directory=request.initial_directory,
            elapsed_seconds=0.1,
            stdout_preview="shell ran",
            stderr_preview="",
            record_output="shell ran",
            exit_code=0,
            terminal_state="exited",
            truncated=False,
            cleanup_proven=True,
        )


def _shell_service(tmp_path, db, chat, approve):
    runtime = _ShellRuntime()
    provider = RawShellToolProvider(
        runtime=runtime,
        console_session_id="console-session",
        initial_directory=lambda: tmp_path,
        resolve_state=lambda _hub: ASK,
        local_tools_enabled=lambda: True,
        kill_switch=lambda: False,
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=build_raw_shell_review_hook(provider, approve),
        review_state_scope=provider.stamp_scope,
    )
    return service, provider, runtime


def test_shell_tool_is_refused_without_an_approval_card_while_the_stop_is_active(
    tmp_path, db, stop_path
):
    """The review's finding: a stop must hold the agent's own shell_exec."""
    cards: list = []

    def approve(rows):
        cards.append(rows)
        return {(row.call_id or row.llm_name): "approve_once" for row in rows}

    chat = _Chat([_fence("shell_exec", {"command": "printf hi"}), "Stopped."])
    service, provider, runtime = _shell_service(tmp_path, db, chat, approve)
    set_emergency_stop(stop_path, reason="runaway agent")

    _run_id, outcome = _run(
        service, _config("shell_exec"), _plan(provider.load_schema("shell_exec"))
    )

    assert runtime.execute_calls == [], "the shell command ran during the stop"
    assert cards == [], "an approval card was raised during the stop"
    assert outcome.status == RUN_DONE and outcome.final_text == "Stopped."
    [result] = _results(outcome)
    assert result.tool_name == "shell_exec"
    assert result.tool_outcome == "blocked"
    assert result.result.startswith(f"{STOP_PREFIX} (runaway agent).")
    assert _model_saw(chat, f"{STOP_PREFIX} (runaway agent)")


def test_an_approval_granted_after_the_stop_lands_does_not_run_the_call(
    tmp_path, db, stop_path
):
    """A card already open when the stop lands: Approve must not execute."""
    cards: list = []

    def stop_then_approve(rows):
        # The user runs /emergency-stop while the card is up, then approves.
        cards.append(rows)
        set_emergency_stop(stop_path, reason="typed while the card was open")
        return {(row.call_id or row.llm_name): "approve_once" for row in rows}

    chat = _Chat([_fence("shell_exec", {"command": "printf hi"}), "Held."])
    service, provider, runtime = _shell_service(tmp_path, db, chat, stop_then_approve)

    _run_id, outcome = _run(
        service, _config("shell_exec"), _plan(provider.load_schema("shell_exec"))
    )

    assert len(cards) == 1, "the stop was clear, so the card is raised"
    assert runtime.execute_calls == [], "the approved command ran after the stop"
    assert outcome.status == RUN_DONE
    [result] = _results(outcome)
    assert result.tool_outcome == "blocked"
    assert result.result.startswith(f"{STOP_PREFIX} (typed while the card was open).")
    assert _model_saw(chat, STOP_PREFIX)


# --- a built-in tool -----------------------------------------------------------


def test_builtin_tool_is_refused_while_the_stop_is_active(db, stop_path, monkeypatch):
    invoked: list[str] = []
    original = BuiltinToolProvider.invoke

    def counting_invoke(self, tool_id, args):
        invoked.append(tool_id)
        return original(self, tool_id, args)

    monkeypatch.setattr(BuiltinToolProvider, "invoke", counting_invoke)
    builtin = BuiltinToolProvider()
    registry = ToolCatalogRegistry()
    registry.register_provider(builtin)
    chat = _Chat([_fence("calculator", {"expression": "2+2"}), "Held."])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    set_emergency_stop(stop_path, reason="")

    _run_id, outcome = _run(
        service,
        _config("calculator"),
        _plan(registry.load_schema("builtin:calculator")),
    )

    assert invoked == [], "the built-in tool ran during the stop"
    assert outcome.status == RUN_DONE
    [result] = _results(outcome)
    assert result.tool_outcome == "blocked"
    # No reason typed: the refusal still names the stop and the way out.
    assert result.result == (
        f"{STOP_PREFIX}. No tool runs until the user clears it with "
        f"{EMERGENCY_STOP_CLEAR_COMMAND}."
    )


# --- a local tool --------------------------------------------------------------


def test_local_tool_is_refused_without_an_approval_card_while_the_stop_is_active(
    tmp_path, db, stop_path
):
    handled: list[dict] = []
    spec = LocalToolSpec(
        name="local_probe",
        description="Record that the local tool ran (test only).",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: handled.append(args) or "local ran",
        exposure=LocalToolExposure.CONSOLE_ONLY,
        approval_effects=(LocalApprovalEffect.PRIVATE_READ,),
    )
    provider = LocalToolProvider(
        workspace_root=tmp_path, specs=[spec], resolve_state=lambda _hub: ASK
    )
    cards: list = []

    def approve(rows):
        cards.append(rows)
        return {(row.call_id or row.llm_name): "approve_once" for row in rows}

    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    chat = _Chat([_fence("local_probe", {}), "Held."])
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=build_local_review_hook(provider, approve),
        review_state_scope=provider.stamp_scope,
    )
    set_emergency_stop(stop_path, reason="local hold")

    _run_id, outcome = _run(
        service,
        _config("local_probe"),
        _plan(registry.load_schema(registry.list_catalog()[0].id)),
    )

    assert handled == [], "the local tool ran during the stop"
    assert cards == [], "an approval card was raised during the stop"
    [result] = _results(outcome)
    assert result.tool_outcome == "blocked"
    assert result.result.startswith(f"{STOP_PREFIX} (local hold).")


# --- an MCP tool -----------------------------------------------------------------


class _MCPService:
    """Signature-faithful MCP service seams the provider uses; records execution."""

    def __init__(self) -> None:
        self.local_service = SimpleNamespace(get_inventory=lambda: {"tools": []})
        self.execute_calls: list[tuple] = []

    def get_kill_switch(self) -> bool:
        return False

    async def local_external_catalog(self) -> list[dict]:
        return [
            {
                "profile_id": "srv",
                "is_connected": True,
                "discovery_snapshot": {
                    "tools": [
                        {
                            "name": "remote_probe",
                            "description": "Record that the MCP tool ran.",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                },
            }
        ]

    def effective_tool_states(self, tools):
        return {(tool.server_key, tool.name): ASK for tool in tools}

    def gate_tool_test(self, _tool):
        return ASK

    def is_session_approved(self, _server_key: str, _tool_name: str) -> bool:
        return False

    def approve_for_session(self, _server_key: str, _tool_name: str) -> None:
        return None

    def set_tool_state(self, *_args, **_kwargs) -> None:
        return None

    def record_tool_decision(self, *_args, **_kwargs) -> None:
        return None

    def _tool_call_timeout(self) -> float:
        return 5.0

    async def execute_hub_tool(
        self,
        server_key: str,
        tool_name: str,
        arguments: dict | None = None,
        *,
        initiator: str = "agent",
        decision: str = "allowed",
        timeout_seconds: float | None = None,
        registered_argument_names: set[str] | None = None,
    ) -> dict:
        self.execute_calls.append((server_key, tool_name))
        return {"content": [{"type": "text", "text": "MCP ran"}]}


@pytest.fixture()
def mcp_loop():
    """The MCP execution coroutine runs on its own loop, as in the Console."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(timeout=2)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def test_mcp_tool_is_refused_without_an_approval_card_while_the_stop_is_active(
    db, stop_path, mcp_loop
):
    mcp_service = _MCPService()
    provider = MCPToolProvider(service=mcp_service, main_loop=mcp_loop)
    asyncio.run(provider.compose_catalog())
    [entry] = provider.list_catalog()
    cards: list = []

    def approve(rows):
        cards.append(rows)
        return {row.llm_name: "approve_once" for row in rows}

    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    chat = _Chat([_fence(entry.name, {}), "Held."])
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=build_mcp_review_hook(provider, approve),
        review_state_scope=provider.stamp_scope,
    )
    set_emergency_stop(stop_path, reason="mcp hold")

    _run_id, outcome = _run(
        service, _config(entry.name), _plan(provider.load_schema(entry.id))
    )

    assert mcp_service.execute_calls == [], "the MCP tool ran during the stop"
    assert cards == [], "an approval card was raised during the stop"
    [result] = _results(outcome)
    assert result.tool_outcome == "blocked"
    assert result.result.startswith(f"{STOP_PREFIX} (mcp hold).")


# --- a sub-agent spawn -------------------------------------------------------------


def test_sub_agent_spawn_is_refused_while_the_stop_is_active(
    db, stop_path, monkeypatch
):
    pin_max_live_subagents(monkeypatch, 1)  # inline: a child would run here
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = _Chat([_fence(SPAWN_TOOL_NAME, {"task": "do the thing"}), "Held."])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    set_emergency_stop(stop_path, reason="no children")

    _run_id, outcome = _run(service, _config(SPAWN_TOOL_NAME), _plan())

    assert chat.child_calls == [], "a sub-agent started during the stop"
    assert STEP_SPAWN not in [step.kind for step in outcome.steps]
    [result] = _results(outcome)
    assert result.tool_name == SPAWN_TOOL_NAME
    assert result.tool_outcome == "blocked"
    assert result.result.startswith(f"{STOP_PREFIX} (no children).")


# --- a call already running when the stop lands ------------------------------------


class _ProbeProvider:
    """Two tools; ``first`` sets the stop while it runs, as a user would."""

    source = "test"

    def __init__(self, stop_path: Path) -> None:
        self._stop_path = stop_path
        self.invoked: list[str] = []

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(id=f"test:{n}", name=n, one_line_description=n, source="test")
            for n in ("first", "second")
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        name = tool_id.split(":", 1)[1]
        return ToolSchema(
            id=tool_id,
            name=name,
            description=name,
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = tool_id.split(":", 1)[1]
        self.invoked.append(name)
        if name == "first":
            set_emergency_stop(self._stop_path, reason="landed mid-call")
        return ToolResult(ok=True, content=f"{name} done")


def test_a_call_running_when_the_stop_lands_settles_and_later_calls_are_held(
    db, stop_path
):
    provider = _ProbeProvider(stop_path)
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    chat = _Chat(
        [
            {
                "content": None,
                "tool_calls": [
                    _native_call("first", {}, "a"),
                    _native_call("second", {}, "b"),
                ],
            },
            "Wound down.",
        ]
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)

    _run_id, outcome = _run(
        service,
        _config("first", "second", native=True),
        _plan(provider.load_schema("test:first"), provider.load_schema("test:second")),
        endpoint="openai",
    )

    assert provider.invoked == ["first"], "a call after the stop landed ran"
    assert outcome.status == RUN_DONE and outcome.final_text == "Wound down."
    first, second = _results(outcome)
    # The running call is not left half-done: it settles with its own result.
    assert (first.tool_name, first.tool_outcome, first.result) == (
        "first",
        "success",
        "first done",
    )
    assert second.tool_name == "second" and second.tool_outcome == "blocked"
    assert second.result.startswith(f"{STOP_PREFIX} (landed mid-call).")
    # Both replies reach the model, paired to their calls.
    tool_messages = {
        message["tool_call_id"]: message["content"]
        for message in chat.calls[1]["messages_payload"]
        if message.get("role") == "tool"
    }
    assert tool_messages["a"] == "first done"
    assert tool_messages["b"].startswith(STOP_PREFIX)


# --- fail safe ---------------------------------------------------------------------


def _calculator_run(db, invoked):
    class _Counting(BuiltinToolProvider):
        def invoke(self, tool_id, args):
            invoked.append(tool_id)
            return super().invoke(tool_id, args)

    registry = ToolCatalogRegistry()
    registry.register_provider(_Counting())
    chat = _Chat([_fence("calculator", {"expression": "2+2"}), "Done."])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    return service, registry, chat


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("directory", "stop state unreadable"),
        ("corrupt", "stop state corrupt"),
        ("non_utf8", "stop state unreadable"),
    ],
)
def test_an_unreadable_stop_file_holds_tool_calls(db, stop_path, damage, reason):
    """Doubt holds, exactly as the send and scheduler gates hold on it."""
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    if damage == "directory":
        stop_path.mkdir()
    elif damage == "non_utf8":
        stop_path.write_bytes(b'{"active": \xff\xfe}')
    else:
        stop_path.write_text("{ not json", encoding="utf-8")
    invoked: list[str] = []
    service, registry, _chat = _calculator_run(db, invoked)

    _run_id, outcome = _run(
        service, _config("calculator"), _plan(registry.load_schema("builtin:calculator"))
    )

    assert invoked == [], "a tool ran while the stop's state could not be read"
    [result] = _results(outcome)
    assert result.tool_outcome == "blocked"
    assert result.result.startswith(f"{STOP_PREFIX} ({reason}).")


@pytest.mark.emergency_stop_unavailable
def test_a_stop_path_that_cannot_be_resolved_holds_tool_calls(db, monkeypatch):
    import tldw_chatbook.emergency_stop as emergency_stop

    def refuse():
        raise PermissionError("data dir refused")

    monkeypatch.setattr(emergency_stop, "default_emergency_stop_path", refuse)
    invoked: list[str] = []
    service, registry, _chat = _calculator_run(db, invoked)

    _run_id, outcome = _run(
        service, _config("calculator"), _plan(registry.load_schema("builtin:calculator"))
    )

    assert invoked == [], "a tool ran while the stop's location was unknown"
    [result] = _results(outcome)
    assert result.tool_outcome == "blocked"
    # /emergency-stop clear cannot help when the lookup fails: say what did.
    assert result.result.startswith(UNAVAILABLE)
    assert "/emergency-stop clear" not in result.result


# --- clear --------------------------------------------------------------------------


def test_tools_run_again_once_the_stop_is_cleared(db, stop_path):
    invoked: list[str] = []
    service, registry, chat = _calculator_run(db, invoked)
    chat.replies.extend([_fence("calculator", {"expression": "3+3"}), "Six."])
    config = _config("calculator")
    plan = _plan(registry.load_schema("builtin:calculator"))
    set_emergency_stop(stop_path, reason="brief hold")

    _first_run, held = _run(service, config, plan, conversation="estop-1")
    clear_emergency_stop(stop_path)
    _second_run, resumed = _run(service, config, plan, conversation="estop-2")

    [held_result] = _results(held)
    assert held_result.result.startswith(f"{STOP_PREFIX} (brief hold).")
    assert invoked == ["builtin:calculator"], "the call after Clear did not run"
    [resumed_result] = _results(resumed)
    assert resumed_result.tool_outcome == "success"
    assert "6" in resumed_result.result


# --- the pure loop ------------------------------------------------------------------


@pytest.mark.emergency_stop_unavailable
def test_a_raising_stop_probe_holds_the_call_in_the_pure_loop():
    """The loop never lets a broken probe open the gate (fail safe)."""
    from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop

    turns = [ModelTurn(text=_fence("calculator", {})), ModelTurn(text="Done.")]
    invoked: list[ToolCall] = []

    def probe():
        raise RuntimeError("probe broke")

    deps = LoopDeps(
        call_model=lambda _messages, _schemas: turns.pop(0),
        invoke_tool=lambda call: invoked.append(call) or ToolResult(ok=True, content="4"),
        spawn=lambda task: ToolResult(ok=True, content="unused"),
        find_tools=lambda _query: [],
        load_schemas=lambda *_args: None,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        emergency_stop_refusal=probe,
    )
    schema = ToolSchema(
        id="builtin:calculator",
        name="calculator",
        description="math",
        parameters={"type": "object"},
    )

    outcome = run_agent_loop(
        _config("calculator"), [{"role": "user", "content": "go"}], [schema], deps
    )

    assert invoked == []
    [result] = _results(outcome)
    assert result.tool_outcome == "blocked"
    assert result.result.startswith(UNAVAILABLE)


def test_the_refusal_names_the_shipped_clear_command():
    from tldw_chatbook.Agents.tool_refusals import emergency_stop_tool_refusal

    assert emergency_stop_tool_refusal("why").endswith(
        f"clears it with {EMERGENCY_STOP_CLEAR_COMMAND}."
    )
    long_reason = "x" * 500 + "\n[b]tail"
    refusal = emergency_stop_tool_refusal(long_reason)
    assert "\n" not in refusal and len(refusal) < 320


# --- the run ends once the stop has held two turns in a row ---------------------------


CALC_SCHEMA = ToolSchema(
    id="builtin:calculator",
    name="calculator",
    description="math",
    parameters={"type": "object"},
)


def _loop(turns, held, invoked):
    """A pure loop whose stop probe answers from ``held`` (one bool per ask)."""
    from tldw_chatbook.Agents.agent_runtime import LoopDeps

    script = list(turns)
    model_calls: list[int] = []

    def call_model(_messages, _schemas):
        model_calls.append(1)
        return script.pop(0)

    def probe():
        return f"{STOP_PREFIX} (scripted)." if held.pop(0) else None

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=lambda call: invoked.append(call) or ToolResult(ok=True, content="4"),
        spawn=lambda task: ToolResult(ok=True, content="unused"),
        find_tools=lambda _query: [],
        load_schemas=lambda *_args: None,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        emergency_stop_refusal=probe,
    )
    return deps, model_calls


def _calls(n):
    return [ModelTurn(text=_fence("calculator", {"expression": f"{i}+1"})) for i in range(n)]


def test_the_run_ends_after_the_stop_holds_two_tool_turns_in_a_row():
    """A model that keeps asking during a stop stops costing model calls.

    A held result resets the tool-failure streak, so before this a model that
    varied its arguments kept calling the model for the run's whole budget
    (the review's probe: 21 model calls, 0 tools run).
    """
    from tldw_chatbook.Agents.agent_models import RUN_STUCK, RunTerminationReason
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop

    invoked: list[ToolCall] = []
    # One ask per batch (before review); held calls are not asked again.
    deps, model_calls = _loop(_calls(10), [True] * 10, invoked)

    outcome = run_agent_loop(
        _config("calculator"), [{"role": "user", "content": "go"}], [CALC_SCHEMA], deps
    )

    assert invoked == []
    assert len(model_calls) == 2, "the run kept calling the model during the stop"
    assert outcome.status == RUN_STUCK
    assert outcome.termination_reason is RunTerminationReason.EMERGENCY_STOP
    assert outcome.steps[-1].summary.startswith(
        "Agent stopped: the emergency stop held its tool calls 2 turns in a row"
    )


def test_one_held_turn_leaves_the_model_its_reply():
    """The first held turn is the model's chance to wind down in text."""
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop

    invoked: list[ToolCall] = []
    deps, model_calls = _loop(_calls(1) + [ModelTurn(text="Held; stopping.")], [True], invoked)

    outcome = run_agent_loop(
        _config("calculator"), [{"role": "user", "content": "go"}], [CALC_SCHEMA], deps
    )

    assert outcome.status == RUN_DONE and outcome.final_text == "Held; stopping."
    assert len(model_calls) == 2 and invoked == []


def test_a_turn_that_runs_a_tool_resets_the_held_count():
    """Held, ran (the stop was cleared), held: not two held turns in a row."""
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop

    invoked: list[ToolCall] = []
    # Turn 1 held (batch ask). Turn 2 clear: batch ask + dispatch ask. Turn 3 held.
    deps, model_calls = _loop(
        _calls(3) + [ModelTurn(text="Done.")], [True, False, False, True], invoked
    )

    outcome = run_agent_loop(
        _config("calculator"), [{"role": "user", "content": "go"}], [CALC_SCHEMA], deps
    )

    assert outcome.status == RUN_DONE and len(model_calls) == 4
    assert len(invoked) == 1


def test_a_goal_run_held_by_the_stop_ends_with_the_emergency_stop_reason(monkeypatch):
    """Goal runs end on any refusal; a stop hold is not a permission refusal."""
    from types import SimpleNamespace

    from tldw_chatbook.Agents import automatic_work_runtime
    from tldw_chatbook.Agents.agent_models import RUN_STUCK, RunTerminationReason
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop

    goal = SimpleNamespace(
        goal=SimpleNamespace(
            permits_call=lambda _name: True,
            observe_tool_result=lambda _name, _args, content: content,
        ),
        workflow=None,
        check=lambda: None,
    )
    monkeypatch.setattr(automatic_work_runtime, "current_automatic_work", lambda: goal)
    invoked: list[ToolCall] = []
    deps, _model_calls = _loop(_calls(2), [True] * 4, invoked)

    outcome = run_agent_loop(
        _config("calculator"), [{"role": "user", "content": "go"}], [CALC_SCHEMA], deps
    )

    assert invoked == []
    assert outcome.status == RUN_STUCK
    assert outcome.termination_reason is RunTerminationReason.EMERGENCY_STOP


def test_the_service_run_ends_after_two_held_turns(db, stop_path):
    """The same through AgentService and the real stop file."""
    from tldw_chatbook.Agents.agent_models import RUN_STUCK, RunTerminationReason

    invoked: list[str] = []
    service, registry, chat = _calculator_run(db, invoked)
    chat.replies[:] = [_fence("calculator", {"expression": f"{i}*2"}) for i in range(6)]
    set_emergency_stop(stop_path, reason="model keeps asking")

    _run_id, outcome = _run(
        service, _config("calculator"), _plan(registry.load_schema("builtin:calculator"))
    )

    assert invoked == []
    assert len(chat.calls) == 2
    assert outcome.status == RUN_STUCK
    assert outcome.termination_reason is RunTerminationReason.EMERGENCY_STOP
    chat.replies.clear()  # The rest were never asked for, which is the point.
