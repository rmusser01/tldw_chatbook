# Tests/Agents/test_local_tools_integration.py
"""End-to-end local-tool integration (phases 1-2, ADR-032): a scripted model
emits a ```tool_call fence for fs_list; the run must flow fence -> registry
-> build_local_review_hook -> approval round trip -> LocalToolProvider.invoke
-> fs_list core -> result appended back into the model's next turn.

Phase 2 adds the token-budgeted find_tools/load_tools disclosure path and the
allow-state e2e (zero approval round trips).

Harness pattern mirrors test_agent_service.py (ScriptedChat + real
AgentRunsDB, no network); provider/review-hook wiring mirrors
console_agent_bridge._compose_run_registry_and_allowed +
_combined_review_state_scope (registry with the local provider,
review_tool_calls=hook, review_state_scope=provider.stamp_scope).
"""

import asyncio
import dataclasses
import json
from io import BytesIO
import threading
from types import SimpleNamespace

import pytest

import tldw_chatbook.Agents.local_tool_provider as local_tool_provider
import tldw_chatbook.MCP.local_server_tools as local_server_tools
from tldw_chatbook.Agents.agent_models import RUN_DONE, AgentConfig, RunBudget
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_SERVER_KEY,
    LocalToolProvider,
    _default_specs,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall, MCPToolProvider
from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
from tldw_chatbook.Agents.session_todo_store import SessionTodoStore
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    ToolCatalogRegistry,
    probe_initial_catalog,
)
from tldw_chatbook.Chat.console_chat_controller import (
    USER_DENIED_REFUSAL,
    build_local_review_hook,
    build_mcp_review_hook,
)
from tldw_chatbook.Chat.console_agent_bridge import _compose_run_registry_and_allowed
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import (
    EffectiveToolState,
    resolve_effective_state,
)
from tldw_chatbook.Tools.workspace_tool_executor import (
    WorkspaceToolExecutionError,
    WorkspaceToolExecutor,
)
from tldw_chatbook.Tools.workspace_tool_protocol import WorkspaceToolResponse
from tldw_chatbook.Tools.workspace_tool_worker import run_workspace_worker


class InProcessWorkspaceExecutor:
    """Test-only real worker dispatch for linked checkouts not importable under -I."""

    def __init__(self, workspace_root):
        self._executor = WorkspaceToolExecutor(workspace_root)

    def execute(self, operation, arguments, *, intent):
        request = self._executor._build_request(operation, arguments, intent=intent)
        stdout = BytesIO()
        run_workspace_worker(BytesIO(request.to_bytes()), stdout, BytesIO())
        response = WorkspaceToolResponse.from_bytes(
            stdout.getvalue().splitlines()[-1],
            expected_operation_id=request.operation_id,
        )
        if response.outcome != "success":
            raise WorkspaceToolExecutionError(response.code, response.error)
        return response.result or ""


def _test_default_specs(workspace, **kwargs):
    return _default_specs(
        workspace,
        workspace_executor=InProcessWorkspaceExecutor(workspace),
        **kwargs,
    )


def fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _catalog_result_id_names(result: str) -> list[tuple[str, str]]:
    """Extract the catalog identity columns from a find_tools result."""
    rows = []
    for line in result.splitlines():
        tool_id, separator, remainder = line.partition(" — ")
        assert separator == " — "
        name, separator, _description = remainder.partition(": ")
        assert separator == ": "
        rows.append((tool_id, name))
    return rows


class ScriptedChat:
    """Returns scripted replies; records every call's kwargs."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        message = item if isinstance(item, dict) else {"content": item}
        return {"choices": [{"message": message}]}


class ReachabilityMCPService:
    """Small signature-faithful MCP service for the production path test."""

    def __init__(self) -> None:
        self.local_service = SimpleNamespace(get_inventory=lambda: {"tools": []})
        self.execute_calls: list[tuple[str, str, dict, str, str]] = []
        self.decision_calls: list[tuple[str, str, str]] = []

    def get_kill_switch(self) -> bool:
        return False

    async def local_external_catalog(self) -> list[dict]:
        return [
            {
                "profile_id": "late-server",
                "is_connected": True,
                "discovery_snapshot": {
                    "tools": [
                        {
                            "name": "reachable",
                            "description": "Return proof from the last MCP provider.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                            },
                        }
                    ]
                },
            }
        ]

    def effective_tool_states(self, tools):
        return {
            (tool.server_key, tool.name): EffectiveToolState(
                state="ask", origin="global_default"
            )
            for tool in tools
        }

    def gate_tool_test(self, _tool):
        return EffectiveToolState(state="ask", origin="global_default")

    def is_session_approved(self, _server_key: str, _tool_name: str) -> bool:
        return False

    def approve_for_session(self, _server_key: str, _tool_name: str) -> None:
        raise AssertionError("approve_once must not persist session authority")

    def set_tool_state(self, *_args, **_kwargs) -> None:
        raise AssertionError("approve_once must not persist tool authority")

    def record_tool_decision(
        self,
        server_key: str,
        tool_name: str,
        *,
        decision: str,
        initiator: str = "agent",
        error: str | None = None,
    ) -> None:
        assert initiator == "agent" and error is None
        self.decision_calls.append((server_key, tool_name, decision))

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
        # The provider bounds its cross-thread wait from the service timeout;
        # it does not override the service coroutine's optional timeout.
        assert timeout_seconds is None
        assert registered_argument_names == {"value"}
        self.execute_calls.append(
            (server_key, tool_name, dict(arguments or {}), initiator, decision)
        )
        return {"content": [{"type": "text", "text": "MCP reached"}]}


@pytest.fixture()
def mcp_main_loop():
    """Run the MCP execution coroutine on the same kind of loop as Console."""
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


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


@pytest.fixture()
def workspace(tmp_path):
    """The confined workspace root: exactly one file, so its name MUST show
    up in a successful fs_list result."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("hello", encoding="utf-8")
    return root


def test_external_mcp_fs_read_uses_the_pinned_workspace_executor(
    workspace, monkeypatch
):
    calls: list[tuple[str, dict, str]] = []

    class RecordingWorkspaceExecutor:
        def __init__(self, workspace_root):
            assert workspace_root == workspace.resolve()

        def execute(self, operation, arguments, *, intent):
            calls.append((operation, dict(arguments), intent))
            return "leased external read"

    class PermissionStore:
        def load(self):
            return {}

        def get_kill_switch(self):
            return False

    monkeypatch.setattr(
        local_tool_provider,
        "WorkspaceToolExecutor",
        RecordingWorkspaceExecutor,
        raising=False,
    )
    monkeypatch.setattr(
        local_server_tools,
        "resolve_effective_state",
        lambda _payload, _hub: EffectiveToolState(
            state="allow",
            origin="tool_override",
        ),
    )
    provider = local_server_tools.build_server_local_provider(
        workspace,
        PermissionStore(),
    )
    registration = next(
        item
        for item in local_server_tools._local_agent_tool_registrations(provider)
        if item.name == "fs_read"
    )

    result = registration.handler({"path": "notes.txt"})

    assert result.ok and result.content == "leased external read"
    assert calls == [("fs_read", {"path": "notes.txt"}, "read")]


def make_service(
    db,
    workspace,
    replies,
    approvals,
    approval_calls,
    *,
    state=None,
    extra_specs=(),
    specs=None,
    todo_store: SessionTodoStore | None = None,
    watchlists_service=None,
    resolve_state=None,
):
    """Assemble the run exactly as the bridge does: registry with builtins +
    the local provider, the build_local_review_hook batch hook, and the
    provider's stamp_scope as review_state_scope.

    ``specs`` replaces the default local spec set (used to keep approval-flow
    fixtures compact enough for direct disclosure); ``extra_specs`` appends
    to whichever base set is in use.
    ``todo_store`` wires a live stable-ID session task store into the default
    spec set (the four ``todo_*`` operations are only registered then)."""
    base = (
        list(specs)
        if specs is not None
        else _test_default_specs(
            workspace,
            todo_store=todo_store,
            watchlists_service=watchlists_service,
        )
    )
    provider = LocalToolProvider(
        workspace_root=workspace,
        specs=base + list(extra_specs),
        resolve_state=(
            resolve_state
            if resolve_state is not None
            else lambda hub: (
                state or EffectiveToolState(state="ask", origin="global_default")
            )
        ),
    )

    def request_approvals(pending):
        approval_calls.append(pending)
        return dict(approvals)

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(provider)
    chat = ScriptedChat(replies)
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=build_local_review_hook(provider, request_approvals),
        review_state_scope=provider.stamp_scope,
    )
    return service, chat


CFG = AgentConfig(
    model="test-model",
    system_prompt="You are helpful.",
    allowed_tools=("fs_list",),
)


def test_fs_list_fence_flow_executes_after_approve_once(db, workspace):
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [fence("fs_list", {"path": "."}), "The workspace has notes.txt."],
        {"fs_list": "approve_once"},
        approval_calls,
        # Approval-flow test, not a disclosure test: use the compact fs-only
        # catalog so its complete schema request is directly callable.
        specs=fs_only_specs(workspace),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "what files are here?"}],
        config=CFG,
        api_endpoint="llama_cpp",  # fence-protocol endpoint (harness pattern)
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "The workspace has notes.txt."
    assert len(chat.calls) == 2  # tool turn + final turn

    # 1. The tool step's result contains the filename (not an error).
    tool_results = [s for s in outcome.steps if s.kind == "tool_result"]
    assert [s.tool_name for s in tool_results] == ["fs_list"]
    assert "notes.txt" in tool_results[0].result
    assert not tool_results[0].result.startswith("ERROR")

    # 2. Exactly ONE approval round trip, gated on the local server key.
    assert len(approval_calls) == 1
    assert len(approval_calls[0]) == 1
    pending = approval_calls[0][0]
    assert isinstance(pending, MCPPendingCall)
    assert pending.server_key == LOCAL_SERVER_KEY == "local:__local__"
    assert pending.llm_name == "fs_list"
    assert pending.tool_name == "fs_list"
    assert pending.arguments == {"path": "."}

    # 3. The tool result went back to the model (fence convention: a
    # user-role "Tool result for {name}: ..." line in the second turn).
    second_payload = chat.calls[1]["messages_payload"]
    assert any(
        m["role"] == "user"
        and m["content"].startswith("Tool result for fs_list: ")
        and "notes.txt" in m["content"]
        for m in second_payload
    )


def test_fs_list_fence_flow_denied_still_completes(db, workspace):
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [fence("fs_list", {"path": "."}), "I could not list the files."],
        {"fs_list": "deny"},
        approval_calls,
        specs=fs_only_specs(workspace),  # see above: direct disclosure wanted
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "what files are here?"}],
        config=CFG,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    # The run completes; the model's second turn still runs.
    assert outcome.status == RUN_DONE
    assert outcome.final_text == "I could not list the files."
    assert len(chat.calls) == 2

    # The review hook returns the canonical user-denial copy without dispatch.
    tool_results = [s for s in outcome.steps if s.kind == "tool_result"]
    assert [s.tool_name for s in tool_results] == ["fs_list"]
    refusal = USER_DENIED_REFUSAL.format(name="fs_list")
    assert tool_results[0].result == refusal
    assert "notes.txt" not in tool_results[0].result

    # The approval gate was still consulted exactly once.
    assert len(approval_calls) == 1
    assert approval_calls[0][0].server_key == LOCAL_SERVER_KEY

    # The denial is fed back to the model with the same "Tool result for
    # {name}: ..." line shape a success produces, so turn 2 can react to it.
    second_payload = chat.calls[1]["messages_payload"]
    assert any(
        m["role"] == "user"
        and m["content"] == f"Tool result for fs_list: {refusal}"
        for m in second_payload
    )


# --- Phase 2: disclosure policy + allow-state coverage ----------------------

# Original six fs_* tools only; fs_patch (added 3b-i) is deliberately excluded
# so fs_only_specs remains a compact direct-disclosure fixture. Do NOT add
# newer fs_* tools here; update LOCAL_TOOL_NAMES below instead.
FS_TOOL_NAMES = {"fs_list", "fs_read", "fs_write", "fs_edit", "fs_glob", "fs_grep"}
# phase-3b-ii: read-only git tools (no risk tags per ADR-033).
GIT_TOOL_NAMES = {"git_status", "git_diff", "git_log", "git_blame", "git_branches"}
LOCAL_TOOL_NAMES = (
    FS_TOOL_NAMES | {"fs_patch", "web_fetch", "web_search"} | GIT_TOOL_NAMES
)  # phase-3b-ii default set (14 local tools)
BUILTIN_TOOL_NAMES = {"calculator", "get_current_datetime"}


def fs_only_specs(workspace):
    """The original 6 fs_* specs (fs_patch and later tools deliberately
    excluded), providing a compact direct-disclosure fixture."""
    return [s for s in _test_default_specs(workspace) if s.name in FS_TOOL_NAMES]


def _extra_schema_cost_specs(count: int):
    """Inert local specs used to make discovery mode explicit in tests."""
    from tldw_chatbook.Agents.local_tool_provider import LocalToolSpec
    from tldw_chatbook.Agents.local_tool_provider import LocalToolExposure

    return [
        LocalToolSpec(
            name=f"fs_pad_{i}",
            description=f"Inert padding spec {i} (test only).",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: "noop",
            exposure=LocalToolExposure.CONSOLE_ONLY,
            approval_effects=(),
        )
        for i in range(count)
    ]


def production_registry(workspace, extra_specs=(), specs=None):
    base = list(specs) if specs is not None else _test_default_specs(workspace)
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(
        LocalToolProvider(
            workspace_root=workspace,
            specs=base + list(extra_specs),
        )
    )
    return registry


def test_direct_disclosure_uses_complete_schema_cost(workspace):
    """The shipped registry is disclosed or deferred by set cost, not count."""
    registry = production_registry(workspace, specs=fs_only_specs(workspace))
    allowed = tuple(entry.name for entry in registry.list_catalog())
    schemas = probe_initial_catalog(registry, allowed, 100, lambda _schemas: 99)
    assert schemas is not None
    assert {s.name for s in schemas} == FS_TOOL_NAMES | BUILTIN_TOOL_NAMES
    assert (
        probe_initial_catalog(registry, allowed, 100, lambda _schemas: 101) is None
    )


def test_mcp_registered_last_remains_reachable_through_discovery_and_approval(
    db, workspace, mcp_main_loop, monkeypatch
):
    """A late MCP provider survives discovery, load, approval, and dispatch."""
    mcp_service = ReachabilityMCPService()
    mcp_provider = MCPToolProvider(service=mcp_service, main_loop=mcp_main_loop)
    asyncio.run(mcp_provider.compose_catalog())
    mcp_name = mcp_provider.list_catalog()[0].name

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(
        LocalToolProvider(
            workspace_root=workspace,
            specs=_test_default_specs(workspace),
        )
    )
    registry.register_provider(LibraryToolProvider(object()))
    # The external provider is deliberately registered last. Reachability is
    # proved by catalog identity, never by an unstable registration index.
    registry.register_provider(mcp_provider)

    approval_calls: list[list[MCPPendingCall]] = []

    def request_approvals(pending: list[MCPPendingCall]) -> dict[str, str]:
        approval_calls.append(pending)
        return {call.llm_name: "approve_once" for call in pending}

    chat = ScriptedChat(
        [
            fence("find_tools", {"query": mcp_name}),
            fence("load_tools", {"ids": [mcp_name]}),
            fence(mcp_name, {"value": "proof"}),
            "The MCP tool was reached.",
        ]
    )
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=build_mcp_review_hook(mcp_provider, request_approvals),
        review_state_scope=mcp_provider.stamp_scope,
    )

    # Cost, not provider order or entry count, forces first-turn discovery.
    # The same singleton is above the 10% automatic threshold but remains
    # loadable because its complete next request still fits.
    monkeypatch.setattr(
        agent_service, "get_model_token_limit", lambda *_args, **_kwargs: 10_000
    )
    monkeypatch.setattr(
        agent_service, "catalog_schema_tokens", lambda *_args, **_kwargs: 1_001
    )
    monkeypatch.setattr(
        agent_service, "_count_model_messages", lambda *_args, **_kwargs: 50
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=tuple(entry.name for entry in registry.list_catalog()),
        budget=RunBudget(max_steps=16, max_model_turns=8),
        response_reserve_tokens=100,
    )

    _run_id, outcome = service.run_turn(
        conversation_id="mcp-last",
        messages=[{"role": "user", "content": "Reach the MCP proof tool."}],
        config=config,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert [
        step.tool_name for step in outcome.steps if step.kind == "tool_call"
    ] == ["find_tools", "load_tools", mcp_name]
    assert mcp_name in next(
        step.result
        for step in outcome.steps
        if step.kind == "tool_result" and step.tool_name == "find_tools"
    )
    assert len(approval_calls) == 1
    assert [(call.server_key, call.tool_name) for call in approval_calls[0]] == [
        ("local:late-server", "reachable")
    ]
    assert mcp_service.execute_calls == [
        (
            "local:late-server",
            "reachable",
            {"value": "proof"},
            "agent",
            "approved",
        )
    ]
    assert any(
        step.kind == "tool_result"
        and step.tool_name == mcp_name
        and "MCP reached" in step.result
        for step in outcome.steps
    )


def test_raw_shell_provider_joins_the_local_registry_partition(workspace):
    runtime = SimpleNamespace(
        permitted=True,
        armed=True,
        model_session_granted=lambda _session_id: False,
        grant_model_session=lambda _session_id: None,
    )
    provider = RawShellToolProvider(
        runtime=runtime,
        console_session_id="console-session",
        initial_directory=lambda: workspace,
    )

    registry, allowed, _builtin_names, local_names = (
        _compose_run_registry_and_allowed({}, raw_shell_provider=provider)
    )

    assert "shell_exec" in allowed
    assert "shell_exec" in local_names
    assert registry.load_schema("raw_shell:shell_exec").name == "shell_exec"


def test_find_load_path_executes_fs_edit_after_approve_once(db, workspace):
    """In discovery mode the model reaches fs_edit via find_tools ->
    load_tools -> call; the edit lands on disk behind one approval gate."""
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [
            fence("find_tools", {"query": "edit"}),
            fence("load_tools", {"ids": ["local:fs_edit"]}),
            fence(
                "fs_edit",
                {"path": "notes.txt", "old_string": "hello", "new_string": "goodbye"},
            ),
            "Edited the file.",
        ],
        {"fs_edit": "approve_once"},
        approval_calls,
        # Add schema cost so the planner genuinely offers find/load rather
        # than relying on the fence loop's name-based dispatch alone.
        extra_specs=_extra_schema_cost_specs(1),
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("fs_edit",),
        # 3 tool turns + the final answer blow past the 8-step default.
        budget=RunBudget(max_steps=16, max_model_turns=8),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "change hello to goodbye"}],
        config=config,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "Edited the file."
    assert len(chat.calls) == 4  # find + load + edit + final

    # The scripted discovery sequence ran in order.
    called = [s.tool_name for s in outcome.steps if s.kind == "tool_call"]
    assert called == ["find_tools", "load_tools", "fs_edit"]

    # find_tools surfaced the catalog id the model then loaded.
    second_payload = chat.calls[1]["messages_payload"]
    assert any(
        m["role"] == "user"
        and m["content"].startswith("Tool result for find_tools: ")
        and "local:fs_edit" in m["content"]
        for m in second_payload
    )

    # The edit actually happened on disk.
    assert (workspace / "notes.txt").read_text(encoding="utf-8") == "goodbye"

    # Exactly ONE approval round trip: the fs_edit gate. find_tools and
    # load_tools are runtime tools the local provider doesn't gate.
    assert len(approval_calls) == 1
    assert len(approval_calls[0]) == 1
    pending = approval_calls[0][0]
    assert isinstance(pending, MCPPendingCall)
    assert pending.server_key == LOCAL_SERVER_KEY
    assert pending.llm_name == "fs_edit"


def test_allow_state_executes_without_approval_round_trip(db, workspace):
    """resolve_state -> allow (tool_override): fs_read runs with ZERO
    approval round trips and its content reaches the model's next turn."""
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [fence("fs_read", {"path": "notes.txt"}), "The file says hello."],
        {},  # no scripted approvals: any round trip would fail loudly here
        approval_calls,
        state=EffectiveToolState(state="allow", origin="tool_override"),
        specs=fs_only_specs(workspace),  # direct disclosure (see above)
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("fs_read",),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "read notes.txt"}],
        config=config,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "The file says hello."
    assert len(chat.calls) == 2

    # Zero approval round trips — an allow-state tool never enters the batch.
    assert approval_calls == []

    # The read executed and its (line-numbered) content came back.
    tool_results = [s for s in outcome.steps if s.kind == "tool_result"]
    assert [s.tool_name for s in tool_results] == ["fs_read"]
    assert "hello" in tool_results[0].result
    assert not tool_results[0].result.startswith("ERROR")

    second_payload = chat.calls[1]["messages_payload"]
    assert any(
        m["role"] == "user"
        and m["content"].startswith("Tool result for fs_read: ")
        and "hello" in m["content"]
        for m in second_payload
    )


# --- Phase 3a Task 5: research-tool + todo e2e over the find/load path -----


def test_find_load_path_executes_web_fetch_after_approve_once(db, workspace):
    """find_tools("fetch") -> load_tools("local:web_fetch") -> web_fetch call:
    the discovered tool executes behind ONE approval round trip.

    The network is cut at the handler seam (dataclasses.replace on the
    frozen LocalToolSpec), not the httpx layer: Tests/Tools/test_web_tool_
    impls.py already covers web_fetch's real transport/DNS behavior; this
    test covers the discovery + gating flow."""
    fetched = []

    def fake_fetch(args):
        fetched.append(dict(args))
        return "Example Domain body text"

    specs = [
        dataclasses.replace(s, handler=fake_fetch) if s.name == "web_fetch" else s
        for s in _test_default_specs(workspace)
    ]
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [
            fence("find_tools", {"query": "fetch"}),
            fence("load_tools", {"ids": ["local:web_fetch"]}),
            fence("web_fetch", {"url": "http://example.com/"}),
            "Fetched the page.",
        ],
        {"web_fetch": "approve_once"},
        approval_calls,
        specs=specs,
        # Add schema cost so find/load is genuinely the live mode.
        extra_specs=_extra_schema_cost_specs(1),
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("web_fetch",),
        # find + load + fetch + final: past the 8-step default (precedent:
        # max_steps=16 in the fs_edit find/load test above).
        budget=RunBudget(max_steps=16, max_model_turns=8),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "fetch http://example.com/"}],
        config=config,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "Fetched the page."
    assert len(chat.calls) == 4  # find + load + fetch + final

    # The scripted discovery sequence ran in order.
    called = [s.tool_name for s in outcome.steps if s.kind == "tool_call"]
    assert called == ["find_tools", "load_tools", "web_fetch"]

    # find_tools surfaced the catalog id the model then loaded.
    second_payload = chat.calls[1]["messages_payload"]
    assert any(
        m["role"] == "user"
        and m["content"].startswith("Tool result for find_tools: ")
        and "local:web_fetch" in m["content"]
        for m in second_payload
    )

    # The stubbed handler ran with the model's URL — no network touched.
    assert fetched == [{"url": "http://example.com/"}]

    # The fetch result went back to the model for the final turn.
    final_payload = chat.calls[3]["messages_payload"]
    assert any(
        m["role"] == "user"
        and m["content"].startswith("Tool result for web_fetch: ")
        and "Example Domain body text" in m["content"]
        for m in final_payload
    )

    # Exactly ONE approval round trip: the web_fetch gate. find_tools and
    # load_tools are runtime tools the local provider doesn't gate.
    assert len(approval_calls) == 1
    assert len(approval_calls[0]) == 1
    pending = approval_calls[0][0]
    assert isinstance(pending, MCPPendingCall)
    assert pending.server_key == LOCAL_SERVER_KEY
    assert pending.llm_name == "web_fetch"
    assert pending.arguments == {"url": "http://example.com/"}


def test_todo_create_mutates_session_store_after_approve_once(db, workspace):
    """find_tools("todo") -> load_tools("local:todo_create") -> todo_create:
    the discovered tool creates a stable-ID task in the injected session store,
    behind ONE approval round trip. With a store wired, the complete schema
    request selects discovery, so todo_create is loaded before execution.

    resolve_state is constructed deliberately as what
    MCP.permission_store.resolve_effective_state returns for todo_create
    (tags=("mutates",), which intersects HIGH_RISK_TAGS) under an INHERITED
    allow (origin="global_default"): the high-risk floor downgrades it to
    ``ask`` with ``risk_floored=True``. An explicit tool_override allow is
    never floored and would skip the gate entirely — that path is already
    pinned by test_allow_state_executes_without_approval_round_trip."""
    todo_store = SessionTodoStore()
    assert "todo_write" not in {
        spec.name for spec in _test_default_specs(workspace, todo_store=todo_store)
    }
    create_args = {
        "content": "Write the report",
        "activeForm": "Writing the report",
    }
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [
            fence("find_tools", {"query": "todo"}),
            fence("load_tools", {"ids": ["local:todo_create"]}),
            fence("todo_create", create_args),
            "Task created.",
        ],
        {"todo_create": "approve_once"},
        approval_calls,
        state=EffectiveToolState(
            state="ask", origin="global_default", risk_floored=True
        ),
        todo_store=todo_store,
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("todo_create",),
        # find + load + create + final: past the 8-step default (precedent:
        # max_steps=16 in the fs_edit find/load test above).
        budget=RunBudget(max_steps=16, max_model_turns=8),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "track these tasks"}],
        config=config,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "Task created."
    assert len(chat.calls) == 4  # find + load + create + final

    # The scripted discovery sequence ran in order.
    called = [s.tool_name for s in outcome.steps if s.kind == "tool_call"]
    assert called == ["find_tools", "load_tools", "todo_create"]
    assert "todo_write" not in called

    created = {
        "id": "1",
        "version": 1,
        "content": "Write the report",
        "status": "pending",
        "activeForm": "Writing the report",
    }
    assert todo_store.get("1") == created

    # The tool step's result is the confirmation line, not an error.
    tool_results = [s for s in outcome.steps if s.kind == "tool_result"]
    assert [s.tool_name for s in tool_results] == [
        "find_tools",
        "load_tools",
        "todo_create",
    ]
    assert _catalog_result_id_names(tool_results[0].result) == [
        ("local:todo_create", "todo_create")
    ]
    assert tool_results[1].result == "loaded: todo_create"
    assert tool_results[-1].result == json.dumps(
        created, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )

    # Exactly ONE approval round trip, gated on the local server key with
    # the risk floor as the stated reason.
    assert len(approval_calls) == 1
    assert len(approval_calls[0]) == 1
    pending = approval_calls[0][0]
    assert isinstance(pending, MCPPendingCall)
    assert pending.server_key == LOCAL_SERVER_KEY
    assert pending.llm_name == "todo_create"
    assert pending.arguments == create_args
    assert pending.reason == "risk_floored"


def test_find_load_path_todo_get_reads_created_task_without_mutation_floor(
    db, workspace
):
    """A read follows a gated create through the same real provider/store.

    The inherited server allow is deliberately resolved by the production
    permission function. It floors ``todo_create`` because that HubTool carries
    ``mutates``, while the empty-tag ``todo_get`` remains allowed and therefore
    adds no second approval round trip.
    """
    todo_store = SessionTodoStore()
    permission_payload = {
        "profiles": {
            "default": {
                "global_default": "ask",
                "servers": {LOCAL_SERVER_KEY: {"default": "allow"}},
            }
        }
    }
    resolved = {}

    def resolve_state(hub):
        effective = resolve_effective_state(permission_payload, hub)
        resolved[hub.name] = (hub.tags, effective)
        return effective

    create_args = {
        "content": "Write the report",
        "activeForm": "Writing the report",
    }
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [
            fence("find_tools", {"query": "todo"}),
            fence(
                "load_tools",
                {"ids": ["local:todo_create", "local:todo_get"]},
            ),
            fence("todo_create", create_args),
            fence("todo_get", {"id": "1"}),
            "Task created and read back.",
        ],
        {"todo_create": "approve_once"},
        approval_calls,
        todo_store=todo_store,
        resolve_state=resolve_state,
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("todo_create", "todo_get"),
        budget=RunBudget(max_steps=20, max_model_turns=10),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "track and reread this task"}],
        config=config,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "Task created and read back."
    called = [s.tool_name for s in outcome.steps if s.kind == "tool_call"]
    assert called == ["find_tools", "load_tools", "todo_create", "todo_get"]

    created = {
        "id": "1",
        "version": 1,
        "content": "Write the report",
        "status": "pending",
        "activeForm": "Writing the report",
    }
    compact_created = json.dumps(
        created, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    tool_results = [s for s in outcome.steps if s.kind == "tool_result"]
    assert [s.tool_name for s in tool_results] == called
    assert _catalog_result_id_names(tool_results[0].result) == [
        ("local:todo_create", "todo_create"),
        ("local:todo_get", "todo_get"),
    ]
    assert tool_results[1].result == "loaded: todo_create, todo_get"
    assert tool_results[-2].result == compact_created
    assert tool_results[-1].result == compact_created
    assert todo_store.get("1") == created
    assert compact_created in str(chat.calls[-1]["messages_payload"])

    assert len(approval_calls) == 1
    assert [pending.llm_name for pending in approval_calls[0]] == ["todo_create"]
    create_tags, create_state = resolved["todo_create"]
    assert create_tags == ("mutates",)
    assert create_state.state == "ask"
    assert create_state.risk_floored is True
    get_tags, get_state = resolved["todo_get"]
    assert get_tags == ()
    assert get_state.state == "allow"
    assert get_state.origin == "server_default"
    assert get_state.risk_floored is False


# --- Phase 3b-ii: git tool reachable over the find/load path -----------------


def test_find_load_path_executes_git_log_after_approve_once(db, workspace):
    """find_tools("git") -> load_tools("local:git_log") -> git_log call:
    a phase-3b-ii git tool stays reachable when schema cost selects discovery
    and executes behind ONE approval round trip.

    Git is cut at the handler seam (dataclasses.replace on the frozen
    LocalToolSpec), not by running a real repo in the harness — the web_fetch
    e2e precedent: Tests/Agents/test_local_tool_provider.py and
    Tests/Tools/test_git_tool_impls.py cover the real git behavior; this test
    covers the discovery + gating flow."""
    logged = []

    def fake_log(args):
        logged.append(dict(args))
        return "abc1234 2026-08-01 Test User: initial commit"

    specs = [
        dataclasses.replace(s, handler=fake_log) if s.name == "git_log" else s
        for s in _test_default_specs(workspace)
    ]
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [
            fence("find_tools", {"query": "git"}),
            fence("load_tools", {"ids": ["local:git_log"]}),
            fence("git_log", {"count": 5}),
            "The repo has one commit.",
        ],
        {"git_log": "approve_once"},
        approval_calls,
        specs=specs,
        # Add schema cost so find/load is genuinely the live mode.
        extra_specs=_extra_schema_cost_specs(1),
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("git_log",),
        # find + load + log + final: past the 8-step default (precedent:
        # max_steps=16 in the fs_edit find/load test above).
        budget=RunBudget(max_steps=16, max_model_turns=8),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "show the recent commits"}],
        config=config,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "The repo has one commit."
    assert len(chat.calls) == 4  # find + load + log + final

    # The scripted discovery sequence ran in order.
    called = [s.tool_name for s in outcome.steps if s.kind == "tool_call"]
    assert called == ["find_tools", "load_tools", "git_log"]

    # find_tools surfaced the catalog id the model then loaded.
    second_payload = chat.calls[1]["messages_payload"]
    assert any(
        m["role"] == "user"
        and m["content"].startswith("Tool result for find_tools: ")
        and "local:git_log" in m["content"]
        for m in second_payload
    )

    # The stubbed handler ran with the model's args — no real git touched.
    assert logged == [{"count": 5}]

    # The log result went back to the model for the final turn.
    final_payload = chat.calls[3]["messages_payload"]
    assert any(
        m["role"] == "user"
        and m["content"].startswith("Tool result for git_log: ")
        and "initial commit" in m["content"]
        for m in final_payload
    )

    # Exactly ONE approval round trip: the git_log gate. find_tools and
    # load_tools are runtime tools the local provider doesn't gate.
    assert len(approval_calls) == 1
    assert len(approval_calls[0]) == 1
    pending = approval_calls[0][0]
    assert isinstance(pending, MCPPendingCall)
    assert pending.server_key == LOCAL_SERVER_KEY
    assert pending.llm_name == "git_log"
    assert pending.arguments == {"count": 5}


class RecordingWatchlistsService:
    def __init__(self, result):
        self.result = json.dumps(result, separators=(",", ":"))
        self.calls = []

    def search_items(self, arguments):
        self.calls.append(("search_items", dict(arguments)))
        return self.result

    def get_item(self, arguments):
        self.calls.append(("get_item", dict(arguments)))
        return self.result

    def _unexpected(self, _arguments):
        raise AssertionError("an unrelated Watchlists tool was invoked")

    list_sources = _unexpected
    list_collections = _unexpected
    list_briefings = _unexpected
    get_briefing = _unexpected
    get_operations_status = _unexpected
    get_operation_status = _unexpected


@pytest.mark.parametrize(
    ("tool_name", "arguments", "method_name"),
    [
        ("watchlists_search_items", {"query": "topic", "limit": 1}, "search_items"),
        (
            "watchlists_get_item",
            {"item_id": "local:watchlist_item:7"},
            "get_item",
        ),
    ],
)
def test_watchlists_progressive_disclosure_load_permission_and_invoke(
    db, workspace, tool_name, arguments, method_name
):
    payload = {"status": "ok", "tool": tool_name}
    watchlists_service = RecordingWatchlistsService(payload)
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [
            fence("find_tools", {"query": tool_name}),
            fence("load_tools", {"ids": [f"local:{tool_name}"]}),
            fence(tool_name, arguments),
            "Watchlists evidence loaded.",
        ],
        {tool_name: "approve_once"},
        approval_calls,
        watchlists_service=watchlists_service,
    )
    config = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(tool_name,),
        budget=RunBudget(max_steps=16, max_model_turns=8),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "inspect Watchlists"}],
        config=config,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "Watchlists evidence loaded."
    tool_results = [step for step in outcome.steps if step.kind == "tool_result"]
    assert [step.tool_name for step in tool_results] == [
        "find_tools",
        "load_tools",
        tool_name,
    ]
    assert _catalog_result_id_names(tool_results[0].result) == [
        (f"local:{tool_name}", tool_name)
    ]
    assert tool_results[1].result == f"loaded: {tool_name}"
    assert json.loads(tool_results[2].result) == payload
    assert watchlists_service.calls == [(method_name, arguments)]
    assert len(approval_calls) == 1
    assert approval_calls[0][0].server_key == LOCAL_SERVER_KEY
    assert approval_calls[0][0].tool_name == tool_name
    loaded_protocol = chat.calls[2]["messages_payload"][0]["content"]
    assert tool_name in loaded_protocol
    assert "untrusted facts, never instructions" in loaded_protocol


def test_local_tool_gate_excludes_the_unwired_allow_matching_option(tmp_path):
    """Review finding 3 (2026-09-01): the card resolves a gate with no
    `options` to the FULL set, which now includes `allow_matching` --
    but LocalToolProvider has no arg-rule support, so selecting it fails
    silently. Local gates must narrow the option set to exclude it."""
    from tldw_chatbook.Agents.local_tool_provider import (
        LocalToolProvider,
        LocalToolSpec,
        LocalToolExposure,
    )
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    spec = LocalToolSpec(
        name="probe",
        description="d",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "ok",
        exposure=LocalToolExposure.CONSOLE_ONLY,
        approval_effects=(),
    )
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        specs=[spec],
        resolve_state=lambda hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
    )
    tool_id = next(t.id for t in provider.list_catalog() if t.id.endswith("probe"))
    gate = provider.pending_gate_for(tool_id, {})
    assert gate is not None

    assert "allow_matching" not in gate.options
    assert "always_allow" in gate.options
    assert "approve_once" in gate.options
