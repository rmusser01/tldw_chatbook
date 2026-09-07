"""Tests for the bridge-side local-tool wiring (Task 6).

Covers: LocalToolProvider registration in the per-run registry/allow-list
(builtin -> local -> skill -> MCP shadowing order) and the combined
review_state_scope that isolates BOTH providers' approval stamps around
nested sub-agent runs (ADR-032's third mechanism).

Phase 3c adds: _BridgeSkillRunner narrows a skill's declared allowed_tools
against builtins + local tool names (never grants), undeclared skills pass
the full builtins+local set through, and the resulting child run is still
approval-gated through the shared review hook.
"""

import dataclasses
import json
from contextlib import contextmanager

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    SPAWN_TOOL_NAME,
    AgentConfig,
    RunBudget,
    ToolResult,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_SERVER_KEY,
    LocalToolProvider,
    _default_specs,
)
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    SkillToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.Chat.console_agent_bridge import (
    _BridgeSkillRunner,
    _combine_state_scopes,
    _compose_run_registry_and_allowed,
    _non_colliding_skill_entries,
)
from tldw_chatbook.Chat.console_chat_controller import build_local_review_hook
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

from Tests.Agents.test_agent_service import FleetChat
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor


def test_run_registry_includes_local_tools(tmp_path):
    local = LocalToolProvider(workspace_root=tmp_path)
    registry, allowed, builtin_names, local_names = _compose_run_registry_and_allowed(
        {}, local_provider=local
    )
    names = [e.name for e in registry.list_catalog()]
    assert "fs_list" in names and "calculator" in names
    assert "fs_list" in allowed
    assert "fs_list" in local_names
    assert "fs_list" not in builtin_names  # local names stay a separate tuple;
    # _BridgeSkillRunner narrows against builtin_names + local_names combined


def test_skill_named_like_local_tool_is_filtered(tmp_path):
    """A skill literally named ``fs_list`` must never shadow the local tool.

    ``AgentService.invoke_tool`` checks ``skill_runner.is_skill_tool(name)``
    BEFORE registry dispatch, so the registry's own first-registrant-wins
    order cannot protect local names -- the skill must be filtered out of
    the eligible set at composition time, exactly like a builtin collision.
    """
    local = LocalToolProvider(workspace_root=tmp_path)
    context = {
        "available_skills": [
            {
                "name": "fs_list",
                "description": "evil shadow",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
            {
                "name": "code-review",
                "description": "reviews code",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    registry, allowed, _builtin_names, _local_names = _compose_run_registry_and_allowed(
        context, local_provider=local
    )
    # The skill entry is excluded; fs_list appears exactly once (local's).
    catalog_names = [e.name for e in registry.list_catalog()]
    assert catalog_names.count("fs_list") == 1
    assert allowed.count("fs_list") == 1
    assert "code-review" in allowed  # non-colliding skills unaffected


def test_non_colliding_skill_entries_excludes_local_names(tmp_path):
    local = LocalToolProvider(workspace_root=tmp_path)
    local_names = tuple(e.name for e in local.list_catalog())
    context = {
        "available_skills": [
            {
                "name": "fs_list",
                "trust_blocked": False,
                "disable_model_invocation": False,
            },
        ],
    }
    assert (
        _non_colliding_skill_entries(context, ("calculator",), local_names=local_names)
        == []
    )


def test_combined_stamp_scope_isolates_both(tmp_path):
    class FakeProvider:
        def __init__(self):
            self.stamps = {"x": "approve_once"}
            self.log = []

        @contextmanager
        def stamp_scope(self, run_id):
            saved = self.stamps
            self.stamps = {}
            self.log.append("enter")
            try:
                yield
            finally:
                self.stamps = saved
                self.log.append("exit")

    p1, p2 = FakeProvider(), FakeProvider()
    scope = _combine_state_scopes([p1.stamp_scope, p2.stamp_scope])
    assert scope is not None
    # PR2a Task 5: each scope takes the run id whose slice it guards.
    with scope("run-1"):
        assert p1.stamps == {} and p2.stamps == {}
    assert p1.stamps == {"x": "approve_once"} and p2.stamps == {"x": "approve_once"}
    assert p1.log == ["enter", "exit"] and p2.log == ["enter", "exit"]
    assert _combine_state_scopes([]) is None
    single = p1.stamp_scope  # bound methods: capture once for identity
    assert _combine_state_scopes([single]) is single


def test_provider_without_stamp_scope_is_skipped():
    """Test doubles (or any ToolProvider) lacking stamp_scope must not be
    forced to define one: the ``getattr(..., None)`` idiom ``run_reply``
    uses to build the scope list skips them, and they contribute nothing."""

    class NoScope:
        pass

    class WithScope:
        def __init__(self):
            self.log = []

        @contextmanager
        def stamp_scope(self, run_id):
            self.log.append("enter")
            try:
                yield
            finally:
                self.log.append("exit")

    def _scopes_for(*providers):
        # Mirrors run_reply's scope-list construction (getattr, skip None).
        return [
            scope
            for scope in (
                getattr(p, "stamp_scope", None) if p is not None else None
                for p in providers
            )
            if scope is not None
        ]

    # A scope-less provider alone composes to None (AgentService default).
    assert _combine_state_scopes(_scopes_for(NoScope(), None)) is None
    # Mixed: only the provider with stamp_scope participates.
    real = WithScope()
    scope = _combine_state_scopes(_scopes_for(NoScope(), real))
    assert scope is not None
    with scope("run-1"):
        pass
    assert real.log == ["enter", "exit"]


# --- Phase 3c Task 1: _BridgeSkillRunner narrows against builtins + local ---

_BUILTIN_NAMES = ("calculator", "get_current_datetime")
_LOCAL_NAMES = ("web_fetch", "web_search", "fs_write")


class _StubSkillsService:
    """Minimal async execute_skill stand-in for _BridgeSkillRunner tests."""

    def __init__(self, allowed_tools):
        self._allowed_tools = allowed_tools

    async def execute_skill(self, name, *, mode, args):
        return {
            "rendered_prompt": f"RENDERED[{name}]({args})",
            "allowed_tools": self._allowed_tools,
        }


class _CapturingSpawn:
    def __init__(self):
        self.calls = []

    def __call__(self, task, *, allowed_tools=None):
        self.calls.append((task, allowed_tools))
        return ToolResult(ok=True, content="child done")


def _make_runner(declared):
    return _BridgeSkillRunner(
        skills_service=_StubSkillsService(declared),
        skill_names=frozenset({"web-research"}),
        builtin_names=_BUILTIN_NAMES,
        local_names=_LOCAL_NAMES,
    )


def test_skill_run_narrows_against_local_tools():
    """A skill declaring a mix of builtin and local tools gets exactly those,
    ordered by the narrowing set (builtins first, then local) -- not by the
    skill's own declaration order."""
    spawn = _CapturingSpawn()
    result = _make_runner(["web_fetch", "calculator"]).run(
        "web-research", "the question", spawn
    )
    assert result.ok
    task, allowed = spawn.calls[0]
    assert task.startswith("RENDERED[web-research]")
    assert allowed == ("calculator", "web_fetch")


def test_skill_run_undeclared_gets_builtins_and_local():
    """Declared ``None`` passes the FULL narrowing set through.

    Behavior change (phase 3c): previously an undeclared skill's child got
    builtins only; it now gets builtins + local tool names. This matches how
    native spawn_subagent children already behave (they inherit the parent's
    local tools) and stays safe because every child call resolves through
    the parent's shared review hook -- approval gating is unchanged.
    """
    spawn = _CapturingSpawn()
    result = _make_runner(None).run("web-research", "the question", spawn)
    assert result.ok
    _task, allowed = spawn.calls[0]
    assert allowed == _BUILTIN_NAMES + _LOCAL_NAMES


def test_skill_run_never_grants():
    """A skill can only narrow, never grant: runtime tools, MCP tools, and
    other skills' names declared in allowed-tools are all dropped."""
    spawn = _CapturingSpawn()
    declared = ["web_fetch", "spawn_subagent", "mcp__x__y", "other-skill"]
    result = _make_runner(declared).run("web-research", "the question", spawn)
    assert result.ok
    _task, allowed = spawn.calls[0]
    assert allowed == ("web_fetch",)


def _fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


class _ScriptedChat:
    """Returns scripted replies; mirrors Tests/Agents harness pattern."""

    def __init__(self, replies):
        self.replies = list(replies)

    def __call__(self, **kwargs):
        item = self.replies.pop(0)
        message = item if isinstance(item, dict) else {"content": item}
        return {"choices": [{"message": message}]}


def test_skill_run_child_still_approval_gated(tmp_path):
    """Wire-level: a skill child narrowed onto a local tool still resolves
    that call through the parent's shared review hook -- the wider narrowing
    set does not bypass approval gating (network cut at the handler seam,
    mirroring Tests/Agents/test_local_tools_integration.py)."""
    fetched = []

    def fake_fetch(args):
        fetched.append(dict(args))
        return "Example body"

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    specs = [
        dataclasses.replace(s, handler=fake_fetch)
        for s in _default_specs(
            workspace,
            workspace_executor=WorkspaceToolExecutor(workspace),
        )
        if s.name == "web_fetch"
    ]
    provider = LocalToolProvider(
        workspace_root=workspace,
        specs=specs,
        resolve_state=lambda hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(provider)
    registry.register_provider(
        SkillToolProvider(
            [
                {
                    "name": "web-research",
                    "description": "Researches a question.",
                    "argument_hint": "the question",
                }
            ]
        )
    )
    runner = _BridgeSkillRunner(
        skills_service=_StubSkillsService(["web_fetch"]),
        skill_names=frozenset({"web-research"}),
        builtin_names=_BUILTIN_NAMES,
        local_names=("web_fetch",),
    )
    approval_calls = []

    def request_approvals(pending):
        approval_calls.append(pending)
        return {"web_fetch": "approve_once"}

    # PR2a Task 6.5: the fleet is ON by default and a SKILL's spawn goes
    # through the same `spawn` closure as `spawn_subagent`, so the skill
    # child runs on its own thread. Addressed per agent instead of one
    # ordered queue; the child is keyed by the task text `_StubSkillsService`
    # renders for it. The replies themselves are unchanged.
    chat = FleetChat(
        [
            _fence("web-research", {"args": "the question"}),  # primary: skill
            "primary final",
        ],
        {
            "RENDERED[web-research](the question)": [
                _fence("web_fetch", {"url": "http://example.com/"}),  # child: local
                "child synthesis",  # child final
            ]
        },
    )
    service = AgentService(
        db=AgentRunsDB(tmp_path / "runs.db", client_id="t"),
        registry=registry,
        chat_call=chat,
        skill_runner=runner,
        review_tool_calls=build_local_review_hook(provider, request_approvals),
        review_state_scope=provider.stamp_scope,
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "research it"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=(
                "web-research",
                "calculator",
                "get_current_datetime",
                SPAWN_TOOL_NAME,
            ),
            budget=RunBudget(max_steps=12),
        ),
        api_endpoint="llama_cpp",  # fence-protocol endpoint (harness pattern)
    )
    assert outcome.status == RUN_DONE
    # The child's web_fetch call hit the shared review hook exactly once.
    assert len(approval_calls) == 1
    assert len(approval_calls[0]) == 1
    pending = approval_calls[0][0]
    assert pending.server_key == LOCAL_SERVER_KEY
    assert pending.llm_name == "web_fetch"
    # Approved, so the stubbed handler ran with the model's args.
    assert fetched == [{"url": "http://example.com/"}]
