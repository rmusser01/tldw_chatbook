"""TASK-34100.5 AC#6 (new-entry-exit-handoff-01): plain chat to a small server.

Live, 2026-10-03: the first 'Reply with just: hi' to a llama.cpp server
started with ``-c 4096`` carried a ~20.8 KB system prompt (the agent operating
prompt plus the fenced protocol for 16 runtime tools) and llama-server
answered ``request (4730 tokens) exceeds the available context size (4096
tokens)``. The agent planner sized that preamble against the 32,000-token
application fallback -- a window nobody had verified -- while the Console's
own send preflight used the window attached to the send's resolution.

The planner now plans against the same window the send resolved. A window
the app cannot vouch for never licenses the full preamble: it is planned as
the smallest common local window, and when the tools do not fit the request
carries only the session's own system prompt.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_runtime import render_tool_protocol
from tldw_chatbook.Agents.tool_catalog import (
    ToolCatalogEntry,
    ToolCatalogRegistry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Chat.console_agent_bridge import build_console_first_request_plan
from tldw_chatbook.Utils.token_counter import ContextWindowResolution

pytestmark = pytest.mark.bootstrap_profile

_MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
_SESSION_PROMPT = "You are a helpful AI assistant."


class _BuiltinLikeProvider:
    """Sixteen ask-gated tools with real-sized descriptions, like the
    built-in MCP source a fresh profile exposes."""

    def __init__(self) -> None:
        self._names = tuple(f"notes_tool_{index}" for index in range(16))

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=f"mcp:{name}",
                name=name,
                one_line_description="Read or change the user's notes.",
                source="mcp",
            )
            for name in self._names
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=tool_id.split(":", 1)[1],
            description="Read or change the user's notes. " * 12,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to find."},
                    "limit": {"type": "integer", "description": "How many."},
                },
            },
        )

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content="ok")


def _plan(window: ContextWindowResolution | None, *, max_tokens: int = 4096):
    registry = ToolCatalogRegistry()
    registry.register_provider(_BuiltinLikeProvider())
    allowed = tuple(entry.name for entry in registry.list_catalog())
    resolution = SimpleNamespace(
        model=_MODEL,
        execution_key="llama_cpp",
        provider="llama_cpp",
        max_tokens=max_tokens,
        context_window=window,
    )
    return build_console_first_request_plan(
        shared_registry=registry,
        shared_allowed_tools=allowed,
        context={},
        skills_present=False,
        mcp_provider=None,
        builtin_gate=None,
        local_provider=None,
        library_provider=None,
        library_authority=None,
        workspace_id=None,
        ephemeral=False,
        diff_sink=None,
        scratch_root=None,
        scratch_lease=None,
        resolution=resolution,
        fallback_model=_MODEL,
        session_system_prompt=_SESSION_PROMPT,
        native_tools=False,
        turn_skill_bindings=(),
        turn_bundle_block="",
        install_skill_enabled=True,
        run_skill_script_enabled=True,
        fork_chat_enabled=True,
        new_chat_enabled=True,
        worktree_merge_enabled=True,
        agent_messages=[{"role": "user", "content": "Reply with just: hi"}],
        fleet_max_live=4,
    )


def _system_content(plan) -> str:
    schemas = [*plan.schemas.runtime_schemas, *plan.schemas.active_schemas]
    protocol = render_tool_protocol(schemas)
    prompt = plan.schemas.system_prompt
    return f"{prompt}\n\n{protocol}" if protocol else prompt


def test_an_unverified_window_never_licenses_the_full_agent_preamble() -> None:
    """The shipped first-run shape: llama.cpp, a model no catalog knows, the
    32,000-token application fallback, max_tokens 4,096."""

    plan = _plan(ContextWindowResolution(32000, "application fallback", False))

    content = _system_content(plan)
    # A llama-server started with -c 4096 must be able to take it with room
    # for the reply: the live failure was 4,730 tokens (~20.8 KB).
    assert len(content) < 1_000, (len(content), content[:300])
    assert plan.schemas.runtime_schemas == ()
    assert plan.schemas.active_schemas == ()
    assert plan.schemas.request_fits
    # The request carries the session's own prompt, not tool instructions
    # for a protocol it no longer describes.
    assert plan.schemas.system_prompt == _SESSION_PROMPT


def test_a_detected_small_window_plans_against_the_server_value() -> None:
    """llama.cpp reported n_ctx 4096 through /props: the planner uses it,
    not the 32,000-token fallback."""

    plan = _plan(
        ContextWindowResolution(4096, "server metadata", True), max_tokens=1024
    )

    assert len(_system_content(plan)) < 1_000
    assert plan.schemas.request_fits


def test_a_verified_large_window_keeps_the_agent_tools() -> None:
    """A model the catalog knows to be large keeps its tools: the fix sizes
    the preamble, it does not remove agent features."""

    plan = _plan(ContextWindowResolution(1_047_576, "model catalog", True))

    names = {schema.name for schema in plan.schemas.runtime_schemas} | {
        schema.name for schema in plan.schemas.active_schemas
    }
    assert "spawn_subagent" in names
    assert any(name.startswith("notes_tool_") for name in names) or (
        "find_tools" in names
    )


@pytest.mark.parametrize("window", [None])
def test_callers_without_a_window_keep_the_legacy_plan(window) -> None:
    """Sub-agents and other callers pass no resolution window; their plans
    are unchanged by this fix."""

    plan = _plan(window)

    assert plan.schemas.runtime_schemas, "legacy planning still discloses tools"
