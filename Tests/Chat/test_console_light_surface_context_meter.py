"""TASK-34100.5.3 AC#5: the context meter counts the tools the send carries.

The Console send preflight (``ConsoleCompactionPreflight``) asks the agent
bridge for ``preview_tool_schemas`` and hands those rows to the request
accounting, whose context breakdown shows a "Tool schemas" row. The bridge
never implemented it, so the meter counted no tools while the request carried
the whole agent surface (TASK-34100.5, AC#6 notes).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_light_tool_surface import (
    LIGHT,
    _fresh_tool_gates,  # noqa: F401 -- autouse: fresh gates per test
    _native_names,
    _production_bridge,
    _send,
)
from tldw_chatbook.Chat.console_agent_bridge import build_console_first_request_plan
from tldw_chatbook.Chat.console_context_compaction import ConsoleCompactionPreflight
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.Utils.token_counter import ContextWindowResolution

pytestmark = pytest.mark.bootstrap_profile


class _Captured(Exception):
    """Stops the preflight once ``prepare`` has seen the tool rows."""


def _preflight_tools(bridge, resolution) -> list[dict]:
    """Drive the real capacity preflight up to ``prepare`` and return the
    ``tools=`` rows it would account."""
    names = inspect.signature(ConsoleCompactionPreflight).parameters
    kwargs = {name: (lambda: None) for name in names}
    kwargs["read_controller__agent_bridge"] = lambda: bridge
    preflight = ConsoleCompactionPreflight(**kwargs)
    seen: list[list[dict]] = []

    def prepare(_resolution, _messages, *, tools=(), **_kwargs):
        seen.append(list(tools))
        raise _Captured

    with pytest.raises(_Captured):
        preflight._assess_request_capacity_only(
            session_id="s",
            owner=SimpleNamespace(context_policy_overrides=None),
            resolution=resolution,
            provider_messages=[{"role": "user", "content": "Reply with just: hi"}],
            prepare=prepare,
            agent_tools_enabled=True,
            assessment_sink=lambda *_args: None,
        )
    return seen[0]


def _openai(window: ContextWindowResolution):
    return SimpleNamespace(
        provider="openai",
        execution_key="openai",
        model="gpt-4.1-nano",
        max_tokens=4096,
        context_window=window,
        reasoning_replay=None,
    )


def _llama(window: ContextWindowResolution):
    return SimpleNamespace(
        provider="llama_cpp",
        execution_key="llama_cpp",
        model="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        max_tokens=4096,
        context_window=window,
        reasoning_replay=None,
    )


class _AskUserProvider:
    """The local provider as a fresh profile composes it for the light
    surface's purposes: ``ask_user`` with its real schema."""

    def __init__(self):
        from tldw_chatbook.Agents.local_tool_provider import (
            light_surface_local_schemas,
        )

        self._schemas = {schema.name: schema for schema in light_surface_local_schemas()}

    def list_catalog(self):
        from tldw_chatbook.Agents.agent_models import ToolCatalogEntry

        return [
            ToolCatalogEntry(
                id=schema.id,
                name=name,
                one_line_description=schema.description,
                source="local",
            )
            for name, schema in self._schemas.items()
        ]

    def load_schema(self, tool_id):
        return next(s for s in self._schemas.values() if s.id == tool_id)

    def invoke(self, tool_id, args):  # pragma: no cover - never dispatched
        raise AssertionError("planning only")


def _row_names(rows) -> set[str]:
    return {row["function"]["name"] for row in rows}


def test_the_preflight_counts_the_light_surface_the_send_carries(tmp_path):
    """A fresh profile, a model with a known large window: the preflight's
    accounting gets exactly the light set's rows."""
    bridge, _gateway, _session, _assistant = _production_bridge(tmp_path, [])

    rows = _preflight_tools(
        bridge, _openai(ContextWindowResolution(1_047_576, "model catalog", True))
    )

    assert _row_names(rows) == LIGHT


def test_a_window_too_small_for_the_light_set_counts_what_the_plan_sends(tmp_path):
    """A llama.cpp server whose window nobody verified is planned as 4,096
    tokens, where the light set does not fit and the request carries no
    tools. The meter agrees with the planner either way."""
    bridge, _gateway, _session, _assistant = _production_bridge(tmp_path, [])
    resolution = _llama(ContextWindowResolution(32_000, "application fallback", False))

    rows = _preflight_tools(bridge, resolution)
    # The same window with a verified, roomy size: the set fits, so the
    # all-or-nothing comparison below is not vacuous.
    roomy = _preflight_tools(
        bridge, _llama(ContextWindowResolution(32_768, "server metadata", True))
    )
    assert _row_names(roomy) == LIGHT
    plan = build_console_first_request_plan(
        shared_registry=bridge._registry,
        shared_allowed_tools=bridge._allowed_tools,
        context={},
        skills_present=False,
        mcp_provider=None,
        builtin_gate=object(),
        local_provider=_AskUserProvider(),
        library_provider=None,
        library_authority=None,
        workspace_id=None,
        ephemeral=False,
        diff_sink=None,
        scratch_root=None,
        scratch_lease=None,
        resolution=resolution,
        fallback_model=resolution.model,
        session_system_prompt="",
        native_tools=True,
        turn_skill_bindings=(),
        turn_bundle_block="",
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        agent_messages=[{"role": "user", "content": "Reply with just: hi"}],
        light_surface=True,
    )
    sent = {
        schema.name
        for schema in (*plan.schemas.runtime_schemas, *plan.schemas.active_schemas)
    }

    assert _row_names(rows) == sent
    # Small local models get the minimal prompt (owner ruling): no tools.
    assert sent == set()


def test_with_full_agent_tools_on_the_meter_counts_the_last_sent_surface(tmp_path):
    """The full surface is composed per send; once one went out, the meter
    counts exactly what it carried."""
    assert save_setting_to_cli_config("tools", "full_agent_tools_enabled", True)
    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])
    _send(bridge, session, assistant_id, _openai_native())

    rows = bridge.preview_tool_schemas(
        _openai(ContextWindowResolution(1_047_576, "model catalog", True))
    )

    assert _row_names(rows) == _native_names(gateway)
    assert "spawn_subagent" in _row_names(rows)


def _openai_native():
    from Tests.Chat.test_console_agent_bridge import _native_resolution

    return _native_resolution()
