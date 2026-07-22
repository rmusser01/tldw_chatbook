"""Registry overrides must reach the agent runtime; identity contracts must
survive overrides. Real code paths; no accessor mocks."""

from tldw_chatbook.Internal_Prompts import get_internal_prompt


def _history_with_system(content: str) -> list[dict]:
    """Build the messages_payload shape ``_is_subagent`` inspects: a list of
    role/content dicts with a leading system message (matches the real
    payload shape assembled by agent_service.call_model / the streaming
    adapter's chat_call: ``[{"role": "system", "content": ...}, ...]``)."""
    return [{"role": "system", "content": content}]


def test_tool_protocol_override_reaches_render(scratch_config):
    from tldw_chatbook.Agents.agent_runtime import (
        FENCE_OPEN,
        _FENCE_CLOSE,
        render_tool_protocol,
    )
    from tldw_chatbook.Agents.agent_models import ToolSchema

    scratch_config(
        "[internal_prompts.agents]\n"
        'tool_protocol = "TOOLS: {tool_list} OPEN {fence_open} CLOSE {fence_close}"\n'
    )
    out = render_tool_protocol(
        [ToolSchema(id="t", name="t", description="d", parameters={})]
    )
    assert out.startswith("TOOLS: ")
    assert FENCE_OPEN in out and _FENCE_CLOSE in out
    assert '"name": "t"' in out


def test_tool_protocol_empty_schemas_still_empty(scratch_config):
    from tldw_chatbook.Agents.agent_runtime import render_tool_protocol

    scratch_config(
        '[internal_prompts.agents]\ntool_protocol = "{tool_list}{fence_open}{fence_close}"\n'
    )
    assert render_tool_protocol([]) == ""


def test_compose_uses_override_and_default(scratch_config):
    from tldw_chatbook.Chat.console_agent_bridge import (
        CONSOLE_AGENT_OPERATING_PROMPT,
        compose_agent_system_prompt,
    )

    assert compose_agent_system_prompt("") == CONSOLE_AGENT_OPERATING_PROMPT
    scratch_config(
        '[internal_prompts.agents]\nconsole_agent_operating = "CUSTOM OPERATING"\n'
    )
    assert compose_agent_system_prompt("") == "CUSTOM OPERATING"
    assert compose_agent_system_prompt("S") == "S\n\nCUSTOM OPERATING"


def test_is_subagent_detects_overridden_and_shipped_prefix(scratch_config):
    from tldw_chatbook.Chat import console_agent_bridge as bridge

    shipped = bridge.SUBAGENT_SYSTEM_PROMPT
    scratch_config(
        '[internal_prompts.agents]\nsubagent_system = "CUSTOM SUBAGENT RULES"\n'
    )
    resolved = get_internal_prompt("agents.subagent_system")
    assert resolved == "CUSTOM SUBAGENT RULES"
    # Detection accepts BOTH the live-resolved and the shipped prefix.
    assert (
        bridge._StreamingModelAdapter._is_subagent(_history_with_system(resolved))
        is True
    )
    assert (
        bridge._StreamingModelAdapter._is_subagent(_history_with_system(shipped))
        is True
    )


def test_is_subagent_false_for_ordinary_system_turn_under_active_override(
    scratch_config,
):
    from tldw_chatbook.Chat import console_agent_bridge as bridge

    scratch_config(
        '[internal_prompts.agents]\nsubagent_system = "CUSTOM SUBAGENT RULES"\n'
    )
    # An ordinary Console session system prompt sharing no prefix with either
    # the override or the shipped subagent prompt must NOT be classified as a
    # subagent turn.
    ordinary = "You are a helpful assistant for this Console session."
    assert (
        bridge._StreamingModelAdapter._is_subagent(_history_with_system(ordinary))
        is False
    )


def test_is_subagent_stable_across_mid_run_override_change(scratch_config):
    """A sub-agent's system prompt is baked into its messages_payload once,
    at spawn time (Agents/agent_service.py:411's own ``get_internal_prompt``
    call), and then stays fixed across that sub-agent's own multi-step
    tool loop. If ``agents.subagent_system`` is edited live (e.g. from
    Settings, on the UI thread, while a run is in flight on a worker
    thread) to a *different* override -- not reverted to the shipped
    default -- a detector that only compares against the CURRENTLY
    resolved value plus the shipped default stops recognizing that
    already-spawned sub-agent's later turns, and its streamed output
    leaks into the primary transcript. Detection must stay stable for any
    ``agents.subagent_system`` value resolved earlier in the same
    process, not just the one active right now."""
    from tldw_chatbook.Chat import console_agent_bridge as bridge

    scratch_config('[internal_prompts.agents]\nsubagent_system = "OVERRIDE A"\n')
    spawn_time_resolved = get_internal_prompt("agents.subagent_system")
    assert spawn_time_resolved == "OVERRIDE A"
    # This is the sub-agent's system message, fixed at spawn time.
    payload = _history_with_system(spawn_time_resolved + "\n\nRun the tool loop.")
    # Turn 1 (spawn turn): the currently active override matches directly.
    assert bridge._StreamingModelAdapter._is_subagent(payload) is True

    # Config is edited mid-run to a *different* override (not removed).
    scratch_config('[internal_prompts.agents]\nsubagent_system = "OVERRIDE B"\n')
    assert get_internal_prompt("agents.subagent_system") == "OVERRIDE B"

    # Turn 2: same sub-agent, same (unchanged) spawn-time system content --
    # must still be classified as a sub-agent turn.
    assert bridge._StreamingModelAdapter._is_subagent(payload) is True
