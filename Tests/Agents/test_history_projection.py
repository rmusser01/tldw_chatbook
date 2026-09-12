"""Accumulated history is provider-shaped and must be projected on a switch.

ADR-110 / TASK-25902. `_append_tool_result` writes a `role="tool"` message
paired by `tool_call_id` for native providers, and a `FENCE_TOOL_RESULT_PREFIX`
user message for everyone else. Handing one protocol's history to the other
does not fail loudly -- it produces a confused model -- so the switch projects.

The invariant that matters most is *totality*: every exchange present before a
switch is present after it, in order. A projection that silently drops a turn
is worse than refusing the fallback.
"""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Agents.agent_models import FENCE_TOOL_RESULT_PREFIX
from tldw_chatbook.Agents.history_projection import (
    ProjectionError,
    project_history_for_protocol,
)


def _native_call(call_id, name, args):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _native_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _fence_call(name, args):
    body = json.dumps({"name": name, "arguments": args})
    return {"role": "assistant", "content": f"```tool_call\n{body}\n```"}


def _fence_result(name, content):
    return {
        "role": "user",
        "content": f"{FENCE_TOOL_RESULT_PREFIX}{name}: {content}",
    }


NATIVE_HISTORY = [
    {"role": "user", "content": "what is 6*7?"},
    _native_call("call_1", "calculator", {"expression": "6*7"}),
    _native_result("call_1", "42"),
    {"role": "assistant", "content": "It is 42."},
]

FENCE_HISTORY = [
    {"role": "user", "content": "what is 6*7?"},
    _fence_call("calculator", {"expression": "6*7"}),
    _fence_result("calculator", "42"),
    {"role": "assistant", "content": "It is 42."},
]


# --- totality: nothing is lost -------------------------------------------


@pytest.mark.parametrize("native", [True, False])
def test_projection_preserves_plain_turns_untouched(native):
    plain = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    assert project_history_for_protocol(plain, native=native) == plain


def test_native_to_fence_keeps_every_exchange_in_order():
    out = project_history_for_protocol(NATIVE_HISTORY, native=False)

    assert len(out) == len(NATIVE_HISTORY)
    assert out[0] == NATIVE_HISTORY[0]
    assert "calculator" in out[1]["content"]
    assert out[2]["role"] == "user"
    assert out[2]["content"].startswith(FENCE_TOOL_RESULT_PREFIX)
    assert "42" in out[2]["content"]
    assert out[3] == NATIVE_HISTORY[3]


def test_fence_to_native_keeps_every_exchange_in_order():
    out = project_history_for_protocol(FENCE_HISTORY, native=True)

    assert len(out) == len(FENCE_HISTORY)
    assert out[1]["tool_calls"][0]["function"]["name"] == "calculator"
    assert out[2]["role"] == "tool"
    assert out[2]["content"] == "42"
    assert out[2]["tool_call_id"] == out[1]["tool_calls"][0]["id"]


def test_no_message_is_ever_dropped():
    """The property the ADR treats as non-negotiable."""
    for history, target in ((NATIVE_HISTORY, False), (FENCE_HISTORY, True)):
        assert len(project_history_for_protocol(history, native=target)) == len(
            history
        )


# --- round trip -----------------------------------------------------------


def test_native_round_trip_is_semantically_stable():
    once = project_history_for_protocol(NATIVE_HISTORY, native=False)
    twice = project_history_for_protocol(once, native=True)

    assert len(twice) == len(NATIVE_HISTORY)
    assert twice[1]["tool_calls"][0]["function"]["name"] == "calculator"
    assert json.loads(twice[1]["tool_calls"][0]["function"]["arguments"]) == {
        "expression": "6*7"
    }
    assert twice[2]["content"] == "42"


def test_fence_round_trip_is_semantically_stable():
    once = project_history_for_protocol(FENCE_HISTORY, native=True)
    twice = project_history_for_protocol(once, native=False)

    assert [m["role"] for m in twice] == [m["role"] for m in FENCE_HISTORY]
    assert "calculator" in twice[1]["content"]
    assert "42" in twice[2]["content"]


def test_projecting_to_the_current_protocol_is_a_no_op():
    assert project_history_for_protocol(NATIVE_HISTORY, native=True) == NATIVE_HISTORY
    assert project_history_for_protocol(FENCE_HISTORY, native=False) == FENCE_HISTORY


# --- the awkward cases ----------------------------------------------------


def test_an_unpaired_call_projects_with_a_no_result_marker():
    """ADR-110: a call whose result never arrived must not vanish."""
    history = [
        {"role": "user", "content": "go"},
        _native_call("call_orphan", "calculator", {"expression": "1+1"}),
    ]

    out = project_history_for_protocol(history, native=False)

    assert len(out) == 2
    assert "calculator" in out[1]["content"]
    assert "no result" in out[1]["content"].lower()


def test_a_multi_call_batch_projects_every_call_and_result():
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "one", "arguments": "{}"},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "two", "arguments": "{}"},
                },
            ],
        },
        _native_result("a", "first"),
        _native_result("b", "second"),
    ]

    out = project_history_for_protocol(history, native=False)

    joined = " ".join(m["content"] for m in out)
    for token in ("one", "two", "first", "second"):
        assert token in joined


def test_look_alike_fence_is_left_as_text():
    """```tool_calls is not ```tool_call -- the loop already guards this."""
    history = [
        {
            "role": "assistant",
            "content": "```tool_calls\nnot a real call\n```",
        }
    ]

    out = project_history_for_protocol(history, native=True)

    assert out == history, "a look-alike must not become a tool call"


def test_assistant_text_alongside_a_native_call_is_preserved():
    history = [
        {
            "role": "assistant",
            "content": "Let me compute that.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": "{}"},
                }
            ],
        },
        _native_result("c1", "42"),
    ]

    out = project_history_for_protocol(history, native=False)

    assert "Let me compute that." in out[0]["content"]


def test_input_is_never_mutated():
    original = json.loads(json.dumps(NATIVE_HISTORY))

    project_history_for_protocol(NATIVE_HISTORY, native=False)

    assert NATIVE_HISTORY == original


def test_unprojectable_history_raises_rather_than_degrading():
    """ADR-110: refuse the fallback rather than send a confused history."""
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "x", "type": "function"}],  # no function body
        }
    ]

    with pytest.raises(ProjectionError):
        project_history_for_protocol(history, native=False)


# --- ADR-110 AC#11: driven from the real provider list, not a copy ---------


def test_every_native_provider_round_trips():
    """Driven from `provider_supports_native_tools`' own source of truth.

    A hand-copied list here would let someone add a provider to the native set
    and quietly create a projection gap. This iterates the real one, so the
    coverage grows with it.
    """
    from tldw_chatbook.Agents.native_tools import (
        NATIVE_TOOLS_PROVIDERS,
        provider_supports_native_tools,
    )
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    # Review I5: the first version iterated a hand-written six-name list and
    # only derived each name's protocol from the function -- adding a provider
    # to the native set grew nothing, while the AC claimed it would. This one
    # is genuinely driven from the sources of truth: every native provider,
    # plus every fence provider the dispatcher knows.
    fence_providers = sorted(
        p for p in API_CALL_HANDLERS if not provider_supports_native_tools(p)
    )
    candidates = sorted(NATIVE_TOOLS_PROVIDERS) + fence_providers
    checked_native = checked_fence = 0

    for provider in candidates:
        native = provider_supports_native_tools(provider)
        source = NATIVE_HISTORY if native else FENCE_HISTORY

        out = project_history_for_protocol(source, native=native)
        assert len(out) == len(source), provider

        other = project_history_for_protocol(source, native=not native)
        assert len(other) == len(source), provider

        back = project_history_for_protocol(other, native=native)
        assert len(back) == len(source), provider
        joined = json.dumps(back)
        assert "calculator" in joined and "42" in joined, provider

        if native:
            checked_native += 1
        else:
            checked_fence += 1

    assert checked_native >= 1, "no native provider exercised"
    assert checked_fence >= 1, "no fence provider exercised"


def test_mixed_narration_and_fence_projects_to_native():
    """Review I-1: the DOMINANT real fence shape is narration + fence in one
    assistant turn (the loop appends fence turns verbatim). The first
    implementation's strict parser returned None for it, so real fence
    histories silently stayed fence-shaped after a fence->native switch."""
    mixed = {
        "role": "assistant",
        "content": "Let me compute that first.\n" + _fence_call(
            "calculator", {"expression": "6*7"}
        )["content"],
    }
    history = [
        {"role": "user", "content": "6*7?"},
        mixed,
        _fence_result("calculator", "42"),
    ]

    out = project_history_for_protocol(history, native=True)

    assert out[1].get("tool_calls"), "the fence must become a native call"
    assert out[1]["tool_calls"][0]["function"]["name"] == "calculator"
    assert "Let me compute that first." in out[1]["content"], (
        "the narration must survive beside the call"
    )
    assert out[2]["role"] == "tool"
    assert out[2]["content"] == "42"
    assert out[2]["tool_call_id"] == out[1]["tool_calls"][0]["id"]


def test_mixed_turn_round_trip_is_structurally_stable():
    mixed_history = [
        {"role": "user", "content": "6*7?"},
        {
            "role": "assistant",
            "content": "Working on it.\n"
            + _fence_call("calculator", {"expression": "6*7"})["content"],
        },
        _fence_result("calculator", "42"),
    ]

    once = project_history_for_protocol(mixed_history, native=True)
    twice = project_history_for_protocol(once, native=False)

    assert len(twice) == len(mixed_history)
    assert [m["role"] for m in twice] == ["user", "assistant", "user"]
    # structural, not substring: the fence body must parse back to the call
    from tldw_chatbook.Agents.agent_runtime import (
        split_visible_text_and_tool_call,
    )

    visible, call = split_visible_text_and_tool_call(twice[1]["content"])
    assert call is not None and call.name == "calculator"
    assert "Working on it." in visible
    assert twice[2]["content"].startswith(FENCE_TOOL_RESULT_PREFIX)


# --- review A-1 (2026-08-31): content after the first fence must survive ----


def test_no_result_marker_round_trips_without_dangling_tool_calls():
    """A no-result call projects with a marker INSIDE the assistant message;
    round-tripping back to native must not lose it -- the first version
    parsed only up to the first fence close, yielding an assistant
    `tool_calls` turn with NO role:"tool" follower, a shape OpenAI-compatible
    backends reject outright. That failure sat on exactly the second fallback
    hop -- the disaster-recovery path."""
    native = [
        {"role": "user", "content": "go"},
        _native_call("call_orphan", "calculator", {"expression": "1+1"}),
    ]

    fence = project_history_for_protocol(native, native=False)
    back = project_history_for_protocol(fence, native=True)

    dangling = [
        m for m in back
        if m.get("tool_calls")
        and not any(
            r.get("role") == "tool"
            and r.get("tool_call_id") == m["tool_calls"][0]["id"]
            for r in back
        )
    ]
    assert not dangling, (
        "round-trip produced an assistant tool_calls turn with no paired "
        "role:'tool' message -- providers 400 on this shape"
    )


def test_two_call_batch_round_trips_both_calls():
    """2 calls in must be 2 calls out -- the first version dropped call b's
    request entirely, making the model believe it never asked (the precise
    failure ADR-110 decision 2's marker exists to prevent)."""
    native = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "one", "arguments": "{}"},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "two", "arguments": "{}"},
                },
            ],
        },
        _native_result("a", "first"),
        _native_result("b", "second"),
    ]

    fence = project_history_for_protocol(native, native=False)
    back = project_history_for_protocol(fence, native=True)

    names = [
        entry["function"]["name"]
        for m in back
        for entry in (m.get("tool_calls") or ())
    ]
    assert sorted(names) == ["one", "two"], f"a call vanished: {names}"
    tool_contents = sorted(
        m["content"] for m in back if m.get("role") == "tool"
    )
    assert tool_contents == ["first", "second"]
