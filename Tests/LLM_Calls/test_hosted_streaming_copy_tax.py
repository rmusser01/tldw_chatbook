"""Copy-strategy contracts for the hosted streaming pipeline.

The hosted stream path pays a per-chunk copy tax. These tests pin the
isolation contract the copies exist to provide, and the protocol-level
guarantees that must hold when the deepcopies are replaced with fresh
per-level shallow construction:

- Mutation isolation: a consumer may deeply mutate every event it
  receives; neither later events nor the engine's terminal state may
  change (the raw SSE record is re-parsed fresh for every chunk, internal
  accumulation stores validated strings, and terminal accounting keeps
  its own copies).
- Bit-identical visible stream: the engine-visible events serialize
  byte-identically to a golden corpus pinned on the deepcopy
  implementation, so the copy-strategy surgery cannot change protocol
  behavior.
- Enumerated mutable leaves: the only mutable (dict/list) values in the
  filtered events are the structural levels the code rebuilds fresh
  (event/choice/delta/tool-call/function) plus the ``usage`` mappings,
  whose provider payloads may nest arbitrarily (OpenRouter
  ``prompt_tokens_details``/``cost``) -- usage keeps its deepcopy.
- Per-chunk deepcopy count on the hot path is zero.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

import tldw_chatbook.LLM_Calls.hosted_provider_engine as hosted_engine
from tldw_chatbook.LLM_Calls import hosted_chat
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatStream
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedPresetFinishPolicy,
    HostedProviderStream,
    _normalize_messages,
    _normalize_tools,
    provider_event_check,
)
from tldw_chatbook.provider_registry import GROQ

# Groq's registry preset: default finish sets, reasoning disposition
# "ignored" (the engine strips reasoning_content from visible deltas).
# The displayable twin covers the keep path (zai/moonshot style).
_RECORD = GROQ
_DISPLAYABLE = replace(GROQ, reasoning_disposition="displayable")

# Realistic nested usage (OpenRouter-style): scalar counters plus nested
# mappings. The nested mappings are the mutable leaves that keep deepcopy.
_USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15,
    "prompt_tokens_details": {"cached_tokens": 0},
    "cost": 0.000125,
}

# A full text turn: identity frame with every top-level key, a bare
# content delta, a reasoning frame (stripped for ignored/proprietary
# records), a terminal choice frame carrying choice-level usage (Mistral
# shape), a trailing empty-choices usage frame (include_usage shape), and
# the sentinel.
_TEXT_LINES: tuple[dict[str, Any] | str, ...] = (
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1730000000,
        "model": "llama-3.3-70b-versatile",
        "system_fingerprint": "fp_1",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hel"},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1730000000,
        "model": "llama-3.3-70b-versatile",
        "choices": [{"index": 0, "delta": {"content": "lo"}}],
    },
    {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "usage": _USAGE}]},
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1730000000,
        "model": "llama-3.3-70b-versatile",
        "choices": [],
        "usage": _USAGE,
    },
    "[DONE]",
)

# A full tool turn: identity frame, tool-call identity delta, two
# argument-continuation deltas (index-only continuation, TASK-34364
# shape), terminal tool_calls frame with choice-level usage, trailing
# usage frame, sentinel.
_TOOL_LINES: tuple[dict[str, Any] | str, ...] = (
    {
        "id": "chatcmpl-2",
        "object": "chat.completion.chunk",
        "created": 1730000001,
        "model": "llama-3.3-70b-versatile",
        "system_fingerprint": "fp_2",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": ""},
                        }
                    ]
                },
            }
        ]
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"timezone":'}}
                    ]
                },
            }
        ]
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": '"UTC"}'}}]
                },
            }
        ]
    },
    {
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "tool_calls", "usage": _USAGE}
        ]
    },
    {
        "id": "chatcmpl-2",
        "object": "chat.completion.chunk",
        "created": 1730000001,
        "model": "llama-3.3-70b-versatile",
        "choices": [],
        "usage": _USAGE,
    },
    "[DONE]",
)

# Minimal corpora for per-chunk deepcopy counting (hot-path chunks and
# the two usage-bearing cold frames).
_CONTENT_FRAME = _TEXT_LINES[0]
_REASONING_FRAME = _TEXT_LINES[2]
_TOOL_START_FRAME = _TOOL_LINES[1]
_TOOL_ARG_FRAME = _TOOL_LINES[2]
_TERMINAL_STOP_FRAME = _TEXT_LINES[3]
_TERMINAL_TOOL_FRAME = _TOOL_LINES[4]
_TRAILING_USAGE_FRAME = _TEXT_LINES[4]


def _records(
    lines: tuple[dict[str, Any] | str, ...],
) -> Any:
    """Build the SSE record iterator exactly as the wire would deliver it."""
    return iter(
        [
            SSERecord(
                event=None,
                data=item if isinstance(item, str) else json.dumps(item),
            )
            for item in lines
        ]
    )


def _boundary_stream(record: Any, lines: tuple[dict[str, Any] | str, ...]) -> Any:
    """Compose the boundary stream with the engine's exact record wiring."""
    return HostedChatStream(
        _records(lines),
        finish_policy=HostedPresetFinishPolicy(record),
        allowed_extra_keys=record.response_allowances,
        allowed_choice_keys=record.choice_allowances,
        allowed_message_keys=record.message_allowances,
        tolerant_top_level_extras=record.tolerant_response_extras,
        usage_optional=record.stream_usage_optional,
        event_check=provider_event_check(record),
        annotation_key=record.stream_annotation_key,
    )


def _engine_stream(record: Any, lines: tuple[dict[str, Any] | str, ...]) -> Any:
    """Wrap the boundary stream in the engine's visibility layer."""
    return HostedProviderStream(_boundary_stream(record, lines), record=record)


def _canonical(event: Any) -> str:
    """Serialize one visible event canonically for byte comparison."""
    return json.dumps(event, sort_keys=True, separators=(",", ":"), default=repr)


def _poison(value: Any) -> None:
    """Deeply mutate everything reachable: overwrite every dict value,
    append to every list, so any shared structure is corrupted loudly."""
    if isinstance(value, dict):
        for key in list(value):
            _poison(value[key])
            value[key] = "<poison>"
        value["<poison-key>"] = "<poison>"
    elif isinstance(value, list):
        for item in list(value):
            _poison(item)
        value.append("<poison>")


def _container_paths(value: Any, path: tuple[Any, ...] = ()) -> set[tuple[Any, ...]]:
    """Collect the paths of every dict/list value ("*" for list items)."""
    found: set[tuple[Any, ...]] = set()
    if isinstance(value, dict):
        found.add(path)
        for key, item in value.items():
            found |= _container_paths(item, path + (key,))
    elif isinstance(value, list):
        found.add(path)
        for item in value:
            found |= _container_paths(item, path + ("*",))
    return found


# The structural levels the filter chain rebuilds fresh per level, plus
# the retained-usage subtree: the only dict/list paths a filtered event
# may contain. Everything else must be a scalar leaf.
_EXPECTED_CONTAINER_PATHS = {
    (),
    ("choices",),
    ("choices", "*"),
    ("choices", "*", "delta"),
    ("choices", "*", "delta", "tool_calls"),
    ("choices", "*", "delta", "tool_calls", "*"),
    ("choices", "*", "delta", "tool_calls", "*", "function"),
    ("choices", "*", "usage"),
    ("choices", "*", "usage", "prompt_tokens_details"),
    ("usage",),
    ("usage", "prompt_tokens_details"),
}


def _leaf_types(value: Any) -> list[type]:
    """Return the types of every scalar leaf in the structure."""
    if isinstance(value, dict):
        leaves: list[type] = []
        for item in value.values():
            leaves += _leaf_types(item)
        return leaves
    if isinstance(value, list):
        leaves = []
        for item in value:
            leaves += _leaf_types(item)
        return leaves
    return [type(value)]


# Golden pin: canonical serializations of the engine-visible events for
# the corpus above, generated by the deepcopy implementation. The
# copy-strategy surgery must keep every frame byte-identical.
_GOLDEN: dict[str, list[str]] = {
    "groq_text": [
        '{"choices":[{"delta":{"content":"Hel","role":"assistant"},"finish_reason":null,"index":0}],"created":1730000000,"id":"chatcmpl-1","model":"llama-3.3-70b-versatile","object":"chat.completion.chunk","system_fingerprint":"fp_1"}',
        '{"choices":[{"delta":{"content":"lo"},"index":0}],"created":1730000000,"id":"chatcmpl-1","model":"llama-3.3-70b-versatile","object":"chat.completion.chunk"}',
        '{"choices":[{"delta":{"content":""},"index":0}]}',
        '{"choices":[{"delta":{"content":""},"finish_reason":"stop","index":0,"usage":{"completion_tokens":5,"cost":0.000125,"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":15}}]}',
        '{"choices":[],"created":1730000000,"id":"chatcmpl-1","model":"llama-3.3-70b-versatile","object":"chat.completion.chunk","usage":{"completion_tokens":5,"cost":0.000125,"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":15}}',
    ],
    "groq_tools": [
        '{"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":null,"index":0}],"created":1730000001,"id":"chatcmpl-2","model":"llama-3.3-70b-versatile","object":"chat.completion.chunk","system_fingerprint":"fp_2"}',
        '{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"","name":"get_time"},"id":"call_1","index":0,"type":"function"}]},"index":0}]}',
        '{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\\"timezone\\":"},"index":0}]},"index":0}]}',
        '{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\\"UTC\\"}"},"index":0}]},"index":0}]}',
        '{"choices":[{"delta":{"content":""},"finish_reason":"tool_calls","index":0,"usage":{"completion_tokens":5,"cost":0.000125,"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":15}}]}',
        '{"choices":[],"created":1730000001,"id":"chatcmpl-2","model":"llama-3.3-70b-versatile","object":"chat.completion.chunk","usage":{"completion_tokens":5,"cost":0.000125,"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":15}}',
    ],
    "displayable_text": [
        '{"choices":[{"delta":{"content":"Hel","role":"assistant"},"finish_reason":null,"index":0}],"created":1730000000,"id":"chatcmpl-1","model":"llama-3.3-70b-versatile","object":"chat.completion.chunk","system_fingerprint":"fp_1"}',
        '{"choices":[{"delta":{"content":"lo"},"index":0}],"created":1730000000,"id":"chatcmpl-1","model":"llama-3.3-70b-versatile","object":"chat.completion.chunk"}',
        '{"choices":[{"delta":{"content":"","reasoning_content":"thinking"},"index":0}]}',
        '{"choices":[{"delta":{"content":""},"finish_reason":"stop","index":0,"usage":{"completion_tokens":5,"cost":0.000125,"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":15}}]}',
        '{"choices":[],"created":1730000000,"id":"chatcmpl-1","model":"llama-3.3-70b-versatile","object":"chat.completion.chunk","usage":{"completion_tokens":5,"cost":0.000125,"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":15}}',
    ],
    "displayable_tools": [
        '{"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":null,"index":0}],"created":1730000001,"id":"chatcmpl-2","model":"llama-3.3-70b-versatile","object":"chat.completion.chunk","system_fingerprint":"fp_2"}',
        '{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"","name":"get_time"},"id":"call_1","index":0,"type":"function"}]},"index":0}]}',
        '{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\\"timezone\\":"},"index":0}]},"index":0}]}',
        '{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\\"UTC\\"}"},"index":0}]},"index":0}]}',
        '{"choices":[{"delta":{"content":""},"finish_reason":"tool_calls","index":0,"usage":{"completion_tokens":5,"cost":0.000125,"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":15}}]}',
        '{"choices":[],"created":1730000001,"id":"chatcmpl-2","model":"llama-3.3-70b-versatile","object":"chat.completion.chunk","usage":{"completion_tokens":5,"cost":0.000125,"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":15}}',
    ],
}


def test_engine_stream_matches_golden_bytes() -> None:
    """Bit-identical golden pin: the visible stream frames must serialize
    byte-for-byte like the deepcopy implementation's output."""
    cases = {
        "groq_text": (_RECORD, _TEXT_LINES),
        "groq_tools": (_RECORD, _TOOL_LINES),
        "displayable_text": (_DISPLAYABLE, _TEXT_LINES),
        "displayable_tools": (_DISPLAYABLE, _TOOL_LINES),
    }
    for name, (record, lines) in cases.items():
        serialized = [
            json.dumps(event, sort_keys=True, separators=(",", ":"))
            for event in _engine_stream(record, lines)
        ]
        assert serialized == _GOLDEN[name], name


def test_consumer_mutation_cannot_corrupt_stream_or_terminal_state() -> None:
    """Deeply mutating every received event must not leak anywhere.

    Two identical streams run in lockstep: the witness is untouched, the
    victim's every yielded event (and its terminal turn) is deeply
    poisoned immediately after receipt. Each victim event must equal the
    witness event, and the victim's terminal turn must equal the
    witness's -- pinning the isolation contract the copy strategy
    provides.
    """
    for record in (_RECORD, _DISPLAYABLE):
        for lines in (_TEXT_LINES, _TOOL_LINES):
            witness = _engine_stream(record, lines)
            victim = _engine_stream(record, lines)
            for witness_event in witness:
                victim_event = next(victim)
                assert _canonical(victim_event) == _canonical(witness_event)
                _poison(victim_event)
            # The victim must still reach its own clean terminal record
            # (StopIteration) after every received event was poisoned.
            assert list(victim) == []
            witness_turn = witness.terminal_turn
            victim_turn = victim.terminal_turn
            assert victim_turn.text == witness_turn.text
            assert victim_turn.finish_reason == witness_turn.finish_reason
            assert victim_turn.reasoning_content == witness_turn.reasoning_content
            assert _canonical(victim_turn.tool_calls) == _canonical(
                witness_turn.tool_calls
            )
            assert _canonical(victim_turn.usage) == _canonical(witness_turn.usage)
            # The terminal turn is handed to the caller too: poisoning it
            # must not reach back into the stream's private state.
            witness_snapshot = deepcopy(witness_turn)
            _poison(victim_turn.assistant_message)
            if victim_turn.usage is not None:
                _poison(victim_turn.usage)
            assert witness_turn.text == witness_snapshot.text
            assert witness_turn.usage == witness_snapshot.usage
            assert witness_turn.assistant_message == witness_snapshot.assistant_message


def test_filtered_events_expose_only_enumerated_containers() -> None:
    """Leaf-immutability walk over the boundary stream's filtered events.

    Every dict/list path in the visible events must be one of the
    structural levels the filter rebuilds fresh (event/choice/delta/
    tool-call/function) or the ``usage`` subtree (whose provider payloads
    nest arbitrarily and therefore keeps its deepcopy). Every scalar leaf
    must be str/int/float/bool/None. A new mutable leaf (say, a list
    under a delta key) fails here and must either be dropped, copied, or
    added to this contract with a copy strategy decision.
    """
    for record in (_RECORD, _DISPLAYABLE):
        for lines in (_TEXT_LINES, _TOOL_LINES):
            for event in _boundary_stream(record, lines):
                assert _container_paths(event) <= _EXPECTED_CONTAINER_PATHS, (
                    f"unexpected mutable path in {event!r}"
                )
                for leaf in _leaf_types(event):
                    assert leaf in (str, int, float, bool, type(None)), (
                        f"non-scalar leaf {leaf} in {event!r}"
                    )
    # The usage-bearing frames really do exercise the nested-usage path
    # the retained deepcopy protects (guard against corpus rot).
    seen = {
        path
        for event in _boundary_stream(_RECORD, _TEXT_LINES)
        for path in _container_paths(event)
    }
    assert {("usage",), ("usage", "prompt_tokens_details")} <= seen
    assert {("choices", "*", "usage", "prompt_tokens_details")} <= {
        path
        for event in _boundary_stream(_RECORD, _TOOL_LINES)
        for path in _container_paths(event)
    }


def test_hot_path_chunks_pay_zero_deepcopy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-chunk deepcopy count: content/reasoning/tool-delta chunks pay 0.

    The two usage-bearing cold frames (terminal choice frame, trailing
    usage frame) may each retain at most two copies: the deliberate
    deepcopy kept for the mutable ``usage`` key path in the filtered
    event, plus the stream's pre-existing internal usage-accounting copy.
    """
    calls = {"count": 0}
    real_deepcopy = deepcopy

    def counting_deepcopy(value: Any) -> Any:
        calls["count"] += 1
        return real_deepcopy(value)

    monkeypatch.setattr(hosted_chat, "deepcopy", counting_deepcopy)
    monkeypatch.setattr(hosted_engine, "deepcopy", counting_deepcopy)

    def per_chunk_counts(lines: tuple[dict[str, Any] | str, ...]) -> list[int]:
        counts: list[int] = []
        stream = _engine_stream(_RECORD, lines)
        exhausted = False
        while not exhausted:
            # Zero before each pull so the terminal record's turn-building
            # copies (made while raising StopIteration on [DONE]) never
            # land on a yielded event's count.
            calls["count"] = 0
            try:
                next(stream)
            except StopIteration:
                exhausted = True
            counts.append(calls["count"])
        counts.pop()  # the [DONE] record yields no event
        return counts

    assert per_chunk_counts(
        (_CONTENT_FRAME, _TERMINAL_STOP_FRAME, _TRAILING_USAGE_FRAME, "[DONE]")
    ) == [0, 2, 2]
    assert per_chunk_counts(
        (
            _CONTENT_FRAME,
            _REASONING_FRAME,
            _TERMINAL_STOP_FRAME,
            _TRAILING_USAGE_FRAME,
            "[DONE]",
        )
    ) == [0, 0, 2, 2]
    assert per_chunk_counts(
        (
            _TOOL_START_FRAME,
            _TOOL_ARG_FRAME,
            _TOOL_LINES[3],
            _TERMINAL_TOOL_FRAME,
            _TRAILING_USAGE_FRAME,
            "[DONE]",
        )
    ) == [0, 0, 0, 2, 2]


def test_normalize_messages_survives_caller_mutation() -> None:
    """Request-message normalization is isolated from the caller's dicts."""
    caller_messages = [
        {"role": "user", "content": "what time is it in UTC?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "arguments": '{"timezone": "UTC"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
        {"role": "user", "content": "thanks"},
    ]
    normalized = _normalize_messages(_RECORD, caller_messages, system_message=None)
    expected = [
        {"role": "user", "content": "what time is it in UTC?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "arguments": '{"timezone": "UTC"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
        {"role": "user", "content": "thanks"},
    ]
    assert normalized == expected
    golden = _canonical(normalized)
    _poison(caller_messages)
    assert _canonical(normalized) == golden


def test_normalize_tools_survives_caller_mutation() -> None:
    """Tool-schema normalization is isolated from the caller's dicts.

    ``parameters`` is a caller-owned JSON schema whose leaves may nest
    arbitrarily (``required`` is a list), so that key path keeps its
    deepcopy even after the surrounding levels switch to fresh shallow
    construction.
    """
    caller_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time for a timezone.",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                },
            },
        }
    ]
    normalized = _normalize_tools(_RECORD, caller_tools)
    assert normalized == [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time for a timezone.",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                },
            },
        }
    ]
    golden = _canonical(normalized)
    _poison(caller_tools)
    assert _canonical(normalized) == golden
