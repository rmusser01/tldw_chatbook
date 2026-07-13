"""Pure streaming fence-gate: incremental visible text, fence suppression."""
import pytest

from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_stream import StreamGate


def drain(chunks):
    """Feed chunks; return (streamed_visible, full_text, tool_call)."""
    gate = StreamGate()
    streamed = "".join(gate.feed(c) for c in chunks)
    streamed += gate.flush_tail()
    visible, call = gate.result()
    return streamed, gate.full_text, visible, call


def test_plain_text_streams_verbatim():
    streamed, full, visible, call = drain(["Tok", "yo is", " the capital."])
    assert streamed == "Tokyo is the capital."
    assert full == "Tokyo is the capital."
    assert visible == "Tokyo is the capital." and call is None


def test_leading_fence_streams_nothing():
    fence = FENCE_OPEN + '\n{"name": "calculator", "arguments": {"expression": "6*7"}}\n```'
    streamed, full, visible, call = drain([fence[:6], fence[6:20], fence[20:]])
    assert streamed == ""                       # nothing visible for a tool turn
    assert full == fence                        # loop still gets the full text
    assert call is not None and call.name == "calculator"


def test_leading_fence_split_across_chunks_is_never_partially_shown():
    fence = FENCE_OPEN + '\n{"name": "x", "arguments": {}}\n```'
    gate = StreamGate()
    # First chunk is only "``" — undecided, must not stream.
    assert gate.feed("``") == ""
    assert gate.feed("`tool_call") == ""        # still undecided (could be tool_calls)
    rest = "".join(gate.feed(c) for c in [fence[len(FENCE_OPEN):]])
    assert rest == "" and gate.flush_tail() == ""


def test_mid_stream_fence_truncates_visible_at_fence():
    tail = FENCE_OPEN + '\n{"name": "calculator", "arguments": {"expression": "1"}}\n```'
    streamed, full, visible, call = drain(["Let me compute. ", tail])
    assert streamed == "Let me compute."        # rstripped prefix, fence content withheld
    assert visible == "Let me compute." and call is not None
    assert full == "Let me compute. " + tail


def test_holdback_prevents_streaming_a_fence_prefix_then_completes():
    # A message ending in text whose tail coincidentally starts like a fence prefix.
    streamed, full, visible, call = drain(["answer ", "``"])
    assert streamed == "answer ``"              # no real fence → tail flushed at end
    assert call is None and visible == "answer ``"


def test_lookalike_fence_is_treated_as_visible_text():
    streamed, full, visible, call = drain(["```python\nprint(1)\n```"])
    assert call is None
    assert streamed == "```python\nprint(1)\n```"


# --- Reviewer repro 1: a whole-turn look-alike fence must never seal the
# gate. `buf.find(FENCE_OPEN)` alone matches "```tool_calls" (plural) and
# "```tool_call_schema", neither of which is a real fence.

def test_lookalike_whole_turn_streams_everything_single_chunk():
    text = "```tool_calls\nSome explanation of the tools available.\n```"
    streamed, full, visible, call = drain([text])
    assert call is None
    assert visible == text
    assert streamed == text


def test_lookalike_whole_turn_streams_everything_char_by_char():
    text = "```tool_calls\nSome explanation of the tools available.\n```"
    streamed, full, visible, call = drain(list(text))
    assert call is None
    assert visible == text
    assert streamed == text


# --- Reviewer repro 2: a malformed mid-stream fence (clean tag line, but
# the JSON body doesn't parse) must not drop the text that follows it —
# `split_visible_text_and_tool_call` scans past it and finds no real fence,
# so the whole turn is visible.

def test_malformed_mid_stream_fence_flushes_everything_single_chunk():
    text = "Before text. ```tool_call\nnot json\n``` after text."
    streamed, full, visible, call = drain([text])
    assert call is None
    assert visible == text
    assert streamed == text


def test_malformed_mid_stream_fence_flushes_everything_char_by_char():
    text = "Before text. ```tool_call\nnot json\n``` after text."
    streamed, full, visible, call = drain(list(text))
    assert call is None
    assert visible == text
    assert streamed == text


# --- Invariant: no matter how a turn is chunked, whatever streamed live
# plus whatever flush_tail() releases at the end must equal the
# authoritative visible text. This is what "finalize is the single
# authority, nothing is ever silently dropped" means operationally.

_INVARIANT_TEXTS = [
    "Tokyo is the capital.",
    FENCE_OPEN + '\n{"name": "x", "arguments": {}}\n```',
    FENCE_OPEN + '\nnot json\n```',
    "Let me compute. " + FENCE_OPEN
        + '\n{"name": "calculator", "arguments": {"expression": "1"}}\n```',
    "Before text. ```tool_call\nnot json\n``` after text.",
    "```tool_calls\nSome explanation of the tools available.\n```",
    "```python\nprint(1)\n```",
    "answer ``",
    # A disproven look-alike immediately followed by a genuine confirmed
    # fence — stresses whitespace-holdback interacting with two candidates.
    "```tool_calls\n``` Actually, let me call it: " + FENCE_OPEN
        + '\n{"name": "x", "arguments": {}}\n```',
]


def _chunkings(text):
    yield [text]
    yield list(text)
    mid = len(text) // 2
    if 0 < mid < len(text):
        yield [text[:mid], text[mid:]]


@pytest.mark.parametrize("text", _INVARIANT_TEXTS)
def test_streamed_plus_tail_always_equals_visible(text):
    for chunks in _chunkings(text):
        gate = StreamGate()
        streamed = "".join(gate.feed(c) for c in chunks)
        tail = gate.flush_tail()
        visible, _call = gate.result()
        assert streamed + tail == visible, chunks
