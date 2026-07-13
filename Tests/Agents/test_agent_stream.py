"""Pure streaming fence-gate: incremental visible text, fence suppression."""
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
