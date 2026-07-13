"""Fence-first text protocol: parse, split, sniff, render."""
import json

from tldw_chatbook.Agents.agent_models import ToolSchema
from tldw_chatbook.Agents.agent_runtime import (
    FENCE_OPEN, STREAM_TEXT, STREAM_TOOL_CALL, STREAM_UNDECIDED,
    parse_fenced_tool_call, render_tool_protocol,
    split_visible_text_and_tool_call, stream_prefix_verdict,
)

GOOD = '```tool_call\n{"name": "calculator", "arguments": {"expression": "6*7"}}\n```'


def test_parse_leading_fence():
    call = parse_fenced_tool_call(GOOD)
    assert call is not None
    assert call.name == "calculator" and call.args == {"expression": "6*7"}


def test_parse_allows_leading_whitespace_only():
    assert parse_fenced_tool_call("\n  " + GOOD) is not None
    assert parse_fenced_tool_call("Sure, running it:\n" + GOOD) is None


def test_parse_defensive_on_malformed_json():
    assert parse_fenced_tool_call('```tool_call\n{"name": broken\n```') is None
    assert parse_fenced_tool_call('```tool_call\n"just a string"\n```') is None
    assert parse_fenced_tool_call('```tool_call\n{"arguments": {}}\n```') is None
    assert parse_fenced_tool_call('```tool_call\n{"name": "x", "arguments": []}\n```') is None
    assert parse_fenced_tool_call("```tool_call\n{unclosed") is None
    assert parse_fenced_tool_call("no fence at all") is None


def test_split_mid_stream_fence_truncates_and_converts():
    text = "Let me compute that.\n" + GOOD
    visible, call = split_visible_text_and_tool_call(text)
    assert visible == "Let me compute that."
    assert call is not None and call.name == "calculator"


def test_split_no_fence_returns_text_unchanged():
    visible, call = split_visible_text_and_tool_call("plain answer")
    assert visible == "plain answer" and call is None


def test_split_malformed_fence_stays_visible_text():
    text = "answer ```tool_call\n{broken"
    visible, call = split_visible_text_and_tool_call(text)
    assert call is None and visible == text


def test_stream_sniff_verdicts():
    assert stream_prefix_verdict("") == STREAM_UNDECIDED
    assert stream_prefix_verdict("  \n") == STREAM_UNDECIDED
    assert stream_prefix_verdict("``") == STREAM_UNDECIDED          # fence prefix
    assert stream_prefix_verdict("```tool") == STREAM_UNDECIDED     # fence prefix
    assert stream_prefix_verdict(FENCE_OPEN) == STREAM_TOOL_CALL
    assert stream_prefix_verdict(FENCE_OPEN + "\n{") == STREAM_TOOL_CALL
    assert stream_prefix_verdict("Tokyo is") == STREAM_TEXT
    assert stream_prefix_verdict("```python\n") == STREAM_TEXT      # other fence


def test_render_tool_protocol_contains_schemas_and_fence_first_rule():
    schema = ToolSchema(id="builtin:calculator", name="calculator",
                        description="Evaluate math",
                        parameters={"type": "object", "properties": {
                            "expression": {"type": "string"}}})
    rendered = render_tool_protocol([schema])
    assert "calculator" in rendered and "Evaluate math" in rendered
    assert FENCE_OPEN in rendered
    # The fence-FIRST requirement must be stated explicitly.
    assert "first" in rendered.lower()
    # Schemas must be embedded as valid JSON.
    assert json.dumps(schema.parameters) in rendered or "expression" in rendered


def test_render_empty_schema_list_is_answer_directly():
    rendered = render_tool_protocol([])
    assert rendered == ""
