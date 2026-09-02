"""ADR-090: rationale capture normalization + ToolCall field."""

from tldw_chatbook.Agents.agent_models import (
    RATIONALE_CAPTURE_CAP,
    ToolCall,
    normalize_rationale,
)


def test_normalize_strips_control_chars_and_collapses_whitespace():
    assert normalize_rationale("line1\n\tline2\x00\x1fend") == "line1 line2 end"


def test_normalize_keeps_the_tail_when_over_cap():
    out = normalize_rationale("A" * 300 + "B" * 300)
    assert len(out) == RATIONALE_CAPTURE_CAP
    assert out.startswith("\N{HORIZONTAL ELLIPSIS}")
    assert out.endswith("B")


def test_normalize_ignores_non_strings_and_blank_text():
    assert normalize_rationale(None) == ""
    assert normalize_rationale(123) == ""
    assert normalize_rationale("  \n \t ") == ""


def test_tool_call_rationale_defaults_empty():
    assert ToolCall(name="fs_list", args={}).rationale == ""


# ---------------------------------------------------------------------------
# with_preamble_rationale + parse_fenced_tool_call (ADR-090 hybrid capture)
# ---------------------------------------------------------------------------

from tldw_chatbook.Agents.agent_models import with_preamble_rationale
from tldw_chatbook.Agents.agent_runtime import parse_fenced_tool_call


def _fence(json_body: str) -> str:
    return "```tool_call\n" + json_body + "\n```"


def test_with_preamble_fills_empty_and_preserves_explicit():
    filled = ToolCall(name="a", args={})
    explicit = ToolCall(name="b", args={}, rationale="explicit")
    out = with_preamble_rationale([filled, explicit], "Checking the config")
    assert out[0].rationale == "Checking the config"
    assert out[1].rationale == "explicit"


def test_with_preamble_noop_on_blank_text():
    call = ToolCall(name="a", args={})
    assert with_preamble_rationale([call], "  ") == (call,)


def test_fence_parses_explicit_rationale_key():
    call = parse_fenced_tool_call(
        _fence(
            '{"name": "fs_read", "arguments": {"path": "x"}, '
            '"rationale": "Reading the config"}'
        )
    )
    assert call is not None
    assert call.rationale == "Reading the config"


def test_fence_wrong_typed_rationale_is_ignored_not_fatal():
    call = parse_fenced_tool_call(
        _fence('{"name": "fs_read", "arguments": {}, "rationale": 123}')
    )
    assert call is not None
    assert call.rationale == ""


def test_fence_oversized_rationale_is_capped():
    call = parse_fenced_tool_call(
        _fence('{"name": "fs_read", "arguments": {}, "rationale": "%s"}' % ("x" * 900))
    )
    assert call is not None
    assert len(call.rationale) == 500
    assert call.rationale.startswith("\N{HORIZONTAL ELLIPSIS}")
