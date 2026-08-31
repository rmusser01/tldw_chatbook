"""ADR-080: rationale capture normalization + ToolCall field."""

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
