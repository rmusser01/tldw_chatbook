"""task-31382: the display label a sub-agent run carries on its actor."""

from __future__ import annotations

from tldw_chatbook.Agents.agent_service import (
    subagent_display_label,
)
from tldw_chatbook.Agents.run_context import (
    SUBAGENT_LABEL_MAX_CHARS,
    CurrentRunActor,
    clean_subagent_label,
)


def test_actor_label_defaults_to_none_and_is_settable():
    assert CurrentRunActor("primary", "r1", None).label is None
    assert CurrentRunActor("subagent", "r2", "r1", label="researcher").label == "researcher"


def test_named_agent_wins_over_the_task_and_is_cleaned_too():
    assert subagent_display_label(" researcher ", "Survey the schema") == "researcher"
    assert subagent_display_label("bad\nname\x07 here", "task") == "bad name here"


def test_clean_subagent_label_is_the_one_sanitizer_the_renderers_share():
    assert clean_subagent_label(None) == ""
    assert clean_subagent_label("  a \n b\x1b[31m c  ") == "a b [31m c"
    cut = clean_subagent_label("y" * (SUBAGENT_LABEL_MAX_CHARS + 5))
    assert len(cut) == SUBAGENT_LABEL_MAX_CHARS and cut.endswith("…")


def test_task_first_line_is_the_fallback_and_is_cut_to_one_line():
    assert subagent_display_label(None, "Survey the schema\nthen report") == "Survey the schema"
    long = "x" * (SUBAGENT_LABEL_MAX_CHARS + 20)
    label = subagent_display_label("", long)
    assert len(label) == SUBAGENT_LABEL_MAX_CHARS and label.endswith("…")
    assert subagent_display_label(None, "   ") is None
    assert subagent_display_label(None, None) is None
