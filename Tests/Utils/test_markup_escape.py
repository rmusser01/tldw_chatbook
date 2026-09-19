"""The escaper must match the parser the app actually renders through.

TASK-32802.1. `rich.markup.escape` only escapes tags matching
``\\[[a-z#/@]...]``, but Textual 8 renders a ``str`` handed to a widget
through ``Content.from_markup``, which consumes ANY ``[...]`` token. Every
pinning test that existed for the old escaper used a lowercase tag -- the one
shape rich does escape -- so the whole defect class was invisible to the
suite. These tests use uppercase-initial brackets on purpose.
"""

from __future__ import annotations

import pytest
from rich.markup import escape as rich_escape
from rich.text import Text
from textual.content import Content

from tldw_chatbook.Utils.input_validation import escape_markup


# The tokens users actually type into titles, and what the narrow escape
# does to each one on Textual 8.
USER_TOKENS = [
    ("[TODO] Q3 plan", " Q3 plan"),
    ("[IMPORTANT]", ""),
    ("[WIP] plan", " plan"),
    ("[Draft] notes", " notes"),
    ("[PR-6] follow-up", " follow-up"),
]


@pytest.mark.parametrize("raw,lost_to_the_narrow_escape", USER_TOKENS)
def test_a_bracketed_token_survives_a_textual_surface(
    raw, lost_to_the_narrow_escape
):
    assert Content.from_markup(escape_markup(raw)).plain == raw
    # Negative control: the same input through the escape this replaced.
    assert Content.from_markup(rich_escape(raw)).plain == lost_to_the_narrow_escape


def test_a_real_markup_tag_is_still_shown_literally():
    """The case the old escape did handle must keep working."""
    raw = "[bold]hi[/bold]"
    assert Content.from_markup(escape_markup(raw)).plain == raw
    assert rich_escape(raw) == escape_markup(raw)


@pytest.mark.parametrize("raw,_lost", USER_TOKENS)
def test_rich_renders_the_wider_escape_identically(raw, _lost):
    """Adopting this repo-wide is safe for a Rich markup consumer too.

    ``\\[`` is a literal ``[`` to Rich's parser as well, so a surface that
    renders through Rich sees exactly what the narrow escape gave it.
    """
    assert Text.from_markup(escape_markup(raw)).plain == raw
    assert (
        Text.from_markup(escape_markup(raw)).plain
        == Text.from_markup(rich_escape(raw)).plain
    )


def test_a_non_string_is_coerced_not_raised():
    """Stored JSON hands this ints and None; a label must not crash."""
    assert escape_markup(5) == "5"
    assert escape_markup(None) == "None"


def test_every_bracket_is_escaped_not_only_the_first():
    raw = "[A] and [B] and [c]"
    assert Content.from_markup(escape_markup(raw)).plain == raw
