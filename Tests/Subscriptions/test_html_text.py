"""Unit tests for `html_text.py` (TASK-2307).

Two properties matter here, and each gets its own hostile-payload coverage:
inertness (nothing this module produces can be interpreted by a terminal or
a Rich markup parser) and readability (a real feed's HTML actually turns
into prose a person can read, not a mangled mess).
"""

import pytest

pytestmark = pytest.mark.unit

from tldw_chatbook.Subscriptions.html_text import (
    html_to_display_text,
    looks_like_html,
    readable_body_text,
    strip_control_characters,
)


# --- looks_like_html ---------------------------------------------------


def test_a_plain_sentence_is_not_treated_as_html():
    assert not looks_like_html("Just an ordinary sentence.")


def test_a_bare_less_than_sign_is_not_mistaken_for_a_tag():
    """A mathematical `<` in plain prose must not trigger the converter --
    the whole reason `_HTML_SHAPED` requires a closing `>` and a letter (or
    `/`) right after `<`."""
    assert not looks_like_html("1 < 2 and 2 < 3, all in plain text.")


def test_a_real_tag_is_detected():
    assert looks_like_html("<p>hello</p>")


def test_an_html_comment_is_detected():
    assert looks_like_html("before <!-- a comment --> after")


def test_a_named_entity_reference_is_detected():
    assert looks_like_html("Tom &amp; Jerry")


def test_a_numeric_entity_reference_is_detected():
    assert looks_like_html("caf&#233;")
    assert looks_like_html("caf&#xE9;")


def test_empty_and_none_are_not_html():
    assert not looks_like_html("")
    assert not looks_like_html(None)


# --- html_to_display_text: readability ----------------------------------


def test_paragraphs_become_readable_prose_with_blank_lines_between_them():
    out = html_to_display_text("<p>First paragraph.</p><p>Second paragraph.</p>")
    assert "<p>" not in out and "</p>" not in out
    assert "First paragraph." in out
    assert "Second paragraph." in out
    assert "\n\n" in out


def test_a_link_keeps_its_label_and_shows_its_destination_as_text():
    """The UAT's exact shape: `<a href>` must not vanish into a hidden
    hyperlink -- both halves must be visible, plain text."""
    out = html_to_display_text('Article URL: <a href="https://example.test/x">read more</a>')
    assert "<a href" not in out
    assert "read more" in out
    assert "https://example.test/x" in out


def test_a_link_whose_label_already_is_the_url_is_not_printed_twice():
    out = html_to_display_text('<a href="https://example.test/x">https://example.test/x</a>')
    assert out.count("https://example.test/x") == 1


def test_list_items_become_bullets():
    out = html_to_display_text("<ul><li>First</li><li>Second</li></ul>")
    assert "- First" in out
    assert "- Second" in out
    assert "<li>" not in out


def test_line_breaks_become_newlines():
    out = html_to_display_text("Line one<br>Line two")
    assert "Line one" in out
    assert "Line two" in out
    assert "<br" not in out


def test_image_alt_text_is_kept_as_a_caption():
    out = html_to_display_text('<img src="x.png" alt="A chart of Q3 revenue">')
    assert "A chart of Q3 revenue" in out
    assert "<img" not in out


def test_image_with_no_alt_text_produces_nothing_visible():
    out = html_to_display_text('<img src="x.png">')
    assert "<img" not in out


def test_entities_are_unescaped_exactly_once():
    out = html_to_display_text("Tom &amp; Jerry")
    assert out == "Tom & Jerry"


def test_entities_are_not_double_unescaped():
    """`&amp;lt;` must become `&lt;`, not `<` -- double-unescaping would
    manufacture a tag-shaped fragment out of literal feed text."""
    out = html_to_display_text("literal: &amp;lt;not-a-tag&amp;gt;")
    assert "&lt;not-a-tag&gt;" in out


def test_script_and_style_content_is_dropped_not_shown_as_text():
    out = html_to_display_text(
        "<p>Visible</p><script>alert('x')</script><style>.a{color:red}</style>"
    )
    assert "Visible" in out
    assert "alert" not in out
    assert "color:red" not in out
    assert "<script>" not in out and "<style>" not in out


def test_nested_drop_content_is_not_reenabled_by_the_inner_close_tag():
    """`<svg><style>...</style>...</svg>`: the inner `</style>` must not
    turn dropping back off while still inside the outer `<svg>`."""
    out = html_to_display_text("<p>Before</p><svg><style>.a{}</style>leaked?</svg><p>After</p>")
    assert "Before" in out and "After" in out
    assert "leaked?" not in out


def test_malformed_nesting_does_not_raise():
    """`html.parser` tolerates bad markup; a stray close tag must not crash."""
    out = html_to_display_text("<p>Unclosed paragraph <b>bold <i>italic</p> stray </a> tail")
    assert "Unclosed paragraph" in out
    assert "tail" in out


def test_a_plain_text_value_with_no_markup_survives_essentially_unchanged():
    out = html_to_display_text("Just a sentence with no markup at all.")
    assert out == "Just a sentence with no markup at all."


def test_runs_of_inline_whitespace_are_collapsed():
    out = html_to_display_text("<p>too    much     space</p>")
    assert "too much space" in out
    assert "  " not in out


def test_more_than_one_blank_line_is_collapsed_to_one():
    out = html_to_display_text("<p>A</p>" + "<div></div>" * 5 + "<p>B</p>")
    assert "\n\n\n" not in out


# --- Hostile payloads: inertness -----------------------------------------


def test_bracket_shaped_text_survives_as_literal_characters():
    """Rich markup shape inside an HTML body must come out as plain text,
    for `Text.append` to render verbatim rather than interpret."""
    out = html_to_display_text("<p>[bold red]not a style[/]</p>")
    assert "[bold red]not a style[/]" in out


def test_raw_esc_control_byte_does_not_survive():
    """A raw ESC in a feed body must never reach the terminal -- an OSC-8
    hyperlink whose label lies about its destination, or a cursor-control
    sequence, both start here."""
    out = html_to_display_text("<p>before \x1b[31mred\x1b[0m after</p>")
    assert "\x1b" not in out
    assert "before" in out and "red" in out and "after" in out


def test_osc8_hyperlink_sequence_loses_its_control_bytes():
    """The exact payload the module's own docstring names: an OSC-8
    hyperlink whose visible label lies about where it actually points."""
    payload = "<p>\x1b]8;;http://evil.test\x07Anthropic docs\x1b]8;;\x07 tail</p>"
    out = html_to_display_text(payload)
    assert "\x1b" not in out
    assert "\x07" not in out
    assert "Anthropic docs" in out
    assert "tail" in out


def test_javascript_href_is_shown_as_inert_text_not_executed_or_hidden():
    out = html_to_display_text('<a href="javascript:alert(1)">click</a>')
    assert "click" in out
    assert "javascript:alert(1)" in out


def test_c1_control_range_is_also_stripped():
    """`strip_control_characters` covers C1 (0x80-0x9F), not just C0/DEL --
    an 8-bit CSI/OSC introducer is just as capable as the 7-bit form."""
    out = strip_control_characters("before \x9b31mfake-csi after")
    assert "\x9b" not in out
    assert "before" in out and "after" in out


def test_tab_and_newline_survive_control_stripping():
    out = strip_control_characters("line one\nline two\tindented")
    assert out == "line one\nline two\tindented"


def test_carriage_return_does_not_survive_control_stripping():
    """Batch-4 review, M1. The docstring's own claim ("C0 minus tab and
    newline") always meant to cover CR (0x0D, neither tab nor newline), but
    the regex range used to jump `\\x0c` -> `\\x0e-\\x1f` and skip it by one
    code point -- a much weaker primitive than ESC/OSC (a bare CR can only
    overwrite characters earlier on the same terminal line, a line-overwrite
    spoof), but a real mismatch between what this module documents and what
    it did.
    """
    out = strip_control_characters("Real Title\rEVIL OVERWRITE")
    assert "\r" not in out
    assert out == "Real TitleEVIL OVERWRITE"


def test_a_crlf_feed_body_is_not_left_with_doubled_blank_lines():
    """The reviewer's own condition for the fix: stripping CR must not turn
    a `\\r\\n` line ending into a doubled blank line -- the `\\n` in the pair
    is the one line break and it must survive untouched, exactly as if the
    body had used `\\n` alone.
    """
    with_crlf = strip_control_characters("line one\r\nline two\r\nline three")
    with_lf = strip_control_characters("line one\nline two\nline three")
    assert with_crlf == with_lf == "line one\nline two\nline three"
    assert "\n\n" not in with_crlf


def test_none_becomes_empty_string():
    assert strip_control_characters(None) == ""
    assert html_to_display_text(None) == ""
    assert readable_body_text(None) == ""


# --- readable_body_text: the dispatch the reader actually calls ----------


def test_readable_body_text_converts_an_html_body():
    out = readable_body_text("<p>Article URL: <a href=\"https://x.test\">here</a></p>")
    assert "<p>" not in out
    assert "here" in out and "https://x.test" in out


def test_readable_body_text_leaves_plain_text_alone_apart_from_control_bytes():
    """Only the control BYTE is removed, not the printable characters that
    happened to follow it -- `strip_control_characters` strips control
    characters, not ANSI escape sequences as a unit (see its own docstring:
    "nothing is escaped or substituted -- the characters simply do not
    survive"). The literal `[31m` that remains is exactly that: ordinary
    text with no control byte left to interpret it as a colour code.
    """
    out = readable_body_text("1 < 2, plainly written, no markup at all.\x1b[31m")
    assert out == "1 < 2, plainly written, no markup at all.[31m"
    assert "\x1b" not in out


def test_readable_body_text_hostile_end_to_end():
    """The full path a real hostile feed body takes: HTML detection, tag
    stripping, entity unescaping, and control-byte removal, all together."""
    payload = (
        '<p>[bold red]injected[/] <a href="javascript:alert(1)">click</a></p>'
        "<p>\x1b]8;;http://evil.test\x07label\x1b]8;;\x07</p>"
        "<script>steal()</script>"
    )
    out = readable_body_text(payload)
    assert "[bold red]injected[/]" in out
    assert "click" in out and "javascript:alert(1)" in out
    assert "label" in out
    assert "steal()" not in out
    assert "\x1b" not in out and "\x07" not in out


# --- Batch-4 review, I3: self-closed (XHTML-style) void tags -------------
#
# Real RSS/Atom feeds commonly emit the self-closed form of a void element
# (`<br/>`, `<img .../>`, `<hr/>`) rather than the bare `<br>`/`<img>` shape
# every other test in this file uses -- these arrive at
# `handle_startendtag`, not `handle_starttag` (`_DisplayTextExtractor`'s own
# docstring says so), and that method had zero test coverage: mutation-
# verified by the review to gut to a no-op with all 84 tests in this file
# plus `test_watchlists_content_pane.py` staying green.


def test_self_closed_br_becomes_a_line_break():
    out = html_to_display_text("Line one<br/>Line two")
    assert out == "Line one\nLine two", (
        "the self-closed <br/> must become an actual line break, not a "
        f"space-join or dropped tag: {out!r}"
    )


def test_self_closed_img_alt_text_is_kept_as_a_caption():
    out = html_to_display_text('<p>Before</p><img alt="A chart of Q3 revenue"/><p>After</p>')
    assert "A chart of Q3 revenue" in out
    assert "<img" not in out
    assert "Before" in out and "After" in out


def test_self_closed_img_with_no_alt_text_produces_nothing_visible():
    out = html_to_display_text('<img src="x.png"/>')
    assert "<img" not in out


def test_self_closed_hr_becomes_a_line_break_not_dropped_content():
    out = html_to_display_text("<p>Above</p><hr/><p>Below</p>")
    assert "Above" in out and "Below" in out
    assert "<hr" not in out


def test_self_closed_void_tag_inside_drop_content_is_still_dropped():
    """The self-closed path must respect `_drop_depth` too -- an `<img/>`
    inside a `<script>` must not leak an `[image: ...]` caption."""
    out = html_to_display_text('<script>var x = "<img alt=leak/>";</script><p>Safe</p>')
    assert "leak" not in out
    assert "Safe" in out


def test_a_self_closed_void_tag_is_bracket_shaped_text_that_still_survives_inert():
    """The hostile-payload discipline this whole file exists for, through
    the self-closed path specifically: an `alt` attribute shaped like Rich
    markup must reach the caller as literal characters."""
    out = html_to_display_text('<img alt="[bold red]not a style[/]"/>')
    assert "[bold red]not a style[/]" in out
