"""TASK-1995: markdown rendering hygiene.

1. The About pane feeds real markdown (not Rich console markup) to its
   Markdown widget.
2. Media reading-mode and search-highlight transforms leave the inside of
   fenced code blocks untouched.
"""

from tldw_chatbook.UI.Tools_Settings_Window import ABOUT_MARKDOWN
from tldw_chatbook.Widgets.Media.media_viewer_panel import (
    _fenced_ranges,
    format_reading_text,
    highlight_match_spans,
)

FENCED_DOC = (
    "Intro sentence one. Intro sentence two.\n"
    "\n"
    "```python\n"
    "x = 1. Then more code? No.\n"
    "1. not a heading\n"
    "```\n"
    "\n"
    "1. a real numbered line\n"
    "Outro. Done.\n"
)


def test_about_text_is_markdown_not_rich_markup():
    assert "[bold]" not in ABOUT_MARKDOWN
    assert "[italic]" not in ABOUT_MARKDOWN
    assert "[link=" not in ABOUT_MARKDOWN
    assert "**tldw-chatbook**" in ABOUT_MARKDOWN
    assert "<https://github.com/rmusser01/tldw>" in ABOUT_MARKDOWN
    # Bullets are list markers the markdown parser understands.
    assert "\n- Multi-provider LLM support" in ABOUT_MARKDOWN


def test_fenced_ranges_finds_closed_and_unclosed_fences():
    ranges = _fenced_ranges(FENCED_DOC)
    assert len(ranges) == 1
    start, end = ranges[0]
    assert FENCED_DOC[start:].startswith("```python")
    assert FENCED_DOC[start:end].rstrip().endswith("```")

    unclosed = "text\n```\ncode to the end"
    (only,) = _fenced_ranges(unclosed)
    assert unclosed[only[0] : only[1]] == "```\ncode to the end"


def test_reading_mode_leaves_fenced_code_untouched():
    formatted = format_reading_text(FENCED_DOC)
    # Code block content is byte-identical.
    assert "x = 1. Then more code? No.\n" in formatted
    assert "## 1. not a heading" not in formatted
    # Prose transforms still apply outside the fence.
    assert "Intro sentence one.\n\nIntro sentence two." in formatted
    assert "## 1. a real numbered line" in formatted
    assert "Outro.\n\nDone." in formatted


def test_highlight_skips_matches_inside_fences():
    target = "code"
    inside = FENCED_DOC.index("more code")
    outside = FENCED_DOC.index("Intro")
    matches = [
        (outside, outside + 5),  # "Intro" — prose
        (inside + 5, inside + 9),  # "code" — inside the fence
    ]
    highlighted = highlight_match_spans(FENCED_DOC, matches, current_index=0)
    # Prose match is wrapped as the current match.
    assert "**`▶ Intro ◀`**" in highlighted
    # In-fence match is left verbatim — no backticks injected around it.
    assert "x = 1. Then more code? No.\n" in highlighted
    assert "`code`" not in highlighted


def test_highlight_wraps_non_current_prose_matches():
    doc = "alpha beta alpha"
    matches = [(0, 5), (11, 16)]
    highlighted = highlight_match_spans(doc, matches, current_index=1)
    assert " `alpha` " in highlighted
    assert "**`▶ alpha ◀`**" in highlighted
