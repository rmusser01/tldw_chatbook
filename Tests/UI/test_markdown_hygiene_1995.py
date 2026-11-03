"""TASK-1995: markdown rendering hygiene.

The About pane feeds real markdown (not Rich console markup) to its
Markdown widget. (The media reading-mode/search-highlight companion tests
were removed with the Widgets/Media package, TASK-38605 dead-code purge.)
"""

from tldw_chatbook.Utils.about_text import ABOUT_MARKDOWN


def test_about_text_is_markdown_not_rich_markup():
    assert "[bold]" not in ABOUT_MARKDOWN
    assert "[italic]" not in ABOUT_MARKDOWN
    assert "[link=" not in ABOUT_MARKDOWN
    assert "**tldw-chatbook**" in ABOUT_MARKDOWN
    assert "<https://github.com/rmusser01/tldw>" in ABOUT_MARKDOWN
    # Bullets are list markers the markdown parser understands.
    assert "\n- Multi-provider LLM support" in ABOUT_MARKDOWN
