"""Console display rows must show an ampersand, not an HTML entity.

TASK-32802.4. `_safe_display_text` HTML-entity-escaped every user string it
normalized, but every destination those strings reach is a terminal surface
that renders them literally: `Static(..., markup=False)` in the staged
context tray, the staged evidence strip, the Run Inspector and the settings
estimate, or a `rich.text.Text` built directly. Rich never decodes `&amp;`
back to `&`, so a Library source titled `R&D Report` reached the user as
`R&amp;D Report`, and already-encoded text was doubled.

The Library side hit this in live UAT and fixed it (RAG-30/31). Console did
not, which is what this closes.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_display_state import _safe_display_text


@pytest.mark.parametrize(
    "raw",
    [
        "R&D Report",
        "Tom & Jerry",
        "a < b and b > c",
        "quote \" and apostrophe '",
        "R&amp;D already encoded upstream",
    ],
)
def test_display_text_is_not_entity_escaped(raw):
    assert _safe_display_text(raw) == raw


def test_the_fallback_still_applies():
    assert _safe_display_text(None, "Untitled source") == "Untitled source"
    assert _safe_display_text("   ", "Untitled source") == "Untitled source"
    assert _safe_display_text("  kept  ", "Untitled source") == "kept"


def test_markup_is_left_for_the_sink_to_handle():
    """This normalizer never claimed to neutralize markup, and still does not.

    The `[/]`-raises-MarkupError failure is fixed at the sinks with
    `markup=False`; pinning that here would misplace the contract.
    """
    assert _safe_display_text("[/] title") == "[/] title"
