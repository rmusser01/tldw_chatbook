"""Model- and user-authored labels must render literally, never as markup — TASK-38585.

Two live-markup injection sites found by the 2026-11-02 audit (section 1.5),
both reproduced live on Textual 8.2.8:

* the Console canvas card label, whose `card.title` comes straight from the
  model's canvas tool-call arguments (length-validated only), and
* the STTS conversation-selection dialog title, which is a user-authored
  conversation title.

Rendered through a markup-parsing Static/Label:

* a title containing ``[/]`` raises ``MarkupError`` out of the app run, and
  because a canvas title persists in message metadata the crash recurs on
  EVERY transcript render of that message;
* bracketed text such as ``Budget [draft]`` is silently swallowed.

These tests mount only the widget under test in a minimal app (widget-level,
per the B6 brief: the full app cannot boot in this environment) and assert
BOTH survival and content preservation — the brackets the author wrote must
still be on screen, not merely "did not crash".
"""

from __future__ import annotations

from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Region
from textual.widgets import Label, Static

from tldw_chatbook.Widgets.Console.console_canvas_card import (
    ConsoleCanvasCard,
    ConsoleCanvasCardPresentation,
)
from tldw_chatbook.Widgets.conversation_selection_dialog import (
    ConversationItem,
    ConversationSelectionDialog,
)

# These tests mount bare widgets and read no config; keep the collection-time
# profile so the per-test sandbox redirect's config-participant admission
# (RecoveryRequired on this dev baseline, see the B6 brief) stays out of the
# way. Registered marker: Tests/conftest.py (TASK-32873).
pytestmark = pytest.mark.bootstrap_profile

# Every bracket shape Textual's tokenizer can choke on or swallow. The
# canvas label embeds the title as f"{title} · revision N · status" (see
# console_transcript.py's canvas presentation builder); these tests mirror
# that composition so the hostile bytes sit exactly where they do in the
# product.
HOSTILE_TITLES = [
    "Budget [draft]",
    "Bad [/] close tag",
    "[/]",
    "[/][/]",
    "Lone [ bracket",
    "[link]styled[/link]",
]


def _visible_text(widget: Static, width: int = 120) -> str:
    """The characters the first rendered line of ``widget`` actually shows."""
    strip = widget.render_lines(Region(0, 0, width, 1))[0]
    return strip.text


def _presentation(title: str) -> ConsoleCanvasCardPresentation:
    return ConsoleCanvasCardPresentation(
        canvas_id="canvas-a",
        revision_id="revision-7",
        label=f"{title} · revision 7 · updated",
        digest="a" * 64,
        reopenable=True,
        error_code=None,
    )


class _CardApp(App[None]):
    CSS = ConsoleCanvasCard.BUNDLED_CSS

    def __init__(self, card: ConsoleCanvasCard) -> None:
        super().__init__()
        self._card = card

    def compose(self) -> ComposeResult:
        yield self._card


@pytest.mark.asyncio
@pytest.mark.parametrize("title", HOSTILE_TITLES)
async def test_canvas_card_label_renders_hostile_titles_literally(title: str):
    """No MarkupError, and the bracketed title text stays on screen."""
    card = ConsoleCanvasCard(
        _presentation(title), session_id="session-a", message_id="m", card_index=0
    )

    async with _CardApp(card).run_test(size=(120, 30)):
        label = card.query_one(".console-canvas-card-label", Static)
        visible = _visible_text(label)

    assert title in visible, (
        f"the card label must show the model-authored title verbatim; "
        f"got {visible!r}"
    )
    assert "revision 7" in visible, "the label's own suffix must survive too"


def _conversation(conversation_id: int, title: str) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "title": title,
        "model_name": "model-a",
        "message_count": 3,
        "created_at": "2026-01-01",
        "updated_at": "2026-01-02",
    }


class _ItemApp(App[None]):
    def __init__(self, item: ConversationItem) -> None:
        super().__init__()
        self._item = item

    def compose(self) -> ComposeResult:
        yield self._item


@pytest.mark.asyncio
@pytest.mark.parametrize("title", HOSTILE_TITLES)
async def test_conversation_item_title_renders_hostile_titles_literally(title: str):
    """The user-authored conversation title must survive verbatim."""
    item = ConversationItem(
        conversation_id=7,
        title=title,
        model_name="model-a",
        message_count=3,
        created_at="2026-01-01",
        updated_at="2026-01-02",
    )

    async with _ItemApp(item).run_test(size=(120, 30)):
        label = item.query_one(".conversation-title", Label)
        visible = _visible_text(label)

    assert title in visible, (
        f"the conversation title must render verbatim; got {visible!r}"
    )


@pytest.mark.asyncio
async def test_selection_dialog_mounts_with_a_crashing_title():
    """Dialog-level smoke: the whole dialog renders hostile titles without error.

    Exercises the `load_conversations` → mount path, not just a bare
    ConversationItem.
    """
    dialog = ConversationSelectionDialog(
        [_conversation(11, "Crash [/] title"), _conversation(22, "Keep [draft]")]
    )

    class _DialogApp(App[None]):
        CSS = ConversationSelectionDialog.BUNDLED_SCREEN_CSS

        def compose(self) -> ComposeResult:
            yield Static("beneath the modal")

    async with _DialogApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(dialog)
        await pilot.pause()

        assert dialog.conversation_items, "the dialog must have mounted its items"
        crashing = dialog.conversation_items[0]
        visible = _visible_text(crashing.query_one(".conversation-title", Label))
        assert "Crash [/] title" in visible, f"got {visible!r}"
