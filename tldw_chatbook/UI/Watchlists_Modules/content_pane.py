"""The Watchlists reader: one pane, two renderers, chosen by `content_kind`.

Both kinds share this pane, its keys and its actions, so a site change reads
like a feed article while still showing what was honestly captured.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Static

from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard

# Every item persisted before Phase A carries `content = NULL`, and it cannot
# be recovered without re-fetching the source. Say so rather than rendering an
# empty pane the reader will mistake for a bug.
_NO_BODY = "no body captured for this item — re-check this source to fetch it"

# Task 5: opening an item marks it read, and this is the explicit unread
# toggle's tooltip, so the *scope* of both actions is stated (fix round 1:
# moved off the permanent body -- see the comment at its one call site)
# rather than discovered as a bug later. `mark_item_status` (see
# `SubscriptionsDB`) updates one `subscription_items` row by its own id --
# there is no per-watchlist copy of an item's status -- so this must NOT read
# as "in this watchlist"; it is the same article everywhere the source it
# came from is included.
_GLOBAL_STATUS_NOTE = (
    "Read status is shared: marking this item read (or unread) changes it "
    "everywhere it appears, in every watchlist that includes its source."
)


def render_article(item: dict[str, Any]) -> Text:
    """Render a feed item: title, source, date, word count, body.

    Remote feed/site content is untrusted and reaches this Rich renderable
    verbatim -- every field pulled from the item is passed through
    `rich.markup.escape` before being appended, so it can never be
    interpreted as Rich/Textual markup. This repo has shipped markup
    injection through tooltips and button labels before; escape at the
    boundary rather than trusting the source.
    """
    body = item.get("content")
    out = Text()
    out.append(escape_markup(str(item.get("title") or "Untitled")), style="bold")
    out.append("\n")
    meta = [str(item.get("source_name") or "unknown source")]
    if item.get("published_date"):
        meta.append(str(item["published_date"]))
    if body:
        meta.append(f"{len(str(body).split())} words")
    out.append(escape_markup(" · ".join(meta)), style="dim")
    out.append("\n\n")
    out.append(escape_markup(str(body)) if body else _NO_BODY)
    return out


def render_change(item: dict[str, Any]) -> Text:
    """Render a site item: what changed, by how much, and the diff lines."""
    out = Text()
    out.append(escape_markup(str(item.get("title") or "Untitled")), style="bold")
    out.append("\n")

    headline: list[str] = []
    pct = item.get("change_percentage")
    if pct is not None:
        # `change_percentage` is always written as a Python float by
        # `baseline_manager.py`/`monitoring_engine.py`, never parsed from raw
        # remote text, so this cast is not currently reachable with a
        # non-numeric value. Guard it anyway: a raise here would escape
        # `compose()` and exit the whole application over a single headline
        # field, so degrade by omitting the percent rather than raising.
        try:
            headline.append(f"{float(pct):.0f}% changed")
        except (TypeError, ValueError):
            pass
    if item.get("change_type"):
        headline.append(str(item["change_type"]))
    out.append(escape_markup(" · ".join(headline) or "changed"), style="dim")
    out.append("\n\n")

    body = item.get("content")
    if not body:
        out.append(_NO_BODY)
        return out

    # Colour the diff, but escape each line first: these lines are remote
    # content, and styling them must not mean interpreting them as markup.
    for line in str(body).splitlines():
        style = "green" if line.startswith("+") else "red" if line.startswith("-") else None
        out.append(escape_markup(line), style=style)
        out.append("\n")
    return out


_RENDERERS = {"article": render_article, "change": render_change}


def render_for(item: dict[str, Any]) -> Text:
    """Dispatch on `content_kind`, falling back rather than raising.

    An exception escaping `compose()` exits the application, so an unexpected
    kind degrades to the article renderer instead of taking the app down.
    """
    return _RENDERERS.get(str(item.get("content_kind") or ""), render_article)(item)


class UnreadToggleRequested(Message):
    """Posted when the user reverses a mark-read via the explicit unread toggle.

    Opening an item marks it read, which destroys its place in whatever
    unread list surfaced it -- an accidental open must be recoverable, so
    this is that escape hatch. Carries the full item dict (not just an id)
    so the screen's handler does not have to re-look it up from a list that
    may already have moved on.
    """

    def __init__(self, item: dict[str, Any] | None) -> None:
        self.item = item
        super().__init__()


class ContentPane(RecomposeCaptureGuard, Vertical):
    """Hosts the reader for the currently selected item.

    Follows `ItemsPane`'s conventions (see `items_pane.py`): the
    `RecomposeCaptureGuard` mixin sits ahead of the concrete Textual
    container because `item` is a `reactive(..., recompose=True)` field and
    this pane is a descendant widget of the Watchlists screen rather than
    the screen itself -- exactly the mouse-capture hazard
    `RecomposeCaptureGuard` closes (task-627/637): a recompose triggered by
    a reactive field on a non-screen widget can otherwise strand mouse
    capture on a removed child.
    """

    item: reactive[dict[str, Any] | None] = reactive(None, recompose=True)

    def compose(self):
        if self.item is None:
            yield Static("Select an item to read it.", id="content-empty")
            return
        # Task 5: the reader marks an item read on open (see the screen's
        # `_mark_item_read_on_open`); this button is the deliberate way
        # back. `_GLOBAL_STATUS_NOTE` states the scope on the tooltip
        # (fix round 1, Important) rather than as a permanent line in the
        # body: a permanent `Static` here measured 3 of CONTENT's 8 visible
        # rows (max-height 12, minus the border and heading), leaving the
        # actual article only 4 of 14 rows -- the note earned its point
        # once, not on every single line of every item read afterward.
        yield Button(
            "Mark unread", id="content-mark-unread-button", tooltip=_GLOBAL_STATUS_NOTE
        )
        yield Static(render_for(self.item), id="content-body")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "content-mark-unread-button":
            event.stop()
            self.post_message(UnreadToggleRequested(self.item))
