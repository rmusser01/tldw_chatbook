"""The Watchlists reader: one pane, two renderers, chosen by `content_kind`.

Both kinds share this pane, its keys and its actions, so a site change reads
like a feed article while still showing what was honestly captured.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static

from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard

# Every item persisted before Phase A carries `content = NULL`, and it cannot
# be recovered without re-fetching the source. Say so rather than rendering an
# empty pane the reader will mistake for a bug.
_NO_BODY = "no body captured for this item — re-check this source to fetch it"


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
        headline.append(f"{float(pct):.0f}% changed")
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
        yield Static(render_for(self.item), id="content-body")
