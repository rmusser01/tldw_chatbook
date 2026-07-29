"""The Watchlists reader: one pane, two renderers, chosen by `content_kind`.

Both kinds share this pane, its keys and its actions, so a site change reads
like a feed article while still showing what was honestly captured.
"""

from __future__ import annotations

from typing import Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
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

# Remote markdown bodies must not produce real terminal hyperlinks (PR #1091
# review, F3; TASK-1348 AC#2 -- this is that decision, recorded).
#
# `rich.markdown.Markdown` defaults to `hyperlinks=True`, which emits OSC-8
# escape sequences: `[Anthropic docs](https://evil.test/steal)` from a feed
# becomes a clickable "Anthropic docs" whose real destination the reader
# cannot see. The label is attacker-chosen and the destination is hidden --
# the terminal equivalent of a phishing anchor, in content this app fetched
# from a URL the user subscribed to years ago and no longer thinks about.
#
# `hyperlinks=False` is the whole fix: the label renders, and the URL renders
# beside it as ordinary visible text. Nothing is lost -- a terminal reader
# still has the address and can copy it -- and the user judges the
# destination they can actually read.
#
# The alternative, sanitizing or allow-listing URLs before rendering, was
# rejected: it means owning a URL policy inside a renderer (which schemes,
# which hosts, punycode, redirectors), it fails open on every case the list
# did not anticipate, and even a perfectly-filtered `https://` link still
# hides its destination behind an attacker's label. This also keeps the rule
# `render_article` already states: defend at the boundary where the parser
# actually is. `Markdown` IS that parser, so the argument goes here rather
# than as escaping upstream, which was removed from this file for corrupting
# ordinary content while protecting nothing.
_MARKDOWN_HYPERLINKS = False


def _is_markdown(item: dict[str, Any]) -> bool:
    """Whether this item's body was captured as markdown source.

    `content_format` is written by the ingest path (`item_persist.py` validates
    the `content_kind`/`content_format` pairing) and, until this fix, nothing
    read it back -- so a markdown body was shown to the user as raw `##` /
    `[text](url)` / `*emphasis*` source (whole-branch review, Minor).

    Args:
        item: A normalized watchlist item (see `normalize_watchlist_item`).

    Returns:
        `True` when `content_format` is `"markdown"`, case- and
        whitespace-insensitively; `False` for anything else, including a
        missing or NULL format.
    """
    return str(item.get("content_format") or "").strip().lower() == "markdown"


def render_article(item: dict[str, Any]) -> RenderableType:
    """Render a feed item: title, source, date, word count, body.

    Remote feed/site content is untrusted and reaches this Rich renderable
    verbatim, and that is safe *because* it is appended to a `Text` rather
    than parsed: `Text.append` never interprets Rich markup, and
    `Static(Text)` does not re-parse it either -- verified directly, a body of
    `[bold red]x[/]` renders as those literal characters.

    An earlier revision routed every field through `rich.markup.escape`
    "defensively". That protected nothing (there was no markup parser on this
    path to protect from) and actively corrupted ordinary content, since
    `escape` prefixes a backslash on anything bracket-shaped: every markdown
    link `[docs](url)`, every `[sic]`, every `[citation needed]` in a real
    feed grew a stray backslash for real users on the common path. It is
    gone. Do not reintroduce it here -- if a genuinely markup-parsing sink is
    ever added (a `DataTable` `str` cell, a tooltip, a `Button` label), escape
    at THAT boundary, where the parser actually is.

    The markdown branch below is the one place on this path where a parser
    genuinely is, and it is defended there rather than upstream -- see
    `_MARKDOWN_HYPERLINKS`.

    Args:
        item: A normalized watchlist item. `title`, `source_name`,
            `published_date`, `content` and `content_format` are read; all
            may be missing or NULL.

    Returns:
        A `Text` for a plain-text body, or a `Group` whose body half is a
        `rich.markdown.Markdown` when `content_format` says markdown.
    """
    body = item.get("content")
    out = Text()
    out.append(str(item.get("title") or "Untitled"), style="bold")
    out.append("\n")
    meta = [str(item.get("source_name") or "unknown source")]
    if item.get("published_date"):
        meta.append(str(item["published_date"]))
    if body:
        meta.append(f"{len(str(body).split())} words")
    out.append(" · ".join(meta), style="dim")
    out.append("\n")
    if body and _is_markdown(item):
        # `Markdown` is a block renderable, so it cannot be appended into the
        # `Text` above -- group the two instead. `Markdown` does not evaluate
        # Rich markup either; it parses CommonMark, and `[bold red]x[/]` is
        # just link-shaped text to it.
        return Group(out, Markdown(str(body), hyperlinks=_MARKDOWN_HYPERLINKS))
    out.append("\n")
    out.append(str(body) if body else _NO_BODY)
    return out


def render_change(item: dict[str, Any]) -> Text:
    """Render a site item: what changed, by how much, and the diff lines.

    Args:
        item: A normalized watchlist item. `title`, `change_percentage`,
            `change_type`, `diff_summary` and `content` are read; all may be
            missing or NULL, and a non-numeric `change_percentage` is
            dropped from the headline rather than raised.

    Returns:
        A `Text` with the title, a one-line headline, and the diff body
        coloured by leading `+`/`-`; `_NO_BODY` in place of the diff when the
        item carries no content.
    """
    out = Text()
    out.append(str(item.get("title") or "Untitled"), style="bold")
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
    if item.get("diff_summary"):
        # Whole-branch review (Minor): `diff_summary` was carried through
        # normalization with no consumer at all. It is the monitoring
        # engine's own one-line account of the change ("2 lines changed"),
        # which is exactly what this headline is for.
        headline.append(str(item["diff_summary"]))
    out.append(" · ".join(headline) or "changed", style="dim")
    out.append("\n\n")

    body = item.get("content")
    if not body:
        out.append(_NO_BODY)
        return out

    # Colour the diff. These lines are remote content appended to a `Text`,
    # which never parses markup -- see `render_article` on why the former
    # `escape_markup` call here was corruption without protection.
    for line in str(body).splitlines():
        style = "green" if line.startswith("+") else "red" if line.startswith("-") else None
        out.append(line, style=style)
        out.append("\n")
    return out


_RENDERERS = {"article": render_article, "change": render_change}


def render_for(item: dict[str, Any]) -> RenderableType:
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
        """Build the reader for `item`, or the empty-state placeholder.

        Yields:
            A single `#content-empty` `Static` when no item is selected;
            otherwise the `#content-mark-unread-button` and a
            `#content-body` `Static` holding `render_for(self.item)`.
        """
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
        # `compact=True` (whole-branch review, Minor): a default `Button` is
        # three rows tall (top border, label, bottom border) and CONTENT has
        # only about nine usable ones -- the same third of the pane the
        # tooltip fix above just reclaimed, spent again on a button's chrome.
        yield Button(
            "Mark unread",
            id="content-mark-unread-button",
            tooltip=_GLOBAL_STATUS_NOTE,
            compact=True,
        )
        yield Static(render_for(self.item), id="content-body")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "content-mark-unread-button":
            event.stop()
            self.post_message(UnreadToggleRequested(self.item))
