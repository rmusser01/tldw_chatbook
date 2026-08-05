"""The reader's stored-page viewer (task-1494).

The Watchlists design spec's Content-pane mockup promised `[full page]` and
`[previous snapshot]` affordances on a change item, reading from
`url_snapshots`. Phase D shipped the change renderer without them, leaving
the data stored and unreachable -- the reader could show *what* changed but
never the page it changed on. `content_pane.ContentPane` now posts
`ViewSnapshotRequested` for those two buttons on a `change`-kind item;
`WatchlistsCollectionsScreen`'s handler resolves it against `url_snapshots`
(`LocalWatchlistsService.get_url_snapshots`) and pushes this modal with the
row it found. This modal owns no DB handle of its own and issues no query --
everything it shows was already fetched before it was constructed, matching
`ContentPane`'s own "no DB in the widget" rule.

**AC#3, the reason this is its own modal rather than a `Static` bolted onto
`ContentPane`:** `extracted_content` is text scraped from a page this app
does not control, at a URL the user subscribed to possibly years ago. It
must render as literal characters, never as Rich markup and never as a live
hyperlink -- the identical doctrine `content_pane.render_article`/TASK-1348
already state for the reader's own body, and `KeptBriefingsModal`'s module
docstring restates for LLM-authored content. The mechanism is the same one
both of those use: build a `rich.text.Text` with `.append`/the constructor,
which never parses markup, and hand it to `Static` -- never a bare `str`,
never `Text.from_markup`, never `rich.markdown.Markdown` (a real parser,
and this snapshot's raw extracted text was never validated as being
written for one). `url`/`created_at` are this app's own values (the
subscription's configured source, and a DB timestamp this process wrote),
not remote content, but they are wrapped through `Text` here too rather
than an f-string handed to `Static` -- one rendering rule for the whole
modal is simpler to keep correct than a rule that only applies to one of
its three fields.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

#: `_store_snapshot` never writes a NULL `extracted_content` on its one live
#: write path, but a pre-migration or hand-edited row is not ruled out --
#: degrade honestly rather than showing a blank scroll region.
_NO_CONTENT = "This snapshot recorded no page text."


def _snapshot_header(url: Any, created_at: Any) -> Text:
    """The modal's header line: the page's URL, then when it was captured."""
    header = Text()
    header.append(str(url) if url else "Unknown URL", style="bold")
    header.append("\n")
    header.append(f"Captured {created_at or 'at an unknown time'}", style="dim")
    return header


def _snapshot_body(content: Any) -> Text:
    """The snapshot body, rendered as literal text (AC#3 -- see the module
    docstring for the full doctrine this restates).

    `Text(str(...))` never parses Rich markup: a captured page's own
    `[bold red]x[/]`-shaped fragment, or an HTML-extraction artefact that
    happens to be `[link=...]`-shaped, paints as those exact characters.
    """
    return Text(str(content) if content else _NO_CONTENT)


class SnapshotViewModal(ModalScreen[None]):
    """Read-only viewer for one `url_snapshots` row.

    Always dismisses `None` -- like `KeptBriefingsModal`, this modal holds
    no state the pushing screen needs back; every field it shows was handed
    to it at construction time.

    Args:
        url: The page's URL (the item's `url` field).
        created_at: The snapshot row's `created_at` timestamp, or `None`.
        content: The snapshot row's `extracted_content`, or `None`.
    """

    BINDINGS = [("escape", "close", "Close")]

    def __init__(self, *, url: Any, created_at: Any, content: Any) -> None:
        super().__init__()
        self._url = url
        self._created_at = created_at
        self._content = content

    def compose(self) -> ComposeResult:
        with Vertical(id="svm-dialog"):
            yield Static(
                _snapshot_header(self._url, self._created_at), id="svm-header"
            )
            with VerticalScroll(id="svm-body-scroll"):
                yield Static(_snapshot_body(self._content), id="svm-body")
            with Horizontal(id="svm-actions"):
                yield Button("Close", id="svm-close")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "svm-close":
            self.dismiss(None)
