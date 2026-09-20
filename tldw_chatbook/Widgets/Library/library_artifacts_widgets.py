"""Artifact reader controls; storage and action ownership stay in the controller."""

from __future__ import annotations

import asyncio

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.widgets import Button, Input, Markdown, OptionList, Static
from textual.widgets.option_list import Option


class ArtifactSearch(Input):
    """Clear both the displayed query and its requested scope before pane return."""

    def on_key(self, event: Key) -> None:
        if event.key == "escape" and self.value:
            event.stop()
            event.prevent_default()
            self.value = ""


class ArtifactItems(Vertical):
    """Search, retention filter, bounded results and paging controls."""

    def __init__(self, controller, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self._page = None
        self._syncing = False

    def compose(self) -> ComposeResult:
        yield Static("Reports · Local", id="library-artifacts-heading", markup=False)
        yield ArtifactSearch(
            value=self.controller.scope.query,
            placeholder="Find reports by Watchlist…",
            id="library-artifacts-search",
        )
        with Horizontal(classes="library-artifacts-toolbar"):
            yield Button("All reports", id="library-artifacts-all", compact=True)
            yield Button("Kept", id="library-artifacts-kept", compact=True)
            yield Button("Newest", id="library-artifacts-sort", compact=True)
        yield Static("Loading reports…", id="library-artifacts-count", markup=False)
        yield OptionList(id="library-artifacts-list")
        with Horizontal(classes="library-artifacts-toolbar"):
            yield Button("First", id="library-artifacts-first", compact=True)
            yield Button("Prev", id="library-artifacts-prev", compact=True)
            yield Button("Next", id="library-artifacts-next", compact=True)
            yield Button("Last", id="library-artifacts-last", compact=True)
        yield Button("Retry", id="library-artifacts-retry", compact=True)
        yield Button(
            "Manage Chatbook packs…", id="library-artifacts-manage", compact=True
        )

    def sync(self) -> None:
        c = self.controller
        page = c.page
        title = {
            "reports": "Reports",
            "chatbooks": "Chatbooks",
            "all": "All artifacts",
        }[c.scope.view]
        self.query_one("#library-artifacts-heading", Static).update(f"{title} · Local")
        self.query_one(Input).placeholder = (
            "Find reports by Watchlist…"
            if c.scope.view == "reports"
            else "Find artifacts by title…"
        )
        self.query_one("#library-artifacts-manage").display = c.scope.view != "reports"
        self.query_one("#library-artifacts-all").display = c.scope.view == "reports"
        self.query_one("#library-artifacts-kept").display = c.scope.view == "reports"
        self._syncing = True
        try:
            rows = self.query_one(OptionList)
            if page is not self._page:
                self._page = page
                rows.clear_options()
                if page:
                    from ...Subscriptions.daily_reports_view import (
                        format_report_timestamp,
                    )

                    rows.add_options(
                        [
                            Option(
                                Text(
                                    f"{row.title}\n{format_report_timestamp(row.created_at)}\n"
                                    f"{row.copy_label} · {row.status} · #{row.key.native_id}",
                                    overflow="ellipsis",
                                ),
                                id=f"{row.key.source}:{row.key.native_id}",
                            )
                            for row in page.items
                        ]
                    )
            if page and c.selected:
                index = next(
                    (i for i, row in enumerate(page.items) if row.key == c.selected),
                    None,
                )
                if index is not None and rows.highlighted != index:
                    rows.highlighted = index
            status = "Loading…" if c.loading else c.error
            if not status and page:
                status = (
                    f"{page.start + 1}–{page.start + len(page.items)} of {page.total} copies"
                    if page.items
                    else "No matches. Clear search or change the filter."
                    if c.scope.query or c.scope.kept_only
                    else "No Chatbooks yet. Open Manage Chatbook packs."
                    if c.scope.view == "chatbooks"
                    else "No reports yet. Open Watchlists or try the report demo."
                )
            self.query_one("#library-artifacts-count", Static).update(status)
            self.query_one("#library-artifacts-retry").display = bool(
                c.error or c.detail_error
            )
            for name, blocked in (
                ("first", not page or page.start == 0),
                ("prev", not page or page.start == 0),
                ("next", not page or page.start + len(page.items) >= page.total),
                ("last", not page or page.start + len(page.items) >= page.total),
            ):
                self.query_one(f"#library-artifacts-{name}", Button).disabled = (
                    c.loading or bool(c.error) or blocked
                )
            self.query_one("#library-artifacts-all", Button).label = (
                "All reports" if c.scope.kept_only else "✓ All reports"
            )
            self.query_one("#library-artifacts-kept", Button).label = (
                "✓ Kept" if c.scope.kept_only else "Kept"
            )
            self.query_one("#library-artifacts-sort", Button).label = (
                "Newest" if c.scope.sort == "newest" else "A–Z"
            )
        finally:
            self._syncing = False

    @on(Input.Changed, "#library-artifacts-search")
    def search_changed(self, event: Input.Changed) -> None:
        event.stop()
        self.controller.search(event.value)

    @on(OptionList.OptionHighlighted, "#library-artifacts-list")
    def highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        if (
            not self._syncing
            and self.controller.page
            and event.option_index < len(self.controller.page.items)
        ):
            self.controller.select(self.controller.page.items[event.option_index].key)

    @on(OptionList.OptionSelected, "#library-artifacts-list")
    def selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.controller.focus_reader()

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.controller.action(
            (event.button.id or "").removeprefix("library-artifacts-")
        )


class ArtifactWork(Vertical):
    """Read-only report body, provenance and capability-specific actions."""

    def __init__(self, controller, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self._markdown_body = None
        self._markdown_update = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="library-artifacts-toolbar"):
            yield Button("‹ Items", id="library-artifacts-back", compact=True)
            yield Button("Preview", id="library-artifacts-preview", compact=True)
            yield Button("Details", id="library-artifacts-details", compact=True)
        yield Static("Select a report", id="library-artifacts-title", markup=False)
        yield Static("", id="library-artifacts-retention", markup=False)
        with VerticalScroll(id="library-artifacts-body"):
            yield Static(
                "Choose an item to read it here.",
                id="library-artifacts-content",
                markup=False,
            )
            yield Markdown("", id="library-artifacts-markdown", open_links=False)
        with Horizontal(classes="library-artifacts-toolbar"):
            yield Button("Keep in Library", id="library-artifacts-keep", compact=True)
            yield Button("Export…", id="library-artifacts-export", compact=True)
            yield Button("Scripts…", id="library-artifacts-scripts", compact=True)
            yield Button("Play", id="library-artifacts-play", compact=True)
        with Horizontal(classes="library-artifacts-toolbar"):
            yield Button("Share…", id="library-artifacts-share", compact=True)
            yield Button("Use in Console", id="library-artifacts-console", compact=True)
            yield Button("Open source", id="library-artifacts-source", compact=True)
        with Horizontal(classes="library-artifacts-toolbar"):
            yield Button("Watchlists", id="library-artifacts-watchlists", compact=True)
            yield Button("Try report demo", id="library-artifacts-demo", compact=True)

    def sync(self) -> None:
        c = self.controller
        detail = c.detail
        row = (
            next((row for row in c.page.items if row.key == c.selected), None)
            if c.page
            else None
        )
        self.query_one("#library-artifacts-title", Static).update(
            row.title if row else "Select an artifact"
        )
        retained = c.selected and c.selected.source == "kept_report"
        chatbook = c.selected and c.selected.source == "chatbook"
        metadata = dict(detail.details) if detail else {}
        self.query_one("#library-artifacts-retention", Static).update(
            " · ".join(
                filter(
                    None,
                    (
                        metadata.get("Preview", "Registered Chatbook"),
                        "Original was truncated" if detail and detail.truncated else "",
                        metadata.get("Sharing", ""),
                    ),
                )
            )
            if chatbook
            else "Kept in Library · independent saved copy"
            if retained
            else "Watchlist copy · disappears if its Watchlist is deleted"
            if c.selected
            else ""
        )
        if c.detail_loading:
            body = Text("Loading artifact…")
        elif not detail:
            body = Text(c.detail_error or "Choose an item to read it here.")
        elif c.mode == "details":
            body = Text(
                "\n".join(f"{label}: {value}" for label, value in detail.details)
            )
        elif detail.body:
            body = Text("")
        else:
            body = Text(
                f"This report is {row.status if row else 'unavailable'}. Open Watchlists for its status."
            )
        preview = bool(
            detail and detail.body and c.mode == "preview" and not c.detail_loading
        )
        content = self.query_one("#library-artifacts-content", Static)
        content.display = not preview
        content.update(body)
        markdown = self.query_one("#library-artifacts-markdown", Markdown)
        markdown.display = preview
        if preview and self._markdown_body != detail.body:
            self._markdown_body = detail.body
            self._markdown_update = markdown.update(detail.body)
        ready = bool(
            detail
            and not c.loading
            and not c.error
            and not c.detail_loading
            and not c.busy
        )
        for name, allowed in (
            ("keep", detail and detail.can_keep),
            ("export", detail and detail.can_export),
            ("play", detail and detail.can_play),
            ("scripts", retained),
            ("share", detail and detail.can_share),
            ("console", chatbook),
            ("source", chatbook and detail and detail.source_available),
        ):
            button = self.query_one(f"#library-artifacts-{name}", Button)
            button.display = bool(allowed)
            button.disabled = not ready
        self.query_one("#library-artifacts-demo").display = (
            c.scope.view != "chatbooks"
            and (not c.selected or bool(row and row.status == "failed"))
        )
        self.query_one("#library-artifacts-watchlists").display = (
            c.scope.view != "chatbooks" and not chatbook
        )
        for toolbar in self.query(Horizontal):
            toolbar.display = any(button.display for button in toolbar.query(Button))

    async def wait_for_body(self) -> None:
        """Wait for the current preview's blocks without cancelling their mount."""
        if self.controller.mode == "preview" and self._markdown_update is not None:
            await asyncio.shield(self._markdown_update)

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.controller.action(
            (event.button.id or "").removeprefix("library-artifacts-")
        )
