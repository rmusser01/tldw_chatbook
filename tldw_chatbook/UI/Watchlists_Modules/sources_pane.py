"""Sources pane for the watchlists screen."""

from __future__ import annotations

from typing import Any

from textual.containers import Grid, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Select, Static, Switch

from .inspector_pane import CheckNowRequested, PreviewRequested


class SourceSelected(Message):
    """Posted when the user selects a source in the sources table."""

    def __init__(self, source: dict[str, Any] | None) -> None:
        self.source = source
        super().__init__()


class CreateSourceRequested(Message):
    """Posted when the user submits the new-source form."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__()


class ImportOpmlRequested(Message):
    """Posted when the user requests an OPML import."""


class ExportOpmlRequested(Message):
    """Posted when the user requests an OPML export."""


class SourcesPane(Vertical):
    """Source list, search/filter, and create form for watchlists."""

    sources = reactive[list[dict[str, Any]]]([], recompose=True)
    selected_source = reactive[dict[str, Any] | None](None)
    search_query = reactive("", recompose=True)
    source_type_filter = reactive("all", recompose=True)
    show_create_form = reactive(False, recompose=True)

    _TYPE_OPTIONS = [
        ("All", "all"),
        ("RSS", "rss"),
        ("Atom", "atom"),
        ("Feed", "feed"),
        ("Playlist", "playlist"),
        ("Channel", "channel"),
    ]

    def compose(self):
        with Vertical(id="sources-toolbar"):
            with Horizontal(classes="destination-filter-strip"):
                yield Input(
                    placeholder="Search sources...",
                    id="sources-search-input",
                    value=self.search_query,
                )
                yield Select(
                    self._TYPE_OPTIONS,
                    value=self.source_type_filter,
                    id="sources-type-select",
                    allow_blank=False,
                )
                yield Button("New Source", id="sources-new-button", variant="primary")
            with Horizontal(classes="destination-filter-strip"):
                yield Button(
                    "Preview",
                    id="sources-preview-button",
                    disabled=self.selected_source is None,
                )
                yield Button(
                    "Check now",
                    id="sources-check-now-button",
                    disabled=self.selected_source is None,
                )
                yield Button("Import OPML", id="sources-import-opml-button")
                yield Button("Export OPML", id="sources-export-opml-button")

        if self.show_create_form:
            with Grid(id="sources-create-form"):
                yield Input(placeholder="Name", id="sources-create-name")
                yield Input(placeholder="URL", id="sources-create-url")
                yield Select(
                    [(label, value) for label, value in self._TYPE_OPTIONS if value != "all"],
                    value="rss",
                    id="sources-create-type",
                    allow_blank=False,
                )
                yield Horizontal(
                    Static("Active"),
                    Switch(value=True, id="sources-create-active"),
                    classes="sources-create-active-row",
                )
                yield Input(placeholder="Tags (comma separated)", id="sources-create-tags")
                yield Button("Create", id="sources-create-submit", variant="success")
                yield Button("Cancel", id="sources-create-cancel", variant="default")

        table = DataTable(id="sources-table")
        table.add_columns("Name", "Type", "Status", "Last scraped", "Active")
        filtered = self._filtered_sources()
        for source in filtered:
            table.add_row(
                str(source.get("name") or source.get("title") or "Untitled"),
                str(source.get("source_type") or "-"),
                str(source.get("status") or "-"),
                str(source.get("last_scraped") or "-"),
                "Yes" if source.get("active") else "No",
                key=str(source.get("id") or id(source)),
            )
        yield table

    def _filtered_sources(self) -> list[dict[str, Any]]:
        query = self.search_query.strip().lower()
        type_filter = self.source_type_filter
        results: list[dict[str, Any]] = []
        for source in self.sources:
            if type_filter != "all" and str(source.get("source_type") or "").lower() != type_filter:
                continue
            if query:
                text = " ".join(
                    str(source.get(key) or "") for key in ("name", "title", "url", "source_type", "status")
                ).lower()
                if query not in text:
                    continue
            results.append(source)
        return results

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "sources-search-input":
            self.search_query = event.value
        event.stop()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sources-type-select":
            self.source_type_filter = str(event.value or "all")
        event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "sources-new-button":
            self.show_create_form = True
        elif button_id == "sources-create-cancel":
            self.show_create_form = False
        elif button_id == "sources-create-submit":
            self._submit_create_form()
        elif button_id == "sources-preview-button" and self.selected_source is not None:
            self.post_message(PreviewRequested(self.selected_source))
        elif button_id == "sources-check-now-button" and self.selected_source is not None:
            self.post_message(CheckNowRequested(self.selected_source))
        elif button_id == "sources-import-opml-button":
            self.post_message(ImportOpmlRequested())
        elif button_id == "sources-export-opml-button":
            self.post_message(ExportOpmlRequested())
        event.stop()

    def _submit_create_form(self) -> None:
        name = self.query_one("#sources-create-name", Input).value.strip()
        url = self.query_one("#sources-create-url", Input).value.strip()
        source_type = str(self.query_one("#sources-create-type", Select).value or "rss")
        active = self.query_one("#sources-create-active", Switch).value
        tags_text = self.query_one("#sources-create-tags", Input).value.strip()
        tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()] if tags_text else []
        self.post_message(
            CreateSourceRequested(
                {
                    "name": name,
                    "url": url,
                    "source_type": source_type,
                    "active": active,
                    "tags": tags,
                }
            )
        )
        self.show_create_form = False

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.select_source_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        event.stop()
        self.select_source_by_id(str(event.cell_key.row_key.value))

    def select_source_by_id(self, source_id: str) -> None:
        """Select the source with the given id and notify listeners."""
        source = None
        for candidate in self.sources:
            if str(candidate.get("id") or "") == source_id:
                source = candidate
                break
        self.selected_source = source

    def watch_selected_source(self, source: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(SourceSelected(source))
