"""Items pane for the watchlists screen."""

from __future__ import annotations

from typing import Any

from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Select, Static


class ItemSelected(Message):
    """Posted when the user selects an item in the items table."""

    def __init__(self, item: dict[str, Any] | None) -> None:
        self.item = item
        super().__init__()


class RefreshItemsRequested(Message):
    """Posted when the user requests a refresh of the items list."""


class ItemsPane(Vertical):
    """Content item list and filter for watchlists."""

    items = reactive[list[dict[str, Any]]]([], recompose=True)
    selected_item = reactive[dict[str, Any] | None](None)
    status_filter = reactive("all", recompose=True)
    search_query = reactive("", recompose=True)
    runtime_backend = reactive("local", recompose=True)

    _STATUS_OPTIONS = [
        ("All statuses", "all"),
        ("New", "new"),
        ("Reviewed", "reviewed"),
        ("Ingested", "ingested"),
        ("Ignored", "ignored"),
        ("Error", "error"),
    ]

    def compose(self):
        with Horizontal(id="items-toolbar", classes="destination-filter-strip"):
            yield Button("Refresh", id="items-refresh-button", variant="primary")
            yield Input(
                placeholder="Search items...",
                id="items-search-input",
                value=self.search_query,
            )
            yield Select(
                self._STATUS_OPTIONS,
                value=self.status_filter,
                id="items-status-select",
                allow_blank=False,
            )

        table = DataTable(id="items-table")
        table.add_columns("Title", "Source", "Status", "Created")
        filtered = self._filtered_items()
        for item in filtered:
            table.add_row(
                str(item.get("title") or "Untitled"),
                str(item.get("source_name") or "-"),
                str(item.get("status") or "-"),
                str(item.get("created_at") or "-"),
                key=str(item.get("id") or id(item)),
            )
        yield table

    def _filtered_items(self) -> list[dict[str, Any]]:
        status_filter = self.status_filter
        query = self.search_query.strip().lower()
        results: list[dict[str, Any]] = []
        for item in self.items:
            if status_filter != "all" and str(item.get("status") or "").lower() != status_filter:
                continue
            if query:
                text = " ".join(
                    str(item.get(key) or "") for key in ("title", "url", "source_name", "status")
                ).lower()
                if query not in text:
                    continue
            results.append(item)
        return results

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "items-search-input":
            self.search_query = event.value
        event.stop()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "items-status-select":
            self.status_filter = str(event.value or "all")
        event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "items-refresh-button":
            self.post_message(RefreshItemsRequested())
        event.stop()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.select_item_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        event.stop()
        self.select_item_by_id(str(event.cell_key.row_key.value))

    def select_item_by_id(self, item_id: str) -> None:
        """Select the item with the given id and notify listeners."""
        item = None
        for candidate in self.items:
            if str(candidate.get("id") or "") == item_id:
                item = candidate
                break
        self.selected_item = item

    def watch_selected_item(self, item: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(ItemSelected(item))
