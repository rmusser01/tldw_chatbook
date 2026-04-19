from __future__ import annotations

from typing import Any, Iterable, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class WorkspaceSourcePicker(ModalScreen[Optional[int]]):
    """Bounded picker for selecting a media item to attach as a workspace source."""

    DEFAULT_CSS = """
    WorkspaceSourcePicker {
        align: center middle;
    }
    #workspace-source-picker-dialog {
        width: 70;
        max-width: 90%;
        height: 24;
        background: $surface;
        border: round $accent;
        padding: 1;
    }
    #workspace-source-results {
        height: 1fr;
        border: round $panel;
        margin: 1 0;
    }
    .workspace-source-actions Button {
        margin-right: 1;
    }
    """

    def __init__(
        self,
        service: Any = None,
        results: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.service = service
        self.results = self._normalize_results(results or [])
        self.selected_media_id: Optional[int] = None

    def compose(self) -> ComposeResult:
        with Container(id="workspace-source-picker-dialog"):
            yield Static("Pick Workspace Source")
            yield Input(placeholder="Search media...", id="workspace-source-search-input")
            yield Label("No source selected", id="workspace-source-selection")
            yield ListView(id="workspace-source-results")
            with Horizontal(classes="workspace-source-actions"):
                yield Button("Search", id="workspace-source-search-button", variant="primary")
                yield Button("Select", id="workspace-source-select-button", variant="success")
                yield Button("Cancel", id="workspace-source-cancel-button", variant="default")

    def _normalize_results(self, results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            media_id = item.get("id") or item.get("media_id")
            if media_id is None:
                continue
            normalized.append(
                {
                    "id": media_id,
                    "title": item.get("title") or item.get("name") or f"Media {media_id}",
                    "type": item.get("type") or item.get("media_type") or item.get("source_type") or "media",
                }
            )
        return normalized

    async def _refresh_results_view(self) -> None:
        if not self.is_mounted:
            return
        results_list = self.query_one("#workspace-source-results", ListView)
        await results_list.clear()
        if not self.results:
            await results_list.append(ListItem(Label("No media results.")))
            return
        for item in self.results:
            list_item = ListItem(Label(f"{item['title']} [{item['type']}]"))
            setattr(list_item, "media_id", item["id"])
            await results_list.append(list_item)

    def _update_selected_label(self) -> None:
        if not self.is_mounted:
            return
        label = self.query_one("#workspace-source-selection", Label)
        if self.selected_media_id is None:
            label.update("No source selected")
        else:
            label.update(f"Selected media id: {self.selected_media_id}")

    def select_result(self, media_id: int) -> bool:
        for item in self.results:
            if item["id"] == media_id:
                self.selected_media_id = media_id
                self._update_selected_label()
                return True
        return False

    async def load_results(self, query: str = "") -> list[dict[str, Any]]:
        if self.service is None:
            return self.results

        method = None
        if query:
            for name in ("search_media", "search_media_list", "search_media_library"):
                method = getattr(self.service, name, None)
                if method is not None:
                    break
        else:
            for name in ("list_media", "list_media_items", "get_media_list"):
                method = getattr(self.service, name, None)
                if method is not None:
                    break

        if method is None:
            return self.results

        payload = await method(query) if query else await method()
        items = payload
        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("results") or payload.get("media") or []

        self.results = self._normalize_results(items or [])
        await self._refresh_results_view()
        return self.results

    @on(ListView.Selected, "#workspace-source-results")
    def handle_result_selected(self, event: ListView.Selected) -> None:
        if event.item and hasattr(event.item, "media_id"):
            self.select_result(event.item.media_id)

    @on(Button.Pressed, "#workspace-source-search-button")
    async def handle_search_pressed(self, event: Button.Pressed) -> None:
        query = self.query_one("#workspace-source-search-input", Input).value.strip()
        await self.load_results(query)

    @on(Button.Pressed, "#workspace-source-select-button")
    def handle_select_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(self.selected_media_id)

    @on(Button.Pressed, "#workspace-source-cancel-button")
    def handle_cancel_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    async def on_mount(self) -> None:
        await self._refresh_results_view()
        self._update_selected_label()
