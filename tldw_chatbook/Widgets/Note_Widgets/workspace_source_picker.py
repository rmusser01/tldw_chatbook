from __future__ import annotations

from typing import Any, Iterable, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from tldw_chatbook.tldw_api.notes_workspace_schemas import MediaSearchRequest


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
        self.error_message: Optional[str] = None

    def _get_client(self) -> Any:
        if self.service is None:
            return None
        explicit_client = getattr(self.service, "__dict__", {}).get("client")
        return explicit_client if explicit_client is not None else self.service

    def compose(self) -> ComposeResult:
        with Container(id="workspace-source-picker-dialog"):
            yield Static("Pick Workspace Source")
            yield Input(placeholder="Search media...", id="workspace-source-search-input")
            yield Label("No source selected", id="workspace-source-selection")
            yield Label("", id="workspace-source-error")
            yield ListView(id="workspace-source-results")
            with Horizontal(classes="workspace-source-actions"):
                yield Button("Search", id="workspace-source-search-button", variant="primary")
                yield Button("Select", id="workspace-source-select-button", variant="success")
                yield Button("Cancel", id="workspace-source-cancel-button", variant="default")

    def _coerce_payload_items(self, payload: Any) -> list[Any]:
        if payload is None:
            return []
        if isinstance(payload, dict):
            return list(payload.get("items") or payload.get("results") or payload.get("media") or [])
        if hasattr(payload, "items"):
            items = getattr(payload, "items")
            if isinstance(items, list):
                return items
        if isinstance(payload, (list, tuple)):
            return list(payload)
        return []

    def _normalize_result_item(self, item: Any) -> Optional[dict[str, Any]]:
        if hasattr(item, "model_dump"):
            item = item.model_dump(mode="json")
        elif not isinstance(item, dict) and hasattr(item, "__dict__"):
            item = dict(item.__dict__)

        if not isinstance(item, dict):
            return None

        media_id = item.get("id") or item.get("media_id")
        if media_id is None:
            return None
        return {
            "id": media_id,
            "title": item.get("title") or item.get("name") or f"Media {media_id}",
            "type": item.get("type") or item.get("media_type") or item.get("source_type") or "media",
        }

    def _normalize_results(self, results: Iterable[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in results:
            normalized_item = self._normalize_result_item(item)
            if normalized_item is not None:
                normalized.append(normalized_item)
        return normalized

    def _revalidate_selection(self) -> None:
        if self.selected_media_id is None:
            self._update_selected_label()
            return
        if any(item["id"] == self.selected_media_id for item in self.results):
            self._update_selected_label()
            return
        self.selected_media_id = None
        self._update_selected_label()

    def _clear_selection(self) -> None:
        self.selected_media_id = None
        self._update_selected_label()

    def _set_error(self, message: str) -> None:
        self.error_message = message
        if self.is_mounted:
            self.query_one("#workspace-source-error", Label).update(message)

    def _clear_error(self) -> None:
        self.error_message = None
        if self.is_mounted:
            self.query_one("#workspace-source-error", Label).update("")

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
        client = self._get_client()
        if client is None:
            return self.results

        try:
            payload: Any
            if query:
                search_media_items = getattr(client, "search_media_items", None)
                if search_media_items is not None:
                    payload = await search_media_items(
                        request_data=MediaSearchRequest(query=query),
                        page=1,
                        results_per_page=10,
                    )
                else:
                    legacy_search = None
                    for name in ("search_media", "search_media_list", "search_media_library"):
                        legacy_search = getattr(client, name, None)
                        if legacy_search is not None:
                            break
                    if legacy_search is None:
                        return self.results
                    payload = await legacy_search(query)
            else:
                list_media_items = getattr(client, "list_media_items", None)
                if list_media_items is not None:
                    payload = await list_media_items(
                        page=1,
                        results_per_page=10,
                        include_keywords=False,
                    )
                else:
                    legacy_list = None
                    for name in ("list_media", "get_media_list"):
                        legacy_list = getattr(client, name, None)
                        if legacy_list is not None:
                            break
                    if legacy_list is None:
                        return self.results
                    payload = await legacy_list()

            items = self._coerce_payload_items(payload)
            self.results = self._normalize_results(items)
            self._clear_error()
            self._revalidate_selection()
        except Exception as exc:
            self.results = []
            self._clear_selection()
            self._set_error(f"Failed to load sources: {exc}")

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
        self._clear_error()
