"""Server-mode ingestion source management panel."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Checkbox, Input, Label, ListItem, ListView, Static

if TYPE_CHECKING:
    from ...app import TldwCli


class MediaIngestionSourcePanel(ScrollableContainer):
    """Manage server-backed ingestion sources from the media ingest view."""

    DEFAULT_CSS = """
    MediaIngestionSourcePanel {
        layout: vertical;
        padding: 1;
        height: 100%;
        background: $panel;
    }

    MediaIngestionSourcePanel #source-panel-disabled {
        padding: 2;
        color: $text-muted;
        text-style: italic;
    }

    MediaIngestionSourcePanel #source-panel-main {
        layout: horizontal;
        height: 100%;
    }

    MediaIngestionSourcePanel .source-column {
        width: 1fr;
        min-width: 24;
        padding: 1;
    }

    MediaIngestionSourcePanel .source-section-title {
        text-style: bold;
        margin-bottom: 1;
    }

    MediaIngestionSourcePanel ListView {
        border: solid $secondary;
        background: $boost;
        height: 18;
        margin-bottom: 1;
    }

    MediaIngestionSourcePanel #source-detail {
        border: solid $secondary;
        background: $boost;
        min-height: 10;
        padding: 1;
        margin-bottom: 1;
    }

    MediaIngestionSourcePanel .source-actions {
        layout: horizontal;
        height: auto;
        margin-top: 1;
        margin-bottom: 1;
    }

    MediaIngestionSourcePanel .source-actions Button {
        margin-right: 1;
    }

    MediaIngestionSourcePanel Input {
        margin-bottom: 1;
    }
    """

    runtime_backend: reactive[str] = reactive("local")

    def __init__(self, app_instance: "TldwCli", **kwargs: Any):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.scope_service = getattr(app_instance, "media_reading_scope_service", None)
        self.runtime_state = getattr(app_instance, "media_runtime_state", None)
        self.sources: List[Dict[str, Any]] = []
        self.selected_source: Optional[Dict[str, Any]] = None

    def compose(self) -> ComposeResult:
        yield Static("Server ingestion sources require server mode.", id="source-panel-disabled")
        with Horizontal(id="source-panel-main"):
            with Vertical(classes="source-column"):
                yield Label("Sources", classes="source-section-title")
                yield ListView(id="source-list")
                with Horizontal(classes="source-actions"):
                    yield Button("Refresh", id="refresh-sources-btn")
                    yield Button("Sync Now", id="sync-source-btn", disabled=True)

            with Vertical(classes="source-column"):
                yield Label("Source Detail", classes="source-section-title")
                yield Static("No source selected.", id="source-detail")
                yield Input(placeholder="Policy", id="source-policy-input")
                yield Checkbox("Enabled", id="source-enabled-checkbox")
                with Horizontal(classes="source-actions"):
                    yield Button("Save Settings", id="save-source-btn", disabled=True)
                    yield Button("Upload Archive", id="upload-archive-btn", disabled=True)
                yield Input(placeholder="Archive path", id="archive-path-input")
                yield Label("Source Items", classes="source-section-title")
                yield ListView(id="source-items-list")

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _source_id_value(self, source: Dict[str, Any]) -> Any:
        source_id = source.get("source_id")
        if source_id not in (None, ""):
            return source_id
        source_key = str(source.get("id") or "")
        if ":" in source_key:
            return source_key.rsplit(":", 1)[-1]
        return source_key

    def _source_row_id(self, index: int) -> str:
        return f"source-row-{index}"

    def _source_index_from_widget(self, item: ListItem) -> int:
        return int(str(item.id).replace("source-row-", "", 1))

    def _archive_upload_supported(self, source: Dict[str, Any]) -> bool:
        source_type = str(source.get("source_type") or "")
        sink_type = str(source.get("sink_type") or "")
        return "archive" in source_type or "archive" in sink_type

    async def _clear_list_view(self, selector: str) -> ListView:
        list_view = self.query_one(selector, ListView)
        await list_view.clear()
        return list_view

    def _show_server_ui(self, enabled: bool) -> None:
        disabled_copy = self.query_one("#source-panel-disabled", Static)
        main_panel = self.query_one("#source-panel-main", Horizontal)
        disabled_copy.display = not enabled
        main_panel.display = enabled

    async def refresh_for_mode(self) -> None:
        """Refresh the panel for the current runtime backend."""
        if self.runtime_state is not None:
            self.runtime_backend = str(getattr(self.runtime_state, "runtime_backend", self.runtime_backend) or "local")

        if self.runtime_backend != "server":
            self._show_server_ui(False)
            self.sources = []
            self.selected_source = None
            await self._clear_list_view("#source-list")
            await self._clear_list_view("#source-items-list")
            self.query_one("#source-detail", Static).update("Server ingestion sources require server mode.")
            self.query_one("#save-source-btn", Button).disabled = True
            self.query_one("#sync-source-btn", Button).disabled = True
            self.query_one("#upload-archive-btn", Button).disabled = True
            return

        self._show_server_ui(True)

        if self.scope_service is None:
            self.query_one("#source-detail", Static).update("Media source service is unavailable.")
            return

        sources = await self._maybe_await(self.scope_service.list_ingestion_sources(mode="server"))
        self.sources = [dict(source) for source in list(sources or [])]
        await self._load_source_list()

        if self.sources:
            await self.select_source(0)
        else:
            self.selected_source = None
            self.query_one("#source-detail", Static).update("No server ingestion sources found.")
            await self._clear_list_view("#source-items-list")
            self.query_one("#save-source-btn", Button).disabled = True
            self.query_one("#sync-source-btn", Button).disabled = True
            self.query_one("#upload-archive-btn", Button).disabled = True

    async def _load_source_list(self) -> None:
        list_view = await self._clear_list_view("#source-list")
        if not self.sources:
            await list_view.append(ListItem(Static("No sources configured")))
            return

        for index, source in enumerate(self.sources):
            label = f"{source.get('source_type', 'source')} -> {source.get('sink_type', 'sink')}"
            if source.get("enabled"):
                label = f"{label} [enabled]"
            await list_view.append(ListItem(Static(label), id=self._source_row_id(index)))

    async def select_source(self, index: int) -> None:
        if index < 0 or index >= len(self.sources):
            return

        self.selected_source = dict(self.sources[index])
        self._update_source_detail()
        await self._load_source_items()

    def _update_source_detail(self) -> None:
        detail = self.query_one("#source-detail", Static)
        policy_input = self.query_one("#source-policy-input", Input)
        enabled_checkbox = self.query_one("#source-enabled-checkbox", Checkbox)
        save_button = self.query_one("#save-source-btn", Button)
        sync_button = self.query_one("#sync-source-btn", Button)
        upload_button = self.query_one("#upload-archive-btn", Button)

        if not self.selected_source:
            detail.update("No source selected.")
            policy_input.value = ""
            enabled_checkbox.value = False
            save_button.disabled = True
            sync_button.disabled = True
            upload_button.disabled = True
            return

        source = self.selected_source
        lines = [
            f"ID: {source.get('id', '')}",
            f"Source Type: {source.get('source_type', '')}",
            f"Sink Type: {source.get('sink_type', '')}",
            f"Enabled: {source.get('enabled', False)}",
            f"Last Sync Status: {source.get('last_sync_status', 'unknown')}",
        ]
        if source.get("last_error"):
            lines.append(f"Last Error: {source['last_error']}")
        detail.update("\n".join(lines))

        policy_input.value = str(source.get("policy") or "")
        enabled_checkbox.value = bool(source.get("enabled", False))
        save_button.disabled = False
        sync_button.disabled = False
        upload_button.disabled = not self._archive_upload_supported(source)

    async def _load_source_items(self) -> None:
        items_view = await self._clear_list_view("#source-items-list")
        if not self.selected_source or self.scope_service is None:
            return

        try:
            items = await self._maybe_await(
                self.scope_service.list_ingestion_source_items(
                    mode="server",
                    source_id=self._source_id_value(self.selected_source),
                )
            )
        except Exception as exc:
            logger.error(f"Failed to load ingestion source items: {exc}", exc_info=True)
            await items_view.append(ListItem(Static("Failed to load source items")))
            return

        loaded_items = list(items or [])
        if not loaded_items:
            await items_view.append(ListItem(Static("No source items")))
            return

        for item in loaded_items:
            label = f"{item.get('normalized_relative_path', 'artifact')} [{item.get('sync_status', 'unknown')}]"
            await items_view.append(ListItem(Static(label)))

    @on(ListView.Selected, "#source-list")
    async def handle_source_selected(self, event: ListView.Selected) -> None:
        if not event.item or not event.item.id:
            return
        try:
            await self.select_source(self._source_index_from_widget(event.item))
        except Exception as exc:
            logger.error(f"Error selecting ingestion source: {exc}", exc_info=True)

    @on(Button.Pressed, "#refresh-sources-btn")
    def handle_refresh_sources(self) -> None:
        self.run_worker(self.refresh_for_mode(), exclusive=True)

    @on(Button.Pressed, "#sync-source-btn")
    def handle_sync_source(self) -> None:
        self.run_worker(self._sync_selected_source(), exclusive=True)

    async def _sync_selected_source(self) -> None:
        if not self.selected_source or self.scope_service is None:
            return
        await self._maybe_await(
            self.scope_service.trigger_ingestion_source_sync(
                mode="server",
                source_id=self._source_id_value(self.selected_source),
            )
        )
        self.notify("Source sync triggered", severity="information")

    @on(Button.Pressed, "#save-source-btn")
    def handle_save_source(self) -> None:
        self.run_worker(self._save_selected_source(), exclusive=True)

    async def _save_selected_source(self) -> None:
        if not self.selected_source or self.scope_service is None:
            return

        enabled = self.query_one("#source-enabled-checkbox", Checkbox).value
        policy = self.query_one("#source-policy-input", Input).value.strip() or None
        updated = await self._maybe_await(
            self.scope_service.patch_ingestion_source(
                mode="server",
                source_id=self._source_id_value(self.selected_source),
                enabled=enabled,
                policy=policy,
            )
        )
        self.selected_source = dict(updated)
        for index, source in enumerate(self.sources):
            if source.get("id") == self.selected_source.get("id"):
                self.sources[index] = dict(self.selected_source)
                break
        self._update_source_detail()
        await self._load_source_list()
        self.notify("Source settings updated", severity="information")

    @on(Button.Pressed, "#upload-archive-btn")
    def handle_upload_archive(self) -> None:
        self.run_worker(self._upload_selected_archive(), exclusive=True)

    async def _upload_selected_archive(self) -> None:
        if not self.selected_source or self.scope_service is None:
            return

        archive_path = self.query_one("#archive-path-input", Input).value.strip()
        if not archive_path:
            self.notify("Enter an archive path first", severity="warning")
            return

        await self._maybe_await(
            self.scope_service.upload_ingestion_source_archive(
                mode="server",
                source_id=self._source_id_value(self.selected_source),
                archive_path=archive_path,
            )
        )
        self.notify("Archive upload triggered", severity="information")
