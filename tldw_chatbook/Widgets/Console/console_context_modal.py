"""Modal viewer for the native Console chat context snapshot."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Label,
    LoadingIndicator,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.worker import Worker, WorkerState

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot


SIZE_THRESHOLD_BYTES = 1 * 1024 * 1024


class ConsoleContextModal(ModalScreen[None]):
    """Display the current transcript and assembled next-send payload."""

    DEFAULT_CSS = """
    ConsoleContextModal { align: center middle; }
    #console-context-modal { width: 95; height: 40; border: tall gray; }
    #console-context-header { height: auto; }
    #console-context-warning { height: auto; color: yellow; }
    #console-context-loading { display: none; }
    #console-context-loading.loading { display: block; }
    #console-context-tabs { height: 1fr; }
    #console-context-actions { height: auto; }
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("r", "refresh", "Refresh"),
    ]

    snapshot = reactive(
        ConsoleContextSnapshot(current_messages=[], next_send_payload={})
    )
    raw_json = reactive(False)
    in_progress = reactive(False)
    token_estimate = reactive(None)
    loading = reactive(False)

    def __init__(
        self,
        snapshot_factory: Callable[[], Awaitable[ConsoleContextSnapshot]],
        *,
        token_estimate: int | None = None,
        in_progress: bool = False,
    ) -> None:
        super().__init__()
        self._snapshot_factory = snapshot_factory
        self.token_estimate = token_estimate
        self.in_progress = in_progress

    def compose(self) -> ComposeResult:
        with Vertical(id="console-context-modal"):
            yield Static("Chat Context", id="console-context-header")
            yield Static("", id="console-context-warning")
            yield LoadingIndicator(id="console-context-loading")

            with TabbedContent(id="console-context-tabs"):
                with TabPane("Current", id="console-context-current"):
                    yield Vertical(id="console-context-current-body")
                with TabPane("Next Send", id="console-context-next-send"):
                    yield Vertical(id="console-context-next-send-body")

            with Horizontal(id="console-context-actions"):
                yield Checkbox("Raw JSON", id="console-context-raw")
                yield Button(
                    "Refresh",
                    id="console-context-refresh",
                    disabled=self.in_progress,
                )
                yield Button("Copy JSON", id="console-context-copy")
                yield Button("Save to File", id="console-context-save")
                yield Button("Close", id="console-context-close")

    def on_mount(self) -> None:
        self.run_worker(self._load_snapshot, exclusive=True)

    def watch_snapshot(self) -> None:
        self._update_view()

    def watch_raw_json(self) -> None:
        self._update_view()

    def watch_loading(self) -> None:
        loading = self.query_one("#console-context-loading", LoadingIndicator)
        if self.loading:
            loading.add_class("loading")
        else:
            loading.remove_class("loading")

    def _update_view(self) -> None:
        warning = self.query_one("#console-context-warning", Static)
        if self.in_progress:
            warning.update("A response is in progress; snapshot may change.")
        else:
            warning.update("")

        header = self.query_one("#console-context-header", Static)
        header_text = "Chat Context"
        if self.token_estimate is not None:
            header_text += f" (~{self.token_estimate} tokens)"
        header.update(header_text)

        current_container = self.query_one(
            "#console-context-current-body", Vertical
        )
        current_container.remove_children()
        for widget in self._build_current_context_widgets():
            current_container.mount(widget)

        next_container = self.query_one(
            "#console-context-next-send-body", Vertical
        )
        next_container.remove_children()
        for widget in self._build_next_send_widgets():
            next_container.mount(widget)

    def _build_current_context_widgets(self) -> list:
        if not self.snapshot.current_messages:
            return [Label("No conversation context.")]
        return [
            Collapsible(
                TextArea(msg.content, read_only=True),
                title=f"[{msg.role}] {msg.status}",
                collapsed=True,
            )
            for msg in self.snapshot.current_messages
        ]

    def _build_next_send_widgets(self) -> list:
        widgets: list = []
        payload = self.snapshot.next_send_payload
        text = self._format_next_send_text()

        if len(text.encode("utf-8")) > SIZE_THRESHOLD_BYTES:
            widgets.append(
                Label(
                    "Context exceeds 1 MiB. Use Save to File to view the full payload."
                )
            )
            return widgets

        if self.raw_json:
            widgets.append(TextArea(text, read_only=True))
            return widgets

        widgets.append(
            Collapsible(
                Label(str(payload.get("model", "unknown"))),
                title="Model",
                collapsed=False,
            )
        )

        widgets.append(
            Collapsible(
                TextArea(self._json_block(payload.get("system")), read_only=True),
                title="System",
                collapsed=True,
            )
        )

        message_widgets = []
        for i, msg in enumerate(payload.get("messages", [])):
            message_widgets.append(
                Collapsible(
                    TextArea(self._json_block(msg), read_only=True),
                    title=f"Message {i}",
                    collapsed=True,
                )
            )
        widgets.append(
            Collapsible(
                *message_widgets,
                title="Messages",
                collapsed=False,
            )
        )

        tools = payload.get("tools")
        if tools:
            widgets.append(
                Collapsible(
                    TextArea(self._json_block(tools), read_only=True),
                    title="Tools",
                    collapsed=True,
                )
            )

        staged = payload.get("staged_sources")
        if staged:
            widgets.append(
                Collapsible(
                    TextArea(self._json_block(staged), read_only=True),
                    title="Staged Sources",
                    collapsed=True,
                )
            )

        return widgets

    def _format_next_send_text(self) -> str:
        import json

        return json.dumps(self.snapshot.next_send_payload, indent=2, default=str)

    @staticmethod
    def _json_block(obj: Any) -> str:
        import json

        return json.dumps(obj, indent=2, default=str)

    @on(Button.Pressed, "#console-context-close")
    def _close(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#console-context-refresh")
    def _refresh(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(self._load_snapshot, exclusive=True)

    async def _load_snapshot(self) -> None:
        self.loading = True
        try:
            self.snapshot = await self._snapshot_factory()
        finally:
            self.loading = False

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.ERROR:
            self.loading = False
            self.notify("Failed to refresh context.", severity="error")

    @on(Checkbox.Changed, "#console-context-raw")
    def _toggle_raw(self, event: Checkbox.Changed) -> None:
        event.stop()
        self.raw_json = event.value

    @on(Button.Pressed, "#console-context-copy")
    def _copy_json(self, event: Button.Pressed) -> None:
        event.stop()
        import json

        text = json.dumps(self.snapshot.next_send_payload, indent=2, default=str)
        try:
            import pyperclip

            pyperclip.copy(text)
            self.notify("JSON copied to clipboard.")
        except Exception:
            self.notify("Copy failed: pyperclip unavailable.", severity="warning")

    @on(Button.Pressed, "#console-context-save")
    def _save_json(self, event: Button.Pressed) -> None:
        event.stop()
        import json
        from datetime import datetime
        from pathlib import Path

        text = json.dumps(self.snapshot.next_send_payload, indent=2, default=str)
        filename = f"chatbook_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = Path.home() / "Downloads" / filename
        path.write_text(text, encoding="utf-8")
        self.notify(f"Saved to {path}")

    def action_refresh(self) -> None:
        self.run_worker(self._load_snapshot, exclusive=True)

    def action_dismiss(self) -> None:
        self.dismiss(None)
