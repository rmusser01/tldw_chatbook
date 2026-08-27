"""Compact durable Watchlists operation receipt for Console."""

from __future__ import annotations

from typing import Any, Mapping

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static


_STATUS_PRESENTATION = {
    "queued": ("◌", "QUEUED"),
    "running": ("◐", "RUNNING"),
    "generating": ("◐", "RUNNING"),
    "completed": ("✓", "COMPLETE"),
    "complete": ("✓", "COMPLETE"),
    "empty": ("○", "EMPTY"),
    "failed": ("!", "FAILED"),
    "cancelled": ("×", "CANCELLED"),
    "canceled": ("×", "CANCELLED"),
}


class WatchlistsOperationCard(Vertical):
    """Render one receipt without retaining tool arguments or result bodies."""

    class InspectRequested(Message):
        def __init__(self, operation_id: str, destination: str) -> None:
            super().__init__()
            self.operation_id = operation_id
            self.destination = destination

    class StopFollowingRequested(Message):
        def __init__(self, operation_id: str) -> None:
            super().__init__()
            self.operation_id = operation_id

    class RetryRequested(Message):
        def __init__(self, operation_id: str) -> None:
            super().__init__()
            self.operation_id = operation_id

    class CancelRequested(Message):
        def __init__(self, operation_id: str) -> None:
            super().__init__()
            self.operation_id = operation_id

    def __init__(
        self,
        operation_id: str,
        *,
        operation: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.operation_id = operation_id
        self._initial_operation = dict(operation or {})
        self.destination = (
            "artifacts" if operation_id.startswith("local:briefing:") else "runs"
        )

    def compose(self) -> ComposeResult:
        yield Static("◌  QUEUED", classes="watchlists-operation-status")
        yield Static("Watchlists operation", classes="watchlists-operation-title")
        yield Static(self.operation_id, classes="watchlists-operation-id")
        yield Static("", classes="watchlists-operation-error")
        with Horizontal(classes="watchlists-operation-actions"):
            yield Button(
                "Artifacts" if self.destination == "artifacts" else "Runs",
                classes="watchlists-operation-inspect compact",
            )
            yield Button(
                "Stop following",
                classes="watchlists-operation-stop-following compact",
            )
            yield Button("Retry", classes="watchlists-operation-retry compact")
            yield Button("Cancel", classes="watchlists-operation-cancel compact")

    def on_mount(self) -> None:
        """Apply the initial durable projection after child widgets exist."""
        self.set_operation(self._initial_operation)

    def set_operation(self, operation: Mapping[str, Any]) -> None:
        """Refresh visible metadata from one durable status projection."""
        status = str(operation.get("status_detail") or "queued").strip().casefold()
        symbol, word = _STATUS_PRESENTATION.get(status, ("?", "UNKNOWN"))
        self.query_one(".watchlists-operation-status", Static).update(
            f"{symbol}  {word}"
        )
        source = operation.get("source")
        collection = operation.get("collection")
        if isinstance(source, Mapping):
            title = source.get("name") or "Source check"
        elif isinstance(collection, Mapping):
            title = collection.get("name") or "News briefing"
        else:
            title = "News briefing" if self.destination == "artifacts" else "Source check"
        self.query_one(".watchlists-operation-title", Static).update(str(title)[:120])
        error = str(operation.get("error_category") or "")[:164]
        error_widget = self.query_one(".watchlists-operation-error", Static)
        error_widget.update(error)
        error_widget.display = bool(error)
        self.query_one(".watchlists-operation-retry", Button).display = bool(
            operation.get("retry_capable")
        )
        self.query_one(".watchlists-operation-cancel", Button).display = bool(
            operation.get("cancel_capable")
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button = event.button
        if button.has_class("watchlists-operation-inspect"):
            self.post_message(self.InspectRequested(self.operation_id, self.destination))
        elif button.has_class("watchlists-operation-stop-following"):
            self.post_message(self.StopFollowingRequested(self.operation_id))
        elif button.has_class("watchlists-operation-retry"):
            self.post_message(self.RetryRequested(self.operation_id))
        elif button.has_class("watchlists-operation-cancel"):
            self.post_message(self.CancelRequested(self.operation_id))
