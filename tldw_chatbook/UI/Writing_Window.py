"""Writing Suite window shell.

Task 8 replaces this placeholder with the source-switched browse and outline UI.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Label, Static


class WritingWindow(Container):
    """Minimal Writing Suite container used by the screen/navigation wiring."""

    def __init__(self, app_instance: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.source = "local"

    def compose(self) -> ComposeResult:
        yield Label("Writing Suite")
        yield Static("Writing Suite is initializing.", id="writing-status")

    def save_state(self) -> dict[str, Any]:
        return {"source": self.source}

    def restore_state(self, state: dict[str, Any]) -> None:
        source = str((state or {}).get("source") or "local").strip().lower()
        self.source = source if source in {"local", "server"} else "local"
