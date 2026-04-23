"""Writing Suite screen implementation."""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger
from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..Writing_Window import WritingWindow


class WritingScreen(BaseAppScreen):
    """Screen wrapper for Writing Suite functionality."""

    CSS_PATH = None
    BINDINGS = []

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "writing", **kwargs)
        self.writing_window: WritingWindow | None = None

    def compose_content(self) -> ComposeResult:
        self.writing_window = WritingWindow(
            self.app_instance,
            id="writing-window",
            classes="window",
        )
        yield self.writing_window

    def save_state(self) -> Dict[str, Any]:
        try:
            window = self.query_one("#writing-window", WritingWindow)
        except Exception:
            window = self.writing_window
        if window is None:
            return {"source": "local"}
        return window.save_state()

    def restore_state(self, state: Dict[str, Any]) -> None:
        try:
            window = self.query_one("#writing-window", WritingWindow)
        except Exception:
            window = self.writing_window
        if window is None:
            logger.debug("WritingScreen restore_state called before window was composed")
            return
        window.restore_state(state)
