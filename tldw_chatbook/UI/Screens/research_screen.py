"""Research Sessions screen implementation."""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger
from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..Research_Window import ResearchWindow


class ResearchScreen(BaseAppScreen):
    """Screen wrapper for Research Sessions functionality."""

    CSS_PATH = None
    BINDINGS = []

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "research", **kwargs)
        self.research_window: ResearchWindow | None = None
        self._pending_restore_state: Dict[str, Any] | None = None

    def compose_content(self) -> ComposeResult:
        """Compose the screen's content: the ResearchWindow plus any pending
        state restoration."""
        self.research_window = ResearchWindow(
            self.app_instance,
            id="research-window",
            classes="window",
        )
        if self._pending_restore_state is not None:
            self.research_window.restore_state(self._pending_restore_state)
            self._pending_restore_state = None
        yield self.research_window

    def save_state(self) -> Dict[str, Any]:
        """Persist the hosted window's state for navigation restore.

        Returns:
            The window's save-state dict, or the pending restore state when
            the window has not been composed yet.
        """
        try:
            window = self.query_one("#research-window", ResearchWindow)
        except Exception:
            window = self.research_window
        if window is None:
            return self._pending_restore_state or {"source": "local"}
        return window.save_state()

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore the hosted window's state after navigation.

        Args:
            state: The state dict previously returned by ``save_state``;
                deferred until compose when the window is not mounted yet.
        """
        try:
            window = self.query_one("#research-window", ResearchWindow)
        except Exception:
            window = self.research_window
        if window is None:
            logger.debug("Deferring ResearchScreen restore_state until window is composed")
            self._pending_restore_state = dict(state or {})
            return
        window.restore_state(state)
