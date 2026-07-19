"""Search/RAG screen implementation."""

from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger
from textual.app import ComposeResult
from textual.widgets import Input, Select, TabbedContent

from ..Navigation.base_app_screen import BaseAppScreen
from ..SearchRAGWindow import SearchRAGWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


# Static, compose-time-fixed option sets used to validate a restored value
# before assigning it to a widget -- never trust a saved-state dict's shape,
# and Select/TabbedContent raise on an out-of-range assignment rather than
# tolerating one.
_SEARCH_MODE_VALUES = {"plain", "contextual", "hybrid"}
_SEARCH_TAB_IDS = {"search-tab", "saved-tab", "history-tab", "maintenance-tab"}


class SearchScreen(BaseAppScreen):
    """
    Search/RAG screen wrapper.
    """

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "search", **kwargs)
        self.search_window = None
        # Stashed by restore_state on this (pre-mount) instance; applied to
        # the freshly composed SearchRAGWindow once it exists (see
        # on_mount). The window is rebuilt fresh every visit, so there is
        # no earlier seam to inject restored state into before it mounts.
        self._pending_search_restore: Optional[Dict[str, Any]] = None

    def compose_content(self) -> ComposeResult:
        """Compose the search window content."""
        self.search_window = SearchRAGWindow(self.app_instance)
        # Add the window class after creation
        self.search_window.add_class("window")
        # Yield the window widget directly
        yield self.search_window

    def on_mount(self) -> None:
        super().on_mount()
        if self._pending_search_restore and self.search_window is not None:
            self._apply_pending_search_restore(self._pending_search_restore)
            self._pending_search_restore = None

    def _apply_pending_search_restore(self, restore: Dict[str, Any]) -> None:
        """Apply a stashed restore dict to the now-mounted search window.

        Mirrors ``SearchRAGWindow.handle_saved_search_load_requested``'s own
        "apply a config to the live controls" pattern. Every assignment is
        individually guarded so one bad/stale value (e.g. a saved-state dict
        from a build that used different tab ids) can't abort the rest of
        the restore or crash navigation.
        """
        window = self.search_window
        query = restore.get("query")
        if query:
            try:
                window.query_one("#search-query-input", Input).value = str(query)
            except Exception:
                logger.opt(exception=True).debug("Could not restore Search query input")

        mode = restore.get("mode")
        if mode in _SEARCH_MODE_VALUES:
            try:
                window.query_one("#search-mode-select", Select).value = mode
            except Exception:
                logger.opt(exception=True).debug("Could not restore Search mode select")

        active_tab = restore.get("active_tab")
        if active_tab in _SEARCH_TAB_IDS:
            try:
                window.query_one("#search-tabs", TabbedContent).active = active_tab
            except Exception:
                logger.opt(exception=True).debug("Could not restore Search active tab")

    def save_state(self) -> Dict[str, Any]:
        """Save the Search window's user-facing view state.

        Reads live values off the mounted window; every read is
        individually guarded so this can never raise -- the app calls it
        unconditionally while navigating away.
        """
        state = super().save_state()
        window = self.search_window
        if window is None:
            return state

        try:
            state["search_query"] = window.query_one("#search-query-input", Input).value
        except Exception:
            logger.opt(exception=True).debug("Could not read Search query input for save_state")
        try:
            state["search_mode"] = window.query_one("#search-mode-select", Select).value
        except Exception:
            logger.opt(exception=True).debug("Could not read Search mode select for save_state")
        try:
            state["search_active_tab"] = window.query_one("#search-tabs", TabbedContent).active
        except Exception:
            logger.opt(exception=True).debug("Could not read Search active tab for save_state")
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Stash restored Search view state for ``on_mount`` to apply.

        Called on a freshly-constructed, not-yet-mounted screen -- the
        SearchRAGWindow it will compose does not exist yet, so the values
        are held here and handed to ``on_mount`` once ``compose_content``
        has run.
        """
        super().restore_state(state)
        if not isinstance(state, dict):
            return
        self._pending_search_restore = {
            "query": state.get("search_query"),
            "mode": state.get("search_mode"),
            "active_tab": state.get("search_active_tab"),
        }
