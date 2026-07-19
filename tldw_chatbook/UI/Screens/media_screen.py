"""Media screen implementation."""

from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger
from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..MediaWindow_v2 import MediaWindow
from .media_runtime_state import MediaRuntimeState

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class MediaScreen(BaseAppScreen):
    """
    Media management screen wrapper.
    """

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "media", **kwargs)
        self.media_window = None
        self.media_runtime_state: MediaRuntimeState = app_instance.media_runtime_state
        # Stashed by restore_state on this (pre-mount) instance; applied to
        # the freshly composed MediaWindow once it exists (see on_mount).
        # MediaWindow itself is rebuilt fresh every visit -- there is no
        # earlier seam to inject restored view state into before it mounts.
        self._pending_media_restore: Optional[Dict[str, Any]] = None

    def compose_content(self) -> ComposeResult:
        """Compose the media window content."""
        self.media_window = MediaWindow(self.app_instance, classes="window")
        self.media_window.runtime_state = self.media_runtime_state
        # Yield the window widget directly
        yield self.media_window

    def on_mount(self) -> None:
        super().on_mount()
        if self._pending_media_restore and self.media_window is not None:
            # Nothing else seeds MediaWindow's active_media_type on a fresh
            # visit under screen navigation (the legacy watch_current_tab ->
            # activate_initial_view path is a deliberate no-op once
            # ``_use_screen_navigation`` is set -- see its own early return),
            # so this is the only place a returning visit's type/search/
            # selection gets re-applied.
            try:
                self.media_window.apply_restored_view_state(self._pending_media_restore)
            except Exception:
                logger.opt(exception=True).error("Error applying restored Media view state")
            self._pending_media_restore = None

    def save_state(self) -> Dict[str, Any]:
        """Save the Media window's user-facing view state.

        Reads live values off the mounted window rather than duplicating
        them in screen-level attrs, so this can never drift from what the
        user is actually looking at. Every read is individually guarded --
        this must never raise, since the app calls it unconditionally while
        navigating away.
        """
        state = super().save_state()
        window = self.media_window
        if window is None:
            return state

        try:
            state["media_active_type"] = window.active_media_type
        except Exception:
            logger.opt(exception=True).debug("Could not read Media active_media_type for save_state")
        try:
            state["media_selected_id"] = window.selected_media_id
        except Exception:
            logger.opt(exception=True).debug("Could not read Media selected_media_id for save_state")

        search_panel = getattr(window, "search_panel", None)
        if search_panel is not None:
            try:
                state["media_search_term"] = search_panel.search_term
            except Exception:
                logger.opt(exception=True).debug("Could not read Media search_term for save_state")
            try:
                state["media_keyword_filter"] = search_panel.keyword_filter
            except Exception:
                logger.opt(exception=True).debug("Could not read Media keyword_filter for save_state")
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Stash restored Media view state for ``on_mount`` to apply.

        Called on a freshly-constructed, not-yet-mounted screen -- the
        MediaWindow it will compose does not exist yet, so the values are
        held here and handed to ``MediaWindow.apply_restored_view_state``
        once ``compose_content``/``on_mount`` have run.
        """
        super().restore_state(state)
        if not isinstance(state, dict):
            return
        self._pending_media_restore = {
            "active_media_type": state.get("media_active_type"),
            "selected_media_id": state.get("media_selected_id"),
            "search_term": state.get("media_search_term"),
            "keyword_filter": state.get("media_keyword_filter"),
        }
