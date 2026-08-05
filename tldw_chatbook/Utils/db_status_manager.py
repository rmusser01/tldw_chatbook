"""
Database Status Manager - Centralized management for database size telemetry.

Computes the local database file sizes on a timer and caches them on the
app (``db_sizes_status``) for the Library Details disclosure, plus logs
them. Token-count footer updates route through here as well.
"""

from typing import Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from textual.app import App
    from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


class DBStatusManager:
    """Manages database size telemetry and footer token-count updates."""

    def __init__(self, app: "App"):
        """
        Initialize the database status manager.

        Args:
            app: The Textual app instance
        """
        self.app = app
        self._update_timer = None

    async def update_db_sizes(self) -> None:
        """
        Compute the local database sizes for telemetry consumers.

        F-014: the sizes no longer render in the app footer (a fresh
        install's "Prompts: N/A | Chats/Notes: N/A | Media: N/A" triplet
        read as "broken" in user chrome). They are cached on the app as
        ``db_sizes_status`` -- the Library rail's Details disclosure
        renders them from there -- and logged (the Logs destination is
        the other telemetry home).
        """
        logger.debug("Computing DB sizes for the db_sizes_status cache.")

        try:
            # Import here to avoid circular imports
            from tldw_chatbook.config import (
                get_prompts_db_path,
                get_chachanotes_db_path,
                get_media_db_path,
            )
            from tldw_chatbook.Utils.Utils import get_formatted_file_size

            # Get database sizes
            db_sizes = {
                "prompts": self._get_db_size(
                    get_prompts_db_path, get_formatted_file_size
                ),
                "chachanotes": self._get_db_size(
                    get_chachanotes_db_path, get_formatted_file_size
                ),
                "media": self._get_db_size(get_media_db_path, get_formatted_file_size),
            }

            self.app.db_sizes_status = db_sizes
            # task-1714: spell the labels -- single letters decoded only by a
            # hover tooltip fail keyboard-first/low-vision users (critique r4).
            logger.info(
                "DB sizes: "
                f"Prompts: {db_sizes['prompts']} | "
                f"Chats/Notes: {db_sizes['chachanotes']} | "
                f"Media: {db_sizes['media']}"
            )

        except Exception as e:
            logger.opt(exception=True).error(f"Error computing DB sizes: {e}")

    async def update_token_count_display(self) -> None:
        """
        Update the token count in the footer when on Chat tab.

        This method checks if the current tab is the chat tab and updates
        the token count display accordingly.
        """
        # Import here to avoid circular imports
        from tldw_chatbook.Constants import TAB_CHAT

        db_status_widget = self._get_db_status_widget()
        if not db_status_widget:
            return

        # Check if we're on the chat tab
        if hasattr(self.app, "current_tab") and self.app.current_tab != TAB_CHAT:
            # Clear token count when not on chat tab
            db_status_widget.update_token_count("")
            return

        try:
            # Do the real update
            from tldw_chatbook.Event_Handlers.Chat_Events.chat_token_events import (
                update_chat_token_counter,
            )

            await update_chat_token_counter(self.app)
        except Exception as e:
            logger.opt(exception=True).error(f"Error updating token count: {e}")
            if db_status_widget:
                db_status_widget.update_token_count("Token count error")

    def start_periodic_updates(self, interval_seconds: float = 5.0) -> None:
        """
        Start periodic database size updates.

        Args:
            interval_seconds: Update interval in seconds
        """
        if self._update_timer:
            self.stop_periodic_updates()

        self._update_timer = self.app.set_interval(
            interval_seconds, lambda: self.app.call_later(self.update_db_sizes)
        )
        logger.info(
            f"Started periodic DB size updates every {interval_seconds} seconds"
        )

    def stop_periodic_updates(self) -> None:
        """Stop periodic database size updates."""
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer = None
            logger.info("Stopped periodic DB size updates")

    def _get_db_status_widget(self) -> Optional["AppFooterStatus"]:
        """
        Get the database status widget from the app.

        Resolves the currently active screen's own ``AppFooterStatus`` first
        (task-264: every ``BaseAppScreen`` mounts one), since the cached
        ``_db_size_status_widget`` -- acquired once from the app's default
        screen at startup -- is occluded as soon as any screen is pushed.
        Falls back to that cache when there's no active-screen resolver
        (e.g. lightweight test doubles) or no active-screen match.

        Returns:
            The AppFooterStatus widget if found, None otherwise
        """
        resolver = getattr(self.app, "_active_footer_status", None)
        if callable(resolver):
            widget = resolver()
            if widget is not None:
                return widget
        if hasattr(self.app, "_db_size_status_widget"):
            return self.app._db_size_status_widget
        return None

    def _get_db_size(self, path_func: callable, formatter_func: callable) -> str:
        """
        Get the formatted size of a database file.

        Args:
            path_func: Function to get the database path
            formatter_func: Function to format the file size

        Returns:
            Formatted size string or "N/A" if unavailable
        """
        try:
            db_path = path_func()
            size_str = formatter_func(db_path)
            return size_str if size_str is not None else "N/A"
        except Exception as e:
            logger.error(f"Error getting DB size: {e}")
            return "Error"
