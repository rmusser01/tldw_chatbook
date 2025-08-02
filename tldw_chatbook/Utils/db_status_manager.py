"""
Database Status Manager - Centralized management for database size status updates.

This module provides unified handling of database size calculations and
status display updates for the application footer.
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from textual.app import App
    from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


class DBStatusManager:
    """Manages database size status updates for the application."""
    
    def __init__(self, app: 'App'):
        """
        Initialize the database status manager.
        
        Args:
            app: The Textual app instance
        """
        self.app = app
        self._update_timer = None
        
    async def update_db_sizes(self) -> None:
        """
        Update the database size information in the AppFooterStatus widget.
        
        This method calculates the sizes of all tracked databases and updates
        the footer status display.
        """
        logger.debug("Attempting to update DB sizes in AppFooterStatus.")
        
        # Get the footer status widget
        db_status_widget = self._get_db_status_widget()
        if not db_status_widget:
            logger.warning("_db_size_status_widget (AppFooterStatus) is None, cannot update DB sizes.")
            return
        
        try:
            # Import here to avoid circular imports
            from tldw_chatbook.config import (
                get_prompts_db_path, 
                get_chachanotes_db_path, 
                get_media_db_path
            )
            from tldw_chatbook.Utils.Utils import get_formatted_file_size
            
            # Get database sizes
            db_sizes = {
                'prompts': self._get_db_size(get_prompts_db_path(), get_formatted_file_size),
                'chachanotes': self._get_db_size(get_chachanotes_db_path(), get_formatted_file_size),
                'media': self._get_db_size(get_media_db_path(), get_formatted_file_size)
            }
            
            # Format status string
            status_string = f"P: {db_sizes['prompts']} | C/N: {db_sizes['chachanotes']} | M: {db_sizes['media']}"
            logger.debug(f"DB size status string to display in AppFooterStatus: '{status_string}'")
            
            # Update the display
            db_status_widget.update_db_sizes_display(status_string)
            logger.info(f"Successfully updated DB sizes in AppFooterStatus: {status_string}")
            
        except Exception as e:
            logger.error(f"Error updating DB sizes in AppFooterStatus: {e}", exc_info=True)
            if db_status_widget:  # Check again in case it became None somehow
                db_status_widget.update_db_sizes_display("Error loading DB sizes")
    
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
        if hasattr(self.app, 'current_tab') and self.app.current_tab != TAB_CHAT:
            # Clear token count when not on chat tab
            db_status_widget.update_token_count("")
            return
        
        try:
            # Do the real update
            from tldw_chatbook.Event_Handlers.Chat_Events.chat_token_events import update_chat_token_counter
            await update_chat_token_counter(self.app)
        except Exception as e:
            logger.error(f"Error updating token count: {e}", exc_info=True)
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
            interval_seconds, 
            lambda: self.app.call_later(self.update_db_sizes)
        )
        logger.info(f"Started periodic DB size updates every {interval_seconds} seconds")
    
    def stop_periodic_updates(self) -> None:
        """Stop periodic database size updates."""
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer = None
            logger.info("Stopped periodic DB size updates")
    
    def _get_db_status_widget(self) -> Optional['AppFooterStatus']:
        """
        Get the database status widget from the app.
        
        Returns:
            The AppFooterStatus widget if found, None otherwise
        """
        if hasattr(self.app, '_db_size_status_widget'):
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