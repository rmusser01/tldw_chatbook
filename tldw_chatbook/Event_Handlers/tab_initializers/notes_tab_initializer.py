"""
Notes Tab Initializer - Handles initialization for the notes tab.

This module manages the initialization logic when the notes tab is shown
or hidden, including loading notes and managing auto-save functionality.
"""

from typing import TYPE_CHECKING

from .base_initializer import BaseTabInitializer

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class NotesTabInitializer(BaseTabInitializer):
    """Handles notes tab initialization and cleanup."""
    
    def get_tab_id(self) -> str:
        """Return the notes tab ID."""
        from tldw_chatbook.Constants import TAB_NOTES
        return TAB_NOTES
    
    async def on_tab_shown(self) -> None:
        """Initialize the notes tab when shown."""
        self.log_initialization("Notes tab shown, refreshing active notes scope...")

        from tldw_chatbook.UI.Screens.notes_screen import NotesScreen

        active_screen = getattr(self.app, "screen", None)
        if isinstance(active_screen, NotesScreen):
            await active_screen.refresh_current_scope()
        else:
            self.log_initialization("Active screen is not NotesScreen; skipping scope refresh.")

        self.log_initialization("Notes tab initialization complete")
    
    async def on_tab_hidden(self) -> None:
        """Clean up when the notes tab is hidden."""
        self.log_initialization("Notes tab hidden, performing cleanup...")

        from tldw_chatbook.UI.Screens.notes_screen import NotesScreen

        active_screen = getattr(self.app, "screen", None)
        if isinstance(active_screen, NotesScreen):
            await active_screen.finalize_for_hide()
        else:
            self.log_initialization("Active screen is not NotesScreen; skipping screen finalization.")

        self.log_initialization("Notes tab cleanup complete")
