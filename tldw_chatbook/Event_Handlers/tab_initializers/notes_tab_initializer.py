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
        self.log_initialization("Notes tab shown, loading notes...")
        
        # Import here to avoid circular imports
        from tldw_chatbook.Event_Handlers import notes_events
        
        # Load and display notes
        self.call_async_handler(
            notes_events.load_and_display_notes_handler,
            self.app
        )
        
        self.log_initialization("Notes tab initialization complete")
    
    async def on_tab_hidden(self) -> None:
        """Clean up when the notes tab is hidden."""
        self.log_initialization("Notes tab hidden, performing cleanup...")
        
        # Cancel any pending auto-save timer
        if hasattr(self.app, 'notes_auto_save_timer') and self.app.notes_auto_save_timer is not None:
            self.app.notes_auto_save_timer.stop()
            self.app.notes_auto_save_timer = None
            self.log_initialization("Cancelled auto-save timer")
        
        # Perform final auto-save if needed
        if (hasattr(self.app, 'notes_auto_save_enabled') and self.app.notes_auto_save_enabled and
            hasattr(self.app, 'notes_unsaved_changes') and self.app.notes_unsaved_changes and 
            self.app.current_selected_note_id):
            
            self.log_initialization("Performing final auto-save before leaving Notes tab")
            
            # Import here to avoid circular imports
            from tldw_chatbook.Event_Handlers.notes_events import _perform_auto_save
            
            # Schedule the auto-save as a background task
            self.app.run_worker(_perform_auto_save(self.app), name="notes_final_autosave")
        
        self.log_initialization("Notes tab cleanup complete")