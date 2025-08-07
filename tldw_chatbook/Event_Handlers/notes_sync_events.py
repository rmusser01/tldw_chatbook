# notes_sync_events.py
# Description: Event handlers for note synchronization
#
# Imports
from pathlib import Path
from typing import TYPE_CHECKING, Optional
#
# Third-Party Imports
from loguru import logger
from textual.widgets import Button, Input, Select, ListView
from textual.css.query import QueryError
#
# Local Imports
from ..Notes.sync_service import NotesSyncService, SyncDirection, ConflictResolution
from ..Notes.sync_engine import SyncProgress
from tldw_chatbook.Widgets.Note_Widgets.notes_sync_widget import NotesSyncWidget, SyncProgressWidget
from ..Third_Party.textual_fspicker import SelectDirectory
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:

async def handle_sync_browse_button(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle the browse button for selecting sync folder."""
    logger.info("Sync folder browse button pressed")
    
    async def folder_selected(path: Optional[Path]) -> None:
        if path:
            try:
                folder_input = app.query_one("#sync-folder-input", Input)
                folder_input.value = str(path)
                logger.info(f"Sync folder selected: {path}")
            except QueryError as e:
                logger.error(f"Could not find sync folder input: {e}")
    
    await app.push_screen(
        SelectDirectory(str(Path.home()), title="Select Sync Folder"),
        callback=folder_selected
    )


async def handle_sync_start_button(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle the start sync button."""
    logger.info("Start sync button pressed")
    
    try:
        # Get sync widget
        sync_widget = app.query_one(NotesSyncWidget)
        
        # Get sync parameters
        folder_input = sync_widget.query_one("#sync-folder-input", Input)
        direction_select = sync_widget.query_one("#sync-direction-select", Select)
        conflict_select = sync_widget.query_one("#sync-conflict-select", Select)
        
        if not folder_input.value:
            app.notify("Please select a folder to sync", severity="warning")
            return
        
        folder = Path(folder_input.value)
        if not folder.exists():
            app.notify(f"Folder does not exist: {folder}", severity="error")
            return
        
        if not folder.is_dir():
            app.notify(f"Path is not a directory: {folder}", severity="error")
            return
        
        direction = SyncDirection(direction_select.value)
        conflict_resolution = ConflictResolution(conflict_select.value)
        
        # Initialize sync service if not already done
        if not hasattr(app, 'sync_service'):
            app.sync_service = NotesSyncService(app.notes_service, app.notes_service._get_db(app.notes_user_id))
        
        # Get progress widget
        progress_widget = sync_widget.query_one("#sync-progress-widget", SyncProgressWidget)
        
        # Define progress callback
        def progress_callback(progress: SyncProgress):
            """Update UI with sync progress."""
            app.call_from_thread(
                progress_widget.update_progress,
                progress.processed_files,
                progress.total_files,
                f"Processing: {progress.processed_files}/{progress.total_files}"
            )
        
        # Start sync in background
        progress_widget.start_sync(100)  # Initial estimate
        
        # Run sync
        session_id, progress = await app.sync_service.sync_folder(
            root_folder=folder,
            user_id=app.notes_user_id,
            direction=direction,
            conflict_resolution=conflict_resolution,
            progress_callback=progress_callback
        )
        
        # Show completion
        summary = {
            'created_notes': len(progress.created_notes),
            'updated_notes': len(progress.updated_notes),
            'created_files': len(progress.created_files),
            'updated_files': len(progress.updated_files),
            'conflicts': len(progress.conflicts),
            'errors': len(progress.errors)
        }
        
        progress_widget.complete_sync(summary)
        
        # Refresh UI
        sync_widget.refresh_notes_status()
        sync_widget.load_sync_history()
        
        # Reload notes list if any changes
        if progress.created_notes or progress.updated_notes:
            from .notes_events import load_and_display_notes_handler
            await load_and_display_notes_handler(app)
        
        # Show notification
        if progress.conflicts:
            app.notify(f"Sync completed with {len(progress.conflicts)} conflicts", severity="warning")
        elif progress.errors:
            app.notify(f"Sync completed with {len(progress.errors)} errors", severity="error")
        else:
            app.notify("Sync completed successfully", severity="information")
        
    except Exception as e:
        logger.error(f"Error during sync: {e}", exc_info=True)
        app.notify(f"Sync failed: {str(e)}", severity="error")


async def handle_sync_save_profile_button(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle saving current sync configuration as a profile."""
    logger.info("Save sync profile button pressed")
    
    try:
        sync_widget = app.query_one(NotesSyncWidget)
        
        # Get current configuration
        folder_input = sync_widget.query_one("#sync-folder-input", Input)
        direction_select = sync_widget.query_one("#sync-direction-select", Select)
        conflict_select = sync_widget.query_one("#sync-conflict-select", Select)
        
        if not folder_input.value:
            app.notify("Please select a folder before saving profile", severity="warning")
            return
        
        # TODO: Show dialog to get profile name
        profile_name = f"Profile {len(app.sync_service.profiles) + 1}"
        
        # Create profile
        profile = app.sync_service.create_profile(
            name=profile_name,
            root_folder=Path(folder_input.value),
            direction=SyncDirection(direction_select.value),
            conflict_resolution=ConflictResolution(conflict_select.value)
        )
        
        # Refresh profiles list
        sync_widget.load_sync_profiles()
        
        app.notify(f"Profile '{profile_name}' saved", severity="information")
        
    except Exception as e:
        logger.error(f"Error saving sync profile: {e}", exc_info=True)
        app.notify(f"Failed to save profile: {str(e)}", severity="error")


async def handle_sync_profile_run_button(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle running a selected sync profile."""
    logger.info("Run sync profile button pressed")
    
    try:
        sync_widget = app.query_one(NotesSyncWidget)
        profiles_list = sync_widget.query_one("#sync-profiles-list", ListView)
        
        selected_item = profiles_list.highlighted_child
        if not selected_item:
            app.notify("Please select a profile to sync", severity="warning")
            return
        
        # Extract profile name from the item (this is a simplified example)
        # In real implementation, we'd store profile data with the ListItem
        profile_name = selected_item.children[0].renderable.split(" - ")[0]
        
        # Get progress widget
        progress_widget = sync_widget.query_one("#sync-progress-widget", SyncProgressWidget)
        
        # Define progress callback
        def progress_callback(progress: SyncProgress):
            """Update UI with sync progress."""
            app.call_from_thread(
                progress_widget.update_progress,
                progress.processed_files,
                progress.total_files,
                f"Syncing profile: {profile_name}"
            )
        
        # Start sync
        progress_widget.start_sync(100)
        
        session_id, progress = await app.sync_service.sync_with_profile(
            profile_name=profile_name,
            user_id=app.notes_user_id,
            progress_callback=progress_callback
        )
        
        # Show completion
        summary = {
            'created_notes': len(progress.created_notes),
            'updated_notes': len(progress.updated_notes),
            'created_files': len(progress.created_files),
            'updated_files': len(progress.updated_files),
            'conflicts': len(progress.conflicts),
            'errors': len(progress.errors)
        }
        
        progress_widget.complete_sync(summary)
        
        # Refresh UI
        sync_widget.refresh_notes_status()
        sync_widget.load_sync_history()
        
    except Exception as e:
        logger.error(f"Error running sync profile: {e}", exc_info=True)
        app.notify(f"Profile sync failed: {str(e)}", severity="error")


async def handle_sync_profile_delete_button(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle deleting a sync profile."""
    logger.info("Delete sync profile button pressed")
    
    try:
        sync_widget = app.query_one(NotesSyncWidget)
        profiles_list = sync_widget.query_one("#sync-profiles-list", ListView)
        
        selected_item = profiles_list.highlighted_child
        if not selected_item:
            app.notify("Please select a profile to delete", severity="warning")
            return
        
        # Extract profile name
        profile_name = selected_item.children[0].renderable.split(" - ")[0]
        
        # Delete profile
        if app.sync_service.delete_profile(profile_name):
            sync_widget.load_sync_profiles()
            app.notify(f"Profile '{profile_name}' deleted", severity="information")
        else:
            app.notify(f"Failed to delete profile '{profile_name}'", severity="error")
            
    except Exception as e:
        logger.error(f"Error deleting sync profile: {e}", exc_info=True)
        app.notify(f"Failed to delete profile: {str(e)}", severity="error")


async def handle_sync_refresh_status_button(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle refreshing the sync status display."""
    logger.info("Refresh sync status button pressed")
    
    try:
        sync_widget = app.query_one(NotesSyncWidget)
        sync_widget.refresh_notes_status()
        app.notify("Sync status refreshed", severity="information", timeout=2)
        
    except Exception as e:
        logger.error(f"Error refreshing sync status: {e}", exc_info=True)
        app.notify(f"Failed to refresh status: {str(e)}", severity="error")


async def handle_sync_cancel_button(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle cancelling or closing sync progress."""
    logger.info("Sync cancel/close button pressed")
    
    try:
        sync_widget = app.query_one(NotesSyncWidget)
        progress_widget = sync_widget.query_one("#sync-progress-widget", SyncProgressWidget)
        
        if event.button.label == "Cancel":
            # Cancel ongoing sync
            # TODO: Implement sync cancellation
            app.notify("Sync cancellation requested", severity="warning")
        else:  # "Close"
            progress_widget.hide_progress()
            
    except Exception as e:
        logger.error(f"Error handling sync cancel/close: {e}", exc_info=True)


async def handle_sync_history_details_button(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle viewing sync history details."""
    logger.info("View sync history details button pressed")
    
    try:
        sync_widget = app.query_one(NotesSyncWidget)
        history_list = sync_widget.query_one("#sync-history-list", ListView)
        
        selected_item = history_list.highlighted_child
        if not selected_item:
            app.notify("Please select a sync session to view details", severity="warning")
            return
        
        # TODO: Show detailed sync session information in a modal
        app.notify("Sync session details (not yet implemented)", severity="information")
        
    except Exception as e:
        logger.error(f"Error viewing sync history details: {e}", exc_info=True)


async def handle_sync_history_conflicts_button(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle viewing conflicts from sync history."""
    logger.info("View sync conflicts button pressed")
    
    try:
        sync_widget = app.query_one(NotesSyncWidget)
        history_list = sync_widget.query_one("#sync-history-list", ListView)
        
        selected_item = history_list.highlighted_child
        if not selected_item:
            app.notify("Please select a sync session to view conflicts", severity="warning")
            return
        
        # TODO: Show conflicts in a modal with resolution options
        app.notify("Sync conflicts view (not yet implemented)", severity="information")
        
    except Exception as e:
        logger.error(f"Error viewing sync conflicts: {e}", exc_info=True)


async def handle_sync_notes_filter_changed(app: 'TldwCli', event_value: str) -> None:
    """Handle changes to the notes filter in sync status."""
    logger.debug(f"Sync notes filter changed to: '{event_value}'")
    
    try:
        sync_widget = app.query_one(NotesSyncWidget)
        # TODO: Implement filtering logic
        sync_widget.refresh_notes_status()
        
    except Exception as e:
        logger.error(f"Error filtering sync notes: {e}", exc_info=True)


async def handle_sync_status_filter_changed(app: 'TldwCli', event) -> None:
    """Handle changes to the sync status filter."""
    if hasattr(event, 'select'):
        logger.debug(f"Sync status filter changed to: {event.select.value}")
        
        try:
            sync_widget = app.query_one(NotesSyncWidget)
            # TODO: Implement status filtering logic
            sync_widget.refresh_notes_status()
            
        except Exception as e:
            logger.error(f"Error filtering by sync status: {e}", exc_info=True)


# Button handler map for sync events
SYNC_BUTTON_HANDLERS = {
    "sync-browse-button": handle_sync_browse_button,
    "sync-start-button": handle_sync_start_button,
    "sync-save-profile-button": handle_sync_save_profile_button,
    "sync-profile-run-button": handle_sync_profile_run_button,
    "sync-profile-delete-button": handle_sync_profile_delete_button,
    "sync-refresh-status-button": handle_sync_refresh_status_button,
    "sync-cancel-button": handle_sync_cancel_button,
    "sync-history-details-button": handle_sync_history_details_button,
    "sync-history-conflicts-button": handle_sync_history_conflicts_button,
}

#
# End of notes_sync_events.py
########################################################################################################################