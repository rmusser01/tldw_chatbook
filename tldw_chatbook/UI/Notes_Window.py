# Notes_Window.py
# Description: This file contains the UI components for the Notes Window 
#
# Imports
from typing import TYPE_CHECKING, Optional # Added Optional
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, TextArea, Static, Label
#
# Local Imports
from ..Widgets.notes_sidebar_left import NotesSidebarLeft
from ..Widgets.notes_sidebar_right import NotesSidebarRight
from ..Widgets.notes_sync_widget import NotesSyncWidget
# Import EmojiSelected and EmojiPickerScreen
from ..Widgets.emoji_picker import EmojiSelected, EmojiPickerScreen
# from ..Constants import TAB_NOTES # Not strictly needed if IDs are hardcoded here
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class NotesWindow(Container):
    """
    Container for the Notes Tab's UI.
    """
    
    DEFAULT_CSS = """
    NotesWindow {
        height: 100%;
    }
    
    #notes-controls-area {
        height: 3;
        align: center middle;
        overflow-x: auto;
    }
    
    .unsaved-indicator {
        color: $warning;
        text-style: bold;
        margin: 0 1;
    }
    
    .unsaved-indicator.has-unsaved {
        color: $error;
    }
    
    .word-count {
        color: $text-muted;
        margin: 0 1;
    }
    
    #notes-preview-toggle {
        margin: 0 1;
    }
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance # Not strictly used in compose below, but good practice if needed later

    def compose(self) -> ComposeResult:
        yield NotesSidebarLeft(id="notes-sidebar-left")

        with Container(id="notes-main-content"):
            yield TextArea(id="notes-editor-area", classes="notes-editor")
            with Horizontal(id="notes-controls-area"):
                yield Button("☰ L", id="toggle-notes-sidebar-left", classes="sidebar-toggle")
                yield Static()  # Spacer
                yield Button("😊", id="open-emoji-picker-button", tooltip="Insert Emoji")
                yield Static() # Spacer
                yield Label("", id="notes-unsaved-indicator", classes="unsaved-indicator")
                yield Button("Save Note", id="notes-save-button", variant="primary")
                yield Button("Sync 🔄", id="notes-sync-button", variant="default")
                yield Button("Preview", id="notes-preview-toggle", variant="default")
                yield Label("Words: 0", id="notes-word-count", classes="word-count")
                yield Static()  # Spacer
                yield Button("R ☰", id="toggle-notes-sidebar-right", classes="sidebar-toggle")

        yield NotesSidebarRight(id="notes-sidebar-right")

    # New method to handle the result from EmojiPickerScreen
    def _handle_emoji_picker_result(self, emoji_char: str) -> None:
        """Callback for when the EmojiPickerScreen is dismissed."""
        if emoji_char: # If an emoji was selected (not cancelled)
            self.post_message(EmojiSelected(emoji_char))

    # Added on_button_pressed to handle the new button
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handles button presses within the NotesWindow."""
        if event.button.id == "open-emoji-picker-button":
            # Use a unique ID for the picker if needed, e.g., "notes-modal-emoji-picker"
            self.app.push_screen(EmojiPickerScreen(id="notes_emoji_modal_picker"), self._handle_emoji_picker_result)
            event.stop()
        elif event.button.id == "notes-sync-button":
            # Push the sync widget as a modal screen
            self.app.push_screen(NotesSyncWidget(self.app_instance))
            event.stop()
        # Add other button ID checks here if necessary for this window's specific buttons.
        # For example, the sidebar toggles are often handled at the app level via actions,
        # but if they were to be handled here, it would look like:
        # elif event.button.id == "toggle-notes-sidebar-left":
        #     self.app.action_toggle_notes_sidebar_left() # Assuming such an action exists
        #     event.stop()
        # elif event.button.id == "toggle-notes-sidebar-right":
        #     self.app.action_toggle_notes_sidebar_right()
        #     event.stop()
        # elif event.button.id == "notes-save-button":
        #     # This is likely handled by an event in notes_events.py or app.py,
        #     # but if handled here, it would be:
        #     # self.app.action_save_current_note() # Assuming such an action
        #     pass # Let other handlers catch it if not stopped


    def on_emoji_picker_emoji_selected(self, message: EmojiSelected) -> None:
        """Handles the EmojiSelected message posted after an emoji is picked."""
        # The message is now posted by _handle_emoji_picker_result,
        # so we don't need to check message.sender.id as strictly if this NotesWindow
        # is the one initiating and handling the modal emoji picker.
        # If multiple sources could send EmojiSelected to NotesWindow,
        # the picker_id attribute on EmojiSelected could be used.
        notes_editor = self.query_one("#notes-editor-area", TextArea)
        notes_editor.insert_text_at_cursor(message.emoji)
        notes_editor.focus()
        message.stop()

#
# End of Notes_Window.py
#######################################################################################################################
