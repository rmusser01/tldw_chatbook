# Notes_Window.py
# Description: This file contains the UI components for the Notes Window 
#
# Imports
from typing import TYPE_CHECKING  # Added Optional
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, TextArea, Label
#
# Local Imports
from tldw_chatbook.Widgets.Note_Widgets.notes_sidebar_left import NotesSidebarLeft
from tldw_chatbook.Widgets.Note_Widgets.notes_sidebar_right import NotesSidebarRight
from tldw_chatbook.Widgets.Note_Widgets.notes_sync_widget_improved import NotesSyncWidgetImproved
# Import EmojiSelected and EmojiPickerScreen
from ..Widgets.emoji_picker import EmojiSelected
from ..Event_Handlers.Audio_Events.dictation_integration_events import InsertDictationTextEvent
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
    
    .unsaved-indicator.auto-saving {
        color: $primary;
        text-style: italic;
    }
    
    .unsaved-indicator.saved {
        color: $success;
        text-style: bold;
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
                yield Button("☰ L", id="toggle-notes-sidebar-left", classes="sidebar-toggle", tooltip="Toggle left sidebar")
                yield Label("", id="notes-unsaved-indicator", classes="unsaved-indicator")
                yield Button("Save Note", id="notes-save-button", variant="primary")
                yield Button("Preview", id="notes-preview-toggle", variant="default")
                yield Button("Sync 🔄", id="notes-sync-button", variant="default")
                yield Button("R ☰", id="toggle-notes-sidebar-right", classes="sidebar-toggle", tooltip="Toggle right sidebar")

        yield NotesSidebarRight(id="notes-sidebar-right")

    # New method to handle the result from EmojiPickerScreen
    def _handle_emoji_picker_result(self, emoji_char: str) -> None:
        """Callback for when the EmojiPickerScreen is dismissed."""
        if emoji_char: # If an emoji was selected (not cancelled)
            self.post_message(EmojiSelected(emoji_char))
    
    def on_insert_dictation_text_event(self, event: InsertDictationTextEvent) -> None:
        """Handle dictation text insertion."""
        if event.text:
            try:
                editor = self.query_one("#notes-editor-area", TextArea)
                # Get current text and cursor position
                current_text = editor.text
                cursor_location = editor.cursor_location
                row, col = cursor_location
                
                # Split text into lines
                lines = current_text.split('\n') if current_text else ['']
                
                # Ensure we have enough lines
                while len(lines) <= row:
                    lines.append('')
                
                # Insert text at cursor position
                line = lines[row]
                lines[row] = line[:col] + event.text + line[col:]
                
                # Rejoin and update
                new_text = '\n'.join(lines)
                editor.load_text(new_text)
                
                # Move cursor after inserted text
                new_col = col + len(event.text)
                editor.cursor_location = (row, new_col)
                
            except Exception as e:
                self.app.notify(f"Failed to insert voice input: {e}", severity="error")
    

    # Added on_button_pressed to handle the new button
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handles button presses within the NotesWindow."""
        if event.button.id == "notes-sync-button":
            # Push the sync widget as a modal screen
            self.app.push_screen(NotesSyncWidgetImproved(self.app_instance))
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
        notes_editor.insert(message.emoji)
        notes_editor.focus()
        message.stop()

#
# End of Notes_Window.py
#######################################################################################################################
