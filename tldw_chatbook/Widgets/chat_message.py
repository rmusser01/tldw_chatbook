# tldw_app/widgets/chat_message.py
# Description: This file contains the ChatMessage widget for tldw_cli
#
# Imports
#
# 3rd-party Libraries
import logging
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, Label # Added Label
from textual.reactive import reactive
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

class ChatMessage(Widget):
    """A widget to display a single chat message with action buttons."""

    class Action(Message):
        """Posted when a button on the message is pressed."""
        def __init__(self, message_widget: "ChatMessage", button: Button) -> None:
            super().__init__()
            self.message_widget = message_widget
            self.button = button

    DEFAULT_CSS = """
    ChatMessage {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }
    ChatMessage > Vertical {
        border: round $surface;
        background: $panel;
        padding: 0 1;
        width: 100%;
        height: auto;
    }
    ChatMessage.-user > Vertical {
        background: $boost; /* Different background for user */
        border: round $accent;
    }
    .message-header {
        width: 100%;
        padding: 0 1;
        background: $surface-darken-1;
        text-style: bold;
    }
    .message-text {
        padding: 1;
        width: 100%;
        height: auto;
    }
    .message-actions {
        height: auto;
        width: 100%;
        padding: 0 1;
        /* margin-top: 1; */ /* Removed top margin */      
        border-top: solid $surface-lighten-1;
        align: right middle; /* Align buttons to the right */
        display: block; /* Default display state */
    }
    .message-actions Button {
        min-width: 8;
        height: 1;
        margin: 0 0 0 1; /* Space between buttons */
        border: none;
        background: $surface-lighten-2;
        color: $text-muted;
    }
    .message-actions Button:hover {
        background: $surface;
        color: $text;
    }
    /* Specific hover style for delete */
    .message-actions .delete-button:hover {
        background: $error; /* Use the theme's error color (usually red) */
        color: white; /* Adjust text color for contrast if needed */
    }
    /* Initially hide AI actions until generation is complete */
    ChatMessage.-ai .message-actions.-generating {
        display: none;
    }
    """

    # Store the raw text content
    message_text = reactive("", repaint=True)
    role = reactive("User", repaint=True)
    # Use an internal reactive to manage generation status and trigger UI updates
    _generation_complete_internal = reactive(True)

    # -- Internal state for message metadata ---
    message_id_internal: reactive[Optional[str]] = reactive(None)
    message_version_internal: reactive[Optional[int]] = reactive(None)
    # Store timestamp if provided, e.g. when loading from DB
    timestamp: reactive[Optional[str]] = reactive(None) # Store as ISO string
    # Store image data if message has an image
    image_data: reactive[Optional[bytes]] = reactive(None)
    image_mime_type: reactive[Optional[str]] = reactive(None)
    # Store feedback (thumbs up/down)
    feedback: reactive[Optional[str]] = reactive(None)

    def __init__(self,
                 message: str,
                 role: str,
                 generation_complete: bool = True,
                 message_id: Optional[str] = None,
                 message_version: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 image_data: Optional[bytes] = None,
                 image_mime_type: Optional[str] = None,
                 feedback: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.message_text = message
        self.role = role
        self._generation_complete_internal = generation_complete

        self.message_id_internal = message_id
        self.message_version_internal = message_version
        self.timestamp = timestamp
        self.image_data = image_data
        self.image_mime_type = image_mime_type
        self.feedback = feedback

        #self.add_class(f"-{role.lower()}") # Add role-specific class
        # For CSS styling, we use a generic class based on whether it's the user or not.
        # The actual self.role (e.g., "Default Assistant") is used for display in the header.
        if role.lower() == "user":
            self.add_class("-user")
        else: # Any role other than "user" (e.g., "AI", "Default Assistant", "Character Name") gets the -ai style
            self.add_class("-ai")

    @property
    def generation_complete(self) -> bool:
        """Public property to access the generation status."""
        return self._generation_complete_internal

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"{self.role}", classes="message-header")
            yield Static(self.message_text, classes="message-text", markup=False)
            actions_class = "message-actions"
            # Logic for the '-generating' class on the actions container
            # This should only apply if it's an AI message AND generation is not complete
            if self.has_class("-ai") and not self.generation_complete:
                actions_class += " -generating"

            with Horizontal(classes=actions_class) as actions_bar:
                actions_bar.id = f"actions-bar-{self.id or self.message_id_internal or 'new'}"
                # Common buttons
                yield Button("Edit", classes="action-button edit-button")
                yield Button("📋", classes="action-button copy-button", id="copy") # Emoji for copy
                yield Button("🔊", classes="action-button speak-button", id="speak") # Emoji for speak

                # AI-specific buttons
                if self.has_class("-ai"):
                    # Display feedback state on thumb buttons
                    thumb_up_label = "👍✓" if self.feedback == "1;" else "👍"
                    thumb_down_label = "👎✓" if self.feedback == "2;" else "👎"
                    yield Button(thumb_up_label, classes="action-button thumb-up-button", id="thumb-up")
                    yield Button(thumb_down_label, classes="action-button thumb-down-button", id="thumb-down")
                    yield Button("🔄", classes="action-button regenerate-button", id="regenerate") # Emoji for regenerate
                    yield Button("↪️", id="continue-response-button", classes="action-button continue-button")

                # Add delete button for all messages at very end
                yield Button("🗑️", classes="action-button delete-button")  # Emoji for delete ; Label: Delete, Class: delete-button

    def watch__generation_complete_internal(self, complete: bool) -> None:
        """
        Watcher for the internal generation status.
        Updates the actions bar visibility and the continue button visibility for AI messages.
        """
        logging.info(f"watch__generation_complete_internal called with complete={complete} for ChatMessage (ID: {self.id}, Role: {self.role})")
        if self.has_class("-ai"):
            logging.info(f"ChatMessage has -ai class")
            try:
                actions_container = self.query_one(".message-actions")
                logging.info(f"Found actions_container: {actions_container}")
                if complete:
                    logging.info(f"Setting actions_container to visible")
                    actions_container.remove_class("-generating")
                    actions_container.styles.display = "block"
                    # Force update the DOM to ensure changes are applied
                    self.refresh()
                else:
                    logging.info(f"Setting actions_container to hidden")
                    actions_container.add_class("-generating")

                # Separately handle the continue button in its own try...except block
                # This prevents an error here from stopping the whole function.
                try:
                    continue_button = self.query_one("#continue-response-button", Button)
                    continue_button.display = complete
                except QueryError:
                    # It's okay if the continue button doesn't exist, as it's commented out.
                    logging.debug("Continue button not found in ChatMessage, skipping visibility toggle.")

            except QueryError as qe:
                # This might happen if the query runs before the widget is fully composed or if it's being removed.
                logging.warning(f"ChatMessage (ID: {self.id}, Role: {self.role}): QueryError in watch__generation_complete_internal: {qe}. Widget might not be fully ready or is not an AI message with these components.")
            except Exception as e:
                logging.error(f"Error in watch__generation_complete_internal for ChatMessage (ID: {self.id}): {e}", exc_info=True)
        else: # Not an AI message
            logging.info(f"ChatMessage does not have -ai class")
            try: # Ensure continue button is hidden for non-AI messages if it somehow got queried
                continue_button = self.query_one("#continue-response-button", Button)
                continue_button.display = False
            except QueryError:
                pass # Expected for non-AI messages as the button isn't composed.

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when a button inside this message is pressed."""
        # Post our custom Action message so the app can handle it.
        # The message carries the button and this widget instance.
        self.post_message(self.Action(self, event.button))
        # Stop the event from bubbling up to the app's on_button_pressed.
        event.stop()

    def mark_generation_complete(self):
        """
        Marks the AI message generation as complete.
        This will trigger the watcher for _generation_complete_internal to update UI.
        """
        logging.info(f"mark_generation_complete called for ChatMessage (ID: {self.id}, Role: {self.role})")
        if self.has_class("-ai"):
            logging.info(f"Setting _generation_complete_internal to True")
            self._generation_complete_internal = True
            # Force a refresh to ensure UI updates
            self.refresh()

    def on_mount(self) -> None:
        """Ensure initial state of continue button and actions bar is correct after mounting."""
        # Trigger the watcher logic based on the initial state.
        self.watch__generation_complete_internal(self._generation_complete_internal)

    def update_message_chunk(self, chunk: str):
        """Appends a chunk of text to an AI message during streaming."""
        # This method is called by handle_streaming_chunk.
        # The _generation_complete_internal should be False during streaming.
        if self.has_class("-ai") and not self._generation_complete_internal:
            # The static_text_widget.update is handled in handle_streaming_chunk
            # This method primarily updates the internal message_text.
            self.message_text += chunk
        # If called at other times, ensure it doesn't break if static_text_widget not found.
        # For streaming, handle_streaming_chunk updates the Static widget directly.

#
#
#######################################################################################################################
