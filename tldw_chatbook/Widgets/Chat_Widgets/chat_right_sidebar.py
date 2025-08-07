# chat_right_sidebar.py
# Description: chat right sidebar widget
#
# Imports
#
# 3rd-Party Imports
import logging

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal
from textual.widgets import Static, Collapsible, Placeholder, Select, Input, Label, TextArea, Button, Checkbox, ListView

from tldw_chatbook.config import settings


#
# Local Imports
# (Add any necessary local imports here if needed for actual content later)
#
#######################################################################################################################
#
# Functions:

def create_chat_right_sidebar(id_prefix: str, initial_ephemeral_state: bool = True) -> ComposeResult:
    """
    Yield the widgets for the character and chat session settings sidebar.
    id_prefix is typically "chat".
    initial_ephemeral_state determines the initial state of controls related to saving.
    """
    with VerticalScroll(id="chat-right-sidebar", classes="sidebar"): # Main ID for the whole sidebar
        # Sidebar header with resize controls
        with Horizontal(classes="sidebar-header-with-resize"):
            yield Button("<<<", id=f"{id_prefix}-sidebar-expand", classes="sidebar-resize-button", tooltip="Expand sidebar")
            yield Static("Session & Character", classes="sidebar-title flex-grow")
            yield Button(">>>", id=f"{id_prefix}-sidebar-shrink", classes="sidebar-resize-button", tooltip="Collapse sidebar")

        # Section for current chat session details (title, keywords, etc.)
        with Collapsible(title="Current Chat Details", collapsed=False, id=f"{id_prefix}-chat-details-collapsible"):
            # "New Chat" Button
            yield Button(
                "New Temp Chat",
                id=f"{id_prefix}-new-temp-chat-button",  # New ID
                classes="sidebar-button",
                variant="primary"
            )
            yield Button(
                "New Chat",
                id=f"{id_prefix}-new-conversation-button",
                classes="sidebar-button"
            )
            yield Label("Conversation ID:", classes="sidebar-label", id=f"{id_prefix}-uuid-label-displayonly")
            yield Input(
                id=f"{id_prefix}-conversation-uuid-display", # Matches app.py query
                value="Temp Chat" if initial_ephemeral_state else "N/A",
                disabled=True, # Always disabled display
                classes="sidebar-input"
            )

            yield Label("Chat Title:", classes="sidebar-label", id=f"{id_prefix}-title-label-displayonly") # Keep consistent ID for query if needed elsewhere
            yield Input(
                id=f"{id_prefix}-conversation-title-input", # Matches app.py query
                placeholder="Chat title...",
                disabled=initial_ephemeral_state,
                classes="sidebar-input"
            )
            yield Label("Keywords (comma-sep):", classes="sidebar-label", id=f"{id_prefix}-keywords-label-displayonly")
            yield TextArea(
                "",
                id=f"{id_prefix}-conversation-keywords-input", # Matches app.py query
                classes="sidebar-textarea chat-keywords-textarea",
                disabled=initial_ephemeral_state
            )
            # Button to save METADATA (title/keywords) of a PERSISTENT/ALREADY EXISTING chat
            yield Button(
                "Save Details",
                id=f"{id_prefix}-save-conversation-details-button", # ID for app.py handler
                classes="sidebar-button save-details-button", # Specific class
                variant="primary", # Or "default"
                disabled=initial_ephemeral_state # Disabled if ephemeral, enabled if persistent
            )
            # Button to make an EPHEMERAL chat PERSISTENT (Save Chat to DB)
            yield Button(
                "Save Temp Chat",
                id=f"{id_prefix}-save-current-chat-button", # Matches app.py's expected ID
                classes="sidebar-button save-chat-button",
                variant="success",
                disabled=not initial_ephemeral_state # Enabled if ephemeral, disabled if already saved
            )
            
            # Button to clone/fork current chat
            yield Button(
                "🔄 Clone Current Chat",
                id=f"{id_prefix}-clone-current-chat-button",
                classes="sidebar-button clone-chat-button",
                variant="default",
                tooltip="Create a copy of the current chat to explore different paths"
            )

            # Button to convert entire conversation to note
            yield Button(
                "📋 Convert to Note",
                id=f"{id_prefix}-convert-to-note-button",
                classes="sidebar-button convert-to-note-button",
                variant="default"
            )

            # Retrieve initial value for strip_thinking_tags checkbox
            initial_strip_value = settings.get("chat_defaults", {}).get("strip_thinking_tags", True)
            yield Checkbox(
                "Strip Thinking Tags",
                value=initial_strip_value,
                id=f"{id_prefix}-strip-thinking-tags-checkbox",
                classes="sidebar-checkbox" # Add a class if specific styling is needed
            )
        # ===================================================================
        # Search Media (only for chat tab)
        # ===================================================================
        if id_prefix == "chat":
            with Collapsible(title="Search Media", collapsed=True, id=f"{id_prefix}-media-collapsible"):
                yield Label("Search Term:", classes="sidebar-label")
                yield Input(
                    id="chat-media-search-input",
                    placeholder="Search title, content...",
                    classes="sidebar-input"
                )
                yield Label("Filter by Keywords (comma-sep):", classes="sidebar-label")
                yield Input(
                    id="chat-media-keyword-filter-input",
                    placeholder="e.g., python, tutorial",
                    classes="sidebar-input"
                )
                yield Button(
                    "Search",
                    id="chat-media-search-button",
                    classes="sidebar-button"
                )
                yield ListView(id="chat-media-search-results-listview", classes="sidebar-listview")

                with Horizontal(classes="pagination-controls", id="chat-media-pagination-controls"):
                    yield Button("Prev", id="chat-media-prev-page-button", disabled=True)
                    yield Label("Page 1/1", id="chat-media-page-label")
                    yield Button("Next", id="chat-media-next-page-button", disabled=True)

                yield Static("--- Selected Media Details ---", classes="sidebar-label", id="chat-media-details-header")

                media_details_view = VerticalScroll(id="chat-media-details-view")
                media_details_view.styles.height = 35  # Set height to 35 lines to fit content without excess space
                with media_details_view:
                    with Horizontal(classes="detail-field-container"):
                        yield Label("Title:", classes="detail-label")
                        yield Button("Copy", id="chat-media-copy-title-button", classes="copy-button", disabled=True)
                    title_display_ta = TextArea("", id="chat-media-title-display", read_only=True, classes="detail-textarea")
                    title_display_ta.styles.height = 3  # Set height to 3 lines for title
                    yield title_display_ta

                    with Horizontal(classes="detail-field-container"):
                        yield Label("Content:", classes="detail-label")
                        yield Button("Copy", id="chat-media-copy-content-button", classes="copy-button", disabled=True)
                    content_display_ta = TextArea("", id="chat-media-content-display", read_only=True,
                                   classes="detail-textarea content-display")
                    content_display_ta.styles.height = 20  # Set height to 20 lines minimum
                    yield content_display_ta

                    with Horizontal(classes="detail-field-container"):
                        yield Label("Author:", classes="detail-label")
                        yield Button("Copy", id="chat-media-copy-author-button", classes="copy-button", disabled=True)
                    author_display_ta = TextArea("", id="chat-media-author-display", read_only=True, classes="detail-textarea")
                    author_display_ta.styles.height = 2  # Set height to 2 lines for author
                    yield author_display_ta

                    with Horizontal(classes="detail-field-container"):
                        yield Label("URL:", classes="detail-label")
                        yield Button("Copy", id="chat-media-copy-url-button", classes="copy-button", disabled=True)
                    url_display_ta = TextArea("", id="chat-media-url-display", read_only=True, classes="detail-textarea")
                    url_display_ta.styles.height = 2  # Set height to 2 lines for URL
                    yield url_display_ta
        # ===================================================================
        # Prompts (only for chat tab)
        # ===================================================================
        if id_prefix == "chat":
            with Collapsible(title="Prompts", collapsed=True, id=f"{id_prefix}-prompts-collapsible"):  # Added ID
                yield Label("Search Prompts:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-prompt-search-input",
                    placeholder="Enter search term...",
                    classes="sidebar-input"
                )
                # Consider adding a search button if direct input change handling is complex
                # yield Button("Search", id=f"{id_prefix}-prompt-search-button", classes="sidebar-button")

                results_list_view = ListView(
                    id=f"{id_prefix}-prompts-listview",
                    classes="sidebar-listview"
                )
                # USER-SETTING: Set height for Prompts Search Results ListView in Chat tab
                results_list_view.styles.height = 15  # Set height for ListView
                yield results_list_view

                yield Button(
                    "Load Selected Prompt",
                    id=f"{id_prefix}-prompt-load-selected-button",
                    variant="default",
                    classes="sidebar-button"
                )
                yield Label("System Prompt:", classes="sidebar-label")

                system_prompt_display = TextArea(
                    "",  # Initial content
                    id=f"{id_prefix}-prompt-system-display",
                    classes="sidebar-textarea prompt-display-textarea",
                    read_only=True
                )
                # USER-SETTING: Set height for Prompts System Prompt in Chat tab
                system_prompt_display.styles.height = 15  # Set height for TextArea
                yield system_prompt_display
                yield Button(
                    "Copy System",
                    id="chat-prompt-copy-system-button",
                    classes="sidebar-button copy-button",
                    disabled=True
                )

                yield Label("User Prompt:", classes="sidebar-label")

                user_prompt_display = TextArea(
                    "",  # Initial content
                    id=f"{id_prefix}-prompt-user-display",
                    classes="sidebar-textarea prompt-display-textarea",
                    read_only=True
                )
                # USER-SETTING: Set height for Prompts User Prompt in Chat tab
                user_prompt_display.styles.height = 15  # Set height for TextArea
                yield user_prompt_display
                yield Button(
                    "Copy User",
                    id="chat-prompt-copy-user-button",
                    classes="sidebar-button copy-button",
                    disabled=True
                )

        # ===================================================================
        # Notes (only for chat tab)
        # ===================================================================
        if id_prefix == "chat":
            with Collapsible(title="Notes", collapsed=True, id=f"{id_prefix}-notes-collapsible"):
                yield Label("Search Notes:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-notes-search-input",
                    placeholder="Search notes...",
                    classes="sidebar-input"
                )
                yield Button(
                    "Search",
                    id=f"{id_prefix}-notes-search-button",
                    classes="sidebar-button"
                )

                notes_list_view = ListView(
                    id=f"{id_prefix}-notes-listview",
                    classes="sidebar-listview"
                )
                notes_list_view.styles.height = 7
                yield notes_list_view

                yield Button(
                    "Load Note",
                    id=f"{id_prefix}-notes-load-button",
                    classes="sidebar-button"
                )
                yield Button(
                    "Create New Note",
                    id=f"{id_prefix}-notes-create-new-button",
                    variant="primary",
                    classes="sidebar-button"
                )

                yield Label("Note Title:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-notes-title-input",
                    placeholder="Note title...",
                    classes="sidebar-input"
                )

                # Expand button above note content
                yield Button(
                    "Expand Notes",
                    id=f"{id_prefix}-notes-expand-button",
                    classes="notes-expand-button sidebar-button"
                )
                
                # Note content label
                yield Label("Note Content:", classes="sidebar-label")
                
                note_content_area = TextArea(
                    id=f"{id_prefix}-notes-content-textarea",
                    classes="sidebar-textarea notes-textarea-normal"
                )
                note_content_area.styles.height = 10
                yield note_content_area

                yield Button(
                    "Save Note",
                    id=f"{id_prefix}-notes-save-button",
                    variant="success",
                    classes="sidebar-button"
                )
                
                yield Button(
                    "Copy Note",
                    id=f"{id_prefix}-notes-copy-button",
                    variant="default",
                    classes="sidebar-button"
                )

        # Placeholder for actual character details (if a specific character is active beyond default)
        # This part would be more relevant if the chat tab directly supported switching active characters
        # for the ongoing conversation, rather than just for filtering searches.
        # For now, keeping it minimal.
        with Collapsible(title="Active Character Info", collapsed=True, id=f"{id_prefix}-active-character-info-collapsible"): # Changed to collapsed=True to start collapsed
            if id_prefix == "chat":
                yield Input(
                    id="chat-character-search-input",
                    placeholder="Search all characters..."
                )
                character_search_results_list = ListView(  # Assign to variable
                    id="chat-character-search-results-list"
                )
                character_search_results_list.styles.height = 7
                yield character_search_results_list
                yield Button(
                    "Load Character",
                    id="chat-load-character-button"
                )
                yield Button(
                    "Clear Active Character",
                    id="chat-clear-active-character-button", # New ID
                    classes="sidebar-button",
                    variant="warning" # Optional: different styling
                )
                yield Label("Character Name:", classes="sidebar-label")
                yield Input(
                    id="chat-character-name-edit",
                    placeholder="Name"
                )
                yield Label("Description:", classes="sidebar-label")
                description_edit_ta = TextArea(id="chat-character-description-edit")
                description_edit_ta.styles.height = 30
                yield description_edit_ta

                yield Label("Personality:", classes="sidebar-label")
                personality_edit_ta = TextArea(id="chat-character-personality-edit")
                personality_edit_ta.styles.height = 30
                yield personality_edit_ta

                yield Label("Scenario:", classes="sidebar-label")
                scenario_edit_ta = TextArea(id="chat-character-scenario-edit")
                scenario_edit_ta.styles.height = 30
                yield scenario_edit_ta

                yield Label("System Prompt:", classes="sidebar-label")
                system_prompt_edit_ta = TextArea(id="chat-character-system-prompt-edit")
                system_prompt_edit_ta.styles.height = 30
                yield system_prompt_edit_ta

                yield Label("First Message:", classes="sidebar-label")
                first_message_edit_ta = TextArea(id="chat-character-first-message-edit")
                first_message_edit_ta.styles.height = 30
                yield first_message_edit_ta
            # yield Placeholder("Display Active Character Name") # Removed placeholder
            # Could add a select here to change the character for the *current* chat,
            # which would then influence the AI's persona for subsequent messages.
            # This is a more advanced feature than just for filtering searches.

        # ===================================================================
        # Chat Dictionaries (only for chat tab)
        # ===================================================================
        if id_prefix == "chat":
            with Collapsible(title="Chat Dictionaries", collapsed=True, id=f"{id_prefix}-dictionaries-collapsible"):
                # Search for available dictionaries
                yield Label("Search Dictionaries:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-dictionary-search-input",
                    placeholder="Search dictionaries...",
                    classes="sidebar-input"
                )
                
                # List of available dictionaries
                yield Label("Available Dictionaries:", classes="sidebar-label")
                dictionary_available_list = ListView(
                    id=f"{id_prefix}-dictionary-available-listview",
                    classes="sidebar-listview"
                )
                dictionary_available_list.styles.height = 5
                yield dictionary_available_list
                
                # Add button for dictionaries
                yield Button(
                    "Add to Chat",
                    id=f"{id_prefix}-dictionary-add-button",
                    classes="sidebar-button",
                    variant="primary",
                    disabled=True
                )
                
                # Currently associated dictionaries
                yield Label("Active Dictionaries:", classes="sidebar-label")
                dictionary_active_list = ListView(
                    id=f"{id_prefix}-dictionary-active-listview",
                    classes="sidebar-listview"
                )
                dictionary_active_list.styles.height = 5
                yield dictionary_active_list
                
                # Remove button for active dictionaries
                yield Button(
                    "Remove from Chat",
                    id=f"{id_prefix}-dictionary-remove-button",
                    classes="sidebar-button",
                    variant="warning",
                    disabled=True
                )
                
                # Quick enable/disable for dictionary processing
                yield Checkbox(
                    "Enable Dictionary Processing",
                    value=True,
                    id=f"{id_prefix}-dictionary-enable-checkbox",
                    classes="sidebar-checkbox"
                )
                
                # Selected dictionary details
                yield Label("Selected Dictionary Details:", classes="sidebar-label")
                dictionary_details = TextArea(
                    "",
                    id=f"{id_prefix}-dictionary-details-display",
                    classes="sidebar-textarea",
                    read_only=True
                )
                dictionary_details.styles.height = 8
                yield dictionary_details

        # ===================================================================
        # World Books (only for chat tab)
        # ===================================================================
        if id_prefix == "chat":
            with Collapsible(title="World Books", collapsed=True, id=f"{id_prefix}-worldbooks-collapsible"):
                # Search for available world books
                yield Label("Search World Books:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-worldbook-search-input",
                    placeholder="Search world books...",
                    classes="sidebar-input"
                )
                
                # List of available world books
                yield Label("Available World Books:", classes="sidebar-label")
                worldbook_available_list = ListView(
                    id=f"{id_prefix}-worldbook-available-listview",
                    classes="sidebar-listview"
                )
                worldbook_available_list.styles.height = 5
                yield worldbook_available_list
                
                # Association controls
                with Horizontal(classes="worldbook-association-controls"):
                    yield Button(
                        "Add to Chat",
                        id=f"{id_prefix}-worldbook-add-button",
                        classes="sidebar-button",
                        variant="primary",
                        disabled=True
                    )
                    yield Select(
                        [(f"Priority {i}", str(i)) for i in range(11)],
                        value="5",
                        id=f"{id_prefix}-worldbook-priority-select",
                        classes="worldbook-priority-select"
                    )
                
                # Currently associated world books
                yield Label("Active World Books:", classes="sidebar-label")
                worldbook_active_list = ListView(
                    id=f"{id_prefix}-worldbook-active-listview",
                    classes="sidebar-listview"
                )
                worldbook_active_list.styles.height = 5
                yield worldbook_active_list
                
                # Remove button for active world books
                yield Button(
                    "Remove from Chat",
                    id=f"{id_prefix}-worldbook-remove-button",
                    classes="sidebar-button",
                    variant="warning",
                    disabled=True
                )
                
                # Quick enable/disable for all world books
                yield Checkbox(
                    "Enable World Info Processing",
                    value=True,
                    id=f"{id_prefix}-worldbook-enable-checkbox",
                    classes="sidebar-checkbox"
                )
                
                # Selected world book details
                yield Label("Selected World Book Details:", classes="sidebar-label")
                worldbook_details = TextArea(
                    "",
                    id=f"{id_prefix}-worldbook-details-display",
                    classes="sidebar-textarea",
                    read_only=True
                )
                worldbook_details.styles.height = 8
                yield worldbook_details

        with Collapsible(title="Other Character Tools", collapsed=True):
            yield Placeholder("Tool 1")
            yield Placeholder("Tool 2")

        logging.debug(f"Character sidebar (id='chat-right-sidebar', prefix='{id_prefix}') created with ephemeral state: {initial_ephemeral_state}")

#
# End of chat_right_sidebar.py
#######################################################################################################################
