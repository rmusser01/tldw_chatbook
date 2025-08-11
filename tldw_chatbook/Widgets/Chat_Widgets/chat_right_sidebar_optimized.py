# chat_right_sidebar_optimized.py
# Performance-optimized version of chat right sidebar with lazy loading

import logging
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Container
from textual.widgets import Static, Button, Input, TextArea, Checkbox
from ..lazy_widgets import LazyCollapsible

logger = logging.getLogger(__name__)

def create_chat_right_sidebar_optimized(
    id_prefix: str,
    initial_ephemeral_state: bool = True
) -> ComposeResult:
    """Create an optimized right sidebar with lazy loading.
    
    This version loads only essential elements initially and defers
    the rest until needed, significantly improving startup time.
    """
    sidebar_id = f"{id_prefix}-right-sidebar"
    
    with VerticalScroll(id=sidebar_id, classes="sidebar"):
        # Header
        yield Static("Session Details", classes="sidebar-title")
        
        # Essential conversation info - always loaded
        yield from _create_essential_info(id_prefix, initial_ephemeral_state)
        
        # Character details - lazy loaded
        yield LazyCollapsible(
            title="🎭 Character Details",
            collapsed=True,
            id=f"{id_prefix}-character-details-collapsible",
            classes="sidebar-collapsible",
            content_factory=lambda: _create_character_details(id_prefix)
        )
        
        # Prompt templates - lazy loaded
        yield LazyCollapsible(
            title="📝 Prompt Templates",
            collapsed=True,
            id=f"{id_prefix}-prompt-templates-collapsible",
            classes="sidebar-collapsible",
            content_factory=lambda: _create_prompt_templates(id_prefix)
        )
        
        # Media review - lazy loaded
        yield LazyCollapsible(
            title="🎬 Media Review",
            collapsed=True,
            id=f"{id_prefix}-media-review-collapsible",
            classes="sidebar-collapsible",
            content_factory=lambda: _create_media_review(id_prefix)
        )
        
        # Notes section - lazy loaded
        yield LazyCollapsible(
            title="📔 Notes",
            collapsed=True,
            id=f"{id_prefix}-notes-collapsible",
            classes="sidebar-collapsible",
            content_factory=lambda: _create_notes_section(id_prefix)
        )
        
        # Advanced features - lazy loaded
        yield LazyCollapsible(
            title="⚙️ Advanced",
            collapsed=True,
            id=f"{id_prefix}-advanced-collapsible",
            classes="sidebar-collapsible advanced-only",
            content_factory=lambda: _create_advanced_section(id_prefix)
        )


def _create_essential_info(id_prefix: str, initial_ephemeral: bool) -> ComposeResult:
    """Create essential conversation info that's always visible."""
    # Conversation status
    with Container(classes="conversation-status"):
        yield Static("Status:", classes="sidebar-label")
        yield Static(
            "Ephemeral" if initial_ephemeral else "Saved",
            id=f"{id_prefix}-conversation-status",
            classes="status-indicator ephemeral" if initial_ephemeral else "status-indicator saved"
        )
    
    # Conversation UUID/ID
    yield Static("Session ID", classes="sidebar-label")
    yield Input(
        value="Ephemeral Chat" if initial_ephemeral else "",
        id=f"{id_prefix}-conversation-uuid-display",
        disabled=True,
        classes="conversation-uuid"
    )
    
    # Quick actions
    with Container(classes="quick-actions"):
        yield Button(
            "💾 Save",
            id=f"{id_prefix}-save-current-chat-button",
            classes="save-button",
            disabled=not initial_ephemeral
        )
        
        yield Button(
            "🗑️ Clear",
            id=f"{id_prefix}-clear-chat-button",
            classes="clear-button"
        )


def _create_character_details(id_prefix: str) -> ComposeResult:
    """Create character details section - loaded on demand."""
    logger.debug("Creating character details (lazy loaded)")
    
    # Character name
    yield Static("Character Name", classes="sidebar-label")
    yield Input(
        placeholder="No character loaded",
        id=f"{id_prefix}-character-name-display",
        disabled=True,
        classes="character-name"
    )
    
    # Character description
    yield Static("Description", classes="sidebar-label")
    yield TextArea(
        "",
        id=f"{id_prefix}-character-description-display",
        disabled=True,
        classes="character-description"
    )
    
    # Character traits
    yield Static("Traits", classes="sidebar-label")
    yield Container(
        id=f"{id_prefix}-character-traits-container",
        classes="character-traits"
    )
    
    # Load/Clear buttons
    with Container(classes="character-actions"):
        yield Button(
            "Load Character",
            id=f"{id_prefix}-load-character-button",
            classes="load-character-button"
        )
        
        yield Button(
            "Clear Character",
            id=f"{id_prefix}-clear-active-character-button",
            classes="clear-character-button"
        )


def _create_prompt_templates(id_prefix: str) -> ComposeResult:
    """Create prompt templates section - loaded on demand."""
    logger.debug("Creating prompt templates (lazy loaded)")
    
    # Template search
    yield Static("Search Templates", classes="sidebar-label")
    yield Input(
        placeholder="Type to search...",
        id=f"{id_prefix}-prompt-search",
        classes="prompt-search"
    )
    
    # Template list container
    yield Container(
        id=f"{id_prefix}-prompt-list-container",
        classes="prompt-list-container"
    )
    
    # Template actions
    with Container(classes="template-actions"):
        yield Button(
            "Apply Template",
            id=f"{id_prefix}-apply-template-button",
            classes="apply-template-button"
        )
        
        yield Button(
            "Copy to System",
            id=f"{id_prefix}-prompt-copy-system-button",
            classes="copy-system-button"
        )
        
        yield Button(
            "Copy to User",
            id=f"{id_prefix}-prompt-copy-user-button",
            classes="copy-user-button"
        )


def _create_media_review(id_prefix: str) -> ComposeResult:
    """Create media review section - loaded on demand."""
    logger.debug("Creating media review section (lazy loaded)")
    
    # Current media item
    yield Static("Current Media", classes="sidebar-label")
    yield Container(
        Static("No media selected", classes="no-media-message"),
        id=f"{id_prefix}-media-display-container",
        classes="media-display-container"
    )
    
    # Media metadata
    yield Static("Media Info", classes="sidebar-label")
    yield Container(
        id=f"{id_prefix}-media-metadata-container",
        classes="media-metadata-container"
    )
    
    # Media actions
    with Container(classes="media-actions"):
        yield Button(
            "Load Media",
            id=f"{id_prefix}-load-media-button",
            classes="load-media-button"
        )
        
        yield Button(
            "Clear Media",
            id=f"{id_prefix}-clear-media-button",
            classes="clear-media-button"
        )
        
        yield Checkbox(
            "Include in Context",
            value=False,
            id=f"{id_prefix}-include-media-context",
            classes="include-media-checkbox"
        )


def _create_notes_section(id_prefix: str) -> ComposeResult:
    """Create notes section - loaded on demand."""
    logger.debug("Creating notes section (lazy loaded)")
    
    # Notes header with expand button
    with Container(classes="notes-header"):
        yield Static("Session Notes", classes="sidebar-label")
        yield Button(
            "⬆",
            id=f"{id_prefix}-notes-expand-button",
            classes="expand-button",
            tooltip="Expand notes area"
        )
    
    # Notes content
    yield TextArea(
        "",
        id=f"{id_prefix}-notes-content-textarea",
        classes="notes-textarea"
    )
    
    # Notes metadata
    with Container(classes="notes-metadata"):
        yield Static("Last saved: Never", id=f"{id_prefix}-notes-last-saved", classes="last-saved")
        yield Checkbox(
            "Auto-save",
            value=True,
            id=f"{id_prefix}-notes-autosave",
            classes="autosave-checkbox"
        )
    
    # Notes actions
    with Container(classes="notes-actions"):
        yield Button(
            "Save Notes",
            id=f"{id_prefix}-save-notes-button",
            classes="save-notes-button"
        )
        
        yield Button(
            "Export",
            id=f"{id_prefix}-export-notes-button",
            classes="export-notes-button"
        )


def _create_advanced_section(id_prefix: str) -> ComposeResult:
    """Create advanced section - loaded on demand."""
    logger.debug("Creating advanced section (lazy loaded)")
    
    # Message editing
    yield Static("Message Editing", classes="sidebar-label")
    yield Checkbox(
        "Enable Message Editing",
        value=False,
        id=f"{id_prefix}-enable-message-editing",
        classes="message-editing-checkbox"
    )
    
    # Export options
    yield Static("Export Options", classes="sidebar-label")
    with Container(classes="export-options"):
        yield Button(
            "Export as Markdown",
            id=f"{id_prefix}-export-markdown",
            classes="export-button"
        )
        
        yield Button(
            "Export as JSON",
            id=f"{id_prefix}-export-json",
            classes="export-button"
        )
        
        yield Button(
            "Export as Text",
            id=f"{id_prefix}-export-text",
            classes="export-button"
        )
    
    # Debug options
    yield Static("Debug Options", classes="sidebar-label")
    yield Checkbox(
        "Show Raw Messages",
        value=False,
        id=f"{id_prefix}-show-raw-messages",
        classes="debug-checkbox"
    )
    
    yield Checkbox(
        "Log API Calls",
        value=False,
        id=f"{id_prefix}-log-api-calls",
        classes="debug-checkbox"
    )
    
    # Token usage
    yield Static("Token Usage", classes="sidebar-label")
    yield Container(
        Static("Input: 0", id=f"{id_prefix}-input-tokens", classes="token-count"),
        Static("Output: 0", id=f"{id_prefix}-output-tokens", classes="token-count"),
        Static("Total: 0", id=f"{id_prefix}-total-tokens", classes="token-count"),
        id=f"{id_prefix}-token-usage-container",
        classes="token-usage-container"
    )