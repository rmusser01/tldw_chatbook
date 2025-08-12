# unified_chat_sidebar.py
# Description: Unified single sidebar for chat interface with tabbed organization
#
# Imports
from typing import TYPE_CHECKING, Optional, Dict, Any, Set
import logging
#
# 3rd-Party Imports
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Static, TabbedContent, TabPane, Button, Input, Label, 
    TextArea, Checkbox, Select, ListView, ListItem
)
from textual.reactive import reactive
from textual.message import Message
from textual import on
#
# Local Imports
from ...config import get_cli_setting, save_setting_to_cli_config, get_cli_providers_and_models
from ...Utils.Emoji_Handling import get_char

if TYPE_CHECKING:
    from ...app import TldwCli

#
#######################################################################################################################
#
# State Management
#

class ChatSidebarState:
    """Centralized state management for the unified sidebar."""
    
    def __init__(self):
        self.active_tab: str = "session"
        self.search_query: str = ""
        self.search_filter: str = "all"
        self.collapsed_sections: Set[str] = set()
        self.sidebar_width: int = 30  # percentage
        self.sidebar_position: str = "left"  # Settings traditionally on left
        self.advanced_mode: bool = False
        self.current_page: Dict[str, int] = {}  # pagination state per content type
        
    def save_preferences(self):
        """Persist user preferences to config."""
        save_setting_to_cli_config("chat_sidebar", "width", self.sidebar_width)
        save_setting_to_cli_config("chat_sidebar", "position", self.sidebar_position)
        save_setting_to_cli_config("chat_sidebar", "active_tab", self.active_tab)
        save_setting_to_cli_config("chat_sidebar", "advanced_mode", self.advanced_mode)
    
    def load_preferences(self):
        """Load user preferences from config."""
        self.sidebar_width = get_cli_setting("chat_sidebar", "width", 30)
        self.sidebar_position = get_cli_setting("chat_sidebar", "position", "right")
        self.active_tab = get_cli_setting("chat_sidebar", "active_tab", "session")
        self.advanced_mode = get_cli_setting("chat_sidebar", "advanced_mode", False)

#
#######################################################################################################################
#
# Compound Widgets
#

class CompactField(Horizontal):
    """Space-efficient form field combining label and input."""
    
    def __init__(self, label: str, field_id: str, value: str = "", 
                 input_type: str = "input", placeholder: str = "", **kwargs):
        super().__init__(**kwargs)
        self.label_text = label
        self.field_id = field_id
        self.value = value
        self.input_type = input_type
        self.placeholder = placeholder
    
    def compose(self) -> ComposeResult:
        yield Label(self.label_text, classes="compact-label")
        if self.input_type == "input":
            yield Input(id=self.field_id, value=self.value, 
                       placeholder=self.placeholder if self.placeholder else "", 
                       classes="compact-input")
        elif self.input_type == "textarea":
            yield TextArea(self.value, id=self.field_id, classes="compact-textarea")


class SearchableList(Container):
    """Reusable search interface for any content type."""
    
    def __init__(self, content_type: str, placeholder: str = "Search...", **kwargs):
        super().__init__(**kwargs)
        self.content_type = content_type
        self.placeholder = placeholder
        self.current_page = 1
        self.total_pages = 1
        self.results_per_page = 10
    
    def compose(self) -> ComposeResult:
        with Container(classes="searchable-list-container"):
            # Search input with integrated button
            with Horizontal(classes="search-bar"):
                yield Input(
                    placeholder=self.placeholder,
                    id=f"{self.content_type}-search-input",
                    classes="search-input"
                )
                yield Button("🔍", id=f"{self.content_type}-search-btn", classes="search-button")
            
            # Results list
            yield ListView(
                id=f"{self.content_type}-results",
                classes="search-results-list"
            )
            
            # Pagination controls
            with Horizontal(classes="pagination-controls"):
                yield Button("◀", id=f"{self.content_type}-prev", classes="page-btn", disabled=True)
                yield Label(f"Page {self.current_page}/{self.total_pages}", 
                          id=f"{self.content_type}-page-label", 
                          classes="page-label")
                yield Button("▶", id=f"{self.content_type}-next", classes="page-btn", disabled=True)


class SmartCollapsible(Container):
    """Collapsible section with auto-collapse and state memory."""
    
    collapsed = reactive(False)
    
    def __init__(self, title: str, section_id: str, auto_collapse: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.section_id = section_id
        self.auto_collapse = auto_collapse
        self.has_unsaved_changes = False
    
    def compose(self) -> ComposeResult:
        with Container(classes="smart-collapsible"):
            # Header with toggle
            with Horizontal(classes="collapsible-header"):
                yield Button(
                    f"{'▶' if self.collapsed else '▼'} {self.title}",
                    id=f"{self.section_id}-toggle",
                    classes="collapsible-toggle"
                )
            
            # Content container
            if not self.collapsed:
                with Container(id=f"{self.section_id}-content", classes="collapsible-content"):
                    yield from self.compose_content()
    
    def compose_content(self) -> ComposeResult:
        """Override in subclasses to provide content."""
        yield Static("Override compose_content in subclass")
    
    def toggle(self):
        """Toggle collapsed state."""
        self.collapsed = not self.collapsed
        self.refresh()

#
#######################################################################################################################
#
# Tab Content Components
#

class SessionTab(Container):
    """Session management tab content."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
    
    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="tab-content session-tab"):
            # Current chat info
            yield Static("Current Chat", classes="section-title")
            
            # Chat ID (read-only) - Using old ID for compatibility
            yield Label("Chat ID:", classes="sidebar-label")
            yield Input(
                id="chat-conversation-uuid-display",
                value="Temp Chat",
                disabled=True,
                classes="sidebar-input"
            )
            
            # Editable fields - Using old IDs for compatibility
            yield Label("Title:", classes="sidebar-label")
            yield Input(
                id="chat-conversation-title-input",
                placeholder="Enter chat title...",
                classes="sidebar-input"
            )
            
            yield Label("Keywords (comma-sep):", classes="sidebar-label")
            yield TextArea(
                "",
                id="chat-conversation-keywords-input",
                classes="sidebar-textarea chat-keywords-textarea"
            )
            
            # Action buttons
            yield Static("Actions", classes="section-title")
            with Horizontal(classes="button-group"):
                yield Button("New Temp Chat", id="chat-new-temp-chat-button", variant="primary")
                yield Button("New Chat", id="chat-new-conversation-button", variant="success")
            
            with Horizontal(classes="button-group"):
                yield Button("Save Details", id="chat-save-conversation-details-button", variant="primary")
                yield Button("Save Temp Chat", id="chat-save-current-chat-button", variant="success")
            
            with Horizontal(classes="button-group"):
                yield Button("🔄 Clone Chat", id="chat-clone-current-chat-button", variant="default")
                yield Button("📋 Convert to Note", id="chat-convert-to-note-button", variant="default")
            
            # Options
            yield Static("Options", classes="section-title")
            yield Checkbox("Strip Thinking Tags", id="chat-strip-thinking-tags-checkbox", value=True)


class SettingsTab(Container):
    """LLM settings tab with progressive disclosure."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.advanced_mode = reactive(False)
    
    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="tab-content settings-tab"):
            # Quick Settings (always visible)
            yield Static("Quick Settings", classes="section-title")
            
            # Provider and model selection
            providers_models = get_cli_providers_and_models()
            providers = list(providers_models.keys())
            
            yield Label("Provider", classes="field-label")
            provider_options = [(p, p) for p in providers]
            yield Select(
                options=provider_options,
                id="settings-provider",
                value=providers[0] if providers else None,
                classes="settings-select"
            )
            
            yield Label("Model", classes="field-label")
            yield Select(
                options=[],
                id="settings-model",
                classes="settings-select"
            )
            
            yield CompactField("Temperature:", "settings-temperature", value="0.7")
            
            # Advanced mode toggle
            yield Checkbox("Show Advanced Settings", id="settings-advanced-toggle", value=False)
            
            # Advanced settings (conditionally visible)
            with Container(id="advanced-settings", classes="hidden"):
                yield Static("Advanced Settings", classes="section-title")
                
                yield Label("System Prompt", classes="field-label")
                yield TextArea(
                    "",
                    id="settings-system-prompt",
                    classes="settings-textarea"
                )
                
                yield CompactField("Top-p:", "settings-top-p", value="0.95")
                yield CompactField("Top-k:", "settings-top-k", value="50")
                yield CompactField("Min-p:", "settings-min-p", value="0.05")
            
            # RAG Settings in collapsible
            yield Static("", classes="section-spacer")
            yield Button("▶ RAG Settings", id="rag-toggle", classes="collapsible-toggle")
            
            with Container(id="rag-settings", classes="hidden"):
                yield Checkbox("Enable RAG", id="settings-rag-enabled", value=False)
                
                yield Label("Pipeline", classes="field-label")
                pipeline_options = [
                    ("Manual Configuration", "none"),
                    ("Speed Optimized", "speed_optimized_v2"),
                    ("High Accuracy", "high_accuracy"),
                    ("Hybrid Search", "hybrid"),
                ]
                yield Select(
                    options=pipeline_options,
                    id="settings-rag-pipeline",
                    value="none",
                    classes="settings-select"
                )
                
                yield Button("Configure RAG", id="settings-rag-configure", variant="default")


class ContentTab(Container):
    """Unified content search and management tab."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.current_filter = "all"
    
    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="tab-content content-tab"):
            # Media Search Section (preserving old functionality)
            yield Static("Search Media", classes="section-title")
            
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
            
            # Results List
            yield ListView(id="chat-media-search-results-listview", classes="sidebar-listview")
            
            # Pagination controls
            with Horizontal(classes="pagination-controls", id="chat-media-pagination-controls"):
                yield Button("Prev", id="chat-media-prev-page-button", disabled=True)
                yield Label("Page 1/1", id="chat-media-page-label")
                yield Button("Next", id="chat-media-next-page-button", disabled=True)
            
            # Selected Media Details
            yield Static("--- Selected Media Details ---", classes="sidebar-label")
            
            # Media detail fields with copy buttons
            with Horizontal(classes="detail-field-container"):
                yield Label("Title:", classes="detail-label")
                yield TextArea(
                    "",
                    id="chat-media-title-display",
                    classes="detail-textarea",
                    disabled=True
                )
                yield Button("Copy", id="chat-media-copy-title-button", disabled=True, classes="copy-button")
            
            with Horizontal(classes="detail-field-container"):
                yield Label("Author:", classes="detail-label")
                yield TextArea(
                    "",
                    id="chat-media-author-display",
                    classes="detail-textarea",
                    disabled=True
                )
                yield Button("Copy", id="chat-media-copy-author-button", disabled=True, classes="copy-button")
            
            with Horizontal(classes="detail-field-container"):
                yield Label("URL:", classes="detail-label")
                yield TextArea(
                    "",
                    id="chat-media-url-display",
                    classes="detail-textarea",
                    disabled=True
                )
                yield Button("Copy", id="chat-media-copy-url-button", disabled=True, classes="copy-button")
            
            yield Label("Content:", classes="detail-label")
            yield TextArea(
                "",
                id="chat-media-content-display",
                classes="content-preview",
                disabled=True
            )
            yield Button("Copy Content", id="chat-media-copy-content-button", disabled=True, classes="copy-button")

#
#######################################################################################################################
#
# Main Unified Sidebar Widget
#

class UnifiedChatSidebar(Container):
    """Single unified sidebar for all chat functionality."""
    
    BINDINGS = [
        ("alt+1", "switch_tab('session')", "Session Tab"),
        ("alt+2", "switch_tab('settings')", "Settings Tab"),
        ("alt+3", "switch_tab('content')", "Content Tab"),
        ("ctrl+\\", "toggle_sidebar", "Toggle Sidebar"),
    ]
    
    DEFAULT_CSS = """
    UnifiedChatSidebar {
        dock: left;
        width: 30%;
        min-width: 250;
        max-width: 50%;
        background: $surface;
        border-right: solid $primary-darken-2;
        padding: 1;
    }
    
    UnifiedChatSidebar.collapsed {
        width: 0;
        min-width: 0;
        padding: 0;
        border: none;
        display: none;
    }
    
    .sidebar-header {
        height: 3;
        background: $boost;
        padding: 1;
        margin-bottom: 1;
    }
    
    .section-title {
        text-style: bold;
        margin: 1 0;
        color: $text;
    }
    
    .tab-content {
        height: 100%;
        padding: 1;
    }
    
    .button-group {
        margin: 1 0;
    }
    
    .hidden {
        display: none;
    }
    """
    
    collapsed = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.state = ChatSidebarState()
        # Only load preferences if not in test mode
        import sys
        if 'pytest' not in sys.modules:
            self.state.load_preferences()
        logger.debug("UnifiedChatSidebar initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the unified sidebar with tabbed interface."""
        with Container(classes="sidebar-container"):
            # Header with collapse button
            with Horizontal(classes="sidebar-header"):
                yield Static("Chat Controls", classes="sidebar-title")
                yield Button("◀", id="sidebar-collapse", classes="collapse-btn")
            
            # Main tabbed content
            with TabbedContent(id="sidebar-tabs", initial=self.state.active_tab):
                with TabPane("Session", id="session"):
                    yield SessionTab(self.app_instance)
                
                with TabPane("Settings", id="settings"):
                    yield SettingsTab(self.app_instance)
                
                with TabPane("Content", id="content"):
                    yield ContentTab(self.app_instance)
    
    def on_mount(self):
        """Initialize sidebar on mount."""
        # Apply saved width
        self.styles.width = f"{self.state.sidebar_width}%"
        
        # Apply docking position (default is left, which is already in CSS)
        if self.state.sidebar_position == "right":
            self.styles.dock = "right"
            self.styles.border_right = None
            self.styles.border_left = "solid $primary-darken-2"
    
    def action_toggle_sidebar(self):
        """Toggle sidebar visibility."""
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.add_class("collapsed")
        else:
            self.remove_class("collapsed")
    
    def action_switch_tab(self, tab_id: str):
        """Switch to specified tab."""
        try:
            tabs = self.query_one("#sidebar-tabs", TabbedContent)
            # Only switch if the tab exists
            if tab_id in ["session", "settings", "content"]:
                tabs.active = tab_id
                self.state.active_tab = tab_id
                # Only save preferences if not in test mode
                import sys
                if 'pytest' not in sys.modules:
                    self.state.save_preferences()
            else:
                logger.warning(f"Invalid tab ID: {tab_id}")
        except Exception as e:
            logger.error(f"Error switching tab: {e}")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses within the sidebar."""
        button_id = event.button.id
        
        if button_id == "sidebar-collapse":
            self.action_toggle_sidebar()
        elif button_id == "settings-advanced-toggle":
            await self._toggle_advanced_settings()
        elif button_id == "rag-toggle":
            await self._toggle_rag_settings()
        else:
            # Forward button press to the main app for handling
            # Map new IDs to old IDs for compatibility
            id_mapping = {
                "session-save-chat": "chat-save-current-chat-button",
                "session-new-chat": "chat-new-conversation-button",
                "session-clone-chat": "chat-clone-current-chat-button",
                "session-to-note": "chat-convert-to-note-button",
                "content-search-btn": "chat-media-search-button",
                "content-load": "chat-load-selected-note-button",
                "settings-rag-configure": "chat-rag-configure-button"
            }
            
            # If this is a mapped button, create a fake event with the old ID
            if button_id in id_mapping:
                old_id = id_mapping[button_id]
                # Post a message to the app with the old button ID
                from textual.message import Message
                class MappedButtonPress(Message):
                    def __init__(self, button_id: str):
                        super().__init__()
                        self.button_id = button_id
                
                # Find the chat window and trigger its handler
                try:
                    chat_window = self.app_instance.query_one("ChatWindowEnhanced")
                    # Create a fake button with the old ID for the event handler
                    fake_button = Button("", id=old_id)
                    fake_event = Button.Pressed(fake_button)
                    await chat_window.on_button_pressed(fake_event)
                except Exception as e:
                    logger.error(f"Error forwarding button press {button_id} -> {old_id}: {e}")
            else:
                # For non-mapped buttons, still forward to chat window handler
                # This ensures all sidebar buttons are properly handled
                try:
                    chat_window = self.app_instance.query_one("ChatWindowEnhanced")
                    await chat_window.on_button_pressed(event)
                except Exception as e:
                    logger.error(f"Error forwarding button press {button_id}: {e}")
    
    @on(Select.Changed)
    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes."""
        select_id = event.select.id if event.select else None
        
        if select_id == "settings-provider":
            # Update the model dropdown based on provider
            provider = event.value
            if provider:
                providers_models = get_cli_providers_and_models()
                models = providers_models.get(provider, [])
                model_select = self.query_one("#settings-model", Select)
                model_options = [(m, m) for m in models]
                model_select.set_options(model_options)
                if models:
                    model_select.value = models[0]
                    
                # Save the provider setting
                save_setting_to_cli_config("chat_defaults", "provider", provider)
                
        elif select_id == "settings-model":
            # Save the model setting
            if event.value:
                save_setting_to_cli_config("chat_defaults", "model", event.value)
                
        elif select_id == "settings-rag-pipeline":
            # Save RAG pipeline setting
            if event.value:
                save_setting_to_cli_config("rag", "pipeline", event.value)
    
    @on(Input.Changed)
    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input widget changes."""
        input_id = event.input.id if event.input else None
        
        if input_id == "settings-temperature":
            # Validate and save temperature
            try:
                temp = float(event.value) if event.value else 0.7
                temp = max(0.0, min(2.0, temp))  # Clamp between 0 and 2
                save_setting_to_cli_config("chat_defaults", "temperature", temp)
            except ValueError:
                pass  # Invalid input, ignore
                
        elif input_id == "settings-top-p":
            try:
                top_p = float(event.value) if event.value else 0.95
                top_p = max(0.0, min(1.0, top_p))
                save_setting_to_cli_config("chat_defaults", "top_p", top_p)
            except ValueError:
                pass
                
        elif input_id == "settings-top-k":
            try:
                top_k = int(event.value) if event.value else 50
                top_k = max(1, top_k)
                save_setting_to_cli_config("chat_defaults", "top_k", top_k)
            except ValueError:
                pass
                
        elif input_id == "settings-min-p":
            try:
                min_p = float(event.value) if event.value else 0.05
                min_p = max(0.0, min(1.0, min_p))
                save_setting_to_cli_config("chat_defaults", "min_p", min_p)
            except ValueError:
                pass
    
    @on(Checkbox.Changed)
    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        checkbox_id = event.checkbox.id if event.checkbox else None
        
        if checkbox_id == "settings-rag-enabled":
            save_setting_to_cli_config("rag", "enabled", event.value)
        elif checkbox_id == "session-strip-tags":
            save_setting_to_cli_config("chat_defaults", "strip_thinking_tags", event.value)
    
    async def _toggle_advanced_settings(self):
        """Toggle advanced settings visibility."""
        try:
            advanced = self.query_one("#advanced-settings")
            checkbox = self.query_one("#settings-advanced-toggle", Checkbox)
            
            if checkbox.value:
                advanced.remove_class("hidden")
            else:
                advanced.add_class("hidden")
                
            self.state.advanced_mode = checkbox.value
            self.state.save_preferences()
        except Exception as e:
            logger.error(f"Error toggling advanced settings: {e}")
    
    async def _toggle_rag_settings(self):
        """Toggle RAG settings visibility."""
        try:
            rag_container = self.query_one("#rag-settings")
            toggle_btn = self.query_one("#rag-toggle", Button)
            
            if "hidden" in rag_container.classes:
                rag_container.remove_class("hidden")
                toggle_btn.label = "▼ RAG Settings"
            else:
                rag_container.add_class("hidden")
                toggle_btn.label = "▶ RAG Settings"
        except Exception as e:
            logger.error(f"Error toggling RAG settings: {e}")

#
# End of unified_chat_sidebar.py
#######################################################################################################################