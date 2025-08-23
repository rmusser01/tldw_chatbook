# enhanced_settings_sidebar.py
# Description: Enhanced settings sidebar with improved UX and organization
#
# Imports
#
# 3rd-Party Imports
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Container
from textual.widgets import (
    Static, Select, TextArea, Input, Collapsible, Button, 
    Checkbox, ListView, TabbedContent, TabPane, Label
)
from textual.reactive import reactive
from textual.message import Message
from textual import on
from textual.css.query import NoMatches

#
# Local Imports
from ..config import get_cli_providers_and_models, get_cli_setting
from ..state.ui_state import UIState

# Try to import pipeline integration
try:
    from ..RAG_Search.pipeline_integration import get_pipeline_manager
    from ..RAG_Search.pipeline_builder_simple import get_pipeline, BUILTIN_PIPELINES
    PIPELINE_INTEGRATION_AVAILABLE = True
except ImportError:
    PIPELINE_INTEGRATION_AVAILABLE = False
    get_pipeline = None
    BUILTIN_PIPELINES = {}

#
#######################################################################################################################
#
# Data Classes for Organization

@dataclass
class SettingGroup:
    """Represents a group of related settings."""
    name: str
    icon: str
    priority: str  # "essential", "common", "advanced"
    collapsed_default: bool
    settings: List[str] = field(default_factory=list)
    description: str = ""
    
@dataclass
class SettingPreset:
    """Represents a preset configuration."""
    name: str
    icon: str
    description: str
    values: Dict[str, Any]

#
# Enhanced Sidebar Component
#

class EnhancedSettingsSidebar(Container):
    """Enhanced settings sidebar with improved UX and organization.
    
    Features:
    - Tabbed interface for better organization
    - Smart search and filtering
    - Visual hierarchy with icons and colors
    - Preset configurations
    - Lazy loading for performance
    - Persistent state management
    """
    
    # CSS classes for styling - removed as they are now in the main CSS files
    
    # Reactive properties
    search_query = reactive("", layout=False)
    active_preset = reactive("custom", layout=False)
    show_advanced = reactive(False, layout=False)
    
    def __init__(self, id_prefix: str, config: dict, **kwargs):
        """Initialize the enhanced sidebar.
        
        Args:
            id_prefix: Prefix for widget IDs (e.g., "chat")
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.id_prefix = id_prefix
        self.config = config
        self.ui_state = UIState()
        self.modified_settings = set()
        
        # Define setting groups for better organization
        self.setting_groups = self._define_setting_groups()
        
        # Define presets
        self.presets = self._define_presets()
        
        # Track loaded tabs for lazy loading
        self.loaded_tabs = set()
        
        # Initialize logging
        import logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _define_setting_groups(self) -> Dict[str, SettingGroup]:
        """Define the organization of settings into groups."""
        return {
            "provider": SettingGroup(
                name="Provider & Model",
                icon="🎯",
                priority="essential",
                collapsed_default=False,
                settings=["provider", "model", "temperature"],
                description="Core LLM configuration"
            ),
            "chat_config": SettingGroup(
                name="Chat Configuration", 
                icon="💬",
                priority="essential",
                collapsed_default=False,
                settings=["system_prompt", "streaming", "conversation"],
                description="Basic chat settings"
            ),
            "rag": SettingGroup(
                name="RAG Settings",
                icon="🔍",
                priority="common",
                collapsed_default=True,
                settings=["rag_enable", "rag_preset", "rag_pipeline"],
                description="Retrieval-Augmented Generation"
            ),
            "model_params": SettingGroup(
                name="Model Parameters",
                icon="⚙️",
                priority="advanced",
                collapsed_default=True,
                settings=["top_p", "top_k", "min_p", "max_tokens"],
                description="Fine-tune model behavior"
            ),
            "tools": SettingGroup(
                name="Tools & Extensions",
                icon="🛠️",
                priority="advanced",
                collapsed_default=True,
                settings=["tools", "templates", "dictionaries"],
                description="Advanced features and integrations"
            )
        }
    
    def _define_presets(self) -> Dict[str, SettingPreset]:
        """Define preset configurations."""
        return {
            "basic": SettingPreset(
                name="Basic",
                icon="📝",
                description="Simple chat with default settings",
                values={
                    "temperature": 0.7,
                    "streaming": True,
                    "rag_enable": False,
                    "show_advanced": False
                }
            ),
            "research": SettingPreset(
                name="Research",
                icon="🔬",
                description="Optimized for research with RAG",
                values={
                    "temperature": 0.3,
                    "streaming": True,
                    "rag_enable": True,
                    "rag_preset": "high_accuracy",
                    "max_tokens": 4096
                }
            ),
            "creative": SettingPreset(
                name="Creative",
                icon="🎨",
                description="Higher temperature for creative tasks",
                values={
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "streaming": True,
                    "rag_enable": False
                }
            ),
            "custom": SettingPreset(
                name="Custom",
                icon="⚡",
                description="Your custom configuration",
                values={}
            )
        }
    
    def compose(self) -> ComposeResult:
        """Compose the enhanced sidebar UI."""
        with VerticalScroll(id=f"{self.id_prefix}-left-sidebar", classes="sidebar enhanced-sidebar"):
            # Header with title and controls
            with Container(classes="sidebar-header"):
                yield Static("⚙️ Chat Settings", classes="sidebar-title")
                yield Button("↩️", id=f"{self.id_prefix}-reset-all", 
                           classes="reset-button", 
                           tooltip="Reset all settings to defaults")
            
            # Preset selector bar
            with Horizontal(classes="preset-bar"):
                for preset_id, preset in self.presets.items():
                    yield Button(
                        f"{preset.icon} {preset.name}",
                        id=f"{self.id_prefix}-preset-{preset_id}",
                        classes=f"preset-button {'active' if preset_id == self.active_preset else ''}",
                        tooltip=preset.description
                    )
            
            # Search bar
            with Container(classes="search-container"):
                yield Input(
                    placeholder="🔍 Search settings...",
                    id=f"{self.id_prefix}-settings-search",
                    classes="search-input"
                )
            
            # Tabbed content for organization
            with TabbedContent(id=f"{self.id_prefix}-settings-tabs"):
                # Essentials Tab
                with TabPane("⭐ Essentials", id=f"{self.id_prefix}-tab-essentials"):
                    yield from self._compose_essentials_tab()
                
                # Features Tab  
                with TabPane("🚀 Features", id=f"{self.id_prefix}-tab-features"):
                    # Lazy load - content added when tab activated
                    yield Container(id=f"{self.id_prefix}-features-content")
                
                # Advanced Tab
                with TabPane("🔧 Advanced", id=f"{self.id_prefix}-tab-advanced"):
                    # Lazy load - content added when tab activated
                    yield Container(id=f"{self.id_prefix}-advanced-content")
                
                # Search Tab (hidden by default, shown when searching)
                with TabPane("🔍 Search Results", id=f"{self.id_prefix}-tab-search"):
                    yield Container(id=f"{self.id_prefix}-search-results", classes="search-results")
    
    def _compose_essentials_tab(self) -> ComposeResult:
        """Compose the essentials tab content."""
        self.logger.debug("Composing essentials tab content")
        
        # Provider & Model section
        yield from self._compose_provider_section()
        
        # Core Settings section
        yield from self._compose_core_settings()
        
        # Current Chat section
        yield from self._compose_chat_details()
    
    def _compose_provider_section(self) -> ComposeResult:
        """Compose the provider and model selection section."""
        self.logger.debug("Composing provider section")
        defaults = self.config.get(f"{self.id_prefix}_defaults", 
                                  self.config.get("chat_defaults", {}))
        providers_models = get_cli_providers_and_models()
        available_providers = list(providers_models.keys())
        default_provider = defaults.get("provider", 
                                       available_providers[0] if available_providers else "")
        self.logger.debug(f"Available providers: {available_providers}, default: {default_provider}")
        
        with Collapsible(
            title="🎯 Provider & Model",
            collapsed=False,
            id=f"{self.id_prefix}-provider-section",
            classes="setting-group setting-group-essential"
        ):
            # Provider selection
            yield Label("Provider", classes="setting-label")
            provider_options = [(provider, provider) for provider in available_providers]
            provider_select = Select(
                options=provider_options,
                prompt="Select Provider...",
                allow_blank=False,
                id=f"{self.id_prefix}-api-provider",
                classes="setting-input"
            )
            # Set value after creating with options
            if default_provider in available_providers:
                provider_select.value = default_provider
            yield provider_select
            
            # Model selection
            yield Label("Model", classes="setting-label")
            initial_models = providers_models.get(default_provider, [])
            model_options = [(model, model) for model in initial_models]
            default_model = defaults.get("model", "")
            
            model_select = Select(
                options=model_options,
                prompt="Select Model...",
                allow_blank=True,
                id=f"{self.id_prefix}-api-model",
                classes="setting-input"
            )
            # Set value after creating with options
            if default_model in initial_models:
                model_select.value = default_model
            elif initial_models and len(initial_models) > 0:
                model_select.value = initial_models[0]
            yield model_select
            
            # Temperature with visual indicator
            yield Label("Temperature", classes="setting-label")
            with Horizontal(classes="temperature-container"):
                yield Input(
                    placeholder="0.7",
                    id=f"{self.id_prefix}-temperature",
                    value=str(defaults.get("temperature", 0.7)),
                    classes="temperature-input"
                )
                yield Static("🌡️", id=f"{self.id_prefix}-temp-indicator", 
                           classes="temp-indicator")
    
    def _compose_core_settings(self) -> ComposeResult:
        """Compose core settings section."""
        self.logger.debug("Composing core settings")
        defaults = self.config.get(f"{self.id_prefix}_defaults", 
                                  self.config.get("chat_defaults", {}))
        
        with Collapsible(
            title="⚡ Core Settings",
            collapsed=False,
            id=f"{self.id_prefix}-core-section",
            classes="setting-group setting-group-essential"
        ):
            # Temperature
            yield Label("Temperature", classes="setting-label")
            yield Input(
                placeholder="e.g., 0.7",
                id=f"{self.id_prefix}-temperature",
                value=str(defaults.get("temperature", 0.7)),
                classes="setting-input"
            )
            
            # System prompt
            yield Label("System Prompt", classes="setting-label")
            system_prompt_classes = "sidebar-textarea"
            if self.id_prefix == "chat":
                system_prompt_classes += " chat-system-prompt-styling"
            yield TextArea(
                id=f"{self.id_prefix}-system-prompt",
                text=defaults.get("system_prompt", ""),
                classes=system_prompt_classes
            )
            
            # Streaming toggle
            yield Checkbox(
                "Enable Streaming",
                id=f"{self.id_prefix}-streaming-enabled-checkbox",
                value=True,
                classes="streaming-toggle",
                tooltip="Enable/disable streaming responses. When disabled, responses appear all at once."
            )
            
            # Show attach button toggle (only for chat)
            if self.id_prefix == "chat":
                try:
                    from ..config import get_cli_setting
                    show_attach_button = get_cli_setting("chat.images", "show_attach_button", True)
                    self.logger.debug(f"Got show_attach_button: {show_attach_button}")
                    yield Checkbox(
                        "Show Attach File Button",
                        id=f"{self.id_prefix}-show-attach-button-checkbox",
                        value=show_attach_button,
                        classes="attach-toggle"
                    )
                except Exception as e:
                    self.logger.error(f"Error getting show_attach_button setting: {e}", exc_info=True)
                    # Fallback checkbox with default value
                    yield Checkbox(
                        "Show Attach File Button",
                        id=f"{self.id_prefix}-show-attach-button-checkbox",
                        value=True,
                        classes="attach-toggle"
                    )
    
    def _compose_chat_details(self) -> ComposeResult:
        """Compose current chat details section."""
        with Collapsible(
            title="💬 Current Chat",
            collapsed=False,
            id=f"{self.id_prefix}-chat-details",
            classes="setting-group setting-group-essential"
        ):
            # New chat buttons with clear hierarchy
            with Horizontal(classes="new-chat-buttons"):
                yield Button(
                    "➕ New Chat",
                    id=f"{self.id_prefix}-new-chat",
                    classes="primary-button",
                    variant="primary"
                )
                yield Button(
                    "📋 Clone",
                    id=f"{self.id_prefix}-clone-chat",
                    classes="secondary-button"
                )
            
            # Chat info display
            yield Label("Chat ID", classes="setting-label")
            yield Input(
                id=f"{self.id_prefix}-chat-id",
                value="Temp Chat",
                disabled=True,
                classes="info-display"
            )
            
            yield Label("Title", classes="setting-label")
            yield Input(
                id=f"{self.id_prefix}-chat-title",
                placeholder="Enter chat title...",
                classes="setting-input"
            )
    
    # Event Handlers
    
    @on(TabbedContent.TabActivated)
    async def handle_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Lazy load tab content when activated."""
        tab_id = event.tab.id
        self.logger.debug(f"Tab activated: {tab_id}")
        
        # Check if tab needs loading
        if tab_id not in self.loaded_tabs:
            if "features" in tab_id:
                await self._load_features_tab()
            elif "advanced" in tab_id:
                await self._load_advanced_tab()
            
            self.loaded_tabs.add(tab_id)
            self.logger.debug(f"Loaded tabs: {self.loaded_tabs}")
    
    @on(Input.Changed)
    def handle_search(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        # Check if this is the search input
        if event.input.id and "settings-search" in event.input.id:
            self.search_query = event.value
            if self.search_query:
                self._perform_search()
            else:
                self._clear_search()
    
    @on(Button.Pressed, ".preset-button")
    def handle_preset_selection(self, event: Button.Pressed) -> None:
        """Handle preset button clicks."""
        # Extract preset ID from button ID
        preset_id = event.button.id.split("-preset-")[-1]
        self._apply_preset(preset_id)
    
    @on(Checkbox.Changed)
    def handle_advanced_toggle(self, event: Checkbox.Changed) -> None:
        """Handle advanced settings toggle."""
        # Check if this is the advanced settings checkbox
        if event.checkbox.id and "show-advanced-checkbox" in event.checkbox.id:
            self.show_advanced = event.value
            self._update_visibility()
    
    # Helper Methods
    
    async def _load_features_tab(self) -> None:
        """Lazy load the features tab content."""
        self.logger.debug("Loading features tab content")
        try:
            container = self.query_one(f"#{self.id_prefix}-features-content")
            self.logger.debug("Found features content container")
            
            # Create and mount simple collapsible sections
            await self._mount_features_widgets(container)
            self.logger.debug("Successfully loaded features tab content")
                    
        except NoMatches as e:
            self.logger.error(f"Features content container not found: {e}")
        except Exception as e:
            self.logger.error(f"Error loading features tab: {e}", exc_info=True)
    
    async def _mount_features_widgets(self, container) -> None:
        """Mount features tab widgets to container."""
        from textual.widgets import Collapsible, Label, ListView
        
        # RAG Settings
        rag_section = Collapsible(
            title="🔍 RAG Settings",
            collapsed=True,
            id=f"{self.id_prefix}-rag-panel",
            classes="setting-group setting-group-common"
        )
        await container.mount(rag_section)
        await rag_section.mount(Label("RAG search and indexing settings", classes="setting-label"))
        
        # Notes section  
        notes_section = Collapsible(
            title="📝 Notes",
            collapsed=True,
            id=f"{self.id_prefix}-notes-collapsible",
            classes="setting-group setting-group-common"
        )
        await container.mount(notes_section)
        await notes_section.mount(Label("Notes management settings", classes="setting-label"))
        
        # Image Generation (if chat)
        if self.id_prefix == "chat":
            image_section = Collapsible(
                title="🎨 Image Generation",
                collapsed=True,
                id=f"{self.id_prefix}-image-generation-collapsible",
                classes="setting-group setting-group-common"
            )
            await container.mount(image_section)
            await image_section.mount(Label("Image generation settings", classes="setting-label"))
        
        # Search Media
        media_section = Collapsible(
            title="🔍 Search Media",
            collapsed=True,
            id=f"{self.id_prefix}-media-collapsible",
            classes="setting-group setting-group-common"
        )
        await container.mount(media_section)
        await media_section.mount(Label("Search media content", classes="setting-label"))
        await media_section.mount(ListView(id=f"{self.id_prefix}-media-search-results-listview", classes="sidebar-listview"))
    
    def _build_features_content(self) -> list:
        """Build features tab content widgets."""
        self.logger.debug("Building features content widgets")
        widgets = []
        
        try:
            # RAG Settings
            rag_collapsible = Collapsible(
                title="🔍 RAG Settings",
                collapsed=True,
                id=f"{self.id_prefix}-rag-panel",
                classes="setting-group setting-group-common"
            )
            rag_collapsible.mount(Label("RAG search and indexing settings", classes="setting-label"))
            widgets.append(rag_collapsible)
            
            # Notes section
            notes_collapsible = Collapsible(
                title="📝 Notes",
                collapsed=True,
                id=f"{self.id_prefix}-notes-collapsible",
                classes="setting-group setting-group-common"
            )
            notes_collapsible.mount(Label("Notes management settings", classes="setting-label"))
            widgets.append(notes_collapsible)
            
            # Image Generation (if chat)
            if self.id_prefix == "chat":
                image_collapsible = Collapsible(
                    title="🎨 Image Generation",
                    collapsed=True,
                    id=f"{self.id_prefix}-image-generation-collapsible",
                    classes="setting-group setting-group-common"
                )
                image_collapsible.mount(Label("Image generation settings", classes="setting-label"))
                widgets.append(image_collapsible)
            
            # Search Media
            media_collapsible = Collapsible(
                title="🔍 Search Media",
                collapsed=True,
                id=f"{self.id_prefix}-media-collapsible",
                classes="setting-group setting-group-common"
            )
            media_collapsible.mount(Label("Search media content", classes="setting-label"))
            media_collapsible.mount(ListView(id=f"{self.id_prefix}-media-search-results-listview", classes="sidebar-listview"))
            widgets.append(media_collapsible)
            
        except Exception as e:
            self.logger.error(f"Error building features content: {e}", exc_info=True)
            
        return widgets

    def _compose_features_content(self) -> ComposeResult:
        """Compose features tab content."""
        self.logger.debug("Composing features content")
        try:
            # RAG Settings
            self.logger.debug("Composing RAG section")
            yield from self._compose_rag_section()
            
            # Notes section
            self.logger.debug("Composing notes section")
            yield from self._compose_notes_section()
            
            # Image Generation (if chat)
            if self.id_prefix == "chat":
                self.logger.debug("Composing image generation section")
                yield from self._compose_image_generation_section()
            
            # Search Media
            self.logger.debug("Composing search media section")
            yield from self._compose_search_media_section()
        except Exception as e:
            self.logger.error(f"Error in _compose_features_content: {e}", exc_info=True)
            raise
    
    def _compose_rag_section(self) -> ComposeResult:
        """Compose RAG settings section."""
        with Collapsible(
            title="🔍 RAG Settings",
            collapsed=True,
            id=f"{self.id_prefix}-rag-panel",
            classes="setting-group setting-group-common"
        ):
            # RAG Enable checkbox
            yield Checkbox(
                "Enable RAG",
                id=f"{self.id_prefix}-rag-enabled-checkbox",
                value=False,
                tooltip="Enable Retrieval-Augmented Generation"
            )
            
            # RAG Search checkboxes
            yield Label("Search in:", classes="setting-label")
            yield Checkbox("Media Items", id=f"{self.id_prefix}-rag-search-media-checkbox", value=True)
            yield Checkbox("Conversations", id=f"{self.id_prefix}-rag-search-conversations-checkbox", value=False)
            yield Checkbox("Notes", id=f"{self.id_prefix}-rag-search-notes-checkbox", value=False)
    
    def _compose_notes_section(self) -> ComposeResult:
        """Compose notes section."""
        with Collapsible(
            title="📝 Notes",
            collapsed=True,
            id=f"{self.id_prefix}-notes-collapsible",
            classes="setting-group setting-group-common"
        ):
            yield Label("Notes content will be loaded here", classes="setting-label")
            yield TextArea(
                id=f"{self.id_prefix}-notes-content",
                classes="notes-textarea-normal"
            )
    
    def _compose_image_generation_section(self) -> ComposeResult:
        """Compose image generation section."""
        with Collapsible(
            title="🎨 Image Generation",
            collapsed=True,
            id=f"{self.id_prefix}-image-generation-collapsible",
            classes="setting-group setting-group-common"
        ):
            yield Label("Image generation settings", classes="setting-label")
    
    def _compose_search_media_section(self) -> ComposeResult:
        """Compose search media section."""
        with Collapsible(
            title="🔍 Search Media",
            collapsed=True,
            id=f"{self.id_prefix}-media-collapsible",
            classes="setting-group setting-group-common"
        ):
            yield Label("Search media content", classes="setting-label")
            yield ListView(id=f"{self.id_prefix}-media-search-results-listview", classes="sidebar-listview")
    
    async def _load_advanced_tab(self) -> None:
        """Lazy load the advanced tab content."""
        self.logger.debug("Loading advanced tab content")
        try:
            container = self.query_one(f"#{self.id_prefix}-advanced-content")
            self.logger.debug("Found advanced content container")
            
            # Create and mount advanced settings sections
            await self._mount_advanced_widgets(container)
            self.logger.debug("Successfully loaded advanced tab content")
                    
        except NoMatches as e:
            self.logger.error(f"Advanced content container not found: {e}")
        except Exception as e:
            self.logger.error(f"Error loading advanced tab: {e}", exc_info=True)
    
    async def _mount_advanced_widgets(self, container) -> None:
        """Mount advanced tab widgets to container."""
        from textual.widgets import Collapsible, Label, Input
        
        defaults = self.config.get(f"{self.id_prefix}_defaults", 
                                  self.config.get("chat_defaults", {}))
        
        # Model Parameters
        model_params = Collapsible(
            title="⚙️ Model Parameters",
            collapsed=True,
            id=f"{self.id_prefix}-model-params",
            classes="setting-group setting-group-advanced"
        )
        await container.mount(model_params)
        
        # Add parameter inputs
        await model_params.mount(Label("Top P", classes="setting-label"))
        await model_params.mount(Input(
            placeholder="e.g., 0.95",
            id=f"{self.id_prefix}-top-p",
            value=str(defaults.get("top_p", 0.95)),
            classes="setting-input"
        ))
        
        await model_params.mount(Label("Min P", classes="setting-label"))
        await model_params.mount(Input(
            placeholder="e.g., 0.05",
            id=f"{self.id_prefix}-min-p",
            value=str(defaults.get("min_p", 0.05)),
            classes="setting-input"
        ))
        
        # Advanced Settings
        advanced_settings = Collapsible(
            title="🔧 Advanced Settings",
            collapsed=True,
            id=f"{self.id_prefix}-advanced-settings",
            classes="setting-group setting-group-advanced"
        )
        await container.mount(advanced_settings)
        
        await advanced_settings.mount(Label("Custom Token Limit", classes="setting-label"))
        await advanced_settings.mount(Input(
            placeholder="0 = use Max Tokens",
            id=f"{self.id_prefix}-custom-token-limit",
            value="12888",
            classes="setting-input"
        ))
        
        # Tools & Templates
        tools_section = Collapsible(
            title="🛠️ Tools & Templates",
            collapsed=True,
            id=f"{self.id_prefix}-tools",
            classes="setting-group setting-group-advanced"
        )
        await container.mount(tools_section)
        await tools_section.mount(Label("Tools and templates configuration", classes="setting-label"))
    
    def _compose_advanced_content(self) -> ComposeResult:
        """Compose advanced tab content."""
        self.logger.debug("Composing advanced content")
        try:
            defaults = self.config.get(f"{self.id_prefix}_defaults", 
                                      self.config.get("chat_defaults", {}))
            
            # Model Parameters
            self.logger.debug("Composing model parameters section")
            yield from self._compose_model_parameters_section(defaults)
            
            # Advanced Settings
            self.logger.debug("Composing advanced settings section")
            yield from self._compose_advanced_settings_section()
            
            # Tools & Templates
            self.logger.debug("Composing tools section")
            yield from self._compose_tools_section()
        except Exception as e:
            self.logger.error(f"Error in _compose_advanced_content: {e}", exc_info=True)
            raise
    
    def _compose_model_parameters_section(self, defaults: dict) -> ComposeResult:
        """Compose model parameters section."""
        with Collapsible(
            title="⚙️ Model Parameters",
            collapsed=True,
            id=f"{self.id_prefix}-model-params",
            classes="setting-group setting-group-advanced"
        ):
            # Top-p
            yield Label("Top P", classes="setting-label")
            yield Input(
                placeholder="e.g., 0.95",
                id=f"{self.id_prefix}-top-p",
                value=str(defaults.get("top_p", 0.95)),
                classes="setting-input"
            )
            
            # Min-p  
            yield Label("Min P", classes="setting-label")
            yield Input(
                placeholder="e.g., 0.05",
                id=f"{self.id_prefix}-min-p", 
                value=str(defaults.get("min_p", 0.05)),
                classes="setting-input"
            )
            
            # Top-k
            yield Label("Top K", classes="setting-label")
            yield Input(
                placeholder="e.g., 50",
                id=f"{self.id_prefix}-top-k",
                value=str(defaults.get("top_k", 50)),
                classes="setting-input"
            )
            
            # Max tokens
            yield Label("Max Tokens", classes="setting-label")
            yield Input(
                placeholder="e.g., 2048",
                id=f"{self.id_prefix}-llm-max-tokens",
                value="2048",
                classes="setting-input"
            )
            
            # Seed
            yield Label("Seed", classes="setting-label")
            yield Input(
                placeholder="e.g., 42",
                id=f"{self.id_prefix}-llm-seed",
                value="0",
                classes="setting-input"
            )
    
    def _compose_advanced_settings_section(self) -> ComposeResult:
        """Compose advanced settings section."""
        with Collapsible(
            title="🔧 Advanced Settings",
            collapsed=True,
            id=f"{self.id_prefix}-advanced-settings",
            classes="setting-group setting-group-advanced"
        ):
            # Custom token limit
            yield Label("Custom Token Limit", classes="setting-label")
            yield Input(
                placeholder="0 = use Max Tokens",
                id=f"{self.id_prefix}-custom-token-limit",
                value="12888",
                classes="setting-input"
            )
            
            # Stop sequences
            yield Label("Stop Sequences", classes="setting-label") 
            yield Input(
                placeholder="e.g., <|endoftext|>,<|eot_id|>",
                id=f"{self.id_prefix}-llm-stop",
                classes="setting-input"
            )
            
            # Frequency penalty
            yield Label("Frequency Penalty", classes="setting-label")
            yield Input(
                placeholder="e.g., 0.0 to 2.0", 
                id=f"{self.id_prefix}-llm-frequency-penalty",
                value="0.0",
                classes="setting-input"
            )
            
            # Presence penalty
            yield Label("Presence Penalty", classes="setting-label")
            yield Input(
                placeholder="e.g., 0.0 to 2.0",
                id=f"{self.id_prefix}-llm-presence-penalty", 
                value="0.0",
                classes="setting-input"
            )
    
    def _compose_tools_section(self) -> ComposeResult:
        """Compose tools and templates section."""
        with Collapsible(
            title="🛠️ Tools & Templates",
            collapsed=True,
            id=f"{self.id_prefix}-tools",
            classes="setting-group setting-group-advanced"
        ):
            yield Label("Tools and templates configuration", classes="setting-label")
    
    def _perform_search(self) -> None:
        """Perform search and show results."""
        # Implementation for searching through settings
        # Would highlight matching settings and show in search tab
        pass
    
    def _clear_search(self) -> None:
        """Clear search results and return to normal view."""
        pass
    
    def _apply_preset(self, preset_id: str) -> None:
        """Apply a preset configuration."""
        if preset_id in self.presets:
            preset = self.presets[preset_id]
            # Apply preset values to settings
            for key, value in preset.values.items():
                self._set_setting_value(key, value)
            
            self.active_preset = preset_id
            self._update_preset_buttons()
    
    def _set_setting_value(self, key: str, value: Any) -> None:
        """Set a setting value programmatically."""
        # Implementation to set various setting types
        pass
    
    def _update_preset_buttons(self) -> None:
        """Update preset button styles based on active preset."""
        for preset_id in self.presets:
            try:
                button = self.query_one(f"#{self.id_prefix}-preset-{preset_id}")
                if preset_id == self.active_preset:
                    button.add_class("active")
                else:
                    button.remove_class("active")
            except NoMatches:
                pass
    
    def _update_visibility(self) -> None:
        """Update visibility of advanced settings."""
        # Show/hide advanced settings based on toggle
        pass
    
    def mark_setting_modified(self, setting_id: str) -> None:
        """Mark a setting as modified from default."""
        self.modified_settings.add(setting_id)
        try:
            widget = self.query_one(f"#{setting_id}")
            widget.add_class("setting-modified")
        except NoMatches:
            pass
    
    def reset_all_settings(self) -> None:
        """Reset all settings to defaults."""
        # Implementation to reset all settings
        self.modified_settings.clear()
        self._apply_preset("basic")

#
# Factory function for backwards compatibility
#

def create_enhanced_settings_sidebar(id_prefix: str, config: dict) -> EnhancedSettingsSidebar:
    """Create an enhanced settings sidebar instance.
    
    This function provides backwards compatibility with the existing codebase
    while offering the new enhanced sidebar.
    
    Args:
        id_prefix: Prefix for widget IDs
        config: Configuration dictionary
        
    Returns:
        EnhancedSettingsSidebar instance
    """
    return EnhancedSettingsSidebar(
        id_prefix=id_prefix,
        config=config,
        id=f"{id_prefix}-enhanced-sidebar"
    )

#
# End of enhanced_settings_sidebar.py
#######################################################################################################################