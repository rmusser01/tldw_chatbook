# settings_sidebar_optimized.py
# Performance-optimized version of settings sidebar with lazy loading

import logging
from typing import Dict, Any
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Container
from textual.widgets import Static, Select, TextArea, Input, Button, Checkbox, Label, Switch
from ..config import get_cli_providers_and_models
from .lazy_widgets import LazyCollapsible, VirtualListView

logger = logging.getLogger(__name__)

def create_settings_sidebar_optimized(id_prefix: str, config: dict) -> ComposeResult:
    """Create an optimized settings sidebar with lazy loading.
    
    This version significantly improves startup performance by:
    1. Deferring creation of collapsed sections
    2. Loading only basic settings initially
    3. Using lazy loading for advanced features
    """
    sidebar_id = f"{id_prefix}-left-sidebar"
    
    with VerticalScroll(id=sidebar_id, classes="sidebar"):
        # Header
        yield Static("Chat Settings", classes="sidebar-title")
        
        # Mode toggle - controls what's visible
        with Container(classes="mode-toggle-container"):
            yield Label("Mode: ", classes="mode-label")
            yield Switch(value=False, id=f"{id_prefix}-advanced-mode-switch", 
                        tooltip="Toggle advanced settings")
            yield Label("Basic", id=f"{id_prefix}-mode-indicator", classes="mode-indicator")
        
        # Get configuration
        defaults = config.get(f"{id_prefix}_defaults", config.get("chat_defaults", {}))
        
        # Basic settings - always visible and loaded immediately
        yield from _create_basic_settings(id_prefix, defaults)
        
        # Advanced settings - loaded lazily
        yield LazyCollapsible(
            title="🔍 RAG Settings",
            collapsed=True,
            id=f"{id_prefix}-rag-panel",
            classes="settings-collapsible advanced-only",
            content_factory=lambda: _create_rag_settings(id_prefix, defaults)
        )
        
        yield LazyCollapsible(
            title="⚙️ Advanced Options",
            collapsed=True,
            id=f"{id_prefix}-advanced-options",
            classes="settings-collapsible advanced-only",
            content_factory=lambda: _create_advanced_options(id_prefix, defaults)
        )
        
        yield LazyCollapsible(
            title="🎭 Character Settings",
            collapsed=True,
            id=f"{id_prefix}-character-settings",
            classes="settings-collapsible",
            content_factory=lambda: _create_character_settings(id_prefix, defaults)
        )
        
        yield LazyCollapsible(
            title="📝 Conversation Management",
            collapsed=True,
            id=f"{id_prefix}-conversation-settings",
            classes="settings-collapsible",
            content_factory=lambda: _create_conversation_settings(id_prefix, defaults)
        )


def _create_basic_settings(id_prefix: str, defaults: dict) -> ComposeResult:
    """Create basic settings that are always visible."""
    # Get provider/model info
    providers_models = get_cli_providers_and_models()
    available_providers = list(providers_models.keys())
    default_provider = defaults.get("provider", available_providers[0] if available_providers else "")
    default_model = defaults.get("model", "")
    
    # Quick Settings - minimal set for basic usage
    with Container(classes="basic-settings-container"):
        yield Static("Provider & Model", classes="sidebar-label")
        
        # Provider selection
        yield Select(
            [(provider, provider) for provider in available_providers],
            prompt="Select Provider…",
            value=default_provider if default_provider in available_providers else None,
            id=f"{id_prefix}-api-provider",
            classes="provider-select"
        )
        
        # Model selection (will be populated based on provider)
        provider_models = providers_models.get(default_provider, [])
        model_options = [(model, model) for model in provider_models]
        
        yield Select(
            model_options,
            prompt="Select Model…",
            value=default_model if default_model in provider_models else None,
            id=f"{id_prefix}-api-model",
            classes="model-select"
        )
        
        # Temperature - essential setting
        yield Static("Temperature", classes="sidebar-label")
        yield Input(
            value=str(defaults.get("temperature", 0.7)),
            placeholder="0.0 - 2.0",
            id=f"{id_prefix}-temperature",
            classes="temperature-input"
        )
        
        # System prompt - commonly used
        yield Static("System Prompt", classes="sidebar-label")
        yield TextArea(
            defaults.get("system_prompt", ""),
            id=f"{id_prefix}-system-prompt",
            classes="system-prompt-textarea"
        )
        
        # Simple checkboxes for common options
        yield Checkbox(
            "Stream Responses",
            value=defaults.get("streaming", True),
            id=f"{id_prefix}-streaming-checkbox",
            classes="streaming-checkbox"
        )


def _create_rag_settings(id_prefix: str, defaults: dict) -> ComposeResult:
    """Create RAG settings - loaded on demand."""
    logger.debug("Creating RAG settings (lazy loaded)")
    
    # RAG enable checkbox
    yield Checkbox(
        "Enable RAG (Retrieval Augmented Generation)",
        value=defaults.get("rag_enabled", False),
        id=f"{id_prefix}-rag-enabled-checkbox",
        classes="rag-enabled-checkbox"
    )
    
    # RAG presets
    rag_presets = [
        ("none", "None - No RAG"),
        ("simple", "Simple - Basic keyword search"),
        ("speed_optimized", "Speed Optimized - Fast BM25 search"),
        ("high_accuracy", "High Accuracy - Semantic search"),
        ("research_focused", "Research - Advanced pipeline"),
    ]
    
    yield Static("RAG Preset", classes="sidebar-label")
    yield Select(
        rag_presets,
        prompt="Select preset...",
        value=defaults.get("rag_preset", "none"),
        id=f"{id_prefix}-rag-preset-select",
        classes="rag-preset-select"
    )
    
    # Search scope checkboxes
    yield Static("Search Scope", classes="sidebar-label")
    with Container(classes="rag-scope-options"):
        yield Checkbox("Media Items", id=f"{id_prefix}-rag-search-media-checkbox", value=True)
        yield Checkbox("Conversations", id=f"{id_prefix}-rag-search-conversations-checkbox", value=False)
        yield Checkbox("Notes", id=f"{id_prefix}-rag-search-notes-checkbox", value=False)
    
    # Basic RAG controls
    yield Static("Top Results", classes="sidebar-label")
    yield Input(
        value=str(defaults.get("rag_top_k", 10)),
        placeholder="Number of results",
        id=f"{id_prefix}-rag-top-k",
        classes="rag-top-k-input"
    )
    
    yield Static("Filter Keywords", classes="sidebar-label")
    yield Input(
        placeholder="Optional keywords...",
        id=f"{id_prefix}-rag-filter-keywords",
        classes="rag-filter-input"
    )


def _create_advanced_options(id_prefix: str, defaults: dict) -> ComposeResult:
    """Create advanced options - loaded on demand."""
    logger.debug("Creating advanced options (lazy loaded)")
    
    # Advanced model parameters
    yield Static("Advanced Model Parameters", classes="section-header")
    
    yield Static("Max Tokens", classes="sidebar-label")
    yield Input(
        value=str(defaults.get("max_tokens", 4096)),
        placeholder="Max response tokens",
        id=f"{id_prefix}-max-tokens",
        classes="max-tokens-input"
    )
    
    yield Static("Top P", classes="sidebar-label")
    yield Input(
        value=str(defaults.get("top_p", 1.0)),
        placeholder="0.0 - 1.0",
        id=f"{id_prefix}-top-p",
        classes="top-p-input"
    )
    
    yield Static("Frequency Penalty", classes="sidebar-label")
    yield Input(
        value=str(defaults.get("frequency_penalty", 0.0)),
        placeholder="-2.0 - 2.0",
        id=f"{id_prefix}-frequency-penalty",
        classes="frequency-penalty-input"
    )
    
    yield Static("Presence Penalty", classes="sidebar-label")
    yield Input(
        value=str(defaults.get("presence_penalty", 0.0)),
        placeholder="-2.0 - 2.0",
        id=f"{id_prefix}-presence-penalty",
        classes="presence-penalty-input"
    )
    
    # Additional options
    yield Checkbox(
        "Show Token Count",
        value=defaults.get("show_token_count", True),
        id=f"{id_prefix}-show-token-count",
        classes="token-count-checkbox"
    )
    
    yield Checkbox(
        "Save Conversation History",
        value=defaults.get("save_history", True),
        id=f"{id_prefix}-save-history",
        classes="save-history-checkbox"
    )
    
    yield Checkbox(
        "Enable Tool Calling",
        value=defaults.get("tool_calling", False),
        id=f"{id_prefix}-tool-calling",
        classes="tool-calling-checkbox"
    )


def _create_character_settings(id_prefix: str, defaults: dict) -> ComposeResult:
    """Create character settings - loaded on demand."""
    logger.debug("Creating character settings (lazy loaded)")
    
    yield Static("Active Character", classes="sidebar-label")
    yield Input(
        placeholder="No character loaded",
        id=f"{id_prefix}-active-character",
        classes="active-character-input",
        disabled=True
    )
    
    yield Button(
        "Load Character",
        id=f"{id_prefix}-load-character-button",
        classes="load-character-button"
    )
    
    yield Button(
        "Clear Character",
        id=f"{id_prefix}-clear-character-button",
        classes="clear-character-button"
    )
    
    # Character search
    yield Static("Search Characters", classes="sidebar-label")
    yield Input(
        placeholder="Type to search...",
        id=f"{id_prefix}-character-search",
        classes="character-search-input"
    )
    
    # Character list will be populated dynamically
    yield Container(
        id=f"{id_prefix}-character-list-container",
        classes="character-list-container"
    )


def _create_conversation_settings(id_prefix: str, defaults: dict) -> ComposeResult:
    """Create conversation settings - loaded on demand."""
    logger.debug("Creating conversation settings (lazy loaded)")
    
    yield Static("Current Conversation", classes="sidebar-label")
    yield Input(
        placeholder="New Conversation",
        id=f"{id_prefix}-conversation-title",
        classes="conversation-title-input"
    )
    
    yield Button(
        "New Conversation",
        id=f"{id_prefix}-new-conversation-button",
        classes="new-conversation-button"
    )
    
    yield Button(
        "Save Conversation",
        id=f"{id_prefix}-save-conversation-button",
        classes="save-conversation-button"
    )
    
    # Conversation search
    yield Static("Search Conversations", classes="sidebar-label")
    yield Input(
        placeholder="Type to search...",
        id=f"{id_prefix}-conversation-search",
        classes="conversation-search-input"
    )
    
    # Conversation list will be populated dynamically
    yield Container(
        id=f"{id_prefix}-conversation-list-container",
        classes="conversation-list-container"
    )