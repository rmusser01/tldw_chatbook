"""
Minimal, actually usable settings sidebar.

Design principles:
1. Show only what's needed RIGHT NOW
2. Everything else hidden but accessible
3. Smart defaults - user shouldn't need to touch most settings
4. Quick access to common actions
"""

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Container
from textual.widgets import Static, Select, TextArea, Input, Button, Checkbox, Label
from textual.reactive import reactive
from textual import on

from ..config import get_cli_providers_and_models


class MinimalSettingsSidebar(Container):
    """A sidebar that doesn't suck."""
    
    def __init__(self, id_prefix: str, config: dict, **kwargs):
        super().__init__(**kwargs)
        self.id_prefix = id_prefix
        self.config = config
        self.defaults = config.get(f"{id_prefix}_defaults", config.get("chat_defaults", {}))
        
        # Track state
        self.expanded_mode = reactive(False)
        
    def compose(self) -> ComposeResult:
        """Create a MINIMAL sidebar."""
        
        with VerticalScroll(id=f"{self.id_prefix}-sidebar", classes="minimal-sidebar"):
            # --- HEADER: Just the essentials ---
            yield Static("Chat Settings", classes="sidebar-title")
            
            # --- QUICK ACTIONS: What users actually use ---
            with Horizontal(classes="quick-actions"):
                yield Button("New Chat", id=f"{self.id_prefix}-new-chat", variant="primary")
                yield Button("Clear", id=f"{self.id_prefix}-clear-chat", variant="warning")
            
            # --- CORE SETTINGS: Only what matters ---
            with Container(classes="core-settings"):
                # Provider & Model - THE most important
                yield from self._compose_provider_model()
                
                # Temperature - Second most tweaked setting
                yield Label("Temperature", classes="setting-label")
                yield Input(
                    value=str(self.defaults.get("temperature", 0.7)),
                    id=f"{self.id_prefix}-temperature",
                    placeholder="0.7",
                    classes="setting-input"
                )
                
                # System prompt - Occasionally needed
                yield Label("System Prompt (optional)", classes="setting-label")
                yield TextArea(
                    text=self.defaults.get("system_prompt", ""),
                    id=f"{self.id_prefix}-system-prompt",
                    classes="system-prompt-small"
                )
                
                # Streaming - Simple toggle
                yield Checkbox(
                    "Stream responses",
                    value=True,
                    id=f"{self.id_prefix}-streaming"
                )
            
            # --- EXPAND BUTTON: For power users ---
            yield Button(
                "⚙️ Advanced Settings",
                id=f"{self.id_prefix}-toggle-advanced",
                classes="expand-button"
            )
            
            # --- ADVANCED SECTION: Hidden by default ---
            with Container(
                id=f"{self.id_prefix}-advanced-section",
                classes="advanced-section hidden"
            ):
                yield Static("Advanced Settings", classes="section-title")
                
                # RAG toggle - Keep it simple
                yield Checkbox(
                    "Enable RAG",
                    value=False,
                    id=f"{self.id_prefix}-rag-enable"
                )
                
                # Max tokens
                yield Label("Max Tokens", classes="setting-label")
                yield Input(
                    value="2048",
                    id=f"{self.id_prefix}-max-tokens",
                    placeholder="2048"
                )
                
                # Top-p
                yield Label("Top P", classes="setting-label")
                yield Input(
                    value=str(self.defaults.get("top_p", 0.95)),
                    id=f"{self.id_prefix}-top-p",
                    placeholder="0.95"
                )
                
                # Note: Add more ONLY if actually needed
    
    def _compose_provider_model(self) -> ComposeResult:
        """Just provider and model selection."""
        providers_models = get_cli_providers_and_models()
        available_providers = list(providers_models.keys())
        default_provider = self.defaults.get("provider", available_providers[0] if available_providers else "")
        
        # Provider
        yield Label("Provider", classes="setting-label")
        provider_select = Select(
            options=[(p, p) for p in available_providers],
            prompt="Select Provider...",
            id=f"{self.id_prefix}-api-provider",
            classes="setting-select"
        )
        if default_provider in available_providers:
            provider_select.value = default_provider
        yield provider_select
        
        # Model
        yield Label("Model", classes="setting-label")
        initial_models = providers_models.get(default_provider, [])
        model_select = Select(
            options=[(m, m) for m in initial_models],
            prompt="Select Model...",
            id=f"{self.id_prefix}-api-model",
            classes="setting-select"
        )
        if initial_models:
            model_select.value = initial_models[0]
        yield model_select
    
    @on(Button.Pressed)
    def toggle_advanced(self, event: Button.Pressed) -> None:
        """Show/hide advanced settings."""
        if event.button.id and "toggle-advanced" in event.button.id:
            try:
                advanced = self.query_one(f"#{self.id_prefix}-advanced-section")
                if "hidden" in advanced.classes:
                    advanced.remove_class("hidden")
                    event.button.label = "⚙️ Hide Advanced"
                else:
                    advanced.add_class("hidden")
                    event.button.label = "⚙️ Advanced Settings"
            except:
                pass
    
    @on(Select.Changed)
    def handle_provider_change(self, event: Select.Changed) -> None:
        """Update models when provider changes."""
        if not event.select.id or "api-provider" not in event.select.id:
            return
        if not event.value:
            return
            
        from ..config import get_cli_providers_and_models
        providers_models = get_cli_providers_and_models()
        
        try:
            model_select = self.query_one(f"#{self.id_prefix}-api-model", Select)
            new_models = providers_models.get(str(event.value), [])
            model_select.set_options([(m, m) for m in new_models])
            if new_models:
                model_select.value = new_models[0]
        except:
            pass


# CSS for the minimal sidebar
MINIMAL_SIDEBAR_CSS = """
/* Minimal Sidebar Styles */
.minimal-sidebar {
    padding: 1 2;
    background: $boost;
    height: 100%;
}

.sidebar-title {
    text-style: bold;
    margin-bottom: 1;
    text-align: center;
}

.quick-actions {
    height: 3;
    margin-bottom: 2;
    align: center middle;
}

.quick-actions Button {
    width: 45%;
    margin: 0 1;
}

.core-settings {
    padding: 1;
    background: $surface 10%;
    border: round $surface-lighten-1;
    margin-bottom: 2;
}

.setting-label {
    margin-top: 1;
    color: $text-muted;
    text-style: bold;
}

.setting-input, .setting-select {
    width: 100%;
    margin-bottom: 1;
}

.system-prompt-small {
    width: 100%;
    height: 5;
    margin-bottom: 1;
}

.expand-button {
    width: 100%;
    margin: 1 0;
    background: $primary 10%;
}

.expand-button:hover {
    background: $primary 20%;
}

.advanced-section {
    padding: 1;
    background: $warning 5%;
    border: round $warning 30%;
}

.advanced-section.hidden {
    display: none;
}

.section-title {
    text-style: bold underline;
    margin-bottom: 1;
}
"""