"""Sidebar widget with tabbed interface."""

from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Button, Input, Select, TabbedContent, TabPane, Label
from textual.reactive import reactive
from textual import on
from typing import List, Tuple


class ChatSidebar(Container):
    """Sidebar with tabs for sessions, settings, and history."""
    
    DEFAULT_CSS = """
    ChatSidebar {
        background: $panel;
        border-right: solid $primary;
        width: 30;
        height: 100%;
    }
    
    ChatSidebar.-hidden {
        display: none;
    }
    
    .sidebar-section {
        padding: 1;
        margin-bottom: 1;
        background: $surface;
        border: round $primary;
    }
    
    .sidebar-title {
        text-style: bold;
        margin-bottom: 1;
        color: $accent;
        text-align: center;
    }
    
    .sidebar-input {
        width: 100%;
        margin-bottom: 1;
        background: $surface;
    }
    
    .sidebar-button {
        width: 100%;
        margin-bottom: 1;
    }
    
    .sidebar-button.primary {
        background: $primary;
    }
    
    .sidebar-button.secondary {
        background: $surface;
        border: solid $primary;
    }
    
    .session-item {
        padding: 1;
        margin-bottom: 1;
        background: $surface;
        border: round $primary;
    }
    
    .session-item:hover {
        background: $panel;
        border: round $accent;
    }
    
    .session-item.active {
        background: $primary 20%;
        border: solid $primary;
    }
    
    TabbedContent {
        height: 100%;
    }
    
    TabPane {
        padding: 1;
    }
    """
    
    # Reactive state
    current_provider: reactive[str] = reactive("openai")
    current_model: reactive[str] = reactive("gpt-4")
    session_history: reactive[List[Tuple[str, str]]] = reactive(list)
    
    def compose(self):
        """Compose sidebar tabs using proper pattern."""
        with TabbedContent():
            with TabPane("Sessions", id="sessions-tab"):
                yield from self._compose_sessions_tab()
            
            with TabPane("Settings", id="settings-tab"):
                yield from self._compose_settings_tab()
            
            with TabPane("History", id="history-tab"):
                yield from self._compose_history_tab()
    
    def _compose_sessions_tab(self):
        """Compose sessions tab content."""
        with VerticalScroll():
            with Container(classes="sidebar-section"):
                yield Static("Current Session", classes="sidebar-title")
                
                yield Input(
                    placeholder="Session title...",
                    id="session-title",
                    classes="sidebar-input"
                )
                
                yield Button(
                    "✨ New Session",
                    id="new-session",
                    classes="sidebar-button primary"
                )
                
                yield Button(
                    "💾 Save Session",
                    id="save-session",
                    classes="sidebar-button secondary"
                )
                
                yield Button(
                    "📂 Load Session",
                    id="load-session",
                    classes="sidebar-button secondary"
                )
                
                yield Button(
                    "🗑️ Clear Messages",
                    id="clear-messages",
                    classes="sidebar-button secondary"
                )
    
    def _compose_settings_tab(self):
        """Compose settings tab content."""
        with VerticalScroll():
            with Container(classes="sidebar-section"):
                yield Static("LLM Settings", classes="sidebar-title")
                
                # Provider selection
                yield Label("Provider:")
                providers = [
                    ("openai", "OpenAI"),
                    ("anthropic", "Anthropic"),
                    ("local", "Local Model")
                ]
                yield Select(
                    options=providers,
                    id="provider-select",
                    allow_blank=False
                )
                
                # Model selection
                yield Label("Model:")
                yield Select(
                    options=[("gpt-4", "GPT-4")],  # Default options
                    id="model-select",
                    allow_blank=False
                )
                
                # Temperature input
                yield Label("Temperature (0-1):")
                yield Input(
                    value="0.7",
                    id="temperature",
                    placeholder="0.7",
                    classes="sidebar-input"
                )
                
                # Max tokens input
                yield Label("Max Tokens:")
                yield Input(
                    value="",
                    id="max-tokens",
                    placeholder="Leave empty for default",
                    classes="sidebar-input"
                )
                
                # Streaming toggle
                yield Button(
                    "🔄 Streaming: ON",
                    id="streaming-toggle",
                    classes="sidebar-button secondary"
                )
    
    def _compose_history_tab(self):
        """Compose history tab content."""
        with VerticalScroll():
            yield Static("Recent Conversations", classes="sidebar-title")
            
            # Placeholder for session history
            with Container(id="history-list"):
                # This would be populated from database
                for i in range(5):
                    yield Button(
                        f"Session {i + 1}: Chat about Textual",
                        classes="session-item",
                        id=f"history-{i}"
                    )
    
    def on_mount(self):
        """Initialize sidebar after mounting."""
        # Load initial models for selected provider
        self._update_models_for_provider("openai")
    
    @on(Select.Changed, "#provider-select")
    def handle_provider_change(self, event: Select.Changed):
        """Update models when provider changes."""
        self.current_provider = str(event.value)
        self._update_models_for_provider(self.current_provider)
    
    @on(Select.Changed, "#model-select")
    def handle_model_change(self, event: Select.Changed):
        """Update current model selection."""
        self.current_model = str(event.value)
        
        # Notify app of model change
        self.notify(f"Model changed to: {self.current_model}")
    
    def _update_models_for_provider(self, provider: str):
        """Update model options based on selected provider.
        
        Args:
            provider: The selected provider
        """
        model_select = self.query_one("#model-select", Select)
        
        # Model options based on provider
        if provider == "openai":
            models = [
                ("gpt-4", "GPT-4"),
                ("gpt-4-turbo", "GPT-4 Turbo"),
                ("gpt-3.5-turbo", "GPT-3.5 Turbo")
            ]
        elif provider == "anthropic":
            models = [
                ("claude-3-opus", "Claude 3 Opus"),
                ("claude-3-sonnet", "Claude 3 Sonnet"),
                ("claude-3-haiku", "Claude 3 Haiku")
            ]
        else:  # local
            models = [
                ("llama-2-7b", "Llama 2 7B"),
                ("mistral-7b", "Mistral 7B"),
                ("custom", "Custom Model")
            ]
        
        model_select.set_options(models)
        
        # Don't set value here - it will be set after options are loaded
    
    @on(Button.Pressed, "#new-session")
    def handle_new_session(self):
        """Handle new session button."""
        # Post action to app
        self.app.action_new_session()
    
    @on(Button.Pressed, "#save-session")
    def handle_save_session(self):
        """Handle save session button."""
        self.app.action_save_session()
    
    @on(Button.Pressed, "#load-session")
    def handle_load_session(self):
        """Handle load session button."""
        self.app.action_open_session()
    
    @on(Button.Pressed, "#clear-messages")
    def handle_clear_messages(self):
        """Handle clear messages button."""
        self.app.action_clear_messages()
    
    @on(Button.Pressed, "#streaming-toggle")
    def handle_streaming_toggle(self, event: Button.Pressed):
        """Toggle streaming mode."""
        button = event.button
        if "ON" in button.label:
            button.label = "🔄 Streaming: OFF"
            self.app.settings.streaming = False
        else:
            button.label = "🔄 Streaming: ON"
            self.app.settings.streaming = True
    
    @on(Input.Changed, "#temperature")
    def handle_temperature_change(self, event: Input.Changed):
        """Validate and update temperature."""
        try:
            temp = float(event.value)
            if 0 <= temp <= 1:
                self.app.settings.temperature = temp
            else:
                event.input.value = str(self.app.settings.temperature)
                self.notify("Temperature must be between 0 and 1", severity="warning")
        except ValueError:
            if event.value:  # Only reset if not empty
                event.input.value = str(self.app.settings.temperature)
    
    @on(Input.Changed, "#max-tokens")
    def handle_max_tokens_change(self, event: Input.Changed):
        """Validate and update max tokens."""
        if not event.value:
            self.app.settings.max_tokens = None
            return
        
        try:
            tokens = int(event.value)
            if tokens > 0:
                self.app.settings.max_tokens = tokens
            else:
                event.input.value = ""
                self.notify("Max tokens must be positive", severity="warning")
        except ValueError:
            event.input.value = ""
    
    def load_session_history(self, sessions: List[Tuple[str, str]]):
        """Load session history into the history tab.
        
        Args:
            sessions: List of (session_id, session_title) tuples
        """
        self.session_history = sessions
        
        # Update history list
        history_list = self.query_one("#history-list", Container)
        history_list.remove_children()
        
        for session_id, title in sessions:
            history_list.mount(Button(
                title,
                classes="session-item",
                id=f"history-{session_id}"
            ))