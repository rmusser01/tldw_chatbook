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
                    ("google", "Google (Gemini)"),
                    ("mistral", "Mistral AI"),
                    ("cohere", "Cohere"),
                    ("groq", "Groq"),
                    ("deepseek", "DeepSeek"),
                    ("openrouter", "OpenRouter"),
                    ("huggingface", "HuggingFace"),
                    ("moonshot", "Moonshot (Kimi)"),
                    ("zai", "01.AI (Yi)"),
                    ("ollama", "Ollama (Local)"),
                    ("llama_cpp", "Llama.cpp (Local)"),
                    ("koboldcpp", "KoboldCPP (Local)"),
                    ("oobabooga", "Oobabooga (Local)"),
                    ("tabbyapi", "TabbyAPI (Local)"),
                    ("vllm", "vLLM (Local)"),
                    ("local-llm", "Local LLM (Generic)"),
                    ("custom_openai", "Custom OpenAI API")
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
                
                # API Key input (for non-local providers)
                yield Label("API Key:")
                yield Input(
                    value="",
                    id="api-key",
                    placeholder="Enter API key or use config",
                    password=True,
                    classes="sidebar-input"
                )
                
                # System prompt
                yield Label("System Prompt:")
                yield Input(
                    value="",
                    id="system-prompt",
                    placeholder="Optional system prompt",
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
        
        # Comprehensive model options for ALL providers
        model_map = {
            "openai": [
                ("gpt-4-turbo-preview", "GPT-4 Turbo Preview"),
                ("gpt-4", "GPT-4"),
                ("gpt-4-32k", "GPT-4 32K"),
                ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
                ("gpt-3.5-turbo-16k", "GPT-3.5 Turbo 16K")
            ],
            "anthropic": [
                ("claude-3-opus-20240229", "Claude 3 Opus"),
                ("claude-3-sonnet-20240229", "Claude 3 Sonnet"),
                ("claude-3-haiku-20240307", "Claude 3 Haiku"),
                ("claude-2.1", "Claude 2.1"),
                ("claude-2.0", "Claude 2.0")
            ],
            "google": [
                ("gemini-pro", "Gemini Pro"),
                ("gemini-pro-vision", "Gemini Pro Vision"),
                ("gemini-ultra", "Gemini Ultra")
            ],
            "mistral": [
                ("mistral-large-latest", "Mistral Large"),
                ("mistral-medium-latest", "Mistral Medium"),
                ("mistral-small-latest", "Mistral Small"),
                ("mixtral-8x7b", "Mixtral 8x7B"),
                ("mistral-7b", "Mistral 7B")
            ],
            "cohere": [
                ("command", "Command"),
                ("command-light", "Command Light"),
                ("command-nightly", "Command Nightly")
            ],
            "groq": [
                ("mixtral-8x7b-32768", "Mixtral 8x7B"),
                ("llama2-70b-4096", "Llama 2 70B"),
                ("gemma-7b-it", "Gemma 7B")
            ],
            "deepseek": [
                ("deepseek-chat", "DeepSeek Chat"),
                ("deepseek-coder", "DeepSeek Coder")
            ],
            "openrouter": [
                ("auto", "Auto (Best Available)"),
                ("anthropic/claude-3-opus", "Claude 3 Opus"),
                ("openai/gpt-4-turbo", "GPT-4 Turbo"),
                ("google/gemini-pro", "Gemini Pro")
            ],
            "huggingface": [
                ("HuggingFaceH4/zephyr-7b-beta", "Zephyr 7B"),
                ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral 7B Instruct"),
                ("meta-llama/Llama-2-7b-chat-hf", "Llama 2 7B Chat")
            ],
            "moonshot": [
                ("moonshot-v1-8k", "Moonshot v1 8K"),
                ("moonshot-v1-32k", "Moonshot v1 32K"),
                ("moonshot-v1-128k", "Moonshot v1 128K")
            ],
            "zai": [
                ("yi-large", "Yi Large"),
                ("yi-medium", "Yi Medium"),
                ("yi-34b", "Yi 34B")
            ],
            "ollama": [
                ("llama2", "Llama 2"),
                ("mistral", "Mistral"),
                ("codellama", "Code Llama"),
                ("phi", "Phi"),
                ("mixtral", "Mixtral"),
                ("gemma", "Gemma"),
                ("custom", "Custom Model")
            ],
            "llama_cpp": [
                ("default", "Default Model"),
                ("custom", "Custom Path")
            ],
            "koboldcpp": [
                ("default", "Default Model")
            ],
            "oobabooga": [
                ("default", "Default Model")
            ],
            "tabbyapi": [
                ("default", "Default Model")
            ],
            "vllm": [
                ("default", "Default Model"),
                ("custom", "Custom Model")
            ],
            "local-llm": [
                ("default", "Default Local Model")
            ],
            "custom_openai": [
                ("gpt-3.5-turbo", "GPT-3.5 Compatible"),
                ("gpt-4", "GPT-4 Compatible"),
                ("custom", "Custom Model")
            ]
        }
        
        # Get models for selected provider
        models = model_map.get(provider, [("default", "Default Model")])
        model_select.set_options(models)
        
        # Update app settings
        self.app.settings.provider = provider
    
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