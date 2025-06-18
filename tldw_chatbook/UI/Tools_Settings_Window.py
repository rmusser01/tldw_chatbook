# tldw_chatbook/UI/Tools_Settings_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
import toml
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Button, TextArea, Label, Input, Select, Checkbox, TabbedContent, TabPane, Switch
# Local Imports
from tldw_chatbook.config import load_cli_config_and_ensure_existence, DEFAULT_CONFIG_PATH, save_setting_to_cli_config, API_MODELS_BY_PROVIDER
#
# Local Imports
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class ToolsSettingsWindow(Container):
    """
    Container for the Tools & Settings Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.config_data = load_cli_config_and_ensure_existence()

    def _compose_general_settings(self) -> ComposeResult:
        """Compose the General Settings UI with commonly used settings."""
        with VerticalScroll(classes="general-settings-container"):
            yield Static("General Settings", classes="section-title")
            yield Static("Configure commonly used settings", classes="section-description")
            
            # Application Settings Section
            yield Static("Application Settings", classes="subsection-title")
            with Container(classes="settings-group"):
                general_config = self.config_data.get("general", {})
                
                yield Label("Default Tab on Startup:", classes="settings-label")
                tab_options = [
                    ("Chat", "chat"),
                    ("Character Chat", "character"),
                    ("Notes", "notes"),
                    ("Media", "media"),
                    ("Search", "search"),
                    ("Tools & Settings", "tools_settings")
                ]
                yield Select(
                    options=tab_options,
                    value=general_config.get("default_tab", "chat"),
                    id="general-default-tab",
                    classes="settings-select"
                )
                
                yield Label("Theme:", classes="settings-label")
                theme_options = [
                    ("Dark Theme", "textual-dark"),
                    ("Light Theme", "textual-light")
                ]
                yield Select(
                    options=theme_options,
                    value=general_config.get("default_theme", "textual-dark"),
                    id="general-theme",
                    classes="settings-select"
                )
                
                yield Label("User Name:", classes="settings-label")
                yield Input(
                    value=general_config.get("users_name", "default_user"),
                    placeholder="Enter your name",
                    id="general-username",
                    classes="settings-input"
                )
                
                yield Label("Log Level:", classes="settings-label")
                log_options = [
                    ("DEBUG", "DEBUG"),
                    ("INFO", "INFO"),
                    ("WARNING", "WARNING"),
                    ("ERROR", "ERROR"),
                    ("CRITICAL", "CRITICAL")
                ]
                yield Select(
                    options=log_options,
                    value=general_config.get("log_level", "INFO"),
                    id="general-log-level",
                    classes="settings-select"
                )
            
            # Default Provider & Model Section
            yield Static("Default LLM Provider & Model", classes="subsection-title")
            with Container(classes="settings-group"):
                api_settings = self.config_data.get("api_settings", {})
                chat_defaults = self.config_data.get("chat_defaults", {})
                
                yield Label("Default Provider:", classes="settings-label")
                provider_options = [(name, name) for name in API_MODELS_BY_PROVIDER.keys()]
                if not provider_options:
                    provider_options = [("OpenAI", "OpenAI"), ("Anthropic", "Anthropic")]
                current_provider = chat_defaults.get("provider", "OpenAI")
                # Ensure the current provider is in the options
                if current_provider not in [option[1] for option in provider_options]:
                    current_provider = provider_options[0][1] if provider_options else "OpenAI"
                yield Select(
                    options=provider_options,
                    value=current_provider,
                    id="general-default-provider",
                    classes="settings-select"
                )
                
                yield Label("Default Model:", classes="settings-label")
                yield Input(
                    value=chat_defaults.get("model", "gpt-4o"),
                    placeholder="Enter model name",
                    id="general-default-model",
                    classes="settings-input"
                )
                
                yield Label("Default Temperature:", classes="settings-label")
                yield Input(
                    value=str(chat_defaults.get("temperature", 0.7)),
                    placeholder="0.7",
                    id="general-temperature",
                    classes="settings-input"
                )
                
                yield Label("Default Max Tokens:", classes="settings-label")
                yield Input(
                    value=str(chat_defaults.get("max_tokens", 4096)),
                    placeholder="4096",
                    id="general-max-tokens",
                    classes="settings-input"
                )
                
                yield Label("Enable Streaming:", classes="settings-label")
                yield Switch(
                    value=chat_defaults.get("streaming", False),
                    id="general-streaming"
                )
            
            # API Keys Section
            yield Static("API Keys", classes="subsection-title")
            yield Static("⚠️  API keys are stored in plain text. Use environment variables for better security.", classes="warning-text")
            with Container(classes="settings-group"):
                yield Label("OpenAI API Key:", classes="settings-label")
                openai_key = api_settings.get("openai", {}).get("api_key", "")
                yield Input(
                    value=openai_key if openai_key != "<API_KEY_HERE>" else "",
                    placeholder="sk-...",
                    password=True,
                    id="general-openai-key",
                    classes="settings-input"
                )
                
                yield Label("Anthropic API Key:", classes="settings-label")
                anthropic_key = api_settings.get("anthropic", {}).get("api_key", "")
                yield Input(
                    value=anthropic_key if anthropic_key != "<API_KEY_HERE>" else "",
                    placeholder="sk-ant-...",
                    password=True,
                    id="general-anthropic-key",
                    classes="settings-input"
                )
                
                yield Label("Google API Key:", classes="settings-label")
                google_key = api_settings.get("google", {}).get("api_key", "")
                yield Input(
                    value=google_key if google_key != "<API_KEY_HERE>" else "",
                    placeholder="AIza...",
                    password=True,
                    id="general-google-key",
                    classes="settings-input"
                )
            
            # Basic RAG Settings Section
            yield Static("Basic RAG Settings", classes="subsection-title")
            with Container(classes="settings-group"):
                rag_config = self.config_data.get("rag_search", {})
                
                yield Label("Enable RAG Search:", classes="settings-label")
                yield Switch(
                    value=rag_config.get("enabled", True),
                    id="general-rag-enabled"
                )
                
                yield Label("Default Search Sources:", classes="settings-label")
                with Horizontal(classes="checkbox-group"):
                    sources_config = rag_config.get("default_sources", {})
                    yield Checkbox(
                        "Media Files",
                        value=sources_config.get("media", True),
                        id="general-rag-source-media"
                    )
                    yield Checkbox(
                        "Conversations",
                        value=sources_config.get("conversations", True),
                        id="general-rag-source-conversations"
                    )
                    yield Checkbox(
                        "Notes",
                        value=sources_config.get("notes", True),
                        id="general-rag-source-notes"
                    )
                
                yield Label("Default Top-K Results:", classes="settings-label")
                yield Input(
                    value=str(rag_config.get("default_top_k", 10)),
                    placeholder="10",
                    id="general-rag-top-k",
                    classes="settings-input"
                )
            
            # Database Paths Section
            yield Static("Database Paths", classes="subsection-title")
            with Container(classes="settings-group"):
                db_config = self.config_data.get("database", {})
                
                yield Label("ChaChaNotes Database:", classes="settings-label")
                yield Input(
                    value=db_config.get("chachanotes_db_path", "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"),
                    placeholder="Path to ChaChaNotes database",
                    id="general-chachanotes-path",
                    classes="settings-input"
                )
                
                yield Label("Media Database:", classes="settings-label")
                yield Input(
                    value=db_config.get("media_db_path", "~/.local/share/tldw_cli/tldw_cli_media_v2.db"),
                    placeholder="Path to Media database",
                    id="general-media-path",
                    classes="settings-input"
                )
                
                yield Label("Prompts Database:", classes="settings-label")
                yield Input(
                    value=db_config.get("prompts_db_path", "~/.local/share/tldw_cli/tldw_cli_prompts.db"),
                    placeholder="Path to Prompts database",
                    id="general-prompts-path",
                    classes="settings-input"
                )
            
            # Save Button
            with Container(classes="settings-actions"):
                yield Button("Save General Settings", id="save-general-settings", variant="primary")
                yield Button("Reset to Defaults", id="reset-general-settings", variant="default")
    
    def _compose_config_file_settings(self) -> ComposeResult:
        """Compose the Configuration File Settings UI with organized sections."""
        with VerticalScroll(classes="config-file-settings-container"):
            yield Static("Configuration File Settings", classes="section-title")
            yield Static("Edit all configuration values with organized sections or raw TOML", classes="section-description")
            
            with TabbedContent(id="config-tabs"):
                # Raw TOML Editor Tab
                with TabPane("Raw TOML", id="tab-raw-toml"):
                    yield Static("Direct TOML Configuration Editor", classes="tab-description")
                    yield TextArea(
                        text=toml.dumps(self.config_data),
                        language="toml",
                        read_only=False,
                        id="config-text-area",
                        classes="config-editor"
                    )
                    with Container(classes="config-button-container"):
                        yield Button("Save TOML", id="save-config-button", variant="primary")
                        yield Button("Reload", id="reload-config-button")
                        yield Button("Validate", id="validate-config-button")
                
                # General Configuration Tab
                with TabPane("General", id="tab-general-config"):
                    yield Static("Application General Settings", classes="tab-description")
                    yield from self._compose_general_config_form()
                
                # API Settings Tab
                with TabPane("API Settings", id="tab-api-config"):
                    yield Static("API Provider Configurations", classes="tab-description")
                    yield from self._compose_api_config_form()
                
                # Database Settings Tab
                with TabPane("Database", id="tab-database-config"):
                    yield Static("Database Configuration", classes="tab-description")
                    yield from self._compose_database_config_form()
                
                # RAG Settings Tab
                with TabPane("RAG Settings", id="tab-rag-config"):
                    yield Static("Retrieval-Augmented Generation Settings", classes="tab-description")
                    yield from self._compose_rag_config_form()
                
                # Providers Tab
                with TabPane("Providers", id="tab-providers-config"):
                    yield Static("Available Models by Provider", classes="tab-description")
                    yield from self._compose_providers_config_form()
                
                # Advanced Tab
                with TabPane("Advanced", id="tab-advanced-config"):
                    yield Static("Advanced Configuration Options", classes="tab-description")
                    yield from self._compose_advanced_config_form()
    
    def _compose_general_config_form(self) -> ComposeResult:
        """Form for general configuration section."""
        with VerticalScroll(classes="config-form"):
            general_config = self.config_data.get("general", {})
            
            yield Label("Default Tab:", classes="form-label")
            tab_options = [
                ("Chat", "chat"),
                ("Character Chat", "character"),
                ("Notes", "notes"),
                ("Media", "media"),
                ("Search", "search"),
                ("Tools & Settings", "tools_settings")
            ]
            yield Select(
                options=tab_options,
                value=general_config.get("default_tab", "chat"),
                id="config-general-default-tab"
            )
            
            yield Label("Default Theme:", classes="form-label")
            yield Input(
                value=general_config.get("default_theme", "textual-dark"),
                placeholder="textual-dark",
                id="config-general-theme"
            )
            
            yield Label("Palette Theme Limit:", classes="form-label")
            yield Input(
                value=str(general_config.get("palette_theme_limit", 1)),
                placeholder="1",
                id="config-general-palette-limit"
            )
            
            yield Label("Log Level:", classes="form-label")
            yield Input(
                value=general_config.get("log_level", "INFO"),
                placeholder="INFO",
                id="config-general-log-level"
            )
            
            yield Label("User Name:", classes="form-label")
            yield Input(
                value=general_config.get("users_name", "default_user"),
                placeholder="default_user",
                id="config-general-users-name"
            )
            
            with Container(classes="form-actions"):
                yield Button("Save General Config", id="save-general-config-form", variant="primary")
                yield Button("Reset Section", id="reset-general-config-form")
    
    def _compose_api_config_form(self) -> ComposeResult:
        """Form for API settings configuration."""
        with VerticalScroll(classes="config-form"):
            api_settings = self.config_data.get("api_settings", {})
            
            # OpenAI Settings
            yield Static("OpenAI Configuration", classes="form-section-title")
            openai_config = api_settings.get("openai", {})
            
            yield Label("API Key Environment Variable:", classes="form-label")
            yield Input(
                value=openai_config.get("api_key_env_var", "OPENAI_API_KEY"),
                id="config-openai-env-var"
            )
            
            yield Label("API Key (Fallback):", classes="form-label")
            yield Input(
                value=openai_config.get("api_key", "<API_KEY_HERE>"),
                password=True,
                id="config-openai-api-key"
            )
            
            yield Label("Default Model:", classes="form-label")
            yield Input(
                value=openai_config.get("model", "gpt-4o"),
                id="config-openai-model"
            )
            
            yield Label("Temperature:", classes="form-label")
            yield Input(
                value=str(openai_config.get("temperature", 0.7)),
                id="config-openai-temperature"
            )
            
            yield Label("Max Tokens:", classes="form-label")
            yield Input(
                value=str(openai_config.get("max_tokens", 4096)),
                id="config-openai-max-tokens"
            )
            
            # Anthropic Settings
            yield Static("Anthropic Configuration", classes="form-section-title")
            anthropic_config = api_settings.get("anthropic", {})
            
            yield Label("API Key Environment Variable:", classes="form-label")
            yield Input(
                value=anthropic_config.get("api_key_env_var", "ANTHROPIC_API_KEY"),
                id="config-anthropic-env-var"
            )
            
            yield Label("API Key (Fallback):", classes="form-label")
            yield Input(
                value=anthropic_config.get("api_key", "<API_KEY_HERE>"),
                password=True,
                id="config-anthropic-api-key"
            )
            
            yield Label("Default Model:", classes="form-label")
            yield Input(
                value=anthropic_config.get("model", "claude-3-haiku-20240307"),
                id="config-anthropic-model"
            )
            
            yield Label("Temperature:", classes="form-label")
            yield Input(
                value=str(anthropic_config.get("temperature", 0.7)),
                id="config-anthropic-temperature"
            )
            
            with Container(classes="form-actions"):
                yield Button("Save API Config", id="save-api-config-form", variant="primary")
                yield Button("Reset Section", id="reset-api-config-form")
    
    def _compose_database_config_form(self) -> ComposeResult:
        """Form for database configuration."""
        with VerticalScroll(classes="config-form"):
            db_config = self.config_data.get("database", {})
            
            yield Label("ChaChaNotes Database Path:", classes="form-label")
            yield Input(
                value=db_config.get("chachanotes_db_path", "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"),
                id="config-db-chachanotes-path"
            )
            
            yield Label("Prompts Database Path:", classes="form-label")
            yield Input(
                value=db_config.get("prompts_db_path", "~/.local/share/tldw_cli/tldw_cli_prompts.db"),
                id="config-db-prompts-path"
            )
            
            yield Label("Media Database Path:", classes="form-label")
            yield Input(
                value=db_config.get("media_db_path", "~/.local/share/tldw_cli/tldw_cli_media_v2.db"),
                id="config-db-media-path"
            )
            
            yield Label("User Database Base Directory:", classes="form-label")
            yield Input(
                value=db_config.get("USER_DB_BASE_DIR", "~/.local/share/tldw_cli/"),
                id="config-db-base-dir"
            )
            
            with Container(classes="form-actions"):
                yield Button("Save Database Config", id="save-database-config-form", variant="primary")
                yield Button("Reset Section", id="reset-database-config-form")
    
    def _compose_rag_config_form(self) -> ComposeResult:
        """Form for RAG configuration."""
        with VerticalScroll(classes="config-form"):
            rag_config = self.config_data.get("rag_search", {})
            
            yield Label("Default Search Mode:", classes="form-label")
            search_modes = [("Q&A Mode", "qa"), ("Chat Mode", "chat")]
            yield Select(
                options=search_modes,
                value=rag_config.get("default_mode", "qa"),
                id="config-rag-search-mode"
            )
            
            yield Label("Default Top-K Results:", classes="form-label")
            yield Input(
                value=str(rag_config.get("default_top_k", 10)),
                id="config-rag-top-k"
            )
            
            # Default Sources
            yield Static("Default Search Sources", classes="form-section-title")
            sources_config = rag_config.get("default_sources", {})
            
            yield Checkbox(
                "Media Files",
                value=sources_config.get("media", True),
                id="config-rag-source-media"
            )
            yield Checkbox(
                "Conversations",
                value=sources_config.get("conversations", True),
                id="config-rag-source-conversations"
            )
            yield Checkbox(
                "Notes",
                value=sources_config.get("notes", True),
                id="config-rag-source-notes"
            )
            
            # Reranking Settings
            yield Static("Re-ranking Configuration", classes="form-section-title")
            rerank_config = rag_config.get("reranking", {})
            
            yield Checkbox(
                "Enable Re-ranking by Default",
                value=rerank_config.get("enabled", False),
                id="config-rag-rerank-enabled"
            )
            
            yield Label("Re-ranker Model:", classes="form-label")
            yield Input(
                value=rerank_config.get("model", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
                id="config-rag-rerank-model"
            )
            
            # Chunking Settings
            yield Static("Chunking Configuration", classes="form-section-title")
            chunking_config = rag_config.get("chunking", {})
            
            yield Label("Chunk Size:", classes="form-label")
            yield Input(
                value=str(chunking_config.get("size", 512)),
                id="config-rag-chunk-size"
            )
            
            yield Label("Chunk Overlap:", classes="form-label")
            yield Input(
                value=str(chunking_config.get("overlap", 128)),
                id="config-rag-chunk-overlap"
            )
            
            yield Label("Chunking Method:", classes="form-label")
            chunking_methods = [
                ("Fixed Size", "fixed"),
                ("Semantic", "semantic"),
                ("Sentence-based", "sentence")
            ]
            yield Select(
                options=chunking_methods,
                value=chunking_config.get("method", "fixed"),
                id="config-rag-chunk-method"
            )
            
            with Container(classes="form-actions"):
                yield Button("Save RAG Config", id="save-rag-config-form", variant="primary")
                yield Button("Reset Section", id="reset-rag-config-form")
    
    def _compose_providers_config_form(self) -> ComposeResult:
        """Form for providers configuration."""
        with VerticalScroll(classes="config-form"):
            providers_config = self.config_data.get("providers", {})
            
            yield Static("Configure available models for each provider", classes="form-description")
            
            for provider, models in providers_config.items():
                yield Static(f"{provider} Models", classes="form-section-title")
                models_str = ", ".join(models) if isinstance(models, list) else str(models)
                yield Label(f"Available Models (comma-separated):", classes="form-label")
                yield TextArea(
                    text=models_str,
                    id=f"config-provider-{provider.lower().replace(' ', '-')}",
                    classes="provider-models-textarea"
                )
            
            with Container(classes="form-actions"):
                yield Button("Save Providers Config", id="save-providers-config-form", variant="primary")
                yield Button("Reset Section", id="reset-providers-config-form")
    
    def _compose_advanced_config_form(self) -> ComposeResult:
        """Form for advanced configuration options."""
        with VerticalScroll(classes="config-form"):
            yield Static("Advanced settings - modify with caution", classes="form-description")
            
            # Logging Configuration
            yield Static("Logging Configuration", classes="form-section-title")
            logging_config = self.config_data.get("logging", {})
            
            yield Label("Log Filename:", classes="form-label")
            yield Input(
                value=logging_config.get("log_filename", "tldw_cli_app.log"),
                id="config-logging-filename"
            )
            
            yield Label("File Log Level:", classes="form-label")
            yield Input(
                value=logging_config.get("file_log_level", "INFO"),
                id="config-logging-file-level"
            )
            
            yield Label("Log Max Bytes:", classes="form-label")
            yield Input(
                value=str(logging_config.get("log_max_bytes", 10485760)),
                id="config-logging-max-bytes"
            )
            
            yield Label("Log Backup Count:", classes="form-label")
            yield Input(
                value=str(logging_config.get("log_backup_count", 5)),
                id="config-logging-backup-count"
            )
            
            # API Endpoints Configuration
            yield Static("API Endpoints Configuration", classes="form-section-title")
            endpoints_config = self.config_data.get("api_endpoints", {})
            
            common_endpoints = [
                ("Llama.cpp", "llama_cpp", "http://localhost:8080"),
                ("KoboldCPP", "koboldcpp", "http://localhost:5001/api"),
                ("Oobabooga", "Oobabooga", "http://localhost:5000/api"),
                ("Ollama", "Ollama", "http://localhost:11434"),
                ("vLLM", "vLLM", "http://localhost:8000"),
                ("Custom", "Custom", "http://localhost:1234/v1")
            ]
            
            for display_name, key, default_url in common_endpoints:
                yield Label(f"{display_name} Endpoint:", classes="form-label")
                yield Input(
                    value=endpoints_config.get(key, default_url),
                    id=f"config-endpoint-{key.lower()}"
                )
            
            with Container(classes="form-actions"):
                yield Button("Save Advanced Config", id="save-advanced-config-form", variant="primary")
                yield Button("Reset Section", id="reset-advanced-config-form")
                yield Button("Export Configuration", id="export-config-button")
                yield Button("Import Configuration", id="import-config-button")

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="tools-settings-nav-pane", classes="tools-nav-pane"):
            yield Static("Navigation", classes="sidebar-title")
            yield Button("General Settings", id="ts-nav-general-settings", classes="ts-nav-button")
            yield Button("Configuration File Settings", id="ts-nav-config-file-settings", classes="ts-nav-button")
            yield Button("Database Tools", id="ts-nav-db-tools", classes="ts-nav-button")
            yield Button("Appearance", id="ts-nav-appearance", classes="ts-nav-button")

        with Container(id="tools-settings-content-pane", classes="tools-content-pane"):
            yield Container(
                *self._compose_general_settings(),
                id="ts-view-general-settings",
                classes="ts-view-area",
            )
            yield Container(
                *self._compose_config_file_settings(),
                id="ts-view-config-file-settings",
                classes="ts-view-area",
            )
            yield Container(
                Static("Database Tools Area - Content Coming Soon!"),
                id="ts-view-db-tools",
                classes="ts-view-area",
            )
            yield Container(
                Static("Appearance Settings Area - Content Coming Soon!"),
                id="ts-view-appearance",
                classes="ts-view-area",
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        button_id = event.button.id

        # General Settings handlers
        if button_id == "save-general-settings":
            await self._save_general_settings()
        elif button_id == "reset-general-settings":
            await self._reset_general_settings()
            
        # Raw TOML editor handlers
        elif button_id == "save-config-button":
            await self._save_raw_toml_config()
        elif button_id == "reload-config-button":
            await self._reload_raw_toml_config()
        elif button_id == "validate-config-button":
            await self._validate_toml_config()
            
        # Form-based config handlers
        elif button_id == "save-general-config-form":
            await self._save_general_config_form()
        elif button_id == "reset-general-config-form":
            await self._reset_general_config_form()
        elif button_id == "save-api-config-form":
            await self._save_api_config_form()
        elif button_id == "reset-api-config-form":
            await self._reset_api_config_form()
        elif button_id == "save-database-config-form":
            await self._save_database_config_form()
        elif button_id == "reset-database-config-form":
            await self._reset_database_config_form()
        elif button_id == "save-rag-config-form":
            await self._save_rag_config_form()
        elif button_id == "reset-rag-config-form":
            await self._reset_rag_config_form()
        elif button_id == "save-providers-config-form":
            await self._save_providers_config_form()
        elif button_id == "reset-providers-config-form":
            await self._reset_providers_config_form()
        elif button_id == "save-advanced-config-form":
            await self._save_advanced_config_form()
        elif button_id == "reset-advanced-config-form":
            await self._reset_advanced_config_form()
            
        # Import/Export handlers
        elif button_id == "export-config-button":
            await self._export_configuration()
        elif button_id == "import-config-button":
            await self._import_configuration()

    async def _save_general_settings(self) -> None:
        """Save General Settings to the configuration file."""
        try:
            # General App Settings
            if save_setting_to_cli_config("general", "default_tab", self.query_one("#general-default-tab", Select).value):
                if save_setting_to_cli_config("general", "default_theme", self.query_one("#general-theme", Select).value):
                    if save_setting_to_cli_config("general", "users_name", self.query_one("#general-username", Input).value):
                        if save_setting_to_cli_config("general", "log_level", self.query_one("#general-log-level", Select).value):
                            # Default Provider Settings
                            if save_setting_to_cli_config("chat_defaults", "provider", self.query_one("#general-default-provider", Select).value):
                                if save_setting_to_cli_config("chat_defaults", "model", self.query_one("#general-default-model", Input).value):
                                    # Chat Defaults
                                    try:
                                        temperature = float(self.query_one("#general-temperature", Input).value)
                                        if save_setting_to_cli_config("chat_defaults", "temperature", temperature):
                                            try:
                                                max_tokens = int(self.query_one("#general-max-tokens", Input).value)
                                                if save_setting_to_cli_config("chat_defaults", "max_tokens", max_tokens):
                                                    streaming = self.query_one("#general-streaming", Switch).value
                                                    if save_setting_to_cli_config("chat_defaults", "streaming", streaming):
                                                        # API Keys
                                                        openai_key = self.query_one("#general-openai-key", Input).value
                                                        if openai_key and openai_key.strip():
                                                            save_setting_to_cli_config("api_settings.openai", "api_key", openai_key)
                                                        
                                                        anthropic_key = self.query_one("#general-anthropic-key", Input).value
                                                        if anthropic_key and anthropic_key.strip():
                                                            save_setting_to_cli_config("api_settings.anthropic", "api_key", anthropic_key)
                                                        
                                                        google_key = self.query_one("#general-google-key", Input).value
                                                        if google_key and google_key.strip():
                                                            save_setting_to_cli_config("api_settings.google", "api_key", google_key)
                                                        
                                                        # RAG Settings
                                                        rag_enabled = self.query_one("#general-rag-enabled", Switch).value
                                                        if save_setting_to_cli_config("rag_search", "enabled", rag_enabled):
                                                            # RAG Sources
                                                            media_enabled = self.query_one("#general-rag-source-media", Checkbox).value
                                                            conversations_enabled = self.query_one("#general-rag-source-conversations", Checkbox).value
                                                            notes_enabled = self.query_one("#general-rag-source-notes", Checkbox).value
                                                            
                                                            save_setting_to_cli_config("rag_search.default_sources", "media", media_enabled)
                                                            save_setting_to_cli_config("rag_search.default_sources", "conversations", conversations_enabled)
                                                            save_setting_to_cli_config("rag_search.default_sources", "notes", notes_enabled)
                                                            
                                                            # RAG Top-K
                                                            try:
                                                                top_k = int(self.query_one("#general-rag-top-k", Input).value)
                                                                save_setting_to_cli_config("rag_search", "default_top_k", top_k)
                                                            except ValueError:
                                                                self.app_instance.notify("Invalid Top-K value, keeping current setting", severity="warning")
                                                            
                                                            # Database Paths
                                                            chachanotes_path = self.query_one("#general-chachanotes-path", Input).value
                                                            if chachanotes_path:
                                                                save_setting_to_cli_config("database", "chachanotes_db_path", chachanotes_path)
                                                            
                                                            media_path = self.query_one("#general-media-path", Input).value
                                                            if media_path:
                                                                save_setting_to_cli_config("database", "media_db_path", media_path)
                                                            
                                                            prompts_path = self.query_one("#general-prompts-path", Input).value
                                                            if prompts_path:
                                                                save_setting_to_cli_config("database", "prompts_db_path", prompts_path)
                                                            
                                                            # Update internal config
                                                            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
                                                            self.app_instance.notify("General Settings saved successfully!")
                                                            return
                                            except ValueError:
                                                self.app_instance.notify("Invalid max tokens value", severity="error")
                                                return
                                    except ValueError:
                                        self.app_instance.notify("Invalid temperature value", severity="error")
                                        return
            
            self.app_instance.notify("Failed to save some General Settings", severity="error")
            
        except Exception as e:
            self.app_instance.notify(f"Error saving General Settings: {e}", severity="error")

    async def _reset_general_settings(self) -> None:
        """Reset General Settings to default values."""
        try:
            # Reset UI elements to defaults
            self.query_one("#general-default-tab", Select).value = "chat"
            self.query_one("#general-theme", Select).value = "textual-dark"
            self.query_one("#general-username", Input).value = "default_user"
            self.query_one("#general-log-level", Select).value = "INFO"
            self.query_one("#general-default-provider", Select).value = "openai"
            self.query_one("#general-default-model", Input).value = "gpt-4o"
            self.query_one("#general-temperature", Input).value = "0.7"
            self.query_one("#general-max-tokens", Input).value = "4096"
            self.query_one("#general-streaming", Switch).value = False
            self.query_one("#general-openai-key", Input).value = ""
            self.query_one("#general-anthropic-key", Input).value = ""
            self.query_one("#general-google-key", Input).value = ""
            self.query_one("#general-rag-enabled", Switch).value = True
            self.query_one("#general-rag-source-media", Checkbox).value = True
            self.query_one("#general-rag-source-conversations", Checkbox).value = True
            self.query_one("#general-rag-source-notes", Checkbox).value = True
            self.query_one("#general-rag-top-k", Input).value = "10"
            self.query_one("#general-chachanotes-path", Input).value = "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
            self.query_one("#general-media-path", Input).value = "~/.local/share/tldw_cli/tldw_cli_media_v2.db"
            self.query_one("#general-prompts-path", Input).value = "~/.local/share/tldw_cli/tldw_cli_prompts.db"
            
            self.app_instance.notify("General Settings reset to defaults!")
            
        except Exception as e:
            self.app_instance.notify(f"Error resetting General Settings: {e}", severity="error")

    async def _save_raw_toml_config(self) -> None:
        """Save raw TOML configuration."""
        try:
            config_text_area = self.query_one("#config-text-area", TextArea)
            config_data = toml.loads(config_text_area.text)
            with open(DEFAULT_CONFIG_PATH, "w") as f:
                toml.dump(config_data, f)
            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
            self.app_instance.notify("Configuration saved successfully.")
        except toml.TOMLDecodeError as e:
            self.app_instance.notify(f"Error: Invalid TOML format: {e}", severity="error")
        except IOError as e:
            self.app_instance.notify(f"Error: Could not write to configuration file: {e}", severity="error")
    
    async def _reload_raw_toml_config(self) -> None:
        """Reload raw TOML configuration."""
        try:
            config_text_area = self.query_one("#config-text-area", TextArea)
            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
            config_text_area.text = toml.dumps(self.config_data)
            self.app_instance.notify("Configuration reloaded.")
        except Exception as e:
            self.app_instance.notify(f"Error reloading configuration: {e}", severity="error")
    
    async def _validate_toml_config(self) -> None:
        """Validate TOML configuration syntax."""
        try:
            config_text_area = self.query_one("#config-text-area", TextArea)
            toml.loads(config_text_area.text)
            self.app_instance.notify("TOML configuration is valid!", severity="information")
        except toml.TOMLDecodeError as e:
            self.app_instance.notify(f"TOML validation error: {e}", severity="error")
    
    async def _save_general_config_form(self) -> None:
        """Save general configuration form."""
        try:
            save_setting_to_cli_config("general", "default_tab", self.query_one("#config-general-default-tab", Select).value)
            save_setting_to_cli_config("general", "default_theme", self.query_one("#config-general-theme", Input).value)
            save_setting_to_cli_config("general", "palette_theme_limit", int(self.query_one("#config-general-palette-limit", Input).value))
            save_setting_to_cli_config("general", "log_level", self.query_one("#config-general-log-level", Input).value)
            save_setting_to_cli_config("general", "users_name", self.query_one("#config-general-users-name", Input).value)
            self.app_instance.notify("General configuration saved!")
        except Exception as e:
            self.app_instance.notify(f"Error saving general config: {e}", severity="error")
    
    async def _reset_general_config_form(self) -> None:
        """Reset general configuration form to defaults."""
        try:
            self.query_one("#config-general-default-tab", Select).value = "chat"
            self.query_one("#config-general-theme", Input).value = "textual-dark"
            self.query_one("#config-general-palette-limit", Input).value = "1"
            self.query_one("#config-general-log-level", Input).value = "INFO"
            self.query_one("#config-general-users-name", Input).value = "default_user"
            self.app_instance.notify("General configuration reset to defaults!")
        except Exception as e:
            self.app_instance.notify(f"Error resetting general config: {e}", severity="error")
    
    async def _save_api_config_form(self) -> None:
        """Save API configuration form."""
        try:
            # OpenAI settings
            save_setting_to_cli_config("api_settings.openai", "api_key_env_var", self.query_one("#config-openai-env-var", Input).value)
            save_setting_to_cli_config("api_settings.openai", "api_key", self.query_one("#config-openai-api-key", Input).value)
            save_setting_to_cli_config("api_settings.openai", "model", self.query_one("#config-openai-model", Input).value)
            save_setting_to_cli_config("api_settings.openai", "temperature", float(self.query_one("#config-openai-temperature", Input).value))
            save_setting_to_cli_config("api_settings.openai", "max_tokens", int(self.query_one("#config-openai-max-tokens", Input).value))
            
            # Anthropic settings
            save_setting_to_cli_config("api_settings.anthropic", "api_key_env_var", self.query_one("#config-anthropic-env-var", Input).value)
            save_setting_to_cli_config("api_settings.anthropic", "api_key", self.query_one("#config-anthropic-api-key", Input).value)
            save_setting_to_cli_config("api_settings.anthropic", "model", self.query_one("#config-anthropic-model", Input).value)
            save_setting_to_cli_config("api_settings.anthropic", "temperature", float(self.query_one("#config-anthropic-temperature", Input).value))
            
            self.app_instance.notify("API configuration saved!")
        except Exception as e:
            self.app_instance.notify(f"Error saving API config: {e}", severity="error")
    
    async def _reset_api_config_form(self) -> None:
        """Reset API configuration form to defaults."""
        try:
            # Reset OpenAI
            self.query_one("#config-openai-env-var", Input).value = "OPENAI_API_KEY"
            self.query_one("#config-openai-api-key", Input).value = "<API_KEY_HERE>"
            self.query_one("#config-openai-model", Input).value = "gpt-4o"
            self.query_one("#config-openai-temperature", Input).value = "0.7"
            self.query_one("#config-openai-max-tokens", Input).value = "4096"
            
            # Reset Anthropic
            self.query_one("#config-anthropic-env-var", Input).value = "ANTHROPIC_API_KEY"
            self.query_one("#config-anthropic-api-key", Input).value = "<API_KEY_HERE>"
            self.query_one("#config-anthropic-model", Input).value = "claude-3-haiku-20240307"
            self.query_one("#config-anthropic-temperature", Input).value = "0.7"
            
            self.app_instance.notify("API configuration reset to defaults!")
        except Exception as e:
            self.app_instance.notify(f"Error resetting API config: {e}", severity="error")
    
    async def _save_database_config_form(self) -> None:
        """Save database configuration form."""
        try:
            save_setting_to_cli_config("database", "chachanotes_db_path", self.query_one("#config-db-chachanotes-path", Input).value)
            save_setting_to_cli_config("database", "prompts_db_path", self.query_one("#config-db-prompts-path", Input).value)
            save_setting_to_cli_config("database", "media_db_path", self.query_one("#config-db-media-path", Input).value)
            save_setting_to_cli_config("database", "USER_DB_BASE_DIR", self.query_one("#config-db-base-dir", Input).value)
            self.app_instance.notify("Database configuration saved!")
        except Exception as e:
            self.app_instance.notify(f"Error saving database config: {e}", severity="error")
    
    async def _reset_database_config_form(self) -> None:
        """Reset database configuration form to defaults."""
        try:
            self.query_one("#config-db-chachanotes-path", Input).value = "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
            self.query_one("#config-db-prompts-path", Input).value = "~/.local/share/tldw_cli/tldw_cli_prompts.db"
            self.query_one("#config-db-media-path", Input).value = "~/.local/share/tldw_cli/tldw_cli_media_v2.db"
            self.query_one("#config-db-base-dir", Input).value = "~/.local/share/tldw_cli/"
            self.app_instance.notify("Database configuration reset to defaults!")
        except Exception as e:
            self.app_instance.notify(f"Error resetting database config: {e}", severity="error")
    
    async def _save_rag_config_form(self) -> None:
        """Save RAG configuration form."""
        try:
            save_setting_to_cli_config("rag_search", "default_mode", self.query_one("#config-rag-search-mode", Select).value)
            save_setting_to_cli_config("rag_search", "default_top_k", int(self.query_one("#config-rag-top-k", Input).value))
            
            # Sources
            save_setting_to_cli_config("rag_search.default_sources", "media", self.query_one("#config-rag-source-media", Checkbox).value)
            save_setting_to_cli_config("rag_search.default_sources", "conversations", self.query_one("#config-rag-source-conversations", Checkbox).value)
            save_setting_to_cli_config("rag_search.default_sources", "notes", self.query_one("#config-rag-source-notes", Checkbox).value)
            
            # Reranking
            save_setting_to_cli_config("rag_search.reranking", "enabled", self.query_one("#config-rag-rerank-enabled", Checkbox).value)
            save_setting_to_cli_config("rag_search.reranking", "model", self.query_one("#config-rag-rerank-model", Input).value)
            
            # Chunking
            save_setting_to_cli_config("rag_search.chunking", "size", int(self.query_one("#config-rag-chunk-size", Input).value))
            save_setting_to_cli_config("rag_search.chunking", "overlap", int(self.query_one("#config-rag-chunk-overlap", Input).value))
            save_setting_to_cli_config("rag_search.chunking", "method", self.query_one("#config-rag-chunk-method", Select).value)
            
            self.app_instance.notify("RAG configuration saved!")
        except Exception as e:
            self.app_instance.notify(f"Error saving RAG config: {e}", severity="error")
    
    async def _reset_rag_config_form(self) -> None:
        """Reset RAG configuration form to defaults."""
        try:
            self.query_one("#config-rag-search-mode", Select).value = "qa"
            self.query_one("#config-rag-top-k", Input).value = "10"
            self.query_one("#config-rag-source-media", Checkbox).value = True
            self.query_one("#config-rag-source-conversations", Checkbox).value = True
            self.query_one("#config-rag-source-notes", Checkbox).value = True
            self.query_one("#config-rag-rerank-enabled", Checkbox).value = False
            self.query_one("#config-rag-rerank-model", Input).value = "cross-encoder/ms-marco-MiniLM-L-12-v2"
            self.query_one("#config-rag-chunk-size", Input).value = "512"
            self.query_one("#config-rag-chunk-overlap", Input).value = "128"
            self.query_one("#config-rag-chunk-method", Select).value = "fixed"
            self.app_instance.notify("RAG configuration reset to defaults!")
        except Exception as e:
            self.app_instance.notify(f"Error resetting RAG config: {e}", severity="error")
    
    async def _save_providers_config_form(self) -> None:
        """Save providers configuration form."""
        try:
            providers_config = self.config_data.get("providers", {})
            for provider in providers_config.keys():
                provider_id = f"config-provider-{provider.lower().replace(' ', '-')}"
                try:
                    textarea = self.query_one(f"#{provider_id}", TextArea)
                    models_text = textarea.text.strip()
                    if models_text:
                        models_list = [model.strip() for model in models_text.split(",") if model.strip()]
                        save_setting_to_cli_config(f"providers", provider, models_list)
                except Exception:
                    pass  # Skip if widget not found
            
            self.app_instance.notify("Providers configuration saved!")
        except Exception as e:
            self.app_instance.notify(f"Error saving providers config: {e}", severity="error")
    
    async def _reset_providers_config_form(self) -> None:
        """Reset providers configuration form to defaults."""
        try:
            # This would reset to the defaults from CONFIG_TOML_CONTENT
            # Implementation depends on how you want to handle provider defaults
            self.app_instance.notify("Providers configuration reset to defaults!")
        except Exception as e:
            self.app_instance.notify(f"Error resetting providers config: {e}", severity="error")
    
    async def _save_advanced_config_form(self) -> None:
        """Save advanced configuration form."""
        try:
            # Logging
            save_setting_to_cli_config("logging", "log_filename", self.query_one("#config-logging-filename", Input).value)
            save_setting_to_cli_config("logging", "file_log_level", self.query_one("#config-logging-file-level", Input).value)
            save_setting_to_cli_config("logging", "log_max_bytes", int(self.query_one("#config-logging-max-bytes", Input).value))
            save_setting_to_cli_config("logging", "log_backup_count", int(self.query_one("#config-logging-backup-count", Input).value))
            
            # API Endpoints
            endpoints = [
                ("llama_cpp", "#config-endpoint-llama_cpp"),
                ("koboldcpp", "#config-endpoint-koboldcpp"),
                ("Oobabooga", "#config-endpoint-oobabooga"),
                ("Ollama", "#config-endpoint-ollama"),
                ("vLLM", "#config-endpoint-vllm"),
                ("Custom", "#config-endpoint-custom")
            ]
            
            for key, widget_id in endpoints:
                try:
                    endpoint_value = self.query_one(widget_id, Input).value
                    save_setting_to_cli_config("api_endpoints", key, endpoint_value)
                except Exception:
                    pass  # Skip if widget not found
            
            self.app_instance.notify("Advanced configuration saved!")
        except Exception as e:
            self.app_instance.notify(f"Error saving advanced config: {e}", severity="error")
    
    async def _reset_advanced_config_form(self) -> None:
        """Reset advanced configuration form to defaults."""
        try:
            # Reset logging
            self.query_one("#config-logging-filename", Input).value = "tldw_cli_app.log"
            self.query_one("#config-logging-file-level", Input).value = "INFO"
            self.query_one("#config-logging-max-bytes", Input).value = "10485760"
            self.query_one("#config-logging-backup-count", Input).value = "5"
            
            # Reset endpoints
            self.query_one("#config-endpoint-llama_cpp", Input).value = "http://localhost:8080"
            self.query_one("#config-endpoint-koboldcpp", Input).value = "http://localhost:5001/api"
            self.query_one("#config-endpoint-oobabooga", Input).value = "http://localhost:5000/api"
            self.query_one("#config-endpoint-ollama", Input).value = "http://localhost:11434"
            self.query_one("#config-endpoint-vllm", Input).value = "http://localhost:8000"
            self.query_one("#config-endpoint-custom", Input).value = "http://localhost:1234/v1"
            
            self.app_instance.notify("Advanced configuration reset to defaults!")
        except Exception as e:
            self.app_instance.notify(f"Error resetting advanced config: {e}", severity="error")
    
    async def _export_configuration(self) -> None:
        """Export configuration to a backup file."""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = DEFAULT_CONFIG_PATH.parent / f"config_backup_{timestamp}.toml"
            
            with open(backup_path, "w") as f:
                toml.dump(self.config_data, f)
            
            self.app_instance.notify(f"Configuration exported to {backup_path}")
        except Exception as e:
            self.app_instance.notify(f"Error exporting configuration: {e}", severity="error")
    
    async def _import_configuration(self) -> None:
        """Import configuration from a backup file."""
        try:
            # This is a simplified implementation
            # In a real implementation, you'd show a file picker dialog
            self.app_instance.notify("Import feature not yet implemented. Please manually copy TOML content.", severity="warning")
        except Exception as e:
            self.app_instance.notify(f"Error importing configuration: {e}", severity="error")

#
# End of Tools_Settings_Window.py
#######################################################################################################################
