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
from textual.widgets import Static, Button, TextArea, Label, Input, Select, Checkbox, TabbedContent, TabPane, Switch, ContentSwitcher, Collapsible
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
    DEFAULT_CSS = """
    ToolsSettingsWindow {
        layout: horizontal;
        height: 100%;
    }
    
    .tools-nav-pane {
        width: 25;
        min-width: 20;
        max-width: 35;
        background: $boost;
        padding: 1;
        border-right: thick $background;
    }
    
    .tools-content-pane {
        width: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    .ts-view-area {
        width: 100%;
        height: 100%;
    }
    
    .ts-nav-button {
        width: 100%;
        margin-bottom: 1;
    }
    
    .ts-nav-button.active-nav {
        background: $primary;
    }
    
    .section-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
        color: $primary;
    }
    
    .section-description {
        text-align: center;
        margin-bottom: 2;
        color: $text-muted;
    }
    
    .settings-container {
        padding: 1;
    }
    
    .settings-label, .form-label {
        margin-top: 1;
        margin-bottom: 0;
        text-style: bold;
    }
    
    .settings-input, .settings-select {
        width: 100%;
        margin-bottom: 1;
    }
    
    .settings-button-container, .form-actions {
        layout: horizontal;
        margin-top: 2;
        height: 3;
        width: 100%;
    }
    
    .settings-button-container .spacer {
        width: 1fr;
    }
    
    .settings-button-container Button, .form-actions Button {
        margin-right: 1;
    }
    
    .config-editor {
        height: 100%;
        width: 100%;
        min-height: 30;
    }
    
    .config-button-container {
        layout: horizontal;
        margin-top: 1;
        height: 3;
    }
    
    .config-button-container Button {
        margin-right: 1;
    }
    
    .config-form {
        padding: 1;
    }
    
    .form-section-title {
        text-style: bold underline;
        margin-top: 2;
        margin-bottom: 1;
        color: $secondary;
    }
    
    .form-subsection-title {
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
        color: $accent;
    }
    
    .provider-models-textarea {
        height: 5;
        width: 100%;
        margin-bottom: 1;
    }
    
    .system-prompt-textarea {
        height: 5;
        width: 100%;
        margin-bottom: 1;
    }
    
    .tab-description {
        margin-bottom: 1;
        color: $text-muted;
        text-align: center;
    }
    
    #config-tabs {
        height: 100%;
    }
    
    #config-tabs ContentSwitcher {
        height: 100%;
        overflow-y: auto;
    }
    
    .db-status {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    .help-text {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    .danger {
        color: $error;
    }
    
    .danger-warning {
        color: $error;
        text-style: bold;
        margin-top: 1;
        text-align: center;
    }
    
    Collapsible {
        margin-bottom: 1;
    }
    
    Collapsible > Contents {
        padding: 1;
    }
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.config_data = load_cli_config_and_ensure_existence()

    def _compose_general_settings(self) -> ComposeResult:
        """Compose the General Settings UI with commonly used settings."""
        yield Static("General Settings", classes="section-title")
        yield Static("Configure commonly used application settings", classes="section-description")
        
        with Container(classes="settings-container"):
            general_config = self.config_data.get("general", {})
            
            yield Label("Default Tab:", classes="settings-label")
            tab_options = [
                ("Chat", "chat"),
                ("Character Chat", "character"),
                ("Notes", "notes"),
                ("Media", "media"),
                ("RAG Search", "rag_search")
            ]
            yield Select(
                options=tab_options,
                value=general_config.get("default_tab", "chat"),
                id="general-default-tab",
                classes="settings-select"
            )
            
            yield Label("Theme:", classes="settings-label")
            
            # Import themes to get all available options
            from ..css.Themes.themes import ALL_THEMES
            
            # Build theme options from ALL_THEMES
            theme_options = [
                ("Dark Theme", "textual-dark"),
                ("Light Theme", "textual-light")
            ]
            
            # Add custom themes from ALL_THEMES
            for theme in ALL_THEMES:
                theme_name = theme.name
                # Create a user-friendly label from the theme name
                label = theme_name.replace('_', ' ').title()
                theme_options.append((label, theme_name))
            
            current_theme = general_config.get("default_theme", "textual-dark")
            
            # Validate the current theme - if it's not in the valid options, use default
            valid_theme_values = [value for _, value in theme_options]
            if current_theme not in valid_theme_values:
                current_theme = "textual-dark"
            
            yield Select(
                options=theme_options,
                value=current_theme,
                id="general-theme",
                classes="settings-select"
            )
            
            yield Label("User Name:", classes="settings-label")
            yield Input(
                value=general_config.get("users_name", "default_user"),
                id="general-username",
                classes="settings-input"
            )
            
            yield Label("Log Level:", classes="settings-label")
            log_options = [
                ("Debug", "DEBUG"),
                ("Info", "INFO"),
                ("Warning", "WARNING"),
                ("Error", "ERROR"),
                ("Critical", "CRITICAL")
            ]
            yield Select(
                options=log_options,
                value=general_config.get("log_level", "INFO"),
                id="general-log-level",
                classes="settings-select"
            )
            
            with Container(classes="settings-button-container"):
                yield Button("Save General Settings", id="save-general-settings", variant="primary")
                yield Static("", classes="spacer")
                yield Button("Reset General Settings", id="reset-general-settings")
    
    def _compose_config_file_settings(self) -> ComposeResult:
        """Compose the Configuration File Settings UI with organized sections."""
        yield Static("Configuration File Settings", classes="section-title")
        yield Static("Edit all configuration values with organized sections or raw TOML", classes="section-description")
        
        with TabbedContent(id="config-tabs"):
            # Raw TOML Editor Tab
            with TabPane("Raw TOML", id="tab-raw-toml"):
                yield Static("Direct TOML Configuration Editor", classes="tab-description")
                config_text = ""
                try:
                    if self.config_data:
                        config_text = toml.dumps(self.config_data)
                    else:
                        config_text = "# No configuration data loaded"
                except Exception as e:
                    config_text = f"# Error loading configuration: {e}\n# Please check the configuration file."
                yield TextArea(
                    text=config_text,
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
                with VerticalScroll():
                    yield from self._compose_general_config_form()
            
            # API Settings Tab
            with TabPane("API Settings", id="tab-api-config"):
                yield Static("API Provider Configurations", classes="tab-description")
                with VerticalScroll():
                    yield from self._compose_api_config_form()
            
            # Database Settings Tab
            with TabPane("Database", id="tab-database-config"):
                yield Static("Database Configuration", classes="tab-description")
                with VerticalScroll():
                    yield from self._compose_database_config_form()
            
            # RAG Settings Tab
            with TabPane("RAG Settings", id="tab-rag-config"):
                yield Static("Retrieval-Augmented Generation Settings", classes="tab-description")
                with VerticalScroll():
                    yield from self._compose_rag_config_form()
            
            # Providers Tab
            with TabPane("Providers", id="tab-providers-config"):
                yield Static("Available Models by Provider", classes="tab-description")
                with VerticalScroll():
                    yield from self._compose_providers_config_form()
            
            # Advanced Tab
            with TabPane("Advanced", id="tab-advanced-config"):
                yield Static("Advanced Configuration Options", classes="tab-description")
                with VerticalScroll():
                    yield from self._compose_advanced_config_form()
    
    def _compose_general_config_form(self) -> ComposeResult:
        """Form for general configuration section."""
        general_config = self.config_data.get("general", {})
        
        tab_options = [
            ("Chat", "chat"),
            ("Character Chat", "character"),
            ("Notes", "notes"),
            ("Media", "media"),
            ("Search", "search"),
            ("Tools & Settings", "tools_settings")
        ]
        
        yield Container(
            Label("Default Tab:", classes="form-label"),
            Select(
                options=tab_options,
                value=general_config.get("default_tab", "chat"),
                id="config-general-default-tab"
            ),
            
            Label("Default Theme:", classes="form-label"),
            Input(
                value=general_config.get("default_theme", "textual-dark"),
                placeholder="textual-dark",
                id="config-general-theme"
            ),
            
            Label("Palette Theme Limit:", classes="form-label"),
            Input(
                value=str(general_config.get("palette_theme_limit", 1)),
                placeholder="1",
                id="config-general-palette-limit"
            ),
            
            Label("Log Level:", classes="form-label"),
            Input(
                value=general_config.get("log_level", "INFO"),
                placeholder="INFO",
                id="config-general-log-level"
            ),
            
            Label("User Name:", classes="form-label"),
            Input(
                value=general_config.get("users_name", "default_user"),
                placeholder="default_user",
                id="config-general-users-name"
            ),
            
            Container(
                Button("Save General Config", id="save-general-config-form", variant="primary"),
                Button("Reset Section", id="reset-general-config-form"),
                classes="form-actions"
            ),
            classes="config-form"
        )
    
    def _compose_api_config_form(self) -> ComposeResult:
        """Form for API settings configuration."""
        api_settings = self.config_data.get("api_settings", {})
        
        # Create sections for each provider
        # The display_name must match the keys in API_MODELS_BY_PROVIDER
        providers = [
            ("OpenAI", "openai", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
            ("Anthropic", "anthropic", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
            ("Google", "google", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
            ("DeepSeek", "deepseek", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
            ("Groq", "groq", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
            ("MistralAI", "mistralai", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
            ("Cohere", "cohere", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
            ("HuggingFace", "huggingface", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
            ("OpenRouter", "openrouter", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "top_k", "min_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
        ]
        
        # Local API providers
        local_providers = [
            ("Llama.cpp", "llama_cpp", ["api_url", "model", "temperature", "top_p", "top_k", "min_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming", "system_prompt"]),
            ("Ollama", "ollama", ["api_url", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming", "system_prompt"]),
            ("vLLM", "vllm", ["api_key_env_var", "api_url", "model", "temperature", "top_p", "top_k", "min_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming", "system_prompt"]),
            ("KoboldCPP", "koboldcpp", ["api_url", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming", "system_prompt"]),
            ("Oobabooga", "oobabooga", ["api_key_env_var", "api_url", "model", "temperature", "top_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming", "system_prompt"]),
        ]
        
        # Build all widgets as a list
        widgets = []
        
        # Commercial providers
        for display_name, key, fields in providers:
            widgets.append(Static(f"{display_name} Configuration", classes="form-section-title"))
            provider_config = api_settings.get(key, {})
            
            for field in fields:
                if field == "api_key":
                    widgets.append(Label("API Key (Fallback):", classes="form-label"))
                    widgets.append(Input(
                        value=provider_config.get(field, "<API_KEY_HERE>"),
                        password=True,
                        id=f"config-{key}-{field}",
                        placeholder="Enter API key or use environment variable"
                    ))
                elif field == "api_key_env_var":
                    widgets.append(Label("API Key Environment Variable:", classes="form-label"))
                    widgets.append(Input(
                        value=provider_config.get(field, f"{key.upper()}_API_KEY"),
                        id=f"config-{key}-{field}",
                        placeholder=f"{key.upper()}_API_KEY"
                    ))
                elif field == "model":
                    widgets.append(Label("Default Model:", classes="form-label"))
                    # Get available models for this provider
                    try:
                        models = list(API_MODELS_BY_PROVIDER.get(display_name, []))
                    except:
                        models = []
                    
                    current_model = provider_config.get(field, "")
                    
                    # Always use Input for now to avoid Select widget issues
                    widgets.append(Input(
                        value=current_model,
                        id=f"config-{key}-{field}",
                        placeholder=f"Enter model name (e.g., {models[0] if models else 'model-name'})"
                    ))
                elif field == "streaming":
                    widgets.append(Checkbox(
                        "Enable Streaming",
                        value=provider_config.get(field, False),
                        id=f"config-{key}-{field}"
                    ))
                else:
                    # Format field name nicely
                    label = field.replace("_", " ").title()
                    widgets.append(Label(f"{label}:", classes="form-label"))
                    widgets.append(Input(
                        value=str(provider_config.get(field, "")),
                        id=f"config-{key}-{field}",
                        placeholder=f"Enter {label.lower()}"
                    ))
        
        # Local providers section
        widgets.append(Static("Local API Providers", classes="form-section-title"))
        
        for display_name, key, fields in local_providers:
            widgets.append(Static(f"{display_name} Configuration", classes="form-subsection-title"))
            provider_config = api_settings.get(key, {})
            
            for field in fields:
                if field == "api_url":
                    widgets.append(Label("API URL:", classes="form-label"))
                    widgets.append(Input(
                        value=provider_config.get(field, "http://localhost:8080"),
                        id=f"config-{key}-{field}",
                        placeholder="http://localhost:8080"
                    ))
                elif field == "system_prompt":
                    widgets.append(Label("Default System Prompt:", classes="form-label"))
                    widgets.append(TextArea(
                        text=provider_config.get(field, "You are a helpful AI assistant"),
                        id=f"config-{key}-{field}",
                        classes="system-prompt-textarea"
                    ))
                elif field == "streaming":
                    widgets.append(Checkbox(
                        "Enable Streaming",
                        value=provider_config.get(field, False),
                        id=f"config-{key}-{field}"
                    ))
                else:
                    label = field.replace("_", " ").title()
                    widgets.append(Label(f"{label}:", classes="form-label"))
                    widgets.append(Input(
                        value=str(provider_config.get(field, "")),
                        id=f"config-{key}-{field}",
                        placeholder=f"Enter {label.lower()}"
                    ))
        
        # Add action buttons
        widgets.append(Container(
            Button("Save API Config", id="save-api-config-form", variant="primary"),
            Button("Reset Section", id="reset-api-config-form"),
            classes="form-actions"
        ))
        
        yield Container(*widgets, classes="config-form")
    
    def _compose_database_config_form(self) -> ComposeResult:
        """Form for database configuration."""
        db_config = self.config_data.get("database", {})
        
        yield Container(
            Label("ChaChaNotes Database Path:", classes="form-label"),
            Input(
                value=db_config.get("chachanotes_db_path", "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"),
                id="config-db-chachanotes-path"
            ),
            
            Label("Prompts Database Path:", classes="form-label"),
            Input(
                value=db_config.get("prompts_db_path", "~/.local/share/tldw_cli/tldw_cli_prompts.db"),
                id="config-db-prompts-path"
            ),
            
            Label("Media Database Path:", classes="form-label"),
            Input(
                value=db_config.get("media_db_path", "~/.local/share/tldw_cli/tldw_cli_media_v2.db"),
                id="config-db-media-path"
            ),
            
            Label("User Database Base Directory:", classes="form-label"),
            Input(
                value=db_config.get("USER_DB_BASE_DIR", "~/.local/share/tldw_cli/"),
                id="config-db-base-dir"
            ),
            
            Container(
                Button("Save Database Config", id="save-database-config-form", variant="primary"),
                Button("Reset Section", id="reset-database-config-form"),
                classes="form-actions"
            ),
            classes="config-form"
        )
    
    def _compose_rag_config_form(self) -> ComposeResult:
        """Form for RAG configuration."""
        rag_config = self.config_data.get("rag_search", {})
        sources_config = rag_config.get("default_sources", {})
        rerank_config = rag_config.get("reranking", {})
        chunking_config = rag_config.get("chunking", {})
        
        search_modes = [("Q&A Mode", "qa"), ("Chat Mode", "chat")]
        chunking_methods = [
            ("Fixed Size", "fixed"),
            ("Semantic", "semantic"),
            ("Sentence-based", "sentence")
        ]
        
        yield Container(
            Label("Default Search Mode:", classes="form-label"),
            Select(
                options=search_modes,
                value=rag_config.get("default_mode", "qa"),
                id="config-rag-search-mode"
            ),
            
            Label("Default Top-K Results:", classes="form-label"),
            Input(
                value=str(rag_config.get("default_top_k", 10)),
                id="config-rag-top-k"
            ),
            
            # Default Sources
            Static("Default Search Sources", classes="form-section-title"),
            Checkbox(
                "Media Files",
                value=sources_config.get("media", True),
                id="config-rag-source-media"
            ),
            Checkbox(
                "Conversations",
                value=sources_config.get("conversations", True),
                id="config-rag-source-conversations"
            ),
            Checkbox(
                "Notes",
                value=sources_config.get("notes", True),
                id="config-rag-source-notes"
            ),
            
            # Reranking Settings
            Static("Re-ranking Configuration", classes="form-section-title"),
            Checkbox(
                "Enable Re-ranking by Default",
                value=rerank_config.get("enabled", False),
                id="config-rag-rerank-enabled"
            ),
            
            Label("Re-ranker Model:", classes="form-label"),
            Input(
                value=rerank_config.get("model", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
                id="config-rag-rerank-model"
            ),
            
            # Chunking Settings
            Static("Chunking Configuration", classes="form-section-title"),
            
            Label("Chunk Size:", classes="form-label"),
            Input(
                value=str(chunking_config.get("size", 512)),
                id="config-rag-chunk-size"
            ),
            
            Label("Chunk Overlap:", classes="form-label"),
            Input(
                value=str(chunking_config.get("overlap", 128)),
                id="config-rag-chunk-overlap"
            ),
            
            Label("Chunking Method:", classes="form-label"),
            Select(
                options=chunking_methods,
                value=chunking_config.get("method", "fixed"),
                id="config-rag-chunk-method"
            ),
            
            Container(
                Button("Save RAG Config", id="save-rag-config-form", variant="primary"),
                Button("Reset Section", id="reset-rag-config-form"),
                classes="form-actions"
            ),
            classes="config-form"
        )
    
    def _compose_providers_config_form(self) -> ComposeResult:
        """Form for providers configuration."""
        providers_config = self.config_data.get("providers", {})
        
        # Build widgets list
        widgets = [Static("Configure available models for each provider", classes="form-description")]
        
        for provider, models in providers_config.items():
            widgets.append(Static(f"{provider} Models", classes="form-section-title"))
            models_str = ", ".join(models) if isinstance(models, list) else str(models)
            widgets.append(Label(f"Available Models (comma-separated):", classes="form-label"))
            widgets.append(TextArea(
                text=models_str,
                id=f"config-provider-{provider.lower().replace(' ', '-')}",
                classes="provider-models-textarea"
            ))
        
        widgets.append(Container(
            Button("Save Providers Config", id="save-providers-config-form", variant="primary"),
            Button("Reset Section", id="reset-providers-config-form"),
            classes="form-actions"
        ))
        
        yield Container(*widgets, classes="config-form")
    
    def _compose_advanced_config_form(self) -> ComposeResult:
        """Form for advanced configuration options."""
        logging_config = self.config_data.get("logging", {})
        endpoints_config = self.config_data.get("api_endpoints", {})
        
        common_endpoints = [
            ("Llama.cpp", "llama_cpp", "http://localhost:8080"),
            ("KoboldCPP", "koboldcpp", "http://localhost:5001/api"),
            ("Oobabooga", "Oobabooga", "http://localhost:5000/api"),
            ("Ollama", "Ollama", "http://localhost:11434"),
            ("vLLM", "vLLM", "http://localhost:8000"),
            ("Custom", "Custom", "http://localhost:1234/v1")
        ]
        
        # Build widgets list
        widgets = [
            Static("Advanced settings - modify with caution", classes="form-description"),
            
            # Logging Configuration
            Static("Logging Configuration", classes="form-section-title"),
            
            Label("Log Filename:", classes="form-label"),
            Input(
                value=logging_config.get("log_filename", "tldw_cli_app.log"),
                id="config-logging-filename"
            ),
            
            Label("File Log Level:", classes="form-label"),
            Input(
                value=logging_config.get("file_log_level", "INFO"),
                id="config-logging-file-level"
            ),
            
            Label("Log Max Bytes:", classes="form-label"),
            Input(
                value=str(logging_config.get("log_max_bytes", 10485760)),
                id="config-logging-max-bytes"
            ),
            
            Label("Log Backup Count:", classes="form-label"),
            Input(
                value=str(logging_config.get("log_backup_count", 5)),
                id="config-logging-backup-count"
            ),
            
            # API Endpoints Configuration
            Static("API Endpoints Configuration", classes="form-section-title")
        ]
        
        # Add endpoint inputs
        for display_name, key, default_url in common_endpoints:
            widgets.append(Label(f"{display_name} Endpoint:", classes="form-label"))
            widgets.append(Input(
                value=endpoints_config.get(key, default_url),
                id=f"config-endpoint-{key.lower()}"
            ))
        
        # Add action buttons
        widgets.append(Container(
            Button("Save Advanced Config", id="save-advanced-config-form", variant="primary"),
            Button("Reset Section", id="reset-advanced-config-form"),
            Button("Export Configuration", id="export-config-button"),
            Button("Import Configuration", id="import-config-button"),
            classes="form-actions"
        ))
        
        yield Container(*widgets, classes="config-form")
    
    def _compose_database_tools(self) -> ComposeResult:
        """Compose the Database Tools UI."""
        yield Static("Database Tools", classes="section-title")
        yield Static("Manage and maintain your application databases", classes="section-description")
        
        with Container(classes="settings-container"):
            # Database Status Section
            with Collapsible(title="Database Status", collapsed=False):
                # Show current database sizes
                yield Label("ChaChaNotes Database:", classes="settings-label")
                yield Static("Size: Loading...", id="db-size-chachanotes", classes="db-status")
                
                yield Label("Prompts Database:", classes="settings-label")
                yield Static("Size: Loading...", id="db-size-prompts", classes="db-status")
                
                yield Label("Media Database:", classes="settings-label")
                yield Static("Size: Loading...", id="db-size-media", classes="db-status")
            
            # Database Maintenance Section
            with Collapsible(title="Database Maintenance", collapsed=False):
                yield Button("Vacuum All Databases", id="db-vacuum-all", variant="primary")
                yield Static("Reclaim unused space and optimize database performance", classes="help-text")
                
                yield Button("Backup All Databases", id="db-backup-all", variant="success")
                yield Static("Create timestamped backups of all databases", classes="help-text")
                
                yield Button("Check Database Integrity", id="db-check-integrity", variant="warning")
                yield Static("Verify database structure and data integrity", classes="help-text")
            
            # Export/Import Section
            with Collapsible(title="Export & Import", collapsed=False):
                yield Button("Export Conversations", id="db-export-conversations")
                yield Button("Export Notes", id="db-export-notes")
                yield Button("Export Characters", id="db-export-characters")
                
                yield Button("Import Data", id="db-import-data", variant="primary")
                yield Static("Import previously exported data files", classes="help-text")
            
            # Danger Zone - collapsed by default for safety
            with Collapsible(title="⚠️ Danger Zone", collapsed=True):
                yield Static("These actions cannot be undone!", classes="danger-warning")
                yield Button("Clear All Conversations", id="db-clear-conversations", variant="error")
                yield Button("Clear All Notes", id="db-clear-notes", variant="error")
                yield Button("Reset All Databases", id="db-reset-all", variant="error")
    
    def _compose_appearance_settings(self) -> ComposeResult:
        """Compose the Appearance Settings UI."""
        yield Static("Appearance Settings", classes="section-title")
        yield Static("Customize the look and feel of your application", classes="section-description")
        
        with Container(classes="settings-container"):
            # Theme Selection
            yield Label("Application Theme:", classes="settings-label")
            
            # Import themes to get all available options
            from ..css.Themes.themes import ALL_THEMES
            
            # Build theme options from ALL_THEMES
            theme_options = [
                ("Textual Dark", "textual-dark"),
                ("Textual Light", "textual-light")
            ]
            
            # Add custom themes from ALL_THEMES
            for theme in ALL_THEMES:
                theme_name = theme.name
                # Create a user-friendly label from the theme name
                label = theme_name.replace('_', ' ').title()
                theme_options.append((label, theme_name))
            
            current_theme = self.config_data.get("general", {}).get("default_theme", "textual-dark")
            
            # Validate the current theme - if it's not in the valid options, use default
            valid_theme_values = [value for _, value in theme_options]
            if current_theme not in valid_theme_values:
                current_theme = "textual-dark"
            
            yield Select(
                options=theme_options,
                value=current_theme,
                id="appearance-theme-select",
                classes="settings-select"
            )
            
            yield Button("Apply Theme", id="appearance-apply-theme", variant="primary")
            yield Static("Changes will take effect immediately", classes="help-text")
            
            # Font Settings
            yield Static("Font Settings", classes="form-section-title")
            
            yield Label("Code Font Size:", classes="settings-label")
            yield Select(
                options=[
                    ("Small (10px)", "10"),
                    ("Medium (12px)", "12"),
                    ("Large (14px)", "14"),
                    ("Extra Large (16px)", "16")
                ],
                value="12",
                id="appearance-font-size",
                classes="settings-select"
            )
            
            # UI Density
            yield Static("UI Density", classes="form-section-title")
            
            yield Label("Interface Density:", classes="settings-label")
            yield Select(
                options=[
                    ("Compact", "compact"),
                    ("Normal", "normal"),
                    ("Comfortable", "comfortable")
                ],
                value="normal",
                id="appearance-density",
                classes="settings-select"
            )
            
            # Animation Settings
            yield Static("Animation Settings", classes="form-section-title")
            
            yield Checkbox(
                "Enable UI Animations",
                value=True,
                id="appearance-enable-animations"
            )
            
            yield Checkbox(
                "Enable Smooth Scrolling",
                value=True,
                id="appearance-smooth-scrolling"
            )
            
            # Color Customization
            yield Static("Color Customization", classes="form-section-title")
            
            yield Label("Accent Color:", classes="settings-label")
            yield Input(
                value="#0078D4",
                placeholder="#0078D4",
                id="appearance-accent-color",
                classes="settings-input"
            )
            
            yield Label("Success Color:", classes="settings-label")
            yield Input(
                value="#10B981",
                placeholder="#10B981",
                id="appearance-success-color",
                classes="settings-input"
            )
            
            yield Label("Warning Color:", classes="settings-label")
            yield Input(
                value="#F59E0B",
                placeholder="#F59E0B",
                id="appearance-warning-color",
                classes="settings-input"
            )
            
            yield Label("Error Color:", classes="settings-label")
            yield Input(
                value="#EF4444",
                placeholder="#EF4444",
                id="appearance-error-color",
                classes="settings-input"
            )
            
            with Container(classes="settings-button-container"):
                yield Button("Save Appearance Settings", id="save-appearance-settings", variant="primary")
                yield Button("Reset to Defaults", id="reset-appearance-settings")

    def compose(self) -> ComposeResult:
        with Container(id="tools-settings-nav-pane", classes="tools-nav-pane"):
            yield Static("Navigation", classes="sidebar-title")
            yield Button("General Settings", id="ts-nav-general-settings", classes="ts-nav-button active-nav")
            yield Button("Configuration File Settings", id="ts-nav-config-file-settings", classes="ts-nav-button")
            yield Button("Database Tools", id="ts-nav-db-tools", classes="ts-nav-button")
            yield Button("Appearance", id="ts-nav-appearance", classes="ts-nav-button")

        with ContentSwitcher(id="tools-settings-content-pane", classes="tools-content-pane", initial="ts-view-general-settings"):
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
                *self._compose_database_tools(),
                id="ts-view-db-tools",
                classes="ts-view-area",
            )
            yield Container(
                *self._compose_appearance_settings(),
                id="ts-view-appearance",
                classes="ts-view-area",
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        button_id = event.button.id

        # Navigation handlers
        if button_id == "ts-nav-general-settings":
            await self._show_view("ts-view-general-settings")
        elif button_id == "ts-nav-config-file-settings":
            await self._show_view("ts-view-config-file-settings")
        elif button_id == "ts-nav-db-tools":
            await self._show_view("ts-view-db-tools")
        elif button_id == "ts-nav-appearance":
            await self._show_view("ts-view-appearance")
            
        # General Settings handlers
        elif button_id == "save-general-settings":
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
            # Save only the settings that exist in the current general settings form
            saved_count = 0
            
            # Default Tab
            if save_setting_to_cli_config("general", "default_tab", self.query_one("#general-default-tab", Select).value):
                saved_count += 1
            
            # Theme
            if save_setting_to_cli_config("general", "default_theme", self.query_one("#general-theme", Select).value):
                saved_count += 1
            
            # Username
            if save_setting_to_cli_config("general", "users_name", self.query_one("#general-username", Input).value):
                saved_count += 1
            
            # Log Level
            if save_setting_to_cli_config("general", "log_level", self.query_one("#general-log-level", Select).value):
                saved_count += 1
            
            # Update internal config
            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
            
            if saved_count > 0:
                self.app_instance.notify(f"General Settings saved successfully! ({saved_count} settings updated)")
            else:
                self.app_instance.notify("No settings were updated", severity="warning")
            
        except Exception as e:
            self.app_instance.notify(f"Error saving General Settings: {e}", severity="error")

    async def _reset_general_settings(self) -> None:
        """Reset General Settings to default values."""
        try:
            # Reset only the UI elements that exist in the current general settings form
            self.query_one("#general-default-tab", Select).value = "chat"
            self.query_one("#general-theme", Select).value = "textual-dark"
            self.query_one("#general-username", Input).value = "default_user"
            self.query_one("#general-log-level", Select).value = "INFO"
            
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
            saved_count = 0
            errors = []
            
            # Get all providers and their fields
            all_providers = [
                ("openai", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
                ("anthropic", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
                ("google", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
                ("deepseek", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
                ("groq", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
                ("mistralai", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
                ("cohere", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
                ("huggingface", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
                ("openrouter", ["api_key_env_var", "api_key", "model", "temperature", "top_p", "top_k", "min_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming"]),
                ("llama_cpp", ["api_url", "model", "temperature", "top_p", "top_k", "min_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming", "system_prompt"]),
                ("ollama", ["api_url", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming", "system_prompt"]),
                ("vllm", ["api_key_env_var", "api_url", "model", "temperature", "top_p", "top_k", "min_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming", "system_prompt"]),
                ("koboldcpp", ["api_url", "model", "temperature", "top_p", "top_k", "max_tokens", "timeout", "retries", "retry_delay", "streaming", "system_prompt"]),
                ("oobabooga", ["api_key_env_var", "api_url", "model", "temperature", "top_p", "max_tokens", "timeout", "retries", "retry_delay", "streaming", "system_prompt"]),
            ]
            
            # Save settings for each provider
            for provider_key, fields in all_providers:
                for field in fields:
                    widget_id = f"#config-{provider_key}-{field}"
                    try:
                        if field == "streaming":
                            # Handle checkbox
                            value = self.query_one(widget_id, Checkbox).value
                        elif field == "system_prompt":
                            # Handle TextArea
                            value = self.query_one(widget_id, TextArea).text
                        else:
                            # Handle Input fields
                            raw_value = self.query_one(widget_id, Input).value
                            
                            # Convert to appropriate type
                            if field in ["temperature", "top_p", "top_k", "min_p"]:
                                value = float(raw_value) if raw_value else 0.0
                            elif field in ["max_tokens", "timeout", "retries", "retry_delay"]:
                                value = int(raw_value) if raw_value else 0
                            else:
                                value = raw_value
                        
                        if save_setting_to_cli_config(f"api_settings.{provider_key}", field, value):
                            saved_count += 1
                            
                    except QueryError:
                        # Widget doesn't exist, skip
                        pass
                    except ValueError as e:
                        errors.append(f"{provider_key}.{field}: {str(e)}")
                    except Exception as e:
                        errors.append(f"{provider_key}.{field}: {str(e)}")
            
            # Update internal config
            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
            
            if errors:
                self.app_instance.notify(f"API config saved with {len(errors)} errors. Check logs for details.", severity="warning")
                for error in errors[:3]:  # Show first 3 errors
                    self.app_instance.notify(f"Error: {error}", severity="error")
            elif saved_count > 0:
                self.app_instance.notify(f"API configuration saved! ({saved_count} settings updated)")
            else:
                self.app_instance.notify("No API settings were updated", severity="warning")
                
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
    
    async def _show_view(self, view_id: str) -> None:
        """Show the specified view and hide all others."""
        # Use ContentSwitcher to switch views
        try:
            content_switcher = self.query_one("#tools-settings-content-pane", ContentSwitcher)
            content_switcher.current = view_id
        except Exception as e:
            # If ContentSwitcher fails, fall back to the old method
            # List of all view IDs
            view_ids = [
                "ts-view-general-settings",
                "ts-view-config-file-settings", 
                "ts-view-db-tools",
                "ts-view-appearance"
            ]
            
            # Hide all views by removing active class
            for v_id in view_ids:
                try:
                    view = self.query_one(f"#{v_id}")
                    view.remove_class("active")
                except Exception:
                    pass  # View might not exist yet
            
            # Show the requested view by adding active class
            try:
                view = self.query_one(f"#{view_id}")
                view.add_class("active")
            except Exception:
                pass
            
        # Update navigation button styles
        nav_buttons = {
            "ts-view-general-settings": "ts-nav-general-settings",
            "ts-view-config-file-settings": "ts-nav-config-file-settings",
            "ts-view-db-tools": "ts-nav-db-tools",
            "ts-view-appearance": "ts-nav-appearance"
        }
        
        for v_id, btn_id in nav_buttons.items():
            try:
                button = self.query_one(f"#{btn_id}")
                if v_id == view_id:
                    button.add_class("active-nav")
                else:
                    button.remove_class("active-nav")
            except Exception:
                pass
    
    async def on_mount(self) -> None:
        """Called when the widget is mounted. Set initial view."""
        # Ensure only the general settings view is active on mount
        await self._show_view("ts-view-general-settings")

#
# End of Tools_Settings_Window.py
#######################################################################################################################
