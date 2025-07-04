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
            
            # Chat Configuration Tab
            with TabPane("Chat", id="tab-chat-config"):
                yield Static("Chat Default Settings", classes="tab-description")
                with VerticalScroll():
                    yield from self._compose_chat_config_form()
            
            # Character Configuration Tab
            with TabPane("Character", id="tab-character-config"):
                yield Static("Character Default Settings", classes="tab-description")
                with VerticalScroll():
                    yield from self._compose_character_config_form()
            
            # Notes Configuration Tab
            with TabPane("Notes", id="tab-notes-config"):
                yield Static("Notes Synchronization Settings", classes="tab-description")
                with VerticalScroll():
                    yield from self._compose_notes_config_form()
            
            # TTS Configuration Tab
            with TabPane("TTS", id="tab-tts-config"):
                yield Static("Text-to-Speech Settings", classes="tab-description")
                with VerticalScroll():
                    yield from self._compose_tts_config_form()
            
            # Embedding Configuration Tab
            with TabPane("Embeddings", id="tab-embedding-config"):
                yield Static("Embedding Model Settings", classes="tab-description")
                with VerticalScroll():
                    yield from self._compose_embedding_config_form()
            
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
        retriever_config = rag_config.get("retriever", {})
        processor_config = rag_config.get("processor", {})
        generator_config = rag_config.get("generator", {})
        chroma_config = rag_config.get("chroma", {})
        
        search_modes = [("Q&A Mode", "qa"), ("Chat Mode", "chat")]
        chunking_methods = [
            ("Fixed Size", "fixed"),
            ("Semantic", "semantic"),
            ("Sentence-based", "sentence")
        ]
        combination_methods = [
            ("Weighted", "weighted"),
            ("Simple", "simple"),
            ("Reciprocal Rank Fusion", "rrf")
        ]
        
        widgets = []
        
        # Basic Settings
        widgets.append(Static("Basic RAG Settings", classes="form-section-title"))
        
        widgets.append(Label("Default Search Mode:", classes="form-label"))
        widgets.append(Select(
            options=search_modes,
            value=rag_config.get("default_mode", "qa"),
            id="config-rag-search-mode"
        ))
        
        widgets.append(Label("Default Top-K Results:", classes="form-label"))
        widgets.append(Input(
            value=str(rag_config.get("default_top_k", 10)),
            id="config-rag-top-k"
        ))
        
        # Default Sources
        widgets.append(Static("Default Search Sources", classes="form-section-title"))
        widgets.append(Checkbox(
            "Media Files",
            value=sources_config.get("media", True),
            id="config-rag-source-media"
        ))
        widgets.append(Checkbox(
            "Conversations",
            value=sources_config.get("conversations", True),
            id="config-rag-source-conversations"
        ))
        widgets.append(Checkbox(
            "Notes",
            value=sources_config.get("notes", True),
            id="config-rag-source-notes"
        ))
        
        # Retriever Settings
        retriever_widgets = []
        retriever_widgets.append(Label("FTS Top-K:", classes="form-label"))
        retriever_widgets.append(Input(
            value=str(retriever_config.get("fts_top_k", 10)),
            id="config-rag-retriever-fts-top-k"
        ))
        
        retriever_widgets.append(Label("Vector Top-K:", classes="form-label"))
        retriever_widgets.append(Input(
            value=str(retriever_config.get("vector_top_k", 10)),
            id="config-rag-retriever-vector-top-k"
        ))
        
        retriever_widgets.append(Label("Hybrid Alpha (0.0-1.0):", classes="form-label"))
        retriever_widgets.append(Input(
            value=str(retriever_config.get("hybrid_alpha", 0.5)),
            id="config-rag-retriever-hybrid-alpha"
        ))
        
        retriever_widgets.append(Label("Media Collection Name:", classes="form-label"))
        retriever_widgets.append(Input(
            value=retriever_config.get("media_collection", "media_embeddings"),
            id="config-rag-retriever-media-collection"
        ))
        
        retriever_widgets.append(Label("Chat Collection Name:", classes="form-label"))
        retriever_widgets.append(Input(
            value=retriever_config.get("chat_collection", "chat_embeddings"),
            id="config-rag-retriever-chat-collection"
        ))
        
        retriever_widgets.append(Label("Notes Collection Name:", classes="form-label"))
        retriever_widgets.append(Input(
            value=retriever_config.get("notes_collection", "notes_embeddings"),
            id="config-rag-retriever-notes-collection"
        ))
        
        widgets.append(Collapsible(
            *retriever_widgets,
            title="Retriever Configuration",
            collapsed=True
        ))
        
        # Processor Settings
        processor_widgets = []
        processor_widgets.append(Checkbox(
            "Enable Re-ranking",
            value=processor_config.get("enable_reranking", True),
            id="config-rag-processor-enable-reranking"
        ))
        
        processor_widgets.append(Label("Re-ranker Model:", classes="form-label"))
        processor_widgets.append(Input(
            value=processor_config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
            id="config-rag-processor-reranker-model"
        ))
        
        processor_widgets.append(Label("Re-ranker Top-K:", classes="form-label"))
        processor_widgets.append(Input(
            value=str(processor_config.get("reranker_top_k", 5)),
            id="config-rag-processor-reranker-top-k"
        ))
        
        processor_widgets.append(Label("Deduplication Threshold (0.0-1.0):", classes="form-label"))
        processor_widgets.append(Input(
            value=str(processor_config.get("deduplication_threshold", 0.85)),
            id="config-rag-processor-deduplication"
        ))
        
        processor_widgets.append(Label("Max Context Length:", classes="form-label"))
        processor_widgets.append(Input(
            value=str(processor_config.get("max_context_length", 4096)),
            id="config-rag-processor-max-context"
        ))
        
        processor_widgets.append(Label("Combination Method:", classes="form-label"))
        processor_widgets.append(Select(
            options=combination_methods,
            value=processor_config.get("combination_method", "weighted"),
            id="config-rag-processor-combination"
        ))
        
        widgets.append(Collapsible(
            *processor_widgets,
            title="Processor Configuration",
            collapsed=True
        ))
        
        # Generator Settings
        generator_widgets = []
        generator_widgets.append(Label("Default Model:", classes="form-label"))
        generator_widgets.append(Input(
            value=generator_config.get("default_model", ""),
            id="config-rag-generator-model",
            placeholder="Leave empty to use chat default"
        ))
        
        generator_widgets.append(Label("Default Temperature:", classes="form-label"))
        generator_widgets.append(Input(
            value=str(generator_config.get("default_temperature", 0.7)),
            id="config-rag-generator-temperature"
        ))
        
        generator_widgets.append(Label("Max Tokens:", classes="form-label"))
        generator_widgets.append(Input(
            value=str(generator_config.get("max_tokens", 1024)),
            id="config-rag-generator-max-tokens"
        ))
        
        generator_widgets.append(Checkbox(
            "Enable Streaming",
            value=generator_config.get("enable_streaming", True),
            id="config-rag-generator-streaming"
        ))
        
        generator_widgets.append(Label("Stream Chunk Size:", classes="form-label"))
        generator_widgets.append(Input(
            value=str(generator_config.get("stream_chunk_size", 10)),
            id="config-rag-generator-chunk-size"
        ))
        
        widgets.append(Collapsible(
            *generator_widgets,
            title="Generator Configuration",
            collapsed=True
        ))
        
        # Chunking Settings (kept from original)
        widgets.append(Static("Chunking Configuration", classes="form-section-title"))
        
        widgets.append(Label("Chunk Size:", classes="form-label"))
        widgets.append(Input(
            value=str(chunking_config.get("size", 512)),
            id="config-rag-chunk-size"
        ))
        
        widgets.append(Label("Chunk Overlap:", classes="form-label"))
        widgets.append(Input(
            value=str(chunking_config.get("overlap", 128)),
            id="config-rag-chunk-overlap"
        ))
        
        widgets.append(Label("Chunking Method:", classes="form-label"))
        widgets.append(Select(
            options=chunking_methods,
            value=chunking_config.get("method", "fixed"),
            id="config-rag-chunk-method"
        ))
        
        # ChromaDB Settings
        chroma_widgets = []
        chroma_widgets.append(Label("Persist Directory:", classes="form-label"))
        chroma_widgets.append(Input(
            value=chroma_config.get("persist_directory", ""),
            id="config-rag-chroma-persist-dir",
            placeholder="Leave empty for default"
        ))
        
        chroma_widgets.append(Label("Collection Prefix:", classes="form-label"))
        chroma_widgets.append(Input(
            value=chroma_config.get("collection_prefix", "tldw_rag"),
            id="config-rag-chroma-prefix"
        ))
        
        chroma_widgets.append(Label("Embedding Model:", classes="form-label"))
        chroma_widgets.append(Input(
            value=chroma_config.get("embedding_model", "all-MiniLM-L6-v2"),
            id="config-rag-chroma-embedding-model"
        ))
        
        chroma_widgets.append(Label("Embedding Dimension:", classes="form-label"))
        chroma_widgets.append(Input(
            value=str(chroma_config.get("embedding_dimension", 384)),
            id="config-rag-chroma-dimension"
        ))
        
        chroma_widgets.append(Label("Distance Metric:", classes="form-label"))
        distance_options = [("Cosine", "cosine"), ("L2", "l2"), ("IP", "ip")]
        chroma_widgets.append(Select(
            options=distance_options,
            value=chroma_config.get("distance_metric", "cosine"),
            id="config-rag-chroma-distance"
        ))
        
        widgets.append(Collapsible(
            *chroma_widgets,
            title="ChromaDB Configuration",
            collapsed=True
        ))
        
        # Action buttons
        widgets.append(Container(
            Button("Save RAG Config", id="save-rag-config-form", variant="primary"),
            Button("Reset Section", id="reset-rag-config-form"),
            classes="form-actions"
        ))
        
        yield Container(*widgets, classes="config-form")
    
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
    
    def _compose_chat_config_form(self) -> ComposeResult:
        """Form for chat default configuration."""
        chat_config = self.config_data.get("chat_defaults", {})
        chat_images_config = self.config_data.get("chat", {}).get("images", {})
        
        # Get providers list for dropdown
        providers = list(self.config_data.get("providers", {}).keys())
        provider_options = [(p, p) for p in providers]
        
        widgets = []
        
        # Main chat settings
        widgets.append(Static("Chat Default Settings", classes="form-section-title"))
        
        widgets.append(Label("Default Provider:", classes="form-label"))
        current_provider = chat_config.get("provider", "OpenAI")
        if current_provider not in providers:
            current_provider = providers[0] if providers else "OpenAI"
        widgets.append(Select(
            options=provider_options,
            value=current_provider,
            id="config-chat-provider"
        ))
        
        widgets.append(Label("Default Model:", classes="form-label"))
        widgets.append(Input(
            value=chat_config.get("model", "gpt-4o"),
            id="config-chat-model",
            placeholder="Enter model name"
        ))
        
        widgets.append(Label("System Prompt:", classes="form-label"))
        widgets.append(TextArea(
            text=chat_config.get("system_prompt", "You are a helpful AI assistant."),
            id="config-chat-system-prompt",
            classes="system-prompt-textarea"
        ))
        
        widgets.append(Label("Temperature:", classes="form-label"))
        widgets.append(Input(
            value=str(chat_config.get("temperature", 0.6)),
            id="config-chat-temperature",
            placeholder="0.0 - 2.0"
        ))
        
        widgets.append(Label("Top P:", classes="form-label"))
        widgets.append(Input(
            value=str(chat_config.get("top_p", 0.95)),
            id="config-chat-top-p",
            placeholder="0.0 - 1.0"
        ))
        
        widgets.append(Label("Min P:", classes="form-label"))
        widgets.append(Input(
            value=str(chat_config.get("min_p", 0.05)),
            id="config-chat-min-p",
            placeholder="0.0 - 1.0"
        ))
        
        widgets.append(Label("Top K:", classes="form-label"))
        widgets.append(Input(
            value=str(chat_config.get("top_k", 50)),
            id="config-chat-top-k",
            placeholder="1 - 1000"
        ))
        
        widgets.append(Checkbox(
            "Strip Thinking Tags",
            value=chat_config.get("strip_thinking_tags", True),
            id="config-chat-strip-thinking"
        ))
        
        widgets.append(Checkbox(
            "Use Enhanced Window (with image support)",
            value=chat_config.get("use_enhanced_window", False),
            id="config-chat-enhanced-window"
        ))
        
        # Image settings in collapsible
        widgets.append(Static("Image Support Settings", classes="form-section-title"))
        
        image_widgets = []
        image_widgets.append(Checkbox(
            "Enable Image Support",
            value=chat_images_config.get("enabled", True),
            id="config-chat-images-enabled"
        ))
        
        image_widgets.append(Checkbox(
            "Show Attach Button",
            value=chat_images_config.get("show_attach_button", True),
            id="config-chat-images-show-button"
        ))
        
        image_widgets.append(Label("Default Render Mode:", classes="form-label"))
        render_options = [("Auto", "auto"), ("Pixels", "pixels"), ("Regular", "regular")]
        image_widgets.append(Select(
            options=render_options,
            value=chat_images_config.get("default_render_mode", "auto"),
            id="config-chat-images-render-mode"
        ))
        
        image_widgets.append(Label("Max Size (MB):", classes="form-label"))
        image_widgets.append(Input(
            value=str(chat_images_config.get("max_size_mb", 10.0)),
            id="config-chat-images-max-size",
            placeholder="10.0"
        ))
        
        image_widgets.append(Checkbox(
            "Auto Resize Images",
            value=chat_images_config.get("auto_resize", True),
            id="config-chat-images-auto-resize"
        ))
        
        image_widgets.append(Label("Resize Max Dimension:", classes="form-label"))
        image_widgets.append(Input(
            value=str(chat_images_config.get("resize_max_dimension", 2048)),
            id="config-chat-images-max-dimension",
            placeholder="2048"
        ))
        
        image_widgets.append(Label("Save Location:", classes="form-label"))
        image_widgets.append(Input(
            value=chat_images_config.get("save_location", "~/Downloads"),
            id="config-chat-images-save-location",
            placeholder="~/Downloads"
        ))
        
        image_widgets.append(Label("Supported Formats (comma-separated):", classes="form-label"))
        formats = chat_images_config.get("supported_formats", [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"])
        formats_str = ", ".join(formats) if isinstance(formats, list) else str(formats)
        image_widgets.append(Input(
            value=formats_str,
            id="config-chat-images-formats",
            placeholder=".png, .jpg, .jpeg, .gif"
        ))
        
        widgets.append(Collapsible(
            *image_widgets,
            title="Image Support Settings",
            collapsed=True
        ))
        
        # Action buttons
        widgets.append(Container(
            Button("Save Chat Config", id="save-chat-config-form", variant="primary"),
            Button("Reset Section", id="reset-chat-config-form"),
            classes="form-actions"
        ))
        
        yield Container(*widgets, classes="config-form")
    
    def _compose_character_config_form(self) -> ComposeResult:
        """Form for character default configuration."""
        char_config = self.config_data.get("character_defaults", {})
        
        # Get providers list for dropdown
        providers = list(self.config_data.get("providers", {}).keys())
        provider_options = [(p, p) for p in providers]
        
        widgets = []
        
        widgets.append(Static("Character Default Settings", classes="form-section-title"))
        
        widgets.append(Label("Default Provider:", classes="form-label"))
        current_provider = char_config.get("provider", "Anthropic")
        if current_provider not in providers:
            current_provider = providers[0] if providers else "Anthropic"
        widgets.append(Select(
            options=provider_options,
            value=current_provider,
            id="config-character-provider"
        ))
        
        widgets.append(Label("Default Model:", classes="form-label"))
        widgets.append(Input(
            value=char_config.get("model", "claude-3-haiku-20240307"),
            id="config-character-model",
            placeholder="Enter model name"
        ))
        
        widgets.append(Label("System Prompt:", classes="form-label"))
        widgets.append(TextArea(
            text=char_config.get("system_prompt", "You are roleplaying as a witty pirate captain."),
            id="config-character-system-prompt",
            classes="system-prompt-textarea"
        ))
        
        widgets.append(Label("Temperature:", classes="form-label"))
        widgets.append(Input(
            value=str(char_config.get("temperature", 0.8)),
            id="config-character-temperature",
            placeholder="0.0 - 2.0"
        ))
        
        widgets.append(Label("Top P:", classes="form-label"))
        widgets.append(Input(
            value=str(char_config.get("top_p", 0.9)),
            id="config-character-top-p",
            placeholder="0.0 - 1.0"
        ))
        
        widgets.append(Label("Min P:", classes="form-label"))
        widgets.append(Input(
            value=str(char_config.get("min_p", 0.0)),
            id="config-character-min-p",
            placeholder="0.0 - 1.0"
        ))
        
        widgets.append(Label("Top K:", classes="form-label"))
        widgets.append(Input(
            value=str(char_config.get("top_k", 100)),
            id="config-character-top-k",
            placeholder="1 - 1000"
        ))
        
        # Action buttons
        widgets.append(Container(
            Button("Save Character Config", id="save-character-config-form", variant="primary"),
            Button("Reset Section", id="reset-character-config-form"),
            classes="form-actions"
        ))
        
        yield Container(*widgets, classes="config-form")
    
    def _compose_notes_config_form(self) -> ComposeResult:
        """Form for notes configuration."""
        notes_config = self.config_data.get("notes", {})
        
        conflict_options = [
            ("Newer Wins", "newer_wins"),
            ("Ask User", "ask"),
            ("Disk Wins", "disk_wins"),
            ("Database Wins", "db_wins")
        ]
        
        sync_direction_options = [
            ("Bidirectional", "bidirectional"),
            ("Disk to Database", "disk_to_db"),
            ("Database to Disk", "db_to_disk")
        ]
        
        widgets = []
        
        widgets.append(Static("Notes Synchronization Settings", classes="form-section-title"))
        
        widgets.append(Label("Sync Directory:", classes="form-label"))
        widgets.append(Input(
            value=notes_config.get("sync_directory", "~/Documents/Notes"),
            id="config-notes-sync-directory",
            placeholder="~/Documents/Notes"
        ))
        
        widgets.append(Checkbox(
            "Enable Auto Sync on Startup",
            value=notes_config.get("auto_sync_enabled", False),
            id="config-notes-auto-sync"
        ))
        
        widgets.append(Checkbox(
            "Sync on Close",
            value=notes_config.get("sync_on_close", False),
            id="config-notes-sync-on-close"
        ))
        
        widgets.append(Label("Conflict Resolution:", classes="form-label"))
        widgets.append(Select(
            options=conflict_options,
            value=notes_config.get("conflict_resolution", "newer_wins"),
            id="config-notes-conflict-resolution"
        ))
        
        widgets.append(Label("Sync Direction:", classes="form-label"))
        widgets.append(Select(
            options=sync_direction_options,
            value=notes_config.get("sync_direction", "bidirectional"),
            id="config-notes-sync-direction"
        ))
        
        # Action buttons
        widgets.append(Container(
            Button("Save Notes Config", id="save-notes-config-form", variant="primary"),
            Button("Reset Section", id="reset-notes-config-form"),
            classes="form-actions"
        ))
        
        yield Container(*widgets, classes="config-form")
    
    def _compose_tts_config_form(self) -> ComposeResult:
        """Form for TTS configuration."""
        tts_config = self.config_data.get("app_tts", {})
        
        widgets = []
        
        widgets.append(Static("Text-to-Speech Configuration", classes="form-section-title"))
        
        # OpenAI TTS
        widgets.append(Static("OpenAI TTS", classes="form-subsection-title"))
        widgets.append(Label("OpenAI API Key (Fallback):", classes="form-label"))
        widgets.append(Input(
            value=tts_config.get("OPENAI_API_KEY_fallback", "sk-..."),
            id="config-tts-openai-key",
            password=True,
            placeholder="sk-..."
        ))
        
        # Kokoro ONNX
        widgets.append(Static("Kokoro ONNX", classes="form-subsection-title"))
        widgets.append(Label("Model Path:", classes="form-label"))
        widgets.append(Input(
            value=tts_config.get("KOKORO_ONNX_MODEL_PATH_DEFAULT", "path/to/kokoro-v0_19.onnx"),
            id="config-tts-kokoro-model-path",
            placeholder="path/to/kokoro-v0_19.onnx"
        ))
        
        widgets.append(Label("Voices JSON Path:", classes="form-label"))
        widgets.append(Input(
            value=tts_config.get("KOKORO_ONNX_VOICES_JSON_DEFAULT", "path/to/voices.json"),
            id="config-tts-kokoro-voices-path",
            placeholder="path/to/voices.json"
        ))
        
        widgets.append(Label("Device:", classes="form-label"))
        device_options = [("CPU", "cpu"), ("CUDA", "cuda"), ("CUDA:0", "cuda:0")]
        widgets.append(Select(
            options=device_options,
            value=tts_config.get("KOKORO_DEVICE_DEFAULT", "cpu"),
            id="config-tts-kokoro-device"
        ))
        
        # ElevenLabs
        widgets.append(Static("ElevenLabs", classes="form-subsection-title"))
        widgets.append(Label("ElevenLabs API Key (Fallback):", classes="form-label"))
        widgets.append(Input(
            value=tts_config.get("ELEVENLABS_API_KEY_fallback", "el-..."),
            id="config-tts-elevenlabs-key",
            password=True,
            placeholder="el-..."
        ))
        
        # Action buttons
        widgets.append(Container(
            Button("Save TTS Config", id="save-tts-config-form", variant="primary"),
            Button("Reset Section", id="reset-tts-config-form"),
            classes="form-actions"
        ))
        
        yield Container(*widgets, classes="config-form")
    
    def _compose_embedding_config_form(self) -> ComposeResult:
        """Form for embedding configuration."""
        embedding_config = self.config_data.get("embedding_config", {})
        models = embedding_config.get("models", {})
        
        widgets = []
        
        widgets.append(Static("Embedding Configuration", classes="form-section-title"))
        
        # Default settings
        widgets.append(Label("Default Model ID:", classes="form-label"))
        model_options = [(model_id, model_id) for model_id in models.keys()]
        widgets.append(Select(
            options=model_options if model_options else [("e5-small-v2", "e5-small-v2")],
            value=embedding_config.get("default_model_id", "e5-small-v2"),
            id="config-embedding-default-model"
        ))
        
        widgets.append(Label("Default LLM for Contextualization:", classes="form-label"))
        widgets.append(Input(
            value=embedding_config.get("default_llm_for_contextualization", "gpt-3.5-turbo"),
            id="config-embedding-default-llm",
            placeholder="gpt-3.5-turbo"
        ))
        
        # Model-specific settings in collapsibles
        for model_id, model_config in models.items():
            model_widgets = []
            
            model_widgets.append(Label("Provider:", classes="form-label"))
            model_widgets.append(Input(
                value=model_config.get("provider", ""),
                id=f"config-embedding-{model_id}-provider",
                placeholder="huggingface, openai, etc."
            ))
            
            model_widgets.append(Label("Model Name/Path:", classes="form-label"))
            model_widgets.append(Input(
                value=model_config.get("model_name_or_path", ""),
                id=f"config-embedding-{model_id}-name",
                placeholder="model name or path"
            ))
            
            model_widgets.append(Label("Dimension:", classes="form-label"))
            model_widgets.append(Input(
                value=str(model_config.get("dimension", 384)),
                id=f"config-embedding-{model_id}-dimension",
                placeholder="384"
            ))
            
            if model_config.get("provider") == "openai":
                model_widgets.append(Label("API Key:", classes="form-label"))
                model_widgets.append(Input(
                    value=model_config.get("api_key", ""),
                    id=f"config-embedding-{model_id}-api-key",
                    password=True,
                    placeholder="Your API key"
                ))
            
            widgets.append(Collapsible(
                *model_widgets,
                title=f"Model: {model_id}",
                collapsed=True
            ))
        
        # Action buttons
        widgets.append(Container(
            Button("Save Embedding Config", id="save-embedding-config-form", variant="primary"),
            Button("Reset Section", id="reset-embedding-config-form"),
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
            yield Collapsible(
                Label("ChaChaNotes Database:", classes="settings-label"),
                Static("Size: Loading...", id="db-size-chachanotes", classes="db-status"),
                Label("Prompts Database:", classes="settings-label"),
                Static("Size: Loading...", id="db-size-prompts", classes="db-status"),
                Label("Media Database:", classes="settings-label"),
                Static("Size: Loading...", id="db-size-media", classes="db-status"),
                title="Database Status",
                collapsed=False
            )
            
            # Database Maintenance Section
            yield Collapsible(
                Button("Vacuum All Databases", id="db-vacuum-all", variant="primary"),
                Static("Reclaim unused space and optimize database performance", classes="help-text"),
                Button("Backup All Databases", id="db-backup-all", variant="success"),
                Static("Create timestamped backups of all databases", classes="help-text"),
                Button("Check Database Integrity", id="db-check-integrity", variant="warning"),
                Static("Verify database structure and data integrity", classes="help-text"),
                title="Database Maintenance",
                collapsed=False
            )
            
            # Export/Import Section
            yield Collapsible(
                Button("Export Conversations", id="db-export-conversations"),
                Button("Export Notes", id="db-export-notes"),
                Button("Export Characters", id="db-export-characters"),
                Button("Import Data", id="db-import-data", variant="primary"),
                Static("Import previously exported data files", classes="help-text"),
                title="Export & Import",
                collapsed=False
            )
            
            # Danger Zone - collapsed by default for safety
            yield Collapsible(
                Static("These actions cannot be undone!", classes="danger-warning"),
                Button("Clear All Conversations", id="db-clear-conversations", variant="error"),
                Button("Clear All Notes", id="db-clear-notes", variant="error"),
                Button("Reset All Databases", id="db-reset-all", variant="error"),
                title="⚠️ Danger Zone",
                collapsed=True
            )
    
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
            
        # New config tab handlers
        elif button_id == "save-chat-config-form":
            await self._save_chat_config_form()
        elif button_id == "reset-chat-config-form":
            await self._reset_chat_config_form()
        elif button_id == "save-character-config-form":
            await self._save_character_config_form()
        elif button_id == "reset-character-config-form":
            await self._reset_character_config_form()
        elif button_id == "save-notes-config-form":
            await self._save_notes_config_form()
        elif button_id == "reset-notes-config-form":
            await self._reset_notes_config_form()
        elif button_id == "save-tts-config-form":
            await self._save_tts_config_form()
        elif button_id == "reset-tts-config-form":
            await self._reset_tts_config_form()
        elif button_id == "save-embedding-config-form":
            await self._save_embedding_config_form()
        elif button_id == "reset-embedding-config-form":
            await self._reset_embedding_config_form()

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
            saved_count = 0
            
            # Basic settings
            if save_setting_to_cli_config("rag_search", "default_mode", self.query_one("#config-rag-search-mode", Select).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search", "default_top_k", int(self.query_one("#config-rag-top-k", Input).value)):
                saved_count += 1
            
            # Sources
            if save_setting_to_cli_config("rag_search.default_sources", "media", self.query_one("#config-rag-source-media", Checkbox).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.default_sources", "conversations", self.query_one("#config-rag-source-conversations", Checkbox).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.default_sources", "notes", self.query_one("#config-rag-source-notes", Checkbox).value):
                saved_count += 1
            
            # Retriever settings
            if save_setting_to_cli_config("rag_search.retriever", "fts_top_k", int(self.query_one("#config-rag-retriever-fts-top-k", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.retriever", "vector_top_k", int(self.query_one("#config-rag-retriever-vector-top-k", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.retriever", "hybrid_alpha", float(self.query_one("#config-rag-retriever-hybrid-alpha", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.retriever", "media_collection", self.query_one("#config-rag-retriever-media-collection", Input).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.retriever", "chat_collection", self.query_one("#config-rag-retriever-chat-collection", Input).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.retriever", "notes_collection", self.query_one("#config-rag-retriever-notes-collection", Input).value):
                saved_count += 1
            
            # Processor settings
            if save_setting_to_cli_config("rag_search.processor", "enable_reranking", self.query_one("#config-rag-processor-enable-reranking", Checkbox).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.processor", "reranker_model", self.query_one("#config-rag-processor-reranker-model", Input).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.processor", "reranker_top_k", int(self.query_one("#config-rag-processor-reranker-top-k", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.processor", "deduplication_threshold", float(self.query_one("#config-rag-processor-deduplication", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.processor", "max_context_length", int(self.query_one("#config-rag-processor-max-context", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.processor", "combination_method", self.query_one("#config-rag-processor-combination", Select).value):
                saved_count += 1
            
            # Generator settings
            if save_setting_to_cli_config("rag_search.generator", "default_model", self.query_one("#config-rag-generator-model", Input).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.generator", "default_temperature", float(self.query_one("#config-rag-generator-temperature", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.generator", "max_tokens", int(self.query_one("#config-rag-generator-max-tokens", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.generator", "enable_streaming", self.query_one("#config-rag-generator-streaming", Checkbox).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.generator", "stream_chunk_size", int(self.query_one("#config-rag-generator-chunk-size", Input).value)):
                saved_count += 1
            
            # Chunking settings
            if save_setting_to_cli_config("rag_search.chunking", "size", int(self.query_one("#config-rag-chunk-size", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.chunking", "overlap", int(self.query_one("#config-rag-chunk-overlap", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.chunking", "method", self.query_one("#config-rag-chunk-method", Select).value):
                saved_count += 1
            
            # ChromaDB settings
            if save_setting_to_cli_config("rag_search.chroma", "persist_directory", self.query_one("#config-rag-chroma-persist-dir", Input).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.chroma", "collection_prefix", self.query_one("#config-rag-chroma-prefix", Input).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.chroma", "embedding_model", self.query_one("#config-rag-chroma-embedding-model", Input).value):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.chroma", "embedding_dimension", int(self.query_one("#config-rag-chroma-dimension", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("rag_search.chroma", "distance_metric", self.query_one("#config-rag-chroma-distance", Select).value):
                saved_count += 1
            
            # Update internal config
            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
            
            if saved_count > 0:
                self.app_instance.notify(f"RAG configuration saved! ({saved_count} settings updated)")
            else:
                self.app_instance.notify("No RAG settings were updated", severity="warning")
                
        except ValueError as e:
            self.app_instance.notify(f"Invalid value: {e}", severity="error")
        except Exception as e:
            self.app_instance.notify(f"Error saving RAG config: {e}", severity="error")
    
    async def _reset_rag_config_form(self) -> None:
        """Reset RAG configuration form to defaults."""
        try:
            # Basic settings
            self.query_one("#config-rag-search-mode", Select).value = "qa"
            self.query_one("#config-rag-top-k", Input).value = "10"
            
            # Sources
            self.query_one("#config-rag-source-media", Checkbox).value = True
            self.query_one("#config-rag-source-conversations", Checkbox).value = True
            self.query_one("#config-rag-source-notes", Checkbox).value = True
            
            # Retriever settings
            self.query_one("#config-rag-retriever-fts-top-k", Input).value = "10"
            self.query_one("#config-rag-retriever-vector-top-k", Input).value = "10"
            self.query_one("#config-rag-retriever-hybrid-alpha", Input).value = "0.5"
            self.query_one("#config-rag-retriever-media-collection", Input).value = "media_embeddings"
            self.query_one("#config-rag-retriever-chat-collection", Input).value = "chat_embeddings"
            self.query_one("#config-rag-retriever-notes-collection", Input).value = "notes_embeddings"
            
            # Processor settings
            self.query_one("#config-rag-processor-enable-reranking", Checkbox).value = True
            self.query_one("#config-rag-processor-reranker-model", Input).value = "cross-encoder/ms-marco-MiniLM-L-12-v2"
            self.query_one("#config-rag-processor-reranker-top-k", Input).value = "5"
            self.query_one("#config-rag-processor-deduplication", Input).value = "0.85"
            self.query_one("#config-rag-processor-max-context", Input).value = "4096"
            self.query_one("#config-rag-processor-combination", Select).value = "weighted"
            
            # Generator settings
            self.query_one("#config-rag-generator-model", Input).value = ""
            self.query_one("#config-rag-generator-temperature", Input).value = "0.7"
            self.query_one("#config-rag-generator-max-tokens", Input).value = "1024"
            self.query_one("#config-rag-generator-streaming", Checkbox).value = True
            self.query_one("#config-rag-generator-chunk-size", Input).value = "10"
            
            # Chunking settings
            self.query_one("#config-rag-chunk-size", Input).value = "512"
            self.query_one("#config-rag-chunk-overlap", Input).value = "128"
            self.query_one("#config-rag-chunk-method", Select).value = "fixed"
            
            # ChromaDB settings
            self.query_one("#config-rag-chroma-persist-dir", Input).value = ""
            self.query_one("#config-rag-chroma-prefix", Input).value = "tldw_rag"
            self.query_one("#config-rag-chroma-embedding-model", Input).value = "all-MiniLM-L6-v2"
            self.query_one("#config-rag-chroma-dimension", Input).value = "384"
            self.query_one("#config-rag-chroma-distance", Select).value = "cosine"
            
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
    
    async def _save_chat_config_form(self) -> None:
        """Save chat configuration form."""
        try:
            saved_count = 0
            
            # Save chat defaults
            if save_setting_to_cli_config("chat_defaults", "provider", self.query_one("#config-chat-provider", Select).value):
                saved_count += 1
            if save_setting_to_cli_config("chat_defaults", "model", self.query_one("#config-chat-model", Input).value):
                saved_count += 1
            if save_setting_to_cli_config("chat_defaults", "system_prompt", self.query_one("#config-chat-system-prompt", TextArea).text):
                saved_count += 1
            if save_setting_to_cli_config("chat_defaults", "temperature", float(self.query_one("#config-chat-temperature", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("chat_defaults", "top_p", float(self.query_one("#config-chat-top-p", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("chat_defaults", "min_p", float(self.query_one("#config-chat-min-p", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("chat_defaults", "top_k", int(self.query_one("#config-chat-top-k", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("chat_defaults", "strip_thinking_tags", self.query_one("#config-chat-strip-thinking", Checkbox).value):
                saved_count += 1
            if save_setting_to_cli_config("chat_defaults", "use_enhanced_window", self.query_one("#config-chat-enhanced-window", Checkbox).value):
                saved_count += 1
            
            # Save chat.images settings
            if save_setting_to_cli_config("chat.images", "enabled", self.query_one("#config-chat-images-enabled", Checkbox).value):
                saved_count += 1
            if save_setting_to_cli_config("chat.images", "show_attach_button", self.query_one("#config-chat-images-show-button", Checkbox).value):
                saved_count += 1
            if save_setting_to_cli_config("chat.images", "default_render_mode", self.query_one("#config-chat-images-render-mode", Select).value):
                saved_count += 1
            if save_setting_to_cli_config("chat.images", "max_size_mb", float(self.query_one("#config-chat-images-max-size", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("chat.images", "auto_resize", self.query_one("#config-chat-images-auto-resize", Checkbox).value):
                saved_count += 1
            if save_setting_to_cli_config("chat.images", "resize_max_dimension", int(self.query_one("#config-chat-images-max-dimension", Input).value)):
                saved_count += 1
            if save_setting_to_cli_config("chat.images", "save_location", self.query_one("#config-chat-images-save-location", Input).value):
                saved_count += 1
            
            # Parse and save supported formats
            formats_str = self.query_one("#config-chat-images-formats", Input).value
            formats_list = [f.strip() for f in formats_str.split(",") if f.strip()]
            if save_setting_to_cli_config("chat.images", "supported_formats", formats_list):
                saved_count += 1
            
            # Update internal config
            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
            
            if saved_count > 0:
                self.app_instance.notify(f"Chat configuration saved! ({saved_count} settings updated)")
            else:
                self.app_instance.notify("No chat settings were updated", severity="warning")
                
        except ValueError as e:
            self.app_instance.notify(f"Invalid value: {e}", severity="error")
        except Exception as e:
            self.app_instance.notify(f"Error saving chat config: {e}", severity="error")
    
    async def _reset_chat_config_form(self) -> None:
        """Reset chat configuration form to defaults."""
        try:
            self.query_one("#config-chat-provider", Select).value = "DeepSeek"
            self.query_one("#config-chat-model", Input).value = "deepseek-chat"
            self.query_one("#config-chat-system-prompt", TextArea).text = "You are a helpful AI assistant."
            self.query_one("#config-chat-temperature", Input).value = "0.6"
            self.query_one("#config-chat-top-p", Input).value = "0.95"
            self.query_one("#config-chat-min-p", Input).value = "0.05"
            self.query_one("#config-chat-top-k", Input).value = "50"
            self.query_one("#config-chat-strip-thinking", Checkbox).value = True
            self.query_one("#config-chat-enhanced-window", Checkbox).value = False
            
            # Reset image settings
            self.query_one("#config-chat-images-enabled", Checkbox).value = True
            self.query_one("#config-chat-images-show-button", Checkbox).value = True
            self.query_one("#config-chat-images-render-mode", Select).value = "auto"
            self.query_one("#config-chat-images-max-size", Input).value = "10.0"
            self.query_one("#config-chat-images-auto-resize", Checkbox).value = True
            self.query_one("#config-chat-images-max-dimension", Input).value = "2048"
            self.query_one("#config-chat-images-save-location", Input).value = "~/Downloads"
            self.query_one("#config-chat-images-formats", Input).value = ".png, .jpg, .jpeg, .gif, .webp, .bmp"
            
            self.app_instance.notify("Chat configuration reset to defaults!")
        except Exception as e:
            self.app_instance.notify(f"Error resetting chat config: {e}", severity="error")
    
    async def _save_character_config_form(self) -> None:
        """Save character configuration form."""
        try:
            save_setting_to_cli_config("character_defaults", "provider", self.query_one("#config-character-provider", Select).value)
            save_setting_to_cli_config("character_defaults", "model", self.query_one("#config-character-model", Input).value)
            save_setting_to_cli_config("character_defaults", "system_prompt", self.query_one("#config-character-system-prompt", TextArea).text)
            save_setting_to_cli_config("character_defaults", "temperature", float(self.query_one("#config-character-temperature", Input).value))
            save_setting_to_cli_config("character_defaults", "top_p", float(self.query_one("#config-character-top-p", Input).value))
            save_setting_to_cli_config("character_defaults", "min_p", float(self.query_one("#config-character-min-p", Input).value))
            save_setting_to_cli_config("character_defaults", "top_k", int(self.query_one("#config-character-top-k", Input).value))
            
            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
            self.app_instance.notify("Character configuration saved!")
        except ValueError as e:
            self.app_instance.notify(f"Invalid value: {e}", severity="error")
        except Exception as e:
            self.app_instance.notify(f"Error saving character config: {e}", severity="error")
    
    async def _reset_character_config_form(self) -> None:
        """Reset character configuration form to defaults."""
        try:
            self.query_one("#config-character-provider", Select).value = "Anthropic"
            self.query_one("#config-character-model", Input).value = "claude-3-haiku-20240307"
            self.query_one("#config-character-system-prompt", TextArea).text = "You are roleplaying as a witty pirate captain."
            self.query_one("#config-character-temperature", Input).value = "0.8"
            self.query_one("#config-character-top-p", Input).value = "0.9"
            self.query_one("#config-character-min-p", Input).value = "0.0"
            self.query_one("#config-character-top-k", Input).value = "100"
            
            self.app_instance.notify("Character configuration reset to defaults!")
        except Exception as e:
            self.app_instance.notify(f"Error resetting character config: {e}", severity="error")
    
    async def _save_notes_config_form(self) -> None:
        """Save notes configuration form."""
        try:
            save_setting_to_cli_config("notes", "sync_directory", self.query_one("#config-notes-sync-directory", Input).value)
            save_setting_to_cli_config("notes", "auto_sync_enabled", self.query_one("#config-notes-auto-sync", Checkbox).value)
            save_setting_to_cli_config("notes", "sync_on_close", self.query_one("#config-notes-sync-on-close", Checkbox).value)
            save_setting_to_cli_config("notes", "conflict_resolution", self.query_one("#config-notes-conflict-resolution", Select).value)
            save_setting_to_cli_config("notes", "sync_direction", self.query_one("#config-notes-sync-direction", Select).value)
            
            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
            self.app_instance.notify("Notes configuration saved!")
        except Exception as e:
            self.app_instance.notify(f"Error saving notes config: {e}", severity="error")
    
    async def _reset_notes_config_form(self) -> None:
        """Reset notes configuration form to defaults."""
        try:
            self.query_one("#config-notes-sync-directory", Input).value = "~/Documents/Notes"
            self.query_one("#config-notes-auto-sync", Checkbox).value = False
            self.query_one("#config-notes-sync-on-close", Checkbox).value = False
            self.query_one("#config-notes-conflict-resolution", Select).value = "newer_wins"
            self.query_one("#config-notes-sync-direction", Select).value = "bidirectional"
            
            self.app_instance.notify("Notes configuration reset to defaults!")
        except Exception as e:
            self.app_instance.notify(f"Error resetting notes config: {e}", severity="error")
    
    async def _save_tts_config_form(self) -> None:
        """Save TTS configuration form."""
        try:
            save_setting_to_cli_config("app_tts", "OPENAI_API_KEY_fallback", self.query_one("#config-tts-openai-key", Input).value)
            save_setting_to_cli_config("app_tts", "KOKORO_ONNX_MODEL_PATH_DEFAULT", self.query_one("#config-tts-kokoro-model-path", Input).value)
            save_setting_to_cli_config("app_tts", "KOKORO_ONNX_VOICES_JSON_DEFAULT", self.query_one("#config-tts-kokoro-voices-path", Input).value)
            save_setting_to_cli_config("app_tts", "KOKORO_DEVICE_DEFAULT", self.query_one("#config-tts-kokoro-device", Select).value)
            save_setting_to_cli_config("app_tts", "ELEVENLABS_API_KEY_fallback", self.query_one("#config-tts-elevenlabs-key", Input).value)
            
            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
            self.app_instance.notify("TTS configuration saved!")
        except Exception as e:
            self.app_instance.notify(f"Error saving TTS config: {e}", severity="error")
    
    async def _reset_tts_config_form(self) -> None:
        """Reset TTS configuration form to defaults."""
        try:
            self.query_one("#config-tts-openai-key", Input).value = "sk-..."
            self.query_one("#config-tts-kokoro-model-path", Input).value = "path/to/kokoro-v0_19.onnx"
            self.query_one("#config-tts-kokoro-voices-path", Input).value = "path/to/voices.json"
            self.query_one("#config-tts-kokoro-device", Select).value = "cpu"
            self.query_one("#config-tts-elevenlabs-key", Input).value = "el-..."
            
            self.app_instance.notify("TTS configuration reset to defaults!")
        except Exception as e:
            self.app_instance.notify(f"Error resetting TTS config: {e}", severity="error")
    
    async def _save_embedding_config_form(self) -> None:
        """Save embedding configuration form."""
        try:
            save_setting_to_cli_config("embedding_config", "default_model_id", self.query_one("#config-embedding-default-model", Select).value)
            save_setting_to_cli_config("embedding_config", "default_llm_for_contextualization", self.query_one("#config-embedding-default-llm", Input).value)
            
            # Save model-specific settings
            embedding_config = self.config_data.get("embedding_config", {})
            models = embedding_config.get("models", {})
            
            for model_id in models.keys():
                try:
                    save_setting_to_cli_config(f"embedding_config.models.{model_id}", "provider", 
                                             self.query_one(f"#config-embedding-{model_id}-provider", Input).value)
                    save_setting_to_cli_config(f"embedding_config.models.{model_id}", "model_name_or_path", 
                                             self.query_one(f"#config-embedding-{model_id}-name", Input).value)
                    save_setting_to_cli_config(f"embedding_config.models.{model_id}", "dimension", 
                                             int(self.query_one(f"#config-embedding-{model_id}-dimension", Input).value))
                    
                    # Save API key if it's an OpenAI model
                    if models[model_id].get("provider") == "openai":
                        api_key_widget = self.query_one(f"#config-embedding-{model_id}-api-key", Input)
                        if api_key_widget:
                            save_setting_to_cli_config(f"embedding_config.models.{model_id}", "api_key", api_key_widget.value)
                except Exception:
                    pass  # Skip if widget not found
            
            self.config_data = load_cli_config_and_ensure_existence(force_reload=True)
            self.app_instance.notify("Embedding configuration saved!")
        except Exception as e:
            self.app_instance.notify(f"Error saving embedding config: {e}", severity="error")
    
    async def _reset_embedding_config_form(self) -> None:
        """Reset embedding configuration form to defaults."""
        try:
            self.query_one("#config-embedding-default-model", Select).value = "e5-small-v2"
            self.query_one("#config-embedding-default-llm", Input).value = "gpt-3.5-turbo"
            
            # Reset model-specific settings would be complex, so just notify
            self.app_instance.notify("Embedding configuration partially reset. Model-specific settings retained.")
        except Exception as e:
            self.app_instance.notify(f"Error resetting embedding config: {e}", severity="error")
    
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
