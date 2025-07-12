# settings_sidebar.py
# Description: settings sidebar widget with enhanced UX features
#
# Imports
#
# 3rd-Party Imports
import logging

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Container
from textual.widgets import Static, Select, TextArea, Input, Collapsible, Button, Checkbox, ListView, Switch, Label
#
# Local Imports
from ..config import get_cli_providers_and_models

#
#######################################################################################################################
#
# Functions:

# Sidebar visual constants ---------------------------------------------------
SIDEBAR_WIDTH = "30%"


def create_settings_sidebar(id_prefix: str, config: dict) -> ComposeResult:
    """Yield the widgets for the settings sidebar with enhanced UX.

    Enhanced features:
        1. Mode toggle (Basic/Advanced) at the top
        2. Search functionality for settings
        3. Better organization with prominent RAG panel
        4. All existing functionality preserved
    """
    sidebar_id = f"{id_prefix}-left-sidebar"

    with VerticalScroll(id=sidebar_id, classes="sidebar"):
        # -------------------------------------------------------------------
        # Retrieve defaults / provider information
        # -------------------------------------------------------------------
        defaults = config.get(f"{id_prefix}_defaults", config.get("chat_defaults", {}))
        providers_models = get_cli_providers_and_models()
        logging.info(
            "Sidebar %s: Received providers_models. Count: %d. Keys: %s",
            id_prefix,
            len(providers_models),
            list(providers_models.keys()),
        )

        available_providers = list(providers_models.keys())
        default_provider: str = defaults.get(
            "provider", available_providers[0] if available_providers else ""
        )
        default_model: str = defaults.get("model", "")
        default_system_prompt: str = defaults.get("system_prompt", "")
        default_temp = str(defaults.get("temperature", 0.7))
        default_top_p = str(defaults.get("top_p", 0.95))
        default_min_p = str(defaults.get("min_p", 0.05))
        default_top_k = str(defaults.get("top_k", 50))

        # -------------------------------------------------------------------
        # Enhanced Header with Mode Toggle and Search
        # -------------------------------------------------------------------
        yield Static("Chat Settings", classes="sidebar-title")
        
        # Mode toggle container
        with Horizontal(id=f"{id_prefix}-mode-toggle-container", classes="mode-toggle-container"):
            yield Label("Basic", classes="mode-label")
            yield Switch(
                value=False,  # False = Basic, True = Advanced
                id=f"{id_prefix}-settings-mode-toggle",
                classes="settings-mode-toggle"
            )
            yield Label("Advanced", classes="mode-label")

        # -------------------------------------------------------------------
        # Quick Settings (Always visible)
        # -------------------------------------------------------------------
        with Collapsible(title="Quick Settings", collapsed=False, id=f"{id_prefix}-quick-settings", classes="settings-collapsible basic-mode advanced-mode"):
            yield Static("Provider & Model", classes="sidebar-label")
            provider_options = [(provider, provider) for provider in available_providers]
            yield Select(
                options=provider_options,
                prompt="Select Provider…",
                allow_blank=False,
                id=f"{id_prefix}-api-provider",
                value=default_provider,
            )

            initial_models = providers_models.get(default_provider, [])
            model_options = [(model, model) for model in initial_models]
            current_model_value = (
                default_model if default_model in initial_models else (initial_models[0] if initial_models else None)
            )
            yield Select(
                options=model_options,
                prompt="Select Model…",
                allow_blank=True,
                id=f"{id_prefix}-api-model",
                value=current_model_value,
            )

            yield Static("Temperature", classes="sidebar-label")
            yield Input(
                placeholder="e.g., 0.7",
                id=f"{id_prefix}-temperature",
                value=default_temp,
                classes="sidebar-input",
            )
            
            yield Static("System Prompt", classes="sidebar-label")
            system_prompt_classes = "sidebar-textarea"
            if id_prefix == "chat":
                system_prompt_classes += " chat-system-prompt-styling"
            yield TextArea(
                id=f"{id_prefix}-system-prompt",
                text=default_system_prompt,
                classes=system_prompt_classes,
            )
            
            # Streaming toggle
            yield Checkbox(
                "Enable Streaming",
                id=f"{id_prefix}-streaming-enabled-checkbox",
                value=True,  # Default to enabled for better UX
                classes="streaming-toggle",
                tooltip="Enable/disable streaming responses. When disabled, responses appear all at once."
            )
            
            # Show attach button toggle (only for chat)
            if id_prefix == "chat":
                from ..config import get_cli_setting
                show_attach_button = get_cli_setting("chat.images", "show_attach_button", True)
                yield Checkbox(
                    "Show Attach File Button",
                    id="chat-show-attach-button-checkbox",
                    value=show_attach_button,
                    classes="attach-button-toggle",
                    tooltip="Show/hide the file attachment button in chat"
                )
            
            # User Identifier for personalization
            yield Static("User Identifier", classes="sidebar-label")
            yield Input(
                id=f"{id_prefix}-llm-user-identifier", 
                placeholder="e.g., user-123",
                classes="sidebar-input",
                tooltip="Optional user identifier for personalizing context"
            )

        # -------------------------------------------------------------------
        # RAG Settings (Prominent Panel - Always visible)
        # -------------------------------------------------------------------
        with Collapsible(title="🔍 RAG Settings", collapsed=True, id=f"{id_prefix}-rag-panel", classes="settings-collapsible rag-settings-panel basic-mode advanced-mode"):
            # Main RAG toggle
            yield Checkbox(
                "Enable RAG",
                id=f"{id_prefix}-rag-enable-checkbox",
                value=False,
                classes="rag-enable-toggle"
            )
            
            # RAG preset selection
            yield Static("RAG Preset", classes="sidebar-label")
            yield Select(
                options=[
                    ("None", "none"),
                    ("Light (BM25)", "light"),
                    ("Full (Embeddings)", "full"),
                    ("Custom", "custom")
                ],
                value="none",
                id=f"{id_prefix}-rag-preset",
                prompt="Select preset...",
                classes="rag-preset-select sidebar-select"
            )
            
            # Search scope
            yield Static("Search Scope", classes="sidebar-label")
            with Container(classes="rag-scope-options"):
                yield Checkbox("Media Items", id=f"{id_prefix}-rag-search-media-checkbox", value=True)
                yield Checkbox("Conversations", id=f"{id_prefix}-rag-search-conversations-checkbox", value=False)
                yield Checkbox("Notes", id=f"{id_prefix}-rag-search-notes-checkbox", value=False)
            
            # Keyword filter
            yield Static("Filter by Keywords", classes="sidebar-label")
            yield Input(
                id=f"{id_prefix}-rag-keyword-filter",
                placeholder="Enter keywords (comma-separated)",
                classes="sidebar-input rag-keyword-filter"
            )
            
            # Basic RAG parameters
            yield Static("Top Results", classes="sidebar-label")
            yield Input(
                id=f"{id_prefix}-rag-top-k",
                value="5",
                placeholder="Number of results",
                classes="sidebar-input"
            )
            
            # Query Expansion Settings
            yield Static("", classes="sidebar-separator")
            yield Static("Query Expansion", classes="sidebar-label")
            yield Checkbox(
                "Enable Query Expansion",
                id=f"{id_prefix}-rag-query-expansion-checkbox",
                value=False,
                classes="rag-query-expansion-toggle"
            )
            
            yield Static("Expansion Method", classes="sidebar-label")
            yield Select(
                options=[
                    ("Remote LLM", "llm"),
                    ("Local Model (Llamafile)", "llamafile"),
                    ("Keywords", "keywords")
                ],
                value="llm",
                id=f"{id_prefix}-rag-expansion-method",
                prompt="Select method...",
                classes="rag-expansion-select sidebar-select"
            )
            
            # Provider & Model selection (shown when Remote LLM is selected)
            yield Static("Expansion Provider", classes="sidebar-label rag-expansion-provider-label")
            rag_provider_options = [(provider, provider) for provider in available_providers]
            yield Select(
                options=rag_provider_options,
                prompt="Select Provider…",
                allow_blank=False,
                id=f"{id_prefix}-rag-expansion-provider",
                value=default_provider,
                classes="rag-expansion-provider sidebar-select"
            )
            
            yield Static("Expansion Model", classes="sidebar-label rag-expansion-llm-label")
            rag_initial_models = providers_models.get(default_provider, [])
            rag_model_options = [(model, model) for model in rag_initial_models]
            yield Select(
                options=rag_model_options,
                prompt="Select Model…",
                allow_blank=True,
                id=f"{id_prefix}-rag-expansion-llm-model",
                value=rag_initial_models[0] if rag_initial_models else Select.BLANK,
                classes="rag-expansion-llm-model sidebar-select"
            )
            
            # Local model selection (shown when Local Model is selected)
            yield Static("Llamafile Model", classes="sidebar-label rag-expansion-local-label hidden")
            yield Input(
                placeholder="e.g., Qwen3-0.6B-Q6_K.gguf",
                value="Qwen3-0.6B-Q6_K.gguf",
                id=f"{id_prefix}-rag-expansion-local-model",
                classes="sidebar-input rag-expansion-local-model hidden"
            )
            
            yield Static("Max Sub-queries", classes="sidebar-label")
            yield Input(
                id=f"{id_prefix}-rag-expansion-max-queries",
                value="3",
                placeholder="1-5",
                classes="sidebar-input"
            )
            
            # Chunking Settings
            yield Static("", classes="sidebar-separator")
            yield Static("Chunking Settings", classes="sidebar-label")
            
            yield Static("Chunk Type", classes="sidebar-label")
            yield Select(
                [
                    ("Words", "words"),
                    ("Sentences", "sentences"),
                    ("Paragraphs", "paragraphs")
                ],
                id=f"{id_prefix}-rag-chunk-type",
                value="words",
                prompt="Select chunk type...",
                allow_blank=False,
                classes="sidebar-select"
            )
            
            yield Static("Chunk Size", classes="sidebar-label")
            yield Input(
                id=f"{id_prefix}-rag-chunk-size",
                value="400",
                placeholder="e.g., 400",
                classes="sidebar-input"
            )
            
            yield Static("Chunk Overlap", classes="sidebar-label")
            yield Input(
                id=f"{id_prefix}-rag-chunk-overlap",
                value="100",
                placeholder="e.g., 100",
                classes="sidebar-input"
            )
            
            # Advanced RAG Settings Separator
            yield Static("", classes="sidebar-separator")
            yield Static("Advanced RAG Settings", classes="sidebar-label sidebar-section-header")
            
            # Re-ranking Options
            yield Checkbox(
                "Enable Re-ranking",
                id=f"{id_prefix}-rag-rerank-enable-checkbox",
                value=True,
                classes="sidebar-checkbox"
            )
            
            yield Static("Re-ranker Model", classes="sidebar-label")
            yield Select(
                [
                    ("FlashRank (Local)", "flashrank"),
                    ("Cohere Rerank", "cohere"),
                    ("None", "none")
                ],
                id=f"{id_prefix}-rag-reranker-model",
                value="flashrank",
                prompt="Select Re-ranker...",
                allow_blank=False
            )
            
            yield Static("Max Context Length (chars)", classes="sidebar-label")
            yield Input(
                id=f"{id_prefix}-rag-max-context-length",
                value="10000",
                placeholder="e.g., 10000",
                classes="sidebar-input"
            )
            
            yield Checkbox(
                "Include Context Metadata",
                id=f"{id_prefix}-rag-include-metadata-checkbox",
                value=True,
                classes="sidebar-checkbox"
            )

        # -------------------------------------------------------------------
        # Advanced Model Parameters (Hidden in Basic Mode)
        # -------------------------------------------------------------------
        with Collapsible(title="Model Parameters", collapsed=True, id=f"{id_prefix}-model-params", classes="settings-collapsible advanced-mode advanced-only"):
            yield Static("Top P", classes="sidebar-label")
            yield Input(
                placeholder="e.g., 0.95",
                id=f"{id_prefix}-top-p",
                value=default_top_p,
                classes="sidebar-input",
            )

            yield Static("Min P", classes="sidebar-label")
            yield Input(
                placeholder="e.g., 0.05",
                id=f"{id_prefix}-min-p",
                value=default_min_p,
                classes="sidebar-input",
            )

            yield Static("Top K", classes="sidebar-label")
            yield Input(
                placeholder="e.g., 50",
                id=f"{id_prefix}-top-k",
                value=default_top_k,
                classes="sidebar-input",
            )
            
            # Token Settings
            yield Static("Max Tokens", classes="sidebar-label")
            yield Input(id=f"{id_prefix}-llm-max-tokens", value="2048", placeholder="e.g., 1024",
                        classes="sidebar-input")
            yield Static("Custom Token Limit (Display)", classes="sidebar-label")
            yield Input(id=f"{id_prefix}-custom-token-limit", value="12888", placeholder="0 = use Max Tokens",
                        classes="sidebar-input", 
                        tooltip="Set a custom limit for the token counter display. 0 = use Max Tokens value above.")
            yield Checkbox("Fixed Tokens (Kobold)", id=f"{id_prefix}-llm-fixed-tokens-kobold", value=False)
            
            # Generation Settings
            yield Static("Seed", classes="sidebar-label")
            yield Input(id=f"{id_prefix}-llm-seed", value="0", placeholder="e.g., 42", classes="sidebar-input")
            yield Static("Stop Sequences (comma-sep)", classes="sidebar-label")
            yield Input(id=f"{id_prefix}-llm-stop", placeholder="e.g., <|endoftext|>,<|eot_id|>",
                        classes="sidebar-input")
            yield Static("Response Format", classes="sidebar-label")
            yield Select(options=[("text", "text"), ("json_object", "json_object")],
                         id=f"{id_prefix}-llm-response-format", value="text", allow_blank=False)

        # -------------------------------------------------------------------
        # Conversation Management (Always visible)
        # -------------------------------------------------------------------
        with Collapsible(title="Conversations", collapsed=True, id=f"{id_prefix}-conversations", classes="settings-collapsible basic-mode advanced-mode"):
            yield Input(
                id=f"{id_prefix}-conversation-search-bar",
                placeholder="Search by title...",
                classes="sidebar-input"
            )
            yield Input(
                id=f"{id_prefix}-conversation-keyword-search-bar",
                placeholder="Search by content keywords...",
                classes="sidebar-input"
            )
            yield Input(
                id=f"{id_prefix}-conversation-tags-search-bar",
                placeholder="Filter by tags (comma-separated)...",
                classes="sidebar-input"
            )
            yield Checkbox(
                "Include Character Chats",
                id=f"{id_prefix}-conversation-search-include-character-checkbox"
            )
            yield Select(
                [],
                id=f"{id_prefix}-conversation-search-character-filter-select",
                allow_blank=True,
                prompt="Filter by Character...",
                classes="sidebar-select"
            )
            yield Checkbox(
                "All Characters",
                id=f"{id_prefix}-conversation-search-all-characters-checkbox",
                value=True
            )
            yield ListView(
                id=f"{id_prefix}-conversation-search-results-list",
                classes="sidebar-listview"
            )
            yield Button(
                "Load Selected Chat",
                id=f"{id_prefix}-conversation-load-selected-button",
                variant="default",
                classes="sidebar-button",
                tooltip="Load the selected conversation"
            )

        # -------------------------------------------------------------------
        # Advanced Settings (Hidden in Basic Mode)
        # -------------------------------------------------------------------
        with Collapsible(title="Advanced Settings", collapsed=True, id=f"{id_prefix}-advanced-settings", classes="settings-collapsible advanced-mode advanced-only"):
            # More token parameters
            yield Static("N (Completions)", classes="sidebar-label")
            yield Input(id=f"{id_prefix}-llm-n", value="1", placeholder="e.g., 1", classes="sidebar-input")
            yield Checkbox("Logprobs", id=f"{id_prefix}-llm-logprobs", value=False)
            yield Static("Top Logprobs", classes="sidebar-label")
            yield Input(id=f"{id_prefix}-llm-top-logprobs", value="0", placeholder="e.g., 5",
                        classes="sidebar-input")
            yield Static("Logit Bias (JSON)", classes="sidebar-label")
            yield TextArea(id=f"{id_prefix}-llm-logit-bias", text="{}", classes="sidebar-textarea")
            yield Static("Presence Penalty", classes="sidebar-label")
            yield Input(id=f"{id_prefix}-llm-presence-penalty", value="0.0", placeholder="e.g., 0.0 to 2.0",
                        classes="sidebar-input")
            yield Static("Frequency Penalty", classes="sidebar-label")
            yield Input(id=f"{id_prefix}-llm-frequency-penalty", value="0.0", placeholder="e.g., 0.0 to 2.0",
                        classes="sidebar-input")

        # -------------------------------------------------------------------
        # Tools & Templates (Hidden in Basic Mode)
        # -------------------------------------------------------------------
        with Collapsible(title="Tools & Templates", collapsed=True, id=f"{id_prefix}-tools", classes="settings-collapsible advanced-mode advanced-only"):
            yield Static("Tool Usage", classes="sidebar-label")
            yield TextArea(id=f"{id_prefix}-llm-tools", text="[]", classes="sidebar-textarea")
            yield Static("Tool Choice", classes="sidebar-label")
            yield Input(id=f"{id_prefix}-llm-tool-choice", placeholder="e.g., auto, none, or specific tool",
                        classes="sidebar-input")
            
            yield Static("Chat Templates", classes="sidebar-label")
            yield Input(
                id=f"{id_prefix}-template-search-input",
                placeholder="Search templates...",
                classes="sidebar-input"
            )
            template_list_view = ListView(
                id=f"{id_prefix}-template-list-view",
                classes="sidebar-listview"
            )
            template_list_view.styles.height = 7
            yield template_list_view
            yield Button(
                "Apply Template",
                id=f"{id_prefix}-apply-template-button",
                classes="sidebar-button"
            )

#
# End of settings_sidebar.py
#######################################################################################################################