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
from textual.message import Message
#
# Local Imports
from ..config import get_cli_providers_and_models
from ..Widgets.Media_Creation.swarmui_widget import SwarmUIWidget

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
# Functions:

# Sidebar visual constants ---------------------------------------------------
SIDEBAR_WIDTH = "30%"


def get_pipeline_description(pipeline_id: str) -> str:
    """Get a description for a pipeline configuration."""
    descriptions = {
        "none": "Manual configuration mode. Set all RAG parameters individually.",
        "speed_optimized_v2": "Optimized for fast response times using BM25 search with minimal processing overhead.",
        "high_accuracy": "Semantic search with embeddings, re-ranking, and comprehensive processing for best accuracy.",
        "hybrid": "Balanced approach combining BM25 and semantic search for good performance and accuracy.",
        "research_focused_v2": "Advanced pipeline with query expansion and multi-stage retrieval for research tasks.",
        "adaptive_v2": "Dynamically adjusts search strategy based on query complexity and available resources.",
        "plain": "Simple keyword-based search using FTS5 full-text search.",
        "semantic": "Pure semantic search using embeddings for conceptual matching.",
    }
    
    # Try to get description from pipeline config if available
    if PIPELINE_INTEGRATION_AVAILABLE and get_pipeline:
        try:
            pipeline_config = get_pipeline(pipeline_id)
            if pipeline_config and 'description' in pipeline_config:
                return pipeline_config['description']
        except:
            pass
    
    return descriptions.get(pipeline_id, f"Pipeline configuration: {pipeline_id}")


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
        # Quick Actions Bar at the top
        # -------------------------------------------------------------------
        with Horizontal(classes="quick-actions-bar"):
            yield Button("➕", id=f"{id_prefix}-expand-all", classes="quick-action-btn", tooltip="Expand all sections")
            yield Button("➖", id=f"{id_prefix}-collapse-all", classes="quick-action-btn", tooltip="Collapse all sections")
            yield Button("🔄", id=f"{id_prefix}-reset-settings", classes="quick-action-btn", tooltip="Reset to defaults")
        
        # Search bar for filtering settings
        yield Input(
            placeholder="🔍 Search settings...",
            id=f"{id_prefix}-settings-search",
            classes="sidebar-search-input"
        )
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

        # -------------------------------------------------------------------
        # ESSENTIAL GROUP - Always visible, priority high
        # -------------------------------------------------------------------
        with Container(classes="settings-group primary-group"):
            yield Static("ESSENTIAL", classes="group-header")
            
            # Quick Settings (Always visible)
            with Collapsible(title="🎯 Quick Settings", collapsed=False, id=f"{id_prefix}-quick-settings", classes="settings-collapsible priority-high basic-mode advanced-mode"):
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
                default_model if default_model in initial_models else (initial_models[0] if initial_models else Select.BLANK)
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
                
                # Show dictation button toggle
                show_mic_button = get_cli_setting("chat.voice", "show_mic_button", True)
                yield Checkbox(
                    "Show Dictation button",
                    id="chat-show-dictation-button-checkbox",
                    value=show_mic_button,
                    classes="dictation-button-toggle",
                    tooltip="Show/hide the dictation/microphone button in chat"
                )
            
            # User Identifier for personalization
            yield Static("User Identifier", classes="sidebar-label")
            yield Input(
                id=f"{id_prefix}-llm-user-identifier", 
                placeholder="e.g., user-123",
                classes="sidebar-input",
                tooltip="Optional user identifier for personalizing context"
            )
            
            # Advanced Settings toggle checkbox
            yield Checkbox(
                "Advanced Settings",
                id=f"{id_prefix}-settings-mode-toggle",
                value=False,  # Unchecked by default (Basic mode)
                classes="advanced-settings-checkbox",
                tooltip="Enable advanced settings and options"
            )

            # Current Chat Details - also in essential group
            with Collapsible(title="💬 Current Chat", collapsed=False, id=f"{id_prefix}-chat-details-collapsible", classes="settings-collapsible priority-high basic-mode advanced-mode"):
                # "New Chat" Buttons
                yield Button(
                "New Temp Chat",
                id=f"{id_prefix}-new-temp-chat-button",
                classes="sidebar-button",
                variant="primary"
            )
            yield Button(
                "New Chat",
                id=f"{id_prefix}-new-conversation-button",
                classes="sidebar-button"
            )
            yield Label("Conversation ID:", classes="sidebar-label", id=f"{id_prefix}-uuid-label-displayonly")
            yield Input(
                id=f"{id_prefix}-conversation-uuid-display",
                value="Temp Chat",
                disabled=True,
                classes="sidebar-input"
            )

            yield Label("Chat Title:", classes="sidebar-label", id=f"{id_prefix}-title-label-displayonly")
            yield Input(
                id=f"{id_prefix}-conversation-title-input",
                placeholder="Chat title...",
                disabled=True,
                classes="sidebar-input"
            )
            yield Label("Keywords (comma-sep):", classes="sidebar-label", id=f"{id_prefix}-keywords-label-displayonly")
            yield TextArea(
                "",
                id=f"{id_prefix}-conversation-keywords-input",
                classes="sidebar-textarea chat-keywords-textarea",
                disabled=True
            )
            # Button to save METADATA
            yield Button(
                "Save Details",
                id=f"{id_prefix}-save-conversation-details-button",
                classes="sidebar-button save-details-button",
                variant="primary",
                disabled=True
            )
            # Button to make an EPHEMERAL chat PERSISTENT
            yield Button(
                "Save Temp Chat",
                id=f"{id_prefix}-save-current-chat-button",
                classes="sidebar-button save-chat-button",
                variant="success",
                disabled=False
            )
            
            # Clone chat button
            yield Button(
                "🔄 Clone Current Chat",
                id=f"{id_prefix}-clone-current-chat-button",
                classes="sidebar-button clone-chat-button",
                variant="default",
                tooltip="Create a copy of the current chat to explore different paths"
            )

            # Convert to note button
            yield Button(
                "📋 Convert to Note",
                id=f"{id_prefix}-convert-to-note-button",
                classes="sidebar-button convert-to-note-button",
                variant="default"
            )

            # Strip thinking tags checkbox
            initial_strip_value = config.get("chat_defaults", {}).get("strip_thinking_tags", True)
            yield Checkbox(
                "Strip Thinking Tags",
                value=initial_strip_value,
                id=f"{id_prefix}-strip-thinking-tags-checkbox",
                classes="sidebar-checkbox"
            )

        # -------------------------------------------------------------------
        # FEATURES GROUP - Secondary importance
        # -------------------------------------------------------------------
        yield Static(classes="sidebar-section-divider")
        
        with Container(classes="settings-group secondary-group"):
            yield Static("FEATURES", classes="group-header")
            
            # Image Generation (only for chat tab)
            if id_prefix == "chat":
                with Collapsible(title="🎨 Image Generation", collapsed=True, id=f"{id_prefix}-image-generation-collapsible", classes="settings-collapsible basic-mode advanced-mode"):
                    yield SwarmUIWidget(id=f"{id_prefix}-swarmui-widget")

            # RAG Settings (Prominent Panel)
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
            
            # Pipeline selection (for pre-configured RAG pipelines)
            yield Static("RAG Pipeline Configuration", classes="sidebar-label")
            
            # Build pipeline options - organized by use case
            pipeline_options = []
            
            # Add "No Pipeline" option first for manual configuration
            pipeline_options.append(("🔧 Manual Configuration (No Pipeline)", "none"))
            
            # Add built-in optimized pipelines
            pipeline_options.extend([
                ("⚡ Speed Optimized", "speed_optimized_v2"),
                ("🎯 High Accuracy", "high_accuracy"),
                ("🔀 Balanced (Default)", "hybrid"),
                ("🔬 Research Mode", "research_focused_v2"),
                ("🤖 Adaptive", "adaptive_v2"),
            ])
            
            # Load additional pipelines from TOML if available
            if PIPELINE_INTEGRATION_AVAILABLE:
                try:
                    pipeline_manager = get_pipeline_manager()
                    all_pipelines = pipeline_manager.list_available_pipelines()
                    
                    # Filter and categorize custom pipelines
                    custom_pipelines = []
                    for pipeline in all_pipelines:
                        # Skip built-in pipelines we already added
                        if pipeline["id"] in ["speed_optimized_v2", "high_accuracy", "hybrid", 
                                              "research_focused_v2", "adaptive_v2", "plain", 
                                              "semantic", "full"]:
                            continue
                            
                        if pipeline["enabled"]:
                            # Determine emoji based on tags or type
                            emoji = "🛠️"  # Default custom
                            tags = pipeline.get("tags", [])
                            if "technical" in tags or "documentation" in tags:
                                emoji = "📖"
                            elif "support" in tags or "customer" in tags:
                                emoji = "💬"
                            elif "medical" in tags or "health" in tags:
                                emoji = "🏥"
                            elif "legal" in tags or "compliance" in tags:
                                emoji = "⚖️"
                            elif "academic" in tags or "research" in tags:
                                emoji = "🎓"
                            elif "fast" in tags or "speed" in tags:
                                emoji = "🚀"
                            
                            label = f"{emoji} {pipeline['name']}"
                            custom_pipelines.append((label, pipeline["id"]))
                    
                    # Add separator and custom pipelines if any exist
                    if custom_pipelines:
                        pipeline_options.append(("─" * 20, "separator"))
                        pipeline_options.extend(sorted(custom_pipelines, key=lambda x: x[0]))
                        
                except Exception as e:
                    logging.warning(f"Failed to load pipeline configurations: {e}")
            
            # Add fallback legacy options at the end
            pipeline_options.extend([
                ("─" * 20, "separator"),
                ("📊 Legacy: Plain Search", "plain"),
                ("🧠 Legacy: Semantic Search", "semantic"),
            ])
            
            yield Select(
                options=pipeline_options,
                value="none",  # Default to manual configuration
                id=f"{id_prefix}-rag-search-mode",
                prompt="Select RAG pipeline...",
                classes="rag-pipeline-select sidebar-select"
            )
            
            # Pipeline description display
            yield Static(
                "Select a pipeline to see its configuration",
                id=f"{id_prefix}-rag-pipeline-description",
                classes="sidebar-label rag-pipeline-description"
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
        # Notes (from right sidebar)
        # -------------------------------------------------------------------
        if id_prefix == "chat":
            with Collapsible(title="Notes", collapsed=True, id=f"{id_prefix}-notes-collapsible", classes="settings-collapsible basic-mode advanced-mode"):
                yield Label("Search Notes:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-notes-search-input",
                    placeholder="Search notes...",
                    classes="sidebar-input"
                )
                yield Button(
                    "Search",
                    id=f"{id_prefix}-notes-search-button",
                    classes="sidebar-button"
                )

                notes_list_view = ListView(
                    id=f"{id_prefix}-notes-listview",
                    classes="sidebar-listview"
                )
                notes_list_view.styles.height = 7
                yield notes_list_view

                yield Button(
                    "Load Note",
                    id=f"{id_prefix}-notes-load-button",
                    classes="sidebar-button"
                )
                yield Button(
                    "Create New Note",
                    id=f"{id_prefix}-notes-create-new-button",
                    variant="primary",
                    classes="sidebar-button"
                )

                yield Label("Note Title:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-notes-title-input",
                    placeholder="Note title...",
                    classes="sidebar-input"
                )

                # Expand button above note content
                yield Button(
                    "Expand Notes",
                    id=f"{id_prefix}-notes-expand-button",
                    classes="notes-expand-button sidebar-button"
                )
                
                # Note content label
                yield Label("Note Content:", classes="sidebar-label")
                
                note_content_area = TextArea(
                    id=f"{id_prefix}-notes-content-textarea",
                    classes="sidebar-textarea notes-textarea-normal"
                )
                note_content_area.styles.height = 10
                yield note_content_area

                yield Button(
                    "Save Note",
                    id=f"{id_prefix}-notes-save-button",
                    variant="success",
                    classes="sidebar-button"
                )
                
                yield Button(
                    "Copy Note",
                    id=f"{id_prefix}-notes-copy-button",
                    variant="default",
                    classes="sidebar-button"
                )

        # -------------------------------------------------------------------
        # Prompts (from right sidebar)
        # -------------------------------------------------------------------
        if id_prefix == "chat":
            with Collapsible(title="Prompts", collapsed=True, id=f"{id_prefix}-prompts-collapsible", classes="settings-collapsible basic-mode advanced-mode"):
                yield Label("Search Prompts:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-prompt-search-input",
                    placeholder="Enter search term...",
                    classes="sidebar-input"
                )

                results_list_view = ListView(
                    id=f"{id_prefix}-prompts-listview",
                    classes="sidebar-listview"
                )
                results_list_view.styles.height = 15
                yield results_list_view

                yield Button(
                    "Load Selected Prompt",
                    id=f"{id_prefix}-prompt-load-selected-button",
                    variant="default",
                    classes="sidebar-button"
                )
                yield Label("System Prompt:", classes="sidebar-label")

                system_prompt_display = TextArea(
                    "",
                    id=f"{id_prefix}-prompt-system-display",
                    classes="sidebar-textarea prompt-display-textarea",
                    read_only=True
                )
                system_prompt_display.styles.height = 15
                yield system_prompt_display
                yield Button(
                    "Copy System",
                    id="chat-prompt-copy-system-button",
                    classes="sidebar-button copy-button",
                    disabled=True
                )

                yield Label("User Prompt:", classes="sidebar-label")

                user_prompt_display = TextArea(
                    "",
                    id=f"{id_prefix}-prompt-user-display",
                    classes="sidebar-textarea prompt-display-textarea",
                    read_only=True
                )
                user_prompt_display.styles.height = 15
                yield user_prompt_display
                yield Button(
                    "Copy User",
                    id="chat-prompt-copy-user-button",
                    classes="sidebar-button copy-button",
                    disabled=True
                )

        # -------------------------------------------------------------------
        # ADVANCED GROUP - Hidden by default, technical settings
        # -------------------------------------------------------------------
        yield Static(classes="sidebar-section-divider")
        
        with Container(classes="settings-group advanced-group"):
            yield Static("ADVANCED", classes="group-header")
            
            # Model Parameters
            with Collapsible(title="⚙️ Model Parameters", collapsed=True, id=f"{id_prefix}-model-params", classes="settings-collapsible advanced-mode advanced-only"):
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
        # Character Info (from right sidebar)
        # -------------------------------------------------------------------
        with Collapsible(title="Active Character Info", collapsed=True, id=f"{id_prefix}-active-character-info-collapsible", classes="settings-collapsible basic-mode advanced-mode"):
            if id_prefix == "chat":
                yield Input(
                    id="chat-character-search-input",
                    placeholder="Search all characters...",
                    classes="sidebar-input"
                )
                character_search_results_list = ListView(
                    id="chat-character-search-results-list",
                    classes="sidebar-listview"
                )
                character_search_results_list.styles.height = 7
                yield character_search_results_list
                yield Button(
                    "Load Character",
                    id="chat-load-character-button",
                    classes="sidebar-button"
                )
                yield Button(
                    "Clear Active Character",
                    id="chat-clear-active-character-button",
                    classes="sidebar-button",
                    variant="warning"
                )
                yield Label("Character Name:", classes="sidebar-label")
                yield Input(
                    id="chat-character-name-edit",
                    placeholder="Name",
                    classes="sidebar-input"
                )
                yield Label("Description:", classes="sidebar-label")
                description_edit_ta = TextArea(id="chat-character-description-edit", classes="sidebar-textarea")
                description_edit_ta.styles.height = 30
                yield description_edit_ta

                yield Label("Personality:", classes="sidebar-label")
                personality_edit_ta = TextArea(id="chat-character-personality-edit", classes="sidebar-textarea")
                personality_edit_ta.styles.height = 30
                yield personality_edit_ta

                yield Label("Scenario:", classes="sidebar-label")
                scenario_edit_ta = TextArea(id="chat-character-scenario-edit", classes="sidebar-textarea")
                scenario_edit_ta.styles.height = 30
                yield scenario_edit_ta

                yield Label("System Prompt:", classes="sidebar-label")
                system_prompt_edit_ta = TextArea(id="chat-character-system-prompt-edit", classes="sidebar-textarea")
                system_prompt_edit_ta.styles.height = 30
                yield system_prompt_edit_ta

                yield Label("First Message:", classes="sidebar-label")
                first_message_edit_ta = TextArea(id="chat-character-first-message-edit", classes="sidebar-textarea")
                first_message_edit_ta.styles.height = 30
                yield first_message_edit_ta

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
        # Search Media (from right sidebar)
        # -------------------------------------------------------------------
        if id_prefix == "chat":
            with Collapsible(title="Search Media", collapsed=True, id=f"{id_prefix}-media-collapsible", classes="settings-collapsible basic-mode advanced-mode"):
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
                yield ListView(id="chat-media-search-results-listview", classes="sidebar-listview")

                with Horizontal(classes="pagination-controls", id="chat-media-pagination-controls"):
                    yield Button("Prev", id="chat-media-prev-page-button", disabled=True)
                    yield Label("Page 1/1", id="chat-media-page-label")
                    yield Button("Next", id="chat-media-next-page-button", disabled=True)

                yield Static("--- Selected Media Details ---", classes="sidebar-label", id="chat-media-details-header")

                media_details_view = VerticalScroll(id="chat-media-details-view")
                media_details_view.styles.height = 35
                with media_details_view:
                    with Horizontal(classes="detail-field-container"):
                        yield Label("Title:", classes="detail-label")
                        yield Button("Copy", id="chat-media-copy-title-button", classes="copy-button", disabled=True)
                    title_display_ta = TextArea("", id="chat-media-title-display", read_only=True, classes="detail-textarea")
                    title_display_ta.styles.height = 3
                    yield title_display_ta

                    with Horizontal(classes="detail-field-container"):
                        yield Label("Content:", classes="detail-label")
                        yield Button("Copy", id="chat-media-copy-content-button", classes="copy-button", disabled=True)
                    content_display_ta = TextArea("", id="chat-media-content-display", read_only=True,
                                   classes="detail-textarea content-display")
                    content_display_ta.styles.height = 20
                    yield content_display_ta

                    with Horizontal(classes="detail-field-container"):
                        yield Label("Author:", classes="detail-label")
                        yield Button("Copy", id="chat-media-copy-author-button", classes="copy-button", disabled=True)
                    author_display_ta = TextArea("", id="chat-media-author-display", read_only=True, classes="detail-textarea")
                    author_display_ta.styles.height = 2
                    yield author_display_ta

                    with Horizontal(classes="detail-field-container"):
                        yield Label("URL:", classes="detail-label")
                        yield Button("Copy", id="chat-media-copy-url-button", classes="copy-button", disabled=True)
                    url_display_ta = TextArea("", id="chat-media-url-display", read_only=True, classes="detail-textarea")
                    url_display_ta.styles.height = 2
                    yield url_display_ta

        # -------------------------------------------------------------------
        # Chat Dictionaries (from right sidebar)
        # -------------------------------------------------------------------
        if id_prefix == "chat":
            with Collapsible(title="Chat Dictionaries", collapsed=True, id=f"{id_prefix}-dictionaries-collapsible", classes="settings-collapsible advanced-mode advanced-only"):
                # Search for available dictionaries
                yield Label("Search Dictionaries:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-dictionary-search-input",
                    placeholder="Search dictionaries...",
                    classes="sidebar-input"
                )
                
                # List of available dictionaries
                yield Label("Available Dictionaries:", classes="sidebar-label")
                dictionary_available_list = ListView(
                    id=f"{id_prefix}-dictionary-available-listview",
                    classes="sidebar-listview"
                )
                dictionary_available_list.styles.height = 5
                yield dictionary_available_list
                
                # Add button for dictionaries
                yield Button(
                    "Add to Chat",
                    id=f"{id_prefix}-dictionary-add-button",
                    classes="sidebar-button",
                    variant="primary",
                    disabled=True
                )
                
                # Currently associated dictionaries
                yield Label("Active Dictionaries:", classes="sidebar-label")
                dictionary_active_list = ListView(
                    id=f"{id_prefix}-dictionary-active-listview",
                    classes="sidebar-listview"
                )
                dictionary_active_list.styles.height = 5
                yield dictionary_active_list
                
                # Remove button for active dictionaries
                yield Button(
                    "Remove from Chat",
                    id=f"{id_prefix}-dictionary-remove-button",
                    classes="sidebar-button",
                    variant="warning",
                    disabled=True
                )
                
                # Quick enable/disable for dictionary processing
                yield Checkbox(
                    "Enable Dictionary Processing",
                    value=True,
                    id=f"{id_prefix}-dictionary-enable-checkbox",
                    classes="sidebar-checkbox"
                )
                
                # Selected dictionary details
                yield Label("Selected Dictionary Details:", classes="sidebar-label")
                dictionary_details = TextArea(
                    "",
                    id=f"{id_prefix}-dictionary-details-display",
                    classes="sidebar-textarea",
                    read_only=True
                )
                dictionary_details.styles.height = 8
                yield dictionary_details

        # -------------------------------------------------------------------
        # World Books (from right sidebar)
        # -------------------------------------------------------------------
        if id_prefix == "chat":
            with Collapsible(title="World Books", collapsed=True, id=f"{id_prefix}-worldbooks-collapsible", classes="settings-collapsible advanced-mode advanced-only"):
                # Search for available world books
                yield Label("Search World Books:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-worldbook-search-input",
                    placeholder="Search world books...",
                    classes="sidebar-input"
                )
                
                # List of available world books
                yield Label("Available World Books:", classes="sidebar-label")
                worldbook_available_list = ListView(
                    id=f"{id_prefix}-worldbook-available-listview",
                    classes="sidebar-listview"
                )
                worldbook_available_list.styles.height = 5
                yield worldbook_available_list
                
                # Add button for world books
                yield Button(
                    "Add to Chat",
                    id=f"{id_prefix}-worldbook-add-button",
                    classes="sidebar-button",
                    variant="primary",
                    disabled=True
                )
                
                # Currently associated world books
                yield Label("Active World Books:", classes="sidebar-label")
                worldbook_active_list = ListView(
                    id=f"{id_prefix}-worldbook-active-listview",
                    classes="sidebar-listview"
                )
                worldbook_active_list.styles.height = 5
                yield worldbook_active_list
                
                # Remove button for active world books
                yield Button(
                    "Remove from Chat",
                    id=f"{id_prefix}-worldbook-remove-button",
                    classes="sidebar-button",
                    variant="warning",
                    disabled=True
                )
                
                # Quick enable/disable for world book processing
                yield Checkbox(
                    "Enable World Book Processing",
                    value=True,
                    id=f"{id_prefix}-worldbook-enable-checkbox",
                    classes="sidebar-checkbox"
                )
                
                # Selected world book details
                yield Label("Selected World Book Details:", classes="sidebar-label")
                worldbook_details = TextArea(
                    "",
                    id=f"{id_prefix}-worldbook-details-display",
                    classes="sidebar-textarea",
                    read_only=True
                )
                worldbook_details.styles.height = 8
                yield worldbook_details

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