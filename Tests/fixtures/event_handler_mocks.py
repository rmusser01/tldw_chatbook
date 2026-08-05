"""
Common mock fixtures for Event Handler tests.
This module provides a comprehensive mock architecture that correctly handles
async/sync methods, query_one behavior, and all necessary mock attributes.
"""

from unittest.mock import AsyncMock, MagicMock
from textual.widgets import (
    Button,
    Input,
    TextArea,
    Static,
    Select,
    Checkbox,
    ListView,
    Label,
    RichLog,
    ProgressBar,
    LoadingIndicator,
    DataTable,
)
from textual.containers import VerticalScroll
from textual.css.query import QueryError


def create_comprehensive_app_mock():
    """Create a comprehensive mock of the TldwCli app with proper async/sync handling."""
    app = AsyncMock()

    # Add thread lock for chat state management
    app._chat_state_lock = MagicMock()

    # Sync accessor read by resolve_runtime_backend_mode(); the blanket
    # AsyncMock would hand back an unawaited coroutine (RuntimeWarning noise).
    app.get_authoritative_runtime_source = MagicMock(return_value="local")

    # Mock services and DBs
    app.chachanotes_db = MagicMock()
    app.notes_service = MagicMock()
    app.notes_service._get_db.return_value = app.chachanotes_db
    app.media_db = MagicMock()
    app.prompts_db = MagicMock()

    # Mock core app properties
    app.API_IMPORTS_SUCCESSFUL = True
    app.app_config = {
        "api_settings": {
            "openai": {"streaming": True, "api_key_env_var": "OPENAI_API_KEY"},
            "anthropic": {"streaming": False, "api_key": "xyz-key"},
            "ollama": {"streaming": True},
            "local-llm": {"streaming": True},
        },
        "chat_defaults": {
            "system_prompt": "Default system prompt.",
            "strip_thinking_tags": True,
        },
        "character_defaults": {},
        "USERS_NAME": "Tester",
        "rag": {"enabled": False, "max_tokens": 4000},
    }

    # Mock app state
    app.current_chat_conversation_id = None
    app.current_chat_is_ephemeral = True
    app.current_chat_active_character_data = None
    app.current_ai_message_widget = None
    app.current_sidebar_media_item = None
    app.current_chat_is_streaming = False
    app.current_chat_worker = None  # No worker running initially
    app.current_tab = "Chat"
    app.current_world_info_active = False  # Default to no world info
    app.current_world_info_count = 0

    # Mock app methods
    app.query_one = MagicMock()
    app.query = MagicMock()
    app.notify = MagicMock()  # notify is called synchronously in event handlers
    app.copy_to_clipboard = MagicMock()
    app.set_timer = MagicMock()
    app.run_worker = MagicMock()
    app.chat_wrapper = MagicMock()  # Called synchronously in worker thread
    app.call_from_thread = MagicMock(side_effect=lambda func, *args: func(*args))

    # Thread-safe methods are synchronous
    app.get_current_ai_message_widget = MagicMock(return_value=None)
    app.set_current_ai_message_widget = MagicMock()
    app.set_current_chat_worker = MagicMock()
    app.set_current_chat_is_streaming = MagicMock()
    app.get_current_chat_is_streaming = MagicMock(return_value=False)

    # Timers
    app._media_sidebar_search_timer = None
    app._character_search_timer = None

    # Logger
    app.loguru_logger = MagicMock()

    # Setup mock widgets
    setup_mock_widgets(app)

    return app


def setup_mock_widgets(app):
    """Set up comprehensive mock widgets for the app."""
    # Create mock widgets with proper async/sync methods
    mock_chat_log = create_widget_mock(
        VerticalScroll,
        async_methods=["mount", "remove_children", "clear", "append", "scroll_end"],
    )
    mock_chat_log.query = MagicMock(return_value=[])
    # scroll_end is actually synchronous in Textual, override the async setting
    mock_chat_log.scroll_end = MagicMock()

    # Create text area mocks with proper methods
    create_widget_mock(TextArea, sync_methods=["clear"])

    # Create all required widgets
    widgets = {
        # Chat main UI
        "#chat-input": create_widget_mock(
            TextArea, text="User message", sync_methods=["clear"]
        ),
        "#chat-log": mock_chat_log,
        # Chat settings
        "#chat-api-provider": create_widget_mock(Select, value="OpenAI"),
        "#chat-api-model": create_widget_mock(Select, value="gpt-4"),
        "#chat-system-prompt": create_widget_mock(TextArea, text="UI system prompt"),
        "#chat-temperature": create_widget_mock(Input, value="0.7"),
        "#chat-top-p": create_widget_mock(Input, value="0.9"),
        "#chat-min-p": create_widget_mock(Input, value="0.1"),
        "#chat-top-k": create_widget_mock(Input, value="40"),
        "#chat-llm-max-tokens": create_widget_mock(Input, value="1024"),
        "#chat-llm-seed": create_widget_mock(Input, value=""),
        "#chat-llm-stop": create_widget_mock(Input, value=""),
        "#chat-llm-response-format": create_widget_mock(Select, value="text"),
        "#chat-llm-n": create_widget_mock(Input, value="1"),
        "#chat-llm-user-identifier": create_widget_mock(Input, value=""),
        "#chat-llm-logprobs": create_widget_mock(Checkbox, value=False),
        "#chat-llm-top-logprobs": create_widget_mock(Input, value=""),
        "#chat-llm-logit-bias": create_widget_mock(TextArea, text="{}"),
        "#chat-llm-presence-penalty": create_widget_mock(Input, value="0.0"),
        "#chat-llm-frequency-penalty": create_widget_mock(Input, value="0.0"),
        "#chat-llm-tools": create_widget_mock(TextArea, text="[]"),
        "#chat-llm-tool-choice": create_widget_mock(Input, value=""),
        "#chat-llm-fixed-tokens-kobold": create_widget_mock(Checkbox, value=False),
        "#chat-strip-thinking-tags-checkbox": create_widget_mock(Checkbox, value=True),
        # Media sidebar
        "#chat-media-search-results-listview": create_widget_mock(
            ListView, async_methods=["clear", "append"]
        ),
        "#chat-media-search-input": create_widget_mock(Input, value=""),
        "#chat-media-keyword-filter-input": create_widget_mock(Input, value=""),
        "#chat-media-title-display": create_widget_mock(
            TextArea, sync_methods=["clear", "load_text"]
        ),
        "#chat-media-content-display": create_widget_mock(
            TextArea, sync_methods=["clear", "load_text"]
        ),
        "#chat-media-author-display": create_widget_mock(
            TextArea, sync_methods=["clear", "load_text"]
        ),
        "#chat-media-url-display": create_widget_mock(
            TextArea, sync_methods=["clear", "load_text"]
        ),
        "#chat-media-copy-title-button": create_widget_mock(Button, disabled=False),
        "#chat-media-copy-content-button": create_widget_mock(Button, disabled=False),
        "#chat-media-copy-author-button": create_widget_mock(Button, disabled=False),
        "#chat-media-copy-url-button": create_widget_mock(Button, disabled=False),
        "#chat-media-page-label": create_widget_mock(Label),
        "#chat-media-prev-page-button": create_widget_mock(Button),
        "#chat-media-next-page-button": create_widget_mock(Button),
        # RAG checkboxes
        "#chat-rag-enabled-checkbox": create_widget_mock(Checkbox, value=False),
        "#chat-rag-search-results-checkbox": create_widget_mock(Checkbox, value=False),
        # LLM Management
        "#llm-provider-select": create_widget_mock(Select, value="ollama"),
        "#llm-server-status": create_widget_mock(Static),
        "#llm-model-status": create_widget_mock(Static),
        "#llm-server-log": create_widget_mock(RichLog, async_methods=["write"]),
        "#llm-start-server-button": create_widget_mock(Button),
        "#llm-stop-server-button": create_widget_mock(Button),
        "#llm-model-select": create_widget_mock(Select),
        "#llm-load-model-button": create_widget_mock(Button),
        "#llm-unload-model-button": create_widget_mock(Button),
        "#llm-refresh-models-button": create_widget_mock(Button),
        "#llm-server-address-input": create_widget_mock(
            Input, value="http://localhost:11434"
        ),
        "#llm-model-path-input": create_widget_mock(Input, value=""),
        "#llm-gpu-layers-input": create_widget_mock(Input, value="0"),
        "#llm-context-size-input": create_widget_mock(Input, value="2048"),
        "#llm-loading-progress": create_widget_mock(ProgressBar),
        "#llm-loading-indicator": create_widget_mock(LoadingIndicator),
        # Ingest UI
        "#media-import-progress": create_widget_mock(
            ProgressBar, total=100, progress=0
        ),
        "#media-import-log": create_widget_mock(RichLog, async_methods=["write"]),
        "#media-type-dropdown": create_widget_mock(Select, value="article"),
        "#media-url-input": create_widget_mock(Input, value=""),
        "#media-title-input": create_widget_mock(Input, value=""),
        "#media-author-input": create_widget_mock(Input, value=""),
        "#media-tags-input": create_widget_mock(Input, value=""),
        "#media-import-button": create_widget_mock(Button),
        "#batch-size-input": create_widget_mock(Input, value="5"),
        "#media-content-preview": create_widget_mock(TextArea),
        # Search/RAG UI
        "#search-query-input": create_widget_mock(Input, value=""),
        "#search-results-table": create_widget_mock(DataTable),
        "#search-button": create_widget_mock(Button),
        "#result-preview": create_widget_mock(TextArea),
        "#copy-result-button": create_widget_mock(Button),
        "#open-result-button": create_widget_mock(Button),
    }

    def query_one_side_effect(selector, widget_type=None):
        """Mock query_one with proper error handling and type checking."""
        # Handle class-based queries (e.g., app.query_one(TitleBar))
        if isinstance(selector, type):
            # Return a mock of the requested widget type
            mock_widget = MagicMock(spec=selector)
            if hasattr(selector, "reset_title"):
                mock_widget.reset_title = MagicMock()
            return mock_widget

        # Handle ID-based queries
        if selector in widgets:
            widget = widgets[selector]
            # Type check if requested
            if widget_type and hasattr(widget, "__class__"):
                # For mock objects, we trust the spec
                pass
            return widget

        # Check for sub-widget queries (e.g., widget.query_one)
        if hasattr(selector, "query_one"):
            return selector.query_one(selector, widget_type)

        raise QueryError(f"Widget not found by mock: {selector}")

    app.query_one.side_effect = query_one_side_effect
    app.screen = MagicMock()
    app.screen.query_one = MagicMock(
        side_effect=lambda selector, widget_type=None: app.query_one(
            selector, widget_type
        )
    )
    app.screen.query = MagicMock(side_effect=lambda selector: app.query(selector))

    # Add query method that returns multiple widgets
    def query_side_effect(selector):
        """Mock query that returns a list of widgets."""
        if selector == ".chat-message":
            return []  # Return empty list by default
        return []

    app.query.side_effect = query_side_effect


def create_widget_mock(widget_class, **kwargs):
    """Create a mock widget with proper async/sync method configuration."""
    async_methods = kwargs.pop("async_methods", [])
    sync_methods = kwargs.pop("sync_methods", [])

    # Create the base mock
    mock = MagicMock(spec=widget_class)

    # Set default attributes
    mock.is_mounted = True
    mock.disabled = False

    # Configure any passed attributes
    for key, value in kwargs.items():
        setattr(mock, key, value)

    # Configure async methods
    for method_name in async_methods:
        if hasattr(mock, method_name):
            setattr(mock, method_name, AsyncMock())

    # Configure sync methods (ensure they're not AsyncMock)
    for method_name in sync_methods:
        if hasattr(mock, method_name):
            setattr(mock, method_name, MagicMock())

    # Special handling for common widget methods
    if hasattr(widget_class, "clear"):
        # TextArea.clear is sync
        if widget_class in [TextArea, Input]:
            mock.clear = MagicMock()

    if hasattr(widget_class, "mount"):
        # mount is always async
        mock.mount = AsyncMock()

    if hasattr(widget_class, "remove"):
        # remove is always async
        mock.remove = AsyncMock()

    if hasattr(widget_class, "query_one"):
        mock.query_one = MagicMock()

    if hasattr(widget_class, "update"):
        # Static.update is sync
        if widget_class == Static:
            mock.update = MagicMock()

    return mock
