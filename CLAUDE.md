# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**tldw_chatbook** is a sophisticated Terminal User Interface (TUI) application built with the Textual framework for interacting with various Large Language Model APIs. It provides a complete ecosystem for AI-powered interactions including conversation management, character/persona chat, notes with bidirectional file sync, media ingestion, and advanced RAG (Retrieval-Augmented Generation) capabilities.

**Core Design Principles**:
- **Modular Architecture**: Clear separation of concerns with dedicated modules for each feature
- **Event-Driven Design**: Decoupled components communicate via Textual's event system
- **Security-First**: Comprehensive input validation, path sanitization, and SQL injection prevention
- **Extensibility**: Plugin-like system for adding new LLM providers and features
- **Performance**: Async operations, worker threads, and memory management for large datasets

**License**: AGPLv3+  
**Python Version**: ≥3.11  
**Main Framework**: Textual (≥3.3.0)  
**Database**: SQLite with FTS5 for full-text search  
**Key Dependencies**: httpx, loguru, rich, pydantic, toml, prometheus_client, keyring, markdownify, aiofiles, jinja2, pycryptodomex

## Common Development Commands

### Running the Application
```bash
# Activate virtual environment first
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the application
python3 -m tldw_chatbook.app
```

### Installation and Dependencies
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install with development dependencies
pip install -e ".[dev]"

# Install optional feature sets
pip install -e ".[embeddings_rag]"  # For RAG functionality
pip install -e ".[websearch]"       # For web search
pip install -e ".[local_vllm]"      # For vLLM support
pip install -e ".[ebook]"            # For e-book processing (includes defusedxml)
pip install -e ".[pdf]"              # For PDF processing
```

### Testing
```bash
# Run all tests
pytest ./Tests

# Run specific test modules
pytest ./Tests/Chat/
pytest ./Tests/Media_DB/
pytest ./Tests/Character_Chat/

# Run only unit tests
pytest -m unit

# Run a single test file
pytest ./Tests/Chat/test_chat_functions.py

# Run a specific test function
pytest ./Tests/Chat/test_chat_functions.py::test_specific_function

# Run with coverage
pytest --cov=tldw_chatbook --cov-report=html
```

### Building
```bash
# Build the package
python -m build

# Install in development mode
pip install -e .

# Install with specific optional dependencies
pip install -e ".[dev,embeddings_rag,websearch]"
```

## Architecture Overview

The codebase follows a sophisticated modular architecture with clear separation of concerns:

### Core Application Structure

#### Entry Points and Configuration
- **`tldw_chatbook/app.py`** - Main entry point, initializes Textual app, manages global state, and coordinates all subsystems
  - Implements `TldwCli` class extending Textual's `App`
  - Manages application lifecycle, theme switching, and global event routing
  - Coordinates worker threads for async operations
- **`tldw_chatbook/config.py`** - Centralized configuration management
  - TOML-based configuration with environment variable fallbacks
  - Provider-specific settings and API key management
  - Path resolution and database initialization
- **`tldw_chatbook/Constants.py`** - Application-wide constants and identifiers
  - Tab identifiers including `TAB_CODING`, `TAB_EVALS`, `TAB_EMBEDDINGS`
  - CSS styling constants for theming including responsive sidebar patterns
  - Help text for LLM server configurations (including MLX-LM)
  - UI dimension constants and TLDW API form container IDs
- **`tldw_chatbook/model_capabilities.py`** - Configuration-based model capability detection
  - Vision support detection for multimodal models
  - Provider-specific capability mapping
- **`tldw_chatbook/config_image_addition.py`** - Image configuration management
  - Image handling settings and defaults

#### UI Layer Architecture
- **`UI/`** - Main window components implementing tab functionality
  - `Chat_Window.py` / `Chat_Window_Enhanced.py` - Advanced chat interface with streaming support
  - `Conv_Char_Window.py` - Conversation and character management
  - `Notes_Window.py` - Notes interface with template support
  - `SearchRAGWindow.py` - RAG search interface with real-time results
  - `Ingest_Window.py` - Media ingestion with progress tracking
  - `LLM_Management_Window.py` - Local LLM server management
  - `Tools_Settings_Window.py` - Application settings and tools
  - `Coding_Window.py` - Coding assistance interface with collapsible sidebar
  - `Logs_Window.py` - Application logs viewer with copy functionality
  - `Stats_Window.py` - Statistics and metrics display
  - `SearchWindow.py` - Legacy search implementation (backup)
  - `Evals_Window.py` - Evaluation system UI for LLM benchmarking
  - `Tab_Bar.py` - Custom tab bar navigation component
  - `MediaWindow.py` - Comprehensive media management with sub-tabs for:
    - Video/Audio content
    - Documents (Word, PowerPoint, etc.)
    - PDFs and EPUBs
    - Websites and article scraping
    - MediaWiki integration
    - Analysis review
  - `Embeddings_Window.py` - Embeddings creation interface
  - `Embeddings_Management_Window.py` - Embeddings management and configuration
- **`Widgets/`** - Reusable UI components following Textual patterns
  - `chat_message.py` / `chat_message_enhanced.py` - Rich message display with markdown
  - `chat_right_sidebar.py` - Context-aware sidebar for chat
  - `notes_sidebar_left/right.py` - Dual sidebar navigation for notes
  - `enhanced_file_picker.py` - Advanced file selection with filtering
  - `emoji_picker.py` - Emoji selection widget
  - `eval_additional_dialogs.py` / `eval_config_dialogs.py` / `eval_results_widgets.py` - Evaluation UI components
  - `Evals_Sidebar.py` - Sidebar for evaluation system navigation
  - `AppFooterStatus.py` - Footer status widget
  - `titlebar.py` - Custom title bar widget
  - `template_selector.py` - Template selection for notes
  - `feedback_dialog.py` - User feedback collection dialog
  - `file_extraction_dialog.py` - File extraction and import UI
  - `notes_sync_widget_improved.py` - Enhanced notes synchronization widget
  - `IngestTldwApi*Window.py` - Media type-specific ingestion windows:
    - `IngestTldwApiVideoWindow.py` - Video ingestion
    - `IngestTldwApiAudioWindow.py` - Audio ingestion
    - `IngestTldwApiPdfWindow.py` - PDF processing
    - `IngestTldwApiEbookWindow.py` - E-book handling
    - `IngestTldwApiDocumentWindow.py` - Document ingestion
    - `IngestTldwApiXmlWindow.py` - XML processing
    - `IngestTldwApiMediawikiWindow.py` - MediaWiki import
    - `IngestTldwApiPlaintextWindow.py` - Plain text ingestion
    - `IngestTldwApiTabbedWindow.py` - Tabbed container for all ingestion types
  - `IngestLocal*Window.py` - Local file ingestion windows:
    - `IngestLocalEbookWindow.py` - Local e-book ingestion
    - `IngestLocalPdfWindow.py` - Local PDF ingestion
    - `IngestLocalPlaintextWindow.py` - Local plaintext ingestion
    - `IngestLocalWebArticleWindow.py` - Local web article ingestion
  - `document_generation_modal.py` - Modal dialog for document generation
  - `cost_estimation_widget.py` - UI widget for cost estimation display
  - `eval_error_dialog.py` - Error dialog for evaluation system
  - `loading_states.py` - Unified loading state components
  - `splash_screen.py` - Main splash screen widget with animation support
  - `settings_sidebar.py` - Settings sidebar widget
  - `file_picker_dialog.py` - Advanced file picker dialog
  - Custom list items, dialogs, and specialized widgets

#### Business Logic Layer
- **`Event_Handlers/`** - Decoupled event handling with clear responsibilities
  - `Chat_Events/` - Chat-specific events (streaming, images, RAG integration)
  - `LLM_Management_Events/` - Provider-specific LLM management
    - `llm_management_events_llamafile.py` - Llamafile server management
    - `llm_management_events_onnx.py` - ONNX runtime support
    - `llm_management_events_transformers.py` - Transformers library integration
    - `llm_management_events_mlx_lm.py` - MLX-LM server support
    - `llm_management_events_vllm.py` - vLLM server event handling
  - `notes_sync_events.py` - File synchronization event handling
  - `eval_events.py` - Evaluation system events
  - `subscription_events.py` - Subscription monitoring system events
  - `embeddings_events.py` - Embeddings creation and management events
  - `app_lifecycle.py` - Application lifecycle management
  - `tab_events.py` - Tab navigation and switching events
  - `sidebar_events.py` - Sidebar interaction events
  - `llm_nav_events.py` - LLM navigation and selection events
  - Tab-specific handlers for each major feature
  - Worker event coordination for async operations
- **`Chat/`** - Core chat engine
  - `Chat_Functions.py` - Conversation management and persistence
  - `prompt_template_manager.py` - Dynamic prompt template system
  - `document_generator.py` - Document generation from conversations (timeline, study guide, briefing)
  - Image handling and multimodal support
- **`Coding/`** - Code assistance functionality
  - `code_mapper.py` - Code mapping and analysis utilities
- **`Character_Chat/`** - Character system implementation
  - `Character_Chat_Lib.py` - Character CRUD operations
  - `ccv3_parser.py` - Character card format parsing
- **`Notes/`** - Advanced notes system
  - `Notes_Library.py` - Notes management with template support
  - `sync_engine.py` - Core synchronization engine with conflict resolution
  - `sync_service.py` - Background sync service implementation
  - Template system for structured note creation
- **`Evals/`** - Comprehensive LLM evaluation system
  - `eval_orchestrator.py` - Evaluation orchestration and management
  - `eval_orchestrator_enhanced.py` - Enhanced orchestration with advanced features
  - `eval_runner.py` - Core evaluation execution engine
  - `eval_runner_enhanced.py` - Enhanced runner with improved error handling
  - `specialized_runners.py` - Task-specific evaluation runners
  - `llm_interface.py` - LLM interaction for evaluations
  - `task_loader.py` - Evaluation task loading and management
  - `eval_templates.py` - Evaluation prompt templates
  - `eval_errors.py` - Comprehensive error handling for evaluations
  - `cost_estimator.py` - Cost estimation for LLM usage during evaluations

#### Data Layer
- **`DB/`** - Database abstraction with specialized implementations
  - `base_db.py` - Common database patterns and utilities
  - `ChaChaNotes_DB.py` - Primary database for characters, chats, and notes
    - Optimistic locking with version fields
    - Soft deletion pattern
    - FTS5 full-text search
    - Automated sync logging via triggers
  - `Client_Media_DB_v2.py` - Media storage with metadata
    - Chunking support for large documents
    - Vector embedding storage (when RAG enabled)
  - `RAG_Indexing_DB.py` - Specialized RAG index management
  - `search_history_db.py` - Search query persistence
  - `Evals_DB.py` - Comprehensive evaluation database for LLM benchmarking
    - Model performance tracking
    - Test suite management
    - Results aggregation
  - `Prompts_DB.py` - Dedicated prompts management with versioning
    - Prompt templates storage
    - Version history tracking
    - Category organization
  - `Subscriptions_DB.py` - Subscription management database
    - Subscription tracking and monitoring
    - Update frequency management
  - `Sync_Client.py` - Client-side synchronization for distributed database sync
  - `sqlite_datetime_fix.py` - SQLite datetime handling fixes
  - `sql_validation.py` - SQL injection prevention

#### Service Layer
- **`LLM_Calls/`** - Unified LLM interface
  - `LLM_API_Calls.py` - Commercial provider integrations
  - `LLM_API_Calls_Local.py` - Local model integrations
  - Streaming response handling
  - Provider-specific parameter mapping
- **`RAG_Search/`** - Advanced RAG implementation
  - `simplified/` - Streamlined RAG architecture
    - Simplified service implementations
    - Optimized for performance
  - `chunking_service.py` - Intelligent text chunking (direct in RAG_Search/)
  - Hybrid search (keyword + semantic)
  - Configurable retrieval strategies
- **`tldw_api/`** - TLDW API client implementation
  - `client.py` - API client for TLDW service integration
  - `schemas.py` - Data models and validation schemas
  - `exceptions.py` - Custom exception handling
  - `utils.py` - Utility functions for API operations

- **`Config_Files/`** - Configuration management
  - `create_custom_template.py` - Template creation utilities
  - Default configuration templates

- **`Third_Party/`** - External integrations
  - `aider/` - Aider code assistant integration
  - `textual_fspicker/` - Enhanced file picker widget library
- **`css/`** - Modular CSS architecture
  - `core/` - Base styles and variables
  - `components/` - Component-specific styles
    - `loading_states.css` - Loading state animations and styles
  - `features/` - Feature-specific styles
  - `layout/` - Layout and grid systems
  - `utilities/` - Utility classes
  - `Themes/` - Theme definitions
  - `build_css.py` - CSS build and compilation system
  - `theme_tester.py` - Theme testing utilities

- **`Screens/`** - Complex UI screen implementations
  - Dedicated screen components for multi-step workflows
  - Reusable screen patterns

#### Supporting Systems
- **`Utils/`** - Cross-cutting utilities
  - `path_validation.py` - Security-focused path handling
  - `input_validation.py` - Input sanitization
  - `optional_deps.py` - Dynamic feature detection
  - `secure_temp_files.py` - Safe temporary file handling
  - `terminal_utils.py` - Terminal capability detection (sixel, TGP support)
  - `Splash.py` / `Splash_Strings.py` / `splash_animations.py` - Splash screen functionality
- **`Metrics/`** - Application telemetry
  - `metrics.py` - Core metrics collection
  - `Otel_Metrics.py` - OpenTelemetry integration
  - `metrics_logger.py` - Metrics logging functionality
  - `metrics_wrapper.py` - Metrics wrapper utilities
  - Performance monitoring
  - Usage analytics (local only)
- **`Web_Scraping/`** - Content extraction
  - `WebSearch_APIs.py` - Comprehensive web search integration
    - Google, Bing, DuckDuckGo, Brave
    - Kagi, Tavily, SearX
    - Baidu, Yandex
  - `Article_Scraper/` - Full article scraping module
    - `crawler.py` - Web crawling functionality
    - `processors.py` - Content processing
    - `importers.py` - Content import pipelines
  - `cookie_scraping/` - Cookie cloning for authenticated access
  - Confluence integration
  - Cookie-based authentication support

### Splash Screen System

#### Architecture
The application includes a sophisticated splash screen system with customizable animations:
- **Core Components**:
  - `Widgets/splash_screen.py` - Main splash screen widget with animation support
  - `Utils/Splash.py` - Core splash functionality and configuration
  - `Utils/Splash_Strings.py` - Splash screen message content
  - `Utils/splash_animations.py` - 20+ animation effects including:
    - MatrixRainEffect, GlitchEffect, TypewriterEffect
    - FadeEffect, RetroTerminalEffect, PulseEffect
    - CodeScrollEffect, StarfieldEffect, GameOfLifeEffect
    - And many more creative visual effects

#### Configuration
- Configured via `[splash_screen]` section in config.toml:
  - `enabled` - Enable/disable splash screen
  - `duration` - Display duration in seconds
  - `card_selection` - Selection mode (random, sequential, or specific card)
  - `active_cards` - List of enabled splash cards
  - `animation_speed` - Animation playback speed multiplier

#### Custom Splash Cards
- Support for custom splash card definitions
- Example cards in `examples/custom_splash_cards/`:
  - `cyberpunk_card.toml`, `gaming_card.toml`, `minimalist_card.toml`
  - `custom_animation_effect.py` - Template for custom animations

### Database Design

#### Schema Versioning
- All databases track schema version
- Migration support via SQL scripts
- Backward compatibility maintained
- SQLite datetime handling fixes via `sqlite_datetime_fix.py`

#### Key Relationships
1. **Conversations ↔ Messages**: One-to-many with ordering
2. **Conversations ↔ Characters**: Optional many-to-one
3. **Keywords ↔ Content**: Many-to-many via link tables
4. **Media ↔ Chunks**: One-to-many for large documents
5. **Notes ↔ Files**: One-to-one for sync tracking

### Event-Driven Architecture

#### Event Flow
1. **UI Interaction** → Widget emits custom event
2. **Event Router** → App routes to appropriate handler
3. **Event Handler** → Processes business logic
4. **State Update** → Updates reactive attributes
5. **UI Update** → Textual automatically refreshes

#### Key Event Types
- `ChatEvent` - Message sending, editing, deletion
- `StreamingChunk` / `StreamDone` - LLM streaming
- `RAGSearchEvent` - Search queries and results
- `SyncEvent` - File synchronization status
- `WorkerEvent` - Background task coordination
- `EvalEvent` - Evaluation system events
- `NoteSyncEvent` - Note-specific sync operations
- `AppLifecycleEvent` - Application startup/shutdown events
- `TabEvent` - Tab navigation and switching
- `SidebarEvent` - Sidebar interactions
- `LLMNavEvent` - LLM selection and navigation

### Security Architecture

#### Input Validation
- All user inputs sanitized via `input_validation.py`
- Path traversal prevention in `path_validation.py`
- SQL identifier validation in `sql_validation.py`

#### Data Protection
- API keys stored in config or environment only
- Sensitive data never logged
- Secure temporary file handling

### Performance Optimizations

#### Async Operations
- LLM calls use httpx for async requests
- Background workers for heavy operations
- Streaming responses for real-time feedback

#### Memory Management
- Configurable cache sizes for embeddings
- Chunk-based processing for large files
- Lazy loading for database results

#### Database Performance
- FTS5 indexes for fast search
- Prepared statements prevent SQL injection
- Connection pooling via thread-local storage

## LLM Provider Integration

The app implements a sophisticated plugin-like system for LLM providers:

### Provider Architecture

#### Unified Interface
All providers implement these core methods:
```python
def chat_with_provider(
    messages: List[Dict],
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    stream: bool = False,
    system_prompt: Optional[str] = None,
    **kwargs
) -> Union[str, Generator[str, None, None]]
```

#### Model Capabilities
The system includes intelligent model capability detection via `model_capabilities.py`:
- **Vision Support**: Automatic detection of multimodal models
- **Provider Mapping**: Configuration-based capability definitions
- **Dynamic Updates**: Easy addition of new model capabilities

#### Commercial Providers
Located in `LLM_Calls/LLM_API_Calls.py`:
- **OpenAI**: GPT-3.5/4 models with function calling support
- **Anthropic**: Claude models with system prompt handling
- **Google**: Gemini models with safety settings
- **Cohere**: Command models with web search
- **DeepSeek**: Specialized coding models
- **Mistral**: Open-weight commercial models
- **Groq**: High-speed inference
- **HuggingFace**: Hub model access
- **OpenRouter**: Multi-provider gateway
- **TabbyAPI**: High-performance inference server
- **Custom OpenAI 2**: Alternative OpenAI-compatible implementation

#### Local Providers
Located in `LLM_Calls/LLM_API_Calls_Local.py`:
- **Llama.cpp**: Direct C++ inference
- **Ollama**: Containerized local models
- **Kobold.cpp**: Gaming-focused inference
- **vLLM**: High-throughput serving
- **Aphrodite**: vLLM fork with extensions
- **MLX**: Apple Silicon optimized
- **Custom OpenAI**: Any OpenAI-compatible endpoint
- **Llamafile**: Single-file executable LLM server
- **ONNX**: Cross-platform neural network inference
- **Transformers**: HuggingFace transformers library

### Provider Implementation Guide

#### Adding a New Provider
1. **Create Provider Function**:
   ```python
   def chat_with_newprovider(
       messages, api_key, model, temperature=0.7,
       max_tokens=4096, stream=False, **kwargs
   ):
       # Implementation
   ```

2. **Update Configuration**:
   - Add to `API_MODELS_BY_PROVIDER` in `config.py`
   - Add default models and endpoints
   - Update config.toml template

3. **Implement Streaming**:
   ```python
   if stream:
       for chunk in response:
           yield extract_content(chunk)
   ```

4. **Add UI Integration**:
   - Update provider dropdowns
   - Add to model selection logic
   - Handle provider-specific parameters

5. **Error Handling**:
   - Implement retry logic
   - Provide meaningful error messages
   - Handle rate limits gracefully

### Provider-Specific Features

#### Streaming Implementation
- Chunk-based yielding for real-time display
- Proper cleanup on cancellation
- Token counting for progress indication

#### Parameter Mapping
- Temperature scaling for different providers
- Token limit adjustments
- System prompt handling variations

#### Authentication
- API key validation
- OAuth support where applicable
- Custom header injection

### Local Model Management

#### Server Lifecycle
1. **Startup**: Managed by LLM_Management_Window
2. **Health Checks**: Periodic connectivity tests
3. **Shutdown**: Graceful termination on app exit

#### Model Loading
- Automatic model discovery (Ollama)
- Manual path specification (llama.cpp)
- HuggingFace model downloading (MLX)

## Key Development Patterns

### Database Operations

#### Connection Management
```python
# Thread-safe connection pattern
with db.transaction() as cursor:
    cursor.execute(query, params)
    db.commit()
```

#### Common Patterns
- **Optimistic Locking**: Version field prevents conflicts
- **Soft Deletion**: `deleted_at` timestamp preserves history
- **FTS5 Search**: Automatic indexing via triggers
- **Sync Logging**: Change tracking for distributed sync
- **Prepared Statements**: Parameterized queries only

### UI Component Structure

#### Widget Lifecycle
1. **Initialization**: Set up reactive attributes
2. **Compose**: Define child widgets
3. **Mount**: Post-initialization setup
4. **Event Handling**: Process user interactions
5. **State Updates**: Modify reactive attributes
6. **Rendering**: Automatic via Textual

#### Reactive Patterns
```python
class MyWidget(Widget):
    data = reactive([], recompose=True)  # Triggers recompose
    status = reactive("idle")  # Triggers refresh only
    
    def watch_data(self, old, new):
        """Called when data changes"""
        self.refresh()
```

#### Event Communication
```python
# Emit custom event
self.post_message(MyCustomEvent(data=result))

# Handle in parent
@on(MyCustomEvent)
def handle_custom(self, event: MyCustomEvent):
    process_data(event.data)
```

### Error Handling Strategy

#### Logging Hierarchy
1. **DEBUG**: Detailed flow tracking (loguru)
2. **INFO**: Key operations and state changes
3. **WARNING**: Recoverable issues
4. **ERROR**: Failures requiring attention
5. **CRITICAL**: System-breaking issues

#### User Feedback
```python
try:
    result = risky_operation()
except SpecificError as e:
    # Log for debugging
    logger.error(f"Operation failed: {e}")
    # User-friendly message
    self.notify("Unable to complete action. Please try again.", severity="error")
    # Fallback behavior
    return default_value
```

### Testing Philosophy

#### Test Categories
1. **Unit Tests**: Isolated component testing
2. **Integration Tests**: Multi-component workflows
3. **Property Tests**: Invariant verification
4. **Security Tests**: Input validation and sanitization

#### Database Testing
```python
@pytest.fixture
def test_db():
    """In-memory database for testing"""
    db = ChaChaNotes_DB(":memory:", "test_client")
    yield db
    db.close()
```

#### Key Testing Principles
- **No Mocking Databases**: Use real SQLite in-memory
- **Property-Based Testing**: Find edge cases automatically with Hypothesis
- **Async Test Support**: For streaming operations with pytest-asyncio
- **Fixture Reuse**: Consistent test data
- **Test Markers**: `unit`, `integration`, `optional`, `asyncio`

### Security Patterns

#### Input Validation
```python
# Text input validation
validated = validate_text_input(
    user_input,
    min_length=1,
    max_length=1000,
    allow_special_chars=True
)

# Path validation
safe_path = validate_path(user_path, base_dir)

# SQL identifier validation
safe_column = validate_sql_identifier(column_name)
```

#### Secure Defaults
- Paths restricted to user data directory
- SQL identifiers validated against whitelist
- File uploads size-limited
- API keys never in code or logs

### Performance Patterns

#### Async Operations
```python
# Background worker pattern
def start_heavy_operation(self):
    self.run_worker(
        self._do_heavy_work,
        name="heavy_op",
        exclusive=True
    )

@work(thread=True)
def _do_heavy_work(self):
    # Long-running task
    result = process_large_dataset()
    self.call_from_thread(self.update_ui, result)
```

#### Memory Management
- Chunk large files during processing
- Clear caches when switching contexts
- Use generators for streaming data
- Limit result set sizes with pagination

### Code Organization

#### Module Structure
```
feature/
├── __init__.py          # Public API
├── models.py            # Data models
├── services.py          # Business logic
├── handlers.py          # Event handlers
└── widgets.py           # UI components
```

#### Import Organization
1. Standard library
2. Third-party libraries
3. Local imports (absolute)
4. Type imports (if TYPE_CHECKING)

#### Naming Conventions
- **Classes**: PascalCase (`ChatWindow`)
- **Functions**: snake_case (`send_message`)
- **Constants**: UPPER_SNAKE (`MAX_TOKENS`)
- **Private**: Leading underscore (`_internal_method`)

## Important Development Guidelines

### Code Standards
- **Branch Strategy**: 
  - Submit PRs to `dev` branch, not `main`
  - Feature branches from `dev`
  - Hotfixes from `main`
- **Code Style**: 
  - No enforced linter currently - follow existing patterns
  - Prefer clarity over cleverness
  - Document complex logic inline
- **Type Hints**: 
  - Use for public APIs and complex functions
  - Import from `typing` for compatibility

### Dependency Management
- **Core Philosophy**: Minimal core, optional features
- **Adding Dependencies**:
  1. Evaluate if it should be optional
  2. Add to appropriate group in `pyproject.toml`
  3. Update `optional_deps.py` detection
  4. Document in README feature table
- **Version Pinning**: 
  - Exact versions for security-critical packages
  - Flexible ranges for stable packages

### Configuration Best Practices
- **Settings Hierarchy**:
  1. Environment variables (highest priority)
  2. Config file (`~/.config/tldw_cli/config.toml`)
  3. Defaults in code (lowest priority)
- **Feature Flags**: Check `get_cli_setting()` for runtime config
- **API Keys**: 
  - NEVER in code or version control
  - Use `PROVIDER_API_KEY` env vars
  - Or `[API]` section in config.toml
- **Splash Screen**: Configure via `[splash_screen]` section:
  - `enabled` - Enable/disable splash screen
  - `duration` - Display time in seconds
  - `card_selection` - random, sequential, or specific card name
  - `active_cards` - List of enabled splash card names

### Database Guidelines
- **Schema Changes**:
  - Increment schema version
  - Add migration in `DB/migrations/`
  - Maintain backward compatibility
  - Test with existing databases
- **Query Patterns**:
  - Always use parameterized queries
  - Validate identifiers with `sql_validation.py`
  - Use transactions for multi-step operations

### Testing Requirements
- **Before Submitting PRs**:
  - Run full test suite: `pytest`
  - Test with optional deps: `pytest -m "not optional"`
  - Check specific feature: `pytest Tests/Feature/`
- **Writing Tests**:
  - Match existing test patterns
  - Use fixtures for database setup
  - Include edge cases and error paths
  - Add security tests for new inputs

### Performance Considerations
- **UI Responsiveness**:
  - Use workers for operations >100ms
  - Stream LLM responses
  - Debounce rapid user inputs
- **Memory Usage**:
  - Process large files in chunks
  - Clear caches when switching contexts
  - Monitor embedding cache size
- **Database Performance**:
  - Use FTS5 for text search
  - Limit result sets with pagination
  - Create indexes for frequent queries

### Security Checklist
- **User Input**: Always validate and sanitize
- **File Paths**: Use `path_validation.py`
- **SQL Queries**: Use `sql_validation.py`
- **API Keys**: Never log or display
- **Temporary Files**: Use `secure_temp_files.py`
- **External Content**: Sanitize HTML/Markdown

### Debugging Tools
- **Logging**:
  - Use loguru for structured logging
  - Check `~/.share/tldw_cli/logs/`
  - Adjust level in config.toml
- **Textual Dev Console**: 
  - Run with `textual console`
  - Then `tldw-cli` in another terminal
- **Performance Profiling**:
  - Use `--profile` flag (if implemented)
  - Check metrics in Stats tab

### Evaluation System Architecture

The evaluation system provides comprehensive LLM benchmarking:

#### Components
- **Evals_DB.py**: Database for storing evaluation data
- **Evals_Window.py**: Main evaluation UI tab
- **Evals/**: Core evaluation engine
  - `eval_orchestrator.py` - Manages evaluation workflows
  - `eval_runner.py` - Executes evaluation tasks
  - `specialized_runners.py` - Task-specific implementations
  - `llm_interface.py` - Standardized LLM communication
  - `task_loader.py` - Loads and manages test suites
  - `eval_templates.py` - Evaluation prompt templates
- **Result Widgets**: Specialized widgets for displaying results
- **Configuration Dialogs**: Settings for evaluation parameters

#### Workflow
1. Configure evaluation parameters
2. Select models to evaluate
3. Run benchmark suites
4. View and compare results
5. Export evaluation data

### Notes Synchronization System

#### Architecture
- **Bidirectional Sync**: Files ↔ Database synchronization
- **Conflict Resolution**: Last-write-wins with backup
- **Change Detection**: File system monitoring
- **Sync Service**: Background synchronization

#### Key Features
- Automatic sync on file changes
- Manual sync triggers
- Sync status indicators
- Conflict resolution UI

### Subscription System

#### Architecture
- **Subscriptions_DB.py**: Database for subscription management
- **subscription_events.py**: Event handling for subscriptions
- Periodic update checking
- Notification system for new content

### TLDW API Integration

#### Components
- **tldw_api/**: Complete API client module
  - Schemas for data validation
  - Exception handling
  - Utility functions
- **IngestTldwApi* Windows**: Specialized UI for each media type
  - Video, Audio, PDF, E-book, Document ingestion
  - XML, MediaWiki, Plain text processing
  - Tabbed interface for all media types

### CSS Build System

#### Architecture
- **Modular Structure**: Organized by purpose (core, components, features)
- **Build Process**: `build_css.py` compiles modular CSS
- **Theme System**: Multiple themes with testing utilities
- **Hot Reload**: Development mode with live updates

### Common Patterns

#### Adding a New Tab
1. Create window class in `UI/`
2. Add tab constant to `Constants.py`
3. Register in `app.py` compose method
4. Create event handlers in `Event_Handlers/`
5. Update tab bar navigation in `UI/Tab_Bar.py`
6. Add tab-specific events in `Event_Handlers/tab_events.py`

#### Adding a New Database Table
1. Design schema with version tracking
2. Add creation in database `__init__`
3. Implement CRUD methods
4. Add FTS5 triggers if searchable
5. Update sync_log triggers
6. Write comprehensive tests

#### Implementing a New Feature
1. Check if it needs optional dependencies
2. Create feature module structure
3. Implement business logic separately from UI
4. Create reusable widgets
5. Use event-driven communication
6. Add configuration options
7. Write tests early
8. Document in appropriate README

### Release Process
1. **Version Bump**: Update in `pyproject.toml`
2. **Changelog**: Update with features/fixes
3. **Test Suite**: Full pass on multiple platforms
4. **Documentation**: Update README and guides
5. **Tag Release**: Follow semantic versioning
6. **PyPI Release**: `python -m build && twine upload dist/*`
7. **Update Homebrew Formula**: If applicable

### Deployment and Distribution

#### PyPI Package
- **Package Name**: `tldw-cli`
- **Entry Point**: `tldw-cli` command
- **Build System**: setuptools with pyproject.toml
- **Package Data**: Includes templates, CSS, and default configs

#### Installation Methods
1. **PyPI**: `pip install tldw-cli`
2. **Development**: `pip install -e .`
3. **Docker**: Container support (if implemented)
4. **Homebrew**: macOS formula (if available)

### Troubleshooting

#### Common Issues
- **Import Errors**: Check optional dependencies
- **Database Locked**: Check for hanging processes
- **UI Not Updating**: Verify reactive attributes
- **Streaming Broken**: Check worker state
- **Config Not Loading**: Validate TOML syntax

#### Getting Help
- Check existing issues on GitHub
- Review test files for usage examples
- Enable debug logging
- Use Textual dev console
- Ask in discussions with minimal reproduction


## Code Style
- Use Python type hints for function parameters and return values
- Follow PEP 8 conventions for naming and formatting
- Use docstrings with Args/Returns/Raises sections in Google style
- Group imports: stdlib, third-party, local
- Error handling: Use specific exceptions from mcp.server.fastmcp.exceptions
- Class naming: PascalCase
- Function/variable naming: snake_case
- Use Pydantic models for data validation
- Validate input parameters and handle exceptions with appropriate error codes
- Tests should use pytest fixtures and mocks for external services

## Commit Guidelines
- Use conventional commit message format (Release Please compatible):
  * Format: `<type>(<scope>): <description>`
  * Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
  * Example: `fix(dependencies): update log4j to patch security vulnerability`
  * Breaking changes: Add `!` after type/scope and include `BREAKING CHANGE:` in body
  * Example: `feat(api)!: change response format` with body containing `BREAKING CHANGE: API now returns JSON instead of XML`
- Ensure commit messages are concise and descriptive
- Explain the purpose and impact of changes in the commit message
- Group related changes in a single commit
- Keep commits focused and atomic
- For version bumps, use `chore(release): v1.2.3` format

## PR Description Guidelines
Use the output of the git diff to create the description of the Merge Request. Follow this structure exactly:

1. **Title**  
   *A one-line summary of the change (max 60 characters).*

2. **Summary**  
   *Briefly explain what this PR does and why.*

3. **Changes**  
   *List each major change as a bullet:*  
   - Change A: what was done  
   - Change B: what was done  

4. **Technical Details**
   *Highlight any notable technical details relevant to the changes*

## CHANGELOG Guidelines
- Maintain a CHANGELOG.md file in the root directory
- Add entries under the following sections:
  * `Added` - New features
  * `Changed` - Changes in existing functionality
  * `Fixed` - Bug fixes
  * `Security` - Vulnerability fixes
- Example:
  ```markdown
  ## 2024-03-20
  ### Added
  - New feature X
  ### Fixed
  - Bug fix Y
  ```