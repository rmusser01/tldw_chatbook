# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**tldw_chatbook** - TUI application built with Textual for LLM interactions. Features: conversation management, character chat, notes with file sync, media ingestion, RAG capabilities.

**Tech Stack**: Python ≥3.11, Textual ≥3.3.0, SQLite with FTS5, AGPLv3+  
**Key Dependencies**: httpx, loguru, rich, pydantic, toml, keyring, aiofiles, jinja2

## Quick Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"  # Or specific: .[embeddings_rag,websearch,local_vllm,ebook,pdf]

# Run
python3 -m tldw_chatbook.app

# Test
pytest  # All tests
pytest Tests/Chat/  # Specific module
pytest --cov=tldw_chatbook  # With coverage
```

## Architecture

### Core Structure

- **`app.py`** - Main entry, `TldwCli` class, tab management, global state
- **`config.py`** - TOML config at `~/.config/tldw_cli/config.toml`, env var fallbacks
- **`Constants.py`** - Tab IDs (TAB_CHAT, TAB_CODING, etc.), UI dimensions, provider mappings

### UI Layer (`UI/` and `Widgets/`)

**Main Windows** (all extend Screen):
- `Chat_Window_Enhanced.py` - Streaming chat, images, RAG
- `Conv_Char_Window.py` - Conversation/character CRUD
- `Notes_Window.py` - Notes with templates and sync
- `SearchRAGWindow.py` - RAG search interface
- `Evals_Window.py` - LLM benchmarking
- `MediaWindow.py` - Media management hub
- Other windows follow similar patterns

**Key Widgets**:
- `chat_message_enhanced.py` - Rich messages with actions
- `tool_message_widgets.py` - Tool calling UI (ToolCallMessage, ToolResultMessage)
- `IngestTldwApi*Window.py` - Media-specific ingestion forms
- `form_components.py` - Standardized form builders

### Business Logic

- **`Chat/`** - `Chat_Functions.py` (conversation CRUD), `document_generator.py` (export formats)
- **`Character_Chat/`** - `Character_Chat_Lib.py`, `ccv3_parser.py` (card formats)
- **`Notes/`** - `sync_engine.py` (bidirectional sync), template system
- **`RAG_Search/`** - `simplified/` for streamlined implementation, `chunking_service.py`
- **`Tools/`** - `tool_executor.py`, built-in: DateTimeTool, CalculatorTool
- **`Evals/`** - `eval_orchestrator.py`, `eval_runner.py`, task-specific runners
- **`LLM_Calls/`** - Provider integrations, unified `chat_with_provider()` interface

### Data Layer (`DB/`)

- **`ChaChaNotes_DB.py`** - Main DB (conversations, messages, characters, notes), schema v7
- **`Client_Media_DB_v2.py`** - Media storage with chunking
- **`RAG_Indexing_DB.py`** - Vector storage (when enabled)
- Other DBs: Evals, Prompts, Subscriptions
- Patterns: soft deletion, optimistic locking, FTS5 triggers, parameterized queries only

### Event System (`Event_Handlers/`)

Event flow: Widget → post_message() → @on() handler → workers → reactive updates → UI refresh

Key events: ChatEvent, StreamingChunk, RAGSearchEvent, SyncEvent, EvalEvent, TabEvent

### Key Patterns

**Database Operations**:
```python
with db.transaction() as cursor:
    cursor.execute(query, params)  # Always parameterized
```

**Reactive UI**:
```python
class MyWidget(Widget):
    data = reactive([], recompose=True)  # Rebuilds UI
    status = reactive("idle")  # Refresh only
```

**Background Work**:
```python
self.run_worker(self._heavy_task, exclusive=True)

@work(thread=True)
def _heavy_task(self):
    result = process()
    self.call_from_thread(self.update_ui, result)
```

## Development Guidelines

### Adding Features

**New LLM Provider**:
1. Add to `LLM_Calls/` with `chat_with_provider()` method
2. Register in main caller
3. Add config section

**New Tab**:
1. Create Screen in `UI/`
2. Add TAB_X constant
3. Register in app.py compose()
4. Add event handlers

**New Tool**:
1. Extend Tool class in `Tools/`
2. Implement: get_name(), get_description(), get_parameters(), execute()
3. Register in AVAILABLE_TOOLS

### Security Requirements

- Validate all inputs via `input_validation.py`
- Use `path_validation.py` for file paths
- SQL identifiers through `sql_validation.py`
- API keys from env/config only, never logged
- Sanitize HTML/Markdown content

### Performance Rules

- Workers for operations >100ms
- Stream LLM responses
- Chunk large files
- Paginate DB results
- Clear caches on context switch

### Configuration

Priority: env vars → config.toml → defaults

Key sections:
- `[API]` - Provider keys
- `[splash_screen]` - Animation settings
- `[embeddings]` - RAG config
- Provider-specific sections

### Testing

- Run full suite before PRs
- Use real SQLite in-memory for DB tests
- Property-based testing with Hypothesis
- Markers: unit, integration, optional, asyncio

## Special Systems

### Tool Calling
- Schema v7 adds tool messages
- `tool_executor.py` handles execution
- Provider parsing implemented
- Status: Detection works, execution pending

### Config Encryption
- AES-256 with PBKDF2
- `Utils/config_encryption.py`
- Password dialogs in widgets

### Splash Screen
- 20+ animations in `splash_animations.py`
- Config: `[splash_screen]` section
- Custom cards in `examples/custom_splash_cards/`

### Notes Sync
- Bidirectional file ↔ DB
- Last-write-wins conflict resolution
- Background monitoring

### Pre-commit Hook
- `auto_review.py` for Claude Code integration
- Reviews diffs with LLM
- Exit 0 = pass, 2 = fail

## Project-Specific Gotchas

1. **No localStorage** in artifacts - use React state or JS variables
2. **Tailwind limitations** - Only core utility classes, no compilation
3. **Schema migrations** - Always increment version, add to migrations/
4. **Optional deps** - Check with `optional_deps.py` before importing
5. **Thread safety** - Use transaction() context manager
6. **Tab constants** - Must match IDs in compose()
7. **Streaming** - Always offer non-streaming fallback
8. **FTS5** - Triggers auto-update on text columns
9. **Workers** - Mark exclusive=True to prevent duplicates
10. **Reactive** - recompose=True rebuilds, default just refreshes

## File Reference

Critical files for common tasks:
- Entry: `app.py`, `config.py`, `Constants.py`
- Chat: `Chat_Functions.py`, `chat_message_enhanced.py`
- DB: `base_db.py`, `ChaChaNotes_DB.py`
- LLM: `LLM_API_Calls.py`, `model_capabilities.py`
- Security: `path_validation.py`, `input_validation.py`
- UI: `form_components.py`, reactive patterns in any widget

## Code Style

- Type hints for public APIs
- Docstrings: Google style with Args/Returns/Raises
- Imports: stdlib → third-party → local
- PascalCase classes, snake_case functions
- Pydantic for validation
- Early returns to reduce nesting
- Constants for magic values
- Context managers for resources
- Descriptive test names
- Profile before optimizing
- Validate at boundaries
- Log errors with context