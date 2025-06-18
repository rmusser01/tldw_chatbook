gf# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**tldw_chatbook** is a Terminal User Interface (TUI) application built with the Textual framework for interacting with various Large Language Model APIs. It provides conversation management, character/persona chat, notes, media ingestion, and RAG capabilities.

**License**: AGPLv3+  
**Python Version**: ≥3.11  
**Main Framework**: Textual (≥3.3.0)

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

The codebase follows a modular architecture with clear separation of concerns:

### Core Structure
- **`tldw_chatbook/app.py`** - Main entry point, initializes Textual app and global state
- **`App_Functions/UI/`** - Main window components for different tabs (Chat, Character, Notes, etc.)
- **`Widgets/`** - Reusable UI components (buttons, inputs, lists, etc.)
- **`Event_Handlers/`** - Decoupled event handling logic for UI interactions
- **`App_Functions/Chat/`** - Core chat functionality and conversation management
- **`App_Functions/Character_Chat/`** - Character/persona management and chat
- **`App_Functions/Notes/`** - Notes creation and management
- **`App_Functions/LLM_Calls/`** - LLM provider integrations (commercial and local)
- **`App_Functions/DB/`** - Database layer with SQLite implementations

### Key Databases
1. **ChaChaNotes_DB** - Characters, Chats, and Notes storage
2. **Media_DB_v2** - Ingested media files and metadata
3. **Prompts_DB** - User prompt templates

### Configuration System
- **Config File**: `~/.config/tldw_cli/config.toml` (created on first run)
- **User Data**: `~/.share/tldw_cli/` directory
- **Environment Variables**: Supported for API keys (e.g., `OPENAI_API_KEY`)

### Event-Driven Architecture
The app uses Textual's event system extensively:
- UI components emit custom events
- Event handlers in `Event_Handlers/` process these events
- Clear separation between UI and business logic

## LLM Provider Integration

The app supports multiple LLM providers through a unified interface:

### Commercial Providers
Located in `App_Functions/LLM_Calls/Commercial_APIs/`:
- OpenAI, Anthropic, Cohere, Google, Groq, Mistral, DeepSeek, HuggingFace, OpenRouter

### Local Providers  
Located in `App_Functions/LLM_Calls/Local_APIs/`:
- Llama.cpp, Ollama, Kobold.cpp, vLLM, Aphrodite, Custom OpenAI-compatible

### Adding New Providers
1. Create a new module in the appropriate directory
2. Implement the standard interface methods (chat completion, streaming)
3. Add provider configuration to the config system
4. Update UI components to include the new provider

## Key Development Patterns

### Database Operations
- All database operations use context managers for connection handling
- FTS5 search capabilities for text search
- Soft deletion pattern with `deleted_at` timestamps
- Conversation versioning and forking support

### UI Component Structure
- Components inherit from Textual's Widget classes
- Use reactive attributes for state management
- Emit custom events for decoupled communication
- Follow Textual's CSS styling conventions

### Error Handling
- Comprehensive logging using loguru
- User-friendly error messages in the UI
- Fallback mechanisms for API failures

### Testing Approach
- Unit tests for core functionality
- Use temporary in-memory SQLite databases for testing (not mocks)
- Test fixtures in `Tests/fixtures/`

## Important Notes

- **Development Branch**: Submit PRs to `dev` branch, not `main`
- **Code Style**: No enforced linter/formatter currently - follow existing patterns
- **Dependencies**: Keep core dependencies minimal, use optional dependencies for features
- **Configuration**: Always check config.toml for feature flags and settings
- **API Keys**: Never hardcode API keys - use environment variables or config file
- **Database Migrations**: No formal migration system - handle schema changes carefully
- **Entry Point**: Main CLI command is `tldw-cli` (defined in pyproject.toml)
- **Package Structure**: Main package is `tldw_chatbook` with modular subpackages