# tldw_chatbook Developer's Guide

All below LLM Generated:

This guide provides a more in-depth look into the architecture and development aspects of `tldw_chatbook`. It's intended for developers who want to contribute to the project or understand its internals.

## Project Architecture

`tldw_chatbook` is built using the [Textual](https://textual.textualize.io/) framework for its Terminal User Interface (TUI). The project follows a modular structure to separate concerns.

### Core Components:

*   **`tldw_chatbook/app.py`**:
    *   The main entry point of the application.
    *   Initializes the Textual `App` and sets up the primary UI layout, screens, and bindings.
    *   Handles global application state and events.

*   **`tldw_chatbook/Screens/`**:
    *   Contains different "screens" of the application. Each screen typically represents a major view or mode of operation (e.g., chat screen, notes screen, settings screen).
    *   Examples: `Stats_screen.py` (though more screens are expected to be here or will be added).

*   **`tldw_chatbook/Widgets/`**:
    *   Houses reusable UI components (widgets) used across various screens. These are custom Textual widgets tailored for the application's needs.
    *   Examples: `character_sidebar.py`, `chat_message.py`, `titlebar.py`.

*   **`tldw_chatbook/UI/`**:
    *   Contains modules that define larger, more complex UI structures, often composing multiple widgets or handling the logic for a specific panel or window within a screen.
    *   Examples: `Chat_Window.py`, `Notes_Window.py`.

*   **`tldw_chatbook/Event_Handlers/`**:
    *   Modules dedicated to handling specific types of events within the application. This helps decouple event logic from the UI components themselves.
    *   Examples: `chat_events.py`, `notes_events.py`, `app_lifecycle.py`.

### Backend and Logic:

*   **`tldw_chatbook/Chat/`**:
    *   `Chat_Functions.py`: Core logic for handling chat interactions, processing user input, and preparing LLM requests.
    *   `Chat_Deps.py`: Manages dependencies and configurations related to chat functionalities.
    *   `prompt_template_manager.py`: Handles the loading and management of prompt templates.

*   **`tldw_chatbook/Character_Chat/`**:
    *   `Character_Chat_Lib.py`: Library for managing character-specific chat features, including loading character data (e.g., from cards) and tailoring interactions.
    *   `ccv3_parser.py`: Parser for character card formats. (not actually built out or used - only for v3 character cards, which are not currently fully supported)

*   **`tldw_chatbook/Notes/`**:
    *   `Notes_Library.py`: Provides an API for managing notes, including CRUD operations (Create, Read, Update, Delete) and synchronization.

*   **`tldw_chatbook/DB/`**:
    *   Manages data persistence using SQLite. This includes:
        *   **`ChaChaNotes_DB.py`**: The primary database library for the application. It manages character profiles, chat conversations (including messages with versioning and branching support), notes, keywords, and their interconnections. Key features include optimistic locking, soft deletion, Full-Text Search (FTS5) capabilities, and a synchronization log for tracking changes.
        *   `Client_Media_DB_v2.py`: Database for managing metadata related to user's ingested media files.
        *   `Prompts_DB.py`: Database for storing and managing user-created and imported prompts.
        *   `Sync_Client.py`: Handles client-side logic for data synchronization (potentially with a server or across devices, details TBD based on its usage).

*   **`tldw_chatbook/LLM_Calls/`**:
    *   Modules responsible for making API calls to various Large Language Model (LLM) providers, both local and commercial.
    *   `LLM_API_Calls.py`: Handles calls to commercial/remote LLM APIs.
    *   `LLM_API_Calls_Local.py`: Handles calls to local LLM instances.
    *   `Summarization_General_Lib.py`, `Local_Summarization_Lib.py`: Libraries for text summarization functionalities.

*   **`tldw_chatbook/Prompt_Management/`**:
    *   `Prompt_Engineering.py`: Utilities and functions to help construct and refine prompts.
    *   `Prompts_Interop.py`: Manages the interaction between different prompt sources or formats.

### Configuration:

*   **`config.toml`**: The primary configuration file for users, typically located at `~/.config/tldw_cli/config.toml`. It stores API keys, default settings, and other preferences.
*   **`tldw_chatbook/config.py`**: Module for loading and accessing configuration settings within the application.
*   **`tldw_chatbook/Constants.py`**: Defines global constants used throughout the application.

### Data Flow (Simplified Example: Sending a Chat Message)

1.  User types a message in a chat window (UI component in `tldw_chatbook/UI/` or `tldw_chatbook/Widgets/`).
2.  The UI component triggers an event.
3.  An event handler in `tldw_chatbook/Event_Handlers/` (e.g., `chat_events.py`) captures this event.
4.  The event handler calls functions in `tldw_chatbook/Chat/Chat_Functions.py` to process the message.
5.  `Chat_Functions.py` might interact with `tldw_chatbook/LLM_Calls/` to send the message to an LLM.
6.  The LLM response is received and processed.
7.  The chat history is updated, potentially involving `tldw_chatbook/DB/ChaChaNotes_DB.py`.
8.  The UI is updated to display the new message and response.

## Key Design Principles

*   **Modularity**: Code is organized into distinct modules with specific responsibilities.
*   **Separation of Concerns**: UI logic, business logic, and data access are kept as separate as possible.
*   **Event-Driven**: Textual is an event-driven framework, and the application leverages this for handling user interactions and other occurrences.
*   **Configurability**: Users can configure the application via `config.toml` and environment variables.

## Adding a New LLM Provider

To add support for a new LLM provider:

1.  **Create a new API calling module** in `tldw_chatbook/LLM_Calls/` (if the interaction logic is significantly different from existing ones) or modify an existing one.
    *   This module should handle authentication, request formatting, and response parsing specific to the new provider.
2.  **Update configuration handling** in `tldw_chatbook/config.py` and the default `config.toml` to include settings for the new provider (e.g., API key, endpoint URL).
3.  **Integrate the new provider** into the relevant parts of `tldw_chatbook/Chat/Chat_Functions.py` or a similar dispatcher so users can select and use it.
4.  **Add new entries** to the `LLMProvider` enum in `tldw_chatbook/Constants.py` if applicable.
5.  **Write tests** for the new integration.

## Working with the TUI (Textual)

*   Familiarize yourself with the [Textual documentation](https://textual.textualize.io/docs).
*   UI components are defined as classes inheriting from Textual's `Widget` or other more specific widget types.
*   Layout is managed using Textual's layout system (e.g., vertical, horizontal, grid).
*   Styling is done via CSS-like files (e.g., `css/tldw_cli.tcss`).
*   User interactions are handled through event handlers (e.g., `@on(Button.Pressed)`).

## Data Storage

*   User data (conversations, notes, characters, media metadata, prompts) is stored in SQLite databases, typically located in a user-specific directory like `~/.share/tldw_cli/` (the exact path can be configured).
*   Database interactions are primarily handled by modules in `tldw_chatbook/DB/`. The core database for characters, chat histories, and notes is managed by `ChaChaNotes_DB.py`, which includes schema definitions, versioning, and data access methods.
*   When making changes that affect the database schema, consider how migrations will be handled for existing users (the `ChaChaNotes_DB.py` module includes schema versioning).

## Logging and Debugging

*   The project uses loguru. Configuration can be found in `tldw_chatbook/Logging_Config.py`.
*   Logs can be helpful for debugging. Textual also has its own debugging tools and console. Thankfully, I've added a 'Logs' tab to view logs directly within the TUI.

This guide provides a starting point. The codebase is the ultimate source of truth. Don't hesitate to explore it and ask questions if you're unsure about something.

## Enhancing Inline Code Comments

Clear and concise inline comments are crucial for maintainability and onboarding new developers. While writing self-documenting code is encouraged, comments should be used to explain:

*   Complex logic or algorithms.
*   The purpose of functions and classes, especially public APIs.
*   Non-obvious decisions or workarounds.
*   `TODO` or `FIXME` items with context.

### Suggested Areas for Initial Focus

As a starting point for improving code clarity, the following files/modules are good candidates for a review and enhancement of inline comments due to their complexity or centrality to the project:

*   `tldw_chatbook/app.py` (main entry point, global state management)
*   `tldw_chatbook/Chat/Chat_Functions.py` (core chat logic and LLM interaction orchestration)
*   `tldw_chatbook/DB/ChaChaNotes_DB.py` (database schema and interaction logic for core features like conversations, notes, and characters)
*   `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (handles diverse and potentially complex interactions with various external LLM APIs)
*   Modules within `tldw_chatbook/Event_Handlers/` (as event-driven logic can sometimes be complex to trace without good comments)

This is an ongoing effort, and all contributors are encouraged to add or improve comments as they work on different parts of the codebase.
