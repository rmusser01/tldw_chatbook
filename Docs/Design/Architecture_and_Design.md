# Architecture & Design of tldw_chatbook



## Project Architecture Overview (Simplified)

`tldw_chatbook` is a Textual-based Terminal User Interface (TUI) application. Here's a very high-level overview of its main components:

- **`tldw_chatbook/`**: The main package containing the application code.
- **`tldw_chatbook/app.py`**: The main entry point of the application, setting up the Textual app.
- **`tldw_chatbook/Character_Chat/`**: Logic related to character-based chat.
- **`tldw_chatbook/Chat/`**: Core logic for handling chat interactions, LLM API calls, and conversation management.
- **`tldw_chatbook/Chunking/`**: Advanced text chunking system with template support for document processing.
  - **`Chunk_Lib.py`**: Main chunking implementation with multiple methods (words, sentences, semantic, etc.)
  - **`chunking_templates.py`**: Template-based chunking system for complex pipelines
  - **`language_chunkers.py`**: Language-specific tokenization (Chinese, Japanese, etc.)
  - **`token_chunker.py`**: Token-based chunking with transformer support
  - **`templates/`**: JSON templates for various chunking strategies
- **`tldw_chatbook/DB/`**: Database interaction layer (SQLite).
  - **`ChaChaNotes_DB.py`**: SQLite Database library used for Chats, Characters, and Notes.
  - **`Client_Media_DB_v2.py`**: Manages Media database operations. This is where all ingested media is stored.
  - **`Prompts_DB.py`**: Manages the prompts database related operations.
  - **`Sync_Client.py`**: Handles synchronization of client DBs with the server. (WIP)
- **`tldw_chatbook/Event_Handlers/`**: Contains event handlers for various user interactions.
- **`tldw_chatbook/LLM_Calls/`**: Modules for making API calls to various LLM providers.
- **`tldw_chatbook/Metrics/`**: Contains modules for local application metrics. Local-Only.
- **`tldw_chatbook/Notes/`**: Interop Library for Notes functionality. (Eventually will hold local note syncing)
- **`tldw_chatbook/Prompt_Management/`**: Contains modules for managing prompts.
- **`tldw_chatbook/Screens/`**: Contains a single screen, this and UI should be merged into a single directory.
- **`tldw_chatbook/Third_Party/`**: Contains third-party libraries and utilities.
- **`tldw_chatbook/Tools/`**: Contains various interfaces to 3rd-party tools and services.
- **`tldw_chatbook/TTS/`**: Contains modules for Text-to-Speech functionality.
- **`tldw_chatbook/UI/`**: Contains different views of the application's various tabs (Was supposed to be screens but haven't bothered to rename it all).
- **`tldw_chatbook/Utils/`**: Contains utility functions and classes used across the application.
- **`tldw_chatbook/Web_Scraping/`**: Contains modules for web scraping functionality.
- **`tldw_chatbook/Widgets/`**: Reusable UI components used across different screens.
- **`tldw_chatbook/Config.py`**: Contains all configuration settings for the application, including API keys, database paths, and other settings.
- **`tldw_chatbook/Constants.py`**: Contains all constants used throughout the application, such as default values and error messages.
- **`tldw_chatbook/Logging_Config.py`**: Contains the logging configuration for the application, setting up loggers, handlers, and formatters.





## LLM Backend Integrations

This section details the various Large Language Model (LLM) inference backends integrated into `tldw_chatbook`.

### Llama.cpp Integration
- 

### Llamafile Integration

### Ollama Integration

### vLLM Integration

### Transformers Integration

### ONNX Runtime Integration

### MLX-LM Integration
- https://github.com/ml-explore/mlx-lm/tree/main

The application now supports MLX-LM for running local language models optimized for Apple Silicon hardware.
Users can manage MLX-LM instances via the "LLM Management" tab, allowing configuration of:

*   **Model Path**: Specify a HuggingFace model ID compatible with MLX or a path to a local MLX model.
*   **Server Host & Port**: Configure the network address for the MLX-LM server.
*   **Additional Arguments**: Pass extra command-line arguments to the `mlx_lm.server` process.

The integration starts a local `mlx_lm.server` process and interacts with it, assuming an OpenAI-compatible API endpoint (typically at `/v1`). This allows for efficient local inference leveraging MLX's performance benefits on supported hardware.


## Chunking System

The chunking system provides sophisticated text segmentation capabilities for document processing, RAG (Retrieval-Augmented Generation), and LLM context management.

### Architecture Overview

The chunking module (`tldw_chatbook/Chunking/`) implements a flexible, template-based approach to text chunking with the following components:

- **Core Chunking Engine** (`Chunk_Lib.py`): Provides multiple chunking methods including words, sentences, paragraphs, tokens, semantic, JSON, XML, and specialized methods for e-books and rolling summarization.

- **Template System** (`chunking_templates.py`): Enables declarative configuration of complex chunking pipelines with preprocessing and postprocessing stages. Templates are defined in JSON and support inheritance.

- **Language Support** (`language_chunkers.py`): Language-specific tokenization for Chinese (jieba), Japanese (fugashi), and other languages with NLTK fallback.

- **Token Support** (`token_chunker.py`): Precise token-based chunking using HuggingFace transformers with word-based fallback when transformers are unavailable.

### Key Features

- **Multiple Chunking Methods**: Words, sentences, paragraphs, tokens, semantic similarity, structured data (JSON/XML)
- **Template-Based Configuration**: Define reusable chunking strategies without code
- **Multi-Stage Pipelines**: Preprocessing → Chunking → Postprocessing
- **Language Awareness**: Automatic language detection and appropriate tokenization
- **Extensible Operations**: Register custom operations for specialized needs
- **Graceful Degradation**: Fallbacks for missing optional dependencies

### Integration Points

The chunking system integrates with:
- **Media Ingestion**: Processes documents during import with configurable strategies per media type
- **RAG System**: Creates optimally-sized chunks for embedding and retrieval
- **Chat System**: Manages context windows for LLM interactions
- **Notes System**: Chunks content for synchronization and search

### Usage Example

```python
from tldw_chatbook.Chunking import Chunker

# Simple usage
chunker = Chunker()
chunks = chunker.chunk_text(text, method="semantic")

# Template-based usage
chunker = Chunker(template="academic_paper")
chunks = chunker.chunk_text(research_paper_text)
```

For detailed information, see:
- [Chunking Architecture](Chunking/Chunking_Architecture.md)
- [Chunking API Reference](Chunking/Chunking_API_Reference.md)
- [Chunking Templates Guide](Chunking/Chunking_Templates_Guide.md)
- [Chunking Use Cases](Chunking/Chunking_Use_Cases.md)
