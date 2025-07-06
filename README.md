# tldw_chatbook README

A Textual TUI for interacting with various LLM APIs, managing conversation history, characters, notes, and more.

Current status: Working/In-Progress

![Screenshot](https://github.com/rmusser01/tldw_chatbook/blob/main/static/PoC-Frontpage.PNG?raw=true)

## System Requirements
- Python ≥ 3.11
- Operating System: Windows, macOS, Linux
- Terminal with Unicode support

## Installation

### Quick Start (Core Features Only)
```bash
# Clone the repository
git clone https://github.com/rmusser01/tldw_chatbook
cd tldw_chatbook

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core package
pip install -e .

# Run the application
tldw-cli
# Or: python3 -m tldw_chatbook.app
```

### Installation with Optional Features
The application supports several optional feature sets that can be installed based on your needs:

```bash
# RAG (Retrieval-Augmented Generation) support
pip install -e ".[embeddings_rag]"

# Advanced text chunking and language detection
pip install -e ".[embeddings_rag,chunker]"

# Web search and scraping capabilities
pip install -e ".[websearch]"

# Image display in TUI
pip install -e ".[images]"

# All optional features
pip install -e ".[embeddings_rag,chunker,websearch,images]"

# Development installation
pip install -e ".[dev]"
```

### Optional Feature Groups

| Feature Group | Enables | Key Dependencies |
|--------------|---------|------------------|
| `embeddings_rag` | Vector search, semantic similarity, hybrid RAG | torch, transformers, sentence-transformers*, chromadb* |
| `chunker` | Advanced text chunking, language detection | nltk, langdetect, jieba, fugashi |
| `websearch` | Web scraping, content extraction | beautifulsoup4, playwright, trafilatura |
| `images` | Image display in TUI | textual-image, rich-pixels |
| `coding_map` | Code analysis features | grep_ast, pygments |
| `local_vllm` | vLLM inference support | vllm |
| `local_mlx` | MLX inference (Apple Silicon) | mlx-lm |

*Note: `sentence-transformers` and `chromadb` are detected separately and installed automatically when needed.

## Core Features (Always Available)
These features work out-of-the-box without any optional dependencies:

### General
- **Textual TUI interface** with keyboard navigation and mouse support
- **Configuration management** via `config.toml`
  - Default location: `~/.config/tldw_cli/config.toml`
  - Environment variable support for API keys
- **Multiple database support**
  - ChaChaNotes DB: Conversations, characters, and notes
  - Media DB: Ingested media files and metadata
  - Prompts DB: Saved prompt templates
  - Default location: `~/.local/share/tldw_cli/`

### LLM Support
- **Commercial LLM APIs**: OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, Mistral, OpenRouter, HuggingFace
- **Local LLM APIs**: Llama.cpp, Ollama, Kobold.cpp, vLLM, Aphrodite, Custom OpenAI-compatible endpoints
- **Streaming responses** with real-time display
- **Full conversation management**: Save, load, edit, fork conversations

### RAG (Basic - FTS5)
Even without optional dependencies, you get:
- **Full-text search** across all content using SQLite FTS5
- **BM25 ranking** for keyword relevance
- **Multi-source search**: Media, conversations, notes
- **Basic text chunking** for long documents
- **Dynamic chunking controls** in chat UI

## Enhanced Features (With Optional Dependencies)

### RAG (Advanced - with `embeddings_rag`)
Installing `pip install -e ".[embeddings_rag]"` adds:
- **Vector/Semantic Search**: Find conceptually similar content
- **Hybrid Search**: Combines keyword (FTS5) and vector search
- **ChromaDB Integration**: Persistent vector storage
- **Embeddings Generation**: Using Sentence Transformers
- **Re-ranking Support**: FlashRank or Cohere for better relevance
- **Advanced Caching**: Query and embedding result caching
- **Memory Management**: Automatic cleanup at configurable thresholds

#### Enabling the New Modular RAG System
```bash
# Set environment variable
export USE_MODULAR_RAG=true
# Or in config.toml: use_modular_service = true
```

#### Default Embedding Configuration
The embeddings_rag module comes with sensible defaults that work out of the box:
- **Default Model**: `mxbai-embed-large-v1` (1024 dimensions) - high-quality embeddings
- **Auto-device Detection**: Automatically uses GPU (CUDA/MPS) if available
- **Zero Configuration**: Works immediately after installation
- **Flexible Dimensions**: Supports Matryoshka - can use 512 or 256 dimensions for speed/storage

Common embedding models are pre-configured:
- **High Quality (Default)**: `mxbai-embed-large-v1` (~335MB, 1024d, supports 512d/256d)
- **State-of-the-Art**: 
  - `stella_en_1.5B_v5` (~1.5GB, 512-8192d, security-pinned)
  - `qwen3-embedding-4b` (~4GB, up to 4096d, 32k context)
- **Small/Fast**: `e5-small-v2`, `all-MiniLM-L6-v2` (~100MB, 384d)
- **Balanced**: `e5-base-v2`, `all-mpnet-base-v2` (~400MB, 768d)
- **Large Models**: `e5-large-v2`, `multilingual-e5-large-instruct` (~1.3GB, 1024d)
- **API-based**: OpenAI embeddings (requires API key)

See `tldw_chatbook/Config_Files/EMBEDDING_DEFAULTS_README.md` for detailed configuration options.

### Advanced Text Processing (with `chunker`)
- **Language-aware chunking**: Sentence and paragraph detection
- **Multi-language support**: Chinese (jieba), Japanese (fugashi)
- **Smart text splitting**: Respects linguistic boundaries
- **Chunking strategies**: Words, sentences, paragraphs, semantic units

### Chat Features
<details>
<summary>Full Chat Feature List</summary>

All chat features listed here work with the core installation:
- **Multi-provider LLM support** (see LLM Support section above)
- **Conversation Management**
  - Save, load, edit, delete conversations
  - Fork conversations at any point (Code is in, feature not supported yet)
  - Search by title, keywords, or content
  - Version history and rollback
- **Character/Persona System**
  - Import and manage character cards
  - Apply personas to conversations
  - Character-specific chat modes
- **Advanced Chat Features**
  - Streaming responses with real-time display
  - Message regeneration
  - Auto-generate questions/answers
  - Ephemeral conversations (not saved by default)
  - Strip thinking blocks from responses
- **Prompt Management**
  - Save, edit, clone prompts
  - Bulk import/export
  - Search and apply templates
- **RAG Integration** (enhanced with optional deps)
  - Enable/disable RAG per message
  - Configure chunk size and overlap
  - Select data sources (media, chats, notes)
  - View retrieved context
</details>

### Notes & Media Features
<details>
<summary>Notes & Media Features</summary>

**Notes System** (Core feature):
- Create, edit, and delete notes
- Search by title, keywords, or content
- Organize with keywords/tags
- Load notes into conversations
- Full-text search with FTS5

**Media Management** (Core feature):
- Ingest various media types (text, documents, transcripts)
- Search media by title, content, or metadata
- Integration with tldw API for processing
- Local processing options
- Media metadata tracking
- Full-text search across all media

**Enhanced with optional dependencies**:
- Vector search for semantic similarity (`embeddings_rag`)
- Web content ingestion (`websearch`)
- Advanced text extraction (`chunker`)
</details>

### Web Search & Scraping (with `websearch`)
- **Web content extraction**: Clean text from web pages
- **Advanced parsing**: Using BeautifulSoup and Trafilatura
- **Browser automation**: Playwright for dynamic content
- **Language detection**: For multi-lingual content
- **Integration with RAG**: Web content as knowledge source

### Image Support (with `images`)
- **Image display in TUI**: View images directly in terminal
- **Rich visual output**: Using rich-pixels
- **Screenshot viewing**: For debugging and documentation

### Local LLM Features
<details>
<summary>Local LLM Inference Options</summary>

**Core support** (no extra deps):
- Llama.cpp server integration (User provides local Llama.cpp binary)
- Ollama HTTP API
- Kobold.cpp API
- Any OpenAI-compatible endpoint

**Enhanced support** (with optional deps):
- **vLLM** (`local_vllm`): High-performance inference
- **MLX** (`local_mlx`): Optimized for Apple Silicon
- **Transformers** (`local_transformers`): HuggingFace models

**Management features**:
- Model downloading from HuggingFace (placeholder)
- Server health monitoring
- Automatic model loading
- Performance optimization settings
</details>


### Planned Features
<details>
<summary> Future Features </summary>

- **General**
  - Web Search functionality (e.g., ability to search the web for relevant information based on conversation history or notes or query)
  - Additional LLM provider support (e.g., more local providers, more commercial providers)
  - More robust configuration options (e.g., more environment variable support, more config.toml options)

- **Chat**
  - Conversation Forking + History Management thereof (Already implemented, but needs more testing/UI buildout)
  - Enhanced character chat functionality (e.g., ASCII art for pictures, 'Generate Character' functionality, backgrounds)
  - Improved conversation history management (e.g., exporting conversations, better search functionality)

- **Notes-related**
  - Improved notes and keyword management (Support for syncing notes from a local folder/file - think Obsidian)

- **Media-related**

- **Search Related**
  - Improved search functionality (e.g., more robust search options, better search results)
  - Support for searching across conversations, notes, characters, and media files (in, but needs to be improved)
  - Support for websearch (code is in place, but needs more testing/UI buildout)

- **Tools & Settings**
  - Support for DB backup management/restore
  - General settings management (e.g., ability to change application settings, like theme, font size, etc.)
  - Support for user preferences (e.g., ability to set user preferences, like default LLM provider, default character, etc.)
  - Support for user profiles (e.g., ability to create and manage user profiles, tied into preference sets)

- **LLM Management**
  - Cleanup and bugfixes

- **Stats**
  - I imagine this page as a dashboard that shows various statistics about the user's conversations, notes, characters, and media files.
  - Something fun and lighthearted, but also useful for the user to see how they are using the application.
  - This data will not be stored in the DB, but rather generated on-the-fly from the existing data.
  - This data will also not be uploaded to any external service, but rather kept local to the user's machine.
  - This is not meant for serious analytics, but rather for fun and lighthearted use. (As in it stays local.)

- **Evals**
  - Self-explanatory
  - Support for evaluating LLMs based on user-defined criteria.
  - Support for RAG evals.
  - Jailbreaks?
  - Backend exists, front-end does not.

- **Coding**
    - Why not, right?
    - Build out a take on the agentic coder, will be a longer-term goal, but will be a fun addition.

- **Workflows**
  - Workflows - e.g., Ability to create structured workflows, like a task list or a series of steps to follow, with the ability to execute them in order with checkpoints after each step. (Agents?)
  - Agentic functionality (e.g., ability to create agents that can perform tasks based on conversation history or notes, think workflow automation with checkpoints)
    - First goal will be the groundwork/framework for building it out more, and then for coding, something like Aider?
    - Separate from the workflows, which are more like structured task lists or steps to follow. Agentic functionality will be more about creating workflows, but not-fully structured, that adapt based on the 'agents' decisions.

- **Other Features**
  - Support for Server Syncing (e.g., ability to sync conversations, notes, characters, Media DB and prompts across devices)
  - Support for audio playback + Generation (e.g., ability to play audio files, generate audio from text - longer term goal, has to run outside of the TUI)
  - Mindmap functionality (e.g., ability to create mindmaps from conversation history or notes)

</details>


## Configuration

### First Run Setup
On first run, the application will:
1. Create a default configuration file at `~/.config/tldw_cli/config.toml`
2. Create necessary databases in `~/.local/share/tldw_cli/`
3. Initialize with core features enabled

### Configuration File
Edit `~/.config/tldw_cli/config.toml` to:
- Add API keys for LLM providers
- Configure RAG settings
- Enable/disable features
- Set UI preferences
- Configure embedding models (defaults to e5-small-v2)

Example embedding configuration:
```toml
[embedding_config]
default_model_id = "mxbai-embed-large-v1"  # High-quality default

[rag.embedding]
model = "mxbai-embed-large-v1"
device = "auto"  # Auto-detects best device (cuda/mps/cpu)
```

### Environment Variables
API keys can also be set via environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `COHERE_API_KEY`
- etc.

### Database Files
Located at `~/.local/share/tldw_cli/`:
- `ChaChaNotes.db`: Conversations, characters, and notes
- `media_v2.db`: Ingested media files and metadata
- `prompts.db`: Saved prompt templates
- `rag_indexing.db`: RAG indexing state (if using RAG features)

### Splash Screen Customization
The application includes a customizable splash screen system inspired by Call of Duty's calling cards. Features include:
- Multiple animation effects (Matrix rain, glitch, retro terminal)
- Progress tracking during startup
- Configurable duration and skip options
- Support for custom splash cards

For detailed information on creating and customizing splash screens, see the [Splash Screen Guide](Docs/Development/SPLASH_SCREEN_GUIDE.md).

## Upgrading from requirements.txt

If you previously installed using `requirements.txt`:
```bash
# Uninstall old dependencies
pip uninstall -r requirements.txt -y

# Install using pyproject.toml
pip install -e .  # or with optional features
```

### Project Structure
<details>
<summary>Here's a brief overview of the main directories in the project:</summary>

```
└── ./
    └── tldw_chatbook
        ├── assets
        │   └── Static Assets
        ├── Character_Chat
        │   └── Libaries relating to character chat functionality/interactions
        ├── Chat
        │   └── Libraries relating to chat functionality/orchestrations
        ├── Chunking
        │   └── Libaries relating to chunking text for LLMs
        ├── css
        │   └── CSS files for the Textual TUI
        ├── DB
        │   └── Core Database Libraries
        ├── Embeddings
        │   └── Embeddings Generation & ChromaDB Libraries
        ├── Event_Handlers
        │   ├── Chat_Events
        │   │   └── Handle all chat-related events
        │   ├── LLM_Management_Events
        │   │   └── Handle all LLM management-related events
        │   └── Event Handling for all pages is done here
        ├── LLM_Calls
        │   └── Libraries for calling LLM APIs (Local and Commercial)
        ├── Local_Inference
        │   └── Libraries for managing local inference of LLMs (e.g., Llama.cpp, llamafile, vLLM, etc.)
        ├── Metrics
        │   └── Library for instrumentation/tracking (local) metrics
        ├── Notes
        │   └── Libraries for managing notes interactions and storage
        ├── Prompt_Management
        │   └── Libraries for managing prompts interactions and storage + Prompt Engineering
        ├── RAG_Search
        │   └── Libraries for RAG (Retrieval-Augmented Generation) search functionality
        ├── Screens
        │   └── First attempt at Unifying the screens into a single directory
        ├── Third_Party
        │   └── All third-party libraries that are not part of the main application
        ├── tldw_api
        │   └── Code for interacting with the tldw API (e.g., for media ingestion/processing/web search)
        ├── TTS
        │   └── Libraries for Text-to-Speech functionality
        ├── UI
        │   └── Libraries containing all screens and panels for the Textual TUI
        ├── Utils
        │   └── All utility libraries that are standalone
        ├── Web_Scraping
        │   └── Libraries for web scraping functionality (e.g., for web search, RAG, etc.)
        ├── Widgets
        │   └── Reusable TUI components/widgets
        ├── app.py - Main application entry point (its big...)
        ├── config.py - Configuration management library
        ├── Constants.py - Constants used throughout the application (Some default values, Config file template, CSS template)
        └── Logging_Config.py - Logging configuration for the application
```
</details>

### Inspiration
https://github.com/darrenburns/elia

## Contributing
- Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.(WIP)
- (Realistically, this is a work in progress, so contributions are welcome, but please be aware that the codebase is still evolving and may change frequently.)
- Make a pull request against the `dev` branch, where development happens prior to being merged into `main`.

## License

This project is licensed under the GNU Affero General Public License - see the [LICENSE](LICENSE) file for details.

### Contact
For any questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/rmusser01/tldw) or contact me directly on the tldw_Project Discord or via the email in my profile.
