# tldw_chatbook

A sophisticated Terminal User Interface (TUI) application built with the Textual framework for interacting with various Large Language Model APIs. It provides a complete ecosystem for AI-powered interactions including conversation management, character/persona chat, notes with bidirectional file sync, media ingestion, advanced RAG (Retrieval-Augmented Generation) capabilities, and comprehensive LLM evaluation system.

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
pip install -e ".[embeddings_rag,chunker,websearch,images,audio,video,pdf,ebook,nemo]"

# Common feature combinations
pip install -e ".[audio,video]"  # Media transcription
pip install -e ".[pdf,ebook]"    # Document processing
pip install -e ".[embeddings_rag,audio]"  # RAG + transcription

# Development installation
pip install -e ".[dev]"
```

### Optional Feature Groups

| Feature Group                  | Enables | Key Dependencies |
|--------------------------------|---------|------------------|
| `embeddings_rag`               | Vector search, semantic similarity, hybrid RAG | torch, transformers, sentence-transformers*, chromadb* |
| `chunker`                      | Advanced text chunking, language detection | nltk, langdetect, jieba, fugashi |
| `websearch`                    | Web scraping, content extraction | beautifulsoup4, playwright, trafilatura |
| `images`                       | Image display in TUI | textual-image, rich-pixels |
| (WIP/Not working) `coding_map` | Code analysis features | grep_ast, pygments |
| `local_vllm`                   | vLLM inference support | vllm |
| `local_mlx`                    | MLX inference (Apple Silicon) | mlx-lm |
| `audio`                        | Audio transcription (Whisper) | faster-whisper, soundfile |
| `video`                        | Video processing & transcription | faster-whisper, yt-dlp |
| `pdf`                          | PDF text extraction | pymupdf, docling |
| `ebook`                        | E-book processing | ebooklib, beautifulsoup4, defusedxml |
| `nemo`                         | NVIDIA Parakeet ASR models | nemo-toolkit[asr] |
| `local_transformers`           | HuggingFace transformers | transformers |

*Note: `sentence-transformers` and `chromadb` are detected separately and installed automatically when needed.

## Core Features (Always Available)

### General
- **Textual TUI interface** with keyboard navigation and mouse support
- **Configuration management** via `config.toml`
  - Default location: `~/.config/tldw_cli/config.toml`
  - Environment variable support for API keys
  - AES-256 encryption for sensitive config data - Option to password protect config file, encrypt on program exit, decrypt in memory at launch
- **Multiple database support**
  - ChaChaNotes DB: Conversations, characters, and notes
  - Media DB: Ingested media files and metadata
  - Prompts DB: Saved prompt templates with versioning
  - Evals DB: LLM evaluation results and benchmarks
  - Subscriptions DB: Content subscription tracking
  - Default location: `~/.local/share/tldw_cli/`

### Main Application Tabs
1. **Chat** - Advanced AI conversation interface with streaming support
2. **Conversations** - Manage, search, and organize chat history
3. **Character Chat** - Character-based interactions with imported personas
4. **Notes** - Advanced note-taking with bidirectional file sync
5. **Search/RAG** - Hybrid search across all content (FTS5 + optional vectors)
6. **Media Ingestion** - Process documents, videos, audio, and web content
7. **Prompts** - Template management with versioning
8. **Coding** - AI-powered coding assistant
9. **Embeddings** - Create and manage vector embeddings
10. **Evaluations** - Comprehensive LLM benchmarking system
11. **Logs** - Application logs and debugging
12. **LLM Management** - Local model server control
13. **Tools & Settings** - Configuration and utilities
14. **Stats** - Usage statistics and metrics

### LLM Support
- **Commercial LLM APIs**: OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, Mistral, OpenRouter, HuggingFace
- **Local LLM APIs**: Llama.cpp, Ollama, Kobold.cpp, vLLM, Aphrodite, MLX-LM, ONNX Runtime, Custom OpenAI-compatible endpoints
- **Streaming responses** with real-time display
- **Full conversation management**: Save, load, edit, fork conversations
- **Model capability detection**: Vision support, tool calling, etc.
- **Custom tokenizer support** for accurate token counting

### RAG (Basic - FTS5)
Even without optional dependencies, you get:
- **Full-text search** across all content using SQLite FTS5
- **BM25 ranking** for keyword relevance
- **Multi-source search**: Media, conversations, notes
- **Basic text chunking** for long documents
- **Dynamic chunking controls** in chat UI

### Tool Calling System (untested)
- **Built-in tools**: DateTimeTool, CalculatorTool
- **Extensible framework**: Abstract Tool base class for custom implementations
- **Safe execution**: Timeouts and concurrency control
- **UI integration**: Dedicated widgets for tool calls and results
- **Provider support**: Multiple LLM providers with tool calling capabilities

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

### Evaluation System (WIP)
A comprehensive LLM benchmarking framework supporting:
- **27+ evaluation task types**: Including:
  - Text understanding and generation
  - Reasoning and logic tasks
  - Language-specific evaluations
  - Code understanding and generation
  - Mathematical reasoning
- **Advanced metrics**: ROUGE, BLEU, F1, semantic similarity, perplexity
- **Comparison tools**: Side-by-side model performance analysis
- **Export capabilities**: Results in various formats
- **Cost estimation**: Token usage and pricing calculations

### Local File Ingestion
Programmatic API for ingesting files without UI interaction:
- **30+ file types supported**: Documents, e-books, text, structured data
- **Batch processing**: Handle multiple files efficiently
- **Directory scanning**: Recursive file discovery
- **Flexible processing**: Chunking, analysis, custom prompts
- **Full integration**: Uses same processors as UI

See `tldw_chatbook/Local_Ingestion/README.md` for API documentation.

### Chat Features
<details>
<summary>Full Chat Feature List</summary>

All chat features listed here work with the core installation:
- **Multi-provider LLM support** (see LLM Support section above)
- **Conversation Management**
  - Save, load, edit, delete conversations
  - Fork conversations at any point
  - Search by title, keywords, or content
  - Version history and rollback
  - Document generation (timeline, study guide, briefing)
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
  - Cost estimation widget (WIP)
  - Tool calling integration
- **Prompt Management**
  - Save, edit, clone prompts
  - Bulk import/export
  - Search and apply templates
  - Version tracking
- **RAG Integration** (enhanced with optional deps)
  - Enable/disable RAG per message
  - Configure chunk size and overlap
  - Select data sources (media, chats, notes)
  - View retrieved context
</details>

### Notes System
**Advanced Features**:
- Create, edit, and delete notes with rich markdown support
- **Bidirectional file synchronization**: (WIP)
  - Automatic sync between database and file system
  - Conflict resolution with backup
  - File system monitoring for changes
  - Background sync service
- **Template system** for structured note creation
- Search by title, keywords, or content
- Organize with keywords/tags
- Load notes into conversations
- Full-text search with FTS5

### Media Management
**Core features**:
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
- Document processing (PDF, EPUB, Word, etc.)
- Audio/video transcription

### Web Search & Scraping (with `websearch`)
- **Web content extraction**: Clean text from web pages
- **Advanced parsing**: Using BeautifulSoup and Trafilatura
- **Browser automation**: Playwright for dynamic content
- **Language detection**: For multi-lingual content
- **Integration with RAG**: Web content as knowledge source
- **Multiple search providers**: Google, Bing, DuckDuckGo, Brave, Kagi, Tavily, SearX, Baidu, Yandex

### Image Support (with `images`)
- **Image display in TUI**: View images directly in terminal
- **Rich visual output**: Using rich-pixels
- **Screenshot viewing**: For debugging and documentation
- **Vision model support**: For multimodal LLMs

### Media Processing Features

#### Audio Transcription (with `audio` or `nemo`)
- **Multiple transcription engines**:
  - **Whisper models** via faster-whisper (default)
  - **NVIDIA Parakeet** models for low-latency transcription (with `nemo`)
  - **Qwen2Audio** for multimodal understanding
- **Parakeet models** (optimized for real-time):
  - TDT (Transducer): Best for streaming applications
  - CTC: Fast batch processing
  - RNN-T: Balance of speed and accuracy
- **Audio format support**: WAV, MP3, M4A, FLAC, and more
- **YouTube/URL audio extraction**: Download and transcribe from URLs
- **Voice Activity Detection**: Filter silence automatically
- **GPU acceleration**: CUDA and Apple Metal support

#### Video Processing (with `video`)
- **Extract audio from videos**: Any format supported by ffmpeg
- **Transcribe video content**: Using any supported ASR model
- **YouTube video support**: Direct download and processing
- **Batch processing**: Handle multiple videos efficiently

#### Document Processing
- **PDF extraction** (with `pdf`): Text, layout, and metadata extraction using PyMuPDF and Docling
- **E-book support** (with `ebook`): EPUB, MOBI, AZW processing
- **Office documents**: Word, PowerPoint, Excel files
- **Advanced chunking**: Preserve document structure
- **Metadata preservation**: Author, title, creation date

### Local LLM Features
<details>
<summary>Local LLM Inference Options</summary>

**Core support** (no extra deps):
- Llama.cpp server integration
- Ollama HTTP API
- Kobold.cpp API
- Any OpenAI-compatible endpoint

**Enhanced support** (with optional deps):
- **vLLM** (`local_vllm`): High-performance inference
- **MLX** (`local_mlx`): Optimized for Apple Silicon
- **Transformers** (`local_transformers`): HuggingFace models
- **ONNX Runtime**: Cross-platform inference

**Management features**:
- Model downloading from HuggingFace
- Server health monitoring
- Automatic model loading
- Performance optimization settings
</details>

### Subscription System
- **Content monitoring**: Track updates to subscribed content
- **Periodic checking**: Automated update detection
- **Notification system**: Alert on new content
- **Flexible scheduling**: Configure update frequencies

### Advanced Configuration
- **Config Encryption**: AES-256 encryption with password protection
- **Custom Tokenizers**: Support for model-specific tokenizer files
- **Model Capabilities**: Flexible configuration-based detection
- **Form Components**: Standardized UI form creation library
- **Theme System**: Multiple themes with CSS customization

### Splash Screen System
Customizable splash screens with 20+ animation effects:
- **Built-in effects**: MatrixRain, Glitch, Typewriter, Fireworks, and more
- **Custom splash cards**: Create your own with examples provided
- **Configuration**: Via `[splash_screen]` section in config.toml
- **Performance**: Async rendering with configurable duration

For detailed customization, see the [Splash Screen Guide](Docs/Development/SPLASH_SCREEN_GUIDE.md).

### Coding Assistant
- **AI-powered code assistance**: In dedicated coding tab
- **Code mapping**: Analysis and understanding of codebases
- **Integration ready**: Framework for future enhancements

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
- Configure embedding models
- Customize splash screens
- Set up config encryption

Example embedding configuration:
```toml
[embedding_config]
default_model_id = "mxbai-embed-large-v1"  # High-quality default

[rag.embedding]
model = "mxbai-embed-large-v1"
device = "auto"  # Auto-detects best device (cuda/mps/cpu)
```

Example audio transcription configuration:
```toml
[transcription]
# Use NVIDIA Parakeet for low-latency transcription
default_provider = "parakeet"  # Options: faster-whisper, qwen2audio, parakeet
default_model = "nvidia/parakeet-tdt-1.1b"  # TDT model for streaming
device = "cuda"  # Use GPU for faster processing
use_vad_by_default = true  # Voice Activity Detection
```

Example splash screen configuration:
```toml
[splash_screen]
enabled = true
duration = 3.0
card_selection = "random"  # Options: random, sequential, or specific card name
active_cards = ["default", "cyberpunk", "minimalist"]
animation_speed = 1.0
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
- `evals.db`: Evaluation results and benchmarks
- `subscriptions.db`: Content subscription tracking
- `search_history.db`: Search query history

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
        │   └── Libraries relating to character chat functionality/interactions
        ├── Chat
        │   └── Libraries relating to chat functionality/orchestrations
        ├── Chunking
        │   └── Libraries relating to chunking text for LLMs
        ├── Coding
        │   └── Code assistance and mapping functionality
        ├── Config_Files
        │   └── Configuration templates and defaults
        ├── css
        │   ├── core/         # Base styles and variables
        │   ├── components/   # Component-specific styles
        │   ├── features/     # Feature-specific styles
        │   ├── layout/       # Layout and grid systems
        │   ├── utilities/    # Utility classes
        │   └── Themes/       # Theme definitions
        ├── DB
        │   └── Core Database Libraries (7 specialized databases)
        ├── Embeddings
        │   └── Embeddings Generation & ChromaDB Libraries
        ├── Evals
        │   └── Comprehensive evaluation system components
        ├── Event_Handlers
        │   ├── Chat_Events
        │   │   └── Handle all chat-related events
        │   ├── LLM_Management_Events
        │   │   └── Handle all LLM management-related events
        │   └── Event Handling for all tabs and features
        ├── Helper_Scripts
        │   ├── Character_Cards/  # Sample character cards
        │   └── Prompts/         # Extensive prompt library
        ├── LLM_Calls
        │   └── Libraries for calling LLM APIs (Local and Commercial)
        ├── Local_Inference
        │   └── Libraries for managing local inference of LLMs
        ├── Local_Ingestion
        │   └── Programmatic file ingestion API
        ├── Metrics
        │   └── Library for instrumentation/tracking (local) metrics
        ├── Notes
        │   └── Libraries for notes management and synchronization
        ├── Prompt_Management
        │   └── Libraries for managing prompts interactions and storage
        ├── RAG_Search
        │   └── Libraries for RAG (Retrieval-Augmented Generation) search
        ├── Screens
        │   └── Complex UI screen implementations
        ├── Third_Party
        │   └── All third-party libraries integrated
        ├── tldw_api
        │   └── Code for interacting with the tldw API
        ├── Tools
        │   └── Tool calling system implementation
        ├── TTS
        │   └── Libraries for Text-to-Speech functionality
        ├── UI
        │   └── Libraries containing all screens and panels for the TUI
        ├── Utils
        │   └── All utility libraries (encryption, splash, validation, etc.)
        ├── Web_Scraping
        │   └── Libraries for web scraping and search functionality
        ├── Widgets
        │   └── Reusable TUI components/widgets
        ├── app.py - Main application entry point
        ├── config.py - Configuration management library
        ├── Constants.py - Constants used throughout the application
        └── model_capabilities.py - Model capability detection
```
</details>

### Inspiration
https://github.com/darrenburns/elia

## Contributing
- Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.(WIP)
- (Realistically, this is a work in progress, so contributions are welcome, but please be aware that the codebase is still evolving and may change frequently.)
- Make a pull request against the `dev` branch, where development happens prior to being merged into `main`.

## License

This project is licensed under the GNU Affero General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.

### Contact
For any questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/rmusser01/tldw_chatbook) or contact me directly on the tldw_Project Discord or via the email in my profile.