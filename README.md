# tldw_chatbook

A sophisticated Terminal User Interface (TUI) application built with the Textual framework for interacting with various Large Language Model APIs. It provides a complete ecosystem for AI-powered interactions including conversation management, character/persona chat, notes with bidirectional file sync, media ingestion, advanced RAG (Retrieval-Augmented Generation) capabilities, comprehensive LLM evaluation system, and optional web browser access.

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
# Or: 
python3 -m tldw_chatbook.app

# Run in web browser (requires 'web' feature)
pip install -e ".[web]"
tldw-cli --serve
# Or use dedicated command:
tldw-serve --port 8080
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

# All optional features
pip install -e ".[embeddings_rag,chunker,websearch,audio,video,pdf,ebook,nemo,mcp,chatterbox,local_tts,higgs_tts,ocr_docext,debugging,mlx_whisper,diarization,coding_map,local_vllm,local_mlx,local_transformers,web]"

# Common feature combinations
pip install -e ".[audio,video]"  # Media transcription (includes faster-whisper)
pip install -e ".[pdf,ebook]"    # Document processing
pip install -e ".[embeddings_rag,audio]"  # RAG + transcription
pip install -e ".[local_tts,chatterbox]"  # Text-to-speech
pip install -e ".[higgs_tts]"  # Higgs Audio V2 TTS (high-quality, voice cloning)
pip install -e ".[mcp]"  # Model Context Protocol integration
pip install -e ".[web]"  # Web server for browser-based access

# Transcription providers (choose one):
pip install -e ".[transcription_faster_whisper]"  # Default, works on all platforms
pip install -e ".[transcription_lightning_whisper]"  # Apple Silicon optimized
pip install -e ".[transcription_parakeet]"  # Real-time ASR for Apple Silicon

# For Apple Silicon users wanting better performance:
pip install -e ".[audio,transcription_lightning_whisper]"  # Audio + optimized transcription
pip install -e ".[video,transcription_parakeet]"  # Video + real-time transcription

# Development installation
pip install -e ".[dev]"
```

### Optional Feature Groups

| Feature Group                  | Enables | Key Dependencies |
|--------------------------------|---------|------------------|
| `embeddings_rag`               | Vector search, semantic similarity, hybrid RAG | torch, transformers, sentence-transformers*, chromadb* |
| `chunker`                      | Advanced text chunking, language detection | nltk, langdetect, jieba, fugashi |
| `websearch`                    | Web scraping, content extraction | beautifulsoup4, playwright, trafilatura |
| `coding_map`                   | Code analysis features | grep_ast, pygments |
| `local_vllm`                   | vLLM inference support | vllm |
| `local_mlx`                    | MLX inference (Apple Silicon) | mlx-lm |
| `transcription_faster_whisper` | CPU/CUDA optimized Whisper transcription | faster-whisper |
| `transcription_lightning_whisper` | Apple Silicon optimized Whisper | lightning-whisper-mlx |
| `transcription_parakeet`       | Real-time ASR for Apple Silicon | parakeet-mlx |
| `mlx_whisper`                  | Legacy: Both Apple Silicon transcription providers | lightning-whisper-mlx, parakeet-mlx |
| `audio`                        | Audio processing with transcription | faster-whisper, soundfile, yt-dlp |
| `video`                        | Video processing with transcription | faster-whisper, soundfile, yt-dlp |
| `media_processing`             | Combined audio/video processing | faster-whisper, soundfile, yt-dlp |
| `pdf`                          | PDF text extraction | pymupdf, docling |
| `ebook`                        | E-book processing | ebooklib, beautifulsoup4, defusedxml |
| `nemo`                         | NVIDIA Parakeet ASR models | nemo-toolkit[asr] |
| `local_transformers`           | HuggingFace transformers | transformers |
| `mcp`                          | Model Context Protocol integration | mcp |
| `chatterbox`                   | Chatterbox TTS model support | chatterbox |
| `local_tts`                    | Local TTS models (Kokoro ONNX) | kokoro-onnx, scipy, pyaudio |
| `ocr_docext`                   | OCR and document extraction | docext, gradio_client |
| `debugging`                    | Metrics and telemetry | prometheus-client, opentelemetry-api |
| `diarization`                  | Speaker diarization for audio | torch, torchaudio, speechbrain |
| `web`                          | Web server for browser access | textual-serve |

*Note: `sentence-transformers` and `chromadb` are detected separately and installed automatically when needed.

### Transcription Providers

The application supports multiple transcription providers. By default, `audio`, `video`, and `media_processing` extras include `faster-whisper` which works on all platforms. For better performance on specific hardware:

#### Available Providers:
- **faster-whisper** (Default): CPU/CUDA optimized implementation, works everywhere
- **lightning-whisper-mlx**: Apple Silicon optimized Whisper implementation
- **parakeet-mlx**: Real-time ASR optimized for Apple Silicon

#### Installation Examples:
```bash
# Default installation (includes faster-whisper)
pip install -e ".[audio]"

# Replace default with Apple Silicon optimized provider
pip install -e ".[audio,transcription_lightning_whisper]"

# Add additional provider alongside default
pip install -e ".[audio,transcription_parakeet]"

# Install only a specific provider (no audio/video processing libs)
pip install -e ".[transcription_parakeet]"
```

**Note for Apple Silicon users**: For the MLX-based providers, you may need to install with `--no-deps` and handle the tiktoken dependency separately if you encounter build errors.

### Special Installation: Higgs Audio TTS

Higgs Audio V2 is a state-of-the-art TTS system with zero-shot voice cloning capabilities. Due to its architecture, it requires manual installation from GitHub before using the pip extras.

#### Prerequisites
- Python 3.11+
- PyTorch (will be installed automatically)
- 8GB+ RAM (16GB+ recommended)
- ~6GB disk space for models

#### Installation Steps

**Option 1: Automated Installation (Recommended)**
```bash
# Unix/Linux/macOS
./scripts/install_higgs.sh

# Windows
scripts\install_higgs.bat
```

**Option 2: Manual Installation**

1. **Clone and install Higgs Audio (REQUIRED FIRST):**
```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio
pip install -r requirements.txt
pip install -e .
cd ..
```

2. **Install tldw_chatbook with Higgs support:**
```bash
pip install -e ".[higgs_tts]"
```

3. **Verify installation:**
```bash
python scripts/verify_higgs_installation.py
```

#### Troubleshooting
- If you get `ImportError: boson_multimodal not found`, ensure you completed step 1
- For CUDA support, install PyTorch with CUDA before step 1
- On macOS, you may need to install additional audio libraries: `brew install libsndfile`

- For detailed Higgs configuration and usage, see [Docs/Higgs-Audio-TTS-Guide.md](Docs/Development/TTS/Higgs-Audio-TTS-Guide.md).

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
- **Image support**
  - View images directly in terminal
  - Screenshot viewing for debugging
  - Vision model support for multimodal LLMs
- **Web server access** (optional)
  - Run the TUI in a web browser
  - Access from any device on your network
  - No terminal emulator required
  - Full functionality via browser interface

### Main Application Tabs
1. **Chat** - Advanced AI conversation interface with streaming support
2. **Chat Tabs** - Multiple concurrent chat sessions (disabled by default)
3. **Conversations, Characters & Prompts** - Manage conversations, character personas, and prompt templates
4. **Notes** - Advanced note-taking with bidirectional file sync
5. **Media** - Browse and manage ingested media content
6. **Search/RAG** - Hybrid search across all content (FTS5 + optional vectors)
7. **Media Ingestion** - Process documents, videos, audio, and web content
8. **Embeddings** - Create and manage vector embeddings
9. **Evaluations** - Comprehensive LLM benchmarking system
10. **LLM Management** - Local model server control
11. **Tools & Settings** - Configuration and utilities
12. **Stats** - Usage statistics and metrics
13. **Logs** - Application logs and debugging
14. **Coding** - AI-powered coding assistant (WIP)
15. **STTS** - Speech-to-Text and Text-to-Speech interface
16. **Subscriptions** - Content subscription tracking and monitoring

### LLM Support
- **Commercial LLM APIs**: OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, Mistral, OpenRouter, HuggingFace
- **Local LLM APIs**: Llama.cpp, Ollama, Kobold.cpp, vLLM, Aphrodite, MLX-LM, ONNX Runtime, Custom OpenAI-compatible endpoints
- **Streaming responses** with real-time display
- **Full conversation management**: Save, load, edit, fork conversations
- **Model capability detection**: Vision support, tool calling, etc.
- **Custom tokenizer support** for accurate token counting
- **Chat Tabs**: Multiple concurrent chat sessions (enable with `enable_chat_tabs = true` in config)

### RAG (Basic - FTS5)
Even without optional dependencies, you get:
- **Full-text search** across all content using SQLite FTS5
- **BM25 ranking** for keyword relevance
- **Multi-source search**: Media, conversations, notes
- **Basic text chunking** for long documents
- **Dynamic chunking controls** in chat UI

### Tool Calling System
- **Built-in tools**: DateTimeTool, CalculatorTool
- **Extensible framework**: Abstract Tool base class for custom implementations
- **Safe execution**: Timeouts and concurrency control
- **UI integration**: Dedicated widgets for tool calls and results
- **Provider support**: Multiple LLM providers with tool calling capabilities
- **Status**: Implementation complete, UI widgets functional, chat integration pending

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

### Evaluation System
A comprehensive LLM benchmarking framework supporting:
- **30+ evaluation task types**: Including:
  - Text understanding and generation
  - Reasoning and logic tasks
  - Language-specific evaluations
  - Code understanding and generation
  - Mathematical reasoning
  - Safety and bias evaluation
  - Creative content evaluation
  - Robustness testing
- **Specialized runners**: Task-specific evaluation implementations
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
- **Bidirectional file synchronization**:
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

### Text-to-Speech System
Comprehensive TTS support with multiple backends:
- **OpenAI TTS**: High-quality cloud-based synthesis
- **ElevenLabs**: Premium voice synthesis with custom voices
- **Kokoro ONNX** (with `local_tts`): Local neural TTS with no internet required
- **Chatterbox** (with `chatterbox`): Advanced local TTS model
- **Unified Interface**: Single API for all backends
- **Voice Selection**: Choose from available voices per backend
- **Audio Output**: Direct playback or save to file
- **STTS Tab**: Dedicated UI for speech synthesis and recognition

### Model Context Protocol (MCP) Integration
With the `mcp` optional dependency:
- **MCP Server**: Expose tldw_chatbook features as MCP tools
- **MCP Client**: Integrate with other MCP-compatible applications
- **Available Tools**: Search, RAG, media processing, conversation management
- **Seamless Integration**: Works with Claude Desktop and other MCP clients
- **Configuration**: Via `[mcp]` section in config.toml

### OCR and Document Extraction (with `ocr_docext`)
- **Advanced OCR**: Extract text from images and scanned documents
- **Document Analysis**: Structure extraction from complex documents
- **Multi-format Support**: PDFs, images, and mixed documents
- **Integration**: Works with media ingestion pipeline

### Debugging and Metrics (with `debugging`)
- **Prometheus Metrics**: Performance and usage tracking
- **OpenTelemetry**: Distributed tracing support
- **Local Metrics**: No external services required
- **Performance Analysis**: Identify bottlenecks and optimize

### Advanced Configuration
- **Config Encryption**: AES-256 encryption with password protection
- **Custom Tokenizers**: Support for model-specific tokenizer files
- **Model Capabilities**: Flexible configuration-based detection
- **Form Components**: Standardized UI form creation library
- **Theme System**: Multiple themes with CSS customization

### Splash Screen System
Customizable splash screens with 50+ animation effects:
- **Built-in effects**: MatrixRain, Glitch, Typewriter, Fireworks, and more
- **Custom splash cards**: Create your own with examples provided
- **Configuration**: Via `[splash_screen]` section in config.toml
- **Performance**: Async rendering with configurable duration

For detailed customization, see the [Splash Screen Guide](Docs/Development/SplashScreen s/SPLASH_SCREEN_GUIDE.md).

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

Example TTS configuration:
```toml
[tts]
default_backend = "openai"  # Options: openai, elevenlabs, kokoro, chatterbox
default_voice = "alloy"  # Backend-specific voice ID
auto_play = true  # Play audio automatically after generation

[tts.kokoro]
model_path = "models/kokoro-v0_19.onnx"  # Path to local model
voice = "af_bella"  # Available voices vary by model
```

Example MCP configuration:
```toml
[mcp]
enabled = true
server_port = 3000
allowed_tools = ["search", "rag", "media_ingest"]
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

Example web server configuration:
```toml
[web_server]
enabled = true
host = "localhost"  # Use "0.0.0.0" to allow external access
port = 8000
title = "tldw chatbook"
debug = false
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

## Web Server Access

The application can be run in a web browser using the optional `textual-serve` integration. This allows you to access the full TUI interface through any modern web browser, making it accessible from devices without terminal access or when SSH is not available.

### Installation
```bash
pip install -e ".[web]"
```

### Usage

#### Method 1: Using the --serve flag
```bash
tldw-cli --serve
# With custom options:
tldw-cli --serve --host 0.0.0.0 --port 8080
```

#### Method 2: Using the dedicated command
```bash
tldw-serve
# With options:
tldw-serve --host 0.0.0.0 --port 8080 --title "My TUI App"
```

### Command-line Options
- `--host`: Host address to bind to (default: localhost)
- `--port`: Port number to bind to (default: 8000)
- `--web-title` / `--title`: Custom title for the web page
- `--debug`: Enable debug mode (only for tldw-serve)

### Security Considerations
- By default, the server binds to `localhost` for local access only
- To allow external access, use `--host 0.0.0.0` but ensure proper firewall configuration
- The web server runs the application in a subprocess with restricted permissions
- No shell access is exposed through the web interface

### Use Cases
- Access the application from tablets or phones
- Use the TUI on systems without proper terminal emulators
- Share the interface with team members on the same network
- Run on a server and access remotely without SSH

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
        ├── MCP
        │   └── Model Context Protocol server and client implementation
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

### Local Models I recommend
<details>
<summary>**Local Models I Can Recommend - Click-Here**</summary>

### Local Models I recommend
- These are just the 'standard smaller' models I recommend, there are many more out there, and you can use any of them with this project.
  - One should also be aware that people create 'fine-tunes' and 'merges' of existing models, to create new models that are more suited to their needs.
  - This can result in models that may be better at some tasks but worse at others, so it's important to test and see what works best for you.
- Llama 3.1 - The native llamas will give you censored output by default, but you can jailbreak them, or use a finetune which has attempted to tune out their refusals. 

For commercial API usage for use with this project: Claude Sonnet 3.5, Cohere Command R+, DeepSeek, gpt4o. 
Flipside I would say none, honestly. The (largest players) will gaslight you and charge you money for it. Fun.
That being said they obviously can provide help/be useful(helped me make this app), but it's important to remember that they're not your friend, and they're not there to help you. They are there to make money not off you, but off large institutions and your data.
You are just a stepping stone to their goals.

From @nrose 05/08/2024 on Threads:
```
No, it’s a design. First they train it, then they optimize it. Optimize it for what- better answers?
  No. For efficiency. 
Per watt. Because they need all the compute they can get to train the next model.So it’s a sawtooth. 
The model declines over time, then the optimization makes it somewhat better, then in a sort of 
  reverse asymptote, they dedicate all their “good compute” to the next bigger model.Which they then 
  trim down over time, so they can train the next big model… etc etc.
None of these companies exist to provide AI services in 2024. They’re only doing it to finance the 
 things they want to build in 2025 and 2026 and so on, and the goal is to obsolete computing in general
  and become a hidden monopoly like the oil and electric companies. 
2024 service quality is not a metric they want to optimize, they’re forced to, only to maintain some 
  directional income
```
</details>


### Inspiration
- https://github.com/darrenburns/elia
- https://github.com/paulrobello/parllama

## Contributing
- Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.(WIP)
- (Realistically, this is a work in progress, so contributions are welcome, but please be aware that the codebase is still evolving and may change frequently.)
- Make a pull request against the `dev` branch, where development happens prior to being merged into `main`.

## License

This project is licensed under the GNU Affero General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.

### Contact
For any questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/rmusser01/tldw_chatbook) or contact me directly on the tldw_Project Discord or via the email in my profile.