<<<<<<< HEAD
# tldw_chatbook

A sophisticated Terminal User Interface (TUI) application built with the Textual framework for interacting with various Large Language Model APIs. The product is organized around a chat-first **master shell**: the **Console** is the main work surface, with **Home** (triage/status) and **Library** (notes, media, study, ingestion, search) as the other top-priority destinations, alongside supporting surfaces (Personas, Artifacts, Watchlists, and model/agent tools) that hand context back into the active conversation.

> 📖 **New here?** The [User Guide](Docs/User_Guide/index.md) walks through
> every screen — what it does and how to use it.

![Screenshot](https://github.com/rmusser01/tldw_chatbook/blob/main/static/PoC-Frontpage.PNG?raw=true)

## Project Status & Recent UI Overhaul

tldw_chatbook is in active development (currently `v0.1.8.0`, pre-1.0). The `dev` branch has landed a major UI/navigation overhaul that reorganizes the app around a **master shell** with a small set of primary destinations instead of a flat tab bar. Older tabs (Chat, Notes, Media, Ingest, Search, Coding, Characters/Prompts, Subscriptions, Chatbooks) remain reachable as routes/aliases during migration, but are no longer separate primary destinations.

**Highlights:**

- **Console** — the chat-first primary work surface (formerly *Chat*). A dual-audience redesign: a first-run **setup card** walks new users to their first message (progressive disclosure, no docs required), while power users get a keyboard-first layer — command palette, session switcher (`Ctrl+K`), quick model popover (`Alt+M`), and direct copy/edit/regenerate on messages. The left rail is organized into collapsible **Session / Context / Model / Details** sections with auto-titled, recent-first conversations.
- **Home** — a triage surface: a rail of **Needs Attention / Running / Recent** rows with a focus canvas that shows the selected item and its actions (approve, reject, retry, open).
- **Library** — the landing page for local content, organized around four verbs: **(re)view · search · ingest · create**. It absorbs media browsing (an in-Library viewer), Search/RAG, ingestion, Study (flashcards/quizzes), and **Notes** — the standalone Notes tab has been retired and its workbench now lives inside Library.
- **Personas** — a Console-parity workbench for characters, personas, and prompts (avatar upload, markdown / character-card import).
- Supporting destinations: **Artifacts** (generated outputs + Chatbooks), **Watchlists** (monitored sources), **Schedules**, **Workflows**, **MCP**, **ACP**, **Skills**, and **Settings**.

Design specs for the overhaul live in [`Docs/superpowers/specs/`](Docs/superpowers/specs/) — see `2026-07-02-console-dual-audience-ux-design.md` and `2026-07-04-home-library-redesign-design.md`.

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
python3 -m tldw_chatbook
# Installed command alternative:
tldw-cli

# Run in web browser (requires 'web' feature)
pip install -e ".[web]"
tldw-cli --serve
# Or use dedicated command:
tldw-serve --port 8080
```

### Local-first baseline

The core local-first app installs with `pip install -e .` and includes the Textual shell, Console, local conversations, notes, personas, Library browsing, Chatbook artifacts, and settings. Missing optional extras should be treated as unavailable advanced capabilities, not as a broken core install.

For packaged installs, use:

```bash
pip install tldw_chatbook
```

### Installation with Optional Features
The application supports several advanced optional capability groups that can be installed based on your needs. Source checkouts use `pip install -e ".[extra]"`; packaged installs use commands such as `pip install "tldw_chatbook[embeddings_rag]"`.

```bash
# RAG (Retrieval-Augmented Generation) support
pip install -e ".[embeddings_rag]"
pip install "tldw_chatbook[embeddings_rag]"

# Advanced text chunking and language detection
pip install -e ".[embeddings_rag,chunker]"

# Web search and scraping capabilities
pip install -e ".[websearch]"

# Most optional features (full extras list lives in pyproject.toml's [project.optional-dependencies])
pip install -e ".[embeddings_rag,chunker,websearch,audio,video,pdf,ebook,nemo,mcp,chatterbox,local_tts,higgs_tts,ocr_docext,debugging,mlx_whisper,diarization,coding_map,local_vllm,local_mlx,local_transformers,web,speech_recording,realtime]"

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

Advanced optional capability groups:

| Feature Group                  | Capability Area | Enables | Key Dependencies |
|--------------------------------|-----------------|---------|------------------|
| `embeddings_rag`               | RAG and retrieval | Vector search, semantic similarity, hybrid RAG | torch, transformers, sentence-transformers*, chromadb* |
| `chunker`                      | RAG and retrieval | Advanced text chunking, language detection | nltk, langdetect, jieba, fugashi |
| `websearch`                    | Server/research | Web search and scraping | beautifulsoup4, playwright, trafilatura |
| `coding_map`                   | Local inference | Code analysis features | grep_ast, pygments |
| `local_vllm`                   | Local inference | vLLM inference support | vllm |
| `local_mlx`                    | Local inference | MLX inference (Apple Silicon) | mlx-lm |
| `transcription_faster_whisper` | Media ingestion and transcription | CPU/CUDA optimized Whisper transcription | faster-whisper |
| `transcription_lightning_whisper` | Media ingestion and transcription | Apple Silicon optimized Whisper | lightning-whisper-mlx |
| `transcription_parakeet`       | Media ingestion and transcription | Parakeet ONNX transcription (cross-platform) | onnx-asr |
| `mlx_whisper`                  | Media ingestion and transcription | Legacy: Both Apple Silicon transcription providers | lightning-whisper-mlx, parakeet-mlx |
| `audio`                        | Media ingestion and transcription | Audio processing with transcription | faster-whisper, soundfile, yt-dlp |
| `video`                        | Media ingestion and transcription | Video processing with transcription | faster-whisper, soundfile, yt-dlp |
| `media_processing`             | Media ingestion and transcription | Combined audio/video processing | faster-whisper, soundfile, yt-dlp |
| `pdf`                          | Media ingestion and transcription | PDF text extraction | pymupdf, docling |
| `ebook`                        | Media ingestion and transcription | E-book processing | ebooklib, beautifulsoup4, defusedxml |
| `nemo`                         | Media ingestion and transcription | NVIDIA Parakeet ASR models | nemo-toolkit[asr] |
| `local_transformers`           | Local inference | HuggingFace transformers | transformers |
| `mcp`                          | MCP integration | Model Context Protocol integration | mcp |
| `chatterbox`                   | Media ingestion and transcription | Chatterbox TTS model support | chatterbox |
| `local_tts`                    | Media ingestion and transcription | Local TTS models (Kokoro ONNX) | kokoro-onnx, scipy, pyaudio |
| `ocr_docext`                   | Media ingestion and transcription | OCR and document extraction | docext, gradio_client |
| `debugging`                    | Server/research | Metrics and telemetry | prometheus-client, opentelemetry-api |
| `diarization`                  | Media ingestion and transcription | Speaker diarization for audio | torch, torchaudio, speechbrain |
| `web`                          | Web access | Web server for browser access | textual-serve |

*Note: `sentence-transformers` and `chromadb` are detected separately and installed automatically when needed.

### Transcription Providers

The application supports multiple transcription providers. By default, `audio`, `video`, and `media_processing` extras include `faster-whisper` which works on all platforms. For better performance on specific hardware:

#### Available Providers:
- **faster-whisper** (Default): CPU/CUDA optimized implementation, works everywhere
- **parakeet-onnx**: Parakeet ONNX transcription, cross-platform (installed by the `transcription_parakeet` extra and by `audio`/`video`)
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
./Helper_Scripts/Higgs-Install/install_higgs.sh

# Windows
Helper_Scripts\Higgs-Install\install_higgs.bat
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
python Helper_Scripts/Higgs-Install/verify_higgs_installation.py
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

### Main Application Destinations
The screen shell is organized around a **master shell** of primary destinations (not a flat, equal-weight tab bar). Listed in navigation order:

1. **Home** — Dashboard, notifications, status, and next actions (triage rail + focus canvas)
2. **Console** — Live agent conversations: streaming chat, approvals, tool use, RAG, and runs
3. **Library** — Workspaces and local source material: notes, media, imported content, conversations, Study (flashcards/quizzes), and Search/RAG — organized around *(re)view · search · ingest · create*
4. **Artifacts** — Generated outputs, bundles, reports, datasets, and Chatbooks
5. **Personas** — Characters, personas, prompts, dictionaries, and behavior profiles
6. **Watchlists** — Monitored sources, runs, alerts, and recovery (formerly *Subscriptions*)
7. **Schedules** — When jobs, watchlists, and workflows run
8. **Workflows** — Reusable procedures, recipes, dry-runs, and outputs
9. **MCP** — MCP servers, tools, permissions, auth, and audit
10. **ACP** — Agent Client Protocol agents, sessions, runtimes, diffs, and terminals
11. **Skills** — Agent Skills packs, discovery, validation, and attachments
12. **Settings** — Global app preferences, appearance, accounts, and storage

> **Migration note:** legacy tabs — `Chat` (now Console), `Notes`, `Media`, `Ingest`, `Search`, `Coding`, `Characters/Prompts` (now Personas), `Subscriptions` (now Watchlists), and `Chatbooks` (now under Artifacts) — still resolve as routes/aliases, but are no longer separate primary destinations. `Coding` in particular is now a thin compatibility stub; agentic programming happens in the Console.

Console exposes three separate ways to use local evidence: **Manual Search
Library** is always available, while each conversation independently stores
**Auto: Never / Automatic** and **Assistant: Blocked / Allowed**. When
assistant access is allowed, **Direct / RAG** selects the built-in Library tool
surface; it is not another permission switch. New conversations default to
Never and Blocked. See [Console context and RAG](Docs/User_Guide/console/context-and-rag.md#per-conversation-library-controls).

### LLM Support
- **Commercial LLM APIs**: OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, Mistral, OpenRouter, QwenCloud, HuggingFace, Moonshot (Kimi), Z.ai (GLM)
- **Local LLM APIs**: Llama.cpp, Ollama, Kobold.cpp, vLLM, Aphrodite, MLX-LM, ONNX Runtime, Custom OpenAI-compatible endpoints
- **Streaming responses** with real-time display
- **Full conversation management**: Save, load, edit, fork conversations
- **Durable tool continuation foundation**: provider integrations can opt in to
  checkpoint an interrupted tool run on its exact assistant branch, then offer
  explicit Resume or Discard recovery without re-running completed tools
- **Model capability detection**: Vision support, tool calling, etc.
- **Custom tokenizer support** for accurate token counting

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
- **Chat integration**: Tool calls, results, and approvals render inline in the Console

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

#### Default Embedding Configuration
The embeddings_rag module comes with sensible defaults that work out of the box:
- **Default Model**: `e5-small-v2` (384 dimensions) - the shipped `[embedding_config] default_model_id`
- **Auto-device Detection**: Automatically uses GPU (CUDA/MPS) if available
- **Zero Configuration**: Works immediately after installation

Common embedding models are pre-configured:
- **High Quality**: `mxbai-embed-large-v1` (~335MB, 1024d, supports 512d/256d)
- **State-of-the-Art**: 
  - `stella_en_1.5B_v5` (~1.5GB, 512-8192d, security-pinned)
  - `qwen3-embedding-4b` (~4GB, up to 4096d, 32k context)
- **Small/Fast (Default)**: `e5-small-v2`, `all-MiniLM-L6-v2` (~100MB, 384d)
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
  - Inline approvals and task resume state for agentic workflows
- **Prompt Management**
  - Save, edit, clone prompts
  - Bulk import/export
  - Search and apply templates
  - Version tracking
- **RAG Integration** (enhanced with optional deps)
  - Search Library manually, or set automatic retrieval per conversation
  - Allow or block assistant-initiated Library access independently
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
- **Multiple search providers**: Google, Bing, DuckDuckGo, Brave, Kagi, Tavily, SearX, Serper, Exa, Baidu, Yandex

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

### Voice Conversation in the Console (Hands-Free)
Talk to the Console instead of typing. Three layers, each usable on its own:
- **Dictation**: press the mic button in the Console composer and speak; the transcript lands in your draft.
- **Spoken commands**: "Console, send.", "Console, stop.", "Console, discard.", and more — drive the capture by voice.
- **Hands-free loop** (`Ctrl+Shift+H` or "Console, hands free."): speak, pause, it sends; the reply is spoken back sentence by sentence; the microphone reopens for your next turn. Any key interrupts the reply; `Esc`, the mic button, or `Ctrl+Shift+H` exit.
- **Realtime engine** (optional, off by default): swap the hands-free loop's engine for a live OpenAI Realtime connection — sub-second turns instead of the pipeline's ~4 s pause-to-send. Install the `realtime` pip extra (`pip install -e ".[realtime]"`), then opt in with the `[realtime]` config section or the Settings screen; see [Realtime engine](Docs/Features/Speech-Services-Guide.md#realtime-engine) for the privacy and cost trade-offs before enabling it.

Quickstart:
1. **Microphone + STT**: `pip install -e ".[speech_recording,transcription_parakeet]"` (macOS; use `transcription_faster_whisper` elsewhere). The first-run wizard's Speech step can also set this up.
2. **A voice for replies** (needed for spoken feedback and the hands-free loop): configure any TTS provider under `[app_tts]` — an OpenAI API key is the fastest start; a local [audio.cpp](https://github.com/0xShug0/audio.cpp) server is the recommended local option.
3. Open the Console, press the mic (or `Ctrl+Shift+H`), and talk.

Full walkthrough — including every spoken command, hands-free timing, barge-in modes, and local TTS server setup — in the [Speech Services User Guide](Docs/Features/Speech-Services-Guide.md).

### Text-to-Speech System
Comprehensive TTS support with multiple backends:
- **OpenAI TTS**: High-quality cloud-based synthesis
- **ElevenLabs**: Premium voice synthesis with custom voices
- **Kokoro ONNX** (with `local_tts`): Local neural TTS with no internet required
- **Chatterbox** (with `chatterbox`): Advanced local TTS model
- **Higgs Audio V2** (with `higgs_tts`): Zero-shot voice cloning (see installation section above)
- **AllTalk**: OpenAI-compatible local TTS server
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

For detailed customization, see the [Splash Screen Guide](Docs/Development/SplashScreens/SPLASH_SCREEN_GUIDE.md).

### Coding Assistant
- **Chat-first programming workflows**: Agentic programming and control are designed to happen in Chat, alongside the rest of the conversation context
- **Inline task continuity**: Approvals, progress, failures, and resume cues can be surfaced directly in the chat shell
- **Code mapping**: Analysis and understanding of codebases remain available as supporting capabilities
- **Legacy compatibility**: The standalone coding surface still exists during migration, but it is no longer the long-term primary UX model

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
default_model_id = "e5-small-v2"  # Shipped default; mxbai-embed-large-v1 and others are documented options

# RAG-side embedding overrides can also be set via environment variables:
# RAG_EMBEDDING_MODEL, RAG_DEVICE
```

Example audio transcription configuration:
```toml
[transcription]
# Use NVIDIA Parakeet for low-latency transcription
default_provider = "parakeet"  # Options: faster-whisper, parakeet-onnx, qwen2audio, parakeet, canary, parakeet-mlx, lightning-whisper-mlx, remote-whisper
default_model = "nvidia/parakeet-tdt-1.1b"  # TDT model for streaming
device = "cuda"  # Use GPU for faster processing
use_vad_by_default = true  # Voice Activity Detection
```

Example TTS configuration:
```toml
[app_tts]
default_provider = "openai"  # Options: openai, elevenlabs, kokoro, chatterbox, alltalk
default_voice = "alloy"  # Backend-specific voice ID

# Kokoro ONNX model locations (used when default_provider = "kokoro")
# KOKORO_ONNX_MODEL_PATH_DEFAULT = "models/kokoro-v0_19.onnx"
# KOKORO_ONNX_VOICES_JSON_DEFAULT = "models/voices.json"
```

Example MCP configuration:
```toml
[mcp]
enabled = true
http_port = 3000
allowed_clients = ["claude-desktop", "localhost"]
```

Example splash screen configuration:
```toml
[splash_screen]
enabled = true
duration = 3.0
card_selection = "random"  # Options: random, sequential, or specific card name
active_cards = ["default", "matrix", "minimal"]

[splash_screen.effects]
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
- `DASHSCOPE_API_KEY` (QwenCloud)
- `MOONSHOT_API_KEY` (Moonshot / Kimi)
- `ZAI_API_KEY` (Z.ai / GLM)
- etc.

### QwenCloud
=======
# tldw_chatbook
>>>>>>> 43e9a20e9f (docs: rewrite README for newcomers)

[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)](#alpha-status)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-green)](LICENSE)

tldw_chatbook is a local-first terminal workbench for conversations with large
language models, personal knowledge, and controllable agent-assisted workflows.
It provides a Textual interface for talking to hosted APIs or local model
servers while keeping conversations and other core data on your machine by
default.

The shortest path through the app is:

1. Install the latest source checkout.
2. Let the first-run wizard connect a provider and model.
3. Open **Console** and send a message.

The core install stays reasonably lightweight. Retrieval, media processing,
web access, and protocol integrations are available as optional dependency
groups when you need them.

> New here? Start with [Quick start](#quick-start), then follow
> [Your first conversation](#your-first-conversation). The detailed
> [User Guide](Docs/User_Guide/index.md) is available when you want to explore
> beyond the first message.

## Alpha status

tldw_chatbook is **Alpha** software. The current package version is `0.1.8.0`.
Expect active development, changing interfaces, incomplete documentation in
some advanced areas, and occasional migration or recovery work.

Maturity is not uniform across the application:

- **Available now:** the core Textual shell, Console connections to hosted
  providers or supported local model servers, local conversations and notes,
  Library workflows, Artifacts and Chatbook workflows, Roleplay, Settings, and
  the source-install path documented below.
- **Still evolving:** advanced optional capabilities, ACP/runtime integration,
  some agent and tool workflows, write synchronization, and complete parity
  between local-only use and configured tldw server workflows.
- **Goal:** a modular, local-first workbench where model access, personal
  knowledge, and agent-assisted work can be combined without making every
  integration part of the core install.

“Local-first” describes the default ownership and storage model; it does not
mean every model executes inside this process. Hosted providers send prompts to
their services. Local models are normally reached through a separately running
server such as Ollama, llama.cpp, or another OpenAI-compatible endpoint.

Current package facts:

| Item | Value |
| --- | --- |
| Release | `0.1.8.0` |
| Classifier | Alpha |
| Python | `>=3.11` |
| Textual runtime | `textual==8.2.8` |
| Installed command | `tldw-cli` |
| Entry point | `tldw_chatbook.cli:main_cli_runner` |

<a id="installation"></a>
## Quick start

The primary installation route is a checkout of the latest source.

### 1. Clone the repository

```bash
git clone https://github.com/rmusser01/tldw_chatbook.git
cd tldw_chatbook
```

### 2. Create a Python 3.11+ virtual environment

Unix and macOS:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
```

Use a `python3` executable whose reported version is 3.11 or newer. If needed,
substitute a versioned executable such as `python3.12` in both commands.

Windows (PowerShell or Command Prompt):

```powershell
py -3 --version
py -3 -m venv .venv
```

If that reports a version below 3.11, install a supported Python or substitute
an installed selector such as `py -3.12` in both commands.

Activate in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate in Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install the core application

```bash
python -m pip install -e .
```

### 4. Launch it

```bash
tldw-cli
```

On first launch, use the setup wizard. It is the primary setup path and can
configure either a hosted model provider or a local model server.

If the command is not found, confirm the virtual environment is active and
retry the install with that environment's `python -m pip`.

## Your first conversation

The hosted and local routes differ only in how the model is reached. Both end
in **Console**, using the provider and model chosen in the first-run wizard.
Completing the wizard returns you to **Home**; click **Console** or press
**Ctrl+2**, then send your first message.

### Option A: Connect a hosted model API

1. In the first-run wizard, choose the quick setup track.
2. Select the hosted provider you already use.
3. Enter its API key and choose one of its available models.
4. Finish setup and return to **Home**.
5. Click **Console** or press **Ctrl+2**.
6. Type a message in the composer and send it.

Prompts and responses travel through the selected provider under that
provider's terms. API keys can be stored through the guided configuration or
supplied through supported environment variables.

### Option B: Connect a local model server

1. Start your local server separately, for example Ollama, llama.cpp, or an
   OpenAI-compatible endpoint.
2. In the first-run wizard, choose the quick setup track.
3. Select the local or compatible provider and confirm its endpoint.
4. Choose a model exposed by that server.
5. Finish setup and return to **Home**.
6. Click **Console** or press **Ctrl+2**, then send a message.

tldw_chatbook does not claim an embedded model runtime. The local server owns
model loading and inference; the app provides the conversation interface.

### Change or repair setup

- Run the guided flow again at **Settings › Diagnostics › Run setup wizard**.
- Repair a provider, endpoint, key, or model directly at
  **Settings › Providers & Models**.
- Open the command palette with **Ctrl+P** or screen help with **F1**.

For a step-by-step explanation, see
[First-Run Setup](Docs/User_Guide/First_Run_Setup.md) and the
[Console guide](Docs/User_Guide/console.md).

## Capability overview

The application is organized around named work surfaces rather than a flat
list of equally mature features. These descriptions stay at the outcome level;
the User Guide records current controls and limitations.

| Destination | What it helps you do |
| --- | --- |
| **Home** | See what needs attention, what is running, and a useful next action. |
| **Console** | Hold model conversations, attach context, and supervise supported tools or agent runs. |
| **Library** | Organize and find conversations, notes, prompts, media, and imported source material. |
| **Artifacts** | Collect generated outputs and Chatbooks. |
| **Roleplay** | Work with characters, personas, dictionaries, and lore. |
| **Watchlists** | Monitor configured sources and review their runs or alerts. |
| **Schedules** | Manage when supported recurring work runs. |
| **Workflows** | Define and run reusable procedures. |
| **MCP** | Configure Model Context Protocol servers, tools, and permissions. |
| **ACP** | Work with compatible agent runtimes and sessions as integration support develops. |
| **Lab** | Explore model, speech, and evaluation workflows. |
| **Logs** | Inspect application activity and diagnostics. |
| **Settings** | Configure providers, models, appearance, storage, and application behavior. |

Core conversations and personal content are designed for local storage.
Capabilities that contact model providers, web services, MCP servers, local
model servers, or a configured tldw server cross the corresponding trust
boundary. Review the relevant settings before using them with sensitive data.

## Project direction

The project is moving toward a modular terminal environment in which a
conversation can draw on local knowledge and invoke explicitly controlled
tools without hiding where data goes or who owns an action.

That direction currently emphasizes:

- a newcomer path that reaches a useful first conversation quickly;
- local ownership of core application data and practical offline workflows;
- equal support for hosted APIs and separately operated local model servers;
- optional installation of heavier retrieval, media, web, and integration
  stacks;
- visible approvals and boundaries for agent-assisted actions;
- recovery paths that explain missing configuration or dependencies.

This is a direction, not a claim that all destinations already provide the
same depth, that every tldw server feature has a local equivalent, or that
every local workflow synchronizes back to a server.

## Optional capabilities

Install extras from the repository root, inside the same virtual environment
as the core application. Add only the capability groups you need.

| Optional group | Representative use |
| --- | --- |
| `embeddings_rag` | Embeddings and retrieval-augmented Library workflows |
| `websearch` | Web search and content extraction dependencies |
| `mcp` | Standalone and in-app MCP integration dependencies |
| `web` | Browser-served Textual access |
| `audio` | Audio import and transcription dependencies |
| `video` | Video import and transcription dependencies |
| `pdf` | PDF extraction dependencies |
| `ebook` | E-book extraction dependencies |

Install one group:

```bash
python -m pip install -e ".[embeddings_rag]"
```

Install a useful combination:

```bash
python -m pip install -e ".[audio,video,pdf,ebook]"
```

Install integration and web-search support together:

```bash
python -m pip install -e ".[mcp,websearch]"
```

Optional groups can add large dependencies, native libraries, model downloads,
or external services. A missing extra should disable or explain the advanced
capability it owns; it should not make the core install unusable. See the
[release recovery and setup guide](Docs/Development/release-recovery-setup.md)
for maintained recovery commands and capability ownership.

## Configuration and data

The main configuration file is:

```text
~/.config/tldw_cli/config.toml
```

The first-run wizard and **Settings** are preferred over hand-editing it.
Environment variables are supported for API keys, which is useful for shells,
CI, and secret managers. Provider-specific names and precedence can change, so
use the current Settings UI and maintained documentation instead of copying an
unverified list of variables.

On a typical Unix-like system, the default base directory for local data is:

```text
~/.local/share/tldw_cli/<user>/
```

User- or profile-specific databases, logs, generated files, caches, and other
state live below that base. Exact paths may vary by platform, profile, and
configuration. Before deleting or moving anything, inspect the active paths in
**Settings** and back up the data you care about.

Local storage does not prevent a configured feature from sending selected
content elsewhere. Hosted model calls, web search, MCP tools, compatible agent
runtimes, and server-backed workflows each have their own data boundary.

## Troubleshooting and documentation

Start with these recovery checks:

- **No setup or wrong provider:** open
  **Settings › Diagnostics › Run setup wizard**.
- **Provider, key, endpoint, or model error:** open
  **Settings › Providers & Models** and save a valid combination.
- **`tldw-cli` is missing:** activate the virtual environment and run
  `python -m pip install -e .` again.
- **An advanced feature is unavailable:** install the optional group named by
  the recovery message, then restart the app.
- **A local model does not respond:** verify its separate server is running,
  its endpoint is reachable, and the configured model name is exposed there.
- **Need runtime detail:** open **Logs**, then consult the relevant guide
  before sharing diagnostic output that may contain private paths or content.

Maintained starting points:

- [User Guide](Docs/User_Guide/index.md) — navigation and task-oriented guides
- [First-Run Setup](Docs/User_Guide/First_Run_Setup.md) — wizard tracks and
  setup recovery
- [Console](Docs/User_Guide/console.md) — conversations, context, and live work
- [Release recovery and setup](Docs/Development/release-recovery-setup.md) —
  optional dependency and blocked-state recovery
- [Changelog](CHANGELOG.md) — release history

Because the project is Alpha, documentation may track the development branch
more closely than an older checkout. When labels differ, compare your package
version with the guide's verification note.

## Contributing, license, and contact

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request, keep changes focused, and include tests or verification
appropriate to the behavior being changed. Use the repository's issue tracker
for reproducible bugs, feature discussion, and documentation gaps.

tldw_chatbook is licensed under the
[GNU Affero General Public License v3.0 or later](LICENSE). If you discover a
security issue, avoid publishing sensitive details in a public issue; contact
the maintainer privately at [contact@rmusser.net](mailto:contact@rmusser.net).

Project links:

- Repository: [github.com/rmusser01/tldw_chatbook](https://github.com/rmusser01/tldw_chatbook)
- Issues: [github.com/rmusser01/tldw_chatbook/issues](https://github.com/rmusser01/tldw_chatbook/issues)
- License: [LICENSE](LICENSE)
