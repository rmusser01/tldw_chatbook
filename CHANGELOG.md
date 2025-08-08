# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Some kind of Versioning
    
## [Unreleased (placeholder for copy/paste)]

### Added
- Initial features pending documentation

## [0.1.7.1] - 2025-08-7
### Fixed 
- Chatbook import logging


## [0.1.7.0] - 2025-08-7
### Added 
- Chat swiping/forking + multiple responses


## [0.1.6.5] - 2025-08-5
### Worked on 
- Evals+Embeddings+Chatbook UIs


## [0.1.6.4] - 2025-08-5
### Mutilated
- Evals module.


## [0.1.6.3] - 2025-08-4
### Added
- Cancellation button for transcription
- Fixes suggested by gemini for Packaging
- Warning dialogs for delete buttons


## [0.1.6.2] - 2025-08-3
### Added
- Textual Serve instructions added to readme


## [0.1.6.1] - 2025-08-3
### Added
- Splashscreen modularization
- Textual-serve - we a webapp now (port 9000, tldw-cli --serve


## [0.1.6.0] - 2025-08-2
### Fixed
- Analysis sub-tab UI + saving/reviewing existing analyses
- Some tests
- Stuff

### Added
- Subscriptions (broken)
- Chatbooks (broken)
- Coding Tab (broken)
- New Embeddings creation workflow (broken)
- Wizard walkthrough widget (broken)
- Extensive mindmap viewer/converter (broken)


## [0.1.5.0] - 2025-07-27
### Fixed
- Stuff

### Added
- Other stuff
- Other Stuff:
  - Theme editor
  - Analysis
  - Study tab
  - Model download via huggingface interface
  - Model view + delete of models downloaded via HF
  - Logits + Logprobs in evals


## [0.1.4.1] - 2025-07-27
### Fixed
- Media Views
- CSS Adjustments
- faster-whisper ingestion


## [0.1.4.0] - 2025-07-24
### Added
- Higgs tts
- clone chat button

### Fixed
- model checkpoints added to gitignore


## [0.1.3.7] - 2025-07-21
### Added
- vibe-coded speaker diarization implementation (Un-tested, need to verify/wire up)
- Audiobook UI that doesn't work
- Improvements to RAG search and evals. Both still don't work.

### Fixed
- RAGSearchWindow.py - endless spiral
- ?
- Audiobook gen is not fixed
- Improved WebSearch API and web scraping libraries. 


## [0.1.3.6] - 2025-07-21
### Fixed
- Ingest Window Transcription model
- Search Window
- Refactor MediaDB version handling
- Refactor encryption of config file + added setting in settings


## [0.1.3.5] - 2025-07-21
### Fixed
- Chatterbox TTS generation
- 'Continue' button
- Datetime import in the chat window


## [0.1.3.4] - 2025-07-20
### Added
- New chat UI Screenshot + Custom Chunkning/RAG enhancements


## [0.1.3.3] - 2025-07-20
### Fixed
- TTS bugfixes (again)
- Fix for background processes not being terminated properly (again)

### Added
- Groundwork for custom chunking


## [0.1.3.2] - 2025-07-20

### Fixed
- TTS bugfixes
- Fix for background processes not being terminated properly


## [0.1.3.1] - 2025-07-20

### Fixed
- Numpy Bugfix


## [0.1.3.0] - 2025-07-20

### Added & Fixed
- TTS Bugfixes
- Groundwork for future features.


## [0.1.2.0] - 2025- 07-18

### Added
- Added more TTS stuff.


## [0.1.1.1] - 2025-07-17

### Added
- Fix for numpy deps in base package


## [0.1.1] - 2025-07-17

### Added
- Fix for numpy deps in base package
- Addition of Splash screen play length in General Settings Window
- 

## [0.1.0] - 2025-07-16

### Added
- Initial release of tldw_chatbook
- Terminal User Interface (TUI) built with Textual framework
- Support for multiple LLM providers (OpenAI, Anthropic, Google, Cohere, etc.)
- Local LLM support (Ollama, llama.cpp, vLLM, MLX)
- Chat interface with streaming responses
- Character/persona chat system
- Notes management with bidirectional file sync
- Media ingestion and processing
- RAG (Retrieval-Augmented Generation) capabilities
- Conversation history and management
- Customizable prompt templates
- Search functionality across conversations and media
- Configuration via TOML files
- Comprehensive keyboard shortcuts
- Multiple themes support

### Security
- Input validation and sanitization
- Path traversal prevention
- SQL injection protection
- Secure temporary file handling

[Unreleased]: https://github.com/rmusser01/tldw_chatbook/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rmusser01/tldw_chatbook/releases/tag/v0.1.0