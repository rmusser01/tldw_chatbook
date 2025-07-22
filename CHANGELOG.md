# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
    
## [Unreleased]

### Added
- Initial features pending documentation

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