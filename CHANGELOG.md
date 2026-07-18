# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Some kind of Versioning
    
## [Unreleased (placeholder for copy/paste)]

### Added
- Initial features pending documentation
- UX efficiency cycle (critique follow-up, ADR-016): the Console composer is now a real
  editable text field with a movable caret (arrows, Home/End, Ctrl+W, mid-draft
  insertion, Shift+Enter newline); destination hotkeys ctrl+1..9,0 jump to the first ten
  shell destinations with matching index labels in the nav; the nav scrolls the active
  destination into view and docks the "More: Ctrl+P" hint outside the scroll area; Console
  readiness chips are keyboard-focusable (focus reveals the full ellipsized label) and the
  Approvals chip plus inspector "Review approval" button now focus the pending approval
  card; F1 shows truthful BINDINGS-generated help on every screen; Lab gains a
  Models | Speech | Evals mode strip, and the "lab" route id resolves correctly, making
  the inline Evals workbench reachable for the first time.
- Lab destination in the shell nav (ADR-015): Models (`llm`), Speech (`stts`), and Evals
  now have a home in the 12-destination rail between ACP and Settings. (Rebase note:
  upstream's retirement of the Skills destination into Library is adopted.)
- Destination identity headers (`DestinationHeader`: title, plain-language subtitle,
  text-labeled status badge) on the Console (now visible), Search, Media, Study, Writing,
  Research, Models, Speech, Logs, Stats, Evals, and Personas screens. Stats also gained the
  standard nav/footer/status chrome and a live Loading/Error/Ready/Empty header badge.

### Removed
- Legacy navigation chrome retired (ADR-014, as amended on rebase): the permanently
  occluded `TitleBar` and the `TabBar`/`TabLinks`/`TabDropdown` legacy nav widgets are
  deleted, along with the dead `general.use_dropdown_navigation` /
  `general.use_link_navigation` config switches. Users who set those options lose nothing
  visible — they only selected which of three invisible nav widgets was mounted.
  (`AppFooterStatus` is NOT deleted: upstream's per-screen mounting in task-264 fixes the
  same occlusion, so the widget stays and the earlier `AppStatusLine` replacement was
  dropped in its favor.)
- Standalone Coding screen retired and merged into Console (ADR-015): the `coding` route,
  `CodingScreen`, and `Coding_Window.py` are gone; legacy `coding` links land on Console.
  `CodeRepoCopyPasteWindow` is unaffected.

### Changed
- Command palette dedupe (ADR-015): one navigation command per destination; legacy route
  names (media, search, study, writing, research, logs, stats, llm, stts, evals, coding,
  ccp, tools_settings, ingest, notes, chatbooks, subscriptions, customize, ...) are
  searchable aliases that land on the owning destination instead of separate labeled
  commands. This removes the duplicated "Personas" and "MCP" palette entries.
- Route folds (ADR-015): Writing and Research now resolve under Library, Logs and Stats
  under Settings, and Models/Speech/Evals under Lab, so the nav boxes the right destination
  on every screen.
- Evals dead-end removed: the destination no longer pushes a separate hub screen on mount
  and no longer shows a permanent "Loading Evaluation Lab..." placeholder. The evaluation
  workbench renders inline under the shell chrome, its cards navigate again, and Escape
  walks the workbench back stack instead of dead-ending. The hub's redundant emoji
  marketing header was dropped in favor of the destination identity header.
- Small-terminal workbench fix: workbench minimum heights no longer exceed the available
  space at ~24-row terminal sizes, so list rows no longer render underneath the status
  line where clicks were intercepted.
- Console top area: control-bar chips carry full-label tooltips so two models sharing a
  name prefix stay distinguishable when ellipsized. (Rebase note: the rail Model readouts
  and transcript copy blocks stay — upstream expanded the Console internals that this
  branch's dedup experiment had pruned, and upstream's version wins.)
- Console session tab strip now scales: the strip scrolls horizontally instead of
  silently clipping past a handful of tabs, the active tab is scrolled into view on
  switch, tabs show a run-in-progress glyph for the session that owns the active
  stream, and middle-click on a tab closes it (the ✕ button stays as the visible,
  keyboard-reachable close path).
- Product naming aligned: the terminal title and first-run welcome now say
  "tldw chatbook" instead of the legacy "tldw CLI" identifier. The Console transcript
  header uses the shell's " | " separator instead of an em dash, and the nav overflow
  hint gained breathing room after the last destination button.

### Fixed
- Search/RAG no longer crashes with NoMatches when the screen is closed while its
  collections loader thread is in flight; the DOM update is now guarded during teardown.
- Study screen no longer races dashboard mounting when applying a pending initial
  section; the sync now retries after refresh instead of raising NoMatches in a worker.
- Seventeen pre-existing test failures repaired: stale config-path monkeypatching in the
  tools/settings API-key tests, outdated Library copy and nav expectations in the shell
  contract tests (including the new Lab destination), a retired notes-mode-chip CSS
  contract, a stale packaging entry-point expectation, a glyph-marker assertion, and a
  ChatScreen test-double missing a new dispatch stub. The phase-5 worker contract now
  detects asyncio.run call sites via AST instead of substring matching.


## [0.1.8.0] - 2026-07-08
### Changed
- Master-shell UI/navigation overhaul: the app is organized around primary destinations
  (Home, Console, Library, Artifacts, Personas, Watchlists, Schedules, Workflows, MCP, ACP,
  Skills, Settings) instead of a flat tab bar. Legacy tabs remain reachable as routes/aliases.
### Added
- Console dual-audience redesign: first-run setup card, keyboard layer (command palette,
  Ctrl+K session switcher, Alt+M model popover, direct message copy/edit/regenerate), and a
  collapsible Session / Context / Model / Details rail with auto-titled, recent-first conversations.
- Home triage surface: Needs Attention / Running / Recent rail + focus canvas with per-item actions.
- Library local-content hub around (re)view · search · ingest · create, with an in-Library media
  viewer; Notes absorbed into Library.
- Personas Console-parity workbench (avatar upload, markdown / character-card import).
- Notes Sync v2 (P1/P2) conformance work.
### Removed
- Standalone Notes tab retired (absorbed into Library); legacy entrypoints retired.


## [0.1.7.3] - 2025-08-7
### Fixed 
- Replaced top tab bar with link bar instead


## [0.1.7.2] - 2025-08-7
### Fixed 
- Numpy requirement in base install


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