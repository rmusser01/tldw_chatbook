# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Some kind of Versioning
    
## [Unreleased (placeholder for copy/paste)]

### Added
- Initial features pending documentation
- Media chunk agent tools (chunking-agent-tools): Console agents and local MCP
  clients get five new `library_*` tools over ingested media's stored chunks —
  the program's student story ("per-chapter notes from an ingested book")
  delivered without blind character-window walking. `library_get_media_structure`
  returns a media item's heading/section tree annotated per node with the
  stored-chunk index span overlapping it, plus an item chunk summary
  (count, families, engine versions, template, stale flag) and the media
  version as a revision token; pagination is by nodes (default 200, max 500)
  through a `node_cursor`, never byte-sliced. `library_get_media_chunk`
  fetches one unit by `chunk_index` **from the stored
  `UnvectorizedMediaChunks` rows verbatim** — the reuse-stored-chunks read
  path; nothing re-chunks implicitly (mutation-tested). Multi-family
  (hierarchical) items name their `chunk_type` families and refuse ambiguous
  addresses with the round-trippable list; neighbors arrive under `context`
  (0–10) inside the 32 KiB result budget with a dropped-neighbor note; a
  stale revision token is the named `content_changed` error. Items ingested
  with chunking off keep the heading tree with an availability hint naming
  `library_rechunk_media`. `library_list_chunk_specs` /
  `library_save_chunk_spec` expose the v7 chunking-template store to agents
  (specs ARE templates): a bounded listing with validity/reserved flags, and
  create-or-update of custom templates through the validated CRUD with the
  validator's full errors array on refusal (built-ins refused with a
  duplicate hint; the reserved `auto` name refused case-insensitively).
  `library_rechunk_media` (opt-in write) re-chunks ONE item synchronously
  through the same per-item machinery as the Library "Re-chunk older-engine
  items" action — flat spec override (`{"template": name}` XOR plain
  `method`/`max_size`/`overlap`; an omitted overlap is 0, not the engine's
  100 default; omitting `spec` re-runs the item's stored config while
  `spec: {}` is an explicit plain override; an unresolvable template is a
  named refusal, never a silent fallback), atomic chunk-row replacement, and
  a separate `reindex: true` opt-in for the forced vector re-index
  (default off; outcome vocabulary `reindexed`/`skipped`/`failed`, never a
  bare "done"; a skipped re-chunk carries no `reindexed` key). The two
  writing tools are policy-gated under new runtime-policy resources —
  `library.templates.save` and `library.media.rechunk` — with denials
  firing before any backend call. Console and MCP advertise identical
  schemas from the one descriptor table (23 Library tools total); the
  student story is pinned end to end by
  `Tests/Library/test_agent_chunk_student_story.py`.
- Note-save agent tool (student-workflow): `library_save_note` closes the
  student story's write loop — the 24th `library_*` tool lets Console agents
  and local MCP clients land their per-chapter study notes where the user
  already reads them (the notes screen). Create by default; update by
  `note_id` + `expected_version` together (exactly one without the other is
  refused; a stale version is the named `content_changed` error). Bounds are
  schema-level (title ≤ 512, content ≤ 100_000, folder ≤ 256). The optional
  one-level folder is created when missing in the notes UI's own local
  scope, concurrent savers converge on one folder, and a folder failure
  never lands an orphaned note. Notes derived from Library media carry the
  documented provenance-header convention (`source`/`revision`/`chapter`/
  `chunks` — revision is load-bearing for staleness). The re-run convention
  is search-based (`library_search_notes` by title, then update by id) since
  the list tool has no folder filter; flashcards are Q/A markdown inside
  notes (the real flashcards rows have no screen route yet). Runs under the
  new `library.notes.save` runtime-policy resource, denied before any
  backend call on both Console and MCP surfaces. The fan-out pattern
  (structure → spawn-per-chapter → fetch → save → re-run, riding the
  existing `spawn_subagent`) is documented in the Console guide, and the
  story test now proves the whole loop — read from stored chunks,
  provenance-headered save, re-read, search-based re-run update without
  duplicates, Q/A flashcard note — end to end against real databases
  (`Tests/Library/test_agent_chunk_student_story.py`).
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
- Chunking-template picker on Library ▸ Import media (chunking-template-parity): a
  "Chunking template" select listing the DB's templates (default "None (manual
  settings)"), hidden in server mode, disabled with a why-label until "Chunk content"
  is on, populated from the media DB via the RAG admin scope service off the mount
  path. The chosen template governs that item's parse and is persisted on the Media
  row (`chunking_config`) and its chunk rows. A new `[chunking]` config section
  (`default_template`) names the fallback template for ingests that did not pick
  one; template resolution order is the picker's stored choice → config default →
  plain options, and an unresolvable template name fails with a named error
  instead of silently chunking with defaults.
- Chunking Auto-selection (chunking-auto-selection): the "Chunking template" picker
  gains an "Auto" option (None stays the default; the name `auto` is reserved —
  template create/rename refuse it). Auto decides per item in three always-terminating
  tiers: a template whose `classifier` block matches the item's media type /
  filename / title / URL and clears its `min_score` wins and runs in full
  (indistinguishable from a manual pick); otherwise the vendored server auto
  planner (`Chunking/engine/auto_planner.py`, moved excluded → vendored) derives
  media-type-aware chunk options (goal hardcoded "balanced", LLM features off);
  otherwise today's plain defaults — Auto never fails an import. Templates
  without a classifier block are never auto-selected (the six built-ins included),
  so nothing changes until a template opts in. The decision is persisted per item
  (`mode: "auto"` + tier/rationale in `Media.chunking_config`; the `template` key
  only on a template-tier win, keeping both existing readers — the usage LIKE and
  the statistics `json_extract` — truthful) and re-resolved, not replayed, on
  re-chunk: changing the template store changes the re-chunk outcome, and the
  stored record is re-stamped to match what the re-chunk actually used. The
  `[chunking] default_template` config key never triggers Auto (picker-only,
  stripped from server-mode ingest). Planner decisions are frozen by byte-pinned
  parity fixtures and a media-type vocabulary test; classifier scoring,
  tie-breaks, and the built-ins-never-auto-selected pin are standing tests.
- Library ▸ Search / RAG legacy-chunk report + re-chunk action
  (chunking-template-parity): the panel shows "Chunked by an older engine: N items"
  when pre-parity chunks exist and offers "Re-chunk older-engine items", which
  re-chunks exactly those items through the template-aware path (honoring each
  item's stored template choice), replaces their chunk rows in one transaction,
  force-reindexes them into the semantic index (deleting the stale vector document
  by deterministic id first), clears the owning service's query cache, and surfaces
  a per-run summary ("N re-chunked, M skipped, K failed" plus notes — never a bare
  "done"). A re-chunk and a RAG index backfill refuse to overlap via a shared
  in-flight guard (a notice, never worker cancellation), and an interrupted
  re-index leaves the item re-indexable on the next backfill rather than
  permanently absent from search.

### Removed
- **BREAKING — file template store deleted (chunking-template-parity, spec §8.1).**
  `tldw_chatbook/Chunking/templates/` (the 13 built-in template JSONs plus
  `README.md` and `example_usage.py`) and `Chunking/chunking_templates.py`
  (`ChunkingTemplateManager`, `ChunkingPipeline`, `ChunkingTemplate`,
  `ChunkingStage`, `ChunkingOperation`) are gone, and those names are no longer
  exported from the `tldw_chatbook.Chunking` package root — a breaking change
  to the package's public import surface. The vendored engine's own
  `ChunkingTemplate` (same public name, different class) is deliberately NOT
  re-exported either: nothing outside the service layer resolves templates.
  Chunking templates are DB rows now. `Chunker`/`improved_chunking_process`
  keep their `template=`/`template_manager=` parameters, but `template=`
  accepts only a pre-resolved template dict (resolve names first via
  `tldw_chatbook.Chunking.template_runtime.resolve_template`); a bare name
  string raises `TemplateError`, and `template_manager=` is accepted and
  ignored. All five packaging sites (pyproject package-data and package
  exclude, MANIFEST.in, `Packaging/check_manifest.py`, and the installed-
  distribution import pin and data contract) moved in the same commit: no
  `Chunking/templates/` path ships in the wheel or sdist. The legacy `Chunker`
  adapter no longer sets its `.template` / `.pipeline` instance attributes
  (they held the deleted `ChunkingTemplate` / `ChunkingPipeline` objects);
  `.template_manager` remains stored for attribute compatibility, never
  consulted.
- Rejected, unreachable dictation-history implementations removed:
  `Audio/transcription_history.py`, `Widgets/transcription_history_viewer.py`,
  and `UI/Dictation_Window.py`.
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
- **Chunking engine swapped for the server's (chunking-engine-parity).** All
  chunking — RAG ingestion, media import, summarization, the works — now runs
  through the same engine tldw_server uses (vendored at dev@385afa95 behind a
  compatibility shim; chunks are stamped `parity-1@385afa95`, media DB schema
  v6). User-visible consequences:
  - **More chunking methods everywhere.** The RAG-service entry point
    previously accepted five methods (words, sentences, paragraphs, tokens,
    semantic); every method the engine implements now works on ingestion
    paths — words, sentences, paragraphs, tokens, semantic, json, xml,
    ebook_chapters, rolling_summarize, fixed_size, code, code_ast,
    structure_aware. `ebook_chapters` in particular now applies to PDFs and
    other documents, not just e-book files.
  - **Imported text is sanitized before chunking (behaviour change).** Null
    bytes and unusual control characters are replaced with spaces, Unicode is
    normalized to NFC where that doesn't shift character positions, and
    bidirectional text override characters are neutralized — so the chunked,
    searchable text can differ slightly from the raw file you imported.
  - **`tiktoken` and `defusedxml` are core dependencies.** The `tokens`
    method now always has a real tokenizer (previously it silently
    approximated token counts by word count when tiktoken was missing — it
    now raises a clear "install tiktoken" error instead), and the `xml`
    method parses with defusedxml.
  - **Overlap handling changed at the edges.** For words/sentences/paragraphs,
    an overlap at or above the chunk size no longer raises an error — the
    engine clamps it just under the size and produces more, smaller chunks.
    The `tokens` method keeps the old strict behavior (it errors when overlap
    ≥ size).
  - **Chunks carry richer metadata.** Each chunk records the engine version
    that produced it plus offsets, word counts, and the method used, and RAG
    Admin diagnostics gained a read-only legacy-chunk report counting chunks
    persisted before the version stamp ("Chunked by an older engine: N
    items").
- **Behaviour change — rolling-summarize fails closed
  (chunking-template-parity).** When the `rolling_summarize` method's per-part
  LLM callback raises, returns the legacy `"Error: …"` failure string, or
  returns a non-string, chunking now raises `ChunkingError` (original cause
  chained) instead of embedding `[Summarization failed for this part: …]`
  marker text into the chunk stream. The markers were silently persisted as
  if they were content — corruption that search, citations, and exports would
  then faithfully reproduce — and nothing depends on them; a failed
  summarization now stops at the failure instead of shipping marker rows.
- **ChunkingTemplates storage rebuilt on media-DB schema v7 — a one-way door
  (chunking-template-parity).** The chunking-template table is now the
  server's shape: rows carry uuid/version and a soft-delete flag, the six
  server built-ins are seeded as the only built-ins, the old chatbook seeds
  are converted to the flat dict contract (three retired seeds survive as
  editable non-builtin rows), unconvertible rows are quarantined under
  "<name> (needs review)" with their original body preserved, and every
  create/update validates the template body (stored-invalid rows stay listed
  with a flag and editable, but are refused at apply with a named error).
  **Once a media DB has been migrated to v7, downgrade is impossible** — v7
  voids ADR-073's revert net for the chunking store. Back up the media DB
  before updating if you may need to return to an older build.
- **Chunking templates govern local ingestion (chunking-template-parity).**
  A resolved template's chunk-stage options now take precedence over the
  ingest form's unchanged defaults on every local ingest seam; only
  explicitly changed form values beat the template (unchanged fields equal
  to the schema default defer to it). Re-chunking honors each item's stored
  template choice with the same resolution order.
- **Behaviour change — reranking now really calls the provider, and really spends
  (TASK-17065).** Reranking has silently no-opped since the feature existed: the
  reranker resolved credentials from a `settings["API"]` table `load_settings()`
  never builds, and dispatched `chat_api_call` with a positional argument list that
  did not match its signature, so it could complete a scoring call for **none** of
  the 29 chat providers the picker offers. It now calls `chat_api_call` the way every
  other caller in this app does — by keyword, resolving no credential of its own and
  letting each provider handler apply the documented precedence. **A profile that
  carries a reranker config therefore begins issuing real provider calls on its next
  search, where it previously failed and skipped.** Nothing else has to be turned on
  for the spend to start.

  *What one search costs* (measured against the real reranker with the provider seam
  faked, not estimated): **pointwise** — the strategy the "Enable reranking" toggle
  creates — issues one call per candidate up to the configured "Rerank results", **and
  retries every failure twice** (`max_retries = 2`, on *any* exception, a bad
  credential included), so the honest ceiling is `top-k × 3`: 3 candidates against a
  failing provider issued 9 calls, 20 issued 60. **listwise** issues exactly **1** call
  per search (10 documents max) — but the same retry rule applies, so a failing
  provider costs **3**. **pairwise** is a merge sort, so it issues `≈ n·log₂n`
  *comparisons*, not `n` scorings — **40–69** calls at top-k 20, ~200 with retries; no
  built-in preset uses it.

  *Who is spending* — the tick is not the only door. Reranking is on for any profile
  that carries a reranker config, and three read-only built-ins ship with one:
  **Hybrid Full** (pointwise, 15), **High Accuracy** (pointwise, 15) and **Research
  Papers** (listwise, 10), all billing `openai` by default. Making one of them active
  is a spend decision with no checkbox in it. The out-of-the-box active profile,
  Hybrid Basic, carries none — so a fresh install still spends nothing until you pick
  or configure otherwise. Untick "Enable reranking" on your own profile to opt out;
  clone a built-in and untick it there.

  This lands on top of TASK-3502's disclosure surfaces, which are what make the spend
  visible: the cost line under the Reranking toggle in Settings ▸ RAG ("Reranking
  scores each result with a separate `<provider>` call — up to `<n>` calls per search,
  or `<n×3>` if calls fail and are retried, billed at that provider's rates"), and the
  skipped/degraded sentence on the Library results screen. Three further consequences
  worth naming: the old broken call put the token cap into the `streaming` slot, so any
  scoring call that *had* resolved a credential would have STREAMED — scoring calls now
  pass `streaming=False` explicitly rather than inheriting a handler default; the
  reranker's configured `max_tokens` (default 100) and `temperature` (default 0.0) now
  reach providers for the first time; and `High Accuracy` and `Research Papers` no
  longer ask for a free-form `reasoning` field under that 100-token cap — nothing
  outside the reranker ever read it, and truncating it turned a billed call into an
  unscored row (listwise: a wholly failed rerank).
- Unsupported direct imports of `get_user_database_path`, `USER_DB_DIR`, and
  `USER_DB_PATH` have been removed.
- Textual 8.x is now required (`>=8.0.0,<9`). This corrects the previously
  overstated Textual 3.3 compatibility range, which could crash when opening
  MCP because the screen uses Textual 8's `Select.NULL` API. Existing source
  checkouts should reinstall dependencies after pulling this update.
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
