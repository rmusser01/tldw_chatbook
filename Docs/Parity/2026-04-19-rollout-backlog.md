# tldw_chatbook Rollout Backlog

## Purpose

Translate the scored capability matrix into concrete implementation waves for `tldw_chatbook`.

## Excluded Domains

- Billing
- Orgs / multi-tenant management
- Multi-user admin and control-plane features
- Platform messaging integrations unless a local workflow depends on them

## Phase 0: Audit And Stabilization

- Finish the capability matrix
- Finish the data compatibility map
- Capture dirty-tree overlap risks in `tldw_chatbook`
- Identify duplicate or legacy UI surfaces
- Use current dirty-tree overlap as an explicit scoring input, not an afterthought

## Phase 1: Core Interoperability Primitives

- Define canonical local/server mapping for conversations, messages, notes, characters, prompts, and chatbooks
- Normalize server connection and auth expectations for future interoperability work
- Identify ID, timestamp, metadata, and deletion-semantic mismatches that would block later sync
- Choose the first code vertical from chat/conversations, prompts/chatbooks, notes/workspaces, or characters

## Phase 2: Highest-Impact Feature Parity

- Chat and conversation workflow alignment
- Character chat session/message alignment
- Notes and workspace-adjacent workflow alignment
- Prompt library and chatbook interoperability improvements

## Completed Verticals

Four parity verticals are now landed in the isolated `codex-prompts-chatbooks-parity` worktree.

### Prompts And Chatbooks

- `tldw_api` now exposes prompt preview/create/version-restore methods and chatbook export/preview/import job methods.
- Local prompt storage now preserves server-shaped structured prompt metadata through `prompt_format`, `prompt_schema_version`, and `prompt_definition`.
- Prompt adapter helpers can round-trip between local prompt records and server request/response payloads without breaking legacy prompts.
- `ChatbooksWindowImproved` is now the primary chatbooks screen surface.
- Server-backed chatbook export/import flows are wired into the creation and import wizards.
- The chatbook management window now shows recent server jobs, which partially closes the Hermes-style job visibility gap for this vertical.
- Server-backed chatbook import is still intentionally limited to conversations, notes, and characters. Prompt, media, and embedding content still fall back to the local import path.

### Notes And Workspace Alignment

- `tldw_api` now exposes notes and workspaces CRUD surfaces needed for local/server compatibility work.
- Local notes services and screen state are scope-aware, which keeps user-space notes and workspace-contained notes separate.
- Workspace CRUD, workspace-scoped note controls, and local-only notes sidebar behavior are aligned with the standalone/offline-first model.
- Follow-on fixes in this branch tightened delete handling, notes dirty-state behavior, and keyword preservation so workspace-contained notes do not leak into the general notes surface.

### Chat And Conversations

- `tldw_api` now exposes chat conversation client contracts for list/get/create/update/delete, tree, and message operations.
- Local chat DB and service layers now preserve conversation metadata, scope fields, assistant/persona context, and server-shaped message payloads through one normalized seam.
- Session state, tab restoration, and existing chat UI plumbing now retain the same conversation contract end-to-end while keeping the visible chat UI unchanged.
- Local message persistence now preserves message variants and metadata during resave flows instead of collapsing them into a lossy linear history.

This closes the current chat/conversations parity vertical and keeps the next wave focused on the remaining entity seams instead of reopening already-landed work.

### Characters, Personas, And Runtime Alignment

- `tldw_api` plus the new mode-aware character/persona services now cover server characters, persona profiles, exemplars, chat greetings, and chat presets without breaking local-first standalone behavior.
- CCP now has separate character and persona management flows, persona-aware history browsing, and backend-aware launch into main chat using explicit runtime/discovery metadata instead of legacy heuristics.
- General chat history excludes CCP-owned character/persona conversations through DB/service filtering rather than UI-only filtering.
- Greeting selection and preset CRUD are wired as chat-scoped server execution helpers through the service and handler seam. The broader chat UI for managing those controls is still intentionally deferred.
- Local persona-profile persistence, sync/dual-write policy, and cross-scope movement remain follow-on work.

### Media, Files, And Ingestion Alignment

- `tldw_api`, the new `Media/` seam, and the UI runtime state now normalize media records onto canonical IDs of the form `<backend>:<entity_kind>:<source_id>`.
- Reading-progress operations now depend on raw `backing_media_id`, which keeps server reading-item IDs out of progress endpoints and keeps local reading progress separate from sync/version semantics.
- `MediaWindow_v2` now uses the shared scope service for browse/detail/progress/document-version operations, and the media list no longer leaks canonical IDs into widget IDs.
- `MediaIngestWindowRebuilt` now uses server ingestion-source management for server mode instead of constructing a direct `TLDWAPIClient` processing path in the UI.
- Unsupported server-mode document-version actions now fail explicitly instead of silently falling back to local-only behavior.

### Retrieval Admin Alignment

- `tldw_api` now exposes chunking-template CRUD/apply/diagnostics contracts plus embedding-collection list/detail/delete contracts needed for admin parity with `tldw_server`.
- The new `RAG_Admin` seam routes chunking-template and embedding-collection actions through the active local/server backend and normalizes both onto one UI-facing contract.
- The chunking templates widget and editor now use that seam, preserve server-compatible template JSON, and keep builtin-template protections intact.
- `EmbeddingsManagementWindow` now performs real collection list/detail/delete operations through the same seam instead of placeholder local-only logic.
- This vertical intentionally stops at retrieval-admin parity. Broader search/chat-documents parity, collection export, embeddings creation alignment, and sync/dual-write behavior remain follow-on work.

### Evaluations Foundation Alignment

- `tldw_api` now exposes evaluation dataset/evaluation/run schemas and client methods aligned with the current `tldw_server` unified evaluation routes.
- The new `Evaluations_Interop` seam normalizes local tasks/datasets/runs and server evaluations/datasets/runs onto one browser-facing contract while keeping local storage authoritative by default.
- Local eval tasks now preserve server-style metadata in a reserved compat payload instead of silently dropping it, and local run records are enriched with flattened metric summaries before normalization.
- Local saved models now normalize into shared `evaluation_target` records, local run creation resolves those targets back to local model IDs, and server run creation routes explicit `target_model` strings through the same scope seam.
- The eval navigation now uses the shared browser in two modes: `Tasks` is the manage surface for definitions/runs plus safe run launch, and `Results` is the non-launching results/detail surface built on the same normalized contracts.
- Local run detail now exposes flattened metrics and sample-level results through the seam, while server runs remain intentionally summary-only because the unified server API does not yet provide a sample-results endpoint.
- Evaluation execution parity is now in place at the seam/browser level. Remaining follow-on work is intentionally narrow: server target discovery/catalog UX, richer server-side run detail if/when the API supports it, and all study-resource mapping. Quick Test remains local-first.

### Study Flashcards And Review Alignment

- `tldw_api` now exposes flashcard deck/card/review schemas and client methods aligned with the current `tldw_server` flashcards routes.
- The new `Study_Interop` seam normalizes local decks/cards and server decks/cards/review candidates onto one backend-aware contract while keeping local storage authoritative by default.
- `StudyScreen` and `StudyWindow` now construct the flashcards surface through a screen-local controller instead of the legacy app-level handler path, which removes the hidden default deck behavior and stabilizes initialization.
- Local DB study parity now includes explicit read helpers for deck/card listing and deck lookup, normalized blank-search behavior, version-threaded local delete/move/deck-delete behavior, and deleted-deck/card filtering instead of inheriting backend-specific quirks.
- The shared flashcards client plus `Study_Interop` now route server card delete/move operations through the current server contract, while server deck delete remains explicitly unsupported and is exposed as a disabled control with explanatory copy in the Study UI.
- The flashcards pane now supports backend-aware deck creation, card creation, due/review flow, lifecycle controls, explicit empty/error states, review-session termination when the queue ends or the user leaves the flashcards view, and review invalidation when the active review card or selected deck is removed.
- The screen-local flashcards controller now owns normalized deck/card caches, selected-record tracking, and lifecycle action wiring so move/delete actions use stored records and optimistic-lock versions rather than widget-label parsing.
- This vertical now covers flashcard lifecycle parity through the Study TUI. Remaining study follow-on work is narrower: study import/export mapping, remediation conversion, any future server deck-delete contract, and later cross-scope move/sync/dual-write policy.

### Study Quizzes And Attempts Alignment

- `tldw_api` now exposes quiz, quiz-question, and quiz-attempt schemas and client methods aligned with the current `tldw_server` quiz routes used for compat.
- Local `ChaChaNotes_DB` study parity now includes quiz/question/attempt tables plus local-first CRUD, question authoring, attempt snapshotting, and grading helpers.
- The quiz side of `Study_Interop` now normalizes local quiz records and server quiz records onto one backend-aware contract without changing the standalone/local-first storage model.
- `StudyScreen` and `StudyWindow` now expose a screen-local quizzes controller that mirrors the flashcards pattern, with explicit empty states, selected-quiz management, and a simple attempt flow that works in both local and server mode.
- The follow-on lifecycle/history slice now adds quiz/question delete routing through the compat seam plus minimal attempt-history browsing/loading in the Study TUI for both local and server mode.
- Remaining study follow-on work is now narrower: quiz remediation conversion, study import/export mapping, and later cross-scope move/sync/dual-write policy.

### Study Workspace Scope And Shell Alignment

- Notes workspace details can now launch Study in an explicit workspace scope, and the Study shell exposes `Back to Workspace` plus `Switch To Global Study` affordances instead of silently mixing scopes.
- Workspace Study is intentionally server-backed only. Local mode now fails closed with explicit unavailable copy, disabled lifecycle controls, and no mixed global/workspace flashcard or quiz lists.
- Flashcards and quizzes now both honor the active Study scope, use workspace-scoped empty-state copy, and hard-reset active review/attempt state on scope transitions and backend-derived availability changes.
- Remaining study follow-on work is narrower: study import/export mapping, remediation conversion, future sync/dual-write behavior, and any later cross-scope move semantics rather than basic workspace Study routing.

## Phase 3: Retrieval And Advanced Workflows

- RAG/search/retrieval workflow alignment beyond the new admin seam
- Embeddings creation/export and retrieval-surface alignment
- Evaluations and study-surface alignment
- MCP/tools/skills compatibility work that depends on the earlier model decisions

## Phase 4: UX Modernization And Hermes-Inspired Enhancements

- Tool progress and tool-result visibility improvements
- Session/history ergonomics upgrades
- Model/provider control improvements
- Background-task and long-running operation visibility
- Approval/safety affordances where they fit Textual
- Only start this wave after at least one core parity vertical lands cleanly

## First Vertical Candidates

- Prompts / chatbooks
- Notes / workspace alignment
- Chat / conversations
- Characters / session alignment

## Recommended First Vertical

- `Prompts / chatbooks` should go first. It has the best combination of user-visible value, `tldw_server` alignment, existing local import/export seams, and low overlap with the current dirty chat UI work.
- `Notes / workspace alignment` should go second. It is high-value and strongly aligned with the offline-first goal, but it needs more entity-shape mapping than prompts/chatbooks.
- `Chat / conversations` remains a top-tier parity domain, but it should not be the next implementation branch until the active chat UI, navigation, and model-control edits are reconciled or isolated in a worktree.
- `Characters` should likely follow chat/session model decisions rather than lead them.

## Recommended Next Vertical

- `Study follow-on alignment` should stay next, but the scope should narrow to the remaining study gaps rather than reopening the flashcards foundation that is now landed.
- Prioritize the remaining study-resource gaps next: remediation/import-export mapping, any broader session-history persistence beyond explicit study attempt objects, any future server deck-delete contract if `tldw_server` adds one, and the later cross-scope move/sync policy that was intentionally deferred.
- Keep remaining evaluation follow-on work tightly scoped to the gaps that still exist after the execution slice: server target discovery/catalog UX, server sample-level results/detail parity if the API grows it, and any later sync/dual-write policy.
- Reuse the same local-first pattern from the earlier verticals: local study/eval artifacts remain authoritative by default, while future sync work layers on top of the normalized seams instead of replacing them.
- Keep Hermes-inspired improvements scoped to concrete workflow pain. Broader job centers, approval flows, and explicit local/server execution controls remain deferred unless a later study vertical forces a minimal compatibility surface.

## Dirty-Tree Overlap Risk

Snapshot from the earlier overlap audit for this branch. Refresh it before starting the next vertical if the active local edit set changes.

- `tldw_chatbook/Constants.py`
- `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- `tldw_chatbook/UI/Navigation/main_navigation.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/UI/Tab_Bar.py`
- `tldw_chatbook/UI/Tab_Links.py`
- `tldw_chatbook/Utils/Emoji_Handling.py`
- `tldw_chatbook/Widgets/enhanced_settings_sidebar.py`
- `tldw_chatbook/app.py`
- `tldw_chatbook/css/features/_chat.tcss`
- `tldw_chatbook/css/layout/_sidebars.tcss`
- `tldw_chatbook/css/layout/_tabs.tcss`
- `tldw_chatbook/css/tldw_cli_modular.tcss`
- `tldw_chatbook/Widgets/compact_model_bar.py` (untracked)

Initial read: any first implementation vertical that touches chat UI, navigation, or model/tool controls is likely to overlap with current local work and should either use a worktree or be explicitly reconciled first.
