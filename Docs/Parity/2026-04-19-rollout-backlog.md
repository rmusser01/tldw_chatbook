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

Three parity verticals are now landed in the isolated `codex-prompts-chatbooks-parity` worktree.

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

## Phase 3: Retrieval And Advanced Workflows

- Media/files/ingestion alignment
- RAG/chunking/template alignment
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

- `Characters / session alignment` should now be the active follow-on branch.
- Start with entity mapping and API-surface comparison before UI work, especially character cards, session ownership, and character-linked message history.
- Reuse the same pattern from the earlier verticals: thin client additions first, local schema compatibility second, adapter/service layer third, then one primary UI surface.
- Keep the application local-first: local character/session edits remain authoritative by default, and future sync work should layer on top of the normalized seams instead of replacing them.
- Keep Hermes-inspired improvements scoped to concrete workflow pain. Broader job centers, approval flows, and explicit local/server execution controls remain deferred unless the character vertical requires a minimal compatibility surface.

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
