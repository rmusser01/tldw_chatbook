# tldw_chatbook Chat And Conversations Parity Vertical Design

**Date:** 2026-04-19  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`

## Goal

Define the next parity vertical after notes/workspaces: align `tldw_chatbook` chat and conversation data models with `tldw_server` while keeping the current chat UI visually unchanged and preserving `tldw_chatbook` as a local-first standalone application.

## Context

The current `tldw_chatbook` chat surface already supports local conversation persistence and message history:

- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- `tldw_chatbook/Chat/Chat_Functions.py`
- `tldw_chatbook/Chat/chat_models.py`
- `tldw_chatbook/Chat/tabs/tab_state_manager.py`
- `tldw_chatbook/DB/ChaChaNotes_DB.py`

That local model is not empty or provisional. It already stores:

- conversations
- messages
- parent/child message relationships
- soft deletes
- optimistic-locking versions
- conversation keywords
- message variants

However, the broader server chat surface is now more structured and exposes higher-level conversation workflows on top of similar primitives:

- `tldw_server/tldw_Server_API/app/api/v1/endpoints/chat.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/messages.py`
- `tldw_server/tldw_Server_API/app/api/v1/schemas/chat_conversation_schemas.py`
- `tldw_server/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

The server adds or normalizes:

- paged conversation search with filters and ranking
- conversation metadata beyond title/rating
- conversation tree retrieval
- assistant identity metadata
- workspace/global conversation scope
- richer topic, cluster, keyword, and source metadata

This is the parity gap this vertical addresses.

## Product Decisions

The following behavior and scoping decisions are fixed for this vertical:

- `tldw_chatbook` remains a standalone local-first application.
- Local persistence remains the default behavior for chat actions in this vertical.
- The main chat UI stays visually unchanged for this pass.
- No dedicated chat `Local / Server` storage-mode toggle is added in this pass.
- Any future server-backed conversation mode must be explicit, not inferred from provider/model settings.
- The current chat screen should continue to work without server availability.
- This vertical is compatibility-first, not live remote-chat-first.
- Local conversation creation must be capable of representing character-backed, persona-backed, and non-character conversations; `character_id` can no longer be treated as universally required in the aligned model.
- Workspace-scoped conversations may exist in the aligned local model, but this pass must not accidentally leak workspace-scoped records into general chat history views.
- The existing destructive rewrite-on-resave workflow is not an acceptable end state for this vertical because it drops message topology and variant fidelity.
- The deferred local/server mode toggle must be tracked as explicit follow-on work in GitHub.

## In Scope

- Extend local conversation storage toward the current server-shaped conversation model.
- Add missing local schema/migration coverage for server-aligned conversation metadata.
- Add local DB/service helpers for conversation list/search, metadata retrieval/update, keyword replacement, message-tree traversal, and message counting where those server seams already exist conceptually.
- Add `tldw_api` client schemas and methods for the current server conversation endpoints.
- Add a chat conversation compatibility service/adaptor layer that can normalize local records and server payloads into one TUI-facing model.
- Refactor local conversation creation so assistant identity is not hard-wired to `character_id` or the default-character fallback.
- Replace the current full-resave message rewrite behavior with a persistence approach that preserves message topology and message-level metadata.
- Preserve current local chat save/load/history workflows where they do not conflict with the aligned data model.
- Add regression tests for local-first behavior and compatibility helpers.
- Document the deferred chat mode toggle as follow-on work.

## Out Of Scope

- Adding a chat `Local / Server` toggle in the UI
- Live server-backed conversation creation from the main chat surface
- Remote-first chat behavior
- Dual-write and sync semantics
- Sync conflict resolution
- Cross-device conversation reconciliation
- Major Hermes-inspired chat UX redesign
- Reworking the active chat layout, sidebars, or navigation structure unless a strictly local compat change requires a small internal adjustment

## Approaches Considered

### Option A: API client parity only

Add `tldw_api` conversation methods and leave the local DB and chat service seams mostly unchanged.

Why not chosen:

- Leaves the offline-first model under-specified
- Creates another round of local cleanup later
- Does not actually reduce the model mismatch between local chat data and server chat data

### Option B: Local-first compatibility layer with unchanged UI

Align the local schema, local DB/service helpers, and `tldw_api` surface to the server model while keeping the current chat UI behavior and layout intact.

Why chosen:

- Matches the requested local-first product direction
- Reduces future sync and interop risk without requiring premature server-mode UX decisions
- Avoids direct overlap with the active chat UI work
- Produces reusable local/server normalization seams for later phases

### Option C: Full local/server chat mode in the main chat UI now

Add the missing chat mode toggle and expose explicit local/server conversation behavior in the main chat surface during this vertical.

Why not chosen:

- Requires product and sync semantics that are intentionally deferred
- Directly overlaps the unstable chat UI area
- Mixes compatibility work with UX and workflow decisions that should come later

## Chosen Model

This vertical treats the local conversation store as the primary execution path and treats the server conversation surface as the compatibility target.

The immediate objective is not to make the main chat screen talk to the server live. The immediate objective is to make local conversation records and local conversation operations look enough like the server model that later remote mode and sync work can build on a stable shared shape.

That means:

- the current chat surface continues to read and write local state
- the local DB grows the missing conversation metadata needed for parity
- local search/list/tree helpers are elevated to first-class service seams
- `tldw_api` gains explicit server conversation coverage
- an adaptor layer becomes the contract boundary between storage backends and the TUI

## Compatibility Targets

### Conversation Metadata

The local conversation model should be extended toward the server conversation contract. The target aligned field set for this vertical is:

- `id`
- `title`
- `created_at`
- `last_modified`
- `version`
- `character_id`
- `assistant_kind`
- `assistant_id`
- `persona_memory_mode`
- `scope_type`
- `workspace_id`
- `state`
- `topic_label`
- `topic_label_source`
- `topic_last_tagged_at`
- `topic_last_tagged_message_id`
- `cluster_id`
- `source`
- `external_ref`

This pass does not need every future field to be fully surfaced in the UI, but the local storage and service layers should be capable of preserving and normalizing them.

### Conversation Identity And Creation

The aligned local creation model must stop assuming every conversation is character-bound.

The minimum supported local conversation identity shapes for this vertical are:

- character-backed conversation:
  - `character_id` is set
  - `assistant_kind='character'`
  - `assistant_id` maps to the selected character identity
- persona-backed conversation:
  - `character_id` is null
  - `assistant_kind='persona'`
  - `assistant_id` maps to the selected persona identity
- non-character or generic conversation:
  - `character_id` is null
  - `assistant_kind` is null
  - `assistant_id` is null

The current main chat UI does not need new visual controls in this pass, but the local DB and local service layer must be capable of representing these shapes. The existing default-character fallback can remain as a legacy behavior only where an older UI path still supplies no explicit assistant identity, but it must no longer be the defining model for conversation creation.

### Conversation Metadata Mapping Rules

The plan should treat the following conversation fields as the minimum explicit mapping contract for local/server alignment:

| Field | Local Requirement In This Vertical | Legacy/Backfill Rule |
|---|---|---|
| `character_id` | Keep as an optional local FK for character-backed conversations only | Existing rows keep their current value; new persona or generic conversations may store null |
| `assistant_kind` | Store locally when present; normalize existing character-backed records to `character` when possible | Existing rows without value may backfill to `character` when `character_id` exists, otherwise remain null |
| `assistant_id` | Store locally when present | Existing rows may backfill from `character_id` when `assistant_kind='character'` |
| `persona_memory_mode` | Store locally when present | Existing rows default to null |
| `scope_type` | Store locally and use in local service filtering | Existing rows default to `global` |
| `workspace_id` | Store locally when scope is `workspace` | Existing rows default to null |
| `state` | Store locally and expose through normalized service payloads | Existing rows default to `in-progress` |
| `topic_label` | Store locally | Existing rows default to null |
| `topic_label_source` | Store locally to preserve manual/auto provenance | Existing rows default to null |
| `topic_last_tagged_at` | Store locally to preserve server-compatible topic tagging timestamps | Existing rows default to null |
| `topic_last_tagged_message_id` | Store locally to preserve the last message associated with topic tagging | Existing rows default to null |
| `cluster_id` | Store locally when present | Existing rows default to null |
| `source` | Store locally when present | Existing rows default to null |
| `external_ref` | Store locally when present | Existing rows default to null |

### Conversation Keywords

Local conversation keyword joins already exist. This vertical should keep them and make them part of the explicit parity model:

- fetch keywords for one conversation
- fetch keywords for a conversation page
- replace the full keyword set atomically when metadata updates require it

### Conversation Search

The server already exposes paged conversation listing and search with filters. This vertical should add a local equivalent service seam even if the current chat UI does not expose every filter immediately.

The local seam should support:

- list active conversations
- title/content search
- paging
- keyword lookup for result rows
- message counts
- future filtering by scope, topic, or state where the schema supports it

The implementation plan should mirror the current server search and tree semantics closely enough that later local/server switching does not require another contract rewrite:

- paged responses should preserve `limit`, `offset`, `total`, and `has_more`
- default local ordering should remain recency-first unless an explicit ranking mode is requested
- the local compatibility seam should accept the server ranking values `recency`, `bm25`, `hybrid`, and `topic`, even if the current UI only relies on recency-first behavior in this pass
- deleted conversations should stay excluded from default list/search results
- any include-deleted behavior added locally must remain explicit and opt-in
- conversation-tree helpers should preserve parent/child ordering by ascending message timestamp

### Conversation Tree

The server exposes a conversation message tree API that reconstructs message threads from parent-child links. Local chatbook already stores `parent_message_id`, so this vertical should make tree retrieval a first-class local helper instead of leaving that structure implicit in raw table access.

### Message Updates And Variants

Local message update, soft delete, feedback, and variant support already exist. This vertical should preserve those capabilities and ensure the compatibility service can present them in a normalized form for future server-aware paths.

### Message Persistence Fidelity

The current local rewrite-on-resave path is incompatible with the aligned conversation model because it soft-deletes all messages and recreates them from flattened history. That behavior is incompatible with stable:

- parent/child message relationships
- message variants and selected-variant state
- message feedback
- future message-level metadata that depends on stable message identities

This vertical therefore requires a persistence refactor. The implementation plan must replace the destructive full-resave path with a model that preserves message structure. Incremental append/update semantics are preferred, but any acceptable approach must preserve stable message topology and message-level metadata for aligned conversations.

## Architecture

The implementation should stay additive and follow the same layered pattern used in the previous parity verticals.

### 1. Local DB Compatibility Layer

Extend `tldw_chatbook/DB/ChaChaNotes_DB.py` with the missing chat conversation schema compatibility work:

- add migrations for missing server-aligned conversation columns
- add any required indexes for new filters or scope handling
- add helper methods for server-shaped conversation retrieval and paging
- add message-tree helpers if they are missing or still only exist in the server copy

The goal is not to duplicate the entire server DB module. The goal is to lift the local DB to the minimum stable contract needed for later interoperability.

### 2. Chat Conversation Service Layer

Add a dedicated local service or adaptor layer between the chat UI and raw DB calls.

This service should:

- normalize local rows into a stable conversation/message payload
- own keyword loading and replacement logic
- centralize scope-aware defaults
- expose message-tree retrieval without requiring UI code to know DB details
- isolate future backend choice between local records and remote payloads

The current UI should not have to understand whether it is reading raw local rows, server payloads, or a future merged source.

### 3. `tldw_api` Client Parity

Extend `tldw_chatbook/tldw_api` with explicit conversation and message-adjacent schemas and methods for the server endpoints already present in `tldw_server`.

The client should cover at least:

- list/search conversations
- fetch one conversation metadata record
- update conversation metadata
- fetch conversation tree

This does not imply the current main chat surface will call those methods yet. It means the client contract should exist before later remote mode work begins.

### 4. UI Preservation

The current main chat UI remains visually unchanged:

- no new mode toggle
- no new server-mode banner
- no reworked layout
- no new chat history panes solely for this vertical

Internal plumbing changes are acceptable if they keep behavior stable and reduce future parity work, but this vertical should not become a stealth UI rewrite.

## Data Rules

### Local-First Persistence

In this vertical:

- creating or updating chats continues to persist locally
- loading chat history continues to read locally
- editing conversation/message metadata continues to be defined first in terms of local state

This preserves the standalone nature of `tldw_chatbook`.

Local-first does not mean preserving every legacy write path unchanged. Where an existing local persistence path destroys aligned metadata or message structure, this vertical should change that path rather than freeze it for compatibility.

### Scope Safety

The server model supports global and workspace-scoped conversations. This vertical should add or align the local schema for those fields, but must preserve current user expectations:

- general chat history views should continue to behave as general user-space history
- workspace-scoped conversations must not silently appear in general history surfaces unless and until the product deliberately defines that behavior

#### Scope Defaults

The local-first service layer must apply explicit scope defaults so implementation does not guess:

| Operation | Default Behavior |
|---|---|
| Create conversation with no workspace context | create with `scope_type='global'` and `workspace_id=null` |
| Create conversation from an explicit workspace-scoped entry point | create with `scope_type='workspace'` and the active `workspace_id` |
| General chat history list/search | return only `scope_type='global'` by default |
| Explicit workspace conversation list/search | return only rows matching `scope_type='workspace'` and the requested `workspace_id` |
| Metadata fetch/update for a known conversation | honor the record's persisted scope and do not silently rewrite it |
| Legacy conversation row with no scope fields populated | normalize as `scope_type='global'` and `workspace_id=null` |

These defaults are intentionally conservative. This vertical should prefer hiding workspace-scoped conversations from general history rather than risking cross-scope leakage.

#### State And Delete Semantics

`state` and soft-delete remain separate concepts in the aligned model:

- allowed `state` values should mirror the current server contract:
  - `in-progress`
  - `resolved`
  - `backlog`
  - `non-viable`
- `state` is a lifecycle classification such as `in-progress` or `resolved`
- soft-delete controls whether a record is active or hidden
- deleted conversations remain excluded from default list/search flows even if their `state` is still populated
- this vertical must not treat deletion as another `state` value

### Provider Selection Is Not Storage Selection

The existing model/provider controls and other `Remote LLM` settings are not a chat storage-mode control. This vertical must keep that distinction explicit in code and docs so later work does not accidentally overload unrelated settings.

## Deferred Future Toggle

This vertical intentionally defers a dedicated chat `Local / Server` storage-mode toggle in the main chat surface.

Current validated state:

- there is no dedicated chat conversation storage-mode toggle in the main chat surface today
- existing provider/model controls must not be repurposed as a storage toggle
- existing media-ingest local/remote controls are unrelated and must not be treated as chat storage mode

That follow-on work must define:

- what `Local` means for create/update/delete operations
- what `Server` means for create/update/delete operations
- whether server mode writes only remotely or also dual-writes locally
- how remote reads populate local state
- how conflicts and offline fallback behave
- how the toggle is displayed without confusing it with provider/model selection

Until those semantics are defined, the toggle should not be added.

### Required Follow-On Issue

This design requires a GitHub issue to track the deferred chat storage-mode control explicitly.

Recommended issue title:

- `Add explicit Local / Server conversation mode control to the main chat surface`

Recommended issue body points:

- the current chat screen has no dedicated storage-mode toggle
- provider/model selection is not storage-mode selection
- local-first remains the default chat persistence mode
- future server mode must define write behavior, local replication behavior, and offline fallback before UI exposure
- the issue should include acceptance criteria for labeling, persistence semantics, and interaction with future sync work

Tracked issue:

- [Issue #141: Add explicit Local / Server conversation mode control to the main chat surface](https://github.com/rmusser01/tldw_chatbook/issues/141)

## Testing Strategy

The implementation plan for this vertical should include regression coverage in four layers:

### Local DB Migration Coverage

- verify new conversation columns are created or migrated correctly
- verify existing local DBs remain readable
- verify defaults preserve current behavior for legacy records

### Local Service Coverage

- verify local rows normalize into the expected compat shape
- verify conversation keyword replacement/fetch behavior
- verify message-tree retrieval from parent/child rows
- verify workspace-scoped records do not leak into general global queries

### `tldw_api` Client Coverage

- verify request/response schema handling for server conversation endpoints
- verify pagination and optional filters serialize correctly

### UI Regression Coverage

- verify the chat screen still mounts and restores state
- verify current local chat history persistence still works
- verify no new chat mode controls appear in this pass

## Risks And Mitigations

### Risk: accidental chat UI overlap

The current chat UI area already has active local edits and was previously identified as high-overlap risk.

Mitigation:

- keep this vertical compatibility-first
- avoid visible UI changes
- prefer DB/service/client seams first

### Risk: over-copying the server DB module

The server chat DB implementation is much broader than what this vertical needs.

Mitigation:

- port only the conversation/message helpers required for stable parity seams
- avoid dragging in unrelated server-only workflows

### Risk: implicit storage-mode confusion

There are already unrelated `Remote` and provider controls elsewhere in the app.

Mitigation:

- explicitly document that provider selection is not conversation storage mode
- track the missing toggle as its own issue instead of quietly overloading another control

## Success Criteria

This vertical is successful when:

- local conversation records can preserve the core server-aligned metadata needed for later interoperability
- local chat persistence remains the default and continues to work without server access
- `tldw_api` exposes the relevant server conversation contracts
- the TUI chat surface remains visually unchanged
- a future chat `Local / Server` mode toggle is explicitly deferred and tracked instead of partially implied
