# tldw_chatbook Notes And Workspaces Parity Vertical Design

**Date:** 2026-04-19  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`

## Goal

Define the next parity vertical after prompts/chatbooks: add live server-backed note CRUD and full workspace CRUD to `tldw_chatbook` while preserving one primary notes surface and enforcing strict separation between user-space notes and workspace-contained notes.

## Context

The current `tldw_chatbook` notes experience is built around one local-first note editor surface:

- `tldw_chatbook/UI/Screens/notes_screen.py`
- `tldw_chatbook/UI/Notes_Window.py`
- `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_left.py`
- `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py`
- `tldw_chatbook/Notes/Notes_Library.py`
- `tldw_chatbook/DB/ChaChaNotes_DB.py`

That local surface supports note editing, searching, keyword handling, and sync-adjacent UI, but it does not expose a workspace model and it does not currently offer a coherent server-backed notes/workspaces flow.

On the server side, notes and workspaces are broader and more structured:

- `tldw_server/tldw_Server_API/app/api/v1/endpoints/notes.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- `tldw_server/tldw_Server_API/app/api/v1/schemas/notes_schemas.py`
- `tldw_server/tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`

The server separates:

- user-space notes
- workspace records
- workspace-contained notes
- workspace sources
- workspace artifacts

This is the parity gap this vertical addresses.

## Product Decisions

The following scope and behavior decisions are fixed for this vertical:

- The vertical adds **live server-backed** note/workspace CRUD, not just compatibility scaffolding.
- The app keeps **one primary notes screen**, not a second server-only notes product.
- The notes screen uses a **mixed navigator**, but not a flattened mixed list.
- User-space notes and workspace-contained notes must remain **visibly and behaviorally separate**.
- A user can browse, search, edit, and manage notes from their user-space note collection independently of workspaces.
- Workspace notes are only visible and searchable **inside the selected workspace context**.
- Workspace notes must not appear in the general user-space notes list or search results.
- This vertical includes full CRUD exposure in the TUI for all four workspace surfaces:
  - workspace record
  - workspace notes
  - workspace sources
  - workspace artifacts
- The server is the source of truth for server-backed notes/workspaces in this vertical.
- No immediate local mirror is written for server-backed notes/workspaces.
- Cross-scope note moves are out of scope.
- Notes graph and manual note-link management are out of scope.
- Sync/import-export are out of scope.

## In Scope

- Add server note CRUD support to the shared `tldw_api` client.
- Add workspace CRUD support to the shared `tldw_api` client.
- Add server contracts for workspace notes, sources, and artifacts.
- Build a service/adaptor layer that the TUI can call for server-backed note and workspace operations.
- Refactor the main notes screen to support three navigator scopes:
  - local notes
  - server user-space notes
  - workspaces
- Add workspace context rendering so a selected workspace exposes:
  - workspace details
  - workspace notes
  - workspace sources
  - workspace artifacts
- Preserve existing local notes behavior.
- Add scope-aware labels, actions, and refresh/error handling.
- Add regression coverage for strict scope separation.

## Out of Scope

- Notes graph endpoints and graph UI
- Manual note links
- Sync and bi-directional file sync
- Import/export for server notes/workspaces
- Cross-scope note movement between user-space and workspaces
- Local mirroring of server-backed resources
- Workspace source batch selection/reorder controls beyond the CRUD surface
- Rewriting unrelated local note features

## Approaches Considered

### Option A: Separate server notes/workspaces screen

Build a second server-focused screen beside the local notes surface.

Why not chosen:

- Splits the notes product into two separate mental models
- Makes parity harder to discover and use
- Increases UI duplication

### Option B: One unified screen with mode switching

Keep one screen but force explicit local/server mode switches.

Why not chosen:

- Better than a second screen, but still too mode-heavy
- Makes workspace context feel bolted on instead of first-class

### Option C: One unified screen with a mixed navigator and strict scope separation

Keep one notes screen, but use a navigator with separate sections for local notes, server notes, and workspaces. Selecting a workspace reveals a workspace-scoped subview instead of flattening its notes into the general note list.

Why chosen:

- Matches the requested product behavior
- Preserves one primary TUI surface
- Keeps workspace-contained notes visibly separate
- Makes future graph and sync work easier because scope boundaries stay explicit

## Chosen UI Model

### Navigator

The left side of the notes screen becomes a mixed navigator with three top-level sections:

- `Local Notes`
- `Server Notes`
- `Workspaces`

Rules:

- `Local Notes` continues to list only local notes.
- `Server Notes` lists only user-space server notes.
- `Workspaces` lists only workspace records.
- Workspace notes must never be merged into either notes list.

### Main Pane Behavior

Selecting an item changes the center/right panes by scope:

- Selecting a local note keeps the existing local-note editor behavior.
- Selecting a server note opens a server-backed note editor.
- Selecting a workspace switches the screen into a workspace context view.

### Workspace Context View

When a workspace is selected, the main notes surface shows workspace-scoped collections and actions:

- `Workspace Details`
- `Workspace Notes`
- `Sources`
- `Artifacts`

This workspace context is self-contained:

- workspace notes are only searchable inside this context
- workspace source operations apply only to the selected workspace
- workspace artifact operations apply only to the selected workspace
- deleting or archiving a workspace does not implicitly convert its notes into user-space notes

### Scope Visibility Requirements

The UI must make scope obvious:

- header labels should identify whether the user is in `Local Note`, `Server Note`, or `Workspace`
- action controls should be scope-specific
- dangerous or unsupported actions should be omitted instead of shown as inert placeholders

## Architecture

The implementation should stay thin and additive, following the same vertical pattern used for prompts/chatbooks.

### 1. Shared API Client Layer

Extend `tldw_chatbook/tldw_api` with server note/workspace contracts and client methods for:

- user-space notes CRUD
- workspace CRUD
- workspace notes CRUD
- workspace sources CRUD
- workspace artifacts CRUD

The client layer should mirror server request/response contracts closely rather than invent a second abstraction.

### 2. Notes/Workspace Service Layer

Add a new service layer between the UI and `tldw_api` that:

- normalizes note/workspace payloads for the TUI
- centralizes optimistic-locking fields and server error handling
- keeps local and server operations clearly separated
- provides a scope-aware interface for the screen

This service should be the only place where the notes screen decides whether it is handling:

- local note operations
- server user-space note operations
- workspace subresource operations

### 3. Unified Notes Screen State

Refactor `notes_screen.py` to use a scope-aware state model.

The state model should track:

- current scope type
- selected local note or selected server note
- selected workspace
- selected workspace subview
- current server-side version fields for editable resources
- loading, refreshing, and error states per server scope

The state should not allow workspace notes to exist in the same selection collection as user-space notes.

### 4. Existing Local Notes Preservation

The local notes path must remain functional and conceptually independent.

The notes screen should reuse local-note behavior where possible, but not by pretending server workspace notes are local notes. Shared editor widgets are acceptable; shared resource identity is not.

## Data Flow

### User-Space Server Notes

When the user selects `Server Notes`:

- load server note listings only
- run server note search only
- create/edit/delete server user-space notes only
- save through the server API with optimistic locking
- use explicit refresh on success/failure instead of assuming local mirror state

### Workspaces

When the user selects `Workspaces`:

- list workspace records only
- selecting a workspace loads:
  - workspace metadata
  - workspace notes
  - workspace sources
  - workspace artifacts

Workspace notes are separate resources from general server notes in the UI model even if the server stores related note structures underneath.

### No Cross-Scope Moves

This vertical does not allow:

- moving a user-space note into a workspace
- moving a workspace note out into user-space notes

Those operations belong to a later vertical because they affect long-term sync and resource ownership semantics.

## Error Handling

The vertical should prefer explicitness over optimistic UI.

- Handle `409` conflicts from optimistic locking with a refresh-and-retry path.
- Surface server validation errors inline where practical.
- Keep separate loading/error states for:
  - note list loading
  - single-resource save/delete
  - workspace context loading
  - workspace subresource operations
- Avoid pretending a server write succeeded before the API confirms it.
- Do not silently fall back to local storage for failed server writes.

## Testing Strategy

This vertical should be implemented test-first.

### API / Schema Tests

Add tests for:

- note request/response contracts
- workspace request/response contracts
- workspace notes/sources/artifacts request shapes

### Service Tests

Add tests for:

- scope-aware CRUD helpers
- optimistic-locking field propagation
- workspace context loading helpers
- separation between server notes and workspace notes

### UI Tests

Add notes screen tests for:

- mixed navigator sections render correctly
- local notes remain isolated
- server notes show only in server-notes scope
- workspace notes render only after selecting a workspace
- sources and artifacts render only inside the selected workspace context
- scope labels/actions change correctly with selection

### Regression Tests

Keep existing local-notes tests running so this vertical proves:

- local notes are not broken
- workspace resources do not leak into general notes
- server-backed flows do not require sync/local mirror to function

## Main Risks

- The current notes screen and sidebars are local-note-oriented and will need a real state cleanup.
- The server notes/workspaces surface is broader than the assumptions in the current local screen.
- A mixed navigator can easily leak scopes unless state and list rendering are explicit.
- Without local mirroring, refresh behavior must be reliable and visible.

## Definition Of Done

This vertical is complete when:

- one unified notes screen supports local notes, server user-space notes, and workspaces
- server notes CRUD works live from the TUI
- workspace CRUD works live from the TUI
- workspace notes, sources, and artifacts CRUD are exposed in the TUI
- workspace notes do not appear in general note lists or searches
- no graph, sync, local mirror, or cross-scope move behavior is introduced
- targeted API, service, UI, and regression tests pass

## Follow-On Work

This vertical intentionally sets up later work without implementing it:

- notes graph and manual links
- sync/import-export
- cross-scope note movement
- local mirroring and eventual offline sync semantics
