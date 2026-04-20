# tldw_chatbook Study Flashcard Lifecycle Parity Design

**Date:** 2026-04-20  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`

## Goal

Define the next narrow study parity vertical after quiz lifecycle/history: add flashcard lifecycle behavior to `tldw_chatbook` through the existing local/server compat seam, covering card deletion, in-backend card moves between decks, and deck deletion where the active backend actually supports it.

## Context

The flashcards parity foundation is already in place in this worktree.

Current `tldw_chatbook` flashcards compatibility work already provides:

- shared `tldw_api` deck/card/review client coverage
- `Study_Interop` local and server services for deck/card create-list and review
- a mode-aware `StudyScopeService`
- a screen-local flashcards controller in the Study TUI
- explicit empty states and backend-aware review-session handling

Relevant local files already in play:

- `tldw_chatbook/DB/ChaChaNotes_DB.py`
- `tldw_chatbook/tldw_api/client.py`
- `tldw_chatbook/tldw_api/flashcards_schemas.py`
- `tldw_chatbook/Study_Interop/local_study_service.py`
- `tldw_chatbook/Study_Interop/server_study_service.py`
- `tldw_chatbook/Study_Interop/study_scope_service.py`
- `tldw_chatbook/Study_Interop/study_normalizers.py`
- `tldw_chatbook/UI/Study_Modules/flashcards_handler.py`
- `tldw_chatbook/UI/Study_Window.py`

Current gap:

- local flashcards only expose create/list/review behavior through the seam
- the Study TUI has no lifecycle controls for decks or cards
- the shared client does not yet expose typed flashcard update/delete methods
- server and local lifecycle support are asymmetric

The current `tldw_server` flashcards API exposes:

- deck create/list/update
- flashcard create/list/get/update/delete
- review/review-session flows

It does **not** currently expose deck deletion in the API router. That asymmetry is a hard design constraint for this vertical.

## Product Decisions

The following decisions are fixed for this slice:

- This vertical is intentionally narrow.
- `tldw_chatbook` remains a standalone, local-first application.
- The active backend remains the only source of truth for this slice.
- No mixed local/server flashcard list or mixed deck picker is introduced.
- No cross-backend moves are allowed.
- No cross-workspace moves are allowed.
- No workspace-aware study behavior changes are included in this slice.
- No import/export work is included in this slice.
- No remediation conversion work is included in this slice.
- No sync or dual-write behavior is included in this slice.

Lifecycle actions included:

- `Delete selected card`
- `Move selected card to another deck`
- `Delete selected deck`

Backend support decisions:

- `Delete selected card`
  - supported in `local` mode
  - supported in `server` mode
- `Move selected card to another deck`
  - supported in `local` mode
  - supported in `server` mode through flashcard update
- `Delete selected deck`
  - supported in `local` mode
  - **not** supported in `server` mode because the current server API does not expose a deck-delete endpoint

Server-mode deck-delete UX is fixed for this slice:

- the control remains visible so the capability gap is obvious
- the control is disabled in `server` mode
- the UI shows an explicit explanation that server deck deletion is not supported by the current API
- the client/service layer must not invent a fake server delete behavior

Local deck-delete semantics are also fixed for this slice:

- local deck delete is a soft-delete operation
- local deck delete must also soft-delete the deck’s flashcards in the same transaction
- local deleted deck/cards must disappear from normal deck lists, flashcard lists, and due-review queries
- local soft-deleted decks must release their original `name` for reuse; the delete transaction must rewrite the stored deck name to a tombstone value rather than leaving the unique name permanently occupied

## In Scope

- Add local DB helpers for:
  - soft-deleting a deck and its child cards
  - soft-deleting a flashcard
  - moving a flashcard by updating `deck_id`
- Add typed `tldw_api` flashcard update/delete request support aligned with current `tldw_server` routes
- Add server-service wrappers for:
  - flashcard move via update
  - flashcard delete
  - explicit unsupported deck delete
- Add scope-service lifecycle methods for:
  - `delete_deck`
  - `delete_flashcard`
  - `move_flashcard`
- Add minimal Study TUI controls for deck/card lifecycle actions
- Keep review-state cleanup explicit when lifecycle actions invalidate current review context
- Add DB, client, interop, and TUI regression coverage
- Update parity docs after the vertical lands

## Out Of Scope

- Server deck deletion
- Cross-backend or cross-workspace card moves
- Flashcard review-history browsing
- Workspace-scoped deck visibility changes
- Import/export of decks or cards
- Remediation conversion flows
- Sync, dual-write, or reconciliation policy
- Broad Study UI redesign

## Approaches Considered

### Option A: Local-only flashcard lifecycle

Add deck/card lifecycle only for local mode and leave server mode unchanged.

Why not chosen:

- leaves an avoidable parity gap for card move/delete even though server routes already exist
- would force another seam expansion later for the same flashcard lifecycle actions

### Option B: Full flashcard lifecycle parity including server deck delete

Attempt to make all three lifecycle actions behave the same in local and server mode.

Why not chosen:

- current `tldw_server` API does not expose deck delete
- inventing a fake server path would break the compat-first rule

### Option C: Compat-first flashcard lifecycle with explicit server deck-delete gap

Implement the lifecycle actions that are genuinely supported by the active backend and surface the one missing server capability explicitly.

Why chosen:

- closes the concrete card lifecycle gap now
- keeps the UI honest about the current server API
- avoids widening into workspace, sync, or import/export policy

## Chosen Model

This vertical keeps the existing flashcards compat seam and extends it only where the current contracts justify it.

The model is:

- local mode remains fully standalone and authoritative
- server mode follows the real `tldw_server` flashcards API
- the UI continues to talk only to the mode-aware scope service
- lifecycle actions must carry optimistic-version data where the backend requires it
- lifecycle actions must explicitly clear or end invalidated review state

For move semantics, this vertical does **not** create a separate “move” entity or workflow. A move is simply a flashcard update that changes the owning `deck_id`.

## Architecture

### 1. Local Persistence

`ChaChaNotes_DB.py` must gain explicit flashcard lifecycle helpers rather than burying lifecycle behavior inside ad hoc UI logic.

Required local helpers:

- `delete_flashcard(card_id, expected_version=None, hard_delete=False)`
- `move_flashcard(card_id, target_deck_id, expected_version=None)`
- `delete_deck(deck_id, expected_version=None, hard_delete=False)`

Local rules:

- `move_flashcard` validates that the target deck exists and is not deleted
- `move_flashcard` updates `deck_id`, `updated_at`, `version`, and modifier metadata
- `delete_flashcard` soft-deletes the card by default
- `delete_deck` soft-deletes the deck and also soft-deletes all non-deleted child cards in one transaction
- `delete_deck` also rewrites the deleted deck row’s unique `name` to a tombstone value in the same transaction so the original user-visible name becomes reusable
- local deck lifecycle paths touched by this slice must keep `decks.card_count` coherent rather than allowing further drift

`card_count` handling is fixed for this slice:

- card create, card move, card delete, and deck delete must update or recalculate `decks.card_count` for every affected local deck in the same transaction
- the UI and normalizers should not start trusting stale `card_count` values while this work is in progress

This slice should not rely on the schema’s `ON DELETE CASCADE`, because that only helps on hard delete and does not solve the soft-delete visibility problem.

### 2. Shared Client And Schemas

`tldw_chatbook` currently lacks typed flashcard update/delete client coverage even though the server already supports those operations.

Add to `tldw_chatbook/tldw_api/flashcards_schemas.py`:

- `FlashcardUpdateRequest`

Add to `tldw_chatbook/tldw_api/client.py`:

- `update_flashcard(card_uuid, request_data)`
- `delete_flashcard(card_uuid, expected_version)`

These must mirror the real current server contract:

- update uses `PATCH /api/v1/flashcards/{card_uuid}`
- delete uses `DELETE /api/v1/flashcards/{card_uuid}?expected_version=...`
- flashcard scheduler-type schema values must accept the current server enum (`sm2_plus`, `fsrs`) rather than preserving a stale client-only alias

Do **not** add a deck-delete client method in this slice.

### 3. Study Interop

The existing flashcards boundary remains the right seam.

`local_study_service.py` adds:

- `delete_deck`
- `delete_flashcard`
- `move_flashcard`

`server_study_service.py` adds:

- `delete_flashcard`
- `move_flashcard`
- `delete_deck` as an explicit unsupported operation

`study_scope_service.py` adds:

- `delete_deck`
- `delete_flashcard`
- `move_flashcard`

Scope-layer behavior:

- route to the active backend only
- normalize successful flashcard move/delete results where needed
- treat server deck delete as unsupported, not as a silent no-op

### 4. Version Threading

This vertical must explicitly carry optimistic-locking information through the lifecycle path.

The normalized flashcard records already expose `version` in `study_normalizers.py`. The normalized deck records also expose `version`. The controller must use those versions when invoking:

- `delete_flashcard`
- `move_flashcard`
- `delete_deck`

Version flow:

- selected card in UI
- flashcards controller
- scope service
- local DB helper or server client

Deck-delete version flow:

- selected deck in UI
- flashcards controller
- scope service
- local DB helper

If this is omitted, server-mode lifecycle actions will fail against the current API contract.

If deck-delete version threading is omitted, local lifecycle semantics become weaker than the rest of the compat seam for no good reason. This slice therefore requires passing `expected_version` for local `delete_deck` from the selected normalized deck record.

### 5. TUI Structure

The Study flashcards pane should gain small lifecycle controls without redesigning the review flow.

Add to the existing flashcards pane:

- `Delete Deck` near the deck controls
- `Delete Selected Card` near the card list
- `Move To Deck` selector near the card list
- `Move Selected Card` button near the card list

UI rules:

- move-target selector is populated from the current backend’s deck list
- the current deck is excluded from valid move targets
- no selected deck means deck/card lifecycle controls are disabled
- no selected card means card lifecycle controls are disabled
- no valid target deck means move is disabled
- server-mode `Delete Deck` stays visible but disabled with explicit explanatory copy

### 6. Review-State Invalidation

Lifecycle actions can invalidate active review state. The controller must handle that explicitly.

Required behavior:

- deleting the currently reviewed card resets the review panel and ends any active review session if needed
- moving the currently reviewed card out of the selected deck resets the review panel and ends any active review session if needed
- deleting the selected deck in local mode resets card list state, selected deck state, and any active review session

This should reuse the existing `end_review_session_if_needed()` path in the flashcards controller rather than creating separate cleanup logic.

## Error Handling

Lifecycle errors must remain explicit and backend-aware.

### Delete Selected Card

- if no card is selected, show a direct user message
- if delete fails because the card no longer exists or the version conflicts, show a direct failure message
- do not mutate list state optimistically before the backend/local DB confirms success

### Move Selected Card

- if no card is selected, show a direct user message
- if no target deck is selected, show a direct user message
- if target deck equals current deck, no-op with a small message
- if target deck is missing/deleted, fail explicitly
- after success, refresh cards for the current deck so moved cards disappear immediately when they leave the current deck

### Delete Selected Deck

- in local mode, perform the real delete, then refresh decks/cards/review state
- in server mode, never call a fake delete path; the disabled control plus explanatory copy is the contract

## Verification Requirements

### DB Coverage

Add tests for:

- soft-deleting a flashcard hides it from default get/list flows
- moving a flashcard to another deck updates ownership and list visibility
- soft-deleting a deck also hides its child cards from:
  - `list_flashcards(...)`
  - `get_due_flashcards(...)`
- soft-deleting a deck releases the original deck name so a new local deck can be created with that same user-visible name
- local `card_count` stays coherent after card create, move, delete, and deck delete

### Client Coverage

Add shared client tests for:

- flashcard update route wiring
- flashcard delete route wiring
- flashcard schema acceptance of the current server scheduler-type enum values

No deck-delete client test should be added in this slice.

### Interop Coverage

Add tests for:

- local study service lifecycle wrappers
- server study service move/delete wrappers
- scope-service routing for card delete and move
- scope-service routing for local deck delete with `expected_version`
- explicit unsupported server deck-delete behavior

### UI Coverage

Add Study flashcards tests for:

- deleting the selected card
- moving the selected card to another deck in the current backend
- deleting the selected deck in local mode
- explicit disabled/unsupported deck-delete behavior in server mode
- review panel reset when the current review card is moved or deleted

## Documentation Updates

When this vertical lands, update:

- `Docs/Parity/2026-04-19-data-compatibility-map.md`
- `Docs/Parity/2026-04-19-rollout-backlog.md`

The docs should record:

- flashcard card lifecycle parity now exists for local/server mode
- local deck deletion is now supported through the compat seam
- server deck deletion remains explicitly unsupported because the current API does not expose it
- workspace-scoped study behavior, import/export, remediation mapping, and sync remain deferred
