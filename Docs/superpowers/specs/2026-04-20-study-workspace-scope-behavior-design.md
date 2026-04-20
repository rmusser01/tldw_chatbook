# tldw_chatbook Study Workspace Scope Behavior Design

**Date:** 2026-04-20  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`

## Goal

Define the next study parity vertical after flashcard lifecycle parity: add strict workspace-scoped Study behavior to `tldw_chatbook` so workspace-owned study items remain isolated from global Study, while keeping `tldw_chatbook` standalone and local-first.

## Context

The current study compatibility work in this worktree already provides:

- server-compatible flashcard deck/card CRUD and review flows through `Study_Interop`
- server-compatible quiz CRUD, question, and attempt flows through `Study_Interop`
- normalized study records that already carry `workspace_id` on server-backed deck/quiz payloads
- a screen-local flashcards controller in `tldw_chatbook/UI/Study_Modules/flashcards_handler.py`
- a screen-local quizzes controller in `tldw_chatbook/UI/Study_Modules/quizzes_handler.py`
- a shared Study screen/window in:
  - `tldw_chatbook/UI/Screens/study_screen.py`
  - `tldw_chatbook/UI/Study_Window.py`

The current notes/workspaces parity work in this worktree already established the product rule that:

- user-space notes are visible in the general notes scope
- workspace-contained notes remain visible only inside workspace context
- workspace-contained content should not leak back into general lists

That same boundary rule now needs to be applied to Study.

Current constraints relevant to this vertical:

- server-backed study records already have a `workspace_id` seam in the shared contracts
- local study persistence does **not** yet implement real workspace ownership
- the shared API client currently does **not** expose workspace-filter query params for flashcard-deck or quiz list endpoints
- `StudyScreen` currently has no scope/context handoff seam and is opened as a normal application screen
- Notes already has a workspace context panel that can be used as the user entry point into workspace-scoped Study

Relevant local files already in play:

- `tldw_chatbook/UI/Screens/study_screen.py`
- `tldw_chatbook/UI/Study_Window.py`
- `tldw_chatbook/UI/Study_Modules/flashcards_handler.py`
- `tldw_chatbook/UI/Study_Modules/quizzes_handler.py`
- `tldw_chatbook/Study_Interop/study_scope_service.py`
- `tldw_chatbook/Study_Interop/quiz_scope_service.py`
- `tldw_chatbook/Study_Interop/study_normalizers.py`
- `tldw_chatbook/Study_Interop/quiz_normalizers.py`
- `tldw_chatbook/UI/Screens/notes_screen.py`
- `tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py`
- `tldw_chatbook/app.py`

## Product Decisions

The following decisions are fixed for this slice:

- This vertical is about **Study scope behavior**, not broad Study feature expansion.
- `tldw_chatbook` remains a standalone product that can sync later.
- The general Study surface remains the default user-space Study surface.
- Workspace-owned study items must never appear in the general Study lists.
- General Study items must never appear in workspace-scoped Study views.
- Workspace-scoped Study is entered from **workspace context**, not through a new Study-side workspace selector.
- This vertical covers both:
  - flashcard decks/cards
  - quizzes/questions/attempt history
- This vertical does **not** add local workspace persistence for study entities.
- In this slice, workspace-scoped Study is **server-backed only**.
- In local mode, workspace-scoped Study must be explicitly unavailable rather than silently falling back to global local study data.
- Create operations must inherit the active Study scope automatically.
- No cross-scope moves are allowed.
- No sync, dual-write, or local mirror policy is included.
- No mixed global/workspace Study view is allowed.
- No Study UI redesign is included beyond the minimum scope banner/header and entry/exit controls needed to make scope explicit.

## In Scope

- Add an app-owned Study scope handoff so Notes can activate the normal Study screen with workspace context.
- Add Study-specific scope state for:
  - `global`
  - `workspace`
- Add a workspace entry action in the workspace context panel.
- Add a scoped Study header/banner that makes workspace context explicit.
- Add explicit scoped exit/navigation actions from workspace-scoped Study.
- Enforce global-vs-workspace filtering for:
  - flashcard deck lists
  - quiz lists
- Thread `workspace_id` into workspace-scoped deck and quiz creation in server mode.
- Add hard action gating so workspace-scoped Study in local mode is unavailable and non-mutating.
- Reset flashcard review state and quiz attempt state on scope transitions.
- Add regression coverage for:
  - strict non-mixed visibility
  - scoped create behavior
  - unavailable-state enforcement
  - app-owned navigation handoff
- Update parity docs after the vertical lands.

## Out Of Scope

- Local DB schema changes for workspace-owned study entities
- Sync or local mirroring of workspace-scoped study entities
- Cross-scope moves between global Study and workspace-scoped Study
- Cross-workspace study moves
- Server API expansion to add new workspace-filter list parameters
- Mixed Study views that combine global and workspace records
- Broader Study screen redesign
- Study import/export mapping
- Quiz remediation conversion

## Approaches Considered

### Option A: Add a Study-side workspace selector

Let the Study screen switch between global Study and workspace-scoped Study through a workspace picker inside Study itself.

Why not chosen:

- creates a second workspace-navigation model beside Notes/workspaces
- weakens the approved product rule that workspace-contained content should be entered from workspace context
- increases the risk of mixed mental models and hidden scope mistakes

### Option B: Add full local workspace study ownership now

Introduce local workspace-backed ownership for decks/quizzes so workspace Study works in both local and server mode immediately.

Why not chosen:

- local workspace ownership for study does not exist yet
- this vertical would become a local data-model expansion instead of a parity boundary pass
- it risks baking incomplete local workspace semantics before sync/graph-era decisions are made

### Option C: App-owned, workspace-context entry with server-only workspace Study

Keep the global Study screen as user-space only, add workspace-scoped Study entry from workspace context, enforce the boundary in compat and controller layers, and make workspace Study unavailable in local mode for now.

Why chosen:

- matches the approved non-mixed product rule
- uses the existing workspace context surface
- keeps the boundary correct without inventing half-finished local workspace persistence
- aligns with the current server contract shape

## Chosen Model

This vertical adds a **Study scope layer** above the current flashcards and quiz compatibility seams.

The model is:

- the Study product has two scopes:
  - `global`
  - `workspace`
- the app, not Notes, owns Study screen activation
- Notes can request workspace-scoped Study, but it does not create a special Study screen instance on its own
- Study controllers consume already-scoped deck/quiz lists from shared services
- local workspace-scoped Study is unavailable in this slice and must fail closed
- global Study remains the default and unchanged surface

## Architecture

### 1. App-Owned Study Activation

The app must own Study activation so scoped Study does not become a Notes-specific navigation exception.

Add an app-level seam in `tldw_chatbook/app.py`:

- `open_study_screen(scope_context: StudyScopeContext | None = None)`

Required behavior:

- Notes calls this helper instead of calling `push_screen(...)` directly
- the helper routes through the normal screen-switch path so `current_tab` and screen lifecycle remain correct
- the helper stores a pending Study scope context for the next Study activation
- the pending scope context must be consumed exactly once and then cleared
- the helper should always route through the normal screen-switch path and create a fresh `StudyScreen`, following the current app architecture
- the design does **not** add a second `apply_scope_context(...)` path for already-mounted Study screens

Pending-context precedence is fixed:

- if both restored screen state and a pending Study scope context exist, the pending Study scope context wins for that activation
- after it is consumed, that resolved Study scope becomes the new persisted Study screen state

This avoids stale workspace scope leaking into future ordinary Study navigation and avoids duplicate activation logic.

### 2. App-Owned Return To Workspace

Workspace-scoped Study needs an explicit exit path back to workspace context.

Add an app-level seam in `tldw_chatbook/app.py`:

- `open_notes_workspace(workspace_id: str, subview: WorkspaceSubview = WorkspaceSubview.DETAILS)`

Required behavior:

- `Back to Workspace` from scoped Study must return to Notes workspace `DETAILS`, not workspace notes
- the helper routes through the normal Notes screen switch path
- Notes receives a pending workspace-selection context from the app and resolves it on activation

Pending-context precedence is fixed here too:

- if both restored Notes state and a pending workspace-selection context exist, the pending workspace-selection context wins for that activation
- after it is consumed, that resolved workspace selection becomes the new persisted Notes state

This keeps navigation consistent and avoids Study making private calls into Notes internals.

### 3. Study Scope Models

Add a new file:

- `tldw_chatbook/UI/Screens/study_scope_models.py`

It should define:

- `StudyScopeType(str, Enum)` with:
  - `GLOBAL`
  - `WORKSPACE`
- `StudyScopeContext`
- `StudyScopeState`

`StudyScopeState` must carry:

- `scope_type`
- `workspace_id`
- `workspace_name`
- `workspace_scope_available`
- `backend`
- `error_message`
- an optional return hint for `Back to Workspace`

The state should be Study-owned and persist across Study screen suspend/resume.

### 4. Study Screen Ownership

`tldw_chatbook/UI/Screens/study_screen.py` must own the active Study scope state.

Required behavior:

- default Study activation with no pending context resolves to `GLOBAL`
- workspace-scoped activation resolves to `WORKSPACE`
- on activation, the screen computes `workspace_scope_available`
  - `True` only when:
    - scope is `WORKSPACE`
    - backend is `server`
    - `workspace_id` is present
  - `False` when:
    - scope is `WORKSPACE`
    - backend is `local`
    - or `workspace_id` is missing
- if workspace scope is requested without a `workspace_id`, the Study screen enters an invalid-entry error state instead of falling back to global Study

The screen should pass scope state into `StudyWindow` rather than forcing flashcards/quizzes controllers to infer scope from unrelated Notes state.

### 5. Study Window Scope UI

`tldw_chatbook/UI/Study_Window.py` must add a minimal scope-aware header.

Required workspace-scoped header content:

- workspace name
- a clear `Workspace Study` label
- backend availability status
- `Back to Workspace`
- `Switch To Global Study`

Rules:

- global Study should remain visually unchanged when no workspace scope is active
- workspace scope should be visually obvious at all times
- local unavailable state should be represented as a true UI state, not just a warning toast
- the scoped header actions have distinct behavior:
  - `Back to Workspace` returns to Notes workspace `DETAILS`
  - `Switch To Global Study` stays in Study but clears workspace scope and returns to ordinary global Study
- workspace-scoped empty states should be explicit and scope-aware:
  - `No study decks in this workspace yet.`
  - `No quizzes in this workspace yet.`
  - these empty states must not fall back to global Study copy

### 6. Notes Workspace Entry

`tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py` must add a single explicit Study entry action in the workspace details section:

- `Open Study`

`tldw_chatbook/UI/Screens/notes_screen.py` handles that action by asking the app to open Study with:

- `scope_type = WORKSPACE`
- `workspace_id = selected_workspace_id`
- `workspace_name = selected workspace name`

Notes should not retain responsibility for Study scope after handoff.

### 7. Hard Scope Transition Reset

Any Study scope change must clear all active screen-local Study state before new data loads.

Flashcards reset requirements:

- clear selected deck
- clear selected card
- clear current deck/card caches
- end any active review session
- clear current review card
- reset review panel state

Quizzes reset requirements:

- clear selected quiz
- clear selected question
- clear quiz question list
- clear attempt history selection
- clear active attempt id
- clear current attempt questions and answers
- clear attempt status widgets

This reset must happen when:

- entering workspace-scoped Study
- leaving workspace-scoped Study
- changing backend while scoped
- applying a new workspace scope context

### 8. Shared Compat-Layer Filtering

The global-vs-workspace boundary must be enforced in the shared study compat/services layer rather than independently in each controller.

#### Flashcards

Extend `tldw_chatbook/Study_Interop/study_scope_service.py` so deck listing accepts Study scope inputs:

- `scope_type`
- `workspace_id`

Behavior:

- `GLOBAL`
  - return only normalized decks where `workspace_id is None`
- `WORKSPACE`
  - return only normalized decks where `workspace_id == selected_workspace_id`

Because the current shared API client list endpoint does not expose a workspace filter, server-backed deck listing must:

- fetch pages using `limit` and `offset`
- normalize each page
- continue paging until exhaustion
- only then apply scope filtering

The service must fail closed:

- if paging errors partway through, raise an error and return no scoped results for that refresh
- do not display a partial filtered list that might hide or mix records incorrectly

Flashcard card listing itself remains deck-based. Once the deck list is correctly scoped, card lists and review flows naturally stay within that selected deck.

Search behavior in this slice is intentionally narrow and must match the current UI:

- deck lists are scope-filtered before the user selects a deck
- flashcard card search remains selected-deck-only
- workspace-scoped flashcard search only searches cards within decks already admitted by the selected workspace scope
- no new cross-deck search UI is introduced in this pass

#### Quizzes

Extend `tldw_chatbook/Study_Interop/quiz_scope_service.py` the same way:

- add `scope_type`
- add `workspace_id`
- page through server results before filtering
- enforce:
  - global only: `workspace_id is None`
  - workspace only: `workspace_id == selected_workspace_id`

Controllers should never filter raw quiz results themselves.

Quiz list behavior is also intentionally narrow:

- quiz lists are scope-filtered before selection
- no new quiz search UI is introduced in this pass
- controllers should only consume already-scoped quiz lists

### 9. Workspace Scope Availability Enforcement

Workspace-scoped Study in local mode must be unavailable by rule, not by suggestion.

Required behavior in workspace scope + local backend:

- do not display global local decks/quizzes
- do not create decks/quizzes with `workspace_id=None`
- do not start flashcard review
- do not start quiz attempts
- do not allow deck/card/question/quiz mutation actions

The controllers should disable actions proactively, and the compat/service path should also fail closed if those actions are called anyway.

### 10. Scoped Create Behavior

Create behavior is fixed:

- global Study:
  - server mode create deck with `workspace_id=None`
  - server mode create quiz with `workspace_id=None`
  - local mode unchanged
- workspace Study:
  - server mode create deck with `workspace_id=selected_workspace_id`
  - server mode create quiz with `workspace_id=selected_workspace_id`
  - local mode unavailable

The user does not manually choose `workspace_id` in this slice. The active Study scope determines it.

### 11. Error Handling

Error handling must preserve scope correctness over convenience.

Rules:

- invalid workspace entry:
  - no `workspace_id`
  - enter scoped error state
  - do not fall back to global Study
- workspace scope + local backend:
  - enter unavailable state
  - do not call local global-study list/create actions
- server load failure in workspace scope:
  - remain in workspace Study
  - show scoped error state
  - do not replace with global results
- paging failure during scoped filtering:
  - fail the refresh
  - surface an error
  - do not show partial data

### 12. Testing Strategy

Add coverage at four layers.

#### Study interop tests

Extend:

- `Tests/Study_Interop/test_study_scope_service.py`
- `Tests/Study_Interop/test_quiz_scope_service.py`

Required cases:

- global study returns only records with `workspace_id is None`
- workspace study returns only records with matching `workspace_id`
- client-side scoped paging continues past the first page
- paging failure fails closed
- local backend + workspace scope does not return global local records

#### Study controller and screen tests

Extend:

- `Tests/UI/test_study_flashcards_screen.py`
- `Tests/UI/test_study_quizzes_screen.py`

Required cases:

- applying workspace scope renders the Study scope header
- local workspace scope disables create/delete/review/attempt controls
- server workspace scope threads `workspace_id` into deck creation
- server workspace scope threads `workspace_id` into quiz creation
- scope transitions reset flashcard selection/review state
- scope transitions reset quiz selection/attempt state
- leaving workspace scope restores ordinary global Study behavior

#### Notes entry tests

Extend:

- `Tests/UI/test_notes_screen.py`

Required cases:

- workspace details render `Open Study`
- clicking `Open Study` asks the app to open Study with workspace scope context
- `Back to Workspace` from Study returns to workspace `DETAILS`

#### Regression tests

Required guarantees:

- global Study in local mode still behaves as before
- workspace-owned decks/quizzes never appear in the general Study lists
- general/global decks/quizzes never appear in workspace Study

## File-Level Impact

Likely files to modify:

- `tldw_chatbook/app.py`
- `tldw_chatbook/UI/Screens/study_screen.py`
- `tldw_chatbook/UI/Screens/study_scope_models.py` (new)
- `tldw_chatbook/UI/Study_Window.py`
- `tldw_chatbook/UI/Study_Modules/flashcards_handler.py`
- `tldw_chatbook/UI/Study_Modules/quizzes_handler.py`
- `tldw_chatbook/Study_Interop/study_scope_service.py`
- `tldw_chatbook/Study_Interop/quiz_scope_service.py`
- `tldw_chatbook/UI/Screens/notes_screen.py`
- `tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py`
- `Tests/Study_Interop/test_study_scope_service.py`
- `Tests/Study_Interop/test_quiz_scope_service.py`
- `Tests/UI/test_study_flashcards_screen.py`
- `Tests/UI/test_study_quizzes_screen.py`
- `Tests/UI/test_notes_screen.py`
- parity docs after the vertical lands

## Success Criteria

This vertical is complete when all of the following are true:

- global Study shows only global/user-space decks and quizzes
- workspace-scoped Study shows only the selected workspace’s decks and quizzes
- Notes can open scoped Study through an app-owned navigation seam
- scoped Study is clearly labeled and can return to workspace details
- workspace-scoped Study is unavailable in local mode without leaking into global local study
- scope transitions reset cached selection and active review/attempt state
- the shared compat layer, not ad hoc controller filtering, enforces non-mixed Study visibility
