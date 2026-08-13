# Console Decomposition Wave 6 Design

**Status:** approved in conversation 2026-08-13; written review pending

## Problem

The one-way Console size ratchet is genuinely red on current `origin/dev`:

- `tldw_chatbook/UI/Screens/chat_screen.py`: **22,204 lines** versus a **17,727**-line ceiling
- `ChatScreen`: **699 methods** versus a **593**-method ceiling

The overage is 4,477 lines and 106 methods. Raising either ceiling is forbidden by
`Tests/Architecture/test_screen_size_ratchet.py`; it would erase the protection that
the earlier decomposition waves established. This wave must earn a reduction by moving
coherent ownership out of the screen.

The growth is concentrated in three feature families added after wave 4:

| Candidate family | Name-based measurement | Intended owner |
|---|---:|---|
| Image generation and MiniMax H3 image edits | 1,261 lines / 33 methods | `ConsoleImageController` |
| Generated-video lifecycle, publication, playback, save and regeneration | 1,303 / 34 | `ConsoleVideoController` |
| Conversation browser plus retrieval/RAG scope and execution | 2,286 / 81 before false-positive removal | `ConsoleConversationBrowserController` and `ConsoleRetrievalController` |

The raw candidate total is 4,850 lines and 148 methods. Not every matched method may
move: Textual handlers resolved by name stay defined on the screen, DOM work stays with
the screen or a region widget, and false-positive name matches are excluded after source
inspection. The implementation plan must therefore retain margin: if the honest movable
set does not clear 4,477/106 after delegation overhead, the next coherent cluster is
chosen by the same ownership rule rather than compressing code or raising the ratchet.

## Existing Architecture

This design is revision work under the already-approved screen decomposition contract:

- `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`
- `DESIGN.md` section 7
- `tldw_chatbook/UI/Console_Modules/wiring.py`

The existing rules remain binding:

1. A region widget owns pixels; a controller owns non-DOM state and behaviour.
2. `action_*`, `@on(...)`, and other Textual entry points stay defined on `ChatScreen`.
   Their bodies may become short, mutation-tested delegations.
3. Controllers do not use `query_one`; screen/region code passes data or named
   operations through explicit callables.
4. Dependencies are named, keyword-only constructor arguments wired as late-binding
   callables in `UI/Console_Modules/wiring.py`.
5. Cross-controller traffic uses those named callables, never a controller reaching
   through the screen to a sibling controller.
6. State formerly assignable on the screen retains read/write proxy compatibility.
7. Existing worker group names, cancellation ownership, persistence ordering, and
   remount/shutdown behaviour are preserved.

No new ADR is required. ADR-like architecture is already canonical in the approved
decomposition spec and `DESIGN.md`; this wave applies it.

## Options Considered

### 1. Strict controller/region extraction — chosen

Move each coherent non-DOM feature family into one focused controller, keep Textual and
DOM boundaries on the screen, and wire dependencies explicitly. This removes actual
responsibilities and methods from `ChatScreen`, makes future image/video/RAG work land in
the owning modules, and preserves the ratchet's meaning.

### 2. Mixin relocation — rejected

Moving methods into base classes would reduce the measured file but leave `ChatScreen`
with implicit inherited responsibilities and hidden dependency access. It would satisfy
the counter while defeating the ownership goal.

### 3. Raise or reset the ratchet — rejected

The current ceiling is intentionally one-way. Updating it upward would convert a hard
architecture failure into accepted growth and contradict TASK-3070's acceptance criteria.

## Component Boundaries

### `ConsoleImageController`

Owns the non-DOM image and H3 lifecycle:

- transcript image-spec construction and remote-image extension/fetch
- generation-card image projection and image request preparation
- pending attachment resolution and image-generation session/message ownership
- H3 reference snapshots, registry lookup, completion reconciliation, failure merge,
  current-screen settlement, and registry-owned operation execution
- generate-image command orchestration after the screen entry point has supplied any DOM
  input

The screen keeps:

- `action_paste_clipboard_image`, because Textual resolves it by name
- clipboard acquisition and any composer/widget mutation that directly touches DOM;
  these become either short screen helpers or data passed into the controller
- message-image save entry points required by external callers, as delegations when the
  implementation is controller-owned

The controller consumes explicit callables for session/store access, transcript/store
sync, attachment snapshots, generation workers, status/control refresh, and app-owned H3
registry access. It never queries the DOM.

### `ConsoleVideoController`

Owns generated-video lifecycle and publication:

- per-session cancellation/in-flight state and video-store resolution
- generation-card spec construction and stable storage identifiers
- pending artifact ownership, publication gates, shielded task execution and drain
- external-copy validation, precommit checks, publication, retry and save resolution
- generation outcome persistence, playback/copy/regeneration orchestration, and stream
  command preparation

Pure path/stat/open helpers move with this controller. The screen keeps named Textual
entry points and any modal/picker interaction that must touch the mounted app; those
handlers pass immutable choices to the controller. The exact same cancellation event,
worker group, storage identity, commit-before-cleanup order, and remount/shutdown drains
must survive the move.

### `ConsoleConversationBrowserController`

Owns the browser's non-DOM data/state pipeline:

- persisted, membership and native row acquisition
- row filtering, identity, starring and merge rules
- query token/timer/results/error state
- background search computation and post-selection refresh
- collapse/config state projection

The existing `on_console_workspace_conversation_search_changed` handler stays on the
screen and delegates after reading the input event. Focus operations and row rendering
remain screen/region responsibilities. Existing workspace/session/message controllers
are reached only through named callables wired at construction.

### `ConsoleRetrievalController`

Owns retrieval/RAG policy and execution that has no DOM:

- staged-RAG capture and effective-scope resolution/cache warming
- scope read/write, clear/save policy and picker input/output preparation
- library-RAG request scope, execution, outcome application and degraded-state policy
- auto-retrieve-on-send decisions and placeholder lifecycle
- RAG source status, active dictionary/world-book scope derivation and summary inputs

The screen keeps `@on`/`@work` entry points where Textual resolves them and any direct
widget synchronization. Those become narrow delegates. Picker/modal presentation stays
on the screen; the controller owns validation and state transitions around the returned
choice.

Conversation-browser and retrieval code are separate controllers because one owns
conversation inventory/search state while the other owns retrieval policy and RAG
execution. Combining them would create a 2,000-line generic "browser/RAG" bucket and
hide their distinct invariants.

## Data and Control Flow

Each extraction follows the same shape:

1. A Textual handler or command remains on `ChatScreen` when framework name resolution
   requires it.
2. The handler reads event/DOM values and passes plain values to its controller.
3. The controller performs state transitions, persistence and asynchronous orchestration
   using explicit dependencies.
4. The controller returns a value or invokes a narrowly named screen callback for the
   required UI refresh.
5. The screen/region applies DOM changes.

Controller construction is added to `build_console_controllers`. Construction order is
not semantically load-bearing: every sibling dependency is late-bound. Controllers may
retain stable app-owned service identities only when the existing binding contract
already permits that snapshot and the constructor docstring records why.

## Error, Cancellation and Privacy Contracts

This is a behavior-preserving refactor. Existing public/sanitized error copy and
metadata-only logging remain byte-equivalent unless a test proves that formatting must
change solely because the owner module name changed.

For image and video operations, cancellation is an ownership contract, not an
implementation detail. The same event instance must flow from screen action through
registry/task/worker/adapter; late outcomes must reconcile only onto the current matching
screen/session/generation. Drains remain bounded and app shutdown still owns definitive
cleanup.

No attachment bytes, prompts, paths, message/session IDs, signed URLs, provider payloads
or exception messages may be added to persistent diagnostics by the move.

## Testing Strategy

Every controller is characterized before extraction and moved in its own commit.

### Cross-cutting architecture tests

- controller modules contain no `query_one` or direct sibling-controller reach-through
- screen entry points remain present with their decorators/binding names
- worker group names and cancellation events are unchanged
- baseline assignable screen attributes retain read/write proxies
- import/re-export consumers in tests are repointed to defining modules before obsolete
  screen imports are removed
- AST ownership inventory proves moved methods are gone from `ChatScreen` except for
  documented short delegations with real callers

### Image/H3 evidence

- generation actions/cards and mounted Console image tests
- H3 lifecycle, cancellation, fresh-screen/remount and attachment-stash tests
- mutation checks removing generation/session gates or current-screen settlement

### Video evidence

- Console generate/play/save/regenerate/stream tests
- video-store, capacity, cancellation, publication and remount tests
- exact container/storage identity and external-copy failure contracts

### Browser/retrieval evidence

- conversation browser search, grouping, selection, collapse and persistence tests
- retrieval-scope picker, RAG scope, library RAG and auto-retrieve tests
- real SQLite-backed scope/session tests where existing suites provide them

### Final gates

After the final rebase, measure actual lines/methods. The ratchet is lowered to the exact
earned numbers; it is never raised. Run the complete relevant suites plus all
architecture tests, Ruff, formatter, py_compile, privacy/diff checks, and the repository
full suite required by the project DoD.

## Delivery Sequence

1. Commit characterization and ownership-map tests.
2. Extract `ConsoleImageController`; verify and commit.
3. Extract `ConsoleVideoController`; verify and commit.
4. Extract `ConsoleConversationBrowserController`; verify and commit.
5. Extract `ConsoleRetrievalController`; verify and commit.
6. Rebase, measure, lower the ratchet, update the canonical decomposition progress and
   TASK-3070 notes, then run final verification.

If any extraction's honest boundary yields too little reduction, stop and revise the
plan before choosing another cluster. Do not widen a controller merely to chase line
count.

## Success Criteria

- The current screen-size and method-count ratchet passes without increasing either
  budget.
- `ChatScreen` loses at least 4,477 lines and 106 methods relative to the final rebased
  starting measurement, accounting for delegation overhead.
- Image/H3, video, conversation-browser and retrieval/RAG behavior is unchanged.
- New work in these families has an obvious owner under `UI/Console_Modules/`.
- All automated, static, privacy, cancellation, persistence and lifecycle gates pass.
