# Library Compose-Once-Per-Visit Design

**Task:** TASK-15459
**Date:** 2026-08-13
**Status:** Approved for implementation planning

## Context

Every navigation to Library constructs a fresh `LibraryScreen`. On a warm visit,
the first `compose_content()` currently runs before the app-scoped source snapshot
cache is applied. `on_mount()` then applies the cache and explicitly requests a
whole-screen recompose so the already-composed loading state becomes visible data.
The fresh source reconciliation can request another whole-screen recompose.
Route-specific work started automatically on mount can add more: Prompt browse,
Collections, Skills trust posture, restored media detail, and pending-source
opening still include whole-screen refresh paths. This turns one destination
visit into a chain of complete compositions and can replace focusable widgets
after the user has started interacting.

TASK-15457 established rail and canvas-owned `sync_state()` seams. TASK-15458
established persistent media-viewer children. TASK-15459 must use those ownership
boundaries to make the first composition authoritative and keep later source
reconciliation below the screen boundary.

## Goals

1. During the automatic entry lifecycle, each Library visit calls
   `LibraryScreen.compose_content()` exactly once, including visits restored or
   deep-linked to a non-default surface.
2. A valid warm cache is present in screen state before the first composition.
3. Cold, expired-cache, timeout, fresh-reconciliation, and automatically started
   entry-worker paths perform zero whole-screen recomposes.
4. Fresh data updates the rail and only the active surface that owns the changed
   snapshot-derived presentation.
5. Unchanged fresh snapshots update cache freshness without touching the DOM when
   the current generation is already rendered cleanly.
6. Targeted updates preserve active-canvas identity, keyboard focus, selection,
   and scroll where the corresponding item or control still exists.
7. Cached-then-fresh behavior, error states, and stale-result protection remain
   intact.

## Non-goals

- Removing every remaining `refresh(recompose=True)` call from Library. TASK-15459
  governs the complete destination-entry lifecycle, including work started
  automatically because of restored or deep-linked entry state. User-initiated
  interactions remain out of scope even if an entry worker is still settling;
  those interactions supersede the old route owner, and every later entry result
  remains subject to the strict targeted-update guard.
- Changing source queries, page sizes, cache TTL, persistence, schemas, or service
  boundaries.
- Redesigning Library layout or copy.
- Replacing the existing screen-per-visit navigation policy.

## Decision

Use **pre-compose cache seeding plus a strict entry-reconciliation router**.

The cache is validated, cloned, and applied at the end of `LibraryScreen.__init__`,
after every snapshot-dependent field has been initialized. The app subsequently
calls `restore_state()` before mount, so saved navigation, selection, filter, and
editor state overlay the cached source data before the first composition.

`on_mount()` no longer reads or applies the cache and never asks the screen to
recompose. It arms the existing timeout and starts the fresh snapshot worker and
any route-specific entry work. When entry data arrives, state mutation and
mounted presentation reconciliation are separate operations. Presentation is
routed to the rail and the active owner. There is no whole-screen fallback on
this lifecycle path.

For this design, a whole-screen recompose request means either
`LibraryScreen.refresh(recompose=True)` or `LibraryScreen.recompose()`. Neither
API may be invoked by destination construction, mount, the source timeout,
snapshot reconciliation, or work automatically launched because of the entry
route. Canvas-owned `sync_state()` may recompose only that retained canvas.

## Entry Lifecycle Boundary

The entry lifecycle begins when a new `LibraryScreen` is constructed and ends
when all work automatically required by its initial restored/navigation state has
either settled or been superseded. It includes:

- cached source seeding and the fresh source snapshot;
- Prompt list browse loading and its terminal result;
- Collections snapshot loading;
- Skills snapshot and trust-posture loading;
- restored or deep-linked note and media detail loading;
- restored Export counts; and
- a pending source-open request supplied by navigation context.

These jobs keep their existing service and cancellation ownership. TASK-15459
changes only how their landed state is projected into the mounted Library. A
user action, including a rail switch, filter, edit, retry, mutation, or explicit
refresh, is not part of this boundary. Such an action can supersede an entry
owner, but cannot make a later entry-worker result eligible for a whole-screen
recompose.

## Cache Contract

### Validation

The cache seed helper accepts a snapshot only when:

- the app exposes both a snapshot and a numeric monotonic timestamp;
- the timestamp age is non-negative and below
  `LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS`;
- the snapshot has the expected six-field outer shape; and
- its known source entries have the shapes consumed by the current builders:
  record tuples for Notes, Media, and Conversations, `(count, ())` for Prompts,
  and `(count, context_payload)` for Skills.

A missing, expired, or malformed cache is ignored. Cache validation cannot prevent
screen construction or navigation.

### Copy-on-read and copy-on-write

Cache reads and writes use one schema-aware clone helper. The new screen must
never alias mutable state back into the app cache. The helper copies every known
mutable container reachable through the snapshot schema: outer mappings, record
mappings inside tuples, the Skills context mapping and its available/blocked
lists and record mappings, counts, total-known flags, study counts, and mutable
recovery payloads. Immutable scalar values may be reused.

This closes the current gap without assuming that a tuple makes the mappings or
lists nested inside it immutable. A repeat visit must not be able to corrupt the
cache used by a third visit.

### Ordering

Cache seeding runs only after constructor initialization is complete. It does not
run inside `compose_content()`, because composition must remain a read-only
projection of settled state. `restore_state()` runs afterward and remains the
owner of saved selection and view state.

## State, Rendered Generation, and Dirtiness

`_apply_local_source_snapshot()` remains the source-snapshot state-mutation seam.
It:

1. normalizes/copies the incoming snapshot;
2. carries a selected out-of-page conversation forward as today;
3. records whether snapshot-derived presentation actually changed;
4. assigns the new records, counts, error/recovery state, study counts, and loaded
   flag;
5. invalidates workspace-depth derived state; and
6. increments a monotonically increasing state generation only for a changed
   presentation.

Fresh successful results refresh the app cache and timestamp even when their
presentation equals the current state. Equality is evaluated against normalized
snapshot values before assignment.

The screen separately tracks the generation last rendered successfully and a
reconciliation-dirty flag. An equal result schedules no rail, canvas, or layout
work only when the current generation is already rendered and not dirty. If a
targeted update failed or was interrupted after state mutation, an equal later
result retries reconciliation until the rendered generation catches the state
generation. This makes the recovery promise compatible with the unchanged-data
fast path.

Every deferred mounted update captures the generation and entry-route identity it
represents. Before touching the DOM it verifies both are still current. This
prevents a timeout update, an immediate worker completion, a superseded entry
worker, or an older retry from rendering over newer state.

## Mount Timing

An entry worker can finish while the Mount message is still being processed. If
changed state arrives before the screen is attached, reconciliation is queued once
for the next screen message-pump turn. The callback rechecks attachment,
generation, and entry-route identity before routing the update. It does not depend
on a whole-screen refresh being scheduled.

Only one reconciliation is pending for a generation and owner. A newer
generation or route supersedes it rather than adding another callback.

## Strict Reconciliation Router

The router first rebuilds normalized Library shell state, then synchronizes the
mounted `LibraryRail`. Rail synchronization also refreshes snapshot-dependent
workspace details. The destination header is patched only if its rendered line
changed. The router returns a typed result: applied, already-current, superseded,
or failed. It never calls either whole-screen recompose API.

The active surface is then handled according to ownership:

| Active surface | Snapshot reconciliation |
| --- | --- |
| Landing hub | Patch counts and reconcile recent-source rows inside a retained landing owner. Preserve the three action buttons. |
| Conversations list | Call `LibraryConversationsCanvas.sync_state()` on the retained canvas. |
| Media list | Call `LibraryMediaCanvas.sync_state()` on the retained canvas. |
| Database Notes list | Call `LibraryNotesCanvas.sync_state()` on the retained canvas. |
| Skills list | Call the existing Skills canvas sync with the fresh skills snapshot/trust state. |
| Search/RAG | Use the existing narrow scope-count and Run-gate synchronizer. Do not rebuild query, results, history, or answer regions. |
| Study/Flashcards/Quizzes handoff | Reconcile carried-source and readiness rows inside a retained handoff owner. Preserve its Open button. |
| Media viewer, Media Trash, note editor, prompt editor/list, skill editor, File Notes, Collections, Ingest, Export | Update the rail only for source-snapshot changes. Route-specific entry results synchronize their own retained owner and do not remount unrelated fields or controls. |

Prompts browse rows come from their browse controller, not the source snapshot.
Collections, Ingest, Media Trash, and Export have separate data owners. A source
snapshot must not overwrite their in-progress forms, selections, receipts, or
workers.

### Route-specific entry results

- Prompt browse loading/results call the retained Prompts list canvas's
  `sync_state()` and restore semantic focus through that canvas's
  post-recompose callback.
- Collections replaces or synchronizes only `#library-collections-panel` inside
  the retained canvas host.
- Skills trust posture calls the retained Skills canvas's `sync_state()`.
- Note detail continues through the retained Notes canvas sync seam.
- Media detail replaces only the active child inside `#library-canvas`; the
  Library screen, rail, header, footer, host, and unrelated state retain identity.
- Export counts continue to patch their always-mounted fields in place.
- A pending source open uses the same narrow route/content replacement seam as a
  settled entry rather than the legacy screen-level navigation fallback.

### Loading and error transitions

On a cold or expired-cache visit, the single initial composition can contain the
existing loading or error placeholder for Conversations, Media, or Database Notes.
When data changes that structural state, the router replaces children only inside
`#library-canvas`. It constructs the appropriate active canvas/viewer/editor state
through the same builders used by initial composition. The screen shell, rail,
header, footer, and canvas host retain identity.

The structural replacement helper must not duplicate conditional rendering logic.
Initial composition and replacement share one focused builder for snapshot-owned
canvas content.

## Focus, Selection, and Scroll

Before any canvas-owned recompose, the router records a shared semantic focus
identity:

- stable widget id for filters and action buttons;
- source record id for row buttons; and
- the active canvas's selection and scroll identity where available.

After the scoped update settles, focus is restored only when:

- the screen and generation are still current;
- the user has not moved focus elsewhere during the update; and
- the same semantic target remains present and enabled.

Selection state continues to come from screen-owned row-selection and selected-id
fields. The strict router uses the existing `PostRecomposeCallback` seam for
Conversations, Media, Notes, Prompts, and Skills, extending TASK-15457's
Notes-specific default into one shared entry-reconciliation contract. Canvas
specializations may add richer selection/scroll details, but none may silently
delegate entry focus preservation to a whole-screen recompose. A structural
loading/error replacement has no outgoing interactive child to restore.

## Failure Behavior

The general `_sync_library_canvas()` helper currently falls back to
`screen.refresh(recompose=True)` when a target is missing. That fallback remains
available for unrelated legacy interaction callers, but the complete entry
lifecycle invokes strict behavior that forbids it.

A missing target caused by an ordinary route race is treated as superseded work.
For a still-current route, the router performs one bounded next-turn targeted
retry. If the target is still unavailable, it marks the current generation dirty,
logs the surface, generation, and exception category, and stops. A later equal or
changed result can repair the surface because dirty state bypasses the equality
no-op gate. It never performs a whole-screen recompose and never loops within one
reconciliation attempt.

Malformed cache data is ignored. Fresh lookup errors continue to render the
existing error copy through the same targeted route. The timeout and fresh worker
share the generation guard, so a late timeout cannot replace a successful result.

## Testing Strategy

All behavior changes follow test-driven development. Each regression is observed
red against the pre-change implementation before product code is written.

### Lifecycle and cache tests

- Warm repeat visit invokes `compose_content()` exactly once and its first rendered
  frame contains cached counts/rows.
- Cold and expired-cache visits invoke `compose_content()` exactly once; fresh data
  replaces only the canvas placeholder.
- Malformed cache data is ignored without blocking navigation.
- Cache read containers do not alias live screen containers.
- A third visit is unaffected by mutations performed on the second visit.
- `restore_state()` selection/filter values overlay cached data before first paint.

### Reconciliation tests

- Equal fresh data updates cache freshness and performs zero DOM synchronization.
- Equal fresh data retries targeted reconciliation when the rendered generation
  is dirty, then returns to the zero-DOM fast path after success.
- Changed fresh data preserves screen, rail, canvas-host, and active-canvas identity.
- Landing counts/recents and handoff readiness update without replacing their
  persistent action buttons.
- Search/RAG preserves query, result, history, answer, focus, and scroll state.
- Editors, viewers, File Notes, Collections, Ingest, Trash, and Export are not
  remounted by source reconciliation.
- A focused filter/action/row is restored after a required scoped canvas update
  when its semantic target remains.

### Route-entry worker tests

- Restored/deep-linked Prompts, Collections, Skills, note editor, media viewer,
  Export, and pending-source visits each call `compose_content()` once.
- Prompt loading/result, Collections load, Skills trust posture, note/media
  detail, Export counts, and pending-source completion update only their retained
  owner.
- Entry-worker completion after a route change is discarded.

### Race and failure tests

- Immediate worker completion during mount reconciles after mount rather than being
  lost.
- Fresh success supersedes a pending timeout update.
- Two changed generations render only the newest generation.
- Missing-widget and route-switch races perform at most one targeted retry per
  attempt and zero whole-screen recomposes.
- Spies on `LibraryScreen.refresh`, `LibraryScreen.recompose`, and
  `compose_content` prove that no entry lifecycle path requests a whole-screen
  recompose after the initial composition.

### Verification and UAT

- Run the focused Library lifecycle, canvas-sync, navigation, media-viewer, and
  snapshot-cache suites using the parent `dev` virtual environment.
- Run scoped Ruff and `git diff --check`.
- Record warm-visit latency before and after with the same seeded snapshot and
  terminal size, reporting median and sample count.
- Render UAT at the supported compact and wide Library sizes. Confirm no loading
  flash on a valid warm cache, no focus jump during fresh reconciliation, honest
  loading/error behavior without a cache, and stable screen/rail/canvas identity.
- Run rendered UAT with an isolated `TLDW_CONFIG_PATH` and data directory, and
  fingerprint the real profile before and after, so verification cannot mutate
  the user's Library.
- Attribute unrelated Windows collection failures against `origin/dev` rather than
  changing out-of-scope platform modules.

## Acceptance Mapping

- **AC1:** compose-count spies across default/restored/deep-linked entry routes,
  retained identity assertions, strict no-screen-recompose failure tests, and
  changed/unchanged reconciliation coverage.
- **AC2:** cache validation, ordering, copy isolation, TTL, restored-state overlay,
  and cached-then-fresh mounted tests.
- **AC3:** identical before/after warm-visit benchmark plus rendered compact/wide
  UAT.
- **AC4:** both whole-screen recompose APIs are forbidden throughout cache,
  timeout, fresh, retry, and route-entry worker paths; semantic focus/scroll
  tests cover every snapshot-owned retained canvas.

## ADR Check

**ADR required:** no
**ADR path:** N/A
**Reason:** This design applies the existing Library screen/rail/canvas ownership,
cache TTL, and targeted-update contracts. It changes no storage, schema, service,
security, dependency, or cross-module runtime boundary.
