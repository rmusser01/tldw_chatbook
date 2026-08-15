# Library Top-Level Source Pagination Design

**Date:** 2026-08-14
**Status:** Approved; TASK-16481 Conversation foundation is complete and in closeout review; TASK-16482–TASK-16488 remain planned
**Scope:** Library top-level flat browse sources

## Context

The Library uses “media” in the broad product sense: Conversations, Media
items, Notes, Prompts, Skills, Collections, and related saved source types.
Several top-level browse canvases currently render only a capped snapshot or a
complete fetched list. A user with more records than the canvas can display may
therefore be unable to reach everything, or may pay the cost of mounting a
large number of Textual widgets at once.

Conversations already have service-backed paging, but their page size and
presentation are not yet a complete Library-wide convention. Prompts have an
exact browse controller but currently use 50-row pages. Media renders at most
the snapshot-fed rows and applies a second display cap. Skills and Collections
render their fetched lists without a shared 20-row presentation limit.

The feature will make flat top-level Library sources consistently reachable in
20-item pages while preserving each source's existing service, selection,
trust, mutation, and detail ownership.

## Goals

1. Make every record in the in-scope top-level sources reachable regardless of
   terminal height or source size.
2. Use a consistent 20-item page contract with Previous and Next controls.
3. Apply existing filters, sorts, and type choices before paging, against the
   complete source rather than the currently visible page.
4. Keep list scrolling independent from a pager that remains visible inside
   the owning list pane.
5. Preserve exact totals, stable deterministic order, stale-request safety,
   focus, deep links, and recoverable errors.
6. Avoid a generic Library data controller or new application-level state
   owner.

## Non-goals

- Notes folder-tree paging.
- Media Trash paging.
- Paging Collection members/content.
- Adding new text-search fields to Media or Collections.
- Direct page-number entry, infinite scrolling, or virtualized lists.
- Freezing an entire browse session into a multi-page database snapshot.
- Changing source authority, storage schema, sync behavior, or runtime mode.

The excluded hierarchical and nested surfaces will be captured as three atomic
follow-up Backlog tasks, one each for Notes, Media Trash, and Collection
members. They may share a workstream, but not one implementation task or PR.

## Scope by Source

| Source | Top-level tranche | Data owner | Required change |
| --- | --- | --- | --- |
| Conversations | Yes | Existing conversation page loader | Retain behavior; align shared 20-row pager presentation and hardening |
| Prompts | Yes | Existing prompt browse controller | Change page size from 50 to 20; align pager presentation and hardening |
| Media items | Yes | Media reading scope/local service | Add authoritative 20-row page state, true database-level offset paging, exact filtered total, and complete type facets |
| Skills | Yes | Skills scope/local service | Filter and sort before 20-row slicing; retain trust metadata and exact total |
| Collections | Yes | Library Collections service | Use the existing exact-total `list_library_collections` page seam in the UI |
| Notes | Follow-up | Placement-aware folder tree | Preserve the existing bounded tree and Load-more behavior; do not flatten or split folder relationships |
| Media Trash | Follow-up | Media Trash service | Nested recovery surface |
| Collection contents | Follow-up | Collection member-page service | Nested detail surface |

## User Experience Contract

### Page presentation

Every in-scope list displays at most 20 source records. Once a page has applied,
the source title shows the exact total for that applied scope, not the rendered
row count:

```text
Media (87)
...
1–20 of 87 · Page 1 of 5
[Previous]  [Next]
```

The row viewport scrolls independently. The range/page status and controls
remain outside that viewport but inside the list pane, including Media and
Collections split workbenches.

At narrow supported sizes, range/page copy occupies its own centered line and
the two controls use a compact row below it. Empty results retain the same
vertical pager space and render:

```text
0 of 0 · Page 1 of 1
```

Before any page has applied, the UI does not invent a zero total. Initial load
uses `Loading page 1…`; initial failure uses `No page loaded · Total
unavailable` with the source-owned Try again action. A title without a count is
used until an exact envelope applies. These unknown/not-loaded states are
distinct from a successfully loaded empty page.

Previous is disabled on page 1. Next is disabled on the final page. Disabled
labels use the repository's non-colour marker convention and a readable reason:

- Already on the first page.
- No more results.
- Page is loading.

The reason appears in visible pager status; a tooltip may repeat it but is
never its sole carrier, because disabled controls are not keyboard focusable.

### Focus and keyboard behavior

- A successful page change resets the list viewport to its top.
- After paging, focus returns to the invoking Previous or Next control while it
  remains enabled. If the result disables it at the first/final boundary,
  focus moves to the opposite enabled pager control. If neither is enabled,
  focus returns to the source filter/chooser when present, otherwise the list.
  The handler captures this focus intent before publishing the loading state,
  because disabling/recomposing the invoking Button may move Textual focus.
- Filter submission retains filter focus after the targeted canvas sync.
- Each source preserves its existing header/filter/action tab order. The pager
  follows the rendered rows, and disabled controls are skipped according to
  Textual behavior; paging does not reorder unrelated toolbar actions.
- Opening a detail/editor and returning preserves the applied page.
- Switching Library categories preserves each category's applied scope during
  the current Library visit.

### Filters, sorts, and type choices

Existing filters, sorts, and Media type choices apply to the complete source
before paging. Any successful scope change starts on page 1. Media type options
come from a complete distinct-type service result, never the current page.

The implementation maintains separate requested and applied scopes:

- The requested scope is the page/filter/type/sort the user most recently
  asked for.
- The applied scope is the scope that produced the visible rows.

Titles, ranges, and active-result copy always describe the applied scope. If a
new filter/type/sort fails, the previous rows remain visible and the canvas says
that the requested change was not applied. The input retains the requested text
so the user can correct or retry it.

### Selection and bulk actions

Conversations and Media retain their current-page row-selection model. “Select
all” names the rendered count, for example `Select all 20 shown`. Paging,
filtering, sorting, or changing Media type exits their Select mode, clears the
selection, and publishes a short `Selection cleared.` notice. This prevents
invisible Conversation/Media selections from carrying into export or
destructive actions.

Prompts deliberately retain their existing immutable cross-page selection
basket and captured versions. Paging does not clear it; the canvas continues to
show `N selected · M on this page`, `Select page` adds only the current page,
and `Clear all`/`Done` remain the explicit exits. Filter, sort, and Prompt
collection changes neither clear nor add entries: the captured basket and
versions remain unchanged. Existing version/conflict checks still govern bulk
actions.

Paging does not introduce Select mode or new bulk actions on sources that do
not already own them.

### Loading, errors, and retry

The last good page remains visible during a page-only request. Controls are
temporarily disabled and the status says which page is loading.

On a page-only failure, the current page remains applied and the canvas reports,
for example, `Couldn’t load page 3.` On a scope-change failure, the current page
remains applied and the canvas reports `Filter wasn’t applied; showing previous
results.`

Every recoverable failure mounts a source-owned `Try again` action that retries
the requested scope. This is required for first-page failures where both pager
buttons are necessarily disabled. Service-unavailable and malformed-response
errors stay inside the owning canvas and never replace the entire Library shell.

## Architecture

### Ownership

Each source retains its existing query, sort, type, selection, request,
mutation, and detail owner. There is no polymorphic Library page controller and
no root application paging state.

A small shared pure function derives immutable pager display values. Existing
source canvases render those values with their own widget IDs; there is no
generic pager widget/composer. The helper owns no workers, source state,
widgets, or messages. Source-specific screen handlers continue to own Previous,
Next, and Retry events.

The broad Library snapshot remains available for rail totals, landing content,
RAG scope preparation, and other existing consumers. A paged canvas uses its
dedicated page state as the authoritative list and cannot be overwritten by a
later broad-snapshot refresh.

### Page metadata

Each in-scope source exposes equivalent immutable display metadata without
sharing a generic data loader:

- Applied page and requested page. Applied page is absent until the first
  successful envelope.
- Fixed page size of 20.
- Exact applied total when freshness is `fresh`; absent whenever freshness is
  `uninitialized` or `stale`.
- Page freshness: `uninitialized`, `fresh`, or `stale`.
- Source-owned stale reason/copy, distinguishing a committed mutation whose
  refresh failed from a source that changed again during automatic clamping.
- Applied and requested filter/type/sort scope.
- Last good page records.
- Loading and recoverable-error copy.
- Monotonic request generation.
- Selection-cleared notice when applicable.

The shared display calculation derives:

- Start/end range.
- Total pages, with the empty state represented as page 1 of 1.
- Unknown/not-loaded and source-specific stale copy without fabricated ranges.
- Previous/Next disabled states and reasons.
- Loading and retry presentation.

### Request flow

1. A filter/type/sort submission or pager action records a requested scope,
   increments the source generation, applies the source-specific selection
   policy above, and publishes loading state through a targeted canvas sync.
2. The source dispatches its existing or new source-specific read through a
   Textual worker. Synchronous local storage calls run off the UI loop.
3. Exclusive worker groups cancel obsolete presentation coroutines. They do not
   claim to stop an already-running `to_thread` call.
4. Ordinary browse limit/offset envelopes must echo the exact requested
   coordinates. The stable-ID locator envelopes used only by Conversation
   deep links and Collection mutation placement instead return their resolved,
   page-aligned offset/page plus the target's zero-based rank. A locator
   validates that the target ID is present at `rank - offset`, that
   `offset = (rank // page_size) * page_size`, that the returned page agrees,
   and that total/cardinality/identity rules all hold under the source order. Prompt
   responses retain their existing normalized contract: `per_page` matches
   the requested `page_size`; `current_page` is the deterministic clamp of the
   request against `total_pages` (or page 1 when empty); and its compatibility
   alias `page` must equal `current_page`. The Prompt request fingerprint/
   generation binds the original requested scope. Every envelope
   validates its exact total and exact expected row cardinality: the lesser of
   page size and remaining total for the returned coordinates. Undersized
   non-final pages and oversized pages fail closed. Before application, each
   source also validates that every summary item has its required stable
   identity and that identities are unique within the page; malformed items
   fail closed rather than being silently dropped by a row builder.
5. Only the current generation may apply. Stale results are discarded.
6. A successful result atomically replaces the source's applied scope, records,
   total, and page-selection projection, then performs a targeted canvas sync.
7. Any fresh exact **limit/offset** response proving the requested page is now
   out of range triggers one generation-guarded reload of the last valid page
   before applying. This covers local mutations and external/concurrent
   deletions; an empty source resolves to page 1. A validated Prompt response
   already contains the coherently clamped page rows and applies directly
   without a redundant second service call. At most one automatic limit/offset
   clamp is attempted.
   If the source shrinks again before the follow-up read and that response is
   also out of range, the last good records remain visible but enter the
   existing `stale` presentation: exact total/range copy is suppressed;
   row, bulk, Previous, and Next actions are disabled; and Retry plus
   scope controls remain available. The canvas shows a recoverable `Source
   changed again; try again.` error rather than applying invalid or empty page
   metadata. Only a successful authoritative request clears that stale state.

Source mutations invalidate the source's outstanding read generation before
they commit, and paging/scope controls remain unavailable for the mutation's
existing ownership interval. A read started before the mutation cannot apply
after it. After a successful mutation, the source performs a fresh
location/page read. If that read fails, the mutation remains truthfully
reported as committed, the prior page is marked `List may be out of date`,
exact-total copy is suppressed, and Try again reloads the authoritative page.
The known mutation result is reconciled locally first: a deleted row is
removed, a returned renamed/restored record replaces the matching row when
present, and selection cannot remain on a removed record. Because other rows
and totals may still be stale, row and bulk actions remain disabled until any
successful authoritative request restores a fresh page; viewing the retained
copy is read-only.
Previous and Next are also disabled because no exact boundary is known. Retry
remains enabled, as do filter/sort/type controls that issue a fresh page-1
request; any successful authoritative request clears the stale state.
The UI never reports the mutation as failed merely because its follow-up read
failed, and never presents a stale total as exact.

Lightweight read-only calls may briefly overlap after rapid scope changes. The
generation guard protects correctness; this design does not add application-
wide locks or queues for short storage reads without profiling evidence.

Unmount invalidates every dedicated source request generation and stops any
source debounce/timer owned by that screen. A local thread read already in
progress may finish, but its result cannot touch the unmounted widget tree or
become the snapshot restored by a fresh Library screen.

### Navigation and state restoration

Paging state is destination view state under ADR-033:

- It remains screen-owned and memory-only.
- The screen snapshot stores the last applied scope, not source records.
- Unsubmitted or failed input drafts remain visible only while the current
  screen instance is alive. They are not persisted across a destination
  re-entry, because restoring them without their explanatory failure state
  would silently mismatch the applied rows. Transient loading/error state and
  a requested page that never applied are likewise not restored.
- A fresh Library screen re-fetches the last successful applied scope before
  treating it as applied.
- Restored page values accept an exact non-boolean integer of at least 1 only.
  Checked page-to-offset arithmetic must remain within the service/SQLite
  signed-integer bound before any call is dispatched; invalid, negative,
  boolean, string, or overflowing values normalize to page 1. A valid but now
  out-of-range page clamps after the exact-total response.
- A missing or now-invalid restored page clamps to the latest valid page.
- Explicit navigation context has precedence over restored scope.

For a source with an existing external deep-link route, the link continues to
resolve its target directly by stable source ID; it does not walk pages. Detail
surfaces open directly. A Conversation list target uses the source-specific
page-location behavior below. If there was no prior detail-list context, Back
returns to a freshly validated page 1. Existing in-list detail/back flows
preserve their page. This task does not add new deep-link producers.

## Source-specific Service Design

### Conversations

Retain the existing exact 20-row conversation controller, generation guard,
full-source filter, page clamp, and data owner. Repair the underlying
Conversation DB page read so count and rows use one read transaction with the
same WHERE/order contract, then adapt the canvas to the shared presentation and
requested/applied error contract.

Replace the current off-page deep-link prepend with a bounded
page-containing-ID query under the exact unfiltered Conversation ordering. The
query returns the target's owning 20-row page and exact total from one coherent
read snapshot. Its zero-based target rank, resolved page/offset, page-local
target position, and requested stable ID must agree or the locator fails closed. A conversation
context clears the in-canvas filter, applies that deterministic page, and
selects the target. It does not inject an extra row into page 1 or change that
page's range semantics. An unavailable target retains the existing warning
behavior.

### Prompts

Retain the existing prompt browse controller and exact service contract. Change
the default Library page size from 50 to 20. Preserve debounced search, Enter
flush, source authority, sort/type scope, mutations, and retained-history
ownership. Adapt only pager presentation and the requested/applied recovery
contract.

Prompt request coordinates remain `page`/`page_size`; the normalized response
retains resolved `current_page`, its equal compatibility alias `page`, and
`per_page`. The controller validates `per_page=20`, agreement between the two
resolved-page fields, and the clamp implied by the requested page and exact
total rather than requiring either field to echo an out-of-range request.

### Media

Use the existing Media search/list boundary with exact totals and optional
`media_types`, but repair local offset execution so page N performs true
database-level paging instead of fetching the first `offset + limit` rows and
slicing in memory.

Requirements:

- Filter before count and page.
- Read rows and exact total coherently.
- `LIMIT 20 OFFSET n` or an equivalent exact page request.
- Deterministic selected sort followed by stable Media ID.
- A small local/scope-service distinct-type seam for the complete active Media
  collection.
- No full content, embeddings, binary data, or filesystem paths in page rows.

The existing Media type button opens a bounded, scrollable `OptionList`-style
chooser rather than mounting one Button per distinct type in the canvas row.
The complete facet set remains keyboard reachable, the active type is marked
without relying on colour, Escape returns focus to the opener, and the chooser
does not alter the list until a choice is committed.

### Skills

Use the existing local `list_skills(limit, offset)` envelope, explicitly with
local mode. Extend its local query/sort inputs so filtering and the trust/status
sort occur before slicing. The response total describes the filtered set and
each summary retains trust-blocked metadata.

Library Skill filtering remains literal, case-insensitive matching over Skill
name and description only, matching the current canvas contract. It does not
search body content, supporting files, argument hints, metadata keys, or trust
diagnostics. A body-only match therefore remains absent.

The page envelope also carries source-wide `blocked_total` and the stable name
of the first blocked Skill, computed from the complete classified index rather
than the current page or filter. The trust header uses that total, and Review
opens that stable target directly even when it is off-page. This preserves the
existing global trust-recovery meaning without displaying a false zero on a
page that happens to contain no blocked row.

The filesystem-backed local service may still scan the installed Skill index to
classify trust before slicing. The UI page and mounted widgets are bounded; the
design does not claim a database-level bounded Skill scan.

### Collections

Use `list_library_collections(limit=20, offset=n)`, which already reads rows and
total in one read transaction. Preserve oldest-first ordering and add a stable
Collection ID tie-breaker.

The service also exposes a bounded page-containing-ID query for a stable
Collection ID under exactly the same ordering. This may use a window/rank CTE;
it returns the target's owning 20-row page and exact total from one coherent
read snapshot, never an unbounded record list. Creation, rename, and restore
validate the zero-based target rank, page-aligned resolved location,
page-local target position, and target presence, then apply
that envelope and select the affected ID. This is necessary because
creation timestamps are second-granularity and name ordering within a timestamp
tie means a new or renamed Collection is not guaranteed to remain on the
final/current page. Deletion reloads/clamps the current page. Restore retains
its existing recovery receipt if the record cannot be located; the UI does not
synthesize placement.

## Determinism and Live Data

Pages are live reads, not a frozen multi-page snapshot. Exact totals describe
the transaction that produced each response. Concurrent mutations may shift
later page boundaries, which is an accepted ADR-030 trade-off.

All service orders require a stable final key:

- Media: requested sort, then Media ID.
- Skills: status/name sort, then normalized Skill name.
- Collections: creation time, case-insensitive name, then Collection ID.
- Conversations and Prompts retain their existing deterministic service order.

## Responsive and Accessibility Requirements

- The list viewport uses `VerticalScroll` or the canvas's existing equivalent
  with `height: 1fr` and `min-height: 0`.
- The pager remains within the visible list-pane geometry at 100×30 and 170×48.
- A long exact total and long localized labels do not push either control
  outside the list pane.
- Important states are text-labelled; colour is never the sole carrier.
- Focus outlines and disabled-label contrast use existing semantic tokens and
  shared disabled-marker helpers.
- User titles, names, and filter values remain literal/markup-safe and bounded
  in visible copy.
- Dynamic status is visible in the canvas rather than toast-only.

## Privacy and Security

- Page calls request only bounded summary projections.
- Search/filter input uses existing sanitization and literal-query boundaries.
- Logs may contain source name, page number, offset, row count, duration, and
  exception type, but never queries, titles, record bodies, paths, credentials,
  or stable private IDs. Because the Media DB method is modified for true
  offset paging, its legacy raw-query/parameter logging on this read path is
  replaced with metadata-only diagnostics in the same change.
- Live verification uses synthetic source records in an isolated profile.
- No data is persisted outside existing source stores and memory-only screen
  snapshots.

## Verification Strategy

### Pure state and service tests

- First, middle, final, and empty pages.
- Exact multiples of 20 and very large totals.
- Page clamping after deletion.
- External/concurrent shrink between page visits reloads the last valid page
  rather than applying an empty out-of-range limit/offset page.
- An out-of-range Prompt request validates and applies its single coherently
  clamped response without dispatching a second service call.
- A coordinated second shrink during that automatic reload performs no second
  automatic clamp, retains the last good records in stale presentation,
  suppresses exact total/range copy, disables row/bulk/pager actions, and
  exposes Retry/scope recovery until an authoritative success.
- Requested versus applied scope on success and failure.
- Restored-page normalization rejects booleans, strings, negatives, zero, and
  page-to-offset overflow before calling a service.
- Malformed/unequal limit-offset echoes, invalid Prompt per-page/resolved-page
  values (including divergent `current_page`/`page` aliases), malformed totals,
  undersized non-final pages, and oversized pages fail closed.
- Missing, malformed, or duplicate source identities in summary items fail
  closed before any row builder can silently discard or merge records.
- Conversation/Collection stable-ID locator envelopes fail closed when the
  target is absent or not at `target_rank - offset`, the resolved offset/page
  disagrees with the rank-derived owner, or total/cardinality/identity
  validation fails.
- Unicode, emoji, RTL, and long labels remain literal and bounded.
- Media: more than 40 records, type-filter totals, at least 60 complete
  distinct-type facets, equal-timestamp stable order, and a deep-page query
  that proves no prefix fetch.
- Skills: more than 40 temporary records, filter-before-slice, trust grouping,
  stable sort, exact filtered total, source-wide blocked total, and off-page
  Review targeting. A marker present only in body/supporting content is not a
  Library filter match.
- Collections: more than 40 real SQLite records, coherent exact total,
  stable-ID owning-page load after create/rename/restore (including equal
  timestamps), delete/refetch, and deterministic ordering.
- Conversations and Prompts existing service/state suites remain green with a
  20-row Prompt expectation.
- Prompt selection tests retain captured versions across multiple 20-row pages
  and prove filter/sort/collection changes leave the basket and captured
  versions unchanged.
- A Conversation service concurrency test coordinates a write between the
  count and row phases and proves one coherent read snapshot rather than a
  mixed total/page.
- A Media log-capture regression exercises both successful and forced-error
  reads with unique synthetic query, result-title, stable-ID, and scratch
  DB-path markers across both Loguru and stdlib logging; none may appear in
  captured diagnostics at enabled levels.

Tests crossing service seams assert against real signatures and real local
services/databases rather than fakes shaped only like the new call site.

### Mounted Textual tests

At 100×30 and 170×48:

- Row 20 is reachable within the independent scroll viewport.
- Page 2 and final-page ranges are exact.
- First Previous and final Next are visibly and semantically disabled, with
  their reasons present in mounted status text rather than tooltip-only copy.
- Both pager controls remain inside the list pane with large totals.
- With at least 60 Media types, one bounded chooser widget can keyboard-scroll
  to and commit the final option, show the active non-colour marker, and avoid
  mounting one Button per type. Escape/cancel restores opener focus and leaves
  requested/applied Media scope unchanged.
- Focus follows the invoking pager control or the defined boundary fallback.
- Filter focus survives submission and canvas sync.
- Conversation/Media selection clears with visible notice on page/scope
  changes.
- Prompt selection remains versioned and cross-page while its summary reports
  both total selected and selected on the visible page.
- Loading, page failure, scope failure, and Try-again behavior are truthful.
- For each in-scope source, a slow old request cannot overwrite its newer
  page/filter/type/sort request.
- For each in-scope source, a gated request that finishes after Library unmount
  cannot update the old or freshly re-entered screen; the fresh screen performs
  its own request.
- Late broad-snapshot overwrite isolation is asserted independently for every
  source whose legacy broad snapshot carries canvas rows—Conversation, Media,
  Skills, and Collections—while rail/landing/RAG consumers still receive that
  snapshot. Prompt retains and re-runs its existing dedicated-request isolation
  coverage.
- Deep links open off-page records without page walking.
- Conversation deep links load the deterministic owning page and never inject
  a record into page 1.
- First-load failure distinguishes unavailable total from a loaded empty page.
- Successful mutation plus failed refresh reports committed-but-stale state,
  suppresses exact-total copy, and recovers through Try again.
- In that stale state, the known mutation is locally reconciled and all row and
  bulk actions remain disabled until a fresh Retry or scope request applies.
- Stale Previous/Next controls show visible unknown-boundary reasons and remain
  disabled; Retry remains enabled. A successful scope-control page-1 request,
  independently of Retry, clears stale state and re-enables valid actions.
- Navigate-away/back tests cover successful applied-page restoration; exclusion
  of failed and unsubmitted drafts; exclusion of transient loading/error state;
  page validation/clamping; and explicit navigation-context precedence.
- Existing detail/back, mutation, trust, and prompt-source behavior remains.

Pilot waits target mounted DOM/state ownership and avoid timing sleeps.

### Mutation checks

The implementation plan must include at least these inverse checks, restored
immediately afterward:

1. Remove a source generation guard; the stale-result race test must fail.
2. Restore Media's prefix-fetch implementation; the bounded deep-page query
   test must fail.
3. Derive Media type options from current page rows; the complete-facet test
   must fail.
4. Restore Conversation off-page prepend behavior; the deterministic deep-link
   page test must fail.
5. Derive the Skills blocked count/Review target from the current page; the
   off-page trust test must fail.
6. Leave one stale-row action enabled after a committed mutation's refresh
   failure; the stale-action safety test must fail.

### Live verification

Build the scratch profile before importing application modules. The clean
launch environment sets `TLDW_TEST_MODE=1`, scratch `HOME`, scratch
`XDG_CONFIG_HOME`, scratch `XDG_DATA_HOME`, exact scratch
`TLDW_CONFIG_PATH`, and `[paths].data_dir` inside that scratch tree. Keep stderr
attached to the real PTY because Textual rendering and tracebacks depend on it.
Before interaction, prove the resolved config/data/model/database paths and
open handles are scratch-owned. Fingerprint the real config and data roots
before launch and after clean shutdown and require byte-identical manifests.

Seed at least 45 synthetic Media items, Skills, and Collections. Verify page 1,
page 2, final page, scrolling, type/filter behavior, focus, and recovery at
100×30 and 170×48. Regress existing Conversations and Prompts paging in the
same isolated run where practical.

Captures and logs contain synthetic labels only and no source bodies or secrets.

### Final gates

1. Pure state/service suites.
2. Exact mounted pager nodes.
3. Mutation checks and restored-green rerun.
4. Relevant Library and UI owner suites.
5. Narrow/wide geometry tests.
6. Isolated live TUI verification.
7. Repository gate after confirming no competing broad pytest run.
8. Focused Ruff and final diff checks.

If a broad gate fails, run the identical command against the exact latest-`dev`
base in an isolated worktree and compare failure node sets, not counts. CSS is
regenerated only when component CSS changes, and timestamp-only bundle churn is
not accepted as a semantic diff.

## Documentation

- Implement the top-level tranche through TASK-16481 (Conversation/pure display),
  TASK-16482 (Prompts), TASK-16483 (Media), TASK-16484 (Skills), and TASK-16485
  (Collections). TASK-16482 through TASK-16485 depend only on TASK-16481.
- Track the deferred nested tranche through TASK-16486 (Notes tree), TASK-16487
  (Media Trash), and TASK-16488 (Collection members), each gated on completion
  of the five top-level tasks.
- Update the Library user guide with the 20-item convention, exact range copy,
  Conversation/Media current-page selection, preserved Prompt cross-page
  selection, and recovery behavior.
- Implementation Notes must name automated, mutation, geometry, and isolated
  live evidence.

## ADR Assessment

**ADR required:** yes
**ADR path:** `backlog/decisions/067-library-top-level-pagination-contracts.md`
**Reason:** The UI ownership follows existing decisions, but the feature adds
durable cross-module page envelopes, stable-ID owning-page queries, complete
facet/trust aggregates, and requested-versus-applied recovery semantics across
multiple source services. Repository governance requires those service
contracts to be recorded before implementation.

Existing decisions:

- ADR-003: Library owns active source browsing and result display.
- ADR-030: Library source reads use exact totals, stable IDs, deterministic
  bounded pages, and bounded summary projections.
- ADR-033: Destination screens own view state; cross-visit snapshots remain
  memory-only and explicit navigation context wins.

## Accepted Trade-offs

- Previous/Next is intentionally simpler than direct page entry.
- Pages can shift under concurrent writes because browse sessions are not
  frozen snapshots.
- Skills still scan their filesystem index before returning a bounded page.
- Notes retains its existing tree-specific Load-more model until the nested
  follow-up.
- Media and Collections do not gain new text-search fields in this top-level
  tranche.
