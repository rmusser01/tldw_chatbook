# TASK-18917 Library Notes Tree Placement-Aware Paging Design

**Date:** 2026-08-29

**Status:** Approved interaction and architecture design; pending written-spec review

**Task:** TASK-18917

**Programme design:**
`Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md`

**Governing ADR:**
`backlog/decisions/067-library-top-level-pagination-contracts.md`

## Summary

Large Database Notes hierarchies must remain reachable without flattening the tree
or borrowing a flat-source pagination controller. Root folders, Unfiled notes, and
each expanded folder will own exact bounded ranges, local loading/retry/stale state,
and stable placement reconciliation.

Normal exploration retains the existing tree-specific Load-more interaction. A
placement locator can open a containing middle range for a deep link or mutation
result without automatically loading every earlier sibling. Existing stable folder,
note, and membership placement identifiers remain authoritative.

## Context and Current Gap

The current Notes navigator already provides:

- stable placement-aware row identities;
- lazy folder expansion;
- bounded folder, note, and membership repository queries;
- independent transport cursors for folders, notes, and memberships;
- mutation routing through `NotesScopeService`;
- generation fencing for whole-tree refreshes;
- filter-specific placement loading;
- focus and Back-return infrastructure in the adaptive reader shell.

However, every expanded folder currently contributes to one aggregate
`_library_notes_tree_expanded_page`. One bottom-of-tree control advances any cursor
that remains. Its totals and errors therefore describe a broad partial snapshot,
not an exact parent branch. This cannot truthfully answer which folder is loading,
exhausted, stale, or retryable, and a distant deep link cannot reveal its placement
without loading preceding pages.

## User Decisions

The following decisions were approved during brainstorming:

1. Root folders, Unfiled, and each expanded folder own inline continuation controls.
2. Normal browsing uses cumulative Load more behavior.
3. Deep-link or mutation recovery may open a containing middle range and expose
   Load earlier as well as Load more.
4. Branch state remains Notes-specific; no generic Library controller is introduced.
5. Existing stable placement identity and hierarchy remain intact.
6. A fixed 20-item range follows the established Library pagination convention.

## Goals

1. Make every active Database Note reachable through bounded folder expansion or
   branch-local paging.
2. Keep folders and note placements under their exact parents.
3. Present exact totals and ranges only when supplied by an authoritative
   parent-scoped query.
4. Preserve deterministic selection, expansion, focus, scroll, deep-link, Back, and
   mutation behavior through stable identities.
5. Keep loading, retry, and stale status local to the affected branch and content
   kind.
6. Cover mounted keyboard, geometry, failure, race, and unmount behavior.

## Non-goals

- Flattening Notes into a top-level list.
- Replacing Load more with page-number navigation.
- Moving Notes state into a generic Library paging controller.
- Changing note-folder storage, membership ownership, sync policy, or schema.
- Freezing a whole browse session into a database snapshot.
- Loading or mounting an entire hierarchy to satisfy a deep link.
- Redesigning the Notes editor or adaptive reader shell.

## Interaction Design

### Branch boundaries

The fixed page size is 20 items per child-folder or note-placement request.

- Root-folder continuation renders after loaded root folders and before Unfiled.
- Unfiled-note continuation renders after loaded Unfiled note placements.
- Within an expanded folder, child-folder continuation renders after loaded child
  folders and before note placements.
- Note continuation renders after loaded note placements.

This ordering prevents newly loaded child folders from appearing above a continuation
control that had been rendered below existing notes.

### Copy and controls

A range starting at the first item uses cumulative copy:

```text
Folders 1–20 of 83  Load more folders
Notes 1–20 of 146  Load more notes
```

A locator-opened middle range uses exact window copy and both directions when they
exist:

```text
Notes 201–220 of 400  Load earlier  Load more
```

Membership continuation is an internal transport concern. The interface reports
note-placement progress and never exposes membership cursor terminology.

### Loading, failure, and stale state

- Activating a pager preserves visible rows and changes only that local action to
  `Loading…`.
- Other branches and content kinds remain interactive.
- Recoverable failure keeps the same boundary control focused and changes it to
  `Couldn’t load more · Retry`.
- A failed mutation refresh retains its prior contiguous range but withdraws the
  exact total: `20 notes loaded · May be out of date · Retry`.
- Exact totals return only after an authoritative refresh succeeds.

### Collapse, filtering, and narrow terminals

- Collapsing a folder retains its loaded range for the current screen lifetime.
- Re-expanding the folder restores it immediately unless a topology change made it
  stale.
- Filter mode owns separate bounded results, totals, generations, and continuation.
  It never replaces browse branch records.
- Clearing a filter restores the prior browse receipt, including expansion, ranges,
  selection, focus, and scroll.
- Titles wrap within available width. Status and actions occupy separate lines when
  necessary rather than squeezing or truncating titles or creating horizontal
  scrolling.

## Notes-Owned State

The screen replaces the one aggregate expanded page with a map keyed by parent
folder ID. Root uses a private sentinel. Each branch contains independent
child-folder and note-placement slice state.

Each slice records:

- one contiguous loaded start/end range;
- the authoritative total while trusted;
- earlier and later offsets;
- the active load direction;
- a recoverable error;
- a stale flag;
- a slice request generation.

The branch map remains private to the Notes navigator. It is not an application
store, a generic controller, or a persistence format.

### Contiguous-range rule

Normal Load more appends an adjacent range. Load earlier prepends an adjacent range.
A target already inside the loaded range reuses it. A distant locator target replaces
the slice with the target’s containing range. Disconnected hidden segments are not
retained because the interface could not describe their gaps truthfully.

Incoming exact totals replace prior totals. They are never combined with `max()`.
Stable-identity item merging is allowed only for an adjacent continuation. A refresh
replaces the authoritative slice.

## Repository and Service Contracts

The repository and `NotesScopeService` gain Notes-specific parent-scoped operations.
The exact names may follow existing module conventions, but the contracts are:

### Load one branch slice

Inputs:

- parent folder ID, or root;
- content kind: child folders or note placements;
- offset;
- limit, fixed to 20 from the screen;
- membership offset when completing one note-placement slice.

Outputs:

- exact items for that parent and content kind;
- exact total even when the requested offset returns no rows;
- previous and next offsets;
- memberships needed to render the returned note slice;
- membership continuation for the same note slice when required.

Folder and note loading must be independently selectable. Loading Unfiled notes must
not reload root folders, and loading child folders must not query exhausted notes.

`NoteFolderPage` remains the transport model unless implementation evidence shows a
smaller extension is required. The screen-owned slice state supplies start offsets,
direction, trust, and generations that do not belong in the repository record.

### Locate one placement

Inputs:

- exact note or folder ID;
- optional preferred placement identity.

Outputs:

- exact surviving placement identity;
- ancestor folder IDs;
- the containing child-folder offset for every ancestor;
- the containing note offset for a note placement;
- exact membership offset when duplicate placements require it.

Resolution order is:

1. exact preferred membership placement;
2. the same note in the preferred folder;
3. a canonical active folder placement;
4. Unfiled when the note has no active placement.

Canonical folder placements sort by normalized folder path and folder ID, then by
membership ID for duplicate placements.

The locator and branch loader use identical ordering and collation:

- folders: normalized name/path followed by folder ID;
- notes: title with SQLite `NOCASE`, followed by note ID;
- duplicate placements: membership ID.

Locator rank and ancestor queries execute in one read transaction so their topology
and offsets are coherent.

## Projection and Rendering

The pure Notes tree projection receives root state plus the parent-keyed branch map.
It recursively renders only loaded folders, their loaded direct note placements, and
branch-boundary status rows.

Projection rows gain a paging/status kind carrying:

- parent branch identity;
- content kind;
- direction or retry action;
- exact range copy or stale copy;
- focus identity;
- disabled/loading state.

Folder, note, and membership placement row identities remain unchanged. Pager
identities derive from parent ID, content kind, and direction; they do not depend on
visible row index.

The widget renders status through text and button semantics, not color alone. Existing
Notes row handlers remain responsible for folders and note placements; dedicated
Notes-tree paging messages or button metadata route continuation actions to the exact
branch slice.

## Request and Race Semantics

Each request captures:

1. its branch-and-content slice generation; and
2. the screen’s Notes topology epoch.

A mismatch in either value discards the result. A newer request supersedes an older
request for the same slice. Requests for independent slices may overlap.

Mutation admission temporarily fences new paging, increments the topology epoch, and
returns any in-flight loading indicators to idle while preserving their visible rows.
Older responses cannot land. Paging resumes after the mutation result is reconciled.

Unmounted results never call query, repaint, or focus APIs. A still-current result may
remain cached for a later remount; stale results are discarded.

## Selection, Focus, Deep Links, and Back

Selection and focus use stable placement or pager identities, never row indexes.

- Expanding a folder loads that folder’s first ranges.
- Restoring a path loads each ancestor’s containing folder range in order, then
  expands it.
- A note deep link opens the editor immediately while its canonical tree placement is
  resolved in the background.
- Editor fetch, placement locator, and Back restoration share one navigation
  generation. Older results cannot overwrite newer navigation.
- If Back occurs before location completes, the navigator renders `Locating note…`
  and applies focus only while the same return receipt remains current.

The live branch map remains the only owner of loaded records. A browse receipt stores
only semantic state:

- selected placement and note ID;
- expanded folder IDs;
- active contiguous range descriptors;
- filter query and result range;
- semantic focus identity;
- navigator and rail scroll offsets;
- captured topology epoch.

Within the same topology epoch the receipt reuses live branch data. After a topology
change it reloads only recorded containing ranges before applying focus. Cross-session
state may store semantic IDs, offsets, expansion, and filter state, but never folder,
note, or membership records.

Focus moves to the first added row only if the activated paging control still owns
focus when loading completes. If the user moved elsewhere, completion does not steal
focus. Failure retains focus on the same stable pager identity. When an exhausted
pager disappears, focus falls to the first added row, then its parent folder when no
row was added.

## Mutation Reconciliation

Mutation invalidation uses exact repository results or exact pre/post lookups, never
partially loaded branch records.

- Create refreshes the containing parent and selects the new folder ID.
- Rename refreshes the containing parent and changed folder while retaining folder ID.
- Move refreshes old and new parents, the moved subtree, and changed ancestor paths;
  the new path expands and retains selection.
- Delete/restore refreshes the containing parent, affected subtree, and Unfiled when
  membership activity changes.
- Placement move/detach refreshes source and destination parents.
- Note create/delete refreshes Unfiled and every exact placement parent returned by
  repository lookup.

A partial placement move is a successful topology change: the destination placement
remains selected, both source and destination refresh, and the user is told the
original remains safely attached.

Deletion captures stable sibling IDs before mutation. Fallback order is:

1. next surviving sibling;
2. previous surviving sibling;
3. surviving parent;
4. canonical first visible placement.

A failed mutation removes the paging fence and retains previously trusted data. A
failed post-success refresh marks only affected slices stale and withdraws their exact
totals.

## Filter State

Filter search is a separate bounded view with authoritative result totals and
continuation. Search results carry exact placement identities and ancestor context but
do not populate or overwrite browse branches.

Opening a filtered placement captures the filter query, result range, selection,
focus, scroll, and topology epoch. Back within the same epoch restores it directly.
After a topology change, the filter result becomes stale and refreshes before exact
totals are presented.

## Error and Privacy Rules

- Branch load failures retain visible rows and expose an inline Retry.
- Locator failure returns to the navigator with actionable status rather than an
  empty permanent loading view.
- A removed deep-link target falls back through the deterministic reconciliation
  order.
- Error logs include operation, branch content kind, generation, and exception class.
  They do not include note titles, folder paths, filter text, or note bodies.
- All status meaning is visible in text, not color alone.

## Performance Boundaries

- Every user-triggered branch fetch returns at most 20 folders or notes.
- Membership continuation remains bounded and must finish the current note slice
  before advancing to another note range.
- Database operations remain off the Textual event loop.
- Locator work scales with ancestor depth and indexed rank queries, not hierarchy
  width.
- No operation mounts or scans the entire hierarchy merely to reveal one placement.

## Verification Design

### Pure state and repository tests

- exact parent-local totals and empty out-of-range totals;
- folder, note, and membership ordering/rank agreement;
- adjacent prepend/append and distant-range replacement;
- decreasing totals replacing old totals;
- duplicate placements and membership continuation;
- root folders, Unfiled, deep ancestry, and independent sibling branches;
- exact mutation-affected parent discovery.

### Screen orchestration tests

- initial root load and one-folder expansion;
- folder-only and note-only continuation;
- simultaneous different-branch requests;
- older/newer same-branch races;
- topology-epoch mutation fencing;
- local loading, retry, stale, and recovery behavior;
- collapse/re-expand retention;
- create, rename, move, restore, delete, and partial placement move;
- external deep-link and Back restoration;
- filter-range return and topology-stale refresh;
- unmount without repaint or focus calls.

### Mounted and geometry tests

- keyboard focus across folder, note, earlier/more, and Retry controls;
- focus preservation when the user moves during an asynchronous request;
- scroll preservation across branch-local canvas sync;
- inline pager placement between child folders and note placements;
- wrapped titles and status controls without horizontal overflow;
- production-shaped adaptive-reader regression coverage;
- 160×50, 120×35, 100×30, and 80×24 terminal sizes.

### Isolated live verification

Use a scratch profile and data root with a synthetic hierarchy containing:

- more than 20 root folders;
- more than 20 Unfiled notes;
- one folder with both paged child folders and paged note placements;
- multiple deep ancestor levels;
- duplicate placements;
- successful mutations;
- a recoverable branch failure and Retry.

Capture identifying content and exact range/status evidence at each required terminal
size. Do not use a developer’s normal profile or database.

## Documentation

- Update the Library user guide with branch-local folder/note paging, Load earlier
  behavior for located ranges, stale copy, Retry, and filter restoration.
- Record targeted automated and isolated live evidence in TASK-18917 Implementation
  Notes.
- Link this design and ADR-067 from the task plan and completion notes.

## ADR Check

**ADR required:** no

**ADR path:** `backlog/decisions/067-library-top-level-pagination-contracts.md`

**Reason:** ADR-067 already requires source-owned paging, exact query-derived totals,
generation fencing, stable identity, and separate follow-up treatment for the Notes
hierarchy. This task applies that established boundary without changing storage,
ownership, sync policy, security, dependencies, or application-level architecture.

## Accepted Trade-offs

- Branch state is more explicit than one aggregate expanded page, but it is necessary
  for truthful parent-local status and races.
- Located middle ranges require Load earlier in addition to the normal Load-more
  interaction.
- Collapsed branch data may consume memory for the current screen lifetime; it avoids
  surprising reloads and is discarded with the screen rather than persisted as data.
- Concurrent writes may shift ranges because the design does not freeze a browsing
  snapshot. Stable identity and exact refresh handle those shifts.
