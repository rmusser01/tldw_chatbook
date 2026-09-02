# TASK-18918 Library Media Trash Paging Design

**Date:** 2026-08-30

**Status:** Approved; independent critique findings incorporated

**Task:** TASK-18918

**Programme design:**
`Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md`

**Governing ADR:**
`backlog/decisions/067-library-top-level-pagination-contracts.md`

## Summary

Library Media Trash will become a local-only, independently filtered, exact-page
recovery surface. It will expose at most 20 deleted Media summaries at once while
keeping every matching item reachable. Search, type, page, selection, loading,
staleness, and recovery remain owned by Trash rather than borrowing the active Media
browse controller.

Restore and per-item permanent deletion will share the existing Media mutation
interlock. A committed mutation updates local presentation before any follow-up read,
so a refresh failure cannot recast success as failure. Exact range and total copy is
shown only for a validated authoritative page.

## Context and Current Gap

The current nested Trash view already provides:

- entry from Library Browse › Media and Back/Escape return;
- a local policy-checked `list_media_trash` service seam;
- deterministic `trash_date DESC, last_modified DESC, id DESC` ordering;
- one scrollable list, stable record IDs, single-row selection, and Restore;
- shared Media mutation exclusion with delete and Undo;
- honest loading, empty, error, and transient restore feedback;
- narrow-height reachability for a large fetched page.

It always requests page 1, mounts up to 200 rows, displays truncation as
`showing X of N`, and has no way to reach later records. Its local service reads the
count and rows separately, so the total and page are not one coherent snapshot. Trash
also lacks complete-source filtering, generation-owned page requests, stale retained
rows, authoritative Retry, page clamping, and per-item permanent deletion.

## Approved Decisions

1. Trash owns an independent filter scope. Entering Trash starts unfiltered rather
   than inheriting Media browse search or type state.
2. The page size is fixed at 20, following ADR-067 and the shipped Library sources.
3. Search matches title metadata only; type is an exact complete-source facet.
   Active-content FTS is not used for deleted records.
4. The surface remains local-only. This task does not change server API behavior or
   expose a source-mode selector.
5. Permanent deletion is per-item, irreversible, and requires inline confirmation.
   Bulk Empty Trash is not added.
6. Back preserves normal Media's applied scope, page, selection, focus, and scroll.
   Restore marks that retained Media page stale; it never inserts an unranked row.
7. Existing Library pager display derivation is reused, but Trash retains its own
   records, workers, selection, mutations, and recovery state.
8. Keyboard authority follows the explicit precedence table below. Deferred page,
   filter, mutation, and focus completions cannot infer a new owner after newer intent.

## Goals

1. Make every matching local Trash item reachable through exact bounded pages.
2. Apply search and type filtering to the complete Trash source before count and
   slicing.
3. Keep exact totals, ranges, page boundaries, selection, and action availability
   truthful through mutations and concurrent source changes.
4. Preserve deterministic keyboard focus, scroll, Back, Retry, and confirmation
   behavior at all supported terminal sizes.
5. Fail closed on malformed service envelopes and fence late requests after newer
   intent or unmount.
6. Keep diagnostics metadata-only and preserve the existing Media mutation owner.

## Non-goals

- Changing Media Trash storage, schema, retention, or restore semantics.
- Adding server-backed Trash paging or changing the `/api/v1/media/trash` contract.
- Sharing page records or selection with normal Media browse.
- Searching deleted content, chunks, keywords, or FTS indexes.
- Adding bulk Restore, bulk permanent delete, or Empty Trash.
- Persisting Trash records, errors, loading state, drafts, or selection across
  Library-screen instances.
- Introducing a generic Library pager widget or data controller.
- Redesigning the Media reader or the shared three-column Library shell.

## Interaction Design

### Entry, hierarchy, and filters

Trash remains a nested Media canvas reached from the normal Media toolbar. Opening it
captures the current Media list's semantic focus and scroll receipt, clears any Media
delete receipt or select-mode confirmation, and requests unfiltered page 1.

The compact Trash header contains:

- `‹ Media` Back;
- `Local Trash` plus `N items` when unfiltered or `N matching` when filtered, only
  while the page is fresh;
- a title-search field;
- a bounded type chooser populated from all Trash types, never the current page;
- exact range and page copy derived by the shared Library pager function.

Submitting search or choosing a type requests page 1 for that independent scope and
clears Trash selection immediately. Search input is a draft until submitted. Failed
or abandoned drafts are not treated as applied scope. Clearing either filter returns
to page 1 of the remaining applied scope.

The type chooser follows the existing Media type-choice interaction instead of
mounting an unbounded row of type buttons. An `All types` choice is always available.
The applied query/type is visible beside the range whenever filters are active. A
draft is never reflected in that applied-scope label until its request succeeds.

### Rows and paging

The page contains at most 20 two-line rows. Titles and secondaries wrap or truncate
according to the existing Media row grammar without horizontal scrolling. The row
list owns remaining vertical height and scrolls independently, keeping the action and
pager bars reachable.

The pager exposes Previous, `Page X of Y`, exact `A–B of N`, Next, and Retry as
appropriate. A page request retains the last good rows but disables row mutation and
pager actions until the request settles. Only an applied fresh scope exposes exact
totals and boundaries.

Disabled controls remain mounted and explain their state. Loading uses
`Trash is refreshing.`; stale mutation actions use
`Refresh Trash before changing this item.`; and an absent selection uses
`Select a Trash item first.` These reasons are exposed through visible status copy or
the shipped tooltip convention as appropriate.

Selection is current-page and current-scope only. It clears before page or filter
intent begins, so an off-page row can never remain invisibly actionable. Page and
filter completion follow the focus table below rather than always selecting the first
row. Background completion never steals newer focus.

### Keyboard and focus precedence

Higher rows in this table win over lower rows:

| State or origin | Escape | Successful completion | Selection/focus outcome |
| --- | --- | --- | --- |
| Permanent-delete confirmation | Cancel confirmation | N/A | Cancel has initial focus; opener focus returns after Cancel |
| Restore/permanent-delete call in flight | Do not leave; status explains that the mutation is finishing | Commit or recoverable pre-commit failure | Mutation action retains authority until the service reports whether it committed |
| Type chooser open | Close chooser | N/A until explicit activation | Type control retains focus; selection is unchanged |
| Type-filter submission | Return to Media and invalidate the request | Page 1 applied | Type control retains focus; selection stays empty |
| Search submission | Return to Media and invalidate the request | Page 1 applied | Search retains focus; selection stays empty |
| Previous/Next pager | Return to Media and invalidate the request | Requested page applied | Invoking pager retains focus when it remains enabled; otherwise its nearest pager/Back fallback receives focus; selection stays empty |
| Retry | Return to Media and invalidate the request | Failed target applies or clamps once | Retry retains focus while visible; success uses the original request-origin rule |
| Initial entry | Return to Media and invalidate the request | Page 1 applied | First row is selected/focused only if entry authority is still current; an initial failure focuses Retry; otherwise Back |
| Post-commit Trash refresh | Return to Media and invalidate only the follow-up read | Authoritative current/clamped page applied | Same page-local row position, then previous row, then Back, only if mutation authority is still current |
| Empty fresh page | Return to Media | N/A | Back receives focus; selection stays empty |
| Idle Trash list | Return to Media | N/A | Row activation selects that visible row; Up/Down follows DOM order |
| Back activation | Already leaving | N/A | Apply the guarded normal-Media return receipt; late Trash completions are fenced |

While confirmation is open, Escape always means Cancel and never Back. Otherwise
Escape exits Trash through the shared Back seam. A keypress that opened or submitted
a control is consumed; it cannot activate a newly mounted successor control. The only
temporary exception is the short mutation-call phase where commit status is unknown:
Back and Escape remain mounted but disabled with `Finishing this action…`. Once the
service reports pre-commit failure or committed success, leaving is available again;
a post-commit authoritative Trash refresh may be abandoned without changing the
mutation result.

### Restore

Restore acts on the visibly selected fresh row. It has no Undo receipt because it is
a recovery action. On commit:

1. remove the stable ID from the Trash page immediately;
2. decrement the known Trash total as internal reconciliation knowledge only;
3. transition Trash immediately to stale/loading, withdraw exact header count, range,
   and page claims, and disable row mutations;
4. preserve normal Media's applied scope, page, records, selection, focus, and scroll,
   while marking that retained page stale through the existing Media mutation
   completion path;
5. never insert or reposition the restored row without an authoritative ranked Media
   read;
6. show `Restored '<title>'.` and request the authoritative current or once-clamped
   Trash page.

If the follow-up read fails, the restore remains successful. Retained rows become
stale, exact total/range copy is withdrawn, actions are disabled, and the status says
`Restored '<bounded title>'. List may be out of date · Retry`.

### Delete permanently

`Delete permanently` is available only for a visibly selected fresh row. Activating it
replaces the action bar with inline confirmation that states the irreversible
consequence and identifies the target by its full title, media type, and Trash
timestamp. The full title is shown in a bounded vertically scrollable detail region,
not shortened into an ambiguous confirmation. Missing type or timestamp is rendered
as `Unknown type` or `Unknown deletion time`, never silently omitted. The captured
stable ID authorizes the mutation; visible text never does. Cancel receives initial
focus. The opener keypress is consumed, so only a later explicit activation of
`Delete permanently` can commit. Cancel or Escape returns to the list and restores
the opener focus without mutation.

The commit calls only
`MediaReadingScopeService.permanently_delete_media_item(mode="local", media_id=...)`.
That seam preserves the existing physical cascade, FTS cleanup, and no-sync-log
semantics; this task does not add a second delete path. A pre-commit not-found,
not-in-Trash, policy, or service failure keeps the row and selection visible and
reports a recoverable action error without changing totals.

On commit, Trash removes the stable ID locally, treats the decremented known total as
internal reconciliation knowledge only, transitions immediately to stale/loading,
withdraws exact header/range/page claims, disables row mutations, shows
`Deleted '<title>' permanently.`, and requests the authoritative current or
once-clamped page. A failed follow-up read cannot report the deletion as failed; it
uses `Deleted '<bounded title>' permanently. List may be out of date · Retry`.

Restore and permanent deletion share the existing Media mutation flag and worker
group with Media delete/Undo. Confirmation is invalidated when scope, selection,
generation, lifecycle, or mounted ownership changes.

### Back and focus

Back and Escape leave Trash through one exit seam. Before entry, Trash captures a
distinct return receipt containing the normal Media stable row identity, list scroll,
and semantic opener/focus identity. It does not reuse the Media Viewer receipt field.
Exit invalidates Trash request generations, dismisses confirmation, drops Trash-only
records and drafts, and applies the receipt through the guarded Media list-return
path. An empty Trash remains on its honest empty state until the user leaves; a
mutation never navigates away implicitly.

The applied Media query, type, page, records, selection, focus, and scroll are never
overwritten by Trash. Restore marks that retained Media page stale and exposes its
source-owned Retry because the restored item's ranked position is unknown. Permanent
deletion does not independently stale normal Media because a trashed item was not in
its active result set. A successful Trash refresh never clears normal Media's stale
flag; only an authoritative normal-Media refresh does.

Within Trash, recomposition restores focus by semantic identity: Back, search, type,
stable Media ID, Previous, Next, Retry, Restore, Delete permanently, or Cancel. If the
identity disappears after mutation, focus moves to the row at the same page-local
position, then the previous row, then Back. Deferred focus checks the same generation
and lifecycle authority as the request that scheduled it.

### Narrow terminals

At 160×50, 120×35, 100×30, and 80×24:

- the shared Library and Items panes retain their existing collapsible behavior;
- Trash uses the same canvas allocation as normal Media;
- the fixed vertical order is Back/title/authority, filters, status, the `1fr` list,
  pager, then action or confirmation;
- at 80×24 the list retains at least four terminal rows, enough for two complete
  two-line entries;
- status copy uses at most two visible lines plus a fold indicator;
- the bounded type chooser temporarily replaces the filter row with a scrollable
  choice strip instead of adding height;
- search/type controls never force horizontal scrolling;
- destructive consequence and confirmation buttons remain fixed while only the full
  target-title detail region scrolls;
- disabled pager and mutation controls remain visible with textual reasons or
  tooltips, never color alone;
- the list, pager, confirmation, and recovery actions remain keyboard reachable.

## State and Authority

Trash uses a source-owned immutable state/reducer separate from normal Media browse.
Its scope contains normalized submitted query, optional media type, page, and fixed
page size. Runtime state contains:

- requested and applied scope;
- last-good immutable page summaries;
- fresh exact total and complete type facets;
- `uninitialized`, `fresh`, or `stale` freshness;
- loading direction and recoverable error/stale copy;
- current-page selected stable ID;
- confirmation stable ID and captured full title, type, and Trash timestamp;
- request generation and one-clamp recovery ownership;
- transient committed-mutation notice.

The pure reducer validates every transition. UI-only focus/scroll receipts and Textual
worker handles remain on `LibraryScreen`; service and storage layers never own them.
After a committed mutation, the reducer may remove the captured stable ID immediately,
but it must transition to stale/loading before scheduling the authoritative read. A
locally decremented count is reconciliation knowledge, not display authority. Only a
validated service result may restore fresh totals, range/page copy, selection, and
enabled row actions.

The existing shared `build_library_pager_display` function derives copy and disabled
state. No generic controller, widget, message type, or application store is added.

## Page Contract

The local Trash page request contains:

- normalized title query;
- exact optional media type;
- page number of at least 1;
- page size exactly 20.

Page and page-size inputs reject booleans and non-integers. The derived offset must
equal `(page - 1) * 20`, remain at most SQLite's signed 64-bit maximum
`2**63 - 1`, and match the coordinates echoed by the response.

The response envelope contains exactly:

- `items`: exact immutable summaries for the requested page;
- `total`: exact total after filters;
- `limit`: echoed page size;
- `offset`: echoed derived offset;
- `types`: complete distinct non-empty Trash types from the same read snapshot.

The facet list covers all local Trash rows independent of the submitted title query,
active type, and current page. This keeps every type discoverable and ensures the
user can always broaden the scope without first clearing hidden state.

Each summary contains exactly `id`, `backing_media_id`, `title`, `media_type`, and
`trash_date`. `backing_media_id` is a positive integer and `id` is its canonical
`local:media:<backing_media_id>` stable ID. `title` is a non-empty display string with
the established `Untitled` fallback, `media_type` is a trimmed string or `None`, and
`trash_date` is an ISO timestamp string or `None`. Both stable IDs and backing IDs
must be unique within the page. Page cardinality must equal
`min(page_size, max(total - offset, 0))`; undersized non-final pages and oversized
pages fail closed.

The title query is trimmed, rejects embedded NUL, and is bounded to 200 characters at
the service boundary. An empty normalized query means no title filter. The UI may
prevent longer input, but service validation remains authoritative.

Ordering is exactly:

```sql
ORDER BY trash_date IS NULL ASC,
         trash_date DESC,
         last_modified IS NULL ASC,
         last_modified DESC,
         id DESC
```

Null timestamps therefore sort last deterministically. Search uses parameterized title
matching with explicit wildcard escaping. Submitted and stored type values use
`TRIM(type)` at the contract boundary; an empty trimmed value means no facet. Facets
are unique sorted non-empty strings, and filtering compares trimmed values by exact
case-sensitive equality. There is no case folding or alias mapping. Filters apply
before `COUNT`, ordering, and `LIMIT/OFFSET`.

The exact contract is implemented through
`MediaDatabase.list_library_media_trash_page`,
`LocalMediaReadingService.list_library_media_trash`, and
`MediaReadingScopeService.list_library_media_trash`. The scope method requires
`mode="local"`; non-local modes fail closed and no server endpoint changes. The
legacy `list_media_trash` compatibility seam remains for other callers.

For the local backend, count, rows, and complete facets execute in one read
transaction. Scope policy enforcement and off-event-loop execution remain in
`MediaReadingScopeService`; `LibraryScreen` never executes raw SQL.

## Validation, Drift, and Recovery

Only the latest Trash generation may apply. The apply gate checks:

- mounted screen and active `media-trash` view;
- unchanged lifecycle and Trash request generation;
- exact requested scope fingerprint;
- valid page envelope and unique stable IDs;
- unchanged mutation/focus authority where the completion may move focus.

A fresh response proving the requested page is beyond the last valid page triggers
one generation-guarded request for `max(1, last_page)`. The out-of-range response is
never displayed. If the clamp response is also out of range, malformed, or fails,
last-good rows become stale and Retry is offered with unknown boundaries.

Initial failure displays no rows and `Could not load Trash · Retry`; Retry repeats the
unfiltered page-1 request. A failed filter submission keeps the prior applied page
fresh and shows `Filter not applied — showing <previous applied scope> · Retry`;
Retry repeats the failed submitted scope. A failed ordinary page request keeps the
prior applied page fresh and shows
`Page <N> not loaded — showing page <M> · Retry`; Retry repeats page N. In both cases
selection stays empty until the user explicitly selects a retained row, and pager
actions operate on the visible applied scope.

A committed mutation or externally detected shrink makes retained rows stale and
shows `List may be out of date · Retry`. Stale state exposes no exact total, range, or
page-boundary claims and disables row mutations. Retry requests the current target or
its single guarded clamp. Query validation uses the fixed copy
`Search is limited to 200 characters.` Malformed payloads are handled as recoverable
read failures and never partially normalized into a page.

Unmount, Back, new filter/page intent, and mutation each invalidate older reads before
starting work. Cancelled Textual workers may finish their local thread reads, but late
results cannot apply to a former or remounted screen.

## Diagnostics and Privacy

Diagnostics may record operation name, page size, requested page, filter-presence
booleans, item count, total, freshness, generation outcome, and exception type. They
must not record search text, titles, raw records, paths, content, credentials, stable
private IDs, or permanent-delete targets.

User-facing failures use bounded fixed copy and never expose raw exception text.

## Verification

### Pure and service tests

- scope normalization and immutable page validation;
- exact cardinality, duplicate stable/backing ID, malformed-coordinate, and
  out-of-range rejection;
- boolean/non-integer coordinate rejection and the SQLite signed-offset bound;
- title/type filtering before slicing, including exact stored-type equality;
- complete unique sorted facets independent of current page;
- explicit NULL-last timestamp ordering;
- coherent count/rows/facets under one read transaction;
- exact local database/service/scope seam and non-local rejection, with the server
  contract untouched;
- pure selection, loading, failure, stale, Retry, and one-clamp transitions;
- restore and permanent-delete success/failure truthfulness.

### Mounted Textual tests

- open, independent filter submit/clear, type selection, Previous/Next, and Retry;
- selection cleared before page/scope changes;
- loading and stale states disable Restore and permanent deletion;
- inline irreversible confirmation with duplicate/truncated/long titles, full target
  identity, Escape/Cancel, consumed opener activation, double-press exclusion, and
  focus;
- every keyboard-precedence row, including initial-error Retry focus and pager
  fallback focus;
- Back/Escape exclusion while mutation commit status is unknown and immediate return
  availability during the post-commit refresh;
- late reads after new intent, Back, mutation, and unmount cannot apply;
- committed mutation plus failed refresh remains successful and recoverable;
- Media browse page/filter/selection/focus/scroll survives a Trash round trip;
- Restore marks retained Media stale without inserting an unranked row, while
  permanent deletion does not independently stale Media;
- semantic focus restoration across recomposition and row removal;
- 160×50, 120×35, 100×30, and 80×24 containment and keyboard reachability in
  ordinary, initial-error, stale, and destructive-confirmation states.

### Isolated live walkthrough

Use a temporary real `MediaDatabase` with more than 40 Trash records spanning
multiple types, long titles, equal/null timestamps, and active non-Trash controls.
Walk through page 1, middle/final pages, title search, type filtering, Restore,
permanent-delete confirmation, concurrent shrink, forced refresh failure/Retry, Back,
and all four supported terminal sizes. Confirm the Local Trash authority label,
filtered total copy, distinct return receipt, stale normal-Media result after Restore,
and unchanged normal-Media result after permanent deletion. Record compositor-visible
evidence and confirm active Media/RAG reads still exclude Trash until restoration.

## ADR Check

ADR required: no

ADR path: `backlog/decisions/067-library-top-level-pagination-contracts.md`

Reason: ADR-067 already selects source-owned exact bounded pages, coherent totals,
complete filtering/facets, generation fencing, clamp-on-shrink, truthful committed
mutation recovery, privacy-safe diagnostics, and a separate Media Trash follow-up.
This design does not change schema, storage ownership, sync/conflict policy, security,
dependencies, runtime boundaries, or the long-lived Library topology.
