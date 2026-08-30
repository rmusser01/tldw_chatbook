# TASK-18918 Library Media Trash Paging Design

**Date:** 2026-08-30

**Status:** Approved in conversation; self-review passed; pending user written-spec review

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
6. Back returns to the untouched Media browse scope and restores the focus/scroll
   receipt captured on entry.
7. Existing Library pager display derivation is reused, but Trash retains its own
   records, workers, selection, mutations, and recovery state.

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
- `Trash` plus the exact total only while the page is fresh;
- a title-search field;
- a bounded type chooser populated from all Trash types, never the current page;
- exact range and page copy derived by the shared Library pager function.

Submitting search or choosing a type requests page 1 for that independent scope and
clears Trash selection immediately. Search input is a draft until submitted. Failed
or abandoned drafts are not treated as applied scope. Clearing either filter returns
to page 1 of the remaining applied scope.

The type chooser follows the existing Media type-choice interaction instead of
mounting an unbounded row of type buttons. An `All types` choice is always available.

### Rows and paging

The page contains at most 20 two-line rows. Titles and secondaries wrap or truncate
according to the existing Media row grammar without horizontal scrolling. The row
list owns remaining vertical height and scrolls independently, keeping the action and
pager bars reachable.

The pager exposes Previous, `Page X of Y`, exact `A–B of N`, Next, and Retry as
appropriate. A page request retains the last good rows but disables row mutation and
pager actions until the request settles. Only an applied fresh scope exposes exact
totals and boundaries.

Selection is current-page and current-scope only. It clears before page or filter
intent begins, so an off-page row can never remain invisibly actionable. A successful
page selects its first row only when Trash still owns entry/list focus; background
completion never steals newer focus.

### Restore

Restore acts on the visibly selected fresh row. It has no Undo receipt because it is
a recovery action. On commit:

1. remove the stable ID from the Trash page immediately;
2. decrement the known Trash total;
3. reconcile the restored summary through the existing Media mutation completion
   path so normal Media browse becomes fresh or stale according to its applied scope;
4. show `Restored '<title>'.`;
5. request the authoritative current or clamped Trash page.

If the follow-up read fails, the restore remains successful. Retained rows become
stale, exact total/range copy is withdrawn, actions are disabled, and the status says
that the item was restored but Trash could not refresh, with Retry.

### Delete permanently

`Delete permanently` is available only for a visibly selected fresh row. Activating it
replaces the action bar with inline confirmation that names the bounded display title
and states that the action cannot be undone. `Delete permanently` confirms; Cancel or
Escape returns to the list and restores the opener focus without mutation.

On commit, Trash removes the stable ID locally, decrements the known total, shows
`Deleted '<title>' permanently.`, and requests the authoritative current or clamped
page. A failed follow-up read cannot report the deletion as failed; it uses the same
stale recovery contract as Restore. A service failure before commit keeps the row and
selection visible and reports a recoverable action error without changing totals.

Restore and permanent deletion share the existing Media mutation flag and worker
group with Media delete/Undo. Confirmation is invalidated when scope, selection,
generation, lifecycle, or mounted ownership changes.

### Back and focus

Back and Escape leave Trash through one exit seam. They invalidate Trash request
generations, dismiss confirmation, drop Trash-only records and drafts, and restore
the exact Media browse semantic focus and scroll receipt captured on entry. An empty
Trash remains on its honest empty state until the user leaves; a mutation never
navigates away implicitly. The applied Media query, type, page, selection, and rows
are never overwritten by Trash.

Within Trash, recomposition restores focus by semantic identity: Back, search, type,
stable Media ID, Previous, Next, Retry, Restore, Delete permanently, or Cancel. If the
identity disappears after mutation, focus moves to the row at the same page-local
position, then the previous row, then Back. Deferred focus checks the same generation
and lifecycle authority as the request that scheduled it.

### Narrow terminals

At 160×50, 120×35, 100×30, and 80×24:

- the shared Library and Items panes retain their existing collapsible behavior;
- Trash uses the same canvas allocation as normal Media;
- search/type controls wrap into compact rows rather than forcing horizontal scroll;
- the list, pager, confirmation, and recovery actions remain reachable;
- long titles cannot push pager or destructive confirmation outside the viewport.

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
- confirmation stable ID and captured title;
- request generation and one-clamp recovery ownership;
- transient committed-mutation notice.

The pure reducer validates every transition. UI-only focus/scroll receipts and Textual
worker handles remain on `LibraryScreen`; service and storage layers never own them.

The existing shared `build_library_pager_display` function derives copy and disabled
state. No generic controller, widget, message type, or application store is added.

## Page Contract

The local Trash page request contains:

- normalized title query;
- exact optional media type;
- page number of at least 1;
- page size exactly 20.

The response contains:

- exact immutable summary items for the requested page;
- exact total after filters;
- echoed page size and offset, or equivalent validated page coordinates;
- complete distinct non-empty Trash types from the same read snapshot.

The facet list covers all local Trash rows independent of the submitted title query,
active type, and current page. This keeps every type discoverable and ensures the
user can always broaden the scope without first clearing hidden state.

Each summary carries canonical stable ID, positive backing Media ID, title, media type,
and Trash timestamp required by the existing relative-age secondary. IDs must be
unique within the page. Page cardinality must equal
`min(page_size, max(total - offset, 0))`; undersized non-final pages and oversized
pages fail closed.

The title query is trimmed, rejects embedded NUL, and is bounded to 200 characters at
the service boundary. An empty normalized query means no title filter. The UI may
prevent longer input, but service validation remains authoritative.

Ordering is exactly:

```text
trash_date DESC, last_modified DESC, id DESC
```

Null timestamp behavior must be deterministic and identical between count/page tests
and production queries. Search uses parameterized title matching with explicit
wildcard escaping. Type matching is exact after existing Media type normalization.
Filters apply before `COUNT`, ordering, and `LIMIT/OFFSET`.

For the local backend, count, rows, and complete facets execute in one read
transaction. The normal `list_media_trash` compatibility seam may remain for other
callers, but Library uses the new exact contract. Scope policy enforcement and
off-event-loop execution remain in `MediaReadingScopeService`; `LibraryScreen` never
executes raw SQL.

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

Initial failure displays no rows and `Could not load Trash · Retry`. A failed page or
filter request retains the last-good applied page but removes exact boundary claims
when its authority may no longer be current. Malformed payloads are handled as
recoverable read failures and never partially normalized into a page.

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
- exact cardinality, duplicate-ID, malformed-coordinate, and out-of-range rejection;
- title/type filtering before slicing;
- complete facets independent of current page;
- coherent count/rows/facets under one read transaction;
- pure selection, loading, failure, stale, Retry, and one-clamp transitions;
- restore and permanent-delete success/failure truthfulness.

### Mounted Textual tests

- open, independent filter submit/clear, type selection, Previous/Next, and Retry;
- selection cleared before page/scope changes;
- loading and stale states disable Restore and permanent deletion;
- inline irreversible confirmation, Escape/Cancel, double-press exclusion, and focus;
- late reads after new intent, Back, mutation, and unmount cannot apply;
- committed mutation plus failed refresh remains successful and recoverable;
- Media browse page/filter/selection/focus/scroll survives a Trash round trip;
- semantic focus restoration across recomposition and row removal;
- 160×50, 120×35, 100×30, and 80×24 containment and keyboard reachability.

### Isolated live walkthrough

Use a temporary real `MediaDatabase` with more than 40 Trash records spanning
multiple types, long titles, equal/null timestamps, and active non-Trash controls.
Walk through page 1, middle/final pages, title search, type filtering, Restore,
permanent-delete confirmation, concurrent shrink, forced refresh failure/Retry, Back,
and all four supported terminal sizes. Record compositor-visible evidence and confirm
active Media/RAG reads still exclude Trash until restoration.

## ADR Check

ADR required: no

ADR path: `backlog/decisions/067-library-top-level-pagination-contracts.md`

Reason: ADR-067 already selects source-owned exact bounded pages, coherent totals,
complete filtering/facets, generation fencing, clamp-on-shrink, truthful committed
mutation recovery, privacy-safe diagnostics, and a separate Media Trash follow-up.
This design does not change schema, storage ownership, sync/conflict policy, security,
dependencies, runtime boundaries, or the long-lived Library topology.
