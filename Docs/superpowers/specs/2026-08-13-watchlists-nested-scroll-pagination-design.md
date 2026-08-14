# Watchlists Nested Scroll and Pagination Design

Date: 2026-08-13
Status: Approved for implementation planning
Builds on: [Watchlists Reader-First Re-IA](2026-08-05-watchlists-reader-first-design.md)
Decision: [ADR-042 — Watchlists reader-first information architecture](../../../backlog/decisions/042-watchlists-reader-first-ia.md)

## Summary

Make the Watchlists Read tab usable as a reader when both the article list and
the selected article are long. The left Watchlists rail and right Inspector
remain fixed. The centre column becomes vertically scrollable and contains two
independently bounded regions: an article list that grows from 10 to 50 terminal
rows and a Content reader that grows from 20 to 50 terminal rows. Once either
region reaches its cap, its body scrolls internally. The article list uses
explicit 50-item pages.

This is a focused UI and pagination change. It reuses the existing region,
scope-service, and `limit`/`offset` boundaries. It adds no database schema,
dependency, global keybinding, or remote-reading capability.

## Problem

The current centre stack is a fixed `Vertical`. The ITEMS region consumes the
flexible height, while CONTENT is content-sized and capped at 12 rows. In a tall
Read list, the reader is pushed to the bottom without a centre-column viewport
that can intentionally reach it. Even when reached, the 12-row cap leaves too
little room for useful reading.

Item loading is also a single 100-row request with no visible page controls.
The UI needs a predictable boundary so a user can scroll the current list,
move to the Content region, scroll the article independently, and explicitly
request the next set of items.

## Goals

- Keep the Watchlists and Inspector rails fixed while the centre column scrolls.
- Let the Read list shrink to a 10-row floor and grow naturally to a 50-row cap.
- Let Content start at 20 rows and grow naturally to a 50-row cap.
- Give the list and article body independent internal scrolling at their caps.
- Page the Read list explicitly in 50-item windows.
- Preserve current scope, selection, mark-read, collapse, solo, and reader
  navigation behaviour unless this design states otherwise.
- Keep short-terminal layouts usable by scrolling instead of crushing either
  region below its minimum.

## Non-goals

- Replacing the current `ItemsPane` `DataTable` with the later reader-first
  `ListView` design.
- Implementing FTS or cross-page search.
- Showing an exact total item count or total page count.
- Adding infinite scroll, automatic page advancement, a draggable splitter, or
  persisted region heights.
- Changing the Sources management tab, Watchlists rail, Inspector, backend
  ownership, or item schema.
- Adding pagination keybindings. Pager buttons remain keyboard-focusable through
  normal focus traversal.

## Layout and Scroll Contract

The workbench keeps its horizontal topology:

```text
fixed Watchlists rail | vertically scrollable centre | fixed Inspector rail
```

The Read tab centre is one stacked scroll document:

```text
centre scroll viewport
├── centre status and tab strip
├── Read list region: 10–50 rows
│   ├── title and toolbar (fixed inside the region)
│   ├── DataTable (independently scrollable)
│   ├── queued legend (fixed inside the region)
│   └── Previous · Page N · Next (fixed inside the region)
└── Content region: 20–50 rows
    ├── Content title (fixed inside the region)
    ├── action row (fixed inside the region)
    └── article body (independently scrollable)
```

The 10- and 50-row Read-list values and the 20- and 50-row Content values
describe each outer region box, including its chrome. They do not promise 50
simultaneously visible data rows. Page size is an independent 50-item data
boundary; at the region cap, the `DataTable` scrolls through the page while the
toolbar, legend, and pager remain visible.

The centre viewport owns movement between the Read list and Content. The
`DataTable` owns scrolling among the current page's rows. The Content body owns
scrolling within the selected article. Pointer-wheel and keyboard scrolling
follow Textual's nearest focused or hovered scroll owner; no custom wheel-event
forwarding is introduced unless live verification proves native nested
scrolling cannot satisfy this contract.

On a terminal that cannot display both region minimums plus centre chrome at
once, the centre viewport scrolls. Neither region shrinks below its minimum.
The existing `z` collapse and `Z` solo/restore interactions remain. Natural
shrinking to the Read list's 10-row floor is the requested “auto-collapse”; it
does not create another persisted collapse state. Explicit solo is the one
height exception: the sole centre region may use the available centre height,
as it does today, because the user deliberately requested a full-region view.

The centre container is scroll-capable on every Watchlists tab. Non-Read panes
retain their existing fill behaviour and only use the outer viewport when their
content genuinely overflows. Read-specific min/max sizing is gated by a class on
the workbench so it does not cap Sources, Runs, Rules, Notifications, Artifacts,
or Overview at 50 rows.

## Explicit Pagination

The page size is 50. A load requests 51 rows at
`offset = page_index * 50`. The first 50 rows are displayed. The lookahead row
is not mounted; its presence only enables `Next`.

The pager is a compact, persistent row below the table:

- `Previous` is disabled on page 1.
- The centre label reads `Page N`.
- `Next` is disabled when no lookahead row exists.
- Both controls are disabled while a page request is in flight.
- No total-count query is added solely to display “of N”.

A page transition is transactional from the UI's perspective. The requested
page index is committed only after its load succeeds. A failure retains the
current page, rows, selection, and article; it shows the existing error
notification and restores the pager controls. Rapid repeated activation cannot
start overlapping page loads.

Changing backend, tree scope, source, status filter, or search text resets to
page 1. Refresh reloads the current page. If a mutation or refresh leaves a
non-first page empty, the screen loads the nearest preceding non-empty page.

The selected article remains in Content when the user changes pages, but it is
not injected into a page where it does not belong. Selecting a row on the new
page replaces it. The current open-item pin remains valid within its own page so
mark-read-on-open does not remove the row from a New-filtered view. The visible
page never exceeds 50 rows to preserve that pin; if a reload must carry the open
item alongside newly queried rows, it replaces the page's last display slot
rather than becoming a 51st visible row.

`j`, `k`, and next-unread navigation operate only on the currently rendered
page. Reaching an edge does not change pages. Pagination is always an explicit
button action.

## Search Scope

The existing search is an in-memory filter over loaded rows. This change keeps
that boundary and makes it honest by relabelling the input to
`Search this page…`. Changing the query resets pagination to page 1 before
filtering. Cross-page search remains part of the separately planned FTS work and
is not pulled into this layout fix.

## Component Responsibilities

### `WatchlistsWorkbench`

- Render `#wl-centre` with Textual's vertical scrolling container instead of a
  fixed `Vertical`.
- Preserve existing region factories, hidden-region gating, focusability,
  collapsed headers, and sole-centre class behaviour.
- Accept no new layout abstraction. The screen adds a Read-mode class to the
  existing workbench instance, which gives CSS the narrow selector it needs.

### `ItemsPane`

- Receive current page number, `has_previous`, `has_next`, and loading state
  from the screen.
- Render the fixed pager below the existing queued legend.
- Post narrowly scoped previous/next request messages; hold no service and
  calculate no offsets.
- Keep toolbar, legend, and pager visible while only the `DataTable` scrolls.
- Change the search placeholder to `Search this page…`.

### `ContentPane`

- Keep the existing renderers and action messages unchanged.
- Wrap the rendered article body in the internal vertical scroll owner so the
  title and action row remain visible at the 50-row cap.
- Keep the empty state within the 20-row region floor.

### `WatchlistsCollectionsScreen`

- Own page index, lookahead, and loading state so workbench recomposition cannot
  erase them.
- Convert pager messages into serialized 51-row requests.
- Reset or retain the page according to the pagination contract.
- Seed every rebuilt `ItemsPane` with page state in the same way it already
  seeds filters, selection, and loaded rows.
- Keep page navigation from clearing the selected Content item.

### Scope and data services

- Reuse existing `list_items(limit, offset, ...)` calls.
- Add no count method and make no storage change.
- Preserve status and scope predicates, including the current open-item pin.

## States and Feedback

- **Empty scope:** the list remains 10 rows and shows its current empty-state
  copy. Pager reads `Page 1`; both buttons are disabled.
- **Loading another page:** current rows and Content remain visible. Pager
  buttons are disabled to prevent concurrent requests.
- **Successful page load:** rows swap as one completed state update; focus moves
  to a predictable list or pager target without affecting the rails.
- **Failed page load:** current state remains visible and an error notification
  explains the failure.
- **Short article:** Content remains 20 rows; no internal scrollbar is needed.
- **Long article:** Content grows to 50 rows and then the article body scrolls.
- **Region rebuild:** page number, rows, filter/search state, selection, centre
  scroll position where practical, and Content survive collapse/solo changes.
- **Manual solo:** the chosen centre region uses available height and Restore
  returns to the bounded stacked layout.

Disabled pager labels remain readable and are not communicated by colour alone.
Tooltips explain boundary states where useful. No new footer hint is added
because no new binding is introduced.

## Error Handling and Edge Cases

- Ignore stale results from a superseded scope/filter/backend request.
- Do not commit a target page index before its rows arrive successfully.
- Clamp an out-of-range page after deletions or status changes by stepping back,
  never by showing a permanent empty page with `Previous` available.
- Keep the lookahead row out of `ItemsPane.items`, selection, navigation, and
  status mutation paths.
- Preserve the selected article even when it is absent from the new page.
- Keep explicit page navigation independent from `j`/`k`; a row-navigation edge
  must not unexpectedly issue I/O.
- Verify that inner scrollbars do not obscure pager buttons or article actions.

## Testing and Verification

### Unit and widget tests

- Empty and short Read lists render at a 10-row minimum.
- Medium lists grow naturally; large lists stop at 50 rows.
- Empty and short Content renders at 20 rows; long Content stops at 50 rows.
- The `DataTable` and article body scroll independently at their caps.
- A short terminal scrolls the centre from the Read list to Content without
  shrinking either below its minimum.
- A 51-row response displays 50 rows and enables `Next`; a response of 50 or
  fewer disables it.
- `Previous`, `Page N`, and `Next` states are correct on first, middle, and last
  pages.
- Offsets are `0`, `50`, `100`, and so on; visible rows never exceed 50.
- Page state survives a workbench rebuild and resets for every agreed context
  change.
- A failed transition preserves rows, page number, selected item, and Content.
- An empty non-first page steps back to a valid page.
- Search copy and page-local behaviour are explicit.
- Geometry assertions cover pager visibility, neighbouring controls, region
  min/max boundaries, and fixed rails at representative terminal sizes.
- Mutation-check the lookahead, disabled-state, and page-reset assertions.

### Live terminal QA

Use an isolated test profile and realistic seeded items to verify:

- centre scrolling reaches Content;
- wheel, scrollbar, focus, Page Up/Down, and arrow behaviour target the intended
  nested scroll owner;
- the current 50-item page scrolls while pager controls remain visible;
- a long article scrolls while Content actions remain visible;
- both fixed rails remain stable while the centre moves;
- compact and large terminal sizes preserve the approved hierarchy.

## ADR Check

ADR required: no.

ADR path: `backlog/decisions/042-watchlists-reader-first-ia.md` (existing).

Reason: this design implements a contained layout and pagination refinement
inside ADR-042's existing Read-tab region ownership and service boundaries. It
does not change storage, sync policy, data ownership, provider/runtime
boundaries, security policy, dependencies, or long-lived application structure.
