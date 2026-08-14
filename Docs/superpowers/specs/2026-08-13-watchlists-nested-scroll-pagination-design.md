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

- Replacing or redesigning the current reader-first `ArticleListPane` /
  `ListView` rows.
- Changing the existing corpus-wide FTS/LIKE search contract.
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
│   ├── article ListView (independently scrollable)
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
boundary; at the region cap, the `ListView` scrolls through the page while the
toolbar, legend, and pager remain visible.

The centre viewport owns movement between the Read list and Content. The
`ListView` owns scrolling among the current page's rows. The Content body owns
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

The pager is a compact, persistent row below the `ListView`:

- `Previous` is disabled on page 1.
- The centre label reads `Page N`.
- `Next` is disabled when no lookahead row exists.
- Both controls are disabled while a page request is in flight.
- No total-count query is added solely to display “of N”.

A page transition requested by `Previous` or `Next` is transactional from the
UI's perspective. The requested page index is committed only after its load
succeeds. A failure retains the current page, rows, selection, article, and
query context; it shows the existing error notification and restores the pager
controls. Rapid repeated activation cannot start overlapping page loads.

Changing backend, tree scope, source, status filter, or search text resets to
page 1. Refresh reloads the current page. If a mutation or refresh leaves a
non-first page empty, the screen loads the nearest preceding non-empty page.

The selected article remains in Content when the user changes pages, but it is
not injected into a page where it does not belong. Each selection records a
page key made from backend, tree scope, normalized status filter, normalized
search query, and page index. The current open-item pin is applied only when a
reload targets that exact key. An explicit page change or any query-context
change therefore preserves Content but invalidates the list pin. A same-key
refresh or mark-read-on-open reload keeps the pin so the open row does not
vanish from an Unread view. The visible page never exceeds 50 rows to preserve
that pin; if a same-key reload must carry the open item alongside 50 queried
rows, it replaces the page's last display slot rather than becoming a 51st
visible row.

`j`, `k`, and next-unread navigation operate only on the currently rendered
page. Reaching an edge does not change pages. Pagination is always an explicit
button action.

## Search Scope

The current Read search is corpus-wide: after its debounce, the screen passes
the query into the existing FTS/LIKE service path, while the live pane applies
an immediate in-memory filter to the rows already mounted. Pagination preserves
that contract. Changing the query immediately invalidates the committed query
context, sets the logical page to 1, disables both pager buttons, and re-arms
the debounce. During the debounce and load, the old rows remain mounted and are
narrowed in memory by the new query; Content remains unchanged. The 51-row
request then carries the search term with offset 0.

On a successful search load, the returned backend page becomes authoritative
for that query and the pane stops applying the in-memory query predicate to
those rows. This distinction is required because a corpus FTS match can occur
in full article content that is intentionally absent from the list projection;
filtering the returned page again from title, author, or preview text could hide
a valid backend match. The input value remains visible and later edits make the
local predicate provisional again until their own backend load succeeds.
If the user selects one of the provisionally filtered old rows during that
window, the selection records the mounted page's prior committed key, never the
pending search key, so the pending result cannot pin that row into a different
query.

On a failed search load, the screen remains logically on page 1, retains the
provisionally filtered old rows and Content, keeps `Next` disabled because the
new context has no successful lookahead yet, and re-enables `Previous` only as
false because page 1 has no predecessor. Refresh retries that same page-1 query.
The placeholder remains `Search items...` because search is not page-local.

## Component Responsibilities

### `WatchlistsWorkbench`

- Render `#wl-centre` with Textual's vertical scrolling container instead of a
  fixed `Vertical`.
- Preserve existing region factories, hidden-region gating, focusability,
  collapsed headers, and sole-centre class behaviour.
- Accept no new layout abstraction. The screen adds a Read-mode class to the
  existing workbench instance, which gives CSS the narrow selector it needs.

### `ArticleListPane`

- Receive current page number, `has_previous`, `has_next`, and loading state
  from the screen.
- Render the fixed pager below the existing queued legend.
- Post narrowly scoped previous/next request messages; hold no service and
  calculate no offsets.
- Keep toolbar, legend, and pager visible while only the `ListView` scrolls.
- Preserve the current reader rows, date headers, in-place filter/search
  painting, and corpus-wide search messaging.
- Track whether the mounted page is authoritative for the current search. A
  search edit restores provisional in-memory filtering; a successful backend
  search load disables that second predicate without clearing the input.
- Provide one programmatic focus operation that highlights the first selectable
  article row without posting `ItemSelected`.

### `ContentPane`

- Keep the existing renderers and action messages unchanged.
- Wrap the rendered article body in the internal vertical scroll owner so the
  title and action row remain visible at the 50-row cap.
- Keep the empty state within the 20-row region floor.

### `WatchlistsCollectionsScreen`

- Own page index, lookahead, and loading state so scoped region replacement
  cannot erase them.
- Own the committed page/query key for the list and the page key where the open
  article was selected.
- Convert pager messages into serialized 51-row requests.
- Reset or retain the page according to the pagination contract.
- Seed every replacement `ArticleListPane` with page and authoritative-search
  state in the same way it already seeds filters, selection, and loaded rows.
- Keep page navigation from clearing the selected Content item.

### Scope and data services

- Reuse existing `list_items(limit, offset, ...)` calls.
- Add no count method and make no storage change.
- Preserve status, scope, and search predicates, including the current
  open-item pin.

## States and Feedback

- **Empty scope:** the list remains 10 rows and shows its current empty-state
  copy. Pager reads `Page 1`; both buttons are disabled.
- **Loading another page:** current rows and Content remain visible. Pager
  buttons are disabled to prevent concurrent requests.
- **Successful explicit page load:** rows swap as one completed state update;
  focus moves to the `ListView` and its first selectable article row becomes the
  cursor target without posting `ItemSelected`. Content therefore remains on
  the previously open article until the user deliberately moves or activates
  the row. Ordinary refreshes preserve the current focus.
- **Failed page load:** current state remains visible and an error notification
  explains the failure.
- **Short article:** Content remains 20 rows; no internal scrollbar is needed.
- **Long article:** Content grows to 50 rows and then the article body scrolls.
- **Scoped region replacement:** page number, rows, filter/search state,
  selection, and Content survive collapse/solo changes because the screen
  reseeds a replacement pane. Inner scroll position may reset when its owning
  widget is actually unmounted; it remains stable for in-place class/layout
  updates that keep the widget mounted.
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
- Keep the lookahead row out of `ArticleListPane.items`, selection, navigation, and
  status mutation paths.
- Preserve the selected article even when it is absent from the new page.
- Apply the open-item pin only when the target backend/scope/filter/search/page
  key equals the key recorded when that article was selected.
- Suppress exactly the programmatic first-row highlight emitted after an
  explicit page success; do not suppress the user's next highlight.
- Keep explicit page navigation independent from `j`/`k`; a row-navigation edge
  must not unexpectedly issue I/O.
- Verify that inner scrollbars do not obscure pager buttons or article actions.

## Testing and Verification

### Unit and widget tests

- Empty and short Read lists render at a 10-row minimum.
- Medium lists grow naturally; large lists stop at 50 rows.
- Empty and short Content renders at 20 rows; long Content stops at 50 rows.
- The `ListView` and article body scroll independently at their caps.
- A short terminal scrolls the centre from the Read list to Content without
  shrinking either below its minimum.
- A 51-row response displays 50 rows and enables `Next`; a response of 50 or
  fewer disables it.
- `Previous`, `Page N`, and `Next` states are correct on first, middle, and last
  pages.
- Offsets are `0`, `50`, `100`, and so on; visible rows never exceed 50.
- Page state survives scoped region replacement and resets for every agreed
  context change.
- A failed explicit `Previous`/`Next` transition preserves rows, page number,
  selected item, and Content.
- An empty non-first page steps back to a valid page.
- Corpus-wide search resets to page 1 and sends the query with every paged load.
- A content-only FTS match remains visible after the backend page becomes
  authoritative, and a subsequent edit restores provisional local filtering.
- Programmatic post-page focus leaves the selected Content article unchanged;
  the next user-driven row move selects normally.
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
