# Watchlists NetNewsWire Reader and Collapsible Rails Design

Date: 2026-08-23

Status: Approved

Amends: [Watchlists reader-first re-IA](2026-08-05-watchlists-reader-first-design.md)

ADR: [ADR-042](../../../backlog/decisions/042-watchlists-reader-first-ia.md)

## Summary

Finish the Watchlists Read experience as a terminal-native, NetNewsWire-shaped reader:

- Navigation combines Smart Feeds with the Folder/Watchlist/Feed hierarchy.
- Feed Items becomes a contextual, date-grouped article list rather than an operations table.
- Reader is the permanent centre anchor and receives every column reclaimed from side panes.
- Navigation, Feed Items, and Inspector collapse independently through narrow, full-height,
  clickable ASCII grips.
- Inspector has one shared open/collapsed preference across every Watchlists tab.
- Manual layout choices persist across restarts; responsive and Article Focus layouts are
  temporary views that never overwrite those choices.

This design evolves the existing four-region Watchlists workbench. It does not build a second
Read-only shell and does not introduce an app-wide split-pane framework while the Media Library
redesign is proceeding independently. The Watchlists implementation keeps clean state and widget
interfaces so common behavior can be extracted after both real consumers exist.

## Current state

The reader-first phases and nested-pagination follow-up are implemented on the approved
`origin/dev` base:

- Read is the landing tab.
- Tree scopes drive item loading.
- Per-source unread counts, mark-all-read with undo, next-unread, and read/unread toggling exist.
- The Inspector starts collapsed for a new user.
- Navigation, Feed Items, Reader, and Inspector are represented by `RegionLayout` and rebuilt by
  `WatchlistsWorkbench` factories.
- The Reader safely converts captured HTML into readable text.
- Read uses `ArticleListPane` with contextual rows, effective-date groups, 50-row pagination, and
  separate list/Reader scroll ownership.
- All Unread, Today, and Starred Smart Feeds, scoped search, refresh-all/new-items feedback,
  star/unstar, and safe Open in browser actions are shipped.

The remaining gap is the pane-layout foundation:

- Reader can still be collapsed, even though the selected article is the purpose of the screen.
- Feed Items and Reader are vertically stacked inside a scrollable centre rather than arranged as
  NetNewsWire-style list and Reader columns.
- Collapsed regions render generic 16-column headers instead of persistent five-column edge grips.
- Responsive layout and persisted manual preference are not distinct state concepts.

## Goals

- Make the daily loop fast: choose a scope, scan useful rows, read, star or catch up, continue.
- Keep Reader continuously present, including when nothing is selected.
- Let users independently reclaim Navigation, Feed Items, and Inspector width.
- Preserve manual choices across Watchlists tabs and application restarts.
- Keep narrow terminals usable without silently changing saved preferences.
- Preserve selection, scroll, focus, search, and stable-list position through every layout change.
- Reuse existing Watchlists services and schema where possible.

## Non-goals

- Redesigning Sources, Runs, Rules, Notifications, Artifacts, or Overview content.
- Changing item read/star state to server-synchronized or watchlist-specific state.
- Adding new database tables or columns.
- Reworking OPML folder round-tripping in this slice.
- Adding enclosure playback, images, or web-page rendering inside Reader.
- Creating a shared application-wide split-pane framework in parallel with Media Library work.
- Reimplementing or changing the shipped ArticleListPane, Smart Feed, scoped search, pagination,
  refresh, item-status, star, or browser-action data contracts except where layout regression
  compatibility requires it.

## Information architecture

The tab order remains Read, Sources, Runs, Rules, Notifications, Artifacts, Overview. Internal
section id `items` remains unchanged for deep links and existing navigation contracts.

Read has four spatial roles:

| Role | Behavior |
| --- | --- |
| Navigation | Collapsible. Smart Feeds above the existing Folder/Watchlist/Feed tree. |
| Feed Items | Collapsible. Contextual list for the selected scope. |
| Reader | Permanent. Selected item or the empty-state message. |
| Inspector | Collapsible. Shared preference across all Watchlists tabs. |

On management tabs, the Feed Items **role** and Reader are not mounted, but the centre host remains.
It renders the active Sources, Runs, Rules, Notifications, Artifacts, or Overview canvas between
Navigation and Inspector, exactly as shipped phase 1 currently uses `Region.ITEMS` as the
management-detail host. That management canvas is the tab's permanent centre anchor and does not
inherit Feed Items' collapsed preference. Returning to Read reapplies the saved Feed Items choice.
Navigation and Inspector retain the same preferred state and grip behavior on every tab, so the
Inspector never becomes a Read-only affordance and no management surface loses its host.

The first-run preferred layout is:

- Navigation open.
- Feed Items open.
- Reader open.
- Inspector collapsed.

Reader empty-state copy is: **“Select a feed item to display it here.”** This deliberately says
“feed item”: selecting a feed in Navigation scopes the list, while selecting a feed item displays
content.

### Backend behavior

Read remains honestly local-only. If the active Watchlists backend is Server, the centre area shows
the existing local-only explanation plus **Switch to Local**. It does not issue unsupported item,
Smart Feed, count, search, or refresh queries and does not display local rows under a Server label.
Management tabs continue to use their existing backend behavior. Switching to Local performs a
normal scope load before replacing the centre state.

## Pane grips

Each collapsible pane has a five-column, full-height grip: four ASCII label columns plus its
divider. The label remains horizontal and is centred vertically. The browser brainstorm mockups'
rotated labels were illustrative only; Textual should render the literal ASCII label.

Direction describes what pressing the grip will do:

| Pane | Collapsed grip | Expanded inside-edge grip |
| --- | --- | --- |
| Navigation | `--->` expands right | `<---` collapses left |
| Feed Items | `--->` expands right | `<---` collapses left |
| Inspector | `<---` expands left | `--->` collapses right |

The grip itself is the intended pointer affordance in both states. It is also focusable, has a
plain-language tooltip naming the pane and action, and exposes a useful accessibility label even
though its visible copy is only an arrow. Focus styling must remain visible without increasing
the grip's geometry.

Reader is not a `RegionLayout` toggle target. A focused Reader cannot be collapsed by `z`, and
help/footer surfaces must not advertise an unavailable Reader-collapse action.

## Preferred, effective, and Article Focus layouts

Layout has three explicit layers:

1. **Preferred layout** — the user's manual Navigation, Feed Items, and Inspector choices. Grip
   clicks and the corresponding keyboard actions update this state and persist it.
2. **Responsive override** — extra collapses required by current terminal width. It is recomputed
   on resize and never persisted.
3. **Article Focus override** — a user-triggered transient mode that collapses all three side panes
   and restores the exact pre-focus preferred layout when toggled off. It is never persisted.

The rendered effective layout is derived from those layers. It is not itself saved.

### Width calculation

Responsive behavior is based on declared component minimums rather than unrelated breakpoint
numbers:

| Component | Target/minimum width |
| --- | --- |
| Navigation | 28 / 24 columns |
| Feed Items | 40 / 32 columns |
| Primary centre (Reader or management canvas) | `1fr` / 44 comfort columns |
| Inspector | 34 / 30 columns |
| Each grip | 5 fixed columns |

Starting from the preferred layout, the resolver collapses side panes until the sum of the primary
centre, mounted grips, and expanded-pane minimums fits. On Read, default responsive collapse
priority is Inspector, Navigation, then Feed Items. On a management tab, Feed Items and its grip
are absent, so the order is Inspector then Navigation. The Reader or management canvas is never a
collapse candidate.

The centre's 44 columns are a **comfort threshold**, not a hard CSS minimum. After all mounted side
panes are collapsed, a terminal narrower than centre comfort plus the fixed grips keeps those grips
and lets the centre consume the remaining width with `min-width: 0`; no grip disappears and no
horizontal overflow is introduced. The supported live-verification floor is 60 columns: Read has
three grips consuming 15 and gives Reader 45; management tabs have two grips consuming 10 and give
their canvas 50. Below the supported floor the centre remains structurally reachable but may
truncate content like the rest of the application; it must still avoid a compositor exception or
horizontal overflow.

If the user explicitly opens a responsively hidden pane, that pane becomes the temporary priority
target at the current width. The preferred state is updated to open and persisted; the effective
resolver collapses lower-priority side panes as necessary. On later expansion, the full preferred
layout naturally returns. Repeated shrink/expand cycles must be idempotent.

Clicking any grip while Article Focus is active first exits Article Focus, restores its baseline,
then applies the requested manual toggle. This prevents a transient mode from owning a second,
ambiguous set of preferences.

## Component ownership

`WatchlistsCollectionsScreen` remains the single controller and owns:

- normalized reader scope;
- pending scope while a replacement snapshot loads;
- selected item id and selected item;
- preferred layout, responsive priority target, and Article Focus state;
- Feed Items search/filter/page state;
- Feed Items selection anchor and Reader scroll position;
- stable-snapshot high-water mark and pending-arrival count;
- shared Inspector preference;
- in-flight per-item action intents.

`WatchlistsWorkbench` renders the permanent Reader and optional side panes from factories. It
accepts the effective layout and emits pane-toggle messages. It performs no persistence, database
queries, or responsive policy decisions.

A Watchlists-local grip widget owns arrow direction, focus/pointer behavior, tooltip, and message
emission. It may reuse established `DestinationRailHandle` vocabulary, but this slice must not
reshape the shared rail framework or touch Media Library layout code. Its public constructor and
toggle message should remain destination-neutral enough for later comparison and extraction.

## Navigation and normalized scope

Smart Feeds mount above the existing tree:

- **All Unread** — `status = 'new'` within all visible local items.
- **Today** — effective date falls on the current local calendar day; ignored/error items are
  excluded.
- **Starred** — `is_flagged = 1`; ignored/error items are excluded.

Tree scopes remain All Sources, Unassigned, Watchlist, and Source. Smart-feed and tree selections
produce one normalized `ReaderScope`; selecting either clears the visual selection in the other.
There is never a separate “smart scope” and “tree scope” active at once.

One normalized `ReaderScope` contract exposes two related but distinct projections.

Its **item predicate** drives:

- item listing;
- unread and Smart Feed counts;
- mark-all-read;
- search;
- pending-arrival counting.

This is the guard against a badge saying one number while the list shows a different population.

Its **refresh source universe** drives which sources are checked and never derives eligibility from
items that already match the item predicate:

- Source refreshes that active, non-paused source.
- Watchlist refreshes its active, non-paused member sources.
- Unassigned refreshes active, non-paused unassigned sources.
- All Sources refreshes all active, non-paused local sources.
- All Unread, Today, and Starred also refresh all active, non-paused local sources, because an empty
  Smart Feed must still be able to discover its first matching item.

Search text and the Unread/All display filter never narrow the refresh source universe.

Publication dates are mixed timezone-aware and naive strings. One shared effective-date helper
parses defensively, treats naive values according to the existing Watchlists convention, falls back
to `created_at`, converts to local time, and supplies Today/Yesterday/calendar groups. Today counts
and row grouping must call the same helper. A bounded SQL prefilter may reduce candidates, but
Python performs the final local-day classification.

The item-query service also exposes a canonical UTC effective-date sort key for keyset pagination.
That database-comparable projection and the Python helper are two projections of the same contract:
parseable `published_date`, otherwise parseable `created_at`, otherwise the deterministic item-id
fallback. Their ordering parity is covered by the same aware, naive, date-only, missing, and malformed
timestamp fixtures; the implementation must not use raw mixed-format string ordering.

## Feed Items

Read replaces the operations `DataTable` with a list-oriented pane. Each article row uses three
lines:

1. unread dot, source, and humane effective time;
2. title, bold while unread;
3. a short plain-text snippet.

Starred and queued-for-briefing markers appear without adding another row. Rows group under Today,
Yesterday, then calendar-date headers. Newest effective date sorts first.

The pane header contains:

- current scope and item/unread count;
- Unread/All toggle;
- scoped search affordance;
- Refresh scope;
- Mark all read.

“All” includes new, reviewed, and ingested items; ignored and error items remain outside the normal
reading list. The existing one-batch `a`/`u` mark-all-read and undo behavior remains.

### Stable snapshot and restoration

Each scope load is a paginated reading snapshot ordered by effective date descending and then item
id descending as the deterministic tie-breaker. The snapshot records the maximum matching item id;
subsequent pages remain constrained to that high-water mark. Pages use a keyset cursor over the
effective-date/item-id pair rather than `OFFSET`, and the screen keeps a seen-id set so an item is
never mounted twice. This is required because marking an item read can remove it from the Unread
predicate while the user is still paging.

The practical snapshot guarantee is deliberately narrower than a database transaction held open for
the whole reading session:

- already mounted row order remains stable;
- rows inserted after the high-water mark do not enter the mounted list;
- no item id appears twice;
- an existing, not-yet-mounted row whose publication metadata changes during a background upsert may
  move to a later page or be deferred until explicit refresh.

An id watermark alone cannot freeze future effective-date order because the existing upsert path may
update `published_date` without allocating a new id. Strictly freezing every future page would require
materializing an unbounded id/sort-key index at scope load, which is disproportionate for this reader.
Rows with larger ids are arrivals, not part of the mounted snapshot. A scope-specific item-creation
high-water mark—not unread-count changes—produces a **“N new items”** affordance. Explicit refresh
replaces the snapshot.

The active Feed Items header reports counts bounded by the mounted snapshot's high-water mark.
Navigation and Smart Feed badges remain live. The new-items affordance explains the difference between
those live badges and the currently mounted snapshot; successful local read/star mutations patch all
affected visible counts without silently admitting new rows.

A successful **scope change clears Reader selection** and shows the empty state until the user
selects an item in the new scope. Only a rebuild of the same scope restores its selected item and
Reader position. This avoids carrying an item into a scope where it is not visible or no longer
matches the Unread filter.

An action that makes the selected item stop matching the active predicate pins that row until the
user navigates away or refreshes. This covers opening/marking an item read in All Unread and
unstarring an item in Starred; neither can disappear under the cursor.

Layout rebuilds restore Feed Items semantically:

- selected item id;
- selected row's visible offset within the viewport;
- loaded page count;
- search/filter state;
- focused child id.

Raw scroll coordinates alone are insufficient because three-line rows and date headers have
variable geometry.

## Reader

Reader keeps the existing safe article/change renderers and becomes a permanent flexing pane.
When an item is selected it shows:

- source eyebrow;
- title;
- author, effective date, and reading metadata when available;
- the safely rendered body;
- footer position and next-unread guidance.

The always-visible action row contains only the approved core actions:

- Star/Unstar (`s`);
- Mark unread/read (`m`);
- Open in browser (`o`).

Ingest, Queue/Unqueue for Briefing, and other advanced actions remain in Inspector. Reader and
Inspector call the same shared item-action helpers so status and star semantics cannot drift.

Selecting/opening a feed item automatically marks it read. Programmatic restoration after a layout
rebuild must not manufacture a second user-selection event. Per-item mutations are serialized so
repeated keys or clicks cannot complete out of order, but status and star retain independent desired
intents: toggling Star must not replace a pending automatic mark-read, or vice versa.

`m` changes only the reversible reader statuses `new` and `reviewed`. For `ingested`, `ignored`, or
`error`, the action is disabled/refused with an explanation and never rewrites the terminal
workflow status. This applies even though the All filter intentionally includes ingested items.

Reader scroll position survives pane toggles and responsive recomposition. Article Focus (`Z`) is
the explicit way to give Reader maximum width, and toggling it again restores the previous manual
layout exactly.

## Search and refresh

`/` focuses scoped full-text search over title, content, and author using the maintained FTS5
index. User input is converted to a safe parameterized MATCH expression; scope predicates remain
bound parameters.

If FTS is unavailable or corrupt, search falls back to a bounded, paginated, scope-aware substring
path running off the UI thread. The UI states that fallback search is active. It must not scan the
entire database on the event loop or search only the currently loaded page while implying global
scope coverage.

Refresh checks the normalized scope's refresh source universe through the existing run pipeline,
skips sources already running, caps concurrency, and produces one aggregate completion notice. It
does not create per-source notification noise. Smart Feed refresh therefore remains useful when the
current Smart Feed has zero matching items.

## State transitions and data flow

Scope navigation and item activation are separate flows. Merely choosing a scope must not
automatically consume its first item:

```text
scope gesture
  → resolve ReaderScope
  → store pending scope; keep committed scope/list visibly active
  → load counts and stable item snapshot in a worker
  → atomically commit active navigation highlight + scope + rows
  → clear item selection and show Reader empty state

explicit item activation
  → select the row
  → render Reader
  → serialize mark-read intent
  → patch row and counts after success
```

A failed scope change never shows old rows under a new heading. The current scope remains active
until the replacement load succeeds. The attempted Navigation control may retain keyboard focus, but
the active-selection styling is restored to the committed scope. Failure copy names both facts, for
example: **“Couldn't open Today; still showing All Unread.”**

Background arrivals update live Navigation badges and the new-items affordance in place; they leave
the active Feed Items header's snapshot-bounded counts unchanged until explicit refresh. They do not
trigger a screen-wide recompose. Layout changes may rebuild factories, but all user position is
restored from screen-owned semantic state.

## Persistence and migration

No database migration is required.

Watchlists layout configuration becomes versioned. The migration:

- preserves saved Navigation, Feed Items, and Inspector values;
- discards any saved Reader/Content collapse because Reader is no longer collapsible;
- removes unknown retired region names;
- writes normalized layout values and the new version in one
  `save_settings_to_cli_config(...)` configuration mutation.

If normalization cannot be persisted, the safe normalized layout still applies in memory and the
migration version remains old so the next launch retries. The implementation must not repeat the
earlier CONTENT-migration failure mode where a marker could advance without the corrected layout
being durable.

Ordinary preference writes have the same durability rule. The in-memory “last persisted” marker may
advance only after the worker reports success. A failed write keeps the latest preferred layout
pending and restores retry eligibility so a later manual gesture can retry it; the current store's
pre-write bookkeeping and ignored worker result must not be carried forward.

Only preferred-layout grip/key gestures write configuration. Responsive recalculation, Article
Focus, background refresh, tab switching, and selection changes never do.

## Failure handling

- **Scope load fails:** keep the previous scope/list active, show attempted-scope failure and Retry.
- **Scope load is pending:** keep the previous active Navigation styling until the new snapshot
  commits; focus alone must not imply that old rows belong to the attempted scope.
- **Read/star write fails:** keep prior visual state and selection; show a concise action-specific
  error. Do not apply a success-looking optimistic state that might not roll back cleanly.
- **Repeated item action:** coalesce per item and mutation field; the latest valid status intent and
  latest valid star intent each win without replacing one another.
- **FTS fails:** bounded off-thread scoped fallback with a visible notice.
- **Refresh partially fails:** keep successful runs and report one aggregate result with failure
  count; Runs retains detailed failures.
- **Missing body:** show an honest “No body was captured for this item” message.
- **Browser open fails:** preserve Reader and notify. Accept only supported `http`/`https` URLs and
  call a browser API directly from a worker, never a shell or the UI event loop.
- **Config write fails:** retain the in-memory choice for the session, notify only when useful, and
  retry persistence on a later manual change or next migration load.

## Delivery isolation

Planning and implementation proceed in the dedicated
`codex/task-21281-watchlists-collapsible-reader-layout` branch at
`.worktrees/task-21281-watchlists-collapsible-reader-layout`, based on `origin/dev` commit
`527152ad3`. The approved Watchlists documentation was transplanted as commits `ade0a5ab7`,
`6af8780f5`, `4561c4199`, and `730d908d2`. The unrelated dirty
`feat/task-3401-video-generation-foundation` worktree remains untouched.

## Security and accessibility

- Remote titles, sources, snippets, categories, and bodies remain markup-safe at every Textual/Rich
  sink.
- FTS terms are sanitized and all scope values remain SQL parameters.
- Browser URLs are scheme-validated; no command interpolation occurs.
- Grips are reachable by pointer and keyboard, have action-specific tooltips and accessibility
  labels, and show a non-obscuring focus state.
- Footer and F1 help advertise only actions valid for the current tab and focused surface.
- Existing terminal-convention and global key reservations remain untouched.

## Keybindings

Read retains or adds:

| Key | Action |
| --- | --- |
| `j` / `k` | Next / previous visible item |
| `space` | Next unread while focus is in Feed Items |
| `m` | Toggle read/unread |
| `s` | Toggle star |
| `o` | Open in browser |
| `a` | Mark current scope read |
| `u` | Undo the latest mark-all-read batch |
| `r` | Refresh current scope |
| `/` | Focus scoped search |
| `z` | Toggle the focused collapsible pane or grip; unavailable on Reader |
| `Z` | Toggle transient Article Focus |
| `[` / `]` | Preserve existing left/right rail shortcuts where they remain unambiguous |

## Implementation decomposition

On the approved `origin/dev` base, the former contextual-list, Smart Feed/search, and
refresh/pagination slices are already shipped under tasks 3072, 3791, and their nested-pagination
follow-up. The remaining programme work is one atomic, independently verifiable slice:

1. **Layout foundation and grips.** Make Reader/management canvas the permanent centre host; add
   Watchlists-local ASCII grips; introduce preferred/effective/Article Focus state, responsive
   resolution, shared Inspector behavior, and versioned layout normalization. Preserve the shipped
   ArticleListPane, Smart Feed, search, refresh, pagination, item-action semantics, selection, and
   scroll contracts while changing their host geometry and simplifying Reader's visible action row
   to the approved three core actions.

This slice receives Backlog task 21281, its own detailed implementation plan, tests, and review.

## Verification

### Pure state and persistence

- Independent preferred toggles and exact restart round-trip.
- Versioned normalization, including unknown/Reader values and failed atomic writes.
- Failed ordinary preference writes remain eligible for a same-session retry.
- Effective layout at width boundaries derived from declared minimums.
- Responsive priority Inspector → Navigation → Feed Items.
- Management responsive priority Inspector → Navigation with only two mounted grips.
- Explicit reopening of each responsively hidden pane.
- All-grips-collapsed behavior from the 60-column supported floor through sub-floor degradation.
- Repeated shrink/expand cycles and Article Focus exact restoration.

### Database and services

- Normalized scope projections make item predicates agree across counts/list/search/bulk-read/
  arrivals while refresh uses the specified source universe.
- Refresh source-universe tests prove empty Smart Feeds still check every eligible local source.
- All Unread, Today, Starred, Watchlist, Source, All Sources, and Unassigned semantics.
- Mixed aware/naive/missing/future publication dates.
- Canonical database sort keys and Python effective-date parsing agree for every supported timestamp
  shape and never fall back to raw string ordering.
- Star persistence across item re-fetch/upsert.
- Safe FTS query construction and bounded fallback pagination.
- Creation high-water mark ignores read/unread transitions.
- Snapshot pagination uses an effective-date/item-id keyset, is bounded by the initial max id, never
  duplicates a seen id, and documents the behavior of mutable metadata on unseen rows.
- Active-pane snapshot counts, live Navigation badges, and the new-items affordance remain mutually
  intelligible.

### Widgets and integration

- Literal five-column ASCII grip geometry and correct arrow direction in every state.
- Pointer, focus, tooltip, keyboard, and accessibility behavior.
- Shared Inspector preference across all seven Watchlists tabs.
- Management tabs keep their centre canvas while Feed Items' preference remains parked for Read.
- Server-backed Read shows local-only recovery and issues no local-reader queries under Server.
- New-user defaults and existing-user migration.
- Contextual rows, date groups, Unread/Starred predicate pinning, pagination, search, stable arrivals,
  and empty states.
- Cursor/visible-row offset, Reader scroll, selected item, focus, and filter survival across every
  manual and responsive layout change.
- Auto-read occurs on genuine selection but not on restoration; serialized mutation order is
  deterministic.
- Scope loading never auto-selects or auto-reads the first row, and a failed load restores committed
  active-selection styling.
- Status and star intent coalescing cannot cancel each other's pending mutation.
- `m` never rewrites ingested, ignored, or error workflow statuses.
- Hostile remote markup and invalid browser URLs fail safely; browser launch runs off the UI thread.

### Production evidence

- Focused Watchlists, Subscriptions, database, and destination-shell suites.
- CSS source and generated bundle integrity.
- Production-style Textual renders at representative wide and narrow sizes, including exact width
  boundaries and both open/closed grip states.
- Live keyboard and pointer checks with an isolated profile. Pointer coordinates must be computed
  by character position, never UTF-8 byte position, per the repository's Watchlists UAT lesson.
- Full regression, lint/static checks, and documentation verification required by the implementing
  Backlog task before it can be marked Done.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/042-watchlists-reader-first-ia.md`

Reason: this is a long-lived amendment to the Watchlists pane structure and layout-state policy.
ADR-042 already owns the reader-first information architecture, so it is amended rather than
creating a competing decision record.
