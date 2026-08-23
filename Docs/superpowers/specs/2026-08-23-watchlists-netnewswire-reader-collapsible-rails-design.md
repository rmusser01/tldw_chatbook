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

Phase 1 of the 2026-08-05 reader-first design is implemented:

- Read is the landing tab.
- Tree scopes drive item loading.
- Per-source unread counts, mark-all-read with undo, next-unread, and read/unread toggling exist.
- The Inspector starts collapsed for a new user.
- Navigation, Feed Items, Reader, and Inspector are represented by `RegionLayout` and rebuilt by
  `WatchlistsWorkbench` factories.
- The Reader safely converts captured HTML into readable text.

The remaining gaps are presentation and capability exposure:

- Feed Items is still a `DataTable`, not a contextual article list.
- There are no All Unread, Today, or Starred Smart Feeds.
- `subscription_items.is_flagged` and its index exist, but no star action or Starred query uses
  them.
- The FTS5 table and maintenance triggers exist, but Read has no scoped full-text search path.
- Reader can still be collapsed, even though the selected article is the purpose of the screen.
- Collapsed regions render generic headers instead of persistent edge grips.
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

On management tabs, Feed Items and Reader are not mounted. Navigation and Inspector retain the
same preferred state and the same grip behavior, so the Inspector never becomes a Read-only
affordance.

The first-run preferred layout is:

- Navigation open.
- Feed Items open.
- Reader open.
- Inspector collapsed.

Reader empty-state copy is: **“Select a feed item to display it here.”** This deliberately says
“feed item”: selecting a feed in Navigation scopes the list, while selecting a feed item displays
content.

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
| Reader | `1fr` / 44 columns |
| Inspector | 34 / 30 columns |
| Each grip | 5 fixed columns |

Starting from the preferred layout, the resolver collapses side panes until the sum of Reader,
grips, and expanded-pane minimums fits. Default responsive collapse priority is Inspector,
Navigation, then Feed Items. Reader is never a candidate.

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

One shared scope-predicate builder must drive:

- item listing;
- unread and Smart Feed counts;
- mark-all-read;
- search;
- refresh eligibility;
- pending-arrival counting.

This is the guard against a badge saying one number while the list shows a different population.

Publication dates are mixed timezone-aware and naive strings. One shared effective-date helper
parses defensively, treats naive values according to the existing Watchlists convention, falls back
to `created_at`, converts to local time, and supplies Today/Yesterday/calendar groups. Today counts
and row grouping must call the same helper. A bounded SQL prefilter may reduce candidates, but
Python performs the final local-day classification.

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

Each scope load is a stable, paginated snapshot. New arrivals do not reorder mounted rows. A
scope-specific item-creation high-water mark—not unread-count changes—produces a **“N new items”**
affordance. Explicit refresh replaces the snapshot.

Opening an unread item marks it read but pins that selected row in an Unread view until the user
navigates away or refreshes. It cannot disappear under the cursor.

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
rebuild must not manufacture a second user-selection event. Per-item read/star mutations are
serialized and deduplicated so repeated keys or clicks cannot complete out of order.

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

Refresh checks active, non-paused sources in the current scope through the existing run pipeline,
skips sources already running, caps concurrency, and produces one aggregate completion notice.
It does not create per-source notification noise.

## State transitions and data flow

The primary flow is:

```text
scope gesture
  → resolve ReaderScope
  → load counts and stable item snapshot in a worker
  → atomically commit scope + rows
  → select a row
  → render Reader
  → serialize mark-read intent
  → patch row and counts after success
```

A failed scope change never shows old rows under a new heading. The current scope remains active
until the replacement load succeeds. Failure copy names both facts, for example: **“Couldn't open
Today; still showing All Unread.”**

Background arrivals update counts and the new-items affordance in place. They do not trigger a
screen-wide recompose. Layout changes may rebuild factories, but all user position is restored from
screen-owned semantic state.

## Persistence and migration

No database migration is required.

Watchlists layout configuration becomes versioned. The migration:

- preserves saved Navigation, Feed Items, and Inspector values;
- discards any saved Reader/Content collapse because Reader is no longer collapsible;
- removes unknown retired region names;
- writes normalized layout values and the new version in one configuration update.

If normalization cannot be persisted, the safe normalized layout still applies in memory and the
migration version remains old so the next launch retries. The implementation must not repeat the
earlier CONTENT-migration failure mode where a marker could advance without the corrected layout
being durable.

Only preferred-layout grip/key gestures write configuration. Responsive recalculation, Article
Focus, background refresh, tab switching, and selection changes never do.

## Failure handling

- **Scope load fails:** keep the previous scope/list active, show attempted-scope failure and Retry.
- **Read/star write fails:** keep prior visual state and selection; show a concise action-specific
  error. Do not apply a success-looking optimistic state that might not roll back cleanly.
- **Repeated item action:** coalesce through the per-item intent queue; latest valid intent wins.
- **FTS fails:** bounded off-thread scoped fallback with a visible notice.
- **Refresh partially fails:** keep successful runs and report one aggregate result with failure
  count; Runs retains detailed failures.
- **Missing body:** show an honest “No body was captured for this item” message.
- **Browser open fails:** preserve Reader and notify. Accept only supported `http`/`https` URLs and
  call a browser API directly, never a shell.
- **Config write fails:** retain the in-memory choice for the session, notify only when useful, and
  retry persistence on a later manual change or next migration load.

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

## Verification

### Pure state and persistence

- Independent preferred toggles and exact restart round-trip.
- Versioned normalization, including unknown/Reader values and failed atomic writes.
- Effective layout at width boundaries derived from declared minimums.
- Responsive priority Inspector → Navigation → Feed Items.
- Explicit reopening of each responsively hidden pane.
- Repeated shrink/expand cycles and Article Focus exact restoration.

### Database and services

- Shared scope predicates make counts, list, search, bulk-read, refresh, and arrival detection agree.
- All Unread, Today, Starred, Watchlist, Source, All Sources, and Unassigned semantics.
- Mixed aware/naive/missing/future publication dates.
- Star persistence across item re-fetch/upsert.
- Safe FTS query construction and bounded fallback pagination.
- Creation high-water mark ignores read/unread transitions.

### Widgets and integration

- Literal five-column ASCII grip geometry and correct arrow direction in every state.
- Pointer, focus, tooltip, keyboard, and accessibility behavior.
- Shared Inspector preference across all seven Watchlists tabs.
- New-user defaults and existing-user migration.
- Contextual rows, date groups, pinning, pagination, search, stable arrivals, and empty states.
- Cursor/visible-row offset, Reader scroll, selected item, focus, and filter survival across every
  manual and responsive layout change.
- Auto-read occurs on genuine selection but not on restoration; serialized mutation order is
  deterministic.
- Hostile remote markup and invalid browser URLs fail safely.

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
