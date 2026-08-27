# ADR-042: Watchlists reader-first information architecture

Status: Accepted
Date: 2026-08-05
Related Task: [backlog/tasks/task-2513 - Watchlists-reader-first-re-IA-phase-1-reading-loop.md](../tasks/task-2513%20-%20Watchlists-reader-first-re-IA-phase-1-reading-loop.md)
Supersedes: N/A (amends ADR-018's pane set and section IA)

## Decision

Re-shape the Watchlists screen around the reading loop instead of operations management:

- **Read-first landing**: the tab-strip `items` tab is relabeled **Read**, reordered to
  first position, and becomes the default landing tab. The ops tabs (sources, runs,
  rules, notifications, artifacts, overview) remain but recede.
- **FEEDS region removed from the workbench**: its non-interactive `Static` rows are
  replaced by the rail tree carrying per-feed unread badges.
- **Inspector collapsed by default for new users**, so the article list and reading
  pane get the width on first run.
- **Scope-plumbed item queries**: picking any rail node (smart feed, watchlist, or
  source) scopes the items list via the existing `list_items(source_id=…)` seam.
- **Bulk mark-all-read** for the current scope, backed by a session undo batch so the
  operation is reversible.
- **New Read keymap**: `m` toggle read/unread, `a` mark all read, `u` undo the
  latest mark-all-read batch, and `space` next unread — single-letter bindings per
  ADR-031.

## 2026-08-23 amendment: permanent Reader and collapsible side panes

The approved
`Docs/superpowers/specs/2026-08-23-watchlists-netnewswire-reader-collapsible-rails-design.md`
finishes the reader-first IA and amends this decision as follows:

- **Reader is the permanent centre anchor on Read.** It is no longer independently
  collapsible. Navigation, Feed Items, and Inspector are the three collapsible
  side panes. On management tabs the existing active management canvas remains the
  permanent centre host; it is not removed or governed by Feed Items' preference.
- **Collapsed panes remain reachable.** Each side pane leaves a narrow,
  full-height, clickable ASCII grip whose arrow points in the action direction.
- **Preferred layout is distinct from effective layout.** Manual grip/key choices
  persist. Responsive collapses and the user-triggered Article Focus mode are
  transient derivations and never overwrite the preferred configuration.
- **Inspector preference is shared across all Watchlists tabs.** It does not gain
  per-tab copies of the same state.
- **Responsive policy protects Reader.** When minimum pane widths no longer fit,
  effective collapse priority is Inspector, Navigation, then Feed Items; an
  explicitly opened pane receives temporary priority and displaces lower-priority
  side panes instead.
- **Existing layout configuration is version-normalized.** Navigation, Feed Items,
  and Inspector choices survive; any persisted Reader/Content collapse is removed.
  The corrected layout and version advance atomically, or the migration retries.
- **Scope navigation commits with its snapshot.** Moving focus to another Navigation
  choice does not relabel old rows. The active scope/highlight changes only after its
  replacement rows load successfully, clears Reader selection, and leaves the empty
  Reader until the user explicitly activates an item.
- **Reading snapshots favor bounded, honest stability.** Pages use an
  effective-date/item-id keyset, an initial item-id high-water mark, and a seen-id
  guard. Mounted rows stay stable and new insertions wait behind the new-items
  affordance; unseen rows with publication metadata changed by a background upsert
  may move or wait for explicit refresh rather than requiring an unbounded
  materialized snapshot index.
- **Item mutation dimensions remain independent.** Status and star writes serialize
  per item but retain separate desired intents, so one action cannot cancel the
  other. Browser launch is scheme-validated and performed off the UI thread.
- **The implementation remains Watchlists-local.** It may reuse shared rail
  vocabulary, but an application-wide split-pane framework is deferred until the
  independently proceeding Media Library redesign provides a second proven shape.

## Context

The 2026-07-25 Watchlists console rebuild shipped a management console with a reader
bolted on. Verified against `origin/dev` (spec §"Starting point"):

- The reading loop (scope → items → catch-up) does not exist: `_load_items` never
  passes the tree scope, so picking a feed cannot show its articles, and the FEEDS
  region renders non-interactive rows.
- There is no mark-all-read and no next-unread; catch-up means opening every item.
- The supporting schema and service seams already exist (`subscription_items.content`,
  grouped unread counts, source-scoped `list_items`); the gap is wiring and
  presentation, not storage.

The goal is to make Watchlists usable as a daily-driver feed reader without
disturbing the ops surfaces built in the rebuild.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep the ops-console IA and add a separate reader screen | Splits one dataset across two destinations and strands the rebuild's content pane; the screen should be one place that leads with reading. |
| Build a new FEEDS pane with interactive rows instead of removing it | Duplicates the rail tree's job; per-feed unread badges in the tree cover the same need with one fewer region. |
| Wait for the full spec (smart feeds, FTS, flagging) before re-IA | The re-IA is the foundation the later phases wire into; landing it first keeps each phase reviewable. |

## Consequences

- Amends ADR-018's pane set and section IA: the seven-tab order and the five-region
  workbench it recorded no longer describe the shipped screen.
- Persisted `feeds` collapse state is silently dropped by the existing unknown-region
  guard in `region_layout_store.py:132-137`; no migration is needed or added.
- Section ids are unchanged, so deep links (e.g. Home notification-review navigation
  into the Notifications section) keep working.
- Phase 1 scope is the reading loop only; smart feeds, FTS search, flagging, and
  content rendering land in later phases of the same spec.
- The 2026-08-23 amendment supersedes the older implication that every workbench
  region, including Reader/Content, remains manually collapsible.

## Links

- Design spec: `Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md`
- NetNewsWire reader and collapsible rails amendment:
  `Docs/superpowers/specs/2026-08-23-watchlists-netnewswire-reader-collapsible-rails-design.md`
- Aggregate feed selection extension:
  `Docs/superpowers/specs/2026-08-25-watchlists-read-aggregate-feed-selection-design.md`
- Phase 1 plan: `Docs/superpowers/plans/2026-08-05-watchlists-reader-first-phase-1-reading-loop.md`
- Amends: `backlog/decisions/018-watchlists-tui-screen.md`
- Keybinding conventions: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`
