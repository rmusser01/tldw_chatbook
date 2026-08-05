# ADR-042: Watchlists reader-first information architecture

Status: Accepted
Date: 2026-08-05
Related Task: [backlog/tasks/task-2511 - Watchlists-reader-first-re-IA-phase-1-reading-loop.md](../tasks/task-2511%20-%20Watchlists-reader-first-re-IA-phase-1-reading-loop.md)
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
- **New Read keymap**: `m` toggle read/unread, `a` mark all read, `u` next unread,
  `space` page the reading pane — single-letter bindings per ADR-031.

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

## Links

- Design spec: `Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md`
- Phase 1 plan: `Docs/superpowers/plans/2026-08-05-watchlists-reader-first-phase-1-reading-loop.md`
- Amends: `backlog/decisions/018-watchlists-tui-screen.md`
- Keybinding conventions: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`
