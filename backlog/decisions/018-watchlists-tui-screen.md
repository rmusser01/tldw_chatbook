# ADR-018: Watchlists TUI Screen

Status: Proposed
Date: 2026-07-18
Related Task: Watchlists module+screen redesign
Supersedes: N/A

## Decision

Replace the placeholder `watchlists_collections` destination shell with a full, three-pane Watchlists management screen that reuses the existing `WatchlistScopeService` local/server split, adds local scraped-item and content-alert storage, and adapts the layout and information architecture from `tldw_server`'s web UI Watchlists page.

## Context

The Chatbook currently has:

- A placeholder `watchlists_collections_screen.py` destination shell that only stages a local snapshot for Console.
- A legacy `SubscriptionWindow.py` with tabbed watchlist management that we want to retire.
- A mature backend seam: `LocalWatchlistsService`, `ServerWatchlistsService`, and `WatchlistScopeService` already support source/run/health-alert CRUD.
- A rich reference implementation in `tldw_server` (`apps/packages/ui/src/components/Option/Watchlists`) with Overview, Sources, Items, Runs, Alerts, Jobs, Outputs, and Templates tabs.

The goal is to give users a single screen where they can monitor sources, view/consume items, inspect runs, and manage alert rules. The screen must work both offline (local `SubscriptionsDB`) and against a connected `tldw_server` API.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Build a single monolithic screen class | Would become a very large file, hard to test, and would entangle list/detail/form logic for four entity types. |
| Adopt the web UI's top tab bar directly in Textual | The Chatbook shell already uses a destination-workbench pattern with a left rail; mixing top tabs inside a destination would be inconsistent. |
| Server-only implementation | Would not work offline and would waste the existing local service seam. |
| Keep `SubscriptionWindow` and add a new screen side-by-side | Per user direction, the old window is being retired and its route folded into the new screen. |

## Consequences

- Extend existing `SubscriptionsDB` tables rather than create parallel ones:
  - `subscription_items` gains `queued_for_briefing` and `run_id` columns for the item reader.
  - `subscription_filters` is reused for source-level filters and local-only content-alert rules (`action='notify'`).
  - `local_watchlist_alert_rules` remains dedicated to run-health alert rules.
- New cross-module UI package: `tldw_chatbook/UI/Watchlists_Modules/` with focused pane/controller modules.
- The `subscriptions` legacy route becomes an alias for `watchlists_collections`.
- `SubscriptionWindow.py` is removed after the new screen is wired and tested.
- Jobs, Outputs, and Templates are intentionally deferred to keep the first slice bounded; the design preserves space for them.

## Links

- Design spec: `docs/superpowers/specs/2026-07-18-watchlists-tui-screen-design.md`
- Reference module: `rmusser01/tldw_server` (`apps/packages/ui/src/components/Option/Watchlists`, `tldw_Server_API/app/core/Watchlists`)
