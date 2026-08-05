# ADR-018: Watchlists TUI Screen

Status: Proposed
Date: 2026-07-18
Related Task: Watchlists module+screen redesign
Supersedes: N/A

## Decision

Replace the placeholder `watchlists_collections` destination shell with a full, three-pane Watchlists management screen that reuses the existing `WatchlistScopeService` local/server split, adds local scraped-item and content-alert storage, retains the client-owned local notifications inbox from `SubscriptionWindow`, and adapts the layout and information architecture from `tldw_server`'s web UI Watchlists page.

## Context

The Chatbook currently has:

- A placeholder `watchlists_collections_screen.py` destination shell that only stages a local snapshot for Console.
- A legacy `SubscriptionWindow.py` with tabbed watchlist management and the client-owned local notifications inbox that we want to retire without losing its user workflows.
- A mature backend seam: `LocalWatchlistsService`, `ServerWatchlistsService`, and `WatchlistScopeService` already support source/run/health-alert CRUD.
- A rich reference implementation in `tldw_server` (`apps/packages/ui/src/components/Option/Watchlists`) with Overview, Sources, Items, Runs, Alerts, Jobs, Outputs, and Templates tabs.

The goal is to give users a single screen where they can monitor sources, view/consume items, inspect runs, manage alert rules, and review local client notifications. Watchlist management must work both offline (local `SubscriptionsDB`) and against a connected `tldw_server` API; the notifications inbox remains explicitly local/client-owned.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Build a single monolithic screen class | Would become a very large file, hard to test, and would entangle list/detail/form logic for four entity types. |
| Adopt the web UI's top tab bar directly in Textual | The Chatbook shell already uses a destination-workbench pattern with a left rail; mixing top tabs inside a destination would be inconsistent. |
| Server-only implementation | Would not work offline and would waste the existing local service seam. |
| Keep `SubscriptionWindow` and add a new screen side-by-side | Per user direction, the old window is being retired and its route folded into the new screen. |
| Drop the notifications inbox with `SubscriptionWindow` | Home already exposes unread client-notification counts and routes users to review them; removing the inbox would leave that action without a truthful destination. |

## Consequences

- Extend existing `SubscriptionsDB` tables rather than create parallel ones:
  - `subscription_items` gains `queued_for_briefing` and `run_id` columns for the item reader.
  - `subscription_filters` is reused for source-level filters and local-only content-alert rules (`action='notify'`).
  - `local_watchlist_alert_rules` remains dedicated to run-health alert rules.
- New cross-module UI package: `tldw_chatbook/UI/Watchlists_Modules/` with focused pane/controller modules.
- The Watchlists navigator includes a **Notifications** section backed by the existing policy-aware `NotificationsInboxController` and `ClientNotificationsDB`; it is local/client-owned regardless of the selected Watchlists backend.
- The `subscriptions` legacy route becomes an alias for `watchlists_collections`.
- Home notification-review navigation targets the retained Notifications section through explicit one-shot navigation context.
- `SubscriptionWindow.py` is removed after the new screen is wired and tested.
- Jobs, Outputs, and Templates are intentionally deferred to keep the first slice bounded; the design preserves space for them.

## Links

- Design spec: `docs/superpowers/specs/2026-07-18-watchlists-tui-screen-design.md`
- Reference module: `rmusser01/tldw_server` (`apps/packages/ui/src/components/Option/Watchlists`, `tldw_Server_API/app/core/Watchlists`)
