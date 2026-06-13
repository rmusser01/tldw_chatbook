# Watchlists And Client Notifications Tranche

Date: 2026-04-21

Source spec: `Docs/superpowers/specs/2026-04-21-watchlists-notifications-vertical-design.md`

Status: source CRUD, the first server control-plane slice, server reminders/feed preferences, and local notification filtering/preferences are landed, with groups, richer outputs, richer feed UX, and sync still deferred.

## Landed Scope

- persisted local notifications store with read and dismiss lifecycle
- filtered local notification queue reads
- local notification delivery preferences for global delivery enablement plus muted categories/severities
- notification dispatch pipeline that records local queue entries and attempts toast or fallback delivery
- delivery suppression for muted local notification categories/severities while preserving queued rows
- policy-gated local notification preference list/configure controller actions
- server watchlist source schemas and API-client methods
- server watchlist source restore API/client/service/UI routing
- server watchlist jobs API/client/service/scope/UI routing for list, create/update, delete, restore, and trigger
- server watchlist runs API/client/service/scope/UI routing for global/per-job list, detail, and cancel
- server watchlist alert-rule API/client/service/scope/UI routing for list, create/update, and delete
- server notification-feed preference get/update service and policy scope routing
- local and server watchlist services plus source-aware scope routing
- app bootstrap wiring for watchlists and client notifications services
- backend-aware subscriptions shell behavior in `SubscriptionWindow`
- source-aware list rendering, create/update/delete/restore routing, server `Jobs`, `Runs`, and `Alert Rules` tabs, and `Notifications` tab behavior in the subscriptions destination

## Explicitly Deferred

- watchlist groups CRUD
- richer structured editors for jobs and alert rules beyond the current JSON payload editor
- richer run outputs, logs, artifact/audio summaries, and historical/live monitoring UX
- richer remote reminders and server notification-feed UX beyond the first landed server surface
- long-running server notification stream worker controls
- local/server sync or mirror behavior

## Verification

Focused verification was run against the watchlists and client-notifications slice with:

```bash
python3 -m pytest \
  Tests/RuntimePolicy/test_runtime_policy_core.py \
  Tests/tldw_api/test_watchlists_schemas.py \
  Tests/tldw_api/test_watchlists_client.py \
  Tests/Subscriptions/test_client_notifications_db.py \
  Tests/Subscriptions/test_notification_dispatch_service.py \
  Tests/Subscriptions/test_notifications_inbox_controller.py \
  Tests/Subscriptions/test_server_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/UI/test_screen_navigation.py \
  Tests/UI/test_subscription_window_watchlists.py -q
```

Result:

- initial first-slice record: `54 passed in 5.53s`
- local notification filtering/preferences focused suites: `10 passed, 1 warning in 0.82s`; runtime-policy registry suite: `16 passed in 0.26s`
- server notification preference scope focused suite: `5 passed in 0.42s`; runtime-policy registry suite: `16 passed in 0.26s`
- current control-plane focused suites: see the latest branch verification for `Tests/tldw_api/test_watchlists_client.py`, `Tests/Subscriptions/test_server_watchlists_service.py`, `Tests/Subscriptions/test_watchlist_scope_service.py`, and `Tests/UI/test_subscription_window_watchlists.py`

## Outcome

Chatbook now has a credible standalone-first watchlists crosswalk:

- `local` mode stays backed by local subscriptions and local notifications
- `server` mode supports live remote watchlist source CRUD, source restore, jobs, runs, and alert-rule administration
- local notification state stays Chatbook-owned even when triggered by server-mode actions
- server-only control-plane tabs show explicit local/offline guidance rather than pretending local subscriptions are server jobs
- local notification delivery configuration remains Chatbook-owned and can mute delivery without losing the local audit queue

The remaining work for this domain is no longer source CRUD, first control-plane alignment, or first local notification configuration. It is group management, richer job/run/alert-rule/feed UX, broader notification producers, and any future sync design.
