# Watchlists And Client Notifications Tranche

Date: 2026-04-21

Source spec: `Docs/superpowers/specs/2026-04-21-watchlists-notifications-vertical-design.md`

Status: first-slice landed, with broader watchlists execution surfaces still deferred.

## Landed Scope

- persisted local notifications store with read and dismiss lifecycle
- notification dispatch pipeline that records local queue entries and attempts toast or fallback delivery
- server watchlist source schemas and API-client methods
- local and server watchlist services plus source-aware scope routing
- app bootstrap wiring for watchlists and client notifications services
- backend-aware subscriptions shell behavior in `SubscriptionWindow`
- source-aware list rendering, create/update/delete routing, and `Notifications` tab behavior in the subscriptions destination

## Explicitly Deferred

- watchlist groups CRUD
- watchlist jobs and runs
- watchlist alert rules
- dedicated restore UI for reversible server deletes
- remote reminders and server notification-feed client surfaces
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
  Tests/Subscriptions/test_server_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/UI/test_screen_navigation.py \
  Tests/UI/test_subscription_window_watchlists.py -q
```

Result:

- `54 passed in 5.53s`

## Outcome

Chatbook now has a credible standalone-first watchlists crosswalk:

- `local` mode stays backed by local subscriptions and local notifications
- `server` mode supports live remote watchlist source CRUD
- local notification state stays Chatbook-owned even when triggered by server-mode actions

The remaining work for this domain is no longer first-slice CRUD alignment. It is the broader server watchlists execution and control-plane surface plus any future sync design.
