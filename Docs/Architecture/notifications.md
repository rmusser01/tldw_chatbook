# Notifications

This document describes the client notification subsystem: the inbox dispatch path (persist + optional toast), the local inbox store and its Watchlists surface, the producer categories, the server parity feed, and the honest state of notification settings, and the separate opt-in Console desktop channel.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Dispatch | `Notifications/notification_dispatch_service.py` | `NotificationDispatchService.dispatch` — the shared path inbox producers use |
| Store | `Notifications/client_notifications_db.py` | `ClientNotificationsDB(BaseDB)` — SQLite inbox; lazy schema on first feature touch; per-thread held connections (file) / one shared connection (`:memory:`) |
| Service | `Notifications/client_notifications_service.py` | `ClientNotificationsService` — policy-enforced queue reads/updates |
| Scope | `Notifications/notifications_scope_service.py` | local-vs-server routing; `ServerEventScopeRequiredError` |
| Server parity | `Notifications/server_notifications_service.py`, `server_notification_events.py`, `event_observer.py`, `event_cursor_store.py`, `event_state_repository.py`, `local_event_producer.py` | server notifications/reminders via `TLDWAPIClient`; durable SSE observation with cursor/dedupe and presentation projection |
| Presentation | `Notifications/notification_presentation.py` | process-local per-event-key delivery/read/dismiss states |
| UI | `UI/Watchlists_Modules/notifications_pane.py`, `UI/Subscription_Modules/notifications_inbox_controller.py` | the Watchlists notifications section + its policy-routed controller |
| Toast helper | `Utils/NotificationHelper.py` | `show_notification` |

Wiring: `app_service_wiring.py` builds the DB (`get_notifications_db_path()`, `:memory:` fallback on failure), the service, and the dispatch service. DB path: `<user_data>/<profile>/tldw_chatbook_notifications.db` (cli setting `notifications_db_path`).

## Dispatch (dataflow)

1. A producer calls `dispatch(app=…, category=…, title=…, message=…, severity=…, source_*…, payload=…)`.
2. Policy gate: `notifications.dispatch.launch.local` — denial raises `PolicyDeniedError`.
3. Settings gate: global `enabled` AND per-category `category_preferences[category].enabled`; the effective `toast_enabled`/`persist_enabled` are ANDed. Disabled returns a skipped result with a reason — nothing written, no toast.
4. `persist_enabled` inserts the inbox row; a provided app plus `toast_enabled` shows a transient toast (`show_notification`, falling back to `app.notify(..., markup=False)` — never Rich markup).
5. Every producer-side dispatch failure is swallowed: a dead notifications store must not fail the job or session it reports on.

### Producer categories (verified)

| Category | Producer |
| --- | --- |
| `briefing` | scheduled briefing completion/failure (with failure-incident dedupe) |
| `automation_run_succeeded/failed/timed_out/skipped` | automation handler, gated by the run's notification policy |
| `guardian` | guardian trend handler |
| `evaluations` | local evaluations service |
| `media_ingestion` | reading service |
| `research` | local research service |
| (reminder bridge) | reminder dispatch posts through the same wiring |

## Inbox UI (Watchlists Notifications section)

Section switch → refresh worker (group `wc_notifications`) → controller `load_rows` behind policy `notifications.queue.list.local` → `list_notifications(limit=100)`; rows render Status/Title/Category/Severity/Created with control characters stripped and local-zone timestamps. Mark-read/dismiss run behind policy `notifications.queue.update.local` with `exit_on_error=False` (a corrupt DB must not kill the app). Row selection feeds the screen's scope reconciliation. Home's "Review notifications" deep-links to this section, and its unread count is the same inbox filtered to `not is_read`.

The controller moves canonical file-backed `ClientNotificationsDB` list/read/dismiss
calls off the event loop with `asyncio.to_thread(run_finite_local_worker, ...)`,
after the existing policy gate. The finite-worker boundary retires only newly
acquired worker connections after their synchronous operation finishes, including
when the awaiting screen worker is cancelled. Borrowed handles and active
transactions retain their owners. In-memory stores keep their constructing-thread
connection; custom methods and awaitable test seams retain their original inline
invocation and awaiting behavior. Mark-read and dismiss use nonexclusive mutation
workers so each admitted native write retains its completion-to-refresh path when
another row is updated. Every successful mutation schedules the existing
latest-request-wins `wc_notifications` loader; exceptions keep the existing
nonfatal worker behavior. Cancelling a UI waiter cannot undo an already running
native write, so cancellation must not supersede these finite mutations.

## Server parity feed

`NotificationsScopeService` defaults to the server backend when a mode is configured; the local queue advertises `notifications.reminders.local` as unsupported (reminders are server-authoritative). The Schedules workbench runs the SSE observation supervisor (documented exit modes, flat-backoff restart, per-exception-class log throttling, `exit_on_error=False`); replay metadata (`retention_gap`, `server_refetch_required`) surfaces as Home's "Requery server events" state.

## Settings state (honest)

Store-level settings (`enabled`, `toast_enabled`, `persist_enabled`, per-category preferences; defaults all-true) live in the DB settings table — **not** config.toml — and **no UI toggles them anywhere on this tree**: enabling/disabling notifications currently requires direct DB writes. The inbox controller's preference methods exist but are unused by any screen. Only the DB path is exposed in Settings.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Store unavailable/corrupt | Dispatch swallowed by the producer; inbox loads fail to an error toast; the app continues |
| `persist_enabled=false` + toast on | Toast shows, no inbox row — consumers must not assume a row exists |
| Policy denial | Dispatch raises; producers that catch it drop the notification |
| Server feed disconnect | Supervisor restarts with backoff; Home shows reconnect states |

## Governing docs

There is **no dedicated notification ADR**. Relevant decisions live inside ADR-018 (the Watchlists inbox surface), ADR-079 (the `briefing` category), ADR-033 (state ownership), ADR-067 (approval-related waits). Design anchors are in-code task tags (lazy schema, connection lifecycle, `exit_on_error`, failure-incident dedupe, markup-free toasts).

## Verified gotchas

1. Silent notification loss on store failure is the **designed** degrade — do not add retry loops at call sites.
2. Toast text concatenates `title: message` and renders `markup=False`; Rich markup in titles is inert in toasts but **live** in Home rail buttons (escaped at build there).
3. `:memory:` inbox uses one shared connection; file-backed is per-thread — Home's exact-type checks gate the threading, and any test double silently forces inline reads.
4. Category settings are per-category trios inside one JSON overlay, not rows — `update_settings` rejects unknown keys.

## Console OS notification channel (ADR-282)

`Utils/os_notifications.py` owns the fixed-argv macOS `osascript` / Linux
`notify-send` dispatcher. Windows and other unsupported platforms skip with a
debug record. `ChatScreen._notify_console_run_outcome`,
`_notify_console_run_failure`, and `_park_console_approval` add this channel at
the existing once-guarded toast boundaries; queue-chain completion stays coalesced.
No focus detection, retry, or durable event replay is introduced.

Canonical Settings -> Console Behavior persists `[console.os_notifications]`:
`enabled=false` by default, with `run_finished`, `run_failed`, and `needs_approval`
independently enabled by default. These preferences are separate from the inbox's
DB-owned settings above; desktop dispatch creates no inbox record.

Only bounded conversation/workspace display names and a fixed outcome category
enter the body. AppleScript quoting and Linux XML escaping keep those names
literal. The child receives PATH and Linux D-Bus/XDG runtime routing variables,
never the inherited provider-key environment. Nonzero exits and spawn/timeout
failures return False and log category plus return code/exception type, without
argv, identity, output, or exception text.

A non-daemon one-shot thread keeps the UI event loop free and retains the
3-second subprocess wait/kill/reap path through orderly interpreter shutdown.
Process exit may wait for the remaining invocation; platform process creation
and forced termination are outside that timeout guarantee. No app/profile
resource is retained by the callback. See [ADR-282](../../backlog/decisions/282-console-os-notification-dispatch.md)
for the separation from the inbox and configurable run hooks, platform limits,
and the shutdown tradeoff.
