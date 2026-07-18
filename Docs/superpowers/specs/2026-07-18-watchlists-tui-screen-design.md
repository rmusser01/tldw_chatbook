# Watchlists TUI Screen Design

Date: 2026-07-18  
Status: Draft (pending spec review)  
ADR: [backlog/decisions/018-watchlists-tui-screen.md](../../../backlog/decisions/018-watchlists-tui-screen.md)  
Reference: `rmusser01/tldw_server` — `apps/packages/ui/src/components/Option/Watchlists` and `tldw_Server_API/app/core/Watchlists`

## Summary

Replace the placeholder `watchlists_collections` destination shell with a full Watchlists management screen. The new screen uses a three-pane destination workbench, supports both local and server backends, and provides Overview, Sources, Items, Runs, and Alert Rules sections. Jobs, Outputs, and Templates are deferred to later slices.

## Goals

- Manage watchlist sources (RSS, site, forum) with create/edit/delete, active toggle, and groups/tags.
- View and consume scraped items with smart counts, batch review, and "queue for briefing".
- Inspect runs: status, stats, filter tallies, fetched items, logs, and cancel/retry actions.
- Create and manage both run-health and content alert rules, with alert notifications surfacing in the app inbox and Home dashboard.
- Support source preview / dry-run and "Check now" actions.
- Support OPML import/export for sources.
- Work offline against local `SubscriptionsDB` and online against a connected `tldw_server` API.
- Retire `SubscriptionWindow.py` and fold the `subscriptions` route into the new screen.

## Non-goals

- Jobs / scheduling UI (parallel Scheduling module migration).
- Briefing output generation and template authoring (deferred).
- Advanced source group editing (groups/tags remain read-only in the first slice).
- Forum sources if disabled by server policy or local capability flags.
- WebSocket live log streaming (poll run detail instead).

## Architecture

### Web UI → TUI mapping

The `tldw_server` web UI uses a top tab bar with primary views. The TUI maps those views into a left-rail section navigator plus a three-pane workbench.

```
┌─ Watchlists ─ [Local ▼] ─ [New Source] ─ [Refresh] ─────────────┐
│                                                                  │
│ ┌────────┐ ┌────────────────────────────────┐ ┌──────────────┐ │
│ │Overview│ │  Middle: Workbench             │ │ Right:       │ │
│ │Sources │ │                                │ │ Inspector /  │ │
│ │ Items  │ │  • Source list / form          │ │ Actions      │ │
│ │ Runs   │ │  • Item reader                 │ │              │ │
│ │ Rules  │ │  • Run inspector               │ │ Context-aware│ │
│ │        │ │  • Alert editor / inbox        │ │ buttons:     │ │
│ │        │ │                                │ │ Preview,     │ │
│ └────────┘ └────────────────────────────────┘ │ Check now,   │ │
│                                               │ Stage,       │ │
│                                               │ Mark read,   │ │
│                                               │ Delete, …    │ │
└───────────────────────────────────────────────┴──────────────┘
```

| Web UI tab | TUI section | First-slice scope |
|---|---|---|
| Overview | **Overview** | Summary cards, health alerts, recent failed runs, latest briefing placeholder |
| Feeds | **Sources** | Source CRUD, groups/tags read-only, OPML import/export, simple source-level filters, preview, Check now |
| Updates | **Items** | Global item reader with smart counts, batch review, queue for briefing |
| Activity | **Runs** | Run history, full run inspector (stats, items, tallies, logs), cancel/retry |
| Alerts | **Rules** | Health + content alert rule editor; alert inbox |
| Monitors | *deferred* | Jobs / scheduling |
| Reports | *deferred* | Outputs and templates |

### Module layout

| Path | Role |
|---|---|
| `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` | Thin shell; left rail, three panes, backend toggle, recovery state |
| `tldw_chatbook/UI/Watchlists_Modules/__init__.py` | Package marker |
| `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py` | Local/server authority switching, policy enforcement, unsupported-capability reporting |
| `tldw_chatbook/UI/Watchlists_Modules/overview_pane.py` | Dashboard cards, health summary, recent failed runs |
| `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py` | Source list, create/edit form, filter editor, preview, OPML import/export |
| `tldw_chatbook/UI/Watchlists_Modules/items_pane.py` | Item reader, smart counts, batch actions |
| `tldw_chatbook/UI/Watchlists_Modules/runs_pane.py` | Run list, full run inspector |
| `tldw_chatbook/UI/Watchlists_Modules/alert_rules_pane.py` | Rule editor + alert inbox |
| `tldw_chatbook/Subscriptions/watchlist_filter_service.py` | Local source-level filter evaluation |
| `tldw_chatbook/Subscriptions/watchlist_content_alert_service.py` | Local content-alert evaluation |
| `tldw_chatbook/Subscriptions/local_watchlists_service.py` | Extended for content alerts, filters, scraped-item persistence, OPML import |

### Backend split

```
WatchlistsBackendController
├── local backend
│   └── LocalWatchlistsService (extended)
│       └── SubscriptionsDB
└── server backend
    └── ServerWatchlistsService
        └── TLDWAPIClient
```

- Source CRUD, runs, and alert rules go through `WatchlistScopeService` (`app.watchlist_scope_service`).
- Server-only operations (OPML server endpoints, group/tag read-only listing) call `app.server_watchlists_service` directly when server backend is active.
- Local-only operations (content-alert evaluation, filter evaluation, OPML parsing) live in new local helper services.

## Data flow

1. Shell mounts → controller checks backend availability → panes load initial data via background workers.
2. User selects a left-rail section → shell updates `active_section` reactive → workbench swaps pane.
3. User selects an entity (source/run/item/rule) → shell updates `selected_entity` reactive → inspector pane updates.
4. Mutations (create/update/delete/preview/check now) → pane calls controller method → controller routes to local/server service → worker executes → reactive state updates → UI refreshes.
5. Run inspector opens a running run → starts a polling worker that refreshes run detail every 3–5 s until terminal.
6. Alert rules trigger notifications via the existing `NotificationDispatchService`; payloads use `source_domain="watchlists"` so Home and the notifications inbox can route them.

### Controller routing for backend-specific operations

| Operation | Local backend | Server backend |
|---|---|---|
| List sources | `LocalWatchlistsService.list_sources` | `ServerWatchlistsService.list_sources` |
| Source CRUD | `WatchlistScopeService.{create/update/delete}_watch_item` | `WatchlistScopeService.{create/update/delete}_watch_item` |
| Preview source | Local fetchers without persistence (new helper) | `TLDWAPIClient.test_watchlist_source` / `test_watchlist_source_draft` |
| Check now | `LocalWatchlistsService.launch_run` + `execute_run` | `TLDWAPIClient.check_watchlist_sources_now` |
| List runs | `WatchlistScopeService.list_runs` | `WatchlistScopeService.list_runs` |
| Run detail | `WatchlistScopeService.get_run` / `observe_run` | `WatchlistScopeService.get_run` / `observe_run` |
| Run cancel | `LocalWatchlistsService.cancel_run` | `TLDWAPIClient.cancel_watchlist_run` |
| Re-run | `LocalWatchlistsService.launch_run` + `execute_run` | `TLDWAPIClient.trigger_watchlist_run` (job-scoped; source-level via check-now) |
| Health alert rules | `WatchlistScopeService.{list/create/update/delete}_alert_rule` | `WatchlistScopeService.{list/create/update/delete}_alert_rule` |
| Content alert rules | `subscription_filters` with `action='notify'` | **Deferred** — server API content-alert endpoints are not yet wired in the Chatbook client |
| OPML import | Local OPML parser + bulk create sources | `TLDWAPIClient.import_watchlist_sources` |
| OPML export | Serialize sources to OPML | `TLDWAPIClient.export_watchlist_sources` |
| List groups/tags | Read-only; not stored locally | `TLDWAPIClient.list_watchlist_groups` / `list_watchlist_tags` |

Content alert rules are local-only in the first slice. The Rules pane shows a capability banner when the server backend is active, disabling content-rule creation with an explanation.

> **Note:** The referenced `TLDWAPIClient` methods (`test_watchlist_source`, `check_watchlist_sources_now`, `import_watchlist_sources`, `export_watchlist_sources`, etc.) already exist in `tldw_chatbook/tldw_api/client.py`.

### Local storage additions

Reuse existing `SubscriptionsDB` tables where possible and extend them with the minimal columns needed for the new screen.

**Schema migration plan**

`SubscriptionsDB` currently initializes tables with `CREATE TABLE IF NOT EXISTS` and has no versioned migration runner. A new `_ensure_watchlists_schema` startup helper performs idempotent migrations:

1. **Add missing columns** by inspecting `PRAGMA table_info` and running `ALTER TABLE ADD COLUMN` for any column not present.
2. **Widen `subscription_filters.action` CHECK constraint.** SQLite cannot drop a CHECK constraint directly, so the helper:
   - Creates a new `subscription_filters_new` table with the same columns but a widened CHECK constraint allowing `('auto_ingest','auto_ignore','tag','priority','notify','include','exclude','flag')`.
   - Copies all rows from `subscription_filters`.
   - Drops `subscription_filters` and renames `subscription_filters_new`.
   - Recreates indexes/triggers.
3. **Add indexes** for the new columns used in list queries:
   - `CREATE INDEX IF NOT EXISTS idx_subscription_items_run_id ON subscription_items(run_id)`
   - `CREATE INDEX IF NOT EXISTS idx_subscription_items_queued ON subscription_items(queued_for_briefing, status)`

| Table | Column | Type | Purpose |
|---|---|---|---|
| `subscription_items` | `queued_for_briefing` | `BOOLEAN DEFAULT 0` | Queue item for future briefing output |
| `subscription_items` | `run_id` | `INTEGER` | Link item to the run that produced it (no FK, because `local_watchlist_runs` is created lazily) |
| `subscription_items` | `alert_matches` | `TEXT` | JSON array of matched local content-alert rule IDs |
| `subscription_filters` | `priority` | `INTEGER DEFAULT 0` | Filter ordering |
| `subscription_filters` | `is_include_required` | `BOOLEAN DEFAULT 0` | Include-only gating |

**Filter and content-alert rule storage**

After the migration, `subscription_filters.action` allows:

- Legacy automatic-processing actions: `auto_ingest`, `auto_ignore`, `tag`, `priority`, `notify`.
- New source-level filter actions: `include`, `exclude`, `flag`.

Source-level filters use `include`/`exclude`/`flag` directly. Content-alert rules are stored as filters with `action = 'notify'` and a `severity` value in `action_params`; `severity` acts as the discriminator between legacy notification filters and new content-alert rules.

Filter type/mode/pattern (keyword/regex/author/date_range) is stored in `conditions` JSON. Example `conditions` rows:

- `{"type": "keyword", "mode": "contains", "pattern": "AI"}`
- `{"type": "regex", "pattern": "^Breaking:"}`

**Run-health rules**

Keep `local_watchlist_alert_rules` for run-health alert rules only (no schema change).

**Execution flow in `LocalWatchlistsService.execute_run`**

1. Fetch items.
2. Apply source-level filters (`subscription_filters` with `action IN ('include','exclude','flag')`) in priority order.
3. Evaluate local content-alert rules (`subscription_filters` with `action = 'notify'`) per item; store matched rule IDs in `alert_matches`.
4. Persist items to `subscription_items` using `INSERT … ON CONFLICT(subscription_id, url, content_hash) DO UPDATE`:
   - Update `run_id` to the current run.
   - Update `alert_matches` to the newly evaluated matches.
   - Preserve existing `status` if it is already `reviewed` or `ignored`; otherwise set to `new`.
5. Evaluate run-health rules (`local_watchlist_alert_rules`) and dispatch notifications.

**Source active toggle on create**

Extend `LocalWatchlistsService.create_source` to honor an `active` boolean in the payload (default `True`), matching the server `SourceCreateRequest` shape.

**Local re-run action**

The UI calls `WatchlistScopeService.launch_run(source_id=…)`. For the local backend, `WatchlistScopeService.launch_run` already calls `LocalWatchlistsService.launch_run` and then `execute_run` in the same flow, so the UI does not need to invoke `execute_run` separately.

## UI Layout Details

### Overview section

```
┌─ Latest Briefing ─┐ ┌─ Health ──────┐ ┌─ Attention Needed ────────┐
│ Run #123 running  │ │ System healthy│ │ 2 feeds need review       │
│ Next: 14:00       │ │               │ │ 1 failed run              │
└───────────────────┘ └───────────────┘ └───────────────────────────┘
┌─ Feeds ─┐ ┌─ Updates ─┐ ┌─ Activity ───────────────┐
│ 12 total│ │ 7 unread  │ │ 1 running / 0 pending    │
└─────────┘ └───────────┘ └──────────────────────────┘
Recent Failed Runs
• Monitor #4  FAILED  2m ago  Connection timeout  [View run]
```

### Sources section

- Left part of workbench: search, type filter, group tree, tag filter.
- Main area: sources `DataTable` with columns Name, Type, Status, Last scraped, Active toggle, Actions.
- Actions per row: Edit, Health/Seen drawer, Clone, Delete, Check now.
- Bulk action bar when rows selected.
- Form modal/inline for create/edit.
- Filter editor: include/exclude/flag rules with keyword/regex matcher, priority order.

### Items section

- Smart filters with counts: All / Today / Today unread / Unread / Reviewed / Queued / Alert matches.
  - `Alert matches` is derived from `subscription_items.alert_matches` JSON, populated during local run execution.
- Source list (left) and item list (middle) with item detail/preview (right inspector).
- Batch controls: mark selected/page/all filtered as reviewed; queue/dequeue for briefing.
- Item actions: open external link, discuss in Console, mark reviewed, queue.

### Runs section

- Runs `DataTable`: Source/Job, Status, Started, Duration, Found, Processed, Filtered, Errors, Actions.
- Attention alert for failed/stalled runs.
- Run detail inspector (right pane):
  - Statistics: status, duration, counts, failure remediation.
  - Items tab: paginated items from this run.
  - Log tab: `log_text` from run detail, with poll indicator.
  - Actions: **Re-run** (re-launch the same source/job via existing `launch_run`), cancel, export tallies. True "retry" that preserves the exact run configuration is deferred.

### Rules section

- Info boundary explaining health vs content alerts.
- Rule list with severity/kind tags, enable toggle, edit/delete.
- Inline rule form: name, kind, match mode, pattern, severity, source scope.
- Alert inbox: filters by status/severity/rule/source; cards with mark read/unread/dismiss.

## Error Handling

- Use a **screen-level recovery state** (consistent with the current placeholder) for backend/service-level errors.
- Policy denials use the existing `DestinationRecoveryState` / `policy_denied_recovery_state` pattern.
- Background workers use `@work(exclusive=True)` per pane to prevent duplicate fetches.
- Pass a backend token into each worker; ignore results if the user switched backends while the worker was in flight.
- Service exceptions are caught in the controller and converted to user-facing messages.
- Form validation errors appear inline next to fields.
- Polling workers stop when a run reaches a terminal status or the user leaves the Runs section.

## Testing Plan

### Unit tests

| Test file | Coverage |
|---|---|
| `Tests/Watchlists/test_watchlists_backend_controller.py` | Backend switching, policy enforcement, unsupported capabilities, error conversion |
| `Tests/Watchlists/test_watchlists_sources_pane.py` | List filtering, form validation, preview/check-now wiring, OPML parsing |
| `Tests/Watchlists/test_watchlists_runs_pane.py` | Run list refresh, inspector binding, cancel/retry guards |
| `Tests/Watchlists/test_watchlists_items_pane.py` | Smart-count filtering, batch review, item row actions |
| `Tests/Watchlists/test_watchlists_alert_rules_pane.py` | Health/content rule forms, rule toggle, inbox filters |

### Service/DB tests

| Test file | Coverage |
|---|---|
| `Tests/Subscriptions/test_local_watchlists_service.py` (extended) | Content-alert evaluation, filter evaluation, scraped-item persistence, run stats |
| `Tests/Subscriptions/test_watchlist_filter_service.py` | Include/exclude/flag matching |
| `Tests/Subscriptions/test_watchlist_content_alert_service.py` | Pattern matching, severity, notification payload |

### UI integration tests

| Test file | Coverage |
|---|---|
| `Tests/UI/test_watchlists_destination_shell.py` | Empty/error/recovery states, backend toggle, section switching, Console handoff |
| `Tests/UI/test_watchlists_sources.py` | CRUD, OPML import, preview |
| `Tests/UI/test_watchlists_runs.py` | List refresh, inspector, cancel |
| `Tests/UI/test_watchlists_items.py` | Filters, mark reviewed, queue |
| `Tests/UI/test_watchlists_alert_rules.py` | Create rules, inbox surfacing |

### Fixtures

- Reuse existing Textual `pilot` fixtures.
- Add `Tests/fixtures/watchlists/sample.opml` for import tests.
- Use an in-memory `SubscriptionsDB` factory for service tests.

## Migration / Cleanup

- Delete `tldw_chatbook/UI/SubscriptionWindow.py`.
- Remove `subscriptions` route from `screen_registry.py`; keep it as a shell alias in `shell_destinations.py` pointing to `watchlists_collections`.
- Update `route_inventory.py` owner mapping if needed.
- Review `tldw_chatbook/Constants.py` (`TAB_SUBSCRIPTIONS`, `ALL_TABS`, `TAB_DISPLAY_LABELS`) and remove or repurpose as appropriate.
- Update `tldw_chatbook/app.py` references:
  - `TAB_SUBSCRIPTIONS` binding/tooltip.
  - `pending_subscription_initial_tab` and `_stage_subscription_watchlist_run_context` staging logic should target the new Watchlists screen route.
  - Subscription button event-handler map around line 4347 should be removed or migrated to the new screen's pane actions.
- Remove or rewrite focused subscription UI tests.

## Decisions / Resolved Questions

1. **Sub-slice split** — Yes, the first slice is split into two reviewable sub-slices:
   - **Slice 1A**: Overview, Sources, Runs (core management and run inspection).
   - **Slice 1B**: Items reader, Alert Rules, source-level filters, OPML import/export, Preview, and Check now.
2. **Content-alert rule storage** — Reuse `subscription_filters` with `action = 'notify'` and store severity in `action_params`. Run-health rules remain in `local_watchlist_alert_rules`.
3. **Scraped-item storage** — Reuse `subscription_items`; add `queued_for_briefing` and `run_id` columns.
4. **Server content-alert rules** — Deferred. The Chatbook client does not yet expose the server's content-alert endpoints, so content rules are local-only in Slice 1B with a backend-capability banner.

## Related Files

- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- `tldw_chatbook/Subscriptions/local_watchlists_service.py`
- `tldw_chatbook/Subscriptions/server_watchlists_service.py`
- `tldw_chatbook/Subscriptions/watchlist_scope_service.py`
- `tldw_chatbook/UI/Navigation/shell_destinations.py`
- `tldw_chatbook/UI/Navigation/screen_registry.py`
- `tldw_chatbook/UI/Workbench/route_inventory.py`
