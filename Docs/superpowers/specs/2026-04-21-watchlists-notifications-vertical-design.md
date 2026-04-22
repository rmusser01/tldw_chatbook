# tldw_chatbook Watchlists And Client Notifications Vertical Design

**Date:** 2026-04-21  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server`

## Goal

Define the next standalone-first parity vertical after the completed media/read-it-later work: extend the existing subscriptions product in `tldw_chatbook` so it can operate as a source-aware `Watchlists + Client Notifications` surface.

This first slice is intentionally narrow:

- local mode continues to use the existing subscriptions product as the authoritative standalone monitoring surface
- server mode adds live remote `watchlist source CRUD`
- Chatbook gains a persisted local notification inbox

This slice does **not** attempt to deliver the entire server watchlists product. It does not include jobs, runs, alert rules, or remote notification feeds.

## Context

This vertical is not greenfield.

Chatbook already has a substantial local subscriptions product:

- `tldw_chatbook/UI/Screens/subscription_screen.py`
- `tldw_chatbook/UI/SubscriptionWindow.py`
- `tldw_chatbook/DB/Subscriptions_DB.py`
- `tldw_chatbook/Event_Handlers/subscription_events.py`
- `tldw_chatbook/Subscriptions/textual_scheduler_worker.py`

That local product is richer than the first remote watchlists slice. It includes:

- local subscription CRUD
- local checking/scheduler behavior
- review items
- dashboard summaries
- briefings
- settings and per-site behavior

Chatbook also has partial local notification plumbing, but not a durable local notification product:

- `tldw_chatbook/Widgets/toast_notification.py`
- `tldw_chatbook/Utils/NotificationHelper.py`

The current notification surface is mostly temporary delivery. It does not provide:

- a persisted local inbox
- read/dismiss lifecycle
- a normalized local notification record
- one authoritative dispatch pipeline for queue-plus-toast delivery

On the server side, the relevant contract is the watchlists source/group surface, not the full watchlists execution stack:

- `tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- `tldw_server/tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py`
- `tldw_server/tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`

The server watchlists contract currently exposes, among other things:

- source CRUD
- source test/import/export/check-now helpers
- groups CRUD
- jobs
- runs
- outputs/templates
- alert rules

This first client slice deliberately narrows that scope to:

- remote `Source` list/detail/create/update/delete
- optional read-only display of returned `group_ids`
- optional read-only lookup of group names for display only
- strict omission of `group_ids` from first-slice create/update requests

It explicitly defers:

- group CRUD
- jobs
- runs
- alert rules
- outputs/templates
- remote notification/reminder/feed surfaces

## Product Decisions

The following decisions are fixed for this vertical:

- The existing `subscription_screen` and `SubscriptionWindow` remain the primary shell.
- The shell becomes source-aware rather than being replaced by a new top-level watchlists screen.
- The shell uses one shared normalized list pane and backend-specific detail/editor panes.
- Local mode remains locally authoritative.
- Server mode remains server-authoritative.
- Server watchlists are treated as live remote-only records.
- No local shadow copy of remote watchlist metadata is introduced in this slice.
- The remote parity promise in this slice is specifically `watchlist source CRUD`.
- The first-slice server editor supports only `rss` and `site` source types.
- `forum` sources are blocked centrally in UI/service/policy even if the server schema can represent them.
- `group_ids` remain read-only/deferred in this slice.
- `group_ids` must never be editable in the first-slice UI.
- first-slice create/update payloads must omit `group_ids` entirely.
- Server group CRUD is explicitly deferred.
- Server jobs, runs, and alert rules are explicitly deferred.
- First-slice server source editing does not expose a free-form raw `settings` editor.
- Existing server-side `settings` values are preserved when the user updates only first-slice editable fields.
- Existing local subscriptions scheduling/review/briefings/dashboard behavior remains local-only in this slice.
- In server mode, unsupported local-only areas must degrade explicitly rather than appearing half-functional.
- Chatbook gains a persisted, Chatbook-owned local notification inbox.
- The notification inbox uses its own dedicated local store rather than piggybacking on the subscriptions database.
- The notification inbox schema is general-purpose, but only `subscriptions/watchlists` events produce into it initially.
- Toasts are delivery, not storage.
- One notification dispatch pipeline must own:
  - queue insert
  - toast attempt
  - fallback notify behavior
- Server delete must be represented accurately as a reversible delete with a restore window, not as an immediate hard delete.
- Dedicated restore UI for remote watchlist sources is deferred even though delete responses carry restore metadata.
- Runtime policy remains the authority for backend mode, permissions, and unsupported-operation behavior.

## User Decisions Captured

- Extend the existing `subscription_screen`, not a new dedicated watchlists screen.
- Use one shared shell with backend-specific detail panes, not one flattened editor.
- The first remote slice covers watchlists plus source CRUD only.
- Client notifications gain a persisted local inbox with read/dismiss state.
- The inbox appears as a new `Notifications` tab inside the existing subscriptions screen.
- Remote watchlists remain live remote-only records with no local shadow copy.
- Server groups stay read-only/deferred for this slice.

## In Scope

- Add a source-aware subscriptions/watchlists scope service.
- Add a dedicated server watchlists client/service layer for watchlist source CRUD.
- Extend `subscription_screen` / `SubscriptionWindow` to become backend-aware.
- Add runtime-backend refresh handling for the subscriptions screen path.
- Split local versus server initialization in the subscriptions window.
- Split local versus server lifecycle/state-sync behavior in the subscriptions screen wrapper.
- Add explicit dirty-state and stale-load handling rules for backend switches.
- Preserve the current local subscriptions product in local mode.
- Add normalized list-row mapping for:
  - local subscriptions
  - server watchlist sources
- Add a backend-specific local subscription editor pane.
- Add a backend-specific server watchlist source editor pane.
- Add source-aware list/detail/create/update/delete behavior.
- Add a persisted local notification queue/store.
- Add a `Notifications` tab backed by the local queue.
- Add a single notification dispatch helper/service that writes queue records and attempts toast delivery.
- Extend runtime-policy registry coverage for local notification queue mutations required by the inbox UI.
- Add runtime-policy enforcement at both UI and service seams.
- Add regression coverage for:
  - backend switching
  - stale mode state
  - remote CRUD routing
  - unsupported server/local operations
  - notification queue read/dismiss behavior

## Out Of Scope

- A new top-level Watchlists screen
- Local shadow copies of remote watchlists
- Watchlist jobs
- Watchlist runs
- Alert rules
- Group CRUD
- Editable remote source restore UI
- Remote reminders / notification feeds
- Whole-app notification migration
- Rewriting all existing app notifications to use the new queue
- Cross-backend sync or mirroring behavior
- A generic server watchlists console

## Approaches Considered

### Option A: Hard-normalized shared watchlist model

Create one fully shared model and one common editor limited to overlapping fields across local subscriptions and server watchlist sources.

Why not chosen:

- local subscriptions already expose richer standalone fields
- flattening to the overlap would weaken the local product
- the server source contract and local subscription contract are not equivalent enough to justify one editor

### Option B: Shared shell with backend-specific detail panes

Use one source-aware list shell and one source-aware screen, but keep detail/edit panes specific to each backend.

Why chosen:

- preserves the stronger existing local subscriptions product
- lets server watchlist sources map honestly to the server contract
- avoids pretending jobs/runs/alert rules are already part of the first slice
- aligns with the runtime-policy source-authority model already in use across other verticals

### Option C: New dedicated Watchlists screen

Create a separate screen for watchlists and leave local subscriptions where they are.

Why not chosen:

- duplicates user-facing navigation and conceptual ownership
- forces a second monitoring destination before sync/mirroring exists
- conflicts with the decision to evolve the current subscriptions product in place

## Chosen Model

This vertical keeps one source-aware monitoring shell inside the existing subscriptions destination.

That shell has:

- a shared normalized list of watch items
- backend-specific detail and editing panes
- a new `Notifications` tab backed by a persisted local queue

The core rule is:

- in `local` mode, the shell operates over local subscriptions and local notification state
- in `server` mode, the shell operates over remote watchlist sources and local notification state

Remote watchlist sources are never silently copied into local subscription storage.

Local notifications remain Chatbook-owned regardless of whether the originating action came from local subscriptions or remote watchlist source mutations.

## Architecture

### 1. Authority Model

This vertical must inherit the runtime-policy authority model rather than introducing an ad hoc backend toggle.

The controlling rules are:

- runtime policy decides whether the active source is `local` or `server`
- `SubscriptionScreen` and `SubscriptionWindow` must refresh from authoritative runtime state on backend change
- local mode is locally authoritative
- server mode is server-authoritative
- stale widget state must not be able to mutate the wrong backend
- unsupported actions must be blocked both in the UI seam and in the scope-service seam

### 1b. Action-Level Policy Contract

This vertical must use the existing action-level runtime-policy vocabulary rather than inventing a second permission layer.

Required action ids for the first slice are:

- `watchlists.list.local`
- `watchlists.detail.local`
- `watchlists.create.local`
- `watchlists.update.local`
- `watchlists.delete.local`
- `watchlists.list.server`
- `watchlists.detail.server`
- `watchlists.create.server`
- `watchlists.update.server`
- `watchlists.delete.server`
- `notifications.queue.list.local`
- `notifications.queue.observe.local`
- `notifications.dispatch.launch.local`

For this vertical, `watchlists.*.local` are the shell/service action ids for the local subscriptions-backed path. The existing local subscriptions product remains the implementation backend in local mode, but policy evaluation at the shared shell and scope-service seam should use the `watchlists` capability namespace rather than inventing parallel `subscriptions.*` action ids.

The current registry does not yet expose queue-mutation actions for inbox state changes, so this vertical must add:

- `notifications.queue.update.local`

That action covers:

- mark read
- mark unread
- dismiss

Clear-all or hard-delete style inbox actions remain out of scope for this slice.

### 1a. Backend-Switch Safety Rules

This is required for planning. The screen cannot rely on best-effort refresh alone.

The backend-switch policy is:

- if the active editor has unsaved local changes, backend switching must not silently discard them
- the user must be prompted to either:
  - discard edits and continue switching, or
  - cancel the backend switch
- no cross-backend draft carryover is allowed
- the screen/window must own one explicit action lock for in-flight write operations
- if a read/list/detail request is in flight when the backend changes, its response must be ignored unless it matches the current refresh generation
- the screen/window must maintain a monotonic refresh generation or equivalent stale-response guard
- write operations must be treated more strictly than reads:
  - while create/update/delete is in flight, backend switching must be temporarily disabled or blocked with a busy message
  - once the write resolves, the screen may refresh into the new backend normally

This prevents two failure modes:

- stale local/server responses repainting the wrong backend view
- unsaved edits disappearing unpredictably on backend switch

### 2. Split Initialization

This is a required design correction.

`SubscriptionWindow` currently initializes local database, scheduler, review/dashboard, and briefings on mount. That behavior cannot remain unconditional once server mode exists.

The window must explicitly split initialization:

- local mode:
  - initialize `SubscriptionsDB`
  - initialize scheduler worker
  - load local review/dashboard/briefing surfaces
- server mode:
  - do not initialize local scheduler/briefing flows as active behaviors
  - initialize the server watchlists service path
  - render explicit local-only unavailable states for unsupported tabs

This split is architectural, not cosmetic.

Teardown rules are equally strict:

- when leaving local mode, any running local scheduler worker must be stopped rather than merely hidden
- local-only refresh or polling behavior must be cancelled before server-mode UI is considered active
- hiding a local-only tab is acceptable for passive UI state, but never for active background behavior

### 2a. Screen Lifecycle Split

`SubscriptionScreen` also requires explicit backend-aware refactoring. It currently mirrors local database state and triggers local-only refresh calls directly.

The screen rules must become:

- `handle_runtime_backend_changed()` delegates to backend-aware refresh paths rather than assuming a local DB-backed window
- `on_screen_resume()` must not unconditionally call local-only review/dashboard/item refresh methods in server mode
- `on_screen_suspend()` and backend-switch teardown paths must explicitly stop local scheduler/worker activity rather than merely hiding local-only tabs
- `_sync_state_from_window()` must stop assuming `window.db` and `window.scheduler_worker` are always present
- screen shell state should be derived from normalized watchlist-scope data or explicit backend-aware window accessors

This is required so server mode does not accidentally execute local-only resume/suspend behavior after the window split lands.

### 3. Core Units

The vertical should be decomposed into the following units.

#### `server watchlists API client additions`

New `tldw_api` client methods for the remote first slice:

- list watchlist sources
- get watchlist source detail
- create watchlist source
- update watchlist source
- delete watchlist source
- optionally list groups for read-only name display if needed

The client/service contract for this first slice must also enforce:

- create/update payloads do not include `group_ids`
- any returned `group_ids` are display-only metadata
- optional group-name lookup, if implemented, is display-only enrichment and must never affect mutation payloads
- server delete responses are modeled as reversible-delete payloads and retain restore-window metadata

These methods must be first-class work, not hidden under UI implementation.

#### `server_watchlists_service`

A dedicated service that wraps the new API client methods and normalizes server source records into a Chatbook-friendly shape.

This first slice covers `Source` CRUD only.

#### `watchlist_scope_service`

A source-aware service above:

- local `SubscriptionsDB`
- the new `server_watchlists_service`

This service owns:

- normalized list operations
- detail load
- create/update/delete routing
- runtime-policy enforcement
- backend-aware unsupported behavior

#### `subscription/watch item normalizers`

A normalization layer for the shared left-pane list.

This layer must normalize only the shared shell-facing fields. It must not force one flattened editor schema.

#### `client_notifications` local store

A local persisted queue/inbox store with read/dismiss behavior.

This store is Chatbook-owned and local-only.

This store should be a dedicated notifications store with its own config/path getter, following the existing DB-path pattern used elsewhere in Chatbook, rather than a new table inside `subscriptions.db`.

Recommended naming:

- `get_notifications_db_path()`
- default file name `tldw_chatbook_notifications.db`

#### `notification_dispatch_service`

One helper/service that:

- inserts a queue record
- attempts toast delivery if available
- falls back to `app.notify`

This avoids parallel notification systems.

### 4. Normalized Shared List Contract

The shared shell list needs only a narrow, list-oriented normalized contract.

Suggested fields:

- `id`
- `backend`
- `entity_kind`
- `source_id`
- `title`
- `source_type`
- `url`
- `active`
- `tags`
- `status_summary`
- `last_checked_or_scraped_at`

Examples:

- `local:subscription:42`
- `server:watchlist_source:17`

The shared list contract should not attempt to encode every backend-specific field.

### 5. Backend-Specific Editors

The right pane must branch by backend.

#### Local editor

Preserve the current local subscription richness:

- type
- source URL
- frequency
- auth
- extraction options
- auto-ingest
- tags/folder/priority
- notification config

#### Server editor

Limit to the actual first-slice server source contract:

- `name`
- `url`
- `source_type`
- `active`
- `tags`
- optional read-only display of `group_ids`
- optional read-only display of resolved group names if group lookup is added
- optional read-only summary that advanced server settings exist

This pane must not expose fake support for:

- jobs
- runs
- alert rules
- group editing
- `forum` source creation/editing
- arbitrary raw `settings` editing

The source-type rule for this pane is strict:

- allow `rss`
- allow `site`
- block `forum` centrally even if returned by server contracts or flags
- if a server row with `source_type=forum` is encountered, render it as unsupported/read-only rather than pretending first-slice edit support exists

The mutation rule for this pane is strict:

- create payloads omit `group_ids`
- update payloads omit `group_ids`
- returned `group_ids` may be rendered for context only
- any optional group-name resolution is display-only and non-authoritative
- when advanced `settings` already exist on a server source, first-slice edits must preserve them unless a later vertical adds an explicit editor

### 6. Notification Queue Contract

The persisted local inbox should use a general-purpose schema, but the first producers are limited to this vertical.

Suggested fields:

- `id`
- `category`
- `title`
- `message`
- `severity`
- `source_backend`
- `source_entity_id`
- `source_entity_kind`
- `is_read`
- `is_dismissed`
- `created_at`
- `payload_json`

Initial producer categories:

- `subscriptions`
- `watchlists`

Initial producer events are intentionally narrow:

- user-triggered create/update/delete success from the subscriptions/watchlists shell
- user-triggered create/update/delete failure from the subscriptions/watchlists shell

The following are explicitly deferred as notification producers for this first slice:

- background local scheduler activity
- backend-switch prompts or warnings
- remote server-originated watchlist events
- remote reminders/notification feeds
- unrelated app-domain notifications

Future domains may produce into the same queue later without redesigning the schema now.

## UI Model

### Shared Shell

The existing subscriptions destination remains the host shell.

Primary structure:

- `Subscriptions` tab
  - shared list pane
  - backend-specific detail/editor pane
- existing local product tabs
  - remain local-first
  - degrade explicitly in server mode if unsupported
- new `Notifications` tab
  - local inbox list
  - read/dismiss controls
  - optional filtering by severity/backend/category

### Backend-Specific Tab Behavior

#### Local mode

All current local subscriptions surfaces continue to operate:

- subscriptions
- review items
- dashboard
- briefings
- settings

#### Server mode

Only the first-slice remote source CRUD surface is active.

Local-only tabs must:

- disable interaction, or
- render a clear “local-only in this slice” state

They must not appear active but silently fail.

### Notifications Tab

This tab lives inside the subscriptions shell for this first slice.

It is backed by the local queue, not by remote notifications/reminders APIs.

The tab must support:

- list notifications
- mark read/unread
- dismiss
- clear dismissed or read items if desired in a later refinement

## Data Flow

### On Mount

1. `SubscriptionScreen` composes `SubscriptionWindow`
2. `SubscriptionWindow` reads authoritative runtime state
3. `SubscriptionWindow.refresh_backend_view()` initializes only the active backend path
4. Shared list loads from the watchlist scope service
5. `Notifications` tab loads from the local notifications store

### On Backend Change

1. App runtime backend changes
2. If the active editor is dirty, the user is prompted to discard or cancel the switch
3. If a create/update/delete operation is still in flight, the switch is blocked until the write resolves
4. If switching proceeds, `SubscriptionScreen.handle_runtime_backend_changed()` is invoked
5. The screen increments the refresh generation and forwards to `SubscriptionWindow.refresh_backend_view()`
6. Window rereads authoritative runtime state
7. Shared list/editor/tabs rebuild for the active backend
8. Any stale read/list/detail responses from an earlier generation are ignored
9. Any active local scheduler/worker state is explicitly stopped before server-mode UI becomes active
10. Passive stale local/server pane state is then torn down or hidden as appropriate

### On Mutation

1. User edits or creates a record in the active backend pane
2. UI path validates the action for the active backend
3. Scope service enforces backend/routing/policy again
4. Underlying local DB or remote service performs mutation
5. Shared list refreshes for that backend
6. Notification dispatch service records a local notification and attempts toast delivery
7. If the mutation was a remote delete, the notification payload preserves the server restore window metadata for accurate copy and future restore UX

## Error Handling

Error handling should stay source-specific and non-magical.

- local validation/storage errors remain local-only failures
- remote connection/auth/validation errors remain server-only failures
- failed remote operations do not create local shadow records
- notification dispatch must not block the underlying successful mutation
- queue insert plus toast attempt are secondary side effects
- in this first slice, notification queue inserts are limited to user-triggered shell actions from this vertical, not background scheduler/server events
- notification queue inserts are limited to CRUD outcome records from this vertical, not prompts, background checks, or remote feed events
- server delete success copy must reflect reversible-delete semantics rather than claiming permanent removal
- first-slice server edits must preserve opaque server `settings` fields they do not expose

Server mode failures should surface clear remote status messages such as:

- connection error
- unauthorized / auth expired
- invalid source payload
- unsupported source type for this slice

Local mode failures should remain ordinary local product errors.

## Testing Strategy

The test plan should mirror the unit boundaries above.

### Local DB/store tests

- local notification queue CRUD
- read/dismiss behavior
- ordering/filtering behavior

### API client/service tests

- new watchlist source client methods
- server watchlists service source normalization
- error classification

### Scope-service tests

- local subscriptions routing
- server watchlist source routing
- backend-specific unsupported operations
- runtime-policy enforcement
- disallowed source type protection in the runtime seam

### UI tests

- subscriptions screen/backend switch callback path
- window refresh on runtime backend change
- local versus server initialization split
- shared list rendering across local and server rows
- backend-specific editor rendering
- server-mode degradation for local-only tabs
- notifications tab read/dismiss behavior
- stale mode state cannot dispatch wrong-backend mutation

### Regression priorities

- backend switch while subscriptions screen is active
- server-mode view should not start local scheduler flows
- failed remote CRUD must not pollute local state
- notifications queue insert plus toast fallback behavior
- server-mode resume/suspend paths should not call local-only refresh flows
- server delete notifications should include restore-window metadata

## Risks And Mitigations

### Risk: local subscriptions shell is too entangled with local-only startup

Mitigation:

- make split initialization an explicit first architectural task
- add backend refresh seam before adding remote CRUD

### Risk: scope creep from “watchlists” into jobs/runs/alert rules

Mitigation:

- name the remote slice precisely as `watchlist source CRUD`
- defer groups editing, jobs, runs, and alert rules explicitly

### Risk: parallel notification systems

Mitigation:

- define one dispatch pipeline that owns queue insert plus toast fallback

### Risk: notification queue mutations bypass runtime policy

Mitigation:

- add explicit `notifications.queue.update.local` policy coverage
- require inbox mark-read/mark-unread/dismiss paths to use the policy seam instead of direct store writes

### Risk: stale backend UI state mutating the wrong backend

Mitigation:

- runtime-backend callback path on the screen
- dirty-editor discard-or-cancel prompt on backend switch
- generation-based stale-response guard for read/list/detail requests
- temporary switch blocking while writes are in flight
- reread authoritative runtime state inside create/update/delete handlers
- runtime-policy enforcement in the scope service

### Risk: accidental server group mutation through first-slice source updates

Mitigation:

- render `group_ids` as read-only display only
- do not include `group_ids` in first-slice create/update payloads
- keep group CRUD and editable group selection deferred to a later vertical

### Risk: first-slice server edits clobber advanced source settings

Mitigation:

- do not expose arbitrary raw `settings` editing in this slice
- preserve existing server `settings` when updating only first-slice editable fields
- block `forum` creation/update centrally instead of partially exposing unsupported settings semantics

## Follow-On Work Deliberately Deferred

- server groups CRUD or editable group picker
- server jobs/runs observation
- alert rules
- remote reminders/notification feeds
- whole-app notification producer migration
- local shadow copies or offline caching of remote watchlists
- sync/mirror behavior between local subscriptions and server watchlists

## Success Criteria

This vertical is successful when:

- local subscriptions still work as a standalone product
- server mode can browse and mutate remote watchlist sources through the same screen shell
- unsupported local-only tabs fail explicitly in server mode
- backend changes refresh this screen correctly
- backend changes do not silently discard dirty editor state
- stale responses from the previous backend cannot repaint the current backend view
- Chatbook has a persisted local notification inbox with read/dismiss state
- inbox read/unread/dismiss actions are covered by runtime policy rather than direct ad hoc store writes
- the inbox is populated only by user-triggered CRUD outcome records from this vertical
- no local shadow copy of remote watchlists is introduced
- local notifications live in a dedicated local notifications store, not in `subscriptions.db`
- server editor only exposes supported first-slice source types and does not expose arbitrary raw `settings` editing
- server source CRUD does not mutate `group_ids` in this slice
- server delete outcomes are presented as reversible deletes with restore-window metadata
- no jobs/runs/alert-rules scope slips into the first slice
