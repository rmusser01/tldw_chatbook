# Watchlists Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add Chatbook client/service/TUI support for server watchlist jobs, runs, restore, and alert-rule administration while preserving Chatbook's standalone local subscriptions behavior.

**Architecture:** Extend the existing source-aware watchlist seam instead of adding a parallel window. `TLDWAPIClient` owns typed remote contracts, `ServerWatchlistsService` normalizes them, `WatchlistScopeService` enforces runtime/source policy, and `SubscriptionWindow` exposes server control-plane tabs with explicit local unavailable/help states. Groups remain read-only/deferred.

**Tech Stack:** Python 3.12, Pydantic v2 schemas, Textual widgets, pytest/pytest-asyncio.

---

### Task 1: Pin The Server Watchlists Contract

**Files:**
- Modify: `tldw_chatbook/tldw_api/watchlists_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_watchlists_client.py`

- [x] **Step 1: Write failing schema/client tests**

Add tests for:
- Job create/update/list/detail/delete/restore.
- Job trigger run.
- Global and per-job run list.
- Run detail and cancel.
- Source restore.
- Alert-rule list/create/update/delete.

- [x] **Step 2: Run contract tests and verify failure**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_watchlists_client.py -q
```

Expected: FAIL on missing schema exports or missing `TLDWAPIClient` methods.

- [x] **Step 3: Implement typed schemas and client methods**

Add Pydantic models aligned to `tldw_server2/tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py` and `watchlist_alert_rules.py`, then add client methods for the endpoint paths:
- `POST /api/v1/watchlists/sources/{source_id}/restore`
- `GET|POST /api/v1/watchlists/jobs`
- `GET|PATCH|DELETE /api/v1/watchlists/jobs/{job_id}`
- `POST /api/v1/watchlists/jobs/{job_id}/restore`
- `POST /api/v1/watchlists/jobs/{job_id}/run`
- `GET /api/v1/watchlists/jobs/{job_id}/runs`
- `GET /api/v1/watchlists/runs`
- `GET /api/v1/watchlists/runs/{run_id}`
- `GET /api/v1/watchlists/runs/{run_id}/details`
- `POST /api/v1/watchlists/runs/{run_id}/cancel`
- `GET|POST /api/v1/watchlists/alert-rules`
- `PATCH|DELETE /api/v1/watchlists/alert-rules/{rule_id}`

- [x] **Step 4: Run contract tests and verify pass**

Run the same command. Expected: PASS.

### Task 2: Normalize Jobs, Runs, Restore, And Alert Rules In Services

**Files:**
- Modify: `tldw_chatbook/Subscriptions/server_watchlists_service.py`
- Modify: `tldw_chatbook/Subscriptions/watchlist_scope_service.py`
- Test: `Tests/Subscriptions/test_server_watchlists_service.py`
- Test: `Tests/Subscriptions/test_watchlist_scope_service.py`

- [x] **Step 1: Write failing service/scope tests**

Cover:
- Server job create/list/detail/update/delete/restore/trigger.
- Server run list/detail/cancel.
- Server alert-rule CRUD.
- Source restore.
- Scope policy action IDs use `watchlists.<action>.server`.
- Local job/run/alert-rule calls return explicit unavailable errors, not fake data.

- [x] **Step 2: Run service/scope tests and verify failure**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/Subscriptions/test_server_watchlists_service.py Tests/Subscriptions/test_watchlist_scope_service.py -q
```

Expected: FAIL on missing service/scope methods.

- [x] **Step 3: Implement minimal service/scope methods**

Normalize server IDs as:
- `server:watchlist_job:<id>`
- `server:watchlist_run:<id>`
- `server:watchlist_alert_rule:<id>`

Expose methods:
- `restore_watch_item`
- `list_jobs`, `get_job_detail`, `save_job`, `delete_job`, `restore_job`, `trigger_job`
- `list_runs`, `get_run_detail`, `cancel_run`
- `list_alert_rules`, `save_alert_rule`, `delete_alert_rule`

- [x] **Step 4: Run service/scope tests and verify pass**

Run the same command. Expected: PASS.

### Task 3: Add The Textual Control-Plane Surface

**Files:**
- Modify: `tldw_chatbook/UI/SubscriptionWindow.py`
- Modify: `tldw_chatbook/UI/Subscription_Modules/subscription_backend_controller.py`
- Test: `Tests/UI/test_subscription_window_watchlists.py`

- [x] **Step 1: Write failing UI tests**

Cover:
- New `Jobs`, `Runs`, and `Alert Rules` tabs mount.
- Server mode refresh populates job/run/alert lists through the scope service.
- Local mode shows explicit local guidance for server-only job/run/alert-rule controls.
- Delete source exposes restore through a controller method.
- Job run/cancel and alert-rule create/delete button handlers call the scope service with server backend.

- [x] **Step 2: Run UI tests and verify failure**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/UI/test_subscription_window_watchlists.py -q
```

Expected: FAIL on missing widgets/handlers.

- [x] **Step 3: Implement first-slice UI controls**

Add:
- `Jobs` tab: list jobs, create/update JSON payload editor, delete/restore, trigger selected job.
- `Runs` tab: list runs, load detail, cancel selected run.
- `Alert Rules` tab: list rules, create/update JSON payload editor, delete selected rule.
- Keep group management read-only/deferred.
- Keep local mode advisory text: local subscriptions/scheduler remain the local execution path.

- [x] **Step 4: Run UI tests and verify pass**

Run the same command. Expected: PASS.

### Task 4: Update Parity Docs And Run Verification

**Files:**
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`

- [x] **Step 1: Update docs**

Record jobs/runs/alert-rule/restore support as landed, with remaining gaps limited to groups read-only/deferred, richer output/audio UX, and sync/mirror semantics.

- [x] **Step 2: Run focused verification**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_watchlists_client.py Tests/Subscriptions/test_server_watchlists_service.py Tests/Subscriptions/test_watchlist_scope_service.py Tests/UI/test_subscription_window_watchlists.py Tests/UI/test_screen_navigation.py Tests/RuntimePolicy/test_boundary_guards.py -q
git diff --check
```

Expected: PASS.

- [x] **Step 3: Compile changed files**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall tldw_chatbook/Subscriptions tldw_chatbook/UI/SubscriptionWindow.py tldw_chatbook/UI/Subscription_Modules tldw_chatbook/tldw_api Tests/Subscriptions Tests/UI/test_subscription_window_watchlists.py Tests/tldw_api/test_watchlists_client.py -q
```

Expected: PASS for changed files.
