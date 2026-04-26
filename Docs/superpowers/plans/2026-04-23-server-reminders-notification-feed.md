# Server Reminders Notification Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the server-owned reminders and notification-feed contract inside Chatbook without replacing the existing Chatbook-local notifications inbox.

**Architecture:** Add typed API models and `TLDWAPIClient` methods for `/api/v1/tasks` and `/api/v1/notifications`, then wrap them in a remote-only service plus a policy-aware scope seam. Mount lightweight `Server Reminders` and `Server Feed` tabs in the existing subscriptions/watchlists window, with local mode showing explicit unavailable guidance.

**Tech Stack:** Python 3, Pydantic v2, Textual, pytest/pytest-asyncio, existing `runtime_policy` action IDs.

---

### Task 1: Typed API Contract

**Files:**
- Create: `tldw_chatbook/tldw_api/server_notifications_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_server_notifications_client.py`

- [x] **Step 1: Write failing client tests**

Cover reminder task create/list/get/update/delete, notification list/unread/mark-read/dismiss/snooze/cancel-snooze/preferences, and SSE stream routing.

- [x] **Step 2: Run the new API test**

Run: `/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_server_notifications_client.py -q`

Expected: fail because schemas and methods do not exist.

- [x] **Step 3: Add schemas and client methods**

Mirror the server response shape from `tldw_server2/tldw_Server_API/app/api/v1/schemas/reminders_schemas.py`; keep unknown stream event payloads as generic `dict | str | None`.

- [x] **Step 4: Re-run the API test**

Expected: pass.

### Task 2: Remote-Only Service And Scope Seam

**Files:**
- Create: `tldw_chatbook/Notifications/server_notifications_service.py`
- Create: `tldw_chatbook/Notifications/server_notifications_scope_service.py`
- Modify: `tldw_chatbook/Notifications/__init__.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/Notifications/test_server_notifications_service.py`

- [x] **Step 1: Write failing service/scope tests**

Cover normalized reminder/feed rows, policy action IDs, local-mode rejection, and stream event normalization.

- [x] **Step 2: Run the new service test**

Run: `/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/Notifications/test_server_notifications_service.py -q`

Expected: fail because service and scope seam do not exist.

- [x] **Step 3: Implement the service and scope seam**

Use `notifications.reminders.list.server`, `notifications.reminders.configure.server`, `notifications.reminders.launch.server`, `notifications.reminders.observe.server`, `notifications.feed.list.server`, and `notifications.feed.observe.server`.

- [x] **Step 4: Wire app bootstrap**

Attach `server_notifications_service` and `server_notifications_scope_service` beside existing local notification and watchlist services.

- [x] **Step 5: Re-run the service test**

Expected: pass.

### Task 3: SubscriptionWindow Remote Tabs

**Files:**
- Modify: `tldw_chatbook/UI/SubscriptionWindow.py`
- Modify: `tldw_chatbook/UI/Subscription_Modules/subscription_backend_controller.py`
- Test: `Tests/UI/test_subscription_window_watchlists.py`

- [x] **Step 1: Write failing UI tests**

Cover server refresh loading reminders/feed, local mode showing remote-only guidance, selected feed mark-read/dismiss/snooze/cancel-snooze actions, selected reminder save/delete actions, and feed watch event rendering.

- [x] **Step 2: Run the UI test**

Run: `/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/UI/test_subscription_window_watchlists.py -q`

Expected: fail because the tabs and handlers do not exist.

- [x] **Step 3: Add lightweight tabs and handlers**

Add `Server Reminders` and `Server Feed` tabs using JSON payload editors to avoid competing with parallel UX work. Keep the existing `Notifications` tab explicitly local.

- [x] **Step 4: Re-run the UI test**

Expected: pass.

### Task 4: Docs And Verification

**Files:**
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`

- [x] **Step 1: Update parity status**

Record the first remote reminders/feed slice as landed and identify remaining gaps as UX depth and sync/mirroring, not missing route discovery.

- [x] **Step 2: Run focused verification**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_server_notifications_client.py Tests/Notifications/test_server_notifications_service.py Tests/UI/test_subscription_window_watchlists.py Tests/UI/test_screen_navigation.py Tests/RuntimePolicy/test_runtime_policy_core.py -q
git diff --check
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall tldw_chatbook/Notifications tldw_chatbook/UI/SubscriptionWindow.py tldw_chatbook/UI/Subscription_Modules tldw_chatbook/tldw_api Tests/Notifications Tests/UI/test_subscription_window_watchlists.py Tests/tldw_api/test_server_notifications_client.py -q
```

Expected: all commands pass before committing.
