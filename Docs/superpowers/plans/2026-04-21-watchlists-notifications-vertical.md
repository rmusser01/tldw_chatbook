# Watchlists And Client Notifications Vertical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend-aware watchlists shell to the existing subscriptions screen, with local subscriptions-backed local mode, server watchlist source CRUD in server mode, and a dedicated persisted local notifications inbox.

**Architecture:** Build the vertical in three layers: first the policy/storage/API foundations, then the watchlists service/scope seams, then the backend-aware `SubscriptionWindow` / `SubscriptionScreen` UI controllers. Keep local subscriptions authoritative in local mode, keep server watchlists live remote-only in server mode, and route all inbox mutations through runtime policy plus one local notification dispatch/store path.

**Tech Stack:** Python, Textual, SQLite via `BaseDB`, Pydantic schema models, `httpx` API client methods, runtime policy registry/enforcer, `pytest`, `unittest.mock`

**Implementation progress:** Foundation and backend service layers are implemented. Completed: runtime-policy notification queue update action, local notification inbox DB, notification dispatch service, watchlist source API schemas/client methods, local/server watchlists services, watchlist scope routing, and app bootstrap wiring. Deferred for the parallel UX rewrite: backend-aware `SubscriptionWindow`/`SubscriptionScreen` controller integration and visual tab work.

---

## Execution Notes

- Execute this plan from a fresh dedicated worktree off `dev`. The current `dev` branch is already dirty and has parallel edits in flight.
- Do not expand scope into watchlist jobs/runs, restore UI, server reminder feeds, raw server `settings` editing, or forum-source editing.
- Keep commits task-scoped. The point of this plan is to make rollback and review cheap.

## File Structure

### Create

- `tldw_chatbook/tldw_chatbook/tldw_api/watchlists_schemas.py`
  Defines first-slice watchlists source request/response models, including reversible delete metadata.
- `tldw_chatbook/tldw_chatbook/Notifications/__init__.py`
  Exports the local notifications store and dispatch service.
- `tldw_chatbook/tldw_chatbook/Notifications/client_notifications_db.py`
  Dedicated SQLite-backed local notification queue/inbox store.
- `tldw_chatbook/tldw_chatbook/Notifications/notification_dispatch_service.py`
  One dispatch path that writes queue records and attempts toast / notify delivery.
- `tldw_chatbook/tldw_chatbook/Subscriptions/watchlist_normalizers.py`
  Shared normalization helpers for local subscriptions rows and server watchlist source rows.
- `tldw_chatbook/tldw_chatbook/Subscriptions/local_watchlists_service.py`
  Thin local adapter over `SubscriptionsDB` so the shared scope layer does not need a second ad hoc local code path.
- `tldw_chatbook/tldw_chatbook/Subscriptions/server_watchlists_service.py`
  Thin server-backed watchlists source service around `TLDWAPIClient`.
- `tldw_chatbook/tldw_chatbook/Subscriptions/watchlist_scope_service.py`
  Source-aware routing layer over local subscriptions + server watchlists.
- `tldw_chatbook/tldw_chatbook/UI/Subscription_Modules/__init__.py`
  Exports backend-aware controllers for the subscriptions/watchlists shell.
- `tldw_chatbook/tldw_chatbook/UI/Subscription_Modules/subscription_backend_controller.py`
  Owns runtime backend refresh generation, list/detail/mutation routing, dirty-state coordination, and worker teardown decisions.
- `tldw_chatbook/tldw_chatbook/UI/Subscription_Modules/notifications_inbox_controller.py`
  Owns inbox list/load/mark/dismiss behavior through policy + store seams.
- `Tests/tldw_api/test_watchlists_schemas.py`
  Schema coverage for first-slice watchlists source models.
- `Tests/tldw_api/test_watchlists_client.py`
  Client-method coverage for watchlists source CRUD.
- `Tests/Subscriptions/test_client_notifications_db.py`
  Local notification queue schema and CRUD tests.
- `Tests/Subscriptions/test_notification_dispatch_service.py`
  Queue insert + toast/notify fallback behavior tests.
- `Tests/Subscriptions/test_server_watchlists_service.py`
  Server watchlists source service tests.
- `Tests/Subscriptions/test_watchlist_scope_service.py`
  Local/server routing, policy enforcement, and normalization tests.
- `Tests/UI/test_subscription_screen.py`
  Screen lifecycle tests for backend switching, resume/suspend, and shell state sync.
- `Tests/UI/test_subscription_window_watchlists.py`
  Window/controller tests for local/server mode behavior and notifications tab behavior.

### Modify

- `tldw_chatbook/tldw_chatbook/tldw_api/client.py`
  Add first-slice watchlists source CRUD methods.
- `tldw_chatbook/tldw_chatbook/tldw_api/__init__.py`
  Export new watchlists schema models.
- `tldw_chatbook/tldw_chatbook/config.py`
  Add `get_notifications_db_path()`.
- `tldw_chatbook/tldw_chatbook/runtime_policy/registry.py`
  Add `notifications.queue.update.local`.
- `Tests/RuntimePolicy/test_runtime_policy_core.py`
  Assert the new notification queue mutation action exists.
- `tldw_chatbook/tldw_chatbook/Subscriptions/__init__.py`
  Export watchlists normalizers/scope/service helpers.
- `tldw_chatbook/tldw_chatbook/app.py`
  Bootstrap server watchlists service, watchlist scope service, notifications store, and dispatch service.
- `tldw_chatbook/tldw_chatbook/UI/SubscriptionWindow.py`
  Delegate backend-aware behavior to new controllers, add `Notifications` tab, keep local-only surfaces disabled in server mode, and expose backend-aware accessors used by the screen.
- `tldw_chatbook/tldw_chatbook/UI/Screens/subscription_screen.py`
  Stop assuming local DB state, add backend-change handling, and use backend-aware window accessors.
- `Tests/UI/test_screen_navigation.py`
  Extend app bootstrap assertions for new services if existing coverage fits better than duplicating app boot tests elsewhere.

## Task Order

### Task 1: Runtime Policy And Local Notifications Foundation

**Files:**
- Create: `tldw_chatbook/tldw_chatbook/Notifications/__init__.py`
- Create: `tldw_chatbook/tldw_chatbook/Notifications/client_notifications_db.py`
- Create: `tldw_chatbook/tldw_chatbook/Notifications/notification_dispatch_service.py`
- Create: `Tests/Subscriptions/test_client_notifications_db.py`
- Create: `Tests/Subscriptions/test_notification_dispatch_service.py`
- Modify: `tldw_chatbook/tldw_chatbook/config.py`
- Modify: `tldw_chatbook/tldw_chatbook/runtime_policy/registry.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_core.py`

- [ ] **Step 1: Write the failing runtime-policy and notifications-store tests**

```python
def test_client_notifications_capability_exposes_queue_update_local():
    action_ids = CAPABILITY_ACTION_MATRIX["client_notifications"]
    assert "notifications.queue.update.local" in action_ids


def test_client_notifications_store_round_trips_read_and_dismiss(tmp_path):
    db = ClientNotificationsDB(tmp_path / "notifications.db")
    row = db.insert_notification(
        category="watchlists",
        title="Deleted source",
        message="Source deleted with restore window.",
        severity="info",
        source_backend="server",
        source_entity_id="17",
        source_entity_kind="watchlist_source",
        payload={"restore_window_seconds": 10},
    )
    db.mark_read(row["id"], is_read=True)
    db.dismiss_notification(row["id"], is_dismissed=True)
    stored = db.list_notifications(limit=10)[0]
    assert stored["is_read"] is True
    assert stored["is_dismissed"] is True
```

- [ ] **Step 2: Run the new red tests**

Run: `pytest Tests/RuntimePolicy/test_runtime_policy_core.py -k "client_notifications" -v`
Expected: FAIL because `notifications.queue.update.local` is not registered yet.

Run: `pytest Tests/Subscriptions/test_client_notifications_db.py Tests/Subscriptions/test_notification_dispatch_service.py -v`
Expected: FAIL because the notifications module does not exist yet.

- [ ] **Step 3: Implement the config getter, local store, dispatch service, and policy registration**

```python
def get_notifications_db_path() -> Path:
    custom_path = get_cli_setting("database", "notifications_db_path", None)
    if custom_path:
        return Path(custom_path).expanduser().resolve()
    return get_user_data_dir() / "tldw_chatbook_notifications.db"


class ClientNotificationsDB(BaseDB):
    def insert_notification(...): ...
    def list_notifications(self, *, limit: int = 100, include_dismissed: bool = False) -> list[dict[str, Any]]: ...
    def mark_read(self, notification_id: int, *, is_read: bool) -> bool: ...
    def dismiss_notification(self, notification_id: int, *, is_dismissed: bool) -> bool: ...


class NotificationDispatchService:
    def dispatch(self, *, app: Any, category: str, title: str, message: str, ... ) -> dict[str, Any]:
        row = self.store.insert_notification(...)
        self._try_toast_or_notify(app=app, message=message, severity=severity)
        return row
```

- [ ] **Step 4: Re-run the targeted tests until they pass**

Run: `pytest Tests/RuntimePolicy/test_runtime_policy_core.py -k "client_notifications" -v`
Expected: PASS

Run: `pytest Tests/Subscriptions/test_client_notifications_db.py Tests/Subscriptions/test_notification_dispatch_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit the notifications foundation**

```bash
git add Tests/RuntimePolicy/test_runtime_policy_core.py \
        Tests/Subscriptions/test_client_notifications_db.py \
        Tests/Subscriptions/test_notification_dispatch_service.py \
        tldw_chatbook/tldw_chatbook/config.py \
        tldw_chatbook/tldw_chatbook/runtime_policy/registry.py \
        tldw_chatbook/tldw_chatbook/Notifications/__init__.py \
        tldw_chatbook/tldw_chatbook/Notifications/client_notifications_db.py \
        tldw_chatbook/tldw_chatbook/Notifications/notification_dispatch_service.py
git commit -m "feat: add client notifications store and policy actions"
```

### Task 2: Watchlists API Schemas And Client Methods

**Files:**
- Create: `tldw_chatbook/tldw_chatbook/tldw_api/watchlists_schemas.py`
- Create: `Tests/tldw_api/test_watchlists_schemas.py`
- Create: `Tests/tldw_api/test_watchlists_client.py`
- Modify: `tldw_chatbook/tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_chatbook/tldw_api/__init__.py`

- [ ] **Step 1: Write the failing schema and client tests**

```python
def test_source_delete_response_keeps_restore_window_metadata():
    payload = SourceDeleteResponse(
        success=True,
        source_id=17,
        restore_window_seconds=10,
        restore_expires_at="2026-04-21T12:00:00Z",
    )
    assert payload.restore_window_seconds == 10


@pytest.mark.asyncio
async def test_client_routes_watchlist_source_crud_calls():
    client = FakeWatchlistsClient()
    created = await client_wrapper.create_watchlist_source(
        SourceCreateRequest(name="AI", url="https://example.com/feed.xml", source_type="rss")
    )
    deleted = await client_wrapper.delete_watchlist_source(17)
    assert created["id"] == 17
    assert deleted["restore_window_seconds"] == 10
```

- [ ] **Step 2: Run the red API tests**

Run: `pytest Tests/tldw_api/test_watchlists_schemas.py Tests/tldw_api/test_watchlists_client.py -v`
Expected: FAIL because the watchlists schemas and client methods do not exist yet.

- [ ] **Step 3: Implement the first-slice watchlists schemas and client methods**

```python
class SourceCreateRequest(BaseModel):
    name: str
    url: AnyUrl
    source_type: Literal["rss", "site"]
    active: bool = True
    tags: list[str] | None = None


class SourceResponse(BaseModel):
    id: int
    name: str
    url: AnyUrl
    source_type: str
    tags: list[str] = Field(default_factory=list)
    group_ids: list[int] = Field(default_factory=list)


async def list_watchlist_sources(self, *, q=None, tags=None, page=1, size=50):
    return await self._request("GET", "/api/v1/watchlists/sources", params={...})


async def delete_watchlist_source(self, source_id: int):
    return await self._request("DELETE", f"/api/v1/watchlists/sources/{source_id}")
```

- [ ] **Step 4: Re-run the watchlists schema/client tests**

Run: `pytest Tests/tldw_api/test_watchlists_schemas.py Tests/tldw_api/test_watchlists_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit the watchlists API client layer**

```bash
git add Tests/tldw_api/test_watchlists_schemas.py \
        Tests/tldw_api/test_watchlists_client.py \
        tldw_chatbook/tldw_chatbook/tldw_api/watchlists_schemas.py \
        tldw_chatbook/tldw_chatbook/tldw_api/client.py \
        tldw_chatbook/tldw_chatbook/tldw_api/__init__.py
git commit -m "feat: add watchlists source api client"
```

### Task 3: Local/Server Watchlists Services, Normalizers, And Scope Routing

**Files:**
- Create: `tldw_chatbook/tldw_chatbook/Subscriptions/watchlist_normalizers.py`
- Create: `tldw_chatbook/tldw_chatbook/Subscriptions/local_watchlists_service.py`
- Create: `tldw_chatbook/tldw_chatbook/Subscriptions/server_watchlists_service.py`
- Create: `tldw_chatbook/tldw_chatbook/Subscriptions/watchlist_scope_service.py`
- Create: `Tests/Subscriptions/test_server_watchlists_service.py`
- Create: `Tests/Subscriptions/test_watchlist_scope_service.py`
- Modify: `tldw_chatbook/tldw_chatbook/Subscriptions/__init__.py`

- [ ] **Step 1: Write the failing service and scope tests**

```python
@pytest.mark.asyncio
async def test_server_watchlists_service_omits_group_ids_and_preserves_settings_on_update():
    client = FakeClient()
    service = ServerWatchlistsService(client=client)
    result = await service.update_source(
        17,
        name="Renamed",
        existing_settings={"rss": {"limit": 50}},
    )
    assert client.calls[-1] == (
        "update_watchlist_source",
        17,
        {"name": "Renamed", "settings": {"rss": {"limit": 50}}},
    )
    assert "group_ids" not in client.calls[-1][2]


@pytest.mark.asyncio
async def test_scope_service_routes_local_and_server_actions_with_watchlists_action_ids():
    policy = FakePolicy()
    scope = WatchlistScopeService(local_service=FakeLocalSubscriptions(), server_service=FakeServerWatchlists(), policy_enforcer=policy)
    await scope.list_watch_items(runtime_backend="server")
    assert policy.calls[0]["action_id"] == "watchlists.list.server"
```

- [ ] **Step 2: Run the red service/scope tests**

Run: `pytest Tests/Subscriptions/test_server_watchlists_service.py Tests/Subscriptions/test_watchlist_scope_service.py -v`
Expected: FAIL because the service, normalizers, and scope service do not exist yet.

- [ ] **Step 3: Implement the normalizers, server service, and scope routing**

```python
def normalize_local_subscription_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": f"local:subscription:{row['id']}",
        "backend": "local",
        "entity_kind": "subscription",
        "source_id": row["id"],
        "title": row["name"],
        "source_type": row["type"],
        "url": row["source"],
        "active": bool(row["is_active"]) and not bool(row["is_paused"]),
        "tags": _coerce_tags(row.get("tags")),
        "status_summary": _local_status_summary(row),
        "last_checked_or_scraped_at": row.get("last_checked"),
    }


class ServerWatchlistsService:
    async def update_source(self, source_id: int, *, name=_UNSET, url=_UNSET, source_type=_UNSET, active=_UNSET, tags=_UNSET, existing_settings=None):
        if source_type == "forum":
            raise ValueError("Forum sources are not supported in the first slice.")
        payload = SourceUpdateRequest(...)
        return normalize_server_watchlist_source(await self._require_client().update_watchlist_source(source_id, payload))


class LocalWatchlistsService:
    def __init__(self, *, db_factory: Callable[[], SubscriptionsDB]):
        self._db_factory = db_factory

    async def list_sources(self) -> list[dict[str, Any]]:
        db = self._db_factory()
        return [normalize_local_subscription_row(row) for row in db.get_all_subscriptions(include_inactive=True)]


class WatchlistScopeService:
    async def list_watch_items(self, *, runtime_backend: str) -> list[dict[str, Any]]: ...
    async def get_watch_item_detail(self, item_id: str, *, runtime_backend: str) -> dict[str, Any]: ...
    async def save_watch_item(self, *, runtime_backend: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...
    async def delete_watch_item(self, *, runtime_backend: str, item_id: str) -> dict[str, Any]: ...
```

- [ ] **Step 4: Re-run the service/scope tests**

Run: `pytest Tests/Subscriptions/test_server_watchlists_service.py Tests/Subscriptions/test_watchlist_scope_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit the watchlists service layer**

```bash
git add Tests/Subscriptions/test_server_watchlists_service.py \
        Tests/Subscriptions/test_watchlist_scope_service.py \
        tldw_chatbook/tldw_chatbook/Subscriptions/__init__.py \
        tldw_chatbook/tldw_chatbook/Subscriptions/watchlist_normalizers.py \
        tldw_chatbook/tldw_chatbook/Subscriptions/local_watchlists_service.py \
        tldw_chatbook/tldw_chatbook/Subscriptions/server_watchlists_service.py \
        tldw_chatbook/tldw_chatbook/Subscriptions/watchlist_scope_service.py
git commit -m "feat: add watchlists scope and server services"
```

### Task 4: App Bootstrap And Shared Service Wiring

**Files:**
- Modify: `tldw_chatbook/tldw_chatbook/app.py`
- Modify: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write the failing bootstrap assertions**

```python
def test_app_initializes_watchlists_and_notifications_services(app):
    assert app.server_watchlists_service is not None
    assert app.watchlist_scope_service is not None
    assert app.client_notifications_db is not None
    assert app.notification_dispatch_service is not None
```

- [ ] **Step 2: Run the red bootstrap test**

Run: `pytest Tests/UI/test_screen_navigation.py -k "watchlists or notifications" -v`
Expected: FAIL because the app does not wire the new services yet.

- [ ] **Step 3: Wire the services into app bootstrap**

```python
try:
    self.server_watchlists_service = ServerWatchlistsService.from_config(self.app_config)
except ValueError:
    self.server_watchlists_service = ServerWatchlistsService(client=None)

self.local_watchlists_service = LocalWatchlistsService(
    db_factory=lambda: SubscriptionsDB(get_subscriptions_db_path(), self.client_id)
)
self.watchlist_scope_service = WatchlistScopeService(
    local_service=self.local_watchlists_service,
    server_service=self.server_watchlists_service,
    policy_enforcer=self,
)
self.client_notifications_db = ClientNotificationsDB(get_notifications_db_path(), self.client_id)
self.notification_dispatch_service = NotificationDispatchService(store=self.client_notifications_db)
```

- [ ] **Step 4: Re-run the bootstrap test**

Run: `pytest Tests/UI/test_screen_navigation.py -k "watchlists or notifications" -v`
Expected: PASS

- [ ] **Step 5: Commit the bootstrap wiring**

```bash
git add Tests/UI/test_screen_navigation.py tldw_chatbook/tldw_chatbook/app.py
git commit -m "feat: wire watchlists and notifications services into app"
```

### Task 5: Backend-Aware Subscription Window Controllers And Notifications Tab

**Files:**
- Create: `tldw_chatbook/tldw_chatbook/UI/Subscription_Modules/__init__.py`
- Create: `tldw_chatbook/tldw_chatbook/UI/Subscription_Modules/subscription_backend_controller.py`
- Create: `tldw_chatbook/tldw_chatbook/UI/Subscription_Modules/notifications_inbox_controller.py`
- Create: `Tests/UI/test_subscription_window_watchlists.py`
- Modify: `tldw_chatbook/tldw_chatbook/UI/SubscriptionWindow.py`

- [ ] **Step 1: Write the failing window/controller tests**

```python
@pytest.mark.asyncio
async def test_server_mode_refresh_skips_local_scheduler_and_loads_watchlist_list():
    window = build_window(runtime_backend="server")
    await window.refresh_backend_view()
    assert window.scheduler_worker is None
    assert window.backend_controller.last_loaded_backend == "server"


@pytest.mark.asyncio
async def test_notifications_tab_marks_read_through_policy_and_store():
    window = build_window(runtime_backend="local")
    row = window.notifications_store.insert_notification(...)
    await window.notifications_controller.mark_read(row["id"], is_read=True)
    assert window.app_instance.require_ui_action_allowed.calls[-1]["action_id"] == "notifications.queue.update.local"


@pytest.mark.asyncio
async def test_remote_delete_notification_preserves_restore_window_metadata():
    window = build_window(runtime_backend="server")
    await window.backend_controller.delete_watch_item("server:source:17")
    notification = window.notifications_store.list_notifications(limit=1)[0]
    assert notification["payload"]["restore_window_seconds"] == 10
    assert notification["payload"]["restore_expires_at"] == "2026-04-21T12:00:00Z"
```

- [ ] **Step 2: Run the red window/controller tests**

Run: `pytest Tests/UI/test_subscription_window_watchlists.py -v`
Expected: FAIL because the controllers and backend-aware window API do not exist yet.

- [ ] **Step 3: Implement the backend controller, inbox controller, and window delegation**

```python
class SubscriptionBackendController:
    async def refresh_backend_view(self, *, runtime_backend: str) -> None: ...
    async def stop_active_backend_workers(self) -> None: ...
    def snapshot_shell_state(self) -> dict[str, Any]: ...
    async def delete_watch_item(self, item_id: str) -> dict[str, Any]:
        result = await self.scope_service.delete_watch_item(runtime_backend="server", item_id=item_id)
        self.notification_dispatch_service.dispatch(
            app=self.app_instance,
            category="watchlists",
            title="Watchlist source deleted",
            message="Server source deleted within restore window.",
            payload={
                "source_id": result["source_id"],
                "restore_window_seconds": result.get("restore_window_seconds"),
                "restore_expires_at": result.get("restore_expires_at"),
            },
        )
        return result


class NotificationsInboxController:
    async def load_rows(self) -> list[dict[str, Any]]: ...
    async def mark_read(self, notification_id: int, *, is_read: bool) -> bool: ...
    async def dismiss(self, notification_id: int, *, is_dismissed: bool) -> bool: ...


class SubscriptionWindow(Container):
    async def refresh_backend_view(self) -> None:
        runtime_backend = self._runtime_backend()
        await self.backend_controller.refresh_backend_view(runtime_backend=runtime_backend)

    async def stop_active_backend_workers(self) -> None:
        await self.backend_controller.stop_active_backend_workers()

    async def delete_selected_watch_item(self) -> dict[str, Any] | None:
        item_id = self._active_watch_item_id()
        if item_id is None:
            return None
        return await self.backend_controller.delete_watch_item(item_id)
```

- [ ] **Step 4: Add the `Notifications` tab and server-mode degradation states**

```python
with TabPane("Notifications", id="notifications"):
    yield ListView(id="notifications-list")
    yield Button("Mark Read", id="notification-mark-read-btn")
    yield Button("Dismiss", id="notification-dismiss-btn")

if runtime_backend == "server":
    self._render_local_only_state(tab_id="review", message="Local-only in this slice.")
```

- [ ] **Step 5: Re-run the window/controller tests**

Run: `pytest Tests/UI/test_subscription_window_watchlists.py -v`
Expected: PASS

- [ ] **Step 6: Commit the window/controller refactor**

```bash
git add Tests/UI/test_subscription_window_watchlists.py \
        tldw_chatbook/tldw_chatbook/UI/Subscription_Modules/__init__.py \
        tldw_chatbook/tldw_chatbook/UI/Subscription_Modules/subscription_backend_controller.py \
        tldw_chatbook/tldw_chatbook/UI/Subscription_Modules/notifications_inbox_controller.py \
        tldw_chatbook/tldw_chatbook/UI/SubscriptionWindow.py
git commit -m "feat: add backend-aware subscription window controllers"
```

### Task 6: Subscription Screen Lifecycle Refactor

**Files:**
- Create: `Tests/UI/test_subscription_screen.py`
- Modify: `tldw_chatbook/tldw_chatbook/UI/Screens/subscription_screen.py`

- [ ] **Step 1: Write the failing screen lifecycle tests**

```python
@pytest.mark.asyncio
async def test_handle_runtime_backend_changed_delegates_to_window_refresh():
    screen = build_screen()
    window = screen.subscription_window
    await screen.handle_runtime_backend_changed("server")
    assert window.calls[:2] == ["stop_active_backend_workers", "refresh_backend_view"]


@pytest.mark.asyncio
async def test_screen_resume_does_not_call_local_only_refreshes_in_server_mode():
    screen = build_screen(runtime_backend="server")
    await screen.on_screen_resume()
    assert screen.subscription_window.refresh_dashboard.await_count == 0
    assert screen.subscription_window.load_new_items.await_count == 0


@pytest.mark.asyncio
async def test_screen_suspend_stops_active_backend_workers():
    screen = build_screen(runtime_backend="server")
    await screen.on_screen_suspend()
    screen.subscription_window.stop_active_backend_workers.assert_awaited_once()
```

- [ ] **Step 2: Run the red screen tests**

Run: `pytest Tests/UI/test_subscription_screen.py -v`
Expected: FAIL because the screen still assumes `window.db`, does not stop active backend workers before server refresh, and does not stop them on suspend.

- [ ] **Step 3: Refactor the screen to use backend-aware window accessors**

```python
async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
    window = self._get_subscription_window()
    if window is None:
        return
    await window.stop_active_backend_workers()
    await window.refresh_backend_view()
    self._sync_state_from_window()


async def on_screen_suspend(self) -> None:
    window = self._get_subscription_window()
    if window is None:
        return
    await window.stop_active_backend_workers()


def _sync_state_from_window(self) -> None:
    window = self._get_subscription_window()
    if window is None:
        return
    snapshot = window.snapshot_shell_state()
    self.subscriptions = snapshot["items"]
    self.active_subscription = snapshot.get("active_item")
    self.is_checking_updates = snapshot.get("is_busy", False)
    self.last_check_time = snapshot.get("last_checked_or_scraped_at")
```

- [ ] **Step 4: Re-run the screen tests**

Run: `pytest Tests/UI/test_subscription_screen.py -v`
Expected: PASS

- [ ] **Step 5: Commit the screen lifecycle refactor**

```bash
git add Tests/UI/test_subscription_screen.py tldw_chatbook/tldw_chatbook/UI/Screens/subscription_screen.py
git commit -m "feat: make subscription screen backend aware"
```

### Task 7: Final Vertical Regression Sweep

**Files:**
- Modify: `Tests/Subscriptions/test_subscriptions_smoke.py` (only if the new shell behavior requires expectation updates)
- Review only: `Docs/superpowers/specs/2026-04-21-watchlists-notifications-vertical-design.md`

- [ ] **Step 1: Add or update the final smoke assertion set only if current subscriptions smoke coverage breaks**

```python
def test_subscription_shell_still_supports_local_mode_after_watchlists_vertical():
    ...
```

- [ ] **Step 2: Run the full targeted regression set**

Run: `pytest Tests/RuntimePolicy/test_runtime_policy_core.py Tests/tldw_api/test_watchlists_schemas.py Tests/tldw_api/test_watchlists_client.py Tests/Subscriptions/test_client_notifications_db.py Tests/Subscriptions/test_notification_dispatch_service.py Tests/Subscriptions/test_server_watchlists_service.py Tests/Subscriptions/test_watchlist_scope_service.py Tests/UI/test_subscription_screen.py Tests/UI/test_subscription_window_watchlists.py Tests/Subscriptions/test_subscriptions_smoke.py -v`
Expected: PASS

- [ ] **Step 3: Run a narrower follow-up rerun for the UI suite if the full sweep fails**

Run: `pytest Tests/UI/test_subscription_screen.py Tests/UI/test_subscription_window_watchlists.py -v`
Expected: PASS

- [ ] **Step 4: Review the implementation against the spec before the final commit**

```text
Confirm all of the following are still true:
- no restore UI
- no jobs/runs
- no raw settings editor
- no forum editing
- no local shadow copy of server watchlists
```

- [ ] **Step 5: Commit the final vertical integration pass only if the regression sweep required code changes**

```bash
git add Tests/Subscriptions/test_subscriptions_smoke.py
git commit -m "test: finalize watchlists notifications vertical regression coverage"
```
