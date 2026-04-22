# Media Read-it-later Combined Vertical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved `Media / Read-it-later / Ingestion Sources` vertical so Chatbook supports local and server `Read-it-later` behavior through one source-aware seam while exposing first-slice server ingestion-source creation from the ingest TUI.

**Architecture:** Build this in two milestones on top of the existing media seam. Milestone A lands the saved-state contract, local persistence, server compatibility mapping, and `Media` browse/viewer behavior. Milestone B extends the existing server ingestion-source panel with first-slice create support for `archive_snapshot` and `git_repository` without redesigning the ingest product or overpromising unsupported server features.

**Tech Stack:** Python 3.11+, Textual, SQLite, existing `MediaDatabase`, existing `tldw_api` client/schemas, pytest

---

## Scope Check

This plan stays within the approved combined vertical because the work is coupled at one seam:

- the normalized media contract
- the source-aware media scope service
- the shared media runtime state
- the two existing media shells: `MediaWindow_v2` and `MediaIngestWindowRebuilt`

Execution is still split into two implementation milestones:

- Milestone A: `Read-it-later` contract, persistence, scope service, and `Media` browse/viewer behavior
- Milestone B: first-slice server ingestion-source create in the existing ingest/source-management panel

This plan does **not** implement:

- a separate `Read-it-later` destination
- mixed local/server browse results
- sync, mirroring, or dual-write behavior
- server ingestion-source delete
- per-media-type server `Read-it-later` saved views
- a broad `Media` or `Ingest` UI redesign

Implementation note for agentic workers:

- Use `@superpowers:test-driven-development` before each code change.
- Use `@superpowers:verification-before-completion` before claiming any task complete.

## File Map

- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
  Responsibility: Add the local-only `Read-it-later` persistence table/helpers and DB-native filtering support that can be reused by the local media adapter.
- Modify: `tldw_chatbook/Media/local_media_reading_service.py`
  Responsibility: Expose local save/remove/list `Read-it-later` operations and route local saved-view browsing through DB-backed filtering instead of in-memory filtering.
- Modify: `tldw_chatbook/Media/server_media_reading_service.py`
  Responsibility: Add server ingestion-source create and keep server `Read-it-later` mutations mapped to reading-item status updates only.
- Modify: `tldw_chatbook/Media/media_reading_normalizers.py`
  Responsibility: Extend normalized records with `supports_read_it_later`, `is_read_it_later`, and `read_it_later_saved_at` while keeping server timestamps honest.
- Modify: `tldw_chatbook/Media/media_reading_scope_service.py`
  Responsibility: Add explicit `Read-it-later` list/save/remove operations, add server ingestion-source create, and enforce the correct runtime-policy action IDs.
- Modify: `tldw_chatbook/UI/Screens/media_runtime_state.py`
  Responsibility: Add source-scoped browse subview state and keep backend-switch resets safe.
- Modify: `tldw_chatbook/Widgets/Media/media_search_panel.py`
  Responsibility: Add the `All` vs `Read-it-later` browse-subview control without collapsing it into media-type navigation.
- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
  Responsibility: Route saved-view browsing and save/remove actions through the explicit scope-service operations, honor the server `All Media` restriction, and keep selection/cache behavior correct.
- Modify: `tldw_chatbook/Event_Handlers/media_events.py`
  Responsibility: Define any new media events needed for saved-state UI actions and keep record identity consistent.
- Modify: `tldw_chatbook/Widgets/Media/media_viewer_panel.py`
  Responsibility: Expose one source-aware save/remove affordance for the current record and surface unsupported-state behavior cleanly.
- Modify: `tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py`
  Responsibility: Extend the existing server-source management panel with first-slice create controls restricted to `archive_snapshot` and `git_repository`.
- Modify: `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
  Responsibility: Keep backend-aware tab refresh behavior correct as the source panel grows create support.

- Create: `Tests/Media/test_local_media_reading_service.py`
  Responsibility: Verify the local adapter uses DB-backed filtering for saved views and routes save/remove persistence correctly.
- Modify: `Tests/Media_DB/test_media_db_v2.py`
  Responsibility: Verify local `Read-it-later` persistence CRUD, saved-at behavior, hidden-by-default deleted rows, and physical-delete cleanup semantics.
- Modify: `Tests/Media/test_media_reading_normalizers.py`
  Responsibility: Verify the normalized saved-state fields and the server no-fake-timestamp rule.
- Modify: `Tests/Media/test_server_media_reading_service.py`
  Responsibility: Verify server create-source wiring plus status-based saved/remove compatibility mapping.
- Modify: `Tests/Media/test_media_reading_scope_service.py`
  Responsibility: Verify explicit `Read-it-later` operations, policy action IDs, and server ingestion-source create routing.
- Modify: `Tests/UI/test_media_runtime_state.py`
  Responsibility: Verify browse-subview runtime-state behavior and backend reset semantics.
- Modify: `Tests/UI/test_media_window_v2_parity.py`
  Responsibility: Verify saved-subview browsing, server `All Media` restriction behavior, save/remove event handling, and filtered selection refresh.
- Modify: `Tests/UI/test_media_ingestion_source_panel.py`
  Responsibility: Verify first-slice create behavior, allowed source-type restriction, refresh/selection after create, and local-mode disabled behavior.
- Modify: `Tests/UI/test_media_ingest_window_rebuilt.py`
  Responsibility: Verify backend-aware ingest refresh still delegates correctly once create support is added.

- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
  Responsibility: Mark the media/read-it-later vertical based on verified behavior, not intended behavior.
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
  Responsibility: Move the specific media/read-it-later gaps from open to landed or narrowed-follow-up state.
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
  Responsibility: Update rollout status and remaining follow-on items after verification.

## Task 1: Add Local `Read-it-later` Persistence And Adapter Support

**Files:**
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
- Modify: `tldw_chatbook/Media/local_media_reading_service.py`
- Create: `Tests/Media/test_local_media_reading_service.py`
- Modify: `Tests/Media_DB/test_media_db_v2.py`

- [ ] **Step 1: Write the failing local DB/adapter tests**

```python
def test_read_it_later_state_round_trips_for_local_media(db_instance):
    media_id, _ = db_instance.add_media_with_keywords(
        title="Reader",
        content="Hello",
        media_type="article",
        keywords=[],
    )

    saved = db_instance.save_media_to_read_it_later(media_id)

    assert saved["media_id"] == media_id
    assert saved["is_read_it_later"] is True
    assert saved["saved_at"] is not None
    assert db_instance.get_media_read_it_later_state(media_id)["is_read_it_later"] is True


def test_local_service_search_media_uses_db_backed_saved_filter(memory_db_factory):
    db = memory_db_factory()
    kept_id, _ = db.add_media_with_keywords(title="Keep", content="A", media_type="article", keywords=[])
    other_id, _ = db.add_media_with_keywords(title="Drop", content="B", media_type="article", keywords=[])
    db.save_media_to_read_it_later(kept_id)

    service = LocalMediaReadingService(db)
    payload = service.search_media(read_it_later_only=True)

    assert [item["id"] for item in payload["items"]] == [kept_id]
    assert all(item["id"] != other_id for item in payload["items"])


def test_soft_deleted_local_media_is_hidden_from_saved_view_by_default(db_instance):
    media_id, _ = db_instance.add_media_with_keywords(
        title="Reader",
        content="Hello",
        media_type="article",
        keywords=[],
    )
    db_instance.save_media_to_read_it_later(media_id)
    db_instance.soft_delete_media(media_id)

    ids = db_instance.list_read_it_later_media_ids()
    assert media_id not in ids
```

- [ ] **Step 2: Run the focused local DB/adapter tests to verify they fail**

Run: `python3 -m pytest Tests/Media_DB/test_media_db_v2.py Tests/Media/test_local_media_reading_service.py -q`
Expected: FAIL with missing local saved-state helpers and missing local adapter support.

- [ ] **Step 3: Implement the minimal local persistence layer**

```python
def save_media_to_read_it_later(self, media_id: int) -> dict[str, Any]:
    current_time = self._get_current_utc_timestamp_str()
    with self.transaction() as conn:
        conn.execute(
            """
            INSERT INTO MediaReadItLaterState (media_id, is_read_it_later, saved_at, updated_at)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(media_id) DO UPDATE SET
                is_read_it_later = 1,
                saved_at = COALESCE(MediaReadItLaterState.saved_at, excluded.saved_at),
                updated_at = excluded.updated_at
            """,
            (media_id, current_time, current_time),
        )
    return self.get_media_read_it_later_state(media_id)


def list_read_it_later_media_ids(self, *, include_deleted: bool = False, include_trash: bool = False) -> list[int]:
    sql = """
        SELECT s.media_id
        FROM MediaReadItLaterState s
        JOIN Media m ON m.id = s.media_id
        WHERE s.is_read_it_later = 1
    """
    # append deleted/trash filters here
```

Implementation constraints for this task:

- Use one local-only table keyed by `media_id`.
- Store `saved_at` and `updated_at` locally.
- Do not write sync-log rows or bump `Media.version`.
- Keep soft-deleted and trashed rows out of the default saved view.
- Prefer DB-backed filtering using `media_ids_filter` or a joined query, not in-memory post-filtering.
- If the table uses a foreign key to `Media(id)`, physical delete cleanup can be automatic; do not add extra deletion code unless the FK path is insufficient.

- [ ] **Step 4: Run the local DB/adapter tests again**

Run: `python3 -m pytest Tests/Media_DB/test_media_db_v2.py Tests/Media/test_local_media_reading_service.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/Client_Media_DB_v2.py tldw_chatbook/Media/local_media_reading_service.py Tests/Media_DB/test_media_db_v2.py Tests/Media/test_local_media_reading_service.py
git commit -m "feat: add local media read-it-later persistence"
```

## Task 2: Extend The Normalizers And Scope Service For Explicit Saved-State Operations

**Files:**
- Modify: `tldw_chatbook/Media/media_reading_normalizers.py`
- Modify: `tldw_chatbook/Media/server_media_reading_service.py`
- Modify: `tldw_chatbook/Media/media_reading_scope_service.py`
- Modify: `Tests/Media/test_media_reading_normalizers.py`
- Modify: `Tests/Media/test_server_media_reading_service.py`
- Modify: `Tests/Media/test_media_reading_scope_service.py`

- [ ] **Step 1: Write the failing service/normalizer tests**

```python
def test_normalize_server_reading_item_exposes_saved_state_without_fake_saved_timestamp():
    normalized = normalize_server_reading_item(
        {
            "id": 41,
            "media_id": 99,
            "title": "Server Article",
            "status": "saved",
            "updated_at": "2026-04-21T10:00:00Z",
        }
    )

    assert normalized["supports_read_it_later"] is True
    assert normalized["is_read_it_later"] is True
    assert normalized["read_it_later_saved_at"] is None


@pytest.mark.asyncio
async def test_scope_service_save_and_remove_use_explicit_reading_list_actions():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(local_service=None, server_service=server, policy_enforcer=policy)

    await scope.save_to_read_it_later(mode="server", media_id=41)
    await scope.remove_from_read_it_later(mode="server", media_id=41)

    assert policy.calls == [
        "collections.reading_list.create.server",
        "collections.reading_list.delete.server",
    ]
    assert ("update_media_metadata", 41, {"status": "saved"}) in server.calls
    assert ("update_media_metadata", 41, {"status": "archived"}) in server.calls


@pytest.mark.asyncio
async def test_scope_service_can_create_server_ingestion_source():
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(local_service=None, server_service=server)

    created = await scope.create_ingestion_source(
        mode="server",
        source_type="git_repository",
        sink_type="media",
        policy="canonical",
        config={"repo_url": "https://example.com/repo.git"},
    )

    assert created["entity_kind"] == "ingestion_source"
    assert created["source_type"] == "git_repository"
```

- [ ] **Step 2: Run the service/normalizer tests to verify they fail**

Run: `python3 -m pytest Tests/Media/test_media_reading_normalizers.py Tests/Media/test_server_media_reading_service.py Tests/Media/test_media_reading_scope_service.py -q`
Expected: FAIL with missing normalized saved-state fields, missing explicit saved-state operations, and missing server create-source support.

- [ ] **Step 3: Implement the minimal contract and service extensions**

```python
async def list_read_it_later(
    self,
    *,
    mode: MediaReadingBackend | str | None = None,
    query: str | None = None,
    limit: int = 20,
    offset: int = 0,
    **filters: Any,
) -> dict[str, Any]:
    normalized_mode = self._normalize_mode(mode)
    self._enforce_policy(f"collections.reading_list.list.{normalized_mode.value}")
    service = self._service_for_mode(normalized_mode)
    payload = await self._maybe_await(
        service.search_media(query=query, limit=limit, offset=offset, read_it_later_only=True, **filters)
        if normalized_mode == MediaReadingBackend.LOCAL
        else service.search_media(query=query, limit=limit, offset=offset, status=["saved"], **filters)
    )
    raw_items = list(payload.get("items", []))
    items = [self._normalize_media_record(normalized_mode, item) for item in raw_items]
    return {"items": items, "total": payload.get("total", len(items)), "offset": offset, "limit": limit}


async def save_to_read_it_later(...):
    # local => dedicated local persistence helper
    # server => update_media_metadata(media_id, status="saved")


async def remove_from_read_it_later(...):
    # local => remove local persistence row
    # server => update_media_metadata(media_id, status="archived")
```

Implementation constraints for this task:

- Normalized local and server records must expose:
  - `supports_read_it_later`
  - `is_read_it_later`
  - `read_it_later_saved_at`
- Server records must never synthesize `read_it_later_saved_at` from `updated_at`.
- `save_to_read_it_later` and `remove_from_read_it_later` must use `collections.reading_list.create.*` and `collections.reading_list.delete.*`.
- `list_read_it_later` must use `collections.reading_list.list.*`.
- Server ingestion-source create must enforce `media.ingestion_sources.create.*`.
- `ServerMediaReadingService` should add `create_ingestion_source(...)` using the existing `IngestionSourceCreateRequest` client contract.
- Local ingestion-source create should continue to fail explicitly.

- [ ] **Step 4: Run the service/normalizer tests again**

Run: `python3 -m pytest Tests/Media/test_media_reading_normalizers.py Tests/Media/test_server_media_reading_service.py Tests/Media/test_media_reading_scope_service.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Media/media_reading_normalizers.py tldw_chatbook/Media/server_media_reading_service.py tldw_chatbook/Media/media_reading_scope_service.py Tests/Media/test_media_reading_normalizers.py Tests/Media/test_server_media_reading_service.py Tests/Media/test_media_reading_scope_service.py
git commit -m "feat: add explicit media read-it-later service operations"
```

## Task 3: Add Browse-Subview State And Saved-View Search Routing To `Media`

**Files:**
- Modify: `tldw_chatbook/UI/Screens/media_runtime_state.py`
- Modify: `tldw_chatbook/Widgets/Media/media_search_panel.py`
- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
- Modify: `Tests/UI/test_media_runtime_state.py`
- Modify: `Tests/UI/test_media_window_v2_parity.py`

- [ ] **Step 1: Write the failing runtime-state and browse tests**

```python
def test_media_runtime_state_backend_reset_restores_safe_saved_view_defaults():
    state = MediaRuntimeState(runtime_backend="server")
    state.active_browse_subview = "read-it-later"
    state.selected_record_id = "server:reading_item:41"

    state.reset_for_backend("local")

    assert state.runtime_backend == "local"
    assert state.active_browse_subview == "all"
    assert state.selected_record_id is None


@pytest.mark.asyncio
async def test_media_window_uses_explicit_saved_view_search_for_read_it_later_subview():
    scope_service = Mock()
    scope_service.list_read_it_later = AsyncMock(return_value={"items": [{"id": "local:media:7", "title": "Saved"}], "total": 1})
    window, _app = _build_media_window(runtime_backend="local", scope_service=scope_service)
    window.runtime_state.active_browse_subview = "read-it-later"

    tasks = []
    window.run_worker = lambda coro, exclusive=True: tasks.append(asyncio.create_task(coro))

    window._perform_search("all-media", "", "")
    await asyncio.gather(*tasks)

    scope_service.list_read_it_later.assert_awaited_once()


@pytest.mark.asyncio
async def test_media_window_forces_server_saved_view_back_to_all_media_when_type_is_not_all_media():
    window, app = _build_media_window(runtime_backend="server", scope_service=Mock())
    window.runtime_state.active_browse_subview = "read-it-later"

    window.activate_media_type("article", "Article")

    assert window.runtime_state.active_browse_subview == "all"
    app.notify.assert_called()
```

- [ ] **Step 2: Run the focused UI tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_media_runtime_state.py Tests/UI/test_media_window_v2_parity.py -q`
Expected: FAIL with missing browse-subview state and missing saved-view routing.

- [ ] **Step 3: Implement the minimal saved-view browse wiring**

```python
@dataclass
class MediaRuntimeState:
    runtime_backend: str = "local"
    active_media_type: str | None = None
    active_browse_subview: str = "all"
    ...


def _saved_view_available_for_context(self) -> bool:
    if self._runtime_backend() != "server":
        return True
    return (self.active_media_type or "all-media") in {None, "all-media"}


if self.runtime_state.active_browse_subview == "read-it-later":
    payload = await scope_service.list_read_it_later(...)
else:
    payload = await scope_service.search_media(...)
```

Implementation constraints for this task:

- Keep media-type navigation and browse-subview selection separate.
- The new subview control should live with search/filter controls, not as a fake media type.
- In server mode, `Read-it-later` is only valid for aggregate `All Media`.
- If the user lands in an invalid server context, reset to `all` and notify once; do not return a silently partial saved view.
- Saved-view browsing must not fetch generic results and then hide rows in memory.

- [ ] **Step 4: Run the UI tests again**

Run: `python3 -m pytest Tests/UI/test_media_runtime_state.py Tests/UI/test_media_window_v2_parity.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/media_runtime_state.py tldw_chatbook/Widgets/Media/media_search_panel.py tldw_chatbook/UI/MediaWindow_v2.py Tests/UI/test_media_runtime_state.py Tests/UI/test_media_window_v2_parity.py
git commit -m "feat: add media read-it-later browse subview"
```

## Task 4: Add The Record-Level Save/Remove Affordance In The Media Viewer

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/media_events.py`
- Modify: `tldw_chatbook/Widgets/Media/media_viewer_panel.py`
- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
- Modify: `Tests/UI/test_media_window_v2_parity.py`

- [ ] **Step 1: Write the failing viewer/action tests**

```python
def test_media_viewer_updates_saved_button_label_from_normalized_record():
    panel = MediaViewerPanel(Mock())
    panel.media_data = {
        "id": "local:media:7",
        "title": "Saved Item",
        "supports_read_it_later": True,
        "is_read_it_later": True,
    }
    panel._update_read_it_later_button = Mock()

    panel.load_media(panel.media_data)

    panel._update_read_it_later_button.assert_called_once()


@pytest.mark.asyncio
async def test_media_window_remove_from_saved_view_clears_selection_when_filtered_out():
    scope_service = Mock()
    scope_service.remove_from_read_it_later = AsyncMock(return_value={"id": "local:media:7", "is_read_it_later": False})
    scope_service.list_read_it_later = AsyncMock(return_value={"items": [], "total": 0})
    window, _app = _build_media_window(runtime_backend="local", scope_service=scope_service)
    window.runtime_state.active_browse_subview = "read-it-later"
    window.runtime_state.selected_record_id = "local:media:7"

    await window._handle_read_it_later_toggle_async(
        MediaReadItLaterToggleEvent(record_id="local:media:7", media_id="7", save_for_later=False)
    )

    assert window.runtime_state.selected_record_id is None
```

- [ ] **Step 2: Run the focused viewer/action tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_media_window_v2_parity.py -q`
Expected: FAIL with missing saved-state event/button behavior.

- [ ] **Step 3: Implement the minimal viewer action flow**

```python
class MediaReadItLaterToggleEvent(Message):
    def __init__(self, media_id: Any, *, record_id: Any = None, save_for_later: bool = True) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.save_for_later = save_for_later


if event.save_for_later:
    updated = await scope_service.save_to_read_it_later(...)
else:
    updated = await scope_service.remove_from_read_it_later(...)
```

Implementation constraints for this task:

- The viewer should expose one button whose label flips between `Save for Later` and `Remove from Read-it-later`.
- Disable or hide the button when `supports_read_it_later` is false.
- After mutation succeeds:
  - update cached detail for that record
  - refresh the current browse results through the correct search path
  - clear selection if the current record no longer belongs in the filtered result set
- Do not leave optimistic saved-state behind after failure.

- [ ] **Step 4: Run the focused viewer/action tests again**

Run: `python3 -m pytest Tests/UI/test_media_window_v2_parity.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Event_Handlers/media_events.py tldw_chatbook/Widgets/Media/media_viewer_panel.py tldw_chatbook/UI/MediaWindow_v2.py Tests/UI/test_media_window_v2_parity.py
git commit -m "feat: add media viewer save for later actions"
```

## Task 5: Extend The Server Ingestion Source Panel With First-Slice Create Support

**Files:**
- Modify: `tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py`
- Modify: `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
- Modify: `Tests/UI/test_media_ingestion_source_panel.py`
- Modify: `Tests/UI/test_media_ingest_window_rebuilt.py`

- [ ] **Step 1: Write the failing ingestion-panel tests**

```python
@pytest.mark.asyncio
async def test_ingestion_source_panel_create_is_disabled_in_local_mode():
    scope_service = Mock()
    app = SourcePanelTestApp(runtime_backend="local", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(MediaIngestionSourcePanel)
        await panel.refresh_for_mode()
        assert panel.query_one("#create-source-btn", Button).disabled is True


@pytest.mark.asyncio
async def test_ingestion_source_panel_creates_allowed_server_source_and_refreshes_selection():
    scope_service = Mock()
    scope_service.list_ingestion_sources = AsyncMock(return_value=[])
    scope_service.create_ingestion_source = AsyncMock(
        return_value={
            "id": "server:ingestion_source:7",
            "source_id": "7",
            "entity_kind": "ingestion_source",
            "source_type": "git_repository",
            "sink_type": "media",
            "policy": "canonical",
            "enabled": True,
        }
    )

    app = SourcePanelTestApp(runtime_backend="server", scope_service=scope_service)
    async with app.run_test() as pilot:
        panel = pilot.app.query_one(MediaIngestionSourcePanel)
        await panel.refresh_for_mode()
        panel.query_one("#create-source-type").value = "git_repository"
        panel.query_one("#create-config-input").value = '{\"repo_url\": \"https://example.com/repo.git\"}'

        await panel._create_source()

        scope_service.create_ingestion_source.assert_awaited_once()
```

- [ ] **Step 2: Run the focused ingest-panel tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_media_ingestion_source_panel.py Tests/UI/test_media_ingest_window_rebuilt.py -q`
Expected: FAIL with missing create controls and missing server create routing.

- [ ] **Step 3: Implement the minimal create-source UI**

```python
ALLOWED_CREATE_SOURCE_TYPES = ("archive_snapshot", "git_repository")

async def _create_source(self) -> None:
    source_type = str(self.query_one("#create-source-type", Select).value)
    if source_type not in ALLOWED_CREATE_SOURCE_TYPES:
        self.notify("This source type is not available from Chatbook yet.", severity="warning")
        return

    config = json.loads(self.query_one("#create-config-input", Input).value or "{}")
    created = await self._maybe_await(
        self.scope_service.create_ingestion_source(
            mode="server",
            source_type=source_type,
            sink_type="media",
            policy=str(self.query_one("#create-policy-type", Select).value),
            config=config,
        )
    )
```

Implementation constraints for this task:

- Keep the panel in the existing `Server Sources` tab.
- Limit create affordances to `archive_snapshot` and `git_repository`.
- Do not expose `local_directory` as a normal remote-client option.
- Keep `sink_type="media"` for this vertical.
- After create succeeds, refresh sources and select the new row if possible.
- Preserve current refresh behavior from `MediaIngestWindowRebuilt`.

- [ ] **Step 4: Run the ingest-panel tests again**

Run: `python3 -m pytest Tests/UI/test_media_ingestion_source_panel.py Tests/UI/test_media_ingest_window_rebuilt.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py tldw_chatbook/UI/MediaIngestWindowRebuilt.py Tests/UI/test_media_ingestion_source_panel.py Tests/UI/test_media_ingest_window_rebuilt.py
git commit -m "feat: add server ingestion source creation to media ingest"
```

## Task 6: Update Parity Docs And Run Vertical Verification

**Files:**
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`

- [ ] **Step 1: Update the parity docs to match verified behavior**

```markdown
- Media / Read-it-later:
  - landed: local saved-state persistence
  - landed: server save/remove compatibility mapping (`saved` / `archived`)
  - landed: aggregate `All Media` server saved view
  - landed: server ingestion-source create for `archive_snapshot` and `git_repository`
  - deferred: per-media-type server saved views
  - deferred: sync/mirror semantics
```

- [ ] **Step 2: Run the focused vertical verification suite**

Run: `python3 -m pytest Tests/Media_DB/test_media_db_v2.py Tests/Media/test_local_media_reading_service.py Tests/Media/test_media_reading_normalizers.py Tests/Media/test_server_media_reading_service.py Tests/Media/test_media_reading_scope_service.py Tests/UI/test_media_runtime_state.py Tests/UI/test_media_window_v2_parity.py Tests/UI/test_media_ingestion_source_panel.py Tests/UI/test_media_ingest_window_rebuilt.py -q`
Expected: PASS

- [ ] **Step 3: Run one higher-level smoke slice for existing media regression coverage**

Run: `python3 -m pytest Tests/UI/test_media_ingestion_tab_integration.py Tests/UI/test_media_window_v88_textual.py -q`
Expected: PASS, or if one existing legacy test fails for unrelated reasons, document that explicitly before merge.

- [ ] **Step 4: Commit**

```bash
git add Docs/Parity/2026-04-21-capability-matrix.md Docs/Parity/2026-04-21-gap-ledger.md Docs/Parity/2026-04-21-execution-roadmap.md
git commit -m "docs: update media read-it-later parity status"
```
