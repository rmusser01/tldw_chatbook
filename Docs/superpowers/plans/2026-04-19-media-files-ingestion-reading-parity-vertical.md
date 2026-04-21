# Media, Files, Ingestion, And Reading Parity Vertical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `tldw_chatbook` media browsing, ingestion-source management, reading-item compatibility, and reading-progress behavior with the current `tldw_server` contracts while preserving local-first storage and the existing media product layout.

**Architecture:** Add a dedicated media/reading contract layer in `tldw_api`, then create a local/server media service seam in a new `tldw_chatbook/Media` package. Extend the local media DB with a non-sync reading-progress table, introduce explicit media runtime state for backend ownership, and refactor the current media browse/ingest windows at the boundary so they consume normalized string IDs and scope-service operations instead of reaching directly into `media_db` or `TLDWAPIClient`.

**Tech Stack:** Python, Textual, Pydantic, `httpx`, SQLite, pytest

---

## File Map

- Create: `tldw_chatbook/tldw_api/media_reading_schemas.py`
  Responsibility: Pydantic request/response models for file artifacts, reference images, ingestion sources, reading items, and reading progress.
- Modify: `tldw_chatbook/tldw_api/client.py`
  Responsibility: Add async client methods for `/files`, `/ingestion-sources`, `/reading/items`, and `/media/{media_id}/progress`.
- Modify: `tldw_chatbook/tldw_api/__init__.py`
  Responsibility: Export the new schemas and client ergonomics alongside existing API surfaces.
- Create: `Tests/tldw_api/test_media_reading_client.py`
  Responsibility: Verify endpoint wiring, query serialization, optimistic parameters, multipart archive upload, and progress-route linkage.
- Create: `Tests/tldw_api/test_media_reading_schemas.py`
  Responsibility: Verify schema validation and model round-tripping for the new contracts.

- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
  Responsibility: Add the narrow local reading-progress store and helpers with explicit non-sync semantics.
- Modify: `Tests/Media_DB/test_media_db_v2.py`
  Responsibility: Verify local reading-progress CRUD, persistence shape, and absence from sync-log/version flows.

- Create: `tldw_chatbook/Media/__init__.py`
  Responsibility: Export the new local/server media seam helpers.
- Create: `tldw_chatbook/Media/media_reading_normalizers.py`
  Responsibility: Normalize local media rows, server reading items, file artifacts, ingestion sources, and reading progress into one media-facing contract.
- Create: `tldw_chatbook/Media/local_media_reading_service.py`
  Responsibility: Wrap `MediaDatabase` browse/detail/update/delete/document-version/reading-progress operations behind a focused local adapter.
- Create: `tldw_chatbook/Media/server_media_reading_service.py`
  Responsibility: Thin service around the new `tldw_api` media-reading methods.
- Create: `tldw_chatbook/Media/media_reading_scope_service.py`
  Responsibility: Route local/server operations, normalize results, and enforce unsupported-operation messages.
- Create: `Tests/Media/test_media_reading_normalizers.py`
  Responsibility: Verify canonical IDs, `backing_media_id`, nested progress data, and timestamp normalization.
- Create: `Tests/Media/test_media_reading_scope_service.py`
  Responsibility: Verify routing, unsupported-operation failures, and use of `backing_media_id` for progress operations.
- Create: `Tests/Media/test_server_media_reading_service.py`
  Responsibility: Verify server-service request shaping and payload normalization helpers where needed.

- Create: `tldw_chatbook/UI/Screens/media_runtime_state.py`
  Responsibility: Own explicit media backend state, selected normalized record, filters, and cache invalidation rules.
- Modify: `tldw_chatbook/app.py`
  Responsibility: Wire the local/server media services and shared media runtime state onto the app instance.
- Modify: `tldw_chatbook/UI/Screens/media_screen.py`
  Responsibility: Attach the shared media runtime state and scope service to the browse surface.
- Modify: `tldw_chatbook/UI/Screens/media_ingest_screen.py`
  Responsibility: Attach the shared media runtime state and scope service to the ingest surface.
- Create: `Tests/UI/test_media_runtime_state.py`
  Responsibility: Verify serialization/defaults/invalidation behavior for media backend ownership.
- Modify: `Tests/UI/test_screen_navigation.py`
  Responsibility: Keep navigation tests aligned with the media screen wrappers after wiring the shared state.

- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
  Responsibility: Move browse/detail/delete/analysis/progress behavior onto the new scope service and normalized record IDs.
- Modify: `tldw_chatbook/Widgets/Media/media_list_panel.py`
  Responsibility: Stop assuming `int media_id` and stop deriving selection from DOM IDs.
- Modify: `tldw_chatbook/Widgets/Media/media_viewer_panel.py`
  Responsibility: Consume normalized record payloads, expose minimal reading-progress controls, and disable unsupported server-mode actions cleanly.
- Modify: `tldw_chatbook/Event_Handlers/media_events.py`
  Responsibility: Update media message payloads for normalized record IDs and add any reading-progress messages needed by the refactored window.
- Modify: `tldw_chatbook/Widgets/Media/__init__.py`
  Responsibility: Re-export any new media widgets/messages added for the parity seam.
- Create: `Tests/UI/test_media_window_v2_parity.py`
  Responsibility: Verify backend invalidation, normalized selection flow, and service-backed browse/detail/progress behavior.
- Create: `Tests/Widgets/test_media_list_panel.py`
  Responsibility: Verify string-ID-safe selection and pagination behavior.

- Create: `tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py`
  Responsibility: Provide the minimal server-mode ingestion-source management surface used in the second ingest-panel slot.
- Modify: `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
  Responsibility: Keep local ingestion intact while replacing the legacy direct remote path with the new source-management panel.
- Create: `Tests/UI/test_media_ingestion_source_panel.py`
  Responsibility: Verify source list/detail/item list/sync/archive flows and local-mode disabled behavior.
- Create: `Tests/UI/test_media_ingest_window_rebuilt.py`
  Responsibility: Verify backend-aware tab behavior and absence of direct `TLDWAPIClient` usage in server mode.

- Modify: `Docs/Parity/2026-04-19-data-compatibility-map.md`
  Responsibility: Record the media/files/ingestion/reading contract and deferred work.
- Modify: `Docs/Parity/2026-04-19-rollout-backlog.md`
  Responsibility: Update rollout status for the media parity vertical.

## Task 1: Add Shared Media/Reading Schemas And `tldw_api` Endpoint Wiring

**Files:**
- Create: `tldw_chatbook/tldw_api/media_reading_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_media_reading_client.py`
- Test: `Tests/tldw_api/test_media_reading_schemas.py`

- [ ] **Step 1: Write the failing schema/client tests**

```python
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    FileCreateOptions,
    FileCreateRequest,
    IngestionSourceCreateRequest,
    IngestionSourcePatchRequest,
    ReadingProgressUpdate,
    ReadingUpdateRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_list_reading_items_serializes_filters(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"items": [], "total": 0, "page": 1, "size": 20})
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_reading_items(status=["saved"], tags=["ai"], q="rag", page=2, size=50)

    args, kwargs = mocked.await_args
    assert args[:2] == ("GET", "/api/v1/reading/items")
    assert kwargs["params"]["status"] == ["saved"]
    assert kwargs["params"]["tags"] == ["ai"]
    assert kwargs["params"]["q"] == "rag"
    assert kwargs["params"]["page"] == 2
    assert kwargs["params"]["size"] == 50


@pytest.mark.asyncio
async def test_progress_routes_use_media_id_not_reading_item_id(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"media_id": 42, "current_page": 3, "total_pages": 10, "zoom_level": 100, "view_mode": "single", "percent_complete": 30.0, "last_read_at": "2026-04-19T12:00:00Z"})
    monkeypatch.setattr(client, "_request", mocked)

    await client.get_reading_progress(42)
    await client.update_reading_progress(42, ReadingProgressUpdate(current_page=4, total_pages=10))

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/media/42/progress")
    assert mocked.await_args_list[1].args[:2] == ("PUT", "/api/v1/media/42/progress")


@pytest.mark.asyncio
async def test_upload_ingestion_source_archive_uses_archive_field(monkeypatch, tmp_path):
    client = TLDWAPIClient("http://localhost:8000")
    archive = tmp_path / "snapshot.zip"
    archive.write_bytes(b"zip")
    mocked = AsyncMock(return_value={"status": "queued", "source_id": 7})
    monkeypatch.setattr(client, "_request", mocked)

    await client.upload_ingestion_source_archive(7, str(archive))

    args, kwargs = mocked.await_args
    assert args[:2] == ("POST", "/api/v1/ingestion-sources/7/archive")
    assert kwargs["files"][0][0] == "archive"


def test_reading_update_request_validates_known_fields():
    payload = ReadingUpdateRequest(status="read", favorite=True, tags=["ai"])
    assert payload.status == "read"
    assert payload.favorite is True
    assert payload.tags == ["ai"]


def test_file_create_request_requires_persist_true():
    request = FileCreateRequest(
        file_type="markdown_table",
        payload={"headers": ["A"], "rows": [["1"]]},
        options=FileCreateOptions(persist=True),
    )
    assert request.options.persist is True


def test_ingestion_source_patch_rejects_extra_fields():
    with pytest.raises(Exception):
        IngestionSourcePatchRequest(unsupported=True)
```

- [ ] **Step 2: Run the schema/client tests to verify they fail**

Run: `python3 -m pytest Tests/tldw_api/test_media_reading_client.py Tests/tldw_api/test_media_reading_schemas.py -q`
Expected: FAIL with missing schema exports and missing `TLDWAPIClient` media-reading methods.

- [ ] **Step 3: Write the minimal schema and client layer**

```python
async def list_ingestion_sources(self) -> Dict[str, Any]:
    return await self._request("GET", "/api/v1/ingestion-sources/")


async def list_reading_items(
    self,
    *,
    status: list[str] | None = None,
    tags: list[str] | None = None,
    q: str | None = None,
    domain: str | None = None,
    page: int = 1,
    size: int = 20,
    sort: str | None = None,
) -> Dict[str, Any]:
    params = {
        "status": status,
        "tags": tags,
        "q": q,
        "domain": domain,
        "page": page,
        "size": size,
        "sort": sort,
    }
    return await self._request("GET", "/api/v1/reading/items", params={k: v for k, v in params.items() if v is not None})


async def get_reading_progress(self, media_id: int) -> Dict[str, Any]:
    return await self._request("GET", f"/api/v1/media/{media_id}/progress")


async def update_reading_progress(self, media_id: int, request_data: ReadingProgressUpdate) -> Dict[str, Any]:
    return await self._request(
        "PUT",
        f"/api/v1/media/{media_id}/progress",
        json_data=request_data.model_dump(exclude_none=True, mode="json"),
    )
```

Required client methods in this task:

- `create_file_artifact`
- `list_reference_images`
- `get_file_artifact`
- `delete_file_artifact`
- `create_ingestion_source`
- `list_ingestion_sources`
- `get_ingestion_source`
- `patch_ingestion_source`
- `list_ingestion_source_items`
- `trigger_ingestion_source_sync`
- `upload_ingestion_source_archive`
- `list_reading_items`
- `get_reading_item`
- `update_reading_item`
- `delete_reading_item`
- `get_reading_progress`
- `update_reading_progress`
- `delete_reading_progress`

Required schema types in this task:

- File artifact request/response models needed by the client seam
- Ingestion source create/patch/list/detail/item/sync models
- Reading item list/detail/update/delete models
- Reading progress get/update/delete models

- [ ] **Step 4: Run the schema/client tests again**

Run: `python3 -m pytest Tests/tldw_api/test_media_reading_client.py Tests/tldw_api/test_media_reading_schemas.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/media_reading_schemas.py tldw_chatbook/tldw_api/client.py tldw_chatbook/tldw_api/__init__.py Tests/tldw_api/test_media_reading_client.py Tests/tldw_api/test_media_reading_schemas.py
git commit -m "feat: add media reading api client support"
```

## Task 2: Extend The Local Media DB With A Non-Sync Reading-Progress Store

**Files:**
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
- Test: `Tests/Media_DB/test_media_db_v2.py`

- [ ] **Step 1: Write the failing DB tests**

```python
def test_upsert_and_get_reading_progress_round_trip(db_instance):
    media_id, media_uuid = db_instance.add_media_with_keywords(
        title="Reader",
        content="Hello",
        media_type="document",
        keywords=["reader"],
    )

    db_instance.upsert_reading_progress(
        media_id,
        current_page=3,
        total_pages=12,
        zoom_level=125,
        view_mode="continuous",
        cfi="epubcfi(/6/2[chapter]!/4/1:0)",
        percentage=25.0,
    )

    progress = db_instance.get_reading_progress(media_id)
    assert progress["current_page"] == 3
    assert progress["total_pages"] == 12
    assert progress["zoom_level"] == 125
    assert progress["view_mode"] == "continuous"
    assert progress["percentage"] == 25.0


def test_reading_progress_delete_removes_row(db_instance):
    media_id, _ = db_instance.add_media_with_keywords(
        title="Reader",
        content="Hello",
        media_type="document",
        keywords=[],
    )

    db_instance.upsert_reading_progress(media_id, current_page=1, total_pages=10)
    assert db_instance.delete_reading_progress(media_id) is True
    assert db_instance.get_reading_progress(media_id) is None


def test_reading_progress_does_not_write_sync_log_or_bump_media_version(db_instance):
    media_id, media_uuid = db_instance.add_media_with_keywords(
        title="Reader",
        content="Hello",
        media_type="document",
        keywords=[],
    )
    initial_log_count = get_log_count(db_instance, media_uuid)
    initial_version = get_entity_version(db_instance, "Media", media_uuid)

    db_instance.upsert_reading_progress(media_id, current_page=2, total_pages=10)

    assert get_log_count(db_instance, media_uuid) == initial_log_count
    assert get_entity_version(db_instance, "Media", media_uuid) == initial_version
```

- [ ] **Step 2: Run the DB test slice to verify it fails**

Run: `python3 -m pytest Tests/Media_DB/test_media_db_v2.py -q`
Expected: FAIL with missing reading-progress helpers or missing table.

- [ ] **Step 3: Implement the minimal DB support**

```python
def _ensure_reading_progress_table(self) -> None:
    self.execute_query(
        """
        CREATE TABLE IF NOT EXISTS MediaReadingProgress (
            media_id INTEGER PRIMARY KEY,
            current_page INTEGER NOT NULL DEFAULT 1,
            total_pages INTEGER NOT NULL DEFAULT 1,
            zoom_level INTEGER NOT NULL DEFAULT 100,
            view_mode TEXT NOT NULL DEFAULT 'single',
            cfi TEXT,
            percentage REAL,
            last_read_at TEXT NOT NULL
        )
        """,
        commit=True,
    )


def get_reading_progress(self, media_id: int) -> dict[str, Any] | None:
    cursor = self.execute_query("SELECT * FROM MediaReadingProgress WHERE media_id = ?", (media_id,))
    row = cursor.fetchone()
    return dict(row) if row else None
```

Implementation constraints in this task:

- Reading-progress rows are keyed by local `media_id`.
- They do not create `sync_log` rows.
- They do not bump `Media.version`.
- They do not create document-version history.

- [ ] **Step 4: Run the DB tests again**

Run: `python3 -m pytest Tests/Media_DB/test_media_db_v2.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/Client_Media_DB_v2.py Tests/Media_DB/test_media_db_v2.py
git commit -m "feat: add local media reading progress store"
```

## Task 3: Create The Local/Server Media Service Seam And Normalizers

**Files:**
- Create: `tldw_chatbook/Media/__init__.py`
- Create: `tldw_chatbook/Media/media_reading_normalizers.py`
- Create: `tldw_chatbook/Media/local_media_reading_service.py`
- Create: `tldw_chatbook/Media/server_media_reading_service.py`
- Create: `tldw_chatbook/Media/media_reading_scope_service.py`
- Test: `Tests/Media/test_media_reading_normalizers.py`
- Test: `Tests/Media/test_media_reading_scope_service.py`
- Test: `Tests/Media/test_server_media_reading_service.py`

- [ ] **Step 1: Write the failing service/normalizer tests**

```python
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Media.media_reading_normalizers import (
    build_media_entity_id,
    normalize_local_media_record,
    normalize_server_reading_item,
)
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService


def test_normalize_server_reading_item_carries_backing_media_id():
    normalized = normalize_server_reading_item(
        {"id": 118, "media_id": 42, "title": "Remote Article", "status": "saved"}
    )

    assert normalized["id"] == "server:reading_item:118"
    assert normalized["source_id"] == "118"
    assert normalized["backing_media_id"] == 42


@pytest.mark.asyncio
async def test_scope_service_uses_backing_media_id_for_server_progress():
    server = type(
        "FakeServer",
        (),
        {
            "get_reading_progress": AsyncMock(return_value={"media_id": 42, "current_page": 3, "total_pages": 10, "zoom_level": 100, "view_mode": "single", "percent_complete": 30.0, "last_read_at": "2026-04-19T12:00:00Z"}),
        },
    )()
    scope = MediaReadingScopeService(local_service=None, server_service=server)
    record = {"id": "server:reading_item:118", "backend": "server", "entity_kind": "reading_item", "source_id": "118", "backing_media_id": 42}

    progress = await scope.get_reading_progress(record, mode="server")

    assert progress["backing_media_id"] == 42
    server.get_reading_progress.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_scope_service_rejects_local_ingestion_source_operations():
    scope = MediaReadingScopeService(local_service=object(), server_service=None)

    with pytest.raises(ValueError, match="Local ingestion sources are not available yet."):
        await scope.list_ingestion_sources(mode="local")
```

- [ ] **Step 2: Run the service/normalizer test slice to verify it fails**

Run: `python3 -m pytest Tests/Media/test_media_reading_normalizers.py Tests/Media/test_media_reading_scope_service.py Tests/Media/test_server_media_reading_service.py -q`
Expected: FAIL with missing module imports and missing seam methods.

- [ ] **Step 3: Implement the minimal media seam**

```python
def build_media_entity_id(backend: str, entity_kind: str, source_id: str | int) -> str:
    return f"{backend}:{entity_kind}:{source_id}"


def normalize_server_reading_item(item: Mapping[str, Any], progress: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": build_media_entity_id("server", "reading_item", item["id"]),
        "backend": "server",
        "entity_kind": "reading_item",
        "source_id": str(item["id"]),
        "backing_media_id": item.get("media_id"),
        "title": item.get("title") or "",
        "media_type": "reading",
        "status": item.get("status"),
        "url": item.get("url"),
        "reading_progress": normalize_progress("server", item.get("media_id"), progress),
    }


class MediaReadingScopeService:
    async def get_reading_progress(self, record: Mapping[str, Any], mode: str = "local") -> dict[str, Any]:
        media_id = record.get("backing_media_id")
        if media_id in {None, ""}:
            raise ValueError("Reading progress is not available for this record.")
        backend = self._backend(mode)
        payload = await self._maybe_await(backend.get_reading_progress(media_id))
        return normalize_progress(mode, media_id, payload)
```

Required scope-service methods in this task:

- Browse/detail: `search_media`, `get_media_detail`
- Local/server edits: `update_media_metadata`, `delete_media`, `undelete_media`
- Reading progress: `get_reading_progress`, `update_reading_progress`, `delete_reading_progress`
- Ingestion sources: `list_ingestion_sources`, `get_ingestion_source`, `patch_ingestion_source`, `list_ingestion_source_items`, `trigger_ingestion_source_sync`, `upload_ingestion_source_archive`
- Local-only document version helpers needed by the current media viewer: `list_document_versions`, `save_analysis_version`, `overwrite_analysis_version`, `delete_analysis_version`

- [ ] **Step 4: Run the service/normalizer tests again**

Run: `python3 -m pytest Tests/Media/test_media_reading_normalizers.py Tests/Media/test_media_reading_scope_service.py Tests/Media/test_server_media_reading_service.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Media/__init__.py tldw_chatbook/Media/media_reading_normalizers.py tldw_chatbook/Media/local_media_reading_service.py tldw_chatbook/Media/server_media_reading_service.py tldw_chatbook/Media/media_reading_scope_service.py Tests/Media/test_media_reading_normalizers.py Tests/Media/test_media_reading_scope_service.py Tests/Media/test_server_media_reading_service.py
git commit -m "feat: add media reading service seam"
```

## Task 4: Add Explicit Media Runtime State And App Wiring

**Files:**
- Create: `tldw_chatbook/UI/Screens/media_runtime_state.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/media_screen.py`
- Modify: `tldw_chatbook/UI/Screens/media_ingest_screen.py`
- Test: `Tests/UI/test_media_runtime_state.py`
- Test: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write the failing runtime-state tests**

```python
from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState


def test_media_runtime_state_defaults_to_local_backend():
    state = MediaRuntimeState()
    assert state.runtime_backend == "local"
    assert state.selected_record_id is None
    assert state.search_term == ""


def test_media_runtime_state_backend_reset_clears_selection_and_caches():
    state = MediaRuntimeState(
        runtime_backend="local",
        selected_record_id="local:media:7",
        browse_items=[{"id": "local:media:7"}],
        reading_progress_by_record_id={"local:media:7": {"current_page": 3}},
        ingestion_source_items_by_id={"server:ingestion_source:2": [{"id": 1}]},
    )

    state.reset_for_backend("server")

    assert state.runtime_backend == "server"
    assert state.selected_record_id is None
    assert state.browse_items == []
    assert state.reading_progress_by_record_id == {}
    assert state.ingestion_source_items_by_id == {}
```

- [ ] **Step 2: Run the runtime-state test slice to verify it fails**

Run: `python3 -m pytest Tests/UI/test_media_runtime_state.py Tests/UI/test_screen_navigation.py -q`
Expected: FAIL with missing runtime-state module or missing app wiring.

- [ ] **Step 3: Implement the runtime-state owner and wiring**

```python
@dataclass
class MediaRuntimeState:
    runtime_backend: str = "local"
    active_media_type: str | None = None
    search_term: str = ""
    keyword_filter: str = ""
    selected_record_id: str | None = None
    browse_items: list[dict[str, Any]] = field(default_factory=list)
    detail_by_record_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    reading_progress_by_record_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    ingestion_source_items_by_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def reset_for_backend(self, runtime_backend: str) -> None:
        self.runtime_backend = runtime_backend
        self.selected_record_id = None
        self.browse_items.clear()
        self.detail_by_record_id.clear()
        self.reading_progress_by_record_id.clear()
        self.ingestion_source_items_by_id.clear()
```

Required wiring outcomes in this task:

- `app.py` creates and stores `self.media_runtime_state`.
- `app.py` wires `self.local_media_reading_service`, `self.server_media_reading_service`, and `self.media_reading_scope_service`.
- `media_screen.py` and `media_ingest_screen.py` both consume the same `MediaRuntimeState` instance.
- If no existing runtime-backend value is available, media defaults to `local`.

- [ ] **Step 4: Run the runtime-state tests again**

Run: `python3 -m pytest Tests/UI/test_media_runtime_state.py Tests/UI/test_screen_navigation.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/media_runtime_state.py tldw_chatbook/app.py tldw_chatbook/UI/Screens/media_screen.py tldw_chatbook/UI/Screens/media_ingest_screen.py Tests/UI/test_media_runtime_state.py Tests/UI/test_screen_navigation.py
git commit -m "feat: wire media runtime state"
```

## Task 5: Refactor The Media Browse Stack To Use Normalized IDs And Progress Controls

**Files:**
- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
- Modify: `tldw_chatbook/Widgets/Media/media_list_panel.py`
- Modify: `tldw_chatbook/Widgets/Media/media_viewer_panel.py`
- Modify: `tldw_chatbook/Event_Handlers/media_events.py`
- Modify: `tldw_chatbook/Widgets/Media/__init__.py`
- Test: `Tests/UI/test_media_window_v2_parity.py`
- Test: `Tests/Widgets/test_media_list_panel.py`

- [ ] **Step 1: Write the failing browse-stack tests**

```python
from unittest.mock import AsyncMock

from tldw_chatbook.Widgets.Media.media_list_panel import MediaItemSelectedEvent


def test_media_list_panel_selection_returns_normalized_record_id(media_list_panel):
    media_list_panel.load_items(
        [{"id": "server:reading_item:118", "title": "Remote Article", "backing_media_id": 42}],
        page=1,
        total_pages=1,
    )
    event = media_list_panel._build_selection_event_for_test(0)
    assert event.record_id == "server:reading_item:118"
    assert event.media_data["backing_media_id"] == 42


@pytest.mark.asyncio
async def test_media_window_backend_change_clears_selected_record_and_viewer(media_window):
    media_window.runtime_state.selected_record_id = "local:media:7"
    media_window.runtime_state.browse_items = [{"id": "local:media:7"}]

    await media_window.handle_runtime_backend_changed("server")

    assert media_window.runtime_state.selected_record_id is None
    assert media_window.viewer_panel.media_data is None


@pytest.mark.asyncio
async def test_media_window_uses_scope_service_for_reading_progress(media_window, mock_scope_service):
    record = {"id": "server:reading_item:118", "backing_media_id": 42, "backend": "server"}
    mock_scope_service.get_reading_progress = AsyncMock(return_value={"backing_media_id": 42, "current_page": 3, "total_pages": 10})

    await media_window.load_reading_progress(record)

    mock_scope_service.get_reading_progress.assert_awaited_once_with(record, mode="server")
```

- [ ] **Step 2: Run the browse-stack test slice to verify it fails**

Run: `python3 -m pytest Tests/UI/test_media_window_v2_parity.py Tests/Widgets/test_media_list_panel.py -q`
Expected: FAIL with integer-ID assumptions, missing normalized selection flow, and missing progress helpers.

- [ ] **Step 3: Refactor the browse stack at the boundary**

```python
class MediaItemSelectedEvent(Message):
    def __init__(self, record_id: str, media_data: dict[str, Any]) -> None:
        super().__init__()
        self.record_id = record_id
        self.media_data = media_data


class MediaListPanel(Container):
    selected_id: reactive[str | None] = reactive(None)

    def _row_widget_id(self, index: int) -> str:
        return f"media-row-{index}"

    def _record_for_row(self, row_index: int) -> dict[str, Any]:
        return self.items[row_index]


@on(ListView.Selected, "#media-list")
def handle_item_selection(self, event: ListView.Selected) -> None:
    row_index = self._row_index_for_widget(event.item)
    record = self._record_for_row(row_index)
    self.selected_id = record["id"]
    self.post_message(MediaItemSelectedEvent(record["id"], record))
```

Browse-stack constraints in this task:

- `MediaListPanel` must stop parsing `int` IDs from widget IDs.
- `MediaWindow_v2.selected_media_id` becomes a normalized string-first selected-record field.
- `MediaWindow_v2` must fetch detail, metadata updates, delete/undelete, document versions, and reading progress through the new scope/local service seam.
- `media_viewer_panel.py` must add a minimal reading-progress section only when the record exposes `backing_media_id`.
- Local-only analysis save/overwrite/delete behavior remains available in local mode and fails explicitly in server mode when unsupported.

- [ ] **Step 4: Run the browse-stack tests again**

Run: `python3 -m pytest Tests/UI/test_media_window_v2_parity.py Tests/Widgets/test_media_list_panel.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MediaWindow_v2.py tldw_chatbook/Widgets/Media/media_list_panel.py tldw_chatbook/Widgets/Media/media_viewer_panel.py tldw_chatbook/Event_Handlers/media_events.py tldw_chatbook/Widgets/Media/__init__.py Tests/UI/test_media_window_v2_parity.py Tests/Widgets/test_media_list_panel.py
git commit -m "feat: refactor media browse stack for parity seam"
```

## Task 6: Replace The Legacy Remote Ingest Path With Server-Mode Source Management

**Files:**
- Create: `tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py`
- Modify: `tldw_chatbook/Widgets/Media/__init__.py`
- Modify: `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
- Test: `Tests/UI/test_media_ingestion_source_panel.py`
- Test: `Tests/UI/test_media_ingest_window_rebuilt.py`

- [ ] **Step 1: Write the failing ingest-panel tests**

```python
from unittest.mock import AsyncMock, Mock

@pytest.mark.asyncio
async def test_ingestion_source_panel_is_disabled_in_local_mode(panel):
    panel.runtime_backend = "local"
    await panel.refresh_for_mode()
    assert panel.query_one("#source-panel-disabled").display is True


@pytest.mark.asyncio
async def test_ingestion_source_panel_lists_sources_in_server_mode(panel, mock_scope_service):
    mock_scope_service.list_ingestion_sources = AsyncMock(
        return_value=[{"id": "server:ingestion_source:7", "title": "Repo", "source_type": "git_repository"}]
    )
    panel.runtime_backend = "server"
    await panel.refresh_for_mode()
    mock_scope_service.list_ingestion_sources.assert_awaited_once_with(mode="server")


@pytest.mark.asyncio
async def test_ingest_window_does_not_construct_api_client_for_server_mode(monkeypatch, ingest_window):
    ctor = Mock()
    monkeypatch.setattr("tldw_chatbook.UI.MediaIngestWindowRebuilt.TLDWAPIClient", ctor)
    ingest_window.runtime_state.runtime_backend = "server"
    await ingest_window.refresh_backend_view()
    ctor.assert_not_called()
```

- [ ] **Step 2: Run the ingest test slice to verify it fails**

Run: `python3 -m pytest Tests/UI/test_media_ingestion_source_panel.py Tests/UI/test_media_ingest_window_rebuilt.py -q`
Expected: FAIL with missing source-management panel and direct API-client usage still present.

- [ ] **Step 3: Implement the backend-aware ingest replacement**

```python
class MediaIngestionSourcePanel(ScrollableContainer):
    runtime_backend: reactive[str] = reactive("local")

    async def refresh_for_mode(self) -> None:
        if self.runtime_backend != "server":
            self.show_disabled_copy("Server ingestion sources require server mode.")
            return
        sources = await self.scope_service.list_ingestion_sources(mode="server")
        self.load_sources(sources)
```

Required outcomes in this task:

- The second ingest-panel slot becomes `Server Sources` or equivalent source-management copy.
- Local mode keeps the existing local ingestion panel unchanged.
- Server mode provides:
  - source list
  - source detail
  - source item list
  - trigger sync action
  - patch mutable settings action
  - archive upload action for archive-backed sources
- `MediaIngestWindowRebuilt.py` no longer constructs `TLDWAPIClient` directly for server-mode behavior.

- [ ] **Step 4: Run the ingest tests again**

Run: `python3 -m pytest Tests/UI/test_media_ingestion_source_panel.py Tests/UI/test_media_ingest_window_rebuilt.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py tldw_chatbook/Widgets/Media/__init__.py tldw_chatbook/UI/MediaIngestWindowRebuilt.py Tests/UI/test_media_ingestion_source_panel.py Tests/UI/test_media_ingest_window_rebuilt.py
git commit -m "feat: add server mode ingestion source management"
```

## Task 7: Update Parity Docs And Run The Focused Regression Sweep

**Files:**
- Modify: `Docs/Parity/2026-04-19-data-compatibility-map.md`
- Modify: `Docs/Parity/2026-04-19-rollout-backlog.md`

- [ ] **Step 1: Update the parity docs for the landed seam**

```markdown
- Media parity now uses canonical IDs of the form `<backend>:<entity_kind>:<source_id>`.
- Reading-progress operations depend on raw `backing_media_id`, not reading-item IDs.
- Local reading progress is local-only and excluded from sync/version semantics in this slice.
- Server-mode ingestion uses ingestion-source management instead of the legacy direct remote-processing tab.
```

- [ ] **Step 2: Run the focused regression suite**

Run: `python3 -m pytest Tests/tldw_api/test_media_reading_client.py Tests/tldw_api/test_media_reading_schemas.py Tests/Media/test_media_reading_normalizers.py Tests/Media/test_media_reading_scope_service.py Tests/Media/test_server_media_reading_service.py Tests/Media_DB/test_media_db_v2.py Tests/UI/test_media_runtime_state.py Tests/UI/test_media_window_v2_parity.py Tests/Widgets/test_media_list_panel.py Tests/UI/test_media_ingestion_source_panel.py Tests/UI/test_media_ingest_window_rebuilt.py -q`
Expected: PASS

- [ ] **Step 3: Run the nearby existing media/UI smoke coverage**

Run: `python3 -m pytest Tests/UI/test_screen_navigation.py Tests/UI/test_media_v88_simple.py Tests/UI/test_ingest_window.py -q`
Expected: PASS, or clearly documented legacy failures unrelated to this vertical.

- [ ] **Step 4: Inspect the final diff before closing the branch**

Run: `git status --short`
Expected: Only the intended media/files/ingestion/reading vertical files remain modified before the final commit.

- [ ] **Step 5: Commit**

```bash
git add Docs/Parity/2026-04-19-data-compatibility-map.md Docs/Parity/2026-04-19-rollout-backlog.md
git commit -m "docs: record media parity vertical"
```

## Notes For Execution

- Keep the UI layout recognizable. Refactor at the boundary instead of redesigning the media product.
- Do not let canonical IDs leak into widget IDs. Use row indexes or side mappings for widget identity.
- Use `backing_media_id` for all reading-progress calls. Do not accidentally call progress endpoints with reading-item IDs.
- Keep local reading progress out of `sync_log`, entity versioning, and document-version history.
- Disable or explain unsupported server-mode actions explicitly instead of silently failing.
