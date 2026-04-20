# RAG, Chunking, And Template Alignment Vertical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `tldw_chatbook` chunking-template management and embedding-collection management with the current `tldw_server` contracts while preserving local-first storage and keeping the current search execution UI unchanged.

**Architecture:** Extend `tldw_api` with chunking-template and embedding-collection contracts, then add a mode-aware retrieval-admin seam that normalizes local and server records onto one admin contract. Refactor the existing chunking template widget and embeddings management window to consume that seam without turning this branch into a full RAG search rewrite.

**Tech Stack:** Python, Textual, Pydantic, `httpx`, SQLite, ChromaDB, pytest

---

## File Map

- Create: `tldw_chatbook/tldw_api/rag_admin_schemas.py`
  Responsibility: Pydantic models for chunking-template and embedding-collection payloads.
- Modify: `tldw_chatbook/tldw_api/client.py`
  Responsibility: Add async methods for chunking-template and embedding-collection endpoints.
- Modify: `tldw_chatbook/tldw_api/__init__.py`
  Responsibility: Export the new schema types and client methods.
- Test: `Tests/tldw_api/test_rag_admin_client.py`
  Responsibility: Verify endpoint wiring, params, payload shaping, and path usage.
- Test: `Tests/tldw_api/test_rag_admin_schemas.py`
  Responsibility: Verify schema validation and normalization expectations.

- Create: `tldw_chatbook/RAG_Admin/__init__.py`
  Responsibility: Export the new retrieval-admin seam helpers.
- Create: `tldw_chatbook/RAG_Admin/rag_admin_normalizers.py`
  Responsibility: Normalize local/server template and collection payloads onto one admin contract.
- Create: `tldw_chatbook/RAG_Admin/local_rag_admin_service.py`
  Responsibility: Wrap local chunking-template and local collection operations behind one adapter.
- Create: `tldw_chatbook/RAG_Admin/server_rag_admin_service.py`
  Responsibility: Thin service around the new `tldw_api` methods.
- Create: `tldw_chatbook/RAG_Admin/rag_admin_scope_service.py`
  Responsibility: Route local/server operations, normalize outputs, and expose explicit unsupported-operation failures.
- Test: `Tests/RAG_Admin/test_rag_admin_normalizers.py`
  Responsibility: Verify normalized record IDs, field mapping, and backend-specific defaults.
- Test: `Tests/RAG_Admin/test_rag_admin_scope_service.py`
  Responsibility: Verify routing, backend mode behavior, and supported/unsupported operations.
- Test: `Tests/RAG_Admin/test_server_rag_admin_service.py`
  Responsibility: Verify server-service request shaping and collection stats handling.

- Modify: `tldw_chatbook/Widgets/chunking_templates_widget.py`
  Responsibility: Resolve backend-aware template records through the new scope service.
- Modify: `tldw_chatbook/Widgets/chunking_template_editor.py`
  Responsibility: Save template changes through the active backend seam.
- Test: `Tests/UI/test_chunking_templates_widget_parity.py`
  Responsibility: Verify local/server template list, selection, and CRUD behavior.

- Modify: `tldw_chatbook/UI/Embeddings_Management_Window.py`
  Responsibility: Replace placeholder collection behavior with real local/server collection loading, stats, and deletion via the scope service.
- Modify: `tldw_chatbook/Widgets/embeddings_list_items.py`
  Responsibility: Ensure collection rows can render normalized collection metadata and status safely.
- Test: `Tests/UI/test_embeddings_management_window_parity.py`
  Responsibility: Verify collection list/detail/delete flows in local and server modes.

- Modify: `tldw_chatbook/app.py`
  Responsibility: Wire the new retrieval-admin services onto the app instance for reuse by the widgets.

- Modify: `Docs/Parity/2026-04-19-data-compatibility-map.md`
  Responsibility: Record the landed retrieval-admin compatibility seam and remaining deferred work.
- Modify: `Docs/Parity/2026-04-19-rollout-backlog.md`
  Responsibility: Mark the vertical complete and name the next recommended vertical.

## Task 1: Add Shared RAG-Admin Schemas And `tldw_api` Endpoint Wiring

**Files:**
- Create: `tldw_chatbook/tldw_api/rag_admin_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_rag_admin_client.py`
- Test: `Tests/tldw_api/test_rag_admin_schemas.py`

- [ ] **Step 1: Write the failing schema/client tests**

```python
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ChunkingTemplateCreateRequest,
    ChunkingTemplateUpdateRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_list_chunking_templates_serializes_filters(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"templates": [], "total": 0})
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_chunking_templates(include_builtin=True, include_custom=False, tags=["notes"], user_id="u1")

    args, kwargs = mocked.await_args
    assert args[:2] == ("GET", "/api/v1/chunking/templates")
    assert kwargs["params"]["include_builtin"] is True
    assert kwargs["params"]["include_custom"] is False
    assert kwargs["params"]["tags"] == ["notes"]
    assert kwargs["params"]["user_id"] == "u1"


@pytest.mark.asyncio
async def test_delete_embedding_collection_uses_collection_name(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value=None)
    monkeypatch.setattr(client, "_request", mocked)

    await client.delete_embedding_collection("demo_collection")

    args, _ = mocked.await_args
    assert args[:2] == ("DELETE", "/api/v1/embeddings/collections/demo_collection")


def test_chunking_template_create_request_round_trips():
    payload = ChunkingTemplateCreateRequest(
        name="demo",
        description="Demo template",
        tags=["rag"],
        template={
            "preprocessing": [],
            "chunking": {"method": "words", "config": {"max_size": 400}},
            "postprocessing": [],
        },
    )

    dumped = payload.model_dump()
    assert dumped["name"] == "demo"
    assert dumped["template"]["chunking"]["method"] == "words"


def test_chunking_template_update_request_is_partial():
    payload = ChunkingTemplateUpdateRequest(description="Updated")
    assert payload.model_dump(exclude_none=True) == {"description": "Updated"}
```

- [ ] **Step 2: Run the schema/client tests to verify they fail**

Run: `python3 -m pytest Tests/tldw_api/test_rag_admin_client.py Tests/tldw_api/test_rag_admin_schemas.py -q`  
Expected: FAIL with missing schema exports and missing `TLDWAPIClient` methods.

- [ ] **Step 3: Write the minimal schema and client layer**

```python
async def list_chunking_templates(
    self,
    *,
    include_builtin: bool = True,
    include_custom: bool = True,
    tags: list[str] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    params = {
        "include_builtin": include_builtin,
        "include_custom": include_custom,
        "tags": tags,
        "user_id": user_id,
    }
    return await self._request(
        "GET",
        "/api/v1/chunking/templates",
        params={k: v for k, v in params.items() if v is not None},
    )


async def delete_embedding_collection(self, collection_name: str) -> None:
    await self._request(
        "DELETE",
        f"/api/v1/embeddings/collections/{collection_name}",
    )
```

Required client methods in this task:

- `list_chunking_templates`
- `get_chunking_template`
- `create_chunking_template`
- `update_chunking_template`
- `delete_chunking_template`
- `apply_chunking_template`
- `get_chunking_template_diagnostics`
- `list_embedding_collections`
- `delete_embedding_collection`
- `get_embedding_collection_stats`

Required schema types in this task:

- chunking-template create/update/response/list/diagnostics models
- embedding-collection response/stats models

- [ ] **Step 4: Run the schema/client tests again**

Run: `python3 -m pytest Tests/tldw_api/test_rag_admin_client.py Tests/tldw_api/test_rag_admin_schemas.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/rag_admin_schemas.py tldw_chatbook/tldw_api/client.py tldw_chatbook/tldw_api/__init__.py Tests/tldw_api/test_rag_admin_client.py Tests/tldw_api/test_rag_admin_schemas.py
git commit -m "feat: add rag admin api client contracts"
```

## Task 2: Add A Mode-Aware Retrieval-Admin Service Seam

**Files:**
- Create: `tldw_chatbook/RAG_Admin/__init__.py`
- Create: `tldw_chatbook/RAG_Admin/rag_admin_normalizers.py`
- Create: `tldw_chatbook/RAG_Admin/local_rag_admin_service.py`
- Create: `tldw_chatbook/RAG_Admin/server_rag_admin_service.py`
- Create: `tldw_chatbook/RAG_Admin/rag_admin_scope_service.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/RAG_Admin/test_rag_admin_normalizers.py`
- Test: `Tests/RAG_Admin/test_rag_admin_scope_service.py`
- Test: `Tests/RAG_Admin/test_server_rag_admin_service.py`

- [ ] **Step 1: Write the failing service tests**

```python
def test_normalize_local_template_maps_is_system_to_is_builtin():
    record = normalize_template_record(
        backend="local",
        payload={
            "id": 7,
            "name": "general",
            "description": "General",
            "template_json": "{}",
            "is_system": 1,
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:00:00Z",
        },
    )

    assert record["record_id"] == "local:chunking_template:general"
    assert record["is_builtin"] is True
    assert record["tags"] == []


@pytest.mark.asyncio
async def test_scope_service_routes_template_list_by_backend():
    local = FakeLocalService(templates=[{"name": "local-demo"}])
    server = FakeServerService(templates=[{"name": "server-demo"}])
    scope = RAGAdminScopeService(local_service=local, server_service=server)

    local_records = await scope.list_templates(mode="local")
    server_records = await scope.list_templates(mode="server")

    assert local_records[0]["backend"] == "local"
    assert server_records[0]["backend"] == "server"


@pytest.mark.asyncio
async def test_scope_service_uses_stats_endpoint_for_server_collection_detail():
    server = FakeServerService(
        collections=[{"name": "demo", "metadata": {"provider": "openai"}}],
        collection_stats={"demo": {"name": "demo", "count": 3, "embedding_dimension": 1536, "metadata": {"provider": "openai"}}},
    )
    scope = RAGAdminScopeService(local_service=FakeLocalService(), server_service=server)

    detail = await scope.get_collection_detail(mode="server", collection_name="demo")

    assert detail["document_count"] == 3
    assert detail["embedding_dimension"] == 1536
```

- [ ] **Step 2: Run the service tests to verify they fail**

Run: `python3 -m pytest Tests/RAG_Admin/test_rag_admin_normalizers.py Tests/RAG_Admin/test_rag_admin_scope_service.py Tests/RAG_Admin/test_server_rag_admin_service.py -q`  
Expected: FAIL with missing package/modules/functions.

- [ ] **Step 3: Write the minimal seam**

```python
def normalize_template_record(*, backend: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    return {
        "backend": backend,
        "record_type": "chunking_template",
        "record_id": f"{backend}:chunking_template:{name}",
        "name": name,
        "description": payload.get("description") or "",
        "template_name": name,
        "tags": list(payload.get("tags") or []),
        "is_builtin": bool(payload.get("is_builtin", payload.get("is_system", False))),
        "version": payload.get("version"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "raw_payload": dict(payload),
    }
```

Required scope methods in this task:

- `list_templates(mode=...)`
- `get_template(mode=..., template_name=...)`
- `create_template(mode=..., ...)`
- `update_template(mode=..., ...)`
- `delete_template(mode=..., template_name=...)`
- `list_collections(mode=...)`
- `get_collection_detail(mode=..., collection_name=...)`
- `delete_collection(mode=..., collection_name=...)`

- [ ] **Step 4: Run the service tests again**

Run: `python3 -m pytest Tests/RAG_Admin/test_rag_admin_normalizers.py Tests/RAG_Admin/test_rag_admin_scope_service.py Tests/RAG_Admin/test_server_rag_admin_service.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Admin/__init__.py tldw_chatbook/RAG_Admin/rag_admin_normalizers.py tldw_chatbook/RAG_Admin/local_rag_admin_service.py tldw_chatbook/RAG_Admin/server_rag_admin_service.py tldw_chatbook/RAG_Admin/rag_admin_scope_service.py tldw_chatbook/app.py Tests/RAG_Admin/test_rag_admin_normalizers.py Tests/RAG_Admin/test_rag_admin_scope_service.py Tests/RAG_Admin/test_server_rag_admin_service.py
git commit -m "feat: add rag admin scope service"
```

## Task 3: Make `ChunkingTemplatesWidget` Backend-Aware

**Files:**
- Modify: `tldw_chatbook/Widgets/chunking_templates_widget.py`
- Modify: `tldw_chatbook/Widgets/chunking_template_editor.py`
- Test: `Tests/UI/test_chunking_templates_widget_parity.py`

- [ ] **Step 1: Write the failing widget tests**

```python
@pytest.mark.asyncio
async def test_chunking_templates_widget_reads_server_templates_when_runtime_backend_is_server(app):
    scope = FakeScopeService(server_templates=[normalized_template("server", "server-demo")])
    app.rag_admin_scope_service = scope
    app.current_runtime_backend = "server"

    widget = ChunkingTemplatesWidget(app_instance=app)
    await pilot_mount(widget)

    assert "server-demo" in widget.templates_cache_by_name


@pytest.mark.asyncio
async def test_chunking_templates_widget_disables_builtin_edit_actions(app):
    scope = FakeScopeService(local_templates=[normalized_template("local", "general", is_builtin=True)])
    app.rag_admin_scope_service = scope

    widget = ChunkingTemplatesWidget(app_instance=app)
    await pilot_mount(widget)
    widget.selected_template_record_id = "local:chunking_template:general"

    assert widget.query_one("#edit-template-btn", Button).disabled is True
```

- [ ] **Step 2: Run the widget tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_chunking_templates_widget_parity.py -q`  
Expected: FAIL because the widget still depends directly on `ChunkingInteropService`.

- [ ] **Step 3: Refactor the widget and editor minimally**

```python
def _runtime_backend(self) -> str:
    candidates = (
        getattr(self.app_instance, "current_runtime_backend", None),
        getattr(self.app_instance, "runtime_backend", None),
    )
    for candidate in candidates:
        if candidate:
            return str(candidate).strip().lower()
    return "local"


async def _fetch_templates(self) -> list[dict[str, Any]]:
    scope = getattr(self.app_instance, "rag_admin_scope_service", None)
    if scope is None:
        return []
    return await scope.list_templates(mode=self._runtime_backend())
```

The editor refactor in this task stays narrow:

- create and update go through the scope service
- the editor continues to work with template JSON as text
- no visual redesign

- [ ] **Step 4: Run the widget tests again**

Run: `python3 -m pytest Tests/UI/test_chunking_templates_widget_parity.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/chunking_templates_widget.py tldw_chatbook/Widgets/chunking_template_editor.py Tests/UI/test_chunking_templates_widget_parity.py
git commit -m "feat: add backend-aware chunking templates widget"
```

## Task 4: Make `EmbeddingsManagementWindow` Collection Management Real And Backend-Aware

**Files:**
- Modify: `tldw_chatbook/UI/Embeddings_Management_Window.py`
- Modify: `tldw_chatbook/Widgets/embeddings_list_items.py`
- Test: `Tests/UI/test_embeddings_management_window_parity.py`

- [ ] **Step 1: Write the failing collection-management tests**

```python
@pytest.mark.asyncio
async def test_embeddings_window_loads_local_collections_from_scope_service(app):
    scope = FakeScopeService(local_collections=[normalized_collection("local", "notes_embeddings", count=5)])
    app.rag_admin_scope_service = scope
    app.current_runtime_backend = "local"

    window = EmbeddingsManagementWindow(app_instance=app)
    await pilot_mount(window)

    assert "notes_embeddings" in [record["name"] for record in window.collections]


@pytest.mark.asyncio
async def test_embeddings_window_loads_server_collection_stats_on_selection(app):
    scope = FakeScopeService(
        server_collections=[normalized_collection("server", "remote_embeddings")],
        server_collection_detail=normalized_collection("server", "remote_embeddings", count=12, dimension=1536),
    )
    app.rag_admin_scope_service = scope
    app.current_runtime_backend = "server"

    window = EmbeddingsManagementWindow(app_instance=app)
    await pilot_mount(window)
    await window._update_collection_info("remote_embeddings")

    count_widget = window.query_one("#embeddings-collection-count", Static)
    assert "12" in str(count_widget.renderable)


@pytest.mark.asyncio
async def test_embeddings_window_deletes_collection_via_scope_service(app):
    scope = FakeScopeService(local_collections=[normalized_collection("local", "demo")])
    app.rag_admin_scope_service = scope

    window = EmbeddingsManagementWindow(app_instance=app)
    window.selected_collection = "demo"

    await window._delete_selected_collection_for_test()

    assert scope.deleted_collections == [("local", "demo")]
```

- [ ] **Step 2: Run the collection-management tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_embeddings_management_window_parity.py -q`  
Expected: FAIL because the collection pane still uses placeholder logic and TODO deletes.

- [ ] **Step 3: Implement the minimal collection-management refactor**

```python
async def _load_collections_list(self) -> None:
    scope = getattr(self.app_instance, "rag_admin_scope_service", None)
    if scope is None:
        return

    mode = self._runtime_backend()
    self.collections = await scope.list_collections(mode=mode)
    ...


async def _update_collection_info(self, collection_name: str) -> None:
    scope = getattr(self.app_instance, "rag_admin_scope_service", None)
    if scope is None:
        return

    detail = await scope.get_collection_detail(
        mode=self._runtime_backend(),
        collection_name=collection_name,
    )
    self.query_one("#embeddings-collection-count", Static).update(str(detail.get("document_count") or 0))
```

This task includes:

- real collection list loading
- real detail/stat display
- single-collection delete through the scope service
- batch delete through the same path

It does not include:

- model-management parity
- collection export parity
- new layout work

- [ ] **Step 4: Run the collection-management tests again**

Run: `python3 -m pytest Tests/UI/test_embeddings_management_window_parity.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Embeddings_Management_Window.py tldw_chatbook/Widgets/embeddings_list_items.py Tests/UI/test_embeddings_management_window_parity.py
git commit -m "feat: add backend-aware embeddings collection management"
```

## Task 5: Final Regression, Docs, And Integration Sweep

**Files:**
- Modify: `Docs/Parity/2026-04-19-data-compatibility-map.md`
- Modify: `Docs/Parity/2026-04-19-rollout-backlog.md`
- Modify: any touched files from Tasks 1-4 if review fixes are needed

- [ ] **Step 1: Write the final focused regression commands into the plan notes**

Run:

```bash
python3 -m pytest Tests/tldw_api/test_rag_admin_client.py Tests/tldw_api/test_rag_admin_schemas.py Tests/RAG_Admin/test_rag_admin_normalizers.py Tests/RAG_Admin/test_rag_admin_scope_service.py Tests/RAG_Admin/test_server_rag_admin_service.py Tests/UI/test_chunking_templates_widget_parity.py Tests/UI/test_embeddings_management_window_parity.py -q
```

Expected: PASS

- [ ] **Step 2: Update parity docs**

Record:

- normalized template identity and field mapping
- normalized collection identity and stats handling
- `SearchRAGWindow` explicitly deferred
- local-first/no-sync policy retained

- [ ] **Step 3: Run the focused regression suite**

Run the command from Step 1 and confirm all tests pass.

- [ ] **Step 4: Run one adjacent smoke check**

Run:

```bash
python3 -m pytest Tests/test_smoke.py -q
```

Expected: PASS or known unrelated failures clearly identified.

- [ ] **Step 5: Commit**

```bash
git add Docs/Parity/2026-04-19-data-compatibility-map.md Docs/Parity/2026-04-19-rollout-backlog.md
git commit -m "docs: record rag admin parity vertical"
```
