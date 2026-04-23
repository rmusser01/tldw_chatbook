# Writing Suite Structural Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone-first Writing Suite to Chatbook with local structural authoring parity and a contract-honest server mode for the current `tldw_server` manuscript API.

**Architecture:** Build a dedicated writing vertical with its own SQLite store, normalized interop models, local/server backend services, and one scope service that routes all actions by explicit source. The TUI exposes one Writing destination with Local and Server panes, no mixed view, and disables unsupported server actions centrally rather than hiding contract gaps in UI code.

**Tech Stack:** Python, SQLite, Pydantic, Textual, existing `tldw_api` async client, existing runtime policy/action registry, pytest.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-04-22-writing-suite-structural-parity-design.md`
- Server endpoint reference: `/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Server schema reference: `/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
- Server route prefix verified from `tldw_server2/tldw_Server_API/app/main.py`: `/api/v1/writing/manuscripts`

## Non-Negotiable Behavior

- Project is required top-level in local and server mode.
- Local and server sources are entirely separate; no mixed browse/search view in v1.
- Writing entities are separate from Notes and Workspaces.
- Local writes never call the server.
- Server writes never mutate local rows.
- Server mode never silently falls back to local mode.
- Server unassigned chapters must render in a project-level `Unassigned Chapters` bucket and must not be dropped or hidden inside a fake manuscript.
- Only scenes have Markdown body drafts.
- Manuscript and chapter working state is metadata plus structure/order only, not authored body content.
- Autosave updates the current working state and does not create a new version number.
- `Create New Version` snapshots the current working state into the next immutable numbered local version.
- Server manual versions and server trash restore remain disabled until verified server endpoints exist.
- Direct manuscript-level scenes are local-only until the server supports parentless scenes.
- Server scene reparenting is disabled until the server supports changing scene parent.

## Worker Guidance

- Use `superpowers:test-driven-development` for every implementation task.
- Use `superpowers:verification-before-completion` before claiming a task is complete or before committing.
- Keep commits small and path-scoped. Do not stage unrelated dirty files from parallel MCP work.
- In this plan, `version` on working rows means optimistic row version for conflict checks. User-facing manual versions are stored as `writing_versions.version_number` and are created only by `Create New Version`.

## File Map

Create:

- `tldw_chatbook/tldw_api/writing_manuscript_schemas.py` - Pydantic mirror of the supported server manuscript project/part/chapter/scene/structure/reorder/search contract.
- `tldw_chatbook/Writing_Interop/__init__.py` - public exports for writing interop services and models.
- `tldw_chatbook/Writing_Interop/writing_models.py` - source-neutral normalized dataclasses/enums for projects, manuscripts, chapters, scenes, versions, trash, outline nodes, and capability results.
- `tldw_chatbook/Writing_Interop/writing_markdown_adapter.py` - deterministic Markdown-to-server-content and server-content-to-Markdown adapter.
- `tldw_chatbook/Writing_Interop/writing_normalizers.py` - local/server record normalizers into `writing_models.py`.
- `tldw_chatbook/Writing_Interop/local_writing_service.py` - local backend service over `WritingDatabase`.
- `tldw_chatbook/Writing_Interop/server_writing_service.py` - server backend adapter over `TLDWAPIClient`.
- `tldw_chatbook/Writing_Interop/writing_scope_service.py` - authoritative source router and capability gate.
- `tldw_chatbook/DB/Writing_DB.py` - dedicated SQLite database for local writing projects.
- `tldw_chatbook/UI/Screens/writing_screen.py` - Textual screen wrapper.
- `tldw_chatbook/UI/Writing_Window.py` - Writing Suite TUI container.
- `tldw_chatbook/UI/Writing_Modules/__init__.py` - module package marker.
- `tldw_chatbook/UI/Writing_Modules/writing_controller.py` - async UI controller between widgets and scope service.
- `tldw_chatbook/Widgets/Writing/__init__.py` - widget package marker.
- `tldw_chatbook/Widgets/Writing/writing_outline_tree.py` - project/manuscript/chapter/scene outline widget.
- `tldw_chatbook/Widgets/Writing/writing_detail_panel.py` - metadata and scene editor panel.
- `tldw_chatbook/Widgets/Writing/writing_source_panel.py` - source switch, project list, search/filter controls.
- `Docs/Development/writing-suite-structural-parity.md` - implementation verification record.

Modify:

- `tldw_chatbook/tldw_api/client.py` - add manuscript API methods under `/api/v1/writing/manuscripts`.
- `tldw_chatbook/tldw_api/__init__.py` - export writing manuscript schemas.
- `tldw_chatbook/config.py` - add `get_writing_db_path()`.
- `tldw_chatbook/app.py` - instantiate writing DB/services/scope service and register Writing screen.
- `tldw_chatbook/Constants.py` - add `TAB_WRITING = "writing"` and include it in `ALL_TABS`.
- `tldw_chatbook/runtime_policy/registry.py` - only if tests reveal missing writing action rows; the current registry already includes `writing_suite`.
- `Docs/Parity/2026-04-21-gap-ledger.md` - update Writing Suite current-state row after implementation.
- `Docs/Parity/2026-04-21-target-state-design.md` - update Writing Suite target-state notes after implementation.
- `Tests/UI/test_screen_navigation.py` - assert Writing screen registration and service wiring.

Create tests:

- `Tests/tldw_api/test_writing_manuscripts_client.py`
- `Tests/Writing_Interop/test_writing_models.py`
- `Tests/Writing_Interop/test_writing_markdown_adapter.py`
- `Tests/Writing_Interop/test_writing_normalizers.py`
- `Tests/DB/test_writing_db.py`
- `Tests/Writing_Interop/test_local_writing_service.py`
- `Tests/Writing_Interop/test_server_writing_service.py`
- `Tests/Writing_Interop/test_writing_scope_service.py`
- `Tests/UI/test_writing_screen.py`

## Implementation Tasks

### Task 1: Server Manuscript Schemas and API Client

**Files:**

- Create: `tldw_chatbook/tldw_api/writing_manuscript_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_writing_manuscripts_client.py`

- [ ] **Step 1: Write failing client tests for project routes**

Add tests that patch `TLDWAPIClient._request` with `AsyncMock` and assert exact method, endpoint, params, headers, and typed response validation.

```python
async def test_list_manuscript_projects_uses_server_prefix():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(return_value={"projects": [], "total": 0})

    response = await client.list_manuscript_projects(status="draft", limit=25, offset=5)

    assert response.total == 0
    client._request.assert_awaited_once_with(
        "GET",
        "/api/v1/writing/manuscripts/projects",
        params={"status": "draft", "limit": 25, "offset": 5},
    )
```

Add equivalent tests for `create_manuscript_project`, `get_manuscript_project`, `update_manuscript_project`, and `delete_manuscript_project`. Update/delete tests must assert `headers={"expected-version": "3"}`.

- [ ] **Step 2: Run the failing API-client project tests**

Run:

```bash
python3 -m pytest Tests/tldw_api/test_writing_manuscripts_client.py -q
```

Expected: fail because `writing_manuscript_schemas.py` and client methods do not exist.

- [ ] **Step 3: Add Pydantic schemas for the supported server contract**

Implement only the structural manuscript routes in `tldw_chatbook/tldw_api/writing_manuscript_schemas.py`. Do not include server characters/world-info/plot/research schemas in this vertical.

Required classes:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


ProjectStatus = Literal["draft", "outlining", "writing", "revising", "complete", "archived"]
NodeStatus = Literal["outline", "draft", "revising", "final"]


class ManuscriptProjectCreateRequest(BaseModel):
    title: str
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: ProjectStatus = "draft"
    synopsis: str | None = None
    target_word_count: int | None = None
    settings: dict[str, Any] | None = None
    id: str | None = None


class ManuscriptProjectUpdateRequest(BaseModel):
    title: str | None = None
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: ProjectStatus | None = None
    synopsis: str | None = None
    target_word_count: int | None = None
    settings: dict[str, Any] | None = None
```

Also add:

- `ManuscriptProjectResponse`
- `ManuscriptProjectListResponse`
- `ManuscriptPartCreateRequest`
- `ManuscriptPartUpdateRequest`
- `ManuscriptPartResponse`
- `ManuscriptChapterCreateRequest`
- `ManuscriptChapterUpdateRequest`
- `ManuscriptChapterResponse`
- `ManuscriptSceneCreateRequest`
- `ManuscriptSceneUpdateRequest`
- `ManuscriptSceneResponse`
- `SceneSummary`
- `ChapterSummary`
- `PartSummary`
- `ManuscriptStructureResponse`
- `ReorderItem`
- `ReorderRequest`
- `ManuscriptSearchResult`
- `ManuscriptSearchResponse`

For server scene responses, support both `content_json` and computed `content`:

```python
class ManuscriptSceneResponse(BaseModel):
    id: str
    chapter_id: str
    project_id: str
    title: str
    sort_order: float
    content_json: str | None = None
    content_plain: str | None = None
    synopsis: str | None = None
    word_count: int = 0
    pov_character_id: str | None = None
    status: str = "draft"
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int

    @computed_field
    @property
    def content(self) -> dict[str, Any] | None:
        ...
```

- [ ] **Step 4: Add client methods under the verified server prefix**

In `tldw_chatbook/tldw_api/client.py`, import the new schemas and add methods:

```python
async def list_manuscript_projects(
    self,
    *,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> ManuscriptProjectListResponse:
    params = {"limit": limit, "offset": offset}
    if status is not None:
        params["status"] = status
    response = await self._request(
        "GET",
        "/api/v1/writing/manuscripts/projects",
        params=params,
    )
    return ManuscriptProjectListResponse.model_validate(response)
```

Add route methods:

- `create_manuscript_project(request_data)`
- `get_manuscript_project(project_id)`
- `update_manuscript_project(project_id, request_data, expected_version)`
- `delete_manuscript_project(project_id, expected_version)`
- `get_manuscript_project_structure(project_id)`
- `reorder_manuscript_entities(project_id, request_data)`
- `search_manuscript_project(project_id, query, limit=20)`
- `create_manuscript_part(project_id, request_data)`
- `list_manuscript_parts(project_id)`
- `get_manuscript_part(part_id)`
- `update_manuscript_part(part_id, request_data, expected_version)`
- `delete_manuscript_part(part_id, expected_version)`
- `create_manuscript_chapter(project_id, request_data)`
- `list_manuscript_chapters(project_id, part_id=None)`
- `get_manuscript_chapter(chapter_id)`
- `update_manuscript_chapter(chapter_id, request_data, expected_version)`
- `delete_manuscript_chapter(chapter_id, expected_version)`
- `create_manuscript_scene(chapter_id, request_data)`
- `list_manuscript_scenes(chapter_id)`
- `get_manuscript_scene(scene_id)`
- `update_manuscript_scene(scene_id, request_data, expected_version)`
- `delete_manuscript_scene(scene_id, expected_version)`

All PATCH/DELETE methods must pass `headers={"expected-version": str(expected_version)}`.

- [ ] **Step 5: Export schemas from `tldw_api.__init__`**

Mirror the existing flashcards/evaluations export style. Do not wildcard-import.

- [ ] **Step 6: Run API-client tests**

Run:

```bash
python3 -m pytest Tests/tldw_api/test_writing_manuscripts_client.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/tldw_api/writing_manuscript_schemas.py tldw_chatbook/tldw_api/client.py tldw_chatbook/tldw_api/__init__.py Tests/tldw_api/test_writing_manuscripts_client.py
git commit -m "feat: add writing manuscript api client"
```

### Task 2: Normalized Models and Markdown Adapter

**Files:**

- Create: `tldw_chatbook/Writing_Interop/__init__.py`
- Create: `tldw_chatbook/Writing_Interop/writing_models.py`
- Create: `tldw_chatbook/Writing_Interop/writing_markdown_adapter.py`
- Create: `tldw_chatbook/Writing_Interop/writing_normalizers.py`
- Test: `Tests/Writing_Interop/test_writing_models.py`
- Test: `Tests/Writing_Interop/test_writing_markdown_adapter.py`
- Test: `Tests/Writing_Interop/test_writing_normalizers.py`

- [ ] **Step 1: Write failing tests for model invariants**

Cover:

- project is required for every entity
- `WritingChapter.manuscript_id` may be `None` for project-level unassigned chapters
- direct manuscript scene requires `manuscript_id` and has no `chapter_id`
- scene under unassigned chapter may have `chapter_id` and no `manuscript_id`
- only scenes carry `body_markdown`
- manuscript/chapter draft payloads reject `body_markdown`

Example:

```python
def test_container_drafts_do_not_accept_body_markdown():
    with pytest.raises(ValueError, match="Only scene drafts"):
        WritingDraft(
            source="local",
            entity_kind="chapter",
            entity_id="chapter-1",
            project_id="project-1",
            metadata={"title": "Chapter 1"},
            body_markdown="# illegal",
        )
```

- [ ] **Step 2: Write failing tests for Markdown adapter**

Cover:

- Markdown round-trips through deterministic wrapper content.
- Plain text is derived from Markdown without silently stripping all content.
- Server TipTap content without wrapper falls back to `content_plain`.
- Unknown/invalid `content_json` falls back to `content_plain`.

Expected wrapper shape:

```python
{
    "type": "doc",
    "content": [
        {
            "type": "paragraph",
            "attrs": {
                "tldw_chatbook_markdown": True,
                "format": "markdown",
                "version": 1,
            },
            "content": [{"type": "text", "text": markdown}],
        }
    ],
}
```

- [ ] **Step 3: Implement normalized dataclasses/enums**

In `writing_models.py`, include:

```python
WritingSource = Literal["local", "server"]
WritingEntityKind = Literal["project", "manuscript", "chapter", "scene"]
WritingStatus = Literal["draft", "outlining", "writing", "revising", "complete", "archived", "outline", "final"]

@dataclass(frozen=True, slots=True)
class WritingProject:
    source: WritingSource
    id: str
    title: str
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: str = "draft"
    synopsis: str | None = None
    target_word_count: int | None = None
    word_count: int = 0
    version: int = 1
    deleted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Add dataclasses:

- `WritingManuscript`
- `WritingChapter`
- `WritingScene`
- `WritingDraft`
- `WritingVersion`
- `WritingTrashEntry`
- `WritingOutlineNode`
- `WritingCapability`

Use `__post_init__` validation to enforce the non-negotiable hierarchy rules.

- [ ] **Step 4: Implement Markdown adapter**

Provide:

- `markdown_to_server_content(markdown: str) -> dict[str, Any]`
- `markdown_to_plain_text(markdown: str) -> str`
- `server_content_to_markdown(content: Mapping[str, Any] | None, content_plain: str | None) -> str`
- `parse_server_content_json(content_json: str | None) -> dict[str, Any] | None`

Keep the adapter deterministic and dependency-light. Do not add a Markdown parsing dependency for v1; simple heading/list marker cleanup is sufficient for `content_plain`.

- [ ] **Step 5: Implement normalizers**

Normalize local dict rows and server schema objects into the dataclasses. Include specific tests that server `unassigned_chapters` become a `WritingOutlineNode(kind="unassigned_chapters")` and not a fake manuscript.

- [ ] **Step 6: Run model/adapter/normalizer tests**

Run:

```bash
python3 -m pytest Tests/Writing_Interop/test_writing_models.py Tests/Writing_Interop/test_writing_markdown_adapter.py Tests/Writing_Interop/test_writing_normalizers.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Writing_Interop Tests/Writing_Interop/test_writing_models.py Tests/Writing_Interop/test_writing_markdown_adapter.py Tests/Writing_Interop/test_writing_normalizers.py
git commit -m "feat: add writing interop models"
```

### Task 3: Local Writing Database

**Files:**

- Create: `tldw_chatbook/DB/Writing_DB.py`
- Modify: `tldw_chatbook/config.py`
- Test: `Tests/DB/test_writing_db.py`

- [ ] **Step 1: Write failing database tests**

Cover:

- schema initializes in a temp DB
- project create/list/get/update/soft-delete/restore
- manuscript create under project
- chapter create with `manuscript_id=None` appears as unassigned
- chapter assign/unassign changes `manuscript_id`
- scene create under chapter
- direct scene create under manuscript with `chapter_id=None`
- soft-deleted records are excluded from normal list and present in trash list
- optimistic version mismatch raises a deterministic exception
- manual version snapshot increments `writing_versions.version_number` but does not change the working row's optimistic `version` unless the working row is also updated

- [ ] **Step 2: Run database tests to verify failure**

Run:

```bash
python3 -m pytest Tests/DB/test_writing_db.py -q
```

Expected: fail because `WritingDatabase` does not exist.

- [ ] **Step 3: Add config path**

In `tldw_chatbook/config.py`, add:

```python
def get_writing_db_path() -> Path:
    custom_path = get_cli_setting("database", "writing_db_path", None)
    if custom_path:
        return Path(custom_path).expanduser().resolve()
    return get_user_data_dir() / "tldw_chatbook_writing.db"
```

- [ ] **Step 4: Implement `WritingDatabase` schema**

Follow the separate-domain pattern in `tldw_chatbook/DB/Subscriptions_DB.py`.

Schema tables:

```sql
CREATE TABLE IF NOT EXISTS writing_projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    subtitle TEXT,
    author TEXT,
    genre TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    synopsis TEXT,
    target_word_count INTEGER,
    settings_json TEXT NOT NULL DEFAULT '{}',
    word_count INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS writing_manuscripts (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES writing_projects(id),
    title TEXT NOT NULL,
    sort_order REAL NOT NULL DEFAULT 0,
    synopsis TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    word_count INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS writing_chapters (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES writing_projects(id),
    manuscript_id TEXT REFERENCES writing_manuscripts(id),
    title TEXT NOT NULL,
    sort_order REAL NOT NULL DEFAULT 0,
    synopsis TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    word_count INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS writing_scenes (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES writing_projects(id),
    manuscript_id TEXT REFERENCES writing_manuscripts(id),
    chapter_id TEXT REFERENCES writing_chapters(id),
    title TEXT NOT NULL,
    body_markdown TEXT NOT NULL DEFAULT '',
    sort_order REAL NOT NULL DEFAULT 0,
    synopsis TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    word_count INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT,
    CHECK (chapter_id IS NOT NULL OR manuscript_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS writing_versions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES writing_projects(id),
    entity_kind TEXT NOT NULL CHECK (entity_kind IN ('manuscript', 'chapter', 'scene')),
    entity_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    body_markdown TEXT,
    created_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    UNIQUE(entity_kind, entity_id, version_number)
);
```

Add indexes for `project_id`, parent IDs, `deleted`, and `entity_kind/entity_id`.

- [ ] **Step 5: Implement database CRUD methods**

Use explicit transaction helpers and expected-version checks. Add methods:

- `create_project`, `list_projects`, `get_project`, `update_project`, `soft_delete_project`, `restore_project`
- `create_manuscript`, `list_manuscripts`, `get_manuscript`, `update_manuscript`, `soft_delete_manuscript`, `restore_manuscript`
- `create_chapter`, `list_chapters`, `get_chapter`, `update_chapter`, `assign_chapter`, `soft_delete_chapter`, `restore_chapter`
- `create_scene`, `list_scenes`, `get_scene`, `update_scene`, `move_scene_local`, `soft_delete_scene`, `restore_scene`
- `get_project_structure`
- `reorder_items`
- `list_trash`
- `create_version`, `list_versions`, `get_version`, `restore_version_to_working_state`

Do not add chapter body methods.

- [ ] **Step 6: Run database tests**

Run:

```bash
python3 -m pytest Tests/DB/test_writing_db.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/DB/Writing_DB.py tldw_chatbook/config.py Tests/DB/test_writing_db.py
git commit -m "feat: add local writing database"
```

### Task 4: Local Writing Service

**Files:**

- Create: `tldw_chatbook/Writing_Interop/local_writing_service.py`
- Modify: `tldw_chatbook/Writing_Interop/__init__.py`
- Test: `Tests/Writing_Interop/test_local_writing_service.py`

- [ ] **Step 1: Write failing local service tests**

Use a temp `WritingDatabase`. Cover:

- project/manuscript/chapter/scene CRUD returns normalized dataclasses
- unassigned chapter appears in outline
- direct manuscript-level scene works locally
- scene under unassigned chapter works locally
- autosave scene updates `body_markdown` and record version
- creating a manual version increments version snapshot number
- restoring a version updates working state but does not create another version
- trash listing and restore work for all entity kinds

- [ ] **Step 2: Run failing local service tests**

Run:

```bash
python3 -m pytest Tests/Writing_Interop/test_local_writing_service.py -q
```

Expected: fail because service does not exist.

- [ ] **Step 3: Implement local service**

Implement a thin async-compatible wrapper over `WritingDatabase`. Match the route/service method names that `WritingScopeService` will call:

```python
class LocalWritingService:
    def __init__(self, db: WritingDatabase | None):
        self.db = db

    def _require_db(self) -> WritingDatabase:
        if self.db is None:
            raise ValueError("Local writing backend is unavailable.")
        return self.db

    async def list_projects(self, *, status: str | None = None, limit: int = 100, offset: int = 0):
        rows = self._require_db().list_projects(status=status, limit=limit, offset=offset)
        return [normalize_local_project(row) for row in rows]
```

Prefer async methods even when the DB work is synchronous so the scope service can treat both sources uniformly.

- [ ] **Step 4: Run local service tests**

Run:

```bash
python3 -m pytest Tests/Writing_Interop/test_local_writing_service.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Writing_Interop/local_writing_service.py tldw_chatbook/Writing_Interop/__init__.py Tests/Writing_Interop/test_local_writing_service.py
git commit -m "feat: add local writing service"
```

### Task 5: Server Writing Service

**Files:**

- Create: `tldw_chatbook/Writing_Interop/server_writing_service.py`
- Modify: `tldw_chatbook/Writing_Interop/__init__.py`
- Test: `Tests/Writing_Interop/test_server_writing_service.py`

- [ ] **Step 1: Write failing server service tests**

Use a fake API client. Cover:

- server projects normalize as `source="server"`
- server parts normalize as manuscripts
- server structure maps `parts` and `unassigned_chapters`
- server unassigned chapters are preserved under an explicit bucket
- creating a server scene requires `chapter_id`
- direct manuscript-level server scene creation raises unsupported capability error
- server scene update sends Markdown wrapper content and plain text
- server update/delete passes expected version through to client methods

- [ ] **Step 2: Run failing server service tests**

Run:

```bash
python3 -m pytest Tests/Writing_Interop/test_server_writing_service.py -q
```

Expected: fail because service does not exist.

- [ ] **Step 3: Implement server service factory**

Follow the existing pattern in `tldw_chatbook/Study_Interop/server_study_service.py`:

```python
class ServerWritingService:
    def __init__(self, client: TLDWAPIClient | None):
        self.client = client

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ServerWritingService":
        return cls(build_tldw_api_client_from_config(config))
```

Use `build_tldw_api_client_from_config` from `tldw_chatbook/Chatbooks/server_chatbook_service.py`.

- [ ] **Step 4: Implement server structural methods**

Add server-backed methods for:

- projects
- manuscripts mapped to parts
- chapters
- scenes under chapters
- project structure
- project search
- reorder parts/chapters/scenes where current server contract supports it
- assign/unassign chapter via chapter update `part_id`

Do not implement:

- direct manuscript-level scenes
- server manual versions
- server version restore
- server trash list
- server restore
- server scene reparenting

Unsupported methods should raise a deterministic exception, for example:

```python
class WritingCapabilityError(RuntimeError):
    def __init__(self, capability: str, source: str, reason: str):
        super().__init__(reason)
        self.capability = capability
        self.source = source
        self.reason = reason
```

- [ ] **Step 5: Run server service tests**

Run:

```bash
python3 -m pytest Tests/Writing_Interop/test_server_writing_service.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Writing_Interop/server_writing_service.py tldw_chatbook/Writing_Interop/__init__.py Tests/Writing_Interop/test_server_writing_service.py
git commit -m "feat: add server writing service"
```

### Task 6: Writing Scope Service and Capability Gates

**Files:**

- Create: `tldw_chatbook/Writing_Interop/writing_scope_service.py`
- Modify: `tldw_chatbook/Writing_Interop/__init__.py`
- Modify: `tldw_chatbook/runtime_policy/registry.py` only if required by failing registry tests
- Test: `Tests/Writing_Interop/test_writing_scope_service.py`
- Test: `Tests/RuntimePolicy/test_runtime_policy_registry.py` if registry expectations are updated

- [ ] **Step 1: Write failing scope service tests**

Cover:

- default mode is local
- invalid mode fails
- local routes only to local backend
- server routes only to server backend
- server unavailable fails visibly
- no fallback from server to local
- capability helper reports server direct scene unsupported
- capability helper reports server manual versions unsupported
- capability helper reports server trash restore unsupported
- capability helper reports server scene reparent unsupported
- policy action ids are checked for top-level CRUD actions

Example:

```python
async def test_server_mode_does_not_fallback_to_local_when_server_missing():
    service = WritingScopeService(local_service=FakeLocal(), server_service=None)

    with pytest.raises(ValueError, match="Server writing backend is unavailable"):
        await service.list_projects(mode="server")

    assert FakeLocal.calls == []
```

- [ ] **Step 2: Run failing scope service tests**

Run:

```bash
python3 -m pytest Tests/Writing_Interop/test_writing_scope_service.py -q
```

Expected: fail because scope service does not exist.

- [ ] **Step 3: Implement source routing**

Implement:

```python
class WritingBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class WritingScopeService:
    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any | None = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer
```

Add `_normalize_mode`, `_service_for_mode`, `_maybe_await`, and action-specific wrappers for all local/server service methods.

- [ ] **Step 4: Implement central capability helper**

The UI must consume this helper instead of duplicating server contract rules:

```python
def get_capability(self, *, mode: str, action: str, entity_kind: str, parent_kind: str | None = None) -> WritingCapability:
    if mode == "server" and action in {"create_version", "restore_version"}:
        return WritingCapability(False, "server_version_history_unavailable")
    if mode == "server" and action == "restore_deleted":
        return WritingCapability(False, "server_trash_restore_unavailable")
    if mode == "server" and entity_kind == "scene" and parent_kind == "manuscript" and action in {"create", "move"}:
        return WritingCapability(False, "server_direct_manuscript_scene_unavailable")
    if mode == "server" and entity_kind == "scene" and action == "reparent":
        return WritingCapability(False, "server_scene_reparent_unavailable")
    return WritingCapability(True, None)
```

Block invalid actions in service methods as well. UI disablement is not sufficient.

- [ ] **Step 5: Integrate runtime policy checks**

Use existing writing action IDs:

- `writing.projects.list.local`
- `writing.projects.list.server`
- `writing.projects.create.local`
- `writing.projects.create.server`
- `writing.manuscripts.update.local`
- `writing.manuscripts.update.server`
- `writing.chapters.delete.local`
- `writing.chapters.delete.server`
- `writing.scenes.create.local`
- `writing.scenes.create.server`

If `validate_registry_completeness()` already passes and these action IDs exist, do not modify `runtime_policy/registry.py`. If a required row is missing, update the existing `writing_suite` seed rather than adding a second writing capability.

- [ ] **Step 6: Run scope and registry tests**

Run:

```bash
python3 -m pytest Tests/Writing_Interop/test_writing_scope_service.py Tests/RuntimePolicy/test_runtime_policy_registry.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Writing_Interop/writing_scope_service.py tldw_chatbook/Writing_Interop/__init__.py Tests/Writing_Interop/test_writing_scope_service.py Tests/RuntimePolicy/test_runtime_policy_registry.py
git commit -m "feat: add writing scope service"
```

If `runtime_policy/registry.py` was not changed, omit it from `git add`.

### Task 7: App Wiring and Navigation

**Files:**

- Modify: `tldw_chatbook/Constants.py`
- Modify: `tldw_chatbook/app.py`
- Create: `tldw_chatbook/UI/Screens/writing_screen.py`
- Create: `tldw_chatbook/UI/Writing_Window.py`
- Test: `Tests/UI/test_screen_navigation.py`
- Test: `Tests/UI/test_writing_screen.py`

- [ ] **Step 1: Write failing navigation tests**

Extend `Tests/UI/test_screen_navigation.py` to assert:

- `"writing"` is in `ALL_TABS`
- screen registry maps `"writing"` to `WritingScreen`
- app wires `writing_db`, `local_writing_service`, `server_writing_service`, and `writing_scope_service`

Add a minimal `Tests/UI/test_writing_screen.py` that mounts `WritingScreen` with a fake app exposing `writing_scope_service`.

- [ ] **Step 2: Run failing UI navigation tests**

Run:

```bash
python3 -m pytest Tests/UI/test_screen_navigation.py Tests/UI/test_writing_screen.py -q
```

Expected: fail because screen and tab are missing.

- [ ] **Step 3: Add constants**

In `tldw_chatbook/Constants.py`:

```python
TAB_WRITING = "writing"
```

Add it to `ALL_TABS` in the same style as existing tabs.

- [ ] **Step 4: Add `WritingScreen` shell**

Create `tldw_chatbook/UI/Screens/writing_screen.py` by following `tldw_chatbook/UI/Screens/study_screen.py` patterns:

```python
class WritingScreen(BaseAppScreen):
    CSS_PATH = None
    BINDINGS = []

    def compose_content(self) -> ComposeResult:
        yield WritingWindow(id="writing-window")

    def save_state(self) -> dict[str, Any]:
        window = self.query_one("#writing-window", WritingWindow)
        return window.save_state()

    def restore_state(self, state: dict[str, Any]) -> None:
        window = self.query_one("#writing-window", WritingWindow)
        window.restore_state(state)
```

- [ ] **Step 5: Add minimal `WritingWindow` shell**

Create `tldw_chatbook/UI/Writing_Window.py` in this task so `WritingScreen` can import and mount cleanly before the full browse UI exists:

```python
class WritingWindow(Container):
    def compose(self) -> ComposeResult:
        yield Label("Writing Suite")
        yield Static("Writing Suite is initializing.", id="writing-status")

    def save_state(self) -> dict[str, Any]:
        return {"source": "local"}

    def restore_state(self, state: dict[str, Any]) -> None:
        return None
```

Task 8 replaces this placeholder with the real source-switched browse layout.

- [ ] **Step 6: Wire services in app**

In `tldw_chatbook/app.py`:

- import `get_writing_db_path`
- import `WritingDatabase`
- import `LocalWritingService`
- import `ServerWritingService`
- import `WritingScopeService`
- import `WritingScreen`
- instantiate `self.writing_db`
- add `_wire_writing_services()`
- call `_wire_writing_services()` near study/evaluation/character service wiring
- add `"writing": WritingScreen` to the screen map

Pattern:

```python
def _wire_writing_services(self) -> None:
    self.local_writing_service = LocalWritingService(self.writing_db) if self.writing_db is not None else None
    try:
        self.server_writing_service = ServerWritingService.from_config(self.app_config)
    except ValueError:
        self.server_writing_service = ServerWritingService(client=None)
    self.writing_scope_service = WritingScopeService(
        local_service=self.local_writing_service,
        server_service=self.server_writing_service,
        policy_enforcer=self.service_policy_enforcer,
    )
```

- [ ] **Step 7: Run navigation tests**

Run:

```bash
python3 -m pytest Tests/UI/test_screen_navigation.py Tests/UI/test_writing_screen.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Constants.py tldw_chatbook/app.py tldw_chatbook/UI/Screens/writing_screen.py tldw_chatbook/UI/Writing_Window.py Tests/UI/test_screen_navigation.py Tests/UI/test_writing_screen.py
git commit -m "feat: wire writing suite screen"
```

### Task 8: Writing Window Browse and Outline

**Files:**

- Modify: `tldw_chatbook/UI/Writing_Window.py`
- Create: `tldw_chatbook/UI/Writing_Modules/__init__.py`
- Create: `tldw_chatbook/UI/Writing_Modules/writing_controller.py`
- Create: `tldw_chatbook/Widgets/Writing/__init__.py`
- Create: `tldw_chatbook/Widgets/Writing/writing_source_panel.py`
- Create: `tldw_chatbook/Widgets/Writing/writing_outline_tree.py`
- Create: `tldw_chatbook/Widgets/Writing/writing_detail_panel.py`
- Test: `Tests/UI/test_writing_screen.py`

- [ ] **Step 1: Write failing UI browse tests**

Cover:

- default source is local
- switching source reloads projects from the selected source only
- server source does not display local projects
- outline renders `Project -> Manuscript -> Chapter -> Scene`
- outline renders `Project -> Unassigned Chapters -> Chapter -> Scene`
- selecting a node loads the detail panel
- missing server configuration shows an unavailable state, not local fallback data

- [ ] **Step 2: Run failing UI browse tests**

Run:

```bash
python3 -m pytest Tests/UI/test_writing_screen.py -q
```

Expected: fail because widgets do not exist.

- [ ] **Step 3: Implement `WritingWindow` composition**

Use existing Textual patterns from `tldw_chatbook/UI/Study_Window.py`, but keep v1 focused:

- source switch: `Local` / `Server`
- project list
- create project button
- outline tree
- detail/editor panel
- status/notice area for unsupported actions

The source switch must call the controller and requery. Do not keep stale local browse payloads when switching to server.

- [ ] **Step 4: Implement controller**

`writing_controller.py` owns async calls to `app.writing_scope_service`. It should expose:

- `load_projects(source)`
- `load_project_structure(source, project_id)`
- `select_node(node_id, kind)`
- `create_project(source, payload)`
- `create_child(source, parent_context, payload)`
- `save_current(source, entity_kind, entity_id, payload, expected_version)`
- `delete_current(source, entity_kind, entity_id, expected_version)`

Use `scope_service.get_capability(...)` before enabling/disabling action buttons.

- [ ] **Step 5: Implement outline widget**

Render stable labels:

- project title
- manuscript title
- literal `Unassigned Chapters`
- chapter title
- scene title

Store source/kind/id/project_id/manuscript_id/chapter_id/version in node data. The unassigned bucket node has no entity ID and is not editable.

- [ ] **Step 6: Run UI browse tests**

Run:

```bash
python3 -m pytest Tests/UI/test_writing_screen.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Writing_Window.py tldw_chatbook/UI/Writing_Modules tldw_chatbook/Widgets/Writing Tests/UI/test_writing_screen.py
git commit -m "feat: add writing suite browse ui"
```

### Task 9: Local Editor, Autosave, Versions, and Trash

**Files:**

- Modify: `tldw_chatbook/UI/Writing_Window.py`
- Modify: `tldw_chatbook/UI/Writing_Modules/writing_controller.py`
- Modify: `tldw_chatbook/Widgets/Writing/writing_detail_panel.py`
- Modify: `tldw_chatbook/Widgets/Writing/writing_outline_tree.py`
- Test: `Tests/UI/test_writing_screen.py`
- Test: `Tests/Writing_Interop/test_local_writing_service.py`

- [ ] **Step 1: Write failing local editor tests**

Cover:

- scene editor autosave calls local scene update with Markdown body
- project/manuscript/chapter autosave calls metadata update only
- chapter detail panel has no body editor
- manuscript detail panel has no body editor
- `Create New Version` appears for local manuscript/chapter/scene
- `Create New Version` does not run on project
- version list is read-only
- restoring a local version updates working state and refreshes detail
- local trash list can restore deleted project/manuscript/chapter/scene

- [ ] **Step 2: Run failing tests**

Run:

```bash
python3 -m pytest Tests/UI/test_writing_screen.py Tests/Writing_Interop/test_local_writing_service.py -q
```

Expected: fail until UI hooks exist.

- [ ] **Step 3: Implement detail panel modes**

Detail panel rules:

- project: title/subtitle/author/genre/status/synopsis/target word count/settings summary
- manuscript: title/status/synopsis/sort order, child preview only
- chapter: title/status/synopsis/sort order, ordered scene preview only
- scene: title/status/synopsis/sort order and Markdown editor

Do not add body text fields to manuscript/chapter.

- [ ] **Step 4: Implement autosave**

Autosave should update working state only. Keep debounce simple and testable; a direct save button plus a debounced callback is acceptable. Tests should call the controller method directly rather than relying on wall-clock sleeps.

- [ ] **Step 5: Implement local versions UI**

Expose:

- `Create New Version`
- version list
- read-only version preview
- `Restore To Working Draft`

For manuscript/chapter versions, preview metadata and ordered child membership. For scene versions, preview Markdown body.

- [ ] **Step 6: Implement local trash UI**

Expose a local-only trash view. It should be visibly unavailable in server mode until server restore endpoints exist.

- [ ] **Step 7: Run tests**

Run:

```bash
python3 -m pytest Tests/UI/test_writing_screen.py Tests/Writing_Interop/test_local_writing_service.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Writing_Window.py tldw_chatbook/UI/Writing_Modules/writing_controller.py tldw_chatbook/Widgets/Writing/writing_detail_panel.py tldw_chatbook/Widgets/Writing/writing_outline_tree.py Tests/UI/test_writing_screen.py Tests/Writing_Interop/test_local_writing_service.py
git commit -m "feat: add local writing editor lifecycle"
```

### Task 10: Server Mutation UI and Unsupported Action States

**Files:**

- Modify: `tldw_chatbook/UI/Writing_Window.py`
- Modify: `tldw_chatbook/UI/Writing_Modules/writing_controller.py`
- Modify: `tldw_chatbook/Widgets/Writing/writing_detail_panel.py`
- Modify: `tldw_chatbook/Widgets/Writing/writing_source_panel.py`
- Test: `Tests/UI/test_writing_screen.py`
- Test: `Tests/Writing_Interop/test_writing_scope_service.py`
- Test: `Tests/Writing_Interop/test_server_writing_service.py`

- [ ] **Step 1: Write failing server UI tests**

Cover:

- server project create/update/delete buttons call server scope methods
- server manuscript maps to part create/update/delete
- server chapter create can target manuscript or unassigned bucket
- server chapter assign/unassign uses verified chapter update path
- server scene create is enabled only under a chapter
- server direct manuscript scene button is disabled with reason
- server `Create New Version` is disabled with reason
- server trash restore is disabled with reason
- server scene reparent is disabled with reason

- [ ] **Step 2: Run failing server UI tests**

Run:

```bash
python3 -m pytest Tests/UI/test_writing_screen.py Tests/Writing_Interop/test_writing_scope_service.py Tests/Writing_Interop/test_server_writing_service.py -q
```

Expected: fail until action states are wired.

- [ ] **Step 3: Wire server mutation flows**

Use `WritingScopeService` for all actions. Do not call `ServerWritingService` directly from widgets.

Supported server UI actions:

- project CRUD
- manuscript/part CRUD
- chapter CRUD
- scene CRUD under chapter
- chapter reorder and assign/unassign to manuscript
- scene reorder within current chapter where the server reorder contract is verified

Unsupported server UI actions:

- direct manuscript scene create/move
- server version create/restore
- server trash restore
- scene reparent between chapters

- [ ] **Step 4: Surface unsupported reasons**

Show the `WritingCapability.reason_code` text in a concise notice. Tests should assert the reason code is present in the widget state, not depend on exact styling.

- [ ] **Step 5: Run server UI tests**

Run:

```bash
python3 -m pytest Tests/UI/test_writing_screen.py Tests/Writing_Interop/test_writing_scope_service.py Tests/Writing_Interop/test_server_writing_service.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Writing_Window.py tldw_chatbook/UI/Writing_Modules/writing_controller.py tldw_chatbook/Widgets/Writing/writing_detail_panel.py tldw_chatbook/Widgets/Writing/writing_source_panel.py Tests/UI/test_writing_screen.py Tests/Writing_Interop/test_writing_scope_service.py Tests/Writing_Interop/test_server_writing_service.py
git commit -m "feat: add server writing action states"
```

### Task 11: Reorder, Move, Search, and Edge-Case Hardening

**Files:**

- Modify: `tldw_chatbook/Writing_Interop/local_writing_service.py`
- Modify: `tldw_chatbook/Writing_Interop/server_writing_service.py`
- Modify: `tldw_chatbook/Writing_Interop/writing_scope_service.py`
- Modify: `tldw_chatbook/UI/Writing_Modules/writing_controller.py`
- Modify: `tldw_chatbook/Widgets/Writing/writing_outline_tree.py`
- Test: `Tests/Writing_Interop/test_local_writing_service.py`
- Test: `Tests/Writing_Interop/test_server_writing_service.py`
- Test: `Tests/Writing_Interop/test_writing_scope_service.py`
- Test: `Tests/UI/test_writing_screen.py`

- [ ] **Step 1: Write failing reorder/move/search tests**

Cover:

- local manuscript reorder
- local chapter reorder
- local scene reorder within chapter
- local scene move from chapter to manuscript
- local scene move from manuscript to chapter
- local scene move between chapters
- server chapter reorder maps to `entity_type="chapters"` with optional `new_parent_id`
- server scene reorder within current chapter maps to `entity_type="scenes"`
- server scene reparent is blocked centrally
- search is source-specific and never returns mixed-source results

- [ ] **Step 2: Run failing tests**

Run:

```bash
python3 -m pytest Tests/Writing_Interop/test_local_writing_service.py Tests/Writing_Interop/test_server_writing_service.py Tests/Writing_Interop/test_writing_scope_service.py Tests/UI/test_writing_screen.py -q
```

Expected: fail until reorder/move/search paths are complete.

- [ ] **Step 3: Implement local reorder and move operations**

Use transactions. Recompute only affected parent/order rows. Preserve hierarchy invariants:

- direct scene requires manuscript
- scene under unassigned chapter may have no manuscript
- scene under assigned chapter derives manuscript display from chapter parent

- [ ] **Step 4: Implement server reorder guardrails**

Server reorder support:

- parts/manuscripts: allowed
- chapters within project or reassigned via `new_parent_id`: allowed
- scenes within current chapter: allowed only as reorder, not reparent

Block server scene reparent even if UI sends it.

- [ ] **Step 5: Implement source-specific search**

Local search can start as `LIKE` against title/synopsis/body. Server search uses `/projects/{project_id}/search`. Scope service requires a source and project ID.

- [ ] **Step 6: Run tests**

Run:

```bash
python3 -m pytest Tests/Writing_Interop/test_local_writing_service.py Tests/Writing_Interop/test_server_writing_service.py Tests/Writing_Interop/test_writing_scope_service.py Tests/UI/test_writing_screen.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Writing_Interop/local_writing_service.py tldw_chatbook/Writing_Interop/server_writing_service.py tldw_chatbook/Writing_Interop/writing_scope_service.py tldw_chatbook/UI/Writing_Modules/writing_controller.py tldw_chatbook/Widgets/Writing/writing_outline_tree.py Tests/Writing_Interop/test_local_writing_service.py Tests/Writing_Interop/test_server_writing_service.py Tests/Writing_Interop/test_writing_scope_service.py Tests/UI/test_writing_screen.py
git commit -m "feat: add writing reorder and search"
```

### Task 12: Final Regression, Docs, and Parity Ledger

**Files:**

- Create: `Docs/Development/writing-suite-structural-parity.md`
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-target-state-design.md`
- Test: all focused tests from this plan

- [ ] **Step 1: Run focused Writing Suite tests**

Run:

```bash
python3 -m pytest Tests/tldw_api/test_writing_manuscripts_client.py Tests/Writing_Interop Tests/DB/test_writing_db.py Tests/UI/test_writing_screen.py Tests/UI/test_screen_navigation.py -q
```

Expected: pass.

- [ ] **Step 2: Run adjacent regression tests**

Run:

```bash
python3 -m pytest Tests/tldw_api Tests/RuntimePolicy Tests/UI/test_screen_navigation.py -q
```

Expected: pass or only known unrelated failures documented with exact failing test names and reasons.

- [ ] **Step 3: Run static sanity checks**

Run:

```bash
python3 -m compileall tldw_chatbook/Writing_Interop tldw_chatbook/DB/Writing_DB.py tldw_chatbook/UI/Screens/writing_screen.py tldw_chatbook/UI/Writing_Window.py tldw_chatbook/Widgets/Writing
git diff --check
```

Expected: no syntax errors and no whitespace errors.

- [ ] **Step 4: Update parity ledger**

In `Docs/Parity/2026-04-21-gap-ledger.md`, update Writing Suite from absent to partial/structural parity. State exactly:

- local project/manuscript/chapter/scene CRUD supported
- local versions/trash supported
- server project/part/chapter/scene structural CRUD supported
- server unassigned chapters rendered
- server direct manuscript scenes, versions, trash restore, and scene reparenting remain blocked pending server endpoints

- [ ] **Step 5: Update target-state design**

In `Docs/Parity/2026-04-21-target-state-design.md`, mark the Writing Suite v1 target as implemented for structural authoring and leave generation/export/collaboration/sync as future rows.

- [ ] **Step 6: Add implementation verification doc**

Create `Docs/Development/writing-suite-structural-parity.md` with:

- scope summary
- local/server capability table
- unsupported server capability list
- test commands and results
- manual smoke-test steps

Manual smoke-test checklist:

```text
1. Start Chatbook.
2. Open Writing.
3. In Local mode, create project -> manuscript -> chapter -> scene.
4. Save Markdown in the scene and create a manual version.
5. Delete the scene and restore it from local trash.
6. Switch to Server mode with a configured server.
7. Create project -> manuscript(part) -> chapter -> scene under chapter.
8. Confirm unassigned server chapter appears in Unassigned Chapters.
9. Confirm direct manuscript-level scene and server version buttons are disabled with reason.
10. Switch back to Local mode and confirm local records are unchanged.
```

- [ ] **Step 7: Commit docs and final verification updates**

```bash
git add Docs/Development/writing-suite-structural-parity.md Docs/Parity/2026-04-21-gap-ledger.md Docs/Parity/2026-04-21-target-state-design.md
git commit -m "docs: record writing suite structural parity"
```

## Final Acceptance Criteria

- `Writing` appears as a first-class Chatbook destination.
- A user can create and edit local writing projects, manuscripts, chapters, and scenes while offline.
- Local scene content is Markdown.
- Local manuscript/chapter state is metadata/structure only, not body prose.
- Local versions are manual and immutable until restore.
- Local trash supports restore.
- A user can switch to server mode and operate against `/api/v1/writing/manuscripts` without local fallback or mixed results.
- Server `parts` display as Chatbook manuscripts.
- Server `unassigned_chapters` display as an explicit project-level bucket.
- Server unsupported actions are blocked in `WritingScopeService` and reflected in UI disabled states.
- Focused test suite passes.
- Parity docs honestly reflect implemented support and remaining server-contract gaps.

## Execution Handoff

After this plan is approved, implement it with one of:

1. **Subagent-Driven (recommended)** - dispatch a fresh worker per task, review after each task, integrate in small commits.
2. **Inline Execution** - execute tasks in this session with `superpowers:executing-plans`, keeping the same task boundaries and commits.
