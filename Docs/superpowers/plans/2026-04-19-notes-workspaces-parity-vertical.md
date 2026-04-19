# Notes And Workspaces Parity Vertical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live server-backed user-space note CRUD and workspace CRUD to the existing notes screen while preserving local note behavior and strict separation between user-space notes and workspace-contained notes.

**Architecture:** Extend the shared `tldw_api` client with note/workspace/media-picker contracts, add a new scope-aware server notes/workspaces service, then refactor the notes screen into an explicit local/server/workspace navigator with guarded scope transitions. Keep the current local notes stack intact by treating the server layer as additive, not as a replacement for `NotesInteropService`.

**Tech Stack:** Python, Textual, Pydantic, `httpx`, pytest

---

## File Map

- Create: `tldw_chatbook/tldw_api/notes_workspace_schemas.py`
  Responsibility: Pydantic request/response models for user-space notes, workspaces, workspace notes, workspace sources, workspace artifacts, and media-picker list/search payloads.
- Modify: `tldw_chatbook/tldw_api/client.py`
  Responsibility: Add note/workspace CRUD, workspace context loading, and media list/search methods.
- Modify: `tldw_chatbook/tldw_api/__init__.py`
  Responsibility: Export the new notes/workspace client models.
- Create: `tests/tldw_api/test_notes_workspace_client.py`
  Responsibility: Verify endpoint wiring and payload serialization for the new client methods.

- Create: `tldw_chatbook/Notes/server_notes_workspace_service.py`
  Responsibility: Normalize server payloads for the TUI, keep user-space notes and workspace resources separate, and expose a single async service boundary for server-backed notes/workspaces.
- Create: `tldw_chatbook/Notes/notes_scope_service.py`
  Responsibility: Route screen actions to either the existing local notes service or the new server notes/workspaces service based on active scope without leaking scope rules into the UI.
- Create: `tests/Notes/test_server_notes_workspace_service.py`
  Responsibility: Verify request normalization, workspace search isolation, optimistic-locking field handling, and workspace-source create payload construction.
- Create: `tests/Notes/test_notes_scope_service.py`
  Responsibility: Verify local/server/workspace routing rules for screen-facing actions.

- Create: `tldw_chatbook/UI/Screens/notes_scope_models.py`
  Responsibility: Hold `ScopeType`, `WorkspaceSubview`, editor selection metadata, and pending-navigation/dirty-state helpers so `notes_screen.py` does not become a larger catch-all.
- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
  Responsibility: Replace the single-note state model with scope-aware state, own all notes-screen button handling, and route actions through local or server services by scope.
- Modify: `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_left.py`
  Responsibility: Render separate `Local Notes`, `Server Notes`, and `Workspaces` navigator sections plus scope-aware search.
- Modify: `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py`
  Responsibility: Render scope-aware note actions and hide local-only controls outside local note scope.
- Create: `tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py`
  Responsibility: Render `Workspace Details`, `Workspace Notes`, `Sources`, and `Artifacts` subviews for a selected workspace.
- Create: `tldw_chatbook/Widgets/Note_Widgets/workspace_source_picker.py`
  Responsibility: Provide a bounded modal/picker that can list/search media and return a valid `media_id` for workspace source creation.

- Modify: `tldw_chatbook/Event_Handlers/tab_initializers/notes_tab_initializer.py`
  Responsibility: Stop calling the legacy local-only notes loader for the new notes screen and instead trigger scope-aware refresh methods on the active screen.
- Modify: `tldw_chatbook/app.py`
  Responsibility: Initialize and expose the new scope-aware server service alongside the existing local notes service while keeping legacy chat-note flows intact.

- Modify: `tests/UI/test_notes_screen.py`
  Responsibility: Cover mixed navigator rendering, dirty-navigation guard behavior, local-only sync visibility, server note selection, workspace search isolation, and delete warnings.
- Create: `tests/Widgets/test_workspace_context_panel.py`
  Responsibility: Verify workspace subview rendering and destructive-action labels.
- Create: `tests/Widgets/test_workspace_source_picker.py`
  Responsibility: Verify the picker loads media rows, filters results, and returns the selected `media_id`.

## Task 1: Add Shared API Contracts And Endpoint Wiring

**Files:**
- Create: `tldw_chatbook/tldw_api/notes_workspace_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `tests/tldw_api/test_notes_workspace_client.py`

- [ ] **Step 1: Write the failing client tests**

```python
@pytest.mark.asyncio
async def test_list_server_notes_hits_notes_endpoint(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"notes": [], "count": 0})
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_server_notes(limit=25, offset=0)

    args, kwargs = mocked.await_args
    assert args[:2] == ("GET", "/api/v1/notes/")
    assert kwargs["params"] == {"limit": 25, "offset": 0, "include_keywords": "true"}


@pytest.mark.asyncio
async def test_search_media_items_posts_to_media_search(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"items": [], "total_items": 0})
    monkeypatch.setattr(client, "_request", mocked)

    await client.search_media_items(MediaSearchRequest(query="paper"))

    args, kwargs = mocked.await_args
    assert args[:2] == ("POST", "/api/v1/media/search")
```

- [ ] **Step 2: Run the client tests to verify they fail**

Run: `python -m pytest tests/tldw_api/test_notes_workspace_client.py -q`
Expected: FAIL with missing client methods and missing schema imports.

- [ ] **Step 3: Implement the minimal client/schema layer**

```python
class WorkspaceNoteUpdateRequest(BaseModel):
    title: str | None = None
    content: str | None = None
    keywords_json: str | None = None
    version: int


async def list_server_notes(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    return await self._request(
        "GET",
        "/api/v1/notes/",
        params={"limit": limit, "offset": offset, "include_keywords": "true"},
    )


async def search_media_items(self, request_data: MediaSearchRequest, page: int = 1, results_per_page: int = 20) -> dict[str, Any]:
    return await self._request(
        "POST",
        "/api/v1/media/search",
        json_data=request_data.model_dump(exclude_none=True),
        params={"page": page, "results_per_page": results_per_page},
    )
```

Required client methods in this task:

- `list_server_notes`
- `search_server_notes`
- `get_server_note`
- `create_server_note`
- `update_server_note`
- `delete_server_note`
- `list_workspaces`
- `get_workspace`
- `create_workspace`
- `update_workspace`
- `delete_workspace`
- `list_workspace_notes`
- `create_workspace_note`
- `update_workspace_note`
- `delete_workspace_note`
- `list_workspace_sources`
- `create_workspace_source`
- `update_workspace_source`
- `delete_workspace_source`
- `list_workspace_artifacts`
- `create_workspace_artifact`
- `update_workspace_artifact`
- `delete_workspace_artifact`
- `list_media_items`
- `search_media_items`

- [ ] **Step 4: Run the client tests again**

Run: `python -m pytest tests/tldw_api/test_notes_workspace_client.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/notes_workspace_schemas.py tldw_chatbook/tldw_api/client.py tldw_chatbook/tldw_api/__init__.py tests/tldw_api/test_notes_workspace_client.py
git commit -m "feat: add notes and workspace api client support"
```

## Task 2: Build The Server Notes And Workspaces Service

**Files:**
- Create: `tldw_chatbook/Notes/server_notes_workspace_service.py`
- Create: `tldw_chatbook/Notes/notes_scope_service.py`
- Test: `tests/Notes/test_server_notes_workspace_service.py`
- Test: `tests/Notes/test_notes_scope_service.py`

- [ ] **Step 1: Write the failing service tests**

```python
@pytest.mark.asyncio
async def test_service_serializes_workspace_note_keywords_for_update():
    service = ServerNotesWorkspaceService(client=FakeClient())

    payload = service.build_workspace_note_update_payload(
        title="Draft",
        content="Body",
        keywords=["alpha", "beta"],
        version=3,
    )

    assert payload.keywords_json == '["alpha", "beta"]'


def test_service_filters_workspace_notes_within_active_workspace_only():
    service = ServerNotesWorkspaceService(client=None)
    notes = [
        {"id": 1, "workspace_id": "ws-1", "title": "Alpha", "content": "One"},
        {"id": 2, "workspace_id": "ws-2", "title": "Alpha", "content": "Two"},
    ]

    filtered = service.filter_workspace_notes(notes, workspace_id="ws-1", query="alpha")
    assert [note["id"] for note in filtered] == [1]


@pytest.mark.asyncio
async def test_scope_service_routes_server_note_save_to_server_service():
    scope_service = NotesScopeService(local_notes_service=FakeLocalNotes(), server_service=FakeServerNotes())

    await scope_service.save_note(
        scope=ScopeType.SERVER_NOTE,
        note_id="note-1",
        title="Remote",
        content="Body",
        version=2,
    )

    assert scope_service.server_service.saved_ids == ["note-1"]
```

- [ ] **Step 2: Run the service tests to verify they fail**

Run: `python -m pytest tests/Notes/test_server_notes_workspace_service.py tests/Notes/test_notes_scope_service.py -q`
Expected: FAIL with missing service class and helper methods.

- [ ] **Step 3: Implement the minimal scope-aware server service**

```python
class ServerNotesWorkspaceService:
    def __init__(self, client: TLDWAPIClient | None):
        self.client = client

    def build_workspace_note_update_payload(self, *, title, content, keywords, version):
        return WorkspaceNoteUpdateRequest(
            title=title,
            content=content,
            keywords_json=json.dumps([kw for kw in keywords if kw]),
            version=version,
        )

    def filter_workspace_notes(self, notes, *, workspace_id: str, query: str) -> list[dict[str, Any]]:
        lowered = query.strip().lower()
        return [
            note for note in notes
            if note.get("workspace_id") == workspace_id
            and (not lowered or lowered in f"{note.get('title', '')} {note.get('content', '')}".lower())
        ]


class NotesScopeService:
    def __init__(self, local_notes_service, server_service):
        self.local_notes_service = local_notes_service
        self.server_service = server_service
```

Required behavior in this task:

- Normalize user-space note payloads into a consistent editor shape.
- Normalize workspace note payloads separately from user-space notes.
- Keep user-space note search API-backed and workspace note search client-side.
- Build valid workspace-source create payloads that include `media_id`.
- Surface `version` fields on all editable server resources.
- Provide a `from_config` or equivalent constructor that can build a server client from `app_config["tldw_api"]`.
- Provide screen-facing methods like `save_note`, `delete_note`, `search_notes`, and `load_workspace_context` that route by scope.

- [ ] **Step 4: Run the service tests again**

Run: `python -m pytest tests/Notes/test_server_notes_workspace_service.py tests/Notes/test_notes_scope_service.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Notes/server_notes_workspace_service.py tldw_chatbook/Notes/notes_scope_service.py tests/Notes/test_server_notes_workspace_service.py tests/Notes/test_notes_scope_service.py
git commit -m "feat: add notes scope services"
```

## Task 3: Introduce Scope-Aware State And App Wiring

**Files:**
- Create: `tldw_chatbook/UI/Screens/notes_scope_models.py`
- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
- Modify: `tldw_chatbook/Event_Handlers/tab_initializers/notes_tab_initializer.py`
- Modify: `tldw_chatbook/app.py`
- Test: `tests/UI/test_notes_screen.py`

- [ ] **Step 1: Write the failing state/navigation tests**

```python
def test_scope_state_defaults():
    state = NotesScreenState()
    assert state.scope_type == ScopeType.LOCAL_NOTE
    assert state.selected_server_note_id is None
    assert state.selected_workspace_id is None


@pytest.mark.asyncio
async def test_switching_scope_with_unsaved_changes_requires_decision(mock_app_instance):
    app = NotesTestApp(notes_service=mock_app_instance.notes_service)
    async with app.run_test() as pilot:
        screen = pilot.app.screen
        screen.state.has_unsaved_changes = True

        blocked = screen.request_scope_transition(ScopeType.SERVER_NOTE, target_id="note-2")
        assert blocked.requires_confirmation is True
```

- [ ] **Step 2: Run the screen tests to verify they fail**

Run: `python -m pytest tests/UI/test_notes_screen.py -q`
Expected: FAIL with missing scope models and missing transition guard behavior.

- [ ] **Step 3: Implement the new state model and app wiring**

```python
class ScopeType(str, Enum):
    LOCAL_NOTE = "local_note"
    SERVER_NOTE = "server_note"
    WORKSPACE = "workspace"


@dataclass
class PendingNavigation:
    target_scope: ScopeType
    target_id: str | int | None = None
    requires_confirmation: bool = False


self.notes_scope_service = NotesScopeService(
    local_notes_service=self.notes_service,
    server_service=ServerNotesWorkspaceService.from_config(self.app_config),
)
```

Required behavior in this task:

- Keep `app.py` legacy `current_selected_note_*` reactives for chat-sidebar/local consumers only.
- Stop using `notes_tab_initializer.py` as a local-notes loader for the new screen; make it call a `refresh_current_scope()` method on `NotesScreen`.
- Move notes-screen-specific save/delete/export/copy handling into `notes_screen.py` so the screen owns scope-aware behavior instead of depending on legacy global handler assumptions.

- [ ] **Step 4: Run the screen tests again**

Run: `python -m pytest tests/UI/test_notes_screen.py -q`
Expected: PASS for the new state and guarded navigation cases created in this task.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/notes_scope_models.py tldw_chatbook/UI/Screens/notes_screen.py tldw_chatbook/Event_Handlers/tab_initializers/notes_tab_initializer.py tldw_chatbook/app.py tests/UI/test_notes_screen.py
git commit -m "feat: add scope aware notes screen state"
```

## Task 4: Build The Mixed Navigator And Workspace Widgets

**Files:**
- Modify: `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_left.py`
- Modify: `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py`
- Create: `tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py`
- Create: `tldw_chatbook/Widgets/Note_Widgets/workspace_source_picker.py`
- Test: `tests/Widgets/test_workspace_context_panel.py`
- Test: `tests/Widgets/test_workspace_source_picker.py`

- [ ] **Step 1: Write the failing widget tests**

```python
async def test_workspace_context_panel_shows_four_subviews():
    panel = WorkspaceContextPanel()
    assert list(panel.SUBVIEWS) == [
        "workspace-details",
        "workspace-notes",
        "workspace-sources",
        "workspace-artifacts",
    ]


@pytest.mark.asyncio
async def test_workspace_source_picker_returns_selected_media_id():
    picker = WorkspaceSourcePicker(results=[{"id": 42, "title": "Paper", "type": "pdf"}])
    picker.select_result(42)
    assert picker.selected_media_id == 42
```

- [ ] **Step 2: Run the widget tests to verify they fail**

Run: `python -m pytest tests/Widgets/test_workspace_context_panel.py tests/Widgets/test_workspace_source_picker.py -q`
Expected: FAIL with missing widget modules/classes.

- [ ] **Step 3: Implement the minimal widgets**

```python
class WorkspaceContextPanel(Static):
    SUBVIEWS = ("workspace-details", "workspace-notes", "workspace-sources", "workspace-artifacts")


class WorkspaceSourcePicker(ModalScreen[int | None]):
    def __init__(self, service: ServerNotesWorkspaceService, results: list[dict[str, Any]] | None = None):
        self.service = service
        self.results = results or []
        self.selected_media_id: int | None = None
```

Required behavior in this task:

- `notes_sidebar_left.py` must render three separate navigator sections and never merge workspace notes into general note lists.
- `notes_sidebar_right.py` must relabel or hide actions by scope.
- `WorkspaceContextPanel` must contain workspace details plus scoped lists for notes, sources, and artifacts.
- `WorkspaceSourcePicker` must list/search media and return a valid `media_id` for source creation.
- `WorkspaceSourcePicker` must stay bounded: no new full media-management workflow.

- [ ] **Step 4: Run the widget tests again**

Run: `python -m pytest tests/Widgets/test_workspace_context_panel.py tests/Widgets/test_workspace_source_picker.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_left.py tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py tldw_chatbook/Widgets/Note_Widgets/workspace_source_picker.py tests/Widgets/test_workspace_context_panel.py tests/Widgets/test_workspace_source_picker.py
git commit -m "feat: add workspace notes ui components"
```

## Task 5: Integrate Scope-Aware UI Behavior And Finish Regression Coverage

**Files:**
- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
- Modify: `tests/UI/test_notes_screen.py`

- [ ] **Step 1: Write the failing integration/regression tests**

```python
@pytest.mark.asyncio
async def test_local_sync_button_hidden_for_server_scope(mock_app_instance):
    app = NotesTestApp(notes_service=mock_app_instance.notes_service)
    async with app.run_test() as pilot:
        screen = pilot.app.screen
        screen.state.scope_type = ScopeType.SERVER_NOTE
        await pilot.pause()
        assert screen.query_one("#notes-sync-button").display is False


@pytest.mark.asyncio
async def test_workspace_delete_warning_mentions_conversation_cascade(mock_app_instance):
    app = NotesTestApp(notes_service=mock_app_instance.notes_service)
    async with app.run_test() as pilot:
        screen = pilot.app.screen
        warning = screen.build_workspace_delete_warning("Research")
        assert "soft-deletes related workspace conversations" in warning
```

- [ ] **Step 2: Run the focused integration suite to verify it fails**

Run: `python -m pytest tests/UI/test_notes_screen.py -q`
Expected: FAIL with missing scope-aware visibility and workspace delete warning behavior.

- [ ] **Step 3: Finish the screen integration**

```python
if self.state.scope_type != ScopeType.LOCAL_NOTE:
    self.query_one("#notes-sync-button", Button).display = False

if self.state.scope_type == ScopeType.WORKSPACE and self.state.workspace_subview == WorkspaceSubview.DETAILS:
    self.mount_or_show_workspace_context()
else:
    self.mount_or_show_note_editor()
```

Required behavior in this task:

- Scope labels and active controls must change with selection.
- User-space server note search must use the server endpoint.
- Workspace note search must filter only the selected workspace note collection already loaded in memory.
- Workspace delete confirmation must mention conversation cascade.
- Export/copy actions must work only for note editors.
- The notes screen must not silently discard unsaved changes during note/scope switches.

- [ ] **Step 4: Run the full focused parity suite**

Run: `python -m pytest tests/tldw_api/test_notes_workspace_client.py tests/Notes/test_server_notes_workspace_service.py tests/Notes/test_notes_scope_service.py tests/UI/test_notes_screen.py tests/Widgets/test_workspace_context_panel.py tests/Widgets/test_workspace_source_picker.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/notes_screen.py tests/UI/test_notes_screen.py
git commit -m "feat: complete notes and workspace parity vertical"
```

## Execution Notes

- Do not broaden this vertical into sync, local mirroring, import/export, notes graph, or cross-scope note moves.
- Do not let workspace notes appear in user-space note search results or note lists at any point.
- Prefer additive UI changes over rewriting the entire notes stack.
- If a code path only exists for the legacy local notes flow outside the main notes screen, leave it alone unless it blocks parity work directly.
