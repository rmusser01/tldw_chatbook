# Prompts And Chatbooks Parity Vertical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add server-compatible prompt and chatbook interoperability to `tldw_chatbook` by extending the existing `tldw_api` client, aligning local prompt storage with `tldw_server`, and consolidating the chatbooks UI onto one primary surface.

**Architecture:** Build on the existing `tldw_chatbook.tldw_api` client instead of creating a parallel networking stack. Extend local prompt storage so structured server prompts can round-trip without losing legacy compatibility, then add thin prompt/chatbook adapter services that the Textual wizards and chatbooks screen can call. Keep live sync, notes/chat parity, and broad Hermes-style UX work out of scope for this vertical.

**Tech Stack:** Python 3, Textual, httpx, SQLite, pytest

**Path Placeholders:** Use `"<path-to-tldw_chatbook-repo>"` for the main repository checkout and `"<path-to-prompts-chatbooks-worktree>"` for the dedicated worktree path used throughout this plan.

---

## File Map

- `tldw_chatbook/tldw_api/prompt_chatbook_schemas.py`: request/response models for prompt CRUD, prompt preview/version restore, chatbook preview/import/export, and chatbook job polling.
- `tldw_chatbook/tldw_api/client.py`: async client methods for the prompt/chatbook endpoints on `tldw_server`.
- `tldw_chatbook/tldw_api/__init__.py`: export the new prompt/chatbook schema/client symbols.
- `tldw_chatbook/DB/Prompts_DB.py`: add structured prompt storage fields while preserving legacy prompt behavior.
- `tldw_chatbook/Prompt_Management/Prompts_Interop.py`: expose the new prompt round-trip helpers to the rest of the app.
- `tldw_chatbook/Prompt_Management/server_prompt_adapter.py`: map local prompt rows to server payloads and server payloads back to local storage.
- `tldw_chatbook/Chatbooks/server_chatbook_service.py`: wrap server preview/import/export/job APIs behind a local service boundary.
- `tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py`: add a server-export path and block unsupported server-only combinations early.
- `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`: add server preview/import mode and explicit unsupported content warnings.
- `tldw_chatbook/UI/Chatbooks_Window_Improved.py`: become the primary chatbooks surface for import/export/manage actions.
- `tldw_chatbook/UI/Screens/chatbooks_screen.py`: mount the improved window instead of the legacy one.
- `Tests/tldw_api/test_prompt_chatbook_schemas.py`: schema coverage for the new request/response models.
- `Tests/tldw_api/test_prompt_chatbook_client.py`: client method and HTTP contract tests using mocks.
- `Tests/Prompts_DB/test_prompts_db_server_parity.py`: migration and round-trip tests for structured prompt storage.
- `Tests/Prompt_Management/test_server_prompt_adapter.py`: adapter tests for prompt payload normalization.
- `Tests/Chatbooks/test_server_chatbook_service.py`: service tests for preview/import/export/job handling.
- `Tests/UI/test_chatbooks_screen_server_actions.py`: UI smoke coverage for the improved chatbooks screen and wizard launch hooks.

## Out Of Scope

- Live sync scheduling, background polling daemons, or offline conflict reconciliation beyond the prompt/chatbook import-export flow.
- Notes, chat, or character parity work.
- A full prompt editor redesign for structured prompt authoring inside every chat surface.
- Hermes-style global job centers or approval frameworks beyond what the chatbooks flow directly needs.

### Task 1: Create The Isolated Worktree And Lock The API Contract

**Files:**
- Create: `Tests/tldw_api/test_prompt_chatbook_schemas.py`
- Create: `Tests/tldw_api/test_prompt_chatbook_client.py`
- Create: `tldw_chatbook/tldw_api/prompt_chatbook_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_prompt_chatbook_schemas.py`
- Test: `Tests/tldw_api/test_prompt_chatbook_client.py`

- [ ] **Step 1: Create the dedicated worktree**

Run:

```bash
git -C "<path-to-tldw_chatbook-repo>" worktree add "<path-to-prompts-chatbooks-worktree>" -b codex-prompts-chatbooks-parity dev
```

Expected: a clean worktree at `"<path-to-prompts-chatbooks-worktree>"`.

- [ ] **Step 2: Write the failing schema tests**

```python
from tldw_chatbook.tldw_api.prompt_chatbook_schemas import (
    PromptCreateRequest,
    PromptPreviewRequest,
    ChatbookExportRequest,
    ChatbookImportRequest,
)


def test_prompt_preview_request_supports_structured_prompts():
    request = PromptPreviewRequest(
        name="prompt",
        prompt_format="structured",
        prompt_schema_version=1,
        prompt_definition={"schema_version": 1, "messages": [{"role": "system", "content": "hi"}]},
    )
    assert request.prompt_format == "structured"
    assert request.prompt_schema_version == 1


def test_chatbook_import_request_exposes_async_and_selection_flags():
    request = ChatbookImportRequest(async_mode=False, import_media=False, import_embeddings=False)
    assert request.async_mode is False
    assert request.import_media is False
```

- [ ] **Step 3: Write the failing client tests**

```python
import pytest
from unittest.mock import AsyncMock

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.prompt_chatbook_schemas import PromptCreateRequest


@pytest.mark.asyncio
async def test_client_create_prompt_posts_to_prompts_endpoint(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"id": 1, "uuid": "abc", "name": "Prompt"})
    monkeypatch.setattr(client, "_request", mocked)

    await client.create_prompt(PromptCreateRequest(name="Prompt", prompt_format="legacy"))

    mocked.assert_awaited_once()
    args, kwargs = mocked.await_args
    assert args[:2] == ("POST", "/api/v1/prompts")
```

- [ ] **Step 4: Run the API contract tests and verify they fail**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/tldw_api/test_prompt_chatbook_schemas.py Tests/tldw_api/test_prompt_chatbook_client.py -q
```

Expected: FAIL on missing imports or missing client methods.

- [ ] **Step 5: Implement the schema module and minimal client methods**

Add schema support for:

```python
class PromptCreateRequest(BaseModel):
    name: str
    prompt_format: Literal["legacy", "structured"] = "legacy"
    prompt_schema_version: int | None = None
    prompt_definition: dict[str, Any] | None = None


class ChatbookExportRequest(BaseModel):
    name: str
    description: str
    content_selections: dict[str, list[str]]
    async_mode: bool = False
```

Add client methods for:

```python
async def list_prompts(self, include_deleted: bool = False) -> dict: ...
async def preview_prompt(self, request_data: PromptPreviewRequest) -> dict: ...
async def create_prompt(self, request_data: PromptCreateRequest) -> dict: ...
async def list_prompt_versions(self, prompt_identifier: str | int) -> dict: ...
async def restore_prompt_version(self, prompt_identifier: str | int, version: int) -> dict: ...
async def preview_chatbook(self, file_path: str) -> dict: ...
async def export_chatbook(self, request_data: ChatbookExportRequest) -> dict: ...
async def import_chatbook(self, file_path: str, request_data: ChatbookImportRequest) -> dict: ...
async def get_chatbook_export_job(self, job_id: str) -> dict: ...
async def get_chatbook_import_job(self, job_id: str) -> dict: ...
```

- [ ] **Step 6: Re-run the API contract tests**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/tldw_api/test_prompt_chatbook_schemas.py Tests/tldw_api/test_prompt_chatbook_client.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the API contract layer**

```bash
cd "<path-to-prompts-chatbooks-worktree>" && git add Tests/tldw_api/test_prompt_chatbook_schemas.py Tests/tldw_api/test_prompt_chatbook_client.py tldw_chatbook/tldw_api/prompt_chatbook_schemas.py tldw_chatbook/tldw_api/client.py tldw_chatbook/tldw_api/__init__.py && git commit -m "feat: add prompt and chatbook API contracts"
```

### Task 2: Upgrade Local Prompt Storage For Structured Server Prompts

**Files:**
- Modify: `tldw_chatbook/DB/Prompts_DB.py`
- Modify: `tldw_chatbook/Prompt_Management/Prompts_Interop.py`
- Create: `Tests/Prompts_DB/test_prompts_db_server_parity.py`
- Test: `Tests/Prompts_DB/test_prompts_db_server_parity.py`

- [ ] **Step 1: Write the failing structured-prompt DB tests**

```python
import json

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase


def test_prompt_db_persists_structured_prompt_metadata():
    db = PromptsDatabase(":memory:", client_id="test-client")
    prompt_id, prompt_uuid, _ = db.add_prompt("Structured", None, None, "legacy system", "legacy user")

    db.update_prompt_by_id(
        prompt_id,
        {
            "prompt_format": "structured",
            "prompt_schema_version": 1,
            "prompt_definition": json.dumps({"schema_version": 1, "messages": [{"role": "system", "content": "hi"}]}),
        },
    )

    prompt = db.fetch_prompt_details(prompt_uuid, include_deleted=True)
    assert prompt["prompt_format"] == "structured"
    assert prompt["prompt_schema_version"] == 1
    assert json.loads(prompt["prompt_definition"])["schema_version"] == 1
```

- [ ] **Step 2: Run the structured-prompt DB test and verify it fails**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/Prompts_DB/test_prompts_db_server_parity.py -q
```

Expected: FAIL on missing columns or missing returned fields.

- [ ] **Step 3: Add the new prompt fields and migration path**

Implement the minimum schema change needed to support server parity:

```python
prompt_format TEXT NOT NULL DEFAULT 'legacy'
prompt_schema_version INTEGER
prompt_definition TEXT
```

Also update:

- `add_prompt`
- `update_prompt_by_id`
- `get_prompt_by_id`
- `get_prompt_by_uuid`
- `fetch_prompt_details`
- export helpers that should preserve the new fields

Preserve `system_prompt` and `user_prompt` for legacy compatibility.

- [ ] **Step 4: Re-run the DB parity test**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/Prompts_DB/test_prompts_db_server_parity.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the prompt DB migration**

```bash
cd "<path-to-prompts-chatbooks-worktree>" && git add Tests/Prompts_DB/test_prompts_db_server_parity.py tldw_chatbook/DB/Prompts_DB.py tldw_chatbook/Prompt_Management/Prompts_Interop.py && git commit -m "feat: store structured prompt metadata locally"
```

### Task 3: Add Prompt Round-Trip Adapters And Interop Helpers

**Files:**
- Create: `tldw_chatbook/Prompt_Management/server_prompt_adapter.py`
- Modify: `tldw_chatbook/Prompt_Management/Prompts_Interop.py`
- Create: `Tests/Prompt_Management/test_server_prompt_adapter.py`
- Test: `Tests/Prompt_Management/test_server_prompt_adapter.py`

- [ ] **Step 1: Write the failing adapter tests**

```python
from tldw_chatbook.Prompt_Management.server_prompt_adapter import (
    local_prompt_to_server_payload,
    server_prompt_to_local_update,
)


def test_server_prompt_to_local_update_preserves_structured_fields():
    payload = {
        "name": "Server Prompt",
        "prompt_format": "structured",
        "prompt_schema_version": 1,
        "prompt_definition": {"schema_version": 1, "messages": [{"role": "user", "content": "hi"}]},
    }
    update = server_prompt_to_local_update(payload)
    assert update["prompt_format"] == "structured"
    assert update["prompt_schema_version"] == 1


def test_local_prompt_to_server_payload_keeps_legacy_snapshot():
    local_prompt = {
        "name": "Legacy Prompt",
        "prompt_format": "legacy",
        "system_prompt": "system",
        "user_prompt": "user",
    }
    payload = local_prompt_to_server_payload(local_prompt)
    assert payload["prompt_format"] == "legacy"
```

- [ ] **Step 2: Run the adapter tests and verify they fail**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/Prompt_Management/test_server_prompt_adapter.py -q
```

Expected: FAIL on missing adapter module/functions.

- [ ] **Step 3: Implement the adapter and interop helpers**

Add helpers for:

- server payload to local DB update dict
- local prompt row to server create/update payload
- preview request assembly
- version-restore application back into local storage

Expose minimal interop entry points from `Prompts_Interop.py`, for example:

```python
def import_prompt_from_server_payload(payload: dict) -> dict: ...
def export_prompt_to_server_payload(prompt_id_or_uuid: int | str) -> dict: ...
def apply_server_prompt_version(prompt_id_or_uuid: int | str, payload: dict) -> dict: ...
```

- [ ] **Step 4: Re-run the adapter tests**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/Prompt_Management/test_server_prompt_adapter.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the prompt adapter layer**

```bash
cd "<path-to-prompts-chatbooks-worktree>" && git add Tests/Prompt_Management/test_server_prompt_adapter.py tldw_chatbook/Prompt_Management/server_prompt_adapter.py tldw_chatbook/Prompt_Management/Prompts_Interop.py && git commit -m "feat: add server prompt round-trip adapters"
```

### Task 4: Add The Server Chatbook Service And Wizard Modes

**Files:**
- Create: `tldw_chatbook/Chatbooks/server_chatbook_service.py`
- Modify: `tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py`
- Modify: `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`
- Create: `Tests/Chatbooks/test_server_chatbook_service.py`
- Test: `Tests/Chatbooks/test_server_chatbook_service.py`

- [ ] **Step 1: Write the failing chatbook service tests**

```python
from tldw_chatbook.Chatbooks.server_chatbook_service import ServerChatbookService


def test_service_rejects_server_unsupported_import_content_types():
    service = ServerChatbookService(client=None)
    unsupported = service.validate_server_import_selection({"prompt": ["1"], "note": ["2"]})
    assert "prompt" in unsupported


def test_service_normalizes_export_selection_keys():
    service = ServerChatbookService(client=None)
    payload = service.build_export_request_payload(
        name="Pack",
        description="Desc",
        selections={"conversation": ["1"], "prompt": ["2"]},
    )
    assert payload.content_selections["conversation"] == ["1"]
```

- [ ] **Step 2: Run the chatbook service tests and verify they fail**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/Chatbooks/test_server_chatbook_service.py -q
```

Expected: FAIL on missing service module or methods.

- [ ] **Step 3: Implement the service and wire the wizards**

The service should wrap:

- preview
- export
- export continuation
- import
- import/export job polling
- validation for currently unsupported server import content types

The wizards should:

- offer `local` and `server` execution modes
- default to `local`
- reuse existing local importer/creator behavior unchanged when `local`
- disable or clearly warn on server-unsupported import selections such as prompt/media/embedding/evaluation content when the server API rejects them

- [ ] **Step 4: Re-run the service tests**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/Chatbooks/test_server_chatbook_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the chatbook service and wizard wiring**

```bash
cd "<path-to-prompts-chatbooks-worktree>" && git add Tests/Chatbooks/test_server_chatbook_service.py tldw_chatbook/Chatbooks/server_chatbook_service.py tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py tldw_chatbook/UI/Wizards/ChatbookImportWizard.py && git commit -m "feat: add server-backed chatbook import export flows"
```

### Task 5: Consolidate The Chatbooks Screen On The Improved UI

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chatbooks_screen.py`
- Modify: `tldw_chatbook/UI/Chatbooks_Window_Improved.py`
- Create: `Tests/UI/test_chatbooks_screen_server_actions.py`
- Test: `Tests/UI/test_chatbooks_screen_server_actions.py`

- [ ] **Step 1: Write the failing chatbooks UI test**

```python
import pytest
from textual.app import App
from unittest.mock import Mock

from tldw_chatbook.UI.Screens.chatbooks_screen import ChatbooksScreen
from tldw_chatbook.UI.Chatbooks_Window_Improved import ChatbooksWindowImproved


@pytest.fixture
def mock_app_instance():
    app = Mock()
    app.notify = Mock()
    return app


class ChatbooksTestApp(App):
    def __init__(self, app_instance):
        super().__init__()
        self._app_instance = app_instance

    def on_mount(self):
        self.push_screen(ChatbooksScreen(self._app_instance))


@pytest.mark.asyncio
async def test_chatbooks_screen_uses_improved_window(mock_app_instance):
    app = ChatbooksTestApp(mock_app_instance)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.screen.query_one(ChatbooksWindowImproved) is not None
```

- [ ] **Step 2: Run the chatbooks UI test and verify it fails**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/UI/test_chatbooks_screen_server_actions.py -q
```

Expected: FAIL because `ChatbooksScreen` still mounts the legacy `ChatbooksWindow`.

- [ ] **Step 3: Make the improved window the primary screen surface**

Update `chatbooks_screen.py` so it mounts `ChatbooksWindowImproved`, and update `Chatbooks_Window_Improved.py` so its action hooks can:

- launch the local/server-aware import wizard
- launch the local/server-aware creation wizard
- refresh after job completion
- expose a minimal manage/jobs view instead of a placeholder action

Do not remove `ChatbooksWindow.py` in this vertical. Leave it as a legacy shell until follow-on cleanup work.

- [ ] **Step 4: Re-run the chatbooks UI test**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/UI/test_chatbooks_screen_server_actions.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the chatbooks UI consolidation**

```bash
cd "<path-to-prompts-chatbooks-worktree>" && git add Tests/UI/test_chatbooks_screen_server_actions.py tldw_chatbook/UI/Screens/chatbooks_screen.py tldw_chatbook/UI/Chatbooks_Window_Improved.py && git commit -m "feat: consolidate chatbooks screen on improved UI"
```

### Task 6: Verify The Vertical End-To-End And Update Docs

**Files:**
- Modify: `tldw_chatbook/tldw_api/README.md`
- Modify: `Docs/Parity/2026-04-19-data-compatibility-map.md`
- Modify: `Docs/Parity/2026-04-19-rollout-backlog.md`
- Test: `Tests/tldw_api/test_prompt_chatbook_schemas.py`
- Test: `Tests/tldw_api/test_prompt_chatbook_client.py`
- Test: `Tests/Prompts_DB/test_prompts_db_server_parity.py`
- Test: `Tests/Prompt_Management/test_server_prompt_adapter.py`
- Test: `Tests/Chatbooks/test_server_chatbook_service.py`
- Test: `Tests/UI/test_chatbooks_screen_server_actions.py`

- [ ] **Step 1: Update the local API docs and parity artifacts**

Document:

- the new prompt/chatbook client methods
- the new structured prompt storage fields
- the chatbooks UI consolidation decision
- any still-unsupported server import content types

- [ ] **Step 2: Run the targeted verification suite**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/tldw_api/test_prompt_chatbook_schemas.py Tests/tldw_api/test_prompt_chatbook_client.py Tests/Prompts_DB/test_prompts_db_server_parity.py Tests/Prompt_Management/test_server_prompt_adapter.py Tests/Chatbooks/test_server_chatbook_service.py Tests/UI/test_chatbooks_screen_server_actions.py -q
```

Expected: PASS.

- [ ] **Step 3: Run one broader regression pass that touches existing prompt/chatbook code**

Run:

```bash
cd "<path-to-prompts-chatbooks-worktree>" && pytest Tests/Chatbooks/test_chatbook_importer.py Tests/Chatbooks/test_chatbook_creator.py Tests/Prompts_DB/test_prompts_db_pytest.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the verified vertical**

```bash
cd "<path-to-prompts-chatbooks-worktree>" && git add tldw_chatbook/tldw_api/README.md Docs/Parity/2026-04-19-data-compatibility-map.md Docs/Parity/2026-04-19-rollout-backlog.md && git commit -m "docs: finalize prompts and chatbooks parity vertical"
```
