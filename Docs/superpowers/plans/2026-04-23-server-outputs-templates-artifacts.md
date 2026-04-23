# Server Outputs Templates Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first authoritative `Collections: Outputs / Templates / Artifacts` parity slice to Chatbook with typed server client coverage, a runtime-policy-aware server scope seam, and a source-aware `Outputs` control panel in the existing TUI.

**Architecture:** Keep this slice source-honest and minimal. Server output templates and output artifacts are exposed through typed client methods plus a thin server service and a scope service that explicitly rejects local mode for now. Mount the first UI surface in `Tools & Settings` beside the other remote-first control panes so the contract is operable without inventing local output parity or a new top-level destination.

**Tech Stack:** Python, Textual, Pydantic, existing `TLDWAPIClient`, existing runtime policy registry/app wiring, pytest.

---

### Task 1: Add Typed Outputs Client Coverage

**Files:**
- Create: `tldw_chatbook/tldw_api/outputs_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_outputs_client.py`

- [ ] **Step 1: Write the failing client contract tests**

```python
@pytest.mark.asyncio
async def test_client_routes_output_template_and_artifact_calls():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(side_effect=[...])

    listed_templates = await client.list_output_templates(q="brief", limit=25, offset=5)
    created_template = await client.create_output_template(
        OutputTemplateCreateRequest(...)
    )
    preview = await client.preview_output_template(
        7,
        OutputTemplatePreviewRequest(template_id=7, item_ids=[1, 2]),
    )
    listed_outputs = await client.list_outputs(type="newsletter_markdown", workspace_tag="workspace:demo")
    created_output = await client.create_output(OutputCreateRequest(...))
    updated_output = await client.update_output(11, OutputUpdateRequest(title="Renamed"))
    deleted_output = await client.delete_output(11, hard=True, delete_file=True)

    assert listed_templates.total == 1
    assert created_output.id == 11
    assert client._request.await_args_list[0].args == ("GET", "/api/v1/outputs/templates")
```

- [ ] **Step 2: Run the client tests to verify they fail**

Run: `python3 -m pytest Tests/tldw_api/test_outputs_client.py -q`

Expected: FAIL because outputs schemas and client methods do not exist yet.

- [ ] **Step 3: Add the minimal typed schemas and client methods**

```python
class OutputTemplateCreateRequest(BaseModel):
    name: str
    type: Literal["newsletter_markdown", "briefing_markdown", "mece_markdown", "newsletter_html", "tts_audio"]
    format: Literal["md", "html", "mp3"]
    body: str
    description: str | None = None
    is_default: bool = False
    metadata: dict[str, Any] | None = None


async def list_output_templates(self, *, q: str | None = None, limit: int = 50, offset: int = 0) -> OutputTemplateListResponse:
    payload = await self._request("GET", "/api/v1/outputs/templates", params={...})
    return OutputTemplateListResponse.model_validate(payload)
```

- [ ] **Step 4: Run the client tests again**

Run: `python3 -m pytest Tests/tldw_api/test_outputs_client.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/outputs_schemas.py tldw_chatbook/tldw_api/client.py tldw_chatbook/tldw_api/__init__.py Tests/tldw_api/test_outputs_client.py
git commit -m "feat: add outputs api client coverage"
```

### Task 2: Add Server Outputs Service and Runtime-Policy Scope Seam

**Files:**
- Create: `tldw_chatbook/Outputs/__init__.py`
- Create: `tldw_chatbook/Outputs/server_outputs_service.py`
- Create: `tldw_chatbook/Outputs/server_outputs_scope_service.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/Outputs/test_server_outputs_service.py`
- Test: `Tests/RuntimePolicy/test_runtime_policy_core.py`

- [ ] **Step 1: Write the failing service and scope tests**

```python
@pytest.mark.asyncio
async def test_server_outputs_service_routes_typed_client_calls():
    client = FakeClient()
    service = ServerOutputsService(client=client)

    templates = await service.list_output_templates(q="brief")
    preview = await service.preview_output_template(7, item_ids=[1, 2])
    outputs = await service.list_outputs(workspace_tag="workspace:demo")
    deleted = await service.delete_output(11, hard=True, delete_file=True)

    assert templates["total"] == 1
    assert deleted["success"] is True


@pytest.mark.asyncio
async def test_outputs_scope_service_rejects_local_mode_and_normalizes_server_ids():
    scope = ServerOutputsScopeService(server_service=ServerOutputsService(client=FakeClient()), policy_enforcer=FakePolicyEnforcer())

    listed = await scope.list_output_templates(mode="server")
    created = await scope.create_output(mode="server", template_id=7, item_ids=[1])

    assert listed["items"][0]["id"] == "server:output_template:7"
    assert created["id"] == "server:output:11"

    with pytest.raises(ValueError, match="Server outputs require server mode"):
        await scope.list_output_templates(mode="local")
```

- [ ] **Step 2: Run the service/scope tests to verify they fail**

Run: `python3 -m pytest Tests/Outputs/test_server_outputs_service.py Tests/RuntimePolicy/test_runtime_policy_core.py -q`

Expected: FAIL because the outputs service and scope seam are not implemented or wired.

- [ ] **Step 3: Implement the minimal service, scope seam, and app bootstrap wiring**

```python
class ServerOutputsScopeService:
    async def list_output_templates(self, *, mode=None, **kwargs):
        self._require_server_mode(mode)
        self._enforce_policy("outputs.templates.list.server")
        payload = await self._maybe_await(self.server_service.list_output_templates(**kwargs))
        return {"items": [self._normalize_template(item) for item in payload.get("items", [])], "total": payload.get("total", 0)}
```

- [ ] **Step 4: Run the service/scope tests again**

Run: `python3 -m pytest Tests/Outputs/test_server_outputs_service.py Tests/RuntimePolicy/test_runtime_policy_core.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Outputs/__init__.py tldw_chatbook/Outputs/server_outputs_service.py tldw_chatbook/Outputs/server_outputs_scope_service.py tldw_chatbook/app.py Tests/Outputs/test_server_outputs_service.py Tests/RuntimePolicy/test_runtime_policy_core.py
git commit -m "feat: add server outputs scope service"
```

### Task 3: Add the Source-Aware Outputs Panel

**Files:**
- Create: `tldw_chatbook/UI/Outputs_Panel.py`
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py`
- Modify: `Tests/UI/test_tools_settings_window.py`
- Modify: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write the failing UI tests**

```python
@pytest.mark.asyncio
async def test_tools_settings_window_mounts_outputs_panel(settings_window):
    nav_button = settings_window.query_one("#ts-nav-outputs", Button)
    assert nav_button.label.plain == "Outputs"
    panel = settings_window.query_one("#outputs-panel", OutputsPanel)
    assert panel is not None


@pytest.mark.asyncio
async def test_outputs_panel_disables_controls_in_local_mode():
    panel = OutputsPanel(app_instance=mock_app_instance)
    ...
    assert panel.query_one("#outputs-disabled", Static).display is True
```

- [ ] **Step 2: Run the UI tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_tools_settings_window.py Tests/UI/test_screen_navigation.py -q`

Expected: FAIL because the outputs panel and navigation entry do not exist.

- [ ] **Step 3: Implement the minimal outputs control panel**

```python
class OutputsPanel(ScrollableContainer):
    """First-slice server outputs control plane."""

    async def refresh_for_mode(self) -> None:
        self.runtime_backend = self._current_runtime_backend()
        enabled = self.runtime_backend == "server" and self.scope_service is not None
        self._show_server_ui(enabled)
        self._set_controls_disabled(not enabled)
```

UI requirements:
- Template section:
  - list
  - create
  - update
  - delete
  - preview
- Output artifact section:
  - list
  - detail/get
  - create/render
  - update
  - delete
- Local mode behavior:
  - explicit unavailable message
  - no hidden mixed-mode fallback
- Status pane:
  - render last payload as formatted JSON for inspection/debugging

- [ ] **Step 4: Run the UI tests again**

Run: `python3 -m pytest Tests/UI/test_tools_settings_window.py Tests/UI/test_screen_navigation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Outputs_Panel.py tldw_chatbook/UI/Tools_Settings_Window.py Tests/UI/test_tools_settings_window.py Tests/UI/test_screen_navigation.py
git commit -m "feat: add outputs tools panel"
```

### Task 4: Update Parity Docs and Run Focused Verification

**Files:**
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`

- [ ] **Step 1: Update the parity docs for the landed first slice**

Document that tranche-one outputs parity now includes:
- typed server outputs/templates client coverage
- runtime-policy-aware server scope seam
- app wiring
- first source-aware TUI panel
- local mode explicit unavailable state

- [ ] **Step 2: Run the focused verification suite**

Run:

```bash
python3 -m pytest Tests/tldw_api/test_outputs_client.py Tests/Outputs/test_server_outputs_service.py Tests/UI/test_tools_settings_window.py Tests/UI/test_screen_navigation.py Tests/RuntimePolicy/test_runtime_policy_core.py -q
python3 -m compileall tldw_chatbook
git diff --check
```

Expected:
- pytest passes
- `compileall` completes without syntax errors
- `git diff --check` is clean

- [ ] **Step 3: Commit**

```bash
git add Docs/Parity/2026-04-21-gap-ledger.md Docs/Parity/2026-04-21-execution-roadmap.md Docs/Parity/2026-04-21-capability-matrix.md
git commit -m "docs: record outputs parity slice"
```
