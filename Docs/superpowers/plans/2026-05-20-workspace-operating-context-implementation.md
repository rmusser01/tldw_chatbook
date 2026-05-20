# Workspace Operating Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Chatbook's local-first workspace operating-context model so Console can expose real workspace state, global Library/Notes visibility remains intact, and later local-to-server handoff has a safe package boundary.

**Architecture:** Add a focused `Workspaces` domain package with pure models, eligibility rules, local registry persistence, manifest creation, and dry-run handoff services before wiring UI actions. Console consumes the active workspace as operating context; Library, Notes, Artifacts, and search remain global browsers that display workspace membership and gate only active-context actions.

**Tech Stack:** Python 3.11+, Textual, SQLite via existing DB patterns, dataclasses/enums, existing `Sync_Interop` envelope seams, pytest mounted UI tests, textual-web/CDP or actual terminal screenshot approval.

---

## Source Documents

- `Docs/superpowers/specs/2026-05-20-workspace-operating-context-handoff-prd-design.md`
- `Docs/superpowers/handoffs/2026-05-08-ui-screenshot-approval-workflow-handoff.md`
- `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Adjacent server issues: `rmusser01/tldw_server#1526`, `#1440`, `#1528`

## Non-Negotiable Scope Rules

- Do not hide Library, Notes, Artifacts, or conversations when the active workspace changes.
- Do not stage, manipulate, RAG-ground, or agent-edit cross-workspace items unless the user explicitly copies or links them into the active workspace.
- Do not add a top-level Workspaces destination in v1.
- Do not implement background write sync in this tranche.
- Do not recreate ACP sessions, containers, VMs, or git worktrees automatically; v1 describes and validates runtime bindings only.
- Do not claim a UI state is visually approved without an actual rendered PNG screenshot and explicit user approval.
- Do not use generated SVGs, ASCII diagrams, geometry dumps, or code layouts as approval evidence.

## File Responsibility Map

### New Workspace Domain

- Create: `tldw_chatbook/Workspaces/__init__.py`
  - Export stable workspace domain APIs.
- Create: `tldw_chatbook/Workspaces/models.py`
  - Own `WorkspaceRecord`, `WorkspaceMembership`, `WorkspaceRuntimeBinding`, `WorkspaceManifest`, authority/sync/eligibility enums, and transfer policy values.
- Create: `tldw_chatbook/Workspaces/eligibility.py`
  - Pure active-context eligibility decisions. This module must never filter browse/search results.
- Create: `tldw_chatbook/Workspaces/registry_service.py`
  - App-facing local workspace registry service: create/list/update/archive workspaces, active workspace selection, membership lookup, and runtime binding lookup.
- Create: `tldw_chatbook/Workspaces/display_state.py`
  - Console/Library/Notes presentation state builders so UI copy remains deterministic and testable.
- Create: `tldw_chatbook/Workspaces/manifest.py`
  - Build local workspace handoff manifests with copy/reference/redaction/audit classification.
- Create: `tldw_chatbook/Workspaces/handoff_service.py`
  - Dry-run local-to-server and server-to-local preflight. First concrete handoff target is ACP task/run packages.

### Persistence

- Create: `tldw_chatbook/DB/Workspace_DB.py`
  - Local SQLite store for workspace registry tables. Use `BaseDB` path handling.
  - Tables: `workspace_records`, `workspace_memberships`, `workspace_runtime_bindings`, `workspace_handoff_audit`.
  - Keep secrets out of stored runtime binding metadata.
- Modify: `tldw_chatbook/app.py`
  - Instantiate `WorkspaceDB` and `LocalWorkspaceRegistryService`.
  - Expose a stable app attribute such as `workspace_registry_service`.
  - Preserve existing `pending_console_launch` and `pending_notes_workspace_context` behavior.

### Existing Anchors To Reuse

- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
  - Keep existing `scope_type` and `workspace_id` inputs.
  - Add workspace existence/eligibility validation at the service boundary after registry exists.
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
  - Do not add new workspace registry tables here unless a later implementation proves cross-table transactional coupling is required.
  - Reuse existing conversation `scope_type` and `workspace_id` columns.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
  - Replace the single left tray composition with a two-section Console context rail.
  - Keep route IDs and Console transcript/composer behavior intact.
- Modify: `tldw_chatbook/Widgets/Console/console_staged_context.py`
  - Keep current staged-context section as the top half of the new rail.
- Create: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
  - Render `Convos & Workspaces` workspace summary, conversation list, disabled switcher/action affordances, and recovery copy.
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
  - Export the new Console workspace widget.
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
  - Display workspace membership tags and active-context eligibility for visible sources without filtering global Library results.
- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
  - Display workspace membership tags and active-context eligibility while keeping cross-workspace notes visible/editable.
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py`
  - Display artifact workspace membership and active-context eligibility once manifest packaging reaches artifacts.
- Modify: `tldw_chatbook/UX_Interop/server_parity_contracts.py`
  - Reuse `WorkspaceIsolationContract` and `FutureSyncStatusContract`; do not replace them with a parallel contract shape.
- Modify: `tldw_chatbook/Sync_Interop/envelope_builder.py`
  - Extend only when manifest/handoff tasks need explicit workspace package envelopes. Existing `build_workspace_source_ref()` remains the first anchor.

### Tests

- Create: `Tests/Workspaces/test_workspace_models.py`
- Create: `Tests/Workspaces/test_workspace_eligibility.py`
- Create: `Tests/Workspaces/test_workspace_registry_service.py`
- Create: `Tests/Workspaces/test_workspace_display_state.py`
- Create: `Tests/Workspaces/test_workspace_manifest.py`
- Create: `Tests/Workspaces/test_workspace_handoff_service.py`
- Create: `Tests/UI/test_console_workspace_context_rail.py`
- Create: `Tests/UI/test_library_workspace_visibility.py`
- Create: `Tests/UI/test_notes_workspace_visibility.py`
- Modify: `Tests/UI/test_chat_screen_state.py`
- Modify: `Tests/UI/test_destination_visual_parity_correction.py`
- Modify: `Tests/Sync_Interop/test_envelope_builder.py`

## UI Approval Gate Template

Every visible UI task below must complete this gate before the screen can be called approved:

- [ ] Start the real app or a focused harness with production TCSS.
- [ ] Prefer textual-web/CDP for browser-observed screenshots. If textual-web does not show the real state, use an actual terminal screenshot.
- [ ] Capture a PNG screenshot from the rendered app surface.
- [ ] Inspect the screenshot locally and reject it if it shows a loader, wrong screen, crash, blank state, or truncated core workflow.
- [ ] Show the screenshot path to the user and wait for explicit approval.
- [ ] Archive approved screenshots under `Docs/superpowers/qa/product-maturity/phase-4/actual-visual-captures/` unless the active roadmap phase specifies a newer folder.
- [ ] Record screenshot path, approval state, verification commands, and residual risks in the task notes or QA summary.

## Task 1: Workspace Models And Eligibility Rules

**Files:**
- Create: `tldw_chatbook/Workspaces/__init__.py`
- Create: `tldw_chatbook/Workspaces/models.py`
- Create: `tldw_chatbook/Workspaces/eligibility.py`
- Test: `Tests/Workspaces/test_workspace_models.py`
- Test: `Tests/Workspaces/test_workspace_eligibility.py`

- [ ] **Step 1: Write model validation tests**

```python
def test_workspace_record_requires_non_empty_identity() -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        WorkspaceRecord(workspace_id="", name="Research")

def test_workspace_membership_supports_multi_workspace_visibility() -> None:
    membership = WorkspaceMembership(
        workspace_id="ws-a",
        item_type="note",
        item_id="note-1",
        role="source",
    )
    assert membership.workspace_id == "ws-a"
```

- [ ] **Step 2: Write eligibility tests**

```python
def test_cross_workspace_note_is_visible_but_not_context_eligible() -> None:
    result = evaluate_workspace_eligibility(
        active_workspace_id="ws-a",
        item_workspace_ids=("ws-b",),
        item_type="note",
        operation="stage_in_console",
    )

    assert result.visible is True
    assert result.active_context_eligible is False
    assert "Copy or link" in result.recovery_copy
```

- [ ] **Step 3: Run tests and confirm they fail**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_models.py Tests/Workspaces/test_workspace_eligibility.py --tb=short`

Expected: fail because `tldw_chatbook.Workspaces` does not exist.

- [ ] **Step 4: Implement minimal models**

Use dataclasses and literal-safe enums. Keep validation in constructors or small helpers.

```python
class WorkspaceAuthority(str, Enum):
    LOCAL_ONLY = "local-only"
    SERVER_BACKED = "server-backed"
    SYNCING_TO_SERVER = "syncing-to-server"
    SYNCING_FROM_SERVER = "syncing-from-server"
    CONFLICT = "conflict"
    DETACHED = "detached"
    REMOTE_ONLY = "remote-only"
    RUNTIME_MISSING = "runtime-missing"
```

- [ ] **Step 5: Implement pure eligibility decisions**

Rules:

- Browse/search visibility is always true for user-owned items.
- Active-context operations require active workspace membership.
- Global conversations are visible in Console but not silently converted into workspace conversations.
- Cross-workspace use returns a recovery path: copy/link into the active workspace.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_models.py Tests/Workspaces/test_workspace_eligibility.py --tb=short`

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Workspaces Tests/Workspaces/test_workspace_models.py Tests/Workspaces/test_workspace_eligibility.py
git commit -m "feat: add workspace operating context models"
```

## Task 2: Local Workspace Registry Persistence

**Files:**
- Create: `tldw_chatbook/DB/Workspace_DB.py`
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/Workspaces/test_workspace_registry_service.py`
- Test: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write persistence tests**

```python
def test_registry_persists_active_workspace(tmp_path: Path) -> None:
    db = WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    service = LocalWorkspaceRegistryService(db)

    service.create_workspace(workspace_id="ws-a", name="Local Research")
    service.set_active_workspace("ws-a")

    reloaded = LocalWorkspaceRegistryService(WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1"))
    assert reloaded.get_active_workspace().workspace_id == "ws-a"
```

- [ ] **Step 2: Write membership tests**

```python
def test_registry_links_note_without_hiding_other_workspaces(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)
    service.link_membership("ws-a", item_type="note", item_id="note-1", role="source")
    service.link_membership("ws-b", item_type="note", item_id="note-1", role="reference")

    assert {m.workspace_id for m in service.get_item_memberships("note", "note-1")} == {"ws-a", "ws-b"}
```

- [ ] **Step 3: Run tests and confirm they fail**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_registry_service.py --tb=short`

Expected: fail because storage/service are missing.

- [ ] **Step 4: Implement `WorkspaceDB`**

Use `BaseDB` for path handling and a local schema version table scoped to this DB. Minimum schema:

```sql
CREATE TABLE IF NOT EXISTS workspace_records (
    workspace_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    authority TEXT NOT NULL,
    sync_status TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

- [ ] **Step 5: Implement `LocalWorkspaceRegistryService`**

Service responsibilities:

- Normalize IDs/names.
- Create a default local workspace only when explicitly requested by the caller.
- Set one active workspace per client.
- Link/unlink memberships idempotently.
- Store runtime binding metadata without secrets.
- Return deterministic ordering for UI lists.

- [ ] **Step 6: Wire app-owned service**

In `tldw_chatbook/app.py`, instantiate `WorkspaceDB` and `LocalWorkspaceRegistryService` next to existing local DB/service setup. Expose as `self.workspace_registry_service`.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_registry_service.py Tests/UI/test_screen_navigation.py --tb=short`

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/DB/Workspace_DB.py tldw_chatbook/Workspaces/registry_service.py tldw_chatbook/app.py Tests/Workspaces/test_workspace_registry_service.py Tests/UI/test_screen_navigation.py
git commit -m "feat: persist local workspace registry"
```

## Task 3: Console Context Rail Read-Only Shell

**Files:**
- Create: `tldw_chatbook/Workspaces/display_state.py`
- Create: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/Workspaces/test_workspace_display_state.py`
- Test: `Tests/UI/test_console_workspace_context_rail.py`
- QA: `Docs/superpowers/qa/product-maturity/phase-4/actual-visual-captures/`

- [ ] **Step 1: Write display-state tests**

```python
def test_console_workspace_state_explains_missing_service() -> None:
    state = build_console_workspace_state(
        registry_service=None,
        current_conversation=None,
    )

    assert state.heading == "Convos & Workspaces"
    assert state.workspace_label == "No workspace selected"
    assert state.change_workspace_enabled is False
    assert "service not ready" in state.recovery_copy.lower()
```

- [ ] **Step 2: Write mounted UI tests**

Assert `#console-staged-context-tray` still exists and the new rail exposes:

- `#console-workspace-context`
- `#console-active-workspace`
- `#console-workspace-conversations`
- `#console-change-workspace`
- Disabled recovery copy when service is missing.

- [ ] **Step 3: Run tests and confirm they fail**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_context_rail.py --tb=short`

Expected: fail because new display state/widget are missing.

- [ ] **Step 4: Implement display state and widget**

`ConsoleWorkspaceContextTray` renders:

- `Convos & Workspaces`
- Active workspace name/authority/sync/runtime.
- Conversation list for the active workspace.
- Disabled `Change workspace` until switch behavior is implemented.
- `New conversation` action only when registry and active workspace are ready.

- [ ] **Step 5: Compose two-section left rail**

In `ChatScreen.compose_content()`, wrap `ConsoleStagedContextTray` and `ConsoleWorkspaceContextTray` in a vertical left rail. Keep the existing left rail width ratio smaller than transcript and inspector because it is context/status, not the primary work area.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_destination_visual_parity_correction.py::test_console_screen_uses_terminal_native_workbench_layout --tb=short`

Expected: pass.

- [ ] **Step 7: Capture actual screenshots**

Capture at least:

- Console with no workspace service ready.
- Console with an active local workspace and empty conversation list.
- Console with active workspace conversations.

Use textual-web/CDP or actual terminal screenshot. Do not continue until the user approves the rendered screenshots.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Workspaces/display_state.py tldw_chatbook/Widgets/Console/console_workspace_context.py tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_context_rail.py Docs/superpowers/qa/product-maturity/phase-4
git commit -m "feat: add console workspace context rail"
```

## Task 4: Active Workspace Selection And Workspace Conversations

**Files:**
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py`
- Test: `Tests/Workspaces/test_workspace_registry_service.py`
- Test: `Tests/UI/test_chat_screen_state.py`
- Test: `Tests/UI/test_console_workspace_context_rail.py`

- [ ] **Step 1: Write conversation-scope regression tests**

```python
def test_workspace_conversation_requires_existing_workspace(tmp_path: Path) -> None:
    registry = build_test_registry(tmp_path)
    service = ChatPersistenceService(db=chat_db, workspace_registry=registry)

    with pytest.raises(ValueError, match="Unknown workspace"):
        service.create_conversation(scope_type="workspace", workspace_id="missing")
```

- [ ] **Step 2: Write active switch tests**

Mounted Console tests should prove selecting workspace B changes eligible conversation rows in `Convos & Workspaces` but does not affect Library/Notes browse state.

- [ ] **Step 3: Run tests and confirm they fail**

Run: `python -m pytest -q Tests/UI/test_chat_screen_state.py Tests/UI/test_console_workspace_context_rail.py --tb=short`

Expected: fail because selection is not wired.

- [ ] **Step 4: Add service APIs**

Add:

- `list_workspace_conversations(workspace_id)`
- `set_active_workspace(workspace_id)`
- `get_active_workspace_context()`
- `fork_conversation_into_workspace(conversation_id, target_workspace_id)` as a service stub that records intent but does not silently mutate global conversations.

- [ ] **Step 5: Wire Console actions**

`Change workspace` may open a lightweight selection overlay/list only after the registry exists. If no workspace exists, show `Create a workspace in Library > Workspaces before switching`.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_registry_service.py Tests/UI/test_chat_screen_state.py Tests/UI/test_console_workspace_context_rail.py --tb=short`

Expected: pass.

- [ ] **Step 7: Capture actual screenshots**

Capture:

- Workspace switch list.
- Active workspace changed.
- Global conversation shown as global, not silently workspace-bound.

Wait for user approval before calling the Console workspace switcher visually approved.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Workspaces/registry_service.py tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Widgets/Console/console_workspace_context.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py Tests/Workspaces/test_workspace_registry_service.py Tests/UI/test_chat_screen_state.py Tests/UI/test_console_workspace_context_rail.py Docs/superpowers/qa/product-maturity/phase-4
git commit -m "feat: wire active workspace conversations"
```

## Task 5: Library And Notes Global Visibility With Context Eligibility

**Files:**
- Create: `tldw_chatbook/Workspaces/item_context.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
- Modify: `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_left.py`
- Modify: `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py`
- Test: `Tests/UI/test_library_workspace_visibility.py`
- Test: `Tests/UI/test_notes_workspace_visibility.py`
- Test: `Tests/UI/test_notes_screen.py`

- [ ] **Step 1: Write Library visibility tests**

```python
async def test_library_shows_cross_workspace_item_but_disables_staging(app) -> None:
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-a", name="A")
    registry.create_workspace(workspace_id="ws-b", name="B")
    registry.set_active_workspace("ws-a")
    registry.link_membership("ws-b", item_type="note", item_id="note-b", role="source")

    # Mount Library and select note-b.
    assert screen.query_one("#library-note-workspace-badge").renderable == "Workspace: B"
    assert screen.query_one("#library-use-in-console").disabled is True
```

- [ ] **Step 2: Write Notes visibility tests**

Assert a note from workspace B remains visible and editable while active workspace A is selected, but `Use in Console` is disabled with copy/link recovery copy.

- [ ] **Step 3: Run tests and confirm they fail**

Run: `python -m pytest -q Tests/UI/test_library_workspace_visibility.py Tests/UI/test_notes_workspace_visibility.py --tb=short`

Expected: fail because membership badges and eligibility gates do not exist.

- [ ] **Step 4: Implement item context helpers**

`item_context.py` should convert item IDs into:

- `workspace_ids`
- `workspace_labels`
- `active_context_eligible`
- `eligibility_reason`
- `recovery_copy`

Do not filter item lists in this helper.

- [ ] **Step 5: Wire Library**

Library should keep all local Notes/Media/Conversations visible. Add badges and disabled active-context action states. `Search/RAG` can search globally, but `Use in Console` must respect active workspace eligibility.

- [ ] **Step 6: Wire Notes**

Notes should keep global browse/edit behavior. Workspace badges and disabled `Use in Chat` recovery copy must be visible in right-side or detail context.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest -q Tests/UI/test_library_workspace_visibility.py Tests/UI/test_notes_workspace_visibility.py Tests/UI/test_notes_screen.py --tb=short`

Expected: pass.

- [ ] **Step 8: Capture actual screenshots**

Capture:

- Library showing a cross-workspace source with disabled staging.
- Notes showing a cross-workspace note still editable but not stageable.
- Console after attempting a blocked cross-workspace staging action, showing recovery copy.

Wait for user approval before closing this UI task.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Workspaces/item_context.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/notes_screen.py tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_left.py tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py Tests/UI/test_library_workspace_visibility.py Tests/UI/test_notes_workspace_visibility.py Tests/UI/test_notes_screen.py Docs/superpowers/qa/product-maturity/phase-4
git commit -m "feat: show workspace eligibility in library and notes"
```

## Task 6: Workspace Manifest And User-Visible Audit

**Files:**
- Modify: `tldw_chatbook/Workspaces/models.py`
- Modify: `tldw_chatbook/Workspaces/manifest.py`
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Test: `Tests/Workspaces/test_workspace_manifest.py`
- Test: `Tests/Sync_Interop/test_envelope_builder.py`

- [ ] **Step 1: Write manifest tests**

```python
def test_manifest_classifies_copy_reference_and_local_only_sources() -> None:
    manifest = build_workspace_manifest(
        workspace=workspace,
        memberships=memberships,
        source_policy=StaticSourcePolicy({
            "file-1": "copy",
            "remote-1": "reference",
            "secret-1": "local-only",
        }),
    )

    assert manifest.sources["file-1"].transfer_mode == "copy"
    assert manifest.sources["remote-1"].transfer_mode == "reference"
    assert manifest.sources["secret-1"].blocked_reason == "local_only"
    assert manifest.audit.redactions
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_manifest.py --tb=short`

Expected: fail because manifest builder does not exist.

- [ ] **Step 3: Implement manifest builder**

Manifest must include:

- Workspace identity and authority.
- Source, conversation, note, artifact, ACP run, MCP/tool, schedule, workflow, runtime binding summaries.
- Transfer mode per source: `copy`, `reference`, `metadata-only`, `local-only`.
- Redaction report.
- User-visible audit summary.
- Non-restorable runtime binding list.

- [ ] **Step 4: Extend sync envelope only if required**

If manifest packages need an envelope, add a small builder method to `SyncEnvelopeBuilder`. Do not alter existing note/chat/source cache envelope behavior.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_manifest.py Tests/Sync_Interop/test_envelope_builder.py --tb=short`

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Workspaces/models.py tldw_chatbook/Workspaces/manifest.py tldw_chatbook/Workspaces/registry_service.py tldw_chatbook/Sync_Interop/envelope_builder.py Tests/Workspaces/test_workspace_manifest.py Tests/Sync_Interop/test_envelope_builder.py
git commit -m "feat: build workspace handoff manifests"
```

## Task 7: ACP Task/Run Package Handoff Dry Run

**Files:**
- Modify: `tldw_chatbook/Workspaces/handoff_service.py`
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `tldw_chatbook/UI/Screens/acp_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/Workspaces/test_workspace_handoff_service.py`
- Test: `Tests/MCP/test_unified_control_plane_service.py`
- Test: `Tests/UI/test_console_live_work_handoffs.py`

- [ ] **Step 1: Write ACP package dry-run tests**

```python
def test_acp_handoff_preflight_includes_task_run_and_required_sources() -> None:
    result = handoff_service.preflight_acp_run_package(
        workspace_id="ws-a",
        acp_run_id="run-1",
    )

    assert result.status == "ready"
    assert result.primary_target == "acp_task_run"
    assert result.audit.user_visible is True
    assert "run-1" in result.manifest.acp_runs
```

- [ ] **Step 2: Write blocked runtime tests**

Assert missing ACP runtime returns `runtime-missing` and clear recovery copy, not a false ready state.

- [ ] **Step 3: Run tests and confirm they fail**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_handoff_service.py Tests/UI/test_console_live_work_handoffs.py --tb=short`

Expected: fail because ACP package preflight is not implemented.

- [ ] **Step 4: Implement dry-run preflight**

The first implementation creates no server records. It returns a manifest, transfer plan, audit details, and blocked runtime reasons.

- [ ] **Step 5: Wire ACP/Console UI states**

ACP and Console may expose `Prepare handoff` only when a package is available. The action opens a preflight summary; it does not perform background sync.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest -q Tests/Workspaces/test_workspace_handoff_service.py Tests/MCP/test_unified_control_plane_service.py Tests/UI/test_console_live_work_handoffs.py --tb=short`

Expected: pass.

- [ ] **Step 7: Capture actual screenshots**

Capture:

- ACP run package preflight ready state.
- ACP runtime missing blocked state.
- Console live-work handoff summary.

Wait for user approval before marking this UI flow accepted.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Workspaces/handoff_service.py tldw_chatbook/MCP/unified_control_plane_service.py tldw_chatbook/UI/Screens/acp_screen.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Workspaces/test_workspace_handoff_service.py Tests/MCP/test_unified_control_plane_service.py Tests/UI/test_console_live_work_handoffs.py Docs/superpowers/qa/product-maturity/phase-4
git commit -m "feat: add acp workspace handoff preflight"
```

## Task 8: Manual Server Handoff Boundary

**Files:**
- Modify: `tldw_chatbook/Workspaces/handoff_service.py`
- Create: `tldw_chatbook/tldw_api/workspace_handoff_schemas.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/Workspaces/test_workspace_handoff_service.py`
- Test: `Tests/tldw_api/test_workspace_handoff_schemas.py`

- [ ] **Step 1: Write manual handoff schema tests**

```python
def test_server_handoff_request_preserves_copy_and_reference_modes() -> None:
    request = WorkspaceHandoffRequest.model_validate({
        "workspace_id": "ws-a",
        "direction": "local-to-server",
        "sources": [
            {"source_id": "file-1", "transfer_mode": "copy"},
            {"source_id": "url-1", "transfer_mode": "reference"},
        ],
    })

    assert [source.transfer_mode for source in request.sources] == ["copy", "reference"]
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `python -m pytest -q Tests/tldw_api/test_workspace_handoff_schemas.py Tests/Workspaces/test_workspace_handoff_service.py --tb=short`

Expected: fail because schemas and manual handoff boundary are missing.

- [ ] **Step 3: Add schemas and service boundary**

Add request/response schemas only. Do not start background sync. Do not assume server endpoint names until server API is confirmed.

- [ ] **Step 4: Add UI preflight copy**

Console/Library can show `Manual handoff prepared` and export/copy manifest JSON. If server transport is unavailable, show `Server handoff transport not configured`.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q Tests/tldw_api/test_workspace_handoff_schemas.py Tests/Workspaces/test_workspace_handoff_service.py --tb=short`

Expected: pass.

- [ ] **Step 6: Capture actual screenshots**

Capture:

- Manual handoff preflight summary.
- Server transport unavailable state.
- Manifest audit details expanded.

Wait for user approval before closing the UI flow.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Workspaces/handoff_service.py tldw_chatbook/tldw_api/workspace_handoff_schemas.py tldw_chatbook/tldw_api/__init__.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/chat_screen.py Tests/tldw_api/test_workspace_handoff_schemas.py Tests/Workspaces/test_workspace_handoff_service.py Docs/superpowers/qa/product-maturity/phase-4
git commit -m "feat: define manual workspace handoff boundary"
```

## Task 9: Cross-Screen Actual-Use QA And Closeout

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-4/workspace-operating-context-qa.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: relevant `backlog/tasks/task-*.md` implementation notes
- Test: `Tests/UI/test_product_maturity_phase3_layout_contracts.py`
- Test: `Tests/UI/test_post_release_ux_hci_validation_plan.py`

- [ ] **Step 1: Write QA checklist before running it**

The QA document must include:

- Console active workspace switch.
- Console staged context eligibility.
- Library global search with workspace badges.
- Notes cross-workspace edit plus blocked staging.
- Artifacts Chatbook membership and resume.
- ACP task/run package preflight.
- Manual server handoff preflight.
- Keyboard-only path through affected controls.

- [ ] **Step 2: Run automated closeout suite**

Run: `python -m pytest -q Tests/Workspaces Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_library_workspace_visibility.py Tests/UI/test_notes_workspace_visibility.py Tests/UI/test_product_maturity_phase3_layout_contracts.py Tests/UI/test_post_release_ux_hci_validation_plan.py --tb=short`

Expected: pass.

- [ ] **Step 3: Run actual-use walkthrough**

Use textual-web/CDP or actual terminal screenshots. Record each screenshot path and whether the user approved it.

- [ ] **Step 4: Update roadmap and backlog**

Only mark tasks Done when:

- Acceptance criteria are checked.
- Automated tests pass.
- Actual rendered screenshots are approved for changed UI states.
- QA note records what was verified and what remains uncertain.

- [ ] **Step 5: Run final hygiene**

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 6: Commit**

```bash
git add Docs/superpowers/qa/product-maturity/phase-4/workspace-operating-context-qa.md Docs/superpowers/trackers/product-maturity-roadmap.md backlog/tasks
git commit -m "docs: close workspace operating context QA"
```

## Recommended PR Slicing

1. `PR A - Workspace domain foundation`: Tasks 1 and 2.
2. `PR B - Console workspace context rail`: Task 3.
3. `PR C - Active workspace conversations`: Task 4.
4. `PR D - Global visibility with eligibility gates`: Task 5.
5. `PR E - Manifest and ACP package preflight`: Tasks 6 and 7.
6. `PR F - Manual server handoff boundary`: Task 8.
7. `PR G - QA closeout`: Task 9.

Do not batch UI PRs unless the user approves screenshots for every affected screen before merge.

## Final Verification Commands

Run these before marking the implementation tranche complete:

```bash
python -m pytest -q Tests/Workspaces --tb=short
python -m pytest -q Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_library_workspace_visibility.py Tests/UI/test_notes_workspace_visibility.py --tb=short
python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py Tests/UI/test_post_release_ux_hci_validation_plan.py --tb=short
python -m pytest -q Tests/Sync_Interop/test_envelope_builder.py Tests/MCP/test_unified_control_plane_service.py --tb=short
git diff --check
```

Expected: all pytest commands pass and `git diff --check` reports no whitespace errors.

## Residual Risks To Track

- Existing Notes workspace services are server-oriented. The local registry must not accidentally make server workspace APIs look ready when only local metadata exists.
- A separate `WorkspaceDB` avoids bloating `ChaChaNotes_DB.py`, but cross-DB transactions are not atomic. If later implementation needs atomic conversation plus workspace membership writes, promote those specific writes through a service-level transaction boundary or migrate narrowly.
- Source copy/reference policy needs server policy confirmation before real upload.
- ACP task/run packages can be preflighted before runtime recreation is safe; do not overpromise resume semantics.
- User-visible audit can be verbose. UI should collapse details but keep export/reveal paths available.
