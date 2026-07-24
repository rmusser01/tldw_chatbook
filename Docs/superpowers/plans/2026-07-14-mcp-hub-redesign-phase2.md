# MCP Hub Redesign — Phase 2 Implementation Plan (Server Mutations + UX Batch)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MCP Hub's Servers mode fully operational — add/edit/delete/import local profiles, connect/test/refresh with timeouts and cancel, server-source mutations, built-in toggles — and apply the deferred Phase 2 UX batch (breadcrumb, actionable callouts, status colors, collapsed Advanced disclosure, footer shortcuts).

**Architecture:** New persisted per-profile runtime state (`LocalMCPStore`) feeds an upgraded readiness derivation. `UnifiedMCPControlPlaneService` grows **typed lifecycle/mutation methods** that wrap the existing `run_action` gating with wall-clock timeouts and runtime-state recording — this is the single seam the Phase 5 chat bridge AND the agent runtime's future `MCPToolProvider` (task-201 seam reserved in `Agents/tool_catalog.py:4`) will consume. UI work stays in `UI/MCP_Modules/` following Phase 1's widgets and the repo's current form/confirm/worker conventions.

**Tech Stack:** Python ≥3.11, Textual 8.2.7 (pinned), pytest + pytest-asyncio. Spec: `Docs/superpowers/specs/2026-07-13-mcp-hub-redesign-design.md` (§7 corrections applied on this branch). UX inputs: `Docs/Design/mcp-hub-phase2-ux-inputs.md`.

## Global Constraints

- Tests: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest <files>` from the worktree root (`.worktrees/mcp-hub-phase2`). The `timeout` shell command does not exist. Only the test files named in this plan must pass; 2 pre-existing failures exist on dev in `Tests/UI/test_destination_visual_parity_correction.py` (`test_library_source_snapshot_times_out_to_stable_error`, `test_library_source_snapshot_timeout_handles_blocking_async_services` — upstream Library regressions, verified at origin/dev; do NOT fix).
- CSS: edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, rebuild `tldw_cli_modular.tcss` via `python3 tldw_chatbook/css/build_css.py`, commit both. **Never hand-edit the bundle** (rebase lesson). App-loaded CSS beats widget DEFAULT_CSS unconditionally — the bundle is the real fix layer; widget DEFAULT_CSS is the harness-only geometry baseline.
- Textual 8.2.7 verified rules: nested `Message` classes in `MCP*`-named widgets need explicit `namespace=`; `Select.BLANK` is not a real API (`Select.NULL` is the sentinel; placeholder `(label, False)` self-consistent; treat both as no-selection); `Select` posts `Changed` for its constructor value AT MOUNT (guard echoes); round borders need ≥2 lines (height-1 buttons collapse); `remove_children()` must be awaited before re-mounting id-bearing children.
- Design system: `ds-*` classes only; status color tokens `$ds-status-ready` (green), `$ds-status-warning` (amber), `$ds-status-error` (red), `$ds-status-info` (accent) from `css/core/_variables.tcss:23-40`. User/remote text into labels: `markup=False` or `rich.markup.escape`. Widget ids: `[a-zA-Z_-][a-zA-Z0-9_-]*` — never raw profile ids in widget ids.
- Frozen files: `unified_mcp_panel.py`, `unified_mcp_sections.py` (retired Phase 6). `MCP/redaction.py` frozen except where a task names it.
- Local profile save payload keys (verified): `profile_id`, `command`, `args`, `env_placeholders`, `env_literals` — a plain `env` key is **silently dropped** by `from_input_dict` (local_store.py:284-298). `env_placeholders` values must match `$NAME`/`${NAME}`; `env_literals` REJECTS secret-shaped keys/values and non-whitelisted values with `ValueError` (store messages are good UI copy — surface them verbatim).
- Local governance action_ids (enforced inside the existing `run_action` path): `mcp.external_profiles.configure.local` (save/delete), `.launch.local` (connect/disconnect), `.trigger.local` (test), `.observe.local` (refresh). Server-source mutations run through the 12 existing `run_action` names (`external_server.create/update/delete`, `external_server.slot.*`, `external_server.secret.set`, `external_server.auth_template.update`) — gated + scope-revalidated internally; **do not pass `owner_scope_type`/`owner_scope_id`** (service injects them; schemas are `extra="forbid"`).
- Conventional commits ending with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Work in `.worktrees/mcp-hub-phase2` on branch `claude/mcp-hub-phase2` only.

## Verified Facts (planning-time, 2026-07-14)

1. **Local mutation surface** (`local_control_service.py`): `save_external_profile(payload)->dict` (raises `ValueError` on validation), `delete_external_profile(id)->bool`, `connect_profile(id)->snapshot dict` (raises `KeyError`/`RuntimeError`), `disconnect_profile(id)->bool`, `test_external_profile(id)->{ok,profile_id,tools,resources,prompts}` (counts), `refresh_external_profile(id)->snapshot`. `run_action` local branch passes `profile.save` payload straight through (line ~901). Saving with changed command/args/env pops the discovery snapshot (`local_store.py:627`).
2. **No per-profile attempt state exists** (`last_error`/`last_validated_at`: zero hits). Natural home: a sixth collection on `LocalMCPStoreState` mirroring `discovery_snapshots`; ~9 `LocalMCPStoreState(...)` constructor call sites must be threaded (each missed site silently resets the field).
3. **Timeouts**: per-RPC 10s (`client.py:_REQUEST_TIMEOUT_SECONDS`), no wall-clock cap (connect worst case ≈40s serial). Right seam: `asyncio.wait_for` around the control-plane call — no `client.py` change.
4. **No `mcpServers` importer exists** — build from scratch; imported literal secrets will be REJECTED by the store, so the importer must convert them to `$VAR` placeholders with explicit warnings.
5. **Server-source mutations fully wired**: all 12 actions reachable via `run_action` with governance + scope-revalidation; `available_actions()` returns `[]` for server source below team scope (UI must mirror). Gap: auth-template GET is unwrapped (template display deferred — recorded).
6. **Agent runtime (PR #623)**: engine-only, no UI, no MCP imports, no collisions with this phase. Implication honored here: typed control-plane methods are the shared gate/execute seam for Phase 5 + task-201.
7. **UI conventions**: forms = hand-composed `Static` label + `Input` + one error `Static` + state-driven `disabled` (model: `Widgets/Library/library_collections_panel.py:30`); destructive = inline arm-then-confirm (`library_screen.py:8724-8747`); in-pane disclosure = `textual.widgets.Collapsible` + `Collapsible.Toggled` (model: `library_ingest_canvas.py:108`), persisted via `save_setting_to_cli_config` (model: `library_screen.py:5502-5535`); footer hints = `AppFooterStatus.set_workbench_shortcuts(source=, shortcuts=)` on mount/resume + `clear_shortcut_context(source=)` on suspend (model: `chat_screen.py:280,825,835,5652,9212,9231`); file pick = `EnhancedFileOpen(location=".", title=..., filters=..., context="...")` via `push_screen(..., callback=)` (model: `chat_screen.py:7242-7279`); masked input = `Input(password=True)` (model: `settings_screen.py:5306`).

## File Structure

| File | Responsibility |
|---|---|
| Modify `tldw_chatbook/MCP/local_store.py` | `profile_runtime_state` collection + accessors + cascade/pop rules |
| Modify `tldw_chatbook/MCP/unified_control_plane_service.py` | Typed local lifecycle/mutation methods (timeout + runtime-state recording); `local_external_catalog()` |
| Modify `tldw_chatbook/MCP/readiness.py` | Runtime-state-aware local derivation; `as_checking()`; `STATE_CSS_CLASSES` |
| Create `tldw_chatbook/MCP/mcp_import.py` | `{"mcpServers": ...}` → import candidates with env classification + warnings |
| Modify `tldw_chatbook/UI/MCP_Modules/mcp_rail.py` | One-shot mount-echo guard (watch-item) |
| Modify `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` | Refresh lock (watch-item); Cancel affordance; wired lifecycle actions; Collapsible Advanced + object label + zero-action hint |
| Create `tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py` | Local profile add/edit form + import panel |
| Create `tldw_chatbook/UI/MCP_Modules/mcp_server_mutations.py` | Server-source create/update/delete + credential-slot forms |
| Modify `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py` | Breadcrumb, compact actionable callouts, per-source columns, form hosting, delete arm-confirm, built-in toggles |
| Modify `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` | Lifecycle workers + cancel + in-flight CHECKING; catalog switch; form/import/mutation orchestration; server-source external records in canvas |
| Modify `tldw_chatbook/UI/Screens/mcp_screen.py` | `a`/`r` bindings; footer shortcuts registration |
| Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ rebuild bundle) | status color classes, `.ds-info-callout`, form/callout geometry |
| Tests | `Tests/MCP/test_local_store_runtime_state.py`, `Tests/MCP/test_control_plane_lifecycle.py`, `Tests/MCP/test_mcp_import.py`, extend `Tests/MCP/test_readiness_derivation.py`, `Tests/UI/test_mcp_rail.py`, `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py`, `Tests/UI/test_mcp_servers_mode.py`; create `Tests/UI/test_mcp_profile_form.py`, `Tests/UI/test_mcp_server_mutations.py` |

Execution order: T1→T2→T3 (foundation) → T4 (hardening) → T5 (lifecycle UI) → T6→T7→T8 (local mutations UI) → T9 (server-source) → T10 (built-in) → T11→T12 (UX batch) → T13 (shortcuts/CSS/gate).

---

### Task 1: Per-profile runtime state in `LocalMCPStore`

**Files:**
- Modify: `tldw_chatbook/MCP/local_store.py`
- Test: `Tests/MCP/test_local_store_runtime_state.py`

**Interfaces:**
- Consumes: existing `LocalMCPStoreState` (5 collections), `LocalMCPStore.save/load`, `_launch_config_changed`.
- Produces (exact, used by T2/T3): `LocalMCPStoreState.profile_runtime_state: dict[str, dict[str, Any]]` (sixth collection, persisted under JSON key `"profile_runtime_state"`); `LocalMCPStore.get_profile_runtime_state(profile_id: str) -> dict | None`; `LocalMCPStore.save_profile_runtime_state(profile_id: str, state: dict) -> dict`; cascade: `delete_profile` removes the entry; `save_profile` pops it when `_launch_config_changed` (same condition as the discovery-snapshot pop). Runtime-state record shape (convention, not enforced): `{"last_attempt_at": iso-str, "last_action": "connect|disconnect|test|refresh", "ok": bool, "last_ok_at": iso-str|None, "last_error": str|None}`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/MCP/test_local_store_runtime_state.py
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.MCP.local_store import LocalMCPStore


@pytest.fixture()
def store(tmp_path: Path) -> LocalMCPStore:
    return LocalMCPStore(tmp_path / "local_mcp_store.json")


def _save_profile(store: LocalMCPStore, profile_id: str = "docs", command: str = "python"):
    return store.save_profile(
        {"profile_id": profile_id, "command": command, "args": ["-m", "demo"]}
    )


def test_runtime_state_roundtrip_persists(store, tmp_path):
    _save_profile(store)
    record = {"last_attempt_at": "2026-07-14T00:00:00Z", "last_action": "connect",
              "ok": False, "last_ok_at": None, "last_error": "boom"}
    saved = store.save_profile_runtime_state("docs", record)
    assert saved == record
    reloaded = LocalMCPStore(tmp_path / "local_mcp_store.json")
    assert reloaded.get_profile_runtime_state("docs") == record
    assert reloaded.get_profile_runtime_state("missing") is None


def test_delete_profile_cascades_runtime_state(store):
    _save_profile(store)
    store.save_profile_runtime_state("docs", {"ok": True})
    assert store.delete_profile("docs") is True
    assert store.get_profile_runtime_state("docs") is None


def test_launch_config_change_pops_runtime_state(store):
    _save_profile(store, command="python")
    store.save_profile_runtime_state("docs", {"ok": False, "last_error": "old"})
    _save_profile(store, command="node")  # command changed -> launch config changed
    assert store.get_profile_runtime_state("docs") is None


def test_other_mutations_do_not_reset_runtime_state(store):
    _save_profile(store)
    store.save_profile_runtime_state("docs", {"ok": True})
    # a snapshot save is one of the other LocalMCPStoreState(...) reconstruction sites
    store.save_discovery_snapshot("docs", {"tools": [{"name": "a"}], "resources": [], "prompts": []})
    assert store.get_profile_runtime_state("docs") == {"ok": True}
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/MCP/test_local_store_runtime_state.py -v`
Expected: FAIL with `AttributeError: 'LocalMCPStore' object has no attribute 'save_profile_runtime_state'`

- [ ] **Step 3: Implement**

In `tldw_chatbook/MCP/local_store.py`:

1. Add to `LocalMCPStoreState` (dataclass, ~line 494): `profile_runtime_state: dict[str, dict[str, Any]] = field(default_factory=dict)`.
2. In `to_dict` (~502) add `"profile_runtime_state": dict(self.profile_runtime_state),`; in `from_dict` (~514) add `profile_runtime_state=dict(payload.get("profile_runtime_state") or {}),`.
3. **Thread the field through EVERY `LocalMCPStoreState(` constructor call site** (grep `LocalMCPStoreState(` — ~9 sites in `save_profile`, `delete_profile`, `save_discovery_snapshot`, `save_governance_rule`, `delete_governance_rule`, `save_approval_request`, `resolve_approval_request`, `delete_approval_request`, `record_runtime_activity`): pass `profile_runtime_state=<current mutated-or-passthrough dict>` at each. A missed site silently wipes the collection — the fourth test above catches the snapshot site; verify the rest by grep count before committing.
4. In `save_profile`, extend the existing launch-config-changed block (~line 627) to also pop runtime state:
```python
        if existing_profile and self._launch_config_changed(existing_profile, saved_profile):
            discovery_snapshots.pop(saved_profile.profile_id, None)
            profile_runtime_state.pop(saved_profile.profile_id, None)
```
5. In `delete_profile`, pop the entry alongside the profile removal.
6. New accessors (mirror `get_discovery_snapshot`/`save_discovery_snapshot` style):
```python
    def get_profile_runtime_state(self, profile_id: str) -> dict[str, Any] | None:
        """Return the persisted lifecycle-attempt record for a profile.

        Args:
            profile_id: Stable local profile identifier.

        Returns:
            The stored record dict, or None if no attempt has been recorded.
        """
        state = self.load()
        record = state.profile_runtime_state.get(str(profile_id))
        return dict(record) if record is not None else None

    def save_profile_runtime_state(self, profile_id: str, record: dict[str, Any]) -> dict[str, Any]:
        """Persist the lifecycle-attempt record for a profile (last-write-wins).

        Args:
            profile_id: Stable local profile identifier.
            record: Attempt record (see module convention: last_attempt_at,
                last_action, ok, last_ok_at, last_error).

        Returns:
            The stored record dict.
        """
        state = self.load()
        runtime = dict(state.profile_runtime_state)
        runtime[str(profile_id)] = dict(record)
        self.save(LocalMCPStoreState(
            profiles=state.profiles,
            discovery_snapshots=state.discovery_snapshots,
            governance_rules=state.governance_rules,
            approval_requests=state.approval_requests,
            runtime_activity=state.runtime_activity,
            profile_runtime_state=runtime,
        ))
        return dict(record)
```
(Adjust the keyword list to the dataclass's actual field names — copy them from the existing `save_discovery_snapshot` body.)

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/MCP/test_local_store_runtime_state.py Tests/MCP/ -q`
Expected: new file 4/4 PASS; whole `Tests/MCP/` green (no regressions from the threading).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/local_store.py Tests/MCP/test_local_store_runtime_state.py
git commit -m "feat(mcp): persist per-profile lifecycle runtime state in LocalMCPStore"
```

---

### Task 2: Typed lifecycle/mutation methods on the control plane

**Files:**
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Test: `Tests/MCP/test_control_plane_lifecycle.py`

**Interfaces:**
- Consumes: T1 accessors (via `self.local_service.store`); existing `run_action` local branches; `get_cli_setting` from `tldw_chatbook.config`.
- Produces (exact — the shared seam for Phase 5 chat bridge and the task-201 agent `MCPToolProvider`):
  - `async save_local_profile(self, payload: dict) -> dict` (raises `ValueError` on store validation — callers surface `str(exc)`)
  - `async delete_local_profile(self, profile_id: str) -> bool`
  - `async connect_local_profile(self, profile_id: str) -> dict`
  - `async disconnect_local_profile(self, profile_id: str) -> bool`
  - `async test_local_profile(self, profile_id: str) -> dict`
  - `async refresh_local_profile(self, profile_id: str) -> dict`
  - `async local_external_catalog(self) -> list[dict]` — `local_service.get_external_servers()` items each merged with `"runtime_state": <record|None>`
  - Timeout: every connect/disconnect/test/refresh wrapped in `asyncio.wait_for(..., timeout=float(get_cli_setting("mcp", "hub_lifecycle_timeout_seconds", 45)))`; on `asyncio.TimeoutError` raises `RuntimeError(f"Timed out after {timeout:.0f}s")`.
  - Recording: connect/disconnect/test/refresh write a runtime-state record (success: `ok=True, last_ok_at=now, last_error=None`; failure incl. timeout: `ok=False, last_error=str(exc)[:300]`) via the store, then re-raise on failure. Timestamps via `datetime.now(timezone.utc).isoformat()`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/MCP/test_control_plane_lifecycle.py
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService


class FakeLocalService:
    def __init__(self, store: LocalMCPStore, *, connect_delay: float = 0.0,
                 connect_error: Exception | None = None) -> None:
        self.store = store
        self.connect_delay = connect_delay
        self.connect_error = connect_error
        self.calls: list[tuple[str, str]] = []

    def get_external_servers(self):
        return [
            {"profile_id": "docs", "command": "python", "args": [],
             "env_placeholders": {}, "discovery_snapshot": None, "is_connected": False}
        ]

    async def run_action(self, name, payload):  # matches control-plane delegation
        raise AssertionError("typed methods must not route through run_action in tests")

    def save_external_profile(self, payload):
        self.calls.append(("save", str(payload.get("profile_id"))))
        return dict(payload)

    def delete_external_profile(self, profile_id):
        self.calls.append(("delete", profile_id))
        return True

    async def connect_profile(self, profile_id):
        self.calls.append(("connect", profile_id))
        if self.connect_delay:
            await asyncio.sleep(self.connect_delay)
        if self.connect_error:
            raise self.connect_error
        return {"server_id": profile_id, "tools": [{"name": "a"}], "resources": [], "prompts": []}

    async def disconnect_profile(self, profile_id):
        self.calls.append(("disconnect", profile_id))
        return True

    async def test_external_profile(self, profile_id):
        self.calls.append(("test", profile_id))
        return {"ok": True, "profile_id": profile_id, "tools": 1, "resources": 0, "prompts": 0}

    async def refresh_external_profile(self, profile_id):
        self.calls.append(("refresh", profile_id))
        return {"server_id": profile_id, "tools": [], "resources": [], "prompts": []}


def _service(tmp_path: Path, **fake_kwargs) -> tuple[UnifiedMCPControlPlaneService, FakeLocalService, LocalMCPStore]:
    store = LocalMCPStore(tmp_path / "store.json")
    fake = FakeLocalService(store, **fake_kwargs)
    service = UnifiedMCPControlPlaneService(
        local_service=fake, server_service=None, target_store=None, context_store=None
    )
    return service, fake, store


@pytest.mark.asyncio
async def test_connect_success_records_ok(tmp_path, monkeypatch):
    service, fake, store = _service(tmp_path)
    result = await service.connect_local_profile("docs")
    assert result["server_id"] == "docs"
    record = store.get_profile_runtime_state("docs")
    assert record["ok"] is True and record["last_error"] is None
    assert record["last_action"] == "connect" and record["last_ok_at"]


@pytest.mark.asyncio
async def test_connect_failure_records_error_and_reraises(tmp_path):
    service, fake, store = _service(tmp_path, connect_error=RuntimeError("spawn failed"))
    with pytest.raises(RuntimeError, match="spawn failed"):
        await service.connect_local_profile("docs")
    record = store.get_profile_runtime_state("docs")
    assert record["ok"] is False and "spawn failed" in record["last_error"]


@pytest.mark.asyncio
async def test_connect_timeout_records_and_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.MCP.unified_control_plane_service.get_cli_setting",
        lambda section, key, default=None: 0.05,
    )
    service, fake, store = _service(tmp_path, connect_delay=1.0)
    with pytest.raises(RuntimeError, match="Timed out"):
        await service.connect_local_profile("docs")
    record = store.get_profile_runtime_state("docs")
    assert record["ok"] is False and "Timed out" in record["last_error"]


@pytest.mark.asyncio
async def test_local_external_catalog_merges_runtime_state(tmp_path):
    service, fake, store = _service(tmp_path)
    store.save_profile_runtime_state("docs", {"ok": False, "last_error": "boom"})
    catalog = await service.local_external_catalog()
    assert catalog[0]["profile_id"] == "docs"
    assert catalog[0]["runtime_state"]["last_error"] == "boom"


@pytest.mark.asyncio
async def test_save_and_delete_delegate(tmp_path):
    service, fake, store = _service(tmp_path)
    saved = await service.save_local_profile({"profile_id": "x", "command": "y"})
    assert saved["profile_id"] == "x"
    assert await service.delete_local_profile("x") is True
    assert ("save", "x") in fake.calls and ("delete", "x") in fake.calls
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/MCP/test_control_plane_lifecycle.py -v`
Expected: FAIL with `TypeError` (constructor kwargs) or `AttributeError: ... has no attribute 'connect_local_profile'` — check the actual `UnifiedMCPControlPlaneService.__init__` signature (line ~17) and adapt the test fixture's constructor call to it (keyword names verified in planning: `local_service`, `server_service`, `target_store`, `context_store`).

- [ ] **Step 3: Implement**

Append to `UnifiedMCPControlPlaneService` (imports: `import asyncio`, `from datetime import datetime, timezone`, `from tldw_chatbook.config import get_cli_setting` — reuse existing imports where present):

```python
    # ---- Typed local lifecycle/mutation seam (Phase 2) ----------------------
    # Shared by the Hub UI now and by the Phase 5 chat bridge / agent-runtime
    # MCPToolProvider (task-201) later. Governance enforcement stays inside
    # the local service exactly as run_action's branches rely on it.

    def _lifecycle_timeout(self) -> float:
        try:
            return float(get_cli_setting("mcp", "hub_lifecycle_timeout_seconds", 45))
        except (TypeError, ValueError):
            return 45.0

    def _record_local_attempt(self, profile_id: str, action: str, *,
                              ok: bool, error: str | None) -> None:
        store = getattr(self.local_service, "store", None)
        if store is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        previous = store.get_profile_runtime_state(profile_id) or {}
        store.save_profile_runtime_state(profile_id, {
            "last_attempt_at": now,
            "last_action": action,
            "ok": ok,
            "last_ok_at": now if ok else previous.get("last_ok_at"),
            "last_error": None if ok else (error or "")[:300],
        })

    async def _run_local_lifecycle(self, action: str, profile_id: str, coro):
        timeout = self._lifecycle_timeout()
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            message = f"Timed out after {timeout:.0f}s"
            self._record_local_attempt(profile_id, action, ok=False, error=message)
            raise RuntimeError(message) from None
        except asyncio.CancelledError:
            self._record_local_attempt(profile_id, action, ok=False, error="Cancelled")
            raise
        except Exception as exc:
            self._record_local_attempt(profile_id, action, ok=False, error=str(exc))
            raise
        self._record_local_attempt(profile_id, action, ok=True, error=None)
        return result

    async def connect_local_profile(self, profile_id: str) -> dict:
        return await self._run_local_lifecycle(
            "connect", profile_id, self.local_service.connect_profile(profile_id))

    async def disconnect_local_profile(self, profile_id: str) -> bool:
        return await self._run_local_lifecycle(
            "disconnect", profile_id, self.local_service.disconnect_profile(profile_id))

    async def test_local_profile(self, profile_id: str) -> dict:
        return await self._run_local_lifecycle(
            "test", profile_id, self.local_service.test_external_profile(profile_id))

    async def refresh_local_profile(self, profile_id: str) -> dict:
        return await self._run_local_lifecycle(
            "refresh", profile_id, self.local_service.refresh_external_profile(profile_id))

    async def save_local_profile(self, payload: dict) -> dict:
        return self.local_service.save_external_profile(dict(payload or {}))

    async def delete_local_profile(self, profile_id: str) -> bool:
        return bool(self.local_service.delete_external_profile(profile_id))

    async def local_external_catalog(self) -> list[dict]:
        records = list(self.local_service.get_external_servers() or [])
        store = getattr(self.local_service, "store", None)
        for record in records:
            profile_id = str(record.get("profile_id") or "")
            record["runtime_state"] = (
                store.get_profile_runtime_state(profile_id) if store else None
            )
        return records
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/MCP/test_control_plane_lifecycle.py Tests/MCP/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/unified_control_plane_service.py Tests/MCP/test_control_plane_lifecycle.py
git commit -m "feat(mcp): typed local lifecycle methods with wall-clock timeout and attempt recording"
```

---

### Task 3: Readiness consumes runtime state; `as_checking`; CSS class map

**Files:**
- Modify: `tldw_chatbook/MCP/readiness.py`
- Test: extend `Tests/MCP/test_readiness_derivation.py`, `Tests/MCP/test_readiness_model.py`

**Interfaces:**
- Consumes: T2's catalog record shape (`record["runtime_state"]`).
- Produces: `local_profile_readiness` honors `runtime_state`: a record with `ok is False and last_error` adds `ReasonCode.DISCOVERY_FAILED` (unless `AUTH_MISSING` already applies — priority table handles ordering) and the message becomes the stored `last_error`; a record with `ok is True` leaves derivation unchanged but sets `detail["last_ok_at"]`. New: `as_checking(snapshot: ReadinessSnapshot, action: str) -> ReadinessSnapshot` (frozen-replace: `state=CHECKING`, `reasons=()`, `message=f"Working — {action}…"`). New: `STATE_CSS_CLASSES: dict[ReadinessState, str]` → `{READY: "mcp-status-ready", CHECKING: "mcp-status-info", NEEDS_SETUP: "mcp-status-warning", NEEDS_ATTENTION: "mcp-status-error", NO_TOOLS: "mcp-status-warning", STALE: "mcp-status-warning"}` (T11/T13 add the CSS rules).

- [ ] **Step 1: Failing tests (append)**

```python
# append to Tests/MCP/test_readiness_derivation.py
def test_runtime_error_drives_needs_attention_with_stored_message():
    record = _local_record(
        discovery_snapshot={"tools": [{"name": "a"}], "resources": [], "prompts": []},
        is_connected=False,
    )
    record["runtime_state"] = {"ok": False, "last_error": "Timed out after 45s",
                               "last_action": "connect", "last_attempt_at": "t", "last_ok_at": None}
    snap = local_profile_readiness(record, environ={})
    assert snap.state is ReadinessState.NEEDS_ATTENTION
    assert snap.primary_reason is ReasonCode.DISCOVERY_FAILED
    assert "Timed out" in snap.message


def test_runtime_ok_keeps_normal_derivation():
    record = _local_record(
        discovery_snapshot={"tools": [{"name": "a"}], "resources": [], "prompts": []},
        is_connected=True,
    )
    record["runtime_state"] = {"ok": True, "last_error": None, "last_ok_at": "2026-07-14T00:00:00Z"}
    snap = local_profile_readiness(record, environ={})
    assert snap.state is ReadinessState.READY
    assert snap.detail["last_ok_at"] == "2026-07-14T00:00:00Z"


def test_auth_missing_outranks_runtime_error():
    record = _local_record(env_placeholders={"API_KEY": "$MISSING"})
    record["runtime_state"] = {"ok": False, "last_error": "boom"}
    snap = local_profile_readiness(record, environ={})
    assert snap.primary_reason is ReasonCode.AUTH_MISSING
```

```python
# append to Tests/MCP/test_readiness_model.py
from tldw_chatbook.MCP.readiness import STATE_CSS_CLASSES, as_checking


def test_state_css_classes_complete():
    assert set(STATE_CSS_CLASSES) == set(ReadinessState)
    assert all(v.startswith("mcp-status-") for v in STATE_CSS_CLASSES.values())


def test_as_checking_replaces_state_and_message():
    snap = _snap(ReadinessState.READY)
    checking = as_checking(snap, "connect")
    assert checking.state is ReadinessState.CHECKING
    assert checking.reasons == ()
    assert "connect" in checking.message
    assert checking.server_key == snap.server_key
```

- [ ] **Step 2: Run to verify failure** — `PYTHONPATH=. .../pytest Tests/MCP/test_readiness_derivation.py Tests/MCP/test_readiness_model.py -q` → FAIL (`ImportError: as_checking`, runtime assertions).

- [ ] **Step 3: Implement** — in `readiness.py`: add `import dataclasses` usage; inside `local_profile_readiness`, after the existing reason collection and before `resolve_state`:

```python
    runtime_state = record.get("runtime_state") or {}
    runtime_error = None
    if isinstance(runtime_state, dict) and runtime_state.get("ok") is False:
        runtime_error = str(runtime_state.get("last_error") or "").strip() or None
    if runtime_error:
        reasons.append(ReasonCode.DISCOVERY_FAILED)
```
Then in the message selection chain, insert the runtime-error branch AFTER the `missing` (auth) branch and BEFORE the snapshot-based branches:
```python
    elif runtime_error:
        message = runtime_error
```
And add to `detail`: `"last_ok_at": (runtime_state.get("last_ok_at") if isinstance(runtime_state, dict) else None),`.

Append module-level:
```python
STATE_CSS_CLASSES: dict[ReadinessState, str] = {
    ReadinessState.READY: "mcp-status-ready",
    ReadinessState.CHECKING: "mcp-status-info",
    ReadinessState.NEEDS_SETUP: "mcp-status-warning",
    ReadinessState.NEEDS_ATTENTION: "mcp-status-error",
    ReadinessState.NO_TOOLS: "mcp-status-warning",
    ReadinessState.STALE: "mcp-status-warning",
}


def as_checking(snapshot: ReadinessSnapshot, action: str) -> ReadinessSnapshot:
    """Return a copy of a snapshot marked as an in-flight lifecycle check.

    Args:
        snapshot: The snapshot being operated on.
        action: Human verb for the in-flight operation (e.g. "connect").

    Returns:
        A frozen copy with state=CHECKING, no reasons, and a working message.
    """
    import dataclasses
    return dataclasses.replace(
        snapshot, state=ReadinessState.CHECKING, reasons=(),
        message=f"Working — {action}…",
    )
```

- [ ] **Step 4: Run** — both files + `Tests/MCP/` green.
- [ ] **Step 5: Commit** — `feat(mcp): readiness consumes persisted attempt state; add as_checking + status CSS map`

---

### Task 4: Watch-item hardening (one-shot echo guard; inspector lock; zero-action hint)

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_rail.py`, `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`
- Test: extend `Tests/UI/test_mcp_rail.py`, `Tests/UI/test_mcp_inspector.py`

**Interfaces:**
- Consumes: existing echo-suppression sentinels in `MCPRail.on_select_changed` (`_displayed_scope_value`/`_displayed_scope_ref_value`), `MCPInspector.update_readiness` (async since Phase 1 fix).
- Produces: acceptance criteria from `Docs/Design/mcp-hub-phase2-ux-inputs.md` — (a) scope sequence A→B→A with no recompose dispatches **three** `ScopeChanged` messages while the mount echo still dispatches zero: implement one-shot consumption (after a suppressed echo match, set the sentinel to a module-level `_ECHO_CONSUMED = object()` so later equal values dispatch); (b) `MCPInspector._refresh_lock: asyncio.Lock` — `update_readiness` body (remove+mount) runs under `async with self._refresh_lock`, so a worker-driven refresh interleaved with a pump-driven one can never produce `DuplicateIds` and the last writer's buttons win exactly once; (c) `#mcp-adv-empty-hint` Static (guidance copy, shown only when the Advanced action select has zero descriptors): "No actions for this section. Select External Servers or Inventory to see runnable actions."

- [ ] **Step 1: Failing tests**

```python
# append to Tests/UI/test_mcp_rail.py
# NOTE: RailApp (Phase 1 harness) collects only ServerSelected/SourceChanged —
# add this handler to it first:
#     def on_mcp_rail_scope_changed(self, event: MCPRail.ScopeChanged) -> None:
#         self.events.append(event)
@pytest.mark.asyncio
async def test_scope_a_b_a_dispatches_three_changes_and_mount_echo_zero():
    app = RailApp()
    async with app.run_test() as pilot:
        rail = app.query_one(MCPRail)
        rail.sync_state(
            source="server",
            snapshots=[_snap("server:main", "Main Server")],
            selected_server_key=None,
            scope_options=[("Personal", "personal"), ("Team", "team")],
            scope_value="personal",
            scope_ref_options=[],
            scope_ref_value=None,
        )
        await pilot.pause()
        changes = [e for e in app.events if isinstance(e, MCPRail.ScopeChanged)]
        assert changes == []  # mount echo suppressed
        select = app.query_one("#mcp-rail-scope-select", Select)
        select.value = "team"       # A -> B
        await pilot.pause()
        select.value = "personal"   # B -> A (must NOT be swallowed as echo)
        await pilot.pause()
        select.value = "team"       # A -> B again
        await pilot.pause()
        changes = [e.scope for e in app.events if isinstance(e, MCPRail.ScopeChanged)]
        assert changes == ["team", "personal", "team"]
```

```python
# append to Tests/UI/test_mcp_inspector.py
@pytest.mark.asyncio
async def test_concurrent_refreshes_serialize_and_last_writer_wins():
    app = InspectorApp()
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        first = _stale_snap()
        second = ReadinessSnapshot(
            server_key="local:web", label="web", source="local",
            state=ReadinessState.READY, reasons=(), message="Connected.",
        )
        await asyncio.gather(
            inspector.update_readiness(first),
            inspector.update_readiness(second),
        )
        await pilot.pause()
        buttons = list(app.query("Button.mcp-inspector-action"))
        assert buttons, "actions must render"
        # last writer wins exactly once: READY action set, no duplicates
        ids = [b.id for b in buttons]
        assert len(ids) == len(set(ids))
        assert inspector._snapshot.server_key == "local:web"


@pytest.mark.asyncio
async def test_zero_descriptor_sections_show_guidance_hint():
    app = InspectorApp()  # FakeAdvService returns one action; override to none
    app.service.available_actions = lambda: []
    async with app.run_test() as pilot:
        inspector = app.query_one(MCPInspector)
        inspector.set_service_context(app.service, [("Overview", "overview")])
        await pilot.pause()
        hint = app.query_one("#mcp-adv-empty-hint", Static)
        assert hint.display
        assert "Inventory" in str(hint.renderable)
```

(Add `import asyncio` and the `ReadinessSnapshot/ReadinessState` imports at the top of the test file if missing.)

- [ ] **Step 2: Run to verify failure** — the A→B→A test fails (second "personal" swallowed by the standing sentinel); the hint test fails (`NoMatches`).

- [ ] **Step 3: Implement**
  - `mcp_rail.py`: module-level `_ECHO_CONSUMED = object()`. In `on_select_changed`'s scope branch: if `event.value == self._displayed_scope_value and self._displayed_scope_value is not _ECHO_CONSUMED`: consume (`self._displayed_scope_value = _ECHO_CONSUMED`) and return; otherwise dispatch. Same for scope-ref sentinel. `compose()` keeps setting the sentinels to the clamped display values (fresh per recompose).
  - `mcp_inspector.py`: `self._refresh_lock = asyncio.Lock()` in `__init__`; wrap the whole `update_readiness` body in `async with self._refresh_lock:`. Add to `compose()`, right after the action Select: `yield Static("", id="mcp-adv-empty-hint", classes="ds-field-row", markup=False)`; in `_refresh_advanced_actions` zero-descriptor branch set text "No actions for this section. Select External Servers or Inventory to see runnable actions." + `display = True`; in the populated branch set `display = False`.

- [ ] **Step 4: Run** — `Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_inspector.py -q` all green.
- [ ] **Step 5: Commit** — `fix(mcp-hub): one-shot scope echo guard, serialized inspector refresh, zero-action guidance`

---

### Task 5: Lifecycle actions wired (workers, CHECKING, cancel)

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`, `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`
- Test: extend `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: T2 typed methods; T3 `as_checking`; existing `HubAction` enum + inspector `_WIRED_ACTIONS`.
- Produces:
  - `_WIRED_ACTIONS` grows `{CONNECT, VALIDATE, REFRESH_DISCOVERY}` **for local-source snapshots only** (inspector receives `snapshot.source`; server-source keeps the disabled tooltip "Managed on the server — use Advanced.").
  - `MCPWorkbench._in_flight: dict[str, Worker]`; handler maps `HubAction.CONNECT → connect_local_profile`, `VALIDATE → test_local_profile`, `REFRESH_DISCOVERY → refresh_local_profile`. Each runs in `run_worker(..., group=f"mcp-lifecycle", exclusive=False)` guarded by "already in flight → notify warning". While in flight, `_snapshot_for` results for that key are wrapped with `as_checking`, and `_sync_children()` renders the CHECKING badge; the inspector shows a **Cancel** button (`#mcp-inspector-cancel`, message `MCPInspector.CancelRequested(server_key)`, namespace already `mcp_inspector`) whenever the rendered snapshot state is CHECKING.
  - On completion: pop in-flight, `notify` success (e.g. "docs: connected — 3 tools.") or error (severity="error", store message already recorded by T2), refresh catalog + `_sync_children()`.
  - `MCPWorkbench` catalog source switches from `load_section("external_servers")` to `service.local_external_catalog()` when available (fallback to the old path via `getattr`).
  - Disconnect: rendered in the server detail (T7 area) — the inspector's readiness actions don't include disconnect (not in the reason→action table); expose it as a detail-view button `#mcp-detail-disconnect` shown only when `snapshot.is_connected` is True, wired to `disconnect_local_profile` through the same in-flight machinery.

- [ ] **Step 1: Failing tests**

```python
# append to Tests/UI/test_mcp_workbench.py
class LifecycleFakeHubService(FakeHubService):
    def __init__(self) -> None:
        super().__init__()
        self.lifecycle_calls: list[tuple[str, str]] = []
        self.connect_gate: asyncio.Event | None = None

    async def local_external_catalog(self):
        return await self.load_section("external_servers")

    async def connect_local_profile(self, profile_id):
        self.lifecycle_calls.append(("connect", profile_id))
        if self.connect_gate is not None:
            await self.connect_gate.wait()
        return {"server_id": profile_id, "tools": [{"name": "a"}], "resources": [], "prompts": []}

    async def test_local_profile(self, profile_id):
        self.lifecycle_calls.append(("test", profile_id))
        return {"ok": True, "profile_id": profile_id, "tools": 1, "resources": 0, "prompts": 0}

    async def refresh_local_profile(self, profile_id):
        self.lifecycle_calls.append(("refresh", profile_id))
        return {"server_id": profile_id, "tools": [], "resources": [], "prompts": []}


class LifecycleApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.unified_mcp_service = LifecycleFakeHubService()

    def compose(self) -> ComposeResult:
        yield MCPWorkbench(app_instance=self, id="mcp-workbench")


@pytest.mark.asyncio
async def test_connect_action_runs_lifecycle_and_notifies():
    app = LifecycleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        # select docs (local profile with snapshot -> stale -> CONNECT is a wired action)
        await pilot.click(f"#{MCP_RAIL_ROW_PREFIX}2")
        await pilot.pause()
        # docs fixture in FakeHubService has a snapshot and is_connected True -> READY;
        # flip fixture to disconnected for this app so CONNECT appears:
        assert True  # (fixture adjustment below makes this meaningful)


@pytest.mark.asyncio
async def test_in_flight_shows_checking_and_cancel_then_completes():
    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        selected = workbench._snapshot_for_display("local:docs")
        assert selected.state.value == "checking"
        assert list(app.query("#mcp-inspector-cancel"))
        app.unified_mcp_service.connect_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("connect", "docs") in app.unified_mcp_service.lifecycle_calls
        assert "local:docs" not in workbench._in_flight


@pytest.mark.asyncio
async def test_cancel_requested_cancels_worker():
    app = LifecycleApp()
    app.unified_mcp_service.connect_gate = asyncio.Event()  # never set -> hangs
    async with app.run_test() as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench._selected_server_key = "local:docs"
        workbench._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        workbench.on_mcp_inspector_cancel_requested(
            MCPInspector.CancelRequested("local:docs")
        )
        await pilot.pause()
        assert "local:docs" not in workbench._in_flight
```

Adjust the first test during implementation: change `LifecycleFakeHubService.load_section`'s docs fixture to `"is_connected": False` so docs derives STALE (CONNECT wired) and assert that clicking the inspector's `#mcp-inspector-action-connect` records `("connect", "docs")`. Import `MCPInspector` and `asyncio` at top of file.

- [ ] **Step 2: Run to verify failure** — `AttributeError: _start_lifecycle` etc.

- [ ] **Step 3: Implement**
  - `mcp_inspector.py`: change `update_readiness` signature usage — `_WIRED_ACTIONS` becomes a function `def _wired_actions(snapshot)` returning the Phase 1 set plus `{HubAction.CONNECT, HubAction.VALIDATE, HubAction.REFRESH_DISCOVERY}` when `snapshot.source == "local"`. Disabled tooltip for still-unwired actions: keep Phase 1 copy; for server-source lifecycle actions use "Managed on the server — use Advanced." Add `class CancelRequested(Message, namespace="mcp_inspector")` with `server_key`. In `update_readiness`, when `snapshot.state is ReadinessState.CHECKING`, mount a single enabled Button "Cancel" (`id="mcp-inspector-cancel"`, classes `mcp-inspector-action console-action-secondary`, tooltip "Cancel the in-flight operation.") instead of the action set; `on_button_pressed` posts `CancelRequested(self._snapshot.server_key)` for it.
  - `mcp_workbench.py`:
    - `self._in_flight: dict[str, Worker] = {}` and `self._in_flight_action: dict[str, str] = {}` in `__init__`.
    - `_snapshot_for_display(key)` = `_snapshot_for(key)` wrapped with `as_checking(snap, self._in_flight_action[key])` when key in `_in_flight`; `_sync_children` uses it for detail/inspector, and `_collect_snapshots` results pass through a list comprehension applying the same wrap for rail/table.
    - `_start_lifecycle(server_key, profile_id, action)`: guard `server_key in self._in_flight` → `self.app.notify(f"{profile_id}: {action} already running.", severity="warning")`; else build coro per action from `{"connect": service.connect_local_profile, "test": service.test_local_profile, "refresh": service.refresh_local_profile, "disconnect": service.disconnect_local_profile}`, wrap in `self._lifecycle_wrapper(...)` coroutine that try/except/finally: pops in-flight, notifies (`severity="error"` with `str(exc)` on failure; success message per action), then `await self.reload()`-lite (`self._snapshots = await self._collect_snapshots(); self._sync_children()`). Store worker: `self._in_flight[server_key] = self.run_worker(wrapper, group="mcp-lifecycle", exclusive=False)`; `self._in_flight_action[server_key] = action`; `self._sync_children()`.
    - Extend `on_mcp_inspector_hub_action_requested`: CONNECT/VALIDATE/REFRESH_DISCOVERY on a `local:` key → `_start_lifecycle(key, key.split(":",1)[1], verb)` with verb map `{CONNECT: "connect", VALIDATE: "test", REFRESH_DISCOVERY: "refresh"}`.
    - `def on_mcp_inspector_cancel_requested(self, event)`: worker = `self._in_flight.pop(event.server_key, None)`; `self._in_flight_action.pop(...)`; `worker.cancel()` if worker; notify "Cancelled."; `self._sync_children()`.
    - Catalog: in `_collect_snapshots`, prefer `catalog = await service.local_external_catalog()` when `callable(getattr(service, "local_external_catalog", None))`, falling back to the Phase 1 `load_section` path.

- [ ] **Step 4: Run** — workbench + inspector + rail suites green.
- [ ] **Step 5: Commit** — `feat(mcp-hub): wire connect/test/refresh lifecycle with in-flight state and cancel`

---

### Task 6: Local profile add/edit form

**Files:**
- Create: `tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py` (host the form), `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (orchestrate)
- Test: `Tests/UI/test_mcp_profile_form.py`

**Interfaces:**
- Consumes: T2 `save_local_profile` (raises `ValueError` with store copy); form conventions (Static label + Input + one error Static + state-driven buttons).
- Produces: `MCPProfileForm(Vertical)`:
  - `__init__(self, *, profile: dict | None = None, **kwargs)` — None = add mode (id editable), dict = edit mode (id disabled, fields prefilled from `profile_id/command/args/env_placeholders/env_literals`).
  - Widgets: `#mcp-form-id` Input, `#mcp-form-command` Input, `#mcp-form-args` TextArea (one arg per line), `#mcp-form-env` TextArea (one `KEY=value` per line; `$VAR`/`${VAR}` values become `env_placeholders`, others `env_literals`), `#mcp-form-error` Static (markup=False), Save `#mcp-form-save` (console-action-primary) / Cancel `#mcp-form-cancel` (console-action-secondary).
  - `build_payload(self) -> dict` — parses fields into the verified save keys: `{"profile_id", "command", "args": [lines], "env_placeholders": {...}, "env_literals": {...}}`; raises `ValueError` for a malformed env line ("Env line 3 must look like KEY=value.").
  - Guidance Static above env field: "Secrets are never stored — reference them as KEY=$ENV_VAR and export the variable before connecting."
  - Messages (namespace `"mcp_profile_form"`): `SubmitRequested(payload: dict)`, `Cancelled()`.
  - `show_error(text: str)` — sets `#mcp-form-error`.
  - Hosting: `MCPServersMode.show_form(profile: dict | None)` mounts the form in a `#mcp-servers-form` container (overview/detail hidden while shown), `hide_form()` restores. Overview gains `#mcp-add-server` Button ("Add server", console-action-primary) posting `MCPServersMode.AddServerRequested()` (namespace `mcp_servers_mode`).
  - Workbench: handles `AddServerRequested` → `show_form(None)`; inspector `HubAction.EDIT_CONFIG` on a local key → `show_form(<catalog record>)`; `SubmitRequested` → `await service.save_local_profile(payload)` — on `ValueError` call `form.show_error(str(exc))`; on success `hide_form()`, notify "Saved {id}.", reload catalog + sync; `Cancelled` → `hide_form()`.

- [ ] **Step 1: Failing tests**

```python
# Tests/UI/test_mcp_profile_form.py
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static, TextArea

from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPProfileForm


class FormApp(App):
    def __init__(self, profile=None) -> None:
        super().__init__()
        self.profile = profile
        self.events: list = []

    def compose(self) -> ComposeResult:
        yield MCPProfileForm(profile=self.profile, id="form")

    def on_mcp_profile_form_submit_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_profile_form_cancelled(self, event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_build_payload_splits_env_into_placeholders_and_literals():
    app = FormApp()
    async with app.run_test() as pilot:
        form = app.query_one(MCPProfileForm)
        app.query_one("#mcp-form-id", Input).value = "docs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one("#mcp-form-args", TextArea).text = "-y\n@modelcontextprotocol/server-filesystem"
        app.query_one("#mcp-form-env", TextArea).text = "API_KEY=$MY_KEY\nDEBUG=true"
        payload = form.build_payload()
        assert payload == {
            "profile_id": "docs", "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "env_placeholders": {"API_KEY": "$MY_KEY"},
            "env_literals": {"DEBUG": "true"},
        }


@pytest.mark.asyncio
async def test_malformed_env_line_raises_with_line_number():
    app = FormApp()
    async with app.run_test() as pilot:
        form = app.query_one(MCPProfileForm)
        app.query_one("#mcp-form-id", Input).value = "docs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one("#mcp-form-env", TextArea).text = "API_KEY=$K\nnot-an-env-line"
        with pytest.raises(ValueError, match="line 2"):
            form.build_payload()


@pytest.mark.asyncio
async def test_save_posts_submit_and_edit_mode_locks_id():
    profile = {"profile_id": "docs", "command": "npx", "args": ["-y"],
               "env_placeholders": {"K": "$V"}, "env_literals": {}}
    app = FormApp(profile=profile)
    async with app.run_test() as pilot:
        assert app.query_one("#mcp-form-id", Input).disabled
        assert app.query_one("#mcp-form-command", Input).value == "npx"
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        assert app.events and app.events[-1].payload["profile_id"] == "docs"


@pytest.mark.asyncio
async def test_show_error_renders_store_copy():
    app = FormApp()
    async with app.run_test() as pilot:
        form = app.query_one(MCPProfileForm)
        form.show_error("Secret-bearing env key 'API_KEY' cannot be stored as a literal")
        await pilot.pause()
        assert "cannot be stored" in str(app.query_one("#mcp-form-error", Static).renderable)
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py
"""Inline add/edit form for local MCP server profiles (stdio-only)."""

from __future__ import annotations

import re
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Static, TextArea

_ENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_PLACEHOLDER_VALUE_RE = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$")


class MCPProfileForm(Vertical):
    """State-driven inline form; validation errors surface in one Static."""

    DEFAULT_CSS = """
    MCPProfileForm { height: auto; min-height: 0; }
    #mcp-form-args, #mcp-form-env { height: 4; min-height: 2; }
    """

    class SubmitRequested(Message, namespace="mcp_profile_form"):
        def __init__(self, payload: dict[str, Any]) -> None:
            super().__init__()
            self.payload = payload

    class Cancelled(Message, namespace="mcp_profile_form"):
        pass

    def __init__(self, *, profile: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._profile = dict(profile) if profile else None

    @property
    def is_edit(self) -> bool:
        return self._profile is not None

    def compose(self) -> ComposeResult:
        title = "Edit server" if self.is_edit else "Add local server (stdio)"
        yield Static(title, classes="destination-section", markup=False)
        profile = self._profile or {}
        yield Static("Profile id", classes="form-label")
        id_input = Input(value=str(profile.get("profile_id") or ""), id="mcp-form-id",
                         placeholder="docs-server")
        id_input.disabled = self.is_edit
        yield id_input
        yield Static("Command", classes="form-label")
        yield Input(value=str(profile.get("command") or ""), id="mcp-form-command",
                    placeholder="npx")
        yield Static("Args — one per line", classes="form-label")
        yield TextArea("\n".join(str(a) for a in profile.get("args") or []),
                       id="mcp-form-args")
        yield Static("Env — one KEY=value per line", classes="form-label")
        yield Static(
            "Secrets are never stored — reference them as KEY=$ENV_VAR and export "
            "the variable before connecting.",
            classes="ds-field-row", markup=False,
        )
        env_lines = [f"{k}={v}" for k, v in (profile.get("env_placeholders") or {}).items()]
        env_lines += [f"{k}={v}" for k, v in (profile.get("env_literals") or {}).items()]
        yield TextArea("\n".join(env_lines), id="mcp-form-env")
        yield Static("", id="mcp-form-error", classes="ds-field-row", markup=False)
        with Horizontal(classes="ds-toolbar"):
            yield Button("Save", id="mcp-form-save", classes="console-action-primary",
                         compact=True, tooltip="Validate and save this profile.")
            yield Button("Cancel", id="mcp-form-cancel", classes="console-action-secondary",
                         compact=True, tooltip="Discard changes.")

    def build_payload(self) -> dict[str, Any]:
        """Parse the form into the store's exact save-payload keys.

        Returns:
            Payload with profile_id/command/args/env_placeholders/env_literals.

        Raises:
            ValueError: A malformed env line (with its 1-based line number).
        """
        placeholders: dict[str, str] = {}
        literals: dict[str, str] = {}
        env_text = self.query_one("#mcp-form-env", TextArea).text
        for index, raw_line in enumerate(env_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = _ENV_LINE_RE.match(line)
            if not match:
                raise ValueError(f"Env line {index} must look like KEY=value.")
            key, value = match.group(1), match.group(2).strip()
            if _PLACEHOLDER_VALUE_RE.match(value):
                placeholders[key] = value
            else:
                literals[key] = value
        args = [line.strip() for line in
                self.query_one("#mcp-form-args", TextArea).text.splitlines() if line.strip()]
        return {
            "profile_id": self.query_one("#mcp-form-id", Input).value.strip(),
            "command": self.query_one("#mcp-form-command", Input).value.strip(),
            "args": args,
            "env_placeholders": placeholders,
            "env_literals": literals,
        }

    def show_error(self, text: str) -> None:
        self.query_one("#mcp-form-error", Static).update(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-form-save":
            event.stop()
            try:
                payload = self.build_payload()
            except ValueError as exc:
                self.show_error(str(exc))
                return
            self.post_message(self.SubmitRequested(payload))
        elif event.button.id == "mcp-form-cancel":
            event.stop()
            self.post_message(self.Cancelled())
```

Hosting (`mcp_servers_mode.py`): add `yield Vertical(id="mcp-servers-form")` alongside overview/detail containers; `async def show_form(self, profile: dict | None)`: hide overview+detail, `await container.remove_children()`, `container.mount(MCPProfileForm(profile=profile))`, `container.display = True`; `def hide_form()`: clear + restore previous view (overview if no `_detail_snapshot` else detail). Overview compose gains `Button("Add server", id="mcp-add-server", classes="console-action-primary", compact=True, tooltip="Create a new local stdio server profile.")` above the summary; `on_button_pressed` posts `AddServerRequested` (new Message, namespace `mcp_servers_mode`). Workbench wiring per the Interfaces block (its `on_mcp_profile_form_submit_requested` runs `run_worker` with a small async closure calling `save_local_profile`, since handlers stay non-blocking).

- [ ] **Step 4: Run** — new file + `Tests/UI/test_mcp_servers_mode.py` + workbench suite green.
- [ ] **Step 5: Commit** — `feat(mcp-hub): local profile add/edit form with placeholder-first env parsing`

---

### Task 7: Delete (arm-then-confirm) + disconnect button in detail

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py`, `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Test: extend `Tests/UI/test_mcp_servers_mode.py`

**Interfaces:**
- Consumes: T2 `delete_local_profile`, `disconnect_local_profile` (via T5 `_start_lifecycle` for disconnect); Library arm-then-confirm precedent.
- Produces: detail view (local snapshots only) gains a `ds-toolbar` row: `#mcp-detail-edit` ("Edit"), `#mcp-detail-disconnect` ("Disconnect", shown only when `snapshot.is_connected`), `#mcp-detail-delete` ("Delete"). First Delete press arms: the row recomposes to `#mcp-detail-delete-confirm` ("Confirm delete", console-action-primary) + `#mcp-detail-delete-cancel` ("Keep", console-action-secondary); confirm posts `MCPServersMode.DeleteConfirmed(server_key)` (namespace `mcp_servers_mode`); any other interaction disarms. Edit posts the existing edit path (workbench `show_form(record)`). Workbench handles `DeleteConfirmed` → `delete_local_profile(profile_id)` in a worker → notify "Deleted {id}.", clear selection, reload catalog, `_sync_children()`. Built-in and server-source detail views do NOT render this toolbar (source checks).

- [ ] **Step 1: Failing tests**

```python
# append to Tests/UI/test_mcp_servers_mode.py
@pytest.mark.asyncio
async def test_delete_requires_arm_then_confirm():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        snap = _snap("local:docs", "docs",
                     detail={"command": "npx", "args": [], "env_placeholders": {},
                             "missing_env": [], "discovery_snapshot": None})
        await canvas.show_detail(snap)
        await pilot.pause()
        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        assert list(app.query("#mcp-detail-delete-confirm"))
        assert not app.events  # nothing posted yet
        await pilot.click("#mcp-detail-delete-cancel")
        await pilot.pause()
        assert list(app.query("#mcp-detail-delete"))  # disarmed
        await pilot.click("#mcp-detail-delete")
        await pilot.pause()
        await pilot.click("#mcp-detail-delete-confirm")
        await pilot.pause()
        confirmed = [e for e in app.events if type(e).__name__ == "DeleteConfirmed"]
        assert confirmed and confirmed[-1].server_key == "local:docs"


@pytest.mark.asyncio
async def test_builtin_detail_has_no_delete_toolbar():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(builtin_readiness(enabled=True))
        await pilot.pause()
        assert not list(app.query("#mcp-detail-delete"))
```

(`CanvasApp` collects `DeleteConfirmed` via `on_mcp_servers_mode_delete_confirmed` appended to its handler set; `show_detail` became async in the Phase 1 fixes — await it.)

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** per Interfaces: `self._delete_armed: bool = False` on `MCPServersMode`, reset in `show_detail`; the toolbar renders from `_detail_toolbar()` into a dedicated `#mcp-detail-toolbar` Horizontal rebuilt (awaited remove + mount) on arm/disarm; every button carries a tooltip ("Delete this profile — asks to confirm.", "Confirm permanent deletion.", "Keep the profile."). Workbench `on_mcp_servers_mode_delete_confirmed` runs the delete worker as described.
- [ ] **Step 4: Run** — servers-mode + workbench suites green.
- [ ] **Step 5: Commit** — `feat(mcp-hub): arm-then-confirm profile delete and detail toolbar`

---

### Task 8: Import from `mcpServers` JSON (paste or file)

**Files:**
- Create: `tldw_chatbook/MCP/mcp_import.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py` (import panel variant), `mcp_servers_mode.py` + `mcp_workbench.py` (hosting/orchestration)
- Test: `Tests/MCP/test_mcp_import.py`, extend `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: `LocalExternalMCPProfile.from_input_dict` for authoritative validation (import it); T2 `save_local_profile`; `EnhancedFileOpen` convention.
- Produces (`tldw_chatbook/MCP/mcp_import.py`):
  - `@dataclass ImportCandidate: profile_id: str; command: str; args: list[str]; env_placeholders: dict[str,str]; env_literals: dict[str,str]; warnings: list[str]`
  - `parse_mcp_servers_json(text: str, *, existing_ids: set[str] = frozenset()) -> list[ImportCandidate]` — parses `{"mcpServers": {name: {command, args?, env?}}}`; raises `ValueError` on invalid JSON ("Not valid JSON: …") or missing `mcpServers` key. Env classification per entry: `$VAR`-shaped values → placeholders; otherwise TRY the value as a literal by round-tripping `LocalExternalMCPProfile.from_input_dict({..., "env_literals": {key: value}})` — on `ValueError`, convert to placeholder `f"${key}"` and append warning `f"{key}: value can't be stored — will reference ${key}; export it before connecting."`. `profile_id in existing_ids` → warning "will overwrite the existing profile." `to_payload(self) -> dict` on the candidate returns the exact save keys.
  - UI: `MCPImportPanel(Vertical)` in `mcp_profile_form.py` — `#mcp-import-text` TextArea, `#mcp-import-file` Button ("From file…") posting `FileRequested()` (workbench pushes `EnhancedFileOpen(location=".", title="Select MCP config JSON", filters=Filters(("JSON", lambda p: p.suffix.lower() == ".json")), context="mcp_import")` and writes the file's text into the TextArea), `#mcp-import-preview` Button → renders candidates into `#mcp-import-list` (one Static per candidate: id, command, warnings — markup=False), `#mcp-import-apply` Button ("Import N servers", disabled until preview succeeds) posting `ImportRequested(candidates)` (namespace `mcp_import_panel`), `#mcp-import-error` Static. Workbench applies each candidate via `save_local_profile(candidate.to_payload())`, collecting per-id failures into a notify summary.

- [ ] **Step 1: Failing tests**

```python
# Tests/MCP/test_mcp_import.py
from __future__ import annotations

import json

import pytest

from tldw_chatbook.MCP.mcp_import import ImportCandidate, parse_mcp_servers_json


def test_parses_command_args_env_and_placeholder_passthrough():
    text = json.dumps({"mcpServers": {"docs": {
        "command": "npx", "args": ["-y", "pkg"], "env": {"WORKSPACE": "$HOME"}}}})
    [candidate] = parse_mcp_servers_json(text)
    assert candidate.profile_id == "docs"
    assert candidate.args == ["-y", "pkg"]
    assert candidate.env_placeholders == {"WORKSPACE": "$HOME"}
    assert candidate.env_literals == {} and candidate.warnings == []


def test_secret_shaped_literal_becomes_placeholder_with_warning():
    text = json.dumps({"mcpServers": {"web": {
        "command": "npx", "env": {"API_KEY": "sk-live-123456"}}}})
    [candidate] = parse_mcp_servers_json(text)
    assert candidate.env_placeholders == {"API_KEY": "$API_KEY"}
    assert candidate.env_literals == {}
    assert any("export it before connecting" in w for w in candidate.warnings)


def test_safe_literal_survives_and_overwrite_warning():
    text = json.dumps({"mcpServers": {"docs": {"command": "npx", "env": {"DEBUG": "true"}}}})
    [candidate] = parse_mcp_servers_json(text, existing_ids={"docs"})
    assert candidate.env_literals == {"DEBUG": "true"}
    assert any("overwrite" in w for w in candidate.warnings)


def test_invalid_json_and_missing_key_raise():
    with pytest.raises(ValueError, match="Not valid JSON"):
        parse_mcp_servers_json("{nope")
    with pytest.raises(ValueError, match="mcpServers"):
        parse_mcp_servers_json(json.dumps({"servers": {}}))


def test_to_payload_uses_exact_store_keys():
    text = json.dumps({"mcpServers": {"docs": {"command": "npx"}}})
    [candidate] = parse_mcp_servers_json(text)
    assert set(candidate.to_payload()) == {
        "profile_id", "command", "args", "env_placeholders", "env_literals"}
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/MCP/mcp_import.py
"""Parse Claude-Desktop-style {"mcpServers": ...} JSON into local profile payloads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile

_PLACEHOLDER_RE = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$")


@dataclass
class ImportCandidate:
    profile_id: str
    command: str
    args: list[str] = field(default_factory=list)
    env_placeholders: dict[str, str] = field(default_factory=dict)
    env_literals: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Return the exact save-payload keys the local store accepts."""
        return {
            "profile_id": self.profile_id,
            "command": self.command,
            "args": list(self.args),
            "env_placeholders": dict(self.env_placeholders),
            "env_literals": dict(self.env_literals),
        }


def _literal_is_storable(profile_id: str, command: str, key: str, value: str) -> bool:
    """Authoritative check: round-trip the store's own validation."""
    try:
        LocalExternalMCPProfile.from_input_dict({
            "profile_id": profile_id or "candidate", "command": command or "cmd",
            "env_literals": {key: value},
        })
    except ValueError:
        return False
    return True


def parse_mcp_servers_json(
    text: str, *, existing_ids: set[str] = frozenset()
) -> list[ImportCandidate]:
    """Parse mcpServers JSON into import candidates with per-entry warnings.

    Args:
        text: Raw JSON text ({"mcpServers": {name: {command, args?, env?}}}).
        existing_ids: Profile ids already present (overwrite warnings).

    Returns:
        One candidate per server entry, in file order.

    Raises:
        ValueError: Invalid JSON, missing/empty "mcpServers", or a non-dict entry.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not valid JSON: {exc}") from None
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict) or not servers:
        raise ValueError('Expected a top-level "mcpServers" object with at least one entry.')
    candidates: list[ImportCandidate] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            raise ValueError(f'Entry "{name}" must be an object.')
        candidate = ImportCandidate(
            profile_id=str(name).strip(),
            command=str(entry.get("command") or "").strip(),
            args=[str(a) for a in entry.get("args") or []],
        )
        for key, raw_value in (entry.get("env") or {}).items():
            value = str(raw_value)
            if _PLACEHOLDER_RE.match(value.strip()):
                candidate.env_placeholders[str(key)] = value.strip()
            elif _literal_is_storable(candidate.profile_id, candidate.command, str(key), value):
                candidate.env_literals[str(key)] = value
            else:
                candidate.env_placeholders[str(key)] = f"${key}"
                candidate.warnings.append(
                    f"{key}: value can't be stored — will reference ${key}; "
                    "export it before connecting."
                )
        if candidate.profile_id in existing_ids:
            candidate.warnings.append(
                f"{candidate.profile_id}: will overwrite the existing profile.")
        candidates.append(candidate)
    return candidates
```

UI panel + workbench wiring per Interfaces (import button in overview next to Add server: `#mcp-import-server` "Import…"; panel hosted in the same `#mcp-servers-form` container; a workbench UI test drives paste → preview → apply against the fake service and asserts `save_local_profile` payloads).

- [ ] **Step 4: Run** — `Tests/MCP/test_mcp_import.py` + UI suites green.
- [ ] **Step 5: Commit** — `feat(mcp-hub): import local profiles from mcpServers JSON with secret-safe env mapping`

---

### Task 9: Server-source mutations (scope-gated forms over run_action)

**Files:**
- Create: `tldw_chatbook/UI/MCP_Modules/mcp_server_mutations.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`, `mcp_servers_mode.py`
- Test: `Tests/UI/test_mcp_server_mutations.py`

**Interfaces:**
- Consumes: existing `service.run_action(name, payload)` with the 12 verified server action names; `service.available_actions()` (returns `[]` below team scope — the UI gate); Phase 1's `server_external_record_readiness` (external records already normalize); `Input(password=True)` for secrets.
- Produces:
  - Workbench (server source, target selected): `_collect_snapshots` additionally loads external records — `payload = await service.load_section("external_servers")`; `records = payload.get("external_servers") or []`; extend snapshots with `server_external_record_readiness(r, server_id=<target id>)`. External-record rows appear in rail + table beneath the target (Phase 1 deferred item).
  - `MCPServerMutationsPanel(Vertical)` (new file) hosted in `#mcp-servers-form`:
    - Add mode: fields `#mcp-srv-id`, `#mcp-srv-name`, `#mcp-srv-transport` (Select: http/sse/stdio, allow_blank=False), `#mcp-srv-url` (config.url), `#mcp-srv-enabled` (Checkbox, default True) → `SubmitRequested(action="external_server.create", payload={"server_id", "name", "transport", "config": {"url": ...} if url else {}, "enabled"})` (namespace `mcp_server_mutations`). **Never include owner_scope keys.**
    - Edit mode (external record): name/enabled → `external_server.update` payload `{"server_id", "name", "enabled"}`.
    - Credentials section (edit mode): slots list rendered from `run_action("external_server.slots.list", {"server_id"})` result (workbench fetches, passes in); per-slot rows with Delete; add-slot subform (`slot_name`, `display_name`, `secret_kind` Select bearer_token/api_key/client_secret, `privilege_class` Select read/write, `is_required` Checkbox) → `external_server.slot.create`; per-slot secret: `Input(password=True)` + Set (`external_server.slot.secret.set`, field cleared after post) + Clear (`external_server.slot.secret.clear`). All posts via the same `SubmitRequested(action=..., payload=...)` message.
  - Workbench handler: `on_mcp_server_mutations_submit_requested` → worker `await service.run_action(event.action, event.payload)`; success → notify + reload; failure → panel `show_error(str(exc))`.
  - Gating: workbench computes `self._server_mutations_available = any(a.get("name") == "external_server.create" for a in service.available_actions() or [])` when source == server (recomputed on scope/section changes); overview's Add server button in server source is disabled with tooltip "Requires team, org, or system-admin scope." when unavailable; external-record detail hides mutation UI likewise.
  - Deferred (recorded): auth-template display/edit (GET is unwrapped server-side — Phase 3 note), external_server.import.

- [ ] **Step 1: Failing tests**

```python
# Tests/UI/test_mcp_server_mutations.py
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Input, Select

from tldw_chatbook.UI.MCP_Modules.mcp_server_mutations import MCPServerMutationsPanel


class MutationsApp(App):
    def __init__(self, record=None, slots=()) -> None:
        super().__init__()
        self.record = record
        self.slots = list(slots)
        self.posted: list = []

    def compose(self) -> ComposeResult:
        yield MCPServerMutationsPanel(record=self.record, slots=self.slots, id="panel")

    def on_mcp_server_mutations_submit_requested(self, event) -> None:
        self.posted.append((event.action, event.payload))


@pytest.mark.asyncio
async def test_add_mode_posts_create_with_exact_payload():
    app = MutationsApp()
    async with app.run_test() as pilot:
        app.query_one("#mcp-srv-id", Input).value = "web-search"
        app.query_one("#mcp-srv-name", Input).value = "Web Search"
        app.query_one("#mcp-srv-transport", Select).value = "http"
        app.query_one("#mcp-srv-url", Input).value = "https://mcp.example/api"
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        action, payload = app.posted[-1]
        assert action == "external_server.create"
        assert payload == {
            "server_id": "web-search", "name": "Web Search", "transport": "http",
            "config": {"url": "https://mcp.example/api"}, "enabled": True,
        }
        assert "owner_scope_type" not in payload and "owner_scope_id" not in payload


@pytest.mark.asyncio
async def test_edit_mode_posts_update_with_name_and_enabled_only():
    record = {"server_id": "web-search", "name": "Web Search", "enabled": True}
    app = MutationsApp(record=record)
    async with app.run_test() as pilot:
        app.query_one("#mcp-srv-name", Input).value = "Search"
        app.query_one("#mcp-srv-enabled", Checkbox).value = False
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        action, payload = app.posted[-1]
        assert action == "external_server.update"
        assert payload == {"server_id": "web-search", "name": "Search", "enabled": False}


@pytest.mark.asyncio
async def test_slot_create_posts_five_fields():
    record = {"server_id": "web-search", "name": "Web Search", "enabled": True}
    app = MutationsApp(record=record)
    async with app.run_test() as pilot:
        app.query_one("#mcp-slot-name", Input).value = "token_readonly"
        app.query_one("#mcp-slot-display", Input).value = "Read-only token"
        app.query_one("#mcp-slot-kind", Select).value = "bearer_token"
        app.query_one("#mcp-slot-privilege", Select).value = "read"
        app.query_one("#mcp-slot-required", Checkbox).value = True
        await pilot.click("#mcp-slot-add")
        await pilot.pause()
        action, payload = app.posted[-1]
        assert action == "external_server.slot.create"
        assert payload == {
            "server_id": "web-search", "slot_name": "token_readonly",
            "display_name": "Read-only token", "secret_kind": "bearer_token",
            "privilege_class": "read", "is_required": True,
        }


@pytest.mark.asyncio
async def test_slot_secret_set_posts_and_clears_input():
    record = {"server_id": "web-search", "name": "Web Search", "enabled": True}
    slots = [{"slot_name": "token_readonly", "display_name": "Read-only token"}]
    app = MutationsApp(record=record, slots=slots)
    async with app.run_test() as pilot:
        secret_input = app.query_one("#mcp-slot-secret-0", Input)
        assert secret_input.password is True
        secret_input.value = "sk-value"
        await pilot.click("#mcp-slot-secret-set-0")
        await pilot.pause()
        action, payload = app.posted[-1]
        assert action == "external_server.slot.secret.set"
        assert payload == {"server_id": "web-search", "slot_name": "token_readonly",
                           "secret": "sk-value"}
        assert secret_input.value == ""
```

Plus one workbench-level test appended to `Tests/UI/test_mcp_workbench.py`: with a fake service whose `available_actions()` returns `[]` in server source, the overview Add-server button is disabled and its tooltip is "Requires team, org, or system-admin scope." (slot rows use index-based ids `#mcp-slot-secret-<i>` — slot names are remote strings, never widget ids).

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** per Interfaces — panel ~200 lines mirroring `MCPProfileForm`'s structure (labels, one error Static, state-driven buttons, tooltips on every Button, `markup=False` on remote-derived Statics, `Select.NULL`-safe handling).
- [ ] **Step 4: Run** — new suite + workbench green.
- [ ] **Step 5: Commit** — `feat(mcp-hub): scope-gated server-source create/update/delete and credential-slot forms`

---

### Task 10: Built-in server expose toggles

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py`, `mcp_workbench.py`
- Test: extend `Tests/UI/test_mcp_servers_mode.py`

**Interfaces:**
- Consumes: `builtin_readiness` detail flags; `save_setting_to_cli_config` + `get_cli_setting` from `tldw_chatbook.config` (Library rail-prefs precedent).
- Produces: built-in detail view renders four `Checkbox` rows (`#mcp-builtin-enabled`, `#mcp-builtin-expose-tools`, `#mcp-builtin-expose-resources`, `#mcp-builtin-expose-prompts`, values from the snapshot detail) + a note Static: "Applies to the next client launch — the built-in server reads config at start." `Checkbox.Changed` posts `MCPServersMode.BuiltinFlagChanged(key: str, value: bool)` (namespace `mcp_servers_mode`; keys: `enabled|expose_tools|expose_resources|expose_prompts`). Workbench handler writes via `save_setting_to_cli_config("mcp", event.key, event.value)` in a thread worker (config I/O), then reloads the catalog so the readiness row updates (enabled=False → NEEDS_SETUP per Phase 1 derivation).

- [ ] **Step 1: Failing test** — builtin detail shows the four checkboxes with values from detail flags; toggling `enabled` posts `BuiltinFlagChanged("enabled", False)`; the note text is present.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** (checkbox block only in the `builtin` branch of the detail renderer — which becomes widget-composed rather than a single text blob for that branch; keep the copy-snippet button).
- [ ] **Step 4: Run.**
- [ ] **Step 5: Commit** — `feat(mcp-hub): built-in server enable/expose toggles with next-launch note`

---

### Task 11: Canvas UX — breadcrumb, actionable callouts, status colors, per-source columns

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py`, `mcp_rail.py`, `mcp_inspector.py`
- Test: extend `Tests/UI/test_mcp_servers_mode.py`, `Tests/UI/test_mcp_rail.py`

**Interfaces:**
- Consumes: T3 `STATE_CSS_CLASSES`; UX-inputs doc decisions (breadcrumb chosen as the single recovery pattern, with selection restoration).
- Produces:
  - **Breadcrumb**: detail view header row gains `#mcp-detail-back` Button ("← All servers", console-action-subdued, tooltip "Return to the overview table.") posting `ServerRowSelected(None)`-equivalent (reuse the existing message with `server_key=None` — workbench already treats None as clear). **Selection restoration**: returning to overview moves the DataTable cursor to the previously selected row (`table.move_cursor(row=<index of previous key>)`) so keyboard users resume where they left.
  - **Callouts**: replace the Static boxes with one-line Buttons (`classes="mcp-callout console-action-subdued"`, height 1, `escape_markup`'d label `f"{glyph} {label}: {message}"`, tooltip "Open {label}."), each posting `ServerRowSelected(key)`; at most 4 rendered + a final Static "+N more — see the table above." when exceeded.
  - **Status colors**: rail row Buttons and the inspector badge Static get `STATE_CSS_CLASSES[state]` added (and stale ones removed) on render; aggregate Static keeps `ds-status-badge` and additionally gets the class for the WORST state present (READY when all ready). DataTable Status cells stay plain text (theme-token colors are not addressable per-cell — documented decision).
  - **Per-source columns**: `update_overview` gains `source` awareness — in Local source the Scope column is omitted (columns rebuilt per call: `clear(columns=True)` then `add_columns(...)`); Auth column copy becomes `"1 env var"/"2 env vars"/"none"` for local snapshots.
  - **Rail count alignment** (UX-inputs polish): rail row labels place the tool count in a fixed right-side column — `_row_label` pads the name to the truncation budget and right-aligns the count (`f"{glyph} {prefix}{name:<{budget}} {count:>3}"`, count blank when None) so counts form one scannable column.
- Tests: breadcrumb posts clear + cursor restored (assert `table.cursor_row` after return); callout click posts the right key; rail row carries the state class; local overview has no Scope column and shows "1 env var".

- [ ] **Steps 1-5**: failing tests → implement → run (`test_mcp_servers_mode.py`, `test_mcp_rail.py`, workbench suite) → commit `feat(mcp-hub): breadcrumb navigation, actionable callouts, status colors, per-source columns`.

---

### Task 12: Advanced disclosure (Collapsible) + object label + info-callout placeholders

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py`, `mcp_workbench.py`
- Test: extend `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: `textual.widgets.Collapsible` (+ `Collapsible.Toggled`); `save_setting_to_cli_config("mcp.hub_state", "advanced_open", bool)` / `get_cli_setting("mcp.hub_state", "advanced_open", False)` (per-user GLOBAL scope per the UX-inputs decision); Phase 1's `_load_advanced_section`.
- Produces:
  - The entire Advanced block (section select, content, action select, payload, run, result, empty-hint) moves inside `Collapsible(title="Advanced (legacy control plane)", collapsed=not persisted_open, id="mcp-adv-collapsible")`. `Collapsible.Toggled` handler persists the new state via a thread worker.
  - **Object label**: `#mcp-adv-object` Static (markup=False) at the top of the collapsible body: "Showing: Local control plane" / "Showing: server <target label>" — updated by `set_service_context` AND whenever the workbench selection changes source/target (`rebind`); the content Static is cleared+reloaded on selection change so a previous object's dump can never linger (UX-inputs acceptance).
  - **Info-callout**: new CSS class `.ds-info-callout` (T13 adds the rule: `border: round $ds-status-info; background: $ds-status-info 10%;` + padding matching `.ds-recovery-callout`); the three placeholder canvases in `mcp_workbench.py` switch from `ds-recovery-callout` to `ds-info-callout`.
- Tests: collapsible starts collapsed by default (fake `get_cli_setting` → False); toggling persists (monkeypatch `save_setting_to_cli_config`, assert called with `("mcp.hub_state", "advanced_open", True)`); object label updates when workbench switches source; placeholders carry `ds-info-callout` not `ds-recovery-callout`.

- [ ] **Steps 1-5**: failing tests → implement → run → commit `feat(mcp-hub): collapsed Advanced disclosure with object label; informational placeholder styling`.

---

### Task 13: Footer shortcuts, `a`/`r` bindings, CSS, bundle, full gate

**Files:**
- Modify: `tldw_chatbook/UI/Screens/mcp_screen.py`, `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ rebuild `tldw_cli_modular.tcss`)
- Test: extend `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: `AppFooterStatus.set_workbench_shortcuts` / `clear_shortcut_context` (Console precedent `chat_screen.py:280,825,835`); T3 `STATE_CSS_CLASSES` names; T12 `.ds-info-callout`.
- Produces:
  - `MCP_SHORTCUTS = (("1-4", "mode"), ("a", "add server"), ("r", "refresh"))` module const in `mcp_screen.py`; `_register_footer_shortcuts` / `_clear_footer_shortcuts` (source="mcp", `query_one(AppFooterStatus)` guarded by try/except NoMatches) called from `on_mount`, `on_screen_resume`, `on_screen_suspend`.
  - New `BINDINGS`: `Binding("a", "mcp_add_server", "Add server", show=False)` → switch to servers mode + post add-form open (workbench method `open_add_server_form()`); `Binding("r", "mcp_refresh", "Refresh", show=False)` → existing exclusive reload worker.
  - CSS block appended to `_agentic_terminal.tcss`:
```css
/* --- MCP Hub Phase 2 ------------------------------------------------- */
.mcp-status-ready { color: $ds-status-ready; }
.mcp-status-warning { color: $ds-status-warning; }
.mcp-status-error { color: $ds-status-error; }
.mcp-status-info { color: $ds-status-info; }

.ds-info-callout {
    border: round $ds-status-info;
    background: $ds-status-info 10%;
    padding: 0 1;
    height: auto;
}

Button.mcp-callout {
    width: 100%;
    height: 1;
    min-height: 1;
    padding: 0 1;
    border: none;
    text-align: left;
}

#mcp-servers-form {
    height: auto;
    min-height: 0;
}

#mcp-detail-toolbar Button {
    height: 1;
    min-height: 1;
    border: none;
}
```
  Rebuild: `python3 tldw_chatbook/css/build_css.py`; commit both files.
  - Full gate:
`PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest Tests/MCP/ Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_servers_mode.py Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py Tests/UI/test_mcp_profile_form.py Tests/UI/test_mcp_server_mutations.py Tests/UI/test_unified_mcp_panel.py Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py -q`
Expected: all green except the 2 documented pre-existing Library failures.
  - Tests: footer registration called with source="mcp" (fake AppFooterStatus via monkeypatched query_one or a mounted harness); `a` binding opens the add form; geometry test extended to assert the form container renders non-zero when shown.

- [ ] **Steps 1-5**: failing tests → implement → rebuild bundle → full gate → commit `feat(mcp-hub): footer shortcuts, add/refresh keys, Phase 2 styles + bundle rebuild`.

**Post-task (controller-owned, not an implementer step):** live screenshot QA round (2050×1240, real CSS, seeded isolated HOME per the Phase 1 recipe in `Docs/superpowers/qa/mcp-hub-phase1-2026-07/README.md`) covering: add-server form, env validation error (store copy), import preview with secret warning, lifecycle in-flight + cancel, delete arm/confirm, built-in toggles, server-source scope-gated state, colored badges, collapsed Advanced, breadcrumb return. User screenshot approval gates the PR.

---

## Out of scope (recorded)

Auth-template display/edit (server GET unwrapped — Phase 3 with Tools mode); external_server.import UI; local HTTP/SSE transport (store has no such fields — spec corrected); Tools/Permissions/Audit modes (Phases 3-5); agent-runtime integration (task-201 consumes T2's typed seam later); DataTable per-cell theme colors (documented limitation).
