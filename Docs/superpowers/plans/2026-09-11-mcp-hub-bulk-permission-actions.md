# MCP Hub Bulk Permission Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two filter-scoped bulk keys on the Permissions matrix — `shift+space` applies the cursor row's next cycled state to a server's visible tool rows; `C` clears its visible overrides (ADR-150).

**Architecture:** The canvas (which alone knows visibility) computes the scope and posts one message per gesture; the workbench remains the single writer, executing the bulk as N ordinary profile-scoped `set_tool_state` calls with raw-shell rows skipped and named in the echo. No new store API; audit granularity unchanged (per-row logging).

**Tech Stack:** Textual key bindings (`shift+space`, `C` — no conflicts with the screen's `1-4/a/r/t`), pytest + Pilot, existing PermissionsApp harness.

**Spec:** `Docs/superpowers/specs/2026-09-11-mcp-hub-bulk-permission-actions-design.md`; ADR: `backlog/decisions/150-mcp-hub-bulk-permission-actions.md`.

## Global Constraints

- Every write goes through `_call_profile_scoped(service.set_tool_state, ...)` under the SAME validated `PermissionProfileContext` as a single press — no batch store API (ADR-150).
- Raw-shell rows are skipped and the skip is named in the echo (`· N skipped (raw shell)`); raw shell stays a two-state control.
- The **global row is excluded** — both keys no-op against it via the legend hint line (not a toast); no-op likewise when the server has zero *visible* tool rows.
- `C` writes only rows that currently hold an override (visible-scoped); the echo says "N visible overrides cleared" and the tooltip/legend teaches the full-clear recipe (clear the filter, then C).
- Wave-B interplay: the canvas computes the next state with the same `cycle_ui_state` a plain press uses, so the first `shift+space` from Inherit applies **Ask** to the visible set.
- On the first write failure: stop, toast `Permission update failed: <reason>` (Wave A's pattern), never partial-silence.
- Copy moves with its pins in the same commit: `_LEGEND_TEXT` (2 verbatim test pins) and the permissions footer hint (`mcp_screen.MCP_MODE_SHORTCUTS` + destination-shell footer pin).
- TDD: every behavior watched RED first; targeted suites only.

---

### Task 1: Canvas — bindings, messages, hint flash, copy

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py`
- Modify: `tldw_chatbook/UI/Screens/mcp_screen.py` (`MCP_MODE_SHORTCUTS["permissions"]`)
- Test: `Tests/UI/test_mcp_permissions_mode.py`, `Tests/UI/test_destination_shells.py` (footer pin)

**Interfaces:**
- Produces:
  - `MCPPermissionsMode.BulkStateRequested(server_key: str, tool_names: tuple[str, ...], new_state: str, profile_context)` — canvas-computed visible tool rows of the cursor row's server plus the cursor row's own `cycle_ui_state` next-state.
  - `MCPPermissionsMode.BulkClearRequested(server_key: str, tool_names: tuple[str, ...], profile_context)` — the server's visible tool rows that currently hold an override.
  - `MCPPermissionsMode.flash_hint(text: str)` — renders one transient line appended to the legend until the next `update_matrix` (the "existing hint Static" the spec names; not a toast).

- [ ] **Step 1: Write the failing tests** (append to `Tests/UI/test_mcp_permissions_mode.py`; the file's `_global_row`, `_server_row`, `_tool_row` helpers and `PermissionsModeApp` harness already exist — check exact helper names with grep before writing):

```python
@pytest.mark.asyncio
async def test_shift_space_on_tool_row_posts_bulk_state_for_visible_tools():
    app = PermissionsModeApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPPermissionsMode)
        rows = [
            _global_row(),
            _server_row(server_key="local:docs", server_label="docs"),
            _tool_row(server_key="local:docs", tool_name="fetch"),
            _tool_row(server_key="local:docs", tool_name="search"),
        ]
        await canvas.update_matrix(rows, kill_switch=False, preview="")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=2)
        await pilot.press("shift+space")
        await pilot.pause()
        assert len(app.events) == 1
        event = app.events[0]
        assert isinstance(event, MCPPermissionsMode.BulkStateRequested)
        assert event.server_key == "local:docs"
        assert sorted(event.tool_names) == ["fetch", "search"]
        assert event.new_state == "ask"  # Wave B: cycle_ui_state(None) == "ask"


@pytest.mark.asyncio
async def test_bulk_clear_posts_only_overridden_visible_rows():
    # Same shape: tool rows with cycle_current="allow"/None mixed; C posts
    # BulkClearNamed with only the overridden names.


@pytest.mark.asyncio
async def test_bulk_keys_noop_on_global_row_with_hint_not_toast():
    # cursor row 0 (global); shift+space AND C post nothing; the legend
    # gains the hint line via flash_hint; no app.notify.
```

(Write all three fully in the file; the second/third follow the first's harness shape. The footer pin at `Tests/UI/test_destination_shells.py:3369` gains the new hint text.)

- [ ] **Step 2: Run to verify RED** — `pytest Tests/UI/test_mcp_permissions_mode.py -k bulk -q` → FAIL (no bindings/messages).

- [ ] **Step 3: Implement** in `mcp_permissions_mode.py`:

1. `BINDINGS` gains `Binding("shift+space", "bulk_set", show=False)` and `Binding("C", "bulk_clear", show=False)`.
2. The two `Message` classes (namespace `mcp_permissions_mode`) per the Interfaces block.
3. `_legend_extras: list[str]` set by `update_matrix` (replacing the current inline legend_text build) and used by `flash_hint(text)` to re-render `#mcp-perm-legend` with the extra line appended (until the next `update_matrix`).
4. `_visible_tool_names_for(server_key) -> tuple[str, ...]` helper over `self._visible_rows`.
5. `action_bulk_set()` / `action_bulk_clear()`: resolve the cursor row exactly like `action_cycle_state` (coordinate_to_cell_key → `_rows_by_key`); global row → `flash_hint("Bulk actions need a server's tool rows — move to one of its rows.")` and return; compute visible tool names (for clear: only rows with `cycle_current is not None`); empty → same hint; else post the message (`new_state` via `cycle_ui_state(row.cycle_current)` for bulk_set).
6. `_LEGEND_TEXT` gains: `" · shift+space bulk set · C clears overrides"` (update the 2 verbatim pins).
7. `mcp_screen.py`: `"permissions": _COMMON_SHORTCUTS + (("space", "cycle permission"), ("shift+space", "bulk set"), ("C", "clear overrides"))` (update the destination footer pin).

- [ ] **Step 4: GREEN** — `pytest Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_destination_shells.py -k "bulk or footer or shortcuts" -q`.

- [ ] **Step 5: Commit.**

---

### Task 2: Workbench — apply/clear with raw-shell skip, echo, error toast

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Test: `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: Task 1's messages.
- Produces: `on_mcp_permissions_mode_bulk_state_requested` / `on_mcp_permissions_mode_bulk_clear_requested` handlers; shared `_apply_bulk_tool_states(server_key, tool_names, state_or_None, context)`.

- [ ] **Step 1: Write the failing tests** (PermissionsApp, real store; mirror the double-press choreography's settle pauses):

```python
@pytest.mark.asyncio
async def test_shift_space_bulk_sets_visible_tools_of_one_server(tmp_path):
    """ADR-150: shift+space applies the cursor row's next state to the
    server's VISIBLE tool rows only (Wave B order: first press from
    Inherit = Ask), with one echo and no other server touched."""
    app = PermissionsApp(tmp_path / "mcp_permissions.json")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        workbench.set_mode("permissions")
        await pilot.pause()
        table = app.query_one("#mcp-perm-table", DataTable)
        table.focus()
        table.move_cursor(row=3)  # local:docs::search
        await pilot.press("shift+space")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        store = app.unified_mcp_service.permission_store.load()
        docs_tools = store["profiles"]["default"]["servers"]["local:docs"]["tools"]
        assert docs_tools["search"]["state"] == "ask"
        assert docs_tools["fetch"]["state"] == "ask"
        preview = str(app.query_one("#mcp-perm-preview", Static).renderable)
        assert preview.startswith("docs: 2 tools → Ask · ")


@pytest.mark.asyncio
async def test_bulk_clear_reverts_only_visible_overrides(tmp_path):
    # Seed search=allow via store; filter the matrix to hide fetch; C on
    # the docs rows clears ONLY search; fetch's seeded override survives;
    # echo "docs: 1 visible overrides cleared".
```

Plus: raw-shell skip test (the `_enable_local_tools(monkeypatch)` + `raw_cli_runtime` harness at test_mcp_workbench.py:10292's pattern: bulk-set local:__local__, assert `shell_exec` untouched and the echo carries `skipped (raw shell)`), and the failure test (service `set_tool_state` raising → toast `Permission update failed: ...`, first-write stop).

- [ ] **Step 2: RED** — `pytest Tests/UI/test_mcp_workbench.py -k "bulk" -q`.

- [ ] **Step 3: Implement** in `mcp_workbench.py`:

```python
    def on_mcp_permissions_mode_bulk_state_requested(self, event) -> None:
        event.stop()
        self.run_worker(
            self._apply_bulk_tool_states(
                event.server_key, list(event.tool_names), event.new_state,
                event.profile_context,
            ),
            group="mcp-perm-bulk", exclusive=True,
        )
    # bulk_clear: same worker path with state=None and echo verb "cleared".
```

`_apply_bulk_tool_states`: validate profile context (stale → the existing "Tool policy profile changed" toast); iterate names, `_tool_for(server_key, name)` for hash-needing allows (None + "allow" + not hash-free → skip-with-count like the vanished-tool guard), `_is_raw_shell_tool` → skip-count; each write via `_call_profile_scoped(service.set_tool_state, ...)` under the SAME context; on exception → the Wave-A reason toast and return (partial writes stand, each idempotent); after the batch one `_sync_permissions_mode(echo=...)` with `"{label}: {N} tools → {label_of_state}"` (or `"{label}: {N} visible overrides cleared"`) plus `" · {K} skipped (raw shell)"` when K>0. Clear path counts only rows written (canvas already sent only overridden rows).

- [ ] **Step 4: GREEN + sweep** — the new tests, then `pytest Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_mcp_workbench.py -q`.

- [ ] **Step 5: Commit.**

---

### Task 3: Docs + final sweep + hygiene

- [ ] **Step 1:** `Docs/User_Guide/mcp.md` — in the tutorial's permissions step, after the server-default sentence, add: "For many rows at once: `shift+space` applies the row's next state to the server's visible tools, and `C` clears its visible overrides — the filter is the scope (clear it to act on all rows)."
- [ ] **Step 2:** Final sweep — all five MCP suites; doc-contract guard (39 pre-existing, zero new); ruff before/after on touched files.
- [ ] **Step 3:** Backlog task Implementation Notes + Done; commit.

---

## Self-Review

- **Spec coverage:** both keys + messages (T1), global-row exclusion + zero-rows no-op via hint (T1), visible-scope semantics + filter-as-scope (T1 helper + T2 filter test), raw-shell skip + echo suffix (T2), first-failure stop + reason toast (T2), single echo + single resync (T2), footer + legend copy with pins (T1), audit per-row unchanged (constraint — no test change needed), tooltip full-clear recipe — covered by the legend clause copy in T1 step 3.6.
- **Placeholder scan:** the two sketch tests in T1/T2 are named-but-abbreviated in this plan for brevity; the executor writes them fully per the shown harness shapes — the first test of each task is complete and the rest follow it mechanically.
- **Type consistency:** message field names (`server_key`, `tool_names`, `new_state`, `profile_context`) used identically in both tasks; `flash_hint(text: str)`.
