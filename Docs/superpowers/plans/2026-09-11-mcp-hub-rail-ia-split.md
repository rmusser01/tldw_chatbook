# MCP Hub Rail IA Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use the superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the agent tool catalog out of the built-in server's rail row into its own "Agent tools" rail section (ADR-148), so one row stops fronting three subsystems.

**Architecture:** A new `agent_tools_readiness()` snapshot keyed `agent:builtin` (the existing `BUILTIN_TOOL_SERVER_KEY` store identity, which has no rail presence today) rides beside — not inside — the server snapshots: the rail renders it as a second section, the workbench routes it by `source == "agent"`, and the `Tool gates` checkbox group moves from the built-in server's detail pane to the agent row's detail pane. No store key, store entry, or permission-resolution semantic changes.

**Tech Stack:** Textual 8.x widgets (Button/Static/DataTable), pytest + Textual Pilot, existing MCP Hub harnesses.

**Spec:** `Docs/superpowers/specs/2026-09-11-mcp-hub-rail-ia-split-design.md` (identity map verified against code; ADR: `backlog/decisions/148-mcp-hub-rail-ia-and-responsive-triad.md`).

## Global Constraints

- No permission/store semantics change: no new store keys beyond the rail-facing snapshot, no resolution changes, ADR-081 posture untouched.
- No two rail rows share a `server_key` (ADR-148); the agent row uses `agent:builtin`, the built-in server keeps `builtin:tldw_chatbook`.
- The agent snapshot NEVER enters `self._snapshots` (overview table, callouts, worst-state summary, and preselection heuristics stay server-only).
- Selecting the agent row renders the UNSCOPED Permissions preview (`agent:builtin` is never in `_last_hub_tools` — verified).
- Label discipline: "Unknown" never renders as "Off"; markup-safe labels; tooltips explain outcomes.
- Tests: targeted runs only (affected MCP suites); TDD — every new behavior's test is watched RED first.
- House keybinding rules (ADR-031) unchanged — no new bindings in this wave.

---

### Task 1: `agent_tools_readiness` snapshot builder

**Files:**
- Modify: `tldw_chatbook/MCP/readiness.py` (after `builtin_readiness`, ~line 640)
- Test: `Tests/UI/test_mcp_servers_mode.py` (new test near the existing readiness tests; `builtin_readiness` is already imported there)

**Interfaces:**
- Consumes: `ReadinessSnapshot`, `ReadinessState`, `ReasonCode` (all in readiness.py).
- Produces: `agent_tools_readiness(*, enabled: bool) -> ReadinessSnapshot` with `server_key=BUILTIN_TOOL_SERVER_KEY_FOR_AGENTS` — define `AGENT_TOOLS_SERVER_KEY = "agent:builtin"` in readiness.py (do NOT import permission_store's `BUILTIN_TOOL_SERVER_KEY` into readiness.py — that would invert the dependency; instead assert equality in a test, see step 1).

- [ ] **Step 1: Write the failing test**

```python
def test_agent_tools_readiness_builder_shapes_the_rail_row():
    from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY
    from tldw_chatbook.MCP.readiness import agent_tools_readiness

    on = agent_tools_readiness(enabled=True)
    assert on.server_key == "agent:builtin"
    assert on.server_key == BUILTIN_TOOL_SERVER_KEY  # same store identity, no new key
    assert on.source == "agent"
    assert on.label == "Agent tools"
    assert on.state is ReadinessState.READY
    assert "Console" in on.message

    off = agent_tools_readiness(enabled=False)
    assert off.state is ReadinessState.OFF_OPT_IN
    assert "Turned off" in off.message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/UI/test_mcp_servers_mode.py -k agent_tools_readiness -q`
Expected: FAIL with `ImportError: cannot import name 'agent_tools_readiness'`

- [ ] **Step 3: Write minimal implementation**

In `tldw_chatbook/MCP/readiness.py`, after `builtin_readiness()`:

```python
#: ADR-148: the rail-facing identity of the in-process agent tool catalog.
#: Deliberately equal to permission_store.BUILTIN_TOOL_SERVER_KEY ("agent:builtin")
#: -- the store identity that already exists but had no rail presence. The
#: equality is pinned by test; readiness.py must not import the store
#: (dependency direction), so the literal is duplicated once and tested.
AGENT_TOOLS_SERVER_KEY = "agent:builtin"


def agent_tools_readiness(*, enabled: bool) -> ReadinessSnapshot:
    """Readiness for the in-process agent tool catalog (ADR-148 Wave D).

    Not a server: nothing connects to it and no client launches it -- the
    state is purely whether the `[console] local_tools_enabled` master
    switch registers the workspace/web/Watchlists/built-in agent tools.
    """
    if enabled:
        state = ReadinessState.READY
        message = "In-process tools for Console agents."
    else:
        state = ReadinessState.OFF_OPT_IN
        message = "Turned off — Console agents get no workspace, web, or Watchlists tools."
    return ReadinessSnapshot(
        server_key=AGENT_TOOLS_SERVER_KEY,
        label="Agent tools",
        source="agent",
        state=state,
        reasons=() if enabled else (ReasonCode.NOT_CONFIGURED,),
        message=message,
        transport="—",
        auth_display="—",
        scope_display="—",
        detail={"enabled": enabled},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/UI/test_mcp_servers_mode.py -k agent_tools_readiness -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/readiness.py Tests/UI/test_mcp_servers_mode.py
git commit -m "feat(mcp): agent_tools_readiness snapshot for the Agent tools rail row (ADR-148)"
```

---

### Task 2: Rail renders the Agent tools section

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_rail.py` (constructor, `sync_state`, `compose`)
- Test: `Tests/UI/test_mcp_rail.py`

**Interfaces:**
- Consumes: `agent_tools_readiness` snapshots passed in by the workbench (Task 4).
- Produces: `MCPRail(..., agent_snapshot: ReadinessSnapshot | None = None)` constructor kwarg; `sync_state(..., agent_snapshot: ReadinessSnapshot | None = None)`; the agent row's click posts the existing `MCPRail.ServerSelected("agent:builtin")`; `_row_keys` ends with the agent key when present.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_mcp_rail.py` (the file's `_snap` helper and `RailApp` harness already exist):

```python
def _agent_snap():
    from tldw_chatbook.MCP.readiness import agent_tools_readiness

    return agent_tools_readiness(enabled=True)


@pytest.mark.asyncio
async def test_rail_renders_agent_tools_section_and_row():
    class AgentRailApp(RailApp):
        def compose(self) -> ComposeResult:
            yield MCPRail(
                source="local",
                snapshots=[_snap("local:docs", "docs")],
                selected_server_key=None,
                scope_options=[("Personal", "personal")],
                scope_value="personal",
                scope_ref_options=[],
                scope_ref_value=None,
                agent_snapshot=_agent_snap(),
                id="mcp-rail",
            )

    app = AgentRailApp()
    async with app.run_test() as pilot:
        headings = [
            str(widget.renderable)
            for widget in app.query(".mcp-rail-heading")
        ]
        assert headings == ["Source", "Servers", "Agent tools"]
        rows = list(app.query("Button.mcp-rail-row"))
        labels = [str(row.label) for row in rows]
        assert any("Agent tools" in label for label in labels)
        # Clicking the agent row posts ServerSelected with the store identity.
        agent_button = next(
            row for row in rows if "Agent tools" in str(row.label)
        )
        await pilot.click(f"#{agent_button.id}")
        await pilot.pause()
        assert app.events, "agent row click must post ServerSelected"
        assert app.events[-1].server_key == "agent:builtin"


@pytest.mark.asyncio
async def test_rail_omits_agent_section_without_snapshot():
    app = RailApp()  # no agent_snapshot kwarg
    async with app.run_test() as pilot:
        headings = [
            str(widget.renderable)
            for widget in app.query(".mcp-rail-heading")
        ]
        assert "Agent tools" not in headings
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_mcp_rail.py -k agent_tools_section -q && pytest Tests/UI/test_mcp_rail.py -k omits_agent_section -q`
Expected: FAIL (`unexpected keyword argument 'agent_snapshot'` / heading missing)

- [ ] **Step 3: Write minimal implementation**

In `MCPRail.__init__`: add `agent_snapshot: ReadinessSnapshot | None = None` parameter, store as `self.agent_snapshot = agent_snapshot`. In `sync_state`: same parameter, same assignment. In `compose`, after the server-rows loop (the `for index, snap in enumerate(self.snapshots, start=1)` block) and before the `if self.source == "server":` scope block:

```python
        # ADR-148 Wave D: the in-process agent tool catalog gets its own
        # rail section -- one row, keyed by the store identity
        # permission_store.BUILTIN_TOOL_SERVER_KEY ("agent:builtin"), so
        # no rail row ever shares a server_key with the built-in server.
        if self.agent_snapshot is not None:
            yield Static(
                "Agent tools", classes="destination-section mcp-rail-heading"
            )
            self._row_keys.append(self.agent_snapshot.server_key)
            agent_row = Button(
                _row_label(self.agent_snapshot, pad_width),
                id=f"{MCP_RAIL_ROW_PREFIX}{len(self._row_keys) - 1}",
                classes=f"mcp-rail-row console-action-subdued {STATE_CSS_CLASSES[self.agent_snapshot.state]}",
                compact=True,
            )
            agent_row.tooltip = escape_markup(
                self.agent_snapshot.message or self.agent_snapshot.label
            )
            agent_row.set_class(
                self.agent_snapshot.server_key == self.selected_server_key,
                "is-active",
            )
            yield agent_row
```

Also include the agent snapshot in the legend's present-states derivation: change the `if self.snapshots:` legend block to

```python
        legend_snaps = list(self.snapshots)
        if self.agent_snapshot is not None:
            legend_snaps.append(self.agent_snapshot)
        if legend_snaps:
            yield Static(
                _present_states_legend(
                    legend_snaps, short=budget < _LEGEND_SHORT_BUDGET
                ),
                id="mcp-rail-state-legend",
                markup=False,
            )
```

(Note: the F15 budget computation sits just above the legend yield — keep it there; the agent block goes after the rows loop where `pad_width` is already computed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/UI/test_mcp_rail.py -q`
Expected: all PASS (existing tests unaffected — no agent snapshot by default)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_rail.py Tests/UI/test_mcp_rail.py
git commit -m "feat(mcp): rail renders the Agent tools section (ADR-148 Wave D)"
```

---

### Task 3: Gates move to the agent detail; built-in detail points there

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py` (`_tool_gate_widgets` gate, `_detail_text` branches)
- Test: `Tests/UI/test_mcp_servers_mode.py` (re-target 4 existing tests, add 2)

**Interfaces:**
- Consumes: `agent_tools_readiness` (Task 1) for test fixtures; `all_tool_gates()` unchanged.
- Produces: `Tool gates` renders for `snapshot.source == "agent"` only; built-in detail shows the pointer line `"Agent tool gates live under Agent tools in the rail."`; `_detail_text` has an `"agent"` branch; gate checkbox ids/`ToolGateChanged` flow unchanged (Wave C/A behavior intact).

- [ ] **Step 1: Re-target the existing pinned tests (they currently pass — after Step 3 they must still pass in their NEW shape; write the new assertions first and watch them fail)**

In `Tests/UI/test_mcp_servers_mode.py`:

1. `test_tool_gate_checkboxes_render_under_builtin_detail_with_subheadings_and_note` → rename to `test_tool_gate_checkboxes_render_under_agent_detail_with_subheadings_and_note`; replace `await canvas.show_detail(builtin_readiness(enabled=True))` with `await canvas.show_detail(agent_tools_readiness(enabled=True))` (import it); the subheading/note/checkbox assertions stay identical.
2. `test_tool_gate_checkboxes_do_not_appear_for_non_builtin_detail` → rename `..._do_not_appear_outside_agent_detail`; extend: gates must NOT render for the built-in snapshot either:

```python
        await canvas.show_detail(builtin_readiness(enabled=True))
        await pilot.pause()
        assert not list(app.query("#mcp-detail-tool-gates Checkbox"))
```

3. `test_showing_builtin_detail_does_not_post_tool_gate_changed` → keep using the built-in snapshot (still valid), and add a sibling assertion that SHOWING agent detail also posts nothing.
4. `test_toggling_tool_gate_checkbox_posts_tool_gate_changed_with_section_and_key` and the master-off dependent test (~line 958 and ~line 961 region): switch `show_detail(builtin_readiness(...))` to `show_detail(agent_tools_readiness(...))`.

Add the new built-in-pointer test:

```python
@pytest.mark.asyncio
async def test_builtin_detail_points_at_the_agent_tools_row():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(builtin_readiness(enabled=True))
        await pilot.pause()
        body = str(app.query_one("#mcp-detail-body", Static).renderable)
        assert "Agent tool gates live under Agent tools in the rail." in body
```

And the agent detail body test:

```python
@pytest.mark.asyncio
async def test_agent_detail_body_explains_console_scope():
    app = CanvasApp()
    async with app.run_test() as pilot:
        canvas = app.query_one(MCPServersMode)
        await canvas.show_detail(agent_tools_readiness(enabled=True))
        await pilot.pause()
        title = str(app.query_one("#mcp-detail-title", Static).renderable)
        assert "Agent tools" in title
        body = str(app.query_one("#mcp-detail-body", Static).renderable)
        assert "Console" in body
        # No [mcp] server toggles and no Edit/Delete toolbar here.
        assert not list(app.query("#mcp-detail-builtin-toggles Checkbox"))
        assert not list(app.query("#mcp-detail-toolbar Button"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_mcp_servers_mode.py -k "agent_detail or points_at_the_agent or outside_agent_detail" -q`
Expected: FAIL (gates don't render for source "agent"; pointer line absent)

- [ ] **Step 3: Write minimal implementation**

In `mcp_servers_mode.py`:

1. `_tool_gate_widgets`: change the gate from

```python
        if snapshot is None or snapshot.source != "builtin":
```

to

```python
        if snapshot is None or snapshot.source != "agent":
```

and update its docstring's first line to "Build the `[tools]`/`[console]` gate Checkbox rows (task-3240, moved to the Agent tools detail by ADR-148 Wave D)."
2. `_detail_text`: add an agent branch before the final `else:  # builtin`:

```python
        elif snapshot.source == "agent":
            lines.append(
                "Registers the in-process tools Console agents can see — it "
                "does not grant permission; the Permissions matrix still "
                "gates every call."
            )
```

3. `_detail_text`'s `else:  # builtin` branch: append the pointer line:

```python
            lines.append("Agent tool gates live under Agent tools in the rail.")
```

- [ ] **Step 4: Run the module suite**

Run: `pytest Tests/UI/test_mcp_servers_mode.py -q`
Expected: all PASS (re-targeted tests included)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py Tests/UI/test_mcp_servers_mode.py
git commit -m "feat(mcp): Tool gates move to the Agent tools detail; built-in detail points there (ADR-148)"
```

---

### Task 4: Workbench wiring — snapshot, rail sync, selection routing

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (`__init__`, `compose`, `_collect_snapshots`, `_sync_children`, `_snapshot_for`)
- Test: `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Consumes: `agent_tools_readiness` (Task 1), rail kwarg (Task 2), detail routing (Task 3).
- Produces: `self._agent_snapshot` on the workbench; `_snapshot_for("agent:builtin")` returns it; the agent rail row is selectable end-to-end; `_snapshots` stays server-only.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_mcp_workbench.py`:

```python
@pytest.mark.asyncio
async def test_agent_tools_rail_row_routes_to_agent_detail(monkeypatch):
    """ADR-148 Wave D: the Agent tools rail row is selectable, opens the
    agent detail (Tool gates visible, no [mcp] toggles), stays OUT of the
    servers overview table, and scopes the Permissions preview to nothing
    (the unscoped summary -- agent:builtin is never in the hub catalog)."""
    monkeypatch.setattr(
        mcp_workbench_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )
    app = ProblemRecordsApp([])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)

        # Not a server row: absent from the overview table...
        table = app.query_one("#mcp-servers-table", DataTable)
        table_keys = [
            table.coordinate_to_cell_key((row, 0))[0].value
            for row in range(table.row_count)
        ]
        assert "agent:builtin" not in table_keys
        # ...but present in the rail and selectable.
        await pilot.click("#mcp-rail-row-2")  # 0 All, 1 built-in, 2 agent
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert workbench._selected_server_key == "agent:builtin"
        assert app.query_one("#mcp-servers-detail").display is True
        assert list(app.query("#mcp-detail-tool-gates Checkbox"))
        assert not list(app.query("#mcp-detail-builtin-toggles Checkbox"))

        workbench.set_mode("permissions")
        await pilot.pause()
        preview = str(app.query_one("#mcp-perm-preview", Static).renderable)
        assert preview.startswith("global default:")


@pytest.mark.asyncio
async def test_agent_snapshot_state_follows_local_master_switch(monkeypatch):
    monkeypatch.setattr(
        mcp_workbench_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if key == "local_tools_enabled" else default
        ),
    )
    app = ProblemRecordsApp([])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        from tldw_chatbook.MCP.readiness import ReadinessState

        assert app.query_one(MCPWorkbench)._agent_snapshot is not None
        assert (
            app.query_one(MCPWorkbench)._agent_snapshot.state
            is ReadinessState.READY
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_mcp_workbench.py -k "agent_tools_rail_row or agent_snapshot_state" -q`
Expected: FAIL (`_agent_snapshot` missing / rail has no third row)

- [ ] **Step 3: Write minimal implementation**

In `mcp_workbench.py`:

1. Import: add `agent_tools_readiness` and `AGENT_TOOLS_SERVER_KEY` to the existing `from tldw_chatbook.MCP.readiness import (...)` block.
2. `__init__`: near `_snapshots`, add `self._agent_snapshot: ReadinessSnapshot | None = None`.
3. `compose`: the initial `MCPRail(...)` call gains `agent_snapshot=None`.
4. `_collect_snapshots`: at the top of the `if self._source == "local":` branch (before appending the built-in), set:

```python
            # ADR-148 Wave D: the agent-tools rail row is derived, not a
            # server -- it rides beside _snapshots so the overview table,
            # callouts, worst-state summary, and preselection heuristics
            # stay server-only.
            self._agent_snapshot = agent_tools_readiness(
                enabled=coerce_bool_setting(
                    get_cli_setting(
                        "console",
                        LOCAL_TOOLS_MASTER_KEY,
                        LOCAL_TOOLS_DEFAULT_ENABLED,
                    ),
                    LOCAL_TOOLS_DEFAULT_ENABLED,
                )
            )
```

and in the server-source branch set `self._agent_snapshot = None`. (Imports for `coerce_bool_setting`, `LOCAL_TOOLS_MASTER_KEY`, `LOCAL_TOOLS_DEFAULT_ENABLED` already exist in this module — verify with grep; `_local_agent_hub_tools` uses all three.)
5. `_snapshot_for`: after the existing loop lookup, add:

```python
        if (
            server_key == AGENT_TOOLS_SERVER_KEY
            and self._agent_snapshot is not None
        ):
            return self._agent_snapshot
```

6. `_sync_children`: the `rail.sync_state(...)` call gains `agent_snapshot=self._agent_snapshot`.

Routing needs NO new branch: `_show_selected_detail` routes through `canvas.show_detail(selected)`, whose source gates (Task 3) do the work; the inspector's `_wired_actions` already gives `source != "local"` the base action set.

- [ ] **Step 4: Run tests to verify they pass, then the full affected suites**

Run: `pytest Tests/UI/test_mcp_workbench.py -k "agent_tools_rail_row or agent_snapshot_state" -q`
Expected: PASS
Run: `pytest Tests/UI/test_mcp_workbench.py Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_servers_mode.py Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_mcp_inspector.py -q`
Expected: all PASS — any test that pinned gates-under-builtin or exact rail row counts is updated in the same commit (grep `mcp-rail-row-` and `mcp-detail-tool-gates` in Tests/ to find them).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_workbench.py Tests/UI/test_mcp_workbench.py
git commit -m "feat(mcp): workbench wires the Agent tools rail row end-to-end (ADR-148 Wave D)"
```

---

### Task 5: Docs truth + final verification

**Files:**
- Modify: `Docs/User_Guide/mcp.md` ("Other registration gates (Servers mode ▸ Tool gates)" section, line ~123)
- Test: `Tests/MCP/test_mcp_documentation_contract.py` (guard: failure count must not increase)

**Interfaces:**
- Consumes: shipped behavior from Tasks 1-4.
- Produces: user-guide truth matching the split.

- [ ] **Step 1: Update the user guide**

In `Docs/User_Guide/mcp.md`, the section currently opens "Select the built-in server's row in Servers mode; its detail pane has a **Tool gates** group under the existing enable/expose checkboxes". Rewrite the opening to:

```markdown
Select the **Agent tools** row in the rail (its own section under
Servers); its detail pane has a **Tool gates** group, split into two
subheadings:
```

and add one sentence after the subheadings list: "The built-in server's own row keeps only the `[mcp]` enable/expose controls — the two panes govern different subsystems by design (ADR-148)."

- [ ] **Step 2: Guard the doc contract**

Run: `pytest Tests/MCP/test_mcp_documentation_contract.py -q`
Expected: the SAME pre-existing failure count as the branch point (39 as of 2026-09-11; 26 mcp.md-scoped) — zero new failures. If any NEW failure names the gates section, fix the doc wording (not the test).

- [ ] **Step 3: Final verification sweep**

Run: `pytest Tests/UI/test_mcp_workbench.py Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_servers_mode.py Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_mcp_inspector.py Tests/UI/test_destination_shells.py -q`
Expected: MCP suites all PASS; destination shells show only the pre-existing failures documented in TASK-32454 (library/schedules/models drift + known flakiness). Ruff on touched files: no NEW findings vs HEAD~.

- [ ] **Step 4: Commit + task hygiene**

```bash
git add Docs/User_Guide/mcp.md
git commit -m "docs(mcp): registration gates live under the Agent tools row (ADR-148)"
```

Create/complete the backlog task (ADR check: covered by ADR-148, linked) with Implementation Notes naming every re-targeted test.

---

## Self-Review

- **Spec coverage:** second rail section (Task 2), gates move + pointer (Task 3), `agent:builtin` key + no-shared-key constraint (Tasks 1-2), unscoped preview (Task 4 test), view-state non-migration (no task needed — `agent:builtin` is new to the rail; old selections restore onto the built-in row unchanged, asserted implicitly by existing restore tests), label honesty `(external MCP)` suffix — **deferred**: the spec's label-suffix work belongs with the duplication-disambiguation follow-up; this plan's Task 5 documents the split only. Noted as an explicit deferral, not a silent gap.
- **Placeholder scan:** none; every step carries code or exact assertions.
- **Type consistency:** `agent_tools_readiness(*, enabled: bool)`, `agent_snapshot: ReadinessSnapshot | None`, `AGENT_TOOLS_SERVER_KEY = "agent:builtin"` used consistently across tasks.
