# MCP Hub Narrow-Width Triad Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Below 120 columns, stack the inspector under the canvas as a bounded scrolling band instead of squeezing three unreadable columns (ADR-148 decision 2).

**Architecture:** One wrapper container — `#mcp-hub-grid` gains `#mcp-hub-main-row` (a `Horizontal` holding the rail and canvas, ids unchanged) with the inspector as its sibling. The `.mcp-compact` class (already toggled by `on_resize`) flips the grid to `layout: vertical`, turning the same tree into main-row-above/inspector-below; the inspector becomes a `max-height: 12` internally scrolling band. Wide layouts are geometrically unchanged (same fr shares, one nesting level deeper). The Advanced collapsible auto-collapses while compact.

**Tech Stack:** Textual 8.x layout (`layout:` CSS override has in-repo precedent: `css/screen_feature_watchlists.tcss:144`), pytest + Pilot.

**Spec:** `Docs/superpowers/specs/2026-09-11-mcp-hub-narrow-width-triad-design.md`; ADR: `backlog/decisions/148-mcp-hub-rail-ia-and-responsive-triad.md`.

## Global Constraints

- Wide layouts (≥120 cols) geometrically unchanged: rail 3fr / canvas 5fr inside the main-row, inspector 3fr of the grid — same shares as today's flat 3/5/3.
- All existing pane ids stable (`#mcp-hub-rail`, `#mcp-hub-canvas`, `#mcp-hub-inspector`) — triad tests query unchanged.
- CSS lands in BOTH places, lockstep: `MCPWorkbench.BUNDLED_CSS` / `MCPInspector.BUNDLED_CSS` and the `MCPWorkbench #mcp-hub-*` block in `tldw_chatbook/css/widget_defaults_scoped.tcss` (~line 789).
- Band: `max-height: 12`, `min-height: 4`, `overflow-y: auto` at compact only; the persisted `advanced_visible` setting is never changed by the auto-collapse (widget state only).
- TDD: every behavior's test watched RED first; targeted suites only.

---

### Task 1: Wrapper container + stacking CSS + geometry tests

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (`compose`, `BUNDLED_CSS`)
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` (`BUNDLED_CSS` width 3fr stays; add nothing here — the band rules live in the workbench block)
- Modify: `tldw_chatbook/css/widget_defaults_scoped.tcss` (the `MCPWorkbench #mcp-hub-*` mirror block)
- Test: `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Produces: `#mcp-hub-main-row` (Horizontal) wrapping rail + canvas inside `#mcp-hub-grid`; `.mcp-compact` stacks the grid vertically and bounds the inspector band.

- [ ] **Step 1: Write the failing geometry tests**

Append to `Tests/UI/test_mcp_workbench.py`:

```python
# -- Wave E (2026-09-11 MCP Hub UX program, ADR-148): narrow-width triad --


@pytest.mark.asyncio
async def test_wide_layout_keeps_inspector_beside_the_main_row():
    """>=120 cols: the triad is geometrically unchanged -- the inspector
    sits BESIDE the main row (rail+canvas), full height, not as a band."""
    app = WorkbenchApp()
    async with app.run_test(size=(160, 44)) as pilot:
        await pilot.pause()
        main_row = app.query_one("#mcp-hub-main-row")
        inspector = app.query_one("#mcp-hub-inspector")
        assert inspector.region.x >= main_row.region.x + main_row.region.width - 1
        assert inspector.region.height > 12  # full pane, not a band


@pytest.mark.asyncio
async def test_compact_layout_stacks_inspector_below_as_bounded_band():
    """<120 cols (ADR-148): the inspector stacks BELOW the main row as a
    bounded, scrollable band (<=12 rows) at full width -- replacing the
    old squeezed third column that broke words mid-token."""
    app = WorkbenchApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        grid = app.query_one("#mcp-hub-grid")
        main_row = app.query_one("#mcp-hub-main-row")
        inspector = app.query_one("#mcp-hub-inspector")
        assert "mcp-compact" in grid.classes
        assert inspector.region.y >= main_row.region.y + main_row.region.height - 1
        assert inspector.region.height <= 12
        assert inspector.region.width >= main_row.region.width - 1
        # The rail+canvas row keeps (nearly) the full width.
        assert main_row.region.width >= 90
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_mcp_workbench.py -k "wide_layout_keeps or compact_layout_stacks" -q`
Expected: FAIL (`#mcp-hub-main-row` NoMatches)

- [ ] **Step 3: Implement**

1. `mcp_workbench.compose()`: wrap the rail and canvas in the wrapper —

```python
        with Horizontal(id="mcp-hub-grid", classes="destination-workbench"):
            with Horizontal(id="mcp-hub-main-row"):
                yield MCPRail(... id="mcp-hub-rail", ...)
                with ContentSwitcher(... id="mcp-hub-canvas", ...):
                    yield MCPServersMode(id="mcp-mode-canvas-servers")
            yield MCPInspector(id="mcp-hub-inspector", classes="destination-workbench-pane")
```

2. `MCPWorkbench.BUNDLED_CSS`: add main-row + stacking rules, and REPLACE the compact inspector squeeze rule:

```css
    #mcp-hub-main-row {
        width: 4fr;   /* 8 of the old 11 share-units; inspector keeps 3 */
        min-width: 0;
        height: 100%;
        min-height: 0;
    }
    /* ADR-148 Wave E: below 120 cols the grid stacks -- main row on top,
    inspector as a bounded, internally scrolling band underneath (the old
    squeezed third column broke words mid-token at ~20 cols). */
    #mcp-hub-grid.mcp-compact {
        layout: vertical;
    }
    #mcp-hub-grid.mcp-compact #mcp-hub-main-row {
        width: 100%;
        height: 1fr;
    }
    #mcp-hub-grid.mcp-compact #mcp-hub-inspector {
        width: 100%;
        height: auto;
        max-height: 12;
        min-height: 4;
        overflow-y: auto;
    }
```

Delete the old `#mcp-hub-grid.mcp-compact #mcp-hub-inspector { width: 2fr; min-width: 20; }` rule (both copies). Keep the compact rail/canvas rules (they now size within the main row).

3. `MCPInspector.BUNDLED_CSS` and its `widget_defaults_scoped.tcss` mirror: change the inspector's own `width: 3fr; min-width: 28` — the width rule MOVES to grid level; set `MCPInspector { width: 3fr; min-width: 28; height: 100%; }` unchanged (the 3fr now competes with main-row's 4fr at grid level, preserving the 3/11 share), and in the workbench BUNDLED_CSS + scoped mirror keep `#mcp-hub-main-row { width: 4fr }` — total grid fr: 4+3 → inspector 3/7 ≈ 43%?? — NO: correct the shares so the ratio matches the old 8:3 — use `#mcp-hub-main-row { width: 8fr }` and leave the inspector at its own 3fr. Verify with the wide geometry test that the inspector share is within ±2 columns of the pre-change layout (assert against a captured baseline from the green run of an untouched wide test, e.g. the existing triad mount test's regions).

4. Mirror every new/changed rule into the `MCPWorkbench #mcp-hub-*` block of `widget_defaults_scoped.tcss` (lockstep).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/UI/test_mcp_workbench.py -k "wide_layout_keeps or compact_layout_stacks" -q`
Expected: PASS. Then `pytest Tests/UI/test_mcp_workbench.py -k "100x30 or mounts_rail" -q` — the existing 100x30 layout test must still pass (master switch reachable inside the band).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_workbench.py tldw_chatbook/UI/MCP_Modules/mcp_inspector.py tldw_chatbook/css/widget_defaults_scoped.tcss Tests/UI/test_mcp_workbench.py
git commit -m "feat(mcp): compact triad stacks the inspector as a bounded band (ADR-148 Wave E)"
```

---

### Task 2: Advanced auto-collapse at compact + intact-content assertion

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` (new `apply_compact_layout()`)
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (`_sync_compact_class` calls it)
- Test: `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py`

**Interfaces:**
- Produces: `MCPInspector.apply_compact_layout(compact: bool) -> None` — when True and `#mcp-adv-collapsible` is open, collapse it (widget state only; the persisted `advanced_visible` flag is untouched).

- [ ] **Step 1: Write the failing tests**

In `Tests/UI/test_mcp_inspector.py` (widget-level):

```python
@pytest.mark.asyncio
async def test_apply_compact_layout_collapses_open_advanced_without_persisting():
    """ADR-148 Wave E: a compact terminal collapses an open Advanced
    collapsible (its JSON dumps are the least band-friendly content) but
    never touches the persisted advanced_visible preference."""
    app = InspectorApp()  # the file's existing bare harness
    async with app.run_test() as pilot:
        await pilot.pause()
        inspector = app.query_one(MCPInspector)
        collapsible = app.query_one("#mcp-adv-collapsible", Collapsible)
        collapsible.collapsed = False
        await pilot.pause()
        persisted = inspector._advanced_visible

        inspector.apply_compact_layout(True)
        await pilot.pause()
        assert app.query_one("#mcp-adv-collapsible", Collapsible).collapsed
        assert inspector._advanced_visible is persisted
```

(Adapt the harness name/monkeypatch of `get_cli_setting("mcp.hub_state", "advanced_visible", ...)` to whatever the file's existing advanced tests use so the collapsible is composed.)

In `Tests/UI/test_mcp_workbench.py` (integration):

```python
@pytest.mark.asyncio
async def test_compact_band_renders_select_prompts_without_mid_word_breaks():
    """The screenshot evidence for ADR-148 showed the Section select
    clipped to 'Overvi|ew' at 100 cols. At the same size with the stacked
    band, the full prompt renders on one line."""
    app = WorkbenchApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        section_select = app.query_one("#mcp-adv-section-select", Select)
        assert section_select.region.width >= 9
        rendered = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in app.screen._compositor.render_strips()
        )
        assert "Overview" in rendered
```

(Guard: this test needs the Advanced area composed — monkeypatch the inspector module's `get_cli_setting` to `advanced_visible=True` exactly as the file's existing advanced-visibility tests do; if that harness pattern differs, mount with the workbench's `_reveal_advanced` path instead.)

- [ ] **Step 2: Run tests to verify they fail** — `pytest ... -k "apply_compact_layout or mid_word_breaks" -q` → FAIL (method missing / select still clipped).

- [ ] **Step 3: Implement**

In `MCPInspector`:

```python
    def apply_compact_layout(self, compact: bool) -> None:
        """ADR-148 Wave E: react to the workbench's compact-mode toggle.

        Compact: collapse an OPEN Advanced collapsible -- its JSON dumps
        are the least band-friendly content at 100 cols -- WITHOUT
        touching the persisted `advanced_visible` preference (widget
        state only; leaving compact never re-expands it for you).
        """
        if not compact:
            return
        try:
            collapsible = self.query_one("#mcp-adv-collapsible", Collapsible)
        except NoMatches:
            return
        if not collapsible.collapsed:
            collapsible.collapsed = True
```

In `MCPWorkbench._sync_compact_class`, after the `set_class(...)` line:

```python
        try:
            self.query_one(MCPInspector).apply_compact_layout(
                0 < width < _COMPACT_WIDTH
            )
        except Exception:
            pass  # pre-compose / torn-down subtree: nothing to collapse
```

- [ ] **Step 4: Run tests to verify they pass** — the two new tests plus `pytest Tests/UI/test_mcp_inspector.py -k advanced -q`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_inspector.py tldw_chatbook/UI/MCP_Modules/mcp_workbench.py Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py
git commit -m "feat(mcp): Advanced auto-collapses in the compact band; prompts render intact (ADR-148 Wave E)"
```

---

### Task 3: Per-mode compact smoke + final sweep + hygiene

**Files:**
- Test: `Tests/UI/test_mcp_workbench.py` (new per-mode smoke)
- Modify: `Docs/User_Guide/mcp.md` (one paragraph in "What this screen is for")

**Interfaces:** none new.

- [ ] **Step 1: Write the failing smoke test** (it should already pass after Tasks 1-2 — if it passes immediately, that is the point: it pins the shipped behavior; keep it as the regression net. If any mode fails, fix the mode's compact CSS before proceeding.)

```python
@pytest.mark.asyncio
async def test_every_mode_renders_usably_at_100x30():
    """ADR-148 Wave E sweep: each mode's canvas renders at 100x30 with the
    stacked band present and no mid-word-broken status line."""
    app = WorkbenchApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        workbench = app.query_one(MCPWorkbench)
        for mode, canvas_id in (
            ("servers", "mcp-mode-canvas-servers"),
            ("tools", "mcp-mode-canvas-tools"),
            ("permissions", "mcp-mode-canvas-permissions"),
            ("audit", "mcp-mode-canvas-audit"),
        ):
            workbench.set_mode(mode)
            await pilot.pause()
            canvas = app.query_one(f"#{canvas_id}")
            assert canvas.display
            assert canvas.region.height > 3
        assert app.query_one("#mcp-hub-inspector").region.height <= 12
```

- [ ] **Step 2: Docs** — append to the end of "What this screen is for" in `Docs/User_Guide/mcp.md`:

```markdown
Below ~120 terminal columns the layout adapts: the detail panel stacks
under the main area as a compact, scrollable band instead of squeezing
three unreadable columns.
```

- [ ] **Step 3: Final sweep** — `pytest Tests/UI/test_mcp_workbench.py Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_rail.py Tests/UI/test_mcp_servers_mode.py Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_destination_shells.py -k "mcp or hub or triad or compact or 100x30" -q` then the full five MCP suites; doc-contract guard (39 pre-existing failures, zero new); ruff before/after on touched files.

- [ ] **Step 4: Commit + backlog task notes + Done.**

---

## Self-Review

- **Spec coverage:** stacking (T1), band bounds + internal scroll (T1 CSS), Advanced auto-collapse + word hygiene verification (T2), per-mode 100x30 assertions (T3), F-057 column dropping unchanged (constraint), rail legend abbreviation already shipped in Wave A. Spec open question 1 (band cap scaling with terminal height) resolved to the fixed 12 default; open question 2 (rail Source select at compact) resolved to keeping current compact rail sizing — both recorded here as the defaults the user may override.
- **Placeholder scan:** none.
- **Type consistency:** `apply_compact_layout(compact: bool)`; `#mcp-hub-main-row` id used consistently; fr shares 8:3 preserve the wide geometry (verification step included in T1).
