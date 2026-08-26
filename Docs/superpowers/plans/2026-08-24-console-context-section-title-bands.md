# Console Context Section Title Bands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every direct Context section header the same raised full-width title-band hierarchy as Inspect without changing geometry, ceilings, scrolling, focus, or disclosure behavior.

**Architecture:** Keep `DestinationRailSectionHeader` as the Context owner and preserve its DOM, ids, two-row measured footprint, and interaction wiring. Add only the existing Inspector group-heading surface, text, and inset tokens to the Context header selector, then regenerate the canonical CSS bundle. A production-shaped Textual test compares computed Context and Inspect styles and pins the unchanged header/body contracts.

**Tech Stack:** Python 3.12, Textual 8.2.8, pytest/Textual `run_test`, TCSS, canonical CSS builder, Ruff.

**Approved design:** `Docs/superpowers/specs/2026-08-22-console-edge-rails-workspace-tree-design.md`

**Backlog task:** `TASK-20937.6`

**ADR required:** no new ADR.

**ADR path:** `backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md`

**Reason:** This is small visual polish within ADR-083's existing Context/Inspect ownership and layout boundaries; it does not change structure, data ownership, interaction contracts, or runtime boundaries.

---

## File responsibility map

- `Tests/UI/test_console_left_rail.py` — production-shaped style parity and preserved Context header/body contracts; its harness gains an opt-in app-bundle path for this computed-style test.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` — source selector for the Context title-band appearance.
- `tldw_chatbook/css/tldw_cli_modular.tcss` — generated canonical bundle; regenerate, never hand-edit.
- `backlog/tasks/task-20937.6 - Verify-and-document-Console-edge-rails-and-workspace-ownership.md` — AC #7 and implementation evidence.

## Task 1: Pin Context/Inspect title-band parity RED

**Files:**

- Modify: `Tests/UI/test_console_left_rail.py`

- [x] **Step 1: Add the production-shaped failing test**

Import `BUNDLED_STYLESHEET` from `Tests.UI.consolidated_css`. Extend the existing
`make_console_pilot` helper with `production_styles: bool = False`; when true,
instantiate a tiny local `ConsoleHarness` subclass whose
`CSS_PATH = str(BUNDLED_STYLESHEET)`. Leave every existing caller on the current
consolidated widget/screen CSS path.

Append a test that opens Inspect through
`make_console_pilot(size=(160, 45), production_styles=True)`, uses
`#console-inspector-run-heading` as the incumbent visual reference, and checks
all seven direct Context headers:

```python
@pytest.mark.asyncio
async def test_context_section_headers_match_inspector_title_band() -> None:
    async with make_console_pilot(
        size=(160, 45), production_styles=True
    ) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#console-inspector-rail-open")
        await pilot.pause()

        inspector_heading = screen.query_one("#console-inspector-run-heading")
        section_ids = (
            "session",
            "workspace",
            "conversations",
            "model",
            "agent",
            "details",
            "character",
        )
        for section_id in section_ids:
            header = screen.query_one(
                f"#console-rail-section-header-{section_id}",
                DestinationRailSectionHeader,
            )
            title = screen.query_one(f"#console-rail-section-title-{section_id}")
            toggle = screen.query_one(
                f"#console-rail-section-toggle-{section_id}", Button
            )

            assert header.styles.background == inspector_heading.styles.background
            assert header.styles.color == inspector_heading.styles.color
            assert header.styles.padding == inspector_heading.styles.padding
            assert title.styles.text_style.bold
            assert title.styles.color == inspector_heading.styles.color
            assert toggle.parent is header
            assert header.region.height == 2
            assert (
                header.region.width
                == header.parent.scrollable_content_region.width
            )
            assert header.content_region.contains_region(toggle.region)

        sections = list(
            screen.query("#console-left-rail-body ConsoleBoundedSection")
        )
        assert [section.max_content_lines for section in sections] == [
            15,
            20,
            20,
            15,
            15,
            15,
            35,
        ]
```

If the mounted Inspector state omits `#console-inspector-run-heading`, use the
first `.console-inspector-group-heading` produced by the same real rail; do not
introduce a fake style oracle or parse TCSS text.

- [x] **Step 2: Run the new node and verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_left_rail.py::test_context_section_headers_match_inspector_title_band \
  -q
```

Expected: FAIL on Context background and/or padding inequality while the real
Inspector heading, all seven Context headers, and their controls mount normally.
Collection, selector, or harness failures are not the intended RED and must be
fixed before production CSS changes.

## Task 2: Apply the minimum shared-token style and regenerate CSS

**Files:**

- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify (generated): `tldw_chatbook/css/tldw_cli_modular.tcss`

- [x] **Step 1: Add only the title-band appearance to the existing selector**

Extend `.console-rail-section-header` without changing its existing height,
minimum height, border, layout, or content alignment:

```tcss
.console-rail-section-header {
    height: auto;
    min-height: 2;
    border-top: solid $ds-column-line;
    padding: 0 1;
    background: $ds-surface-raised;
    color: $ds-text-primary;
    content-align: center middle;
}
```

Do not add a new class, replace `DestinationRailSectionHeader`, or modify the
Inspector selector. The Context title already owns bold primary text and the
toggle already owns focus styling.

- [x] **Step 2: Rebuild the canonical stylesheet**

Run:

```bash
../../.venv/bin/python -m tldw_chatbook.css.build_css
```

Expected: the generated modular bundle changes only by the corresponding
Context-header declarations and its generated metadata.

- [x] **Step 3: Run the RED node and verify GREEN**

Run the Task 1 Step 2 command.

Expected: PASS.

- [x] **Step 4: Prove the test is load-bearing**

Temporarily remove `background`, `color`, and `padding` from the source selector,
rebuild CSS, and rerun the node. Expected: FAIL on computed style parity. Restore
the declarations, rebuild again, and rerun. Expected final: PASS.

## Task 3: Run proportional gates, record the result, and commit

**Files:**

- Verify: `Tests/UI/test_console_left_rail.py`
- Verify: `Tests/UI/test_console_edge_rail_geometry.py`
- Verify: `Tests/UI/test_console_rail_reconciliation.py`
- Verify: `Tests/UI/test_css_build_integrity.py`
- Modify: `backlog/tasks/task-20937.6 - Verify-and-document-Console-edge-rails-and-workspace-ownership.md`

- [x] **Step 1: Run the focused behavior and geometry gate**

```bash
../../.venv/bin/python -B -m pytest \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_edge_rail_geometry.py \
  Tests/UI/test_console_rail_reconciliation.py \
  Tests/UI/test_css_build_integrity.py -q
```

Expected: all selected tests pass with only existing documented warnings.

- [x] **Step 2: Run scoped static and generated-artifact checks**

```bash
../../.venv/bin/python -m ruff check Tests/UI/test_console_left_rail.py
../../.venv/bin/python -m ruff format --check Tests/UI/test_console_left_rail.py
git diff --check
```

Expected: all commands exit 0; generated CSS is synchronized.

- [x] **Step 3: Run the Impeccable mechanical detector once**

```bash
node ../../.agents/skills/impeccable/scripts/detect.mjs --json \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  Tests/UI/test_console_left_rail.py
```

Expected: no blocking UI-quality finding attributable to the title-band change.

- [x] **Step 4: Record implementation evidence without closing the task**

Check AC #7 and append a concise note naming the CSS-only approach, RED/GREEN
node, mutation proof, focused test count, Ruff/CSS/diff results, and unchanged
geometry/ceilings. Keep TASK-20937.6 In Progress because fresh iTerm2 and Windows
Terminal captures are still required on the corrected product source.

- [x] **Step 5: Commit and push the implementation boundary**

```bash
git add \
  Docs/superpowers/plans/2026-08-24-console-context-section-title-bands.md \
  Tests/UI/test_console_left_rail.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  'backlog/tasks/task-20937.6 - Verify-and-document-Console-edge-rails-and-workspace-ownership.md'
git commit -m "style(console): distinguish Context section titles"
git push origin codex/task-20937-6-closeout
```

Expected: the branch is clean and pushed; TASK-20937.6 remains In Progress.
