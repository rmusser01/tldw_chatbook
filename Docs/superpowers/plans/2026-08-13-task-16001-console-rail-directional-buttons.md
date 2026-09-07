# Directional Full-Button Console Rail Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give both Console rails inward-pointing collapsed handles and outward-pointing full-width open-header collapse Buttons with the exact approved ASCII labels.

**Architecture:** Keep rail state, shared glyph ownership, IDs, handlers, widths, and responsive rules unchanged. Translate only the two canonical horizontal collapsed labels in `ConsoleRailHandle`, then replace each Console rail header's `Static` title plus three-cell Button with one existing-ID Button that fills the same one-row header; use instance-local inline sizing/alignment so shared Lab/Personas styling is untouched.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich cell geometry, pytest/Textual Pilot.

**Design spec:** `Docs/superpowers/specs/2026-08-13-task-16001-console-rail-directional-buttons-design.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a reversible Console-only presentation and hit-target correction using existing widgets, IDs, handlers, state, and layout boundaries. ADR-034's shared disclosure-glyph ownership remains unchanged.

---

## File map

- Modify `Tests/UI/test_console_rail_handle.py`: replace the superseded Context/Inspector copy expectations with exact inward-pointing horizontal labels while preserving vertical and noncanonical controls.
- Modify `Tests/UI/test_destination_rail.py`: pin mounted collapsed labels, tooltips, unchanged geometry, and shared-handle isolation.
- Modify `Tests/UI/test_console_inspector_compact_access.py`: keep the real 90-column Context open/persistence path but require `Context->`.
- Modify `Tests/UI/test_console_right_rail.py`: require `<-Inspect`, then prove the open Inspector's title side belongs to the full-width collapse Button.
- Modify `Tests/UI/test_console_left_rail.py`: prove the open Context header is one full-width Button and its title side collapses the rail.
- Modify `Tests/UI/test_console_rail_title.py`: replace the obsolete requirement for a `#console-context-rail-title` Static with the single-Button header contract.
- Modify `Tests/UI/test_console_shell_regions.py`: update only horizontal collapsed copy while preserving widths, stacked mode, and both toggle paths.
- Modify `Tests/UI/test_settings_console_rail_labels.py`: update the failed-save horizontal copy while preserving the active-style contract.
- Modify `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`: pin both real collapsed Button labels and tooltips.
- Modify `Tests/UI/test_workbench_visual_snapshots.py`: replace the superseded Context-only sweep with a real-Console two-viewport state-transition sweep that paints all four approved labels.
- Modify `tldw_chatbook/Widgets/Console/console_rail_handle.py`: translate canonical horizontal Context/Inspector labels only.
- Modify `tldw_chatbook/UI/Console_Modules/left_rail.py`: replace the Context title/tiny glyph pair with one full-width right-aligned Button.
- Modify `tldw_chatbook/UI/Console_Modules/right_rail.py`: replace the Inspector title/tiny glyph pair in `ConsoleInspectorRail` with one full-width left-aligned Button.
- Modify `backlog/tasks/task-16001 - Make-Console-rail-controls-directional-full-buttons.md`: complete ACs and record scoped evidence.

### Task 1: Replace obsolete RED tests with the corrected directional contract

**Files:**
- Modify: `Tests/UI/test_console_rail_handle.py:189-209`
- Modify: `Tests/UI/test_destination_rail.py:350-390`
- Modify: `Tests/UI/test_console_inspector_compact_access.py:161-210`
- Modify: `Tests/UI/test_console_right_rail.py:85-130`
- Modify: `Tests/UI/test_console_left_rail.py`
- Modify: `Tests/UI/test_console_rail_title.py`
- Modify: `Tests/UI/test_console_shell_regions.py:112-165`
- Modify: `Tests/UI/test_settings_console_rail_labels.py:190-205`
- Modify: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py:232-245`
- Modify: `Tests/UI/test_workbench_visual_snapshots.py:307-375`

- [ ] **Step 1: Correct the pure and mounted collapsed-label contracts**

In `test_console_rail_handle.py`, replace the superseded labels while retaining the noncanonical and vertical controls:

```python
def test_horizontal_canonical_labels_use_inward_console_button_copy() -> None:
    context = _handle(label=CONSOLE_RAIL_CONTEXT_LABEL)
    inspector = _handle(label=CONSOLE_RAIL_INSPECTOR_LABEL, side="right")

    assert context._display_label() == "Context->"
    assert inspector._display_label() == "<-Inspect"
```

Keep `test_horizontal_noncanonical_left_label_is_unchanged`, and add the symmetric right-side guard:

```python
def test_horizontal_noncanonical_right_label_is_unchanged() -> None:
    handle = _handle(label="Review", side="right")

    assert handle._display_label() == "Review"
```

In `test_destination_rail.py`, rename/update the mounted canonical tests so they require `Context->` and `<-Inspect`, preserve both fixed tooltips, and retain the existing 13/11 Context and 11/9 Inspector geometry/containment assertions. Keep the generic `DestinationRailHandle` controls unchanged to prove the copy does not leak into shared rails.

Use the explicit names `test_console_handle_uses_inward_context_label_on_the_left` and `test_console_handle_uses_inward_inspector_label_on_the_right` for the two mounted canonical tests.

- [ ] **Step 2: Correct existing real-Console collapsed consumers**

Change only horizontal canonical copy:

```python
# Tests/UI/test_console_inspector_compact_access.py
assert context_button.label == "Context->"

# Tests/UI/test_console_right_rail.py
assert str(open_button.label) == "<-Inspect"

# Tests/UI/test_console_shell_regions.py
(False, 13, 11, "Context->", "<-Inspect"),

# Tests/UI/test_settings_console_rail_labels.py
assert left._display_label() == "Context->"
assert right._display_label() == "<-Inspect"

# Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py
assert context_button.label == "Context->"
assert inspector_button.label == "<-Inspect"
```

Preserve selector-relative handle clicks, tooltips, widths, stacked labels, settings state, Inspector badge behavior, and persistence assertions.

- [ ] **Step 3: Add open Context full-header presentation and non-arrow click coverage**

In `test_console_left_rail.py`, import `Button` and `Horizontal`. Add a real-Console test using the existing `make_console_pilot()` harness:

```python
@pytest.mark.asyncio
async def test_context_header_is_one_full_width_collapse_button() -> None:
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        header = screen.query_one("#console-context-rail-collapse").parent
        button = screen.query_one("#console-context-rail-collapse", Button)

        assert isinstance(header, Horizontal)
        assert list(header.children) == [button]
        assert str(button.label) == "<---------|Context"
        assert button.tooltip == "Collapse Console context rail"
        assert header.content_region.contains_region(button.region)
        assert button.region.width == header.content_region.width
        assert button.region.height == 1
        assert button.styles.text_align == "right"
        assert button.styles.content_align_horizontal == "right"
```

Add a second test that clicks the title end rather than the arrow:

```python
@pytest.mark.asyncio
async def test_clicking_context_header_title_end_collapses_the_rail() -> None:
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        button = screen.query_one("#console-context-rail-collapse", Button)
        title_end = (button.region.width - 2, 0)

        assert await pilot.click(button, offset=title_end)
        await pilot.pause(0.2)
        assert screen.query_one("#console-left-rail").display is False
        assert screen.query_one("#console-context-rail-handle").display is True
```

The existing ID keeps the same production handler and persisted preference path; do not add direct handler calls.

Replace `test_console_rail_title_reads_console_context` in `test_console_rail_title.py` with `test_console_context_rail_header_uses_the_full_collapse_button`. Require the obsolete title selector to be absent and the existing collapse Button to carry `<---------|Context`; this prevents the old structure contract from contradicting the new header after implementation.

- [ ] **Step 4: Add open Inspector full-header presentation and non-arrow click coverage**

In `test_console_right_rail.py`, add `test_inspector_header_is_one_full_width_collapse_button`: open the Inspector through its collapsed handle, re-query after recompose, and require the header has one child Button with exact label `Inspect|--------->`, tooltip, full content-region width, height one, and left text/content alignment. Add `test_clicking_inspector_header_title_start_collapses_the_rail` using a title-side coordinate:

```python
button = pilot.app.screen.query_one("#console-inspector-rail-collapse", Button)
title_start = (1, 0)
assert await pilot.click(button, offset=title_start)
```

Retain the existing post-collapse visibility/focus assertions. The click must target the Button object, not a screen-only coordinate.

- [ ] **Step 5: Replace the superseded visual sweep with all four states**

Rename the TASK-16001 visual test to `test_task_16001_console_directional_rail_buttons_visual_sweep` and parameterize `(140, 42)` and `(160, 45)` plus the four applicable rail states. Each parameter independently drives the real `TldwCli` to its target state using the currently mounted controls, so RED exercises every state instead of stopping at the first expected mismatch. In each run:

1. Open Console and force one of: Context open / Inspector collapsed; both open; Context collapsed / Inspector open; both collapsed.
2. Before any intended copy/structure assertion, require the target rail visibility, handle/header containment, expected unchanged widths, full one-row geometry, and positive transcript width; export a healthy SVG and extract rendered `<text>` independently of its title.
3. Only after those setup/precondition checks, compare the applicable controls to the exact expected label, unfiltered compositor row, title-selector absence, one-child header structure, full-width Button, tooltip, and rendered SVG text.
4. Keep the dedicated left/right rail tests as the authority for non-arrow title clicks; the visual cases use ordinary mounted controls only to reach each independent target state during RED.

At every state assert the transcript remains positive-width and the applicable button/handle stays contained. In TASK-15783, change only the two collapsed Inspector copy assertions from `Inspect->` to `<-Inspect`; preserve its name, six viewport/badge parameters, geometry, frame, badge, containment, transcript, and SVG contracts unchanged.

- [ ] **Step 6: Run the focused RED gates**

Run only the changed/related nodes:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_rail_handle.py::test_horizontal_canonical_labels_use_inward_console_button_copy \
  Tests/UI/test_console_rail_handle.py::test_horizontal_noncanonical_left_label_is_unchanged \
  Tests/UI/test_console_rail_handle.py::test_horizontal_noncanonical_right_label_is_unchanged \
  Tests/UI/test_destination_rail.py::test_console_handle_uses_inward_context_label_on_the_left \
  Tests/UI/test_destination_rail.py::test_console_handle_uses_inward_inspector_label_on_the_right \
  Tests/UI/test_console_inspector_compact_access.py::test_left_handle_opens_left_rail_at_90_cols \
  Tests/UI/test_console_left_rail.py::test_context_header_is_one_full_width_collapse_button \
  Tests/UI/test_console_left_rail.py::test_clicking_context_header_title_end_collapses_the_rail \
  Tests/UI/test_console_rail_title.py::test_console_context_rail_header_uses_the_full_collapse_button \
  Tests/UI/test_console_right_rail.py::test_clicking_open_then_collapse_toggles_visibility_and_persists \
  Tests/UI/test_console_right_rail.py::test_inspector_header_is_one_full_width_collapse_button \
  Tests/UI/test_console_right_rail.py::test_clicking_inspector_header_title_start_collapses_the_rail \
  Tests/UI/test_console_shell_regions.py::test_fresh_console_composes_saved_rail_label_style \
  Tests/UI/test_settings_console_rail_labels.py::test_console_rail_label_failed_save_keeps_draft_and_active_style \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions \
  Tests/UI/test_workbench_visual_snapshots.py::test_task_15783_console_collapsed_inspector_rail_visual_parity_sweep \
  Tests/UI/test_workbench_visual_snapshots.py::test_task_16001_console_directional_rail_buttons_visual_sweep
```

Expected: canonical collapsed-label tests fail on current `Context ▸`/`Inspect->`; all six TASK-15783 parameters fail only on current `Inspect->` versus `<-Inspect`; open-header tests fail on the separate `Static` plus three-cell glyph Button; noncanonical and vertical controls pass. Every TASK-16001 visual case must reach an intended copy/structure mismatch, not fail on setup, containment, clipping, or transcript geometry.

- [ ] **Step 7: Commit the corrected RED tests**

```bash
git add -- \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_rail_title.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_workbench_visual_snapshots.py
git commit -m "test(console): require directional full-button rails"
```

### Task 2: Implement the three Console-local presentation changes

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_rail_handle.py:95-101`
- Modify: `tldw_chatbook/UI/Console_Modules/left_rail.py:52-71,247-273`
- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py:61-69,158-184`

- [ ] **Step 1: Correct the collapsed horizontal display seam**

Keep vertical handling first and translate only canonical labels:

```python
def _display_label(self) -> str:
    """Return compact visible text while preserving full tooltips."""
    if self.vertical:
        return self._stack_vertical_label(self.label)
    if self.side == "left":
        return "Context->" if self.label == CONSOLE_RAIL_CONTEXT_LABEL else self.label
    return "<-Inspect" if self.label == CONSOLE_RAIL_INSPECTOR_LABEL else self.label
```

Do not change `compose()` geometry: `Context->` fits the left handle's existing nine painted cells under its default padding, and `<-Inspect` fits the existing right-side nine-cell content after its current `line_pad=0`. Do not change canonical rail-state constants or shared `DestinationRailHandle` glyphs.

- [ ] **Step 2: Replace the Context title/tiny Button pair**

In `ConsoleLeftRail.compose()`, keep the one-row `Horizontal` header but remove the title `Static`. Create the existing-ID Button directly:

```python
collapse_button = Button(
    "<---------|Context",
    id="console-context-rail-collapse",
    classes="console-rail-collapse-button",
    compact=True,
)
collapse_button.tooltip = "Collapse Console context rail"
collapse_button.styles.width = "100%"
collapse_button.styles.min_width = 0
collapse_button.styles.max_width = "100%"
collapse_button.styles.text_align = "right"
collapse_button.styles.content_align = ("right", "middle")
yield collapse_button
```

Remove the now-unused `GLYPH_COLLAPSE_LEFT` and `resolve_glyph` imports. Keep `Static`; the left rail uses it throughout its body.

- [ ] **Step 3: Replace the Inspector title/tiny Button pair**

Mirror the Context structure in `ConsoleInspectorRail.compose()`:

```python
collapse_button = Button(
    "Inspect|--------->",
    id="console-inspector-rail-collapse",
    classes="console-rail-collapse-button",
    compact=True,
)
collapse_button.tooltip = "Collapse Inspector rail"
collapse_button.styles.width = "100%"
collapse_button.styles.min_width = 0
collapse_button.styles.max_width = "100%"
collapse_button.styles.text_align = "left"
collapse_button.styles.content_align = ("left", "middle")
yield collapse_button
```

Remove the now-unused `Static`, `GLYPH_COLLAPSE_RIGHT`, and `resolve_glyph` imports from `right_rail.py`. Do not change the shared `.console-rail-collapse-button` rule; inline instance styles keep Lab/Personas controls at their existing three-cell width.

- [ ] **Step 4: Run the exact focused GREEN gates from Task 1**

Run the Task 1 Step 6 commands again. Expected: every selected case passes, including both visual parameters and stacked/noncanonical controls.

- [ ] **Step 5: Mutation-check both non-arrow header clicks**

Temporarily change each selector-relative click to its first out-of-bounds x-coordinate:

```python
outside = (button.region.width, 0)
```

Run the two exact title-click nodes separately. Each must fail because the click returns false or its rail remains open. Restore the in-bounds Context `(width - 2, 0)` and Inspector `(1, 0)` coordinates, rerun both nodes, and require PASS.

- [ ] **Step 6: Commit the production implementation**

```bash
git add -- \
  tldw_chatbook/Widgets/Console/console_rail_handle.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Console_Modules/right_rail.py
git commit -m "fix(console): make rail directions and hit targets clear"
```

### Task 3: Run scoped regressions and close TASK-16001

**Files:**
- Modify: `backlog/tasks/task-16001 - Make-Console-rail-controls-directional-full-buttons.md`
- Modify: `Tests/UI/test_console_tab_scope.py`
- Add: `Docs/superpowers/plans/2026-08-13-task-16001-console-rail-directional-buttons.md`

- [ ] **Step 1: Run directly related regression tests only**

First, update the existing Console focus-tour regression to tolerate the
optional in-transcript provider-recovery action while still requiring the
tour to reach the status chips and Inspector within ten stops. This is a
test-harness correction only; do not alter production focus behavior.

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_shell_regions.py::test_fresh_console_composes_saved_rail_label_style \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_tab_scope.py \
  Tests/UI/test_workbench_pane_focus.py \
  Tests/UI/test_css_build_integrity.py \
  Tests/UI/test_console_internals_decomposition.py::test_console_workbench_panes_have_visible_terminal_frames
```

Then run the related compositor tests:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_workbench_visual_snapshots.py \
  -k 'task_15783 or task_16001'
```

Expected: all selected tests pass. Per user instruction, do not run the full repository suite.

- [ ] **Step 2: Run focused static, format, design, and integrity checks**

Run Ruff only over the three production and ten modified test files, then `git diff --check`. Run the Impeccable detector over the three production UI files and require no new finding relative to the pre-edit baseline. Run the exact duplicate-task-ID guard from `.github/workflows/backlog-guard.yml`. No CSS source changes are planned, so `test_css_build_integrity.py` is the bundle guard and no bundle regeneration should occur.

- [ ] **Step 3: Self-review the complete diff**

Run:

```bash
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_chatbook/Widgets/Console/console_rail_handle.py \
  tldw_chatbook/UI/Console_Modules/left_rail.py \
  tldw_chatbook/UI/Console_Modules/right_rail.py \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_left_rail.py \
  Tests/UI/test_console_rail_title.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_workbench_visual_snapshots.py
```

Confirm the final diff contains no active `Context--->` or `Inspect->` expectation, no separate Console rail title Static, no shared CSS/glyph/state/handler change, no dynamic dash logic, and no width/responsive-policy change.

- [ ] **Step 4: Complete the task record**

Edit the five-digit task file directly with `apply_patch` because this repository's Backlog CLI has a known five-digit edit parsing failure. Check AC #1-5, set `status: Done`, and add concise Implementation Notes with exact scoped test counts, mutation results, lint/format/detector/integrity evidence, no-full-suite note, and ADR `no` / path `N/A`. Do not mark Done while any directly related gate is red.

- [ ] **Step 5: Commit closeout documentation and verify cleanliness**

```bash
git add -- \
  'backlog/tasks/task-16001 - Make-Console-rail-controls-directional-full-buttons.md' \
  Docs/superpowers/plans/2026-08-13-task-16001-console-rail-directional-buttons.md
git commit -m "docs(console): close TASK-16001"
git status --short
git diff --check origin/dev...HEAD
```

Expected: clean worktree and no whitespace errors.
