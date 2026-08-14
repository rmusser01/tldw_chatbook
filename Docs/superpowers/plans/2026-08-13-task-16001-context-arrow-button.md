# Full-Width Context Arrow Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render `Context--->` as the complete eleven-cell horizontal collapsed Context button while preserving every existing rail geometry, tooltip, vertical-mode, Inspector, and responsive contract.

**Architecture:** Keep the shared `DestinationRailHandle`, CSS files, canonical rail-state label, and layout geometry unchanged. Use the existing `ConsoleRailHandle._display_label()` presentation seam for one fixed eleven-cell literal and clear Textual's default `line_pad=1` inline on the existing horizontal left Button so the label occupies its existing eleven-cell content region; strengthen component, mounted, interaction, settings, and real-Console compositor regressions around those seams.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich cell geometry, pytest/Textual Pilot.

**Design spec:** `Docs/superpowers/specs/2026-08-13-task-16001-context-arrow-button-design.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a reversible presentation refinement inside the existing Console-specific rail display seam. It changes no ownership boundary, public interface, persistence model, dependency, security policy, or long-lived application structure.

---

## File map

- Modify `Tests/UI/test_console_rail_handle.py`: pin the exact canonical horizontal Context display while preserving vertical, noncanonical-left, and Inspector variants.
- Modify `Tests/UI/test_destination_rail.py`: pin the mounted Context button label, tooltip, and unchanged 13/11 geometry while retaining the shared-handle control.
- Modify `Tests/UI/test_console_inspector_compact_access.py`: click the last cell of the real Context button at compact width and prove the existing open/persistence path still runs.
- Modify `Tests/UI/test_console_shell_regions.py`: update the saved horizontal rail-style expectation while preserving stacked mode, widths, and both open/collapse paths.
- Modify `Tests/UI/test_settings_console_rail_labels.py`: update the mounted horizontal Context expectation in the failed-save state while preserving the settings behavior.
- Modify `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`: pin the real core-loop Context button label and tooltip alongside the existing Inspector assertion.
- Modify `Tests/UI/test_workbench_visual_snapshots.py`: add a three-viewport real-Console compositor sweep that forces Context collapsed and proves one painted `Context--->` row, unchanged geometry, containment, and Inspector preservation.
- Modify `tldw_chatbook/Widgets/Console/console_rail_handle.py`: translate only the canonical horizontal Context label and clear line padding on the existing horizontal left Button.
- Modify `backlog/tasks/task-16001 - Make-Context-arrow-a-full-button-label.md`: complete acceptance criteria and record scoped verification after fresh evidence.

### Task 1: Pin the full Context button and arrow-end hit target

**Files:**
- Modify: `Tests/UI/test_console_rail_handle.py:188-201`
- Modify: `Tests/UI/test_destination_rail.py:315-370`
- Modify: `Tests/UI/test_console_inspector_compact_access.py:161-199`
- Modify: `Tests/UI/test_console_shell_regions.py:112-158`
- Modify: `Tests/UI/test_settings_console_rail_labels.py:190-205`
- Modify: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py:220-242`
- Modify: `Tests/UI/test_workbench_visual_snapshots.py:170-285`

- [ ] **Step 1: Write the pure display regressions**

Rename the horizontal display test to describe both canonical Console labels and require the approved Context copy:

```python
def test_horizontal_canonical_labels_use_console_button_copy() -> None:
    context = _handle(label=CONSOLE_RAIL_CONTEXT_LABEL)
    inspector = _handle(label=CONSOLE_RAIL_INSPECTOR_LABEL, side="right")

    assert context._display_label() == "Context--->"
    assert inspector._display_label() == "Inspect->"
```

Add an explicit noncanonical-left preservation guard:

```python
def test_horizontal_noncanonical_left_label_is_unchanged() -> None:
    context = _handle(label="Sources")

    assert context._display_label() == "Sources"
```

Leave `test_vertical_context_label_stacks_without_direction_glyph` unchanged so vertical Context remains `C\no\nn\nt\ne\nx\nt`.

- [ ] **Step 2: Strengthen the mounted Context contract**

Add a focused mounted Console-handle test next to the Inspector label test in `test_destination_rail.py`:

```python
@pytest.mark.asyncio
async def test_console_handle_uses_the_full_context_arrow_button_on_the_left():
    handle = _console_handle(side="left", label=CONSOLE_RAIL_CONTEXT_LABEL)
    app = _HandleHarness(handle)

    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        button = app.query_one("#console-rail-open", Button)

        assert str(button.label) == "Context--->"
        assert button.tooltip == "Open Context rail"
        assert handle.region.width == 13
        assert handle.content_region.width == 11
        assert button.region.x >= handle.content_region.x
        assert button.region.right <= handle.content_region.right
```

Keep the existing generic `DestinationRailHandle(side="left")` assertions unchanged; they prove this presentation rule does not leak into the shared component.

- [ ] **Step 3: Make the compact Context interaction test target the final arrow cell**

Import `Button` in `test_console_inspector_compact_access.py`. In `test_left_handle_opens_left_rail_at_90_cols`, replace the selector-centered open click with a selector-relative click one cell inside the existing button's right edge:

```python
open_button = console.query_one("#console-context-rail-open", Button)
assert str(open_button.label) == "Context--->"
assert open_button.tooltip == "Open Context rail"
arrow_end = (open_button.region.width - 1, open_button.region.height // 2)
assert await pilot.click(open_button, offset=arrow_end)
```

Keep the existing rail visibility, persistence marker, transcript width, and collapse-back assertions. Passing the widget to `pilot.click` is load-bearing: it proves the rightmost visible arrow cell belongs to this one Button instead of merely sending a coordinate to the screen.

- [ ] **Step 4: Update existing real-Console label consumers**

Change only canonical horizontal Context expectations:

```python
# Tests/UI/test_console_shell_regions.py
(False, 13, 11, "Context--->", "Inspect->"),

# Tests/UI/test_settings_console_rail_labels.py
assert left._display_label() == "Context--->"

# Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py
context_button = console.query_one("#console-context-rail-open", Button)
assert context_button.label == "Context--->"
assert context_button.tooltip == "Open Context rail"
```

Leave stacked `Context`, every `Inspect->` assertion, widths, settings state, and open/collapse assertions unchanged.

- [ ] **Step 5: Add the three-viewport compositor sweep**

Add a new `test_task_16001_console_collapsed_context_button_visual_sweep` in `test_workbench_visual_snapshots.py`, parameterized over `(130, 30)`, `(140, 42)`, and `(160, 45)`. Start from the real `TldwCli`, configure Console readiness through the existing helpers, open Console, and if the Context rail is visible click `#console-context-rail-collapse`; re-query after the resulting recompose. Wait until the collapsed handle has nonzero geometry, then require:

```python
screen = app.screen
workspace = screen.query_one("#console-workspace-grid")
context_handle = screen.query_one("#console-context-rail-handle")
context_button = screen.query_one("#console-context-rail-open", Button)
inspector_button = screen.query_one("#console-inspector-rail-open", Button)
transcript = screen.query_one("#console-transcript-region")

assert context_handle.display is True
assert _painted_region_rows(screen, context_button.region) == ["Context--->"]
assert context_button.label == "Context--->"
assert context_button.tooltip == "Open Context rail"
assert workspace.content_region.contains_region(context_handle.region)
assert context_handle.region.width == 13
assert context_handle.content_region.width == 11
assert context_button.region.x >= context_handle.content_region.x
assert context_button.region.right <= context_handle.content_region.right
assert inspector_button.label == "Inspect->"
assert transcript.region.width > 0
```

Also export the screenshot and require the rendered `<text>` evidence to include `Context--->`, using the file's existing SVG health helpers. Do not reuse or rename TASK-15783; its six-state Inspector/badge provenance remains intact.

- [ ] **Step 6: Run the focused RED gates**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_rail_handle.py::test_horizontal_canonical_labels_use_console_button_copy \
  Tests/UI/test_console_rail_handle.py::test_horizontal_noncanonical_left_label_is_unchanged \
  Tests/UI/test_destination_rail.py::test_console_handle_uses_the_full_context_arrow_button_on_the_left \
  Tests/UI/test_console_inspector_compact_access.py::test_left_handle_opens_left_rail_at_90_cols \
  Tests/UI/test_console_shell_regions.py::test_fresh_console_composes_saved_rail_label_style \
  Tests/UI/test_settings_console_rail_labels.py::test_console_rail_label_failed_save_keeps_draft_and_active_style \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions \
  Tests/UI/test_workbench_visual_snapshots.py::test_task_16001_console_collapsed_context_button_visual_sweep
```

Expected: canonical horizontal Context cases fail because production still returns `Context ▸`; the noncanonical and stacked controls pass. The new visual cases should reach their exact-copy/compositor assertion rather than fail on setup or unrelated geometry.

- [ ] **Step 7: Commit the RED regressions**

```bash
git add -- \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_workbench_visual_snapshots.py
git commit -m "test(console): require full Context arrow button"
```

### Task 2: Implement the fixed eleven-cell Context label

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_rail_handle.py:66-101`

- [ ] **Step 1: Make the minimal presentation changes**

Add one nonvertical left Button branch in `compose()` after the vertical branches and before the existing right-side branch:

```python
elif self.side == "left" and isinstance(child, Button):
    child.styles.line_pad = 0
```

Then translate only the canonical horizontal Context label in `_display_label()` while preserving branch order:

```python
def _display_label(self) -> str:
    """Return compact visible text while preserving full tooltips."""
    if self.vertical:
        return self._stack_vertical_label(self.label)
    if self.side == "left":
        return (
            "Context--->"
            if self.label == CONSOLE_RAIL_CONTEXT_LABEL
            else self.label
        )
    return "Inspect->" if self.label == CONSOLE_RAIL_INSPECTOR_LABEL else self.label
```

`Context--->` is exactly eleven terminal cells. Textual 8's default `line_pad=1` reserves one cell on each side, so the fixed label uses the existing eleven-cell content width only after the inline reset. Do not add constants, dynamic dash calculations, child widgets, CSS rules, width changes, IDs, or classes. The vertical-first branch protects stacked Context, and the equality guard protects noncanonical left labels.

- [ ] **Step 2: Run the focused GREEN gates from Task 1**

Run the exact Task 1 Step 6 command again.

Expected: all selected nodes and parameters pass.

- [ ] **Step 3: Mutation-check the selector-relative arrow-end click**

Temporarily change the compact interaction test's x-coordinate to the first out-of-bounds cell:

```python
arrow_end = (open_button.region.width, open_button.region.height // 2)
```

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_inspector_compact_access.py::test_left_handle_opens_left_rail_at_90_cols
```

Expected: FAIL because the click assertion returns false or the Context rail does not open. Restore `open_button.region.width - 1`, rerun the node, and require PASS.

- [ ] **Step 4: Run the directly related rail regression set**

Run only directly modified/related tests, per user instruction:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_shell_regions.py::test_fresh_console_composes_saved_rail_label_style \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_css_build_integrity.py \
  Tests/UI/test_console_internals_decomposition.py::test_console_workbench_panes_have_visible_terminal_frames
```

Then run both directly related compositor sweeps explicitly:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_workbench_visual_snapshots.py \
  -k 'task_15783 or task_16001'
```

Expected: all selected tests pass. Do not run the full repository suite.

- [ ] **Step 5: Run focused static and integrity checks**

Run:

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/Widgets/Console/console_rail_handle.py \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_workbench_visual_snapshots.py
../../.venv/bin/ruff format --check \
  tldw_chatbook/Widgets/Console/console_rail_handle.py \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_workbench_visual_snapshots.py
git diff --check
```

Run the duplicate backlog task-ID guard from `.github/workflows/backlog-guard.yml`:

```bash
bash -euo pipefail -c '
fail=0
file_dupes=$(ls backlog/tasks | sed -nE "s/^(task-[0-9]+(\\.[0-9]+)*) - .*\\.md$/\\1/p" | sort | uniq -d)
if [ -n "$file_dupes" ]; then
  fail=1
  printf "Duplicate filename task IDs:\n%s\n" "$file_dupes"
fi
fm_dupes=$(awk "FNR==1{seen=0} /^id:/ && !seen {seen=1; print tolower(\$2)}" backlog/tasks/*.md | sort | uniq -d)
if [ -n "$fm_dupes" ]; then
  fail=1
  printf "Duplicate frontmatter task IDs:\n%s\n" "$fm_dupes"
fi
exit "$fail"
'
```

Expected: lint/format/diff checks pass and the duplicate guard exits 0 without output.

- [ ] **Step 6: Commit the implementation**

```bash
git add -- tldw_chatbook/Widgets/Console/console_rail_handle.py
git commit -m "fix(console): make Context arrow fully clickable"
```

### Task 3: Close out TASK-16001 with fresh evidence

**Files:**
- Modify: `backlog/tasks/task-16001 - Make-Context-arrow-a-full-button-label.md`
- Add: `Docs/superpowers/plans/2026-08-13-task-16001-context-arrow-button.md`

- [ ] **Step 1: Self-review the complete task diff**

Run:

```bash
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_chatbook/Widgets/Console/console_rail_handle.py \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_workbench_visual_snapshots.py
```

Confirm the production diff contains only the approved canonical Context literal and inline padding reset on the existing horizontal left Button. Confirm the arrow-end test is restored in bounds and that shared handle code, CSS, widths, Inspector copy/badge, vertical copy, IDs, and classes did not change.

- [ ] **Step 2: Complete the task record without the five-digit Backlog CLI edit path**

After every scoped gate passes, edit `backlog/tasks/task-16001 - Make-Context-arrow-a-full-button-label.md` directly with `apply_patch` because this repository's Backlog CLI has a known five-digit task-edit parsing failure. Check AC #1-5, add an `## Implementation Notes` section containing the exact scoped evidence and deviations (if any), and set frontmatter `status: Done`. Do not mark Done while any directly related gate is red.

Required notes summary:

```markdown
## Implementation Notes

Implemented the eleven-cell `Context--->` display through the existing `ConsoleRailHandle` presentation seam and cleared inline line padding only on the existing horizontal left Button. The shared rail, CSS, geometry, vertical mode, noncanonical labels, Inspector, and responsive behavior remain unchanged. Focused component, mounted, arrow-end interaction, settings, compact-access, frame/CSS integrity, real-Console compositor, Ruff, duplicate-ID, and diff checks passed; per user instruction, no full repository suite was run. ADR required: no; ADR path: N/A.
```

- [ ] **Step 3: Commit closeout documentation**

```bash
git add -- \
  'backlog/tasks/task-16001 - Make-Context-arrow-a-full-button-label.md' \
  Docs/superpowers/plans/2026-08-13-task-16001-context-arrow-button.md
git commit -m "docs(console): close TASK-16001"
```

- [ ] **Step 4: Final cleanliness check**

Run:

```bash
git status --short
git diff --check origin/dev...HEAD
```

Expected: clean worktree and no whitespace errors.
