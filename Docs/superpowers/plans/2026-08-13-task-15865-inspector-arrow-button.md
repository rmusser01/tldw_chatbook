# Full-Width Inspect Arrow Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render `Inspect->` as the complete nine-cell horizontal collapsed Inspector button while preserving every existing rail geometry, badge, tooltip, vertical-mode, and responsive contract.

**Architecture:** Keep the shared `DestinationRailHandle`, CSS files, and layout geometry unchanged. Use the existing `ConsoleRailHandle._display_label()` presentation seam for one fixed nine-cell literal and clear Textual's default `line_pad=1` inline in the existing horizontal right Button block so the label uses all nine content cells; strengthen existing component, mounted, interaction, and real-Console visual regressions around those seams.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich cell geometry, pytest/Textual Pilot.

**Design spec:** `Docs/superpowers/specs/2026-08-13-task-15865-inspector-arrow-button-design.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a reversible display-copy refinement inside an existing component seam; it changes no architecture, persistence, dependency, service, security, or ownership boundary.

---

## File map

- Modify `Tests/UI/test_console_rail_handle.py`: pin the canonical horizontal display copy and preserve vertical/left-side variants.
- Modify `Tests/UI/test_destination_rail.py`: pin the mounted production-stylesheet button label while retaining existing geometry, tooltip, and badge containment assertions.
- Modify `Tests/UI/test_console_right_rail.py`: click the right/arrow end of the existing button and prove the same control opens the Inspector.
- Modify `Tests/UI/test_console_shell_regions.py`: update the saved horizontal rail-style expectation while preserving stacked mode.
- Modify `Tests/UI/test_settings_console_rail_labels.py`: update the mounted horizontal Inspector expectation in the failed-save state.
- Modify `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`: update the core-loop mounted button label while preserving its tooltip contract.
- Modify `Tests/UI/test_workbench_visual_snapshots.py`: require one composited `Inspect->` row across the existing six real-Console viewport/badge states.
- Modify `tldw_chatbook/Widgets/Console/console_rail_handle.py`: change the canonical horizontal Inspector display literal and clear line padding on the existing horizontal right Button.
- Modify `backlog/tasks/task-15865 - Make-Inspect-arrow-a-full-button-label.md`: complete acceptance criteria and record scoped verification after fresh evidence.

### Task 1: Pin the full-button label and arrow-end hit target

**Files:**
- Modify: `Tests/UI/test_console_rail_handle.py:188-194`
- Modify: `Tests/UI/test_destination_rail.py:350-366`
- Modify: `Tests/UI/test_console_right_rail.py:85-119`
- Modify: `Tests/UI/test_console_shell_regions.py:113-145`
- Modify: `Tests/UI/test_settings_console_rail_labels.py:190-206`
- Modify: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py:220-236`
- Modify: `Tests/UI/test_workbench_visual_snapshots.py:202-302`

- [ ] **Step 1: Update the pure display contract to the approved literal**

Change the horizontal Inspector assertion in `test_horizontal_defaults_preserve_context_and_abbreviate_inspector_label` while leaving the Context assertion unchanged:

```python
assert context._display_label() == CONSOLE_RAIL_CONTEXT_LABEL
assert inspector._display_label() == "Inspect->"
```

The existing `test_vertical_inspector_label_stacks_without_direction_glyph` remains unchanged and continues to require stacked `Inspector`.

- [ ] **Step 2: Update the mounted button contract**

In `test_console_handle_abbreviates_the_inspector_label_on_the_right`, require:

```python
button = app.query_one("#console-rail-open", Button)
assert str(button.label) == "Inspect->"
assert button.tooltip == "Open Inspector rail"
```

Do not move the badge into the button. Existing badged geometry tests in the same file remain the authority for the separate row.

- [ ] **Step 3: Strengthen the real click path at the arrow end**

Import the real widget type:

```python
from textual.widgets import Button
```

In `test_clicking_open_then_collapse_toggles_visibility_and_persists`, replace the selector-centered open click with a selector-relative click one cell inside the button's right edge:

```python
open_button = pilot.app.screen.query_one("#console-inspector-rail-open", Button)
assert str(open_button.label) == "Inspect->"
arrow_end = (open_button.region.width - 1, open_button.region.height // 2)
assert await pilot.click(open_button, offset=arrow_end)
```

Keep every existing post-click open/hidden assertion and the collapse-back path. Supplying `open_button` is load-bearing: Textual's coordinate-only `pilot.click(offset=...)` can return success without proving which widget received the event, while the selector-relative call proves the rightmost arrow cells belong to this existing button.

- [ ] **Step 4: Update the six-state real-Console compositor assertion**

In `test_task_15783_console_collapsed_inspector_rail_visual_parity_sweep`, preserve the geometry/frame/badge/transcript assertions and change only the visible copy requirements:

```python
assert _painted_region_rows(screen, inspector_button.region) == ["Inspect->"]
assert inspector_button.label == "Inspect->"
```

Do not rename the historical TASK-15783 parity test; TASK-15865 strengthens its current UI contract without rewriting its provenance.

- [ ] **Step 5: Update every remaining focused label consumer**

Change only the horizontal canonical Inspector expectations in these existing tests:

```python
# Tests/UI/test_console_shell_regions.py
(False, 13, 11, "Context ▸", "Inspect->"),

# Tests/UI/test_settings_console_rail_labels.py
assert right._display_label() == "Inspect->"

# Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py
assert inspector_button.label == "Inspect->"
```

Leave the stacked `Inspector`, left `Context ▸`, width, tooltip, and settings state assertions unchanged.

- [ ] **Step 6: Run the focused RED gates**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_rail_handle.py::test_horizontal_defaults_preserve_context_and_abbreviate_inspector_label \
  Tests/UI/test_destination_rail.py::test_console_handle_abbreviates_the_inspector_label_on_the_right \
  Tests/UI/test_console_right_rail.py::test_clicking_open_then_collapse_toggles_visibility_and_persists \
  Tests/UI/test_console_shell_regions.py::test_fresh_console_composes_saved_rail_label_style \
  Tests/UI/test_settings_console_rail_labels.py::test_console_rail_label_failed_save_keeps_draft_and_active_style \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions \
  Tests/UI/test_workbench_visual_snapshots.py::test_task_15783_console_collapsed_inspector_rail_visual_parity_sweep
```

Expected: every horizontal canonical Inspector copy expectation fails because production still returns `Inspect`; the stacked-mode parameter and unchanged geometry, badge, tooltip, and interaction assertions remain valid up to the copy mismatch.

- [ ] **Step 7: Commit the RED regressions**

```bash
git add -- \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_workbench_visual_snapshots.py
git commit -m "test(console): require full Inspect arrow button"
```

### Task 2: Implement the fixed nine-cell label

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_rail_handle.py:94-100`

- [ ] **Step 1: Make the minimal presentation changes**

Change the canonical horizontal right-side return literal:

```python
return "Inspect->" if self.label == CONSOLE_RAIL_INSPECTOR_LABEL else self.label
```

In the existing nonvertical right Button block, clear Textual's default line padding:

```python
child.styles.line_pad = 0
```

`Inspect->` is exactly nine terminal cells. Textual 8's default `line_pad=1` reserves one cell on each side, and a runtime compositor probe showed the label wrapping until this inline reset; the repository uses this inline pattern because TCSS rejects `line-pad: 0`. Do not add constants, width calculations, child widgets, CSS-file rules, or layout changes. The existing branch order protects vertical and left-side presentation.

- [ ] **Step 2: Run the focused GREEN gates from Task 1**

Run the exact Task 1 Step 6 command again.

Expected: all selected cases pass, including all six parameterized visual states.

- [ ] **Step 3: Mutation-check the selector-relative arrow-end click**

Now that the label expectations are GREEN, temporarily change the selector-relative x-coordinate from `open_button.region.width - 1` to the first out-of-bounds cell:

```python
arrow_end = (open_button.region.width, open_button.region.height // 2)
```

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_right_rail.py::test_clicking_open_then_collapse_toggles_visibility_and_persists
```

Expected: FAIL because the click no longer opens the Inspector (or the click assertion itself returns false). Restore `open_button.region.width - 1`, rerun the node, and require PASS before continuing.

- [ ] **Step 4: Run the directly related rail regression set**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_shell_regions.py::test_fresh_console_composes_saved_rail_label_style \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_css_build_integrity.py \
  Tests/UI/test_console_internals_decomposition.py::test_console_workbench_panes_have_visible_terminal_frames
```

Then run the six-state sweep explicitly:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_workbench_visual_snapshots.py \
  -k task_15783
```

Expected: all selected tests pass. Per user instruction, do not run the full repository suite.

- [ ] **Step 5: Run focused static and integrity checks**

Run:

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/Widgets/Console/console_rail_handle.py \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_workbench_visual_snapshots.py
../../.venv/bin/ruff format --check \
  tldw_chatbook/Widgets/Console/console_rail_handle.py \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_workbench_visual_snapshots.py
git diff --check
```

Run the exact duplicate backlog task-ID guard from `.github/workflows/backlog-guard.yml`:

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

Expected: exit 0 with no duplicate output.

- [ ] **Step 6: Commit the implementation**

```bash
git add -- tldw_chatbook/Widgets/Console/console_rail_handle.py
git commit -m "fix(console): make Inspect arrow fully clickable"
```

### Task 3: Close out TASK-15865 with fresh evidence

**Files:**
- Modify: `backlog/tasks/task-15865 - Make-Inspect-arrow-a-full-button-label.md`

- [ ] **Step 1: Self-review the complete task diff**

Run:

```bash
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_chatbook/Widgets/Console/console_rail_handle.py \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_destination_rail.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_settings_console_rail_labels.py \
  Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py \
  Tests/UI/test_workbench_visual_snapshots.py
```

Confirm the production diff is the approved literal plus the inline line-pad reset on the existing horizontal right Button, the arrow-end click stays in bounds, and no shared handle, CSS file, width, badge composition, or vertical copy changed.

- [ ] **Step 2: Complete the task record**

After every scoped gate passes, run this command exactly once to check AC #1-5, add concise Implementation Notes, and move TASK-15865 to Done:

```bash
backlog task edit 15865 \
  --check-ac 1 \
  --check-ac 2 \
  --check-ac 3 \
  --check-ac 4 \
  --check-ac 5 \
  --notes "Implemented the nine-cell Inspect arrow through the existing ConsoleRailHandle display seam using the fixed Inspect-> literal plus an inline line-pad reset on the existing horizontal right Button; no CSS-file, outer-width, child-widget, ID/class, or layout-structure changes were added. Modified console_rail_handle.py plus test_console_rail_handle.py, test_destination_rail.py, test_console_right_rail.py, test_console_shell_regions.py, test_settings_console_rail_labels.py, test_product_maturity_gate1_core_loop_screen_adaptation.py, and test_workbench_visual_snapshots.py. Directly related rail, interaction, compact-access, CSS-integrity, visual, Ruff, duplicate-ID, and diff checks passed. Per user instruction, no full repository suite was run. ADR required: no." \
  --status Done \
  --plain
```

Expected: TASK-15865 renders as Done with AC #1-5 checked and an Implementation Notes section. Do not run the command if any directly related gate is red. Diff the task file afterwards because Backlog CLI notes replace the notes section.

- [ ] **Step 3: Commit closeout documentation**

```bash
git add -- 'backlog/tasks/task-15865 - Make-Inspect-arrow-a-full-button-label.md'
git commit -m "docs(console): close TASK-15865"
```

- [ ] **Step 4: Final cleanliness check**

Run:

```bash
git status --short
git diff --check origin/dev...HEAD
```

Expected: clean worktree and no whitespace errors.
