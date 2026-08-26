# Inspector Rail Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the collapsed Console Inspector handle a full-height, filled, vertically centered counterpart to the collapsed Context handle without changing other destinations or rail behavior.

**Architecture:** Keep `DestinationRailHandle` unchanged and opt into the new geometry only in `ConsoleRailHandle` when `side == "right"`. Use a Console-specific class for the full-height panel styling, override the base widget's fixed inline button geometry in Python, and switch the Console call site from a quiet frame to its standard solid frame.

**Tech Stack:** Python 3.11+, Textual 8.x widgets/layout, TCSS, pytest/Textual pilot, repository CSS bundle generator.

**Design spec:** `Docs/superpowers/specs/2026-08-12-task-15783-inspector-rail-parity-design.md`

**ADR required:** no

**ADR path:** `backlog/decisions/017-console-left-rail-usability.md`

**Related ADR:** `backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md`

**Reason:** ADR-017 already governs the Console rail's text-only visual language, and ADR-043 governs the compact-width Inspector access preserved by AC #4; this is a reversible presentation refinement within those existing boundaries.

**Approved follow-up:** After reviewing the finished 160×45 render, the user
approved shortening only the horizontal collapsed right-handle label from
`Inspector` to `Inspect`. Tasks 1–3 below describe the completed parity work;
Task 4 is the only active follow-up scope.

---

## File map

- Modify `Tests/UI/test_destination_rail.py`: add production-stylesheet mounted geometry regressions for unbadged and badged Console Inspector handles and the unchanged shared right-handle default.
- Modify `Tests/UI/test_console_internals_decomposition.py`: change the real Console frame contract from quiet/no-border to solid/all-edge for the collapsed Inspector.
- Modify `Tests/UI/test_css_build_integrity.py`: pin the new selector and declarations in both component source and generated bundle.
- Modify `Tests/UI/test_workbench_visual_snapshots.py`: exercise the real Console at the three representative sizes with deterministic unbadged/badged Inspector states and exported SVG health checks.
- Modify `tldw_chatbook/Widgets/Console/console_rail_handle.py`: add the Console-only Inspector class and override fixed inline button geometry.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: frame the collapsed Inspector with the standard solid Console frame.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss`: add the Console Inspector full-height/fill rule after the shared compact right-handle rule.
- Regenerate `tldw_chatbook/css/tldw_cli_modular.tcss`: mechanical production bundle generated from component TCSS.
- Modify `backlog/tasks/task-15783 - Match-collapsed-Inspector-rail-to-Context-rail.md`: complete acceptance criteria and record verification/implementation notes only after fresh evidence.

### Task 1: Pin the broken Inspector geometry

**Files:**
- Modify: `Tests/UI/test_destination_rail.py`
- Modify: `Tests/UI/test_console_internals_decomposition.py:3155-3161`
- Modify: `Tests/UI/test_workbench_visual_snapshots.py`

- [ ] **Step 1: Add a production-stylesheet harness**

Import `Path` and `tldw_chatbook`, then add a focused harness that loads the real bundle:

```python
_BUNDLED_STYLESHEET = (
    Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
)


class _StyledHandleHarness(App[None]):
    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, *handles: DestinationRailHandle) -> None:
        super().__init__()
        self._handles = handles

    def compose(self) -> ComposeResult:
        yield from self._handles
```

Frame each Console handle before passing it to the harness with
`frame_console_region(handle)`, exactly matching the inline solid frame used in
production. Leave the shared `DestinationRailHandle(side="right")` unframed so
its compact/quiet default remains the control case.

- [ ] **Step 2: Add failing unbadged and badged Console Inspector tests**

Mount a separately framed left `ConsoleRailHandle(side="left")` and separate
framed right `ConsoleRailHandle(side="right")` instances at a 20-row viewport,
one right handle with no badge and one with `badge="3 approvals"`. Assert exact
left/right parity and preserved widths:

```python
assert left_handle.region.height == right_handle.region.height == 20
assert left_handle.region.width == 13
assert right_handle.region.width == 11
assert right_handle.content_region.width == 9
assert handle.region.height == 20
assert handle.styles.background.a > 0
assert all(edge[0] == "solid" for edge in handle.styles.border)
assert button.styles.content_align == ("center", "middle")
assert button.region.x >= handle.content_region.x
assert button.region.right <= handle.content_region.right
```

For the badged state also assert
`str(badge.renderable) == "3 appr"`,
`button.region.bottom <= badge.region.y`,
`button.region.height == handle.content_region.height - badge.region.height`,
and badge bottom/right containment. For the unbadged state assert the button
occupies the full content height. Assert the left/right backgrounds and four
border edges match.

- [ ] **Step 3: Pin the shared non-Console right-handle default**

Mount `DestinationRailHandle(side="right")` under the production stylesheet and assert it remains compact (`region.height <= 6`), transparent, and borderless. This proves Lab/Personas behavior is not changed by the Console fix.

- [ ] **Step 4: Update the real Console frame expectation**

In `test_console_workbench_panes_have_visible_terminal_frames`, replace the quiet-frame assertions for `#console-inspector-rail-handle` with `console-frame-solid`, all four `solid` border edges, outer width 11, and content width 9.

- [ ] **Step 5: Add a source/bundle parity regression**

In `Tests/UI/test_css_build_integrity.py`, add a test that extracts
`.console-inspector-rail-handle` with `_rule_body()` from both
`_AGENTIC_SOURCE` and `_BUNDLED_STYLESHEET` and requires:

```python
for css in (source, bundle):
    rule = _rule_body(css, ".console-inspector-rail-handle")
    assert "height: 100%;" in rule
    assert "min-height: 20;" in rule
    assert "max-height: 100%;" in rule
    assert "background: $ds-surface-panel;" in rule
```

- [ ] **Step 6: Add the executable real-Console visual/geometry sweep**

In `Tests/UI/test_workbench_visual_snapshots.py`, add a parameterized test over
`size in ((130, 30), (140, 42), (160, 45))` and
`approval_count in (0, 3)`. Before `run_test`, set
`app.console_pending_approval_count = approval_count`, complete onboarding,
and open Console through the existing `_open_console()` helper. Query
`#console-inspector-rail-handle`, `#console-inspector-rail-open`, and the
optional badge. Assert the handle's height equals `#console-workspace-grid`'s
height, button vertical center equals the available content area's vertical
center (excluding the badge row), background/frame matches the left handle,
and the transcript has positive width. For count 3, assert visible `3 appr`
and non-overlap; for count 0, assert no badge. Call:

```python
svg = app.export_screenshot(
    title=f"TASK-15783 Inspector parity {size[0]}x{size[1]} approvals={approval_count}",
    simplify=True,
)
_assert_svg_healthy(svg)
```

The approved design originally named 100x30. RED-test review found that the
current baseline places the collapsed Inspector at x=1362 there, outside the
workspace's x=2..98 range. That separate horizontal-overflow defect cannot be
made green by this task's vertical/fill change without expanding ADR-043 layout
scope. Use 130x30 instead: it is the narrowest probed baseline-contained size
outside the 118–128 auto-open band (handle x=117..128 inside workspace
x=2..128). Existing compact-access tests remain the authority at 80/90/140.

- [ ] **Step 7: Run every new regression and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_destination_rail.py -k 'inspector or shared_right'
.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_workbench_panes_have_visible_terminal_frames
.venv/bin/python -m pytest -q Tests/UI/test_css_build_integrity.py -k inspector_rail
.venv/bin/python -m pytest -q Tests/UI/test_workbench_visual_snapshots.py -k task_15783
```

Expected: failures show the Inspector is only 3–6 rows, transparent/borderless, the button is three rows, and the real Console handle still uses `console-frame-quiet`. The shared non-Console assertion should pass.

- [ ] **Step 8: Commit the failing regressions**

```bash
git add Tests/UI/test_destination_rail.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_css_build_integrity.py Tests/UI/test_workbench_visual_snapshots.py
git commit -m "test(console): pin full-height Inspector rail parity"
```

### Task 2: Implement the Console-only full-height handle

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_rail_handle.py:5-89`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:12547-12560`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:2673-2701`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Add the Console Inspector class and inline geometry override**

Import `ComposeResult` and `Button`. After the base initializer, add `console-inspector-rail-handle` only for `side == "right"`. Override `compose()` minimally:

```python
def compose(self) -> ComposeResult:
    for child in super().compose():
        if self.side == "right" and isinstance(child, Button):
            child.styles.width = "100%"
            child.styles.max_width = "100%"
            child.styles.height = "1fr"
            child.styles.min_height = 0
            child.styles.max_height = "100%"
        yield child
```

This overrides only the base widget's right-side fixed inline geometry; label, tooltip, badge composition, and every shared default remain intact.

- [ ] **Step 2: Add the Console-only TCSS rule**

Immediately after `.console-rail-handle-right`, add:

```tcss
.console-inspector-rail-handle {
    height: 100%;
    min-height: 20;
    max-height: 100%;
    background: $ds-surface-panel;
}
```

Do not change `.console-rail-handle-right`; it remains the compact shared default.

- [ ] **Step 3: Use the standard Console frame**

In `ChatScreen.compose_content`, change:

```python
yield self._frame_console_region(right_handle, variant="quiet")
```

to:

```python
yield self._frame_console_region(right_handle)
```

- [ ] **Step 4: Regenerate the production stylesheet**

Run:

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` contains the new selector and the normal generated header/timestamp update; do not hand-edit the bundle.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_destination_rail.py
.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_workbench_panes_have_visible_terminal_frames
.venv/bin/python -m pytest -q Tests/UI/test_console_right_rail.py Tests/UI/test_console_inspector_compact_access.py
.venv/bin/python -m pytest -q Tests/UI/test_css_build_integrity.py
.venv/bin/python -m pytest -q Tests/UI/test_workbench_visual_snapshots.py -k task_15783
```

Expected: all selected tests pass with no warnings attributable to TASK-15783.

- [ ] **Step 6: Run focused static checks**

Run:

```bash
.venv/bin/ruff check tldw_chatbook/Widgets/Console/console_rail_handle.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_destination_rail.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_css_build_integrity.py
git diff --check
node .agents/skills/impeccable/scripts/detect.mjs --json --scope layout tldw_chatbook/Widgets/Console/console_rail_handle.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss
```

Expected: zero Ruff errors, no whitespace errors, and no unexplained layout detector findings.

- [ ] **Step 7: Commit the implementation**

```bash
git add tldw_chatbook/Widgets/Console/console_rail_handle.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "fix(console): fill collapsed Inspector rail"
```

### Task 3: Verify the rendered Console and close out the task

**Files:**
- Modify: `backlog/tasks/task-15783 - Match-collapsed-Inspector-rail-to-Context-rail.md`

- [ ] **Step 1: Re-run the real-Console visual/geometry sweep**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_workbench_visual_snapshots.py -k task_15783
```

Expected: six real-Console mounted states pass (three widths × unbadged/badged),
each with a healthy exported SVG and exact geometry assertions.

- [ ] **Step 2: Run the final fresh focused verification set**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_destination_rail.py Tests/UI/test_console_right_rail.py Tests/UI/test_console_inspector_compact_access.py Tests/UI/test_css_build_integrity.py
.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_workbench_panes_have_visible_terminal_frames
.venv/bin/python -m pytest -q Tests/UI/test_workbench_visual_snapshots.py -k task_15783
.venv/bin/ruff check tldw_chatbook/Widgets/Console/console_rail_handle.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_destination_rail.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_css_build_integrity.py Tests/UI/test_workbench_visual_snapshots.py
.venv/bin/ruff format --check tldw_chatbook/Widgets/Console/console_rail_handle.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_destination_rail.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_css_build_integrity.py Tests/UI/test_workbench_visual_snapshots.py
git diff --check
```

Expected: all selected tests pass, Ruff exits zero, and the diff check is clean.

- [ ] **Step 3: Run repository-wide gates required for Done**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Expected for Done: all three exit zero. If an unrelated/environment baseline
prevents any gate from completing green, record the exact command and failure
and leave TASK-15783 In Progress; do not claim the repository Definition of
Done is satisfied.

- [ ] **Step 4: Self-review against every acceptance criterion**

Record the implementation base SHA immediately before the RED-test commit,
then inspect `git diff <implementation-base>..HEAD --` limited to TASK-15783's
listed files. Explicitly confirm full height, filled/bordered
surface, vertical centering, badge containment, unchanged 11-column width and
tooltips, unchanged shared right handles, bundle parity, and focused test
evidence. Do not include unrelated working-tree changes.

- [ ] **Step 5: Update Backlog task hygiene**

Check every acceptance criterion, add concise Implementation Notes with the
red/green commands and visual evidence, link ADR-017, ADR-043, and both
spec/plan files,
and set TASK-15783 to Done only if every repository Definition of Done item is
actually satisfied. Otherwise leave it In Progress and name the unmet gate.

When every gate is green, use the exact CLI form:

```bash
backlog task edit 15783 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --notes "<verified implementation summary>" -s Done --plain
backlog task 15783 --plain
```

If a full gate is not green, omit `-s Done`, keep the task In Progress, and put
the exact unmet gate in `--notes`. Before implementation begins, assign it with
`backlog task edit 15783 -a @codex -s "In Progress" --plain`.

- [ ] **Step 6: Commit closeout documentation**

```bash
git add 'backlog/tasks/task-15783 - Match-collapsed-Inspector-rail-to-Context-rail.md'
git commit -m "docs(console): record Inspector rail parity verification"
```

### Task 4: Keep the horizontal Inspect action on one line

**Files:**
- Modify: `Tests/UI/test_console_rail_handle.py`
- Modify: `Tests/UI/test_destination_rail.py`
- Modify: `Tests/UI/test_console_shell_regions.py`
- Modify: `Tests/UI/test_settings_console_rail_labels.py`
- Modify: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`
- Modify: `Tests/UI/test_workbench_visual_snapshots.py`
- Modify: `tldw_chatbook/Widgets/Console/console_rail_handle.py:94-101`
- Modify: `backlog/tasks/task-15783 - Match-collapsed-Inspector-rail-to-Context-rail.md`

- [ ] **Step 1: Change horizontal contract expectations to `Inspect`**

Update only tests that observe the horizontal collapsed Console right handle:

```python
assert inspector._display_label() == "Inspect"
assert str(right_button.label) == "Inspect"
assert right_button.tooltip == "Open Inspector rail"
```

In
`Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`, replace
the broad `"Inspector" in text` check with an exact widget contract so the
old prefix cannot pass the new expectation:

```python
inspector_button = console.query_one("#console-inspector-rail-open", Button)
assert str(inspector_button.label) == "Inspect"
assert inspector_button.tooltip == "Open Inspector rail"
```

Rename the two focused tests whose names currently say the horizontal default
is preserved or the Inspector label is renamed so they describe the new
horizontal abbreviation contract. In the shell/config parameterizations,
change only the non-stacked right-label expectation. Preserve every stacked
`I\nn\ns\np\ne\nc\nt\no\nr` expectation, settings label, canonical constant,
tooltip, and open Inspector heading.

- [ ] **Step 2: Replace the vacuous visual substring oracle**

In `Tests/UI/test_workbench_visual_snapshots.py`, add a small local helper that
slices a widget's exact rows from `screen._compositor.render_strips()`. Textual
regions use terminal-cell coordinates, so crop the `Strip` by cells rather
than applying Python code-point slicing:

```python
def _composited_rows(widget) -> list[str]:
    strips = widget.screen._compositor.render_strips()
    region = widget.region
    return [
        strips[y].crop(region.x, region.right).text
        for y in range(region.y, region.bottom)
        if 0 <= y < len(strips)
    ]
```

In the six-state TASK-15783 sweep, assert both the semantic label and the final
paint:

```python
assert str(inspector_button.label) == "Inspect"
assert inspector_button.tooltip == "Open Inspector rail"
painted_rows = [
    row.strip()
    for row in _composited_rows(inspector_button)
    if row.strip()
]
assert painted_rows == ["Inspect"]
```

Delete the whole-SVG `"Inspector" in rendered_text` assertion. It is not an
oracle for this behavior because the screenshot title contains `Inspector`,
and `Inspect` is also a prefix of `Inspector`.

- [ ] **Step 3: Run the changed contracts and verify RED**

First record the known formatter baseline before any code/test edit:

```bash
.venv/bin/ruff format --check tldw_chatbook/Widgets/Console/console_rail_handle.py Tests/UI/test_console_rail_handle.py Tests/UI/test_destination_rail.py Tests/UI/test_console_shell_regions.py Tests/UI/test_settings_console_rail_labels.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_workbench_visual_snapshots.py
```

Expected baseline: exit 1 naming exactly these five files and no others:
`test_console_rail_handle.py`, `test_console_shell_regions.py`,
`test_destination_rail.py`, `test_settings_console_rail_labels.py`, and
`console_rail_handle.py`. Do not reformat them as part of this copy-only task.

Then run only the directly affected tests:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_rail_handle.py Tests/UI/test_destination_rail.py Tests/UI/test_console_shell_regions.py Tests/UI/test_settings_console_rail_labels.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py
.venv/bin/python -m pytest -q Tests/UI/test_workbench_visual_snapshots.py -k task_15783
```

Expected: failures are limited to horizontal collapsed right-handle copy still
rendering `Inspector` or two painted rows (`Inspect`, `or`). Vertical-label,
tooltip, geometry, badge, and interaction assertions continue to pass.

- [ ] **Step 4: Commit the RED contracts**

```bash
git add Tests/UI/test_console_rail_handle.py Tests/UI/test_destination_rail.py Tests/UI/test_console_shell_regions.py Tests/UI/test_settings_console_rail_labels.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_workbench_visual_snapshots.py
git commit -m "test(console): require one-line Inspect rail label"
```

- [ ] **Step 5: Implement the one-literal Console-only abbreviation**

In `ConsoleRailHandle._display_label()`, retain the vertical branch and every
noncanonical fallback, changing only the canonical horizontal return value:

```python
if self.vertical:
    return self._stack_vertical_label(self.label)
if self.side != "right":
    return self.label
return "Inspect" if self.label == CONSOLE_RAIL_INSPECTOR_LABEL else self.label
```

Do not change `CONSOLE_RAIL_INSPECTOR_LABEL`, shared
`DestinationRailHandle`, CSS, geometry, ids/classes, state builders, tooltip,
badge vocabulary, or the open Inspector rail.

- [ ] **Step 6: Run the exact focused GREEN verification**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_rail_handle.py Tests/UI/test_destination_rail.py Tests/UI/test_console_shell_regions.py Tests/UI/test_settings_console_rail_labels.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_console_right_rail.py Tests/UI/test_console_inspector_compact_access.py Tests/UI/test_css_build_integrity.py Tests/UI/test_console_internals_decomposition.py::test_console_workbench_panes_have_visible_terminal_frames
.venv/bin/python -m pytest -q Tests/UI/test_workbench_visual_snapshots.py -k task_15783
.venv/bin/ruff check tldw_chatbook/Widgets/Console/console_rail_handle.py Tests/UI/test_console_rail_handle.py Tests/UI/test_destination_rail.py Tests/UI/test_console_shell_regions.py Tests/UI/test_settings_console_rail_labels.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_workbench_visual_snapshots.py
.venv/bin/ruff format --check tldw_chatbook/Widgets/Console/console_rail_handle.py Tests/UI/test_console_rail_handle.py Tests/UI/test_destination_rail.py Tests/UI/test_console_shell_regions.py Tests/UI/test_settings_console_rail_labels.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_workbench_visual_snapshots.py
git diff --check
```

Expected: all selected tests pass; the compositor reports one `Inspect` row at
all six live Console states; the exact tooltip and stacked label remain
unchanged. Ruff check and `git diff --check` exit zero. Ruff format retains the
exact five-file baseline failure set recorded in Step 3, with no new file or
finding attributable to the follow-up. Per the user's explicit instruction,
do not run the full repository suite for this follow-up.

- [ ] **Step 7: Record evidence, close AC #7, and commit**

Append the follow-up evidence manually to TASK-15783's existing Implementation
Notes; do not pass `--notes`, because Backlog replaces the entire notes block
instead of appending. Record the RED/GREEN counts, exact compositor evidence,
preserved invariants, formatter baseline comparison, and user-scoped
verification boundary. Then check AC #7 and return the task to Done only after
every focused gate above passes. Re-read and diff the task before committing to
prove the existing implementation history was preserved.

```bash
git add tldw_chatbook/Widgets/Console/console_rail_handle.py
git commit -m "fix(console): shorten collapsed rail label to Inspect"
backlog task edit 15783 --check-ac 7 -s Done --plain
backlog task 15783 --plain
git diff -- 'backlog/tasks/task-15783 - Match-collapsed-Inspector-rail-to-Context-rail.md'
git add 'backlog/tasks/task-15783 - Match-collapsed-Inspector-rail-to-Context-rail.md'
git commit -m "docs(console): record Inspect label verification"
git push
```
