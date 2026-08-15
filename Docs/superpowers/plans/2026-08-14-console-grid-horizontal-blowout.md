# Console Grid Horizontal Blowout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the Console workspace contained at 120 columns by applying a deterministic Inspector-first rail policy, while preserving rail preferences and keyboard focus across responsive replacements.

**Architecture:** Keep width policy pure in `Chat/console_rail_state.py`: one finalizer resolves two-open rail state after automatic Inspector opens, one pure helper describes the Context-reveal preference update, and the width-band function exposes every geometry boundary. Existing `ChatScreen` compose/current/resize and action handlers consume those helpers; no new `ChatScreen` method, CSS rule, persistence schema, dependency, or Textual patch is introduced.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Ruff, MyPy, Backlog.md CLI

---

## Scope and file responsibilities

- `tldw_chatbook/Chat/console_rail_state.py`: owns all pure responsive rail priority, reveal-update, and width-band decisions.
- `tldw_chatbook/UI/Screens/chat_screen.py`: threads one resolved width through existing auto-open/finalization paths, applies the pure reveal decision from the two existing entry points, and hands focus to a visible handle during responsive replacement.
- `Tests/Chat/test_console_rail_state.py`: pins pure 99/100/149/150 priority and 117/118/128/129/149/150 band boundaries.
- `Tests/UI/test_console_inspector_compact_access.py`: pins mounted 120-column auto-open, Context reveal through both exact entry points, persistence, and real focus handoff.
- `Tests/UI/test_console_resize_reflow.py`: pins resize-event width authority and cold/live convergence at the new boundaries.
- `Tests/UI/test_console_shell_regions.py`: keeps the production hierarchy and stylesheet as the containment oracle for the four existing 120-column regressions.
- `Tests/UI/test_console_rail_width_budget.py`: refreshes only the stale 12-cell label oracle to the shipped 13-cell label-plus-gutter contract.
- `backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md`: records Inspector priority and preference/focus consequences.
- `backlog/tasks/task-16220 - Console-grid-blows-out-horizontally-at-120-columns-with-the-Inspector-rail-open.md`: tracks this plan, acceptance evidence, and closeout notes.

### Task 1: Record the refined responsive contract

**Files:**
- Modify: `backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md`
- Modify: `backlog/tasks/task-16220 - Console-grid-blows-out-horizontally-at-120-columns-with-the-Inspector-rail-open.md`

- [x] **Step 1: Amend ADR-043 before production changes**

Add TASK-16220 as a related task and record:

```text
At 100-149 columns, when both effective rail states are open, Inspector wins:
Context renders collapsed, Inspector owns compact-override authority, and the
stored Context preference is unchanged. A deliberate Context reveal closes
Inspector through the normal preference update. Responsive replacement moves
focus from a hidden rail to its visible reveal handle.
```

Replace the obsolete consequence that automatic opens always retain false compact-override flags. Preserve the existing below-100 explicit-toggle and 150-plus behavior.

- [x] **Step 2: Add this reviewed plan to the Backlog task**

Run:

```bash
backlog task edit 16220 --plan "1. Refine ADR-043 for Inspector-first compact priority and responsive focus handoff.\n2. Add pure rail-priority, reveal-update, and width-band contracts with RED-to-GREEN tests.\n3. Thread one width authority through existing Console compose/current/resize paths and preserve keyboard focus.\n4. Prove production-hierarchy containment at 120 columns and refresh the stale label-width oracle.\n5. Run focused verification, self-review, update docs/task evidence, and close through Backlog CLI."
```

Expected: `backlog task 16220 --plain` shows status `In Progress`, the four unchanged acceptance criteria, this implementation plan, and ADR-043.

- [x] **Step 3: Verify documentation diff hygiene**

Run:

```bash
git diff --check -- Docs/superpowers/specs/2026-08-14-console-grid-horizontal-blowout-design.md Docs/superpowers/plans/2026-08-14-console-grid-horizontal-blowout.md backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md "backlog/tasks/task-16220 - Console-grid-blows-out-horizontally-at-120-columns-with-the-Inspector-rail-open.md"
```

Expected: exit 0 with no output.

- [x] **Step 4: Commit the reviewed contract and plan**

```bash
git add Docs/superpowers/specs/2026-08-14-console-grid-horizontal-blowout-design.md Docs/superpowers/plans/2026-08-14-console-grid-horizontal-blowout.md backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md "backlog/tasks/task-16220 - Console-grid-blows-out-horizontally-at-120-columns-with-the-Inspector-rail-open.md"
git commit -m "docs(console): refine compact rail priority"
```

### Task 2: Add pure priority and boundary contracts

**Files:**
- Modify: `Tests/Chat/test_console_rail_state.py`
- Modify: `tldw_chatbook/Chat/console_rail_state.py`

- [x] **Step 1: Write failing pure-state tests**

Add a parameterized priority test with the exact boundary cases:

```python
@pytest.mark.parametrize(
    ("width", "expected_left", "expected_right", "expected_override"),
    [
        (99, True, True, True),
        (100, False, True, True),
        (149, False, True, True),
        (150, True, True, False),
    ],
)
def test_console_rail_priority_resolves_two_open_rails(...):
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={
            "left_open": True,
            "left_open_explicit": True,
            "right_open": True,
        },
        available_columns=width,
    )
    resolved = resolve_console_rail_priority(state, width)
    assert resolved.left_open is expected_left
    assert resolved.right_open is expected_right
    assert resolved.preferred_left_open is True
    assert resolved.right_compact_override is expected_override
    assert resolved.compact_override is expected_override
```

Pin `console_context_reveal_preferences()` to return `{"left_open": True, "right_open": False}` only when Inspector is effectively open in the 100-149 band, and otherwise `{"left_open": True}`. Extend `test_console_rail_width_band_buckets` across `83/84/99/100/117/118/128/129/149/150` so resize recomputation cannot skip an auto-open or restoration edge.

- [x] **Step 2: Run the tests to verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_rail_state.py -k "priority or reveal_preferences or width_band"
```

Expected: FAIL because the two pure helpers do not exist and the current width-band function merges all widths at or above 100.

- [x] **Step 3: Implement the smallest pure helpers**

In `console_rail_state.py`, import `replace` from `dataclasses`, reuse the existing 100/150 constants, and add no new state field:

```python
def _inspector_priority_width(available_columns: int | None) -> bool:
    return (
        available_columns is not None
        and CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS
        <= available_columns
        < CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS
    )


def resolve_console_rail_priority(
    rail_state: ConsoleRailState,
    available_columns: int | None,
) -> ConsoleRailState:
    if not (
        _inspector_priority_width(available_columns)
        and rail_state.left_open
        and rail_state.right_open
    ):
        return rail_state
    return replace(
        rail_state,
        left_open=False,
        right_compact_override=True,
        compact_override=True,
    )


def console_context_reveal_preferences(
    rail_state: ConsoleRailState,
    available_columns: int | None,
) -> dict[str, bool]:
    changes = {"left_open": True}
    if _inspector_priority_width(available_columns) and rail_state.right_open:
        changes["right_open"] = False
    return changes
```

Split the width buckets into stable names for `<84`, `84-99`, `100-117`, `118-128`, `129-149`, and `>=150`/unknown. Do not introduce a new enum or configuration layer; the strings are only resize deduplication keys.

- [x] **Step 4: Run pure-state GREEN and adjacent contracts**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_rail_state.py
```

Expected: PASS, including the existing below-100 explicit-toggle and 150-plus contracts.

- [x] **Step 5: Commit pure behavior**

```bash
git add tldw_chatbook/Chat/console_rail_state.py Tests/Chat/test_console_rail_state.py
git commit -m "fix(console): resolve compact rail priority"
```

### Task 3: Integrate priority, rail switching, and responsive focus

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_inspector_compact_access.py`
- Modify: `Tests/UI/test_console_resize_reflow.py`

- [x] **Step 1: Write failing mounted behavior tests**

Using `ConsoleHarness` and real `pilot.resize_terminal`, add tests that prove:

1. At 120 columns the standard-width auto-open produces `left_open=False`, `right_open=True`, `right_compact_override=True`, `compact_override=True`, and does not change stored `left_open`.
2. Clicking `#console-context-rail-open` at 120 closes/persists Inspector false and shows Context.
3. Activating the visible Workbench `#console-control-attach-context` action at 120 performs the same switch; do not touch `#console-attach-context` or `#console-staged-context-attach` file-picker behavior.
4. Resize 117-to-118 with focus inside Context moves focus to `#console-context-rail-open` after Context is hidden.
5. Resize 128-to-129 with focus inside Inspector moves focus to `#console-inspector-rail-open` after Inspector is hidden.
6. A resize event width of 120 wins even when `_console_rail_available_columns()` is monkeypatched to a stale non-auto-open width; the effective state and focus use the event width.

Keep the existing `test_clicking_the_collapse_button_clears_focus` manual-collapse characterization unchanged.

- [x] **Step 2: Run mounted tests to verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_inspector_compact_access.py Tests/UI/test_console_resize_reflow.py -k "priority or context or attach or resize or focus"
```

Expected: FAIL because auto-open does not receive compact authority, Context reveal requests two open rails, resize misses 118/129 edges, and hidden-rail focus clears to `None`.

- [x] **Step 3: Thread one resolved width through existing paths**

In existing `ChatScreen` methods only:

- resolve `available_columns` once before building state;
- pass it into `_should_open_standard_width_inspector` instead of rereading `self.size`;
- call `resolve_console_rail_priority()` after standard, pending-launch, and fleet modifiers in both `_current_console_rail_state` and `compose_content`;
- have both existing Context-reveal handlers call `_set_console_rail_preference(**console_context_reveal_preferences(...))`;
- leave the two file-picker button handlers unchanged.

Do not add a `ChatScreen` method. Do not move layout policy into CSS.

- [x] **Step 4: Add responsive focus handoff inside the existing resize handler**

Before applying the new state, capture whether current focus is within the left or right rail using the existing `_is_descendant_or_self`. After `_sync_console_rail_visibility_if_changed`, if that focused rail became hidden and its reveal button is displayed, focus `#console-context-rail-open` or `#console-inspector-rail-open` immediately. Do nothing when the rail remains open, the handle is hidden, or the transition is a manual collapse.

- [x] **Step 5: Run mounted GREEN and manual-collapse regression**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_inspector_compact_access.py Tests/UI/test_console_resize_reflow.py Tests/UI/test_console_right_rail.py -k "priority or context or attach or resize or focus or collapse"
```

Expected: PASS. In particular, responsive replacements land on their reveal handles and explicit Inspector collapse still lands on `None`.

- [x] **Step 6: Record the screen ratchet baseline without weakening it**

Run before and after the screen edit:

```bash
python3 - <<'PY'
import ast
from pathlib import Path

path = Path("tldw_chatbook/UI/Screens/chat_screen.py")
source = path.read_text(encoding="utf-8")
tree = ast.parse(source)
screen = next(
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "ChatScreen"
)
methods = sum(
    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    for node in screen.body
)
print(len(source.splitlines()), methods)
PY
```

Expected: method count is unchanged from the pre-task value. Do not raise or edit `Tests/Architecture/test_screen_size_ratchet.py`; document its current development-baseline line/method ceiling failure separately if it remains red.

- [x] **Step 7: Commit the mounted integration**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_inspector_compact_access.py Tests/UI/test_console_resize_reflow.py
git commit -m "fix(console): preserve compact rail focus"
```

### Task 4: Prove production geometry and refresh the stale oracle

**Files:**
- Modify: `Tests/UI/test_console_shell_regions.py`
- Modify: `Tests/UI/test_console_rail_width_budget.py`

- [x] **Step 1: Strengthen the 120-column production-hierarchy oracle**

Update the module narrative and the existing `_REGIONS` table to describe the approved policy rather than the broken baseline. At 120x30, change only these expected states:

```python
("#console-left-rail", "hittable", "hittable", "hidden"),
("#console-left-rail-body", "hittable", "hittable", "hidden"),
("#console-context-rail-handle", "hidden", "hidden", "hittable"),
```

Keep the Inspector rail open, its handle hidden, and the run-inspector row's existing height-driven `"clipped"` characterization. For the same 120x30 `size2` row, assert every displayed direct child of `#console-workspace-grid` is contained by the grid content region and by the screen viewport:

```python
grid = pilot.app.screen.query_one("#console-workspace-grid")
for child in grid.children:
    if not child.display:
        continue
    assert grid.content_region.contains_region(child.region)
    assert child.region.right <= pilot.app.screen.region.right
```

Keep the real `ConsolidatedCSSApp`/production hierarchy; do not replace it with a simplified three-widget harness.

- [x] **Step 2: Run the known geometry regressions**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_shell_regions.py -k size2
```

Expected before integration: the four known 120-column rows fail with 354/1534/472 geometry. Expected after integration: all `size2` cases pass and the new containment loop is green.

- [x] **Step 3: Refresh only the stale label-width oracle**

In `test_session_rows_fit_inside_the_rail`, change the comment and exact assertion from 12 to the production `ConsoleWorkspaceStatusPair` contract of 13 cells (12-cell label plus one-cell gutter):

```python
assert label.region.width == 13
```

Do not modify the unrelated recovery-copy test in the same module.

- [x] **Step 4: Run the two required width-budget rows**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_rail_width_budget.py::test_session_rows_fit_inside_the_rail
```

Expected: 2 passed.

- [x] **Step 5: Mutation-check the load-bearing branch**

Temporarily replace `resolve_console_rail_priority()` with `return rail_state`, run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_shell_regions.py -k size2
```

Expected: the 120-column geometry regression returns. Restore the implementation, rerun the same command, and require GREEN. The mounted 120-column state assertion from Task 3 separately pins compact-override authority; do not add a synthetic mutation for a production state that cannot occur.

- [x] **Step 6: Commit geometry evidence**

```bash
git add Tests/UI/test_console_shell_regions.py Tests/UI/test_console_rail_width_budget.py
git commit -m "test(console): pin compact grid containment"
```

### Task 5: Focused verification, review, documentation, and closeout

**Files:**
- Modify: `backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md` if implementation details require clarification
- Modify: `backlog/tasks/task-16220 - Console-grid-blows-out-horizontally-at-120-columns-with-the-Inspector-rail-open.md`

- [x] **Step 1: Run the exact related test matrix only**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_rail_state.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_resize_reflow.py \
  Tests/UI/test_console_right_rail.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_console_rail_width_budget.py::test_session_rows_fit_inside_the_rail
```

Expected: PASS. Do not run the full repository or broad test directories.

- [x] **Step 2: Run touched-file static checks**

Run:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Chat/console_rail_state.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_console_rail_state.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_resize_reflow.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_console_rail_width_budget.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Chat/console_rail_state.py \
  Tests/Chat/test_console_rail_state.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_resize_reflow.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_console_rail_width_budget.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Chat/console_rail_state.py \
  tldw_chatbook/UI/Screens/chat_screen.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Chat/console_rail_state.py \
  tldw_chatbook/UI/Screens/chat_screen.py
git diff --check
```

Expected: changed-line checks pass. If a whole legacy file reports baseline debt, compare the exact diagnostic against commit `31d0dab15` and document rather than reformatting unrelated code.

- [x] **Step 3: Perform one bounded visual verification pass**

Run the Console with the production stylesheet at 120x30 and capture the compositor or exported screenshot after Inspector auto-open. Verify Context handle, Transcript, and Inspector are all painted within the viewport; then resize 117→118 and 128→129 and verify the focused reveal handle. One confirmation pass after any fixes; no open-ended polishing loop.

- [x] **Step 4: Self-review and independent code review**

Review the cumulative diff for state/persistence separation, stale-width reads, rapid-resize focus races, unrelated file-picker changes, screen-method growth, and hidden layout overflow. Request a correctness/spec review before closeout and address only verified findings.

- [x] **Step 5: Complete Backlog evidence**

Check all four acceptance criteria, add concise Implementation Notes covering the pure finalizer, shared Context reveal decision, focus handoff, production-hierarchy geometry evidence, ADR-043 refinement, exact test/static results, baseline-only deviations, and modified files. Add a lessons entry only if implementation uncovers a genuinely new recurring trap with incident evidence.

Run:

```bash
backlog task edit 16220 -s Done --notes "Implemented Inspector-first compact rail resolution, responsive focus handoff, exact-width resize handling, and production-hierarchy containment evidence; refined ADR-043 and verified only the focused Console rail/layout matrix."
backlog task 16220 --plain
```

Expected: status `Done`, every AC checked, Implementation Plan retained, Implementation Notes present, and ADR-043 linked.

- [x] **Step 6: Commit closeout**

```bash
git add backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md "backlog/tasks/task-16220 - Console-grid-blows-out-horizontally-at-120-columns-with-the-Inspector-rail-open.md"
git commit -m "docs(console): complete compact grid fix"
```
