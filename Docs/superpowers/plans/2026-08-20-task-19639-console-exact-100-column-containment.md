# Console Exact-100-Column Containment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep every supported Console rail state contained at exactly 100 columns on cold start and live resize while preserving at least 55 outer main-column cells and 40 readable transcript cells.

**Architecture:** Extend the pure Console rail-state classifier so an open Context rail receives existing layout-minimum-waiver authority at exactly 100 columns, and give that exact width its own resize-deduplication band. Reuse the screen's current compose/live-sync consumers; add no new layout system, persistence path, or rail behavior. Lock the result with pure-state tests and the existing production-CSS Textual compositor harness.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Rich/Textual compositor geometry, Ruff.

**Required skills:** `@superpowers:test-driven-development`, `@textual-tui`, `@ponytail`, and `@superpowers:verification-before-completion`.

**Design:** `Docs/superpowers/specs/2026-08-20-task-19639-console-exact-100-column-containment-design.md`

**Backlog task:** `backlog/tasks/task-19639 - Keep-Console-workspace-geometry-inside-the-viewport-at-exactly-100-columns.md`

**ADR required:** no

**ADR path:** `backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md`

**Reason:** This is a defect correction inside ADR-043's existing responsive priority and main-minimum-waiver mechanism. Storage, preference ownership, rail visibility policy, and cross-module interfaces do not change.

---

## File map

- Modify `tldw_chatbook/Chat/console_rail_state.py`: classify exact 100 as minimum-waiver eligible and as a distinct semantic resize band; correct state-level comments.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: correct compose/live-sync comments and exact-width arithmetic only; existing consumers remain behaviorally unchanged.
- Modify `Tests/Chat/test_console_rail_state.py`: pure 99/100/101 state table and exact-width band contract.
- Modify `Tests/UI/test_console_shell_regions.py`: production-CSS four-state geometry matrix, readable-width assertions, adjacent live-resize transitions, focus/reading-state continuity, and zero-save oracle.
- Modify `backlog/tasks/task-19639 - Keep-Console-workspace-geometry-inside-the-viewport-at-exactly-100-columns.md`: check acceptance criteria and add implementation notes only after verification succeeds.
- No CSS source or generated bundle changes are expected.

### Task 1: Pin the pure state and resize boundary in red

**Files:**
- Modify: `Tests/Chat/test_console_rail_state.py`
- Test: `Tests/Chat/test_console_rail_state.py`

- [ ] **Step 1: Strengthen the exact-boundary rail-state test**

Replace the single 100-column assertion with a parameterized table:

```python
@pytest.mark.parametrize(
    ("width", "expected_open", "expected_override"),
    [(99, False, False), (100, True, True), (101, True, False)],
)
def test_console_rail_state_default_left_boundary(
    width: int,
    expected_open: bool,
    expected_override: bool,
) -> None:
    key = build_console_rail_preference_key(
        workspace_id="workspace-1",
        session_id="session-1",
    )
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"left_open": True},
        available_columns=width,
    )

    assert state.left_open is expected_open
    assert state.preferred_left_open is True
    assert state.left_compact_override is expected_override
    assert state.compact_override is expected_override
```

- [ ] **Step 2: Pin exact 100 as its own width-band key**

Update `test_console_rail_width_band_buckets` so the relevant rows are:

```python
(99, "narrow"),
(100, "exact-left-boundary"),
(101, "compact-before-auto-open"),
(117, "compact-before-auto-open"),
```

- [ ] **Step 3: Run the two tests and observe the intended failures**

Run:

```bash
python -B -m pytest \
  Tests/Chat/test_console_rail_state.py::test_console_rail_state_default_left_boundary \
  Tests/Chat/test_console_rail_state.py::test_console_rail_width_band_buckets -q
```

Expected: FAIL at width 100 because `left_compact_override` is false, and FAIL because width 100 still returns `compact-before-auto-open`. Widths 99 and 101 remain green controls.

### Task 2: Implement the minimum pure-state correction

**Files:**
- Modify: `tldw_chatbook/Chat/console_rail_state.py`
- Test: `Tests/Chat/test_console_rail_state.py`

- [ ] **Step 1: Split the exact resize band**

In `console_rail_width_band()` add the exact boundary after the existing narrow branch:

```python
if available_columns < CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS:
    return "narrow"
if available_columns == CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS:
    return "exact-left-boundary"
```

- [ ] **Step 2: Make the left minimum-waiver authority inclusive**

Change only the override comparison; keep the force-collapse comparison exclusive:

```python
left_compact_override = (
    available_columns is not None
    and available_columns <= CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS
    and left_open
)
```

- [ ] **Step 3: Correct the state comments**

Describe `*_compact_override` as layout-minimum-waiver authority. State explicitly that the left flag covers an honored explicit open below 100 and any effective open at exactly 100; it does not imply a persistence write or explicit user intent.

- [ ] **Step 4: Run the pure-state boundary tests**

Run the Task 1 command again.

Expected: PASS.

- [ ] **Step 5: Run the directly adjacent pure-state suite**

Run:

```bash
python -B -m pytest Tests/Chat/test_console_rail_state.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the pure-state correction**

```bash
git add tldw_chatbook/Chat/console_rail_state.py Tests/Chat/test_console_rail_state.py
git commit -m "fix(console): contain exact-width rail state"
```

### Task 3: Pin production-CSS cold-start geometry in red, then green

**Files:**
- Modify: `Tests/UI/test_console_shell_regions.py`
- Test: `Tests/UI/test_console_shell_regions.py`

- [ ] **Step 1: Extend the production pilot helper without adding a second harness**

Allow `make_console_pilot()` to accept a prepared app while retaining the exact `TldwCli.CSS_PATH` stack. The caller configures a ready llama.cpp provider with the existing `_configure_native_ready_console()` helper before `run_test`, so first paint is the Console rather than its setup overlay.

```python
@asynccontextmanager
async def make_console_pilot(*, size, app=None, hide_setup_overlay=True):
    app = app or _build_test_app()
    host = ProductionCSSConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        setup_modal = console.query_one("#console-setup-modal")
        if hide_setup_overlay:
            setup_modal.display = False
            await pilot.pause()
        else:
            assert setup_modal.display is False
        yield pilot
```

Existing setup-blocked tests retain the default overlay hide because their Inspector auto-open fixture depends on the blocked state. Exact-100 regressions pass `hide_setup_overlay=False` with the preconfigured ready app, making the already-hidden first-paint assertion load-bearing.

- [ ] **Step 2: Add one shared geometry assertion**

Add `_assert_workspace_contained(console, *, default_context_only)` that:

```python
screen = console.screen
grid = console.query_one("#console-workspace-grid")
displayed = tuple(child for child in grid.children if child.display)
for child in displayed:
    assert child.region.width > 0 and child.region.height > 0
    assert grid.content_region.x <= child.region.x
    assert child.region.right <= grid.content_region.right
    assert screen.region.x <= child.region.x
    assert child.region.right <= screen.region.right

main = console.query_one("#console-main-column")
transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
if default_context_only:
    assert main.region.width >= 55
assert transcript.content_region.width >= 40

transcript_hit = screen.get_widget_at(*transcript.region.center)[0]
assert (
    transcript_hit is transcript
    or transcript in transcript_hit.ancestors
    or transcript_hit in transcript.walk_children()
)

for child in displayed:
    hit = screen.get_widget_at(*child.region.center)[0]
    assert hit is child or child in hit.ancestors or hit in child.walk_children()
```

- [ ] **Step 3: Add the four stored-preference combinations as true cold starts**

Parameterize `(left_open, right_open)` over all four boolean combinations. Before `run_test`, configure a ready app, install `_stored_console_rail_preferences` on `ChatScreen` to return the row's payload, and replace `_save_console_rail_preferences` with a `Mock`. Assert first-painted geometry without calling `_sync_console_rail_visibility()` manually:

```python
app = _build_test_app()
_configure_native_ready_console(app)
preferences = {"left_open": left_open, "right_open": right_open}
save_spy = Mock()
monkeypatch.setattr(
    ChatScreen,
    "_stored_console_rail_preferences",
    lambda self, key, fallback_key: preferences,
)
monkeypatch.setattr(ChatScreen, "_save_console_rail_preferences", save_spy)

async with make_console_pilot(
    size=(100, 30), app=app, hide_setup_overlay=False
) as pilot:
    console = pilot.app.screen
    rail_state = console._last_console_rail_state
    assert rail_state is not None
    # Assert expected first-painted children and geometry here; do not resync.
```

This patches the existing storage-read seam rather than inventing a harness state path. Because both patches exist before composition, the test proves compose-time consumption and the zero-save cold-start oracle.

Assert the expected displayed set:

```python
expected = {
    (True, False): {"console-left-rail", "console-main-column", "console-inspector-rail-handle"},
    (False, False): {"console-context-rail-handle", "console-main-column", "console-inspector-rail-handle"},
    (False, True): {"console-context-rail-handle", "console-main-column", "console-right-rail"},
    (True, True): {"console-context-rail-handle", "console-main-column", "console-right-rail"},
}[(left_open, right_open)]
```

Also assert preferred values remain equal to the stored booleans, Inspector priority owns the open/open row, and `save_spy.assert_not_called()` after first paint.

- [ ] **Step 4: Run the matrix against pre-fix behavior if Task 2 is temporarily reverted**

Mutation preparation: temporarily restore the exclusive `<` comparison, clear bytecode caches or use `python -B`, and run:

```bash
python -B -m pytest \
  Tests/UI/test_console_shell_regions.py::test_exact_100_workspace_state_matrix_is_contained -q
```

Expected: the Context-open/Inspector-closed row FAILS with the known out-of-bounds geometry; the other three rows are green controls. Restore the inclusive comparison immediately.

- [ ] **Step 5: Run the matrix with the correction restored**

Run the same command.

Expected: all four rows PASS, with at least 55 outer main cells in the default row and at least 40 readable transcript cells in every row.

### Task 4: Pin all adjacent live-resize directions and continuity

**Files:**
- Modify: `Tests/UI/test_console_shell_regions.py`
- Test: `Tests/UI/test_console_shell_regions.py`

- [ ] **Step 1: Add a ready, populated transcript helper**

Reuse the real Console store and public transcript methods, following `Tests/UI/test_console_composer_collapse.py`:

```python
async def _seed_overflowing_transcript(console, pilot):
    store = console._ensure_console_chat_store()
    selected = ""
    for index in range(24):
        message = store.append_message(
            store.active_session_id,
            role=(ConsoleMessageRole.USER if index % 2 == 0 else ConsoleMessageRole.ASSISTANT),
            content="\n".join(f"message {index} line {line}" for line in range(3)),
        )
        selected = message.id
    await console._sync_native_console_chat_ui()
    await pilot.pause()
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    assert transcript.max_scroll_y > 0
    transcript.select_message(selected)
    transcript.release_anchor()
    transcript.scroll_to(y=2, animate=False)
    await pilot.pause()
    return transcript, selected, transcript.scroll_y
```

- [ ] **Step 2: Parameterize the four adjacent transitions**

Use `(101, 100)`, `(100, 101)`, `(99, 100)`, and `(100, 99)`. Before resize, focus the visible Context collapse control when Context is open or the reveal control when it is closed. Spy on `console._save_console_rail_preferences`, then resize and assert:

- `_assert_workspace_contained()` passes.
- 101↔100 keeps focus on `#console-context-rail-collapse`.
- 99→100 hands focus from `#console-context-rail-open` to `#console-context-rail-collapse`.
- 100→99 performs the inverse handoff.
- `selected_message_id` is unchanged.
- semantic tail-follow remains released and `scroll_y == min(reading_y, max_scroll_y)`.
- preferred left/right values are unchanged.
- the preference-save spy has zero calls.

- [ ] **Step 3: Prove the live-resize regression is red without the exact band**

Temporarily remove only the exact-100 band branch, retain the inclusive override, and run:

```bash
python -B -m pytest \
  Tests/UI/test_console_shell_regions.py::test_exact_100_adjacent_resize_converges -q
```

Expected: 101→100 FAILS because the live handler remains in `compact-before-auto-open`; at least one opposite-direction control remains green. Restore the exact band branch immediately.

- [ ] **Step 4: Run the live transition test green**

Run the same command after restoration.

Expected: all four transitions PASS.

- [ ] **Step 5: Commit production geometry and resize coverage**

```bash
git add Tests/UI/test_console_shell_regions.py
git commit -m "test(console): lock exact-width compositor geometry"
```

### Task 5: Correct production comments without changing behavior

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Chat/console_rail_state.py`

- [ ] **Step 1: Correct compose-time width arithmetic**

Update the comment above `#console-main-column` to use the shipped worst-case exact-100 row: 30-column Context + 56-column standard main minimum + 11-column horizontal Inspector handle against 96 grid-content cells after two borders and two padding cells.

- [ ] **Step 2: Correct compact-override semantics in both consumers**

In compose and `_sync_console_rail_visibility()`, say that `compact_override` grants layout-minimum-waiver authority for honored explicit rails below their thresholds and the exact-100 Context boundary. Remove obsolete references to a 24-column Context minimum or a 3-column default handle.

- [ ] **Step 3: Run comment-adjacent architecture and width contracts**

Run:

```bash
python -B -m pytest \
  Tests/UI/test_console_rail_width_budget.py \
  Tests/UI/test_console_internals_decomposition.py -q
```

Expected: width contracts PASS. If the known Console architecture ratchet fails at its unchanged baseline, record the exact measured/budget values; do not weaken it in this task.

- [ ] **Step 4: Commit comment corrections**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Chat/console_rail_state.py
git commit -m "docs(console): clarify compact width authority"
```

### Task 6: Verify, document, and close TASK-19639

**Files:**
- Modify: `backlog/tasks/task-19639 - Keep-Console-workspace-geometry-inside-the-viewport-at-exactly-100-columns.md`
- Test: focused Console and repository governance suites

- [ ] **Step 1: Run import provenance before behavior suites**

```bash
python -B -m pytest Tests/test_probe_import_provenance.py -q
```

Expected: PASS and imports resolve to this worktree.

- [ ] **Step 2: Run focused behavior and geometry suites**

```bash
python -B -m pytest \
  Tests/Chat/test_console_rail_state.py \
  Tests/UI/test_console_shell_regions.py \
  Tests/UI/test_console_resize_reflow.py \
  Tests/UI/test_console_inspector_compact_access.py \
  Tests/UI/test_console_rail_width_budget.py -q
```

Expected: PASS.

- [ ] **Step 3: Run CSS integrity and static checks**

```bash
python -B -m pytest Tests/UI/test_css_build_integrity.py -q
python -m ruff check \
  tldw_chatbook/Chat/console_rail_state.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_console_rail_state.py \
  Tests/UI/test_console_shell_regions.py
python -m ruff format --check \
  tldw_chatbook/Chat/console_rail_state.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_console_rail_state.py \
  Tests/UI/test_console_shell_regions.py
git diff --check
```

Expected: PASS. No CSS bundle rebuild is needed because no TCSS changes are planned.

- [ ] **Step 4: Run backlog and architecture governance checks**

Run:

```bash
python -B -m pytest \
  Tests/UI/test_product_maturity_phase1_harness.py::test_backlog_task_frontmatter_ids_are_unique -q
python -B -m pytest Tests/Architecture/test_screen_size_ratchet.py -q
```

Record unchanged baseline failures—currently TASK-17165's malformed ID and the 21,292 versus 17,727 Console architecture budget—without claiming those checks are green or changing them outside this task.

- [ ] **Step 5: Run the full test suite checkpoint**

```bash
python -B -m pytest
```

Expected: all task-reachable tests pass. Record unrelated unchanged repository baseline failures with exact commands and evidence.

- [ ] **Step 6: Complete the task file**

Check all acceptance criteria, add concise `## Implementation Notes` listing behavior, tests, mutations, modified files, ADR-043 reuse, and any unchanged baseline failures, then set status to `Done` only if every Definition-of-Done item is satisfied.

- [ ] **Step 7: Review the final diff**

Confirm the change contains only the inclusive override, exact resize band, tests, comment corrections, and task documentation. Do not include TASK-18915 or manual-collapse focus behavior.

- [ ] **Step 8: Commit task completion**

```bash
git add 'backlog/tasks/task-19639 - Keep-Console-workspace-geometry-inside-the-viewport-at-exactly-100-columns.md'
git commit -m "docs(backlog): complete TASK-19639"
```
