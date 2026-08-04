# Console Decomposition — Wave 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove both collaborator kinds from the screen-decomposition design on the Console — three rail region widgets and one controller — landing fast enough that the most concurrently-modified file in the repo is not held hostage by a long-lived branch.

**Architecture:** Per `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`. Region widgets own pixels; controllers do not. New collaborators live in `UI/Console_Modules/` (new package, mirroring `UI/Evals/`). Every extraction preserves DOM ids verbatim and changes no behaviour.

**Tech Stack:** Python ≥3.11, Textual, pytest.

**Why a wave, not the whole screen:** `chat_screen.py` is the most concurrently-modified file in this repository — dev took 161 commits in one recent day, several from sessions working in this exact file. A branch carrying nine cluster extractions would rot before review. Wave 1 proves the pattern (three regions + one controller + the compose skeleton); wave 2 (workspace, session, message, agent, character, image/attachment, composer-orchestration controllers) is planned after wave 1 merges, carrying what it learned. This refines the spec's "one implementation plan per screen" into "per screen, in waves that each land independently" — same intent, honest about this file's churn rate.

## Global Constraints

The spec's six migration rules, verbatim — all non-negotiable:

1. **One region per change.** A change moves exactly one region or extracts exactly one controller. No batch moves.
2. **Ids are preserved verbatim.** A region widget composes the same ids in the same nesting. If an id must change, that is its own change with its own review, never a passenger on an extraction.
3. **Painted-geometry assertions before and after.** Every extraction carries a test asserting the moved region's controls are hit-testable — `screen.get_widget_at(*control.region.center)` resolves to the control — at 160x45 AND 235x52. Task 1 writes these against the CURRENT code, so they are proven to pass before anything moves.
4. **CSS moves with its region** into `css/features/`, and the bundle is regenerated via `build_css.py`. The bundle is never hand-edited.
5. **Behaviour changes are forbidden in an extraction.** An extraction that also fixes a bug is two changes.
6. **A characterisation test precedes any extraction whose behaviour is not already covered.**

House rules:

- **Do not start Task 1 while feature work is active in the Console.** This is an
  owner ruling (2026-08-02), not a style preference: refactoring `chat_screen.py`
  concurrently with feature branches in the same file guarantees conflict pain for
  both sides. The pre-flight gate before dispatching Task 1 is:
  `git log origin/dev --oneline --since="24 hours ago" -- tldw_chatbook/UI/Screens/chat_screen.py`
  plus a scan of open `origin/*console*` branches — at the time of the ruling that
  showed 8 commits in 36 hours and four live console feature branches
  (`console-cost-usage-foundation`, `console-voice-control-v2`,
  `console-message-selection-toggle`, `controlbar-save-chatbook-removal`). Start only
  when the churn has visibly settled AND the owner has confirmed the window is open.
- Run tests foreground: `/private/tmp/tldw-venv/bin/python -m pytest <paths> -p no:randomly` from the clone root. Never `-q`. **Pass `timeout: 600000` on the Bash call** — this harness auto-backgrounds anything past its 120s default and a backgrounded pytest has stalled implementers repeatedly.
- COMMIT BEFORE ANY MUTATION CHECK; never `git stash` (shared across 100+ worktrees); never `git checkout --` on uncommitted work.
- **Rebase onto `origin/dev` before starting each task.** This file moves under you; a stale base guarantees conflicts at merge. CSS-bundle conflicts resolve by `git checkout --theirs` the bundle then regenerating via `build_css.py`.
- Line numbers in this plan are anchors measured at dev `073d640ac`; they WILL have drifted. Re-locate by the named anchor (method name, id, `_frame_console_region` call), never by trusting the number.
- Existing DOM-driven tests must pass unchanged. Tests reaching into private methods that moved get mechanically retargeted with assertions kept byte-for-byte.

## Verified facts (measured at dev `073d640ac` — re-verify anchors, not conclusions)

- `ChatScreen` spans lines 1559–20338; 567 methods. `compose_content` is lines 12546–13226.
- `compose_content`'s regions are bounded by **seven `_frame_console_region(...)` calls** at ~12659 (left handle), ~12672 (`left_rail`), ~12771 (main column, a multi-line call), ~13087 (`right_rail`), ~13139 and ~13159 (inspector area, including the `console-run-inspector` Vertical at ~13162), ~13195 (`right_handle`, `variant="quiet"`). These are the extraction seams.
- `_frame_console_region(widget, *, top=True, variant="solid")` is a static styling helper that adds `console-frame-solid`/`console-frame-quiet` classes and sets `widget.styles.border`. It must move to `UI/Console_Modules/` where all regions can import it — it is the one shared piece.
- **`on_button_pressed` (368 lines) is NOT a flat dispatch** — it references only ~18 `console-*` ids with substantial per-branch bodies (`console-conversation-browser-*`, `console-session-tab-*`, `console-dictation`, `console-send-message`, `console-agent-drilldown-back`, …). Rail-owned branches move with their rails; the rest stays for wave 2.
- **`ConsoleComposerBar` already exists** (`#console-native-composer`, see `_console_composer_or_none` ~5137). The composer is NOT a wave-1 region; its screen-side orchestration is a wave-2 controller.
- `ConsoleRunInspector`, `ConsoleStatusChips`, `ConsoleSetupModal` are already widgets, yielded by `compose_content`.
- The rail-section machinery on the screen: `_console_rail_state_config` (~9813, 8 lines), `_console_rail_available_columns` (~10232, 4), `_sync_console_rail_sections` (~10553, 10), `_apply_console_rail_section_open` (~10564, 16), `_toggle_console_rail_section` (~10581, 33), `_console_rail_system_line_state` (~3806, 18).
- Dictation: module-level `ConsoleStreamingDictationSession` (~744, 322 lines), `ConsoleDictationEvent` (~619), `ConsoleDictationLimitSignal` (~643), plus ~20 `*dictation*` methods on the screen (742 lines) and the `self._console_dictation_state` attribute.
- The shell is responsive: `_sync_compact_shell_controls`, `_compact_console_workbench_widget`, `_hidden_console_workbench_widget` exist, so **at 160x45 some regions may legitimately be hidden or compact**. Task 1 pins what IS, not what "should be".
- Test harnesses to reuse (verify their real shape first): `ConsoleHarness` in `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`, `ConsoleNavigationHarness` in `Tests/UI/test_console_native_chat_flow.py`, `_visible_text` alongside them. `Tests/UI/test_console_internals_decomposition.py` imports all of these — read its top for the working import pattern.
- `UI/Console_Modules/` does not exist. Create it in Task 2 with an `__init__.py` carrying a module docstring pointing at the spec.

---

### Task 1: Pin the shell's geometry before anything moves

**Files:**
- Create: `Tests/UI/test_console_shell_regions.py`

**Interfaces:**
- Consumes: `ConsoleHarness` (or `ConsoleNavigationHarness` — whichever mounts the full shell; verify), `_visible_text`.
- Produces: the baseline suite every later task must keep green **byte-identical**.

This is spec rule 3 done properly: the assertions are written against the CURRENT code and proven to pass before anything moves, so when Task 3 relocates the left rail, a failure means the move broke something — not that the test was born wrong.

- [ ] **Step 1: Discover the current truth at both sizes**

Write a THROWAWAY probe (do not commit it) that mounts the Console at 160x45 and at 235x52 and prints, for each of these ids, whether it exists, `display`, and `region`: `#console-shell`, `#console-left-rail`, `#console-left-rail-body`, `#console-main-column`, `#console-context-rail-handle`, `#console-inspector-rail-handle`, `#console-control-bar`, `#console-mode-bar`, `#console-native-composer`, `#console-run-inspector`. The shell is responsive — some of these may be hidden or compact at 160x45. Record what you observe in your report.

- [ ] **Step 2: Write the baseline tests from what you observed**

One parametrized test per region id, shaped like this — with the expectation table filled from Step 1's observations, NOT from assumption:

```python
"""Painted-geometry baseline for the Console shell's regions.

Written BEFORE the wave-1 extractions (spec rule 3, screen-decomposition
design). Every extraction task must keep this file green and byte-identical.
If an extraction needs this file to change, the extraction changed behaviour
-- stop and treat that as a finding.

The expectation table pins what the shell DOES at each size as of the
baseline commit, including regions that are legitimately hidden in compact
mode. It does not pin what anyone thinks it should do.
"""

import pytest

# (id, expected_at_160x45, expected_at_235x52) where expected is
# "hittable" | "hidden" -- FILL FROM THE STEP-1 PROBE, per observation.
_REGIONS: list[tuple[str, str, str]] = [
    ("#console-left-rail", "<observed>", "<observed>"),
    # ... every id from Step 1 ...
]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (235, 52)])
@pytest.mark.parametrize("region_id,expect_small,expect_large", _REGIONS)
async def test_region_geometry_is_stable(region_id, expect_small, expect_large, size):
    expected = expect_small if size == (160, 45) else expect_large
    async with make_console_pilot(size=size) as pilot:  # the real harness, per its own idiom
        nodes = pilot.app.screen.query(region_id)
        if expected == "hidden":
            assert not nodes or not nodes[0].display
            return
        node = nodes[0]
        assert node.display and node.region.width > 0
        hit = pilot.app.screen.get_widget_at(*node.region.center)[0]
        assert hit is node or node in hit.ancestors or hit in node.walk_children()
```

`make_console_pilot` stands in for however the real harness is entered — copy the exact idiom from `test_console_internals_decomposition.py`'s own tests. The `hit in node.walk_children()` arm exists because a container's center usually resolves to a child; hitting any descendant proves the region paints and receives the mouse.

The `"<observed>"` placeholders MUST all be replaced with `"hittable"` or `"hidden"` before commit — a grep for `<observed>` in the committed file is a task failure.

- [ ] **Step 3: Run the file against unmodified code**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_console_shell_regions.py -p no:randomly` (Bash `timeout: 600000`)
Expected: PASS on the first complete run. A failure here means the expectation table does not match reality — fix the table, not the shell.

- [ ] **Step 4: Commit**

```bash
git add Tests/UI/test_console_shell_regions.py
git commit -m "test(console): pin shell region geometry before decomposition (wave 1)"
```

---

### Task 2: The shared frame helper and the package

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/__init__.py`
- Create: `tldw_chatbook/UI/Console_Modules/frame.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_shell_regions.py` (unchanged — that is the assertion)

**Interfaces:**
- Produces: `frame_console_region(widget, *, top=True, variant="solid")` — the same behaviour `ChatScreen._frame_console_region` has today, as a module function all wave-1 regions import.

The frame helper is the one piece every region shares. Moving it first means Tasks 3-5 never import from `chat_screen` (which would be a cycle: the screen imports the regions).

- [ ] **Step 1: Read the real `_frame_console_region`** (~12530s, a `@staticmethod` or plain method — check) including the constants it uses (`CONSOLE_QUIET_FRAME_BORDER`, the `ConsoleComposerBar` special-case) and where those constants live.

- [ ] **Step 2: Create the package and move the helper.** `__init__.py` docstring: one paragraph naming the spec and the rule ("a region widget owns pixels; a controller does not"). `frame.py` holds `frame_console_region` as a module function, moved verbatim, with its constants imported from their current homes (import them — do not copy them). `ChatScreen._frame_console_region` becomes a one-line delegation to it (kept temporarily so the not-yet-moved call sites in `compose_content` are untouched; Task 6 removes it).

- [ ] **Step 3: Run the baseline plus the decomposition suite**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_console_shell_regions.py Tests/UI/test_console_internals_decomposition.py -p no:randomly` (timeout 600000)
Expected: PASS, baseline byte-identical.

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/ tldw_chatbook/UI/Screens/chat_screen.py
git commit -m "refactor(console): shared region frame helper in UI/Console_Modules (wave 1)"
```

---

### Task 3: The left rail becomes a region widget

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/left_rail.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_left_rail.py` (new), `Tests/UI/test_console_shell_regions.py` (unchanged)

**Interfaces:**
- Consumes: `frame_console_region` (Task 2).
- Produces: `ConsoleLeftRail` — composes the block currently at `compose_content` ~12659–12770 (the `left_handle` frame plus the `left_rail` frame's subtree), same ids, same nesting. Messages upward for anything that touches other regions.

- [ ] **Step 1: Map the block.** Read `compose_content` from the `left_handle` frame call to the line before the ~12771 main-column frame call. List every id and every widget class it yields, and every `self.*` it references, in your report. That reference list decides what moves with it.

- [ ] **Step 2: Write the characterisation test first.** `Tests/UI/test_console_left_rail.py`, driving the REAL screen through the real harness: open/collapse a rail section through an actual `pilot.click` on its real control, and assert the persisted outcome (the section's open state after the click, and again after toggling back). Also: pressing a rail section header does not steal focus from the composer if that is today's behaviour — pin whichever way it currently behaves. Run it against unmodified code; it must pass BEFORE the move.

- [ ] **Step 3: Extract.** `ConsoleLeftRail(Vertical)` in `left_rail.py`:
  - `compose()` yields the moved block verbatim — same ids, same nesting, `frame_console_region` from Task 2.
  - The rail-section machinery moves onto it: `_console_rail_state_config`, `_console_rail_available_columns`, `_sync_console_rail_sections`, `_apply_console_rail_section_open`, `_toggle_console_rail_section`, `_console_rail_system_line_state` — *if* Step 1's reference map shows they touch only rail state and rail DOM. Any of them that reaches other regions stays on the screen, and the rail posts a message instead:

```python
    class SectionToggled(Message):
        """A rail section was opened or closed by the user.

        Args:
            section_id: The toggled section's id.
            opened: True when the section is now open.
        """

        def __init__(self, section_id: str, opened: bool) -> None:
            self.section_id = section_id
            self.opened = opened
            super().__init__()
```

  - `on_button_pressed` branches whose ids live inside this block move onto the rail as `@on(Button.Pressed, "#<id>")` handlers. Branches whose bodies touch other regions stay on the screen; the rail's job is only to be the place its own buttons are handled.
  - State the rail needs at construction is passed as named constructor arguments (never `app_instance` wholesale — spec rule).
  - `compose_content` replaces the moved block with `yield ConsoleLeftRail(...)`.

- [ ] **Step 4: Run everything that touches this surface**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_console_shell_regions.py Tests/UI/test_console_left_rail.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py -p no:randomly` (timeout 600000)
Expected: PASS; baseline and characterisation files byte-identical to their pre-move state.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/left_rail.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_left_rail.py
git commit -m "refactor(console): left rail is a region widget (wave 1)"
```

---

### Task 4: The right rail becomes a region widget

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/right_rail.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_right_rail.py` (new), `Tests/UI/test_console_shell_regions.py` (unchanged)

**Interfaces:**
- Consumes: `frame_console_region`.
- Produces: `ConsoleRightRail` — the `right_rail` frame block at ~13087–13158, same contract as Task 3.

Same steps as Task 3, applied to the right-rail block: map the block and its `self.*` references; write the characterisation test against unmodified code (collapse/expand through a real click, pinned persisted state); extract with ids verbatim; move only handlers whose bodies stay inside the rail; message upward otherwise.

One thing to resolve in Step 1 and state in your report: this codebase has ids for BOTH `console-context-rail-*` and `console-inspector-rail-*`. Determine which family lives in this block and name the widget for what it actually is (`ConsoleContextRail` if that is what the block holds) — the class name should match the DOM family, not this plan's guess.

- [ ] **Step 1: Map the block** (as Task 3 Step 1).
- [ ] **Step 2: Characterisation test first**, passing before the move.
- [ ] **Step 3: Extract**, ids verbatim, same message pattern.
- [ ] **Step 4: Run** the same four test files plus the new one (timeout 600000). Expected: PASS.
- [ ] **Step 5: Commit** — `refactor(console): right rail is a region widget (wave 1)`.

---

### Task 5: The dictation controller

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/dictation.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_dictation_controller.py` (new)

**Interfaces:**
- Produces: `ConsoleDictationController` — owns `_console_dictation_state` and the dictation lifecycle; the screen delegates. This is wave 1's proof of the controller kind.

Dictation is the most self-contained cluster: its session class (`ConsoleStreamingDictationSession`, 322 lines) and its event types (`ConsoleDictationEvent`, `ConsoleDictationLimitSignal`) are ALREADY module-level in `chat_screen.py`, and the screen's ~20 `*dictation*` methods (~742 lines) cluster around one state attribute.

- [ ] **Step 1: Map the cluster.** List every method matching `*dictation*` on `ChatScreen`, plus the three module-level types, plus every OTHER method that reads `self._console_dictation_state`. The controller boundary is that state attribute: methods that touch it and nothing region-shaped move; methods that merely call into the cluster stay as one-line delegations.

- [ ] **Step 2: Characterisation.** Existing dictation tests exist (`Tests/UI` and `Tests/Audio` — locate them by grep). Run them against unmodified code and record the counts; they are this task's regression net. Where the mic-button flow lacks a DOM-driven test, add one: a real click on `#console-dictation` asserting the persisted state change, passing BEFORE the move.

- [ ] **Step 3: Extract.** Move the three module-level types and the state-owning methods into `dictation.py`. `ConsoleDictationController.__init__` takes named dependencies only — the screen handle for `run_worker`/`call_from_thread`, and the specific app services the mapped methods actually use (found in Step 1), never `app_instance` itself. Workers it starts use `group="console-dictation"` so `exclusive=True` scopes to dictation (spec rule). The screen keeps its public entry points (`action_*`, event handlers) as one-line delegations to `self._dictation`.

- [ ] **Step 4: Run** the dictation tests found in Step 2, the new test, and the Task-1 baseline (timeout 600000). Expected: PASS, with any retargeted private-method tests keeping their assertions byte-for-byte.

- [ ] **Step 5: Commit** — `refactor(console): dictation controller in UI/Console_Modules (wave 1)`.

---

### Task 6: The compose skeleton, the convention, and the record

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `DESIGN.md`
- Modify: `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`
- Test: full serial sweep

**Interfaces:** none new — this task closes the wave.

- [ ] **Step 1: Remove the scaffolding.** Delete `ChatScreen._frame_console_region`'s delegation shim if no call site remains inside the screen; if call sites remain (the main column still composes inline until wave 2), keep the shim and say so in the report — do not force it.

- [ ] **Step 2: Measure and record.** In the spec's chat table, annotate the extracted rows with "(wave 1, done)" and record the new `chat_screen.py` line count and `ChatScreen` method count next to the old ones. Honest numbers — the file will still be large; the point is the regions and dictation are owned elsewhere now.

- [ ] **Step 3: Write the convention into `DESIGN.md`.** A "Screen decomposition" section: the one rule (region widget owns pixels; controller does not), where collaborators live (`UI/<Screen>_Modules/`), the six migration rules by reference to the spec, and `UI/Evals/` + wave 1 as the two existence proofs.

- [ ] **Step 4: Full serial sweep**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI -p no:randomly -k "console or shell"` then, if green, the full `Tests/UI` serially (timeout 600000 each; the full run takes several minutes).
Expected: PASS. This repo's parallel runs produce cross-test interference — serial is the gate.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py DESIGN.md Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md
git commit -m "refactor(console): wave 1 closed - skeleton, convention, honest numbers"
```

---

## Wave 1 exit criteria

- The left and right rails are region widgets in `UI/Console_Modules/`, composing the same ids in the same nesting, handling their own buttons, messaging upward for anything cross-region.
- Dictation state and lifecycle live in a controller; the screen holds one-line delegations.
- `Tests/UI/test_console_shell_regions.py` is byte-identical to its Task-1 commit and green.
- Every pre-existing DOM-driven Console test passes unchanged; retargeted private-method tests kept their assertions byte-for-byte.
- No id changed, no behaviour changed, no CSS rule changed meaning.
- `DESIGN.md` states the convention; the spec records what wave 1 actually shipped.

## Not in wave 1 (deliberate)

The main-column block (~12771–13054) and its workspace grid; the workspace, session, message, agent, character, and image/attachment controllers; the composer-orchestration controller; `on_key` (190 lines); the `__init__` split. All of that is wave 2, planned after this merges — sized by what these six tasks teach about the real cost of a move. Jump mode and border key hints (tasks 1950/1951) remain separate work that becomes easier after the regions exist.
