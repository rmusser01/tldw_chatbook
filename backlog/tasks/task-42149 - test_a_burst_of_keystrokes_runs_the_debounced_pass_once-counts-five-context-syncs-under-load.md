---
id: TASK-42149
title: "test_a_burst_of_keystrokes_runs_the_debounced_pass_once counts five context syncs under load"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 21:34
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/test_console_rail_search_debounce.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the library stream (round 18), which saw it fail 3 of 4 local runs at base with `assert 5 <= 4`. Reproduced at d85e90bc9e: under a temporary `@private_profile_test` marker (the plain test otherwise stops at the local `RecoveryRequired: raw_source_selection_changed` trap), 2 of 4 runs at `-n 4`, load 11-16, failed `assert counter.context_syncs <= 4` with 5 (`Tests/UI/test_console_rail_search_debounce.py:226`).

The test counts context syncs during a fixed `pilot.pause(0.35)` after six keystrokes. Its comment allows the debounced pass, its worker's sync and one sync from the 0.2 s native Console tick. Under load more ticks can land in the window, so the count does not isolate what the test pins: that six keystrokes do not cost six passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test pins that six keystrokes inside the debounce window run one debounced pass, counted independently of how many native ticks land in the window
- [x] #2 It passes 10 of 10 runs under load (load recorded), and a negative control (no debounce, so each keystroke runs a pass) fails it
<!-- AC:END -->

## Implementation Plan

1. Probe the counted context syncs under load (caller of each sync, with time since the keystrokes).
2. Name the cause from the probe.
3. Count the debounced pass itself (`_start_console_conversation_browser_search`), not context syncs: none before the first await, exactly one, for the final query, after a bounded wait for it.
4. Negative control: no debounce (the keystroke runs the pass at once) fails the test. 10 of 10 under load.

## Implementation Notes

**Cause.** The debounce is not broken. A probe logged the caller of every counted `_sync_console_workspace_context` in the 0.35 s window (temporary `@private_profile_test`, `-n 4`): 6 of 10 runs failed at load 29-55, and every failing run had exactly one debounced pass (`_start_console_conversation_browser_search`, about 0.205 s after the keystrokes). The other syncs were follow-up work landing within 15 ms of it: the persisted-row refresh (the pass drops that cache), the workspace-files availability refresh, and the search worker's two syncs. The files-availability refresh only runs when its 1 s cache has expired. Under load more than 1 s passed between the boot snapshot and the pass, so it added a fifth sync. One run also caught a boot-time persisted-row refresh inside the window (six syncs). No run showed the 0.2 s native tick the old comment blamed.

**Fix.** The test now counts the debounced pass itself: an instance attribute on the workspace controller wraps `_start_console_conversation_browser_search` (the keystroke's timer looks it up there). It asserts none ran and no context sync ran before the first await, then waits (bounded, 250 x 0.02 s) for the pass and asserts it ran once, for "alpha6". The context-sync bound is gone.

**Evidence.** Negative control: `transition_browser_search` calling the pass at once instead of arming the timer fails `assert ['a', 'al', '...ha', 'alpha6'] == []`. Fixed test: 10 of 10 at `-n 4`, load 31-44, under a temporary `@private_profile_test` marker, removed afterwards (the plain test stops at the local `RecoveryRequired` trap). The file is in no PR lane, and this task asks for none.

**Review rev1 (2026-11-21).** After the one pass lands, the test pauses 0.25 s more and asserts `passes == ["alpha6"]` again, so a late duplicate pass fails.
- **Negative controls:**
  - A second timer per keystroke at 0.35 s (a late duplicate pass) passes the first check and fails the re-check: `['alpha6', 'a', ...] == ['alpha6']`, 2 of 2.
  - Leaving the superseded timers running fails the first check, 2 of 2.
- **Green:** 10 of 10 at `-n 4`, load 19-22, under the temporary marker, removed afterwards.

Files: `Tests/UI/test_console_rail_search_debounce.py`.
