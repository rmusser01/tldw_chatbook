---
id: TASK-3316
title: >-
  screen_navigation file-notes collections transition test hangs and kills the
  whole run
status: Done
assignee: []
created_date: '2026-08-08 21:30'
updated_date: '2026-08-11 03:01'
labels:
  - tests
  - file-notes
  - dev-baseline
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during follow-up batch phase A (2026-08-08): `Tests/UI/test_screen_navigation.py::test_file_notes_collections_source_transition_blocks_mutation_through_recompose` hangs on dev base `ebeae1440` (reproduced with the phase's product diff fully reverted). Under the repo's `timeout_method = thread` a hung test dumps stacks and terminates the ENTIRE pytest process (the task-1466 lesson), so any run that collects this file dies — which also hides every test after it. Belongs to the file-notes/collections surface, not the ingest arc; filed from the ingest batch so it does not rot unowned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The hang's mechanism is identified and fixed (or the test bounded at its source) so the file completes under the standard timeout
- [x] #2 Full `Tests/UI/test_screen_navigation.py` completes with a READ pass count
- [x] #3 Failures the hang had been hiding in this file are repaired, not just revealed -- the file runs green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the hang alone with a short --timeout and read the pytest-timeout stack dump.
2. Surface what the fire-and-forget task actually did (bounded diagnostic) rather than inferring from the idle-event-loop dump.
3. Date the regression: git blame the failing production line + the test stub, then bisect by running the test from git archive copies of dev SHAs outside the repo.
4. Fix the stale stub against the current _flush_library_note_save contract.
5. Bound every unbounded 'await an Event a background task must set' in this file at its source, surfacing the task's swallowed exception instead of waiting forever.
6. Run the FULL file, report the READ pass count, and repair whatever failures the hang had been hiding.
7. Mutation-check: restore the stale stub with the bound in place and confirm it now FAILS FAST naming the real cause instead of hanging.
8. Record the lesson (incident + rule) in backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the hang at its source and repaired what it had been hiding.

**Root cause.** The test drives `screen._select_library_rail_row(...)` as a fire-and-forget `asyncio.create_task` and then waits on `sync_returned`, an Event only that coroutine can set. Its `_flush_library_note_save` stub returned `None` -- correct against the seam's `-> None` signature when the test was authored (eb036a6a1, 2026-07-27) -- but PR #1439 retyped the seam to `NoteFlushOutcome` and made the caller read `note_flush.kind`, so the awaited path died one line in on `AttributeError: 'NoneType' object has no attribute 'kind'`. A `create_task` result nobody retrieves swallows that whole, and the signal became unreachable: `await sync_returned.wait()` blocked forever.

The pytest-timeout dump does NOT name it -- the test coroutine is suspended at an await, so the only stack shown is MainThread idle in `selectors.select`. Bounding the wait and asking the task is what talked.

**Landed on dev with PR #1439, before the media-ingest arcs.** Bisected by running the test from `git archive` copies outside the repo: 86e511781 (#1439's first parent) *1 passed in 3.64s*; 6b4ccf475 (#1439's merge) *hang -> process killed*. 6b4ccf475 is an ancestor of ebeae1440 (#1452, the ingest arc), so the arcs are exonerated.

**Fix.** (a) The stub returns a real `NoteFlushOutcome(PERMITTED)`. (b) Two module-level helpers in the test file bound every 'await a signal a background task must set' at its source: `_wait_for_background_signal` returns the moment the Event is set, and otherwise re-raises the task's swallowed exception (or reports its silent early return) instead of waiting; `_await_background_task` bounds the trailing `await task`. Applied to all three tests in the file with that shape, so no future gating change can make any of them unbounded again.

**What the hang had been hiding** (AC#3): with the file finally completing, three more tests surfaced. Two were hard, same defect class -- stale stubs against retyped seams: `test_action_library_note_editor_back_honors_dirty_guard` assigned to `_library_note_dirty`, now a read-only property over the session snapshot (the dirty veto is expressed by the flush OUTCOME), and asserted the old `_focus_library_list_entry` handoff instead of today's focus-identity restore; `test_action_library_prompt_editor_back_honors_dirty_guard` broke because the guarded exit now re-requests the prompts page through the browse controller (needs a running App) and no longer calls `refresh` directly. Both repaired to the current contracts, preserving their original claims. The third, `test_skills_route_lands_on_library_with_skills_row_selected`, is a load-sensitive flake (12/12 alone, 1 failure seen in a contended full run, text not captured); its budget is bounded so it fails loudly rather than fatally -- reported, not chased.

**Evidence.** Full `Tests/UI/test_screen_navigation.py`: **126 passed** (86s). The three bounded tests looped 15x, 15/15 green. Mutation: restoring the stale `return None` with the bound in place fails in **2.2s naming the AttributeError** where it previously hung the whole process.

**Files:** `Tests/UI/test_screen_navigation.py`, `backlog/docs/lessons-testing-evidence.md` (new entry: an `await event.wait()` on a fire-and-forget task hangs on the task's own exception, and the timeout dump names nothing).
<!-- SECTION:NOTES:END -->
