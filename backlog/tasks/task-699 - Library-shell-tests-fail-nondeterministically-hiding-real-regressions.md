---
id: TASK-699
title: 'Library shell tests fail nondeterministically, hiding real regressions'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-26 15:02'
updated_date: '2026-07-26 22:20'
labels:
  - testing
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library shell test file fails around six tests on every run, but not the same six: the set shifts between runs on identical code, and every one of the varying tests passes when run alone. This holds equally on the development branch, so it is not caused by any single change. The cost is that the file cannot answer whether a change broke something -- a genuine regression would be indistinguishable from the day's shuffle, and comparing failure counts between runs actively misleads. Three failures are stable and separately actionable; the rest are order-dependent, clustering around note conflict handling, note save results after a switch, export registry warnings, and ingest canvas isolation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Library shell test file produces the same result on repeated runs of the same code,A test that passes alone does not fail as part of the file,The three consistently failing tests are either fixed or explicitly recorded as known failures with reasons
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reduced this file from 9 failures to a low-rate intermittent, and ruled out two candidate causes. NOT closed.

WHAT WAS FIXED. Three of the original nine were STABLE, not flaky, and were misfiled here as nondeterminism: two over-broad config-precedence traps (task-687) and one assertion comparing a mounted form against a never-mounted one (task-698). Both merged in PR #933. The screen was correct in all three cases.

WHAT WAS RULED OUT. The wait helpers were bounded by a count of pilot.pause(0.02) calls rather than wall clock, which is genuinely wrong -- a pause takes as long as the loop needs, so under contention the budget silently shrinks. Fixed, and worth keeping. But it is NOT the cause of the remaining tail: the very next full-file run still failed, in 289s against an earlier 520s, i.e. under LESS load. Do not re-litigate load sensitivity; it has been tested.

Also ruled out: cross-contamination inside the notes suite. All 81 notes tests pass together, and each failing test passes in isolation. So the polluter, if there is one, is outside that set.

OBSERVED RATES, so the next attempt starts from data. Before the helper change: 4 failed, 2, 0, 1, 0. After: 1, 0. Every failure has been in the notes/note-conflict family -- note_conflict_shows_overwrite_reload_and_keeps_user_text, note_conflict_during_preview_reads_live_text, note_conflict_reload_discards_local_edits, note_save_result_after_switch_is_discarded, notes_sync_now_calls_recording_service_with_chosen_enums -- plus export_registry_failure_* and ingest_canvas_different_canvas_isolation seen earlier.

WHAT I NEVER GOT. The assertion text. Every encounter was a bare FAILED line, and the two runs where I set out to capture the traceback both passed. That is the single most useful next step: run the full file repeatedly with -rf until one fails and the message is captured, since the failing tests have several await points where a prior test's pending autosave or a stale preview snapshot could interfere -- a far better fit for 'passes alone, fails in a full file' than timing.

Suggested approach next time: rather than bisecting 257 tests by hand, run the file under pytest-repeat or in a loop capturing -rf output, get one real traceback, and work back from the assertion. A rate this low makes single-run bisection unreliable.
<!-- SECTION:NOTES:END -->
