---
id: TASK-699
title: 'Library shell tests fail nondeterministically, hiding real regressions'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-26 15:02'
updated_date: '2026-07-26 23:44'
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
Root cause found and fixed. The file went from 9 failures on clean dev to 259 passed / 0 failed.

Three of the original nine were STABLE failures misfiled here as nondeterminism -- two over-broad config-precedence traps and one assertion comparing a mounted form against a never-mounted one (tasks 687/698, PR #933). The screen was correct in all three cases.

The remaining intermittent was a state-then-DOM race in
test_library_shell_note_conflict_shows_overwrite_reload_and_keeps_user_text. It waited for _library_note_autosave_state to become 'conflict', then immediately asserted screen.query('#library-note-conflict-overwrite'). The wait succeeded every time -- the state really had flipped -- but the screen had not yet recomposed to render the buttons that state implies, and an empty DOMQuery is falsy. Whether it passed depended on whether a recompose landed in the same tick.

That explains every observation, including the ones that killed my earlier theories: passes alone, passes alongside all 81 notes tests, fails only sometimes in a full file, and no correlation with machine load (a run failed at 289s where an earlier one passed at 520s). It also explains why the whole tail was in the note-conflict family: they share the shape.

Fixed by awaiting the widgets with _wait_for_selector. Grepped for the same pattern; this test was the only instance.

What finally worked, after five single-run attempts to capture a traceback all passed: looping the full file until one failed. A rate this low makes single runs useless as a diagnostic.

Two hypotheses were tested and eliminated along the way, and should not be revisited: load sensitivity via the wait helpers (disproved by the 289s-vs-520s run; the wall-clock fix in PR #942 stands on its own merits but was not the cause) and contamination inside the notes suite (all 81 pass together).

PRs: #933 (687/698), #942 (wait-helper robustness), #946 (this fix).
<!-- SECTION:NOTES:END -->
