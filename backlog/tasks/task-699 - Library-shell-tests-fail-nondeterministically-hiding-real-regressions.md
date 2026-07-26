---
id: TASK-699
title: 'Library shell tests fail nondeterministically, hiding real regressions'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-26 15:02'
updated_date: '2026-07-26 18:18'
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
Reduced this file from 9 failures on dev to 0-1, by fixing what turned out not to be flakiness at all.

Three of the failures were STABLE across every run and were misfiled as nondeterminism. They were two over-broad precedence traps (task-687) and one assertion that compared a mounted form against a never-mounted one (task-698). Both are fixed; the screen was correct in all three cases. Full-file runs after those fixes: 1 failed / 256 passed, then 0 failed / 257 passed.

What remains is a genuinely order-dependent tail, currently one test:
test_library_shell_note_conflict_shows_overwrite_reload_and_keeps_user_text. It
passes alone, passes alongside its siblings, and fails only in a full-file run --
and not on every full-file run. Its wait carries a 15-second timeout, so simple
slowness does not explain it; the cause is accumulated state across 257 Textual
app instances rather than a slow machine.

Measurement conditions matter for anyone picking this up: this machine was
running fourteen concurrent pytest processes from other agents at load ~12
throughout. That is enough to change which tests lose a race, so compare failure
SETS from identical commands in parallel worktrees, never counts between
different invocations -- a 3-vs-4 count difference across two different commands
briefly looked like a regression during the 684 work and was not one.

Remaining scope for this task: the note-conflict tail, plus the pool observed
earlier under load (note_save_result_after_switch_is_discarded,
export_registry_failure_*, ingest_canvas_different_canvas_isolation), all of
which pass in isolation.
<!-- SECTION:NOTES:END -->
