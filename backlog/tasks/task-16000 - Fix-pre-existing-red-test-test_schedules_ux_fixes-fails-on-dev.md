---
id: TASK-16000
title: 'Fix pre-existing red test: test_schedules_ux_fixes fails on dev'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 01:10'
labels:
  - tests
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`test_schedules_ux_fixes` is red on dev — surfaced while baselining test failures during the TASK-15450 review (it is NOT attributable to the consolidation; it failed identically at the pre-consolidation base) and is absent from the known-red batch task-15766 filed 2026-08-13. Diagnose whether the test or the production surface drifted, fix accordingly, and if the investigation shows a class of similar drift, note it. Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root cause identified (test drift vs production regression) with the introducing commit named
- [x] #2 The test passes, or is corrected to pin current intended behavior with the change justified
- [x] #3 If a production regression: the fix ships with born-red evidence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate the test (`Tests/UI/test_schedules_ux_fixes.py`) and reproduce at HEAD, capturing verbatim output.
2. If it now passes, determine whether a prior fix already landed on this branch's history; if so, verify that fix's diagnosis independently rather than re-deriving from scratch.
3. Reproduce the ORIGINAL (pre-fix) assertions against current production code to obtain born-failing evidence and confirm test-drift vs production-regression.
4. Trace the introducing commit(s) via `git log -S`/blame/ancestry-path on the production widget file to explain *why* the test's expectations diverged from shipped behavior.
5. If a drift class is revealed (distinct from the known migration-fixture family), note it.
6. Baseline the surrounding Scheduling/Schedules suite; run ruff on any touched files; update the task file; no code change if the fix is already shipped and verified.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Current state at HEAD (`task/16000-burn` @ `f857f0dc5`, one merge behind `origin/dev`'s `acc3b0193` via the unrelated task-15782 PR):** `Tests/UI/test_schedules_ux_fixes.py` is GREEN — 9/9 passed (`/tmp/.../16000_repro_head.txt`). The fix already shipped on this branch's history as commit `aaf8bd84f` "test: align Schedules sync-bar copy" (2026-08-14 06:28:29 -0700), filed as **task-16252** (assignee `@codex`, marked Done same day) — a concurrent session independently found and fixed this exact red test ~12h after task-16000 was opened (01:10) but before this worktree started. `aaf8bd84f` touched only the test file; no production code changed.

**Root cause (independently re-derived, not just trusted from task-16252's notes): TEST DRIFT via a merge-conflict-resolution artifact, not a production regression.**

- `9dd2374b5` "feat(ui): Lab/Schedules/Logs UX overhaul + chrome honesty (ADR-031)" (2026-08-04 07:59:38) added `test_schedules_ux_fixes.py` (new file, "Tests: 6 new UX regression files") *and*, in the same commit, rewrote `SyncStatusWidget.compose()`/`set_owner_state()` to a short-copy variant: bare `"Server"` / `"Server (no connection)"` label, `"Clear errors"` button with tooltip `"Dismiss the current sync error messages."`. The new test's `test_sync_bar_labels_and_tooltips` asserted exactly that variant — test and production were consistent at authoring time.
- Separately, on `dev`, commits `559b37b7d` "fix(ui): complete destination action tooltips" (2026-07-24) and `403ae2368` "fix(ui): explain destination action outcomes" (2026-07-25) had already given the same widget a *different*, more explanatory copy: `f"Server ({active_server_id or 'unavailable'})"` label with tooltip `"Use the connected server as the Schedules owner."`, and a `"Clear"` button with tooltip `"Clear the latest scheduling sync error."`.
- Merge commit `a481ecd7c` "Merge origin/dev into chore/harness-review-tasks-320-334" (2026-08-04 20:00:11, same day as `9dd2374b5`) resolved the resulting conflict on `sync_status_widget.py` by taking `origin/dev`'s side wholesale for this file (`git diff a481ecd7c^2 a481ecd7c -- .../sync_status_widget.py` is empty; `git diff a481ecd7c^1 a481ecd7c -- ...` shows the ADR-031 branch's short-copy hunk being discarded) — i.e. the July tooltip-copy won, the ADR-031 branch's Aug-4 copy variant lost. The sibling test file had no conflict (new file, clean add) and passed through with `9dd2374b5`'s original assertions untouched. From that merge forward, `test_sync_bar_labels_and_tooltips` pinned UI copy that had already lost the merge and never actually shipped.
- Confirmed production has been stable since: `git blame` on the current label/tooltip lines in `sync_status_widget.py` attributes them to `559b37b7d` (2026-07-24), and no commit between then and HEAD (`06d8ce492`, `f831e55a8`, `9b52eae9a`/TASK-15450 CSS consolidation) touches that copy — only CSS/layout concerns. This rules out a later production regression.
- **Born-red evidence (reproduced independently for this task):** copied the pre-fix test body (`git show 56c283965:Tests/UI/test_schedules_ux_fixes.py`, the version immediately before `aaf8bd84f`) back into place and ran it against *current, unmodified* production code:
  ```
  >           assert server.label == "Server"
  E           AssertionError: assert Content('Server (http://127.0.0.1:8000)') == 'Server'
  E            +  where Content('Server (http://127.0.0.1:8000)') = Button(id='scheduling-owner-server', ...).label
  ```
  (full output: `/tmp/.../16000_repro_prefix_born_red.txt`). This confirms production never rendered the bare `"Server"`/`"Clear errors"` copy the stale test expected — it has rendered the `"Server (<url>)"`/`"Clear"` copy since July. File was restored via `cp` immediately after (`git status --porcelain` on the test file was clean before and after).
- AC#3 ("if a production regression...") does not apply — this is drift in the other direction (test pinned intended-but-never-shipped copy). `aaf8bd84f` correctly re-pinned the test to the current, long-standing production contract; no further code change needed.

**Drift class worth flagging (distinct from the migration-fixture family in 15765/16197):** a same-commit "new test + matching production change" pair is fragile across a *later* merge that pulls trunk into the feature branch and resolves a conflict on the production file by keeping trunk's side — the merge has no way to know a fresh, still-unmerged-to-trunk test elsewhere in the same tree assumed the discarded side. The test (a clean add, no conflict) survives unedited while its target behavior silently loses the merge. Worth a grep sweep for other same-day-introduced UI-copy assertion tests around 2026-08-04's ADR-031 merges if red tests of this shape turn up again, but no further instances were found in scope for this task.

**Verification:** `Tests/UI/test_schedules_ux_fixes.py` 9/9 passed; broader baseline `Tests/UI/test_schedules_ux_fixes.py Tests/UI/test_schedules_workbench.py Tests/Scheduling Tests/QA/test_scheduling_css_tokens.py` — 322/322 passed, 0 failed (`/tmp/.../16000_baseline_scheduling.txt`). No files outside `backlog/tasks/` were modified (`git status --porcelain` clean save for this task file), so no ruff run was needed.

**Files:** only `backlog/tasks/task-16000 - Fix-pre-existing-red-test-test_schedules_ux_fixes-fails-on-dev.md` changed by this task. The actual fix (`Tests/UI/test_schedules_ux_fixes.py`) shipped earlier via task-16252 / commit `aaf8bd84f`, already present in this branch's history.
<!-- SECTION:NOTES:END -->
