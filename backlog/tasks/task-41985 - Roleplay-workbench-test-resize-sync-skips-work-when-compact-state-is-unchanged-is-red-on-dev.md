---
id: TASK-41985
title: 'Roleplay workbench test test_resize_sync_skips_work_when_compact_state_is_unchanged is red on dev'
status: To Do
assignee: []
created_date: '2026-11-19 13:30'
labels:
  - roleplay
  - tests
  - personas
dependencies: []
references:
  - Tests/UI/test_personas_workbench.py
  - tldw_chatbook/UI/Screens/personas_screen.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_personas_workbench.py::TestWorkbenchShell::test_resize_sync_skips_work_when_compact_state_is_unchanged` fails on dev today (`a6df4d8d48`) with `AssertionError: unchanged compact state should not query panes`. This task's PR is the fix; no open PR fixes it (the three open PRs that touch the file, #3431, #3441 and #3518, do not touch this test).

**When it broke.** It passed at `121e5818aa` (2026-08-21) and failed at `461668df00` (2026-09-25). A `git bisect run` over that range (about 1,860 commits, with commits that could not run skipped, a pass/fail probe per step under mounted-Roleplay config isolation) names `07f5138ee5` "fix: revoke stale repairs and restore character navigation controls" (2026-09-05, merged in PR #2446, `31ce681e71`) as the first bad commit.

**What that commit changed.** In `PersonasScreen._sync_responsive_workbench` it removed the opening `if self._workbench_compact == compact and self._workbench_narrow == narrow: return`. The conversation-actions block (`query_one("#personas-conversation-actions")`, the navigation row, every action button, the transcript `set_compact`) now runs on every call, and the unchanged-state early return was moved below that block, so it only skips the pane and handle restyle. The commit did not touch this test, whose guard raises on any `query_one("#personas-...")` call.

**What the test asserts.** After `app.run_test(size=(80, 40))` mounts the compact workbench, it patches `screen.query_one` to raise on any `#personas-...` selector, then calls `_sync_responsive_workbench()` and expects no pane query.

**CI.** No PR-gate lane runs this test: `scripts/ui_pr_gate_census.txt` lists only `TestPersonaHumanIdentityRemoval` from this file, and nothing under `.github/` names the file. `nightly-deep.yml` has a cron, but `.github/workflows/test.yml` says it "remains disabled (ADR-366)". I did not check any nightly run. By the bisect the test has been red on dev since `07f5138ee5` (2026-09-05).

**Related notes.** TASK-41800 (line ~53) already notes that this test and the buddy-close test "fail on dev too and are on no lane", and TASK-32500 notes this test as pre-existing red. This is the dedicated task for what those notes mention.

**B7.** TASK-33910.12 (B7, To Do) deletes `_sync_responsive_workbench` and `#personas-workbench`; if it lands first, this test pins deleted code and can be retired. This is a note, not a dependency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes on dev, or is corrected to the behaviour dev now intends, or is retired with a recorded reason (including that TASK-33910.12 deleted the code it pins)
- [ ] #2 The behaviour the test pins is stated and still holds: after a resize that leaves the compact and narrow state unchanged, the workbench panes are not restyled
- [ ] #3 The action-row and transcript compact state still updates when navigation controls change (the behaviour `07f5138ee5` added) and a test shows it
- [ ] #4 If the test is kept, it runs somewhere a regression turns a check red (a PR-gate census entry, or a recorded reason it stays out)
<!-- AC:END -->
