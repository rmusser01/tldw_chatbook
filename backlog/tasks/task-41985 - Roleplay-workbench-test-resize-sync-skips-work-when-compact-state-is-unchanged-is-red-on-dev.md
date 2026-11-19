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

**Product or test.** Judged from the diff: the removal looks deliberate, since the commit's purpose was to re-evaluate the action rows and transcript on every sync so navigation controls restore after a row change. So the test pins a property the product no longer has, and is stale as written. What is not settled: `on_resize` still calls this method for every resize event, so the "unchanged state does no work" property is now only half true (pane work skipped, action-row work repeated). Whether the repeated work matters is a product call. I did not measure it. This was judged by reading the diff, not by running a profile.

**CI.** No PR-gate lane runs this test: `scripts/ui_pr_gate_census.txt` lists only `TestPersonaHumanIdentityRemoval` from this file, and nothing under `.github/` names the file. The whole-tree `nightly-deep.yml` run nominally covers it but is killed by its job ceiling with partial results, so dev stayed red for about two and a half months unnoticed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes on dev: either the product again skips the action-row work when the compact and narrow state is unchanged, or the test is corrected to the behaviour dev now intends, with the reason recorded in the task notes
- [ ] #2 The behaviour the test pins is stated and still holds: after a resize that leaves the compact and narrow state unchanged, the workbench panes are not restyled
- [ ] #3 The action-row and transcript compact state still updates when navigation controls change (the behaviour `07f5138ee5` added) and a test shows it
- [ ] #4 If the test is kept, it runs somewhere a regression turns a check red (a PR-gate census entry, or a recorded reason it stays out)
<!-- AC:END -->
