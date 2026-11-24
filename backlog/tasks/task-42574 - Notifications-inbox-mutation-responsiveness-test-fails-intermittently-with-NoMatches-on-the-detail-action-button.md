---
id: TASK-42574
title: "The notifications inbox mutation-responsiveness test fails intermittently with NoMatches on the detail action button"
status: To Do
assignee: []
created_date: 2026-11-24 12:00
labels:
- follow-up
- ci
- tests
- flaky
- watchlists
priority: medium
references:
- Tests/UI/test_notifications_inbox_responsiveness.py
- backlog/tasks/task-34100.3 - Moonshot-replies-survive-provider-continuation-persistence.md
- backlog/tasks/task-36931 - Keep-Watchlists-inbox-SQLite-work-off-the-UI-event-loop.md
dependencies: []
---

## Description

`Tests/UI/test_notifications_inbox_responsiveness.py::test_each_admitted_inbox_mutation_refreshes_after_its_native_commit` fails intermittently, in both parametrisations seen so far. It exists on dev today (origin/dev `5e73e8a2bc`; the test file was last changed 2026-10-25 in `cd6ceea5e4`) and no open PR fixes or quarantines it (`gh pr list --state open --search "notifications inbox"` returns nothing).

Evidence of recurrence, not reproduced locally:
- CI, 2026-11-23: the `[dismiss]` case failed once in UI Fast Lane (12) on PR #3441, a Roleplay-only PR that does not touch Watchlists, with `NoMatches '#notifications-dismiss-button'`.
- TASK-34100.3's notes record the same test run 3 times on a clean base tree: 18, 18, then 17 passed with 1 failed, the failure being the `[mark-read]` case. That note concluded "flaky on dev itself" but filed no task.
- Local probe on a clean dev worktree (`5e73e8a2bc`, config isolated, 2026-11-24): the `[dismiss]` node alone passed 10 of 10 (7.05-10.97 s each). So it needs load or a particular interleaving, as in CI.

Likely cause (read from the test, not proven): `activate()` waits for the button to exist and be enabled, then calls `pilot.pause()` and then `screen.query_one(selector, Button).press()`; if the detail pane rebuilds between the wait and the press, `query_one` raises `NoMatches`.

## Acceptance Criteria

- [ ] The failing interleaving is reproduced (a stress loop, an induced rebuild, or a CI log) and the cause named.
- [ ] The test (or the screen, if the rebuild is a product defect) no longer races: the press targets a button that is present at the moment of the call.
- [ ] Both parametrisations pass 30 of 30 consecutive runs under CPU load on a clean tree.
