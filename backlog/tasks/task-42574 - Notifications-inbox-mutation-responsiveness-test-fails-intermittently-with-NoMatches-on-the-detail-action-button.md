---
id: TASK-42574
title: 'The notifications inbox mutation-responsiveness test fails intermittently with NoMatches on the detail action button'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-24 12:00'
labels:
  - follow-up
  - ci
  - tests
  - flaky
  - watchlists
dependencies: []
references:
  - Tests/UI/test_notifications_inbox_responsiveness.py
  - backlog/tasks/task-34100.3 - Moonshot-replies-survive-provider-continuation-persistence.md
  - backlog/tasks/task-36931 - Keep-Watchlists-inbox-SQLite-work-off-the-UI-event-loop.md
priority: medium
updated_date: '2026-11-27 08:19'
modified_files:
  - Tests/UI/test_notifications_inbox_responsiveness.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_notifications_inbox_responsiveness.py::test_each_admitted_inbox_mutation_refreshes_after_its_native_commit` fails intermittently, in both parametrisations seen so far. It exists on dev today (origin/dev `52351c4e0b`; the test file was last changed 2026-10-25 in `cd6ceea5e4`) and no open PR fixes or quarantines it (`gh pr list --state open --search "notifications inbox"` returns nothing).

Evidence of recurrence, not reproduced locally:
- CI, 2026-11-23: the `[dismiss]` case failed once in UI Fast Lane (12) on PR #3441, a Roleplay-only PR that does not touch Watchlists, with `NoMatches '#notifications-dismiss-button'`.
- TASK-34100.3's notes record the same test run 3 times on a clean base tree: 18, 18, then 17 passed with 1 failed, the failure being the `[mark-read]` case. That note concluded "flaky on dev itself" but filed no task.
- Local probe on a clean dev worktree (`5e73e8a2bc`, an ancestor of dev `52351c4e0b`; the test file is unchanged between them; config isolated, 2026-11-24): the `[dismiss]` node alone passed 10 of 10 (7.05-10.97 s each). So it needs load or a particular interleaving, as in CI.

Likely cause (read from the test, not proven): `activate()` selects the notification, then waits until the button exists, is enabled, the selected notification is the row, and `screen.query_one(selector, Button) is not old_button` (the replacement check, so the first rebuild is already waited out); it then calls `pilot.pause()` and `screen.query_one(selector, Button).press()`. The race left is a second rebuild of the detail pane (for example a refresh landing after the replacement check) between that wait and the press, which makes `query_one` raise `NoMatches`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The failing interleaving is reproduced (a stress loop, an induced rebuild, or a CI log) and the cause named.
- [x] #2 The test (or the screen, if the rebuild is a product defect) no longer races: the press targets a button that is present at the moment of the call.
- [x] #3 Both parametrisations pass 30 of 30 consecutive runs under CPU load on a clean tree.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the missing action button by holding a real Notifications pane replacement in the mounted test.
2. Wait for the selected row and its live action control, then queue the press in the same event-loop turn. Keep the native commit and rendered-result assertions.
3. Repeat both cases 30 times under local CPU load and verify the UI shard in GitHub Actions.

```text
ADR required: no
ADR path: N/A
Reason: The change synchronizes an existing test with the mounted pane.
```
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The hosted dismiss case tried to read the previous action button while the pane was remounting its children. It failed before selection. Holding the real pane mount reproduced the same missing-button error. [GitHub Actions failure](https://github.com/rmusser01/tldw_chatbook/actions/runs/45947878008/job/140348620140)

The test waits for the new selection and the mounted action, then queues the press in the same event-loop turn. The mark-read and dismiss cases both pass with the mount held, and each passed 30 consecutive runs under CPU load from a clean checkout. The 12 tests in the file and the pinned Ruff gate passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
