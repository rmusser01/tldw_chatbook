---
id: TASK-31930
title: Ignore late screen rebuild notifications after app stack teardown
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 06:19'
updated_date: '2026-09-06 15:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Queued BaseAppScreen ContentsRebuilt events can arrive after app teardown empties screen_stack. The handler accesses self.screen before the existing overlay scheduler empty-stack guard, raising ScreenStackError. The intermittent inventory failure is deterministically reproduced by calling the real handler with an empty stack. Await bounded production guard approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A late ContentsRebuilt notification with an empty screen stack does not raise or schedule presentation work.
- [x] #2 An active matching screen still schedules reconciliation, while stale nonmatching screen notifications remain ignored.
- [x] #3 The exact negative regression fails before the guard and passes afterward, with relevant app lifecycle tests verified.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow approved-console-regressions plan TASK31930: RED actual handler on empty stack, guard before current-screen access, verify empty/matching/stale cases and complete buddy/parallel-run files. ADR required: no; routine late-event lifecycle guard.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved late-event repair: short-circuit an empty screen stack before reading self.screen, retaining matching-screen reconciliation and stale-screen rejection. Real empty-stack RED was 2 failed / 2 passed; complete buddy and parallel-run files passed 65. Final buddy/parallel/live-handoff selection passed 134 in 146.97s with no Darwin F_GETPATH retained SQLite lines (/private/tmp/tldw-approved-buddy-handoffs-final.xml and .log). Independent spec/code review and scoped static checks pass; existing app.py lint debt is unchanged. ADR required: no, routine lifecycle guard. Modified app.py and test_persona_buddy_app_mount.py; full repository sweep and screen-size paydown remain outside this repair.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31824 was renumbered to TASK-31930 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
