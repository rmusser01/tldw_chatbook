---
id: TASK-31929
title: Consume character Chat handoffs on cached Console resume
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 06:18'
updated_date: '2026-09-06 15:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A character-chat handoff staged while Console is hidden remains pending on return to the cached ChatScreen because only on_mount schedules its consumer. A scratch tracked-resume-timer probe confirms the missing lifecycle path. Production repair awaits design approval; the UAT also exposes a downstream trace provenance failure after this stage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Returning to the same Console consumes and acknowledges one staged Chat handoff and creates exactly the intended character-bound session.
- [x] #2 First mount and ordered saved-chat startup retain their existing behavior; suspending again stops pending resume timers before hidden consumption.
- [x] #3 Focused real navigation regressions and relevant full lifecycle files pass without direct test-side handoff consumption.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow approved-console-regressions plan TASK31929: RED real warm CHAT handoff plus hide-before-timer control, add existing consumer to tracked ordinary-resume timer list, verify full reuse/handoff/UAT. ADR required: no; direct ADR033 handoff behavior under existing TASK31520 cached-screen lifecycle.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved ADR033 warm-return repair: register the existing CHAT claimant in ordinary-resume tracked timers; ordered startup remains unchanged and suspension stops delivery. Real tests preserve an existing conversation, verify one character session plus greeting and exact settled revision, no-new handoff behavior, same-screen reuse, and hide-before-timer cancellation. Final-fixture negative control suppressing only the new timer fails as expected. Complete handoff/UAT/reuse/ordered-resume selection passed 36 in 83.13s (/private/tmp/tldw-approved-resume-complete.xml); new handoff and complete UAT also passed 8 resource-clean (/private/tmp/tldw-31823-resource-retry.xml). Native handles found in adjacent pre-existing reuse fixtures remain under TASK31927. Independent review acknowledgement gap corrected; scoped lint/format checks pass. No new ADR. Modified chat_screen.py timer list, added test_console_chat_handoff_resume.py, and reused exact-owner cleanup in UAT without changing capture or behavior assertions.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31823 was renumbered to TASK-31929 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
