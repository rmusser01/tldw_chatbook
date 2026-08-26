---
id: TASK-21150
title: 'Wizard/Console review follow-ups from PR #2101'
status: To Do
assignee: []
created_date: '2026-08-26 00:10'
labels:
  - ux
  - console
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred minors and recommendations from the TASK-21139..21149 code review (PR #2101): (a) Summary consent allow-path should schedule the model-catalog refresh in the same session, matching the Console modal's allow behavior; (b) AppearanceStep show-all rebuilds (themes and cards) should re-press the row matching the retained selection; (c) bound the three remaining resolve_for_send awaits outside the send path (continuation replay ~3211, instruction preview ~8066, compaction ~9790 in console_chat_controller.py) — same hang class as UAT H-3; (d) make the composer action-link markup invariant mechanical (escape or assert on the reason literal) instead of comment-enforced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Wizard consent allow triggers the catalog refresh that session
- [ ] #2 Show-all theme/card rebuilds preserve the pressed row
- [ ] #3 No resolve_for_send await can hang unbounded
- [ ] #4 Composer markup safety is enforced by code, not comment
<!-- AC:END -->
