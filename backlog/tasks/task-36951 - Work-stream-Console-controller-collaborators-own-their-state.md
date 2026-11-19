---
id: TASK-36951
title: 'Work stream: Console controller collaborators own their state'
status: To Do
created_date: 2026-10-26 01:13
labels:
- architecture-review-2026-10-25
- console
- architecture
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleChatController's helper hosts were moved out without their state: `InterruptRoundHost` takes 122 keywords (86 controller-state getters) and mutates controller dicts through them; `ConsoleCompactionPreflight` takes 67 (41 injected module globals); and one agent reply call passes 78 keywords.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F4; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every subtask of this stream is Done
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
