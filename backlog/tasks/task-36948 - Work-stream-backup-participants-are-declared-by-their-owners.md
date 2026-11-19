---
id: TASK-36948
title: 'Work stream: backup participants are declared by their owners'
status: To Do
created_date: 2026-10-26 01:10
labels:
- architecture-review-2026-10-25
- backup
- architecture
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backup_Recovery's participant contract is clean, but registration is hard-coded: 24 exact types, two closers tables that disagree (11 entries versus 19 keys, 15 of them with a closer), a policy if-chain, a type table rebuilt at least three times per connection, and an unsaved-editor probe that reads ChatScreen private fields. 66 production modules import participants, many using private names.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F10; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
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
