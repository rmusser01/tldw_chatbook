---
id: TASK-36953
title: 'Work stream: module boundaries for the server client and misfiled packages'
status: Done
created_date: 2026-10-26 01:14
updated_date: '2026-11-20 14:55'
labels:
- architecture-review-2026-10-25
- architecture
priority: low
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The tldw_server client is one class with 1,220 methods, of which a typical server service uses 13. Splash animations make up 96 of Utils' 161 files, and Utils' input validation imports from Agents, Chat and Terminal. Interop helper consolidation is owned by TASK-32898.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F14, F15; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every subtask of this stream is Done
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed 2026-11-20 on dev `d85e90bc9e`. AC #1: all three subtasks (TASK-36953.1 to .3) are Done.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
