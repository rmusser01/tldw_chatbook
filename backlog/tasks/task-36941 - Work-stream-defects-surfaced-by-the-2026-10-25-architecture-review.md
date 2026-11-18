---
id: TASK-36941
title: 'Work stream: defects surfaced by the 2026-10-25 architecture review'
status: To Do
created_date: 2026-10-26 01:05
labels:
- architecture-review-2026-10-25
- defect
priority: high
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The architecture review reproduced four concrete defects while measuring structure: a transaction manager that can later commit writes from an aborted block, an Interop service that always falls back to the server client's private transport, provider credential resolvers that disagree on precedence, and a dead Textual-importing Chat module. Each is small and independent of the structural work.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F11, F13, F14, F17; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every subtask of this stream is Done or explicitly closed with a recorded reason
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
