---
id: TASK-36952
title: 'Work stream: Library screen Phase A2 — controllers without the screen'
status: To Do
created_date: 2026-10-26 01:13
labels:
- architecture-review-2026-10-25
- library
- architecture
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`library_screen.py` is 37,542 lines (the class 36,411) with 1,354 methods. That is 5,312 file lines and 96 methods more than when the decomposition program closed at 32,230 lines. Its `__init__` builds 21 controllers with 413 lambdas, the media and notes controllers take the screen plus 92 and 95 lambdas, and tests pin method bodies to the class (370 unbound calls in 46 files; 66 files reach controllers through the screen). Related: TASK-32463, TASK-35309.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F2; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The media subsystem's controller is constructed without the screen and tested directly
- [ ] #2 The lambda and unbound-call ratchets are lower than at filing
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
