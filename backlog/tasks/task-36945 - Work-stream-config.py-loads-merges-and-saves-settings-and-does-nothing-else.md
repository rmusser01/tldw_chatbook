---
id: TASK-36945
title: 'Work stream: config.py loads, merges and saves settings and does nothing else'
status: To Do
created_date: 2026-10-26 01:08
labels:
- architecture-review-2026-10-25
- config
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`config.py` is 12,258 lines: a 2,584-line inline TOML template, a 1,672-line loader, encryption, canvas and Console capture policy, and three lazily-filled DB singletons that consumers depend on someone else having initialized. Settings are read from 356 call sites, several at import time where hot reload can never reach them.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F9; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every subtask of this stream is Done
- [ ] #2 config.py has a size ratchet row that this stream lowers
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
