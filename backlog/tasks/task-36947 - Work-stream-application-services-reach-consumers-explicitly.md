---
id: TASK-36947
title: 'Work stream: application services reach consumers explicitly'
status: To Do
created_date: 2026-10-26 01:09
labels:
- architecture-review-2026-10-25
- architecture
priority: high
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`TldwCli` holds 397 attributes, about 199 of them services, and code in every layer reaches them by string or by taking the whole app: 352 non-UI functions take the app, the Console controller holds a settable app reference, and `ConsoleRuntime` reads a screen's private projection. MCP's server, the mobile web service and config each compose their own databases.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F1; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every subtask of this stream is Done
- [ ] #2 The app string-lookup ratchet from TASK-36942.4 is lower than at filing
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
