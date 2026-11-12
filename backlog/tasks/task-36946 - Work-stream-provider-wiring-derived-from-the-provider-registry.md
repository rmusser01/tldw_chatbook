---
id: TASK-36946
title: 'Work stream: provider wiring derived from the provider registry'
status: To Do
created_date: 2026-10-26 01:09
labels:
- architecture-review-2026-10-25
- providers
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-179 promises that a new engine preset costs one registry record plus one dispatch line. The newest preset needed 13 edit sites in 9 files, about 8 of them repeating `ProviderRecord` data; TASK-33510 and TASK-33511 were bugs of exactly this drift. Credential precedence is handled in TASK-36941.3.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F11; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Adding an engine preset touches only its registry record and its config template section, demonstrated in the final subtask's notes
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
