---
id: TASK-36942
title: 'Work stream: architecture guardrails that stop silent structural decay'
status: To Do
created_date: 2026-10-26 01:06
labels:
- architecture-review-2026-10-25
- architecture
- tests
priority: high
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Import-time cycles are nearly gone, but function-level imports hide a 1,217-module dependency cycle, nothing checks layer direction globally, 40 modules of 4,000+ lines have no size ratchet row, the 'pure' agent runtime loads Textual and the DB at import, and string lookups on the app object grow unchecked. These guards are cheap and make every later structural task measurable.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F1, F5, F6, F16; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Layer direction, app string lookups, module size and agent_runtime import purity are each enforced by an architecture test that runs in the existing Architecture lane
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
