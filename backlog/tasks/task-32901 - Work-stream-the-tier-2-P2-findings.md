---
id: TASK-32901
title: "Work stream: the tier-2 P2 findings"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 115 P2 findings, grouped by package family rather than by defect class so each PR stays reviewable
and one reviewer can hold the whole diff. Every member was re-validated against `origin/dev d0face3ebe`
before filing; the per-finding verdict, current file:line and the literal command that proves it are in
`qa/tier2-code-review-2026-09-21/validation/`.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every P2 is either fixed, or closed with a recorded reason
- [ ] #2 No P2 PR mixes package families
- [ ] #3 Findings the validation pass marked WRONG are closed as such, not fixed
<!-- AC:END -->
