---
id: TASK-32902
title: "Work stream: the tier-2 P3 polish"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 108 P3 findings. Lowest priority in the review and explicitly the last thing to burn down; several
are one-line corrections (`domain.replace("www.", "")` strips `www.` anywhere in the netloc, so
`newww.example.com` becomes `neexample.com`), several are `re.compile` inside a per-item loop, and a
large share are diagnostics that reach no sink.

Do not let this stream block the others, and do not open it before the P0/P1 streams are closed.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every P3 is either fixed or closed with a recorded reason
- [ ] #2 No P3 change alters behaviour beyond the finding it closes
<!-- AC:END -->
