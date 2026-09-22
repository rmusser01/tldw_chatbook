---
id: TASK-32895
title: "Work stream: untrusted-input bounds"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Five classes of unbounded read on attacker-influenceable input: three image ingresses with no pixel cap
(the guard already exists in eight other modules), plaintext/HTML/MOBI ingestion with no size cap while
audio and video already have one, a voice download whose digest is computed and never compared, a chat
dictionary that `re.compile`s a user pattern without the validator its sibling uses, and six `subprocess`
calls with no timeout.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every image ingress enforces one shared decoded-pixel cap
- [ ] #2 Text ingestion has a size cap symmetric with the existing audio/video caps
- [ ] #3 The computed digest is compared and a byte ceiling is enforced
- [ ] #4 No `subprocess` call in the children runs without a timeout
<!-- AC:END -->
