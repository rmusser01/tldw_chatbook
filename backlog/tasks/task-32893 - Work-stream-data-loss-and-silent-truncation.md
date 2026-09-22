---
id: TASK-32893
title: "Work stream: data loss and silent truncation"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-data-loss
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Six paths that destroy or silently drop user data while reporting success: a rechunk that hard-DELETEs
every chunk and reports `"rechunked"` when the replacement is empty, an extraction refusal swallowed
because `"interrupted"` is missing from the failure-reason set, an `expected_version` parameter accepted
and never threaded through, two voice-profile stores that cannot tell "absent" from "unreadable", and
three silent export truncations.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No path deletes user rows before its replacement is known non-empty
- [ ] #2 Every swallowed failure in the children either surfaces or is logged with its reason
- [ ] #3 `expected_version` is either honoured or removed from the signature
- [ ] #4 Each child has a test that reproduces the loss before the fix
<!-- AC:END -->
