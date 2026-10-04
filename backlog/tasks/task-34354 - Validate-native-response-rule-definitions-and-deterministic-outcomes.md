---
id: TASK-34354
title: Validate native response rule definitions and deterministic outcomes
status: Done
assignee:
  - '@codex'
created_date: '2026-10-04 05:53'
updated_date: '2026-10-04 06:04'
labels: []
dependencies: []
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 1. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Closed native definitions reject unsupported fields and enforce all size limits.
- [x] #2 Chat and Workspace exclusions mask logical rules while revisions remain pinned.
- [x] #3 Literal and Markdown heading checks distinguish violations, unavailable checks and inapplicable rules.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/219-console-learned-response-rules.md
Reason: direct implementation of accepted ADR-219.
Follow implementation plan Task 1: failing boundary/scope/check tests, immutable models and scope resolution, deterministic evaluation, targeted verification and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented closed immutable rule definitions, scope precedence and deterministic literal/Markdown-heading checks. Unknown and skipped results remain distinct from pass; private input bodies stay out of repr. Added rule package and three test files plus fixtures. ADR-219 governs the change. Watched the missing boundary fail; 43 feature cases and 36 command regression cases pass; black py312, ruff and mypy pass. Self-review completed.
<!-- SECTION:NOTES:END -->
