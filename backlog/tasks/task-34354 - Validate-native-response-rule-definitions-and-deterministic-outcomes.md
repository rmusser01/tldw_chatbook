---
id: TASK-34354
title: Validate native response rule definitions and deterministic outcomes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:53'
updated_date: '2026-10-12 03:43'
labels: []
dependencies: []
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/228-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 1. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Closed native definitions reject unsupported fields and enforce all size limits.
- [ ] #2 Chat and Workspace exclusions mask logical rules while revisions remain pinned.
- [ ] #3 Literal and Markdown heading checks distinguish violations, unavailable checks and inapplicable rules.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/228-console-learned-response-rules.md
Reason: direct implementation of accepted ADR-228.
Follow implementation plan Task 1: failing boundary/scope/check tests, immutable models and scope resolution, deterministic evaluation, targeted verification and commit.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented closed immutable rule definitions, scope precedence and deterministic literal/Markdown-heading checks. Unknown and skipped results remain distinct from pass; private input bodies stay out of repr. Added rule package and three test files plus fixtures. ADR-228 governs the change. Watched the missing boundary fail; 43 feature cases and 36 command regression cases pass; black py312, ruff and mypy pass. Self-review completed.
<!-- SECTION:NOTES:END -->
