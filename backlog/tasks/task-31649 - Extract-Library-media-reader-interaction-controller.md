---
id: TASK-31649
title: Extract Library media reader interaction controller
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 16:55'
updated_date: '2026-09-05 16:56'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move media reading interactions and their transient state into a cohesive controller, restoring Library size and method ratchets while preserving existing Reader behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Content search, reading position, display memoization and read-later behavior preserve existing contracts.
- [ ] #2 Controller dependencies are explicit and late-bound and DOM identities remain unchanged.
- [ ] #3 Targeted Reader characterization and existing unchanged screen size and method ceilings pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run existing Reader search, progress, memoization and read-later characterization before extraction.
2. Extract one Reader interaction controller owning its search/progress/memo state; retain DOM structure and explicit screen callbacks.
3. Remove proven-obsolete private delegators and use exact per-field forwarding declarations for transitional state, mirroring the existing Console descriptor.
4. Verify targeted Reader/media/import tests, new controller ports, unchanged architecture ceilings, Ruff/format and diff checks.
ADR required: no
ADR path: N/A
Reason: Direct application of approved screen decomposition design and DESIGN.md section 7; state forwarding mirrors the existing Console convention.
<!-- SECTION:PLAN:END -->
