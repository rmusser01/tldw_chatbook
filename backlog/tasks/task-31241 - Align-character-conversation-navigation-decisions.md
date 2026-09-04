---
id: TASK-31241
title: Align character conversation navigation decisions
status: To Do
assignee: []
created_date: '2026-09-04 02:03'
updated_date: '2026-09-04 02:04'
labels:
  - architecture
  - console
  - roleplay
  - search
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Renumbering provenance

Renumbered from TASK-31233 on 2026-09-04. The final pre-commit worktree sweep
found the older `Review selected opens the review it creates` task created at
01:50; it keeps TASK-31233 under the older-arrival rule. This unshipped task was
created at 02:03 and moves with all plan and dependency references.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the architectural source of truth for the approved character-conversation navigation programme before any implementation begins, so all eight pull requests share one local-only identity, activation, recovery, and indexing contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A new collision-safe ADR records local-only character conversation navigation, typed identity, activation, repair, and semantic-index ownership.
- [ ] #2 ADR-004, ADR-030, ADR-037, ADR-046, ADR-083, and ADR-085 are amended consistently with the approved design.
- [ ] #3 ADR-031 and ADR-033 remain preserved contracts and are linked without contradictory changes.
- [ ] #4 Terminology distinguishes Data Profile authority from RAG configuration profiles in every changed decision.
- [ ] #5 Documentation-only validation and review pass with no implementation behavior change.
<!-- AC:END -->
