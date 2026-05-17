---
id: TASK-13.6
title: 'Phase 6.6: Packaging configuration and data-safety validation'
status: Done
assignee: []
created_date: '2026-05-16 00:00'
labels:
  - product-maturity
  - phase-6-release-hardening
dependencies:
  - TASK-13.1
parent_task_id: TASK-13
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate install, configuration, migration, and data-safety paths enough for release-candidate use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 QA walkthrough verifies packaging, configuration, migration, and data-safety release checks from the current repo state.
- [x] #2 Focused regression evidence exists for packaging/configuration/data-safety seams changed by this task.
- [x] #3 Repo-tracked QA evidence exists under Docs/superpowers/qa/product-maturity/phase-6/.
- [x] #4 P0/P1 findings are fixed or explicitly accepted according to the spec severity policy.
<!-- AC:END -->

## Implementation Plan

1. Add a failing Phase 6.6 regression that requires packaging/configuration/migration/data-safety evidence, QA index tracking, roadmap tracking, portable verification commands, and an in-progress task state.
2. Inspect current packaging metadata, setup/config docs, config path isolation, database migration/versioning seams, and destructive/write-safety affordances.
3. Add durable Phase 6.6 QA evidence recording checks, P0/P1 decisions, accepted residuals, and verification output without introducing machine-specific paths.
4. Update the Phase 6 QA index, product-maturity roadmap, and release-hardening guard to reflect TASK-13.6 completion.
5. Mark acceptance criteria complete, add implementation notes, and run focused verification.

## Implementation Notes

Validated Phase 6.6 packaging, configuration, migration, and data-safety release seams from current dev. Added focused regression coverage and repo-tracked evidence, updated the Phase 6 QA index and product-maturity roadmap, and recorded no P0/P1 findings. Packaging build passes, with a P2 residual for future setuptools license metadata cleanup.
