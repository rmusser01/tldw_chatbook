---
id: TASK-32532
title: >-
  Component-pattern library: catalog, governance, monolith carve-up,
  zero-literal migration
status: Done
assignee:
  - '@codex'
created_date: '2026-09-13 23:51'
updated_date: '2026-09-14 22:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements Docs/superpowers/specs/2026-09-13-component-pattern-library-design.md (ADR-161). Parent task; one subtask per plan task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 9-family catalog + registry live
- [x] #2 Governance tests green and ratchets pinned
- [x] #3 Monolith dissolved under 2,000-line ceiling
- [x] #4 TASK-24451 superseded and closed
- [x] #5 Zero dimension literals in sheets
- [x] #6 Zero Python ad-hoc visual style assignments
- [x] #7 Gallery snapshots dark+light green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR paths: backlog/decisions/161-component-pattern-library.md and backlog/decisions/097-boot-budget-ratchets.md (existing). Reason: finish approved design-system migration and shed generated comment bytes without changing CSS semantics. Retain completed plan steps 1–11; repair Python migration and AST hard-floor enforcement, verify stateful layouts, finish constitution/catalog and task hygiene, then review the full branch. Plan: Docs/superpowers/plans/2026-09-13-component-pattern-library.md. Recovery evidence: .superpowers/sdd/2026-09-13-component-pattern-library/resume-2026-09-14.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the component-pattern library: canonical registry/catalog and dark/light gallery; dead-code deletion and family consolidation; owning-sheet monolith carve-up with 2,000-line ceiling; hard-zero source dimension/hex and scoped Python visual-write guards; TASK-24451 superseded/closed. Recovered steps 1–11 and repaired step 12 syntax bypass, retained-state regressions, mixed shorthand guard and comment-selector semantics. Runtime data exceptions and original property scope are explicit in ADR-161 and design-language §6. Generated comments shed 232,853 boot bytes and banked headroom per ADR-097. Final167 targeted checks, 6 sync/budget checks, independent reviews and native dark/light gallery checks pass. Two pre-existing test/contract mismatches documented, no full-suite claim. All twelve subtasks have evidence-backed notes and Done status. Branch remains available for integration. Implementation commit ae9093a714. Evidence and limits: Docs/superpowers/reports/2026-09-14-component-pattern-library-closeout.md. Existing ADR-161 governs the migration; ADR-097 governs byte paydown. No new ADR required.

Final review follow-up: active components/stats_screen.css was outside the *.tcss-only source inventory. Reopened to migrate its literals and make active build membership part of the guard before final closeout.

Final manifest follow-up complete: active components/stats_screen.css now enters all stylesheet inventories; 43 dimensions tokenized with identical existing values, duplicate section-header defaults consolidated with paired legacy/current computed-geometry proof. Fresh final combined run: 176 passed; independent review approved. Final boot census 609,446 B, limit 634,050 B unchanged. See Docs/superpowers/reports/2026-09-14-component-pattern-library-closeout.md.
<!-- SECTION:NOTES:END -->
