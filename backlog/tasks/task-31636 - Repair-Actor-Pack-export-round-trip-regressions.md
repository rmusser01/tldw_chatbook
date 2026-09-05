---
id: TASK-31636
title: Repair Actor Pack export round-trip regressions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 08:51'
updated_date: '2026-09-05 08:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore Actor Pack export/import activation behavior on the latest dev branch so portable Persona identity round trips remain valid and the subsystem test suite is green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create New activation preserves an incoming Persona portable UUID.
- [x] #2 Actor Pack export and activation preserve valid Persona policy rules and reject malformed rules without weakening canonical validation.
- [x] #3 Deterministic export goldens match the current intentional producer version.
- [x] #4 The complete Actor Pack test directory passes on the latest dev branch.
- [x] #5 Scoped static checks and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce and classify all current Actor Pack failures on the rebased latest dev branch.
2. Trace the failing Persona payload through activation, snapshot capture, and canonical validation to identify the contract drift.
3. Add or adjust focused regression coverage and implement the smallest production correction consistent with ADR-074 and ADR-079.
4. Run the complete Actor Pack suite and scoped static checks, then document evidence.

ADR required: no

ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md and backlog/decisions/079-workspace-assistant-defaults.md

Reason: ADR-074 already defines portable identity preservation, Persona JSON authority, canonical archive validation, and activation semantics; ADR-079 requires identity-scoped Persona policy rules to travel through Actor Packs. This is a regression repair within those existing boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extended the pure Actor Pack Persona contract with ADR-079 policy rules, including strict nested validation for rule kind, name, booleans, call caps, and unknown fields. Added contract and activation/export round-trip coverage proving valid rules travel with a Persona while malformed rules fail closed. Regenerated the reviewed minimal deterministic archives after the intentional package producer version changed from 0.1.8.0 to 0.1.8.1.

Verification: the focused regression selection passed 13 tests; the final complete `Tests/Actor_Packs` run passed 216 tests. Scoped Ruff lint and format checks and `git diff --check` passed. Pytest emitted known cleanup warnings from immutable-publication security tests after the passing result; they did not change the exit status or test outcome.

ADR required: no. ADR-074 and ADR-079 already govern the repaired portability and validation behavior. No new architecture or generalizable testing lesson was introduced.
<!-- SECTION:NOTES:END -->
