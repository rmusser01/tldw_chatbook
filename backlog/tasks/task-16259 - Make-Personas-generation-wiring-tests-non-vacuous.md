---
id: TASK-16259
title: Make Personas generation wiring tests non-vacuous
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 16:40'
updated_date: '2026-08-14 16:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove geometry-dependent activation from the Personas generation wiring contract and prove every generation path reaches its controller seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generation controls use the wiring suite event helper instead of off-screen click geometry.
- [x] #2 Whole-character no-clobber coverage proves the provider seam actually ran.
- [x] #3 The complete Personas generation wiring module and nearby tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a focused test-harness correction with no production boundary change.

1. Preserve the five deterministic RED failures and identify vacuous passing paths.
2. Route generation activations through the suite's existing event-level helper.
3. Add controller-call evidence to the no-clobber case.
4. Run the full module, nearby tests, Ruff, formatter, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Routed every generation-toolbar activation through the suite's existing event-level `_press` helper, leaving user-entry navigation under the real pilot click path.
- Made the no-clobber scenario assert the whole-character controller was actually invoked, removing its prior false-green path.
- The complete 9-test wiring module, Ruff lint/format, and diff hygiene pass; the surrounding checkpoint had 467 other passing Persona/Notes tests.
<!-- SECTION:NOTES:END -->
