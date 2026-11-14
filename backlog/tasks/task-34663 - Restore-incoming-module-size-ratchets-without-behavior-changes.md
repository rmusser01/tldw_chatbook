---
id: TASK-34663
title: Restore incoming module size ratchets without behavior changes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-08 16:29'
updated_date: '2026-10-08 22:27'
labels: []
dependencies: []
priority: high
type: chore
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the nine inherited size-ratchet failures retained during SQLite ownership verification by putting cohesive pure helpers and data contracts beside their existing owners. Do not raise limits or change runtime policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All nine originally failing module-size rows pass at unchanged or lower caps.
- [x] #2 Moved helpers and data contracts preserve original implementations, exports and actual caller behavior with targeted tests.
- [x] #3 No production admission, ownership, deadline, validation, UI geometry or error handling is weakened.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Mechanical extraction preserves existing domain, runtime and public contracts; existing ADR036/094/220/225 ownership rules remain.
1. Inspect all moved helpers, dependencies, imports, exports and monkeypatch routes; retain nine original size REDs.
2. Move only cohesive pure code or data contracts to existing domain helpers or narrow leaf modules; keep existing import aliases and required live bindings.
3. Prove AST/code identity for moved definitions, run original affected helper/caller tests and size/static guards.
4. Lower measured ratchet rows, record actual outcomes and retained baseline limitations; self-review and commit locally.

5. Verify cold-import and warm-ready budgets. Move the exact workspace-source ID constant to the existing stdlib limits owner and preserve its schema/client exports, avoiding a new eager schema graph from API utils. Do not change any bound or validation.

2026-10-08 warm-ready native shutdown is repaired but actual census is 1038 against unchanged 1033 cap. Consolidate pure request projections into the existing prepared-request owner and store dataclasses into existing Console-native models, move the pure fence scan into existing selection helpers, and defer Collections reference-only imports to their actual resolution paths. Keep constructor borrower capture and shutdown imports off first-paint through existing app composition boundaries; preserve original code/exports and test exact mechanical identity plus unchanged budgets. No new budget exception or semantic policy change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Mechanical extraction reuses existing domain owners: 48 moved and 449 retained definitions match reviewed incoming source; public and monkeypatch aliases preserved. All 52 size rows pass without raised caps; original store/generation 441, modules 72, import/API 114, closure 12, fence/selection 26 and Collections reference 23 pass. Actual warm-ready/cold startup nine passed with 1033/1033 and 681/686, expected census warnings retained. No new aggregate Ruff code/message finding, 12 removed; inherited whole-file debt not called clean. Artifact guards pass after legitimate unchanged Mermaid network/pin verification. QA publishes outcomes/hashes only. Genuine closed-chat Send is unresolved; two unused untracked generated helpers excluded after blocked cleanup. Status remains In Progress pending final affected verification/review; no full sweep or integration claim.

Final complete controller outcome remains 362 passed/one original closed-chat failure; do not credit all affected callers green. Exact mechanical proof, preserved exported bindings and unchanged validation/ownership/deadline boundaries support AC3; AC2 remains unchecked pending full affected verification/review. Both Backlog guards and whitespace pass. No new source policy or budget change.

Actual moved-helper caller verification now includes complete controller/per-session/new controls: 385 passed with no pytest warnings after the separately approved the separately approved closed-chat fix closed-chat correction. Previous 48-moved/449-retained mechanical identity and store/generation/API/closure/selection/Collections controls remain unchanged. All 52 size rows pass; the separately approved closed-chat fix further lowers controller cap to 29266. All eleven guards pass; original unfinished caller receipts remain historical. The two unused generated files were removed only after complete green evidence, with recoverable private copies. Remain In Progress for final review/integration and broader qualification; no parent Done or remote publication.

Final whole-branch read-only review at c37690530e confirms authored source/caller coverage, with no Critical or confirmed Important production issue. Deferred Minor: mechanical-move-check.py reads current definitions against fixed8f057657 and later approved closed-Send edits necessarily break retained whole-class equality. The retained successful extraction proof is historical, not claimed reproducible at final HEAD; do not weaken it by omitting the controller. Source385/size52 pass after approved closed-Send fix; generated unused scaffolds were subsequently removed with explicit authorization and recoverable local evidence. In Progress pending integration; inherited debt and broader qualification unwaived.
<!-- SECTION:NOTES:END -->
