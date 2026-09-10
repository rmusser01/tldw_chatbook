---
id: TASK-32161
title: Address Canvas V2 PR review and integration gaps
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 06:07'
updated_date: '2026-09-09 06:59'
labels:
  - canvas
  - review
dependencies:
  - TASK-31941
  - TASK-32160
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve verified review findings and demonstrate that the qualified Canvas Mermaid and SQLite changes remain correct when integrated with current dev.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Backlog CLI initially assigned TASK-32116 on 2026-09-09. The all-ref sweep found
that ID already owned by Persist native goal runs and recoverable launch intent.
Reassigned to TASK-32161 before implementation, above the swept ref/worktree
allocation maximum 32159 and alongside the SQLite collision correction TASK-32160.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Production run-bound Canvas tools deliver exact-profile Mermaid and source-only authoring guidance without broadening run or browser authority.
- [x] #2 Required CI verifies the hash-pinned Mermaid generated artifacts and fails on missing inputs or drift.
- [x] #3 Private SQLite helper source identity, pipe deadlines, dispatch and cleanup have focused behavior coverage in addition to existing real-process coverage.
- [x] #4 Every Qodo finding has an evidenced fix or technical disposition and targeted integration checks and independent review cover the rebased changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR paths: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md and backlog/decisions/125-lock-safe-private-sqlite-validation.md. Reason: implement the existing exact-profile authoring, reproducibility and private-helper contracts without changing runtime bytes or authority. 1. Preserve the reviewed head and rebase the feature-only range onto latest dev. 2. Verify Qodo feedback against production paths; add failing authoring and required-reproduction regressions before minimal fixes. 3. Add focused helper behavior tests and complete API/compatibility documentation. 4. Run scoped static, generated-artifact and Canvas/SQLite/upstream-owner integration gates; obtain scoped independent review and resolve all review threads with evidence. 5. Push with an exact remote lease and merge only after Qodo and current-head required checks are clear; record the actual merged result.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the retained-profile production coordinator seam, complete public authoring docs, direct private-helper unit tests, and required hash-pinned Mermaid reproduction with local preflight parity. Preserved immutable JavaScript ABI under ADR-124 and corrected the Python 3.12 README floor. ADRs124/125 apply; no new ADR. Independent scoped review passed with no findings; all five Qodo threads have evidence replies and are resolved. Qodo exact-head review reports zero bugs. Code head27cf82559a passed required hosted Derived Artifacts run34320576418 including Fast Lane, covering the six locally semaphore-blocked process tests on a healthy runner. Local qualification:1387Canvas passes/2optionalbrowser skips,721SQLite passes/2Windows skips,all7preflightchecks,real public-input6-outputrebuild,final4authoring passes and scoped Ruff/format pass. Three unchanged upstream-owner failures and local SemLock limitations remain documented, not hidden. Implementation/review acceptance is complete; final documentation-head CI and the user-authorized PR merge remain integration steps tracked on PR2537 and in the SDD ledger. Evidence:Docs/superpowers/reviews/2026-09-09-canvas-pr2537-integration.md.
<!-- SECTION:NOTES:END -->
