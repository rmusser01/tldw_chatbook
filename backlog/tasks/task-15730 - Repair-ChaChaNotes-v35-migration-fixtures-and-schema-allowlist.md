---
id: TASK-15730
title: Repair ChaChaNotes v35 migration fixtures and schema allowlist
status: Done
assignee: []
created_date: '2026-08-13 09:08'
updated_date: '2026-08-13 09:20'
labels:
  - database
  - migrations
  - tests
dependencies: []
references:
  - >-
    Docs/superpowers/qa/audio-cpp-clone-voice-bundle-portability-2026-08-11/baseline-red-nodes.txt
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore deterministic ChaChaNotes schema and migration coverage after the production schema advanced to v35, without weakening migration validation or hiding unrelated environment failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The 40 preserved ChaChaNotes/schema baseline-red nodes pass on the latest dev without skips, xfails, or vanished collection.
- [x] #2 Fresh-schema assertions describe v35 and validate the v34/v35 artifacts that production actually owns.
- [x] #3 Historical migration fixtures include the columns/tables required by their declared versions, remove post-version columns that collide with replay, and upgrade through the production migration chain.
- [x] #4 Malformed historical schemas remain rejected; production migration validation is not relaxed to accommodate invalid fixtures.
- [x] #5 The chachanotes SQL validation allowlist exactly covers the live v35 substantive table inventory, including the three console auxiliary/context tables.
- [x] #6 Focused ChaChaNotes and SQL-validation suites plus Ruff, formatting, mypy where applicable, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this is a bounded test-fixture and validation-allowlist repair aligned to the already-shipped ChaChaNotes v35 schema; it changes no storage architecture or migration policy.

1. Reproduce and classify the preserved 40-node deterministic cluster on current dev.
2. Correct stale latest-version assertions and repair historical fixtures to match their declared schemas.
3. Update the central chachanotes table allowlist to match the live v35 inventory.
4. Run the exact node set, affected DB suites, static checks, and mutation checks.
5. Review, document the evidence, close the task, and open a focused follow-up PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Repaired the preserved 40-node ChaChaNotes v35 failure cluster without changing production migration policy.
- Ratcheted fresh/current schema assertions to v35, completed deliberately minimal historical fixtures with required notes/conversation/message fields, and removed the v34 compaction column before replaying older migration chains.
- Review follow-up completed the v24/v28 `notes` fixtures with the v4 bookkeeping and v5 sync columns, added an executable column-contract ratchet, and routed the reviewed repository read through `transaction()`.
- Added the three live v33 Console context tables to the hand-maintained ChaChaNotes SQL allowlist.
- Verification: exact preserved nodes 40 passed; affected plus dedicated v34/v35 suites 240 passed; Ruff check and format, scoped mypy, and git diff check passed.
- ADR check: no ADR required because this aligns fixtures and an existing validation inventory to the shipped v35 schema; no storage boundary or migration policy changed.
<!-- SECTION:NOTES:END -->
