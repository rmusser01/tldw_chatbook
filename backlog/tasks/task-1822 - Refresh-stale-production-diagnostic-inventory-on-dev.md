---
id: TASK-1822
title: Refresh stale production diagnostic inventory on dev
status: Done
assignee:
  - '@codex'
created_date: '2026-08-02 02:15'
updated_date: '2026-08-02 02:35'
labels:
  - testing
  - baseline
  - security
dependencies: []
references:
  - backlog/decisions/029-local-private-data-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the repository architecture gate by reviewing current production diagnostic ownership and persistent-sink topology against ADR-029, then recording the accepted current baseline without changing runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every production diagnostic owner changed since the reviewed baseline is inspected for metadata-only safety under ADR-029.
- [x] #2 Persistent sink topology is verified unchanged or any change is explicitly reviewed and documented.
- [x] #3 The checked production diagnostic inventory exactly matches current dev source.
- [x] #4 The focused diagnostic-inventory architecture tests pass.
- [x] #5 No production runtime behavior changes are introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: This refresh applies the existing metadata-only diagnostic boundary and records current ownership; it makes no new architecture, sink, storage, or security decision.

1. Reproduce the stale-inventory failure and generate a deterministic current inventory for comparison.
2. Review every changed owner entry and the corresponding source diagnostics against ADR-029; confirm persistent-sink topology is unchanged.
3. Replace only the reviewed inventory artifact.
4. Run the focused architecture tests, inventory checker, JSON validation, and diff hygiene.
5. Record verification and review results, check all acceptance criteria, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refreshed the checked production diagnostic inventory from current dev source without changing production runtime code. The audit covered 27 changed owner entries: four added owners, one removed legacy voice-command owner, 45 net additional calls, and digest-only source movement. Each changed entry was traced back to its exact previously reviewed source digest; added and removed diagnostic call segments were inspected against ADR-029. All current changed calls are ordinary Loguru/stdlib diagnostics rejected by the persistent file handler unless emitted through the sole schema-validated metadata owner. The persistent admission marker remains single-owner and the four-file persistent-sink topology is unchanged.

Verification: the inventory, sentinel-matrix, and persistent-boundary suites passed 18/18; JSON parsing confirmed 435 owners, 1,073 TASK-492 calls, 6,700 TASK-494 calls, and four sink files; diff hygiene passed. No Python source changed, so Python lint/format checks are not applicable. A non-gating repository-wide -x exploration proceeded through 2,230 passes and 58 skips without a failure after the original inventory gate cleared, then was manually stopped to avoid an hour-scale unrelated run. Independent review found the generated refresh safe and requested only this task-closeout evidence.

ADR required: no new ADR. ADR-029 remains authoritative. Modified files: Docs/security/production-diagnostic-inventory.json and this task record.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed and refreshed the stale dev production-diagnostic inventory under ADR-029. Current source and the checked baseline now agree, persistent sink topology remains unchanged, and all focused architecture/privacy gates pass without a production runtime change.
<!-- SECTION:FINAL_SUMMARY:END -->
