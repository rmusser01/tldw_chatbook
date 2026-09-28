---
id: TASK-33030
title: 'CI: derive diagnostic-inventory totals from rows (schema 4)'
status: Done
assignee: []
created_date: '2026-09-27 17:52'
updated_date: '2026-09-27 20:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stop committing the inventory summary totals that conflicted on 102 of 127 two-sided sync merges; spec 2026-09-27-ci-conflicts-and-waste-design.md part A.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 build_inventory emits schema 4 with no summary; totals derived by inventory_summary()
- [x] #2 All four consumer test files (plus the review fixture, deliberately not regenerated) keep the dev baseline red set exactly
- [x] #3 Committed inventory regenerated; only schema_version and summary lines changed
- [x] #4 ADR-029 amendment recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: `build_inventory` now emits schema 4 with no stored `summary` block; totals are derived on read by the new `inventory_summary()` helper in scripts/check_persistent_diagnostic_inventory.py, so the summary counts can never drift from the row data and can never conflict on a two-sided merge. Four consumer test files were updated for the new shape (test_derived_artifact_checkers.py, test_diagnostic_path_privacy.py, test_persistent_diagnostic_inventory.py, test_summarization_diagnostic_privacy.py), plus the review fixture -- the whole-inventory hash fixture in the committed Docs/security/production-diagnostic-inventory.json -- that was deliberately NOT regenerated: only schema_version and the (now-derived) summary lines changed there, and the dev baseline red set was preserved exactly. `test_measure_sync_merge_conflicts.py` is the meter's own new test, not a consumer of the schema-4 shape. Recorded as an amendment to ADR-029 (backlog/decisions/029-local-private-data-boundary.md). Added scripts/measure_sync_merge_conflicts.py to measure the sync-merge conflict rate against real dev history (distinct from "two-sided syncs", the term for the separate 127-sync inventory-both-sides subset used for the counterfactual replay in the design doc's Evidence section). The script has no `--positive` flag -- the feature is `--merges` rejecting values below 1 via a `_positive_int` argparse type. Fix rounds covered git-error handling (raise instead of silently treating a non-0/1 exit as clean), that `--merges` rejection, excluding merges whose second parent isn't on origin/dev's first-parent chain, a `--since` cutoff, and an integration test against a real temporary git repo (not a mock). Baseline measured: 572 sync merges / 297 conflicted (51%), `--merges 300`, origin/dev on 2026-09-27, before this change. Merged as PR #2853 (abe8f7d1f2).
<!-- SECTION:NOTES:END -->
