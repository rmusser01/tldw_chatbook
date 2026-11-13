---
id: TASK-40750
title: Reduce repeated parsing in component governance CI
status: Done
created_date: 2026-11-13 18:52
assignee:
- '@codex'
updated_date: 2026-11-13 19:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce avoidable work in the UI PR gate while preserving the complete canonical stylesheet ownership check and its diagnostics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canonical stylesheet ownership still rejects misplaced bare and pseudo selectors while allowing scoped composition.
- [x] #2 Each source is parsed once per ownership check, with measured before-and-after runtime on identical inputs.
- [x] #3 Targeted governance checks pass and the CI cost analysis identifies the next measured opportunities without changing runner deployment or test coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/161-component-pattern-library.md
Reason: Test-only optimization preserving the existing ADR-161 contract and CI boundaries.
1. Read hosted UI duration rows and profile canonical ownership checking on current dev.
2. Add a small correctness/work-budget regression and eliminate repeated stylesheet parsing without persistent caching.
3. Run targeted checks and paired timings, review the diff, and record evidence and remaining CI-cost opportunities.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The canonical stylesheet ownership guard now scans each source once per invocation instead of once per canonical class. It retains the same complete source set, selector rules, class/source diagnostic order, and fresh reads on every invocation. No cross-test cache, app reuse, test removal, timeout increase, or runner/workflow change was introduced.

ADR required: no. Existing contract: backlog/decisions/161-component-pattern-library.md. This is a test-only optimization of that contract.

## Scope and verification

- Tests/UI/test_component_pattern_governance.py: materialize each source's definitions once; add a regression that accepts scoped composition, rejects misplaced bare and pseudo selectors, observes changed content, and checks the parse budget. A pre-existing assert was formatted by current Ruff rules.
- scripts/ui_pr_gate_census.txt: append the regression. All 2,029 existing entries retain their exact positions; with the current 36-shard matrix the new case goes to shard 14.
- scripts/check_ui_pr_gate_census.py: raise the coverage floor from 2,029 to 2,030. An existing zip now spells strict=False explicitly, preserving its behavior while satisfying Ruff B905.
- The new regression failed before the optimization because two sources were parsed four times instead of twice. It passed after the change, including checking changed source text on a second invocation.
- The first census-contract run caught the missing floor increment (103 passed, one failed); after updating the floor all 104 census/shard checks passed.
- Final combined verification: 106 targeted tests passed in 34.32 s. Ruff check/format and git diff --check pass. No full-suite run or hosted workflow was dispatched. No production/runtime behavior or dependency changed.

## Measured improvement

Base: 1455ceeb96 (origin/dev when the isolated worktree was created). Local Python 3.12.11, Textual 8.2.8, pytest 8.4.2; shared development dependencies, plugin autoload disabled, asyncio/timeout plugins enabled. This host had concurrent unrelated work, so these are a local comparison, not a hosted p95 prediction.

Unprofiled targeted pytest comparison:
- Existing ownership case: 42.89 s -> 8.90 s call duration (79% reduction).
- Added regression: 5.91 s call duration, leaving about 28 s net savings in this comparison.

Separate cProfile measurements run directly inside the selected private-profile child:
- Ownership body: 38.987 s -> 5.106 s.
- CSS scanner: 22,638 calls / 33.958 s -> 231 calls / 0.327 s.
- Exactly the same 231 sources and 98 canonical classes are checked. The regression also checks that source changes are not hidden by a persistent cache.

Reproduce the functional run using the project's Python environment:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Tests/UI/test_component_pattern_governance.py::test_canonical_classes_defined_only_in_owning_sheet Tests/UI/test_component_pattern_governance.py::test_ownership_scan_parses_sources_once_and_observes_changes Tests/CI/test_ui_pr_gate_census_checker.py Tests/CI/test_ui_lane_shard_count_statements.py -p pytest_asyncio.plugin -p pytest_timeout -p no:cacheprovider --timeout=180 -q`

For direct-child profiling set TLDW_TEST_PRIVATE_PROFILE_NODE to the ownership test's exact node ID and run the same single node with `python -m cProfile -o /tmp/ownership.prof -m pytest ...`. Treat profiled and ordinary timings separately.

## CI cost evidence and next investigation

Source: https://github.com/rmusser01/tldw_chatbook/actions/runs/43398660691 (2026-11-13, first attempt). Downloaded the existing run's logs, inspected all 36 UI job logs, and summed the complete pytest duration rows. This is one run, not a population baseline. Its UI shards passed; the workflow failed in another lane.

- UI jobs occupied about 449 runner-minutes.
- Reported test-body phases: 324.10 min; setup: 2.80 min; teardown: 58.51 min. The remaining roughly 64 min includes job setup, collection and unreported overhead; it is not all test startup.
- By test filename prefix, body time was Settings 113.04 min, Console 73.94 min, Library 60.35 min. These categories overlap the other views below and must not be summed with them.
- Slowest files by summed setup/call/teardown: test_library_file_notes_workspace.py 17.34 min; test_library_adaptive_reader_closeout.py 11.05; test_destination_visual_parity_correction.py 8.63; test_settings_category_sweep.py 7.51; test_settings_pending_save_drafts.py 6.86.
- 2,463 cases with bodies below 0.1 s paid 19.39 min of teardown. Investigate the cleanup profile before changing it: TASK-1468 documents actual cross-test app-cycle failures when per-test collection was removed. No cleanup was weakened here.
- Matching logged node names to explicit private_profile_test decorators in this worktree found 1,129 cases accounting for 226.28 min of body-phase time. This includes child startup, execution and shutdown; it does not attribute all of that time to startup. The decorator mapping uses current dev, not the historical CI tree. One second saved per matched child would save about 18.8 runner-minutes, a sensitivity estimate only.

Next priority: instrument a few representative Settings/private-profile children to separate interpreter/import/collection, app startup, body, and shutdown costs. Preserve fresh process/profile isolation. Then profile cleanup on a representative mixed UI shard and target repeated work without weakening native-worker retirement or garbage-collection guarantees. Rebalance shards only after reducing their work; reordering alone does not reduce runner-minutes.

The rollout/deployment session owns runner changes. This work neither deploys runners nor changes branch protection, the merge queue, matrix size, or workflow YAML. No new general lesson beyond the already documented profiling and isolation rules was needed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
