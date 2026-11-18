---
id: TASK-38697
title: "Widget CSS: BaseAppScreen's scoped block is emitted after its subclass KanbanBoardScreen"
status: Done
assignee:
- '@claude'
created_date: 2026-11-03 11:40
updated_date: '2026-11-04 01:08'
labels:
- follow-up
- css
- tests
priority: low
references:
- Tests/UI/test_widget_css_consolidation.py
- tldw_chatbook/css/widget_defaults_scoped.tcss
- tldw_chatbook/css/build_css.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-36523's checks, PR #3370). Checked on dev 9bbd5c2515.

`Tests/UI/test_widget_css_consolidation.py::test_base_class_blocks_precede_their_subclasses` fails with `[scoped] BaseAppScreen is a base of KanbanBoardScreen but is emitted after it, inverting the tie-breaker Textual gave them`. In `widget_defaults_scoped.tcss`, KanbanBoardScreen's block starts at :43 and BaseAppScreen's at :748. Within one stream an exact specificity tie goes to the later rule, so a base block must come first (the test docstring explains why).

KanbanBoardScreen's scoped block arrived in ab991c7117 (2026-10-31, "Repair Kanban first-use creation and column identity"). The two blocks share no selector today (`BaseAppScreen #screen-content` against `KanbanBoardScreen #kanban-*`), so nothing renders differently yet. The pin exists so that a later shared selector does not silently flip.

**Evidence.** Locally the file errors at setup with the `RecoveryRequired` trap (TASK-38111). The red above was seen with the test run under a temporary `@private_profile_test` marker, since removed. The census lists only `test_class_level_css_stays_within_the_allowlist` from this file, so no PR lane runs this test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_base_class_blocks_precede_their_subclasses passes on dev because the generated scoped stream emits every base block before its subclasses' blocks, and the generated sheets reproduce from build_css.py
- [x] #2 The test runs on a PR lane (a census entry for this node), or the task notes record why it cannot
<!-- AC:END -->

## Implementation Plan

1. Fix at the source: `widget_css.iter_blocks` (shared by the widget and screen builds) sorts blocks by module path and line. After that sort, stably defer each block until the blocks of all its ancestors have been emitted. Ancestors resolve by class name through the `ClassDef` bases of the files it already parses. Every block with no ancestor waiting keeps its path slot.
2. Add a fixture test that `iter_blocks` emits a base (reached through a CSS-less intermediate) before its subclass, with an unrelated block keeping its slot. Show it red before the fix.
3. Rebuild with `build_css.py`, and check that the only generated change is KanbanBoardScreen's scoped block moving after BaseAppScreen's.
4. Lane: the file's one census node is `@private_profile_test`, and this test errors at setup locally with the RecoveryRequired trap. Mark `test_base_class_blocks_precede_their_subclasses` and the new test `@private_profile_test`, so they run in a fresh profile the same way, and append both node ids at the census tail with the floor set to the new count.
5. Covering tests: the file and the CSS bundle-sync tests, then preflight.

## Implementation Notes

The fix is in `widget_css.iter_blocks`, which both the widget-defaults and the screen builds use. After the path-then-line sort, `_bases_first` defers each block until the blocks of all its ancestors have been emitted; every other block keeps its slot. Ancestors are resolved by class name through the `ClassDef` bases of the files `iter_blocks` already parses, so the build stays stdlib-only. The known ceiling is marked `ponytail:` in the code: a chain through a class in a file with no bundled CSS is missed. The real-`__mro__` test still catches such a chain.

Rebuilt with `build_css.py`. The only generated change is `widget_defaults_scoped.tcss`, where KanbanBoardScreen's block moves from 2nd to 14th, right after BaseAppScreen's. The sheet's sorted line multiset is byte-identical to dev's. No other rule anywhere in the CSS names a `kanban` selector, so the move cannot change any tie. The screen sheets and the bundle are unchanged, and `check_bundle_sync.py` passes.

Tests:
- New: `test_iter_blocks_emits_every_base_before_its_subclasses`, a three-file fixture where `Sub` reaches `Base` through a CSS-less `Mid`.
- Red before the fix: `['Sub', 'Other', 'Base'] == ['Other', 'Base', 'Sub']`, and `test_base_class_blocks_precede_their_subclasses` failed with the task's `[scoped] BaseAppScreen is a base of KanbanBoardScreen ...`. Both are green after the fix.
- Both are now `@private_profile_test`, like the file's existing census node. Run alone, the plain test errored at setup with the RecoveryRequired trap (TASK-38111). It passed under the marker.
- Lane: both node ids are appended at the tail of `scripts/ui_pr_gate_census.txt`, and the floor is 1373.
- Covering run (`-n 4`, load 18-24): `test_widget_css_consolidation.py`, `test_css_bundle_sync_guard.py`, `test_css_generated_comments.py`, `test_css_staleness_manifest.py` and `test_component_pattern_governance.py` gave 314 passed. Census and shard pins (`Tests/CI/test_ui_pr_gate_census_checker.py`, `test_ui_lane_shard_count_statements.py`, `test_ui_lane_shard_replay.py`) passed with the sync guard: 120 passed.

Files: `tldw_chatbook/css/widget_css.py`, `tldw_chatbook/css/widget_defaults_scoped.tcss`, `Tests/UI/test_widget_css_consolidation.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

