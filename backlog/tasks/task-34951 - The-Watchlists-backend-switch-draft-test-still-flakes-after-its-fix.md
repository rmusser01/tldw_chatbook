---
id: TASK-34951
title: The Watchlists backend-switch draft test still flakes after its fix
status: In Progress
created_date: 2026-10-14 14:04
labels:
- tests
- flaky
- watchlists
priority: medium
assignee:
- '@claude'
updated_date: 2026-10-14 15:32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Watchlists/test_watchlists_bulk_source_authoring.py::test_backend_switch_preserves_open_source_draft_widget_and_focus[size1] failed in at least eight required-check attempts across six branches on 2026-10-13 and 2026-10-14, including a docs-only branch and a run that passed on a same-commit re-run. Commit db3a0c3183 ('await tree choices before backend-switch draft') reached dev with PR #3010 on 2026-10-13 18:29Z to fix it, and it failed again about five and a half hours later. A flaky test in the required check can evict a good pull request from the merge queue. Found while re-measuring the required check for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause of the remaining intermittent failure is identified, in the test or the product
- [ ] #2 The test passes 20 consecutive runs in CI
- [x] #3 If the product was at fault, a test that fails deterministically without the fix pins it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read every post-fix CI failure. 2. Find why the create form's watchlist Select lacked the watchlist. 3. Fix and pin with a test that holds the tree load across a section swap.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Cause (product): every CI failure after db3a0c3183 is InvalidSelectValueError: Illegal select value 1 at destination.value = watchlist['id']: the create form's watchlist Select did not offer the watchlist. A section swap calls the new pane's factory, which reads _tree_watchlists, then awaits the old pane's removal and the new one's mount (watchlists_workbench._reconcile_body / _replace_region_content). A tree load landing in that gap pushes its choices through _push_scoped_sources_to_pane, finds no mounted Sources pane and drops them; _reseed_active_section_pane, whose docstring describes exactly this gap, restored only the Sources rows. The Overview's watchlist count had the same hole (it would show the no-watchlists first-run panel). Fix: the reseed calls _push_scoped_sources_to_pane() for Sources and re-applies watchlist_count for Overview (UI/Screens/watchlists_collections_screen.py, +4/-2). Pin: test_tree_load_landing_while_a_section_pane_mounts_reaches_that_pane[sources|overview] holds the tree load until the target pane is built and lets it land before the mount; red on dev for both, green 3 of 3. The original test 5 of 5 after the fix; the whole file 28 passed. AC 2 (20 consecutive CI runs) needs the fix on dev first; the test also failed after db3a0c3183 on 2026-10-14 14:26Z (size0). Note: watchlists_collections_screen.py was already over its module-size budget on dev (14,378 against 14,324) and grows by 2 lines here.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
