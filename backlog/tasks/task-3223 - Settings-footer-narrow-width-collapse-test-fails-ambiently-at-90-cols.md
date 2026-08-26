---
id: TASK-3223
title: Settings footer narrow-width collapse test fails ambiently at 90 cols
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 19:10'
updated_date: '2026-08-09 18:17'
labels: []
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_narrow_footer_collapses_but_f1_help_stays_truthful (Tests/UI/test_settings_footer_hints.py) fails on unmodified dev: at 90 cols the Settings footer's Storage-category hints (s save category | r revert category | t check storage) do not collapse to an ellipsis+globals form the way the test expects -- the full text plus GLOBAL_HINTS_COMPACT still fits under the current width-ladder math. Confirmed ambient (A/B'd against clean HEAD @ c9dfc48cb, unrelated to task-2860's footer dedup fix -- same failure reproduces with AppFooterStatus.py fully reverted to HEAD).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test passes on dev, or its width assumption is corrected to match the real ladder math
- [x] #2 No regression to the other test_settings_footer_hints.py cases (F1 help truthfulness, per-category hint sets)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the failure at HEAD in isolation to get the exact rendered text.
2. Read AppFooterStatus._apply_responsive_footer's width-ladder steps (task-2860/LIB-18) to understand the CURRENT contract: screen-context hints outrank the global cluster -- globals compact first, screen context only collapses to ellipsis once even the compacted-globals form no longer fits.
3. Empirically probe the widget at a range of widths (100 down to 50) to find the actual threshold where the Storage category's context collapses to ellipsis under the real ladder.
4. Repair the test's assumed width (was 90, a width where the CURRENT contract keeps full context alive) to a width inside the confirmed-collapsing range, and update the stale docstrings/comments describing the old <=100-cols assumption.
5. Run the test 5x; confirm no change to production code was needed (this was a stale test assumption, not a footer bug).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Discovered while running the footer regression sweep for task-2860 (Library footer F6 hint fix). Isolated repro: pytest Tests/UI/test_settings_footer_hints.py::test_narrow_footer_collapses_but_f1_help_stays_truthful -q. Not caused by task-2860 -- confirmed by reverting tldw_chatbook/Widgets/AppFooterStatus.py to HEAD and re-running (still fails identically).

**Diagnosis (2026-08-09).** The test's assumption was stale, not the footer. task-2860/LIB-18 (`AppFooterStatus._apply_responsive_footer`) reordered the responsive ladder so a screen's OWN hints outrank the global cluster: the ladder now compacts the GLOBAL half first (`GLOBAL_HINTS_COMPACT`, dropping "F6 panes" entirely and shortening the rest) while keeping the screen's context text fully intact, and only once even that compacted-globals form no longer fits does the screen context itself collapse to an ellipsis. At 90 cols the Storage category's `context_compact_globals` step ("s save category | r revert category | t check storage | F1 · Ctrl+P · Ctrl+Q", 76 chars) fits comfortably -- the test's expectation ("save category" disappears by 90 cols) described the PRE-LIB-18 ladder, which used to jump straight from full text to an ellipsis.

Empirically probed the real widget (mounted, not hand-derived) across widths 100->50: the screen context stays intact through width=84 and collapses to the ellipsis form at width=83, staying collapsed through width=56 (the globals compact further, to `GLOBAL_HINTS_COMPACT`, only at width<=55). Fixed the test by moving its harness width from 90 to 70 (comfortably inside the confirmed-collapsing range) and updating the module/test docstrings that described the old <=100-cols assumption to describe the current LIB-18 contract instead. No production code changed -- confirmed by `git diff --stat tldw_chatbook/` being empty for this task.

Red-before-fix evidence: the original failure (before any edit) was `AssertionError: expected collapsed footer at 90 cols, got 's save category | r revert category | t check storage | F1 · Ctrl+P · Ctrl+Q'`. After the width fix the test passes 5/5 consecutive runs.

**Note (out of scope, filed as a concern, not fixed here):** running the full `Tests/UI/test_settings_footer_hints.py` file (not just the target test) surfaces 4 OTHER failures, all `AttributeError: 'SettingsScreen' object has no attribute '_appearance_bool_label'` (or a downstream consequence of it) when a test's flow touches the Appearance category. Confirmed ambient: `git diff --stat 4d0232358 HEAD` shows zero production-code changes on this branch relative to dev tip, so this is a pre-existing dev-tip defect, unrelated to task-3223's narrow-footer assertion. `_appearance_bool_label` is called at 9 call sites in `settings_screen.py` (`_sync_appearance_widgets` and the appearance action handlers) but is never defined anywhere in the tree -- this looks like a real, currently-shipping crash for the Settings Appearance category, not test debt. Flagging for a follow-up task; out of this task's bounded scope (3223/3800/3801).

**Files changed:** `Tests/UI/test_settings_footer_hints.py` (module docstring + `test_narrow_footer_collapses_but_f1_help_stays_truthful`: width 90->70, updated assertions/docstrings to the current LIB-18 contract).
<!-- SECTION:NOTES:END -->
