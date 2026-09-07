---
id: TASK-31825
title: Wire DestinationHeader's dormant compact density to a height-based trigger
status: Done
assignee:
  - '@claude'
created_date: '2026-09-06 14:15'
updated_date: '2026-09-06 18:14'
labels:
  - ui
  - responsive
  - workbench
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from task-31419's real-shell measurement (2026-09-06, 80x24, scratch profile). The measurement corrected PR-4 Task 6's attribution: of the claimed 13 chrome rows, only 4 are true app shell (MainNavigationBar 3, AppFooterStatus 1 -- both deliberate/minimal); the destination header (5 rows), scheduler liveness (1) and status strip (4) are composed inside SchedulesWorkbench itself. The one real narrow-terminal lever found: the shared DestinationHeader widget (UI/Workbench/workbench_widgets.py:~164) already ships a density="compact" CSS rule that NO caller ever triggers, and no height-based responsive logic exists anywhere in the workbench layer. Wiring a height-based trigger there would reclaim rows for every (~12) workbench screen that uses the widget -- a shared-widget-layer change, not UI/Navigation/ and not a per-screen override. Full measurement in task-31419's Implementation Notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DestinationHeader renders its compact density automatically below a height threshold, via the shared workbench-widget layer (no per-screen overrides)
- [x] #2 Every workbench screen using DestinationHeader benefits without per-screen changes, verified on at least Schedules plus one other
- [x] #3 Geometry is asserted with the bundled stylesheet (compact rows measurably fewer at 80x24; normal density unchanged at standard sizes)
- [x] #4 Tests/UI/test_schedules_responsive_floor.py stays green (minus the known pre-existing dev red)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read DestinationHeader/WorkbenchHeaderState/WorkbenchFrame to find the dormant .density-compact .workbench-header CSS rule and how density classes reach the DOM; inventory all DestinationHeader( call sites.
2. Investigate Textual's native VERTICAL_BREAKPOINTS mechanism (Screen._on_resize) as the trigger: confirm empirically it fires on both terminal resize AND a freshly pushed screen's first layout pass (first paint correct, no custom on_resize/on_mount code needed).
3. Add VERTICAL_BREAKPOINTS + a named floor constant (24) to BaseAppScreen (the common base of every DestinationHeader-using screen), toggling a new marker class distinct from the existing user-preference-driven density-compact/density-normal (which Console already drives itself and which touches unrelated ds-panel/ds-inspector rules) to avoid collision.
4. Add narrow CSS rules in components/_workbench.tcss keyed to the new marker class, scoped to .workbench-header/.workbench-header-subtitle only (reusing the same min-height/padding/display values .density-compact already declares for the header).
5. Regenerate the CSS bundle and verify check_bundle_sync.
6. Write revert-checked tests (ConsolidatedCSSApp + APP_STYLESHEETS harness) covering: compact at 80x24 (pinned height), normal at a standard size (pinned height), the exact 24/25 threshold boundary, first-paint correctness with no resize event, a second screen (Study) benefiting with zero per-screen code, and hysteresis across repeated resizes.
7. Run the mandated regression suites (floor suite, workbench widgets, destination headers, on_mount/on_unmount MRO guards, plus broader screen-navigation/console-workbench/css-class-coverage spot checks) and attribute every pre-existing failure via a throwaway revert (never git stash).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Trigger: Textual's own VERTICAL_BREAKPOINTS on BaseAppScreen (the common base of every DestinationHeader-using screen), not a hand-written on_resize/on_mount handler. Screen._on_resize applies breakpoint classes on every terminal resize AND on the screen's own first layout pass (a freshly pushed screen posts itself a Resize the moment it gets a real size), so a screen mounted straight at 80x24 is compact before the first paint -- verified empirically with a probe script before committing to the design. This sidesteps the on_resize/on_mount MRO AST guard entirely (no new handler method is written).

Floor: BaseAppScreen._DESTINATION_HEADER_COMPACT_FLOOR_HEIGHT = 24 (named constant + comment). VERTICAL_BREAKPOINTS = [(0, "shell-header-compact"), (25, "shell-header-normal")] -- compact at/below 24, normal at 25+ (boundary-tested).

CSS: deliberately did NOT reuse the existing `.density-compact` class the task's dormant rule already declares. That class is also the user's own global density preference and, on Console, an inner #console-shell wrapper Console manages itself (`density-{workbench_state.density}` in chat_screen.py) -- putting the same class on the Screen ancestor would fight that signal (both `.density-compact .workbench-header` and `.density-normal .workbench-header` could apply simultaneously to the same header depending on which ancestor's class won), and would ALSO compact every unrelated `.ds-panel`/`.ds-inspector`/`.ds-approval-card`/`.ds-recovery-callout` under that screen (a much bigger blast radius than the header). Added new rules keyed to `shell-header-compact` instead, scoped only to `.workbench-header`/`.workbench-header-subtitle`, reusing the exact min-height/padding/display values `.density-compact` already declares (components/_workbench.tcss). Console's `#console-workbench-header.console-header-inline` (id-qualified) and Lab's `.lab-header-inline` and Research Workspace's `#research-workspace-header .workbench-header` (both higher- or equal-specificity, verified against source order) all keep their existing shape unaffected, same as they already do against `.density-compact` today.

Rows saved at 80x24: Schedules' DestinationHeader measured 5 rows (border 2 + title 1 + subtitle 1 + status 1) before, 4 after (subtitle hidden via `.shell-header-compact .workbench-header-subtitle { display: none }`, `min-height` drops 2->1). Verified via a live probe against the real bundle before writing assertions, then pinned in tests. Confirmed the same 5->4 drop on Study (a second, unmodified DestinationHeader caller) with zero per-screen code -- proves AC#2.

Tests: Tests/UI/test_destination_header_compact_floor.py (new, 7 tests) -- compact at the 80x24 floor (pinned height=4), unchanged at a standard size (pinned height=5, matches today), the exact 24/25 threshold boundary, first-paint correctness with no resize event, Study as the second beneficiary at both sizes, and a repeated-resize hysteresis check (children count stays 3, no duplicate/leaked classes across 6 toggles). Every test in the new file was revert-checked: overwritten the 3 changed files with their HEAD content via `git show`/`cp` (no `git stash`, per the hard rule), confirmed all 7 fail without the wiring, then restored.

Regression sweep (all foreground, revert-checked where the failure count looked surprising): test_schedules_responsive_floor.py (27 passed, 1 known pre-existing red: test_the_docked_task_detail_pane_scrolls_to_reveal_history_past_the_fold, unrelated -- TALL=(235,52) is far above the floor); test_workbench_widgets.py + test_destination_headers.py (1 pre-existing failure, test_folded_screens_box_owning_destination_in_nav, confirmed identical at baseline); test_css_class_coverage_contract.py (162 pre-existing missing-class entries, byte-identical count at baseline, none of them the new shell-header-* classes); test_shell_chrome_contract.py (clean); test_console_workbench_contract.py + test_screen_footer_hints.py (14 pre-existing failures, confirmed identical set at baseline, including the console-header-inline tests -- Console's id-qualified rule protects it as designed); test_destination_shells.py (6 pre-existing failures, confirmed identical at baseline, including the [schedules] case); test_screen_navigation.py + test_screen_navigation_failure_recovery.py + test_settings_nav_active_scroll.py (pre-existing order-dependent flake in this huge shared-state file -- 31-32 failures either way from the same cluster of library-routing tests, confirmed at baseline before touching the wiring); test_on_mount_mro_convention.py + test_on_unmount_mro_convention.py (clean -- no new handler methods were written). CSS bundle regenerated via `python3 -m tldw_chatbook.css.build_css` and `check_bundle_sync` green.

Files: tldw_chatbook/UI/Navigation/base_app_screen.py (VERTICAL_BREAKPOINTS + floor constant), tldw_chatbook/css/components/_workbench.tcss (new rules + comment), tldw_chatbook/css/tldw_cli_modular.tcss (regenerated bundle), Tests/UI/test_destination_header_compact_floor.py (new).
<!-- SECTION:NOTES:END -->
