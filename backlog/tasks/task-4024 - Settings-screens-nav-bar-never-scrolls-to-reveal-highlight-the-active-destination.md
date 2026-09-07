---
id: TASK-4024
title: >-
  Settings screen's nav bar never scrolls to reveal/highlight the active
  destination
status: Done
assignee:
  - '@claude'
created_date: '2026-08-09 21:40'
updated_date: '2026-08-10 21:41'
labels:
  - navigation
  - regression
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during task-4020's (Library re-critique RC-02) live/headless verification -- a separate,
real defect from that task's mid-word-clip ghosting scope.

Navigating to Settings (via the overflow menu; also reproduced headlessly via `NavigateToScreen`
against a real `TldwCli` app) leaves `MainNavigationBar`'s strip stuck at `scroll_x=0`: the active
destination is never scrolled into view, so its `is-active` highlight is never visible anywhere on
screen. Reproduced live (persisted 15+ seconds, both at 80 and 120 cols, via tmux) and headlessly
(`Tests/UI/app_factory._build_test_app` + a real `NavigateToScreen("settings")`):
`active_destination_id` resolves correctly to `"settings"` and `nav-settings` carries the
`is-active` class, but `strip.scroll_to_widget(button)` (called from
`_scroll_active_destination_into_view`) returns `True` without actually moving `scroll_x` on the
first two invocations after `SettingsScreen` mounts; a third, later call succeeds.

Other destinations reached the same way (Schedules, Lab, MCP, ACP -- including other F-key routes)
scroll correctly, so this looks specific to Settings' unusually heavy initial layout
(`settings_screen.py` is ~18k lines) racing the nav bar's own settle chain (`on_mount`'s single
`call_after_refresh`) rather than a `NavOverflowMenu` or ghosting defect. Filed separately from
task-4020 because it does not itself produce a mid-word-clipped label: everything at the stuck
`scroll_x=0` position is either fully visible or correctly ghosted by the bar's initial settle
pass. It is a lost active-tab-highlight / discoverability bug, not a ghosting regression.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Navigating to the Settings screen (via any path: hotkey, overflow menu, command palette) leaves the nav bar scrolled to show the active Settings destination, highlighted, within a bounded time after the screen mounts
- [x] #2 Root cause stated: why `_scroll_active_destination_into_view`'s `scroll_to_widget` call needs multiple invocations to take effect specifically after `SettingsScreen` mounts, when it succeeds on the first call for other destinations
- [x] #3 Regression test reproduces the stuck state headlessly (real `NavigateToScreen` to `settings` against a real `TldwCli` app) and pins the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify at HEAD with an instrumented headless repro (done): SettingsScreen.on_mount -> _refresh_sync_rows() sets recompose=True reactives ~300ms after mount, recomposing the whole screen; BaseAppScreen.compose mints a NEW MainNavigationBar, discarding the first bar's successful scroll; the new bar's _mark_mount_settled (one call_after_refresh tick) fires BEFORE the post-recompose automatic focus restoration lands on nav-home, so on_descendant_focus records that automatic landing as a DELIBERATE focus; every later _recenter_strip (interval/resize) then targets always-visible nav-home instead of active nav-settings -- scroll_x pinned at 0, manual scrolls snapped back.
2. Fix in MainNavigationBar only (shared chrome, bounded): close the settle window on evidence, not time -- if the screen has no focused widget when _mark_mount_settled runs, the automatic focus placement has not landed yet; defer settling until the next DescendantFocus and consume that event as automatic (never deliberate). First-mount ordering (AUTO_FOCUS before the marker) is unchanged.
3. TDD: RED regression test (real TldwCli + real NavigateToScreen('settings')) that waits for the recompose replacement and requires the active button fully visible AND stable past an interval tick; convert/delete the scratch harness.
4. Regression-run the master-shell nav suite; live tmux verification at 80 cols.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (AC#2), established with an instrumented headless repro before fixing: SettingsScreen.on_mount -> _refresh_sync_rows() sets the screen-level recompose=True reactives ~300ms after mount; the whole-screen recompose mints a REPLACEMENT MainNavigationBar (BaseAppScreen.compose), discarding the first bar's already-successful scroll-to-active. For the replacement, the screen is already laid out, so its _mark_mount_settled marker (first call_after_refresh tick) fires a few ms BEFORE the post-recompose automatic focus placement lands on the bar's first button (nav-home) -- the OPPOSITE ordering from a first mount, where AUTO_FOCUS empirically lands before the marker. on_descendant_focus therefore recorded that automatic landing as a DELIBERATE focus, and every subsequent _recenter_strip pass (the 0.5s interval, resizes) recentered on always-visible nav-home instead of active nav-settings: scroll_x pinned at 0, manual scroll_to_widget calls snapped back within one tick (which is why the filed repro saw True-but-no-move calls). Fix, bounded to MainNavigationBar: close the settle window on EVIDENCE, not time -- if screen.focused is None when the marker runs, the automatic placement has not landed; arm _settle_after_next_focus and let on_descendant_focus consume the next focus event as automatic (never deliberate), then settle. First-mount ordering unchanged (focused non-None at marker time -> settle immediately). Failure bias is safe: an unrecorded focus only ever means recenter prefers ACTIVE. TDD: Tests/UI/test_settings_nav_active_scroll.py (real TldwCli + real NavigateToScreen('settings'), waits for the recompose replacement, requires the active button fully visible AND still visible after a >0.5s hold) -- RED at HEAD with the exact stuck signature (replaced=True scroll_x=0.0 button x=136 vs strip width 70), GREEN with fix. Regressions: test_master_shell_navigation (41 passed), test_screen_navigation (123 passed; 2 failures + 1 timeout A/B-confirmed pre-existing at clean HEAD), destination-headers/study/state-ownership suites green. Live tmux at 80 AND 120 cols: More ▾ -> F9 Settings; strip scrolled with 'F9 Settings' boxed within ~3s and still boxed 15s+ later; ANSI capture shows the is-active background (48;2;0;101;190) on the label. Files: main_navigation.py, Tests/UI/test_settings_nav_active_scroll.py, Docs/User_Guide/settings.md (stale pager bullet fixed + stamp). Predecessor's scratch harness deleted after conversion into the pinned test.
<!-- SECTION:NOTES:END -->

## Notes

Filed from task-4020 (Library re-critique RC-02) live/headless verification, 2026-08-09.

Headless repro: `Tests/UI/app_factory._build_test_app()`, post `NavigateToScreen("settings")`,
poll for `SettingsScreen`, then read `MainNavigationBar`'s `strip.scroll_x` (stays `0.0` well
past 1s) and `nav-settings`' region (off-screen, e.g. `Region(x=136, y=0, width=15, height=3)`
against a `strip.region` of `Region(x=0, y=0, width=70, height=2)`). Manually re-invoking
`strip.scroll_to_widget(button, animate=False)` a third time (after two prior no-op-but-`True`
calls) does scroll correctly, which is why this reads as a settle-timing race against Settings'
unusually heavy construction rather than a geometry or routing bug -- `active_destination_id`
and the `is-active` CSS class are both already correct throughout.

Live repro: tmux scratch profile (`TLDW_CONFIG_PATH` pointing at a fresh `[general] users_name`),
80 and 120 cols, click "F9 Settings" in the `NavOverflowMenu` (`More ▾`), strip stays at
`scroll_x=0` (Home visible, nothing highlighted) for 15+ seconds -- confirmed at both widths.
