---
id: TASK-4024
title: Settings screen's nav bar never scrolls to reveal/highlight the active destination
status: To Do
assignee: []
created_date: '2026-08-09 21:40'
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
- [ ] #1 Navigating to the Settings screen (via any path: hotkey, overflow menu, command palette) leaves the nav bar scrolled to show the active Settings destination, highlighted, within a bounded time after the screen mounts
- [ ] #2 Root cause stated: why `_scroll_active_destination_into_view`'s `scroll_to_widget` call needs multiple invocations to take effect specifically after `SettingsScreen` mounts, when it succeeds on the first call for other destinations
- [ ] #3 Regression test reproduces the stuck state headlessly (real `NavigateToScreen` to `settings` against a real `TldwCli` app) and pins the fix
<!-- AC:END -->

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
