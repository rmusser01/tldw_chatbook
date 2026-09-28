---
id: TASK-33241
title: Returning to the theme picker after Save repaints faster
status: Done
assignee: []
created_date: '2026-09-28 08:00'
labels:
  - settings
  - theme
  - perf
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from PR #2877's review: Save and Save as return to the picker, and that view switch (like Back) costs ~170 ms of restyle and render at 211x44 — not stylesheet parsing (TASK-33120 already removed that).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The dominant cost of the editor-to-picker switch is measured and identified
- [x] #2 The switch is cut to under 100 ms at 211x44, or a documented reason why not and the best achievable number
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Profile Back/Save editor->picker at 211x44 on a private profile (cProfile + wrapped update_nodes/_refresh_layout, UI-thread CPU time).
2. Cut the dominant cost in this repo's code (expected: ThemePane's -picker class toggle restyling the whole pane subtree, incl. the 136-node editor).
3. Save: one catalog rescan instead of two.
4. Guard test counting restyled nodes on a switch + a CSS guard that keeps the restyle-scope assumption true.
5. Record numbers in Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ThemePane restyles only itself on a view switch, and Save rescans the themes folder once.

**Measured (AC#1).** 211x44, private profile, `TLDW_TEST_CSS_CACHE=0`, 20 saved themes; 8 interleaved before/after rounds in one process, medians. The machine was loaded (load average 36-48), so the CPU numbers (`time.thread_time` on the UI thread) are the reliable ones. The worst event-loop gap uses the `_StallMeter` from the file-actions test.
- The dominant cost was `ThemePane.watch_current`'s `set_class(..., "-picker")`. Textual restyles a node's whole subtree on any class change. Here that was 181 nodes, 136 of them the hidden editor, at ~0.35 ms per node. It cost 59 ms of CPU inside the synchronous `show_picker()` call, ~165 ms wall on a quiet machine. The first layout and render afterwards cost ~17 ms of CPU.
- Stylesheet parsing played no part (0 reparses).

**Fix (AC#2).**
- `-picker` now styles only the pane's own compound. The four descendant height rules (and their compact overrides) are `1fr` or `auto` unconditionally. They only matter while the picker shows.
- `watch_current` sets the class with `update=False` and restyles just the pane via `app.stylesheet.update_nodes([self])`.
- `test_picker_class_styles_only_the_pane` scans the built bundle, so no rule can use `-picker` as an ancestor again.
- `_saved` passes its highlight through `show_picker(highlight=...)`. It used to start two thread scans, one wasted.

**After.**

| Metric | Before | After |
|---|---|---|
| Nodes restyled | 181 | 3 (the pane and a :focus change) |
| Synchronous switch CPU | 58.8 ms | 1.0 ms |
| First frame CPU (switch + first layout/render) | 74.0 ms | 17.3 ms |
| Worst UI stall, wall | 115 ms | 58 ms |

What remains is Textual's layout and render of the switched view, spread over several frames (~70 ms of CPU in total, unchanged). Its worst single frame is well under 100 ms.

**Files.**
- `Widgets/settings_theme_picker.py`: `watch_current`, `show_picker(highlight)`, `_saved`
- `css/components/_settings_splash_theme.tcss` and the rebuilt bundle
- `Tests/UI/test_settings_theme_switch_cost.py`: CSS guard, restyle count with the list still filling, single-rescan Save

All three tests fail on the base code.
<!-- SECTION:NOTES:END -->
