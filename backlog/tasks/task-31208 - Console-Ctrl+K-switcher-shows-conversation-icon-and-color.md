---
id: TASK-31208
title: 'Console Ctrl+K switcher: show conversation icon and color'
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-03 12:00'
labels:
  - console
  - ui
dependencies:
  - TASK-31207
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Once conversations carry a custom icon+color (TASK-31207), the Ctrl+K session
switcher (`Widgets/Console/console_session_switcher_modal.py`) still renders
plain rows. Show the same colored icon to the left of each row title so the
visual anchor works in the keyboard-driven switcher too, not just in the
Context tab.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ctrl+K switcher rows for conversations with a set appearance render the colored icon to the left of the title; rows without one render as today
- [x] #2 Switcher layout unchanged otherwise (subtitle, filtering, selection highlight) — geometry/render evidence, not just text asserts
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Thread icon/color through `ConsoleSwitcherEntry` + builder.
2. Render a colored icon prefix left of the title in the modal (ASCII-glyph
   mode aware), escaping the sanitized title under `Text.from_markup`.
3. Route the switcher's row gathering through `_merge_console_browser_rows`
   so rows carry appearance (and dedupe) exactly like the rail.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `ConsoleSwitcherEntry` gained `icon`/`color` (defaulted, so existing
  constructors are untouched); `build_console_switcher_entries` copies them
  from browser rows.
- `_switcher_icon_prefix` renders the colored icon (or the fixed ASCII `*`
  substitute) left of the title; the label now goes through
  `Text.from_markup` with the sanitized title/subtitle escaped.
- `action_open_console_session_switcher` passes its three row sources through
  `_merge_console_browser_rows`, the appearance/star/dedupe choke point,
  instead of concatenating raw builder output.
- Tests: entry passthrough (`test_console_switcher_state.py`), modal render
  + prefix-helper contract (`test_console_appearance_picker.py`), and both
  Ctrl+K integration tests in `test_console_native_chat_flow.py` green.
- Modified: `Chat/console_switcher_state.py`,
  `Widgets/Console/console_session_switcher_modal.py`,
  `UI/Screens/chat_screen.py` (row gathering only).
<!-- SECTION:NOTES:END -->
