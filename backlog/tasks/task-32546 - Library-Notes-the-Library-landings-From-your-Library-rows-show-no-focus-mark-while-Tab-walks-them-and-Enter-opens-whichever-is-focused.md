---
id: TASK-32546
title: >-
  Library Notes: the Library landing's "From your Library" rows show no focus
  mark while Tab walks them, and Enter opens whichever is focused
status: In Progress
assignee: []
created_date: '2026-09-13 06:47'
updated_date: '2026-09-14 16:58'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, persona Sam, the Library landing on the way into Notes (Library-wide, not Notes-specific; not owned by 32210–32237 or 32341–32393 — #2603 fixed rail-row focus shape only). D16.

**What happened.** From a cold landing (palette "Switch to Library") twelve Tabs never reached a rail row: the landing's "From your Library" rows come first and show no focus mark — an ANSI diff over the twelve captures changes only the nav-bar line — yet Tab×1 + Enter opened the Media item "Meeting recording 2026-09-05" (B 25-tabwalk-*.ansi, 28). F6 on the landing toggles between the rail search field ("typing in field") and an unmarked target (B 27). Captures: B 25-tabwalk-*.ansi, 27, 28.

**Cause.** INFERRED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Landing "From your Library" rows show a shape-based focus cue when focused by Tab, and the footer names the focused row
- [x] #2 F6 on the landing lands on a visibly marked target
- [x] #3 A test pins the landing row's focus class and the footer label
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: F6 on the landing toggles the rail search field and library-hub-recent-notes; the recents row changes only its BACKGROUND COLOUR (rgb(30,30,30) -> rgb(28,70,102)), no shape, and the footer never names it.
2. Add .library-hub-recent:focus with the house left bar, mirroring .library-rail-row:focus (task-32359), in the screen-owned sheet; rebuild the bundle.
3. Teach _library_focus_enter_label the landing rows and let the landing footer carry the enter chip.
4. Pin the focus rule and the footer label.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced, and the critique's wording needed one correction.

The rows are not unmarked -- they mark focus with a BACKGROUND COLOUR and nothing else. Decoded off an ANSI capture: rgb(30,30,30) unfocused, rgb(28,70,102) focused, no shape, which is the very cue task-32359 removed from the rail rows one pane away. The Tab-walk part of the report is also explained: from a cold landing Tab walks the top TAB BAR (⌃2 → ⌃3 → ⌃4 …), which is why twelve captures differed only on the nav line; the rows are reached by F6, whose landing canvas target is `library-hub-recent-notes` (`_WORKBENCH_FOCUS_TARGETS`). F6 toggling 'between the rail search field and an unmarked target' is exactly that.

Fix: `.library-hub-recent:focus` takes the house left bar, mirroring `.library-rail-row:focus` -- added to `components/_agentic_terminal.tcss` (the SOURCE; `screen_agentic_library.tcss` is generated and hand-editing it is what TASK-395 exists to prevent) and the bundle regenerated. `_library_focus_enter_label` learns the landing rows, and `_library_footer_shortcuts_for_current_state` now appends the enter chip on a surface whose static set has no 'enter' chip to replace (the Search/RAG branch replaced one). AC#2 falls out of the same rule: F6's landing target IS one of these rows.

Live at 235x52: F6 x2 paints '█ Notes · Import checklist' and the footer reads '… | enter open notes |' (wave4-caps/layout/layout-15-f6-after-2; before: layout-27-f6-2, layout-04-landing-tab1..6).

Files: css/components/_agentic_terminal.tcss (+ regenerated css/screen_agentic_library.tcss), UI/Screens/library_screen.py, Tests/UI/test_library_notes_w4_layout.py.
<!-- SECTION:NOTES:END -->
