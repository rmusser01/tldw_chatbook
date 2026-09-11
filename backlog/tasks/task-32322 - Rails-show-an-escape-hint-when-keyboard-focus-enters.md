---
id: TASK-32322
title: >-
  Rails show an escape hint when keyboard focus enters
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review A3. Tab is region-locked per CONSOLE_TAB_REGIONS (chat_screen.py ~656-661); F6/Esc are the exits but nothing teaches that at the point of need. When focus enters a rail, surface a brief contextual hint naming the exit keys, without permanently spending a row.

Filed from the 2026-09-10 Console rail UX review (review item A3).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: test that focusing either rail prepends the escape hint and composer focus drops it. 2. Add left-rail on_focus/on_blur footer refresh. 3. Add _console_rail_focus_active + prepended hint in the screen. 4. Run navigation + left rail + setup suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When keyboard focus moves into either rail, a transient or dismissible hint identifies F6 (next pane) and Esc (composer) as exits
- [x] #2 The hint does not appear for mouse-only interaction and does not change rail layout height persistently
- [x] #3 Hint copy matches the actual bindings (truthfulness rule, TASK-2154 FR-06)
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** Extended the Inspector's existing footer-hint mechanism
(right_rail `_set_inspector_focus_active` -> screen
`_register_console_footer_shortcuts`) to both rails:
`ConsoleLeftRail` gained `on_focus`/`on_blur` that ask the screen to
re-register footer hints (blur deferred via `call_after_refresh` so exit
truth waits for replacement focus, mirroring the right rail's
`_finish_descendant_blur`). The screen prepends
`("Esc", "composer · F6 panes")` while focus is inside EITHER rail (new
`_console_rail_focus_active` checking both rail subtrees). Prepend order
follows the established degradation rule (hints drop from the END). No
permanent layout cost; mouse-only use never triggers it. Copy matches
actual bindings (Esc->composer is a screen binding; F6 next pane in
footer/F1), per TASK-2154 FR-06 truthfulness.

**ADR check.** Not required — footer hint following an existing pattern.

**Modified.** `UI/Console_Modules/left_rail.py` (focus handlers),
`UI/Screens/chat_screen.py` (hint + helper), `Tests/UI/
test_console_inspector_navigation.py` (+1 test: hint appears on left-rail
focus, persists alongside n/p on right-rail focus, drops on composer
focus). Verified: navigation suite + left rail + setup polish — 55
passed; the 2 navigation failures (staged owner sync, responsive handoff)
are pre-existing dev drift (verified on stashed clean tree).

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE: 10 Tab presses walked focus Conversations header -> filter input -> rows, never leaving the rail. Current state: right rail gets a footer 'n/p Sections' hint only (chat_screen.py:5016-5017); no Esc/F6 escape hint, left rail gets nothing. Scope: extend the footer-hint mechanism.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
