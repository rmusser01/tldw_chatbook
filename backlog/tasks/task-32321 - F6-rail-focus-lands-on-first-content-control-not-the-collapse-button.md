---
id: TASK-32321
title: >-
  F6 rail focus lands on first content control not the collapse button
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
UX review A2. CONSOLE_FOCUS_TARGETS_BY_PANE (chat_screen.py ~636-644) makes the rail collapse button the first F6 focus target inside each rail, so a reflexive Enter hides the pane the user just entered. The target should be the first content control; the collapse button stays one Tab away.

Filed from the 2026-09-10 Console rail UX review (review item A2).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: repin the F6 cycle test to content controls. 2. Reorder CONSOLE_FOCUS_TARGETS_BY_PANE (content first, collapse as fallback). 3. Update remaining pins + docstring + user guide. 4. Run all four focus suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 F6 into the Context rail focuses a content control (first section header or equivalent), not #console-context-rail-collapse
- [x] #2 F6 into the Inspector rail focuses a content control, not #console-inspector-rail-collapse
- [x] #3 The collapse buttons remain reachable by Tab from the content target
- [x] #4 Existing F6 pane-cycle order (left rail -> transcript -> right rail -> composer) is unchanged
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** `CONSOLE_FOCUS_TARGETS_BY_PANE` now lists each rail's first
CONTENT control before its collapse button: left rail targets the pinned
Terminal action (`console-terminal-open`), right rail the pinned send
authority summary (`console-send-authority-summary`, the same target
TASK-24703 chose for alt+i). The collapse buttons remain as fallback stops
(and `console-left-rail`/`console-right-rail` as last-resort fallbacks),
so nothing is lost — the destructive control is just no longer the FIRST
thing Enter would hit after entering the pane. Verified live during the
walkthrough: previously one F6 + one reflexive Enter collapsed the whole
rail silently.

Pinned tests updated to the new first stops:
test_workbench_pane_focus (F6 cycle), test_console_tab_scope (focus tour +
F6-from-composer). test_console_inspector_keyboard_route's alt+i test was
already asserting the authority summary — its docstring now notes the map
caught up. The 14 failures in test_console_workbench_contract are
pre-existing dev drift (verified by stashing my change and re-running on
the clean tree: same 14 fail), filed separately.

**ADR check.** Not required — ordering change inside one focus map,
behaviour pinned by existing tests.

**Modified.** `UI/Screens/chat_screen.py` (map + comment),
`Tests/UI/test_workbench_pane_focus.py`, `Tests/UI/test_console_tab_scope.py`,
`Tests/UI/test_console_inspector_keyboard_route.py` (docstring),
`Docs/User_Guide/console.md`. Verified: pane-focus, tab-scope, keyboard-
route, inspector-focus-visibility suites — 32 passed.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE: one F6 press focused the left rail; a reflexive Enter collapsed the whole rail silently. Current map chat_screen.py:876-884 still targets the collapse buttons. alt+i open path bypasses to #console-send-authority-summary (TASK-24703) - follow that pattern.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
