---
id: TASK-32320
title: >-
  Console rails keyboard toggles and palette entries
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review A1 (High). There is no keybinding to open, close, or toggle either Console rail; verified across ChatScreen BINDINGS (chat_screen.py ~1395-1486) and the rail widgets (no BINDINGS/on_key). Collapse/reopen is pointer-only (header buttons, edge handles, status chips), and the chips are unavailable before the first send and in single-pane mode -- so a keyboard user cannot reach the Inspector at all in some states. ADR-031 governs keybinding choices; the palette (Ctrl+P) already carries 'Console: ...' entries.

Filed from the 2026-09-10 Console rail UX review (review item A1).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: binding exists, F1 advertises, palette offers both, alt+c round trip. 2. Add binding + mirror action; footer + single-pane + F1 entries; palette entries. 3. Update the user-guide keyboard table. 4. Run keyboard-route + wiring + right-rail suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A keybinding toggles the Context (left) rail, mirroring the existing alt+i Inspector toggle (conforming to ADR-031), and works when focus is in the transcript or composer
- [x] #2 Both rail toggles (left and Inspector) are reachable as 'Console: ...' command palette entries
- [x] #3 The bindings do not shadow terminal-convention keys or ADR-031 globals, and the composer still receives all printable input
- [x] #4 F1 help panel lists the left-rail binding alongside the existing alt+i entry; user-guide docs updated
- [x] #5 The left-rail toggle respects the same reveal/preference semantics as the existing Inspector toggle (explicit marker, width floors)
<!-- AC:END -->
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** The Context rail now has the keyboard dignity alt+i gave
the Inspector (TASK-24604's contract): `alt+c` binding +
`action_toggle_console_context_rail` — not gated on display (a collapsed
rail at narrow widths is when the way back matters), opening focuses the
rail via TASK-32321's content-first map, closing returns focus to the
composer. Footer vocabulary gains ("Alt+C", "context rail") and the
single-pane promotion now fronts BOTH rail accelerators (the 80-col
degradation lesson). F1's Panes group lists it beside Alt+I. Command
palette gains "Console: Toggle Context rail" AND "Console: Toggle
Inspector rail" (the Alt+M-eaten-by-multiplexer fallback). A bare 'c'
htop key would be swallowed by the transcript's own bindings, so it
rides the alt chord like its sibling.

**ADR check.** Conforms to ADR-031 (no terminal-convention or global
shadowing; alt chord pattern follows alt+m/alt+w/alt+i precedent) — no
new ADR required.

**Modified.** `UI/Screens/chat_screen.py` (binding, action, footer,
single-pane, F1), `UI/console_command_provider.py` (two entries),
`Tests/UI/test_console_inspector_keyboard_route.py` (+4 tests: binding
exists, F1 advertises, palette offers both, full alt+c round trip),
`Docs/User_Guide/console.md` (keyboard table). Verified:
keyboard-route suite 13 passed; environment-wiring + right-rail suites
85 passed.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (live + code): left rail has NO toggle binding/action/palette entry; right rail has alt+i (chat_screen.py:1881). Scope: LEFT-rail binding + palette entries for BOTH rails.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
