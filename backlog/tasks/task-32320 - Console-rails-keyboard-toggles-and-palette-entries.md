---
id: TASK-32320
title: >-
  Console rails keyboard toggles and palette entries
status: To Do
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

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A keybinding toggles the Context (left) rail, mirroring the existing alt+i Inspector toggle (conforming to ADR-031), and works when focus is in the transcript or composer
- [ ] #2 Both rail toggles (left and Inspector) are reachable as 'Console: ...' command palette entries
- [ ] #3 The bindings do not shadow terminal-convention keys or ADR-031 globals, and the composer still receives all printable input
- [ ] #4 F1 help panel lists the left-rail binding alongside the existing alt+i entry; user-guide docs updated
- [ ] #5 The left-rail toggle respects the same reveal/preference semantics as the existing Inspector toggle (explicit marker, width floors)
<!-- AC:END -->
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (live + code): left rail has NO toggle binding/action/palette entry; right rail has alt+i (chat_screen.py:1881). Scope: LEFT-rail binding + palette entries for BOTH rails.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
