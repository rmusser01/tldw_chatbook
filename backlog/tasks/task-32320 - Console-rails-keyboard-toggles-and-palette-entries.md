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
- [ ] #1 A keybinding exists that toggles the Context (left) rail, conforming to ADR-031 conventions, and works when focus is in the transcript or composer
- [ ] #2 A keybinding exists that toggles the Inspector (right) rail under the same constraints
- [ ] #3 Both toggles are reachable as 'Console: ...' command palette entries
- [ ] #4 The bindings do not shadow terminal-convention keys or the globals listed in ADR-031, and the composer still receives all printable input
- [ ] #5 F1 help panel lists the new bindings; user-guide docs updated
<!-- AC:END -->
