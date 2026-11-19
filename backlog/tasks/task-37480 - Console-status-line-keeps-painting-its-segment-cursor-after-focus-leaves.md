---
id: TASK-37480
title: Console status line keeps painting its segment cursor after focus leaves
status: To Do
assignee: []
created_date: '2026-10-29 08:18'
labels:
  - console
  - keyboard-focus
  - ux-review-2026-09-29
dependencies:
  - TASK-33622.3
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The status line (ConsoleStatusText) paints its keyboard segment cursor in reverse video whenever a cursor name is set, focused or not. Once the line has had focus, the cursor stays painted after focus moves on, so the screen shows a second focus-like highlight on a control the keyboard no longer reaches. TASK-33622.3's type-to-compose makes this more common: a letter typed on the line moves focus to the composer, and the reverse-video segment stays (seen live 2026-10-29, tools/evidence/bd5/U10/r2/18-fix-strip-esc-yes.png). Dev already shows it after Esc returns focus to the composer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When focus leaves the status line, its segment cursor is no longer painted, and it reappears on the same segment when the line regains focus
- [ ] #2 A Pilot test on the real ChatScreen focuses the status line, moves focus away (Esc and a typed letter) and asserts no reverse-video segment remains; it fails on the pre-fix code
<!-- AC:END -->
