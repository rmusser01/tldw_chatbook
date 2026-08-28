---
id: TASK-23114
title: Add mouse reporting to persistent terminal sessions
status: To Do
assignee: []
created_date: '2026-08-28 00:00'
updated_date: '2026-08-28 00:00'
labels:
  - console
  - terminal
  - input
  - ux
dependencies:
  - TASK-22512
references:
  - backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let explicitly mouse-aware programs running in a persistent terminal receive bounded, correctly encoded mouse input without breaking Chatbook selection, scrollback, focus release, or host-terminal interaction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Terminal clicks, motion, and wheel events remain local to Chatbook unless the focused terminal has enabled a supported mouse-reporting mode.
- [ ] #2 Supported terminal mouse modes encode coordinates, buttons, release, wheel direction, and modifiers correctly within the visible terminal viewport.
- [ ] #3 Local scrollback, text selection, Ctrl+] focus release, rail interaction, and global Chatbook controls remain usable before, during, and after terminal mouse reporting.
- [ ] #4 Mouse capture and reporting state clears on focus release, alternate-screen exit, session close, Terminal disarm, navigation, and app shutdown without leaving input routed to a stale session.
- [ ] #5 Invalid, oversized, off-viewport, reordered, or unsupported mouse events fail closed and cannot inject arbitrary terminal bytes or escape sequences into unrelated UI.
- [ ] #6 POSIX PTY and Windows ConPTY sessions exhibit equivalent user-visible mouse behavior for the supported protocol subset.
- [ ] #7 Focused unit, mounted Textual, and real-terminal verification cover event-shape, capture, coordinate, resize, scrollback, and late-click races that synthetic Pilot events cannot certify alone.
- [ ] #8 User documentation identifies supported mouse modes and explains how to release terminal input when a nested program captures the mouse.
- [ ] #9 The implementation plan records whether the persistent-terminal ADR requires an amendment before code work.
<!-- AC:END -->
