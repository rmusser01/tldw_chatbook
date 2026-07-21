---
id: TASK-346
title: Guard Console layout at small terminal sizes - composer clipped out entirely at 97x30 cells
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, keyboard]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cold start at 700x480 px (97x30 cells): rail, transcript and Inspector chip render, but the composer row does not exist anywhere on screen (workbench boxes are clipped open-ended at the bottom); typing produced no visible echo. No 'terminal too small' notice is shown. At 125x38 (900x620) the composer is present and usable. Footer at small widths truncates the key hints ('Ctrl+K switch se', 'F1 help | E') while memory stats (P:/C\/N:/M:) keep their full space.

**Repro:** Serve the app and open it in a 700x480 viewport (97x30 cells). Observe no composer row and no warning; type text and observe nothing change. Compare with 900x620 where the composer renders at row 34.

**Verifier note:** Evidence j6-b05-cold-700x480.png confirms: no composer row exists at 97x30, boxes clip open-ended, no too-small warning, and footer truncates key hints while P:/C\/N:/M: memory stats keep full width. chat_screen.py has no on_resize/breakpoint handling and no minimum-size overlay. No ledger or backlog coverage (task-226/task-108 are Personas-specific). 97x30 is larger than the 80x24 default terminal, so the core loop is silently broken at common real sizes — P1 stands.

**Source:** Console UX expert review 2026-07-20 (finding j6-small-terminal-composer-clipped; P1, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-b05-cold-700x480.png`, `j6-b06-700x480-typing.png`, `j6-b01-900x620.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Below the minimum viable height, either the layout drops lower-priority panes (rail/inspector) to preserve transcript+composer, or an explicit 'terminal too small' overlay appears. Key hints should win over debug stats in footer truncation
<!-- AC:END -->
