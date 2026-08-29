---
id: TASK-23195
title: 'Console Context rail: give the rail a title distinct from its collapse control'
status: To Do
assignee: []
created_date: '2026-08-29 21:56'
labels:
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail's entire header is a single Button labelled '<---------|Context'. There is no rail title; the word Context exists only as part of the control that collapses it. The literal is hard-coded ASCII art that bypasses the ascii_glyphs fallback system every other Console glyph routes through. Separately, the overflow hint says 'more sections - scroll' without naming what is hidden.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rail shows a title that is not the collapse control
- [ ] #2 The collapse affordance resolves its glyph through resolve_glyph so ASCII mode works
- [ ] #3 The overflow hint names the hidden sections
<!-- AC:END -->
