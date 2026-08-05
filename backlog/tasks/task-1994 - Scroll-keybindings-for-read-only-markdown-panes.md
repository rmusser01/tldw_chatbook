---
id: TASK-1994
title: Scroll keybindings (j/k/space/b) for read-only markdown panes
status: To Do
assignee: []
created_date: '2026-08-02 22:30'
labels:
  - ux
  - keyboard
  - markdown
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Frogmouth's viewer binds `j`/`k` (line scroll), `space` (page down) and `b` (page up) on its document container — cheap, expected-by-terminal-users navigation. Chatbook's read-only markdown surfaces (HF README pane, media content/analysis panes, Library note preview) support only mouse/native scroll when focused.

Deliberate exclusion: the Console transcript already binds `j`/`k` for message SELECTION — it is out of scope and must not change. Scope is read-only viewer panes only, and bindings must be discoverable through the existing footer/key-hint convention.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The HF README pane, media content/analysis panes, and Library note preview scroll by line with j/k and by page with space/b when focused
- [ ] #2 Console transcript selection keys are untouched (existing transcript tests stay green)
- [ ] #3 The bindings are discoverable via the footer/key-hint convention on those panes
<!-- AC:END -->
