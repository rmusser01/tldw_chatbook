---
id: TASK-24612
title: Inspect rail Tab cycle stops on two non-interactive containers
status: To Do
assignee: []
created_date: '2026-08-30 00:55'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
  - a11y
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A focus walk of the open rail measured a closed Tab cycle in which two of the stops are containers, not controls: the rail root region widget and the outer scroll body. Both accept focus, neither has a dedicated focus treatment, and live capture showed their focus indication as a single border glyph and a lit scrollbar column. Separately, every one of the 11 bounded sections reported can_focus False on its viewport in the empty state, so the n and p section-jump accelerator has no focusable target in any section and repeatedly leaves focus where it was.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Tab stop inside the Inspect rail has a focus treatment a user can see
- [ ] #2 A focus stop that is a scroll container is either given a visible treatment or removed from the Tab cycle
- [ ] #3 Pressing n or p moves focus to a target the user can see in every section, including sections that do not overflow
- [ ] #4 The way to leave the rail's Tab cycle is discoverable without prior knowledge
<!-- AC:END -->
