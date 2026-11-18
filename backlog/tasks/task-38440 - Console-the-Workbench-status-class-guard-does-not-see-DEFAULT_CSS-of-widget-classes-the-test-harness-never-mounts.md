---
id: TASK-38440
title: >-
  Console: the Workbench status-class guard does not see DEFAULT_CSS of widget
  classes the test harness never mounts
status: To Do
assignee: []
created_date: '2026-10-31 07:16'
labels:
  - console
  - performance
  - testing
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36571 made a Workbench status flip restyle only the node that carries the status class. That is correct only while no rule keys a descendant on a status class and no status-keyed rule sets a property the node's children read from it (colour, background, text style, opacity, tint, link styles). Its guard scans every rule the mounted Console loads plus every built top-level sheet, so a widget class that the harness never mounts and that ships unscoped DEFAULT_CSS (SCOPED_CSS = False) could add such a rule without the guard seeing it, and the header's children would then paint a stale style after a status flip. Low risk today: the two DEFAULT_CSS status rules found in the review (the dictation window and the audio troubleshooting dialog) are scoped.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The guard covers the DEFAULT_CSS of every Widget subclass defined under tldw_chatbook, mounted by the harness or not
- [ ] #2 A seeded unscoped DEFAULT_CSS rule that keys a descendant on a status class, or sets an ancestor-read property on one, fails the guard and names the widget class and selector
- [ ] #3 The guard passes on the shipped CSS and stays under a few seconds
<!-- AC:END -->
