---
id: TASK-32323
title: >-
  Inspector rail per-section collapse
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review A4. The Inspector rail is one giant scroll unit with no per-section collapse (right_rail.py module docstring notes the whole rail is a single collapse/expand unit), unlike the left rail's seven disclosure sections. Introduce per-section headers for at least Sources / Run / Session Settings, persisted like the left rail's section flags.

Filed from the 2026-09-10 Console rail UX review (review item A4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Inspector body is divided into named collapsible sections (Sources+Scope, Run, Session Settings at minimum) using the established DestinationRailSectionHeader/ConsoleInspectorSection pattern
- [ ] #2 Section open/closed state persists per workspace alongside existing rail preferences
- [ ] #3 A collapsed section still renders its header and a one-line summary when one exists
- [ ] #4 Existing ids consumed by screen sync methods (console-staged-context-tray, retrieval scope ROW_ID, console-run-inspector-state, console-settings-summary) keep working without changes to those sync methods
<!-- AC:END -->
