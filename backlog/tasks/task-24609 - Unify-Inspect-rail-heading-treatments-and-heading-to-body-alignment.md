---
id: TASK-24609
title: Unify Inspect rail heading treatments and heading-to-body alignment
status: To Do
assignee: []
created_date: '2026-08-30 00:54'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
  - css
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail uses five different heading treatments, and console-settings-title declares neither text-style nor color with no rule in scope to supply them, so the Session Settings title renders identically to the eight rows it heads. The focus cue adds a sixth bold treatment on top. Separately, section headings render at a two-column indent while their own body rows render at one, so every heading is off-axis from its content by one cell.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every section title in the Inspect rail uses one shared treatment
- [ ] #2 The raised-background treatment is reserved for run-inspector sub-groups only
- [ ] #3 Headings and their body rows share a left alignment column
- [ ] #4 The focus cue remains visually distinct from every heading treatment
<!-- AC:END -->
