---
id: TASK-32598
title: Keep Library section toggles visibly focused above the rail scroll cue
status: To Do
assignee: []
created_date: '2026-09-15 02:48'
labels:
  - library
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-library-workflow-audit.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 80x24, keyboard Tab from Search/RAG focuses Create underneath the opaque scroll-for-more row. Its glyph and focus cue disappear while Enter still collapses the section. Reproduced on baseline 2939afda63 in both themes with production styles and in a native private-profile app. The focused toggle region is (22,21,3,1), its painted crop is three spaces, and the cue covers (2,21,24,1). Task-32219 established the useful fold cue; the repair should retain its discoverability while keeping keyboard targets visible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 80x24 in textual-dark and textual-light, every rail section toggle reached by Tab paints its label or glyph and an unambiguous focus cue before it can be activated.
- [ ] #2 Tab and Shift+Tab bring focused controls above the docked fold cue, including Create, Import/Export and Details Diagnostics, without changing the selected destination.
- [ ] #3 Enter activates the visibly focused section and preserves usable focus through 120-to-80-to-120 resizing.
- [ ] #4 The fold cue still truthfully indicates additional content, and verification includes compositor paint and a native keyboard reproduction rather than region containment alone.
<!-- AC:END -->

## ID allocation provenance

The local CLI initially offered TASK-32597. Before references or publication, this new task moved to TASK-32598 after a fresh all-ref history and worktree sweep found IDs through 32597. No existing task was renumbered.
