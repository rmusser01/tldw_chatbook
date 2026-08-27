---
id: TASK-22513
title: Polish Library Notes work-first editors and Folder Files shell
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-27 06:04'
labels:
  - library
  - notes
  - ui
dependencies:
  - TASK-22032
  - TASK-19001
references:
  - Docs/superpowers/specs/2026-08-26-library-notes-ux-improvements-design.md
  - backlog/decisions/086-library-adaptive-reader-shell.md
  - backlog/decisions/076-library-lifecycle-progressive-disclosure.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make both Library Notes authorities calmer and more efficient for sustained writing while preserving database and on-disk authority, state, recovery, and every incumbent capability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Database Notes and Folder Files use the shared adaptive reader shell as their only geometry and focus-evacuation owner, including an independently collapsible Folder Files tree.
- [ ] #2 Work-first Library collapse activates once per approved Notes work session, manual expansion wins, responsive changes never persist, and all reset predicates are deterministic.
- [ ] #3 Database list and Folder tree visibility and width preferences use independent normalized keys, Settings controls, environment overrides, and race-safe persistence authorities.
- [ ] #4 Database Notes retains Edit, Preview, Info, and all navigator workflows; Folder Files exposes Edit and Manage, retains autosave and recovery, and gains neither Markdown Preview nor a manual Save control.
- [ ] #5 Only the two note-body editors retain their resting background on focus and use a geometry-stable heavy outline with verified theme contrast.
- [ ] #6 Primary headers preserve authority and consequential status, apply the approved status precedence, and keep safe recovery actions visible without truncation.
- [ ] #7 The Notes-specific Ctrl+S binding and hints are removed without adding a replacement shortcut; visible Save remains keyboard reachable.
- [ ] #8 Targeted reducer, configuration, Settings, shell, Notes, Folder Files, accessibility, CSS, and live isolated TUI verification pass.
- [ ] #9 Library Notes and Folder Files user documentation reflects the final pane, mode, focus, save, and shortcut behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-08-26-library-notes-work-first-ux.md
Design: Docs/superpowers/specs/2026-08-26-library-notes-ux-improvements-design.md
ADR required: no
ADR path: N/A
Reason: this directly implements ADR-086's shared adaptive-reader shell and ADR-076's progressive-disclosure boundaries without changing storage, sync/conflict policy, service ownership, security, dependencies, or cross-module contracts.
Existing ADRs: backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/076-library-lifecycle-progressive-disclosure.md
<!-- SECTION:PLAN:END -->
