---
id: TASK-19026
title: Preserve wide Library Notes browse continuity
status: In Progress
assignee: []
created_date: '2026-08-21 15:29'
labels:
  - library
  - ux
  - notes
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Library rail visible for wide Notes browsing while giving note editing and Files workspaces a focused full-width task surface that returns to the exact prior browse context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Wide Notes browse retains the Library rail at the existing breakpoint.
- [ ] #2 The note editor and Files workspace use a focused task surface with a persistent Library/Notes return cue.
- [ ] #3 Back restores Notes source, scope, selected identity, scroll, rail position, and semantic focus.
- [ ] #4 Dirty, sync, conflict, mutation, and Escape guards remain authoritative.
- [ ] #5 Compact Notes remains navigation-first, and resizing does not reset draft or browse context.
- [ ] #6 Only touched Notes and direct-owner tests are run; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

See `Docs/superpowers/plans/2026-08-21-library-notes-wide-browse-continuity.md`.

ADR required: no

ADR path: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`

Reason: this task implements ADR-076's approved responsive Notes presentation
inside the existing Library screen, Notes canvas, source, focus, and guard
owners. It changes no storage, sync/conflict policy, service contract, or
cross-module interface.
