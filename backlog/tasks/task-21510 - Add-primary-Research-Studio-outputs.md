---
id: TASK-21510
title: Add primary Research Studio outputs
status: To Do
assignee: []
created_date: '2026-08-24 05:54'
updated_date: '2026-08-24 05:54'
labels:
  - research
  - workspace
  - studio
  - artifacts
dependencies:
  - TASK-21507
  - TASK-21508
  - TASK-21509
references:
  - Docs/superpowers/specs/2026-08-23-research-workspace-design.md
  - Docs/superpowers/plans/2026-08-23-research-workspace-primary-studio.md
  - backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Summary, Flashcards, Quiz, Report, and Compare Sources as traceable Research Studio creations that save to and reopen from their existing canonical owners instead of a duplicate workspace-output database.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The five primary output cards project explicit availability, owner, reason, and recovery from current authority, source readiness, processing route, and capability revision; Compare requires at least two ready sources.
- [ ] #2 A field-level owner mapping covers ID, version, source snapshot/provenance, generation configuration, workspace association, reopen route, replacement/new-version behavior, and deletion for every primary output before implementation.
- [ ] #3 Local Summary, Report, and Compare save as Local Chatbook artifacts plus workspace membership; local Flashcards save as Study deck/cards plus membership; local Quiz saves as Quiz record/questions plus membership.
- [ ] #4 Server outputs generate and persist through the supported server workspace artifact/study contracts and never masquerade a client download as server persistence.
- [ ] #5 Generation captures an immutable qualified workspace/source/version/config snapshot, fences stale UI results, supports cancel/retry, and reports exact processing route and saved destination.
- [ ] #6 Studio history is derived from canonical records and memberships through `WorkspaceOutputRef`; no parallel output row becomes the version, delete, or list owner.
- [ ] #7 View/edit/export, regenerate-replace, regenerate-new-version, discuss in Grounded Chat, save/append to Quick Notes, delete, safe undo, and canonical reopen are available only where the owning service supports them.
- [ ] #8 Source-version changes mark existing outputs `Sources changed since generation` while preserving inspection; regeneration uses a fresh source snapshot.
- [ ] #9 Targeted owner-adapter, field-mapping, generation, persistence, reopen, version, delete, no-blending, mounted Studio, and real canonical-owner round-trip tests pass without a full-suite claim.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 already chooses each canonical local/server owner and forbids a duplicate output store. This task implements those mappings; a new ADR is required if any primary output receives a new canonical database.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-primary-studio.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->
