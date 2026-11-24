---
id: TASK-21510
title: Add primary Research Studio outputs
status: In Progress
assignee:
- '@codex'
created_date: 2026-08-24 05:54
updated_date: 2026-11-24 18:56
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
- backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Summary, Flashcards, Quiz, Report, and Compare Sources as traceable Research Studio creations that save to and reopen from their existing canonical owners instead of a duplicate workspace-output database.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The five primary output cards project explicit availability, owner, reason, and recovery from current authority, source readiness, processing route, and capability revision; Compare requires at least two ready sources.
- [x] #2 A field-level owner mapping covers ID, version, source snapshot/provenance, generation configuration, workspace association, reopen route, replacement/new-version behavior, and deletion for every primary output before implementation.
- [x] #3 Local Summary, Report, and Compare save as Local Chatbook artifacts plus workspace membership; local Flashcards save as Study deck/cards plus membership; local Quiz saves as Quiz record/questions plus membership.
- [ ] #4 Server outputs generate and persist through the supported server workspace artifact/study contracts and never masquerade a client download as server persistence.
- [x] #5 Generation captures an immutable qualified workspace/source/version/config snapshot, fences stale UI results, supports cancel/retry, and reports exact processing route and saved destination.
- [x] #6 Studio history is derived from canonical records and memberships through `WorkspaceOutputRef`; no parallel output row becomes the version, delete, or list owner.
- [x] #7 View/edit/export, regenerate-replace, regenerate-new-version, discuss in Grounded Chat, save/append to Quick Notes, delete, safe undo, and canonical reopen are available only where the owning service supports them.
- [x] #8 Source-version changes mark existing outputs `Sources changed since generation` while preserving inspection; regeneration uses a fresh source snapshot.
- [x] #9 Targeted owner-adapter, field-mapping, generation, persistence, reopen, version, delete, no-blending, mounted Studio, and real canonical-owner round-trip tests pass without a full-suite claim.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-322 already chooses each canonical local/server owner and forbids a duplicate output store. This task implements those mappings; a new ADR is required if any primary output receives a new canonical database.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-primary-studio.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implements the five Local Studio outputs and the three versioned Server text outputs under ADR-322. Summary, Report, and Compare Sources save in Local Artifacts or Server workspace artifacts. Local Flashcards and Quiz save in Study and reopen the exact canonical record. WorkspaceDB v11 stores bounded generation metadata on the Local membership. Server text outputs retain the source and citation identities on the artifact. The review and export actions use exact Server artifact versions.

Studio exposes processing and owner availability, source-version warnings, cancel and retry, owner history, versioned edits and replacement, new versions, Grounded Chat draft handoff, Quick Note save and append, and owner-filtered delete and export. A retry repairs a replacement whose owner committed before the workspace link did. Pane tests exercised 160×40 down to 60×20. Local integration used real Library Media, Chatbook, Study, Quiz, and workspace stores. The loopback check used the real Server HTTP routers, temporary databases, and the built-in test provider. It confirmed server generation, artifact save, replay, replacement, accepted export, and review invalidation after edit.

The Server Flashcards and Quiz criterion remains open. The current Server Study routes assign owner IDs on separate create requests and provide no reliable operation lookup or complete generation provenance to recover a lost create response. Studio displays an unavailable capability for those two Server cards. The adjacent Study screen tests reproduced the same 14 profile-bootstrap failures on the unchanged base. Targeted Studio, owner, design-token, and module-size checks passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
