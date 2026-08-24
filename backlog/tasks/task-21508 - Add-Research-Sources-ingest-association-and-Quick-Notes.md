---
id: TASK-21508
title: Add Research Sources ingest association and Quick Notes
status: To Do
assignee: []
created_date: '2026-08-24 05:54'
updated_date: '2026-08-24 05:54'
labels:
  - research
  - workspace
  - sources
  - notes
dependencies:
  - TASK-21507
references:
  - Docs/superpowers/specs/2026-08-23-research-workspace-design.md
  - Docs/superpowers/plans/2026-08-23-research-workspace-sources-quick-notes.md
  - backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Research Sources and Quick Notes durable, authority-correct workbench features, with every intake first landing in the selected authority's general catalog and then being associated to the captured workspace by stable identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import/Upload, URL, Paste, and existing Library/My Media attachment converge on one durable source-operation contract that captures the qualified target before work starts.
- [ ] #2 Local intake creates or reuses a general local Library item and then idempotently links `WorkspaceMembership(role=source)`; Server intake creates or reuses server Media and then idempotently creates the server workspace-source row, with no cross-authority item creation or fallback.
- [ ] #3 Restart and retry resume from the stored canonical item and failed stage; receipts independently report catalog, association, and readiness/indexing outcomes, and a successful catalog write survives later-stage failure.
- [ ] #4 Removing a source unlinks only the workspace association and selected retrieval scope; Library/Media deletion is a separate owner-routed action and multi-workspace ownership is disclosed before deletion.
- [ ] #5 Source search, filters, sorting, pagination, selection, preview, readiness, status, reorder, and batch association controls operate through the selected authority's real adapter and fail closed when unsupported.
- [ ] #6 Desired source selection is persisted separately from effective retrieval readiness; FTS-only state is labeled honestly and never presented as Hybrid/vector ready.
- [ ] #7 Source folders and annotations persist only in the private device overlay, are labeled device-only, and never become filesystem roots, canonical memberships, or cross-device server content.
- [ ] #8 Quick Notes create, list, search, edit, delete, and resolve conflicts through Local Notes plus workspace membership or the canonical Server workspace-notes API; message/source provenance is retained without a parallel note store.
- [ ] #9 Historical database migration, duplicate reuse, late completion after navigation, partial failure, idempotent replay, unlink-without-delete, no-blending, mounted UI, and real Library/Media round-trip tests pass within the targeted verification boundary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 already fixes canonical catalog ownership, stable workspace association, device-overlay ownership, partial-failure behavior, and no-blending. The migrations implement that accepted contract without changing it.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-sources-quick-notes.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->
