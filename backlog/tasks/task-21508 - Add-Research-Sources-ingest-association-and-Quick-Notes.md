---
id: TASK-21508
title: Add Research Sources ingest association and Quick Notes
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 05:54'
updated_date: '2026-08-24 21:01'
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
- [x] #1 Import/Upload, URL, Paste, and existing Library/My Media attachment converge on one durable source-operation contract that captures the qualified target before work starts.
- [x] #2 Local intake creates or reuses a general local Library item and then idempotently links `WorkspaceMembership(role=source)`; Server intake creates or reuses server Media and then idempotently creates the server workspace-source row, with no cross-authority item creation or fallback.
- [x] #3 Restart and retry resume from the stored canonical item and failed stage; receipts independently report catalog, association, and readiness/indexing outcomes, and a successful catalog write survives later-stage failure.
- [x] #4 Removing a source unlinks only the workspace association and selected retrieval scope; Library/Media deletion is a separate owner-routed action and multi-workspace ownership is disclosed before deletion.
- [x] #5 Source search, filters, sorting, pagination, selection, preview, readiness, status, reorder, and batch association controls operate through the selected authority's real adapter and fail closed when unsupported.
- [x] #6 Desired source selection is persisted separately from effective retrieval readiness; FTS-only state is labeled honestly and never presented as Hybrid/vector ready.
- [x] #7 Source folders and annotations persist only in the private device overlay, are labeled device-only, and never become filesystem roots, canonical memberships, or cross-device server content.
- [x] #8 Quick Notes create, list, search, edit, delete, and resolve conflicts through Local Notes plus workspace membership or the canonical Server workspace-notes API; message/source provenance is retained without a parallel note store.
- [x] #9 Historical database migration, duplicate reuse, late completion after navigation, partial failure, idempotent replay, unlink-without-delete, no-blending, mounted UI, and real Library/Media round-trip tests pass within the targeted verification boundary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 already fixes canonical catalog ownership, stable workspace association, device-overlay ownership, partial-failure behavior, and no-blending. The migrations implement that accepted contract without changing it.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-sources-quick-notes.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented the ADR-078 contract across the separate F10 Research Workspace screen: explicit qualified Local/Server authority, durable source operations, canonical Library/Media ownership before idempotent workspace association, independent readiness, association-only removal, and device-only source organization.
- Added the complete Sources workbench and canonical Quick Notes surface, including owner-routed search and mutations, receipts and retry recovery, desired-versus-effective readiness, optimistic conflicts, provenance, navigation guards, and honest disabled states where the canonical Server contract cannot safely provide an operation.
- Added real SQLite and Server-fake round-trip coverage for duplicate reuse, captured-workspace late completion, bounded restart resume, association failure retention, unlink-without-delete, tags-not-membership, exact Server identity, and no Local blending. Required inverse mutations were each observed failing their named guard and restored.
- Corrected installed-distribution migration packaging so the v40-to-v43 runtime chain is present and directly exercised from source, sdist, and wheel installations. Updated the Research Workspace user guide with the exact controls, ownership, recovery, privacy, and current limitations.
- Targeted verification: DB/Workspace/Research/integration 444 passed; App/Library/Notes 600 passed; Library runner 146 passed with one Windows-only skip; Research UI 171 passed; shell Research 6 passed; API/private-path 131 passed; Packaging 43 passed; Library canvas 136 passed; CSS/parity 65 passed. Ruff lint passed for all 105 changed Python files; scoped format, changed-production compileall, migration artifact parity, privacy/no-blend scans, Impeccable detector, and `git diff --check` passed. The broad legacy formatter inventory still reports 47 pre-existing whole-file candidates and was not mechanically rewritten.
- Isolated live verification used temporary config, data, cache, and XDG directories and opened the production F10 destination as `ResearchWorkspaceScreen`. No test Server API was available, so live Server verification was not attempted and is not claimed. The full pytest suite was not run, per repository policy.
- ADR check: no new ADR was required; `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md` remains the governing decision.
<!-- SECTION:NOTES:END -->
