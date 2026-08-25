---
id: TASK-21508
title: Add Research Sources ingest association and Quick Notes
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 05:54'
updated_date: '2026-08-25 04:55'
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
- Implemented ADR-078 across the separate F10 Research Workspace: qualified Local or Server ownership, durable staged source operations, canonical catalog ownership before idempotent workspace association, association-only removal, independent readiness, device-only organization, and canonical Quick Notes with conflict and navigation guards.
- Added real SQLite Local round trips and a production-seam Server round trip through app dispatch, registry reconciliation, terminal listener, scheduler, coordinator, generated My Media catalog, and exact workspace-source association. Injected Local Media and registry spies prove zero Server-path Local calls; profile, principal, and result-ID mismatches fail closed.
- Fixed asynchronous Library backend switching so the canvas keeps rendering its persisted owner while pending intent sequences rapid clicks; writes serialize, and completion repaints only on the Textual UI loop for the current generation. Delayed success, failure recovery, and rapid Server to Local to Server tests use the real decorated worker.
- Corrected installed migration packaging through v43 and updated the Research Workspace guide with exact navigation, authority, controls, canonical ownership, recovery, privacy, and limitations.
- Fix-round verification: 188 changed-area UI and integration checks passed; remote or Server runner slice 20 passed; installed-distribution packaging 43 passed. Four new review inverses each failed and were restored: missing completion repaint, missing generation fence, Server ID copied to Local media, and constant fake catalog.
- Default Ruff passed across the fix inventory and UP017 passed across the six Task 6 files plus the Fix 1 unit test. The broader UP selector retains 11 findings only on lines unchanged from the pre-Task-6 base; a full production-screen UP017 probe retains four pre-existing findings outside the fix hunks. No whole-file upgrade-selector claim is made. Changed-range format, compileall, package artifact probes, privacy or no-blend scans, Impeccable detector, and diff checks passed.
- The full Library screen file has 31 passing checks and one stale top-button selector failure reproduced identically at exact base ebf80f954; all four changed backend or owner tests pass. The full pytest suite was not run under repository policy.
- Isolated F10 live smoke used temporary config, XDG, and data roots. No test Server API was available, so live Server behavior was not attempted or claimed.
- ADR check: no new ADR was required; backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md remains governing.
<!-- SECTION:NOTES:END -->
