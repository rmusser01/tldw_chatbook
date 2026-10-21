---
id: TASK-36003
title: Scan project folders in bounded resumable pages
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 14:30'
updated_date: '2026-10-21 16:27'
labels: []
dependencies:
  - TASK-36002
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-keyword-index.md
  - backlog/decisions/240-workspace-project-context-and-folder-bound-index.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Discover large local project trees without unbounded memory, stale authority or leaked excluded content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pages stay within256metadata entries, two reads,4MiB queued text and closed bounded pipe frames; eligible UTF8 files are at most1MiB.
- [x] #2 Sensitive/excluded, binary/generated and symlink/junction/reparse content is skipped before descent/read; changed root or revoked unit authority stops new reads.
- [x] #3 Incomplete directories and interruptions remain truthful and resumable without duplicate publication or false deletion proof.
- [x] #4 Real worker retirement and20k wide/many-directory corpus receipts demonstrate resource bounds and report actual platform qualification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement approved K2 bounded stdlib-only scanner and closed protocol with real contained-process/root-pin transport, per-unit authority admission and narrowing exclusion floor, same-read capture helper and private Capture operation, durable completion receipts and bounded backpressure. Consume K1 immutable records; emit raw ScanPage and leave chunk preparation/publication/claims to K3. TDD first, then actual worker/root-pin and 20k-file/directory/wide-corpus qualification with resource/cancellation/retirement receipts, scoped lint and independent review. ADR required: yes, existing ADR240-workspace-project-context-and-folder-bound-index applies with ADR101/102/174/181; no new ADR. Preserve Library choices; no semantic/index UI/service or ordinary one-shot protocol changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented closed stdlib-only native scanner, bounded per-unit authority/narrowing admission, descriptor-based shared scan/Capture, strict UTF8 and immutable page handoff. Commits db10df9a25 and 6cbee38fe0 independently approved; I1 actual sender-slice overlap fixed with audited64KiB headroom and deterministic real-pipe ASCII/UTF8 regression.30 runnable cases have passing final evidence across covering/corrected-node runs;3 Windows platform skips and inherited warnings retained. Existing root-pin covering37passed3skipped; native/profile and exact source timing limits recorded. Three20k corpora completed with ~31-32ms cancellation and actual PID/tree/pipes retirement; historical sampledRSS/timing predates headroom arithmetic, not exact OS peaks. Scoped Ruff/format/diff pass. Evidence Docs/superpowers/qa/2026-10-20-workspace-project-context/keyword/task-2-report.md, task-2-review.md, task-2-re-review-1.md and k2-evidence.json. ADR240/101/102/174/181 linked; saved Library behavior untouched, semantic work excluded. K3 owns durable writer/restart/quota/production-publication gates; K4 owns permission-reviewed disclosure. Windows report-encoding incident added to existing testing evidence lesson.
<!-- SECTION:NOTES:END -->
