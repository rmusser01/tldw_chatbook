---
id: TASK-32721
title: Implement verified llama.cpp launch and Console adoption
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 16:11'
updated_date: '2026-09-17 16:57'
labels:
  - llamacpp
  - lab
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make local or explicitly selected llama.cpp servers usable in Console only after current health and exact-model verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local launches use a reserved non-path model alias and canonical absent-value port 8080 while preserving explicit values.
- [x] #2 Health and exact-model checks fence stale results and distinguish process liveness from API readiness.
- [x] #3 Use in Console adopts only the active session; Make default uses canonical Settings and never silently persists.
- [x] #4 Existing GGUF leases and manual snapshots remain correct through launch, stop, cancellation, and navigation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/165-llamacpp-tuning-profiles-and-startup-diagnostics.md
Reason: implements ADR-114 with explicit tuning persistence and bounded diagnostics under ADR-165.

Follow the corresponding slice of Docs/superpowers/plans/2026-09-17-llamacpp-management-milestone.md and its linked design. Write failing targeted behavioral tests, implement the minimal owned interfaces, integrate the mounted flow, and verify the affected lifecycle, source, snapshot and settings boundaries. Record review and qualification limits before closeout.

## Renumbering provenance

The Backlog CLI assigned 32712 below the live all-ref/all-worktree maximum 32720. Renumbered immediately before implementation to 32721 to avoid existing branch reservations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact-generation health/model verification, safe model IDs, reserved launch alias and absent-value port 8080, occupied-listener admission, and session-only Console / explicit Settings draft handoffs. Reused existing compensation and process-lease ownership; preserved manual snapshots. Added mounted navigation and live llama-server launch/verify/chat/Stop qualification. ADR: backlog/decisions/165-llamacpp-tuning-profiles-and-startup-diagnostics.md (extends ADR-114/119). Files: llama connection/handoff modules, pending store, launcher/lifecycle, Models, Console and Settings integration. Evidence and known unrelated failures: Docs/superpowers/reviews/2026-09-17-llamacpp-milestone-verification.md. User guide: Docs/User_Guide/lab.md. No full sweep or commit; unrelated work preserved.
<!-- SECTION:NOTES:END -->
