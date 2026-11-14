---
id: TASK-33015
title: Search local archive recovery candidates with bounded results
status: Done
assignee:
  - '@codex'
created_date: '2026-09-27 15:22'
updated_date: '2026-09-27 16:40'
labels: []
dependencies:
  - TASK-33014
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Find missing archived chats across local workspaces without an embedding index.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Search includes chat and workspace-only archives while excluding deleted records and private control content.
- [x] #2 Either applicable archive event can match date filters; unknown times and live continuation are explicit.
- [x] #3 Storage work and output are bounded; stale or incomplete queries never return false complete results.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md. Reason: approved bounded cross-workspace archive search contract. Follow Task 2 in Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md: real-store discovery/date red cases, immutable contracts, bound workspace snapshot query and role-safe text projection, keyset/byte/deadline matrix, scoped validation and commit. Use /tmp/archive-recovery-venv/bin/python; targeted tests only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded cross-workspace recovery search with immutable contracts, parameterized pre-limit eligibility/date filtering, role-safe literal matching, lifecycle workspace digest, keyset continuation and bounded SQL/text projections. Five-second deadline covers both stores and projection; temporary handlers and remaining-budget lock waits restore connection settings. Added 47 passing scoped real-SQLite tests including cancellation, locks, query plans and byte-boundary continuation. ADR-166 applies; no new ADR. Detailed evidence and downstream 30-KiB page/32-KiB final-envelope and dual-store worker contracts are in .superpowers/sdd/2026-09-27-agent-archive-recovery/task-2-report.md. New modules/tests pass Ruff lint/format; narrow legacy delegates add zero baseline diagnostics. Independent review pending; leave In Progress.

Review fix round 1: shared body matching/projection now trims the full bound Unicode whitespace/BOM set before rejecting ambiguous serialized private content. Added opt-in native operation_deadline through both owning connection APIs and existing private SQLite actor admission, with remaining-budget cold setup/probes and cleanup/unregistration of failed newly opened handles; default callers retain normal timeouts and ownership gates. RED: 38 review regressions failed, including real 1.059s/1.027s cold waits with a 0.15s budget. GREEN: 38 focused regressions passed; final scoped archive plus affected connection coverage passed 131 tests, and 3 existing Chat liveness tests passed. Cold waits now measure 0.1646s/0.1720s with failed-handle cleanup and subsequent reuse verified. Zero new legacy lint diagnostics. Full evidence appended to task-2-report.md; independent re-review pending, remain In Progress.

Independent Task 2 review and fix-round re-review approved the implementation. Fixed Unicode-whitespace private-body leakage and cold-connection deadline gaps with real-store RED/GREEN evidence. Final fix qualification: 131 targeted tests plus three liveness tests pass; no introduced lint diagnostics. Final envelope fitting and dual-store runtime dispatch are assigned to later plan tasks. ADR-166 applies; task complete.
<!-- SECTION:NOTES:END -->
