---
id: TASK-32675
title: Delete plugin data only after exact-root users drain
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:21'
updated_date: '2026-10-01 03:09'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32673
  - TASK-32674
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove only the reviewed saved data after every owned user has stopped, including processes that can write while idle.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deletion review binds installation identity, exact owned roots, generations and all affected workspaces; a persisted access fence precedes destructive deletion.
- [x] #2 Shared-root users and idle writer-capable processes are tracked until confirmed stopped; unresolved writers or handles leave deletion pending without forced cross-workspace cancellation.
- [x] #3 Reattachment, stale root generations, symlink replacement and PID reuse cannot redirect cleanup; partial cleanup/restart retains the fence and honest progress.
- [x] #4 Successful deletion advances root generation before fresh authorized use; cancellation cannot claim data restoration or revive cancelled work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: implements accepted exact-root custody, authenticated clean/dirty runtime checkpoints and retained destructive phases through existing authority/runtime owners.
1. Port reviewed F8 fixtures with owned-checkout child isolation and establish RED after real install preconditions.
2. Integrate reviewed F8 increment preserving current worker/profile, consent, continuation and retention contracts.
3. Qualify exact-root creation/deletion, native identity, idle/shared users, failed persistence and cleanup, surviving-process recovery, shutdown and managed-resume constraints with positive controls.
4. Run targeted neighbors, owned lint/format, shared baseline comparison and packaged migration checks; self-review ACs, document platform limits, complete via CLI and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated reviewed F8 exact-root creation, grants, deletion/attachment/reconciliation and clean/dirty runtime checkpoints through existing authority/runtime owners. Registry v4 retains original process-root joins; shutdown fences ordinary work before final clean publication, and F7 resume compares actual root custody. Preserved isolated child provenance and fresh-process fork guard. ADR-162/163. Core 55 passed 289.47s; neighbors 250 passed 513.89s, no warnings/skips. All 46 owned Python files lint/format; changed syntax/whitespace pass. Built uninstalled wheel contains all four exact migration resources. Self-reviewed all ACs and documented local macOS/APFS, simulated boot-change and external-writer limits in Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md. Actual hook/MCP grant producers remain H6/M4.
<!-- SECTION:NOTES:END -->
