---
id: TASK-32677
title: Execute v2 command hooks with bounded resource ownership
status: To Do
assignee: []
created_date: '2026-09-16 04:23'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32676
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run reviewed hook commands with fair application-wide limits and cancellation that retains ownership until actual cleanup.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both synchronous agent-thread and asynchronous Console entries execute argv-only handlers with bounded input/output capture and no implicit shell.
- [ ] #2 The exact per-runtime and application execution, reservation and observation limits include provisional, nested, late and cleanup-pending work with fair admission.
- [ ] #3 Cancellation closes admission immediately, keeps local children counted until reaped, and separates notification deadlines from the five-second host-reap allowance.
- [ ] #4 Real controlled child processes establish successful execution and kill/reap controls; timeout, failed launch, cancelled waiters and surviving-child paths preserve honest outcomes and metadata-only diagnostics.
<!-- AC:END -->
