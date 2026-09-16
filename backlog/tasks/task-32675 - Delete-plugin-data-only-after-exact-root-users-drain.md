---
id: TASK-32675
title: Delete plugin data only after exact-root users drain
status: To Do
assignee: []
created_date: '2026-09-16 04:21'
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
- [ ] #1 Deletion review binds installation identity, exact owned roots, generations and all affected workspaces; a persisted access fence precedes destructive deletion.
- [ ] #2 Shared-root users and idle writer-capable processes are tracked until confirmed stopped; unresolved writers or handles leave deletion pending without forced cross-workspace cancellation.
- [ ] #3 Reattachment, stale root generations, symlink replacement and PID reuse cannot redirect cleanup; partial cleanup/restart retains the fence and honest progress.
- [ ] #4 Successful deletion advances root generation before fresh authorized use; cancellation cannot claim data restoration or revive cancelled work.
<!-- AC:END -->
