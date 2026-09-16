---
id: TASK-32673
title: Stop and revoke plugin work within the requested scope
status: To Do
assignee: []
created_date: '2026-09-16 04:20'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32672
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users immediately stop plugin activity while preserving unrelated authorized work and reporting persistence honestly.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Disable here, global-default changes, Disable everywhere and uninstall invalidate exactly their specified scopes; old approvals and callbacks never revive on re-enable.
- [ ] #2 Live admission closes and host cancellation starts before waiting for trust unlock or storage writes; affected plugin cleanup callbacks are suppressed immediately.
- [ ] #3 Durable disable/removal success is distinct from a session-only block and confirmed local stop; retry and failure cannot reopen the live scope or release surviving resource ownership.
- [ ] #4 Uninstall removes only installation-owned grants and registrations, retains data by default and leaves independent credentials/connections owned by their existing services.
<!-- AC:END -->
