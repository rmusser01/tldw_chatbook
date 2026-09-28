---
id: TASK-32674
title: Drain plugin work before applying updates and rollback
status: To Do
assignee: []
created_date: '2026-09-16 04:21'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32673
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply one reviewed package revision without overlapping incompatible active work or silently restoring historical permissions.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Applying update fences new old-revision admission, shows leases and approvals, and prevents Stop continuations from extending the drain.
- [ ] #2 Users can wait, cancel before commitment or explicitly cancel affected work; updates publish only after confirmed drain and cleanup.
- [ ] #3 Rollback uses the same review and commit gates with current disables, mappings and grants; shared-data and external-effect rollback remain explicitly unsupported.
- [ ] #4 New or newly supported components stay unselected, immutable prior packages remain available for comparison, and quota/retention protects live and recovery material.
<!-- AC:END -->
