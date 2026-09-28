---
id: TASK-32688
title: Acquire bounded Git and local plugin snapshots
status: To Do
assignee: []
created_date: '2026-09-16 04:31'
labels:
  - plugins
  - implementation
  - delivery
dependencies:
  - TASK-32668
  - TASK-32674
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-marketplaces-and-ui.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fetch inspectable immutable package content from user-selected Git hosts without running repository-controlled checkout behavior.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 GitHub shorthand, explicit HTTPS/SSH origins, refs and subdirectories resolve to immutable commits with separate source and package provenance.
- [ ] #2 Acquisition uses checked Git argv, restricted transports and host authentication, with repository hooks, filters, LFS, recursive submodules and build/install execution disabled.
- [ ] #3 Fetch/materialization limits apply during work; traversal, remote local-file authority, escaping links, special files and cross-platform path collisions are rejected.
- [ ] #4 Cancellation, missing Git, moving refs, quota and authentication errors preserve the installed revision and redact credentials without blocking startup.
<!-- AC:END -->
