---
id: TASK-32671
title: Commit and recover reviewed plugin installation changes
status: To Do
assignee: []
created_date: '2026-09-16 04:18'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32670
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make reviewed local installation and authority changes recoverable without accidentally publishing uncommitted capabilities.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Review binds exact package/catalog digests, installation identity, interpretation, selection, workspace and authority inputs; stale reviews cannot commit.
- [ ] #2 Publication follows prepared snapshot and intent, durable SQLite commit, post-commit certificate, secure marker, then projections; retries use the same operation identity.
- [ ] #3 Crash recovery follows every specified evidence state in another process, preserving unrelated installations and refusing automatic promotion when commitment proof is missing.
- [ ] #4 Full disks, missing responses and interrupted writes preserve old or fenced committed state without recreating grants or executing package code; recovery material cannot be pruned.
<!-- AC:END -->
