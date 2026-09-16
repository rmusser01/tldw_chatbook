---
id: TASK-32670
title: Authenticate complete plugin authority snapshots
status: To Do
assignee: []
created_date: '2026-09-16 04:17'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32669
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent package or registry tampering from changing activation, mappings or revocation without reviewed authority.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A separate plugin trust namespace authenticates complete registry snapshots, exact marker tuples and domain-separated prepared intents and commit certificates without resetting standalone skill trust.
- [ ] #2 Activation overrides, selection, requirements, execution mappings, credential-binding generations, revocations and data-root fences are authenticated; missing or mismatched evidence blocks use.
- [ ] #3 Snapshots are encrypted in the protected store outside package content and SQLite, and secrets remain credential references.
- [ ] #4 Locked or unavailable trust follows ADR-009 posture rules; successful trust, offline tamper, rollback, reset and standalone-skill controls use real crypto with an isolated marker store.
<!-- AC:END -->
