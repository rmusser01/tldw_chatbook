---
id: TASK-32678
title: Integrate hook input transformations and post-event barriers
status: To Do
assignee: []
created_date: '2026-09-16 04:24'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32677
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply structured hook effects at the real tool boundary without bypassing review or allowing later model steps to overtake required context.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Transformers run once in order, final arguments are frozen, qualified guards and context handlers run, then ordinary permission review binds the actual call.
- [ ] #2 Preauthorized and durable tool paths retain their existing exemptions while all controlling restrictions and fresh dispatch checks apply.
- [ ] #3 Dispatched results establish required PostToolUse and known-error PostToolUseFailure checkpoints before subsequent model admission or normal settlement.
- [ ] #4 Validated effect acceptance and checkpoint release are atomic; failure, cancellation, master-off or stale results cannot erase a requirement or replay settled tool work.
<!-- AC:END -->
