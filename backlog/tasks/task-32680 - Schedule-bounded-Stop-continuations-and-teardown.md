---
id: TASK-32680
title: Schedule bounded Stop continuations and teardown
status: To Do
assignee: []
created_date: '2026-09-16 04:25'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32679
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow useful automatic follow-up while preserving user priority, cancellation and finite scheduler ownership.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Stop proposals combine into at most one scheduler turn keyed by parent turn and event, with three-turn and 120-second chain caps and inherited restrictions.
- [ ] #2 Foreground user work, vetoes, update drain, revocation, closure and uncertain dispatch prevent stale continuation or automatic replay; continuations do not fire UserPromptSubmit.
- [ ] #3 Interrupt and SessionEnd are bounded observations after admission sealing, cannot prompt/connect or extend cleanup, and revoked plugin handlers are suppressed.
- [ ] #4 Mounted and viewless tests exercise continuation admission, deduplication, concurrent settlement and teardown with legacy hook behavior preserved.
<!-- AC:END -->
