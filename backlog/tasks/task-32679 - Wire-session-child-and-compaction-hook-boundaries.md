---
id: TASK-32679
title: Wire session child and compaction hook boundaries
status: To Do
assignee: []
created_date: '2026-09-16 04:24'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32678
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the approved lifecycle events where their effects can be applied safely to the owning run and context.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SessionStart uses cancellable provisional admission and publishes dependent capabilities/context only after controlling requirements succeed; tab focus and package inspection emit no session events.
- [ ] #2 SubagentStart narrows inherited tools and budgets, and SubagentStop contributes only to an active parent checkpoint without restarting settled work.
- [ ] #3 PreCompact supplies required compactor input and PostCompact fences subsequent input after committed summaries; runtime context blocks are reassembled without duplication.
- [ ] #4 Manual-only UserPromptSubmit, idle hook-set replacement, SessionEnd, required-context failures and independent component success are covered at actual Console and agent boundaries.
<!-- AC:END -->
