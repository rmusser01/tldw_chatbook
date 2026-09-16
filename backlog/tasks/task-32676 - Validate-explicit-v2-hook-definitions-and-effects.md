---
id: TASK-32676
title: Validate explicit v2 hook definitions and effects
status: To Do
assignee: []
created_date: '2026-09-16 04:22'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32645
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make richer hook declarations inspectable and deterministic while preserving the legacy six-event configuration.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Legacy hooks.hook parsing and stdout behavior remain unchanged; hooks.handler and native v2 files validate the exact event/effect/matcher/template contracts.
- [ ] #2 Required success, required nonempty context and dependency-scoped requirements stay distinct, and invalid controlling requirements never disappear through optional parsing.
- [ ] #3 PreToolUse phase classification is exhaustive, including mixed transformer/deny, final guards, context-only handlers and effect-free required completion.
- [ ] #4 Payload and result bounds reject whole invalid batches, reserved ownership fields cannot be supplied by output, and malformed teardown/approval/Stop requirements are refused.
<!-- AC:END -->
