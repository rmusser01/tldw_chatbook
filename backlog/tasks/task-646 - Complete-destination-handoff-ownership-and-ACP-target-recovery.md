---
id: TASK-646
title: Complete destination handoff ownership and ACP target recovery
status: To Do
assignee:
  - '@codex'
created_date: '2026-07-26 13:37'
labels:
  - architecture
  - state
  - reliability
dependencies:
  - TASK-645
references:
  - backlog/decisions/026-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-application-session-state-ownership-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining Study, Artifacts, and ACP navigation handoffs to the revisioned memory-only owner, repair the missing ACP target consumer, remove the dead Notes slot, and close the application-level ownership boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typed Study, Artifacts, and ACP slots preserve revisioned single-slot replacement semantics
- [ ] #2 Artifacts transient failures release for later lifecycle or user retry while success and terminal missing-target outcomes acknowledge
- [ ] #3 ACP consumes its target by matching the current runtime session and provides explicit stale or unsupported recovery instead of silently losing the target
- [ ] #4 All listed raw application pending fields and the dead pending_notes_workspace_context slot are removed and an AST ownership guard prevents their return
- [ ] #5 Focused and static checks plus the installed-wheel gate, product-maturity UI sentinel, and full suite pass after the integrated tranche
<!-- AC:END -->
