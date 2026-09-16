---
id: TASK-32686
title: Register all native plugin capability types
status: To Do
assignee: []
created_date: '2026-09-16 04:29'
labels:
  - plugins
  - implementation
  - delivery
dependencies:
  - TASK-32672
  - TASK-32685
  - TASK-32684
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-marketplaces-and-ui.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose commands, rules, agents, hooks and complete skill metadata consistently through Chatbook runtime services.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Namespaced manual commands, always/manual rules, validated agent presets and owned hooks join the same immutable run snapshot with explicit component selection.
- [ ] #2 Instruction content remains attributed untrusted context through inline/forked skills and agents; required context overflow refuses whole selected material.
- [ ] #3 Tool inheritance, empty allowlists, model mappings, alias collisions and dependency closure preserve their declared semantics across composer/catalog/agent paths.
- [ ] #4 New enablement never adds capability midway through an existing run, and disabled or missing requirements remain visible without degrading unrelated selected components.
<!-- AC:END -->
