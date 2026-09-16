---
id: TASK-32685
title: Invoke MCP-backed hooks through normal tool authority
status: To Do
assignee: []
created_date: '2026-09-16 04:29'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32680
  - TASK-32684
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete hook interoperability with MCP handlers that preserve recursion, initialization and permission boundaries.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MCP hooks use explicit typed input templates, current owned definitions and normal schema/profile/approval checks on already-connected eligible servers.
- [ ] #2 Error-first result normalization accepts only the specified structured, single-text, exact mirrored or empty forms and rejects ambiguous fallback, extra blocks and oversized metadata.
- [ ] #3 Provisional initialization resolves independently eligible dependencies and rejects static/dynamic cycles without dispatch or guard bypass.
- [ ] #4 Approval observations cannot recurse into prompts; teardown cannot connect or request approval; nested suspension respects resource tickets and causal depth four with cancellation controls.
<!-- AC:END -->
