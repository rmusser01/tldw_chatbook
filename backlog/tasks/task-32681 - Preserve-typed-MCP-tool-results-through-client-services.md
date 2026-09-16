---
id: TASK-32681
title: Preserve typed MCP tool results through client services
status: To Do
assignee: []
created_date: '2026-09-16 04:26'
labels:
  - plugins
  - implementation
  - mcp
dependencies:
  - TASK-32645
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-mcp.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retain the protocol information needed to distinguish tool errors from successful structured results while keeping ordinary display compatible.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Stdio/client/service tool results retain content, structuredContent, isError and metadata before any display projection, with separate transport-failure identity.
- [ ] #2 Malformed error flags, serialization overflow and tool errors cannot turn into successful hook effects.
- [ ] #3 Existing non-hook consumers retain compatible presentation through an explicit projection while typed consumers receive the complete result.
- [ ] #4 Production client and service tests cover structured-only, text-only, mirrored, error-bearing and oversized results without executing third-party plugins.
<!-- AC:END -->
