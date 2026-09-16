---
id: TASK-32683
title: Bind MCP credentials to stable reviewed authority
status: To Do
assignee: []
created_date: '2026-09-16 04:27'
labels:
  - plugins
  - implementation
  - mcp
dependencies:
  - TASK-32682
  - TASK-32670
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-mcp.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep normal token renewal usable while ensuring account, endpoint or scope changes cannot inherit stale plugin authority.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owning credential service exposes stable reference, principal, issuer, audience/endpoint and scope bindings with separate authority and storage revisions.
- [ ] #2 Verified renewal within unchanged authority resolves current credentials at dispatch without invalidating plugin trust; changed or unknown authority requires reconciliation.
- [ ] #3 Explicit header/token mappings and supported OAuth flows use existing host credential services; unsupported auth is a specific unready state and no vendor connector grant is imported.
- [ ] #4 Secret sentinels stay out of trust snapshots, catalogs, receipts, debug defaults and errors; origin changes, failed refresh, opaque replacement and store migration have successful and refusal controls.
<!-- AC:END -->
