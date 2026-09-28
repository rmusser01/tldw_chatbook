---
id: TASK-32682
title: Add qualified direct Streamable HTTP MCP transport
status: To Do
assignee: []
created_date: '2026-09-16 04:27'
labels:
  - plugins
  - implementation
  - mcp
dependencies:
  - TASK-32681
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-mcp.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect directly to generic MCP servers using explicit protocol profiles instead of treating the tldw_server wrapper as equivalent support.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Direct stdio and Streamable HTTP profiles qualify 2026-07-28, 2025-11-25 and 2025-03-26 behavior with correct per-request or initialize/session lifecycle and JSON/SSE handling.
- [ ] #2 Detection uses side-effect-free discovery; unsupported versions, pagination failures and optional capabilities produce explicit readiness diagnostics.
- [ ] #3 Cancellation and reconnect retain uncertain invocation outcomes without automatic replay, and connection readiness includes protocol and discovery success.
- [ ] #4 HTTPS is required except explicitly selected loopback development origins; redirects cannot forward credentials or weaken transport, and legacy stdio profile storage remains readable.
<!-- AC:END -->
