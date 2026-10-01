---
id: TASK-32682
title: Add qualified direct Streamable HTTP MCP transport
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:27'
updated_date: '2026-10-01 04:46'
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
- [x] #1 Direct stdio and Streamable HTTP profiles qualify 2026-07-28, 2025-11-25 and 2025-03-26 behavior with correct per-request or initialize/session lifecycle and JSON/SSE handling.
- [x] #2 Detection uses side-effect-free discovery; unsupported versions, pagination failures and optional capabilities produce explicit readiness diagnostics.
- [x] #3 Cancellation and reconnect retain uncertain invocation outcomes without automatic replay, and connection readiness includes protocol and discovery success.
- [x] #4 HTTPS is required except explicitly selected loopback development origins; redirects cannot forward credentials or weaken transport, and legacy stdio profile storage remains readable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/111-mcp-remote-transport-and-client-dependency.md
Reason: implements the accepted direct client transport boundary and reciprocal partial supersession of ADR-111, with explicit protocol profiles and schema migration.
1. Read current client/store/control-plane custody and pinned official protocol sources; reconcile accepted ADR ownership before implementation.
2. Port reviewed owned stdio/HTTP fixtures and establish RED at the actual new direct-transport entry after the M1 successful controls.
3. Integrate the reviewed M2 increment while preserving current producer/recovery guards, native cleanup custody and typed results; qualify the three explicit eras, JSON/SSE, cancellation/no replay, readiness and profile migration.
4. Run affected transport/store/control-plane neighbors, static comparison and resource checks; record evidence and platform limits, self-review ACs and commit only M2-owned files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated explicit stdio/Streamable HTTP profiles for three qualified MCP eras, strict bounded JSON/SSE discovery/results and no replay. Legacy profile schema 2 migration reuses the current protected writer. Preserved actual recovery/producer guards and stdio settlement; added concrete HTTP maintenance custody with retained pool-close proof, raw storage-pause refusal and fresh-connect-only resume. ADR-162/111 reciprocal partial supersession recorded; no new dependency or permission owner. Controlled peer fixtures advertise actual capabilities and forward dispatch observations. Changed client/local_store/local_control_service, new protocol_profiles/streamable_http and targeted tests/docs. Evidence: 180 transport/store/lifecycle tests, 319 typed/catalog/tool neighbors (four documented pre-M1 deadline baselines deselected), 12 actual native maintenance controls, all no warnings/skips; new Ruff/formatter and shared no-add lint/syntax/whitespace pass. Full evidence and controlled macOS/non-vendor limits: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.
<!-- SECTION:NOTES:END -->
