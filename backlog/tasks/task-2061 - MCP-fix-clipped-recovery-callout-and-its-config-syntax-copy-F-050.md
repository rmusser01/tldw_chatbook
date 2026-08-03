---
id: TASK-2061
title: 'MCP: fix clipped recovery callout and its config-syntax copy (F-050)'
status: In Progress
assignee: []
created_date: '2026-08-03 16:23'
updated_date: '2026-08-03 16:28'
labels:
  - mcp
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Servers-mode problem callout renders the built-in server's disabled message on one clipped line with config-file syntax as user copy (readiness.py 'Disabled in config ([mcp].enabled = false).'). Make the callout copy short, plain, and fully rendered at 100 cols; keep the technical detail in a tooltip; de-jargon the same string everywhere it is user-facing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Callout copy is short plain language with no config-file syntax and renders fully at 100 cols,Long technical detail remains available via tooltip,Clicking the callout still jumps to/opens the server row,The 'Disabled in config' jargon string is replaced wherever else it is user-facing (e.g. mcp_rail tooltips),Relevant MCP tests updated and passing plus ruff clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI/copy-only fix; no schema, boundary, or contract change). Steps: 1. Add failing tests: builtin_readiness off-message is short plain copy without config syntax, technical detail retained in snapshot.detail; servers-mode callout label renders <=100 cols with no '[mcp]' syntax and tooltip carries the technical detail. 2. Change readiness.py builtin disabled message to short plain copy; keep '[mcp].enabled = false' detail in detail dict. 3. mcp_servers_mode callout tooltip includes technical detail when present; keep click -> ServerRowSelected jump. 4. Rail tooltip de-jargoned automatically via short message. 5. Update tests asserting the old string (test_product_maturity_phase6_recovery_docs.py, test_mcp_inspector.py if needed). 6. Run MCP test files + ruff.
<!-- SECTION:PLAN:END -->
