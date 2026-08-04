---
id: TASK-2152
title: 'MCP: fix clipped recovery callout and its config-syntax copy (F-050)'
status: Done
assignee: []
created_date: '2026-08-03 16:23'
updated_date: '2026-08-03 16:39'
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
- [x] #1 Callout copy is short plain language with no config-file syntax and renders fully at 100 cols
- [x] #2 Long technical detail remains available via tooltip
- [x] #3 Clicking the callout still jumps to/opens the server row
- [x] #4 The 'Disabled in config' jargon string is replaced wherever else it is user-facing (e.g. mcp_rail tooltips)
- [x] #5 Relevant MCP tests updated and passing plus ruff clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI/copy-only fix; no schema, boundary, or contract change). Steps: 1. Add failing tests: builtin_readiness off-message is short plain copy without config syntax, technical detail retained in snapshot.detail; servers-mode callout label renders <=100 cols with no '[mcp]' syntax and tooltip carries the technical detail. 2. Change readiness.py builtin disabled message to short plain copy; keep '[mcp].enabled = false' detail in detail dict. 3. mcp_servers_mode callout tooltip includes technical detail when present; keep click -> ServerRowSelected jump. 4. Rail tooltip de-jargoned automatically via short message. 5. Update tests asserting the old string (test_product_maturity_phase6_recovery_docs.py, test_mcp_inspector.py if needed). 6. Run MCP test files + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: changed builtin_readiness(enabled=False) message to short plain copy 'Turned off — open to enable.' (callout line ~56 chars, fully rendered at 100 cols) and kept the config syntax as detail['technical_detail']; the Servers-mode callout tooltip now prefixes that technical detail ahead of 'Open {label}.' (_callout_tooltip helper). Clicking the callout still posts ServerRowSelected -> opens the server row/detail (pinned by test). Rail tooltips and inspector/detail views inherit the de-jargoned message automatically since they render snap.message. Files: tldw_chatbook/MCP/readiness.py, tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py; tests updated: Tests/MCP/test_readiness_derivation.py (new plain-copy test), Tests/UI/test_mcp_servers_mode.py (new callout label/tooltip/click test), Tests/UI/test_product_maturity_phase6_recovery_docs.py + Tests/UI/test_mcp_inspector.py (old string retired). TDD: both new tests failed RED before the change. Verification: 203 passed across readiness derivation/model + servers_mode + rail + inspector + phase6 recovery; 311 passed in test_mcp_workbench.py + visual parity; ruff clean. ADR: not required (UI/copy-only). Deliberately not done: reframing the disabled built-in as opt-in (that is TASK-2153); commit 813c2e5ae.
<!-- SECTION:NOTES:END -->
