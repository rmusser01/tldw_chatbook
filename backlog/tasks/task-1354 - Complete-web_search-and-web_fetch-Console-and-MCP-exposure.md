---
id: TASK-1354
title: Complete web_search and web_fetch Console and MCP exposure
status: Done
assignee: []
created_date: '2026-08-05 05:49'
updated_date: '2026-08-12'
labels:
  - web-tools
dependencies: []
references:
  - ADR-032
  - ADR-053
documentation:
  - Docs/superpowers/specs/2026-08-05-web-search-fetch-tools-design.md
  - Docs/superpowers/plans/2026-08-05-web-search-fetch-tools.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish and truthfully reconcile the shipped web_search and web_fetch capability across the Console local-tool provider and the standalone MCP server. The original FastMCP-era builtin/per-domain proposal was superseded by ADR-032's local-tool permission boundary and ADR-053's mcp-unified opt-in exposure: fresh calls remain Ask, external Ask fails closed, and web_fetch permits only public HTTP(S) targets with redirect-hop revalidation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console agents can discover and call web_search and web_fetch through LocalToolProvider, with each invocation governed by the existing On/Off/Ask permission, kill-switch, and definition-hash checks
- [x] #2 The standalone mcp-unified server exposes web_search and web_fetch only when mcp.expose_local_tools is enabled, routes calls through the same LocalToolProvider, and fails closed when an external client cannot satisfy Ask
- [x] #3 web_fetch rejects non-HTTP(S), private, loopback, link-local, reserved, multicast, metadata, and unresolvable targets before transport and revalidates every redirect hop
- [x] #4 The configuration template documents the tools and webfetch sections used by the shipped runtime, including local-tool availability and robots.txt behavior
- [x] #5 Focused Console, provider, external MCP, egress, redirect, and configuration tests pass without restoring the retired FastMCP/builtin design
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md; backlog/decisions/053-mcp-unified-standalone-runtime-boundary.md
Reason: ADR-032 is the canonical Console registration, permission, and egress boundary for local web tools; ADR-053 governs their optional external mcp-unified exposure. Amend and link these existing records rather than create a duplicate ADR.

1. Audit TASK-1354, its original FastMCP-era spec/plan, the shipped phase-3a local-provider implementation, TASK-2828 external exposure, and later permission/config hardening.
2. Reconcile ADR-032 and the stale TASK-1354 spec/plan with the implemented split egress contract: both web tools use the Ask-preserving local permission model; `web_fetch` enforces public HTTP(S) targets and redirect-hop validation; and for each `web_search` invocation the caller/model selects one allowlisted `search_engine`, which determines the destination. The operator supplies supported per-engine credentials and configurable endpoints where available; fixed-endpoint engines remain implementation-defined; a configured Searx endpoint may be local; and `web_search` does not apply public-target validation.
3. Run focused existing tests for Console discovery/invocation, LocalToolProvider schemas and approvals, egress and redirect blocking, configuration contracts, and optional external MCP registration/refusal.
4. Fix only evidence-backed gaps within the reconciled acceptance criteria; otherwise avoid duplicating already-shipped production logic or tests.
5. Run scoped static checks, self-review the exact diff, record implementation notes and verification evidence, check all acceptance criteria, and mark TASK-1354 Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reconciled the stale FastMCP-era task, design, and plan with the implementation that actually shipped through the local-agent-tools phase work. The Console owns `local:web_search` and `local:web_fetch` through `LocalToolProvider` under ADR-032; fresh calls remain Ask and retain the kill switch, definition-hash protection, and approval flow.
- Confirmed TASK-2828/ADR-053 provide opt-in standalone mcp-unified exposure through the same provider. `[mcp] expose_local_tools` defaults off, and external Ask fails closed because no Console approval callback exists.
- Amended ADR-032 to record the shipped split egress contract. Both tools remain permission-gated; only `web_fetch` rejects non-HTTP(S), non-public DNS answers, and unsafe redirect hops before transport. For each `web_search` call, the caller/model selects one allowlisted `search_engine`, which determines the destination. The operator supplies supported per-engine credentials and configurable endpoints where available; fixed-endpoint engines remain implementation-defined; a configured Searx endpoint may be local; and `web_search` does not apply public-target validation. Linked TASK-1354 from ADR-032 and ADR-053 and marked the abandoned builtin/default-Allow/domain-scoped/Playwright draft non-normative.
- No production code or duplicate tests were added: the feature and its regression coverage already existed. Verification: 414 passed across web core, provider, Console integration, external MCP, egress, config, and deep-search contracts; 21 passed across Console bridge and MCP server UI integration; template probe confirmed `[tools]`, `[webfetch]`, and documented robots.txt behavior; diff checks were clean.
- Plan deviation: implementation was already delivered by the approved phase-3a/phase-4 plans before this parent task was closed, so this pass performed the planned audit/reconciliation and verification only. No new lesson was added because `lessons-backlog-hygiene.md` already records the exact shipped-but-still-To-Do board trap.
<!-- SECTION:NOTES:END -->
