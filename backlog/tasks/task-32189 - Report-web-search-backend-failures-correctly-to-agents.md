---
id: TASK-32189
title: Report web-search backend failures correctly to agents
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 19:51'
updated_date: '2026-09-09 20:04'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent failed or challenge-blocked web searches from being presented as successful tool calls that encourage repeated searches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DuckDuckGo challenge and HTTP failure responses produce explicit failures rather than empty search results.
- [x] #2 Backend and malformed-response search failures are recorded as failed tool outcomes; genuine empty searches remain distinguishable.
- [x] #3 Agents receive actionable guidance against repeating unavailable searches; targeted Console and external-tool regressions pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/078-structured-agent-tool-outcome-provenance.md; backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: ordinary search failures use the existing LocalToolError and structured failed-result contracts.
1. Reproduce challenge and backend errors through search and local provider.
2. Detect challenge/HTTP failures in DuckDuckGo and propagate explicit errors from web_search while preserving genuine empty results and permission gates.
3. Include actionable stop/configuration guidance and verify success/error/cache controls.
4. Run focused tests, lint and review; document remaining credential requirements.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Search failures now raise existing LocalToolError, preserving failed outcomes through Console and external local-tool adapters. DuckDuckGo rejects HTTP errors, structural challenge pages and a missing parser. Genuine empty Searx and documented Yandex code15 responses stay empty successes; quota/auth errors remain failures. Successful caching remains unchanged. Updated web tool, search backends, provider tests and one runtime integration.

RED reproduced backend, challenge ordering, Searx/Yandex empty and provider failures. GREEN: 102 focused checks, three opt-in live skips; external local-server controls passed. Combined search/runtime/provider/local-server run: 637 passed, three skips, one unrelated filesystem-read ledger failure also reproduced with original web module (ledger/provider implementation unchanged). Independent review found and verified the Searx correction; final Yandex/Searx review passed 24 checks. No introduced lint findings; changed ranges formatted. Reviewed the sole diagnostic removal (missing-lxml logger now raises) with --statements and regenerated its inventory, verified clean. Existing ADR078/032; user docs and live-verification lesson updated. No credentials changed; an available/configured search backend is still required.
<!-- SECTION:NOTES:END -->
