---
id: TASK-32712
title: Keep Search RAG failure recovery visible beside retry controls
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 05:34'
updated_date: '2026-09-17 05:45'
labels:
  - library
  - search-rag
  - recovery
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Search/RAG failure review found that a query can settle with its failure details below the compact viewport while query focus is retained and Run re-enables. Users need an immediate explanation and an accurate next step beside the query, with retry preserving input and clearing stale failure feedback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failed or unavailable retrieval is explained beside the query in wide and compact layouts without hiding provider-cost disclosure.
- [x] #2 Retry preserves the query and source choices, releases the busy gate, and clears the prior failure after a successful local result.
- [x] #3 Mounted timing checks and native dark/light keyboard journeys verify failure then retry; raw exception details are not shown to users.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A; existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/003-settings-library-rag-defaults.md apply.
Reason: repair visible recovery feedback in the existing query region, preserving retrieval ownership, provider disclosure, and retry behavior.
1. Add mounted failure/unavailable then retry probes with real screen/controller rendering, both sizes/themes, retained query/focus, and provider disclosure. Demonstrate the invisible failure before implementation.
2. Render a brief outcome notice next to Run using existing token-backed status styling, keeping detailed Evidence recovery available. Synchronize the notice with the existing query gate and clear it on retry/success without stale refresh overwrites.
3. Verify focused Search/RAG tests and static checks, native injected-failure then real local retry journeys, private persistence/shutdown, and independent review. Update evidence/audit/task and commit locally. No full suite, push or merge.
Task allocation: CLI offered 32708, already below the all-local/remote-ref and worktree maximum 32711 (267 refs scanned); this new untracked task was assigned 32712 before implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retrieval failures and unavailable-service recovery now appear beside Run using the existing token-backed callout. Provider disclosure remains separate and the detailed Evidence recovery stays available. The notice remains mounted, updates synchronously with the Run gate, and clears on retry/success; delayed conditional refreshes cannot restore stale feedback.

Changed the Library Search/RAG panel helper/exports and controller, added 13 mounted regressions, and updated the workflow audit plus Docs/superpowers/qa/2026-09-17-rag-retry-recovery/README.md. The red run reproduced 6 failures / 2 passes; the final targeted gate passes 132 tests without exclusions. Ruff checks pass for new files, changed production ranges are formatted, and baseline comparison shows no added diagnostics. Independent review has no actionable findings.

Eight native failure/unavailable-to-real-local-retry journeys pass in dark/light at 170x48 and 80x24. All 12 captures were inspected. Query/scope/focus survive, raw exception text stays out of the rendered UI, the exact Media result returns, source/default files remain unchanged, all 10 databases pass quick_check, and normal shutdown/process absence are verified. Provider disclosure is mounted-harness evidence; provider generation and semantic/remote retrieval remain unqualified. No full suite, push or merge.

ADR required: no; existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/003-settings-library-rag-defaults.md apply. No plan deviations or new generalizable lesson. Next bounded review: Search/RAG answer-generation and recovery states.
<!-- SECTION:NOTES:END -->
