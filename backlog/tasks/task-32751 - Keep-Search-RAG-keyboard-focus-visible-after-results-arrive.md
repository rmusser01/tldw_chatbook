---
id: TASK-32751
title: Keep Search RAG keyboard focus visible after results arrive
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 05:00'
updated_date: '2026-09-17 05:22'
labels:
  - library
  - search-rag
  - keyboard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During TASK-2377 native verification, submitting a real local keyword query from the focused Search/RAG query field scrolls Evidence into view while focus remains on library-rag-query-input. At 80x24 the retained input region is (29, -19, 50, 3), completely outside the viewport; the footer still says typing in field. The same condition occurs in both themes; at 170x48 the query is clipped above the panel viewport. Evidence: Docs/superpowers/qa/2026-09-17-rag-scope-recovery/result.json and the ready SVG captures. This differs from TASK-32053's card focus cue and TASK-4023's intended Evidence reveal: users now have keyboard focus in an invisible input after a successful query. Review the result-arrival reveal/focus contract, including a newer user focus decision while retrieval is in flight.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a query completes, the focused control remains visibly painted in wide and compact layouts in both themes.
- [x] #2 Evidence remains discoverable and query text is retained; a newer user focus choice during retrieval is not overwritten.
- [x] #3 Mounted timing and native keyboard checks cover query submission and result arrival without provider calls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A; existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/003-settings-library-rag-defaults.md apply.
Reason: fix the existing reveal callback so keyboard focus remains visible; no new application structure, ownership, persistence, or visual values.
1. Reproduce the focused-query viewport loss with the real mounted Search panel, deferred outcomes, and production CSS. Cover current focus at callback time, ready/empty results and keyboard reachability of Evidence.
2. Make the existing post-refresh reveal honor the live focused descendant before scrolling to Evidence. Preserve query text and existing keyboard navigation; do not transfer focus or add persistent focus state.
3. Run focused Search/RAG tests and static checks; verify real local keyword queries and newer focus choices in native dark/light at wide/compact sizes. Review, record evidence, close task and commit locally. No full suite, push or merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Search run-start and result-arrival reveals now keep the live focused panel control visible. The callback reads current focus after refresh, so query editing and newer keyboard choices survive retrieval; when focus is outside the panel or unset, Evidence still reveals automatically. Query text and the existing Tab route to result cards remain intact. At compact widths the focused control takes viewport priority over the automatic result reveal.

Modified library_rag_search_controller.py and added test_library_rag_result_focus.py. The original regression reproduced invisible focus (7 failed / 1 passed); the expanded 20-case matrix covers both themes/sizes, ready/empty outcomes, newer focus and no/foreign-focus fallback. Final targeted gate: 118 passed, no exclusions. New files pass Ruff lint/format; changed production ranges pass formatting and existing controller diagnostics remain unchanged. Independent review has no remaining findings.

Eight real local keyword searches pass natively in dark/light at 170x48 and 80x24. Enter submits, Shift+Tab changes focus during a gated real retrieval, and Tab reaches the exact returned Media record. All twelve final captures were inspected. Ten private databases pass quick_check, the seeded source/default profile files are unchanged, and app exit/PID absence are verified. The first native attempt stalled on a runner-wide worker wait; it was quit normally, and two corrected runs passed using domain completion state. The existing worker-wait lesson covers this incident.

Evidence: Docs/superpowers/qa/2026-09-17-rag-result-focus/README.md; Library audit updated. ADR required: no; existing ADR-150 focus rules and ADR-003 ownership apply. No new styles, tokens, storage or provider contract. Semantic retrieval, provider generation and broader failure/retry journeys remain unqualified here. No full suite, push or merge.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

TASK-32707 became TASK-32751 during TASK-32749 integration on 2026-09-17.
The task already landed on dev retains the old ID; this unmerged design-review
task moves under the landed-keeps-ID rule. Original add: bddafc724d 2026-09-16T22:06:50-07:00.
Historical capture payloads and probe filenames retain their original identity;
current task references use TASK-32751.
