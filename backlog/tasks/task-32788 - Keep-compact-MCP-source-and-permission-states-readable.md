---
id: TASK-32788
title: Keep compact MCP source and permission states readable
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 12:43'
updated_date: '2026-09-18 13:02'
labels:
  - mcp
  - ui
  - layout
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let keyboard users identify a tool and its permission state together in compact MCP layouts while keeping source controls and introductory guidance readable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 80x24 and wider supported layouts the MCP introduction is complete and both Local and Server source values remain fully painted and usable.
- [x] #2 Permission rows show their complete tool or server label through wrapping beside a fully visible State value at the left scroll position; tags remain accessible.
- [x] #3 Resize and filtering preserve selected row identity and exact permission action authority without unexpected writes or newer focus takeover.
- [x] #4 Targeted production-styled tests and real private native theme/size journeys qualify the change; evidence and review ledgers are updated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the production-styled compact baseline and add failing visible-content, row-identity and resize tests. 2. Use existing token-backed auto height for MCP purpose and reduced compact Source SelectCurrent padding; retain the three-pane layout. 3. Measure permission table space and current labels to wrap only Tool cells as needed while keeping State visible; preserve column order, tags and key-based selection. 4. Run targeted MCP/governance checks and independent review; qualify dark/light compact/wide native permission reads and exact mutation. 5. Update QA and component ledgers, then save on draft PR2707. ADR required: no. ADR path: N/A (existing ADR-150 and ADR-161). Reason: Bounded responsive presentation repair preserving existing navigation, service authority and component boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the MCP introduction auto-height and compact Source padding token-backed. Permission rows measure cell widths and wrap long Tool labels while reserving State; tags, row keys, filtering and exact profile authority remain intact. Resize does not reload the service or reclaim newer focus.

134 distinct targeted cases pass (6 final readability, 92 existing MCP, 36 governance/performance). Independent review found no introduced blocker. Four real private native dark/light compact/wide journeys and sixteen rendered/inspected captures verify full labels, exact policy persistence and return at the fresh revision. All 29 source hashes match; normal exit, lock release, eleven database integrity checks and unchanged default profile are recorded. The 92-case run began before the last scrollbar reservation correction; final six cases and native evidence qualify that correction. No full suite or provider requests.

Updated production MCP permission mode, source/generated CSS, targeted regression tests, QA README and component/Tool Profiles ledgers. Added the reproduced scrollbar measurement lesson. ADR required: no; existing ADR-150/161 govern this bounded presentation repair. Broader MCP server/tool/audit/runtime workflows remain open. Evidence: Docs/superpowers/qa/2026-09-18-mcp-compact-readability/README.md. Draft PR2707 remains subject to separate visual review and merge approval.
<!-- SECTION:NOTES:END -->
