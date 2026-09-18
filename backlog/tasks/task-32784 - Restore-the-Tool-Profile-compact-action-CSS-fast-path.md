---
id: TASK-32784
title: Restore the Tool Profile compact action CSS fast path
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 11:02'
updated_date: '2026-09-18 11:09'
labels:
  - settings
  - ui
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep compact Tool Profile actions fully visible without adding broadly indexed Button rules that exceed the existing CSS performance guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The parsed ancestor-scoped bare-type rule count stays within the existing 274 limit without weakening it.
- [x] #2 Compact profile actions retain their existing size, specificity, action coverage and visible keyboard continuation; Import remains outside the rule.
- [x] #3 The generated bundle matches source and targeted performance, layout and governance checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the failed PR guard and exact added selector with the real parsed stylesheet. 2. Give profile action buttons a dedicated class and target that class at equal specificity; rebuild CSS from modules. 3. Verify the unchanged ratchet, existing compact focus journeys, mounted before/after computed style and paint equivalence, and CSS/token governance; review and save to draft PR 2707. ADR required: no. ADR path: backlog/decisions/150-design-token-system-and-design-language.md. Reason: Mechanical selector-index repair preserving visual and behavior contracts under the existing design language.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the compact Tool Profiles bare Button selector with Button.tool-profile-action and added that class only through the profile action factory. Specificity (0,2,2), width declaration, compact grid and Import exclusion are unchanged. Rebuilt the generated bundle from source.

The actual parsed-style guard reproduced 275 > 274 before repair and now reports 274. 80 targeted checks pass, including compact focus and CSS/governance coverage. Four mounted theme/size comparisons produce identical computed styles, geometry, visibility and painted text. Fixed-scroll comparison separates initial animation variance from layout; existing focus tests cover automatic reveal. Ruff, formatting, backlog and diff guards pass; independent review found no actionable issue. No full suite or new native captures were run for this mechanical change.

Evidence: Docs/superpowers/qa/2026-09-18-tool-profile-css-fastpath/README.md. Updated Tool Profiles/component ledgers and draft PR description. Existing ADR-150 governs the repair; no new ADR required. Remote CI must run on the new saved head; MCP Edit remains the next UI repair.
<!-- SECTION:NOTES:END -->
