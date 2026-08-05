---
id: TASK-526
title: Refresh latest-dev MCP first-use smoke copy
status: Done
assignee: []
created_date: '2026-07-24 19:09'
updated_date: '2026-07-24 19:11'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the first-use route smoke gate aligned with the current MCP Hub destination so it verifies meaningful live copy instead of waiting for the retired Unified MCP label.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The MCP route smoke expects copy rendered by the current MCP Hub screen
- [x] #2 The full latest-dev core app usability smoke module passes
- [x] #3 The merge-base masking failure and no-ADR decision are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the feature-branch timeout and compare the expected copy with MCPScreen.
2. Replace only the retired MCP copy assertion with a stable current destination-purpose phrase.
3. Run the full latest-dev smoke module, Ruff, format, diff checks, and review.
4. Document merge-base evidence, verification, and the no-ADR decision before completion.

ADR required: no
ADR path: N/A
Reason: This corrects a stale smoke-test copy pin and changes no production interface or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the retired Unified MCP smoke pin with the stable current MCPScreen destination-purpose phrase, Manage MCP servers. This keeps the route gate user-visible and meaningful while matching the MCP Hub screen introduced by the destination redesign.

The feature branch timed out at mcp expected copy because Unified MCP is no longer rendered. The merge-base test is masked earlier by its production-path sqlite readonly failure, repaired by TASK-522; inspection confirms the merge-base MCPScreen already renders the current purpose phrase. Verification: the full latest-dev core app usability smoke module passes (3 passed); combined with command-palette coverage, 63 tests pass. Ruff, format, and diff checks pass, and independent review approved the phrase as stable because focused MCP tests cover the same full purpose copy.

ADR required: no. This refreshes a test-only copy assertion and changes no production UI or architecture.
<!-- SECTION:NOTES:END -->
