---
id: TASK-32833
title: Keep MCP tool results current and raw responses readable
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 03:19'
updated_date: '2026-09-19 03:35'
labels:
  - mcp
  - ui
  - design-system
dependencies:
  - TASK-32832
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need to distinguish the current tool attempt from earlier output and inspect raw responses in compact terminals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Raw response disclosure label and body remain reachable at compact and wide sizes in both themes.
- [x] #2 A failed argument validation replaces all previous result details and retains the argument draft for correction.
- [x] #3 Subsequent real tool execution restores current output, and close/reopen and outcome identity checks remain intact.
- [x] #4 Targeted tests, design guards and private native execution evidence qualify the bounded repair without raising CSS budgets.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce compact disclosure clipping and success-to-invalid stale output on the existing inspector.
2. Apply existing token-backed sizing to the disclosure and reuse the complete result-rendering path for local validation failures.
3. Verify output transitions, keyboard disclosure/body access and correction with targeted tests and real private stdio execution across both themes and sizes.
4. Run derived/design guards and independent review, retain evidence and update ledgers, then save a bounded draft stacked on PR2718.

ADR required: no
ADR path: N/A (existing ADR-150/161 and ADR-031 apply)
Reason: Repair existing inspector presentation and result replacement; no runtime, authority, storage or service contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reused the complete result renderer for local validation failures so previous raw output and interpretation disappear while drafts, previews and exact permission-profile context survive. Scoped token rules wrap Raw response and bound the compact body without changing wide sizing. Verified 50 distinct targeted cases, seven preflight guards and 16 inspected native captures with eight real stdio executions, four invalid attempts that did not execute, and clean private lifecycle. No added Ruff diagnostics or budget increases; independent review clear. ADR required: no; applies ADR-150/161 and ADR-031. Evidence: Docs/superpowers/qa/2026-09-18-mcp-inspector-results/README.md. Stacked on PR2718 because the compact workflow needs its scrolling fix; retarget dev after parent merge and recheck integration/visual approval. Wider schema/runtime and Audit workflows remain separate.
<!-- SECTION:NOTES:END -->
