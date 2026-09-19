---
id: TASK-32789
title: Keep MCP Tools controls and rows reachable in compact layouts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 13:07'
updated_date: '2026-09-18 13:23'
labels:
  - mcp
  - ui
  - layout
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let keyboard users read the local-tools switch, operate catalog filters and reach tool rows when the MCP workbench is narrow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The complete local-tools switch label and on/off state are painted at compact and wide sizes in dark and light themes.
- [x] #2 Focused filter controls and first/last tool rows stay visible and usable through resize; the pane can scroll its content.
- [x] #3 Filtering and resize remain read-only and delayed reveals respect newer focus and modes.
- [x] #4 Targeted regressions, independent review and real private native journeys qualify the repair; review ledgers record remaining MCP workflows.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve production-styled baseline and add failing visible-label/control/row tests. 2. Use the existing VerticalScroll and current-focus reveal pattern; wrap the local-tools button with token-backed constraints and stack filters based on measured pane space. 3. Verify filtering, resize, selection and newer focus without tool execution or config writes. 4. Run targeted MCP/design/CSS checks, independent review and native dark/light compact/wide journeys. 5. Update QA and component ledgers, then save to draft PR2707. ADR required: no. ADR path: N/A (existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md). Reason: Responsive presentation and focus repair within the existing workbench and authority boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made MCP Tools scroll its content, reveal current focused controls after layout and retain the visible selected table row after resize. The complete on/off toggle uses full pane width and auto height; filters stack under the existing compact workbench class. No new breakpoint, token value or persistence authority was introduced.

88 distinct targeted cases pass: 6 final access cases, 46 existing MCP regressions and 36 governance/performance cases. The 36 existing Tools cases were repeated after the selected-row correction (overlap, not additive). Independent review caught a long-catalog cursor failure masked by post-resize test navigation; two red regressions now pass, and an independent direct resize replay confirmed the fix. Ruff is clean; changed methods and new files pass formatting, with two pre-existing unchanged formatting differences retained.

Four real private native theme/size journeys and sixteen rendered/inspected captures verify off/on setting persistence and restoration, first/last rows, preserved selection through resizing, exact tool inspection, and text/server filtering. Permission profiles remain unchanged. Final runner and 31 production hashes match; normal exit, released lock, ten healthy private databases, unchanged default fingerprints and no errors are recorded. No full suite, tool execution or provider requests.

Updated production mode and source/generated CSS, regression tests, QA gallery, MCP/component ledgers and the selected-row testing lesson. Plan refinement: reused the existing measured compact workbench class instead of adding a new pane breakpoint. ADR required: no; existing ADR-150 design-token system and ADR-161 apply. Remaining Tool/State readability and incorrect workspace-root guidance are explicitly recorded for further review. Evidence: Docs/superpowers/qa/2026-09-18-mcp-tools-access/README.md. Draft PR2707 still requires separate visual review and merge approval.
<!-- SECTION:NOTES:END -->
