---
id: TASK-32812
title: Keep MCP rail labels and server navigation visible in compact terminals
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 17:04'
updated_date: '2026-09-18 17:33'
labels:
  - mcp
  - ui
  - responsive
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep server navigation understandable and reachable in compact MCP panes without clipping labels or hiding the selected target.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All servers is fully readable in compact and wide terminals with existing focus and active states.
- [x] #2 Server rows fit their available viewport with truthful ellipsis, visible readiness and counts, and intact selection identity for Unicode labels.
- [x] #3 Long server lists remain reachable by keyboard and pointer across resize and refresh without losing the users active target.
- [x] #4 Targeted rendered-geometry checks, governance, independent review and native dark/light compact/wide evidence pass and are recorded in the draft PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce clipped rail controls and row paint in the real three-pane shell at compact/wide sizes, including Unicode names, counts and long lists. 2. Apply existing token-backed compact button sizing, readable All servers wrapping and scrollable rail constraints; derive label budgets from final available viewport rather than outer width. Preserve target identity and genuine focus/activation through resize and redraw; reject delayed presses from replaced rail controls instead of resolving their old numeric IDs through a new row list. 3. Run targeted rail and adjacent component checks, token/CSS governance and independent review. 4. Verify native dark/light compact/wide navigation, publish the gallery and update draft PR evidence. ADR required: no. ADR path: backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md. Reason: Routine repair within the existing three-pane/navigation and token contracts; no new storage, runtime or cross-module API boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Compact MCP rows now wrap inside the actual pane, retain visible All servers and Source labels, and scroll long lists. Literal terminal-cell truncation handles Unicode and markup-like names; control-bound server keys reject delayed presses on removed rows. Ordinary same-catalog refresh and resize preserve focus without restoring stale focus. Source/scope/list structural changes keep the existing capture guard. ADR check: no new ADR; routine repair under ADR-150 and ADR-161. Evidence: Docs/superpowers/qa/2026-09-18-mcp-rail-navigation/README.md records 119 passing scoped cases, six original/new regressions, eight inspected real-terminal captures in dark/light compact/wide, clean exit and private-data checks, independent review, Ruff, generated-CSS, Backlog and diagnostic checks. Two explicit exclusions remain: 26 baseline CSS-ratchet offenders reproduced at 9a4ba3cef3, and an unqualified destination-tour profile-setup failure. No full suite ran. Updated MCP/completion ledgers and draft PR2707 evidence; wider review and separate merge approval remain open.
<!-- SECTION:NOTES:END -->
