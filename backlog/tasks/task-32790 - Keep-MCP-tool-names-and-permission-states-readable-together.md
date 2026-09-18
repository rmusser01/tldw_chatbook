---
id: TASK-32790
title: Keep MCP tool names and permission states readable together
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 13:27'
updated_date: '2026-09-18 13:44'
labels:
  - mcp
  - ui
  - layout
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users identify catalog tools and read their permission state together in compact MCP panes without losing selection while resizing or filtering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Long and Unicode tool names wrap beside a complete State value at the left scroll position in compact and wide layouts; other columns remain accessible.
- [x] #2 Resize, filtering and catalog/state refresh retain selected tool identity when it remains visible and preserve explicit selection behavior without unexpected service reloads or permission writes.
- [x] #3 Current focus and active filters survive responsive reflow; removed or filtered-out selections fall back safely.
- [x] #4 Targeted tests, independent review and real private native theme/size evidence qualify the change and update review ledgers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce real compact Tool/State clipping and add failing production-styled and Unicode/selection regressions. 2. Measure available table cells and reserve State width before wrapping Tool names; retain existing columns, token styling and key-based action authority. 3. Reflow only when measured width changes and restore cursor by tool ID through filtering/refresh; preserve newer focus and stable filter controls. 4. Run targeted MCP/design/CSS tests, independent review and native compact/wide dark/light journeys. 5. Update QA and component ledgers and save to draft PR2707. ADR required: no. ADR path: N/A (existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md). Reason: Bounded responsive rendering and selection repair in the existing catalog without new service or authority boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Long MCP Tool names now wrap beside complete permission State using measured viewport geometry; final child-size notifications handle scrollbar changes. Rows retain tool IDs across resize, filtering and refresh, with safe removed-row fallback and fresh Enter activation. Existing inspector click tests use private process profiles. 88 final targeted cases, independent review and four real private native theme/size cells pass; all 12 captures inspected, 31 sources pinned, normal exit0 and ten healthy private DBs confirmed. QA: Docs/superpowers/qa/2026-09-18-mcp-tools-readability/README.md. Existing ADR150 design-token and ADR161 apply; no new ADR. Review ledgers and scrollbar lesson updated. Root guidance, draft retention and overlapping saves remain next bounded repairs; no tool execution qualification.
<!-- SECTION:NOTES:END -->
