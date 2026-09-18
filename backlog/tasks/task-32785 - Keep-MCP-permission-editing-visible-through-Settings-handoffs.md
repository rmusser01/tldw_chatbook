---
id: TASK-32785
title: Keep MCP permission editing visible through Settings handoffs
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 11:12'
updated_date: '2026-09-18 11:34'
labels:
  - mcp
  - settings
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users review and edit the requested Tool Profile with visible keyboard focus and reachable controls at compact and wide terminal sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cold and restored Settings Edit handoffs retain exact current profile authority and focus a fully painted permission table.
- [x] #2 First, last and filtered permission rows stay visible during keyboard navigation; profile, filter and policy feedback remain reachable without clipping.
- [x] #3 Profile revision and permission mutation boundaries stay unchanged; stale authority remains refused and newer navigation retains focus.
- [x] #4 Targeted routed and permission-state tests plus native compact/wide dark/light journeys qualify the behavior and clean private-profile lifecycle.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the clipped table through real Settings-to-MCP routing, including restored MCP state and keyboard row navigation. 2. Give the existing Permissions canvas vertical scrolling within its allocated workbench pane while retaining bounded table sizing and current controls; rebuild extracted/source CSS as required. 3. Add focused routed visibility/keyboard regressions and verify filtering, policy persistence, stale authority and newer navigation. 4. Inspect real native compact/wide dark/light journeys, review independently and save evidence to draft PR 2707. ADR required: no. ADR path: backlog/decisions/107-portable-tool-use-packs.md; backlog/decisions/150-design-token-system-and-design-language.md. Reason: Restore visible focus and overflow access within the existing MCP workbench and permission contracts; no new navigation, storage or authority boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Settings Edit now reveals the current permission table after content layout. The scrollable canvas keeps profile/filter/feedback reachable, respects newer focus and permits Space mutations only while the table owns focus. Existing profile authority remains unchanged. 129 distinct targeted cases pass (4 routed, 31 workbench, 62 mode and 32 governance); older mode tests use existing process isolation and one stale assertion now pins the existing 70% token. Four final native dark/light compact/wide cells and 20 inspected captures qualify real imports, state edits, horizontal State access, fresh-revision return and clean private lifecycle. Independent review found no introduced blocker. Existing Ruff baselines unchanged; new files/changed methods formatted; backlog/diagnostic guards pass. QA: Docs/superpowers/qa/2026-09-18-mcp-permission-handoff/README.md. ADR required: no; existing ADR-107 and ADR-150 apply. Broader MCP layout and concurrent-workflow review remain open.
<!-- SECTION:NOTES:END -->
