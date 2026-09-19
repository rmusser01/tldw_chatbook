---
id: TASK-32822
title: Keep MCP tool switch labels readable in compact panes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 21:27'
updated_date: '2026-09-18 21:41'
labels:
  - mcp
  - ui
  - design-system
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MCP Servers still clips non-master tool-switch names and on/off states inside compact detail panes. Make every existing switch readable without changing which configuration or permission it controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every built-in and local tool-switch label paints its complete name and on/off state at compact and wide sizes in both themes.
- [x] #2 Keyboard focus and scroll reveal, exact toggle routing, disabled local dependencies and the existing master-switch behavior remain intact.
- [x] #3 Targeted regression checks, unchanged CSS budgets, source-bound native visuals and private lifecycle checks qualify the repair; review ledgers and draft PR record its scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md. Reason: bounded layout repair using established master-switch wrapping; no new settings, runtime boundary or permission policy. 1. Read existing tool-gate consumer, wrapping rule, tests and live evidence. 2. Reproduce complete-label clipping using the real workbench with application styles. 3. Extend the token-backed wrap rule to the existing gate group and rebuild CSS. 4. Verify label paint, keyboard traversal/activation, disabled dependencies and existing master consumers in isolated targeted tests; inspect compact/wide dark/light native captures and lifecycle. 5. Independent review, task/ledgers/evidence and save to the draft PR. Fresh 230-ref/33-worktree scan found max32821 before reserving32822.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extended the existing token-backed MCP tool-switch wrapping rule from the master control to every gate Button; rebuilt the generated stylesheet. Complete name/state paint, keyboard traversal and disabled dependencies pass at compact/wide sizes in both themes. No settings handlers, permission policies, token values or CSS registrations changed. ADR required: no; existing ADR-150 and ADR-161 apply. Four intended baseline compact failures become green; 37 distinct targeted cases and all seven preflight guards pass. Eight rendered native captures were inspected, with real private Deep research save/reversal, restored focus, unchanged permission profiles, normal shutdown, ten healthy private databases, unchanged defaults and twelve matching source hashes. CSS bytes 584112/608090, selector candidates 274/274; limits unchanged. Independent review found no actionable issue. Initial runner selection failure and interpreter-specific lifecycle probe failure remain documented. Ruff and formatting pass. Evidence: Docs/superpowers/qa/2026-09-18-mcp-gate-labels/README.md. Both review ledgers and draft PR2707 description updated; broader MCP/destination review and this PR own visual merge approval remain open. No full suite run.
<!-- SECTION:NOTES:END -->
