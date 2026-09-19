---
id: TASK-32823
title: Keep selected MCP tool details current during catalog refresh
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-18 21:46'
updated_date: '2026-09-19 00:24'
labels:
  - mcp
  - ui
  - selection
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A refreshed Tools catalog must not leave obsolete selected-tool metadata or test controls in the inspector, and unchanged catalogs must preserve the user argument draft and focus.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Catalog refresh updates the selected tool details and clears removed tools without selecting a replacement.
- [x] #2 Unchanged definitions preserve the mounted argument draft and focus; changed definitions retire old test controls and permission previews with visible guidance.
- [ ] #3 Targeted refresh, selection and prepared-test regressions plus bounded native evidence qualify the behavior; review ledgers and the draft PR record remaining scope.
- [x] #4 Selected-tool details, argument controls and refresh guidance remain keyboard-reachable and visibly scroll into view at compact and wide sizes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: backlog/decisions/161-component-pattern-library.md and backlog/decisions/170-table-repopulation-selection-boundary.md.
Reason: repair selected-detail projection and panel preview lifetime inside existing service admission, ownership and table-selection contracts.
1. Resume the preserved investigation on fresh merged dev after PR2707 closes; retain its red/green evidence.
2. Reproduce delayed preview completion during held form teardown, including no existing nonce. Invalidate the retiring panel synchronously before the first await, and reject late publication without changing service admission.
3. Preserve unchanged schema/raw drafts and focus; verify changed/removed metadata, newer selections and newer focus during refresh, and adjacent prepared-test ownership.
4. Run targeted private-profile tests serially, scoped static checks, and native dark/light compact/wide catalog-refresh journeys with clean lifecycle evidence.
5. Independent review, updated receipts/ledgers, and save this bounded follow-up on its own draft PR against dev. Broader server lifecycle and execution reviews remain subsequent work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled the selected MCP tool with each catalog refresh. Equal definitions and profile context preserve the mounted schema/raw form, preview, draft and focus; changed definitions retire controls with reopening guidance, and removed tools clear without choosing a replacement. Newer selection/focus wins over waiting refresh work.

Synchronous opaque form identity invalidation closes delayed preview publication during teardown. Preview request generation advances before refusal paths, and queued retired-form requests do not mint. Existing service admission and execution authority remain unchanged.

Native verification exposed and repaired inspector scrolling, narrow button width, natural form height and current-focus reveal after content shrink. Four final dark/light 80x24 and 170x48 native journeys pass with twelve inspected captures, clean exit, released instance lock, ten healthy private databases, unchanged default fingerprints and matching production/runner hashes. Failed native runs and deterministic red cases are retained.

Validation: 353 distinct targeted cases pass across the final 320-case inspector/token/bundle run, 42-case ownership run and two CSS budget/ratchet guards. No introduced production Ruff diagnostics; new tests and native runner pass lint/format, changed production ranges pass formatting. Seven preflight derived-artifact checks passed before final layout repairs; affected CSS/token/budget guards passed afterward. Independent review found no remaining blocker. No full repository suite was requested or run.

Modified MCPInspector/MCPWorkbench, generated widget CSS, focused regressions, QA receipts, review ledgers and incident lessons. Existing ADR-161/170 apply; no new ADR is required. Evidence: Docs/superpowers/qa/2026-09-18-mcp-inspector-refresh/README.md. Broader server lifecycles, connected tools, argument validation and actual execution remain subsequent bounded reviews. Draft PR creation and its remote CI/visual approval are still pending.
<!-- SECTION:NOTES:END -->
