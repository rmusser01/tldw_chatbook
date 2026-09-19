---
id: TASK-32835
title: Keep MCP Audit filters readable and reachable in compact terminals
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 04:02'
updated_date: '2026-09-19 04:21'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users read and operate all Audit filters and reach the filtered execution table in a compact terminal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Text filter values and full decision and initiator labels paint within their controls in dark and light compact layouts.
- [x] #2 Keyboard focus can reach each filter and the execution table, including after resize, without losing filter values.
- [x] #3 Wide layouts remain usable; targeted tests, design governance and private native evidence verify the supported sizes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce filter value/label clipping and focus reachability with the real app styles at 80x24 through 170x48.
2. Stack compact filters using existing tokens, provide execution-pane scrolling and retain focused controls through reflow as needed.
3. Rebuild generated CSS; run targeted layout/behavior/governance and budget checks, independent review and private native dark/light journeys.
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: bounded repair of the existing Audit controls within the current responsive workbench; no new application structure, interaction contract or runtime boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stacked Audit filters at full width across supported pane sizes and enabled execution-pane scrolling. Deferred focus/resize reveal keeps the current control and table cursor reachable without changing filter values. Existing ADR-150/161 apply; no new ADR, tokens, policy or runtime boundary. Plan refinement: the same column is used at wide sizes because intermediate triad widths also crowd the controls.

107 distinct targeted cases pass, including all 69 legacy Audit cases and ten new real-app dark/light cases at 80x24, 100x30, 120x40, 170x48. Seven preflight guards pass; no introduced Ruff diagnostics; new/changed ranges formatted. All 18 private native captures were inspected, with clean exit/lock release, ten healthy databases and unchanged defaults. Independent review found no actionable issue. Initial and intermediate failures, including the standalone legacy fixture import-order limitation, remain in Docs/superpowers/qa/2026-09-18-mcp-audit-filters/README.md.

Modified Audit UI/CSS, generated sheets, rendered layout tests and review ledgers. Existing CSS literal pins now test computed geometry. Based on fresh dev; PR2720 owns the separate selection repair. Next review is inspector guidance ownership. Follow-up PR requires its own current-head CI and final visual approval; wider component review stays open.
<!-- SECTION:NOTES:END -->
