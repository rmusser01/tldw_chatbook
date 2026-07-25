---
id: TASK-561
title: Contain stylesheet-backed destination layout regressions
status: Done
assignee: []
created_date: '2026-07-25 18:19'
updated_date: '2026-07-25 18:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore honest destination geometry under the committed application stylesheet for MCP, Settings, Workflows, and ACP after the parity harness began loading production CSS.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP inspector remains usable and inside the workbench in ready unavailable and loading states
- [x] #2 Settings exposes at least one visible primary or recovery action in its default workbench
- [x] #3 Workflows and ACP compact panes remain inside the viewport without weakening desktop pane ratios
- [x] #4 The reproduced failures and complete destination visual-parity module pass
- [x] #5 Focused destination suites stylesheet integrity static checks and implementation notes verify the correction
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the seven RED failures under the production stylesheet and inspect actual widget regions and visible actions.
2. Correct MCP vertical allocation so ready, unavailable, and loading inspector states retain usable height inside the workbench.
3. Reconcile the Settings visible-action contract with the current mounted controls, changing production layout only if all valid actions are genuinely off-screen.
4. Remove compact-only minimum-width overconstraints from Workflows and ACP while preserving their desktop fr ratios.
5. Rebuild the generated stylesheet and run the seven focused regressions, complete visual-parity module, focused destination suites, stylesheet integrity, Ruff, formatter, and diff checks.
6. Self-review the production geometry and close TASK-560 only after its complete-module criterion is independently satisfied.

ADR required: no (existing decisions apply)
ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md and backlog/decisions/015-shell-destination-ia.md
Reason: Existing ADRs require visible workflow controls, stable destination composition, and responsive workbench geometry; this task repairs implementation drift without changing ownership or application structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Contained the production-stylesheet regressions revealed by the corrected parity harness. The MCP inspector now overrides the generic intrinsic-height inspector style and fills its horizontal hub grid. Workflows and ACP retain their 2:4:2 fr ratios but drop incompatible fixed minimum widths so compact terminals do not expand the workbench past the viewport. Settings parity accepts its current visible Save/manual-sync/Appearance actions and uses the mounted draft-status sentinel; Workflows compact parity accepts the visible empty-state recovery before its below-fold launcher. Rebuilt the generated stylesheet from source modules.

Verification: 9/9 focused regression slice; 82/82 complete production-stylesheet parity module; 7/7 targeted MCP/Settings tests; 55/55 destination and Console-handoff tests; 9/9 stylesheet integrity tests; Ruff, formatter, and diff checks pass.

ADR required: no. Existing ADR-011 workbench UI and ADR-015 shell destination IA decisions require these responsive geometry and visible-action contracts.

Modified files: Tests/UI/test_destination_visual_parity_correction.py; tldw_chatbook/css/components/_agentic_terminal.tcss; generated tldw_chatbook/css/tldw_cli_modular.tcss.
<!-- SECTION:NOTES:END -->
