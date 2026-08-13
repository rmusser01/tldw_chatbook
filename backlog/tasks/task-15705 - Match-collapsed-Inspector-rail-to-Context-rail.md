---
id: TASK-15705
title: Match collapsed Inspector rail to Context rail
status: Done
assignee:
  - '@codex'
created_date: '2026-08-13 06:07'
updated_date: '2026-08-13 16:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the collapsed Console Inspector rail visually match the established Context rail treatment so its label is centered vertically and its background fills the full rail column.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The collapsed Inspector handle fills the Console workspace column vertically.
- [x] #2 The collapsed Inspector column uses the same filled panel and border treatment as the collapsed Context column.
- [x] #3 The Inspector label is centered vertically while its optional badge remains visible and legible.
- [x] #4 Existing Inspector width, tooltip, badge abbreviation, open behavior, and compact-width access remain unchanged.
- [x] #5 Component TCSS and the generated stylesheet remain in sync.
- [x] #6 Focused Console rail and stylesheet integrity tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/017-console-left-rail-usability.md
Related ADR: backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md
Reason: Existing ADR-017 governs the rail visual language and ADR-043 governs compact-width access preserved by AC #4; no new architectural decision is introduced.

1. Add solid-framed, production-stylesheet regressions for Context/Inspector height parity, deterministic unbadged/badged geometry, shared-right-handle isolation, and source/bundle CSS parity.
2. Add a Console-only Inspector handle class and Python button geometry override; switch its Console frame from quiet to solid.
3. Update component TCSS, regenerate the production CSS bundle, and run focused rail/CSS/static checks.
4. Run a real-Console six-state SVG/geometry sweep at 130x30, 140x42, and 160x45 with zero/three approvals. RED review replaced 100x30 because its baseline Inspector is already horizontally off-screen, a separate ADR-043 layout defect outside this visual-parity change; existing compact-access tests retain 80/90/140 coverage.
5. Run full-suite Ruff/check/format gates; verify the task diff; close only if every Definition of Done gate is green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the collapsed Console Inspector as the full-height, filled, solid-framed right-side counterpart to Context while preserving the shared DestinationRailHandle defaults. ConsoleRailHandle now applies Inspector-only full-height button geometry, ChatScreen uses the standard Console frame, component TCSS defines the filled 100% height surface, and the generated modular stylesheet contains the same rule.

Files changed for implementation and verification: tldw_chatbook/Widgets/Console/console_rail_handle.py; tldw_chatbook/UI/Screens/chat_screen.py; tldw_chatbook/css/components/_agentic_terminal.tcss; tldw_chatbook/css/tldw_cli_modular.tcss; Tests/UI/test_destination_rail.py; Tests/UI/test_console_internals_decomposition.py; Tests/UI/test_css_build_integrity.py; Tests/UI/test_workbench_visual_snapshots.py; plus the approved spec, plan, and this task record.

TDD evidence: the initial RED run was 11 failed / 1 passed. Review then tightened the real-Console sweep and exposed six height failures. Latent harness corrections configured the native-ready Console after startup and moved the narrow visual state from the baseline-overflowing 100x30 case to the scoped 130x30 case without changing compact-access coverage. GREEN evidence from fresh closeout: 38 passed for destination rail, Console right rail, compact Inspector access, and CSS integrity; 1 passed for the live Console terminal-frame contract; 6 passed / 5 deselected for TASK-15705 visual states at 130x30, 140x42, and 160x45 with approval counts 0 and 3. The sweep verifies full workspace height, filled/solid parity with Context, centered label, contained abbreviated badge, preserved 11-column outer and 9-column content widths, and positive transcript width.

Static and integrity evidence: git diff --check passed. The scoped Ruff command reported 31 findings and scoped Ruff format reported four files; an identical command against base ae089ae711be980bb4116068386893f121004c5c produced the exact same 31 findings and same four files, so TASK-15705 added no static or format regression. The Impeccable detector reported two advisory pre-existing literal colors; the identical base scan reported the same two findings. CSS source/bundle parity is pinned by a passing integrity regression, and the generated bundle contains the component selector verbatim apart from its normal generated header.

Review and scope: implementation, specification, and quality reviews were approved through HEAD. Self-review of ae089ae711be980bb4116068386893f121004c5c..HEAD found only the ten planned task paths and confirmed AC 1-6. The change adds no input/data boundary, dependency, persistence, service contract, or license change; performance impact is limited to one lightweight composition-time style override.

Verification scope was explicitly narrowed by user override to directly modified/related functionality. A repository-wide pytest run that had already begun was terminated on request at approximately 42% and has no final count; repository-wide Ruff/format results are not completion gates for this closeout. Focused gates and changed-line/base-differential evidence are green.

ADR required: no. Existing backlog/decisions/017-console-left-rail-usability.md governs the text-only bordered rail language; backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md governs the compact-width access preserved here. Design: Docs/superpowers/specs/2026-08-12-task-15705-inspector-rail-parity-design.md. Plan: Docs/superpowers/plans/2026-08-12-task-15705-inspector-rail-parity.md. No generalizable new lesson was produced; existing testing-evidence and backlog-hygiene lessons were followed.
<!-- SECTION:NOTES:END -->
