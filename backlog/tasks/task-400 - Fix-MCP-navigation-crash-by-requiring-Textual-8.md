---
id: TASK-400
title: Fix MCP navigation crash by requiring Textual 8
status: In Progress
assignee: []
created_date: '2026-07-23 22:51'
updated_date: '2026-07-23 23:10'
labels: []
dependencies: []
references:
  - backlog/decisions/022-textual-8-runtime-floor.md
documentation:
  - Docs/superpowers/specs/2026-07-23-mcp-textual-runtime-floor-design.md
  - Docs/superpowers/plans/2026-07-23-mcp-textual-runtime-floor.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the application from crashing when users navigate to the MCP destination after installing a Textual version that predates the Select.NULL API used by the MCP UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The MCP destination mounts without a Select.NULL AttributeError
- [ ] #2 A regression test prevents lowering the Textual runtime floor below 8.0.0
- [ ] #3 Focused MCP tests pass on the supported runtime
- [ ] #4 Package metadata and development requirements support Textual 8.x (>=8.0.0,<9)
- [ ] #5 CI runs the focused MCP runtime suites against exactly Textual 8.0.0
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the accepted Textual 8.x runtime boundary in ADR-022 and the approved design/implementation documents.
2. Add red semantic regressions for both dependency manifests and update the existing Phase 6 packaging seam.
3. Change pyproject.toml and requirements.txt to textual>=8.0.0,<9, add the Unreleased changelog notice, and make the dependency tests green.
4. Add a red workflow contract, then a focused GitHub Actions lane that installs Textual 8.0.0 and runs the MCP workbench/tools suites.
5. Verify packaging, workflow, and MCP suites on the normal and exact-minimum runtimes; run diff hygiene; record evidence and close the task.

ADR required: yes
ADR path: backlog/decisions/022-textual-8-runtime-floor.md
Reason: this changes the supported framework/runtime policy, ends Textual 3-7 support, and fails closed on unreviewed future Textual major versions.
<!-- SECTION:PLAN:END -->
