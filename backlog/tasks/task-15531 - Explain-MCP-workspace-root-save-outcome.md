---
id: TASK-15531
title: Explain MCP workspace-root save outcome
status: Done
assignee:
  - '@codex'
created_date: '2026-08-12 02:55'
updated_date: '2026-08-12 02:56'
labels:
  - ui
  - mcp
  - accessibility
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the MCP Tools workspace-root save action explain its persisted effect so the destination-wide action audit is green and keyboard/mouse users can discover when the change applies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The MCP workspace-root Save action exposes concise, truthful outcome guidance through the existing Textual tooltip affordance.
- [x] #2 The destination action-outcome audit passes for both MCP route aliases, with focused MCP controls remaining green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this is a one-control copy/accessibility correction within the existing Textual and MCP persistence contracts.

1. Reproduce the existing destination audit failure and trace the save outcome.
2. Add the smallest native tooltip at the existing button construction site.
3. Run the two failing audit cases, focused MCP controls, scoped lint/format/diff checks, and review the rendered copy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a native Textual tooltip to the existing MCP Tools workspace-root save button. The copy explains that the saved root applies to the next Console agent run and that blank input uses the app folder; no new state, component, or persistence behavior was introduced.

Verification: the two previously failing destination-route audit cases pass; MCP Tools controls and workspace-save behavior pass 32 focused tests; scoped Ruff lint, changed-range format, py_compile, diff-check, and the Impeccable detector pass. ADR check: no ADR required because this is a one-control copy/accessibility correction within the established UI and persistence contracts.
<!-- SECTION:NOTES:END -->
