---
id: TASK-32832
title: Make compact MCP inspector controls reachable
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 02:48'
updated_date: '2026-09-19 03:12'
labels:
  - mcp
  - ui
  - design-system
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users must be able to reach and operate the existing Test Tool form and its recovery controls in compact terminals without losing their argument draft.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keyboard focus and scrolling reveal Test Tool arguments and actions at compact and wide inspector sizes in both themes.
- [x] #2 Action labels remain readable; form drafts survive compact-to-wide-to-compact resize and Close/Escape restore the existing panel behavior.
- [x] #3 Targeted geometry and behavior checks plus private native terminal journeys verify real tool execution and clean shutdown.
- [x] #4 The repair uses existing token and component rules; generated styles and design governance pass without raising budgets.
- [x] #5 Sibling readiness, permission and Advanced buttons remain readable and reachable when the shared inspector scrollbar is present.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the observed offscreen argument field with real bundled CSS and inspect overflow/content geometry.
2. Give the existing inspector a scrolling viewport and size its form/actions to content using established token rules; preserve handlers and drafts.
3. Verify compact/wide dark/light keyboard reachability, resize, raw/schema forms and Close/Escape with targeted tests, then real private native execution.
4. Run design/CSS/derived guards, independent review, update evidence/ledgers and save a separate draft PR against dev.

ADR required: no
ADR path: N/A (existing ADR-150/161 and ADR-031 apply)
Reason: Routine repair of reachability within the existing inspector and interaction contract; no new UX destination, ownership, runtime, service or storage boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the existing MCP inspector scroll vertically, preserve drafts and reveal current focus after resize. Token-backed form/button sizing keeps approval and sibling actions readable beside the scrollbar. No new ADR: applies ADR-150/161 and ADR-031. Verified 42 distinct targeted cases, seven preflight guards and 12 inspected native captures with real private stdio validation/execution and clean shutdown. Six older selected tests fail identically during profile setup on unchanged source; retained as a harness limit, not passing evidence. Independent review resolved sibling clipping and permission-fixture coverage findings. Evidence: Docs/superpowers/qa/2026-09-18-mcp-inspector-reachability/README.md. MCP and completion ledgers updated; compact Raw response disclosure and wider workflows remain separate. No merge; visual approval still required.
<!-- SECTION:NOTES:END -->
