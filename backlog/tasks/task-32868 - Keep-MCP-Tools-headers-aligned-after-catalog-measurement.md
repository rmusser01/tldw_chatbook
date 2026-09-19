---
id: TASK-32868
title: Keep MCP Tools headers aligned after catalog measurement
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 21:37'
updated_date: '2026-09-19 21:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A catalog refresh can leave Tools column labels painted at their initial widths while the rows use measured widths. Keep the labels aligned with their data through refresh, theme changes and resize.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The painted Tools header aligns with measured body columns when a refresh paints before idle measurement.
- [x] #2 Catalog replacement, filtering, tags, Unicode and compact-wide transitions preserve header alignment, focused selection and the next real selection gesture.
- [x] #3 Targeted regressions and private native dark-light compact-wide captures qualify the repair without executing tools or changing permissions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A (existing ADR-150 and ADR-161 apply)
Reason: restore existing table rendering and interaction behavior; no new architecture, dependency or visual tokens.

1. Reproduce paint-before-idle header misalignment on merged dev and identify the measured-width/cache boundary.
2. Add a regression that compares painted header and body column positions through catalog replacement, filter, theme and resize transitions.
3. Make the smallest Tools-owned repair; retain auto-width sizing, wrapped names, focus, row identity and gesture suppression.
4. Run targeted table/selection and token guards, plus private native dark/light compact/wide journeys and lifecycle checks.
5. Record PR2726 merge proof, new evidence and remaining scope; self-review, obtain independent review and save a bounded PR against dev for visual approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced stale Tools headers on merged dev ad0f76e23b with four composed-screen regressions and a real native before/after journey. MCPToolsTable clears render caches before Textual dimension measurement so new auto-widths and auto-height rows share fresh rendering. The change preserves sizing, tokens, focus, identity and gesture rules; no global DataTable patch or dependency change.

152 targeted tests pass, including CLI validation, existing compact/readability coverage and quiet refresh/real selection gestures. Seven derived-artifact guards, Ruff and formatting checks pass. Four baseline native views fail the composed alignment check; all four fixed views pass, and only the header row differs in terminal/pixel comparisons. Both runs shut down normally with healthy private data, unchanged permission/execution state and user defaults. Independent source/QA review found no blockers; the evidence audit is recorded with the gallery.

Evidence: Docs/superpowers/qa/2026-09-19-mcp-tools-header/README.md and GALLERY.md. Production file: tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py; tests: test_mcp_tools_header_alignment.py and test_mcp_native_qa_args.py. The reports record PR2726's verified merge and keep remaining MCP reviews separate. Added the observed composed-screen-vs-fresh-render trap to lessons-testing-evidence.md.

ADR required: no. Existing backlog/decisions/150-design-token-system-and-design-language.md, 161-component-pattern-library.md and 170-table-repopulation-selection-boundary.md apply. Final owner visual approval and PR current-head CI/review remain merge gates, not claims made by these local checks.
<!-- SECTION:NOTES:END -->
