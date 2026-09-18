---
id: TASK-32816
title: Keep conversation Appearance actions visible at compact terminal sizes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 18:51'
updated_date: '2026-09-18 20:07'
labels:
  - ui
  - console
  - design-system
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-32813 native dark/light review at 80x24 found the conversation Appearance Cancel row below the viewport. All computed stylesheet rules matched the original pre-consolidation CSS. The modal uses height29 with max-height100% and retains other fixed-height children, so focusing the Cancel control does not paint its action row. The narrow palette also wraps the None label. Existing captures are in the CSS consolidation QA report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 80x24 in dark and light themes, Appearance actions remain visibly reachable by keyboard and pointer without widening the terminal.
- [x] #2 The palette labels and controls remain readable at compact and wide sizes while preserving saved icon and color behavior.
- [x] #3 Mounted geometry and native screenshots verify the compact layout and cancellation without committing changes.
- [x] #4 Typing an icon search across debounce refreshes preserves the complete query and cursor, and Enter selects the matching icon.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce compact action clipping, wrapped none swatch and actual icon-grid geometry with mounted compact/wide dark/light cases. 2. Fit the existing modal composition to its 80x24 content budget: remove extra palette/custom-color gaps, reserve scrollbar height, allow a single complete scrollable icon row, and size icon/swatch labels to their cell content. Use existing ADR-150/161 tokens for every touched fixed visual value; preserve result/cancel handlers. 3. Verify keyboard/pointer reachability, palette scrolling, icon filtering, saved-selection result behavior and resize retention with targeted tests. Rebuild generated CSS and check governance and unchanged performance ceilings. 4. Inspect native dark/light compact/wide captures with cancellation-only fixture interactions and clean isolated shutdown; obtain independent review, update ledgers and save draft PR2707. ADR required: no. ADR path: backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md. Reason: Routine correction using the existing modal, scrolling, focus and token patterns; no new UX/application boundary or persisted behavior.
Build dependency found by the tokenized mounted run: widget-default streams cannot see central tokens because Textual scopes variables per source. Resolve core/_variables.tcss through the existing build-time per-block variable isolation before emitting widget defaults; preserve theme aliases and block-local overrides, keep the same stylesheet tiers/source count, and pin rebuild propagation with a focused builder regression. Existing ADR-150/161 govern this direct token-system implementation; no new runtime boundary or ADR.
Native run 001 exposed pre-existing query loss during debounced icon-grid refresh: focus returned to the search input and selected its existing text, so later keystrokes replaced the prefix. Add a deterministic slow-typing regression and preserve the search cursor on programmatic refocus before the one corrective native capture batch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Appearance now keeps all actions inside 80x24, fits twelve icons per row, reserves palette scrolling space and preserves the full none label. Debounced search no longer selects and replaces its prefix on refocus. Existing ADR-150/161 tokens are resolved once at build time and copied per widget block; dynamic theme aliases and local overrides remain isolated, with no new runtime source or cascade tier. The generated diff is confined to Appearance. Evidence: eight mounted layout cases; 61 regression checks; 24 parser checks; five search/result checks (overlapping counts); three budget checks; and two final route/source-budget journeys after TASK-32818 isolates the identified host audio compiler. Eight native dark/light compact/wide captures were inspected, with clean shutdown, ten healthy private DBs, zero conversation/message rows and unchanged default config. All nine captured source hashes still match. Six pre-existing lint diagnostics, zero new; 29 changed format ranges clean; independent review has no actionable finding. Docs/superpowers/qa/2026-09-18-console-appearance/README.md retains every first failure and scope limit. No full suite, real provider or native appearance persistence claim. Existing ADR-150/161 apply; no new ADR.
<!-- SECTION:NOTES:END -->
