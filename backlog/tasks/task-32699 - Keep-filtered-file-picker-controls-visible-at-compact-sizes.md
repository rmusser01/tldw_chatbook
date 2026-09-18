---
id: TASK-32699
title: Keep filtered file picker controls visible at compact sizes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 06:07'
updated_date: '2026-09-16 06:34'
labels:
  - ui
  - file-picker
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The provider-recovery native review found that at 80x24 the GGUF filter leaves the filename field about one column wide and pushes Cancel outside the dialog. Keyboard Escape still cancels, but the controls are not visibly usable. Evidence: Docs/superpowers/qa/2026-09-16-ingest-recovery/picker-80.svg. Existing folder-only compact behavior is recorded in TASK-32665.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 80x24 in both themes the filtered file picker shows a usable filename field, filter, Open and Cancel together with a loaded listing.
- [x] #2 Typing or pasting a file path, changing the filter, selecting and cancelling work through actual mounted controls.
- [x] #3 Compact-wide resize retains typed path, selection, current filter and focus without recreating controls; neighboring folder and save pickers retain their behavior.
- [x] #4 Token-backed production CSS, targeted tests and isolated native captures qualify the layout without broad test sweeps or model operations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/160-progressive-file-picker-listings.md; backlog/decisions/161-component-pattern-library.md
Reason: responsive repair of existing picker controls; no new ownership, persistence or interaction boundary.

1. Reproduce filename starvation and clipped actions under consolidated production CSS.
2. Adapt compact Open/Save footer and chrome with existing tokens, retaining mounted controls during resize.
3. Verify typed/pasted paths, filter changes, Open/Save/Cancel, folder neighbors and resize continuity with targeted tests.
4. Run isolated native dark/light compact/wide journeys, inspect captures, review the diff and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Compact Open/Save dialogs now keep a labeled full-width filename row above filter/actions, using existing tokens and the established compact picker chrome. Resize updates a CSS class without remounting controls. The styles are explicitly scoped away from Enhanced pickers. Rebuilt the CSS bundle.

Verification: 12 new production-CSS journeys and 167 neighboring/governance checks pass; a final 6-check CSS/budget recheck also passes. Eight isolated native journeys and eight final inspected captures cover dark/light 80x24 and wide resize continuity, real bracketed paste, callbacks, selection and cancellation. Private DBs and fixtures remained healthy/unchanged. No full suite or downstream model/import operations.

Zero new Ruff diagnostics (five inherited vendored findings remain); all three touched Python files pass formatting; CSS remains within its unchanged byte budget. Independent review found no actionable issues. Native inspection caught same-edge dock overlap of the filename label; added a failing painted-label assertion and a token margin, then confirmed it. Recorded the incident in lessons-testing-evidence.md.

ADR required: no; direct responsive implementation of ADR-150, ADR-160 and ADR-161 linked in the plan. Evidence, captures, limits and commands: Docs/superpowers/qa/2026-09-16-filtered-picker/README.md. No new task ID or architectural boundary.
<!-- SECTION:NOTES:END -->
