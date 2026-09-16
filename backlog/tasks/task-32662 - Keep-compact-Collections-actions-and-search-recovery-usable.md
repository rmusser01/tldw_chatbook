---
id: TASK-32662
title: Keep compact Collections actions and search recovery usable
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 00:48'
updated_date: '2026-09-16 01:04'
labels:
  - library
  - collections
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow TASK-32659 with the observed compact reader toolbar clipping, keyboard traversal and clearing text search while relevance sorting is active. Keep the existing reader topology and authority behavior while making these paths usable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reader action and mode labels remain complete and keyboard reachable with Items open at compact width, and recover when enlarged.
- [x] #2 Tab, Shift+Tab and Enter reach and operate reader controls with visible focus while capture identity and unsaved annotations remain intact.
- [x] #3 Submitting an empty or whitespace-only text search from relevance sort restores valid unfiltered results within the selected scope without an exception.
- [x] #4 Targeted and private native evidence cover wide and compact layouts in both themes without increasing existing size budgets.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/113-collections-capture-authority-and-legacy-boundary.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Repair destination-local responsive action layout and valid query transitions within existing geometry, authority and token contracts.

1. Reproduce clipped primary/mode controls through actual Tab/Shift+Tab and Enter with production CSS and real Local captures. Cover Items open at 80x24 and wide recovery, both themes; retain an annotation draft during resizing.
2. Reproduce empty and whitespace-only Input submission while relevance is active. Verify truthful two-capture results, valid sort and retained Saved scope after repair.
3. Use existing content-measured wrapping/layout patterns so complete buttons fit without changing the adaptive shell or recreating focused controls. Normalize relevance only when text search becomes empty.
4. Run new journeys and affected existing reader, browse, geometry, query-budget, token/bundle and controller-size checks. Compare static diagnostics against base 72d59ca88f; do not increase budgets or run the full suite.
5. Perform one private native wide/compact inspection and at most one confirmation, check exact persistence and normal shutdown. Obtain focused review, update QA/guide/audit/task notes and commit locally. No provider/server/extraction requests, push or dev integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the compact Collections control and text-search recovery review.

Action and mode bars now stack according to measured button widths, preserving mounted controls, keyboard order and annotation drafts. Clearing text under relevance selects saved desc before request validation; other sorting and selected scope stay intact. Source CSS was rebuilt into the generated bundle.

Verification: 137 distinct targeted checks pass, including both themes, wide/compact resize, actual Tab/Shift+Tab/Enter, persistence, neighboring journeys, query budgets and token/bundle governance. The unchanged LibraryScreen size ceiling remains an inherited failure (35,210 lines / 1,320 methods versus 33,204 / 1,276); all Collections budgets pass without increases. Zero new Ruff diagnostics; new Python formatting, changed-range formatting and whitespace checks pass. Independent read-only review found no actionable issue. Two transient focus assertions were corrected to wait for visible natural focus after recomposition.

Private native checks passed at 170x48 dark and 80x24 light. Four rendered captures were inspected; exact original notes, two Saved captures, ten SQLite integrity checks, zero messages and normal exit 0 were verified. No extraction/provider/server request, full suite, push or dev integration.

QA: Docs/superpowers/qa/2026-09-15-collections-compact/README.md. Updated the Collections guide and Library workflow audit. Next review: Library ingestion/import journeys.

ADR required: no. Existing backlog/decisions/086-library-adaptive-reader-shell.md, 113-collections-capture-authority-and-legacy-boundary.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md apply; no boundary, storage, dependency or token-value change.
<!-- SECTION:NOTES:END -->
