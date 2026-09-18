---
id: TASK-32749
title: Reconcile current dev with the reviewed design-system workstream
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 19:00'
updated_date: '2026-09-17 19:41'
labels:
  - design-system
  - ui
  - integration
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the saved component-review branch reviewable against current dev while preserving the token migration and incoming Console, Library and picker behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The candidate includes the recorded current-dev commit with no unresolved conflicts and preserves both sides of the reviewed behavior.
- [x] #2 Canonical stylesheet ownership, zero-literal and Python-style floors, generated artifacts and unchanged boot budget pass after integration.
- [x] #3 Targeted affected Console, Library, picker and Settings checks pass; any pre-existing failures are separately demonstrated.
- [x] #4 The candidate has unique Backlog IDs, passes applicable static checks and independent review, and has private native rendering and clean lifecycle evidence.
- [x] #5 The integration report, full completion ledger and draft PR reflect the verified candidate without claiming complete feature coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: existing ADR-150/161. Reason: reconcile existing contracts, with no new architecture boundary. Follow Docs/superpowers/plans/2026-09-17-component-current-dev-integration.md: record parents and IDs; merge exact current dev without automatic commit; reconcile source behavior and canonical style owners; rebuild generated artifacts; run targeted affected and governance checks; complete independent review and private native lifecycle evidence; update ledger/report and save to draft PR #2704. No full suite or merge into dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled dev 1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6 (71 commits), preserving incoming Console persistence/covered-layout and File Notes recovery/path behavior alongside the design-system work. Notes rules moved into canonical Library owners; Workflow uses the current split registry and tokenized identical values. Rebuilt styles and diagnostic inventory. Branch collisions 32628/32707 moved to 32750/32751 with provenance.

Incoming source-lifetime failures reproduced on exact dev; affected tests now use existing private-profile subprocess isolation and preserve admission guards. Picker journeys follow incoming path-first folder focus and Select folder copy. Verification: 280 distinct targeted cases, seven derived-artifact checks, 620062/634050 boot bytes, no new scoped lint/format debt. Three unrelated fatal Ruff diagnostics reproduce on dev. Four private native theme/size cells, eight inspected captures, ten healthy DBs, clean process/terminal shutdown and unchanged default files. Independent review found no integration-specific blocker; incoming empty Notes action clipping remains explicitly tracked as TASK-32752.

ADR required: no; existing ADR-150/161 apply. Plan deviations, failures and exact evidence are in Docs/superpowers/reports/2026-09-17-component-current-dev-integration.md and Docs/superpowers/qa/2026-09-17-component-current-dev/README.md. Updated the full completion ledger and draft PR #2704. No full suite or merge into dev; remaining feature reviews stay active.
<!-- SECTION:NOTES:END -->
