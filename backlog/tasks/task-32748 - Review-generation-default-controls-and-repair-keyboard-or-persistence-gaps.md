---
id: TASK-32748
title: Review generation-default controls and repair keyboard or persistence gaps
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 18:29'
updated_date: '2026-09-17 18:58'
labels:
  - ui
  - settings
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the next component-review slice so per-model generation defaults remain readable, editable and truthful through keyboard validation, save, revert and navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Supported generation controls and complete labels are keyboard-visible in dark/light wide and compact layouts; unsupported controls do not create hidden focus stops.
- [x] #2 Invalid edits do not persist; Save, blank override removal and Revert preserve the intended provider/model ownership and unrelated configuration.
- [x] #3 Category return and resize preserve staged values and usable disclosure/focus; failures retain recoverable choices.
- [x] #4 Targeted production-CSS and private native evidence, review, static checks and completion tracking qualify this bounded slice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: existing backlog/decisions/006-provider-aware-generation-settings.md; ADR-031/150/161 govern keyboard and component patterns. Reason: review/repair the existing Settings-owned per-model profile surface without changing profile/runtime ownership or provider contracts.
1. Inspect TASK-83/189, existing provider-profile tests and production styles. Add keyboard/paint and persistence journeys for generation defaults; reproduce any defect before production edits.
2. Repair only confirmed gaps using existing form/disclosure patterns and tokens; update acceptance criteria before expanding scope. Preserve current provider/model contracts.
3. Run focused Settings/profile and relevant CSS checks. Verify real private-profile persistence and dark/light wide/compact rendering; inspect captures and lifecycle evidence. Obtain independent review.
4. Update the guide, workflow audit and full completion ledger, close this atomic task after all gates, and save the verified continuation to draft PR #2704. No full sweep or merge into dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired compact generation fields, retained disclosure/draft state and same-field focus after reflow, and rejected non-finite values before Settings mutation. Existing ADR-006/031/150/161 apply; no new boundary decision. Added 15 production-CSS journeys and repaired five baseline-proven stale CSS-owner tests. All 173 distinct targeted cases pass; no added lint diagnostics. Four final private native journeys completed real saves/blank removal after injected failures, with eight inspected captures, 11 clean databases, unchanged default files and normal shutdown/process absence. QA: Docs/superpowers/qa/2026-09-17-settings-generation-defaults/README.md. Independent review guard addressed, final review clear. Updated guide, workflow audit, finite-input lesson and full completion ledger. No full suite or merge into dev; current-dev integration is next.
<!-- SECTION:NOTES:END -->
