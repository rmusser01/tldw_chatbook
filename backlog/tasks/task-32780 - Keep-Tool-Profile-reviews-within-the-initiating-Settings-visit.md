---
id: TASK-32780
title: Keep Tool Profile reviews within the initiating Settings visit
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 09:46'
updated_date: '2026-09-18 09:57'
labels:
  - settings
  - ui
  - tool-packs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent delayed Tool Pack inspection or export preparation from opening stale review dialogs over a newer Settings visit or another workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Delayed import and export review cannot open after category departure or leave-and-return, Settings removal, or unrelated modal suspension.
- [x] #2 A newer Tool Profile workflow supersedes earlier preparation; only the current review may proceed to import or export publication.
- [x] #3 Owned options, review and file-picker dialogs retain normal revise, cancel and successful workflow behavior without changing immutable-snapshot or service authority.
- [x] #4 Targeted mounted lifecycle and existing workflow tests pass; the review ledger records the qualified scope and remaining focus/native journeys.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce delayed import inspection and export capture across real category navigation, removal, unrelated modal suspension and replacement workflows.
2. Give pending reviews a local intent identity, invalidate it on view departure, and exempt only their own active modal. Recheck after each asynchronous preparation/modal boundary before prompting or admitting a mutation.
3. Run targeted lifecycle, export and existing workflow regressions; obtain independent review and save evidence to draft PR 2707.
ADR required: no
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: Repairs UI review lifetime under the existing review-first contract without changing permission, publication, storage or service boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pending import inspection and export capture now carry a Settings-visit intent. Category changes, screen departure, unrelated modal suspension and replacement workflows invalidate it; owned file/options/review dialogs preserve it. Preparation and modal returns recheck ownership before prompting or admitting a write. Delayed destination capture also cannot publish after departure. Storage, import validation and immutable export authority are unchanged.

Independent review reproduced an initial shared-exception regression: a delayed, already-admitted publication error disappeared after a category roundtrip. A separate mutation-admission flag now retains admitted failure/uncertainty receipts like success, while obsolete preparation errors stay silent. The identical mounted probe passes after repair.

74 distinct targeted cases pass: 23 lifetime, 42 existing workflow/loading, 9 export recovery/outcome. The first lifetime run passed 20 of 21; the remaining import fixture mistakenly used removal category outcome_uncertain. Correcting it to real activation_uncertain and adding same-type replacements yielded four follow-up passes (one overlaps); all 23 lifetime cases are qualified. Production-CSS 80x24 controls/modal navigation are real; controlled service waits/file choices qualify lifecycle, not real activation. No new native visual claim, full suite or provider requests.

Scoped Ruff remains 114 before/after with no introduced diagnostics. New tests and changed workflow methods pass formatting; backlog and diff guards pass. Evidence: Docs/superpowers/qa/2026-09-18-tool-profile-review-lifetime/README.md. Updated Tool Profiles ledger and recorded the admitted-write lesson. Refresh-focus repair and complete native import/removal journeys remain open.

ADR required: no; backlog/decisions/107-portable-tool-use-packs.md governs existing review-first service authority. Draft PR 2707 remains unmerged pending its own visual/merge approval.
<!-- SECTION:NOTES:END -->
