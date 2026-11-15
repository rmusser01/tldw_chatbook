---
id: TASK-40969
title: Investigate Console context-memory change events and preserve live edits
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15T07:55:00Z'
updated_date: '2026-11-15T10:06:00Z'
labels:
  - textual-patterns
  - settings
  - inputs
dependencies: []
references:
  - tldw_chatbook/UI/Screens/settings_screen.py
documentation:
  - backlog/decisions/303-settings-commit-models-three-honestly-labeled.md
  - Docs/superpowers/qa/2026-11-14-textual-pattern-followups/verification.md
  - Docs/superpowers/plans/2026-11-14-textual-pattern-followups.md
priority: medium
type: spike
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Determine whether programmatic refreshes of the four Console context-memory Inputs and five Selects can be mistaken for user edits or overwrite newer raw text. Qualify the current behavior with mounted controls, then repair any confirmed defect using the existing Settings draft and input-tracking contracts. This follows the latest-dev Textual book review; it is not a claim that the defect has already been reproduced. Existing TASK-35867 covers eight other Console inputs, and TASK-32767 covers broader context/background qualification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mounted controls distinguish initialization and programmatic refresh from genuine user edits, including queued Changed messages.
- [x] #2 Newer raw text and caret/selection/focus survive an overlapping refresh; percent conversion, validation, Save and confirmed Revert retain their existing behavior.
- [x] #3 Record the observed baseline and disposition; any confirmed defect has a failing behavioral regression followed by a passing targeted verification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: qualify and repair existing draft/input contracts under ADR-052/161/303; preserve existing commit model and staged-save capture.
1. After Media completes, mount all four Inputs and five Selects and reproduce programmatic refresh, unrelated dirty draft, queued real edits and raw percent input.
2. If a defect is confirmed, add failing behavioral evidence and repair through existing tracked-input/suppression helpers, preserving newer raw text, focus/caret/selection and initialization cleanliness.
3. Verify percent conversion, validation, Save and confirmed Revert using targeted context-memory and relevant existing input-origin tests; record baseline and disposition.
Plan: Docs/superpowers/plans/2026-11-14-textual-pattern-followups.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Confirmed three mounted defects: backing refresh contaminated an unrelated draft with all nine untouched context values, queued real edits could be overwritten, and equivalent refresh reformatted raw percentage text. Extended the existing tracked-control seam to four Inputs/five Selects, suppressed projection messages natively and compared parsed numeric meaning before assigning. Existing staged Save, validation, percentage conversion and persistence remain unchanged.

Changed context composition/handlers/sync and the shared tracking helper in settings_screen.py; added mounted event-origin regressions and updated app-shell architecture and Textual lessons.

Verification: failing behavioral baselines followed by 4 new + 12 existing targeted passes, including keyboard raw/invalid text, focus/caret/selection, all nine queued controls, real normalized Save and confirmed Revert. Parent/private-child receipts verified. New-test lint/format and production F checks pass; no new full-file lint diagnostics. Independent task and final whole-branch reviews approved. Four new cases are in the PR census.

ADR required: no; existing ADR-052/161/303 apply. Evidence and limitations: Docs/superpowers/qa/2026-11-14-textual-pattern-followups/verification.md. Local uncommitted implementation; no full-suite or publish claim.
<!-- SECTION:NOTES:END -->
