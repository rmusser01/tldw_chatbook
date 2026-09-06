---
id: TASK-31696
title: Restore semantic Notes browse focus when returning from Folder Files
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:37'
updated_date: '2026-09-05 18:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the Files-to-Database return path that reads a removed receipt focus field, retaining semantic placement focus, scroll restoration and independent editor authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Files return paths resolve the current semantic Notes receipt without missing-field exceptions
- [x] #2 Semantic focus role, selected note and scroll restore consistently across retained and rebuilt Notes paths
- [x] #3 Pure semantic receipt and missing-field Escape regressions pass; independent admission repaint and responsive scroll failures remain separately tracked
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce existing Files return and editor-authority failures; trace obsolete receipt.focus against current semantic receipt fields.
2. Characterize exact focus identity projection for note/folder/filter roles and scroll offsets in pure state tests.
3. Move the two existing identical screen conversions into a receipt.focus_identity property and use it at the broken Files return, retaining callback guards and lifecycle behavior.
4. Run pure tree state and original return/focus matrix plus relevant Notes focus tests, scoped static checks, screen ratchet and parent review.
ADR required: no
ADR path: N/A
Reason: Existing pure semantic conversion ownership is deduplicated to repair a removed-field read; no lifecycle, focus policy, persisted receipt schema or runtime boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added the pure record-free focus_identity projection to the semantic tree receipt; both existing screen conversions and Files return now share it. Exact note/folder/filter fallback roles and scroll are characterized. No callback generation/guard or persisted dataclass field changed. Screen shrank from41324 to41305 lines, methods unchanged at1301.
Four new projection cases failed before and pass after; complete tree-state file48 passed in2.87s. Both previously failing Escape sizes pass after the missing-field repair. The broader seven-case diagnostic went from7 failures to5 independent failures: outgoing editor repaint causes four identity/focus failures, and wide return has a responsive scroll6vs7 mismatch. Scope split approved by parent: those remain separate work, not claimed fixed here.
State/test Ruff and whole-file format pass. Screen has40 pre-existing Ruff findings, verified exactly identical before/after via JSON comparison; no new findings. git diff --check passes. Parent approved and reviewed the conversion boundary. ADR required:no; existing pure conversion ownership only. No new lifecycle/focus policy. Screen unrelated full-file format debt preserved.
<!-- SECTION:NOTES:END -->
