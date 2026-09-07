---
id: TASK-15670
title: Change review lists one root twice under a single turn entry
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 21:30'
updated_date: '2026-08-27 00:48'
labels:
  - console
  - change-review
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AgentRunsChangeReviewProvider.turns()` groups by run id, and after PR 3a-1 Task 6c a run can hold both a turn row and a post-turn (survivor) row for the same root. Nothing breaks and multi_root labelling still works, but the review screen's file list can show the same path twice with no visible reason. Splitting the selector by `kind` needs a ReviewTurn key that is not the run id, which is a UI change Task 6c deliberately did not fold in.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A run holding multiple change windows for the same canonical root gives every affected file row a concise, distinct, visible window label; the normal turn and survivor labels explain when and by whom the change happened, and unknown or repeated kinds remain distinguishable
- [x] #2 Ordinary single-window and multi-root file labels still behave as they do today
- [x] #3 Review-wide **Undo All** refuses a same-root multi-window run before preflight or mutation and tells the user to revert one labeled window at a time; per-file revert remains available
- [x] #4 Affected turn-selector and tree totals describe summed entries as file changes rather than implying they are unique files
- [x] #5 The provider's existing tests pass, and production-shaped regressions cover same-root windows, narrow rendering, multi-root preservation, and the no-mutation refusal
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a bounded Review-screen clarity and safety correction under accepted ADR-089; it changes no storage, ownership, runtime boundary, or cross-module contract.

1. Add failing real-provider and mounted-screen regressions for distinct same-root window labels, narrow visibility, honest counts, ordinary multi-root preservation, and Review-wide Undo All refusal before provider work.
2. Add the smallest Review-screen row-classification and label presentation needed to distinguish repeated-root windows, with a stable fallback for unexpected kinds.
3. Reuse the same canonical-root comparison to refuse ambiguous whole-turn revert while preserving focused per-file revert.
4. Run the focused provider/screen tests, scoped static checks, and diff hygiene checks; self-review the exact branch diff against ADR-089 and this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented explicit provenance and safe recovery for repeated-root change windows in the full Change Review screen.

- Kept every snapshot window and prefixed affected rows with stable turn, overlap, survivor, or numbered fallback labels that remain visible in the narrow tree.
- Changed only repeated-root aggregates to say `file change(s)`; ordinary single-window and multi-root labels retain their existing copy and ordering.
- Refused Review-wide **Undo All** before provider preflight or mutation when one canonical root has multiple windows, with explicit guidance to use the labeled per-file reverts.
- Added real tracker/database/provider regressions for all persisted label kinds, unknown-kind fallbacks, 80-column rendering, snapshot-specific diff selection, ordinary multi-root preservation, and no-mutation refusal.
- ADR required: no. Accepted ADR-089 already owns this Review-screen safety boundary; no storage or cross-module contract changed.
- Verification: all 58 Change Review tests passed; the 6 focused new/affected tests, Ruff, Python compilation, and diff checks passed. The adjacent 12-test card suite has a pre-existing order-sensitive timeout in its final mounted retry test: the exact timeout reproduced unchanged on `origin/dev`, while that test passes alone.
<!-- SECTION:NOTES:END -->
