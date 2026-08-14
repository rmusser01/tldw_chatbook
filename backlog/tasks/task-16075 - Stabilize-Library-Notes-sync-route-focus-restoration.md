---
id: TASK-16075
title: Stabilize Library Notes sync-route focus restoration
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 04:31'
updated_date: '2026-08-14 04:32'
labels:
  - testing
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore Notes Sync Back focus to the filter after its targeted canvas recompose, preserving the existing focus-ownership contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync-route focus failure reproduces independently and is root-caused
- [x] #2 Sync Back explicitly restores filter focus after the targeted recompose
- [x] #3 The regression test waits for the settled focus and passes deterministically
- [x] #4 Focused static and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the isolated RED and inspect the production focus-restoration sequence.
2. Supply the existing filter-focus callback to the targeted Notes sync and use the bounded condition helper at the settled assertion.
3. Run the named test repeatedly, a focused adjacent focus slice, Ruff, and diff checks; remove the callback once to prove the regression remains discriminating.
4. Record implementation notes and close the task.

ADR required: no
ADR path: N/A
Reason: routine focus-restoration bug fix within the existing targeted-sync boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Root cause: the Notes Sync Back handler was converted to targeted canvas sync but did not provide the explicit post-recompose focus callback that whole-screen recompose previously supplied. Waiting alone remained RED for the full 15-second bound.
- Added the existing `_focus_library_notes_filter_input` callback to that one targeted sync and made the regression assertion wait for settled focus with a diagnostic failure message.
- Verified the named regression and all 21 adjacent Notes focus tests pass. Ruff check and `git diff --check` pass; the test file's unrelated pre-existing formatter drift remains unchanged.
- ADR required: no; this is a one-line restoration of the established focus contract.
<!-- SECTION:NOTES:END -->
