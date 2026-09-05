---
id: TASK-31668
title: Isolate Library conversation snapshot calls from the UI loop
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:00'
updated_date: '2026-09-05 18:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep blocking async conversation sources from defeating the bounded Library source snapshot timeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversation snapshot calls execute off the UI loop like Notes and Media calls.
- [x] #2 The existing blocking-async snapshot timeout stays below0.05seconds and typed source results remain unchanged.
- [x] #3 Relevant snapshot and generation-guard tests pass without increasing timeouts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the existing blocking-async timeout regression after correcting its required fixture configuration, and add a thread-identity/result regression for all three source families.
2. Give conversation listing the same isolate_in_worker=True option already used for Notes and Media, changing no result or generation handling.
3. Verify both timeout cases, typed/source result contracts and current-generation snapshot tests; keep the existing0.05second ceiling.
ADR required: no
ADR path: N/A
Reason: Routine correction of one missing option on the existing worker-isolation seam; no new thread/lifecycle architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added the missing isolate_in_worker=True option to the conversation listing call in the existing source-snapshot gather, matching Notes and Media. The original blocking-async test reached0.2075seconds against its unchanged0.05second ceiling after the fixture prerequisite was repaired. A new thread-identity regression also failed before implementation because conversation execution occurred on the UI thread.
Verification after the one-argument fix:27 passed,64 deselected,27.85s across the new thread/result regression, both original timeout tests and selected current-generation/entry/snapshot recovery coverage. The new regression verifies all three source families retain records, counts and known-total flags while executing off the UI thread. No timeout, result, cancellation or generation assertion changed. New test Ruff/format and diffcheck pass. Screen41325lines/1301methods remains within the existing ceilings.
ADR required:no; applies the existing worker-isolation contract. Textual worker/testing skill guidance confirmed keeping blocking service implementations away from the UI loop.
<!-- SECTION:NOTES:END -->
