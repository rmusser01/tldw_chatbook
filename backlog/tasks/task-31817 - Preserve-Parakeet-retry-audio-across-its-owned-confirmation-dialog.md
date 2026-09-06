---
id: TASK-31817
title: Preserve Parakeet retry audio across its owned confirmation dialog
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 05:42'
updated_date: '2026-09-06 15:21'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The retry confirmation currently triggers Console suspend-time dictation teardown, discarding retained audio and leaving the mounted mic indicator stale. A bounded lifecycle repair is awaiting user design approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Confirming the owned retry dialog can retry the retained real-session audio exactly once; declining clears retry state without changing the draft.
- [x] #2 Ordinary navigation and unmount still abandon dictation, while retry-dialog cancellation and return leave the current mic display consistent with canonical state.
- [x] #3 Strengthened regression doubles enforce real retry availability and the complete dictation/suspension selections verify the approved behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
User approved the five bounded fixes on 2026-09-06. Follow Docs/superpowers/plans/2026-09-06-approved-console-regressions.md TASK31817, with the approved spec alongside it. ADR required: no; existing controller ownership and ordinary teardown remain unchanged. Test retained-audio availability and exact owned-dialog lifetime RED first, then minimal fix, complete dictation/reuse verification and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved exact retry lifetime repair: snapshot the owned dialog at Console suspend before asynchronous cleanup, keep retained audio for its explicit Confirm, and abandon on rejection/cancellation/unmount or an undecided owned-dialog foreground loss. Independent review found the latter transition; real mounted overlay/navigation RED tests now pass. Stale confirmations cannot replay discarded audio and mounted mic repaints canonical idle. Complete dictation21passed57.47s; final seven-file dictation/streaming/first-run/reuse/command selection253passed374.99s with no retained SQLite or descriptor-growth warning (/private/tmp/tldw-31821-dictation-reuse-resources.xml and .log). Resource adapters tracked TASK31821 use existing owners only. Independent review, changed test/controller lint and scoped formatting pass; pre-existing full-controller formatting debt not rewritten. No new ADR; controller/dialog lifecycle contract preserved. Modified dictation.py, ChatScreen suspend call and dictation tests; no broad modal cleanup exemption.
<!-- SECTION:NOTES:END -->
