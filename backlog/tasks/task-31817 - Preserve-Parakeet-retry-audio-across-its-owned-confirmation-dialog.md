---
id: TASK-31817
title: Preserve Parakeet retry audio across its owned confirmation dialog
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-06 05:42'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The retry confirmation currently triggers Console suspend-time dictation teardown, discarding retained audio and leaving the mounted mic indicator stale. A bounded lifecycle repair is awaiting user design approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Confirming the owned retry dialog can retry the retained real-session audio exactly once; declining clears retry state without changing the draft.
- [ ] #2 Ordinary navigation and unmount still abandon dictation, while retry-dialog cancellation and return leave the current mic display consistent with canonical state.
- [ ] #3 Strengthened regression doubles enforce real retry availability and the complete dictation/suspension selections verify the approved behavior.
<!-- AC:END -->
