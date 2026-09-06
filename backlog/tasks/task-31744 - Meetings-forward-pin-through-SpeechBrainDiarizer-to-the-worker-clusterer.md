---
id: TASK-31744
title: 'Meetings: forward pin() through SpeechBrainDiarizer to the worker clusterer'
status: To Do
assignee: []
created_date: '2026-09-06 07:40'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The sticky-pin guarantee (a user-named speaker cluster is never auto-merged) lives in the worker's OnlineClusterer, but the Diarizer protocol and SpeechBrainDiarizer expose no pin() forwarding, so a live rename does not actually pin the cluster in the real backend (the Meetings screen calls pin() only if present). Add pin(cluster_id) to the Diarizer protocol + SpeechBrainDiarizer (send a pin command to the worker, which calls OnlineClusterer.pin), so a renamed speaker is not auto-merged mid-meeting. Deferred from the phase-2 diarization SDD run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A live rename pins the cluster in the real subprocess backend so it is never auto-merged
<!-- AC:END -->
