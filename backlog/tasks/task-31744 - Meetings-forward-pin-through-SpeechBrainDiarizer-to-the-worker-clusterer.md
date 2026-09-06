---
id: TASK-31744
title: 'Meetings: forward pin() through SpeechBrainDiarizer to the worker clusterer'
status: Done
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
- [x] #1 A live rename pins the cluster in the real subprocess backend so it is never auto-merged
<!-- AC:END -->

## Implementation Notes

`Diarizer.pin(cluster_id)` added to the protocol; `SpeechBrainDiarizer.pin` sends `{"cmd": "pin"}` to the worker (best-effort, non-blocking lock acquire, no reply awaited, only the cluster id crosses the pipe); the worker calls the live `OnlineClusterer.pin`. Covered by fake-subprocess tests and a torch-free test that drives the worker's real command loop (`serve()`). Landed in PR #2471 (commits 691f968a9, 3598e070c, 39f046f84).
