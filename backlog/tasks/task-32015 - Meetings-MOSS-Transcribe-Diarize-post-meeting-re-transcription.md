---
id: TASK-32015
title: 'Meetings: MOSS-Transcribe-Diarize post-meeting re-transcription'
status: To Do
assignee: []
created_date: '2026-09-07 23:50'
labels:
  - audio
dependencies:
  - TASK-31827
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Third TASK-31827 sub-project. MOSS-Transcribe-Diarize 0.9B (Apache-2.0) is a batch, CUDA-only joint transcription+diarization model — not a live `Diarizer`. Design a post-meeting 'speaker-attributed re-transcription' action (server-hosted, or local CUDA) that writes a second transcript with MOSS's speaker labels, reconciled onto the meeting's named speakers where possible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design approved (hosting, privacy, how the second transcript is stored and shown, reconciliation with existing names)
- [ ] #2 A finished meeting can be re-transcribed with MOSS and the result appears beside the live transcript
- [ ] #3 No CUDA host → the action is hidden or explains why
<!-- AC:END -->
