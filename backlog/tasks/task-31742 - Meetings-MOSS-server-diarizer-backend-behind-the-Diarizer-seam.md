---
id: TASK-31742
title: 'Meetings: MOSS/server diarizer backend behind the Diarizer seam'
status: To Do
assignee: []
created_date: '2026-09-05 22:49'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a second Diarizer backend using MOSS-Transcribe-Diarize (0.9B, batch, CUDA-first, Apache-2.0), local when a CUDA GPU is present and server-hosted otherwise, selectable via the [meetings] diarizer_backend = server config reserved by the phase-2 design (Docs/superpowers/specs/2026-09-05-meeting-diarization-design.md). Needs its own design, including the off-device-audio privacy model, since the server backend sends meeting audio off the device unlike the local SpeechBrain backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design approved
<!-- AC:END -->
