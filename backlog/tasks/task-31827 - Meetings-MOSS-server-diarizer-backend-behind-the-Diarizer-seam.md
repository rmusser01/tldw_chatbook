---
id: TASK-31827
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

## Renumbering provenance

Renumbered from TASK-31742 to TASK-31827 on 2026-09-06 when `feat/meeting-diarization` was brought up to date with dev: TASK-31742 had already been taken on dev by a Canvas task (the older arrival keeps the id, per the TASK-19601 owner rule). No dependencies: entries or doc/code references pointed at the old id.
