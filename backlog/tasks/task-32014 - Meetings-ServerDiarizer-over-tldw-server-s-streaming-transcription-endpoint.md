---
id: TASK-32014
title: 'Meetings: ServerDiarizer over tldw_server's streaming transcription endpoint'
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
Second TASK-31827 sub-project. tldw_server's `WS /api/v1/audio/stream/transcribe` already returns speaker ids per final segment (Silero VAD + ECAPA on the server). Design a `Diarizer` backend that uses it behind the existing seam, with an explicit off-device-audio consent model (LAN vs remote, `store_audio`), and decide how live labels, the Stop pass and voiceprint matching map onto a server that owns the embeddings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design approved (protocol adapter, consent surface, degradation, what is sent off-device and when)
- [ ] #2 `diarizer_backend = server` produces live labels from the server's speaker ids and degrades to coarse labels when the server is unreachable
- [ ] #3 No audio leaves the device without the consent state being on and visible in the rail
<!-- AC:END -->
