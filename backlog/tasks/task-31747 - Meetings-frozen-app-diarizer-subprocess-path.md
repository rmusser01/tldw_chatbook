---
id: TASK-31747
title: 'Meetings: frozen-app diarizer subprocess path'
status: To Do
assignee: []
created_date: '2026-09-06 07:40'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The local diarizer subprocess is spawned via [sys.executable, -m tldw_chatbook.Audio.diarizer_worker], which works in a dev checkout but not necessarily in the packaged macOS .app / Windows build. Implement and verify the frozen-app path on an actual packaged build (multiprocessing spawn with freeze_support, or a bundled worker entry point), or wire the documented worker-thread fallback (accepting GIL contention). Today a frozen build gracefully degrades to coarse labels. Deferred from the phase-2 diarization SDD run (spec §3.4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Live diarization works in a packaged build, or the documented thread fallback is wired and verified
<!-- AC:END -->
