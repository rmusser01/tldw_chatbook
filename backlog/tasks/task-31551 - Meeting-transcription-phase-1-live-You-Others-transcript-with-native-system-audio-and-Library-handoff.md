---
id: TASK-31551
title: >-
  Meeting transcription phase 1: live You/Others transcript with native system
  audio and Library handoff
status: In Progress
assignee: []
created_date: '2026-09-05 00:42'
updated_date: '2026-09-05 00:44'
labels:
  - audio
  - meetings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Record Zoom or in-person meetings from the TUI with a live labelled transcript, persist crash-safe audio, and hand the recording to Library ingest with diarization. Spec: Docs/superpowers/specs/2026-09-04-meeting-transcription-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Meetings screen records mic plus system audio on macOS and Linux and shows a live transcript labelled You/Others
- [ ] #2 Stopping a meeting produces mixed.wav plus transcript.jsonl and meeting.json in the meetings folder and queues a Library audio ingest with diarization
- [ ] #3 A meeting survives tab switches and app quit without losing recorded audio (headers patched, recovery offered on next visit)
- [ ] #4 Console dictation and hands-free refuse to start while a meeting is active
- [ ] #5 All new logic is covered by hardware-free tests and the suite is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-09-04-meeting-transcription.md tasks 1-12 in order
<!-- SECTION:PLAN:END -->
