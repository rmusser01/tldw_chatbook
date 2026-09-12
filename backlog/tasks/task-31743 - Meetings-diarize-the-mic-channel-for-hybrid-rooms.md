---
id: TASK-31743
title: 'Meetings: diarize the mic channel for hybrid rooms'
status: Done
assignee: []
created_date: '2026-09-05 22:49'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In call mode the whole mic channel is labelled You, so a second person sitting next to the local user in a hybrid meeting is not separated (phase-2 diarization design section 4, hybrid limitation). Add a config option to also run the diarizer over the mic channel so co-located speakers are split, while keeping mic=You as the default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mic-channel diarization is available behind a config option
- [x] #2 Default behaviour (mic = You in call mode) is unchanged
<!-- AC:END -->

## Implementation Notes

`[meetings] diarize_mic_channel` (default false): when on in call mode, mic-channel and overlap segments are also sent to the diarizer and the Stop pass diarizes `mixed.wav`; `render_label(..., diarize_mic=True)` lets a diarized mic segment render its speaker instead of the reserved "You". Stamped into `meeting.json`; documented in `Docs/User_Guide/meetings.md`. Landed in PR #2471 (commits aa72e1f9f, e060f8d27).
