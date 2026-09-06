---
id: TASK-31743
title: 'Meetings: diarize the mic channel for hybrid rooms'
status: To Do
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
- [ ] #1 Mic-channel diarization is available behind a config option
- [ ] #2 Default behaviour (mic = You in call mode) is unchanged
<!-- AC:END -->
