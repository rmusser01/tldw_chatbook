---
id: TASK-31826
title: 'Meetings: cross-meeting voiceprint speaker enrollment'
status: To Do
assignee: []
created_date: '2026-09-05 22:49'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remember a speaker's voice as a named person across meetings so a known voice is auto-named in future recordings. Deferred from the phase-2 diarization design (Docs/superpowers/specs/2026-09-05-meeting-diarization-design.md section 10). Needs its own design: a voiceprint store, match thresholds, and a consent/privacy surface for storing biometric voice data. Also the vehicle for making after-the-fact speaker names portable across devices (store the name map in the synced DB rather than the local meeting folder).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design approved
<!-- AC:END -->

## Renumbering provenance

Renumbered from TASK-31741 to TASK-31826 on 2026-09-06 when `feat/meeting-diarization` was brought up to date with dev: TASK-31741 had already been taken on dev by a Canvas task (the older arrival keeps the id, per the TASK-19601 owner rule). No dependencies: entries or doc/code references pointed at the old id.
