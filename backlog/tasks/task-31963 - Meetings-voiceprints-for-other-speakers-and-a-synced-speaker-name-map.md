---
id: TASK-31963
title: 'Meetings: voiceprints for other speakers and a synced speaker-name map'
status: To Do
assignee: []
created_date: '2026-09-07 05:49'
labels:
  - audio
dependencies:
  - TASK-31826
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-31826 shipped the first slice (the local user's own voiceprint). The remainder of its description is still open: remember OTHER people's voices as named persons across meetings (consent surface, per-person store entries, matching several voiceprints per meeting) and make after-the-fact speaker names portable across devices by storing the name map in the synced DB rather than the local meeting folder. Needs its own design (privacy/consent is central: these are other people's biometrics).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design approved (consent model, storage, matching several voiceprints, threshold policy)
- [ ] #2 A named person enrolled from a past meeting is auto-named in a later meeting with a marker and easy override
- [ ] #3 Speaker names edited after the fact are visible on another device with the same synced DB
<!-- AC:END -->
