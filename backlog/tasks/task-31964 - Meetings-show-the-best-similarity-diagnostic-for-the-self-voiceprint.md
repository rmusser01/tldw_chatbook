---
id: TASK-31964
title: 'Meetings: show the best-similarity diagnostic for the self voiceprint'
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
The voiceprint spec (section 3.5) asks for a privacy-neutral diagnostic 'best similarity seen for you in the last meeting' so users can calibrate `voice_match_threshold`. The encrypted record already has a `last_best_similarity` field but nothing computes it: the worker would track the best cosine similarity to the enrolled vector per meeting, the backend expose it, the owner persist it at Stop, and the Voice row show it. Deferred from TASK-31826 (recorded plan deviation).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The worker tracks the best similarity to the enrolled vector per meeting and the backend exposes it
- [ ] #2 The owner persists it into the encrypted record at Stop (no vectors)
- [ ] #3 The Meetings Voice row shows it as a number with a one-line hint about the threshold
<!-- AC:END -->
