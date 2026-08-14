---
id: TASK-16199
title: 'TTS shutdown sweep can still zero an in-flight export source'
status: To Do
assignee: []
created_date: '2026-08-14 03:05'
labels:
  - bug
  - concurrency
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15471's fix round made TTS export and the 5s per-file cleanup mutually exclusive: the export claims the message id under `_audio_files_lock` and `_cleanup_audio_file` defers its secure-delete (which ZEROES the file in place before unlinking, `Utils/secure_temp_files.py:216-224`) while a claim is held. The re-review verified the claim logic on every path but found the one uncovered window (its N1): `cleanup_tts_resources`'s shutdown sweep (`Event_Handlers/TTS_Events/tts_events.py:2728-2742`) deletes every artifact WITHOUT consulting `_exporting_audio_refcounts`, and cancelling the export task does not stop its already-running pool thread — so quitting the app mid-copy can still produce a zeroed export file. Present before TASK-15471 too (the window was just narrower). Fix direction: make the shutdown sweep honor the claims (skip-and-log or bounded-wait), and add the one-sentence bound rationale to the cleanup poll loop the review asked for (its N2). Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The shutdown sweep cannot destroy a file an export currently claims (test or forced-interleave probe as evidence)
- [ ] #2 Shutdown still completes promptly when an export hangs (bounded wait or skip, stated)
- [ ] #3 The cleanup poll loop's termination bound is documented at the loop
<!-- AC:END -->
