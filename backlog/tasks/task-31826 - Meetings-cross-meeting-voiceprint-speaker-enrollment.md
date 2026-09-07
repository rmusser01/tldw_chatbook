---
id: TASK-31826
title: 'Meetings: cross-meeting voiceprint speaker enrollment'
status: Done
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
- [x] #1 Design approved
- [x] #2 A user can enroll their own voice once (~30 s from the mic, countdown + cancel) and the voiceprint is stored encrypted at rest with export/import/delete
- [x] #3 In later meetings the user's cluster is auto-named with the display name and a visible marker; renaming it is an override; the Stop pass applies the match when no live match happened
- [x] #4 After a meeting a single non-blocking learning offer can merge an accepted clean sample into the voiceprint (with a per-meeting cap) or be declined / turned off
- [x] #5 Every failure (no keyring, locked keyring, corrupt or mismatched store, worker crash, bad vector) degrades to 'matching off' with a visible reason and never blocks Start/Stop/ingest or breaks speaker labels
- [x] #6 No vectors, audio, transcript text, speaker names, passphrases or user paths reach logs, worker descriptions or notifications; the store module is never imported at app boot
- [x] #7 User guide documents enrollment, matching, the offer, export/import/delete, privacy and the live-labels requirement
<!-- AC:END -->

## Renumbering provenance

Renumbered from TASK-31741 to TASK-31826 on 2026-09-06 when `feat/meeting-diarization` was brought up to date with dev: TASK-31741 had already been taken on dev by a Canvas task (the older arrival keeps the id, per the TASK-19601 owner rule). No dependencies: entries or doc/code references pointed at the old id.

## Implementation Plan

1. Encrypted single-voiceprint store with keyring/keyfile key modes and passphrase export/import (T1)
2. Worker-side matching ops via the command loop, torch-free tests (T2)
3. Backend enrolls after READY, fixes the first self match, exposes centroid ops (T3)
4. Session applies the match / override / Stop-pass; owner loads at Start, offers learning, enrolls from the mic (T4)
5. Meetings rail UI: status line, marker, offer, Enroll, Voice row (T5)
6. Docs, boot invariant, opt-in real-voice test (T6); whole-branch review + one fix wave

## Implementation Notes

Delivered the first slice the user chose: **only the local user's** voiceprint. Design in
`Docs/superpowers/specs/2026-09-06-meeting-voiceprint-design.md`, plan in
`Docs/superpowers/plans/2026-09-06-meeting-voiceprint.md`; built with subagent-driven development
(6 tasks, per-task reviews, whole-branch final review, one fix wave), branch `feat/meeting-voiceprint`
(head 8ec8a0f3e). Key decisions: matching runs inside the diarizer worker (Approach A); the store uses its own
random key (keyring, else an owner-only key file) so it never depends on the config-encryption password;
the key is read at meeting Start on a bounded worker thread and created only during explicit enrollment
(macOS Keychain prompt timing); voice match requires live speaker labels; a learning offer only exists once a
voiceprint does; offers lapse when the screen is left. Recorded deviations: no Settings › Meetings section
(controls on the Meetings rail; toggles are config keys); the best-similarity diagnostic is stored, not shown.
The remaining scope of this task's description (other people's voiceprints; a synced name map for
cross-device portability) is filed as TASK-31963; the diagnostic as TASK-31964; the rail's
small-terminal layout as TASK-31965.
