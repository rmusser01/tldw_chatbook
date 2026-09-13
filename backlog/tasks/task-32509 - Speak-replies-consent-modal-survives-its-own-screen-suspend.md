---
id: TASK-32509
title: Speak-replies consent modal survives its own screen suspend
status: Done
assignee:
  - '@robert'
created_date: '2026-09-12 21:46'
updated_date: '2026-09-12 21:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT on merged dev (PR #2638 follow-up) found the Speak-replies enable flow broken: pushing the destination-consent modal SUSPENDS the ChatScreen underneath, and on_screen_suspend unmounts the ConsoleAutoSpeakCoordinator -- whose unmount tombstones the pending modal callback. The user's Enable press arrives dead: consent is silently discarded and the switch snaps back OFF. Introduced by TASK-31520 (fed0b7257a) moving the coordinator unmount into the suspend path; invisible to the suite because the consent tests mock the modal opener without a real push. Live-verified repro and fix against a real app boot (Kokoro ONNX TTS, DeepSeek chat): consent persists, disposition computes SPEAK, Kokoro synthesizes WAV, and the streaming sink opens the real output device.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mounted test with a REAL pushed consent modal pins that the coordinator stays mounted while its modal is open and that Enable persists auto_speak
- [x] #2 Verified by the test failing on dev HEAD and passing with the fix
- [x] #3 Suspend quiesce still unmounts the coordinator when no consent modal is open (original TASK-31520 behavior preserved)
- [x] #4 Existing consent/auto-speak suites stay green (43-file suite 44 passed; modal-dismissal failures identical to pristine dev)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: mounted test with real modal push (coordinator unmounted mid-consent on dev HEAD)\n2. Fix: modal_open property + suspend-path guard\n3. GREEN + auto-speak/fleet-wake/hands-free suites\n4. Live audible verification (consent persists, Kokoro WAV speaks through the sink)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed and live-verified; see Implementation Notes.
<!-- SECTION:NOTES:END -->
