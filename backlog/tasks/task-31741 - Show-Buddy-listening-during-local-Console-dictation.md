---
id: TASK-31741
title: Show Buddy listening during local Console dictation
status: To Do
created_date: 2026-09-05 21:48
labels:
- buddy
- voice
- uat
priority: medium
references:
- qa/buddy-uat-2026-09-05/merged-live-uat/README.md
- backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Human microphone UAT on merged Chatbook completed local faster-whisper transcription, a DeepSeek reply and audible Kokoro playback, but Migu stayed idle throughout the 20-second recording. Realtime and trusted playback already publish Buddy voice state; local dictation lacks that connection. Preserve existing ADR-074 request-owned lifecycle leases and avoid implying an active microphone merely because preparation began.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Migu shows listening while an actual local Console dictation capture is recording.
- [ ] #2 Stop, cancel, capture failure and session/screen teardown release only that capture's Buddy ownership, preserving another voice owner.
- [ ] #3 Model preparation does not falsely show listening before microphone startup succeeds.
- [ ] #4 Targeted lifecycle tests and a bounded live microphone replay verify the state transitions and terminal cleanup.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
