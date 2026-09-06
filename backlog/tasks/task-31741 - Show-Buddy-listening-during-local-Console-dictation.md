---
id: TASK-31741
title: Show Buddy listening during local Console dictation
status: Done
assignee: []
created_date: '2026-09-05 21:48'
updated_date: '2026-09-05 23:54'
labels:
  - buddy
  - voice
  - uat
dependencies: []
references:
  - qa/buddy-uat-2026-09-05/merged-live-uat/README.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Human microphone UAT on merged Chatbook completed local faster-whisper transcription, a DeepSeek reply and audible Kokoro playback, but Migu stayed idle throughout the 20-second recording. Realtime and trusted playback already publish Buddy voice state; local dictation lacks that connection. Preserve existing ADR-074 request-owned lifecycle leases and avoid implying an active microphone merely because preparation began.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Migu shows listening while an actual local Console dictation capture is recording.
- [x] #2 Stop, cancel, capture failure and session/screen teardown release only that capture's Buddy ownership, preserving another voice owner.
- [x] #3 Model preparation does not falsely show listening before microphone startup succeeds.
- [x] #4 Targeted lifecycle tests and a bounded live microphone replay verify the state transitions and terminal cleanup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: Directly implement the existing request-owned Buddy voice lifecycle.

1. Trace capture startup and terminal paths against realtime/trusted playback.
2. Add focused failing capture/preparation/cleanup/concurrent-owner tests.
3. Wire capture-owned listening with scoped release and stale callback protection.
4. Run targeted tests and scoped Ruff/Bandit baseline comparison. Root performs bounded live microphone UAT before completing AC4.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Local dictation publishes a capture-owned listening lease through the existing ADR074 Buddy voice seam. Preparation does not acquire it; stop, failure, cancel and teardown release only that owner. Retry-dialog ownership is covered separately by TASK31756. On revision e9a1543d2, intentional human capture recognized the requested phrase locally, DeepSeek completed the reply, and Kokoro delivered 68608 bytes to a drained sink; the user confirmed hearing it clearly. Buddy was listening during capture and idle afterward. The retained successful dictation session object is intentional reuse, not the recorder handle; the live probe did not directly inspect that handle. Process exit and normal configuration integrity are verified. Fresh focused Buddy lifecycle tests: 10 passed, 1 existing dependency warning, 12.89 seconds. Earlier touched-code Ruff/Bandit comparison found no added findings; no production code changed in this acceptance update. Evidence: qa/buddy-uat-2026-09-05/merged-live-uat/README.md. Existing ADR074 applies; no new ADR required. Server browser voice and full OpenAI realtime interaction remain outside this task.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Human dictation, local transcription, DeepSeek response, audible Kokoro playback and Buddy listening-to-idle acceptance passed; 10 focused lifecycle regressions passed.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
