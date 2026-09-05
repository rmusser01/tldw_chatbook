---
id: TASK-31741
title: Show Buddy listening during local Console dictation
status: In Progress
created_date: 2026-09-05 21:48
labels:
- buddy
- voice
- uat
priority: medium
references:
- qa/buddy-uat-2026-09-05/merged-live-uat/README.md
- backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
updated_date: 2026-09-05 21:59
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented local dictation Buddy listening through the existing ADR-074 request-owned voice event seam. Successful microphone startup acquires a unique dictation owner; preparation acquires nothing. State exits and screen/suspend teardown release that exact owner through the captured sink, preserving concurrent realtime/playback ownership. Session switching retains the existing capture-to-origin-draft behavior; subsequent cleanup remains owner-scoped.

Changed tldw_chatbook/UI/Console_Modules/dictation.py and added Tests/UI/test_console_dictation_buddy.py. Focused TDD: initial 5 failures (idle instead of listening), 3 passes; final focused run 9 passed (10.85s), covering preparation, failed/cancelled startup, stop/cancel/capture failure, teardown, context change, and stale previous-capture errors. No microphone or server accessed by these tests.

Scoped Ruff: 9 pre-existing findings before/after, no new findings; new test file Ruff check/format pass. Production formatter debt exists in both baseline/current file. Bandit production scan: zero findings before/after. git diff --check passes.

Bounded human microphone replay remains pending with the main task. Broader targeted dictation/streaming/readback run is being finalized; two retry-dialog tests reproduce the same Dictate… versus Dictate failure against unchanged HEAD dictation.py and are outside this change. Task remains In Progress until live UAT and main-task completion checks.
Final broader targeted command: .venv/bin/python -m pytest Tests/UI/test_console_dictation_buddy.py Tests/UI/test_console_dictation.py Tests/UI/test_console_dictation_streaming.py Tests/UI/test_console_readback_lifecycle.py -q --tb=short => 111 passed, 2 failed, 1 existing requests dependency warning in 188.11s. The two failures are test_retryable_parakeet_failure_confirms_one_replay_and_normal_insertion and test_declining_parakeet_retry_preserves_draft_and_clears_retained_audio; both independently reproduce against unchanged HEAD dictation.py (2 failed in 13.50s), preserving the same Dictate… instead of Dictate assertion. No new targeted regression identified. Final focused Buddy suite remains 9 passed. Evidence logs: /private/tmp/task31741-{red,green,final-buddy,targeted,retry-baseline}.log; baseline runner/source in /private/tmp/task31741-baseline/. No commits made; root owns live UAT and integration.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
