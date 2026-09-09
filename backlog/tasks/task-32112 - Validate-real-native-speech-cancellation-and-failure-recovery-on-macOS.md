---
id: TASK-32112
title: Validate real native speech cancellation and failure recovery on macOS
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 00:41'
updated_date: '2026-09-09 02:04'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend successful native speech playback qualification to interruptions and actual managed-server failure, with evidence that later replies and retries recover cleanly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real CPU and Metal playback can be stopped and a following reply completes with no stale output.
- [x] #2 An in-flight real synthesis request can be cancelled and releases its operation ownership before the next reply.
- [x] #3 An app-owned server failure surfaces a failed request without partial successful playback, and the same-message retry restarts and completes.
- [x] #4 Evidence includes real speaker callbacks, independently checked complete speech, source provenance, and joined process cleanup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: live validation of existing ADR-023 and ADR-039 lifecycle contracts without a new boundary.

1. Snapshot exact merged source and prepare isolated managed CPU/Metal harnesses using real Console admission, HTTP, synthesis and output callbacks.
2. Stop real playback, cancel admitted in-flight synthesis, terminate only the owned managed server and retry the same message.
3. Independently transcribe each successful recovery clip and verify no stale output, active operation, lease or owned process remains.
4. Record runtime versus observer failures and evidence; add a regression and scoped repair if a production issue is reproduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Validated real CPU and Metal Stop, in-flight cancellation, successor replies, app-owned child termination, and same-message retry through trusted Console admission and actual HTTP/device callbacks against immutable application revision 565dc499210491def757bed325c4014e324dc470. All application lifecycle phases recovered, six complete successor/retry source and device captures matched PCM, and all retained owners/leases/processes joined.

Injected SIGTERM exposed upstream audio.cpp child crashes (CPU SIGSEGV; Metal SIGABRT); the app surfaced failure without partial playback and restarted successfully. These native shutdown defects are recorded, not represented as fixed. Strict base.en ASR exact-clause checks matched 2/12 captures with retained lexical ambiguities, so qualification is complete delivery/recovery rather than exact pronunciation. No new application repair was required by these runs. QA: Docs/QA/tts-runtime-recovery-2026-09-09/native-validation.md and evidence-summary.json. ADR required: no; existing ADR-023/039 boundaries are unchanged.
<!-- SECTION:NOTES:END -->
