---
id: TASK-32115
title: Settle Buddy speech when synthesis is superseded
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 04:04'
updated_date: '2026-09-09 04:32'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Console speech request can replace a Buddy utterance before its first audio, canceling synthesis without settling the Buddy playback lifecycle. The waiting utterance then blocks its serial queue. Preserve exact playback ownership while completing the interrupted request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Replacing a guarded Buddy utterance before audio begins settles it as unsuccessful and leaves the replacement request active.
- [x] #2 The Buddy speech queue can continue after an interrupted generation without leaking a pending utterance or stopping another owner.
- [x] #3 A regression reproduces the missing terminal event before the fix; targeted Buddy, Console admission, cancellation and playback lifecycle tests pass.
- [x] #4 Cancellation of a replacement admission while old synthesis cleanup is pending still settles the old utterance and allows its queue to continue, without admitting or stopping the cancelled replacement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing speech ownership remains governed by ADR-139 and the established TTS lifecycle contracts.
Reason: routine terminal-state bug fix at the existing exact-owner cancellation boundary; no new provider, persistence, permission or queue contract.

1. Reproduce replacement before first audio through real handler admission/cancellation and the guarded utterance lifecycle.
2. Add a failing regression that also checks replacement ownership and queue continuation.
3. Settle the canceled owner at the narrow existing terminal boundary while preserving cancellation outcome semantics and exact-stop isolation.
4. Run targeted Buddy/Console lifecycle tests, review independently, then requalify the combined installed TTS source with actual playback before the PR merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Settled interrupted guarded speech through a callback on the exact cancelled generation task, registered after cancellation is accepted. A cancelled replacement waiter cannot bypass the terminal acknowledgement. Existing completed-task/mismatch early returns and successful supersession bookkeeping remain unchanged.

The real-handler/real-queue regression reproduced both ordinary supersession and cancellation during held provider cleanup, then passed all four current/stale owner and normal/cancelled replacement cases. It checks cleanup-before-terminal ordering, queue continuation, cancelled-replacement non-admission, late old Stop isolation, repeated Stop idempotence and resource cleanup. All 11 Buddy handler tests and 461 targeted integration tests passed; independent review/probes found no remaining issue. The test passes Ruff/format; the handler retains the same 57 pre-existing Ruff findings and formatting debt.

Rebased onto dev 5655c4820733d24754f872c21cc6196d83bf1504; the tested handler/test hashes remained unchanged. Final source e9866c11e8207aa637397cb1ed4624d6314b2f3e was built and installed into isolated Python 3.12/3.13 environments, with all 2,253 packaged Chatbook Python hashes matching. Nine real Speech Lab/Speak replies clips passed playback and complete-file content checks across MPS WAV, British CPU MP3 and Python 3.13 ONNX; two installed dependency-guidance checks passed. User settings unchanged and cleanup joined. All six local artifact guards pass; the only new Library diagnostic is a reviewed fixed warning string with no new sink.

Modified the shared TTS handler, Buddy regression, QA evidence and testing lesson. ADR required: no; routine repair of existing exact-owner lifecycle boundaries under ADR-139. No provider, storage, permission or queue contract changed. Evidence: Docs/QA/tts-runtime-recovery-2026-09-09/buddy-integration-validation.json; raw receipts under /private/tmp/tts-runtime-recovery-validation/buddy-rebase/. Full suite not run.
<!-- SECTION:NOTES:END -->
