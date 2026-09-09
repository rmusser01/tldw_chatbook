---
id: TASK-32115
title: Settle Buddy speech when synthesis is superseded
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:04'
updated_date: '2026-09-09 04:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Console speech request can replace a Buddy utterance before its first audio, canceling synthesis without settling the Buddy playback lifecycle. The waiting utterance then blocks its serial queue. Preserve exact playback ownership while completing the interrupted request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Replacing a guarded Buddy utterance before audio begins settles it as unsuccessful and leaves the replacement request active.
- [ ] #2 The Buddy speech queue can continue after an interrupted generation without leaking a pending utterance or stopping another owner.
- [ ] #3 A regression reproduces the missing terminal event before the fix; targeted Buddy, Console admission, cancellation and playback lifecycle tests pass.
- [ ] #4 Cancellation of a replacement admission while old synthesis cleanup is pending still settles the old utterance and allows its queue to continue, without admitting or stopping the cancelled replacement.
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
