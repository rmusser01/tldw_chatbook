---
id: TASK-32076
title: Verify and repair TTS backend audio delivery parity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 18:21'
updated_date: '2026-09-08 20:31'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need each retained TTS provider to produce complete playable replies and preserve request ownership across failure or cancellation after similar Kokoro defects were fixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every registered TTS provider is checked through effective selection and repeated request admission with actual output contracts documented.
- [x] #2 Confirmed Chatterbox complete-file encoding, non-WAV conversion, cancellation ownership and retry-state defects have regression coverage and are repaired.
- [x] #3 Confirmed provider format mismatches and shared codec failures are repaired without mislabeling raw PCM or publishing errors as successful audio.
- [x] #4 Targeted regression tests and available real playback checks document full decoded duration, content, and model or credential limitations.
- [x] #5 Supported output formats reach a compatible file player, and completion or failure belongs only to the process that played that clip.
- [x] #6 The repaired delivery paths preserve boot module and CSS budgets, with PR checks passing before integration.
- [x] #7 Qodo review findings are verified and addressed, including failed playback cleanup, retained Higgs shutdown ownership, buffer boundaries, and application-path coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md; backlog/decisions/039-global-and-studio-tts-settings-ownership.md; backlog/decisions/040-speech-lab-current-result-and-auto-play.md
Reason: Restore existing complete-file, declared-format, effective-selection and operation-ownership contracts; no new provider or storage boundary.
1. Run the targeted backend baseline and reproduce container truncation, format mismatch, cancellation and false-success cases with real codecs and transport boundaries.
2. Add failing regressions for confirmed defects; repair Chatterbox output and operation ownership, shared codec mappings, provider format handling and Higgs failure propagation.
3. Exercise every registered provider from mounted fresh controls through repeated admission and response consumption; check native audio.cpp contracts.
4. Validate available local inference and playback with complete decoding and transcript checks, documenting unavailable services or model weights.
5. Run targeted regressions, formatter/lint and required derived checks; record exact evidence and remaining limitations.
6. Address verified PR review findings with failing regressions for terminal playback failures and model shutdown ownership; complete public playback and buffer-boundary coverage before rerunning targeted checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired delivery parity across registered TTS providers: Chatterbox produces one complete encoded utterance, uses a real conversion API, retains canceled inference/process ownership, resets fallback state and bounds retained audio; Higgs rejects failed/missing/nonfinite audio and retains loading/generation threads. ElevenLabs and AllTalk now deliver their declared formats, and shared PCM/AAC muxing is correct. Provider switching preserves Global/Studio model and voice ownership. Explicit known-rate PCM supports live/fallback playback without altering raw exports or default request formats. Mac Opus uses FFplay; exact-process completion honors failure exit codes.

Validation: main Python 3.12 targeted cohort: 744 passed, 1 optional Chatterbox skip; final Chatterbox: 30 passed; Python 3.13 shared/PCM/UI: 115 passed, 16 Torch-dependent skips, plus independent PCM/adjacent/UI cohorts: 116 + 242 + 129 passed. Six real device-playback format paths retained full decoded duration and complete independently transcribed content using recorded speech at model/HTTP boundaries. No live Chatterbox/Higgs/remote/audio.cpp inference claim. All derived preflight checks pass; changed production Python and new test modules pass formatting; all changed Python syntax checks pass, new files full Ruff clean, zero introduced legacy lint diagnostics. Independent review found no actionable issues and 8 targeted review regressions passed.

ADR required: no; existing 023, 039, 040 govern these repaired contracts. Diagnostic statements reviewed before regeneration: obsolete logs removed; only fixed Opus guidance and numeric process status added, no new logging destination/content interpolation. Evidence and limits: Docs/superpowers/qa/tts-backend-parity-2026-09-08/verification.md. Updated Speech Services guide and a lesson from the afplay false-success incident. No full test sweep. TASK-1880 remains partly open for caller-scoped/default PCM selection. Task renumbered from 32051 after a 189 remote-ref / all-worktree audit found an unrelated concurrent task with that ID; highest observed was 32075.

PR #2520 follow-up: CI caught the new PCM helper on the boot path (974 modules against 973). Call-site imports and an explicit absent-at-ready pin restore 973/973; 4 census tests and 94 PCM/UI/streaming regressions pass. The unchanged dev CSS base was already 241 bytes over its 804,000 limit; shortening one existing comment saves 497 bytes, preserves every non-comment CSS token, and restores 803,744 bytes. The CSS regression passes. No budget constants were raised; existing ADR-097 governs this fix. All PR CI jobs passed on 272f074 before the review follow-up.

Qodo follow-up addresses all eight comments: failed playback now terminates both UI/event consumers and releases artifacts; Higgs retains cleanup through host cancellation and bounds actual source/float32 buffers; public playback, limit-boundary and error-field tests cover the missing paths; PCM protocol constants and Chatterbox request/yield docs are explicit. Higgs/manager/bridge cohort: 148 passed; shared limits/PCM: 43 passed; focused playback: 17 passed. All 304 distinct playback/lifecycle cases passed across broad and focused runs after stabilizing one existing scheduler-racy ordering test (38 passed module rerun). Six actual device formats again preserved full duration and complete transcripts. Fresh independent review found no further actionable issues. All 6 preflight checks, focused formatting/syntax and baseline-relative lint pass; no new logging calls and no inventory regeneration needed. Review fixes e208c06b53 passed all PR CI gates, including PR Fast Lane, derived artifacts, UI latency, CSS, backlog IDs and Linux/macOS/Windows import evidence. All eight Qodo threads are answered and resolved; its summary reports zero unresolved bugs or rule violations. The final task-record/rebase revision must retain these gates before merge. Existing formatting debt in the old utterance test is unchanged; its awaited fixture cleanup now exits without shutdown messages.

Final integration: 46156a9565 passed every hosted gate, but dev advanced with PR #2522 before merge. Rebased onto e222813571 and preserved both appended lesson entries (the sole conflict). TTS runtime, Speech UI, event-handler and test trees remain identical. Additional shared-logging/app integration checks: 387 passed; all six preflight checks pass. Boot CSS is 803,443 / 804,000 bytes after the upstream comment reduction; CSS rules are unchanged.
<!-- SECTION:NOTES:END -->
