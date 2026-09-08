---
id: TASK-32076
title: Verify and repair TTS backend audio delivery parity
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-08 18:21'
updated_date: '2026-09-08 19:27'
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
- [ ] #6 The repaired delivery paths preserve boot module and CSS budgets, with PR checks passing before integration.
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired delivery parity across registered TTS providers: Chatterbox produces one complete encoded utterance, uses a real conversion API, retains canceled inference/process ownership, resets fallback state and bounds retained audio; Higgs rejects failed/missing/nonfinite audio and retains loading/generation threads. ElevenLabs and AllTalk now deliver their declared formats, and shared PCM/AAC muxing is correct. Provider switching preserves Global/Studio model and voice ownership. Explicit known-rate PCM supports live/fallback playback without altering raw exports or default request formats. Mac Opus uses FFplay; exact-process completion honors failure exit codes.

Validation: main Python 3.12 targeted cohort: 744 passed, 1 optional Chatterbox skip; final Chatterbox: 30 passed; Python 3.13 shared/PCM/UI: 115 passed, 16 Torch-dependent skips, plus independent PCM/adjacent/UI cohorts: 116 + 242 + 129 passed. Six real device-playback format paths retained full decoded duration and complete independently transcribed content using recorded speech at model/HTTP boundaries. No live Chatterbox/Higgs/remote/audio.cpp inference claim. All derived preflight checks pass; all changed Python formats/syntax checks pass, new files full Ruff clean, zero introduced legacy lint diagnostics. Independent review found no actionable issues and 8 targeted review regressions passed.

ADR required: no; existing 023, 039, 040 govern these repaired contracts. Diagnostic statements reviewed before regeneration: obsolete logs removed; only fixed Opus guidance and numeric process status added, no new logging destination/content interpolation. Evidence and limits: Docs/superpowers/qa/tts-backend-parity-2026-09-08/verification.md. Updated Speech Services guide and a lesson from the afplay false-success incident. No full test sweep. TASK-1880 remains partly open for caller-scoped/default PCM selection. Task renumbered from 32051 after a 189 remote-ref / all-worktree audit found an unrelated concurrent task with that ID; highest observed was 32075.

PR #2520 follow-up: CI caught the new PCM helper on the boot path (974 modules against 973). Call-site imports and an explicit absent-at-ready pin restore 973/973; 4 census tests and 94 PCM/UI/streaming regressions pass. The unchanged dev CSS base was already 241 bytes over its 804,000 limit; shortening one existing comment saves 497 bytes, preserves every non-comment CSS token, and restores 803,744 bytes. The CSS regression passes. No budget constants were raised; existing ADR-097 governs this fix. PR CI/review results remain pending.
<!-- SECTION:NOTES:END -->
