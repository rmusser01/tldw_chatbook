---
id: TASK-660
title: Make external audio.cpp Console TTS settings coherent
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-26 04:41'
updated_date: '2026-07-26 05:12'
labels:
  - tts
  - audio-cpp
  - console
  - settings
dependencies:
  - TASK-569
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
documentation:
  - >-
    Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md
  - Docs/superpowers/plans/2026-07-25-external-audio-cpp-console-tts.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make newly saved external audio.cpp preferences immediately usable by Console Speak while preserving one application-owned TTS runtime, complete-WAV delivery through the asynchronous response interface, legacy-provider compatibility, and user ownership of the external server process.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Blank legacy audio.cpp model and voice values resolve to explicit compatible modes, while saves persist authoritative mode keys, dual-write exact values, and atomically remove stale canonical and legacy exact keys for dynamic modes.
- [ ] #2 Preference or request selection and the matching provider revision and lease are admitted atomically, settings completion remains bounded during active speech, admitted speech is not silently cancelled, and old and replacement audio.cpp instances never coexist.
- [ ] #3 Console Speak routes audio.cpp through the native TTSService and plays one validated complete WAV through the existing asynchronous response and playback lifecycle, while unassigned legacy providers retain their compatibility path.
- [ ] #4 The installed audio.cpp build passes the pinned-contract characterization gate before UAT, and Chatbook never launches, restarts, signals, supervises, or stops the external server.
- [ ] #5 Deterministic tests cover sentinel persistence, mixed-generation admission races, pending and superseded reconfiguration, native Console routing, complete-WAV cleanup, legacy regressions, and external-process non-ownership.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend ADR-023 and record the pre-existing static-analysis baseline before production changes.
2. Characterize the installed Homebrew audio.cpp build against the pinned contract; stop before runtime code if incompatible.
3. Add immutable global TTS preferences plus one atomic set/delete config mutation whose structured result distinguishes pre-replacement failure from post-replacement cache-refresh failure.
4. Make STTS settings translate Select sentinels into explicit modes and persist authoritative mode/value mutations.
5. Add distinct safe revision/reconfiguring/unavailable errors, revision-checked registry admission, and split TTSService resource admission from execution.
6. Add one app-owned request-admission coordinator that freezes preferences and acquires the matching lease under a writer-preferred gate.
7. Run config persistence off-loop inside one service-retained, serialized publication task, then perform a two-second bounded latest-generation audio.cpp handoff without cancelling admitted speech or overlapping adapters.
8. Route Console Speak through native audio.cpp complete-WAV synthesis while retaining all legacy providers behind LegacyTTSAdapter.
9. Prove external-process non-ownership with exact PID checks, run isolated first-time-user Console UAT, execute focused/repository-wide and baseline-aware static verification, update docs, and record evidence.

ADR required: yes
ADR path: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
Reason: the task strengthens the accepted provider/runtime service contract and configuration lifecycle.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Automated unit, integration, Textual, race, and cleanup tests cover every acceptance criterion and pass.
- [ ] #2 Ruff checks and formatting, compileall, focused typing checks where configured, and git diff --check pass.
- [ ] #3 ADR-023, user documentation, compatibility limitations, external-process ownership, and UAT evidence are current.
- [ ] #4 Self-review confirms the implementation stays within Slice 1 and adds no managed process or character-profile behavior.
- [ ] #5 All acceptance criteria and DoD items are checked, concise implementation notes are added, and status changes to Done only after all evidence exists.
<!-- DOD:END -->
