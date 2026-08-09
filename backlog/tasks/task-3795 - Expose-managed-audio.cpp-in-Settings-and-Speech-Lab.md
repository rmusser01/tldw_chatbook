---
id: TASK-3795
title: Expose managed audio.cpp in Settings and Speech Lab
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-09 02:38'
updated_date: '2026-08-09 02:49'
labels:
  - tts
  - audio-cpp
  - settings
  - speech-lab
  - uat
dependencies:
  - TASK-3792
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
  - backlog/decisions/040-speech-lab-current-result-and-auto-play.md
documentation:
  - Docs/superpowers/specs/2026-08-02-audio-cpp-managed-lifecycle-design.md
  - Docs/superpowers/specs/2026-08-02-speech-lab-current-result-ux-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the completed managed audio.cpp runtime usable through the canonical Global Settings and Speech Lab surfaces, with truthful saved-versus-active lifecycle state, safe diagnostics, and a verified first-time character-roleplay workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Global Settings exposes explicit External server and Managed local server modes, shows only the selected mode primary fields, and keeps advanced lifecycle and common safety limits discoverable.
- [ ] #2 A user can detect or pick a prebuilt audiocpp_server binary and pick an existing server.json, receives field-specific safe validation and trust/loopback/working-directory guidance, and saving performs no launch, probe, catalog refresh, or synthesis.
- [ ] #3 Saving changed Managed settings or External mode while a managed child is active preserves that child and truthfully presents saved, applied, and pending state with a clear handoff to Speech Lab.
- [ ] #4 Speech Lab presents one audio.cpp runtime card with truthful saved/applied/process generations, process and capability state, active endpoint, catalog freshness, pending relation, and a link to Global Settings without duplicating durable fields or launching on mount.
- [ ] #5 Speech Lab exposes the specified state-specific Start/Test, Restart/Apply, Shutdown, Refresh, and Generate behavior asynchronously, keeps incompatible actions visibly disabled with reasons, and clearly distinguishes playback Stop from server Shutdown.
- [ ] #6 Recent managed diagnostics remain collapsed by default, bounded and sanitized, identify process generation and stream, warn that output may be sensitive and clears on restart, and cannot launch, persist, or alter lifecycle state.
- [ ] #7 The managed Settings and Speech Lab surfaces preserve focus, non-color status cues, keyboard accessibility, and the status/primary-action/current-result hierarchy at narrow terminal widths.
- [ ] #8 User setup and recovery documentation covers binary/server.json ownership, lazy start, saved-versus-active behavior, diagnostics privacy, restart, shutdown, crash recovery, and switching back to External mode.
- [ ] #9 A real-binary first-time-user UAT saves without launching, starts exactly one managed process, discovers the configured multi-model catalog, generates and audibly plays a character-roleplay WAV, proves restart/apply, unexpected-exit recovery, explicit shutdown/lazy restart, External apply, and final app cleanup without modifying user-provided artifacts.
- [ ] #10 External audio.cpp behavior and complete-WAV playback remain unchanged, and normal automated tests require no audio.cpp binary, models, downloads, external network, or audio hardware.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve a full two-mode audio.cpp Settings draft and durable mapping while keeping runtime projection active-mode-only; cover dormant-field round trips and backward compatibility first.
2. Add safe managed artifact validation, explicit binary detection, and executable/server.json picker behaviors with no process or network work on save.
3. Render the External/Managed global Settings form, lifecycle timing and common limits, focus-safe responsive behavior, and truthful saved/applied pending guidance.
4. Add one coherent passive TTSService runtime observation for saved/applied modes and generations plus process state, with no adapter materialization or launch.
5. Add the Speech Lab runtime card, state-specific lifecycle actions, passive refresh fencing, collapsed sanitized diagnostics, Settings link, and responsive/accessibility coverage.
6. Document first-time setup and recovery, run focused and broader automated verification plus privacy/mutation checks, then complete real-binary character-roleplay audible UAT and task closeout.

ADR required: no new ADR.
ADR path: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md; backlog/decisions/039-global-and-studio-tts-settings-ownership.md; backlog/decisions/040-speech-lab-current-result-and-auto-play.md.
Reason: the accepted ADRs and managed-lifecycle spec already decide ownership, persistence, lifecycle behavior, diagnostics, and complete-WAV UX; this task implements approved Slice 5.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Focused Settings, Speech Lab, lifecycle, diagnostics, responsive, accessibility, and no-passive-launch tests pass and mutation checks prove the new guards can fail.
- [ ] #2 Affected broader TTS, Settings, Console, Roleplay, and application-ownership suites pass or exact unchanged dev failures are documented from identical commands.
- [ ] #3 Ruff, formatting, mypy, compileall, boundary/privacy scans, and cumulative branch diff checks pass.
- [ ] #4 The real-binary UAT evidence records only sanitized binary/version, capability, generation, WAV metadata, process-cleanup facts, and the user audible-playback confirmation.
- [ ] #5 ADR-023, ADR-039, ADR-040, the managed-lifecycle spec, user documentation, and task implementation notes remain traceable and current.
<!-- DOD:END -->
