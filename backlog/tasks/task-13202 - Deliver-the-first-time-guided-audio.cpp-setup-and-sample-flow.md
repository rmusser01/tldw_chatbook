---
id: TASK-13202
title: Deliver the first-time guided audio.cpp setup and sample flow
status: To Do
assignee: []
created_date: '2026-08-09 17:39'
labels:
  - tts
  - audio-cpp
  - settings
  - speech-lab
  - uat
dependencies:
  - TASK-13201
references:
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
  - backlog/decisions/040-speech-lab-current-result-and-auto-play.md
  - backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a first-time user configure a supported local package in Global Settings and generate, play, and reuse a complete WAV from Speech Lab without editing JSON.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Global Settings clearly separates External, user-provided server.json, and Guided Managed sources; Guided setup can detect or browse to the separately installed server, add/scan reviewed packages, select a default model and backend, and preserves dormant source values.
- [ ] #2 The package review shows exact family/variant, task capability, compatibility/evidence state, model ID, path-safe summary, lazy-load/memory behavior, and recovery guidance without claiming Chatbook downloads the server or supports an unevidenced tuple.
- [ ] #3 Save performs no launch, socket, HTTP, catalog, synthesis, generated-artifact, or model-file side effect; it reports Configuration saved — ready to test and hands focus to the single current Speech Lab primary action while leaving Studio preferences separate.
- [ ] #4 Speech Lab derives label, operation, enabled state, reason, tooltip, progress copy, and post-action focus from one immutable observation, including Start & Generate Sample, Restart & Apply Settings, Retry Sample, Test Connection, and Shutdown without duplicate restart actions.
- [ ] #5 A deliberate first sample starts one shared child, verifies the selected catalog entry, generates a structurally valid complete WAV, and presents prominent Play/Pause, duration/status, safe provenance, Generate again, and Save WAV controls; autoplay changes only through the existing optional Studio preference.
- [ ] #6 Multiple accepted models remain registered in the one lazy child, changing the selected model reuses that child, and the UI truthfully warns that loaded models may remain resident until shutdown rather than promising unload.
- [ ] #7 Console and Roleplay Speak actions lazily use the exact captured global or character selection without passive browsing launches, and provider switches or late lifecycle/catalog/generation results cannot relabel, disable, or execute a stale visible action.
- [ ] #8 The full guided flow is keyboard-operable and screen-reader/non-color legible at supported narrow widths, with current disabled reasons, focus restoration, bounded scrollable diagnostics, and live announcements that do not spam progress ticks.
- [ ] #9 User documentation and pinned first-time macOS/Linux UAT demonstrate install-server-separately, package selection, side-effect-free Save, sample generation, audible playback, multi-model reuse, restart/apply, crash recovery, explicit shutdown, and unchanged External/manual-json behavior without editing JSON.
<!-- AC:END -->
