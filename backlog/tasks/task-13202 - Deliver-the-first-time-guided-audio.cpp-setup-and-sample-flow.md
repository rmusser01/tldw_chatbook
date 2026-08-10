---
id: TASK-13202
title: Deliver the first-time guided audio.cpp setup and sample flow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-09 17:39'
updated_date: '2026-08-10 19:31'
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

## Implementation Plan

1. Follow `Docs/superpowers/plans/2026-08-10-task-13202-audio-cpp-first-time-guided-setup.md` test-first and keep every change within this task's acceptance criteria.
2. Round-trip and validate External, manual `server.json`, and Guided audio.cpp settings while preserving dormant values and proving Save remains side-effect free.
3. Add the keyboard-accessible guided Settings controls, bounded package scan/review, default/backend selection, truthful copy, and exact Speech Lab handoff.
4. Derive the Speech Lab primary action from one immutable runtime observation and add the fenced Start & Generate Sample flow without duplicating lifecycle actions.
5. Complete the current-result experience with WAV playback state, safe provenance, duration, Generate again, Save WAV, and the existing Studio-only autoplay preference.
6. Verify one-child multi-model reuse plus exact captured Console and Roleplay selections, changing production code only where regressions expose a task-scoped gap.
7. Complete narrow-width/accessibility checks, user documentation, and pinned macOS/Linux UAT evidence without touching unrelated audio.cpp processes.
8. Run focused and proportionate verification, self-review the diff against all acceptance criteria, record implementation notes, and close the task only after the Definition of Done is satisfied.

ADR required: no

ADR path: N/A (`ADR-039`, `ADR-040`, and `ADR-050` already apply)

Reason: TASK-13202 implements the already-approved guided setup, onboarding, and product seam without changing settings ownership, Studio preference ownership, generated-configuration ownership, or managed-lifecycle boundaries.
