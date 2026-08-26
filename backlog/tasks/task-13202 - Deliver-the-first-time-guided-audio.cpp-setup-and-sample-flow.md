---
id: TASK-13202
title: Deliver the first-time guided audio.cpp setup and sample flow
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 17:39'
updated_date: '2026-08-10 22:35'
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
- [x] #1 Global Settings clearly separates External, user-provided server.json, and Guided Managed sources; Guided setup can detect or browse to the separately installed server, add/scan reviewed packages, select a default model and backend, and preserves dormant source values.
- [x] #2 The package review shows exact family/variant, task capability, compatibility/evidence state, model ID, path-safe summary, lazy-load/memory behavior, and recovery guidance without claiming Chatbook downloads the server or supports an unevidenced tuple.
- [x] #3 Save performs no launch, socket, HTTP, catalog, synthesis, generated-artifact, or model-file side effect; it reports Configuration saved — ready to test and hands focus to the single current Speech Lab primary action while leaving Studio preferences separate.
- [x] #4 Speech Lab derives label, operation, enabled state, reason, tooltip, progress copy, and post-action focus from one immutable observation, including Start & Generate Sample, Restart & Apply Settings, Retry Sample, Test Connection, and Shutdown without duplicate restart actions.
- [x] #5 A deliberate first sample starts one shared child, verifies the selected catalog entry, generates a structurally valid complete WAV, and presents prominent Play/Pause, duration/status, safe provenance, Generate again, and Save WAV controls; autoplay changes only through the existing optional Studio preference.
- [x] #6 Multiple accepted models remain registered in the one lazy child, changing the selected model reuses that child, and the UI truthfully warns that loaded models may remain resident until shutdown rather than promising unload.
- [x] #7 Console and Roleplay Speak actions lazily use the exact captured global or character selection without passive browsing launches, and provider switches or late lifecycle/catalog/generation results cannot relabel, disable, or execute a stale visible action.
- [x] #8 The full guided flow is keyboard-operable and screen-reader/non-color legible at supported narrow widths, with current disabled reasons, focus restoration, bounded scrollable diagnostics, and live announcements that do not spam progress ticks.
- [x] #9 User documentation and pinned first-time macOS/Linux UAT demonstrate install-server-separately, package selection, side-effect-free Save, sample generation, audible playback, multi-model reuse, restart/apply, crash recovery, explicit shutdown, and unchanged External/manual-json behavior without editing JSON.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the complete Global Settings Guided source flow: separately installed server selection, bounded reviewed-package scanning, exact model/backend/default selection, dormant-source preservation, passive validation/Save, and the direct Speech Lab handoff.
- Added one immutable Speech Lab audio.cpp action projection and the deliberate first-sample journey, including complete-WAV playback, duration and safe provenance, regenerate/save controls, one-child multi-model reuse, lifecycle recovery, and captured Console/Roleplay selections.
- Kept downloads, model-library management, clone-profile authoring, streaming audio, and lifecycle redesign outside this task. Real `release-0.5.1` evidence showed the standalone PocketTTS GGUF requires separate voice material, so it remains registered and truthfully marked voice-required instead of being presented as text-ready.
- Independent review found and verified fixes for host/package backend admission, async Save snapshot fidelity, lifecycle-busy result fencing, and reserved-key shortcut conflicts. The expanded affected suite passed with 662 tests; Ruff, compileall, CSS bundle sync, `git diff --check`, scoped mypy, and privacy checks passed. Existing broad Settings-model mypy findings remain pre-existing baseline errors.
- Exact-commit UAT at `3ad24a5180579d91924f8829d9953d48a5653589` passed on macOS arm64 with Homebrew audio.cpp `0.5.1` and on provisioned Linux arm64 with the pinned `release-0.5.1` CPU build. Evidence covers side-effect-free Save, a structurally valid Supertonic WAV, two-model reuse, restart/apply, crash recovery, shutdown, manual JSON, External ownership, and zero leaked task-owned children; the macOS human audible checkpoint is recorded separately in the QA evidence.
- ADR required: no. Existing ADR-039, ADR-040, and ADR-050 remain authoritative; this implementation did not change their ownership or lifecycle boundaries.
<!-- SECTION:NOTES:END -->
