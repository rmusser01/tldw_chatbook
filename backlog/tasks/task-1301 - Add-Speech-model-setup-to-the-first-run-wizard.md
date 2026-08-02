---
id: TASK-1301
title: Add Speech model setup to the first-run wizard
status: Done
assignee: []
created_date: '2026-07-29 00:30'
labels:
  - stt
  - artifacts
  - onboarding
  - ui
dependencies:
  - TASK-596
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Use the reusable model-artifact setup controls in the Full first-run track so users can explicitly download, activate, and configure a transcription model without creating a second onboarding flow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Full setup track includes a skippable Speech transcription step while Quick setup remains unchanged.
- [x] #2 The step defaults to English and presents Parakeet v2 INT8 as the recommended managed model, with supported language and precision changes using the canonical STT policy and catalog.
- [x] #3 Before downloading, the step shows the exact dependency closure, source revision, license, precision, bytes, destination, staging requirement, and free-space result and requires explicit consent.
- [x] #4 Download, verification, activation, cancellation, retry, and installed-state behavior reuse the TASK-596 controls and TASK-595 service without duplicate artifact or network logic.
- [x] #5 Transcription configuration is persisted only after a verified artifact is active; re-running the wizard prefills the persisted language, model, and precision without exposing secrets.
- [x] #6 Skip and failures never trap the user, and the final Summary reports persisted transcription configuration and installed readiness rather than transient widget state.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Claimed 2026-08-01 (voice-stream session, branch feat/wizard-speech-setup-1301) -- check this note
before parallel work, per the TASK-595/596 duplicate-implementation guard.

1. Scout the shipped TASK-596 Phase 1 controls (Widgets/ModelArtifacts/{install_modal,
   install_progress,activation_controls}.py, UI/Screens/model_browser_state.py) and the TASK-595
   ModelArtifactService: their real construction/consent/progress/activation contracts.
2. Scout the Full first-run wizard track (UI/Wizards/FirstRunSetupWizard.py): step registration,
   skip semantics, summary composition, Quick-track isolation.
3. Add a skippable Speech transcription step to the Full track only: recommended Parakeet v2 INT8,
   language/precision changes through the canonical STT policy + catalog (tldw_chatbook/STT/),
   consent modal showing the full preflight (closure, revision, license, precision, bytes,
   destination, staging, free-space), reusing the 596 controls verbatim -- no duplicate artifact or
   network logic.
4. Persist transcription config ONLY after a verified artifact is active; wizard re-run prefills
   from persisted config without secrets; Summary reports persisted config + installed readiness,
   never transient widget state.
5. RED-first tests: step present+skippable in Full track, Quick track unchanged, consent gate,
   persist-only-after-active, prefill-on-rerun, skip/failure never traps, summary honesty.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped on feat/wizard-speech-setup-1301 (12 commits, three review rounds). A skippable Speech
transcription step in the Full track only (Quick byte-identical, pinned): recommended Parakeet v2
INT8 via the curated registry, language/precision from the canonical STT policy+catalog, consent
through the shared ModelInstallModal preflight, download/verify/activate/cancel/retry/installed
state entirely through the TASK-596 controls + TASK-595 service (zero duplicate artifact/network
logic, reviewer-verified), config persisted ONLY after a verified artifact is active.

Review rounds mattered: round 1 found five Importants — copy pointing at a nonexistent
"Settings ▸ Speech" (real destination is Lab ▸ Models, verified from routes), the Install button
below the fold at the wizard's own 120x40 budget, commit() clobbering an existing remote-whisper
config on a silent re-run (AC#5's prefill clause was unimplemented), no runtime gate (would download
~660MB onto installs without onnx-asr and report ✓), and no Retry/Delete on a broken artifact.
Round 2 verified all five fixed (mutation evidence) and caught three residuals from the fixes:
Rich markup eating the extras name from the recovery instruction (markup=False), a promise no
control could deliver in the installed+active+config-elsewhere state (resolved with a real
"Use Parakeet v2 as my default" affordance riding the existing commit path — new UI surface,
called out in the PR), and Summary claiming ✓ while the step said "runtime not installed"
(Summary now reflects the runtime probe). Round 3: Approved, AC#1-6 all met.

Parked as follow-ups (reviewer-endorsed): third hardcoded v3 language table awaiting a real
catalog consumer; service_factory seam not covering the install path; ~115ms wizard-open import;
cosmetic reload-window button greying. Tests: 756 across the sweep; per-file counts and all
RED/mutation evidence in .superpowers/sdd/2026-08-01-speech-setup/.
<!-- SECTION:NOTES:END -->
