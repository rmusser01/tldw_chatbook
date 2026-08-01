---
id: TASK-1301
title: Add Speech model setup to the first-run wizard
status: In Progress
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
- [ ] #1 The Full setup track includes a skippable Speech transcription step while Quick setup remains unchanged.
- [ ] #2 The step defaults to English and presents Parakeet v2 INT8 as the recommended managed model, with supported language and precision changes using the canonical STT policy and catalog.
- [ ] #3 Before downloading, the step shows the exact dependency closure, source revision, license, precision, bytes, destination, staging requirement, and free-space result and requires explicit consent.
- [ ] #4 Download, verification, activation, cancellation, retry, and installed-state behavior reuse the TASK-596 controls and TASK-595 service without duplicate artifact or network logic.
- [ ] #5 Transcription configuration is persisted only after a verified artifact is active; re-running the wizard prefills the persisted language, model, and precision without exposing secrets.
- [ ] #6 Skip and failures never trap the user, and the final Summary reports persisted transcription configuration and installed readiness rather than transient widget state.
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
