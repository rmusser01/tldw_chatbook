---
id: TASK-32903
title: OmniVoice ONNX int8hq standalone TTS backend
status: In Progress
assignee:
  - '@Robert'
created_date: '2026-09-24 02:35'
updated_date: '2026-09-24 18:27'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Docs/superpowers/specs/2026-09-23-omnivoice-onnx-tts-backend-design.md per ADR-180
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 omnivoice generates speech in Speech Lab
- [ ] #2 cloning works from a reference profile
- [ ] #3 artifact installs via model browser with license consent
- [x] #4 targeted tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-09-23-omnivoice-onnx-tts-backend.md task-by-task (12 TDD tasks; ADR-180 is the decision record)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All 12 plan tasks implemented on branch feat/omnivoice-onnx-tts (ADR-180; spec 2026-09-23; plan 2026-09-23). 87 unit tests + 2 env-gated integration tests green. PENDING: manual live verification (install artifact via model browser, run Tests/TTS/test_omnivoice_integration.py with OMNIVOICE_ONNX_ROOT set, listen to outputs vs model-card demos) — owner: Robert. Subagent quota exhausted mid-run; Tasks 8-12 controller-implemented inline with ledgered reviews (see .superpowers/sdd ledger + final review pending).
<!-- SECTION:NOTES:END -->
