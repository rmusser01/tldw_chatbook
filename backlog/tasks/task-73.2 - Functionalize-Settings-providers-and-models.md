---
id: TASK-73.2
title: Functionalize Settings Providers and Models
status: Done
labels:
- settings
- providers
- console
- configuration
dependencies:
- TASK-73.1
priority: high
parent_task_id: TASK-73
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Providers and Models the first fully functional Settings category by supporting provider catalog selection, provider+model default profiles, custom provider escape hatches, credential source status, endpoint validation, save/revert, and Console readiness agreement for all Console-supported `chat_api_call()` providers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings lists or accepts every provider identity Console can send through via `chat_api_call()`.
- [x] #2 Provider, default model, endpoint/base URL, credential source, and selected provider+model default profile values load from config and can be saved or reverted safely.
- [x] #3 API keys and tokens are masked in UI, tests, notifications, validation output, and screenshots.
- [x] #4 Provider readiness states cover ready, missing key, missing model, invalid endpoint, unsupported, WIP, and keyless local providers.
- [x] #5 Console and Settings use the same effective provider/model/config source after save.
- [x] #6 When a Console user switches models, the new model inherits its provider+model default profile unless the user explicitly applies a session override.
- [x] #7 Global sampling and transport defaults are not saved by this category; only selected provider+model profiles may save model-scoped sampling or transport defaults, and global fallbacks are routed to Console Defaults.
- [x] #8 Focused mounted and pure tests cover catalog, credentials, endpoint validation, provider+model default profiles, save/revert, Console readiness agreement, Console inheritance, and reset-to-model-defaults behavior.
- [x] #9 Actual CDP/Textual-web screenshots verify missing-key, local-endpoint, model-default profile, Console-after-save, and Console-after-model-switch inheritance states and are approved before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing pure and mounted regressions for provider+model default profiles.
2. Implement shared provider/model profile resolution for Console settings defaults.
3. Move Settings Providers & Models sampling controls from global chat defaults to selected provider+model profiles.
4. Preserve existing provider/model/endpoint save behavior and add profile save/revert behavior.
5. Run focused verification and capture actual CDP/Textual-web screenshots for user approval before PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added provider+model profile resolution so Console defaults prefer `api_settings.<provider>.model_defaults.<model>` before provider/global fallbacks.
- Updated Settings Providers & Models to own provider, model, endpoint, credential env var, and selected model profile fields while routing global sampling/transport defaults to Console Defaults.
- Added safe save/revert handling for provider endpoints, credential env vars, and model profile values without persisting unedited stale profile values after model switches.
- Added focused pure and mounted regressions for model profile loading/saving, Console model-switch inheritance, credential validation, endpoint handling, and provider readiness agreement.
- Captured actual Textual-web/CDP screenshots for Settings provider/model profile and Console inherited defaults under `Docs/superpowers/qa/product-maturity/screen-qa/settings/`; user approved both screenshots before PR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Providers & Models is now a functional Settings category for Console-backed provider defaults. It can load, validate, save, and revert provider identity, model, endpoint, credential env var, and selected provider+model profile defaults, and Console inherits those defaults when the selected model changes unless the session has an explicit override.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Unit, mounted, and integration tests relevant to changed behavior pass
- [x] #3 Static analysis and diff hygiene checks pass
- [x] #4 Actual app QA walkthrough completed with screenshots
- [x] #5 User approval recorded for visible Settings changes
- [x] #6 Documentation and task notes updated
- [x] #7 Task status moved to Done after implementation notes are added
<!-- DOD:END -->
