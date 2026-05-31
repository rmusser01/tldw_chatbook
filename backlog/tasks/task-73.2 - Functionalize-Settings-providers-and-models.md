---
id: TASK-73.2
title: Functionalize Settings Providers and Models
status: To Do
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
- [ ] #1 Settings lists or accepts every provider identity Console can send through via `chat_api_call()`.
- [ ] #2 Provider, default model, endpoint/base URL, credential source, and selected provider+model default profile values load from config and can be saved or reverted safely.
- [ ] #3 API keys and tokens are masked in UI, tests, notifications, validation output, and screenshots.
- [ ] #4 Provider readiness states cover ready, missing key, missing model, invalid endpoint, unsupported, WIP, and keyless local providers.
- [ ] #5 Console and Settings use the same effective provider/model/config source after save.
- [ ] #6 When a Console user switches models, the new model inherits its provider+model default profile unless the user explicitly applies a session override.
- [ ] #7 Global sampling and transport defaults are not saved by this category; only selected provider+model profiles may save model-scoped sampling or transport defaults, and global fallbacks are routed to Console Defaults.
- [ ] #8 Focused mounted and pure tests cover catalog, credentials, endpoint validation, provider+model default profiles, save/revert, Console readiness agreement, Console inheritance, and reset-to-model-defaults behavior.
- [ ] #9 Actual CDP/Textual-web screenshots verify missing-key, local-endpoint, model-default profile, Console-after-save, and Console-after-model-switch inheritance states and are approved before PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Unit, mounted, and integration tests relevant to changed behavior pass
- [ ] #3 Static analysis and diff hygiene checks pass
- [ ] #4 Actual app QA walkthrough completed with screenshots
- [ ] #5 User approval recorded for visible Settings changes
- [ ] #6 Documentation and task notes updated
- [ ] #7 Task status moved to Done after implementation notes are added
<!-- DOD:END -->
