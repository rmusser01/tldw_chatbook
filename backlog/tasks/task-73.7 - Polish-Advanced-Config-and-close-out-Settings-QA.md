---
id: TASK-73.7
title: Polish Advanced Config and close out Settings QA
status: To Do
labels:
- settings
- advanced-config
- qa
- release-hardening
dependencies:
- TASK-73.2
- TASK-73.3
- TASK-73.4
- TASK-73.5
- TASK-73.6
priority: high
parent_task_id: TASK-73
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Polish Advanced Config as the expert escape hatch and close out Settings with actual-use QA that proves category configuration, recovery, keyboard use, and cross-screen behavior work end to end.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Advanced Config keeps validation-before-save, stale-validation warning, backup-on-save, redacted errors, and recovery copy.
- [ ] #2 Raw TOML is positioned as expert fallback, with guided category paths preferred where available.
- [ ] #3 Full Settings walkthrough verifies provider configuration to Console, Console defaults, Storage, Privacy, Diagnostics, Advanced Config, and keyboard-only operation.
- [ ] #4 Final CDP/Textual-web screenshot evidence covers every functional Settings category and is approved.
- [ ] #5 Focused Settings, Console, provider readiness, and diff hygiene checks pass.
- [ ] #6 Parent Settings configuration hub task is updated with final implementation notes and residual risks.
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
