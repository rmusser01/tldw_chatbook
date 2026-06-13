---
id: TASK-73.7
title: Polish Advanced Config and close out Settings QA
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-01 05:29'
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
documentation:
  - Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
parent_task_id: TASK-73
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Polish Advanced Config as the expert escape hatch and close out Settings with actual-use QA that proves category configuration, recovery, keyboard use, and cross-screen behavior work end to end.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Advanced Config keeps validation-before-save, stale-validation warning, backup-on-save, redacted errors, and recovery copy.
- [x] #2 Raw TOML is positioned as expert fallback, with guided category paths preferred where available.
- [x] #3 Full Settings walkthrough verifies provider configuration to Console, Console defaults, Storage, Privacy, Diagnostics, Advanced Config, and keyboard-only operation.
- [x] #4 Final CDP/Textual-web screenshot evidence covers every functional Settings category and is approved.
- [x] #5 Focused Settings, Console, provider readiness, and diff hygiene checks pass.
- [x] #6 Parent Settings configuration hub task is updated with final implementation notes and residual risks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing Advanced Config regressions for safe backup recovery and guided category escape paths.
2. Implement minimal Advanced Config affordances: load backup into editor without saving, require validation before save, and provide guided category jump buttons.
3. Preserve existing validation-before-save, stale validation, backup-on-save, and redaction behavior.
4. Run focused Settings tests and diff hygiene.
5. Capture actual CDP/Textual-web screenshot evidence for user approval before PR closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added Advanced Config guided category path buttons for Providers, Console, Storage, Privacy, and Diagnostics so users can leave Raw TOML for safer guided settings.
- Added a `Load Backup` recovery action that loads the `.bak` file into the editor as a preview without writing the active config file.
- Kept Save disabled after backup preview until the exact preview text is validated, preserving validation-before-save and stale-validation behavior.
- Added mounted regressions for backup-preview recovery and guided category escape paths.
- Captured and approved actual Textual-web/CDP screenshot evidence at `Docs/superpowers/qa/settings-configuration-hub/closeout-advanced-config-recovery-cdp.png`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Advanced Config is now a safer expert escape hatch: Raw TOML remains guarded by validation and backup policy, while users have clear paths back to guided Settings categories and can preview backups before saving.
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
