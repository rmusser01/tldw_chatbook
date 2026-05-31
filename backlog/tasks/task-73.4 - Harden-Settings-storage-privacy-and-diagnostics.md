---
id: TASK-73.4
title: Harden Settings Storage Privacy and Diagnostics
status: Done
labels:
- settings
- storage
- privacy
- diagnostics
dependencies:
- TASK-73.1
priority: high
parent_task_id: TASK-73
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Storage, Privacy, and Diagnostics actionable and safe by exposing real validation, redaction, reload, and recovery paths without moving files or leaking secrets unless a later task explicitly adds a guarded mutation path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Storage validates current config/data/database paths and reports writable, not writable, missing, or invalid states.
- [x] #2 Privacy reports secret redaction, key source, encryption status, and local/server data boundaries without exposing raw credentials.
- [x] #3 Diagnostics validates and reloads config with redacted errors and clear config source copy.
- [x] #4 Any unsupported mutation path is labeled unavailable/WIP instead of appearing actionable.
- [x] #5 Focused tests cover path validation, diagnostics reload, and no secret leakage.
- [x] #6 Actual CDP/Textual-web screenshot QA verifies Storage, Privacy, and Diagnostics and is approved before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit existing Storage, Privacy, Diagnostics helpers and mounted tests against TASK-73.4 acceptance criteria.
2. Add failing regressions for actionable path status rows, privacy redaction/key-source/encryption posture, diagnostics config-source/redacted reload output, and disabled/WIP mutation affordances.
3. Implement the smallest Settings adapter/screen changes to pass the regressions without adding unsafe storage mutation or raw secret exposure.
4. Run focused Settings tests, static diff checks, and targeted compile checks.
5. Capture actual Textual-web/CDP screenshots for Storage, Privacy, and Diagnostics and wait for user approval before PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added Storage path readiness reporting for existing, missing, invalid, and unwritable config/data/database locations without creating or moving files.
- Expanded Privacy checks to show redacted secret posture, provider key sources, encryption status, and local/server data boundaries without exposing raw credential values.
- Expanded Diagnostics validation/reload output with redacted config source copy and preserved Advanced Config as the only raw-edit path.
- Labeled unsupported Storage, Privacy, and Diagnostics mutation paths as unavailable/WIP and kept guided Save/Revert disabled for those read-only categories.
- Added mounted/helper regressions in `Tests/UI/test_settings_configuration_hub.py` for WIP labels, missing-path parent readiness, privacy boundary output, and redacted diagnostics config-source output.
- Captured approved Textual-web QA screenshots in `Docs/superpowers/qa/settings-configuration-hub/`: `stage-4-storage.png`, `stage-4-privacy.png`, and `stage-4-diagnostics.png`.
- Verification run: `python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short` passed with 114 tests.
- Verification run: `python -m py_compile tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_configuration_hub.py` passed.
- Verification run: `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Settings Storage, Privacy, and Diagnostics now act as safe validation hubs: they surface actionable read-only readiness, redacted privacy and credential-source posture, and config validation/reload evidence while explicitly marking unsupported mutation paths as unavailable/WIP.
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
