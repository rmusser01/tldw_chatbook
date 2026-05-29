---
id: TASK-73.4
title: Harden Settings Storage Privacy and Diagnostics
status: To Do
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
- [ ] #1 Storage validates current config/data/database paths and reports writable, not writable, missing, or invalid states.
- [ ] #2 Privacy reports secret redaction, key source, encryption status, and local/server data boundaries without exposing raw credentials.
- [ ] #3 Diagnostics validates and reloads config with redacted errors and clear config source copy.
- [ ] #4 Any unsupported mutation path is labeled unavailable/WIP instead of appearing actionable.
- [ ] #5 Focused tests cover path validation, diagnostics reload, and no secret leakage.
- [ ] #6 Actual CDP/Textual-web screenshot QA verifies Storage, Privacy, and Diagnostics and is approved before PR.
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
