---
id: TASK-41956
title: "106 voice tests fail on dev under the isolated test wrapper, 48 with VoicePreparationError('unexpected')"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- voice
priority: medium
references:
- Tests/Audio
- Tests/Chat/test_console_voice_preflight.py
dependencies:
- TASK-41640
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** During TASK-41640, 106 nodes across 15 voice files failed on dev d0486978dd under the scratch wrapper; 48 are in `test_console_voice_preflight.py` with `VoicePreparationError("unexpected")` from `validate_speculative_voice_entry`. Not investigated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is recorded per group (environment, drift or defect)
- [ ] #2 Drift is repaired and defects fixed red-first
<!-- AC:END -->
