---
id: TASK-42024
title: "Native credential recovery children close TTS without waiting for it"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- tests
priority: low
references:
- Tests/ProductionApp/test_native_credential_recovery.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41717 (round 17). Several child scripts in `test_native_credential_recovery.py` call `tts_service.close()` without `await tts_service.wait_closed()`, the cause TASK-41717 fixed for the startup-handoff children (a TTSService shutdown `CancelledError` printed on every run).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each child awaits `wait_closed()` after `close()`, and its log shows no TTS shutdown traceback
<!-- AC:END -->
