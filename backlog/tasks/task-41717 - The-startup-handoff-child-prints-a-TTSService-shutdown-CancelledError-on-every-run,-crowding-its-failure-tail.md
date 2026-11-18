---
id: TASK-41717
title: "The startup-handoff child prints a TTSService shutdown CancelledError on every run, crowding its failure tail"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- tests
- backup
priority: low
references:
- Tests/Backup_Recovery
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41532 (round 16). The startup-handoff child prints `unhandled exception during asyncio.run() shutdown` for `TTSService._complete_shutdown` (CancelledError) on every run, passing or failing. `_run` shows only the last 6000 characters of a failure, so this noise pushes the real assertion out of view.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The child's shutdown prints no CancelledError traceback, by fixing TTS shutdown or the child's teardown
- [ ] #2 A failing child's tail shows its own assertion first
<!-- AC:END -->
