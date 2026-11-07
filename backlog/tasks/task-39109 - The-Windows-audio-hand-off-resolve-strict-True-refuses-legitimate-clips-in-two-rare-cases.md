---
id: TASK-39109
title: 'The Windows audio hand-off resolve(strict=True) refuses legitimate clips in two rare cases'
status: To Do
assignee: []
created_date: '2026-11-06 09:00'
labels:
  - tts
  - windows
dependencies: []
references:
  - tldw_chatbook/Utils/path_validation.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Windows hand-off's `resolve(strict=True)` refuses legitimate clips in two rare cases: drives where `GetFinalPathNameByHandle` fails (some RAM-disk or virtual filesystems), and paths over 260 characters without LongPathsEnabled, which keep their `\\?\` prefix. It exists on dev today; this task's PR does not fix it. Found by the Roleplay B2b follow-up burn-down (2026-11-05), by reasoning from the code, not reproduced on Windows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both cases are handled, or documented in the User Guide as a known limit with the user-facing refusal text
- [ ] #2 The decision is recorded in the task notes
<!-- AC:END -->
