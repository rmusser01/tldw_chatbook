---
id: TASK-39108
title: 'audio_file_path_is_safe logs raw and resolved DB paths at WARNING on refusal'
status: To Do
assignee: []
created_date: '2026-11-06 09:00'
labels:
  - security
  - privacy
dependencies: []
references:
  - tldw_chatbook/Utils/path_validation.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`audio_file_path_is_safe` logs the raw and resolved DB path at WARNING when it refuses a path, for callers that do not redact (`path_validation.py:274-276`; for example the `daily_reports_view` listing). It exists on dev today; this task's PR does not fix it. Found by the Roleplay B2b follow-up burn-down (2026-11-05), read on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A refusal logs no raw or resolved path unless the caller asked for it
- [ ] #2 A test pins the log text for a non-redacting caller
<!-- AC:END -->
