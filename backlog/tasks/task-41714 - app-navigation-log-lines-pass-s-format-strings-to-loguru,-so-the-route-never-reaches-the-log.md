---
id: TASK-41714
title: "app_navigation log lines pass %s format strings to loguru, so the route never reaches the log"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- logging
priority: low
references:
- tldw_chatbook/app_navigation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41500 and confirmed by its review (round 16). Several navigation log lines (for example "Reusing installed screen instance ... route: %s") pass `%s` to loguru, which does not substitute it, so the log prints a literal `%s`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every loguru call in `app_navigation.py` uses `{}` placeholders or f-strings and logs its values
- [ ] #2 A grep or test guards against `%s` in loguru calls in that module
<!-- AC:END -->
