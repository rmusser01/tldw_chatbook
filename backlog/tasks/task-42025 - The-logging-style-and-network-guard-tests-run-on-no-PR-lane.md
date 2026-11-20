---
id: TASK-42025
title: "The logging style and network guard tests run on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- tests
- ci
priority: low
references:
- Tests/test_stdlib_logging_format_style.py
- Tests/test_network_guard.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41708 and TASK-41714 (round 17). `Tests/test_stdlib_logging_format_style.py` (the loguru `%s`-style and stdlib brace-style scans) and `Tests/test_network_guard.py` (including the new DNS refusal) run on no PR lane, so a regression in either guard would land unseen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both files run on a PR lane, with their times recorded and the CI pins updated
<!-- AC:END -->
