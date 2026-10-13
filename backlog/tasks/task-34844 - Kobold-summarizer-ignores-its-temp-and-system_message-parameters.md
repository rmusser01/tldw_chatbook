---
id: TASK-34844
title: Kobold summarizer ignores its temp and system_message parameters
status: To Do
assignee: []
created_date: '2026-10-12 12:10'
labels: [bug, providers]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-34812 review: summarize_with_kobold accepts temp and system_message but never reads either — the payload hardcodes "temperature": 0.7 (Local_Summarization_Lib.py:638) and the prompt ignores the system message. The TASK-34812 routing fix was therefore behaviorally inert for those two parameters. Either wire them into the payload/prompt (matching sibling local summarizers) or delete the dead parameters so the next reader does not assume they matter; pick one and pin it with a test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 temp and system_message are either honored in the kobold payload/prompt or removed from the signature, with the choice recorded in Implementation Notes.
- [ ] #2 A test pins the chosen behavior.
<!-- AC:END -->
