---
id: TASK-41531
title: "A manual mix-in refusal after a binding or persistence change still reads \"Mix-in attempt could not be saved\""
status: To Do
assignee: []
created_date: 2026-11-17 05:39
labels:
- follow-up
- console
- copy
priority: low
references:
- tldw_chatbook/Chat/console_chat_controller.py
dependencies:
- TASK-41137
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-41137 fixed only the close-during-save case. A refusal caused by a binding or persistence change still shows "Mix-in attempt could not be saved; response was not sent." (`console_chat_controller.py:24627`), which does not say what changed or what to do. Noted in the Console review (round 15) as pre-existing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each reason the snapshot is unavailable (binding changed, persistence failed) shows copy naming it and the next step
- [ ] #2 A test per reason pins the copy
<!-- AC:END -->
