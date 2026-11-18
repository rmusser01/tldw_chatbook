---
id: TASK-41722
title: "Console refuses a pasted QwenCloud /v1/responses base that the adapter accepts"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- providers
priority: low
references:
- tldw_chatbook/LLM_Calls/qwencloud.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Noted by TASK-41522 (round 16), predates it. The QwenCloud adapter checks its normalised base, so a pasted `.../compatible-mode/v1/responses` works (pinned by `Tests/LLM_Calls/test_qwencloud.py`), while Console's endpoint check reads the raw value and refuses it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console and the adapter reach the same decision for a pasted QwenCloud `/v1/responses` base
- [ ] #2 A test pins both surfaces
<!-- AC:END -->
