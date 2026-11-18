---
id: TASK-41649
title: "Accept a queued Send's capture by a runtime-owned token, not structural equality"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- security
priority: low
references:
- tldw_chatbook/UI/Console_Modules/prompt_queue.py
- tldw_chatbook/Chat/console_configuration_preparation.py
dependencies:
- TASK-41309
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41309 fixed queued Sends made before the Console attach finished. `PromptQueue._with_user_origin` copies the captured snapshot, so the seal's identity check could never hold. The seal (`Chat/console_configuration_preparation.py`, near `:2304`) now accepts the copy when it differs only by `recovery_user_origin`. That is structural equality, and ADR-356 rule 4 prefers runtime-owned authority. A token issued with the capture and carried by the queue would prove the copy came from the queue.

Also noted: queued-turn completion cannot be tested in the harness (the local chat dictionary backend is unavailable there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The queue hands the seal a runtime-owned token issued with the capture, and the seal accepts on the token
- [ ] #2 A structurally equal copy without the token is refused (test)
<!-- AC:END -->
