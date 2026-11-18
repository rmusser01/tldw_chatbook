---
id: TASK-41507
title: "Headless Console sends (launch-wake, received dispatch, mobile) hit the cold-start skill trust refusal"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- console
- skills
- headless
priority: high
references:
- tldw_chatbook/Chat/console_received_dispatch.py
- tldw_chatbook/Web_Server/mobile_service.py
- tldw_chatbook/Chat/console_chat_controller.py
dependencies:
- TASK-41138
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-41138 (round 15) made a cold-Home Buddy send build the skill trust service before capture, because a send otherwise refuses with `RecoveryRequired('console_capture_source_not_ready')`. The same refusal very likely reaches the other headless senders, which call `controller.submit_draft` without building the trust service: `console_received_dispatch.py:793` and `:802`, `Web_Server/mobile_service.py:1162`, and launch-wake deliveries from a cold Console (found in the testinfra fix round; untested).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A launch-wake delivery, a received dispatch and a mobile send into a cold Console each reach the provider (or a visible, correct refusal), never `console_capture_source_not_ready`
- [ ] #2 The trust service is built in the shared submit path or each caller, with one decision recorded in the notes
- [ ] #3 Each of the three paths has a test that starts cold and is red before the fix
<!-- AC:END -->
