---
id: TASK-42587
title: "Regenerate and Edit & resend recheck the emergency stop before dispatch"
status: To Do
assignee: []
created_date: 2026-11-27 04:56
labels:
- follow-up
- console
- emergency-stop
priority: medium
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_turn_resend.py
- tldw_chatbook/Chat/console_automatic_submission.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-42560's controller review (round 27, 2026-11-26), on dev 9926834bc8 with #3786.

**Why.** #3786's in-window emergency-stop recheck (`emergency_stop_turn_refusal`) runs for Send (`console_chat_controller.py`, around line 11204) and for `_replay_message` (`console_turn_resend.py`, which Continue and Resend use). Regenerate and Edit & resend check the stop only at the screen gate, before the controller is called. A stop pressed while either action awaits its pre-dispatch work (mix-in configuration, provider resolution, skill substitution, chat dictionaries, world info) does not refuse it before dispatch, so the provider can still be called after the user pressed the stop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A stop pressed while Regenerate or Edit & resend awaits its pre-dispatch work refuses the action before any provider call, with the same copy as a stopped Send, shown by a test through the real controller that presses the stop during that await, red before and green after
- [ ] #2 A refused action leaves the transcript as it was before the action (no sibling reply or edited message is left behind)
- [ ] #3 Send, Continue and Resend keep their current stop behaviour
<!-- AC:END -->
