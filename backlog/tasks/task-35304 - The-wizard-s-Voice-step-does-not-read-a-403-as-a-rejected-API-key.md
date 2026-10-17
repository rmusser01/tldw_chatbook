---
id: TASK-35304
title: 'The wizard''s Voice step does not read a 403 as a rejected API key'
status: To Do
assignee: []
created_date: '2026-10-17 09:11'
labels:
- model-config-redesign
- follow-up
- first-run
- voice
dependencies:
- TASK-33008
references:
- tldw_chatbook/UI/Wizards/first_run_voice_step_state.py
- tldw_chatbook/UI/Wizards/first_run_voice_status.py
- Tests/Wizards/test_first_run_voice_prefill.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen while checking PR #3144 (Phase 8). The owner ruling of 2026-10-02 (PR #2958 review) is that only a 401 rejects a key; a 403 means the key may not do that request. Phase 8 applied it to the wizard's Provider and Model steps, but not to the Voice step.

At origin/dev 111f8e48fc, `first_run_voice_step_state.py` (about line 338) maps a status code in {401, 403} to the failure kind `key_rejected`, and `first_run_voice_status.py` (about line 362) renders that kind as "the speech server rejected the API key". Tests/Wizards/test_first_run_voice_prefill.py::test_server_answers_are_classified pins it: its 403 case expects `key_rejected` and "rejected the API key". The ruling was made for model listings, and the Voice Test is a speech request, so the wording a 403 gets here is for the owner to confirm.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A 403 to the Voice step's Test reads as a refusal of that request (for example that the key or server does not allow speech), not "rejected the API key", in wording the owner has confirmed
- [ ] #2 A 401 to the Voice step's Test still reads "rejected the API key"
- [ ] #3 The classification test pins the 401 and the 403 as different kinds
<!-- AC:END -->
