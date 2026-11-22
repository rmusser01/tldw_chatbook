---
id: TASK-42417
title: "Five Console agent-start tests on no PR lane are red on dev: four chat-start nodes refused OriginalSourceChanged, and the live report save whose sub-agent never starts"
status: To Do
assignee: []
created_date: 2026-11-22 21:17
updated_date: 2026-11-22 21:17
labels:
- follow-up
- console
- agents
- tests
- dev-red
priority: medium
references:
- Tests/Chat/test_console_chat_start.py
- Tests/UI/test_console_agent_progress.py
- tldw_chatbook/Chat/console_chat_start.py
- tldw_chatbook/Agents/native_source_storage.py
- tldw_chatbook/Agents/activation.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22): TASK-42342's covering run reported 12 local reds on dev ddd3c02052. Re-checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports), whose ports touch none of these files; paths are under `tldw_chatbook/` unless they start with `Tests/` or `.github/`.

**Why.** These five nodes fail every time, in their files and again in an isolated `-n 4` run of them at load 12. Of the other seven reds: three in `test_console_agent_progress.py` are the known `RecoveryRequired: raw_source_selection_changed` trap and are not part of this task; three in `test_console_staged_evidence_sessions.py` are recorded on TASK-33620.7 (its AC#5 tests); and `test_console_tab_publication_read_barrier.py::test_actual_close_publishes_real_tabs_before_stock_read_release[same_session_drain]` is recorded on TASK-37040. None of them runs in a PR lane: the only `test_console_chat_start.py` nodes in `derived-artifacts.yml` are two `test_stop_keeps_native_claim_*` nodes, and `test_console_agent_progress.py` is in no step and not in the UI census. So nothing reports them.

**Failing nodes and messages.**
- `Tests/Chat/test_console_chat_start.py::test_native_start_child_and_wake_share_original_allowance[generation]` and `[deadline]` (line 1755): `assert 'not_started' == 'started'`.
- `Tests/Chat/test_console_chat_start.py::test_native_configured_v2_initialization_and_stop_proposal[False]` and `[True]` (line 2727): `AssertionError: required native initialization never ran` (the `native-initialized` marker file is absent).
- In all four, the log shows `Chat start failed (phase=preaccept, exception_type=OriginalSourceChanged)` from `Chat/console_chat_start.py:498`. That exception is raised only in `Agents/native_source_storage.py` (lines 148-342), when an original agent storage, scratch, log or writer source no longer matches its capture. 62c4cc0762 and 7447dca26e (TASK-41315, TASK-41639, 2026-11-17/18) changed those checks to compare admitted sources as a set.
- `Tests/UI/test_console_agent_progress.py::test_real_live_report_save_preserves_open_selection_and_navigation` (line 387, a `@private_profile_test`): in its child process, `spawn_subagent` returns `outcome=failed` and the held gateway is never entered, so `assert await asyncio.to_thread(entered.wait, 10)` fails with the parent's `ConsoleSubmitResult(accepted=True, visible_copy='done', ...)`. The child also logs `run-hooks: ... hook=unavailable engine error (exception_type=AgentActivationRequired)` for UserPromptSubmit.

TASK-41958 (open) records `OriginalSourceChanged: original agent storage source changed` in other Console send files after #3480. It may be the same cause; its criteria cover only its own files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the two groups (chat-start OriginalSourceChanged, live report sub-agent start) has a recorded cause: test drift after a named product change, or a product defect
- [ ] #2 Drift is repaired in the tests; a defect is fixed in product code with a test shown red first
- [ ] #3 The five nodes pass, alone and in a `-n 4` run of their two files
- [ ] #4 The two files (or these nodes) run in a PR lane, with their serial time under the lane's flags recorded in the notes
<!-- AC:END -->
