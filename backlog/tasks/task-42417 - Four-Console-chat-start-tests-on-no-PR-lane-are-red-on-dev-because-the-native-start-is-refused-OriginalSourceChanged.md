---
id: TASK-42417
title: "Four Console chat-start tests on no PR lane are red on dev because the native start is refused OriginalSourceChanged"
status: To Do
assignee: []
created_date: 2026-11-22 21:17
updated_date: 2026-11-22 22:10
labels:
- follow-up
- console
- agents
- tests
- dev-red
priority: medium
references:
- Tests/Chat/test_console_chat_start.py
- tldw_chatbook/Chat/console_chat_start.py
- tldw_chatbook/Agents/native_source_storage.py
- tldw_chatbook/Agents/activation.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22): TASK-42342's covering run reported 12 local reds on dev ddd3c02052 (its notes, line 84). Re-checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports), whose ports touch none of these files; paths are under `tldw_chatbook/` unless they start with `Tests/` or `.github/`.

**Why.** These four nodes fail every time, in their file and again in an isolated `-n 4` run of them at load 12. None runs in a PR lane: the only `test_console_chat_start.py` nodes in `derived-artifacts.yml` are two `test_stop_keeps_native_claim_*` nodes. So nothing reports them.

**Failing nodes and messages.**
- `Tests/Chat/test_console_chat_start.py::test_native_start_child_and_wake_share_original_allowance[generation]` and `[deadline]`: `assert (await controller._chat_start.start(request)).launch_status == "started"` at line 1850 fails with `'not_started' == 'started'`.
- `Tests/Chat/test_console_chat_start.py::test_native_configured_v2_initialization_and_stop_proposal[False]` and `[True]`: `assert initialized.exists(), "required native initialization never ran"` at line 2764.
- In all four, the log shows `Chat start failed (phase=preaccept, exception_type=OriginalSourceChanged)` from `Chat/console_chat_start.py:498`. That exception is raised only in `Agents/native_source_storage.py` (lines 148-342), when an original agent storage, scratch, log or writer source no longer matches its capture. 62c4cc0762 and 7447dca26e (TASK-41315, TASK-41639, 2026-11-17/18) changed those checks to compare admitted sources as a set.

TASK-41958 (open) records `OriginalSourceChanged: original agent storage source changed` in other Console send files after #3480. It may be the same cause; its criteria cover only its own files.

The other local reds from the same run are filed as their own round-21 tasks (the agent-progress live report save, the three staged-evidence tests and the read-barrier `same_session_drain` case); three agent-progress nodes are the known `RecoveryRequired: raw_source_selection_changed` trap.

**Coordination.** Laning these nodes edits `.github/workflows/derived-artifacts.yml`: coordinate with tldw-chatbook-30 (#3683 / TASK-42281) before editing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the preaccept OriginalSourceChanged refusal is recorded: test drift after a named product change, or a product defect
- [ ] #2 Drift is repaired in the tests; a defect is fixed in product code with a test shown red first
- [ ] #3 The four nodes pass, alone and in a `-n 4` run of their file
- [ ] #4 The four nodes run in a PR lane, with their serial time under the lane's flags recorded in the notes
<!-- AC:END -->
