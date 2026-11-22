---
id: TASK-42423
title: "Three Console staged-evidence session tests are red on dev and run on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-22 22:10
updated_date: 2026-11-22 22:10
labels:
- follow-up
- console
- context-sources
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_console_staged_evidence_sessions.py
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/console_received_dispatch.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22): three of the 12 local reds TASK-42342's covering run reported on dev ddd3c02052 (its notes, line 84). Re-checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports), whose ports touch none of these files; paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** `Tests/UI/test_console_staged_evidence_sessions.py` is the evidence for TASK-33620.7's per-chat staged-evidence fix (its AC#5): one chat's staging must not leak into, or be consumed by, another chat's send. Three of its tests fail every time, in the file and in an isolated `-n 4` run. The file is in no PR step and not in the UI census, so nothing reports them. TASK-33620.7 has been In Progress since 2026-10-31 and its subtasks (TASK-33620.7.1-.3) cover other work, so these reds are filed here rather than parked on it.

**Failing nodes and messages.**
- `test_a_send_in_one_chat_leaves_another_chats_staging_alone` (line 304): `1 replies did not arrive: [('user', 'complete', 'unrelated'), ('assistant', 'complete', 'accepted')]`.
- `test_staged_evidence_follows_one_turn_and_one_chat` (line 369): `2 replies did not arrive: [('user', 'complete', 'question'), ('assistant', 'complete', 'accepted'), ...]`.
- `test_original_stock_handler_accepts_b_without_consuming_a_staging` (line 795): `assert record.inputs.staged_evidence_revision is None` fails with `0`.

The file was last changed by 371407f1e4 (TASK-41643, 2026-11-18). The same runs log a handled `ValueError: Local chat dictionary backend is unavailable.` from the Active dictionaries summary refresh (`UI/Console_Modules/retrieval.py:497`); whether it is related is not known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the three failures has a recorded cause: test drift after a named product change, or a product defect (a staged revision consumed by the wrong chat's send would be one)
- [ ] #2 Drift is repaired in the tests; a defect is fixed in product code with a test shown red first
- [ ] #3 The file passes alone and in a `-n 4` run
- [ ] #4 The file runs in a PR lane, with its serial time under the lane's flags recorded in the notes
<!-- AC:END -->
