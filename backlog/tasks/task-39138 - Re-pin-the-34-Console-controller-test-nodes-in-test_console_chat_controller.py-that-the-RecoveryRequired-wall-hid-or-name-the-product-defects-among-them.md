---
id: TASK-39138
title: "Re-pin the 34 Console controller test nodes in test_console_chat_controller.py that the RecoveryRequired wall hid, or name the product defects among them"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/Chat/test_console_chat_controller.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_chat_store.py
- tldw_chatbook/Backup_Recovery/raw_participants.py
dependencies:
- TASK-38111
- TASK-39045
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-38111's notes (round-11 branch fu/r11-recovery) list them. Line numbers are dev's (`Tests/Chat/test_console_chat_controller.py`).

**How they surfaced.**
- With TASK-38111's profile marker, the file has 330 passed and 35 failed.
- The same 35 are red on dev under a marker plugin, and red alone: 38 nodes across three files, 38 failed at `-n 4` (load 49).
- One of the 35, `TestUserPromptSubmitHooks::test_wake_notice_fires_nothing`, is TASK-39045's.

**The other 34, by cause:**
- Test fakes refuse new keyword arguments (5):
  - `invocation_source`: `:1143`, `:7431`, `:8110[changed-history]`;
  - `validated`: `:9805`;
  - `expected_continuation`: `:11182`.
- Named-profile fakes have the wrong shape (2): `:9562` and `:9629`.
- Images and vision (8): `:2824`, `:2863`, `:2881`, `:2935`, `:3122`, `:3195`, `:3283`, `:3367`.
  - Three are sends refused with "Console send blocked: vision-model can't a…".
  - Five are media counts or history-part shapes.
- Session close and dispatch gap (3): `:1911` and `:1951` read "Accepted turn ownership changed before settlement." instead of "Session closed.", and `:1993` raises `KeyError 'Unknown Console chat session'`.
- Canvas and agent composition (9):
  - `test_real_canvas_controller_allows_exact_failed_assistant_retry` (`:6000`, 6 cases) raises `StopIteration`;
  - `:5764` and `:5825` read "Couldn't save the prepared turn";
  - `:8776` ends with its agent call state ERROR.
- Usage and the stop path (4):
  - `:7072` has no signals object, and `:7272` reads `'complete' == 'failed'`;
  - `:7138` and `:7209` time out at the 1 s provider-start wait (`:7172`, `:7248`), even run alone, so the provider never starts.
- Character emote metadata (2): `:1444` and `:1484`; the metadata now carries a `mixin_attempt_id`.
- The prompt-history stub (1): `test_accepted_send_survives_prompt_history_maintenance_refusal` (`:9783`) raises `RecoveryRequired` from its own `raw._raw_participant(history)` call, because its `default_prompt_history_path` stub cannot satisfy the guard's selected-path check.

**Why this needs care.** Some of these may be product defects rather than drift: vision-capable image sends are refused, a closed session reports ownership-changed copy, and canvas turns cannot save. A blind re-pin would hide them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 34 nodes passes against current production behaviour under the file's plain one-file command at `-n 4`, or is named in the notes as a product defect with a follow-up proposed, rather than re-pinned to the defect
- [ ] #2 No node is deleted or loosened unless the notes name the commit that changed the behaviour it pinned
- [ ] #3 The two stop-path usage nodes fail, if they fail, on their real assertion rather than on the provider-start wait; the before evidence is the `TimeoutError`
- [ ] #4 The prompt-history node still proves the send is accepted while history persistence is refused: it fails under a temporary edit that lets the history write through
- [ ] #5 The whole file passes 3 times at `-n 4` with the load recorded, apart from TASK-39045's node if that task has not landed
<!-- AC:END -->
