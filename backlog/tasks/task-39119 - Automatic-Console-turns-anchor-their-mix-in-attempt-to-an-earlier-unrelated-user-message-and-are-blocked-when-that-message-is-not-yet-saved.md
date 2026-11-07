---
id: TASK-39119
title: "Automatic Console turns anchor their mix-in attempt to an earlier, unrelated user message, and are blocked when that message is not yet saved"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: '2026-11-07 07:14'
labels:
- follow-up
- console
- agents
priority: medium
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_chat_models.py
- tldw_chatbook/DB/console_prompt_mixins.py
- tldw_chatbook/DB/recovery_core_schema.py
- Tests/UI/test_console_fleet_wake_ui_freshness.py
dependencies:
- TASK-39044
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-39044's notes (round-11 branch fu/r11-modals) report it as the remaining load flake in the fleet freshness node. Line numbers are dev's.

**The anchor.**
- Before a reply, the controller picks the last USER row before the assistant row as the mix-in attempt's anchor (`console_chat_controller.py:25022-25034`), with no check of the turn's origin. It then writes the attempt against it (`:25035-25085`).
- Automatic origins are `AGENT_WAKE`, `AGENT_CHAT_START` and `GOAL_ITERATION` (`console_chat_models.py:96-97`). Their nearest USER row belongs to an earlier, manual turn.

**What blocks the turn.**
- The insert trigger `console_prompt_mixin_attempt_anchor_insert` (`DB/recovery_core_schema.py:1167`) refuses an anchor that is not yet a saved message (`mixin_anchor_ownership`), and so do `DB/console_prompt_mixins.py:173`, `:421` and `:443`.
- The turn is then blocked with "Mix-in attempt could not be saved; response was not sent." The anchor requirement dates from 7970fbcada (2026-11-01).

**Seen in practice.**
- `test_wake_turn_in_a_nonviewed_session_flips_the_tab_glyph_off_running` failed this way in a round-11 stress run on the modals branch ("BLOCKED … 'Mix-in attempt could not be saved…'"). TASK-39044's notes record 2 of 6 failures at load 58.
- The list also said `test_composer_blocked_copy_names_the_wake_not_provider_setup` "fails on base the same way". That did not reproduce: it passed 2 of 2 on dev under a bootstrap marker, and no log shows it failing with this copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A wake, agent chat start or goal iteration never records its mix-in attempt against a user message from an earlier turn
- [x] #2 Such a turn is never blocked because an earlier user message is not yet saved; a deterministic test holds that save back and is red on dev
- [x] #3 Manual-turn anchoring and refusal are unchanged, and the existing mix-in tests pass by name
- [x] #4 The freshness node passes 10 of 10 at `-n 4`, with the load recorded
<!-- AC:END -->

## Implementation Plan

1. Check each automatic origin against the code. A wake and a goal iteration write no USER row (invariant 5), so the last USER row before their assistant row always belongs to an earlier turn. An agent chat start writes and durably commits its own opening USER row in an agent-created chat, so its anchor is already its own turn's row.
2. ADR-286 point 5 says automatic new turns do not resolve mix-ins (their snapshot is already forced empty). So a wake or goal iteration takes no anchor: in `_stream_assistant_response_inner`, a turn carrying a wake authorization (passed only for those two origins) skips the backwards USER-row scan and with it the attempt write, the path such a turn already took in a chat with no USER rows. Manual and queued turns, retries and chat starts are untouched.
3. Tests, red on base: a wake (`Tests/Chat/test_console_fleet_wake.py`) and a goal iteration (`Tests/Chat/test_console_goal_dispatch.py`) in a chat whose earlier USER row is held unsaved (`persist=False`). Each must reach the provider and record no mix-in attempt. The native chat-start test also asserts that every attempt anchors to the start's own opening row.
4. Run the mix-in suites by name (manual anchoring and refusal), and the freshness node 10 times at `-n 4` with the load recorded.

## Implementation Notes

**Which turns had the wrong anchor.** A wake and a goal iteration write no USER row (invariant 5), so the last USER row before their assistant row always belongs to an earlier turn. An agent chat start writes its opening prompt as a USER row in an agent-created chat and commits it durably, so its anchor is already its own row. The pinned native chat-start test now asserts this (`anchors == {opening row}`); that assertion passes on base as well.

**Fix.** In `_stream_assistant_response_inner`, a turn carrying a wake authorization (the FRESH path passes one only for `AGENT_WAKE` and `GOAL_ITERATION`) skips the backwards USER-row scan. With no anchor, there is no attempt write and no mix-in composition. That is the path such a turn already took in a chat with no USER rows, and ADR-286 point 5 says automatic new turns resolve no mix-ins (their snapshot is already forced empty for automatic origins). Manual, queued, retry and chat-start turns keep the scan unchanged. A negative control keyed on `WorkOrigin.AUTOMATIC` (which would also drop chat starts) still passed the chat-start pin, because the durable commit writes the chat start's attempt. So the rule keys on the wake authorization, the turns that have no USER row of their own.

**Tests (red on base, green on head).**
- `Tests/Chat/test_console_fleet_wake.py::test_a_wake_takes_no_mixin_anchor_from_an_earlier_unsaved_user_row`: the earlier USER row is never persisted (its save held back). Base: `BLOCKED, visible_copy='Mix-in attempt could not be saved; response was not sent.'`, with no provider call. Head: the wake reaches the provider and `console_prompt_mixin_attempts` is empty.
- `Tests/Chat/test_console_goal_dispatch.py::test_a_goal_iteration_takes_no_mixin_anchor_from_an_unsaved_earlier_turn`: the same for a goal iteration. Base: `native_run_id` None, `MixinSnapshotUnavailableError` rollback, `blocker=turn_refused`. Head: done, no attempt rows. The sibling test's TASK-39044 comment, which said the turn refuses an unsaved row, now points at this test.

**AC#3.** Mix-in suites by name (`Tests/Chat/test_console_prompt_mixin_{admission,caching,composition,forks,journey,retry,values}.py`, `Tests/DB/test_console_prompt_mixins.py`, `Tests/UI/test_console_prompt_mixins.py`, `Tests/UI/test_library_prompt_mixins.py`), with the wake, goal-dispatch, chat-start, freshness, progress-wake and goal light-surface files and the module size ratchet, at `-n 4`. Head: 785 passed, 2 failed. Base 8b762b52fa, same command: 779 passed, 6 failed. Both head failures (`test_library_prompt_mixins.py::test_literal_history_restore_keeps_kind_definition_and_guards[mixin-mixin]` and `[prompt-mixin]`) also failed on base; base's other four were more Library mix-in nodes and the freshness file's `test_wake_reply_reaches_the_viewed_transcript_without_a_switch` (a turn_refused block). Load 45-150, shared machine. A rerun of `test_library_prompt_mixins.py` alone gave head 24 passed / 4 failed and base 25 passed / 3 failed, the `test_literal_history_restore_keeps_kind_definition_and_guards` cases failing on both with `NoMatches: '#library-prompt-history-restore'` on the Library screen, a load-sensitive Library UI failure outside this change.

**AC#4.** `test_wake_turn_in_a_nonviewed_session_flips_the_tab_glyph_off_running`, under a temporary `parametrize('_rep', range(10))` that was removed afterwards: 10 passed at `-n 4` in 105.65 s, load 61-88.

**Size.** `console_chat_controller.py` grows by 2 lines (comment); its row in `Tests/Architecture/test_module_size_ratchet.py` was raised 29180 -> 29182 under the owner size waiver, with a dated comment.

**Lanes.** Both new tests sit in files whose nodes run in no PR lane outside `pr-fast-lane`, which takes no additions. Proposed for the lanes stream.

Files: `tldw_chatbook/Chat/console_chat_controller.py`, `Tests/Chat/test_console_fleet_wake.py`, `Tests/Chat/test_console_goal_dispatch.py`, `Tests/Chat/test_console_chat_start.py`, `Tests/Architecture/test_module_size_ratchet.py`.
